// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "optcontactproblem.hpp"
#include "solver_utils.hpp"
#include "general/scan.hpp"

namespace mfem
{

namespace
{

class ScopedHypreHostPolicy
{
private:
   HYPRE_MemoryLocation old_memory = HYPRE_MEMORY_UNDEFINED;
   HYPRE_ExecutionPolicy old_execution = HYPRE_EXEC_UNDEFINED;
   bool restore = false;

public:
   ScopedHypreHostPolicy()
   {
      if (HYPRE_Initialized())
      {
         HYPRE_GetMemoryLocation(&old_memory);
         HYPRE_GetExecutionPolicy(&old_execution);
         HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
         HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
         restore = true;
      }
   }

   void Restore()
   {
      if (restore)
      {
         HYPRE_SetMemoryLocation(old_memory);
         HYPRE_SetExecutionPolicy(old_execution);
         restore = false;
      }
   }

   ~ScopedHypreHostPolicy() { Restore(); }
};

}

void OptContactProblem::ReleaseMemory()
{
   delete J; J = nullptr;
   delete Jt; Jt = nullptr;
   delete Pc; Pc = nullptr;
   delete NegId; NegId = nullptr;
   delete Iu; Iu = nullptr;
   delete negIu; negIu = nullptr;
   delete Mv; Mv = nullptr;
   delete Mcs; Mcs = nullptr;
   if (dcdu)
   {
      delete dcdu;
      dcdu = nullptr;
   }
}

void OptContactProblem::ComputeGapJacobian()
{
   if (J) { delete J; }
   J = const_cast<HypreParMatrix *>(SetupTribol(problem->GetMesh(),coords,
                                                problem->GetEssentialDofs(),
                                                mortar_attrs, nonmortar_attrs,gapv, tribol_ratio));

   dof_starts.SetSize(2);
   dof_starts[0] = J->ColPart()[0];
   dof_starts[1] = J->ColPart()[1];

   constraints_starts.SetSize(2);
   if (bound_constraints_activated)
   {
      constraints_starts[0] = J->RowPart()[0] + 2 * J->ColPart()[0];
      constraints_starts[1] = J->RowPart()[1] + 2 * J->ColPart()[1];
   }
   else
   {
      constraints_starts[0] = J->RowPart()[0];
      constraints_starts[1] = J->RowPart()[1];
   }
}

OptContactProblem::OptContactProblem(ElasticityOperator * problem_,
                                     const std::set<int> & mortar_attrs_,
                                     const std::set<int> & nonmortar_attrs_,
                                     real_t tribol_ratio_,
                                     bool bound_constraints_)
   : problem(problem_), mortar_attrs(mortar_attrs_),
     nonmortar_attrs(nonmortar_attrs_),
     tribol_ratio(tribol_ratio_),
     bound_constraints(bound_constraints_)
{
   comm = problem->GetComm();
   vfes = problem->GetFESpace();
   block_offsetsg.SetSize(4);

   grad_ref.UseDevice(true);
   xref.UseDevice(true);
   xrefbc.UseDevice(true);
   gapv.UseDevice(true);
   Mvlump.UseDevice(true);
   Mcslumpfull.UseDevice(true);
   Mcslump.UseDevice(true);
   dl.UseDevice(true);
   eps.UseDevice(true);
}

void OptContactProblem::FormContactSystem(ParGridFunction * coords_,
                                          const Vector & xref_)
{
   ReleaseMemory();

   coords = coords_;
   xref.NewMemoryAndSize(xref_.GetMemory(), xref_.Size(), false);
   xref.UseDevice(true);

   ComputeGapJacobian();

   problem->Getxrefbc(xrefbc);
   const Array<int> & ess_tdof_list = problem->GetEssentialDofs();
   const int num_ess_tdofs = ess_tdof_list.Size();
   Vector ess_values(num_ess_tdofs); ess_values.UseDevice(true);
   const bool use_dev = Device::IsEnabled();
   const int *ess_tdofs = ess_tdof_list.Read(use_dev);
   const real_t *xrefbc_in = xrefbc.Read(use_dev);
   real_t *ess_values_data = ess_values.Write(use_dev);
   mfem::forall(num_ess_tdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = ess_tdofs[i];
      ess_values_data[i] = dof >= 0 ? xrefbc_in[dof] : -xrefbc_in[-dof-1];
   });

   xrefbc = xref;
   ess_tdofs = ess_tdof_list.Read(use_dev);
   const real_t *ess_values_read = ess_values.Read(use_dev);
   real_t *xrefbc_out = xrefbc.ReadWrite(use_dev);
   mfem::forall(num_ess_tdofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = ess_tdofs[i];
      if (dof >= 0)
      {
         xrefbc_out[dof] = ess_values_read[i];
      }
      else
      {
         xrefbc_out[-dof-1] = -ess_values_read[i];
      }
   });

   if (problem->IsNonlinear())
   {
      energy_ref = problem->GetEnergy(xrefbc);
      problem->GetGradient(xrefbc,grad_ref);
      Kref = problem->GetHessian(xrefbc);
   }
   dimU = J->Width();
   dimG = J->Height();
   num_constraints = J->GetGlobalNumRows();

   block_offsetsg[0] = 0;
   block_offsetsg[1] = dimG;
   block_offsetsg[2] = dimU;
   block_offsetsg[3] = dimU;
   block_offsetsg.PartialSum();

   Vector diagVec(dimU); diagVec.UseDevice(true); diagVec = 0.0;
   diagVec = 1.0;
   Iu = MakeDiagonalHypreParMatrix(comm, GetGlobalNumDofs(), GetDofStarts(),
                                   diagVec);

   diagVec = -1.0;
   negIu = MakeDiagonalHypreParMatrix(comm, GetGlobalNumDofs(), GetDofStarts(),
                                      diagVec);

   dimM = dimG;
   dimC = dimM;

   ParBilinearForm MassForm(vfes);
   MassForm.AddDomainIntegrator(new VectorMassIntegrator);
   MassForm.Assemble();

   MassForm.Finalize(0);
   Mv = MassForm.ParallelAssemble();

   Vector onev(Mv->Width()); onev.UseDevice(true); onev = 1.0;
   Mvlump.SetSize(Mv->Height());
   Mv->Mult(onev, Mvlump);

   if (!dl.Size())
   {
      dl.SetSize(dimU); dl = 0.0;
      if (bound_constraints)
      {
         eps.SetSize(dimU); eps = 0.0; // minimum size of eps controlled by eps_min
      }
   }
}

HypreParMatrix * OptContactProblem::Duc(const BlockVector & x)
{
   if (bound_constraints_activated)
   {
      Array2D<const HypreParMatrix *> dcduBlockMatrix(3, 1);
      dcduBlockMatrix(0, 0) = J;
      dcduBlockMatrix(1, 0) = Iu;
      dcduBlockMatrix(2, 0) = negIu;
      if (dcdu) { delete dcdu; }
      dcdu = HypreParMatrixFromBlocks(dcduBlockMatrix);
      return dcdu;
   }
   else
   {
      return J;
   }
}

HypreParMatrix * OptContactProblem::Dmc(const BlockVector &)
{
   if (!NegId)
   {
      Vector negone(dimM); negone.UseDevice(true); negone = -1.0;
      NegId = MakeDiagonalHypreParMatrix(comm, GetGlobalNumConstraints(),
                                         GetConstraintsStarts(), negone);
   }
   return NegId;
}

HypreParMatrix * OptContactProblem::GetContactSubspaceTransferOperator()
{
   if (!Pc)
   {
      if (!Jt)
      {
         Jt = J->Transpose();
         Jt->EliminateRows(problem->GetEssentialDofs());
      }
      int hJt = Jt->Height();
      const bool use_dev = Device::IsEnabled();
      const HYPRE_Int *Jt_diag_i =
         mfem::Read(Jt->GetDiagMemoryI(), hJt + 1, use_dev);
      const HYPRE_Int *Jt_offd_i =
         mfem::Read(Jt->GetOffdMemoryI(), hJt + 1, use_dev);
      Array<int> row_ids(hJt), row_flags(hJt), nonzerorows(hJt),
            num_nonzerorows(1);
      row_ids.UseDevice(use_dev);
      row_flags.UseDevice(use_dev);
      nonzerorows.UseDevice(use_dev);
      num_nonzerorows.UseDevice(use_dev);

      int *row_ids_data = row_ids.Write(use_dev);
      int *row_flags_data = row_flags.Write(use_dev);
      mfem::forall(hJt, [=] MFEM_HOST_DEVICE (int i)
      {
         row_ids_data[i] = i;
         row_flags_data[i] = (Jt_diag_i[i+1] > Jt_diag_i[i]) ||
                             (Jt_offd_i[i+1] > Jt_offd_i[i]);
      });

      CopyFlagged(use_dev, row_ids.Read(use_dev), row_flags.Read(use_dev),
                  nonzerorows.Write(use_dev),
                  num_nonzerorows.Write(use_dev), hJt);
      int nrows_c = num_nonzerorows.HostRead()[0];
      nonzerorows.SetSize(nrows_c);
      SparseMatrix Pct;
      Pct.OverrideSize(nrows_c, vfes->GetTrueVSize());
      Pct.GetMemoryI().New(nrows_c + 1, MemoryType::HOST);
      Pct.GetMemoryJ().New(nrows_c, MemoryType::HOST);
      Pct.GetMemoryData().New(nrows_c, MemoryType::HOST);

      {
         int *I = Pct.WriteI(false);
         int *J = Pct.WriteJ(false);
         real_t *A = Pct.WriteData(false);
         const int *rows = nonzerorows.HostRead();
         for (int i = 0; i <= nrows_c; i++)
         {
            I[i] = i;
         }
         for (int i = 0; i < nrows_c; i++)
         {
            J[i] = rows[i];
            A[i] = 1.0;
         }
      }

      HYPRE_BigInt rows_c[2];

      HYPRE_BigInt row_offset_c;
      HYPRE_BigInt nrows_c_bigint = nrows_c;
      MPI_Scan(&nrows_c_bigint,&row_offset_c,1,MPITypeMap<HYPRE_BigInt>::mpi_type,
               MPI_SUM,comm);

      row_offset_c-=nrows_c_bigint;
      rows_c[0] = row_offset_c;
      rows_c[1] = row_offset_c+nrows_c;

      HYPRE_BigInt glob_nrows_c;
      HYPRE_BigInt glob_ncols_c = vfes->GlobalTrueVSize();
      MPI_Allreduce(&nrows_c_bigint, &glob_nrows_c,1,
                    MPITypeMap<HYPRE_BigInt>::mpi_type,
                    MPI_SUM,comm);

      HypreParMatrix *P_ct = new HypreParMatrix(comm, glob_nrows_c,
                                                glob_ncols_c, rows_c,
                                                vfes->GetTrueDofOffsets(),
                                                &Pct);
      HypreStealOwnership(*P_ct, Pct);
      Pc = P_ct->Transpose();
      delete P_ct;
   }
   return Pc;
}

void OptContactProblem::g(const Vector & d, Vector & gd)
{
   Vector temp(dimU); temp.UseDevice(true); temp = 0.0;
   temp.Set(1.0, d);
   temp.Add(-1.0, xref);
   J->Mult(temp, gd);
   gd.Add(1.0, gapv);
}

//           [     g1(d)      ]
// c(d, s) = [ eps + (d - dl) ] - s
//           [ eps - (d - dl) ]
void OptContactProblem::c(const BlockVector & x, Vector & y)
{
   x.SyncToBlocks();
   const Vector disp  = x.GetBlock(0);
   const Vector slack = x.GetBlock(1);

   if (bound_constraints_activated)
   {
      BlockVector yblock(block_offsetsg);
      yblock.UseDevice(true);
      for (int i = 0; i < yblock.NumBlocks(); i++)
      {
         yblock.GetBlock(i).UseDevice(true);
      }
      yblock = 0.0;

      g(disp, yblock.GetBlock(0));
      yblock.GetBlock(1).Set( 1.0, disp );
      yblock.GetBlock(1).Add(-1.0, dl);
      yblock.GetBlock(2).Set(-1.0, yblock.GetBlock(1));
      yblock.GetBlock(1).Add(1.0, eps);
      yblock.GetBlock(2).Add(1.0, eps);
      yblock.SyncFromBlocks();
      y.Set(1.0, yblock);
      y.Add(-1.0, slack);
   }
   else
   {
      g(disp, y);
      y.Add(-1., slack);
   }
}

real_t OptContactProblem::CalcObjective(const BlockVector & x, int & eval_err)
{
   x.SyncToBlocks();
   return E(x.GetBlock(0), eval_err);
}

void OptContactProblem::CalcObjectiveGrad(const BlockVector & x,
                                          BlockVector & y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
   y.SyncFromBlocks();
}

real_t OptContactProblem::E(const Vector & d, int & eval_err)
{
   if (problem->IsNonlinear())
   {
      // (d - xref)^T [ 1/2 K * (d - xref) + gradEQP] + EQP
      real_t energy = 0.0;
      Vector dx(dimU); dx.UseDevice(true); dx = 0.0;
      Vector temp(dimU); temp.UseDevice(true); temp = 0.0;
      dx.Set(1.0, d);
      dx.Add(-1.0, xrefbc);
      Kref->Mult(dx, temp);
      temp *= 0.5;
      temp.Add(1.0, grad_ref);
      energy = InnerProduct(comm, dx, temp);
      energy += energy_ref;
      eval_err = 0;
      return energy;
   }
   else
   {
      real_t energy = problem->GetEnergy(d);
      if (IsFinite(energy))
      {
         eval_err = 0;
      }
      else
      {
         eval_err = 1;
      }
      if (Mpi::Root() && eval_err == 1)
      {
         mfem::out << "energy = " << energy << std::endl;
         mfem::out << "eval_err = " << eval_err << std::endl;
      }
      return energy;
   }
}

void OptContactProblem::DdE(const Vector & d, Vector & gradE)
{
   if (problem->IsNonlinear())
   {
      // KQP * (d - xref) + gradEQP
      Vector dx(dimU); dx.UseDevice(true); dx = 0.0;
      dx.Set(1.0, d);
      dx.Add(-1.0, xrefbc);
      Kref->Mult(dx, gradE);
      gradE.Add(1.0, grad_ref);
   }
   else
   {
      return problem->GetGradient(d, gradE);
   }
}

HypreParMatrix * OptContactProblem::DddE(const Vector & d)
{
   return (problem->IsNonlinear()) ? Kref : problem->GetHessian(d);
}

void OptContactProblem::SetDisplacement(const Vector & dx,
                                        bool activate_constraints)
{
   const bool use_dev = eps.UseDevice() || dx.UseDevice();
   real_t *eps_data = eps.ReadWrite(use_dev);
   if (activate_constraints)
   {
      eps_min = std::max(eps_min, GlobalLpNorm(infinity(), eps.Normlinf(), comm));
      const real_t eps_floor = eps_min;
      mfem::forall(dimU, [=] MFEM_HOST_DEVICE (int j)
      {
         eps_data[j] = fmax(eps_floor, eps_data[j]);
      });
   }
   else
   {
      const real_t *dx_data = dx.Read(use_dev);
      mfem::forall(dimU, [=] MFEM_HOST_DEVICE (int j)
      {
         eps_data[j] = fmax(eps_data[j], fabs(dx_data[j]));
      });
   }
}

void OptContactProblem::ActivateBoundConstraints()
{
   dl.Set(1.0, xrefbc);
   bound_constraints_activated = true;
   constraints_starts[0] = J->RowPart()[0] + 2 * J->ColPart()[0];
   constraints_starts[1] = J->RowPart()[1] + 2 * J->ColPart()[1];
   num_constraints = J->GetGlobalNumRows() + 2 * J->GetGlobalNumCols();
   dimM = dimG + 2 * dimU;
   dimC = dimM;
}

HypreParMatrix * OptContactProblem::SetupTribol(ParMesh * pmesh,
                                                ParGridFunction * coords,
                                                const Array<int> & ess_tdofs, const std::set<int> & mortar_attrs,
                                                const std::set<int> & non_mortar_attrs,
                                                Vector &gap, real_t ratio)
{
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // Tribol and its MFEM adapter are host-only in this build. Keep both
   // HYPRE allocation and execution on the host for the complete Tribol
   // lifetime; switching only the allocation policy leaves HYPRE kernels
   // operating on host pointers.
   ScopedHypreHostPolicy hypre_host_policy;

   int coupling_scheme_id = 0;
   int mesh1_id = 0; int mesh2_id = 1;

   // Tribol's implicit mortar path is host-only. Make the coordinates current
   // on the host before handing the grid function to Tribol.
   coords->HostRead();
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id,
      *pmesh, *coords, mortar_attrs, non_mortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   tribol::setBinningProximityScale(coupling_scheme_id, ratio);

   // Access Tribol's pressure grid function (on the contact surface)
   auto& pressure = tribol::getMfemPressure(coupling_scheme_id);

   ParBilinearForm acs_form(pressure.ParFESpace());
   acs_form.AddDomainIntegrator(new MassIntegrator);
   acs_form.Assemble();
   acs_form.Finalize(0);
   Mcs = acs_form.ParallelAssemble();

   Vector onecs(Mcs->Width()); onecs.UseDevice(true); onecs = 1.0;
   Mcslumpfull.SetSize(Mcs->Height()); //Vector
   Mcs->Mult(onecs, Mcslumpfull);

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle = 1;   // pseudo cycle
   real_t t = 1.0;  // pseudo time
   real_t dt = 1.0; // pseudo dt
   tribol::update(cycle, t, dt);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);

   HypreParMatrix * Mfull = (HypreParMatrix *)(&A_blk->GetBlock(1,0));
   Mfull->InvScaleRows(Mcslumpfull); // scaling
   HypreParMatrix * Me = Mfull->EliminateCols(ess_tdofs);
   delete Me;

   FilteredContactMatrix filtered =
      FilterContactMatrixAndGap(*Mfull, *pressure.ParFESpace(),
                                coupling_scheme_id, gap);
   MPI_Comm matrix_comm = Mfull->GetComm();

   A_blk.reset();
   delete Mcs;
   Mcs = nullptr;
   tribol::finalize();
   hypre_host_policy.Restore();

   HypreParMatrix *M = new HypreParMatrix(matrix_comm,
                                          filtered.global_num_rows,
                                          filtered.global_num_cols,
                                          filtered.row_starts,
                                          filtered.col_starts,
                                          filtered.diag,
                                          filtered.offd,
                                          filtered.offd_cmap,
                                          true);
   M->CopyRowStarts();
   M->CopyColStarts();
   M->SetOwnerFlags(M->OwnsDiag(), M->OwnsOffd(),
                    filtered.offd_num_cols ? 1 : 0);
   return M;
}

OptContactProblem::FilteredContactMatrix
OptContactProblem::FilterContactMatrixAndGap(
   HypreParMatrix &contact_matrix,
   ParFiniteElementSpace &pfes,
   int coupling_scheme_id,
   Vector &gap)
{
   const int h = contact_matrix.Height();
   HYPRE_BigInt *offd_cmap = nullptr;
   HYPRE_Int offd_num_cols = 0;
   contact_matrix.GetOffdColMap(offd_cmap, offd_num_cols);

   const bool use_dev = Device::IsEnabled();
   const int diag_nnz = contact_matrix.GetDiagMemoryData().Capacity();
   const int offd_nnz = contact_matrix.GetOffdMemoryData().Capacity();
   const HYPRE_Int *diag_i =
      mfem::Read(contact_matrix.GetDiagMemoryI(), h + 1, use_dev);
   const real_t *diag_data =
      mfem::Read(contact_matrix.GetDiagMemoryData(), diag_nnz, use_dev);
   const HYPRE_Int *offd_i =
      mfem::Read(contact_matrix.GetOffdMemoryI(), h + 1, use_dev);
   const real_t *offd_data =
      mfem::Read(contact_matrix.GetOffdMemoryData(), offd_nnz, use_dev);

   Array<int> row_ids(h), row_flags(h), nonzero_rows(h), num_nonzero_rows(1);
   Vector row_l1_norms(h);
   row_ids.UseDevice(use_dev);
   row_flags.UseDevice(use_dev);
   nonzero_rows.UseDevice(use_dev);
   num_nonzero_rows.UseDevice(use_dev);
   row_l1_norms.UseDevice(true);

   int *row_ids_data = row_ids.Write(use_dev);
   real_t *row_l1_norms_data = row_l1_norms.Write(use_dev);
   mfem::forall(h, [=] MFEM_HOST_DEVICE (int i)
   {
      real_t row_l1_norm = 0.0;
      for (int j = diag_i[i]; j < diag_i[i+1]; j++)
      {
         row_l1_norm += fabs(diag_data[j]);
      }
      for (int j = offd_i[i]; j < offd_i[i+1]; j++)
      {
         row_l1_norm += fabs(offd_data[j]);
      }
      row_ids_data[i] = i;
      row_l1_norms_data[i] = row_l1_norm;
   });

   const real_t max_l1_row_norm = h ? row_l1_norms.Max() : 0.0;
   const real_t rel_row_norm_threshold = 1.e-5;
   const real_t row_norm_threshold = rel_row_norm_threshold * max_l1_row_norm;
   const real_t *row_l1_norms_read = row_l1_norms.Read(use_dev);
   int *row_flags_data = row_flags.Write(use_dev);
   mfem::forall(h, [=] MFEM_HOST_DEVICE (int i)
   {
      row_flags_data[i] = row_l1_norms_read[i] > row_norm_threshold;
   });

   CopyFlagged(use_dev, row_ids.Read(use_dev), row_flags.Read(use_dev),
               nonzero_rows.Write(use_dev), num_nonzero_rows.Write(use_dev),
               h);
   const int hnew = num_nonzero_rows.HostRead()[0];
   nonzero_rows.SetSize(hnew);

   Array<int> diag_row_nnz(hnew), offd_row_nnz(hnew);
   Array<int> diag_offsets(hnew + 1), offd_offsets(hnew + 1);
   diag_row_nnz.UseDevice(use_dev);
   offd_row_nnz.UseDevice(use_dev);
   diag_offsets.UseDevice(use_dev);
   offd_offsets.UseDevice(use_dev);

   int *diag_offsets_data = diag_offsets.Write(use_dev);
   int *offd_offsets_data = offd_offsets.Write(use_dev);
   mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
   {
      diag_offsets_data[0] = 0;
      offd_offsets_data[0] = 0;
   });
   if (hnew)
   {
      int *diag_row_nnz_data = diag_row_nnz.Write(use_dev);
      int *offd_row_nnz_data = offd_row_nnz.Write(use_dev);
      const int *nonzero_rows_data = nonzero_rows.Read(use_dev);
      mfem::forall(hnew, [=] MFEM_HOST_DEVICE (int i)
      {
         const int row = nonzero_rows_data[i];
         diag_row_nnz_data[i] = diag_i[row+1] - diag_i[row];
         offd_row_nnz_data[i] = offd_i[row+1] - offd_i[row];
      });
      InclusiveScan(use_dev, diag_row_nnz.Read(use_dev),
                    diag_offsets_data + 1, hnew);
      InclusiveScan(use_dev, offd_row_nnz.Read(use_dev),
                    offd_offsets_data + 1, hnew);
   }

   const int diag_filtered_nnz = diag_offsets.HostRead()[hnew];
   const int offd_filtered_nnz = offd_offsets.HostRead()[hnew];

   SparseMatrix *diag = new SparseMatrix;
   diag->OverrideSize(hnew, contact_matrix.Width());
   diag->GetMemoryI().New(hnew + 1, Device::GetDeviceMemoryType());
   diag->GetMemoryJ().New(diag_filtered_nnz, Device::GetDeviceMemoryType());
   diag->GetMemoryData().New(diag_filtered_nnz, Device::GetDeviceMemoryType());

   SparseMatrix *offd = new SparseMatrix;
   offd->OverrideSize(hnew, offd_num_cols);
   offd->GetMemoryI().New(hnew + 1, Device::GetDeviceMemoryType());
   offd->GetMemoryJ().New(offd_filtered_nnz, Device::GetDeviceMemoryType());
   offd->GetMemoryData().New(offd_filtered_nnz, Device::GetDeviceMemoryType());

   int *diag_filtered_i = diag->WriteI(use_dev);
   int *offd_filtered_i = offd->WriteI(use_dev);
   const int *diag_offsets_read = diag_offsets.Read(use_dev);
   const int *offd_offsets_read = offd_offsets.Read(use_dev);
   mfem::forall(hnew + 1, [=] MFEM_HOST_DEVICE (int i)
   {
      diag_filtered_i[i] = diag_offsets_read[i];
      offd_filtered_i[i] = offd_offsets_read[i];
   });

   const HYPRE_Int *diag_j =
      mfem::Read(contact_matrix.GetDiagMemoryJ(), diag_nnz, use_dev);
   const HYPRE_Int *offd_j =
      mfem::Read(contact_matrix.GetOffdMemoryJ(), offd_nnz, use_dev);
   int *diag_filtered_j = diag->WriteJ(use_dev);
   int *offd_filtered_j = offd->WriteJ(use_dev);
   real_t *diag_filtered_data = diag->WriteData(use_dev);
   real_t *offd_filtered_data = offd->WriteData(use_dev);
   const int *filtered_rows_data = nonzero_rows.Read(use_dev);
   mfem::forall(hnew, [=] MFEM_HOST_DEVICE (int i)
   {
      const int row = filtered_rows_data[i];
      int row_nnz_offset = diag_offsets_read[i];
      for (int j = diag_i[row]; j < diag_i[row+1]; j++)
      {
         diag_filtered_j[row_nnz_offset] = diag_j[j];
         diag_filtered_data[row_nnz_offset] = diag_data[j];
         row_nnz_offset++;
      }

      row_nnz_offset = offd_offsets_read[i];
      for (int j = offd_i[row]; j < offd_i[row+1]; j++)
      {
         offd_filtered_j[row_nnz_offset] = offd_j[j];
         offd_filtered_data[row_nnz_offset] = offd_data[j];
         row_nnz_offset++;
      }
   });

   const int nrows = hnew;
   HYPRE_BigInt nrows_bigint = nrows;
   HYPRE_BigInt row_offset;
   MPI_Scan(&nrows_bigint,&row_offset,1,MPITypeMap<HYPRE_BigInt>::mpi_type,
            MPI_SUM,contact_matrix.GetComm());

   FilteredContactMatrix filtered;
   row_offset-=nrows;
   filtered.row_starts[0] = row_offset;
   filtered.row_starts[1] = row_offset+nrows;
   filtered.col_starts[0] = contact_matrix.ColPart()[0];
   filtered.col_starts[1] = contact_matrix.ColPart()[1];

   MPI_Allreduce(&nrows_bigint, &filtered.global_num_rows,1,
                 MPITypeMap<HYPRE_BigInt>::mpi_type,
                 MPI_SUM,contact_matrix.GetComm());
   filtered.global_num_cols = contact_matrix.GetGlobalNumCols();
   filtered.offd_num_cols = offd_num_cols;

   if (offd_num_cols)
   {
      filtered.offd_cmap = new HYPRE_BigInt[offd_num_cols];
      for (int i = 0; i < offd_num_cols; i++)
      {
         filtered.offd_cmap[i] = offd_cmap[i];
      }
   }

   filtered.diag = diag;
   filtered.offd = offd;

   Vector gap_full;
   tribol::getMfemGap(coupling_scheme_id, gap_full);
   gap_full.HostRead();
   gap_full.UseDevice(true);

   auto& P_submesh = *pfes.GetProlongationMatrix();
   Vector gap_true(P_submesh.Width());
   gap_true.UseDevice(true);

   P_submesh.MultTranspose(gap_full,gap_true);
   gap.SetSize(nrows);
   gap.UseDevice(true);
   Mcslump.SetSize(nrows);
   Mcslump.UseDevice(true);

   const real_t *gap_true_data = gap_true.Read(use_dev);
   const real_t *Mcslumpfull_data = Mcslumpfull.Read(use_dev);
   real_t *gap_data = gap.Write(use_dev);
   real_t *Mcslump_data = Mcslump.Write(use_dev);
   const int *nonzero_rows_data = nonzero_rows.Read(use_dev);
   mfem::forall(nrows, [=] MFEM_HOST_DEVICE (int i)
   {
      const int row = nonzero_rows_data[i];
      gap_data[i] = gap_true_data[row];
      Mcslump_data[i] = Mcslumpfull_data[row];
   });
   gap /= Mcslump;

   return filtered;
}

}
