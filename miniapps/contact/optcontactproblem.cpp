// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

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
}

void OptContactProblem::FormContactSystem(ParGridFunction * coords_,
                                          const Vector & xref_)
{
   ReleaseMemory();

   coords = coords_;
   xref.SetDataAndSize(xref_.GetData(), xref_.Size());

   ComputeGapJacobian();

   problem->Getxrefbc(xrefbc);
   const Array<int> & ess_tdof_list = problem->GetEssentialDofs();
   Vector ess_values;

   xrefbc.GetSubVector(ess_tdof_list, ess_values);
   xrefbc=xref;
   xrefbc.SetSubVector(ess_tdof_list, ess_values);

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

   Vector diagVec(dimU); diagVec = 0.0;
   SparseMatrix * tempSparse;

   diagVec = 1.0;
   tempSparse = new SparseMatrix(diagVec);
   Iu = new HypreParMatrix(comm, GetGlobalNumDofs(), GetDofStarts(), tempSparse);
   HypreStealOwnership(*Iu, *tempSparse);
   delete tempSparse;

   diagVec = -1.0;
   tempSparse = new SparseMatrix(diagVec);
   negIu = new HypreParMatrix(comm, GetGlobalNumDofs(), GetDofStarts(),
                              tempSparse);
   HypreStealOwnership(*negIu, *tempSparse);
   delete tempSparse;

   dimM = dimG;
   dimC = dimM;

   ParBilinearForm MassForm(vfes);
   MassForm.AddDomainIntegrator(new VectorMassIntegrator);
   MassForm.Assemble();

   Array<int> empty_tdof_list;
   Mv = new HypreParMatrix();
   MassForm.FormSystemMatrix(empty_tdof_list,*Mv);

   Vector onev(Mv->Width()); onev = 1.0;
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
      Vector negone(dimM); negone = -1.0;
      SparseMatrix diag(negone);
      NegId = new HypreParMatrix(comm, GetGlobalNumConstraints(),
                                 GetConstraintsStarts(), &diag);
      HypreStealOwnership(*NegId, diag);
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
      SparseMatrix mergedJt;
      Jt->MergeDiagAndOffd(mergedJt);
      Array<int> nonzerorows;
      for (int i = 0; i<hJt; i++)
      {
         if (!mergedJt.RowIsEmpty(i))
         {
            nonzerorows.Append(i);
         }
      }
      int nrows_c = nonzerorows.Size();
      SparseMatrix Pct(nrows_c,vfes->GlobalTrueVSize());

      for (int i = 0; i<nrows_c; i++)
      {
         int col = nonzerorows[i]+vfes->GetMyTDofOffset();
         Pct.Set(i,col,1.0);
      }
      Pct.Finalize();

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

      HYPRE_BigInt * J;
#ifndef HYPRE_BIGINT
      J = Pct.GetJ();
#else
      J = new HYPRE_BigInt[Pct.NumNonZeroElems()];
      for (int i = 0; i < Pct.NumNonZeroElems(); i++)
      {
         J[i] = Pct.GetJ()[i];
      }
#endif
      HypreParMatrix * P_ct = new HypreParMatrix(comm, nrows_c, glob_nrows_c,
                                                 glob_ncols_c, Pct.GetI(), J,
                                                 Pct.GetData(), rows_c,vfes->GetTrueDofOffsets());
      Pc = P_ct->Transpose();
      delete P_ct;
#ifdef HYPRE_BIGINT
      delete [] J;
#endif
   }
   return Pc;
}

void OptContactProblem::g(const Vector & d, Vector & gd)
{
   Vector temp(dimU); temp = 0.0;
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
   const Vector disp  = x.GetBlock(0);
   const Vector slack = x.GetBlock(1);

   if (bound_constraints_activated)
   {
      BlockVector yblock(block_offsetsg); yblock = 0.0;

      g(disp, yblock.GetBlock(0));
      yblock.GetBlock(1).Set( 1.0, disp );
      yblock.GetBlock(1).Add(-1.0, dl);
      yblock.GetBlock(2).Set(-1.0, yblock.GetBlock(1));
      yblock.GetBlock(1).Add(1.0, eps);
      yblock.GetBlock(2).Add(1.0, eps);
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
   return E(x.GetBlock(0), eval_err);
}

void OptContactProblem::CalcObjectiveGrad(const BlockVector & x,
                                          BlockVector & y)
{
   DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
}

real_t OptContactProblem::E(const Vector & d, int & eval_err)
{
   if (problem->IsNonlinear())
   {
      // (d - xref)^T [ 1/2 K * (d - xref) + gradEQP] + EQP
      real_t energy = 0.0;
      Vector dx(dimU); dx = 0.0;
      Vector temp(dimU); temp = 0.0;
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
      Vector dx(dimU); dx = 0.0;
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
   if (activate_constraints)
   {
      eps_min = std::max(eps_min, GlobalLpNorm(infinity(), eps.Normlinf(), comm));
      for (int j = 0; j < dimU; j++)
      {
         eps(j) = std::max(eps_min, eps(j));
      }
   }
   else
   {
      for (int j = 0; j < dimU; j++)
      {
         eps(j) = std::max(eps(j), std::abs(dx(j)));
      }
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

   int coupling_scheme_id = 0;
   int mesh1_id = 0; int mesh2_id = 1;

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
   Array<int> empty_tdof_list;
   Mcs = new HypreParMatrix();
   acs_form.FormSystemMatrix(empty_tdof_list,*Mcs);

   Vector onecs(Mcs->Width()); onecs = 1.0;
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

   int h = Mfull->Height();
   SparseMatrix merged;
   Mfull->MergeDiagAndOffd(merged);
   Array<int> nonzero_rows;

   real_t max_l1_row_norm = 0.0;
   real_t rel_row_norm_threshold = 1.e-5;
   for (int i = 0; i < h; i++)
   {
      if (!merged.RowIsEmpty(i))
      {
         max_l1_row_norm = std::max( max_l1_row_norm, merged.GetRowNorml1(i));
      }
   }

   for (int i = 0; i<h; i++)
   {
      if (!merged.RowIsEmpty(i))
      {
         if (merged.GetRowNorml1(i) > rel_row_norm_threshold * max_l1_row_norm)
         {
            nonzero_rows.Append(i);
         }
      }
   }

   int hnew = nonzero_rows.Size();
   SparseMatrix P(hnew,h);

   for (int i = 0; i<hnew; i++)
   {
      int col = nonzero_rows[i];
      P.Set(i,col,1.0);
   }
   P.Finalize();

   SparseMatrix * reduced_merged = Mult(P,merged);

   HYPRE_BigInt rows[2];
   int nrows = reduced_merged->Height();
   HYPRE_BigInt nrows_bigint = nrows;
   HYPRE_BigInt row_offset;
   MPI_Scan(&nrows_bigint,&row_offset,1,MPITypeMap<HYPRE_BigInt>::mpi_type,
            MPI_SUM,Mfull->GetComm());

   row_offset-=nrows;
   rows[0] = row_offset;
   rows[1] = row_offset+nrows;
   HYPRE_BigInt glob_nrows;
   MPI_Allreduce(&nrows_bigint, &glob_nrows,1,MPITypeMap<HYPRE_BigInt>::mpi_type,
                 MPI_SUM,Mfull->GetComm());

   HYPRE_BigInt glob_ncols = reduced_merged->Width();

   HYPRE_BigInt * J;
#ifndef HYPRE_BIGINT
   J = reduced_merged->GetJ();
#else
   J = new HYPRE_BigInt[reduced_merged->NumNonZeroElems()];
   for (int i = 0; i < reduced_merged->NumNonZeroElems(); i++)
   {
      J[i] = reduced_merged->GetJ()[i];
   }
#endif

   HypreParMatrix * M = new HypreParMatrix(Mfull->GetComm(), nrows, glob_nrows,
                                           glob_ncols, reduced_merged->GetI(), J,
                                           reduced_merged->GetData(), rows, Mfull->ColPart());
   delete reduced_merged;

#ifdef HYPRE_BIGINT
   delete [] J;
#endif

   Vector gap_full;
   tribol::getMfemGap(coupling_scheme_id, gap_full);

   auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
   Vector gap_true(P_submesh.Width());

   P_submesh.MultTranspose(gap_full,gap_true);
   gap.SetSize(nrows);
   Mcslump.SetSize(nrows);

   for (int i = 0; i<nrows; i++)
   {
      gap[i] = gap_true[nonzero_rows[i]];
      Mcslump(i) = Mcslumpfull(nonzero_rows[i]);
   }
   gap /= Mcslump;
   tribol::finalize();
   return M;
}

}
