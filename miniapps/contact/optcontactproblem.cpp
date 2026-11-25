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
#include "coulomblaw.hpp"

namespace mfem
{

// Vector mortar data shares the problem's mechanics and parent coordinates.
struct OptContactProblem::CoulombData
{
   real_t coefficient;      ///< Nonnegative Coulomb coefficient.
   bool registered = false; ///< Whether this data owns a live coupling scheme.
   /// Reference configuration and last accepted coordinates supplied to Tribol.
   ParGridFunction reference_coordinates, previous_coordinates;
   Vector base_displacement, base_gap, normals; ///< Frozen affine kinematics.
   /// Frozen kinematic derivative G and force coupling G^T.
   std::unique_ptr<HypreParMatrix> gap_jacobian, force_jacobian;
   Array2D<int> node_dofs; ///< Cartesian true DOFs of each owned dual node.
   Array<int> supported;   ///< Whether mortar integration supports each node.
   static constexpr int scheme = 0; ///< Owned coupling scheme ID.
   /// Pressure per weighted gap/slip, used only to scale projection equations.
   static constexpr real_t equation_scale = 100.0;

   /** @brief Initialize the reference fields before any load is applied.
       @param space Borrowed parent displacement space.
       @param friction Validated Coulomb coefficient. */
   CoulombData(ParFiniteElementSpace *space, real_t friction)
      : coefficient(friction), reference_coordinates(space),
        previous_coordinates(space)
   {
      space->GetParMesh()->GetNodes(reference_coordinates);
      previous_coordinates = reference_coordinates;
   }

   /// Destroy transfer matrices and the scheme before its borrowed fields.
   ~CoulombData()
   {
      gap_jacobian.reset();
      force_jacobian.reset();
      if (registered)
      {
         tribol::finalize();
      }
   }
};

OptContactProblem::~OptContactProblem()
{
   ReleaseMemory();
}

void OptContactProblem::EnableCoulomb(real_t coefficient)
{
   MFEM_VERIFY(std::isfinite(coefficient) && coefficient >= 0.0,
               "Coulomb friction must be finite and nonnegative.");
   MFEM_VERIFY(!problem->IsNonlinear() && !bound_constraints,
               "Coulomb mortar requires linear mechanics without IP bounds.");
   MFEM_VERIFY(!coulomb_data,
               "Select Coulomb mortar once before forming contact.");
   coulomb_data = std::make_unique<CoulombData>(vfes, coefficient);
}

Vector &OptContactProblem::GetContactMultipliers()
{
   MFEM_VERIFY(coulomb_data && coulomb_data->registered,
               "Form the Coulomb contact system before accessing multipliers.");
   return tribol::getMfemContactPressure(CoulombData::scheme);
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
}

void OptContactProblem::FormContactSystem(ParGridFunction * coords_,
                                          const Vector & xref_)
{
   ReleaseMemory();

   coords = coords_;
   xref.SetDataAndSize(xref_.GetData(), xref_.Size());

   if (coulomb_data)
   {
      FormCoulombSystem(xref_);
      return;
   }

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
      *pmesh, *coords, non_mortar_attrs, mortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_CASE,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Formulations place multipliers on the first, nonmortar surface.
   ParGridFunction reference_coordinates(coords->ParFESpace());
   pmesh->GetNodes(reference_coordinates);
   tribol::registerMfemReferenceCoords(coupling_scheme_id, reference_coordinates);
   tribol::setMfemUseSingleMortarFormulation(coupling_scheme_id, true, false);
   tribol::getMfemContactPressure(coupling_scheme_id) = 0.0;
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

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Calculate scalar gaps and their frozen-geometry derivative.
   int cycle = 1;   // pseudo cycle
   real_t t = 1.0;  // pseudo time
   real_t dt = 1.0; // pseudo dt
   MFEM_VERIFY(tribol::update(cycle, t, dt) == 0,
               "Tribol scalar mortar calculation failed.");

   // Normalize the weighted gap derivative for the scalar IP constraints.
   auto Mfull = tribol::getMfemWeightedGapJacobian(coupling_scheme_id);
   MFEM_VERIFY(Mfull, "The scalar mortar formulation must supply a gap Jacobian.");
   Mfull->InvScaleRows(Mcslumpfull); // scaling
   HypreParMatrix * Me = Mfull->EliminateCols(ess_tdofs);
   delete Me;

   int h = Mfull->Height();
   SparseMatrix merged;
   Mfull->MergeDiagAndOffd(merged);
   // Remove stored zeros before they inflate the IP matrix-product graph.
   if (merged.NumNonZeroElems() > 0) { merged.Threshold(0.0); }
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

   const auto gap_true = tribol::getMfemContactGap(coupling_scheme_id);
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

void OptContactProblem::FormCoulombSystem(const Vector &displacement)
{
   auto &data = *coulomb_data;
   const int scheme = CoulombData::scheme;
   if (!data.registered)
   {
      // Formulations place scalar and vector multipliers on the nonmortar side.
      tribol::registerMfemCouplingScheme(
          scheme, 0, 1, *problem->GetMesh(), *coords, nonmortar_attrs,
          mortar_attrs, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
          tribol::SINGLE_MORTAR, tribol::FRICTION_COULOMB,
          tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
          tribol::ExecutionMode::Sequential);
      data.registered = true;
      tribol::registerMfemReferenceCoords(scheme, data.reference_coordinates);
      tribol::registerMfemPreviousCoords(scheme, data.previous_coordinates);
      tribol::setMfemUseSingleMortarFormulation(scheme, true, false);
      tribol::setBinningProximityScale(scheme, tribol_ratio);
      tribol::getMfemContactPressure(scheme) = 0.0;
   }
   data.previous_coordinates = *coords;
   data.base_displacement = displacement;
   // Repartition contact onto one rank for these small benchmark surfaces.
   tribol::updateMfemParallelDecomposition(1, true);
   real_t dt = 1.0;
   if (tribol::update(0, 0.0, dt))
   {
      throw std::runtime_error("Tribol frozen mortar calculation failed.");
   }
   // Copy through lvalues: moving a HypreParVector into its Vector base would
   // leave the data tied to the temporary Hypre object's lifetime.
   const auto gap_values = tribol::getMfemContactGap(scheme);
   const auto normal_values = tribol::getMfemNodalNormals(scheme);
   data.base_gap = gap_values;
   data.normals = normal_values;
   data.gap_jacobian = tribol::getMfemWeightedGapJacobian(scheme);
   data.force_jacobian = tribol::getMfemDfDp(scheme);
   MFEM_VERIFY(data.gap_jacobian && data.force_jacobian,
               "Frozen frictional mortar must supply both coupling matrices.");

   auto &space = *tribol::getMfemPressure(scheme).ParFESpace();
   dimU = vfes->GetTrueVSize();
   dimM = dimC = dimG = space.GetTrueVSize();
   num_constraints = space.GlobalTrueVSize();
   Array<int> owned;
   for (int node = 0; node < space.GetNDofs(); ++node)
   {
      if (space.GetLocalTDofNumber(space.DofToVDof(node, 0)) >= 0)
      {
         owned.Append(node);
      }
   }
   data.node_dofs.SetSize(owned.Size(), 3);
   data.supported.SetSize(owned.Size());
   SparseMatrix rows;
   data.gap_jacobian->MergeDiagAndOffd(rows);
   for (int i = 0; i < owned.Size(); ++i)
   {
      data.supported[i] = 0;
      for (int d = 0; d < 3; ++d)
      {
         const int tdof =
             space.GetLocalTDofNumber(space.DofToVDof(owned[i], d));
         data.node_dofs(i, d) = tdof;
         if (rows.GetRowNorml1(tdof) > 0.0)
         {
            data.supported[i] = 1;
         }
      }
   }
}

OptContactProblem::CoulombLawResult
OptContactProblem::EvaluateCoulombLaw(const Vector &displacement,
                                      bool derivatives) const
{
   const auto &data = *coulomb_data;
   const int scheme = CoulombData::scheme;
   const auto &multiplier = tribol::getMfemContactPressure(scheme);
   Vector increment(displacement);
   increment -= data.base_displacement;
   Vector kinematics(data.base_gap.Size());
   data.gap_jacobian->Mult(increment, kinematics);
   kinematics += data.base_gap;
   CoulombLawResult result;
   result.equation.SetSize(multiplier.Size());
   result.equation = 0.0;
   result.gap.SetSize(data.supported.Size());
   result.pressure.SetSize(data.supported.Size());
   result.slip.SetSize(multiplier.Size());
   result.slip = 0.0;
   result.state.SetSize(data.supported.Size());
   SparseMatrix dk(multiplier.Size()), dl(multiplier.Size());
   Array<real_t> normal(3), kinematic(3), multiplier_node(3);
   for (int node = 0; node < data.supported.Size(); ++node)
   {
      for (int d = 0; d < 3; ++d)
      {
         const int i = data.node_dofs(node, d);
         normal[d] = data.normals[i];
         kinematic[d] = kinematics[i];
         multiplier_node[d] = multiplier[i];
      }
      const auto law = internal::EvaluateCoulombProjection(
          normal, kinematic, multiplier_node, data.coefficient,
          CoulombData::equation_scale, data.supported[node]);
      result.gap[node] = law.gap;
      result.pressure[node] = law.pressure;
      result.state[node] = law.state;
      for (int d = 0; d < 3; ++d)
      {
         const int row = data.node_dofs(node, d);
         result.equation[row] = law.equation[d];
         result.slip[row] = law.slip[d];
         if (derivatives)
         {
            for (int e = 0; e < 3; ++e)
            {
               dk.Set(row, data.node_dofs(node, e), law.dk(d, e));
               dl.Set(row, data.node_dofs(node, e), law.dl(d, e));
            }
         }
      }
   }
   if (derivatives)
   {
      auto &space = *tribol::getMfemPressure(scheme).ParFESpace();
      auto assemble = [&](SparseMatrix &local)
      {
         local.Finalize();
         auto matrix = std::make_unique<HypreParMatrix>(
             comm, space.GlobalTrueVSize(), space.GetTrueDofOffsets(), &local);
         HypreStealOwnership(*matrix, local);
         matrix->CopyRowStarts();
         matrix->CopyColStarts();
         return matrix;
      };
      result.dk = assemble(dk);
      result.dl = assemble(dl);
   }
   return result;
}

Vector OptContactProblem::CoulombResidual(const Vector &displacement) const
{
   const auto &data = *coulomb_data;
   const int scheme = CoulombData::scheme;
   auto law = EvaluateCoulombLaw(displacement, false);
   const auto &multiplier = tribol::getMfemContactPressure(scheme);
   const int ndofs = displacement.Size();
   Vector residual(ndofs + multiplier.Size());
   Vector bulk(residual, 0, ndofs), force(ndofs);
   problem->GetGradient(displacement, bulk);
   data.force_jacobian->Mult(multiplier, force);
   bulk += force;
   bulk.SetSubVector(problem->GetEssentialDofs(), 0.0);
   Vector constraints(residual, ndofs, multiplier.Size());
   constraints = law.equation;
   return residual;
}

std::unique_ptr<BlockOperator>
OptContactProblem::CoulombJacobian(const Vector &displacement) const
{
   const auto &data = *coulomb_data;
   auto law = EvaluateCoulombLaw(displacement, true);
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = displacement.Size();
   offsets[2] = offsets[1] + law.equation.Size();
   auto jacobian = std::make_unique<BlockOperator>(offsets);
   jacobian->owns_blocks = 1;
   auto stiffness =
       std::make_unique<HypreParMatrix>(*problem->GetHessian(displacement));
   auto j01 = std::make_unique<HypreParMatrix>(*data.force_jacobian);
   auto j10 = std::unique_ptr<HypreParMatrix>(
       ParMult(law.dk.get(), data.gap_jacobian.get(), true));
   j01->EliminateRows(problem->GetEssentialDofs());
   delete j10->EliminateCols(problem->GetEssentialDofs());
   jacobian->SetBlock(0, 0, stiffness.release());
   jacobian->SetBlock(0, 1, j01.release());
   jacobian->SetBlock(1, 0, j10.release());
   jacobian->SetBlock(1, 1, law.dl.release());
   return jacobian;
}

bool OptContactProblem::CheckCoulombContact(const Vector &displacement) const
{
   const auto &data = *coulomb_data;
   const int scheme = CoulombData::scheme;
   auto law = EvaluateCoulombLaw(displacement, false);
   const auto &multiplier = tribol::getMfemContactPressure(scheme);
   Array<int> counts(4), global_counts(4);
   counts = 0;
   int invalid = 0, any_invalid = 0;
   constexpr real_t tolerance = 1.0e-6;
   for (int node = 0; node < data.supported.Size(); ++node)
   {
      // Count unsupported dual nodes separately while checking their zero LM.
      ++counts[data.supported[node] ? law.state[node] : 3];
      real_t tangent_squared = 0.0, dissipation = 0.0;
      for (int d = 0; d < 3; ++d)
      {
         const int i = data.node_dofs(node, d);
         const real_t tangent =
             multiplier[i] + law.pressure[node] * data.normals[i];
         tangent_squared += tangent * tangent;
         dissipation += tangent * law.slip[i];
      }
      const real_t limit = data.coefficient * std::max(0.0, law.pressure[node]);
      if (!std::isfinite(law.pressure[node]) || !std::isfinite(law.gap[node]) ||
          !std::isfinite(tangent_squared) || !std::isfinite(dissipation) ||
          law.pressure[node] < -tolerance || law.gap[node] < -tolerance ||
          std::abs(law.pressure[node] * law.gap[node]) > tolerance ||
          std::sqrt(tangent_squared) > limit + tolerance ||
          dissipation < -tolerance)
      {
         invalid = 1;
      }
   }
   MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm);
   MPI_Allreduce(counts.GetData(), global_counts.GetData(), 4, MPI_INT, MPI_SUM,
                 comm);
   if (!any_invalid && Mpi::Root())
   {
      out << " Coulomb contact: open=" << global_counts[0]
          << " stick=" << global_counts[1] << " slip=" << global_counts[2]
          << " unsupported=" << global_counts[3]
          << '\n';
   }
   return !any_invalid;
}

}
