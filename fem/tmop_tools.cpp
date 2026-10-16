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

#include "tmop_tools.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "../general/osockstream.hpp"

namespace mfem
{

using namespace mfem;

void AdvectorCG::SetInitialField(const Vector &init_nodes,
                                 const Vector &init_field)
{
   nodes0 = init_nodes;
   field0 = init_field;
}

void AdvectorCG::ComputeAtNewPosition(const Vector &new_mesh_nodes,
                                      Vector &new_field,
                                      int nodes_ordering)
{
   MFEM_VERIFY(nodes0.Size() == new_mesh_nodes.Size(),
               "AdvectorCG assumes fixed mesh topology!");

   FiniteElementSpace *space = fes;
#ifdef MFEM_USE_MPI
   if (pfes) { space = pfes; }
#endif
   int fes_ordering = space->GetOrdering(),
       ncomp = space->GetVDim();

   const int dof_cnt = field0.Size() / ncomp;

   new_field = field0;
   Vector new_field_temp;
   for (int i = 0; i < ncomp; i++)
   {
      if (fes_ordering == Ordering::byNODES)
      {
         new_field_temp.MakeRef(new_field, i*dof_cnt, dof_cnt);
      }
      else
      {
         new_field_temp.SetSize(dof_cnt);
         for (int j = 0; j < dof_cnt; j++)
         {
            new_field_temp(j) = new_field(i + j*ncomp);
         }
      }
      ComputeAtNewPositionScalar(new_mesh_nodes, new_field_temp);
      if (fes_ordering == Ordering::byVDIM)
      {
         for (int j = 0; j < dof_cnt; j++)
         {
            new_field(i + j*ncomp) = new_field_temp(j);
         }
      }
   }

   field0 = new_field;
   nodes0 = new_mesh_nodes;
}

void AdvectorCG::ComputeAtNewPositionScalar(const Vector &new_mesh_nodes,
                                            Vector &new_field)
{
   Mesh *m = mesh;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
#endif

   MFEM_VERIFY(m != NULL, "No mesh has been given to the AdaptivityEvaluator.");

   // This will be used to move the positions.
   GridFunction *mesh_nodes = m->GetNodes();
   *mesh_nodes = nodes0;
   real_t minv = new_field.Min(), maxv = new_field.Max();

   // Velocity of the positions.
   GridFunction u(mesh_nodes->FESpace());
   subtract(new_mesh_nodes, nodes0, u);

   // Define a scalar FE space for the solution, and the advection operator.
   TimeDependentOperator *oper = NULL;
   FiniteElementSpace *fess = NULL;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfess = NULL;
#endif
   if (fes)
   {
      fess = new FiniteElementSpace(fes->GetMesh(), fes->FEColl(), 1);
      oper = new SerialAdvectorCGOper(nodes0, u, *fess, al);
   }
#ifdef MFEM_USE_MPI
   else if (pfes)
   {
      pfess = new ParFiniteElementSpace(pfes->GetParMesh(), pfes->FEColl(), 1);
      oper  = new ParAdvectorCGOper(nodes0, u, *pfess, al, opt_mt);
   }
#endif
   MFEM_VERIFY(oper != NULL,
               "No FE space has been given to the AdaptivityEvaluator.");
   ode_solver.Init(*oper);

   // Compute some time step [mesh_size / speed].
   real_t h_min = std::numeric_limits<real_t>::infinity();
   for (int i = 0; i < m->GetNE(); i++)
   {
      h_min = std::min(h_min, m->GetElementSize(i));
   }
   real_t v_max = 0.0;
   const int s  = u.Size()/m->Dimension();

   u.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      real_t vel = 0.;
      for (int j = 0; j < m->Dimension(); j++)
      {
         vel += u(i+j*s)*u(i+j*s);
      }
      v_max = std::max(v_max, vel);
   }

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      real_t v_loc = v_max, h_loc = h_min;
      MPI_Allreduce(&v_loc, &v_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    pfes->GetComm());
      MPI_Allreduce(&h_loc, &h_min, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pfes->GetComm());
   }
#endif

   if (v_max == 0.0) // No need to change the field.
   {
      delete oper;
      delete fess;
#ifdef MFEM_USE_MPI
      delete pfess;
#endif
      return;
   }

   v_max = std::sqrt(v_max);
   real_t dt = dt_scale * h_min / v_max;

   real_t t = 0.0;
   bool last_step = false;
   while (!last_step)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }
      ode_solver.Step(new_field, t, dt);
   }

   real_t glob_minv = minv,
          glob_maxv = maxv;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(&minv, &glob_minv, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pfes->GetComm());
      MPI_Allreduce(&maxv, &glob_maxv, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    pfes->GetComm());
   }
#endif

   // Trim the overshoots and undershoots.
   new_field.HostReadWrite();
   for (int i = 0; i < new_field.Size(); i++)
   {
      if (new_field(i) < glob_minv) { new_field(i) = glob_minv; }
      if (new_field(i) > glob_maxv) { new_field(i) = glob_maxv; }
   }

   delete oper;
   delete fess;
#ifdef MFEM_USE_MPI
   delete pfess;
#endif
}

SerialAdvectorCGOper::SerialAdvectorCGOper(const Vector &x_start,
                                           GridFunction &vel,
                                           FiniteElementSpace &fes,
                                           AssemblyLevel al)
   : TimeDependentOperator(fes.GetVSize()),
     x0(x_start), x_now(*fes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&fes), K(&fes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void SerialAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const real_t t = GetTime();
   add(x0, t, u, x_now);
   K.FESpace()->GetMesh()->NodesUpdated();

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   Vector rhs(K.Size());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

#ifdef MFEM_USE_SINGLE
   const real_t rtol = 1e-4;
#else
   const real_t rtol = 1e-12;
#endif

   // Solve.
   di_dt = 0.0;
   if (al == AssemblyLevel::PARTIAL)
   {
      // Solve mat-free on the tdofs as the JacobiSmoother operates on tdofs.
      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      M.FormLinearSystem(ess_tdof_list, di_dt, rhs, A, X, B);
      OperatorJacobiSmoother S(M, ess_tdof_list);
      PCG(*A, S, B, X, 0, 100, rtol, 0.0);
      M.RecoverFEMSolution(X, rhs, di_dt);
   }
   else
   {
      // Solve the SpMat directly on the ldofs.
      DSmoother S(M.SpMat());
      PCG(M.SpMat(), S, rhs, di_dt, 0, 100, rtol, 0.0);
   }
}

#ifdef MFEM_USE_MPI
ParAdvectorCGOper::ParAdvectorCGOper(const Vector &x_start,
                                     GridFunction &vel,
                                     ParFiniteElementSpace &pfes,
                                     AssemblyLevel al,
                                     MemoryType mt)
   : TimeDependentOperator(pfes.GetVSize()),
     x0(x_start), x_now(*pfes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&pfes), K(&pfes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   if (al == AssemblyLevel::PARTIAL)
   {
      Kinteg->SetPAMemoryType(mt);
   }
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   if (al == AssemblyLevel::PARTIAL)
   {
      Minteg->SetPAMemoryType(mt);
   }
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void ParAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const real_t t = GetTime();
   add(x0, t, u, x_now);
   K.ParFESpace()->GetParMesh()->NodesUpdated();

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   ParGridFunction rhs(K.ParFESpace());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   HypreParVector *RHS = rhs.ParallelAssemble();
   HypreParVector X(K.ParFESpace());
   X = 0.0;

   OperatorHandle Mop;
   Solver *prec = nullptr;
   Array<int> ess_tdof_list;
   if (al == AssemblyLevel::PARTIAL)
   {
      M.FormSystemMatrix(ess_tdof_list, Mop);
      prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   }
   else
   {
      Mop.Reset(M.ParallelAssemble());
      prec = new HypreSmoother;
      static_cast<HypreSmoother*>(prec)->SetType(HypreSmoother::Jacobi, 1);
   }

   CGSolver lin_solver(M.ParFESpace()->GetParMesh()->GetComm());
   lin_solver.SetPreconditioner(*prec);
   lin_solver.SetOperator(*Mop);
#ifdef MFEM_USE_SINGLE
   const real_t rtol = 1e-4;
#else
   const real_t rtol = 1e-8;
#endif
   lin_solver.SetRelTol(rtol); lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(*RHS, X);
   K.ParFESpace()->GetProlongationMatrix()->Mult(X, di_dt);

   delete RHS;
   delete prec;
}
#endif

#ifdef MFEM_USE_GSLIB
void InterpolatorFP::SetInitialField(const Vector &init_nodes,
                                     const Vector &init_field)
{
   nodes0 = init_nodes;
   Mesh *m = mesh;
   FiniteElementSpace *f = fes;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
   if (pfes)  { f = pfes; }
#endif
   m->SetNodes(nodes0);

   if (m->GetNodes()->FESpace()->IsDGSpace())
   {
      MFEM_ABORT("InterpolatorFP is not supported for periodic meshes yet.");
   }

   const real_t rel_bbox_el = 0.1;
   const real_t newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;

   if (finder)
   {
      finder->FreeData();
      delete finder;
   }

#ifdef MFEM_USE_MPI
   if (pfes) { finder = new FindPointsGSLIB(pfes->GetComm()); }
   else      { finder = new FindPointsGSLIB(); }
#else
   finder = new FindPointsGSLIB();
#endif
   finder->Setup(*m, rel_bbox_el, newton_tol, npts_at_once);

   field0_gf.SetSpace(f);
   field0_gf = init_field;
}

void InterpolatorFP::ComputeAtNewPosition(const Vector &new_mesh_nodes,
                                          Vector &new_field, int nodes_ordering)
{
   // TODO - this is here only to prevent breaking user codes. To be removed.
   // If the meshes are different, one has to call SetNewFieldFESpace().
   // If only some positions are interpolated, use ComputeAtGivenPositions().
   if (fes_new_field == nullptr && new_mesh_nodes.Size() != nodes0.Size())
   {
      MFEM_WARNING("Deprecated -- use ComputeAtGivenPositions() instead!");
      ComputeAtGivenPositions(new_mesh_nodes, new_field, nodes_ordering);
      return;
   }

   const FiniteElementSpace *fes_field =
      (fes_new_field) ? fes_new_field : field0_gf.FESpace();
   const int dim = fes_field->GetMesh()->Dimension();

   if (new_mesh_nodes.Size() / dim != fes_field->GetNDofs())
   {
      // The nodes of the FE space don't coincide with the mesh nodes.
      Vector mapped_nodes;
      fes_field->GetNodePositions(new_mesh_nodes, mapped_nodes);
      finder->Interpolate(mapped_nodes, field0_gf, new_field);
   }
   else
   {
      finder->Interpolate(new_mesh_nodes, field0_gf, new_field, nodes_ordering);
   }
}

void InterpolatorFP::ComputeAtGivenPositions(const Vector &positions,
                                             Vector &values, int p_ordering)
{
   finder->Interpolate(positions, field0_gf, values, p_ordering);
}

#endif

real_t TMOPNewtonSolver::ComputeScalingFactor(const Vector &d_in,
                                              const Vector &b) const
{
   const FiniteElementSpace *fes = NULL;
   real_t energy_in = 0.0;
#ifdef MFEM_USE_MPI
   const ParNonlinearForm *p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(!(parallel && p_nlf == NULL), "Invalid Operator subclass.");
   if (parallel)
   {
      fes = p_nlf->FESpace();
      energy_in = p_nlf->GetEnergy(d_in);
   }
#endif
   const bool serial = !parallel;
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(!(serial && nlf == NULL), "Invalid Operator subclass.");
   if (serial)
   {
      fes = nlf->FESpace();
      energy_in = nlf->GetEnergy(d_in);
   }

   // Get the local prolongation of the solution vector.
   Vector d_loc(fes->GetVSize(), (temp_mt == MemoryType::DEFAULT) ?
                /* */            Device::GetDeviceMemoryType() : temp_mt);
   if (serial)
   {
      const SparseMatrix *cP = fes->GetConformingProlongation();
      if (!cP) { d_loc = d_in; }
      else     { cP->Mult(d_in, d_loc); }
   }
#ifdef MFEM_USE_MPI
   else
   {
      fes->GetProlongationMatrix()->Mult(d_in, d_loc);
   }
#endif

   real_t scale = 1.0;
   bool fitting = IsSurfaceFittingEnabled();
   real_t init_fit_avg_err, init_fit_max_err = 0.0;
   if (fitting && surf_fit_converge_error)
   {
      GetSurfaceFittingError(d_loc, init_fit_avg_err, init_fit_max_err);
      // Check for convergence
      if (init_fit_max_err < surf_fit_max_err_limit)
      {
         if (print_options.iterations || print_options.warnings)
         {
            mfem::out << "TMOPNewtonSolver converged "
                      "based on the surface fitting error.\n";
         }
         scale = 0.0;
         return scale;
      }
   }

   if (surf_fit_adapt_count >= surf_fit_adapt_count_limit)
   {
      if (print_options.iterations)
      {
         mfem::out << "TMOPNewtonSolver terminated "
                   "based on max number of times surface fitting weight can"
                   "be increased. \n";
      }
      scale = 0.0;
      return scale;
   }

   // Check if the starting mesh (given by x) is inverted. Note that x hasn't
   // been modified by the Newton update yet.
   const real_t min_detT_in = ComputeMinDet(d_loc, *fes);
   const bool untangling = (min_detT_in <= 0.0) ? true : false;
   const real_t untangle_factor = 1.5;
   if (untangling)
   {
      // Needed for the line search below. The untangling metrics see this
      // reference to detect deteriorations.
      MFEM_VERIFY(min_det_ptr != NULL, " Initial mesh was valid, but"
                  " intermediate mesh is invalid. Contact TMOP Developers.");
      MFEM_VERIFY(min_detJ_limit == 0.0,
                  "This setup is not supported. Contact TMOP Developers.");
      *min_det_ptr = untangle_factor * min_detT_in;
   }

   const bool have_b = (b.Size() == Height());

   Vector d_out(d_in.Size());
   bool x_out_ok = false;
   real_t energy_out = 0.0, min_detT_out;
   const real_t norm_in = Norm(r);
   real_t avg_fit_err, max_fit_err = 0.0;

   const real_t detJ_factor = (solver_type == 1) ? 0.25 : 0.5;
   compute_metric_quantile_flag = false;
   // TODO:
   // - Customized line search for worst-quality optimization.
   // - What is the Newton exit criterion for worst-quality optimization?

   // Perform the line search.
   for (int i = 0; i < 12; i++)
   {
      avg_fit_err = 0.0;
      max_fit_err = 0.0;

      //
      // Update the mesh and get the L-vector in x_out_loc.
      //
      // Form limited (line-search) displacement d_out = d_in - scale * c,
      // and the corresponding mesh positions x_out = x_0 + d_out.
      add(d_in, -scale, c, d_out);
      if (serial)
      {
         const SparseMatrix *cP = fes->GetConformingProlongation();
         if (!cP) { d_loc = d_out; }
         else     { cP->Mult(d_out, d_loc); }
      }
#ifdef MFEM_USE_MPI
      else { fes->GetProlongationMatrix()->Mult(d_out, d_loc); }
#endif

      // Check the changes in detJ.
      min_detT_out = ComputeMinDet(d_loc, *fes);
      if (untangling == false && min_detT_out <= min_detJ_limit)
      {
         // No untangling, and detJ got negative (or small) -- no good.
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Neg det(J) found.\n";
         }
         scale *= detJ_factor; continue;
      }
      if (untangling == true && min_detT_out < *min_det_ptr)
      {
         // Untangling, and detJ got even more negative -- no good.
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Neg det(J) decreased.\n";
         }
         scale *= detJ_factor; continue;
      }

      // Skip the energy and residual checks when we're untangling. The
      // untangling metrics change their denominators, which can affect the
      // energy and residual, so their increase/decrease is not relevant.
      if (untangling) { x_out_ok = true; break; }

      // Update mesh-dependent quantities.
      ProcessNewState(d_out);

      // Ensure sufficient decrease in fitting error if we are trying to
      // converge based on error.
      if (fitting && surf_fit_converge_error)
      {
         GetSurfaceFittingError(d_loc, avg_fit_err, max_fit_err);
         if (max_fit_err >= 1.2*init_fit_max_err)
         {
            if (print_options.iterations)
            {
               mfem::out << "Scale = " << scale << " Surf fit err increased.\n";
            }
            scale *= 0.5; continue;
         }
      }

      // Check the changes in total energy.
      if (serial)
      {
         energy_out = nlf->GetEnergy(d_out);
      }
#ifdef MFEM_USE_MPI
      else
      {
         energy_out = p_nlf->GetEnergy(d_out);
      }
#endif
      if (energy_out > energy_in + 0.2*fabs(energy_in) ||
          std::isnan(energy_out) != 0)
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Increasing energy: "
                      << energy_in << " --> " << energy_out << '\n';
         }
         scale *= 0.5; continue;
      }

      // Check the changes in the Newton residual.
      oper->Mult(d_out, r);
      if (have_b) { r -= b; }
      real_t norm_out = Norm(r);

      if (norm_out > 1.2*norm_in)
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Norm increased: "
                      << norm_in << " --> " << norm_out << '\n';
         }
         scale *= 0.5; continue;
      }
      else { x_out_ok = true; break; }
   } // end line search

   if (untangling)
   {
      // Update the global min detJ. Untangling metrics see this min_det_ptr.
      if (min_detT_out > 0.0)
      {
         *min_det_ptr = 0.0;
         if (print_options.summary || print_options.iterations ||
             print_options.first_and_last)
         { mfem::out << "The mesh has been untangled at the used points!\n"; }
      }
      else { *min_det_ptr = untangle_factor * min_detT_out; }
   }

   if (print_options.summary || print_options.iterations ||
       print_options.first_and_last)
   {
      if (untangling)
      {
         mfem::out << "Min det(T) change: "
                   << min_detT_in << " -> " << min_detT_out
                   << " with " << scale << " scaling.\n";
      }
      else
      {
         mfem::out << "Energy decrease: "
                   << energy_in << " --> " << energy_out << " or "
                   << (energy_in - energy_out) / energy_in * 100.0
                   << "% with " << scale << " scaling.\n";
      }
   }

   if (x_out_ok == false) { scale = 0.0; }

   if (surf_fit_scale_factor > 0.0) { surf_fit_coeff_update = true; }
   compute_metric_quantile_flag = true;

   return scale;
}

void TMOPNewtonSolver::Mult(const Vector &b, Vector &x) const
{
   // Prolongate x to ldofs.
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   auto fes_mesh_nodes = nlf->FESpace()->GetMesh()->GetNodes()->FESpace();
   const Operator *P = fes_mesh_nodes->GetProlongationMatrix();
   x_0.SetSpace(fes_mesh_nodes);
   periodic = fes_mesh_nodes->IsDGSpace();
   if (P)
   {
      MFEM_VERIFY(x.Size() == P->Width(),
                  "The input's size must be the tdof size of the mesh nodes.");
      P->Mult(x, x_0);
   }
   else
   {
      MFEM_VERIFY(x.Size() == x_0.Size(),
                  "The input's size must match the size of the mesh nodes.");
      x_0 = x;
   }

   // Pass down the initial position to the integrators.
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   for (int i = 0; i < integs.Size(); i++)
   {
      auto ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti) { ti->SetInitialMeshPos(&x_0); }
      auto co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co) { co->SetInitialMeshPos(&x_0); }
   }

   // Solve for the displacement, which always starts from zero.
   Vector dx(height); dx = 0.0;
   if (solver_type == 0)      { NewtonSolver::Mult(b, dx); }
   else if (solver_type == 1) { LBFGSSolver::Mult(b, dx); }
   else { MFEM_ABORT("Invalid solver_type"); }

   // Form the final mesh using the computed displacement.
   if (periodic)
   {
      Vector dx_loc(nlf->FESpace()->GetVSize());
      const Operator *Pd = nlf->FESpace()->GetProlongationMatrix();
      if (Pd) { Pd->Mult(dx, dx_loc); }
      else    { dx_loc = dx; }

      GetPeriodicPositions(x_0, dx_loc, *fes_mesh_nodes, *nlf->FESpace(), x);
   }
   else { x += dx; }

   // Make sure the pointers don't use invalid memory (x_0_loc is gone).
   for (int i = 0; i < integs.Size(); i++)
   {
      auto ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti) { ti->SetInitialMeshPos(nullptr); }
      auto co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co) { co->SetInitialMeshPos(nullptr); }
   }
}

void TMOPNewtonSolver::UpdateSurfaceFittingWeight(real_t factor) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->UpdateSurfaceFittingWeight(factor);
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->UpdateSurfaceFittingWeight(factor);
         }
      }
   }
}

void TMOPNewtonSolver::GetSurfaceFittingWeight(Array<real_t> &weights) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   weights.SetSize(0);
   real_t weight;

   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti && ti->IsSurfaceFittingEnabled())
      {
         weight = ti->GetSurfaceFittingWeight();
         weights.Append(weight);
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            if (ati[j]->IsSurfaceFittingEnabled())
            {
               weight = ati[j]->GetSurfaceFittingWeight();
               weights.Append(weight);
            }
         }
      }
   }
}

void TMOPNewtonSolver::GetSurfaceFittingError(const Vector &d_loc,
                                              real_t &err_avg,
                                              real_t &err_max) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;

   err_avg = 0.0;
   err_max = 0.0;
   real_t err_avg_loc, err_max_loc;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         if (ti->IsSurfaceFittingEnabled())
         {
            ti->GetSurfaceFittingErrors(d_loc, err_avg_loc, err_max_loc);
            err_avg = std::max(err_avg_loc, err_avg);
            err_max = std::max(err_max_loc, err_max);
         }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            if (ati[j]->IsSurfaceFittingEnabled())
            {
               ati[j]->GetSurfaceFittingErrors(d_loc, err_avg_loc, err_max_loc);
               err_avg = std::max(err_avg_loc, err_avg);
               err_max = std::max(err_max_loc, err_max);
            }
         }
      }
   }
}

bool TMOPNewtonSolver::IsSurfaceFittingEnabled() const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;

   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         if (ti->IsSurfaceFittingEnabled())
         {
            return true;
         }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            if (ati[j]->IsSurfaceFittingEnabled())
            {
               return true;
            }
         }
      }
   }
   return false;
}

void TMOPNewtonSolver::ProcessNewState(const Vector &dx) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();

   // Reset the update flags of all TargetConstructors. This is done to avoid
   // repeated updates of shared TargetConstructors.
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->ResetUpdateFlags(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->ResetUpdateFlags(); }
         }
      }
   }

   Vector dx_loc;
   const Operator *P = nlf->GetProlongation();
   if (P)
   {
      dx_loc.SetSize(P->Height());
      P->Mult(dx, dx_loc);
   }
   else { dx_loc = dx; }

   const FiniteElementSpace *dx_fes = nlf->FESpace();
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->UpdateAfterMeshPositionChange(dx_loc, *dx_fes);
         if (compute_metric_quantile_flag)
         {
            ti->ComputeUntangleMetricQuantiles(dx_loc, *dx_fes);
         }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->UpdateAfterMeshPositionChange(dx_loc, *dx_fes);
            if (compute_metric_quantile_flag)
            {
               ati[j]->ComputeUntangleMetricQuantiles(dx_loc, *dx_fes);
            }
         }
      }
   }

   // Constant coefficient associated with the surface fitting terms if
   // adaptive surface fitting is enabled. The idea is to increase the
   // coefficient if the surface fitting error does not sufficiently
   // decrease between subsequent TMOPNewtonSolver iterations.
   if (surf_fit_coeff_update)
   {
      // Get surface fitting errors.
      GetSurfaceFittingError(dx_loc, surf_fit_avg_err, surf_fit_max_err);
      // Get array with surface fitting weights.
      Array<real_t> fitweights;
      GetSurfaceFittingWeight(fitweights);

      if (print_options.iterations)
      {
         mfem::out << "Avg/Max surface fitting error: " <<
                   surf_fit_avg_err << " " <<
                   surf_fit_max_err << "\n";
         mfem::out << "Min/Max surface fitting weight: " <<
                   fitweights.Min() << " " << fitweights.Max() << "\n";
      }

      real_t change_surf_fit_err = surf_fit_avg_err_prvs-surf_fit_avg_err;
      real_t rel_change_surf_fit_err = change_surf_fit_err/surf_fit_avg_err_prvs;

      // Increase the surface fitting coefficient if the surface fitting error
      // does not decrease sufficiently. If we are converging based on residual,
      // also make sure we have not reached the maximum fitting weight and
      // error threshold.
      if (rel_change_surf_fit_err < surf_fit_err_rel_change_limit &&
          (surf_fit_converge_error ||
           (fitweights.Max() < surf_fit_weight_limit &&
            surf_fit_max_err > surf_fit_max_err_limit)))
      {
         real_t scale_factor = std::min(surf_fit_scale_factor,
                                        surf_fit_weight_limit/fitweights.Max());
         UpdateSurfaceFittingWeight(scale_factor);
         surf_fit_adapt_count += 1;
      }
      else
      {
         surf_fit_adapt_count = 0;
      }
      surf_fit_avg_err_prvs = surf_fit_avg_err;
      surf_fit_coeff_update = false;
   }
}

real_t TMOPNewtonSolver::ComputeMinDet(const Vector &d_loc,
                                       const FiniteElementSpace &fes) const
{
   real_t min_detJ = infinity();
   const int NE = fes.GetNE(), dim = fes.GetMesh()->Dimension();
   Array<int> xdofs;
   DenseMatrix Jpr(dim);
   const bool mixed_mesh = fes.GetMesh()->GetNumGeometries(dim) > 1;
   if (dim == 1 || mixed_mesh ||
       UsesTensorBasis(fes) == false || fes.IsVariableOrder())
   {
      for (int i = 0; i < NE; i++)
      {
         const int dof = fes.GetFE(i)->GetDof();
         DenseMatrix dshape(dof, dim), pos(dof, dim);
         Vector posV(pos.Data(), dof * dim);

         x_0.GetElementDofValues(i, posV);
         if (periodic)
         {
            auto n_el = dynamic_cast<const NodalFiniteElement *>(fes.GetFE(i));
            n_el->ReorderLexToNative(dim, posV);
         }

         Vector d_loc_el;
         fes.GetElementVDofs(i, xdofs);
         d_loc.GetSubVector(xdofs, d_loc_el);
         posV += d_loc_el;

         const IntegrationRule &irule = GetIntegrationRule(*fes.GetFE(i));
         const int nsp = irule.GetNPoints();
         for (int j = 0; j < nsp; j++)
         {
            fes.GetFE(i)->CalcDShape(irule.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jpr);
            min_detJ = std::min(min_detJ, Jpr.Det());
         }
      }
   }
   else
   {
      min_detJ = dim == 2 ? MinDetJpr_2D(&fes, d_loc) :
                 dim == 3 ? MinDetJpr_3D(&fes, d_loc) : 0.0;
   }
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
      MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, p_nlf->ParFESpace()->GetComm());
   }
#endif
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(fes.GetMesh()->GetTypicalElementGeometry());
   min_detJ /= Wideal.Det();

   return min_detJ;
}

#ifdef MFEM_USE_MPI
// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_p(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, ParMesh &pmesh,
                       char *title, int position)
{
   L2_FECollection fec(order, pmesh.Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   ParGridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, pmesh, metric);
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   metric.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '"<< title << "'\n"
           << "window_geometry "
           << position << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA\n";
   }
}
#endif

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_s(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, Mesh &mesh,
                       char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA\n";
}

void GetPeriodicPositions(const Vector &x_0, const Vector &dx,
                          const FiniteElementSpace &fesL2,
                          const FiniteElementSpace &fesH1, Vector &x)
{
   x = x_0;
   Vector dx_r(x.Size());
   const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
   auto R_H1 = fesH1.GetElementRestriction(ord);
   auto R_L2 = fesL2.GetElementRestriction(ord);
   R_H1->Mult(dx, dx_r);
   R_L2->AddMultTranspose(dx_r, x);
}

}
