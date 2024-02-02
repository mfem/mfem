// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "extrapolator.hpp"
#include "../common/mfem-common.hpp"
#include "marking.hpp"

using namespace std;

namespace mfem
{

const char vishost[] = "localhost";
const int  visport   = 19916;
int wsize            = 350; // glvis window size

AdvectionOper::AdvectionOper(Array<bool> &zones, ParBilinearForm &Mbf,
                             ParBilinearForm &Kbf, const Vector &rhs)
   : TimeDependentOperator(Mbf.Size()),
     active_zones(zones),
     M(Mbf), K(Kbf), K_mat(NULL), b(rhs),
     lo_solver(NULL), lumpedM(NULL)
{
   K_mat = K.ParallelAssemble(&K.SpMat());

   ParBilinearForm M_Lump(M.ParFESpace());
   lumpedM = new Vector;
   M_Lump.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   M_Lump.Assemble();
   M_Lump.Finalize();
   M_Lump.SpMat().GetDiag(*lumpedM);
   lo_solver = new DiscreteUpwindLOSolver(*M.ParFESpace(),
                                          K.SpMat(), *lumpedM);
}

AdvectionOper::~AdvectionOper()
{
   delete lo_solver;
   delete lumpedM;
   delete K_mat;
}

void AdvectionOper::Mult(const Vector &x, Vector &dx) const
{
   ParFiniteElementSpace &pfes = *M.ParFESpace();
   const int NE = pfes.GetNE();
   const int nd = pfes.GetFE(0)->GetDof();
   Array<int> dofs(nd);

   if (adv_mode == LO)
   {
      lo_solver->CalcLOSolution(x, b, dx);
      for (int k = 0; k < NE; k++)
      {
         pfes.GetElementDofs(k, dofs);
         if (active_zones[k] == false)
         {
            dx.SetSubVector(dofs, 0.0);
            continue;
         }
      }
      return;
   }

   MFEM_VERIFY(adv_mode == HO, "Wrong input for advection mode (-dg).");

   Vector rhs(x.Size());
   K_mat->Mult(x, rhs);
   rhs += b;

   DenseMatrix M_loc(nd);
   DenseMatrixInverse M_loc_inv(&M_loc);
   Vector rhs_loc(nd), dx_loc(nd);
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);

      if (active_zones[k] == false)
      {
         dx.SetSubVector(dofs, 0.0);
         continue;
      }

      rhs.GetSubVector(dofs, rhs_loc);
      M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
      M_loc_inv.Factor();
      M_loc_inv.Mult(rhs_loc, dx_loc);
      dx.SetSubVector(dofs, dx_loc);
   }
}

void AdvectionOper::ComputeElementsMinMax(const ParGridFunction &gf,
                                          Vector &el_min, Vector &el_max) const
{
   ParFiniteElementSpace &pfes = *gf.ParFESpace();
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   for (int k = 0; k < NE; k++)
   {
      el_min(k) = numeric_limits<double>::infinity();
      el_max(k) = -numeric_limits<double>::infinity();

      for (int i = 0; i < ndof; i++)
      {
         el_min(k) = min(el_min(k), gf(k*ndof + i));
         el_max(k) = max(el_max(k), gf(k*ndof + i));
      }
   }
}

void AdvectionOper::ComputeBounds(const ParFiniteElementSpace &pfes,
                                  const Vector &el_min, const Vector &el_max,
                                  Vector &dof_min, Vector &dof_max) const
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction el_min_gf(&pfes_bounds), el_max_gf(&pfes_bounds);
   const int NE = pmesh->GetNE(), ndofs = dof_min.Size() / NE;

   el_min_gf = el_min;
   el_max_gf = el_max;

   el_min_gf.ExchangeFaceNbrData(); el_max_gf.ExchangeFaceNbrData();
   const Vector &min_nbr = el_min_gf.FaceNbrData();
   const Vector &max_nbr = el_max_gf.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int k = 0; k < NE; k++)
   {
      double k_min = el_min_gf(k), k_max = el_max_gf(k);

      el_to_el.GetRow(k, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            k_min = std::min(k_min, el_min_gf(face_nbr_el[n]));
            k_max = std::max(k_max, el_max_gf(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            k_min = std::min(k_min, min_nbr(face_nbr_el[n] - NE));
            k_max = std::max(k_max, max_nbr(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         dof_min(k*ndofs + j) = k_min;
         dof_max(k*ndofs + j) = k_max;
      }
   }
}

void Extrapolator::Extrapolate(Coefficient &level_set,
                               const ParGridFunction &input,
                               const double time_period,
                               ParGridFunction &xtrap)
{
   ParMesh &pmesh = *input.ParFESpace()->GetParMesh();
   const int order = input.ParFESpace()->GetOrder(0),
             dim   = pmesh.Dimension(), NE = pmesh.GetNE();

   // Get a ParGridFunction and mark elements.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes_H1(&pmesh, &fec);
   ParGridFunction ls_gf(&pfes_H1);
   ls_gf.ProjectCoefficient(level_set);
   if (visualization)
   {
      socketstream sock1, sock2;
      common::VisualizeField(sock1, vishost, visport, ls_gf,
                             "Domain level set", 0, 0, wsize, wsize,
                             "rRjlmm********A");
      common::VisualizeField(sock2, vishost, visport, input,
                             "Input u", 0, wsize+60, wsize, wsize,
                             "rRjlmm********A");
      MPI_Barrier(pmesh.GetComm());
   }
   // Mark elements.
   Array<int> elem_marker;
   ShiftedFaceMarker marker(pmesh, pfes_H1, false);
   ls_gf.ExchangeFaceNbrData();
   marker.MarkElements(ls_gf, elem_marker);

   // The active zones are where we extrapolate (where the PDE is solved).
   Array<bool> active_zones(NE);
   for (int k = 0; k < NE; k++)
   {
      // Extrapolation is done in zones that are CUT or OUTSIDE.
      active_zones[k] =
         (elem_marker[k] == ShiftedFaceMarker::INSIDE) ? false : true;
   }

   // Setup a VectorCoefficient for n = - grad_ls / |grad_ls|.
   // The sign makes it point out of the known region.
   // The coefficient must be continuous to have well-defined transport.
   LevelSetNormalGradCoeff ls_n_coeff_L2(ls_gf);
   ParFiniteElementSpace pfes_H1_vec(&pmesh, &fec, dim);
   ParGridFunction lsn_gf(&pfes_H1_vec);
   ls_gf.ExchangeFaceNbrData();
   lsn_gf.ProjectDiscCoefficient(ls_n_coeff_L2, GridFunction::ARITHMETIC);
   VectorGridFunctionCoefficient ls_n_coeff(&lsn_gf);

   // Initial solution.
   // Trim to the known values (only elements inside the known region).
   Array<int> dofs;
   L2_FECollection fec_L2(order, dim);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2);
   ParGridFunction u(&pfes_L2), vis_marking(&pfes_L2);
   u.ProjectGridFunction(input);
   for (int k = 0; k < NE; k++)
   {
      pfes_L2.GetElementDofs(k, dofs);
      if (elem_marker[k] != ShiftedFaceMarker::INSIDE)
      { u.SetSubVector(dofs, 0.0); }
      vis_marking.SetSubVector(dofs, elem_marker[k]);
   }
   if (visualization)
   {
      socketstream sock1, sock2;
      common::VisualizeField(sock1, vishost, visport, u,
                             "Fixed (known) u values", wsize, 0,
                             wsize, wsize, "rRjlmm********A");
      common::VisualizeField(sock2, vishost, visport, vis_marking,
                             "Element markings", 0, 2*wsize+60,
                             wsize, wsize, "rRjlmm********A");
   }

   // Normal derivative function.
   ParGridFunction n_grad_u(&pfes_L2);
   NormalGradCoeff n_grad_u_coeff(u, ls_n_coeff);
   n_grad_u.ProjectCoefficient(n_grad_u_coeff);
   if (visualization && xtrap_degree >= 1)
   {
      socketstream sock;
      common::VisualizeField(sock, vishost, visport, n_grad_u,
                             "n.grad(u)", 2*wsize, 0, wsize, wsize,
                             "rRjlmm********A");
   }

   // 2nd normal derivative function.
   ParGridFunction n_grad_n_grad_u(&pfes_L2);
   NormalGradCoeff n_grad_n_grad_u_coeff(n_grad_u, ls_n_coeff);
   n_grad_n_grad_u.ProjectCoefficient(n_grad_n_grad_u_coeff);
   if (visualization && xtrap_degree == 2)
   {
      socketstream sock;
      common::VisualizeField(sock, vishost, visport, n_grad_n_grad_u,
                             "n.grad(n.grad(u))", 3*wsize, 0, wsize, wsize,
                             "rRjmm********A");
   }

   ParBilinearForm lhs_bf(&pfes_L2), rhs_bf(&pfes_L2);
   lhs_bf.AddDomainIntegrator(new MassIntegrator);
   const double alpha = -1.0;
   rhs_bf.AddDomainIntegrator(new ConvectionIntegrator(ls_n_coeff, alpha));
   auto trace_i = new NonconservativeDGTraceIntegrator(ls_n_coeff, alpha);
   rhs_bf.AddInteriorFaceIntegrator(trace_i);
   rhs_bf.KeepNbrBlock(true);

   ls_gf.ExchangeFaceNbrData();
   lhs_bf.Assemble();
   lhs_bf.Finalize();
   rhs_bf.Assemble(0);
   rhs_bf.Finalize(0);

   // Compute a CFL time step.
   double h_min = std::numeric_limits<double>::infinity();
   for (int k = 0; k < NE; k++)
   {
      h_min = std::min(h_min, pmesh.GetElementSize(k));
   }
   MPI_Allreduce(MPI_IN_PLACE, &h_min, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   // The propagation speed is 1.
   double dt = 0.25 * h_min / order / 1.0;
   double half_dt = 0.5 * dt;
   if (advection_mode == AdvectionOper::LO)
   {
      dt = half_dt;
   }

   // Time loops.
   Vector rhs(pfes_L2.GetVSize());
   AdvectionOper adv_oper(active_zones, lhs_bf, rhs_bf, rhs);
   adv_oper.adv_mode = advection_mode;
   RK2Solver ode_solver(1.0);
   ode_solver.Init(adv_oper);

   if (xtrap_degree == 0)
   {
      // Constant extrapolation of u (always LO).
      rhs = 0.0;
      adv_oper.adv_mode = AdvectionOper::LO;
      TimeLoop(u, ode_solver, time_period, half_dt,
               wsize, "Extrap const u -- LO");
      xtrap.ProjectGridFunction(u);
      return;
   }

   std::string mode_text = "HO";
   if (advection_mode == AdvectionOper::LO)  { mode_text = "LO"; }

   MFEM_VERIFY(xtrap_degree == 1 || xtrap_degree == 2, "Wrong order input.");
   if (xtrap_type == ASLAM)
   {
      if (xtrap_degree == 1)
      {
         // Constant extrapolation of [n.grad_u] (always LO).
         rhs = 0.0;
         adv_oper.adv_mode = AdvectionOper::LO;
         TimeLoop(n_grad_u, ode_solver, time_period, half_dt,
                  2*wsize, "Extrap const n.grad(u) -- Aslam -- LO");

         adv_oper.adv_mode = advection_mode;

         // Linear extrapolation of u.
         lhs_bf.Mult(n_grad_u, rhs);
         TimeLoop(u, ode_solver, time_period, dt,
                  wsize, "Extrap linear u -- Aslam -- " + mode_text);
      }

      if (xtrap_degree == 2)
      {
         // Constant extrapolation of [n.grad(n.grad(u))] (always LO).
         rhs = 0.0;
         adv_oper.adv_mode = AdvectionOper::LO;
         TimeLoop(n_grad_n_grad_u, ode_solver, time_period, half_dt,
                  3*wsize, "Extrap const n.grad(n.grad(u)) -- Aslam -- LO");

         adv_oper.adv_mode = advection_mode;

         // Linear extrapolation of [n.grad_u].
         lhs_bf.Mult(n_grad_n_grad_u, rhs);
         TimeLoop(n_grad_u, ode_solver, time_period, dt,
                  2*wsize, "Extrap linear n.grad(u) -- Aslam -- " + mode_text);

         // Quadratic extrapolation of u.
         lhs_bf.Mult(n_grad_u, rhs);
         TimeLoop(u, ode_solver, time_period, dt,
                  wsize, "Extrap quadratic u -- Aslam -- " + mode_text);
      }
   }
   else if (xtrap_type == BOCHKOV)
   {
      if (xtrap_degree == 1)
      {
         // Constant extrapolation of all grad(u) components (always LO).
         rhs = 0.0;
         adv_oper.adv_mode = AdvectionOper::LO;
         ParGridFunction grad_u_0(&pfes_L2), grad_u_1(&pfes_L2);
         GradComponentCoeff grad_u_0_coeff(u, 0), grad_u_1_coeff(u, 1);
         grad_u_0.ProjectCoefficient(grad_u_0_coeff);
         grad_u_1.ProjectCoefficient(grad_u_1_coeff);
         TimeLoop(grad_u_0, ode_solver, time_period, half_dt,
                  2*wsize, "Extrap const du_dx -- Bochkov -- LO");
         TimeLoop(grad_u_1, ode_solver, time_period, half_dt,
                  3*wsize, "Extrap const du_dy -- Bochkov -- LO");

         adv_oper.adv_mode = advection_mode;

         // Linear extrapolation of u.
         ParLinearForm rhs_lf(&pfes_L2);
         NormalGradComponentCoeff grad_u_n(grad_u_0, grad_u_1, ls_n_coeff);
         rhs_lf.AddDomainIntegrator(new DomainLFIntegrator(grad_u_n));
         rhs_lf.Assemble();
         rhs = rhs_lf;
         TimeLoop(u, ode_solver, time_period, dt,
                  wsize, "Extrap linear u -- Bochkov -- " + mode_text);
      }

      if (xtrap_degree == 2)
      {
         MFEM_ABORT("Quadratic Bochkov method is not implemented.");
      }
   }
   else { MFEM_ABORT("Wrong input for extrapolation type (-et)."); }

   xtrap.ProjectGridFunction(u);
}

// Errors in cut elements.
void Extrapolator::ComputeLocalErrors(Coefficient &level_set,
                                      const ParGridFunction &exact,
                                      const ParGridFunction &xtrap,
                                      double &err_L1, double &err_L2,
                                      double &err_LI)
{
   ParMesh &pmesh = *exact.ParFESpace()->GetParMesh();
   const int order = exact.ParFESpace()->GetOrder(0),
             dim   = pmesh.Dimension(), NE = pmesh.GetNE();

   // Get a ParGridFunction and mark elements.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes_H1(&pmesh, &fec);
   ParGridFunction ls_gf(&pfes_H1);
   ls_gf.ProjectCoefficient(level_set);
   // Mark elements.
   Array<int> elem_marker;
   ShiftedFaceMarker marker(pmesh, pfes_H1, false);
   ls_gf.ExchangeFaceNbrData();
   marker.MarkElements(ls_gf, elem_marker);

   Vector errors_L1(NE), errors_L2(NE), errors_LI(NE);
   GridFunctionCoefficient exact_coeff(&exact);

   xtrap.ComputeElementL1Errors(exact_coeff, errors_L1);
   xtrap.ComputeElementL2Errors(exact_coeff, errors_L2);
   xtrap.ComputeElementMaxErrors(exact_coeff, errors_LI);
   err_L1 = 0.0, err_L2 = 0.0, err_LI = 0.0;
   double cut_volume = 0.0;
   for (int k = 0; k < NE; k++)
   {
      if (elem_marker[k] == ShiftedFaceMarker::CUT)
      {
         err_L1 += errors_L1(k);
         err_L2 += errors_L2(k);
         err_LI = std::max(err_LI, errors_LI(k));
         cut_volume += pmesh.GetElementVolume(k);
      }
   }
   MPI_Comm comm = pmesh.GetComm();
   MPI_Allreduce(MPI_IN_PLACE, &err_L1, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &err_L2, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &err_LI, 1, MPI_DOUBLE, MPI_MAX, comm);
   MPI_Allreduce(MPI_IN_PLACE, &cut_volume, 1, MPI_DOUBLE, MPI_SUM, comm);
   err_L1 /= cut_volume;
   err_L2 /= cut_volume;
}

void Extrapolator::TimeLoop(ParGridFunction &sltn, ODESolver &ode_solver,
                            double t_final, double dt,
                            int vis_x_pos, std::string vis_name)
{
   socketstream sock;

   const int myid  = sltn.ParFESpace()->GetMyRank();
   bool done = false;
   double t = 0.0;
   for (int ti = 0; !done;)
   {
      double dt_real = min(dt, t_final - t);
      ode_solver.Step(sltn, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << vis_name+" / time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            common::VisualizeField(sock, vishost, visport, sltn,
                                   vis_name.c_str(), vis_x_pos, wsize+60,
                                   wsize, wsize, "rRjlmm********A");
            MPI_Barrier(sltn.ParFESpace()->GetComm());
         }
      }
   }
}



DiscreteUpwindLOSolver::DiscreteUpwindLOSolver(ParFiniteElementSpace &space,
                                               const SparseMatrix &adv,
                                               const Vector &Mlump)
   : pfes(space), K(adv), D(adv), K_smap(), M_lumped(Mlump)
{
   // Assuming it is finalized.
   const int *I = K.GetI(), *J = K.GetJ(), n = K.Size();
   K_smap.SetSize(I[n]);
   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { K_smap[j] = _j; break; }
         }
      }
   }

   ComputeDiscreteUpwindMatrix();
}

void DiscreteUpwindLOSolver::CalcLOSolution(const Vector &u, const Vector &rhs,
                                            Vector &du) const
{
   ParGridFunction u_gf(&pfes);
   u_gf = u;
   ApplyDiscreteUpwindMatrix(u_gf, du);

   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      du(i) = (du(i) + rhs(i)) / M_lumped(i);
   }
}

void DiscreteUpwindLOSolver::ComputeDiscreteUpwindMatrix() const
{
   const int *I = K.HostReadI(), *J = K.HostReadJ(), n = K.Size();

   const double *K_data = K.HostReadData();

   double *D_data = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k];
         double kij = K_data[k];
         double kji = K_data[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         D_data[k] = kij + dij;
         D_data[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

void DiscreteUpwindLOSolver::ApplyDiscreteUpwindMatrix(ParGridFunction &u,
                                                       Vector &du) const
{
   const int s = u.Size();
   const int *I = D.HostReadI(), *J = D.HostReadJ();
   const double *D_data = D.HostReadData();

   u.ExchangeFaceNbrData();
   const Vector &u_np = u.FaceNbrData();

   for (int i = 0; i < s; i++)
   {
      du(i) = 0.0;
      for (int k = I[i]; k < I[i + 1]; k++)
      {
         int j = J[k];
         double u_j  = (j < s) ? u(j) : u_np[j - s];
         double d_ij = D_data[k];
         du(i) += d_ij * u_j;
      }
   }
}

}
