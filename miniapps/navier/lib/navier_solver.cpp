// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "navier_solver.hpp"
#include "kernels/stress_integrator.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace navier;

static inline void vmul(const Vector &x, Vector &y)
{
   auto d_x = x.Read();
   auto d_y = y.ReadWrite();
   MFEM_FORALL(i, x.Size(), d_y[i] = d_x[i] * d_y[i];);
}

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst)
{
   Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   {
      dst->AddDomainIntegrator((*bffis)[i]);
   }
}

NavierSolver::NavierSolver(ParMesh *mesh, int order, double kin_vis)
   : pmesh(mesh), order(order), kin_vis(kin_vis),
     gll_rules(0, Quadrature1D::GaussLobatto)
{
   vfec = new H1_FECollection(order, pmesh->Dimension());
   pfec = new H1_FECollection(order);
   vfes = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
   pfes = new ParFiniteElementSpace(pmesh, pfec);

   // Check if fully periodic mesh
   if (!(pmesh->bdr_attributes.Size() == 0))
   {
      vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr = 0;

      pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      pres_ess_attr = 0;
   }

   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();

   un.SetSize(vfes_truevsize);
   un = 0.0;
   un_next.SetSize(vfes_truevsize);
   un_next = 0.0;
   unm1.SetSize(vfes_truevsize);
   unm1 = 0.0;
   unm2.SetSize(vfes_truevsize);
   unm2 = 0.0;
   fn.SetSize(vfes_truevsize);
   Nun.SetSize(vfes_truevsize);
   Nun = 0.0;
   Nunm1.SetSize(vfes_truevsize);
   Nunm1 = 0.0;
   Nunm2.SetSize(vfes_truevsize);
   Nunm2 = 0.0;
   u_ext.SetSize(vfes_truevsize);
   Fext.SetSize(vfes_truevsize);
   FText.SetSize(vfes_truevsize);
   Lext.SetSize(vfes_truevsize);
   resu.SetSize(vfes_truevsize);

   tmp1.SetSize(vfes_truevsize);

   pn.SetSize(pfes_truevsize);
   pn = 0.0;
   resp.SetSize(pfes_truevsize);
   resp = 0.0;
   FText_bdr.SetSize(pfes_truevsize);
   g_bdr.SetSize(pfes_truevsize);

   un_gf.SetSpace(vfes);
   un_gf = 0.0;
   un_next_gf.SetSpace(vfes);
   un_next_gf = 0.0;

   Lext_gf.SetSpace(vfes);
   curlu_gf.SetSpace(vfes);
   curlcurlu_gf.SetSpace(vfes);
   FText_gf.SetSpace(vfes);
   resu_gf.SetSpace(vfes);

   pn_gf.SetSpace(pfes);
   pn_gf = 0.0;
   resp_gf.SetSpace(pfes);

   mv.SetSize(vfes_truevsize);
   mvinv.SetSize(vfes_truevsize);

   kin_vis_gf.SetSpace(pfes);
   ConstantCoefficient kvcoeff_tmp(kin_vis);
   kin_vis_gf.ProjectCoefficient(kvcoeff_tmp);
   kv.SetSize(pfes_truevsize);
   grad_nu_sym_grad_uext.SetSize(vfes_truevsize);

   hpfrt_tdofs.SetSize(vfes_truevsize);
   hpfrt_tdofs = 0.0;

   cur_step = 0;

   PrintInfo();
}

void NavierSolver::Setup(double dt)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
   }

   sw_setup.Start();

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);

   Array<int> empty;

   gll_ir = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order - 1);
   gll_ir_face = gll_rules.Get(vfes->GetMesh()->GetFaceGeometry(0), 2 * order - 1);
   gll_ir_nl = gll_rules.Get(vfes->GetFE(0)->GetGeomType(),
                             (int)(ceil(1.5 * (2 * order - 1))));

   mean_evaluator = new MeanEvaluator(*pfes, gll_ir);
   curl_evaluator = new CurlEvaluator(*vfes);
   stress_evaluator = new StressEvaluator(*kin_vis_gf.ParFESpace(),
                                          *un_gf.ParFESpace(),
                                          gll_ir);

   nlcoeff.constant = -1.0;
   N = new ParNonlinearForm(vfes);
   auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff);
   nlc_nlfi->SetIntRule(&gll_ir_nl);
   N->AddDomainIntegrator(nlc_nlfi);
   N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   N->Setup();

   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   mv_blfi->SetIntRule(&gll_ir);
   Mv_form->AddDomainIntegrator(mv_blfi);
   Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mv_form->Assemble();

   Sp_form = new ParBilinearForm(pfes);
   auto *sp_blfi = new DiffusionIntegrator;
   sp_blfi->SetIntRule(&gll_ir);
   Sp_form->AddDomainIntegrator(sp_blfi);
   Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);

   D_form = new ParMixedBilinearForm(vfes, pfes);
   auto *vd_mblfi = new VectorDivergenceIntegrator();
   vd_mblfi->SetIntRule(&gll_ir);
   D_form->AddDomainIntegrator(vd_mblfi);
   D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, D);

   G_form = new ParMixedBilinearForm(pfes, vfes);
   auto *g_mblfi = new GradientIntegrator();
   g_mblfi->SetIntRule(&gll_ir);
   G_form->AddDomainIntegrator(g_mblfi);
   G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, G);

   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);

   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff);
   hmv_blfi->SetIntRule(&gll_ir);

   BilinearFormIntegrator *hdv_blfi = nullptr;
   kin_vis_gf_coeff.SetGridFunction(&kin_vis_gf);
   hdv_blfi = new StressIntegrator(kin_vis_gf_coeff, gll_ir);
   hdv_blfi->SetIntRule(&gll_ir);
   H_form->AddDomainIntegrator(hmv_blfi);
   H_form->AddDomainIntegrator(hdv_blfi);
   H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   HMv_form = new ParBilinearForm(vfes);
   auto *hmv_blfi2 = new VectorMassIntegrator(H_bdfcoeff);
   hmv_blfi2->SetIntRule(&gll_ir);
   HMv_form->AddDomainIntegrator(hmv_blfi2);
   HMv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   HMv_form->Assemble();

   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(pfes);
   auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
   ftext_bnlfi->SetIntRule(&gll_ir_face);
   FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);
   FText_bdr_form->UseFastAssembly(true);

   g_bdr_form = new ParLinearForm(pfes);
   for (auto &vel_dbc : vel_dbcs)
   {
      auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
      gbdr_bnlfi->SetIntRule(&gll_ir_face);
      g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
   }
   g_bdr_form->UseFastAssembly(true);

   f_form = new ParLinearForm(vfes);
   for (auto &accel_term : accel_terms)
   {
      auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
      // @TODO: This order should always be the same as the nonlinear forms one!
      // const IntegrationRule &ir = IntRules.Get(vfes->GetFE(0)->GetGeomType(),
      //                                          4 * order);
      // vdlfi->SetIntRule(&ir);
      vdlfi->SetIntRule(&gll_ir);
      f_form->AddDomainIntegrator(vdlfi);
   }

   // Build diagonal vector from velocity Mass operator
   Mv_form->AssembleDiagonal(mv);

   // Inverse mass operator is the inverse of the diagonal
   {
      const auto d_mv = mv.Read();
      auto d_mvinv = mvinv.Write();
      MFEM_FORALL(i, mv.Size(), d_mvinv[i] = 1.0 / d_mv[i];);
   }

   lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
   SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
   SpInvPC->SetPrintLevel(pl_amg);
   SpInvPC->Mult(resp, pn);
   SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
   SpInvOrthoPC->SetSolver(*SpInvPC);
   SpInv = new CGSolver(vfes->GetComm());
   SpInv->iterative_mode = true;
   SpInv->SetOperator(*Sp);
   if (pres_dbcs.empty())
   {
      SpInv->SetPreconditioner(*SpInvOrthoPC);
   }
   else
   {
      SpInv->SetPreconditioner(*SpInvPC);
   }
   SpInv->SetPrintLevel(pl_spsolve);
   SpInv->SetRelTol(rtol_spsolve);
   SpInv->SetMaxIter(200);

   HInv = new CGSolver(vfes->GetComm());
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   // HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(300);

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   un_gf.GetTrueDofs(un);
   un_next = un;
   un_next_gf.SetFromTrueDofs(un_next);

   // Set initial time step in the history array
   dthist[0] = dt;

   if (filter_method != FilterMethod::NONE)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes,
                                        pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh,
                                              vfec_filter,
                                              pmesh->Dimension());

      u_filter_basis_gf.SetSpace(vfes_filter);
      u_filter_basis_gf = 0.0;

      u_low_modes_gf.SetSpace(vfes);
      u_low_modes_gf = 0.0;

      hpfrt_gf.SetSpace(vfes);
      hpfrt_gf = 0.0;
   }

   sw_setup.Stop();
}

void NavierSolver::UpdateTimestepHistory(double dt)
{
   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;

   // Rotate values in nonlinear extrapolation history
   Nunm2 = Nunm1;
   Nunm1 = Nun;

   // Rotate values in solution history
   unm2 = unm1;
   unm1 = un;

   // Update the current solution and corresponding GridFunction
   un_next_gf.GetTrueDofs(un_next);
   un = un_next;
   un_gf.SetFromTrueDofs(un);
}

void NavierSolver::Step(double &time, double dt, int current_step,
                        bool provisional)
{
   // RANS
   if (rans_model)
   {
      // kv = rans_model->EvaluateTo(time + dt);
      // kv_gf -= kv;
   }

   sw_step.Start();

   SetTimeIntegrationCoefficients(current_step);

   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs)
   {
      vel_dbc.coeff->SetTime(time + dt);
   }

   // Set current time for pressure Dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs)
   {
      pres_dbc.coeff->SetTime(time + dt);
   }

   H_bdfcoeff.constant = bd0 / dt;
   H_form->Update();
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   HInv->SetOperator(*H);
   delete HInvPC;
   Vector diag_pa(vfes->GetTrueVSize());
   HMv_form->AssembleDiagonal(diag_pa);
   HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   HInv->SetPreconditioner(*HInvPC);

   // Extrapolated f^{n+1}.
   for (auto &accel_term : accel_terms)
   {
      accel_term.coeff->SetTime(time + dt);
   }

   f_form->Assemble();
   f_form->ParallelAssemble(fn);

   // Nonlinear extrapolated terms.
   sw_extrap.Start();

   N->Mult(un, Nun);
   N->Mult(unm1, Nunm1);
   N->Mult(unm2, Nunm2);

   {
      const auto d_Nun = Nun.Read();
      const auto d_Nunm1 = Nunm1.Read();
      const auto d_Nunm2 = Nunm2.Read();
      auto d_Fext = Fext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      MFEM_FORALL(i, Fext.Size(),
                  d_Fext[i] = ab1_ * d_Nun[i] +
                              ab2_ * d_Nunm1[i] +
                              ab3_ * d_Nunm2[i];);
   }

   Fext.Add(1.0, fn);

   if (filter_method == FilterMethod::HPFRT_LOS)
   {
      u_filter_basis_gf.ProjectGridFunction(un_gf);
      u_low_modes_gf.ProjectGridFunction(u_filter_basis_gf);
      const auto d_un_gf = un_gf.Read();
      const auto d_u_low_modes_gf = u_low_modes_gf.Read();
      auto d_hpfrt_gf = hpfrt_gf.ReadWrite();
      MFEM_FORALL(i,
                  un_gf.Size(),
                  d_hpfrt_gf[i] = -10.0 * (d_un_gf[i] - d_u_low_modes_gf[i]););

      hpfrt_gf.ParallelProject(hpfrt_tdofs);
   }
   else if (filter_method == FilterMethod::HPFRT_LPOLY)
   {
      BuildHPFForcing(un_gf);
   }

   // Fext = M^{-1} (F(u^{n}) + f^{n+1} + RT^{n+1})
   if (filter_method != FilterMethod::NONE)
   {
      Fext.Add(1.0, hpfrt_tdofs);
   }
   vmul(mvinv, Fext);

   // Compute BDF terms.
   {
      const double bd1idt = -bd1 / dt;
      const double bd2idt = -bd2 / dt;
      const double bd3idt = -bd3 / dt;
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Fext = Fext.ReadWrite();
      MFEM_FORALL(i, Fext.Size(),
                  d_Fext[i] += bd1idt * d_un[i] +
                               bd2idt * d_unm1[i] +
                               bd3idt * d_unm2[i];);
   }

   sw_extrap.Stop();

   // Pressure Poisson.
   sw_curlcurl.Start();
   {
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_u_ext = u_ext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      MFEM_FORALL(i, u_ext.Size(),
                  d_u_ext[i] = ab1_ * d_un[i] +
                               ab2_ * d_unm1[i] +
                               ab3_ * d_unm2[i];);
   }

   Lext_gf.SetFromTrueDofs(u_ext);

   curl_evaluator->ComputeCurlCurl(u_ext, tmp1);
   curlcurlu_gf.SetFromTrueDofs(tmp1);

   // (curl curl u)_i *= nu
   for (int d = 0; d < curlcurlu_gf.FESpace()->GetVDim(); d++)
   {
      for (int i = 0; i < curlcurlu_gf.FESpace()->GetNDofs(); i++)
      {
         int idx = Ordering::Map<Ordering::byNODES>(
                      curlcurlu_gf.FESpace()->GetNDofs(),
                      curlcurlu_gf.FESpace()->GetVDim(),
                      i,
                      d);
         curlcurlu_gf(idx) *= kin_vis_gf(i);
      }
   }
   curlcurlu_gf.GetTrueDofs(Lext);

   sw_curlcurl.Stop();

   // \tilde{F} = F - \nu CurlCurl(u)
   FText.Set(-1.0, Lext);
   FText.Add(1.0, Fext);

   kin_vis_gf.ParallelAssemble(kv);
   stress_evaluator->Apply(kv, u_ext, grad_nu_sym_grad_uext);

   FText.Add(1.0, grad_nu_sym_grad_uext);

   // p_r = \nabla \cdot FText
   D->Mult(FText, resp);
   resp.Neg();

   // Add boundary terms.
   FText_gf.SetFromTrueDofs(FText);
   FText_bdr_form->Assemble();
   FText_bdr_form->ParallelAssemble(FText_bdr);

   g_bdr_form->Assemble();
   g_bdr_form->ParallelAssemble(g_bdr);
   resp.Add(1.0, FText_bdr);
   resp.Add(-bd0 / dt, g_bdr);

   if (pres_dbcs.empty())
   {
      Orthogonalize(resp);
   }

   for (auto &pres_dbc : pres_dbcs)
   {
      pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
   }

   pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

   Vector X1, B1;
   auto *SpC = Sp.As<ConstrainedOperator>();
   EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
   sw_spsolve.Start();
   SpInv->Mult(B1, X1);
   sw_spsolve.Stop();
   iter_spsolve = SpInv->GetNumIterations();
   res_spsolve = SpInv->GetFinalNorm();
   Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

   // If the boundary conditions on the pressure are pure Neumann remove the
   // nullspace by removing the mean of the pressure solution. This is also
   // ensured by the OrthoSolver wrapper for the preconditioner which removes
   // the nullspace after every application.
   if (pres_dbcs.empty())
   {
      mean_evaluator->MakeMeanZero(pn);
   }

   pn_gf.GetTrueDofs(pn);

   // Project velocity.
   G->Mult(pn, resu);
   resu.Neg();
   vmul(mv, Fext);
   resu.Add(1.0, Fext);

   // un_next_gf = un_gf;

   for (auto &vel_dbc : vel_dbcs)
   {
      un_next_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

   Vector X2, B2;
   auto *HC = H.As<ConstrainedOperator>();
   EliminateRHS(*H_form, *HC, vel_ess_tdof, un_next_gf, resu_gf, X2, B2, 1);
   sw_hsolve.Start();
   HInv->Mult(B2, X2);
   sw_hsolve.Stop();
   iter_hsolve = HInv->GetNumIterations();
   res_hsolve = HInv->GetFinalNorm();
   H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);

   un_next_gf.GetTrueDofs(un_next);

   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      mfem::out << std::setw(5) << "PRES " << std::setw(5) << std::fixed
                << iter_spsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_spsolve << "   " << rtol_spsolve
                << "\n";
      mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed
                << iter_hsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_hsolve << "   " << rtol_hsolve
                << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }
}

int NavierSolver::PredictTimestep(double dt_min, double dt_max,
                                  double cfl_target, double &dt)
{
   int flag = 0;
   double previous_dt = dt;
   double cflmax = 1.2 * cfl_target;
   double cflmin = 0.8 * cfl_target;

   double vel_max = ComputeCFL(un_next_gf, 1.0);
   double cfl = dt * vel_max;
   double vel_cfl_old = vel_cfl;
   vel_cfl = vel_max;

   // first timestep, initialize values
   if (dthist[1] == 0.0)
   {
      cfl_old = cfl;
   }

   double cflpred = 2.0*cfl-cfl_old;

   if (cfl > cflmax || cflpred > cflmax || cfl < cflmin)
   {
      double A = (vel_cfl - vel_cfl_old) / dt;
      double B = vel_cfl;
      double C = -cfl_target;
      double discr = (B*B) - 4.0 * A * C;

      if (discr <= 0.0)
      {
         dt = dt * (cfl_target/cfl);
      }
      else if (std::fabs(vel_cfl - vel_cfl_old)/vel_cfl < 0.001)
      {
         dt = dt * (cfl_target/cfl);
      }
      else
      {
         double dtlow = (-B+sqrt(discr) )/(2.0*A);
         double dthi = (-B-sqrt(discr) )/(2.0*A);
         if (dthi > 0.0 && dtlow > 0.0)
         {
            dt = std::min(dthi, dtlow);
         }
         else if (dthi <= 0.0 && dtlow <= 0.0)
         {
            dt = dt * (cfl_target/cfl);
         }
         else
         {
            dt = std::max(dthi, dtlow);
         }
      }

      if (previous_dt/dt < 0.2)
      {
         dt = previous_dt * 5.0;
      }
   }

   if (Mpi::Root())
   {
      if (dt < dt_min)
      {
         MFEM_ABORT("Minimum timestep reached, likely unstable.");
      }
   }

   if (cfl > cflmax || cflpred > cflmax)
   {
      flag = -1;
   }

   dt = std::min(1.2*previous_dt, dt);
   dt = std::max(0.8*previous_dt, dt);
   dt = std::min(dt_max, dt);
   cfl_old = cfl;

   return flag;
}

void NavierSolver::MeanZero(ParGridFunction &v)
{
   Vector tvec;
   v.GetTrueDofs(tvec);
   mean_evaluator->MakeMeanZero(tvec);
   v.Distribute(tvec);
}

void NavierSolver::EliminateRHS(Operator &A,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A.GetOutputProlongation();
   const Operator *Pi = A.GetProlongation();
   const Operator *Ri = A.GetRestriction();
   A.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_tdof_list, 0.0);
   }
   constrainedA.EliminateRHS(X, B);
}

void NavierSolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pfes->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}

double NavierSolver::ComputeCFL(ParGridFunction &u, double dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   // The integration rule here has to conform with the nonlinear term.
   auto ir = gll_ir;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);

         if (vdim == 3)
         {
            cflz = fabs(dt * ut(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}

void NavierSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((vel_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition deteceted.");
      if (attr[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void NavierSolver::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr)
{
   pres_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Pressure Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((pres_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition deteceted.");
      if (attr[i] == 1)
      {
         pres_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(f), attr);
}

void NavierSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierSolver::AddAccelTerm(VecFuncT *f, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void NavierSolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step + 1, max_bdf_order);

   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   double rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   double rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];

   if (bdf_order == 3)
   {
      rho2 = dthist[1] / dthist[2];
   }

   if (step == 0 && bdf_order == 1)
   {
      bd0 = 1.0;
      bd1 = -1.0;
      bd2 = 0.0;
      bd3 = 0.0;
      ab1 = 1.0;
      ab2 = 0.0;
      ab3 = 0.0;
   }
   else if (step >= 1 && bdf_order == 2)
   {
      bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
      bd1 = -(1.0 + rho1);
      bd2 = pow(rho1, 2.0) / (1.0 + rho1);
      bd3 = 0.0;
      ab1 = 1.0 + rho1;
      ab2 = -rho1;
      ab3 = 0.0;
   }
   else if (step >= 2 && bdf_order == 3)
   {
      bd0 = 1.0 + rho1 / (1.0 + rho1)
            + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
      bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
      bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
      ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
      ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
      ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
   }
}

void NavierSolver::PrintTimingData()
{
   double my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_step.RealTime();
   my_rt[2] = sw_extrap.RealTime();
   my_rt[3] = sw_curlcurl.RealTime();
   my_rt[4] = sw_spsolve.RealTime();
   my_rt[5] = sw_hsolve.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP"
                << std::setw(10) << "EXTRAP" << std::setw(10) << "CURLCURL"
                << std::setw(10) << "PSOLVE" << std::setw(10) << "HSOLVE"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4]
                << std::setw(10) << my_rt[5] << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10)
                << my_rt[1] / my_rt[1] << std::setw(10) << my_rt[2] / my_rt[1]
                << std::setw(10) << my_rt[3] / my_rt[1] << std::setw(10)
                << my_rt[4] / my_rt[1] << std::setw(10) << my_rt[5] / my_rt[1]
                << "\n";

      mfem::out << std::setprecision(8);
   }
}

void NavierSolver::PrintInfo()
{
   int mesh_ne = vfes->GetParMesh()->GetGlobalNE();
   int fes_size0 = vfes->GlobalTrueVSize();
   int fes_size1 = pfes->GlobalTrueVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Mesh #EL: " << mesh_ne << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}

void NavierSolver::EvaluateLegendrePolynomial(const int N, const double x,
                                              Vector &L)
{
   L(0) = 1.0;
   L(1) = x;

   for (int i = 2; i < N; i++)
   {
      L(i) = ((2*i-1) * x * L(i-1) - (i-1) * L(i-2)) / (double)i;
   }
}

void NavierSolver::EvaluateLegendrePolynomialShifted(const int N,
                                                     const double x,
                                                     Vector &L)
{
   EvaluateLegendrePolynomial(N, 2.0 * x - 1.0, L);
}

void NavierSolver::EvaluateMonomialBasis(const int N, const double x, Vector &L)
{
   for (int i = 0; i < N; i++)
   {
      L(i) = std::pow(x, i);
   }
}

void NavierSolver::SetRANSModel(std::shared_ptr<RANSModel> k)
{
   rans_model = k;

   // rans_model->Setup(*pfes);
}

void NavierSolver::EnableFilter(FilterMethod f)
{
   filter_method = f;
}

void NavierSolver::BuildHPFForcing(ParGridFunction &vel_gf)
{
   const int hpf_kco = filter_cutoff_modes;
   const double hpf_weight = filter_alpha;
   const double hpf_chi = -1.0 * hpf_weight;
   const int N = order + 1;

   DenseMatrix D_filter;
   D_filter.Diag(1.0, N);

   const int k0 = N - hpf_kco;
   // Amplitude
   double a;

   for (int k = k0; k < N; k++)
   {
      a = (pow((double)(k+1 - k0), 2.0))/pow((double)(hpf_kco), 2.0);
      D_filter(k, k) = 1.0 - a;
   }

   // Legendre Basis interpolation matrix
   DenseMatrix V(N), Vinv(N);
   Vector Vi(N);

   for (int i = 0; i < N; i++)
   {
      EvaluateLegendrePolynomialShifted(N, gll_ir.IntPoint(i).x, Vi);
      V.SetRow(i, Vi);
   }

   DenseMatrix DfVinv(N), F(N), Ft(N);
   Vinv = V;
   Vinv.Invert();

   // Construct F = V * D * V^-1
   Mult(D_filter, Vinv, DfVinv);
   Mult(V, DfVinv, F);

   Ft.Transpose(F);

   // compute u - F u F'
   Array<int> vdofs, dofs;
   Vector u_e, w1;
   auto nodal_fe = dynamic_cast<const NodalFiniteElement*>(vfes->GetFE(0));
   if (!nodal_fe)
   {
      MFEM_ABORT("internal error");
   }
   auto dofs_lex_ordering = nodal_fe->GetLexicographicOrdering();
   int ndofs = nodal_fe->GetDof();
   dofs.SetSize(ndofs);
   int dim = nodal_fe->GetDim();
   DenseMatrix W1, W2(N);

   // hpfrt E-vector
   auto h1v_element_restriction = vfes->GetElementRestriction(
                                     ElementDofOrdering::LEXICOGRAPHIC);
   Vector hpfrt_e(h1v_element_restriction->Height());
   Vector hpfrt_edual(h1v_element_restriction->Height());

   for (int e = 0; e < vfes->GetNE(); ++e)
   {
      Array <int> vdofs;
      vfes->GetElementVDofs(e, vdofs);
      for (int d = 0; d < dim; d++)
      {
         // Retrieve input field in single dimension d
         for (int lex_idx = 0; lex_idx < ndofs; lex_idx++)
         {
            int scalar_h1_idx = dofs_lex_ordering[lex_idx];
            int vector_h1_idx = scalar_h1_idx + ndofs * d;
            dofs[lex_idx] = vdofs[vector_h1_idx];
         }

         vel_gf.GetSubVector(dofs, u_e);

         // Copy 1D input field
         w1 = u_e;
         // Make 1D input field available as a matrix. Assumes that the 1D
         // field is ordered the same as the quadrature rule.
         W1.UseExternalData(w1.GetData(), N, N);
         // W2 = F * W1
         Mult(F, W1, W2);
         // W1 = W2 * F'
         Mult(W2, Ft, W1);
         // In vector form, perform u_e = u_e - w1
         subtract(u_e, w1, u_e);
         // Set the dofs of the high-pass filtered GridFunction in the E-vector
         hpfrt_e.SetVector(u_e, ndofs * (e * dim + d));
      }
   }
   auto mv_integ = (*Mv_form->GetDBFI())[0];
   hpfrt_edual = 0.0;
   mv_integ->AddMultPA(hpfrt_e, hpfrt_edual);
   h1v_element_restriction->MultTranspose(hpfrt_edual, hpfrt_gf);

   hpfrt_gf.ParallelAssemble(hpfrt_tdofs);
   hpfrt_tdofs *= hpf_chi;
}

NavierSolver::~NavierSolver()
{
   delete FText_gfcoeff;
   delete g_bdr_form;
   delete FText_bdr_form;
   delete mass_lf;
   delete Mv_form;
   delete N;
   delete Sp_form;
   delete D_form;
   delete G_form;
   delete HInvPC;
   delete HInv;
   delete H_form;
   delete SpInv;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete lor;
   delete f_form;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete vfec_filter;
   delete vfes_filter;
   delete HMv_form;
   delete stress_evaluator;
}
