// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace navier;

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst)
{
   Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   {
      dst->AddDomainIntegrator((*bffis)[i]);
   }
}

NavierSolver::NavierSolver(ParMesh *mesh, int order, double kin_vis)
   : pmesh(mesh), order(order), kin_vis(kin_vis)
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

   Lext_gf.SetSpace(vfes);
   curlu_gf.SetSpace(vfes);
   curlcurlu_gf.SetSpace(vfes);
   FText_gf.SetSpace(vfes);
   resu_gf.SetSpace(vfes);

   pn_gf.SetSpace(pfes);
   pn_gf = 0.0;
   resp_gf.SetSpace(pfes);

   cur_step = 0;

   PrintInfo();
}

void NavierSolver::Setup(double dt)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
      if (partial_assembly)
      {
         mfem::out << "Using Partial Assembly" << std::endl;
      }
      else
      {
         mfem::out << "Using Full Assembly" << std::endl;
      }
   }

   sw_setup.Start();

   pmesh_lor = new ParMesh(pmesh, order, BasisType::GaussLobatto);
   pfec_lor = new H1_FECollection(1);
   pfes_lor = new ParFiniteElementSpace(pmesh_lor, pfec_lor);

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);

   Array<int> empty;

   // GLL integration rule (Numerical Integration)
   IntegrationRules rules_ni(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir_ni = rules_ni.Get(vfes->GetFE(0)->GetGeomType(),
                                               2 * order - 1);

   nlcoeff.constant = -1.0;
   N = new ParNonlinearForm(vfes);
   N->AddDomainIntegrator(new VectorConvectionNLFIntegrator(nlcoeff));
   if (partial_assembly)
   {
      N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      N->Setup();
   }

   Mv_form = new ParBilinearForm(vfes);
   BilinearFormIntegrator *mv_blfi = new VectorMassIntegrator;
   if (numerical_integ)
   {
      mv_blfi->SetIntRule(&ir_ni);
   }
   Mv_form->AddDomainIntegrator(mv_blfi);
   if (partial_assembly)
   {
      Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);

   Sp_form = new ParBilinearForm(pfes);
   BilinearFormIntegrator *sp_blfi = new DiffusionIntegrator;
   if (numerical_integ)
   {
      // blfi->SetIntRule(&ir_ni);
   }
   Sp_form->AddDomainIntegrator(sp_blfi);
   if (partial_assembly)
   {
      Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);

   D_form = new ParMixedBilinearForm(vfes, pfes);
   D_form->AddDomainIntegrator(new VectorDivergenceIntegrator);
   if (partial_assembly)
   {
      D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, D);

   G_form = new ParMixedBilinearForm(pfes, vfes);
   G_form->AddDomainIntegrator(new GradientIntegrator);
   if (partial_assembly)
   {
      G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, G);

   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);
   H_form->AddDomainIntegrator(new VectorMassIntegrator(H_bdfcoeff));
   H_form->AddDomainIntegrator(new VectorDiffusionIntegrator(H_lincoeff));
   if (partial_assembly)
   {
      H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(pfes);
   FText_bdr_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                            *FText_gfcoeff),
                                         vel_ess_attr);

   g_bdr_form = new ParLinearForm(pfes);
   for (auto &vel_dbc : vel_dbcs)
   {
      g_bdr_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                           *vel_dbc.coeff),
                                        vel_dbc.attr);
   }

   f_form = new ParLinearForm(vfes);
   for (auto &accel_term : accel_terms)
   {
      auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
      // @TODO: This order should always be the same as the nonlinear forms one!
      // const IntegrationRule &ir = IntRules.Get(vfes->GetFE(0)->GetGeomType(),
      //                                          4 * order);
      // vdlfi->SetIntRule(&ir);
      f_form->AddDomainIntegrator(vdlfi);
   }

   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      Mv_form->AssembleDiagonal(diag_pa);
      MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MvInv = new CGSolver(MPI_COMM_WORLD);
   MvInv->iterative_mode = false;
   MvInv->SetOperator(*Mv);
   MvInv->SetPreconditioner(*MvInvPC);
   MvInv->SetPrintLevel(pl_mvsolve);
   MvInv->SetRelTol(1e-12);
   MvInv->SetMaxIter(200);

   if (partial_assembly)
   {
      Sp_form_lor = new ParBilinearForm(pfes_lor);
      Sp_form_lor->UseExternalIntegrators();
      CopyDBFIntegrators(Sp_form, Sp_form_lor);
      Sp_form_lor->Assemble();
      Sp_form_lor->FormSystemMatrix(pres_ess_tdof, Sp_lor);
      SpInvPC = new HypreBoomerAMG(*Sp_lor.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(pl_amg);
      SpInvPC->Mult(resp, pn);
      SpInvOrthoPC = new OrthoSolver();
      SpInvOrthoPC->SetOperator(*SpInvPC);
   }
   else
   {
      SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(0);
      SpInvOrthoPC = new OrthoSolver();
      SpInvOrthoPC->SetOperator(*SpInvPC);
   }
   SpInv = new CGSolver(MPI_COMM_WORLD);
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

   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      H_form->AssembleDiagonal(diag_pa);
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HInv = new CGSolver(MPI_COMM_WORLD);
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(200);

   un_gf.GetTrueDofs(un);

   if (filter_alpha != 0.0)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes,
                                        pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh,
                                              vfec_filter,
                                              pmesh->Dimension());

      un_NM1_gf.SetSpace(vfes_filter);
      un_NM1_gf = 0.0;

      un_filtered_gf.SetSpace(vfes);
      un_filtered_gf = 0.0;
   }

   sw_setup.Stop();
}

void NavierSolver::Step(double &time, double dt, int cur_step)
{
   sw_step.Start();

   time += dt;

   // Set current time for velocity dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs)
   {
      vel_dbc.coeff->SetTime(time);
   }

   // Set current time for pressure dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs)
   {
      pres_dbc.coeff->SetTime(time);
   }

   SetTimeIntegrationCoefficients(cur_step);

   if (cur_step <= 2)
   {
      H_bdfcoeff.constant = bd0 / dt;
      H_form->Update();
      H_form->Assemble();
      H_form->FormSystemMatrix(vel_ess_tdof, H);

      if (partial_assembly)
      {
         HInv->SetOperator(*H);
         delete HInvPC;
         Vector diag_pa(vfes->GetTrueVSize());
         H_form->AssembleDiagonal(diag_pa);
         HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
         HInv->SetPreconditioner(*HInvPC);
      }
      else
      {
         HInv->SetOperator(*H);
      }
   }

   // Extrapolated f^{n+1}.
   for (auto &accel_term : accel_terms)
   {
      accel_term.coeff->SetTime(time);
   }

   f_form->Assemble();
   f_form->ParallelAssemble(fn);

   // Nonlinear extrapolated terms.
   sw_extrap.Start();

   N->Mult(un, Nun);
   Nun.Add(1.0, fn);

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

   // Rotate the solutions from previous time steps.
   Nunm2 = Nunm1;
   Nunm1 = Nun;

   // Fext = M^{-1} (F(u^{n}) + f^{n+1})
   MvInv->Mult(Fext, tmp1);
   iter_mvsolve = MvInv->GetNumIterations();
   res_mvsolve = MvInv->GetFinalNorm();
   Fext.Set(1.0, tmp1);

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
      auto d_Lext = Lext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      MFEM_FORALL(i, Lext.Size(),
                  d_Lext[i] = ab1_ * d_un[i] +
                              ab2_ * d_unm1[i] +
                              ab3_ * d_unm2[i];);
   }

   Lext_gf.SetFromTrueDofs(Lext);
   if (pmesh->Dimension() == 2)
   {
      ComputeCurl2D(Lext_gf, curlu_gf);
      ComputeCurl2D(curlu_gf, curlcurlu_gf, true);
   }
   else
   {
      ComputeCurl3D(Lext_gf, curlu_gf);
      ComputeCurl3D(curlu_gf, curlcurlu_gf);
   }

   curlcurlu_gf.GetTrueDofs(Lext);
   Lext *= kin_vis;

   sw_curlcurl.Stop();

   // \tilde{F} = F - \nu CurlCurl(u)
   FText.Set(-1.0, Lext);
   FText.Add(1.0, Fext);

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
   if (partial_assembly)
   {
      auto *SpC = Sp.As<ConstrainedOperator>();
      EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
   }
   else
   {
      Sp_form->FormLinearSystem(pres_ess_tdof, pn_gf, resp_gf, Sp, X1, B1, 1);
   }
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
      MeanZero(pn_gf);
   }

   pn_gf.GetTrueDofs(pn);

   // Project velocity.
   G->Mult(pn, resu);
   resu.Neg();
   Mv->Mult(Fext, tmp1);
   resu.Add(1.0, tmp1);

   for (auto &vel_dbc : vel_dbcs)
   {
      un_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

   // Rotate solutions from previous time steps.
   unm2 = unm1;
   unm1 = un;

   Vector X2, B2;
   if (partial_assembly)
   {
      auto *HC = H.As<ConstrainedOperator>();
      EliminateRHS(*H_form, *HC, vel_ess_tdof, un_gf, resu_gf, X2, B2, 1);
   }
   else
   {
      H_form->FormLinearSystem(vel_ess_tdof, un_gf, resu_gf, H, X2, B2, 1);
   }
   sw_hsolve.Start();
   HInv->Mult(B2, X2);
   sw_hsolve.Stop();
   iter_hsolve = HInv->GetNumIterations();
   res_hsolve = HInv->GetFinalNorm();
   H_form->RecoverFEMSolution(X2, resu_gf, un_gf);

   un_gf.GetTrueDofs(un);

   if (filter_alpha != 0.0)
   {
      un_NM1_gf.ProjectGridFunction(un_gf);
      un_filtered_gf.ProjectGridFunction(un_NM1_gf);
      const auto d_un_filtered_gf = un_filtered_gf.Read();
      auto d_un_gf = un_gf.ReadWrite();
      MFEM_FORALL(i,
                  un_gf.Size(),
                  d_un_gf[i] = (1.0 - filter_alpha) * d_un_gf[i]
                               + filter_alpha * d_un_filtered_gf[i];);
   }

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      // If numerical integration is active, there is no solve (thus no
      // iterations), on the inverse velocity mass application.
      if (!numerical_integ)
      {
         mfem::out << std::setw(5) << "MVIN " << std::setw(5) << std::fixed
                   << iter_mvsolve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_mvsolve
                   << "   " << 1e-12 << "\n";
      }
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

void NavierSolver::MeanZero(ParGridFunction &v)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(v.ParFESpace());
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(v.ParFESpace());
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   }

   double integ = mass_lf->operator()(v);

   v -= integ / volume;
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

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}

void NavierSolver::ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         curl.SetSize(3);
         curl(0) = grad(2, 1) - grad(1, 2);
         curl(1) = grad(0, 2) - grad(2, 0);
         curl(2) = grad(1, 0) - grad(0, 1);

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

void NavierSolver::ComputeCurl2D(ParGridFunction &u,
                                 ParGridFunction &cu,
                                 bool assume_scalar)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         if (assume_scalar)
         {
            curl.SetSize(2);
            curl(0) = grad(0, 1);
            curl(1) = -grad(0, 0);
         }
         else
         {
            curl.SetSize(2);
            curl(0) = grad(1, 0) - grad(0, 1);
            curl(1) = 0.0;
         }

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication.

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

double NavierSolver::ComputeCFL(ParGridFunction &u, double dt)
{
   ParMesh *pmesh = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh->GetElementSize(e, 1) / (double) fes->GetOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
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
                 pmesh->GetComm());

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
   if (step == 0)
   {
      bd0 = 1.0;
      bd1 = -1.0;
      bd2 = 0.0;
      bd3 = 0.0;
      ab1 = 1.0;
      ab2 = 0.0;
      ab3 = 0.0;
   }
   else if (step == 1)
   {
      bd0 = 3.0 / 2.0;
      bd1 = -4.0 / 2.0;
      bd2 = 1.0 / 2.0;
      bd3 = 0.0;
      ab1 = 2.0;
      ab2 = -1.0;
      ab3 = 0.0;
   }
   else if (step == 2)
   {
      bd0 = 11.0 / 6.0;
      bd1 = -18.0 / 6.0;
      bd2 = 9.0 / 6.0;
      bd3 = -2.0 / 6.0;
      ab1 = 3.0;
      ab2 = -3.0;
      ab3 = 1.0;
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
   int fes_size0 = vfes->GlobalVSize();
   int fes_size1 = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
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
   delete MvInvPC;
   delete Sp_form_lor;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete f_form;
   delete pfes_lor;
   delete pfec_lor;
   delete pmesh_lor;
   delete MvInv;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete vfec_filter;
   delete vfes_filter;
}
