#include "flow_solver.hpp"
#include "../../general/forall.hpp"
#include <fstream>

using namespace mfem;
using namespace flow;

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst)
{
   Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   {
      dst->AddDomainIntegrator((*bffis)[i]);
   }
}

FlowSolver::FlowSolver(ParMesh *mesh, int order, double kin_vis)
   : pmesh(mesh), order(order), kin_vis(kin_vis), vfec(nullptr), pfec(nullptr),
     vfes(nullptr), pfes(nullptr)
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
   resp.SetSize(pfes_truevsize);
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

void FlowSolver::Setup(double dt)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      std::cout << "Setup" << std::endl;
      if (partial_assembly)
      {
         std::cout << "Using Partial Assembly" << std::endl;
      }
      else
      {
         std::cout << "Using FULL Assembly" << std::endl;
      }
   }

   sw_setup.Start();

   pmesh_lor = new ParMesh(pmesh, order, BasisType::GaussLobatto);
   // vfec_lor = new H1_FECollection(1, pmesh->Dimension());
   pfec_lor = new H1_FECollection(1);
   // vfes_lor = new ParFiniteElementSpace(pmesh_lor, vfec_lor, pmesh->Dimension());
   pfes_lor = new ParFiniteElementSpace(pmesh_lor, pfec_lor);

   // vgt = new InterpolationGridTransfer(*vfes, *vfes_lor);
   pgt = new InterpolationGridTransfer(*pfes, *pfes_lor);

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

   // TODO has to be changed for multiple boundary attribute definitions!
   // Assuming we only set one function for dirichlet on the whole boundary.
   // FText_bdr_form has to be set only on the attributes where there are velocity dirichlet bcs.
   // Maybe use vel_ess_attr?
   // Needs github PR #936 https://github.com/mfem/mfem/pull/936
   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(pfes);
   FText_bdr_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                            *FText_gfcoeff),
                                         vel_ess_attr);

   g_bdr_form = new ParLinearForm(pfes);
   for (auto vdbc = vel_dbcs.begin(); vdbc != vel_dbcs.end(); ++vdbc)
   {
      g_bdr_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                           vdbc->coeff),
                                        vdbc->attr);
   }

   if (partial_assembly)
   {
      // Mv_form_lor = new ParBilinearForm(vfes_lor);
      // CopyDBFIntegrators(Mv_form, Mv_form_lor);
      // Mv_form_lor->Assemble();
      // Mv_form_lor->FormSystemMatrix(empty, Mv_lor);
      // MvInvPC = new HypreSmoother(*Mv_lor.As<HypreParMatrix>());
      // static_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
      Vector diag_pa(vfes->GetTrueVSize());
      Mv_form->AssembleDiagonal(diag_pa);
      MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
      static_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
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
      Sp_form_lor->SetExternBFS(1);
      CopyDBFIntegrators(Sp_form, Sp_form_lor);
      Sp_form_lor->Assemble();
      Sp_form_lor->FormSystemMatrix(pres_ess_tdof, Sp_lor);
      SpInvPC = new HypreBoomerAMG(*Sp_lor.As<HypreParMatrix>());
      HYPRE_Solver amg_precond = static_cast<HYPRE_Solver>(*SpInvPC);
      HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
      HYPRE_BoomerAMGSetRelaxType(amg_precond, 3);
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
      // H_form_lor = new ParBilinearForm(vfes_lor);
      // CopyDBFIntegrators(H_form, H_form_lor);
      // H_form_lor->Assemble();
      // H_form_lor->FormSystemMatrix(empty, H_lor);
      // HInvPC = new HypreSmoother(*H_lor.As<HypreParMatrix>());
      // static_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
      Vector diag_pa(vfes->GetTrueVSize());
      H_form->AssembleDiagonal(diag_pa);
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>());
      static_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HInv = new CGSolver(MPI_COMM_WORLD);
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(200);

   if (0)
   {
      // Timing test
      StopWatch t0, t1;
      int numit = 10;
      for (int i = 0; i < numit; ++i)
      {
         t0.Start();
         SpInvPC->Mult(resp, pn);
         t0.Stop();

         t1.Start();
         Sp->Mult(resp, pn);
         t1.Stop();
      }
      std::cout << "PREC MULT: " << t0.RealTime() / numit << std::endl;
      std::cout << "OP MULT: " << t1.RealTime() / numit << std::endl;
   }

   un_gf.GetTrueDofs(un);

   sw_setup.Stop();
}

void FlowSolver::Step(double &time, double dt, int cur_step)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      std::cout << "Step" << std::endl;
   }
   sw_step.Start();

   time += dt;

   for (auto vdbc = vel_dbcs.begin(); vdbc != vel_dbcs.end(); ++vdbc)
   {
      vdbc->coeff.SetTime(time);
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
         // H_form_lor->Update();
         // H_form_lor->Assemble();
         // H_form_lor->FormSystemMatrix(vel_ess_tdof, H_lor);
         HInv->ClearPreconditioner();
         HInv->SetOperator(*H);
         delete HInvPC;
         // HInvPC = new HypreSmoother(*H_lor.As<HypreParMatrix>());
         // static_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
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

   // Extrapolated f^{n+1}
   // forcing_coeff->SetTime(t - dt);
   // f_form->Assemble();
   // f_form->ParallelAssemble(fn);
   // forcing_coeff->SetTime(t);

   fn = 0.0;

   //
   // Nonlinear EXT terms
   //

   sw_extrap.Start();

   N->Mult(un, Nun);
   Nun.Add(1.0, fn);

   // TEST
   // Fext.Set(ab1, Nun);
   // Fext.Add(ab2, Nunm1);
   // Fext.Add(ab3, Nunm2);
   MFEM_FORALL(i,
               Fext.Size(),
               Fext[i] = ab1 * Nun[i] + ab2 * Nunm1[i] + ab3 * Nunm2[i];);

   Nunm2 = Nunm1;
   Nunm1 = Nun;

   // Fext = M^{-1} (F(u^{n}) + f^{n+1})
   MvInv->Mult(Fext, tmp1);
   Fext.Set(1.0, tmp1);

   // BDF terms
   // Fext.Add(-bd1 / dt, un);
   // Fext.Add(-bd2 / dt, unm1);
   // Fext.Add(-bd3 / dt, unm2);
   double bd1idt = -bd1 / dt;
   double bd2idt = -bd2 / dt;
   double bd3idt = -bd3 / dt;
   MFEM_FORALL(i,
               Fext.Size(),
               Fext[i] += bd1idt * un[i] + bd2idt * unm1[i] + bd3idt * unm2[i];);

   sw_extrap.Stop();

   //
   // Pressure poisson
   //

   sw_curlcurl.Start();

   // Lext.Set(ab1, un);
   // Lext.Add(ab2, unm1);
   // Lext.Add(ab3, unm2);
   MFEM_FORALL(i,
               Lext.Size(),
               Lext[i] = ab1 * un[i] + ab2 * unm1[i] + ab3 * unm2[i];);

   Lext_gf.SetFromTrueDofs(Lext);
   // ComputeCurlCurl(Lext_gf, curlcurlu_gf);
   if (pmesh->Dimension() == 2)
   {
     ComputeCurl2D(Lext_gf, curlu_gf);
     ComputeCurl2D(curlu_gf, curlcurlu_gf, true);
   }
   else 
   {
     MFEM_ABORT("3D Curl not implemented");
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

   // Add boundary terms
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

   // pn_gf = 0.0;

   pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

   Vector X1, B1;
   if (partial_assembly)
   {
      ConstrainedOperator *SpC = Sp.As<ConstrainedOperator>();
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

   if (pres_dbcs.empty())
   {
      MeanZero(pn_gf);
   }

   pn_gf.GetTrueDofs(pn);

   //
   // Project velocity
   //

   G->Mult(pn, resu);
   resu.Neg();
   Mv->Mult(Fext, tmp1);
   resu.Add(1.0, tmp1);

   // un_gf = 0.0;

   for (auto vdbc = vel_dbcs.begin(); vdbc != vel_dbcs.end(); ++vdbc)
   {
      un_gf.ProjectBdrCoefficient(vdbc->coeff, vdbc->attr);
   }

   vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

   unm2 = unm1;
   unm1 = un;

   Vector X2, B2;
   if (partial_assembly)
   {
      ConstrainedOperator *HC = H.As<ConstrainedOperator>();
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

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      printf("PRES %d %.2E %.2E\n", iter_spsolve, res_spsolve, rtol_spsolve);
      printf("HELM %d %.2E %.2E\n", iter_hsolve, res_hsolve, rtol_hsolve);
   }
}

void FlowSolver::MeanZero(ParGridFunction &v)
{
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

void FlowSolver::EliminateRHS(Operator &A,
                              ConstrainedOperator &constrainedA,
                              const Array<int> &ess_tdof_list,
                              Vector &x,
                              Vector &b,
                              Vector &X,
                              Vector &B,
                              int copy_interior)
{
   const Operator *P = A.GetProlongation();
   const Operator *R = A.GetRestriction();
   A.InitTVectors(P, R, x, b, X, B);
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_tdof_list, 0.0);
   }
   constrainedA.EliminateRHS(X, B);
}

void FlowSolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}

void FlowSolver::ComputeCurlCurl(ParGridFunction &u, ParGridFunction &ccu)
{
   ParGridFunction cu(u.ParFESpace());
   CurlGridFunctionCoefficient cu_gfcoeff(&u);

   cu.ProjectDiscCoefficient(cu_gfcoeff, GridFunction::AvgType::ARITHMETIC);

   if (u.ParFESpace()->GetVDim() == 2)
   {
      for (int i = 0; i < u.ParFESpace()->GetNDofs(); i++)
      {
         cu[cu.ParFESpace()->DofToVDof(i, 1)] = 0.0;
      }
   }

   cu_gfcoeff.SetGridFunction(&cu);
   cu_gfcoeff.assume_scalar = true;
   ccu.ProjectDiscCoefficient(cu_gfcoeff, GridFunction::AvgType::ARITHMETIC);
}

void FlowSolver::ComputeCurl2D(ParGridFunction &u,
                                   ParGridFunction &cu,
                                   bool assume_scalar)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation
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
         // Project
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval
         // GetVectorGradientHat
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

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

void FlowSolver::AddVelDirichletBC(
   void (*f)(const Vector &x, double t, Vector &u), Array<int> &attr)
{
   vel_dbcs.push_back(
      VelDirichletBC_T(f,
                       attr,
                       VectorFunctionCoefficient(pmesh->Dimension(), f)));

   if (verbose && pmesh->GetMyRank() == 0)
   {
      out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            out << i << " ";
         }
      }
      out << std::endl;
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

void FlowSolver::AddPresDirichletBC(
   double (*f)(const Vector &x, double t), Array<int> &attr)
{
   pres_dbcs.push_back(
      PresDirichletBC_T(f,
                       attr,
                       FunctionCoefficient(f)));

   if (verbose && pmesh->GetMyRank() == 0)
   {
      out << "Adding Pressure Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            out << i << " ";
         }
      }
      out << std::endl;
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

void FlowSolver::SetTimeIntegrationCoefficients(int step)
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

void FlowSolver::PrintTimingData()
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
      printf("%10s %10s %10s %10s %10s %10s\n",
             "SETUP",
             "STEP",
             "EXTRAP",
             "CURLCURL",
             "PSOLVE",
             "HSOLVE");
      printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
             my_rt[0],
             my_rt[1],
             my_rt[2],
             my_rt[3],
             my_rt[4],
             my_rt[5]);
      printf("%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n",
             " ",
             my_rt[1] / my_rt[1],
             my_rt[2] / my_rt[1],
             my_rt[3] / my_rt[1],
             my_rt[4] / my_rt[1],
             my_rt[5] / my_rt[1]);
   }
}

void FlowSolver::PrintInfo()
{
   int fes_size0 = vfes->GlobalVSize();
   int fes_size1 = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      out << "FLOW version: "
          << "00000" << std::endl
          << "MFEM version: " << MFEM_VERSION << std::endl
          << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
          << "Velocity #DOFs: " << fes_size0 << std::endl
          << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}

FlowSolver::~FlowSolver()
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
   delete pfes_lor;
   delete pfec_lor;
   delete pgt;
   delete pmesh_lor;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete MvInv;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
}
