#include "navierstokes_operator.hpp"
#include "util.hpp"
#include "navier_annotation.hpp"

using namespace mfem;

NavierStokesOperator::NavierStokesOperator(ParFiniteElementSpace &vel_fes,
                                           ParFiniteElementSpace &pres_fes,
                                           std::vector<VelDirichletBC> &velocity_dbcs,
                                           std::vector<PresDirichletBC> &pressure_dbcs,
                                           ParGridFunction &nu_gf,
                                           bool convection,
                                           bool convection_explicit,
                                           bool matrix_free) :
   TimeDependentOperator(vel_fes.GetTrueVSize() + pres_fes.GetTrueVSize()),
   vel_fes(vel_fes),
   pres_fes(pres_fes),
   kinematic_viscosity(nu_gf),
   velocity_dbcs(velocity_dbcs),
   pressure_dbcs(pressure_dbcs),
   convection(convection),
   convection_explicit(convection_explicit),
   matrix_free(matrix_free),
   offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()}),
        intrules(0, Quadrature1D::GaussLobatto),
        zero_coeff(0.0)
{
   if (vel_fes.GetParMesh()->bdr_attributes.Size() > 0)
   {
      vel_ess_bdr.SetSize(vel_fes.GetParMesh()->bdr_attributes.Max());
      vel_ess_bdr = 0.0;
      pres_ess_bdr.SetSize(vel_fes.GetParMesh()->bdr_attributes.Max());
      pres_ess_bdr = 0.0;
   }

   for (auto &bc : velocity_dbcs)
   {
      for (int i = 0; i < bc.second->Size(); i++)
      {
         if ((*bc.second)[i] == 1)
         {
            vel_ess_bdr[i] = 1;
         }
      }
   }

   for (auto &bc : pressure_dbcs)
   {
      for (int i = 0; i < bc.second->Size(); i++)
      {
         if ((*bc.second)[i] == 1)
         {
            pres_ess_bdr[i] = 1;
         }
      }
   }

   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdofs);

   offsets.PartialSum();

   z.Update(offsets);
   w.Update(offsets);
   Y.Update(offsets);

   trans_newton_residual.reset(new TransientNewtonResidual(*this));

   vel_bc_gf.reset(new ParGridFunction(&vel_fes));
   *vel_bc_gf = 0.0;

   pres_bc_gf.reset(new ParGridFunction(&pres_fes));
   *pres_bc_gf = 0.0;

   // The nonlinear convective integrators use over-integration (dealiasing) as
   // a stabilization mechanism.
   ir_nl = intrules.Get(vel_fes.GetFE(0)->GetGeomType(),
                        (int)(ceil(1.5 * 2*(vel_fes.GetOrder(0)+1) - 3)));

   ir = intrules.Get(vel_fes.GetFE(0)->GetGeomType(),
                     (int)(2*(vel_fes.GetOrder(0)+1) - 3));

   ir_face = intrules.Get(vel_fes.GetFaceElement(0)->GetGeomType(),
                          (int)(2*(vel_fes.GetOrder(0)+1) - 3));

   pc.reset(new BlockLowerTriangularPreconditioner(offsets));
}

void NavierStokesOperator::MultImplicit(const Vector &xb, Vector &yb) const
{
   NAVIER_PERF_BEGIN("MultImplicit");

   const BlockVector x(xb.GetData(), offsets);
   BlockVector y(yb.GetData(), offsets);

   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   A->Mult(x, y);
   y.Neg();

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdofs, 0.0);

   NAVIER_PERF_END("MultImplicit");
}

void NavierStokesOperator::MultExplicit(const Vector &xb, Vector &yb) const
{
   NAVIER_PERF_BEGIN("NavierStokesOperator::MultExplicit");

   const BlockVector x(xb.GetData(), offsets);
   BlockVector y(yb.GetData(), offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   n_form->Mult(xu, yu);
   yu.Neg();

   if (fu_rhs.Size())
   {
      yu += fu_rhs;
   }

   yp = 0.0;
   yu.SetSubVector(vel_ess_tdofs, 0.0);

   NAVIER_PERF_END("NavierStokesOperator::MultExplicit");
}

void NavierStokesOperator::Mult(const Vector &xb, Vector &yb) const
{
   if (EvalMode::ADDITIVE_TERM_1)
   {
      MultExplicit(xb, yb);
   }
   else if (EvalMode::ADDITIVE_TERM_2)
   {
      MultImplicit(xb, yb);
   }
   else
   {
      MFEM_ABORT("NavierStokesOperator::Mult >> unknown EvalMode");
   }
}

void NavierStokesOperator::Solve(Vector &b, Vector &x)
{
   BlockVector xb(x.GetData(), offsets);
   Vector &xu = xb.GetBlock(0);
   Vector &xp = xb.GetBlock(1);

   BlockVector bb(b.GetData(), offsets);
   Vector &bu = bb.GetBlock(0);
   Vector &bp = bb.GetBlock(1);

   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-4);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(100);
   krylov.SetMaxIter(100);
   krylov.SetPreconditioner(*pc.get());
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-3);
   newton.SetAbsTol(1e-9);
   newton.SetMaxIter(1);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetOperator(*trans_newton_residual);
   newton.SetPreconditioner(krylov);

   bu.SetSubVector(vel_ess_tdofs, 0.0);
   bp = 0.0;
   ProjectVelocityDirichletBC(xu);
   ProjectPressureDirichletBC(xp);

   NAVIER_PERF_BEGIN("Newton");
   newton.Mult(b, x);
   NAVIER_PERF_END("Newton");
}

int NavierStokesOperator::SUNImplicitSetup(const Vector &x, const Vector &fx,
                                           int jok, int *jcur, double gamma)
{
   Setup(gamma);
   *jcur = 1;

   return 0;
}

int NavierStokesOperator::SUNImplicitSolve(const Vector &bb, Vector &xb,
                                           double tol)
{
   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(tol);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(100);
   krylov.SetMaxIter(100);
   krylov.SetPreconditioner(*pc.get());
   krylov.SetOperator(*Amonoe_matfree);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   krylov.Mult(bb, xb);

   return 0;
}

void NavierStokesOperator::MassMult(const Vector &x, Vector &y)
{
   const BlockVector xb(x.GetData(), offsets);
   BlockVector yb(y.GetData(), offsets);
   Mv->Mult(xb.GetBlock(0), yb.GetBlock(0));
   yb.GetBlock(1) = 0.0;
}

int NavierStokesOperator::SUNMassMult(const Vector &xb, Vector &yb)
{
   MassMult(xb, yb);
   return 0;
}

int NavierStokesOperator::SUNMassSolve(const Vector &bb, Vector &xb, double tol)
{
   const BlockVector b(bb.GetData(), offsets);
   BlockVector x(xb.GetData(), offsets);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(tol);
   cg.SetOperator(*Mv);
   cg.Mult(b.GetBlock(0), x.GetBlock(0));
   x.GetBlock(1) = 0.0;

   return 0;
}

void NavierStokesOperator::Step(BlockVector &X, double &t, const double dt)
{
   NAVIER_PERF_BEGIN("NavierStokesOperator::Step");

   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-4);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(100);
   krylov.SetMaxIter(100);
   krylov.SetPreconditioner(*pc.get());
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-3);
   newton.SetAbsTol(1e-9);
   newton.SetMaxIter(1);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetOperator(*trans_newton_residual);
   newton.SetPreconditioner(krylov);

   int method = 4;

   if (method == 4)
   {
      // IMEX BE-FE
      const double aI_22 = 1.0;
      const double aE_21 = 1.0;
      const double c2 = 1.0;

      BlockVector F1(offsets), F2(offsets);

      // Stage 1
      {
         SetTime(t);
         Setup(dt);
         Y = X;
         MultExplicit(Y, F1);
      }
      // Stage 2
      {
         SetTime(t + c2 * dt);

         Setup(aI_22 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         z.GetBlock(0) += w.GetBlock(0);

         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;
         Y = X;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         NAVIER_PERF_BEGIN("Newton");
         newton.Mult(z, Y);
         NAVIER_PERF_END("Newton");
         X = Y;
      }
   }
   else if (method == 5)
   {
      // IMEX-SSP2(3,2,2)
      const double aI_11 = 0.25, aI_22 = 0.25, aI_31 = 1.0 / 3.0, aI_32 = 1.0 / 3.0,
                   aI_33 = 1.0 / 3.0;
      const double cI_1 = 0.25, cI_2 = 0.25;
      const double aE_21 = 0.5, aE_31 = 0.5, aE_32 = 0.5;
      const double cE_2 = 0.5;

      BlockVector FE1(offsets), FI1(offsets), FI2(offsets), FE2(offsets);

      // Stage 1
      {
         SetTime(t + cI_1 * dt);
         Setup(aI_11 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;

         Y = 0.0;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         SetTime(t + cI_1 * dt);
         MultImplicit(Y, FI1);

         SetTime(t);
         MultExplicit(Y, FE1);
      }
      // Stage 2
      {
         SetTime(t + cI_2 * dt);

         Setup(aI_22 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;

         Y = FE1;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         SetTime(t + cI_2 * dt);
         MultImplicit(Y, FI2);

         SetTime(t + cE_2 * dt);
         MultExplicit(Y, FE2);
      }
      // Stage 3
      {
         SetTime(t + dt);

         Setup(aI_33 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = FI1.GetBlock(0);
         w.GetBlock(0) *= dt * aI_31;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_32;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_31;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_32;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;

         Y = FE2;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);
         X = Y;
      }
   }
   else if (method == 6)
   {
      // IMEX ARS(2,2,2)
      const double gamma = (2.0 - sqrt(2)) / 2.0;
      const double delta = 1.0 - 1.0 / (2.0 * gamma);
      const double aI_22 = gamma, aI_32 = 1.0 - gamma, aI_33 = gamma;
      const double aE_21 = gamma, aE_31 = delta, aE_32 = 1.0 - delta;
      const double c_2 = gamma, c_3 = 1.0;

      BlockVector FE1(offsets), FI1(offsets), FI2(offsets), FE2(offsets);

      // Stage 1
      {
         SetTime(t);
         Setup(dt);
         Y = X;
         MultExplicit(Y, FE1);
      }
      // Stage 2
      {
         SetTime(t + c_2 * dt);

         Setup(aI_22 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;

         Y = FE1;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         NAVIER_PERF_BEGIN("Newton");
         newton.Mult(z, Y);
         NAVIER_PERF_END("Newton");

         MultImplicit(Y, FI2);
         MultExplicit(Y, FE2);
      }
      // Stage 3
      {
         SetTime(t + c_3 * dt);

         Setup(aI_33 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_31;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_32;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_32;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
         z.GetBlock(1) = 0.0;

         Y = FE2;
         ProjectVelocityDirichletBC(Y.GetBlock(0));
         ProjectPressureDirichletBC(Y.GetBlock(1));

         // Orthogonalize(Y.GetBlock(1));
         NAVIER_PERF_BEGIN("Newton");
         newton.Mult(z, Y);
         NAVIER_PERF_END("Newton");
         X = Y;
      }
   }

   t += dt;
   NAVIER_PERF_END("NavierStokesOperator::Step");
}

void NavierStokesOperator::ProjectVelocityDirichletBC(Vector &v)
{
   // Update boundary conditions
   vel_bc_gf->Distribute(v);
   for (auto &bc : velocity_dbcs)
   {
      vel_bc_gf->ProjectBdrCoefficient(*bc.first, *bc.second);
   }
   vel_bc_gf->ParallelProject(v);
}

void NavierStokesOperator::ProjectPressureDirichletBC(Vector &p)
{
   // Update boundary conditions
   pres_bc_gf->Distribute(p);
   for (auto &bc : pressure_dbcs)
   {
      pres_bc_gf->ProjectBdrCoefficient(*bc.first, *bc.second);
   }
   pres_bc_gf->ParallelProject(p);
}

void NavierStokesOperator::SetTime(double t)
{
   for (auto &bc : velocity_dbcs)
   {
      bc.first->SetTime(t);
   }

   if (forcing_form != nullptr)
   {
      forcing_coeff->SetTime(t);

      forcing_form->Update();
      forcing_form->Assemble();
      forcing_form->ParallelAssemble(fu_rhs);
   }

   time = t;
}

int NavierStokesOperator::Setup(const double dt)
{
   if (cached_dt == dt)
   {
      if (Mpi::Root())
      {
         out << "NavierStokesOperator::Setup >> using cached objects" << std::endl;
      }
      return 0;
   }

   cached_dt = dt;

   delete am_coeff;
   am_coeff = new ConstantCoefficient(1.0);

   delete ak_coeff;
   ak_coeff = new GridFunctionCoefficient(&kinematic_viscosity);

   delete ad_coeff;
   ad_coeff = new ConstantCoefficient(1.0);

   delete am_mono_coeff;
   am_mono_coeff = new ProductCoefficient(1.0, *am_coeff);

   delete ak_mono_coeff;
   ak_mono_coeff = new TransformedCoefficient(ak_coeff,
   [this](const double kv) { return cached_dt * kv; });

   SetParameters(am_coeff, ak_coeff, ad_coeff);
   Assemble();

   trans_newton_residual->Setup(cached_dt);

   return 1;
}

void NavierStokesOperator::SetParameters(Coefficient *am,
                                         Coefficient *ak,
                                         Coefficient *ad)
{
   this->am_coeff = am;
   this->ak_coeff = ak;
   this->ad_coeff = ad;

   delete mv_form;
   mv_form = new ParBilinearForm(&vel_fes);
   BilinearFormIntegrator *integrator = new VectorMassIntegrator(*am_coeff);
   integrator->SetIntRule(&ir);
   mv_form->AddDomainIntegrator(integrator);

   delete mp_form;
   mp_form = new ParBilinearForm(&pres_fes);
   auto mp_coeff = new TransformedCoefficient(ak_coeff,
   [](const double kv) { return 1.0 / kv; });
   integrator = new MassIntegrator(*mp_coeff);
   integrator->SetIntRule(&ir);
   mp_form->AddDomainIntegrator(integrator);

   delete lp_form;
   lp_form = new ParBilinearForm(&pres_fes);
   // auto lp_coeff = new ConstantCoefficient(cached_dt);
   integrator = new DiffusionIntegrator;
   integrator->SetIntRule(&ir);
   lp_form->AddDomainIntegrator(integrator);

   delete k_form;
   k_form = new ParBilinearForm(&vel_fes);
   integrator = new VectorDiffusionIntegrator(*ak_coeff);
   integrator->SetIntRule(&ir);
   k_form->AddDomainIntegrator(integrator);

   delete d_form;
   d_form = new ParMixedBilinearForm(&vel_fes, &pres_fes);
   integrator = new VectorDivergenceIntegrator(*ad_coeff);
   integrator->SetIntRule(&ir);
   d_form->AddDomainIntegrator(integrator);

   delete g_form;
   g_form = new ParMixedBilinearForm(&pres_fes, &vel_fes);
   integrator = new GradientIntegrator;
   integrator->SetIntRule(&ir);
   g_form->AddDomainIntegrator(integrator);

   if (matrix_free)
   {
      mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      lp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      d_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   if (convection)
   {
      delete n_form;
      n_form = new ParNonlinearForm(&vel_fes);
      NonlinearFormIntegrator *nl_integrator = new VectorConvectionNLFIntegrator;
      nl_integrator->SetIntRule(&ir_nl);
      n_form->AddDomainIntegrator(nl_integrator);

      if (matrix_free)
      {
         // n_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
   }
}

void NavierStokesOperator::SetForcing(VectorCoefficient *f)
{
   forcing_coeff = f;

   delete forcing_form;
   forcing_form = new ParLinearForm(&vel_fes);
   auto forcing_integrator = new VectorDomainLFIntegrator(*forcing_coeff);
   forcing_integrator->SetIntRule(&ir_nl);
   forcing_form->AddDomainIntegrator(forcing_integrator);
   if (matrix_free)
   {
      forcing_form->UseFastAssembly(true);
   }

   fu_rhs.SetSize(vel_fes.GetTrueVSize());
   fu_rhs = 0.0;
}

/// @brief Assemble all forms and matrices
void NavierStokesOperator::Assemble()
{
   NAVIER_PERF_BEGIN("NavierStokesOperator::Assemble");
   if (k_form == nullptr)
   {
      MFEM_ABORT("use SetParameters() first")
   }

   Array<int> empty;
   OperatorHandle tmphandle;

   mv_form->Update();
   mv_form->Assemble();
   mv_form->Finalize();
   mv_form->FormSystemMatrix(empty, Mv);

   k_form->Update();
   k_form->Assemble();
   k_form->Finalize();
   k_form->FormSystemMatrix(empty, K);

   d_form->Update();
   d_form->Assemble();
   d_form->Finalize();
   d_form->FormRectangularSystemMatrix(empty, empty, D);

   g_form->Update();
   g_form->Assemble();
   g_form->Finalize();
   g_form->FormRectangularSystemMatrix(empty, empty, G);

   delete A;
   A = new BlockOperator(offsets);
   A->SetBlock(0, 0, K.Ptr());
   A->SetBlock(0, 1, G.Ptr());
   A->SetBlock(1, 0, D.Ptr());

   if (convection)
   {
      n_form->Update();
      n_form->Setup();
   }

   if (forcing_form != nullptr)
   {
      forcing_form->Update();
      forcing_form->Assemble();
      forcing_form->ParallelAssemble(fu_rhs);
   }
   NAVIER_PERF_END("NavierStokesOperator::Assemble");
}

void NavierStokesOperator::RebuildPC(const Vector &x)
{
   NAVIER_PERF_BEGIN("NavierStokesOperator::RebuildPC");

   Array<int> empty;
   BilinearFormIntegrator *integrator = nullptr;

   delete mdta_form;
   mdta_form = new ParBilinearForm(&vel_fes);
   integrator = new VectorMassIntegrator;
   integrator->SetIntRule(&ir);
   mdta_form->AddDomainIntegrator(integrator);
   integrator = new VectorDiffusionIntegrator(*ak_mono_coeff);
   integrator->SetIntRule(&ir);
   mdta_form->AddDomainIntegrator(integrator);

   delete gmono_form;
   gmono_form = new ParMixedBilinearForm(&pres_fes, &vel_fes);
   integrator = new GradientIntegrator(new ConstantCoefficient(cached_dt));
   integrator->SetIntRule(&ir);
   gmono_form->AddDomainIntegrator(integrator);

   delete dmono_form;
   dmono_form = new ParMixedBilinearForm(&vel_fes, &pres_fes);
   integrator = new VectorDivergenceIntegrator(*ad_coeff);
   integrator->SetIntRule(&ir);
   dmono_form->AddDomainIntegrator(integrator);

   delete cmono_form;
   cmono_form = new ParBilinearForm(&pres_fes);
   cmono_form->AddDomainIntegrator(new MassIntegrator(zero_coeff));

   if (matrix_free)
   {
      mdta_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      gmono_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      dmono_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      cmono_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   mdta_form->Update();
   mdta_form->Assemble();
   mdta_form->Finalize();
   mdta_form->FormSystemMatrix(vel_ess_tdofs, MdtAe);

   dmono_form->Update();
   dmono_form->Assemble();
   dmono_form->Finalize();
   dmono_form->FormRectangularSystemMatrix(vel_ess_tdofs, pres_ess_tdofs, Dmonoe);

   gmono_form->Update();
   gmono_form->Assemble();
   gmono_form->Finalize();
   gmono_form->FormRectangularSystemMatrix(pres_ess_tdofs, vel_ess_tdofs, Gmonoe);

   cmono_form->Update();
   cmono_form->Assemble();
   cmono_form->Finalize();
   cmono_form->FormSystemMatrix(pres_ess_tdofs, Cmonoe);

   mp_form->Update();
   mp_form->Assemble();
   mp_form->Finalize();
   mp_form->FormSystemMatrix(empty, Mp);

   lp_form->Update();
   lp_form->Assemble();
   lp_form->Finalize();
   lp_form->FormSystemMatrix(pres_ess_tdofs, Lp);

   // Array2D<HypreParMatrix *> blocks(2, 2);
   // blocks(0, 0) = MdtAe.As<HypreParMatrix>();
   // blocks(0, 1) = Gmonoe.As<HypreParMatrix>();
   // blocks(1, 0) = Dmonoe.As<HypreParMatrix>();
   // blocks(1, 1) = nullptr;
   // delete Amonoe;
   // Amonoe = HypreParMatrixFromBlocks(blocks);

   delete Amonoe_matfree;
   Amonoe_matfree = new BlockOperator(offsets);
   Amonoe_matfree->SetBlock(0, 0, MdtAe.Ptr());
   Amonoe_matfree->SetBlock(0, 1, Gmonoe.Ptr());
   Amonoe_matfree->SetBlock(1, 0, Dmonoe.Ptr());
   Amonoe_matfree->SetBlock(1, 1, Cmonoe.Ptr());

   {
      auto amg = new LORSolver<HypreBoomerAMG>(*mdta_form, vel_ess_tdofs);
      amg->GetSolver().SetSystemsOptions(vel_fes.GetMesh()->Dimension(), true);
      HYPRE_BoomerAMGSetAggNumLevels(amg->GetSolver(), 4);
      HYPRE_BoomerAMGSetPMaxElmts(amg->GetSolver(), 2);
      amg->GetSolver().SetPrintLevel(0);

      Vector diag_pa(vel_fes.GetTrueVSize());
      mdta_form->AssembleDiagonal(diag_pa);
      auto jacobi = new OperatorJacobiSmoother(diag_pa, vel_ess_tdofs);

      // MdtAinvPC.reset(amg);
      // auto cg = new CGSolver(MPI_COMM_WORLD);
      // cg->SetRelTol(1e-6);
      // cg->SetMaxIter(10);
      // cg->SetOperator(*MdtAe.Ptr());
      // cg->SetPreconditioner(*MdtAinvPC);
      MdtAinv.reset(jacobi);
   }

   {
      Vector diag_pa(pres_fes.GetTrueVSize());
      mp_form->AssembleDiagonal(diag_pa);
      auto jacobi = new OperatorJacobiSmoother(diag_pa, pres_ess_tdofs);
      // MpinvPC.reset(jacobi);
      // auto cg = new CGSolver(MPI_COMM_WORLD);
      // cg->SetRelTol(1e-4);
      // cg->SetMaxIter(10);
      // cg->SetOperator(*Mp);
      // cg->SetPreconditioner(*MpinvPC);
      Mpinv.reset(jacobi);
   }

   {
      auto amg = new LORSolver<HypreBoomerAMG>(*lp_form, pres_ess_tdofs);
      HYPRE_BoomerAMGSetStrongThreshold(amg->GetSolver(), 0.7);
      amg->GetSolver().SetPrintLevel(0);
      // LpinvPC.reset(amg);
      // auto cg = new CGSolver(MPI_COMM_WORLD);
      // cg->SetRelTol(1e-4);
      // cg->SetMaxIter(10);
      // cg->SetOperator(*Lp);
      // cg->SetPreconditioner(*LpinvPC);
      // cg->SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
      Lpinv.reset(amg);
   }

   Sinv = new CahouetChabardPC(*Mpinv.get(), *Lpinv.get(), cached_dt,
                               pres_ess_tdofs);

   auto block_pc = static_cast<BlockLowerTriangularPreconditioner *>(pc.get());
   block_pc->SetBlock(0, 0, MdtAinv.get());
   block_pc->SetBlock(1, 0, D.Ptr());
   block_pc->SetBlock(1, 1, Sinv);

   NAVIER_PERF_END("NavierStokesOperator::RebuildPC");
}

const Array<int>& NavierStokesOperator::GetOffsets() const
{
   return offsets;
}