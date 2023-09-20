#include "navierstokes_operator.hpp"
#include "util.hpp"

using namespace mfem;

NavierStokesOperator::NavierStokesOperator(ParFiniteElementSpace &vel_fes,
                                           ParFiniteElementSpace &pres_fes,
                                           std::vector<VelDirichletBC> velocity_dbcs,
                                           const Array<int> pres_ess_bdr,
                                           ParGridFunction &nu_gf,
                                           bool convection,
                                           bool convection_explicit,
                                           bool matrix_free) :
   Operator(vel_fes.GetTrueVSize() + pres_fes.GetTrueVSize()),
   vel_fes(vel_fes),
   pres_fes(pres_fes),
   kinematic_viscosity(nu_gf),
   velocity_dbcs(velocity_dbcs),
   pres_ess_bdr(pres_ess_bdr),
   convection(convection),
   convection_explicit(convection_explicit),
   matrix_free(matrix_free),
   offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()}),
   intrules(0, Quadrature1D::GaussLobatto)
{
   // Append individual velocity Dirichlet BC attributes to collective array.
   if (vel_fes.GetParMesh()->bdr_attributes.Size() > 0) {
      vel_ess_bdr.SetSize(vel_fes.GetParMesh()->bdr_attributes.Max());
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

   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdofs);

   offsets.PartialSum();

   z.Update(offsets);
   w.Update(offsets);
   Y.Update(offsets);

   trans_newton_residual.reset(new TransientNewtonResidual(*this));

   vel_bc_gf = std::make_unique<ParGridFunction>(&vel_fes);
   *vel_bc_gf = 0.0;

   ir_nl = intrules.Get(vel_fes.GetFE(0)->GetGeomType(),
                             (int)(ceil(1.5 * 2*(vel_fes.GetOrder(0)+1) - 3)));

   ir = intrules.Get(vel_fes.GetFE(0)->GetGeomType(),
                             (int)(2*(vel_fes.GetOrder(0)+1) - 3));

   strumpack.reset(new STRUMPACKSolver(0, nullptr, MPI_COMM_WORLD));
   mumps.reset(new MUMPSSolver);
}

void NavierStokesOperator::MultImplicit(const Vector &xb, Vector &yb) const
{
   const BlockVector x(xb.GetData(), offsets);
   BlockVector y(yb.GetData(), offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   A->Mult(x, y);
   y.Neg();

   yu += fu_rhs;

   for (int i = 0; i < vel_ess_tdofs.Size(); i++)
   {
      yu[vel_ess_tdofs[i]] = xu[vel_ess_tdofs[i]];
   }

   yp = 0.0;
}

void NavierStokesOperator::MultExplicit(const Vector &xb, Vector &yb) const
{
   const BlockVector x(xb.GetData(), offsets);
   BlockVector y(yb.GetData(), offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   n_form->Mult(xu, yu);

   for (int i = 0; i < vel_ess_tdofs.Size(); i++)
   {
      yu[vel_ess_tdofs[i]] = xu[vel_ess_tdofs[i]];
   }

   yp = 0.0;
}

void NavierStokesOperator::Mult(const Vector &xb, Vector &yb) const
{
   const BlockVector x(xb.GetData(), offsets);
   BlockVector y(yb.GetData(), offsets);

   const Vector &xu = x.GetBlock(0);
   Vector &zu = z.GetBlock(0);
   Vector &yu = y.GetBlock(0);

   if (convection_explicit) {
      n_form->Mult(xu, yu);
   } else {
      A->Mult(x, y);
      y.Neg();

      yu += fu_rhs;

      if (convection)
      {
         n_form->Mult(xu, zu);
         yu -= zu;
      }
   }

   for (int i = 0; i < vel_ess_tdofs.Size(); i++)
   {
      yu[vel_ess_tdofs[i]] = xu[vel_ess_tdofs[i]];
   }
}

void NavierStokesOperator::Step(BlockVector &X, double &t, const double dt)
{
   MyFGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-6);
   krylov.SetMaxIter(5000);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   BlockOrthoSolver krylov_ortho(MPI_COMM_WORLD, offsets);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-4);
   newton.SetAbsTol(1e-12);
   newton.SetMaxIter(1);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetOperator(*trans_newton_residual);
   newton.SetPreconditioner(*mumps.get());
   // newton.SetPreconditioner(krylov);

   int method = 5;

   if (method == 1)
   {
      // Backward Euler

      SetTime(t + dt);

      Setup(dt);

      Mv->Mult(X.GetBlock(0), z.GetBlock(0));
      z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

      ProjectVelocityDirichletBC(X.GetBlock(0));
   
      Y = X;
      Orthogonalize(Y.GetBlock(1));
      newton.Mult(z, Y);
      X = Y;
   }
   else if (method == 2)
   {
      // SDIRK2
      double a11, a22, a33, a44;
      a11 = a22 = a33 = a44 = 1.0 / 4.0;
      double c1 = 1.0 / 4.0, c2 = 11.0 / 28.0, c3 = 1.0 / 3.0, c4 = 1.0;
      double a21 = 1.0 / 7.0;
      double a31 = 61.0 / 144.0, a32 = -49.0 / 144.0;
      double a41 = 0.0, a42 = 0.0, a43 = 3.0 / 4.0;
      BlockVector F1(offsets), F2(offsets), F3(offsets);

      // Stage 1
      {
         SetTime(t + c1 * dt);

         Setup(a11 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(X.GetBlock(0));

         Y = X;
         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         Mult(Y, F1);
      }
      // Stage 2
      {
         SetTime(t + c2 * dt);

         Setup(a22 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * a21;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         Mult(Y, F2);
      }
      // Stage 3
      {
         SetTime(t + c3 * dt);

         Setup(a33 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * a31;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = F2.GetBlock(0);
         w.GetBlock(0) *= dt * a32;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         Mult(Y, F3);
      }
      // Stage 4
      {
         SetTime(t + c4 * dt);

         Setup(a44 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * a41;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = F2.GetBlock(0);
         w.GetBlock(0) *= dt * a42;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = F3.GetBlock(0);
         w.GetBlock(0) *= dt * a43;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);
         X = Y;
      }
   }
   else if (method == 3)
   {
      // TR-BDF2
      double alpha = 2.0 - sqrt(2);
      double a21 = alpha / 2.0,
             a22 = alpha / 2.0;
      double a31 = 1.0 / (2.0 * (2.0 - alpha)),
             a32 = 1.0 / (2.0 * (2.0 - alpha)),
             a33 = (1.0 - alpha) / (2.0 - alpha);
      double c2 = alpha,
             c3 = 1.0;
      
      BlockVector F1(offsets), F2(offsets);

      // Stage 1
      {
         SetTime(t);
         Setup(dt);
         Mult(X, F1);
      }
      // Stage 2
      {
         SetTime(t + c2 * dt);

         Setup(a22 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * a21;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         Mult(Y, F2);
      }
      // Stage 3
      {
         SetTime(t + c3 * dt);

         Setup(a33 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         w.GetBlock(0) = F1.GetBlock(0);
         w.GetBlock(0) *= dt * a31;
         z.GetBlock(0) += w.GetBlock(0);
         w.GetBlock(0) = F2.GetBlock(0);
         w.GetBlock(0) *= dt * a32;
         z.GetBlock(0) += w.GetBlock(0);
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         X = Y;
      }
   } else if (method == 4)
   {
      // IMEX BE-FE
      double aI_22 = 1.0;
      double aE_21 = 1.0;
      double c2 = 1.0;

      BlockVector F1(offsets), F2(offsets);

      convection_explicit = true;

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

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);
         X = Y;
      }
    } else if (method == 5) {
      // IMEX-SSP2(3,2,2)
      double aI_11 = 0.25, aI_22 = 0.25, aI_31 = 1.0 / 3.0, aI_32 = 1.0 / 3.0, aI_33 = 1.0 / 3.0;
      double cI_1 = 0.25, cI_2 = 0.25, cI_3 = 1.0;
      double aE_21 = 0.5, aE_31 = 0.5, aE_32 = 0.5;
      double cE_2 = 0.5, cE_3 = 1.0;

      BlockVector FE1(offsets), FI1(offsets), FI2(offsets), FE2(offsets);

      convection_explicit = true;

      // Stage 1
      {
         SetTime(t + cI_1 * dt);
         Setup(aI_11 * dt);

         Mv->Mult(X.GetBlock(0), z.GetBlock(0));
         z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
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

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);

         SetTime(t + cI_2 * dt);
         MultImplicit(Y, FI2);

         SetTime(t + cE_2 * dt);
         MultExplicit(Y, FE2);
      }
      // Stage 3
      {
         SetTime(t + cI_3 * dt);

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

         ProjectVelocityDirichletBC(Y.GetBlock(0));

         Orthogonalize(Y.GetBlock(1));
         newton.Mult(z, Y);
         X = Y;
      }
   }

   t += dt;
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

void NavierStokesOperator::Setup(const double dt)
{
   if (cached_dt == dt)
   {
      if (Mpi::Root())
      {
         out << "NavierStokesOperator::Setup >> using cached objects" << std::endl;
      }
      return;
   }

   cached_dt = dt;

   delete am_coeff;
   am_coeff = new ConstantCoefficient(1.0);

   delete ak_coeff;
   ak_coeff = new GridFunctionCoefficient(&kinematic_viscosity);

   delete ad_coeff;
   ad_coeff = new ConstantCoefficient(-1.0);

   delete am_mono_coeff;
   am_mono_coeff = new ProductCoefficient(1.0, *am_coeff);

   delete ak_mono_coeff;
   ak_mono_coeff = new ProductCoefficient(cached_dt, *ak_coeff);

   SetParameters(am_coeff, ak_coeff, ad_coeff);
   Assemble();

   trans_newton_residual->Setup(cached_dt);
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
   integrator = new MassIntegrator;
   integrator->SetIntRule(&ir);
   mp_form->AddDomainIntegrator(integrator);

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
         n_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
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
}

/// @brief Assemble all forms and matrices
void NavierStokesOperator::Assemble()
{
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

   mp_form->Update();
   mp_form->Assemble();
   mp_form->Finalize();
   mp_form->FormSystemMatrix(empty, Mp);

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
}

void NavierStokesOperator::RebuildPC(const Vector &x)
{
   Array<int> empty;

   delete mdta_form;
   mdta_form = new ParNonlinearForm(&vel_fes);
   BilinearFormIntegrator *integrator = new VectorMassIntegrator(*am_mono_coeff);
   integrator->SetIntRule(&ir);
   mdta_form->AddDomainIntegrator(integrator);
   integrator = new VectorDiffusionIntegrator(*ak_mono_coeff);
   integrator->SetIntRule(&ir);
   mdta_form->AddDomainIntegrator(integrator);
   if (!convection_explicit)
   {
      auto nl_integrator = new VectorConvectionNLFIntegrator(*ak_mono_coeff);
      nl_integrator->SetIntRule(&ir_nl);
      mdta_form->AddDomainIntegrator(integrator);
   }
   mdta_form->SetEssentialTrueDofs(vel_ess_tdofs);

   delete gmono_form;
   gmono_form = new ParMixedBilinearForm(&pres_fes, &vel_fes);
   integrator = new GradientIntegrator(new ConstantCoefficient(cached_dt));
   integrator->SetIntRule(&ir);
   gmono_form->AddDomainIntegrator(integrator);

   delete dmono_form;
   dmono_form = new ParMixedBilinearForm(&vel_fes, &pres_fes);
   integrator = new VectorDivergenceIntegrator(new ConstantCoefficient(-1.0));
   integrator->SetIntRule(&ir);
   dmono_form->AddDomainIntegrator(integrator);

   mdta_form->Update();

   dmono_form->Update();
   dmono_form->Assemble();
   dmono_form->Finalize();
   dmono_form->FormRectangularSystemMatrix(empty, empty, Dmonoe);

   gmono_form->Update();
   gmono_form->Assemble();
   gmono_form->Finalize();
   gmono_form->FormRectangularSystemMatrix(empty, vel_ess_tdofs, Gmonoe);

   const BlockVector xb(x.GetData(), offsets);
   HypreParMatrix *MdtAeGrad = static_cast<HypreParMatrix *>(&mdta_form->GetGradient(xb.GetBlock(0)));

   Array2D<HypreParMatrix *> blocks(2, 2);
   blocks(0, 0) = MdtAeGrad;
   blocks(0, 1) = Gmonoe.As<HypreParMatrix>();
   blocks(1, 0) = Dmonoe.As<HypreParMatrix>();
   blocks(1, 1) = nullptr;
   delete Amonoe;
   Amonoe = HypreParMatrixFromBlocks(blocks);

   mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps->SetOperator(*Amonoe);

   // delete Amonoe_rowloc;
   // Amonoe_rowloc = new STRUMPACKRowLocMatrix(*Amonoe);
   // strumpack->SetPrintFactorStatistics(false);
   // strumpack->SetPrintSolveStatistics(false);
   // strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
   // strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
   // strumpack->DisableMatching();
   // strumpack->SetOperator(*Amonoe_rowloc);
   // strumpack->SetFromCommandLine();

   // std::ofstream amono("amono.dat");
   // Amonoe->PrintMatlab(amono);
   // amono.close();
}

const Array<int>& NavierStokesOperator::GetOffsets() const
{
   return offsets;
}