#include "navierstokes_operator.hpp"
#include "block_schur_pc.hpp"
#include "schur_pcd.hpp"
#include "util.hpp"

using namespace mfem;

NavierStokesOperator::NavierStokesOperator(ParFiniteElementSpace &vel_fes,
                                           ParFiniteElementSpace &pres_fes,
                                           const Array<int> vel_ess_bdr,
                                           const Array<int> pres_ess_bdr,
                                           ParGridFunction &nu_gf,
                                           bool convection,
                                           bool convection_explicit,
                                           bool matrix_free) :
   TimeDependentOperator(vel_fes.GetTrueVSize() + pres_fes.GetTrueVSize()),
   vel_fes(vel_fes),
   pres_fes(pres_fes),
   kinematic_viscosity(nu_gf),
   vel_ess_bdr(vel_ess_bdr),
   pres_ess_bdr(pres_ess_bdr),
   convection(convection),
   convection_explicit(convection_explicit),
   matrix_free(matrix_free),
   offsets({0, vel_fes.GetTrueVSize(), pres_fes.GetTrueVSize()})
{
   vel_fes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdofs);
   pres_fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdofs);

   offsets.PartialSum();

   z.Update(offsets);
   w.Update(offsets);

   trans_newton_residual.reset(new TransientNewtonResidual(*this));
}

void NavierStokesOperator::Mult(const Vector &x, Vector &y) const
{}

void NavierStokesOperator::ImplicitSolve(const double dt, const Vector &b,
                                         Vector &x)
{
   FGMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-6);
   krylov.SetMaxIter(1000);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-10);
   newton.SetAbsTol(1e-12);
   newton.SetMaxIter(50);
   newton.SetOperator(*trans_newton_residual);
   newton.SetPreconditioner(krylov);
   // newton.SetAdaptiveLinRtol();
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());

   const BlockVector bb(b.GetData(), offsets);
   Mv->Mult(bb.GetBlock(0), z.GetBlock(0));
   z.GetBlock(0).SetSubVector(vel_ess_tdofs, 0.0);
   // Set the RHS for the pressure equation to zero, as it is not involved in
   // time-stepping.
   z.GetBlock(1) = 0.0;
   newton.Mult(z, x);
}

void NavierStokesOperator::SetTime(double t)
{
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
   delete am_coeff;
   am_coeff = new ConstantCoefficient(1.0);

   delete ak_coeff;
   ak_coeff = new GridFunctionCoefficient(&kinematic_viscosity);

   delete ad_coeff;
   ad_coeff = new ConstantCoefficient(-1.0);

   SetParameters(am_coeff, ak_coeff, ad_coeff);
   Assemble();

   trans_newton_residual->Setup(dt);

   cached_dt = dt;
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
   mv_form->AddDomainIntegrator(new VectorMassIntegrator(*am_coeff));

   delete mp_form;
   mp_form = new ParBilinearForm(&pres_fes);
   mp_form->AddDomainIntegrator(new MassIntegrator);

   delete k_form;
   k_form = new ParBilinearForm(&vel_fes);
   k_form->AddDomainIntegrator(new VectorDiffusionIntegrator(*ak_coeff));

   delete d_form;
   d_form = new ParMixedBilinearForm(&vel_fes, &pres_fes);
   d_form->AddDomainIntegrator(new VectorDivergenceIntegrator(*ad_coeff));

   delete g_form;
   g_form = new ParMixedBilinearForm(&pres_fes, &vel_fes);
   g_form->AddDomainIntegrator(new GradientIntegrator());

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
      n_form->AddDomainIntegrator(new VectorConvectionNLFIntegrator);

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
   forcing_form->AddDomainIntegrator(new VectorDomainLFIntegrator(*forcing_coeff));
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
   g_form->FormRectangularSystemMatrix(empty, empty, Dt);

   delete A;
   A = new BlockOperator(offsets);
   A->SetBlock(0, 0, K.Ptr());
   A->SetBlock(0, 1, Dt.Ptr());
   A->SetBlock(1, 0, D.Ptr());

   if (convection)
   {
      n_form->Setup();
   }

   if (forcing_form != nullptr)
   {
      forcing_form->Update();
      forcing_form->Assemble();
      forcing_form->ParallelAssemble(fu_rhs);
   }
}

const Array<int>& NavierStokesOperator::GetOffsets() const
{
   return offsets;
}