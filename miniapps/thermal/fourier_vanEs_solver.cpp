// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fourier_vanEs_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;
/*
void
UnitVectorField::Eval(Vector &V, ElementTransformation &T,
                    const IntegrationPoint &ip)
{
 double x[2];
 Vector transip(x, 2);

 T.Transform(T.GetIntPoint(), transip);

 V.SetSize(2);

 if ( prob_ % 2 == 1 )
 {
    if (unit_vec_type_ == 1)
    {
       double cx = cos(M_PI * x[0]);
 double cy = cos(M_PI * x[1]);
 double sx = sin(M_PI * x[0]);
 double sy = sin(M_PI * x[1]);

 V[0] = -sx * cy;
 V[1] =  sy * cx;
    }
    else
    {
 V[0] = cos(M_PI/6.0);
 V[1] = sin(M_PI/6.0);
    }
 }
 else
 {
    V[0] = -a_ * a_ * x[1];
    V[1] =  b_ * b_ * x[0];
 }

 double nrm = V.Norml2();
 V *= (nrm > 1e-6 * min(a_,b_)) ? (1.0/nrm) : 0.0;
}
*/
void ChiParaCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);

   if (nonlin_)
   {
      K *= pow(T_->Eval(T, ip), 2.5);
   }
   K *= chi_para_;
}

void ChiPerpCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);
   K *= -1.0;
   K(0,0) += 1.0;
   K(1,1) += 1.0;

   if (nonlin_)
   {
      K *= 1.0 / sqrt(T_->Eval(T, ip));
   }
   K *= chi_perp_;
}

void dChiCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                    const IntegrationPoint &ip)
{
   double temp = T_->Eval(T, ip);
   double perp_factor = 0.5 * chi_perp_ * pow(temp, -1.5);
   double para_factor = 2.5 * chi_para_ * pow(temp,  1.5);

   bbT_->Eval(K, T, ip);
   K *= perp_factor + para_factor;
   K(0,0) -= perp_factor;
   K(1,1) -= perp_factor;
}

namespace thermal
{

ThermalDiffusionTDO::ThermalDiffusionTDO(
   ParFiniteElementSpace &H1_FESpace,
   Coefficient & dTdtBdr,
   Array<int> & bdr_attr,
   double chi_perp,
   double chi_para,
   int prob,
   int coef_type,
   VectorCoefficient & UnitB,
   Coefficient & c, bool td_c,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FESpace.GetTrueVSize(), 0.0),
     init_(false),
     nonLinear_(coef_type == 2),
     testGradient_(false),
     multCount_(0), solveCount_(0),
     T_(&H1_FESpace),
     TCoef_(&T_),
     unitBCoef_(&UnitB),
     // ICoef_(2),
     bbTCoef_(*unitBCoef_, *unitBCoef_),
     chiPerpCoef_(bbTCoef_, TCoef_, chi_perp, coef_type != 0),
     chiParaCoef_(bbTCoef_, TCoef_, chi_para, coef_type != 0),
     chiCoef_(chiPerpCoef_, chiParaCoef_),
     dChiCoef_(bbTCoef_, TCoef_, chi_perp, chi_para),
     impOp_(H1_FESpace,
            dTdtBdr, false,
            bdr_attr,
            c, false,
            chiCoef_, coef_type != 0,
            dChiCoef_, coef_type != 0,
            Q,    false, coef_type == 2 ),
     newton_(H1_FESpace.GetComm())
{
   this->init();
}

ThermalDiffusionTDO::~ThermalDiffusionTDO()
{
}

void
ThermalDiffusionTDO::init()
{
   cout << "Entering TDO::Init" << endl;
   if ( init_ ) { return; }

   newton_.SetPrintLevel(2);
   newton_.SetRelTol(1e-10);
   newton_.SetAbsTol(0.0);

   if ( nonLinear_ && testGradient_ )
   {
      Vector x(impOp_.Height());
      Vector dx(impOp_.Height());

      T_.Distribute(x);
      cout << "GetTime " << this->GetTime() << endl;
      impOp_.SetState(T_, this->GetTime(), 0.1);

      cout << "init 0" << endl;
      newton_.SetOperator(impOp_);
      cout << "init 1" << endl;
      cout << "init 2" << endl;
      x.Randomize(1);
      x.Print(cout);
      dx.Randomize(2);
      dx *= 0.01;
      dx.Print(cout);
      cout << "init 3" << endl;
      double ratio = newton_.CheckGradient(x, dx);
      cout << "CheckGradient returns: " << ratio << endl;
   }

   init_ = true;
   cout << "Leaving TDO::Init" << endl;
}

void
ThermalDiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   newTime_ = true;
}

void
ThermalDiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   MFEM_ABORT("ThermalDiffusionTDO::Mult should not be called");
}

void
ThermalDiffusionTDO::ImplicitSolve(const double dt,
                                   const Vector &T, Vector &dT_dt)
{
   dT_dt = 0.0;

   T_.Distribute(T);

   impOp_.SetState(T_, this->GetTime(), dt);

   Solver & solver = impOp_.GetGradientSolver();

   if (!nonLinear_)
   {
      solver.Mult(impOp_.GetRHS(), dT_dt);
   }
   else
   {
      newton_.SetOperator(impOp_);
      newton_.SetSolver(solver);

      newton_.Mult(impOp_.GetRHS(), dT_dt);
   }
   solveCount_++;
}

ImplicitDiffOp::ImplicitDiffOp(ParFiniteElementSpace & H1_FESpace,
                               Coefficient & dTdtBdr, bool tdBdr,
                               Array<int> & bdr_attr,
                               Coefficient & heatCap, bool tdCp,
                               ChiCoef & chi, bool tdChi,
                               dChiCoef & dchi, bool tdDChi,
                               Coefficient & heatSource, bool tdQ,
                               bool nonlinear)
   : Operator(H1_FESpace.GetTrueVSize()),
     first_(true),
     tdBdr_(tdBdr),
     tdCp_(tdCp),
     tdChi_(tdChi),
     tdDChi_(tdDChi),
     tdQ_(tdQ),
     nonLinear_(nonlinear),
     newTime_(true),
     newTimeStep_(true),
     t_(0.0),
     dt_(-1.0),
     ess_bdr_attr_(bdr_attr),
     bdrCoef_(&dTdtBdr),
     cpCoef_(&heatCap),
     chiCoef_(&chi),
     dChiCoef_(&dchi),
     QCoef_(&heatSource),
     dtChiCoef_(1.0, *chiCoef_),
     T0_(&H1_FESpace),
     T1_(&H1_FESpace),
     dT_(&H1_FESpace),
     gradTCoef_(&T0_),
     dtGradTCoef_(-1.0, gradTCoef_),
     dtdChiGradTCoef_(*dChiCoef_, dtGradTCoef_),
     m0cp_(&H1_FESpace),
     s0chi_(&H1_FESpace),
     a0_(&H1_FESpace),
     dTdt_(&H1_FESpace),
     Q_(&H1_FESpace),
     Qs_(&H1_FESpace),
     rhs_(&H1_FESpace),
     RHS_(H1_FESpace.GetTrueVSize()),
     // RHS0_(0),
     AInv_(NULL),
     APrecond_(NULL)
{
   H1_FESpace.GetEssentialTrueDofs(ess_bdr_attr_, ess_bdr_tdofs_);

   m0cp_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
   s0chi_.AddDomainIntegrator(new DiffusionIntegrator(*chiCoef_));

   a0_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
   a0_.AddDomainIntegrator(new DiffusionIntegrator(dtChiCoef_));
   if (nonLinear_)
   {
      a0_.AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(
                                 dtdChiGradTCoef_));
   }

   Qs_.AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
   if (!tdQ_) { Qs_.Assemble(); }
}

ImplicitDiffOp::~ImplicitDiffOp()
{
   delete AInv_;
   delete APrecond_;
}

void ImplicitDiffOp::SetState(ParGridFunction & T, double t, double dt)
{
   T0_ = T;

   newTime_ = fabs(t - t_) > 0.0;
   newTimeStep_= (fabs(1.0-dt/dt_)>1e-6);

   t_  = newTime_     ?  t :  t_;
   dt_ = newTimeStep_ ? dt : dt_;

   if (tdBdr_ && (newTime_ || newTimeStep_))
   {
      bdrCoef_->SetTime(t_ + dt_);
   }

   if (newTimeStep_ || first_)
   {
      dtChiCoef_.SetAConst(dt_);
      dtGradTCoef_.SetAConst(-dt_);
   }

   if ((tdCp_ && newTime_) || first_)
   {
      m0cp_.Update();
      m0cp_.Assemble();
      m0cp_.Finalize();
   }

   if (!tdChi_ && first_)
   {
      s0chi_.Assemble();
      s0chi_.Finalize();

      ofstream ofsS0("s0_const_initial.mat");
      s0chi_.SpMat().Print(ofsS0);

      a0_.Assemble();
      a0_.Finalize();
   }
   else if (tdChi_ && newTime_ && !nonLinear_)
   {
      chiCoef_->SetTemp(T0_);
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);

      ofstream ofsS0("s0_lin_initial.mat");
      s0chi_.SpMat().Print(ofsS0);

      a0_.Update();
      a0_.Assemble(0);
      a0_.Finalize(0);
   }

   if ((tdQ_ && newTime_) || first_)
   {
      cout << "Assembling Q" << endl;
      QCoef_->SetTime(t_ + dt_);
      Qs_.Assemble();
      Qs_.ParallelAssemble(RHS_);
      cout << "Norm of Q: " << Qs_.Norml2() << endl;
   }

   first_       = false;
   newTime_     = false;
   newTimeStep_ = false;
}

void ImplicitDiffOp::Mult(const Vector &dT, Vector &Q) const
{
   dT_.Distribute(dT);

   add(T0_, dt_, dT_, T1_);

   if (tdChi_ && nonLinear_)
   {
      chiCoef_->SetTemp(T1_);
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);
   }
   else
   {
      cout << "Well this is a surprise..." << endl;
   }
   m0cp_.Mult(dT_, Q_);
   s0chi_.AddMult(T1_, Q_);

   Q_.ParallelAssemble(Q);
   Q.SetSubVector(ess_bdr_tdofs_, 0.0);
}

Operator & ImplicitDiffOp::GetGradient(const Vector &dT) const
{
   if (tdChi_)
   {
      if (!nonLinear_)
      {
         chiCoef_->SetTemp(T0_);
      }
      else
      {
         dT_.Distribute(dT);
         add(T0_, dt_, dT_, T1_);

         chiCoef_->SetTemp(T1_);
         dChiCoef_->SetTemp(T1_);
         gradTCoef_.SetGridFunction(&T1_);
      }
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);

      a0_.Update();
      a0_.Assemble(0);
      a0_.Finalize(0);
   }

   if (!nonLinear_)
   {
      s0chi_.Mult(T0_, rhs_);

      rhs_ -= Qs_;
      rhs_ *= -1.0;
   }
   else
   {
      rhs_ = Qs_;
   }

   dTdt_.ProjectBdrCoefficient(*bdrCoef_, ess_bdr_attr_);

   a0_.FormLinearSystem(ess_bdr_tdofs_, dTdt_, rhs_, A_, SOL_, RHS_);

   return A_;
}

Solver & ImplicitDiffOp::GetGradientSolver() const
{
   if (!nonLinear_)
   {
      Operator & A_op = this->GetGradient(T0_); // T0_ will be ignored
      HypreParMatrix & A_hyp = dynamic_cast<HypreParMatrix &>(A_op);

      if (tdChi_)
      {
         delete AInv_;     AInv_     = NULL;
         delete APrecond_; APrecond_ = NULL;
      }

      if ( AInv_ == NULL )
      {
         // A_hyp.Print("A.mat");

         HyprePCG * AInv_pcg = NULL;

         cout << "Building PCG" << endl;
         AInv_pcg = new HyprePCG(A_hyp);
         AInv_pcg->SetTol(1e-10);
         AInv_pcg->SetMaxIter(200);
         AInv_pcg->SetPrintLevel(0);
         if ( APrecond_ == NULL )
         {
            cout << "Building AMG" << endl;
            APrecond_ = new HypreBoomerAMG(A_hyp);
            APrecond_->SetPrintLevel(0);
            AInv_pcg->SetPreconditioner(*APrecond_);
         }
         AInv_ = AInv_pcg;
      }
   }
   else
   {
      if (AInv_ == NULL)
      {
         /*
              HypreSmoother *J_hypreSmoother = new HypreSmoother;
         J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
         J_hypreSmoother->SetPositiveDiagonal(true);
         JPrecond_ = J_hypreSmoother;

              GMRESSolver * AInv_gmres = NULL;

              cout << "Building GMRES" << endl;
              AInv_gmres = new GMRESSolver(T0_.ParFESpace()->GetComm());
              AInv_gmres->SetRelTol(1e-12);
              AInv_gmres->SetAbsTol(0.0);
              AInv_gmres->SetMaxIter(20000);
              AInv_gmres->SetPrintLevel(2);
         AInv_gmres->SetPreconditioner(*JPrecond_);
         AInv_ = AInv_gmres;
         */
         HypreGMRES * AInv_gmres = NULL;

         cout << "Building HypreGMRES" << endl;
         AInv_gmres = new HypreGMRES(T0_.ParFESpace()->GetComm());
         AInv_gmres->SetTol(1e-12);
         AInv_gmres->SetMaxIter(200);
         AInv_gmres->SetPrintLevel(2);
         if ( APrecond_ == NULL )
         {
            cout << "Building AMG" << endl;
            APrecond_ = new HypreBoomerAMG();
            APrecond_->SetPrintLevel(0);
            AInv_gmres->SetPreconditioner(*APrecond_);
         }
         AInv_ = AInv_gmres;
      }
   }

   return *AInv_;
}

} // namespace thermal

void
MatrixInverseCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   M_->Eval(K, T, ip); K.Invert();
}

void
ScaledMatrixCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                              const IntegrationPoint &ip)
{
   M_->Eval(K, T, ip); K *= a_;
}


} // namespace mfem

#endif // MFEM_USE_MPI
