// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "fourier_hybrid_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace miniapps;

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

void dChiParaCoef::Eval(DenseMatrix &K, ElementTransformation &T,
			const IntegrationPoint &ip)
{
   double temp = T_->Eval(T, ip);
   double para_factor = 2.5 * chi_para_ * pow(temp,  1.5);

   bbT_->Eval(K, T, ip);
   K *= para_factor;
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

void ChiInvPerpCoef::Eval(DenseMatrix &K, ElementTransformation &T,
			  const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);
   K *= -1.0;
   K(0,0) += 1.0;
   K(1,1) += 1.0;
   
   if (nonlin_)
   {
     K *= sqrt(T_->Eval(T, ip));
   }
   K *= 1.0 / chi_perp_;
}

void ChiInvParaCoef::Eval(DenseMatrix &K, ElementTransformation &T,
			  const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);

   if (nonlin_)
   {
      K *= pow(T_->Eval(T, ip), -2.5);
   }
   K *= 1.0 / chi_para_;
}

namespace thermal
{

HybridThermalDiffusionTDO::HybridThermalDiffusionTDO(
   ParFiniteElementSpace &H1_FESpace,
   ParFiniteElementSpace &HDiv_FESpace,
   ParFiniteElementSpace &L2_FESpace,
   VectorCoefficient & dqdtBdr,
   Coefficient & dTdtBdr,
   Array<int> & bdr_attr,
   double chi_perp,
   double chi_para,
   int prob,
   int coef_type,
   VectorCoefficient & UnitB,
   Coefficient & c, bool td_c,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FESpace.GetTrueVSize() +
			   HDiv_FESpace.GetTrueVSize(), 0.0),
     init_(false),
     nonLinear_(coef_type == 2),
     testGradient_(false),
     dim_(H1_FESpace.GetParMesh()->Dimension()),
     tsize_(H1_FESpace.GetTrueVSize()),
     qsize_(HDiv_FESpace.GetTrueVSize()),
     multCount_(0), solveCount_(0),
     T_(&H1_FESpace),
     q_(&HDiv_FESpace),
     Q_perp_(&L2_FESpace),
     TCoef_(&T_),
     unitBCoef_(&UnitB),
     bbTCoef_(*unitBCoef_, *unitBCoef_),
     chiPerpCoef_(bbTCoef_, TCoef_, chi_perp, coef_type != 0),
     chiParaCoef_(bbTCoef_, TCoef_, chi_para, coef_type != 0),
     chiCoef_(chiPerpCoef_, chiParaCoef_),
     // dChiCoef_(bbTCoef_, TCoef_, chi_perp, chi_para),
     dChiParaCoef_(bbTCoef_, TCoef_, chi_para),
     chiInvPerpCoef_(bbTCoef_, TCoef_, chi_perp, coef_type != 0),
     chiInvParaCoef_(bbTCoef_, TCoef_, chi_para, coef_type != 0),
     chiInvCoef_(chiInvPerpCoef_, chiInvParaCoef_),
     HDiv_FESpace_(&HDiv_FESpace),
     L2_FESpace_(&L2_FESpace),
     mK_(NULL), sC_(NULL), dC_(NULL), a_(NULL), Div_(NULL),
     dqdt_gf_(NULL), Qs_(NULL),
     MKInv_(NULL), MKDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     dqdt_(&HDiv_FESpace),
     // rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dqdtBdrCoef_(&dqdtBdr),
     tdQ_(td_Q), tdC_(td_c),
     QCoef_(&Q), CCoef_(&c),
     CInvCoef_(new InverseCoefficient(c)),
     dtCInvCoef_(NULL),
     impOp_(H1_FESpace,
            dTdtBdr, false,
            bdr_attr,
            c, td_c,
            chiParaCoef_, coef_type > 0,
	    dChiParaCoef_, coef_type > 0,
            Q, td_Q || true,
            coef_type == 2),
     newton_(H1_FESpace.GetComm())
{
   this->init();
}

HybridThermalDiffusionTDO::~HybridThermalDiffusionTDO()
{
   delete CInvCoef_;
   delete dtCInvCoef_;
   delete Div_;
   delete dC_;
   delete a_;
   delete mK_;
   delete sC_;
   delete dqdt_gf_;
   delete Qs_;
   delete MKInv_;
   delete MKDiag_;
   delete AInv_;
   delete APrecond_;
}

void
HybridThermalDiffusionTDO::init()
{
   cout << "Entering TDO::Init" << endl;
   if ( init_ ) { return; }

   if ( mK_ == NULL )
   {
      mK_ = new ParBilinearForm(HDiv_FESpace_);
      mK_->AddDomainIntegrator(new VectorFEMassIntegrator(chiInvCoef_));
      mK_->Assemble();
   }

   if ( sC_ == NULL )
   {
      sC_ = new ParBilinearForm(HDiv_FESpace_);
      sC_->AddDomainIntegrator(new DivDivIntegrator(*CInvCoef_));
      sC_->Assemble();
   }
   if ( dC_ == NULL )
   {
      dC_ = new ParMixedBilinearForm(L2_FESpace_, HDiv_FESpace_);
      dC_->AddDomainIntegrator(
         new MixedScalarWeakGradientIntegrator(*CInvCoef_));
      dC_->Assemble();
   }
   if ( dqdt_gf_ == NULL )
   {
      dqdt_gf_ = new ParGridFunction(HDiv_FESpace_);
   }
   if ( Qs_ == NULL && QCoef_ != NULL )
   {
      Qs_ = new ParGridFunction(L2_FESpace_);
      Qs_->ProjectCoefficient(*QCoef_);
   }

   Div_ = new ParDiscreteDivOperator(HDiv_FESpace_, L2_FESpace_);
   Div_->Assemble();
   Div_->Finalize();

   rhs_.SetSize(HDiv_FESpace_->GetVSize());
   dQs_.SetSize(HDiv_FESpace_->GetVSize());
   tmp_.SetSize(L2_FESpace_->GetVSize());

   HDiv_FESpace_->GetEssentialTrueDofs(*bdr_attr_, ess_bdr_tdofs_);

   newton_.SetPrintLevel(2);
   newton_.SetRelTol(1e-10);
   newton_.SetAbsTol(0.0);

   if ( nonLinear_ && testGradient_ )
   {
      Vector x(impOp_.Height());
      Vector dx(impOp_.Height());

      T_.Distribute(x);
      Q_perp_ = 0.0;
      cout << "GetTime " << this->GetTime() << endl;
      impOp_.SetState(T_, Q_perp_, this->GetTime(), 0.1);

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
HybridThermalDiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   dqdtBdrCoef_->SetTime(t);

   if ( tdQ_ )
   {
      QCoef_->SetTime(t);
      Qs_->ProjectCoefficient(*QCoef_);
   }

   if ( tdC_ )
   {
      // CCoef_->SetTime(t);
      // CInvCoef_->SetTime(t);
      dtCInvCoef_->SetTime(t);
      sC_->Assemble();
   }

   chiInvCoef_.SetTime(t);
   mK_->Assemble();

   if ( tdC_ && a_ != NULL )
   {
      a_->Assemble();
   }

   newTime_ = true;
}

void
HybridThermalDiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   MFEM_ABORT("HybridThermalDiffusionTDO::Mult should not be called");
}

void
HybridThermalDiffusionTDO::initA(double dt)
{
   cout << "Entering initA" << endl;
   if ( CInvCoef_ != NULL )
   {
      dtCInvCoef_ = new ScaledCoefficient(dt, *CInvCoef_);
   }
   if ( a_ == NULL)
   {
      a_ = new ParBilinearForm(HDiv_FESpace_);
      a_->AddDomainIntegrator(new VectorFEMassIntegrator(chiInvCoef_));
      a_->AddDomainIntegrator(new DivDivIntegrator(*dtCInvCoef_));
      a_->Assemble();
   }
   else
   {
      a_->Update();
      a_->Assemble();
   }
   cout << "Leaving initA" << endl;
}

void
HybridThermalDiffusionTDO::initImplicitSolve()
{
  cout << "Entering initImplicitSolve" << endl;
   if ( tdC_ || AInv_ == NULL || APrecond_ == NULL )
   {
      delete AInv_;
      AInv_ = new HyprePCG(A_);
      AInv_->SetTol(1e-12);
      AInv_->SetMaxIter(200);
      AInv_->SetPrintLevel(0);

      delete APrecond_;
      APrecond_ = (dim_==2) ?
                  (HypreSolver*)(new HypreAMS(A_, HDiv_FESpace_)):
                  (HypreSolver*)(new HypreADS(A_, HDiv_FESpace_));

      if ( dim_ == 2 )
      {
         dynamic_cast<HypreAMS*>(APrecond_)->SetPrintLevel(0);
      }
      else
      {
         dynamic_cast<HypreADS*>(APrecond_)->SetPrintLevel(0);
      }
      AInv_->SetPreconditioner(*APrecond_);
   }
   cout << "Leaving initImplicitSolve" << endl;
}

void
HybridThermalDiffusionTDO::ImplicitSolve(const double dt,
					 const Vector &X, Vector &dX_dt)
{
   cout << "Entering ImplicitSolve" << endl;
   Vector T(X.GetData(), tsize_);
   Vector q(&(X.GetData())[tsize_], qsize_);
   Vector dT_dt(dX_dt.GetData(), tsize_);
   Vector dq_dt(&(dX_dt.GetData())[tsize_], qsize_);
   cout << 1 << endl;   
   dX_dt = 0.0;
   cout << 2 << endl;   

   T_.Distribute(T);

   {
     // q_.MakeRef(const_cast<ParFiniteElementSpace*>(HDiv_FESpace_),
     //		const_cast<Vector&>(y), 0);
     // u_.MakeRef(const_cast<ParFiniteElementSpace*>(L2_FESpace_),
     //		const_cast<Vector&>(y), HDiv_FESpace_->GetVSize());
     q_.Distribute(q);
     // dqdt_.MakeRef(HDiv_FESpace_, dy_dt, 0);
     // dudt_.MakeRef(L2_FESpace_, dy_dt, HDiv_FESpace_->GetVSize());

     // cout << "sC size: " << sC_->Width() << ", q_ size: " << q_.Size() << ", rhs_ size: " << rhs_.Size() << endl;
   cout << 3 << endl;   
     sC_->Mult(q_, rhs_);
     dC_->Mult(*Qs_, dQs_);
     rhs_ += dQs_;
     rhs_ *= -1.0;
   cout << 4 << endl;   
     // dqdt_gf_->ProjectBdrCoefficientNormal(*dqdtBdrCoef_, *bdr_attr_);
   dqdt_.ProjectBdrCoefficientNormal(*dqdtBdrCoef_, *bdr_attr_);
   cout << 5 << endl;   
     this->initA(dt);

     // a_->FormLinearSystem(ess_bdr_tdofs_, *dqdt_gf_, rhs_, A_, X_, RHS_);
     a_->FormLinearSystem(ess_bdr_tdofs_, dqdt_, rhs_, A_, X_, RHS_);

     this->initImplicitSolve();
     
     AInv_->Mult(RHS_, X_);

     a_->RecoverFEMSolution(X_, rhs_, dqdt_);

     // Div_->Mult(q_, dudt_);
     // Div_->Mult(dqdt_, tmp_);
     // tmp_  *= dt;
     // dudt_ += tmp_;
     // dudt_ *= -1.0;
     // dudt_ += *Qs_;
   }
   
   impOp_.SetState(T_, Q_perp_, this->GetTime(), dt);

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
                               ChiParaCoef & chi, bool tdChi,
			       dChiParaCoef & dchi, bool tdDChi,
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
     QPerpCoef_(NULL),
     QCoef_(heatSource, QPerpCoef_),
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
   cout << "Entering ImplicitDiffOp c'tor" << endl;
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

   cout << "Qs 0" << endl;
   Qs_.AddDomainIntegrator(new DomainLFIntegrator(QCoef_));
   cout << "Qs 1 " << tdQ_ << endl;
   if (!tdQ_) { Qs_.Assemble(); }
   cout << "Leaving ImplicitDiffOp c'tor" << endl;
}

ImplicitDiffOp::~ImplicitDiffOp()
{
   delete AInv_;
   delete APrecond_;
}

  void ImplicitDiffOp::SetState(ParGridFunction & T, ParGridFunction & Q_perp,
				double t, double dt)
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
      QCoef_.SetQPerp(Q_perp);
      QCoef_.SetTime(t_ + dt_);
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
