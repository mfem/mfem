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

#include "fourier_nl_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

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

void ChiParaCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);

   if (type_ == 0)
   {
      K *= chi_max_;
   }
   else
   {
      K *= chi_min_ * pow(1.0 + gamma_ * T_->Eval(T, ip), 2.5);
   }
}

void dChiCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                    const IntegrationPoint &ip)
{
   bbT_->Eval(K, T, ip);
   K *= 2.5 * chi_min_ * gamma_ * pow(1.0 + gamma_ * T_->Eval(T, ip), 1.5);
}

namespace thermal
{

ThermalDiffusionTDO::ThermalDiffusionTDO(
   ParFiniteElementSpace &H1_FESpace,
   Coefficient & dTdtBdr,
   Array<int> & bdr_attr,
   double chi_perp,
   double chi_para_min,
   double chi_para_max,
   int prob,
   int unit_vec_type,
   int coef_type,
   Coefficient & c, bool td_c,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FESpace.GetTrueVSize(), 0.0),
     init_(false),
     nonLinear_(coef_type == 2),
     testGradient_(false),
     multCount_(0), solveCount_(0),
     T_(&H1_FESpace),
     TCoef_(&T_),
     unitBCoef_(prob, unit_vec_type),
     ICoef_(2),
     bbTCoef_(unitBCoef_, unitBCoef_),
     chiPerpCoef_(ICoef_, bbTCoef_, chi_perp, -chi_perp),
     chiParaCoef_(bbTCoef_, TCoef_, coef_type, chi_para_min, chi_para_max),
     chiCoef_(chiPerpCoef_, chiParaCoef_),
     dChiCoef_(bbTCoef_, TCoef_, chi_para_min, chi_para_max),
     impOp_(H1_FESpace,
            dTdtBdr, false,
            bdr_attr,
            c, false,
            chiCoef_, coef_type > 0,
            dChiCoef_, coef_type > 0,
            Q,    false,
            coef_type == 2),
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

NLAdvectionDiffusionTDO::NLAdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
						 Coefficient & dTdtBdr,
						 Array<int> & bdr_attr,
						 Coefficient & c, bool td_c,
						 Coefficient & k, bool td_k,
						 VectorCoefficient & V,
						 bool td_v, double nu,
						 Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     myid_(H1_FES.GetMyRank()),
     init_(false), //initA_(false), initAInv_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), aV_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k), tdV_(td_v), nu_(nu),
     QCoef_(&Q), CCoef_(&c), kCoef_(&k), KCoef_(NULL), VCoef_(&V),
     // CInvCoef_(NULL), kInvCoef_(NULL), KInvCoef_(NULL)
     dtkCoef_(NULL), dtKCoef_(NULL), dtnuVCoef_(NULL)
{
   this->init();
}

NLAdvectionDiffusionTDO::NLAdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
						 Coefficient & dTdtBdr,
						 Array<int> & bdr_attr,
						 Coefficient & c, bool td_c,
						 MatrixCoefficient & K,
						 bool td_k,
						 VectorCoefficient & V,
						 bool td_v, double nu,
						 Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     init_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), aV_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k), tdV_(td_v), nu_(nu),
     QCoef_(&Q), CCoef_(&c), kCoef_(NULL), KCoef_(&K), VCoef_(&V),
     dtkCoef_(NULL), dtKCoef_(NULL), dtnuVCoef_(NULL)
{
   this->init();
}

NLAdvectionDiffusionTDO::~NLAdvectionDiffusionTDO()
{
   delete a_;
   delete aV_;
   delete mC_;
   delete sK_;
   delete dTdt_gf_;
   delete Qs_;
   delete MCInv_;
   delete MCDiag_;
   delete AInv_;
   delete APrecond_;
   delete dtkCoef_;
   delete dtKCoef_;
   delete dtnuVCoef_;
}

void
NLAdvectionDiffusionTDO::init()
{
   if ( init_ ) { return; }

   if ( mC_ == NULL )
   {
      mC_ = new ParBilinearForm(H1_FESpace_);
      mC_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      mC_->Assemble();
   }

   if ( sK_ == NULL )
   {
      sK_ = new ParBilinearForm(H1_FESpace_);
      if ( kCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
      }
      else if ( KCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*KCoef_));
      }
      sK_->Assemble();
   }
   if ( aV_ == NULL )
   {
      aV_ = new ParBilinearForm(H1_FESpace_);
      aV_->AddDomainIntegrator(
         new MixedScalarWeakDivergenceIntegrator(*VCoef_));
      aV_->Assemble();
   }
   if ( dTdt_gf_ == NULL )
   {
      dTdt_gf_ = new ParGridFunction(H1_FESpace_);
   }
   if ( Qs_ == NULL && QCoef_ != NULL )
   {
      Qs_ = new ParLinearForm(H1_FESpace_);
      Qs_->AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
      Qs_->Assemble();
      rhs_ = new Vector(Qs_->Size());
   }
   /*
   CInvCoef_ = new InverseCoefficient(*CCoef_);
   if ( kCoef_ != NULL ) kInvCoef_ = new InverseCoefficient(*kCoef_);
   if ( KCoef_ != NULL ) KInvCoef_ = new MatrixInverseCoefficient(*KCoef_);
   */
   H1_FESpace_->GetEssentialTrueDofs(*bdr_attr_, ess_bdr_tdofs_);

   init_ = true;
}

void
NLAdvectionDiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   dTdtBdrCoef_->SetTime(t);

   if ( tdQ_ )
   {
      QCoef_->SetTime(t);
      Qs_->Assemble();
   }

   if ( tdC_ )
   {
      CCoef_->SetTime(t);
      mC_->Assemble();
   }

   if ( tdK_ )
   {
      if ( kCoef_ != NULL ) { kCoef_->SetTime(t); }
      if ( KCoef_ != NULL ) { KCoef_->SetTime(t); }
      sK_->Assemble();
   }

   if ( tdV_ )
   {
      VCoef_->SetTime(t);
      aV_->Assemble();
   }

   if ( ( tdC_ || tdK_ || tdV_ ) && a_ != NULL )
   {
      a_->Assemble();
   }

   newTime_ = true;
}

void
NLAdvectionDiffusionTDO::initMult() const
{
   if ( tdC_ || MCInv_ == NULL || MCDiag_ == NULL )
   {
      if ( MCInv_ == NULL )
      {
         MCInv_ = new HyprePCG(MC_);
         MCInv_->SetTol(1e-12);
         MCInv_->SetMaxIter(200);
         MCInv_->SetPrintLevel(0);
      }
      else
      {
         MCInv_->SetOperator(MC_);
      }
      if ( MCDiag_ == NULL )
      {
         MCDiag_ = new HypreDiagScale(MC_);
         MCInv_->SetPreconditioner(*MCDiag_);
      }
      else
      {
         MCDiag_->SetOperator(MC_);
      }
   }
}

void
NLAdvectionDiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   dT_dt = 0.0;

   sK_->Mult(T, *rhs_);

   *rhs_ -= *Qs_;
   rhs_->Neg();

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   mC_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, MC_, dTdt_, RHS_);

   this->initMult();

   MCInv_->Mult(RHS_, dTdt_);

   mC_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   multCount_++;
}

void
NLAdvectionDiffusionTDO::initA(double dt)
{
   if ( kCoef_ != NULL )
   {
      if (dtkCoef_ == NULL)
      {
         dtkCoef_ = new ProductCoefficient(dt, *kCoef_);
      }
      dtkCoef_->SetAConst(dt);
   }
   else
   {
      if (dtKCoef_ == NULL)
      {
         dtKCoef_ = new ScalarMatrixProductCoefficient(dt, *KCoef_);
      }
      dtKCoef_->SetAConst(dt);
   }
   if (nu_ != 0.0)
   {
      if (dtnuVCoef_ == NULL)
      {
         dtnuVCoef_ = new ScalarVectorProductCoefficient(dt * nu_, *VCoef_);
      }
      dtnuVCoef_->SetAConst(dt * nu_);
   }

   if ( a_ == NULL)
   {
      a_ = new ParBilinearForm(H1_FESpace_);
      a_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      if ( kCoef_ != NULL)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtkCoef_));
      }
      else
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtKCoef_));
      }
      if (nu_ != 0.0)
      {
         a_->AddDomainIntegrator(
            new MixedScalarWeakDivergenceIntegrator(*dtnuVCoef_));
      }
   }
   a_->Update();
   a_->Assemble();
}

void
NLAdvectionDiffusionTDO::initImplicitSolve()
{
   delete AInv_;
   AInv_ = new HyprePCG(A_);
   AInv_->SetTol(1e-12);
   AInv_->SetMaxIter(200);
   AInv_->SetPrintLevel(0);

   delete APrecond_;
   APrecond_ = new HypreBoomerAMG(A_);
   APrecond_->SetPrintLevel(0);
   AInv_->SetPreconditioner(*APrecond_);
   /*
      if ( tdC_ || tdK_ || AInv_ == NULL || APrecond_ == NULL )
      {
         if ( AInv_ == NULL )
         {
            AInv_ = new HyprePCG(A_);
            AInv_->SetTol(1e-12);
            AInv_->SetMaxIter(200);
            AInv_->SetPrintLevel(0);
         }
         else
         {
            AInv_->SetOperator(A_);
         }
         if ( APrecond_ == NULL )
         {
            APrecond_ = new HypreBoomerAMG(A_);
            APrecond_->SetPrintLevel(0);
            AInv_->SetPreconditioner(*APrecond_);
         }
         else
         {
            APrecond_->SetOperator(A_);
         }
      }
     */
}

void
NLAdvectionDiffusionTDO::ImplicitSolve(const double dt,
				       const Vector &T, Vector &dT_dt)
{
   dT_dt = 0.0;
   // cout << "sK size: " << sK_->Width() << ", T size: " << T.Size() << ", rhs_ size: " << rhs_->Size() << endl;

   /*
   ostringstream ossT; ossT << "T_" << solveCount_ << ".vec";
   ofstream ofsT(ossT.str().c_str());
   T.Print(ofsT);
   ofsT.close();
   */
   sK_->Mult(T, *rhs_);
   aV_->AddMult(T, *rhs_);
   /*
   ofstream ofsrhs("rhs.vec");
   rhs_->Print(ofsrhs);

   ofstream ofsQ("Q.vec");
   Qs_->Print(ofsQ);
   */
   *rhs_ -= *Qs_;
   *rhs_ *= -1.0;

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   this->initA(dt);

   a_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, A_, dTdt_, RHS_);
   /*
   A_.Print("A.mat");
   ofstream ofsB("b.vec");
   RHS_.Print(ofsB);
   */
   this->initImplicitSolve();

   AInv_->Mult(RHS_, dTdt_);

   a_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   solveCount_++;
}

} // namespace thermal

} // namespace mfem

#endif // MFEM_USE_MPI
