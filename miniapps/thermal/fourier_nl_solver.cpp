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

#include "fourier_nl_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace miniapps;

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
     double cx = cos(M_PI * x[0]);
     double cy = cos(M_PI * x[1]);
     double sx = sin(M_PI * x[0]);
     double sy = sin(M_PI * x[1]);

     V[0] = -sx * cy;
     V[1] =  sy * cx;
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
   int coef_type,
   Coefficient & c, bool td_c,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FESpace.GetTrueVSize(), 0.0),
     init_(false),
     nonLinear_(coef_type == 2),
     multCount_(0), solveCount_(0),
     // H1_FESpace_(&H1_FESpace),
     T_(&H1_FESpace),
     // dTdt_gf_(NULL),
     TCoef_(&T_),
     unitBCoef_(prob),
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
     newton_(H1_FESpace.GetComm())//,
     // rhs_(NULL),
     // bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0)
{
   this->init();
}

ThermalDiffusionTDO::~ThermalDiffusionTDO()
{
  // delete dTdt_gf_;
}

void
ThermalDiffusionTDO::init()
{
   if ( init_ ) { return; }
   /*
   if ( mC_ == NULL )
   {
      mC_ = new ParBilinearForm(H1_FESpace_);
      mC_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      mC_->Assemble();
   }
   */
   // if ( sK_ == NULL )
   // {
   //  sK_ = new ParBilinearForm(H1_FESpace_);
   //  sK_->AddDomainIntegrator(new DiffusionIntegrator(KCoef_));
      /*
      if ( kCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
      }
      else if ( KCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*KCoef_));
      }
      sK_->Assemble();
      */
   // }
   /*
   if ( dTdt_gf_ == NULL )
   {
      dTdt_gf_ = new ParGridFunction(H1_FESpace_);
   }
   */
   /*
   if ( Qs_ == NULL && QCoef_ != NULL )
   {
      Qs_ = new ParLinearForm(H1_FESpace_);
      Qs_->AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
      Qs_->Assemble();
      rhs_ = new Vector(Qs_->Size());
   }
   */
   /*
   CInvCoef_ = new InverseCoefficient(*CCoef_);
   if ( kCoef_ != NULL ) kInvCoef_ = new InverseCoefficient(*kCoef_);
   if ( KCoef_ != NULL ) KInvCoef_ = new MatrixInverseCoefficient(*KCoef_);
   */
   // H1_FESpace_->GetEssentialTrueDofs(*bdr_attr_, ess_bdr_tdofs_);

   newton_.SetPrintLevel(2);

   // newton_.SetOperator(impOp_);
   // newton_.SetSolver(impOp_.GetGradientSolver());

   init_ = true;
}

void
ThermalDiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   // impOp_.SetTime(time);
   // dTdtBdrCoef_->SetTime(t);
   /*
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
   */
   /*
   if ( tdK_ )
   {
      if ( kCoef_ != NULL ) { kCoef_->SetTime(t); }
      if ( KCoef_ != NULL ) { KCoef_->SetTime(t); }
      sK_->Assemble();
   }
   */
   /*
   if ( ( tdC_ || tdK_ ) && a_ != NULL )
   {
      a_->Assemble();
   }
   */
   newTime_ = true;
}
/*
void
ThermalDiffusionTDO::SetHeatSource(Coefficient & Q, bool time_dep)
{
if ( ownsQ_ )
  {
    delete QCoef_;
  }

tdQ_   = time_dep;
QCoef_ = &Q;
}

void
ThermalDiffusionTDO::SetConductivityCoefficient(Coefficient & k,
                  bool time_dep)
{
if ( ownsK_ )
  {
    delete kCoef_;
    delete KCoef_;
  }

tdK_   = time_dep;
kCoef_ = &k;
KCoef_ = NULL;
}

void
ThermalDiffusionTDO::SetConductivityCoefficient(MatrixCoefficient & K,
                  bool time_dep)
{
if ( ownsK_ )
  {
    delete kCoef_;
    delete KCoef_;
  }

tdK_   = time_dep;
kCoef_ = NULL;
KCoef_ = &K;
}

void
ThermalDiffusionTDO::SetSpecificHeatCoefficient(Coefficient & c,
                  bool time_dep)
{
if ( ownsC_ )
  {
    delete CCoef_;
  }

tdC_   = time_dep;
CCoef_ = &c;
}
*/
/*
void
ThermalDiffusionTDO::initMult() const
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
*/
void
ThermalDiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   MFEM_ABORT("ThermalDiffusionTDO::Mult should not be called");
   /*
   dT_dt = 0.0;

   T_ = T;
   sK_->Assemble();
   sK_->Mult(T, *rhs_);

   *rhs_ -= *Qs_;
   rhs_->Neg();

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   mC_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, MC_, dTdt_, RHS_);

   this->initMult();

   MCInv_->Mult(RHS_, dTdt_);

   mC_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   multCount_++;
   */
}
  /*
void
ThermalDiffusionTDO::initA(double dt)
{
  */
  /*
   if ( kCoef_ != NULL )
   {
      dtkCoef_ = new ScaledCoefficient(dt, *kCoef_);
   }
   else
   {
      dtKCoef_ = new ScaledMatrixCoefficient(dt, *KCoef_);
   }
   */
  // dtKCoef_ = new ScaledMatrixCoefficient(dt, KCoef_);
   /*	
   if ( a_ == NULL)
   {
      a_ = new ParNonlinearForm(H1_FESpace_);
      a_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      */
      /*
      if ( kCoef_ != NULL)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtkCoef_));
      }
      else
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtKCoef_));
      }

      a_->Assemble();
      */
   // }
  //}
  /*
void
ThermalDiffusionTDO::initImplicitSolve()
{
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
}
  */
void
ThermalDiffusionTDO::ImplicitSolve(const double dt,
				   const Vector &T, Vector &dT_dt)
{
  cout << "Entering ImplicitSolve" << endl;
   dT_dt = 0.0;
   // cout << "sK size: " << sK_->Width() << ", T size: " << T.Size() << ", rhs_ size: " << rhs_->Size() << endl;
   //
   // T_ = T;
   ostringstream ossT; ossT << "T_nl_" << solveCount_ << ".vec";
   ofstream ofsT(ossT.str().c_str());
   T.Print(ofsT);
   ofsT.close();

   T_.Distribute(T);
   /*
   sK_->Assemble();
   sK_->Mult(T, *rhs_);

   *rhs_ -= *Qs_;
   *rhs_ *= -1.0;

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   this->initA(dt);

   a_->Assemble();
   a_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, A_, dTdt_, RHS_);

   this->initImplicitSolve();

   // AInv_->Mult(RHS_, dTdt_);

   a_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);
   */
   // impOp_.SetTimeStep(dt);
   cout << "Setting state" << endl;
   impOp_.SetState(T_, this->GetTime(), dt);

   cout << "GetGradSolver" << endl;
   Solver & solver = impOp_.GetGradientSolver();

   if (!nonLinear_)
   {
     cout << "Calling solver.Mult" << endl;
     solver.Mult(impOp_.GetRHS(), dT_dt);
     cout << "Done Calling solver.Mult" << endl;
   }
   else
   {
     cout << "SetOperator" << endl;
     newton_.SetOperator(impOp_);
     cout << "SetSolver" << endl;
     newton_.SetSolver(solver);

     cout << "Mult with b.Size(): " << impOp_.GetRHS().Size() << endl;
     newton_.Mult(impOp_.GetRHS(), dT_dt);
   }
   solveCount_++;
  cout << "Leaving ImplicitSolve" << endl;
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
    gradTCoef_(&T0_),
    dtGradTCoef_(-1.0, gradTCoef_),
    dtdChiGradTCoef_(*dChiCoef_, dtGradTCoef_),
    m0cp_(&H1_FESpace),
    s0chi_(&H1_FESpace),
    a0_(&H1_FESpace),
    dTdt_(&H1_FESpace),
    Q_(&H1_FESpace),
    rhs_(&H1_FESpace),
    // Q_RHS_(H1_FESpace.GetTrueVSize()),
    RHS_(H1_FESpace.GetTrueVSize()),
    RHS0_(0),
    AInv_(NULL),
    APrecond_(NULL)
    // grad_(NULL),
    // solver_(NULL)
{
  H1_FESpace.GetEssentialTrueDofs(ess_bdr_attr_, ess_bdr_tdofs_);

  m0cp_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
  s0chi_.AddDomainIntegrator(new DiffusionIntegrator(*chiCoef_));

  a0_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
  a0_.AddDomainIntegrator(new DiffusionIntegrator(dtChiCoef_));
  if (nonLinear_)
  {
    a0_.AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(dtdChiGradTCoef_));
  }
  
  Q_.AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
  if (!tdQ_) { Q_.Assemble(); }
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
    s0chi_.Assemble();
    s0chi_.Finalize();

    ofstream ofsS0("s0_lin_initial.mat");
    s0chi_.SpMat().Print(ofsS0);

    a0_.Update();
    a0_.Assemble();
    a0_.Finalize();
  }
  
  if ((tdQ_ && newTime_) || first_)
  {
    cout << "Assembling Q" << endl;
     QCoef_->SetTime(t_ + dt_);
     Q_.Assemble();
     cout << "Norm of Q: " << Q_.Norml2() << endl;
  }

  first_       = false;
  newTime_     = false;
  newTimeStep_ = false;
}

void ImplicitDiffOp::Mult(const Vector &dT, Vector &Q) const
{
  cout << "Entering ImplicitDiffOp::Mult" << endl;
  add(T0_, dt_, dT, T1_);

  if (tdChi_ && nonLinear_)
  {
    chiCoef_->SetTemp(T1_);
    s0chi_.Update();
    s0chi_.Assemble();
    s0chi_.Finalize();
  }

  m0cp_.Mult(dT, Q);
  s0chi_.AddMult(T1_, Q);
  Q -= Q_;
  cout << "Leaving ImplicitDiffOp::Mult with Q: " << Q.Norml2() << endl;
}

Operator & ImplicitDiffOp::GetGradient(const Vector &dT) const
{
  cout << "Entering GetGradient" << endl;
  if (tdChi_)
  {
    if (!nonLinear_)
    {
      chiCoef_->SetTemp(T0_);
    }
    else
    {
      add(T0_, dt_, dT, T1_);

      chiCoef_->SetTemp(T1_);
      dChiCoef_->SetTemp(T1_);
      gradTCoef_.SetGridFunction(&T1_);
    }
    s0chi_.Update();
    s0chi_.Assemble();
    s0chi_.Finalize();

    a0_.Update();
    a0_.Assemble();
    a0_.Finalize();
  }

  if (!nonLinear_)
  {
    ofstream ofsS0("s0.mat");
    s0chi_.SpMat().Print(ofsS0);

    // rhs_ = Q_;
    // s0chi_.AddMult(T0_, rhs_, -1.0);
    s0chi_.Mult(T0_, rhs_);

    ofstream ofsT("T0.vec");
    T0_.Print(ofsT);

    ofstream ofsrhs("rhs_nl.vec");
    rhs_.Print(ofsrhs);

    ofstream ofsQ("Q_nl.vec");
    Q_.Print(ofsQ);
    
    rhs_ -= Q_;
    rhs_ *= -1.0;
  }
  
  cout << "Project bdr" << endl;
  dTdt_.ProjectBdrCoefficient(*bdrCoef_, ess_bdr_attr_);
  cout << "Form Lin Sys" << endl;
  a0_.FormLinearSystem(ess_bdr_tdofs_, dTdt_, rhs_, A_, dTdt_, RHS_);
  A_.Print("A_nl.mat");
  ofstream ofsB("b_nl.vec");
  RHS_.Print(ofsB);
  
  cout << "Norm of RHS: " << RHS_.Norml2() << endl;
  cout << "Leaving GetGradient" << endl;
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
         AInv_pcg->SetTol(1e-12);
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
       GMRESSolver * AInv_gmres = NULL;

       cout << "Building GMRES" << endl;
       AInv_gmres = new GMRESSolver(T0_.ParFESpace()->GetComm());
       AInv_gmres->SetRelTol(1e-12);
       AInv_gmres->SetAbsTol(0.0);
       AInv_gmres->SetMaxIter(200);
       AInv_gmres->SetPrintLevel(0);
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
