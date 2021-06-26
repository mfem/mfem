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

#include "fourier_flux_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

namespace thermal
{

ThermalDiffusionFluxOperator::ThermalDiffusionFluxOperator(
   ParMesh & pmesh,
   ParFiniteElementSpace &HDiv_FES,
   ParFiniteElementSpace &L2_FES,
   VectorCoefficient & dqdtBdr,
   Array<int> & bdr_attr,
   Coefficient & c, bool td_c,
   Coefficient & k, bool td_k,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(HDiv_FES.GetVSize() + L2_FES.GetVSize(), 0.0),
     init_(false), //initA_(false), initAInv_(false),
     dim_(pmesh.Dimension()),
     multCount_(0), solveCount_(0),
     HDiv_FESpace_(&HDiv_FES),
     L2_FESpace_(&L2_FES),
     mK_(NULL), sC_(NULL), dC_(NULL), a_(NULL), Div_(NULL),
     dqdt_gf_(NULL), Qs_(NULL),
     MKInv_(NULL), MKDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     // rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dqdtBdrCoef_(&dqdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k),
     QCoef_(&Q), CCoef_(&c), kCoef_(&k), KCoef_(NULL),
     // CInvCoef_(NULL), kInvCoef_(NULL), KInvCoef_(NULL)
     CInvCoef_(new InverseCoefficient(c)),
     kInvCoef_(new InverseCoefficient(k)), KInvCoef_(NULL),
     dtCInvCoef_(NULL)
{
   this->init();
}

ThermalDiffusionFluxOperator::ThermalDiffusionFluxOperator(
   ParMesh & pmesh,
   ParFiniteElementSpace &HDiv_FES,
   ParFiniteElementSpace &L2_FES,
   VectorCoefficient & dqdtBdr,
   Array<int> & bdr_attr,
   Coefficient & c, bool td_c,
   MatrixCoefficient & K, bool td_k,
   Coefficient & Q, bool td_Q)
   : TimeDependentOperator(HDiv_FES.GetVSize() + L2_FES.GetVSize(), 0.0),
     init_(false),
     dim_(pmesh.Dimension()),
     multCount_(0), solveCount_(0),
     HDiv_FESpace_(&HDiv_FES),
     L2_FESpace_(&L2_FES),
     mK_(NULL), sC_(NULL), dC_(NULL), a_(NULL), Div_(NULL),
     dqdt_gf_(NULL), Qs_(NULL),
     MKInv_(NULL), MKDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     // rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dqdtBdrCoef_(&dqdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k),
     QCoef_(&Q), CCoef_(&c), kCoef_(NULL), KCoef_(&K),
     CInvCoef_(new InverseCoefficient(c)),
     kInvCoef_(NULL),
     KInvCoef_(new MatrixInverseCoefficient(K)),
     dtCInvCoef_(NULL)
{
   this->init();
}

ThermalDiffusionFluxOperator::~ThermalDiffusionFluxOperator()
{
   delete CInvCoef_;
   delete kInvCoef_;
   delete KInvCoef_;
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
ThermalDiffusionFluxOperator::init()
{
   if ( init_ ) { return; }

   if ( mK_ == NULL )
   {
      mK_ = new ParBilinearForm(HDiv_FESpace_);
      if ( kCoef_ != NULL )
      {
         mK_->AddDomainIntegrator(new VectorFEMassIntegrator(*kInvCoef_));
      }
      else
      {
         mK_->AddDomainIntegrator(new VectorFEMassIntegrator(*KInvCoef_));
      }
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

   init_ = true;
}

void
ThermalDiffusionFluxOperator::SetTime(const double time)
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

   if ( tdK_ )
   {
      if ( kCoef_ != NULL ) { kCoef_->SetTime(t); kInvCoef_->SetTime(t); }
      if ( KCoef_ != NULL ) { KCoef_->SetTime(t); KInvCoef_->SetTime(t); }
      mK_->Assemble();
   }

   if ( ( tdC_ || tdK_ ) && a_ != NULL )
   {
      a_->Assemble();
   }

   newTime_ = true;
}
/*
void
ThermalDiffusionFluxOperator::SetHeatSource(Coefficient & Q, bool time_dep)
{
if ( ownsQ_ )
  {
    delete QCoef_;
  }

tdQ_   = time_dep;
QCoef_ = &Q;
}

void
ThermalDiffusionFluxOperator::SetConductivityCoefficient(Coefficient & k,
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
ThermalDiffusionFluxOperator::SetConductivityCoefficient(MatrixCoefficient & K,
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
ThermalDiffusionFluxOperator::SetSpecificHeatCoefficient(Coefficient & c,
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
void
ThermalDiffusionFluxOperator::initMult() const
{
   if ( tdC_ || MKInv_ == NULL || MKDiag_ == NULL )
   {
      if ( MKInv_ == NULL )
      {
         MKInv_ = new HyprePCG(MK_);
         MKInv_->SetTol(1e-12);
         MKInv_->SetMaxIter(200);
         MKInv_->SetPrintLevel(0);
      }
      else
      {
         MKInv_->SetOperator(MK_);
      }
      if ( MKDiag_ == NULL )
      {
         MKDiag_ = new HypreDiagScale(MK_);
         MKInv_->SetPreconditioner(*MKDiag_);
      }
      else
      {
         MKDiag_->SetOperator(MK_);
      }
   }
}

void
ThermalDiffusionFluxOperator::Mult(const Vector &y, Vector &dy_dt) const
{
   cout << "Entering Mult" << endl;
   dy_dt = 0.0;

   q_.MakeRef(const_cast<ParFiniteElementSpace*>(HDiv_FESpace_),
              const_cast<Vector&>(y), 0);
   u_.MakeRef(const_cast<ParFiniteElementSpace*>(L2_FESpace_),
              const_cast<Vector&>(y), HDiv_FESpace_->GetVSize());

   dqdt_.MakeRef(HDiv_FESpace_, dy_dt, 0);
   dudt_.MakeRef(L2_FESpace_, dy_dt, HDiv_FESpace_->GetVSize());

   sC_->Mult(q_, rhs_);
   dC_->Mult(*Qs_, dQs_);

   rhs_ += dQs_;
   rhs_.Neg();

   dqdt_gf_->ProjectBdrCoefficientNormal(*dqdtBdrCoef_, *bdr_attr_);

   mK_->FormLinearSystem(ess_bdr_tdofs_, *dqdt_gf_, rhs_, MK_, X_, RHS_);

   this->initMult();

   MKInv_->Mult(RHS_, X_);

   mK_->RecoverFEMSolution(X_, rhs_, dqdt_);

   Div_->Mult(q_, dudt_);
   dudt_ *= -1.0;
   dudt_ += *Qs_;

   multCount_++;

   cout << "Leaving Mult" << endl;
}

void
ThermalDiffusionFluxOperator::initA(double dt)
{
   if ( CInvCoef_ != NULL )
   {
      dtCInvCoef_ = new ScaledCoefficient(dt, *CInvCoef_);
   }
   if ( a_ == NULL)
   {
      a_ = new ParBilinearForm(HDiv_FESpace_);
      if ( kInvCoef_ != NULL)
      {
         a_->AddDomainIntegrator(new VectorFEMassIntegrator(*kInvCoef_));
      }
      else
      {
         a_->AddDomainIntegrator(new VectorFEMassIntegrator(*KInvCoef_));
      }

      a_->AddDomainIntegrator(new DivDivIntegrator(*dtCInvCoef_));
      a_->Assemble();
   }
   else if ( tdK_ )
   {
      a_->Update();
      a_->Assemble();
   }
}

void
ThermalDiffusionFluxOperator::initImplicitSolve()
{
   if ( tdC_ || tdK_ || AInv_ == NULL || APrecond_ == NULL )
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
}

void
ThermalDiffusionFluxOperator::ImplicitSolve(const double dt,
                                            const Vector &y, Vector &dy_dt)
{
   dy_dt = 0.0;

   q_.MakeRef(const_cast<ParFiniteElementSpace*>(HDiv_FESpace_),
              const_cast<Vector&>(y), 0);
   u_.MakeRef(const_cast<ParFiniteElementSpace*>(L2_FESpace_),
              const_cast<Vector&>(y), HDiv_FESpace_->GetVSize());

   dqdt_.MakeRef(HDiv_FESpace_, dy_dt, 0);
   dudt_.MakeRef(L2_FESpace_, dy_dt, HDiv_FESpace_->GetVSize());

   // cout << "sC size: " << sC_->Width() << ", q_ size: " << q_.Size() << ", rhs_ size: " << rhs_.Size() << endl;

   sC_->Mult(q_, rhs_);
   dC_->Mult(*Qs_, dQs_);
   rhs_ += dQs_;
   rhs_ *= -1.0;

   // dqdt_gf_->ProjectBdrCoefficientNormal(*dqdtBdrCoef_, *bdr_attr_);
   dqdt_.ProjectBdrCoefficientNormal(*dqdtBdrCoef_, *bdr_attr_);

   this->initA(dt);

   // a_->FormLinearSystem(ess_bdr_tdofs_, *dqdt_gf_, rhs_, A_, X_, RHS_);
   a_->FormLinearSystem(ess_bdr_tdofs_, dqdt_, rhs_, A_, X_, RHS_);

   this->initImplicitSolve();

   AInv_->Mult(RHS_, X_);

   a_->RecoverFEMSolution(X_, rhs_, dqdt_);

   Div_->Mult(q_, dudt_);
   Div_->Mult(dqdt_, tmp_);
   tmp_  *= dt;
   dudt_ += tmp_;
   dudt_ *= -1.0;
   dudt_ += *Qs_;

   solveCount_++;
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
