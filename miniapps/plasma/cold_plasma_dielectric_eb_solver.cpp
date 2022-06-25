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

#include "cold_plasma_dielectric_eb_solver.hpp"
#include "cold_plasma_dielectric_coefs.hpp"
#include "../common/mesh_extras.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

ElectricEnergyDensityCoef::ElectricEnergyDensityCoef(VectorCoefficient &Er,
                                                     VectorCoefficient &Ei,
                                                     MatrixCoefficient &epsr,
                                                     MatrixCoefficient &epsi)
   : ErCoef_(Er),
     EiCoef_(Ei),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     eps_r_(3),
     eps_i_(3)
{}

double ElectricEnergyDensityCoef::Eval(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   epsrCoef_.Eval(eps_r_, T, ip);
   epsiCoef_.Eval(eps_i_, T, ip);
   /*
   if (T.ElementNo == 1)
   {
      cout << "eps_r" << endl;
      eps_r_.Print(std::cout, 3);
      cout << "eps_i" << endl;
      eps_i_.Print(std::cout, 3);
   }
   */
   eps_r_.Mult(Er_, Dr_);
   eps_i_.AddMult_a(-1.0, Ei_, Dr_);

   eps_i_.Mult(Er_, Di_);
   eps_r_.AddMult(Ei_, Di_);

   double u = (Er_ * Dr_) + (Ei_ * Di_);

   return 0.5 * u;
}

MagneticEnergyDensityCoef::MagneticEnergyDensityCoef(double omega,
                                                     VectorCoefficient &dEr,
                                                     VectorCoefficient &dEi,
                                                     Coefficient &muInv)
   : omega_(omega),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Br_(3),
     Bi_(3)
{}

double MagneticEnergyDensityCoef::Eval(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   dErCoef_.Eval(Bi_, T, ip); Bi_ /=  omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ /= -omega_;

   double muInv = muInvCoef_.Eval(T, ip);

   double u = ((Br_ * Br_) + (Bi_ * Bi_)) * muInv;

   return 0.5 * u;
}

EnergyDensityCoef::EnergyDensityCoef(double omega,
                                     VectorCoefficient &Er,
                                     VectorCoefficient &Ei,
                                     VectorCoefficient &dEr,
                                     VectorCoefficient &dEi,
                                     MatrixCoefficient &epsr,
                                     MatrixCoefficient &epsi,
                                     Coefficient &muInv)
   : omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     Br_(3),
     Bi_(3),
     eps_r_(3),
     eps_i_(3)
{}

double EnergyDensityCoef::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   dErCoef_.Eval(Bi_, T, ip); Bi_ /=  omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ /= -omega_;

   epsrCoef_.Eval(eps_r_, T, ip);
   epsiCoef_.Eval(eps_i_, T, ip);

   eps_r_.Mult(Er_, Dr_);
   eps_i_.AddMult_a(-1.0, Ei_, Dr_);

   eps_i_.Mult(Er_, Di_);
   eps_r_.AddMult(Ei_, Di_);

   double muInv = muInvCoef_.Eval(T, ip);

   double u = (Er_ * Dr_) + (Ei_ * Di_) + ((Br_ * Br_) + (Bi_ * Bi_)) * muInv;

   return 0.5 * u;
}

PoyntingVectorReCoef::PoyntingVectorReCoef(double omega,
                                           VectorCoefficient &Er,
                                           VectorCoefficient &Ei,
                                           VectorCoefficient &dEr,
                                           VectorCoefficient &dEi,
                                           Coefficient &muInv)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Hr_(3),
     Hi_(3)
{}

void PoyntingVectorReCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   double muInv = muInvCoef_.Eval(T, ip);

   dErCoef_.Eval(Hi_, T, ip); Hi_ *=  muInv / omega_;
   dEiCoef_.Eval(Hr_, T, ip); Hr_ *= -muInv / omega_;

   S.SetSize(3);

   S[0] = Er_[1] * Hr_[2] - Er_[2] * Hr_[1] +
          Ei_[1] * Hi_[2] - Ei_[2] * Hi_[1] ;

   S[1] = Er_[2] * Hr_[0] - Er_[0] * Hr_[2] +
          Ei_[2] * Hi_[0] - Ei_[0] * Hi_[2] ;

   S[2] = Er_[0] * Hr_[1] - Er_[1] * Hr_[0] +
          Ei_[0] * Hi_[1] - Ei_[1] * Hi_[0] ;

   S *= 0.5;
}

PoyntingVectorImCoef::PoyntingVectorImCoef(double omega,
                                           VectorCoefficient &Er,
                                           VectorCoefficient &Ei,
                                           VectorCoefficient &dEr,
                                           VectorCoefficient &dEi,
                                           Coefficient &muInv)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Hr_(3),
     Hi_(3)
{}

void PoyntingVectorImCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   double muInv = muInvCoef_.Eval(T, ip);

   dErCoef_.Eval(Hi_, T, ip); Hi_ *=  muInv / omega_;
   dEiCoef_.Eval(Hr_, T, ip); Hr_ *= -muInv / omega_;

   S.SetSize(3);

   S[0] = Er_[1] * Hi_[2] - Er_[2] * Hi_[1] -
          Ei_[1] * Hr_[2] + Ei_[2] * Hr_[1] ;

   S[1] = Er_[2] * Hi_[0] - Er_[0] * Hi_[2] -
          Ei_[2] * Hr_[0] + Ei_[0] * Hr_[2] ;

   S[2] = Er_[0] * Hi_[1] - Er_[1] * Hi_[0] -
          Ei_[0] * Hr_[1] + Ei_[1] * Hr_[0] ;

   S *= 0.5;
}

MinkowskiMomentumDensityReCoef::MinkowskiMomentumDensityReCoef(double omega,
                                                               VectorCoefficient &Er,
                                                               VectorCoefficient &Ei,
                                                               VectorCoefficient &dEr,
                                                               VectorCoefficient &dEi,
                                                               MatrixCoefficient &epsr,
                                                               MatrixCoefficient &epsi)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     Br_(3),
     Bi_(3),
     epsr_(3),
     epsi_(3)
{}

void MinkowskiMomentumDensityReCoef::Eval(Vector &G, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   dErCoef_.Eval(Bi_, T, ip); Bi_ *=  1.0 / omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ *= -1.0 / omega_;

   epsrCoef_.Eval(epsr_, T, ip);
   epsiCoef_.Eval(epsi_, T, ip);

   epsr_.Mult(Er_, Dr_);
   epsi_.AddMult_a(-1.0, Ei_, Dr_);

   epsr_.Mult(Ei_, Di_);
   epsi_.AddMult(Er_, Di_);

   G.SetSize(3);

   G[0] = Dr_[1] * Br_[2] - Dr_[2] * Br_[1] +
          Di_[1] * Bi_[2] - Di_[2] * Bi_[1] ;

   G[1] = Dr_[2] * Br_[0] - Dr_[0] * Br_[2] +
          Di_[2] * Bi_[0] - Di_[0] * Bi_[2] ;

   G[2] = Dr_[0] * Br_[1] - Dr_[1] * Br_[0] +
          Di_[0] * Bi_[1] - Di_[1] * Bi_[0] ;

   G *= 0.5;
}

MinkowskiMomentumDensityImCoef::MinkowskiMomentumDensityImCoef(double omega,
                                                               VectorCoefficient &Er,
                                                               VectorCoefficient &Ei,
                                                               VectorCoefficient &dEr,
                                                               VectorCoefficient &dEi,
                                                               MatrixCoefficient &epsr,
                                                               MatrixCoefficient &epsi)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     Br_(3),
     Bi_(3),
     epsr_(3),
     epsi_(3)
{}

void MinkowskiMomentumDensityImCoef::Eval(Vector &G, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   dErCoef_.Eval(Bi_, T, ip); Bi_ *=  1.0 / omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ *= -1.0 / omega_;

   epsrCoef_.Eval(epsr_, T, ip);
   epsiCoef_.Eval(epsi_, T, ip);

   epsr_.Mult(Er_, Dr_);
   epsi_.AddMult_a(-1.0, Ei_, Dr_);

   epsr_.Mult(Ei_, Di_);
   epsi_.AddMult(Er_, Di_);

   G.SetSize(3);

   G[0] = Dr_[1] * Br_[2] - Dr_[2] * Br_[1] -
          Di_[1] * Bi_[2] + Di_[2] * Bi_[1] ;

   G[1] = Dr_[2] * Br_[0] - Dr_[0] * Br_[2] -
          Di_[2] * Bi_[0] + Di_[0] * Bi_[2] ;

   G[2] = Dr_[0] * Br_[1] - Dr_[1] * Br_[0] -
          Di_[0] * Bi_[1] + Di_[1] * Bi_[0] ;

   G *= 0.5;
}

Maxwell2ndE::Maxwell2ndE(ParFiniteElementSpace & HCurlFESpace,
                         double omega,
                         ComplexOperator::Convention conv,
                         MatrixCoefficient & epsReCoef,
                         MatrixCoefficient & epsImCoef,
                         Coefficient & muInvCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef,
                         bool cyl,
                         bool pa)
   : ParSesquilinearForm(&HCurlFESpace, conv),
     cyl_(cyl),
     pa_(pa),
     epsReCylCoef_(epsReCoef, 1),
     epsImCylCoef_(epsImCoef, 1),
     muInvCylCoef_(muInvCoef, 1),
     massReCoef_(omega, cyl_ ? epsReCylCoef_ : epsReCoef),
     massImCoef_(omega, cyl_ ? epsImCylCoef_ : epsImCoef),
     kmkReCoef_(kReCoef, kImCoef, &muInvCoef, true, -1.0),
     kmkImCoef_(kReCoef, kImCoef, &muInvCoef, false, -1.0),
     kmReCoef_(kReCoef, &muInvCoef, 1.0),
     kmImCoef_(kImCoef, &muInvCoef, -1.0),
     kmkCylReCoef_(kReCoef, kImCoef, &muInvCylCoef_, true, -1.0),
     kmkCylImCoef_(kReCoef, kImCoef, &muInvCylCoef_, false, -1.0),
     kmCylReCoef_(kReCoef, &muInvCylCoef_, 1.0),
     kmCylImCoef_(kImCoef, &muInvCylCoef_, -1.0),
     mkCylReCoef_(&muInvCylCoef_, kReCoef, 1.0),
     mkCylImCoef_(&muInvCylCoef_, kImCoef, -1.0)
{
   if (pa_) { this->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   if (!cyl_)
   {
      this->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef), NULL);
   }
   else
   {
      this->AddDomainIntegrator(new CurlCurlIntegrator(muInvCylCoef_), NULL);
   }
   this->AddDomainIntegrator(new VectorFEMassIntegrator(massReCoef_),
                             new VectorFEMassIntegrator(massImCoef_));

   if ( kReCoef || kImCoef )
   {
      if (pa_)
      {
         MFEM_ABORT("kCoef: Partial Assembly has not yet been implemented for "
                    "MixedCrossCurlIntegrator and MixedWeakCurlCrossIntegrator.");
      }
      if (!cyl_)
      {
         this->AddDomainIntegrator(new VectorFEMassIntegrator(kmkReCoef_),
                                   new VectorFEMassIntegrator(kmkImCoef_));
         this->AddDomainIntegrator(new MixedVectorCurlIntegrator(kmImCoef_),
                                   new MixedVectorCurlIntegrator(kmReCoef_));
         this->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(kmImCoef_),
                                   new MixedVectorWeakCurlIntegrator(kmReCoef_));
      }
      else
      {
         this->AddDomainIntegrator(new VectorFEMassIntegrator(kmkCylReCoef_),
                                   new VectorFEMassIntegrator(kmkCylImCoef_));
         this->AddDomainIntegrator(new MixedVectorCurlIntegrator(kmCylImCoef_),
                                   new MixedVectorCurlIntegrator(kmCylReCoef_));
         this->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(mkCylImCoef_),
                                   new MixedVectorWeakCurlIntegrator(mkCylReCoef_));
      }
   }
}

void Maxwell2ndE::Assemble()
{
   this->ParSesquilinearForm::Assemble();
   if (!pa_) { this->Finalize(); }
}

CurrentSourceE::CurrentSourceE(ParFiniteElementSpace & HCurlFESpace,
                               ParFiniteElementSpace & HDivFESpace,
                               double omega,
                               ComplexOperator::Convention conv,
                               const CmplxVecCoefArray & jsrc,
                               const CmplxVecCoefArray & ksrc,
                               VectorCoefficient * kReCoef,
                               VectorCoefficient * kImCoef,
                               bool cyl,
                               bool pa)
   : ParComplexLinearForm(&HCurlFESpace, conv),
     omega_(omega),
     jt_(&HDivFESpace),
     kt_(&HCurlFESpace),
     jtilde_(jsrc.Size()),
     ktilde_(ksrc.Size()),
     jtildeCyl_(jsrc.Size()),
     ktildeCyl_(ksrc.Size())
{
   // In order to avoid constructing another set of coefficients the following
   // code will create integrators which use the incorrect sign for the real
   // part of the linear form. This will need to be corrected during assembly
   // when we will also scale by a factor of omega.

   for (int i = 0; i < jsrc.Size(); i++)
   {
      jtilde_[i] = new ComplexVectorCoefficientByAttr;
      jtilde_[i]->attr = jsrc[i]->attr;
      jtilde_[i]->attr_marker = jsrc[i]->attr_marker;
      jtilde_[i]->real = new ComplexPhaseVectorCoefficient(kReCoef, kImCoef,
                                                           jsrc[i]->real,
                                                           jsrc[i]->imag,
                                                           true, true);
      jtilde_[i]->imag = new ComplexPhaseVectorCoefficient(kReCoef, kImCoef,
                                                           jsrc[i]->real,
                                                           jsrc[i]->imag,
                                                           false, true);

      jtildeCyl_[i] = new ComplexVectorCoefficientByAttr;
      jtildeCyl_[i]->attr = jsrc[i]->attr;
      jtildeCyl_[i]->attr_marker = jsrc[i]->attr_marker;
      jtildeCyl_[i]->real =
         new MatrixVectorProductCoefficient(rhoSCoef_, *jtilde_[i]->real);
      jtildeCyl_[i]->imag =
         new MatrixVectorProductCoefficient(rhoSCoef_, *jtilde_[i]->imag);

      if (!cyl)
      {
         this->AddDomainIntegrator(new VectorFEDomainLFIntegrator(
                                      *jtilde_[i]->imag),
                                   new VectorFEDomainLFIntegrator(
                                      *jtilde_[i]->real),
                                   jtilde_[i]->attr_marker);
      }
      else
      {
         this->AddDomainIntegrator(new VectorFEDomainLFIntegrator(
                                      *jtildeCyl_[i]->imag),
                                   new VectorFEDomainLFIntegrator(
                                      *jtildeCyl_[i]->real),
                                   jtildeCyl_[i]->attr_marker);
      }
   }

   for (int i = 0; i < ksrc.Size(); i++)
   {
      ktilde_[i] = new ComplexVectorCoefficientByAttr;
      ktilde_[i]->attr = ksrc[i]->attr;
      ktilde_[i]->attr_marker = ksrc[i]->attr_marker;
      ktilde_[i]->real = new ComplexPhaseVectorCoefficient(kReCoef, kImCoef,
                                                           ksrc[i]->real,
                                                           ksrc[i]->imag,
                                                           true, true);
      ktilde_[i]->imag = new ComplexPhaseVectorCoefficient(kReCoef, kImCoef,
                                                           ksrc[i]->real,
                                                           ksrc[i]->imag,
                                                           false, true);

      ktildeCyl_[i] = new ComplexVectorCoefficientByAttr;
      ktildeCyl_[i]->attr = ksrc[i]->attr;
      ktildeCyl_[i]->attr_marker = ksrc[i]->attr_marker;
      ktildeCyl_[i]->real =
         new MatrixVectorProductCoefficient(rhoSCoef_, *ktilde_[i]->real);
      ktildeCyl_[i]->imag =
         new MatrixVectorProductCoefficient(rhoSCoef_, *ktilde_[i]->imag);

      if (!cyl)
      {
         this->AddBoundaryIntegrator(new VectorFEBoundaryTangentialLFIntegrator(
                                        *ktilde_[i]->imag),
                                     new VectorFEBoundaryTangentialLFIntegrator(
                                        *ktilde_[i]->real),
                                     ktilde_[i]->attr_marker);
      }
      else
      {
         this->AddBoundaryIntegrator(new VectorFEBoundaryTangentialLFIntegrator(
                                        *ktildeCyl_[i]->imag),
                                     new VectorFEBoundaryTangentialLFIntegrator(
                                        *ktildeCyl_[i]->real),
                                     ktildeCyl_[i]->attr_marker);
      }
   }

   this->real().Vector::operator=(0.0);
   this->imag().Vector::operator=(0.0);
}

CurrentSourceE::~CurrentSourceE()
{
   for (int i=0; i<jtilde_.Size(); i++)
   {
      delete jtilde_[i]->real;
      delete jtilde_[i]->imag;
      delete jtilde_[i];
   }
   for (int i=0; i<ktilde_.Size(); i++)
   {
      delete ktilde_[i]->real;
      delete ktilde_[i]->imag;
      delete ktilde_[i];
   }
}

void CurrentSourceE::Update()
{
   this->ParComplexLinearForm::Update();

   jt_.Update();
   kt_.Update();
}

void CurrentSourceE::Assemble()
{
   this->ParComplexLinearForm::Assemble();

   this->real() *= -omega_;
   this->imag() *=  omega_;

   jt_ = 0.0;
   for (int i=0; i<jtilde_.Size(); i++)
   {
      for (int j=0; j<jtilde_[i]->attr.Size(); j++)
      {
         int attr = jtilde_[i]->attr[j];
         jt_.real().ProjectCoefficient(*jtilde_[i]->real, attr);
         jt_.imag().ProjectCoefficient(*jtilde_[i]->imag, attr);
      }
   }

   kt_ = 0.0;
   for (int i=0; i<ktilde_.Size(); i++)
   {
      kt_.ProjectBdrCoefficientTangent(*ktilde_[i]->real,
                                       *ktilde_[i]->imag,
                                       ktilde_[i]->attr_marker);
   }
}

FaradaysLaw::FaradaysLaw(const ParComplexGridFunction &e,
                         ParFiniteElementSpace &HDivFESpace,
                         double omega,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef)
   : e_(e),
     b_(&HDivFESpace),
     omega_(omega),
     curl_(const_cast<ParFiniteElementSpace*>(e.ParFESpace()), &HDivFESpace),
     kReCross_(NULL),
     kImCross_(NULL)
{
   ParFiniteElementSpace & HCurlFESpace =
      const_cast<ParFiniteElementSpace&>(*e_.ParFESpace());

   if (kReCoef)
   {
      kReCross_ = new ParDiscreteLinearOperator(&HCurlFESpace, &HDivFESpace);
      kReCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kReCoef));
   }
   if (kImCoef)
   {
      kImCross_ = new ParDiscreteLinearOperator(&HCurlFESpace, &HDivFESpace);
      kImCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kImCoef));
   }
}

void FaradaysLaw::Update()
{
   b_.Update();

   curl_.Update();

   if (kReCross_) { kReCross_->Update(); }
   if (kImCross_) { kImCross_->Update(); }
}

void FaradaysLaw::Assemble()
{
   curl_.Assemble();
   curl_.Finalize();

   if (kReCross_)
   {
      kReCross_->Assemble();
      kReCross_->Finalize();
   }
   if (kImCross_)
   {
      kImCross_->Assemble();
      kImCross_->Finalize();
   }
}

void FaradaysLaw::ComputeB()
{
   // B = Curl(E) / (i omega) = -i Curl(E) / omega
   curl_.Mult(e_.imag(), b_.real()); b_.real() *=  1.0 / omega_;
   curl_.Mult(e_.real(), b_.imag()); b_.imag() *= -1.0 / omega_;

   if (kReCross_)
   {
      // B += i k x E / (i omega) = k x E / omega
      kReCross_->AddMult(e_.real(), b_.real(),  1.0 / omega_);
      kReCross_->AddMult(e_.imag(), b_.imag(),  1.0 / omega_);
   }

   if (kImCross_)
   {
      kImCross_->AddMult(e_.imag(), b_.real(), -1.0 / omega_);
      kImCross_->AddMult(e_.real(), b_.imag(),  1.0 / omega_);
   }
}

GausssLaw::GausssLaw(const ParComplexGridFunction &f,
                     ParFiniteElementSpace & L2FESpace,
                     VectorCoefficient * kReCoef,
                     VectorCoefficient * kImCoef)
   : f_(f),
     df_(&L2FESpace),
     div_(const_cast<ParFiniteElementSpace*>(f.ParFESpace()), &L2FESpace),
     kReDot_(NULL),
     kImDot_(NULL)
{
   ParFiniteElementSpace & HDivFESpace =
      const_cast<ParFiniteElementSpace&>(*f_.ParFESpace());

   if (kReCoef)
   {
      kReDot_ = new ParDiscreteLinearOperator(&HDivFESpace, &L2FESpace);
      kReDot_->AddDomainInterpolator(
         new VectorInnerProductInterpolator(*kReCoef));
   }
   if (kImCoef)
   {
      kImDot_ = new ParDiscreteLinearOperator(&HDivFESpace, &L2FESpace);
      kImDot_->AddDomainInterpolator(
         new VectorInnerProductInterpolator(*kImCoef));
   }
}

void GausssLaw::Update()
{
   df_.Update();
   div_.Update();

   if (kReDot_) { kReDot_->Update(); }
   if (kImDot_) { kImDot_->Update(); }
}

void GausssLaw::Assemble()
{
   div_.Assemble();
   div_.Finalize();

   if (kReDot_)
   {
      kReDot_->Assemble();
      kReDot_->Finalize();
   }
   if (kImDot_)
   {
      kImDot_->Assemble();
      kImDot_->Finalize();
   }
}

void GausssLaw::ComputeDiv()
{
   // df = Div(f)
   div_.Mult(f_.real(), df_.real());
   div_.Mult(f_.imag(), df_.imag());

   if (kReDot_)
   {
      // df += i Re(k) . f
      kReDot_->AddMult(f_.real(), df_.imag(),  1.0);
      kReDot_->AddMult(f_.imag(), df_.real(), -1.0);
   }
   if (kImDot_)
   {
      // df -= Im(k) . f
      kImDot_->AddMult(f_.real(), df_.real(), -1.0);
      kImDot_->AddMult(f_.imag(), df_.imag(), -1.0);
   }
}

Displacement::Displacement(const ParComplexGridFunction &e,
                           ParFiniteElementSpace & HDivFESpace,
                           MatrixCoefficient & epsReCoef,
                           MatrixCoefficient & epsImCoef,
                           bool pa)
   : pa_(pa),
     d_(&HDivFESpace),
     dReCoef_(&epsReCoef, &epsImCoef, &e.real(), &e.imag()),
     dImCoef_(&epsReCoef, &epsImCoef, &e.real(), &e.imag()),
     d_lf_(&HDivFESpace),
     m_(&HDivFESpace)
{
   d_lf_.AddDomainIntegrator(new VectorFEDomainLFIntegrator(dReCoef_),
                             new VectorFEDomainLFIntegrator(dImCoef_));

   if (pa_) { m_.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m_.AddDomainIntegrator(new VectorFEMassIntegrator);
}

void Displacement::Update()
{
   d_.Update();
   d_lf_.Update();
   m_.Update();
}

void Displacement::Assemble()
{
   m_.Assemble();
   if (!pa_) { m_.Finalize(); }
}

void Displacement::ComputeD()
{
   OperatorPtr M;

   Array<int> ess_tdof;
   m_.FormSystemMatrix(ess_tdof, M);

   int tvsize = d_.ParFESpace()->TrueVSize();

   D_.SetSize(tvsize);
   RHS_.SetSize(tvsize);

   Operator *diag = NULL;
   Operator *pcg = NULL;
   if (pa_)
   {
      diag = new OperatorJacobiSmoother(m_, ess_tdof);
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetOperator(*M);
      cg->SetPreconditioner(static_cast<OperatorJacobiSmoother&>(*diag));
      cg->SetRelTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }
   else
   {
      diag = new HypreDiagScale(*M.As<HypreParMatrix>());
      HyprePCG *cg = new HyprePCG(*M.As<HypreParMatrix>());
      cg->SetPreconditioner(static_cast<HypreDiagScale&>(*diag));
      cg->SetTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }

   d_lf_.Assemble();
   d_lf_.SyncAlias();

   d_lf_.real().ParallelAssemble(RHS_);
   D_ = 0.0;
   pcg->Mult(RHS_, D_);
   d_.real().Distribute(D_);

   d_lf_.imag().ParallelAssemble(RHS_);
   if (d_lf_.GetConvention() == ComplexOperator::BLOCK_SYMMETRIC)
   {
      RHS_ *= -1.0;
   }
   D_ = 0.0;
   pcg->Mult(RHS_, D_);
   d_.imag().Distribute(D_);

   delete diag;
   delete pcg;
}

ElectricEnergyDensityVisObject::ElectricEnergyDensityVisObject(
   const std::string & field_name,
   L2_ParFESpace *sfes,
   bool cyl, bool pseudo)
   : ScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void ElectricEnergyDensityVisObject::PrepareVisField(const
                                                     ParComplexGridFunction &e,
                                                     MatrixCoefficient &epsr,
                                                     MatrixCoefficient &epsi)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());

   ElectricEnergyDensityCoef ur(Er, Ei, epsr, epsi);
   ConstantCoefficient ui(0.0);

   this->PrepareVisField(ur, ui, NULL, NULL);
}

MagneticEnergyDensityVisObject::MagneticEnergyDensityVisObject(
   const std::string & field_name,
   L2_ParFESpace *sfes,
   bool cyl, bool pseudo)
   : ScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void MagneticEnergyDensityVisObject::PrepareVisField(const
                                                     ParComplexGridFunction &e,
                                                     double omega,
                                                     Coefficient &muInv)
{
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   MagneticEnergyDensityCoef ur(omega, dEr, dEi, muInv);
   ConstantCoefficient ui(0.0);

   this->PrepareVisField(ur, ui, NULL, NULL);
}

EnergyDensityVisObject::EnergyDensityVisObject(const std::string & field_name,
                                               L2_ParFESpace *sfes,
                                               bool cyl, bool pseudo)
   : ScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void EnergyDensityVisObject::PrepareVisField(const ParComplexGridFunction &e,
                                             double omega,
                                             MatrixCoefficient &epsr,
                                             MatrixCoefficient &epsi,
                                             Coefficient &muInv)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   EnergyDensityCoef ur(omega, Er, Ei, dEr, dEi, epsr, epsi, muInv);
   ConstantCoefficient ui(0.0);

   this->PrepareVisField(ur, ui, NULL, NULL);
}

PoyntingVectorVisObject::PoyntingVectorVisObject(const std::string & field_name,
                                                 L2_ParFESpace *vfes,
                                                 L2_ParFESpace *sfes,
                                                 bool cyl, bool pseudo)
   : ComplexVectorFieldVisObject(field_name, vfes, sfes, cyl, pseudo)
{}

void PoyntingVectorVisObject::PrepareVisField(const ParComplexGridFunction &e,
                                              double omega,
                                              Coefficient & muInvCoef)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   PoyntingVectorReCoef Sr(omega, Er, Ei, dEr, dEi, muInvCoef);
   PoyntingVectorImCoef Si(omega, Er, Ei, dEr, dEi, muInvCoef);

   this->PrepareVisField(Sr, Si, NULL, NULL);
}

MinkowskiMomentumDensityVisObject::MinkowskiMomentumDensityVisObject(
   const std::string & field_name,
   L2_ParFESpace *vfes,
   L2_ParFESpace *sfes,
   bool cyl, bool pseudo)
   : ComplexVectorFieldVisObject(field_name, vfes, sfes, cyl, pseudo)
{}

void MinkowskiMomentumDensityVisObject::PrepareVisField(
   const ParComplexGridFunction &e,
   double omega,
   MatrixCoefficient & epsr,
   MatrixCoefficient & epsi)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   MinkowskiMomentumDensityReCoef Gr(omega, Er, Ei, dEr, dEi, epsr, epsi);
   MinkowskiMomentumDensityImCoef Gi(omega, Er, Ei, dEr, dEi, epsr, epsi);

   this->PrepareVisField(Gr, Gi, NULL, NULL);
}

TensorCompVisObject::TensorCompVisObject(const std::string & field_name,
                                         L2_ParFESpace *sfes,
                                         bool cyl, bool pseudo)
   : ScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void TensorCompVisObject::PrepareVisField(MatrixCoefficient &mr,
                                          MatrixCoefficient &mi,
                                          int i, int j)
{
   TensorCompCoef mrCoef(mr, i, j);
   TensorCompCoef miCoef(mi, i, j);

   this->PrepareVisField(mrCoef, miCoef, NULL, NULL);
}

CPDSolverEB::CPDSolverEB(ParMesh & pmesh, int order, double omega,
                         CPDSolverEB::SolverType sol, SolverOptions & sOpts,
                         CPDSolverEB::PrecondType prec,
                         ComplexOperator::Convention conv,
                         VectorCoefficient & BCoef,
                         MatrixCoefficient & epsReCoef,
                         MatrixCoefficient & epsImCoef,
                         MatrixCoefficient & epsAbsCoef,
                         Coefficient & muInvCoef,
                         Coefficient * etaInvCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef,
                         Array<int> & abcs,
                         StixBCs & stixBCs,
                         bool vis_u, bool cyl, bool pa)
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(1),
     sol_(sol),
     solOpts_(sOpts),
     prec_(prec),
     conv_(conv),
     ownsEtaInv_(etaInvCoef == NULL),
     cyl_(cyl),
     vis_u_(vis_u),
     pa_(pa),
     omega_(omega),
     pmesh_(&pmesh),
     L2FESpace_(new L2_ParFESpace(pmesh_, order-1, pmesh_->Dimension())),
     L2FESpace2p_(new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension())),
     L2VSFESpace_(new L2_ParFESpace(pmesh_,order,pmesh_->Dimension(),
                                    pmesh_->SpaceDimension())),
     L2VSFESpace2p_(new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension(),
                                      pmesh_->SpaceDimension())),
     L2V3FESpace_(NULL),
     HCurlFESpace_(MakeHCurlParFESpace(pmesh, order)),
     HDivFESpace_(MakeHDivParFESpace(pmesh, order)),
     // HDivFESpace2p_(NULL),
     b1_(NULL),
     e_(HCurlFESpace_),
     e_t_(NULL),
     e_b_(NULL),
     e_v_("E", L2VSFESpace_, L2FESpace_, cyl_, false),
     b_v_("B", L2VSFESpace_, L2FESpace_, cyl_, true),
     db_v_("DivB", L2FESpace_, cyl_, true),
     d_v_("D", L2VSFESpace_, L2FESpace_, cyl_, true),
     dd_v_("DivD", L2FESpace_, cyl_, true),
     j_v_("J", L2VSFESpace_, L2FESpace_, cyl_, true),
     k_v_("K", L2VSFESpace_, L2FESpace_, cyl_, true),
     ue_v_("uE", L2FESpace_, cyl_, true),
     ub_v_("uB", L2FESpace_, cyl_, true),
     u_v_("u", L2FESpace_, cyl_, true),
     s_v_("S", L2VSFESpace_, L2FESpace_, cyl_, true),
     g_v_("G", L2VSFESpace_, L2FESpace_, cyl_, true),
     eps_00_v_("eps00", L2FESpace_, cyl_, true),
     eps_01_v_("eps01", L2FESpace_, cyl_, true),
     eps_02_v_("eps02", L2FESpace_, cyl_, true),
     eps_10_v_("eps10", L2FESpace_, cyl_, true),
     eps_11_v_("eps11", L2FESpace_, cyl_, true),
     eps_12_v_("eps12", L2FESpace_, cyl_, true),
     eps_20_v_("eps20", L2FESpace_, cyl_, true),
     eps_21_v_("eps21", L2FESpace_, cyl_, true),
     eps_22_v_("eps22", L2FESpace_, cyl_, true),
     b_hat_(NULL),
     // u_(NULL),
     // uE_(NULL),
     // uB_(NULL),
     // S_(NULL),
     StixS_(NULL),
     StixD_(NULL),
     StixP_(NULL),
     BCoef_(&BCoef),
     epsReCoef_(&epsReCoef),
     epsImCoef_(&epsImCoef),
     epsAbsCoef_(&epsAbsCoef),
     muInvCoef_(&muInvCoef),
     etaInvCoef_(etaInvCoef),
     kReCoef_(kReCoef),
     kImCoef_(kImCoef),
     SReCoef_(NULL),
     SImCoef_(NULL),
     DReCoef_(NULL),
     DImCoef_(NULL),
     PReCoef_(NULL),
     PImCoef_(NULL),
     omegaCoef_(new ConstantCoefficient(omega_)),
     negOmegaCoef_(new ConstantCoefficient(-omega_)),
     omega2Coef_(new ConstantCoefficient(pow(omega_, 2))),
     negOmega2Coef_(new ConstantCoefficient(-pow(omega_, 2))),
     abcCoef_(NULL),
     sbcReCoef_(NULL),
     sbcImCoef_(NULL),
     sinkx_(NULL),
     coskx_(NULL),
     negsinkx_(NULL),
     posMassCoef_(NULL),
     derCoef_(NULL),
     deiCoef_(NULL),
     uCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_,
            *epsReCoef_, *epsImCoef_, *muInvCoef_),
     uECoef_(erCoef_, eiCoef_, *epsReCoef_, *epsImCoef_),
     uBCoef_(omega_, derCoef_, deiCoef_, *muInvCoef_),
     SrCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     SiCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     dbcs_(stixBCs.GetDirichletBCs()),
     nbcs_(stixBCs.GetNeumannBCs()),
     sbcs_(stixBCs.GetSheathBCs()),
     axis_(stixBCs.GetCylindricalAxis()),
     maxwell_(*HCurlFESpace_, omega, conv,
              epsReCoef, epsImCoef, muInvCoef,
              kReCoef, kImCoef, cyl, pa),
     current_(*HCurlFESpace_, *HDivFESpace_, omega, conv,
              stixBCs.GetCurrentSrcs(), nbcs_, kReCoef, kImCoef, cyl, pa),
     faraday_(e_, *HDivFESpace_, omega, kReCoef, kImCoef),
     divB_(faraday_.GetMagneticFlux(), *L2FESpace_, kReCoef, kImCoef),
     displacement_(e_, *HDivFESpace_, epsReCoef, epsImCoef, pa_),
     divD_(displacement_.GetDisplacement(), *L2FESpace_, kReCoef, kImCoef),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Constructing CPDSolverEB ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   if (BCoef_)
   {
      e_b_ = new ParComplexGridFunction(L2FESpace_);
      *e_b_ = 0.0;
      b_hat_ = new ParGridFunction(HDivFESpace_);
   }
   if (kReCoef_ || kImCoef_)
   {
      e_t_ = new ParGridFunction(L2VSFESpace_);
   }
   else
   {
      e_t_ = new ParGridFunction(HCurlFESpace_);
   }

   if (false)
   {
      GridFunction * nodes = pmesh_->GetNodes();
      cout << "nodes is " << nodes << endl;
      for (int i=0; i<HCurlFESpace_->GetNBE(); i++)
      {
         const FiniteElement &be = *HCurlFESpace_->GetBE(i);
         ElementTransformation *eltrans = HCurlFESpace_->GetBdrElementTransformation (i);
         cout << i << '\t' << pmesh_->GetBdrAttribute(i)
              << '\t' << be.GetGeomType()
              << '\t' << eltrans->ElementNo
              << '\t' << eltrans->Attribute
              << endl;
      }
   }

   blockTrueOffsets_.SetSize(3);
   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   if ( dbcs_.Size() > 0 )
   {
      if ( dbcs_.Size() == 1 && dbcs_[0]->attr[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         ess_bdr_ = 0;
         for (int i=0; i<dbcs_.Size(); i++)
         {
            for (int j=0; j<dbcs_[i]->attr.Size(); j++)
            {
               ess_bdr_[dbcs_[i]->attr[j]-1] = 1;
            }
         }
      }
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }

   if (axis_.Size() > 0)
   {
      Array<int> axis_bdr(pmesh_->bdr_attributes.Max());
      Array<int> axis_bdr_tdofs;
      Array<int> axis_bdr_z_tdofs;
      if ( axis_.Size() == 1 && axis_[0]->attr[0] == -1 )
      {
         axis_bdr = 1;
      }
      else
      {
         axis_bdr = 0;
         for (int i=0; i<axis_.Size(); i++)
         {
            for (int j=0; j<axis_[i]->attr.Size(); j++)
            {
               axis_bdr[axis_[i]->attr[j]-1] = 1;
            }
         }
      }
      HCurlFESpace_->GetEssentialTrueDofs(axis_bdr, axis_bdr_tdofs);

      Vector zHat(3); zHat = 0.0; zHat[2] = 1.0;
      VectorConstantCoefficient zHatCoef(zHat);

      e_.real().ProjectCoefficient(zHatCoef);

      HypreParVector *tv = e_.real().GetTrueDofs();

      for (int i=0; i<axis_bdr_tdofs.Size(); i++)
      {
         if ((*tv)(axis_bdr_tdofs[i]) > 0.5)
         {
            axis_bdr_z_tdofs.Append(axis_bdr_tdofs[i]);
         }
      }
      ess_bdr_tdofs_.Append(axis_bdr_z_tdofs);

      delete tv;
   }

   // Setup various coefficients
   posMassCoef_ = new ScalarMatrixProductCoefficient(*omega2Coef_,
                                                     *epsAbsCoef_);

   // Impedance of free space
   if ( abcs.Size() > 0 )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Admittance Coefficient" << endl;
      }

      abc_bdr_marker_.SetSize(pmesh.bdr_attributes.Max());
      if ( abcs.Size() == 1 && abcs[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_bdr_marker_ = 1;
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_bdr_marker_ = 0;
         for (int i=0; i<abcs.Size(); i++)
         {
            abc_bdr_marker_[abcs[i]-1] = 1;
         }
      }
      if ( etaInvCoef_ == NULL )
      {
         etaInvCoef_ = new ConstantCoefficient(sqrt(epsilon0_/mu0_));
      }
      abcCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvCoef_,
                                            prodFunc);
   }
   /*
   // Complex Impedance
   if ( sbcs.Size() > 0 && etaInvReCoef_ != NULL && etaInvReCoef_ != NULL )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Complex Admittance Coefficient" << endl;
      }

      sbc_marker_.SetSize(pmesh.bdr_attributes.Max());

      // Mark select boundaries as absorbing
      sbc_marker_ = 0;
      for (int i=0; i<sbcs.Size(); i++)
      {
         sbc_marker_[sbcs[i]-1] = 1;
      }

      sbcReCoef_ = new TransformedCoefficient(omegaCoef_, etaInvImCoef_,
                                              prodFunc);
      sbcImCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvReCoef_,
                                              prodFunc);
   }
   */

   // Bilinear Forms
   if ( abcCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("abcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      maxwell_.AddBoundaryIntegrator(NULL,
                                     new VectorFEMassIntegrator(*abcCoef_),
                                     abc_bdr_marker_);
   }
   /*
   if ( sbcReCoef_ && sbcImCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("sbcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      a1_->AddBoundaryIntegrator(new VectorFEMassIntegrator(*sbcReCoef_),
                                 new VectorFEMassIntegrator(*sbcImCoef_),
                                 sbc_marker_);
   }
   */
   b1_ = new ParBilinearForm(HCurlFESpace_);
   if (pa_) { b1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   // b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsAbsCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));
   //b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massImCoef_));

   // Build grid functions
   e_ = 0.0;

   // solNorm_ = e_->ComputeL2Error(const_cast<VectorCoefficient&>(erCoef_),
   //                               const_cast<VectorCoefficient&>(eiCoef_));

   if (vis_u_)
   {
      // if (L2FESpace2p_ == NULL)
      // {
      //    L2FESpace2p_ = new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension());
      // }
      // u_ = new ParGridFunction(L2FESpace2p_);
      // uE_ = new ParGridFunction(L2FESpace2p_);
      // uB_ = new ParGridFunction(L2FESpace2p_);

      // HDivFESpace2p_ = new RT_ParFESpace(pmesh_,2*order,pmesh_->Dimension());
      // S_ = new ParComplexGridFunction(HDivFESpace2p_);

      erCoef_.SetGridFunction(&e_.real());
      eiCoef_.SetGridFunction(&e_.imag());

      derCoef_.SetGridFunction(&e_.real());
      deiCoef_.SetGridFunction(&e_.imag());
   }
   {
      StixCoefBase * s = dynamic_cast<StixCoefBase*>(epsReCoef_);

      if (s != NULL)
      {
         SReCoef_ = new StixSCoef(*s);
         SImCoef_ = new StixSCoef(*s);
         DReCoef_ = new StixDCoef(*s);
         DImCoef_ = new StixDCoef(*s);
         PReCoef_ = new StixPCoef(*s);
         PImCoef_ = new StixPCoef(*s);

         dynamic_cast<StixCoefBase*>(SImCoef_)->SetImaginaryPart();
         dynamic_cast<StixCoefBase*>(DImCoef_)->SetImaginaryPart();
         dynamic_cast<StixCoefBase*>(PImCoef_)->SetImaginaryPart();
      }
   }

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

CPDSolverEB::~CPDSolverEB()
{
   for (int i=0; i<vCoefs_.Size(); i++)
   {
      delete vCoefs_[i];
   }

   delete negsinkx_;
   delete coskx_;
   delete sinkx_;
   delete SReCoef_;
   delete SImCoef_;
   delete DReCoef_;
   delete DImCoef_;
   delete PReCoef_;
   delete PImCoef_;
   delete posMassCoef_;
   delete abcCoef_;
   delete sbcReCoef_;
   delete sbcImCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   delete StixS_;
   delete StixD_;
   delete StixP_;
   delete e_b_;
   delete b_hat_;
   delete b_hat_v_;
   // delete u_;
   // delete uE_;
   // delete uB_;
   delete e_t_;

   delete b1_;

   delete L2FESpace_;
   delete L2FESpace2p_;
   delete L2VSFESpace_;
   delete L2VSFESpace2p_;
   delete L2V3FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   // delete HDivFESpace2p_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
CPDSolverEB::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolverEB::PrintSizes()
{
   // HYPRE_Int size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      // cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
CPDSolverEB::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << flush; }

   tic_toc.Clear();
   tic_toc.Start();

   b1_->Assemble();
   if (!pa_) { b1_->Finalize(); }

   maxwell_.Assemble();
   current_.Assemble();
   faraday_.Assemble();
   displacement_.Assemble();
   divB_.Assemble();
   divD_.Assemble();

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolverEB::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   // H1FESpace_->Update(false);
   if (L2FESpace_) { L2FESpace_->Update(); }
   if (L2FESpace2p_) { L2FESpace2p_->Update(); }
   if (L2VSFESpace_) { L2VSFESpace_->Update(); }
   if (L2VSFESpace2p_) { L2VSFESpace2p_->Update(); }
   if (L2V3FESpace_) { L2V3FESpace_->Update(); }
   HCurlFESpace_->Update();
   HDivFESpace_->Update();
   // if (HDivFESpace2p_) { HDivFESpace2p_->Update(false); }

   if ( ess_bdr_.Size() > 0 )
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }

   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // Inform the grid functions that the space has changed.
   e_.Update();

   if (axis_.Size() > 0)
   {
      Array<int> axis_bdr(pmesh_->bdr_attributes.Max());
      Array<int> axis_bdr_tdofs;
      Array<int> axis_bdr_z_tdofs;
      if ( axis_.Size() == 1 && axis_[0]->attr[0] == -1 )
      {
         axis_bdr = 1;
      }
      else
      {
         axis_bdr = 0;
         for (int i=0; i<axis_.Size(); i++)
         {
            for (int j=0; j<axis_[i]->attr.Size(); j++)
            {
               axis_bdr[axis_[i]->attr[j]-1] = 1;
            }
         }
      }
      HCurlFESpace_->GetEssentialTrueDofs(axis_bdr, axis_bdr_tdofs);

      Vector zHat(3); zHat = 0.0; zHat[2] = 1.0;
      VectorConstantCoefficient zHatCoef(zHat);

      e_.real().ProjectCoefficient(zHatCoef);

      HypreParVector *tv = e_.real().GetTrueDofs();

      for (int i=0; i<axis_bdr_tdofs.Size(); i++)
      {
         if ((*tv)(axis_bdr_tdofs[i]) > 0.5)
         {
            axis_bdr_z_tdofs.Append(axis_bdr_tdofs[i]);
         }
      }
      ess_bdr_tdofs_.Append(axis_bdr_z_tdofs);

      delete tv;
   }

   // if (u_) { u_->Update(); }
   // if (uE_) { uE_->Update(); }
   // if (uB_) { uB_->Update(); }
   // if (S_) { S_->Update(); }
   if (e_t_) { e_t_->Update(); }
   if (e_b_) { e_b_->Update(); }
   e_v_.Update();
   b_v_.Update();
   d_v_.Update();
   j_v_.Update();
   k_v_.Update();
   ue_v_.Update();
   ub_v_.Update();
   u_v_.Update();
   s_v_.Update();
   g_v_.Update();
   eps_00_v_.Update();
   eps_01_v_.Update();
   eps_02_v_.Update();
   eps_10_v_.Update();
   eps_11_v_.Update();
   eps_12_v_.Update();
   eps_20_v_.Update();
   eps_21_v_.Update();
   eps_22_v_.Update();
   db_v_.Update();
   dd_v_.Update();
   if (b_hat_) { b_hat_->Update(); }

   if (StixS_) { StixS_->Update(); }
   if (StixD_) { StixD_->Update(); }
   if (StixP_) { StixP_->Update(); }

   // Inform the bilinear forms that the space has changed.
   b1_->Update();

   // Inform the other objects that the space has changed.
   maxwell_.Update();
   current_.Update();
   faraday_.Update();
   displacement_.Update();
   divB_.Update();
   divD_.Update();
   // if ( grad_        ) { grad_->Update(); }
   // if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
   // if ( SurfCur_     ) { SurfCur_->Update(); }
   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolverEB::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   OperatorHandle A1;
   Vector E, RHS;

   if (dbcs_.Size() > 0)
   {
      Array<int> attr_marker(pmesh_->bdr_attributes.Max());
      for (int i = 0; i<dbcs_.Size(); i++)
      {
         attr_marker = 0;
         for (int j=0; j<dbcs_[i]->attr.Size(); j++)
         {
            attr_marker[dbcs_[i]->attr[j] - 1] = 1;
         }

         bool dbc_phase = true;
         if (dbc_phase)
         {
            ComplexPhaseVectorCoefficient re_e_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, true, true);
            ComplexPhaseVectorCoefficient im_e_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, false, true);

            e_.ProjectBdrCoefficientTangent(re_e_dbc, im_e_dbc, attr_marker);
         }
         else
         {
            e_.ProjectBdrCoefficientTangent(*dbcs_[i]->real, *dbcs_[i]->imag,
                                            attr_marker);
         }
      }
   }

   if (axis_.Size() > 0)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVecCoef(zeroVec);

      Array<int> attr_marker(pmesh_->bdr_attributes.Max());
      for (int i = 0; i<axis_.Size(); i++)
      {
         attr_marker = 0;
         for (int j=0; j<axis_[i]->attr.Size(); j++)
         {
            attr_marker[axis_[i]->attr[j] - 1] = 1;
         }

         e_.ProjectBdrCoefficientTangent(zeroVecCoef, zeroVecCoef,
                                         attr_marker);
      }
   }

   maxwell_.FormLinearSystem(ess_bdr_tdofs_, e_, current_, A1, E, RHS);

   tic_toc.Clear();
   tic_toc.Start();

   Operator * pcr = NULL;
   Operator * pci = NULL;
   BlockDiagonalPreconditioner * BDP = NULL;

   if (pa_)
   {
      switch (prec_)
      {
         case INVALID_PC:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "No Preconditioner Requested (PA)" << endl;
            }
            break;
         case DIAG_SCALE:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Diagonal Scaling Preconditioner Requested (PA)" << endl;
            }
            pcr = new OperatorJacobiSmoother(*b1_, ess_bdr_tdofs_);
            break;
         default:
            MFEM_ABORT("Requested preconditioner is not available with PA.");
            break;
      }
   }
   else if (sol_ == GMRES || sol_ == FGMRES || sol_ == MINRES)
   {
      OperatorHandle PCOp;
      b1_->FormSystemMatrix(ess_bdr_tdofs_, PCOp);
      switch (prec_)
      {
         case INVALID_PC:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "No Preconditioner Requested" << endl;
            }
            break;
         case DIAG_SCALE:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Diagonal Scaling Preconditioner Requested" << endl;
            }
            pcr = new HypreDiagScale(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            break;
         case PARASAILS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "ParaSails Preconditioner Requested" << endl;
            }
            pcr = new HypreParaSails(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            dynamic_cast<HypreParaSails*>(pcr)->SetSymmetry(1);
            break;
         case EUCLID:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Euclid Preconditioner Requested" << endl;
            }
            pcr = new HypreEuclid(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            if (solOpts_.euLvl != 1)
            {
               HypreSolver * pc = dynamic_cast<HypreSolver*>(pcr);
               HYPRE_EuclidSetLevel(*pc, solOpts_.euLvl);
            }
            break;
         case AMS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "AMS Preconditioner Requested" << endl;
            }
            pcr = new HypreAMS(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()),
                               HCurlFESpace_);
            break;
         default:
            MFEM_ABORT("Requested preconditioner is not available.");
            break;
      }
   }

   if (pcr && conv_ != ComplexOperator::HERMITIAN)
   {
      pci = new ScaledOperator(pcr, -1.0);
   }
   else
   {
      pci = pcr;
   }
   if (pcr)
   {
      BDP = new BlockDiagonalPreconditioner(blockTrueOffsets_);
      BDP->SetDiagonalBlock(0, pcr);
      BDP->SetDiagonalBlock(1, pci);
      BDP->owns_blocks = 0;
   }

   switch (sol_)
   {
      case GMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "GMRES Solver Requested" << endl;
         }
         GMRESSolver gmres(HCurlFESpace_->GetComm());
         if (BDP) { gmres.SetPreconditioner(*BDP); }
         gmres.SetOperator(*A1.Ptr());
         gmres.SetRelTol(solOpts_.relTol);
         gmres.SetMaxIter(solOpts_.maxIter);
         gmres.SetKDim(solOpts_.kDim);
         gmres.SetPrintLevel(solOpts_.printLvl);

         gmres.Mult(RHS, E);
      }
      break;
      case FGMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "FGMRES Solver Requested" << endl;
         }
         FGMRESSolver fgmres(HCurlFESpace_->GetComm());
         if (BDP) { fgmres.SetPreconditioner(*BDP); }
         fgmres.SetOperator(*A1.Ptr());
         fgmres.SetRelTol(solOpts_.relTol);
         fgmres.SetMaxIter(solOpts_.maxIter);
         fgmres.SetKDim(solOpts_.kDim);
         fgmres.SetPrintLevel(solOpts_.printLvl);

         fgmres.Mult(RHS, E);
      }
      break;
      case MINRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MINRES Solver Requested" << endl;
         }
         MINRESSolver minres(HCurlFESpace_->GetComm());
         if (BDP) { minres.SetPreconditioner(*BDP); }
         minres.SetOperator(*A1.Ptr());
         minres.SetRelTol(solOpts_.relTol);
         minres.SetMaxIter(solOpts_.maxIter);
         minres.SetPrintLevel(solOpts_.printLvl);

         minres.Mult(RHS, E);
      }
      break;
#ifdef MFEM_USE_SUPERLU
      case SUPERLU:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "SuperLU Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         SuperLURowLocMatrix A_SuperLU(*A1C);
         SuperLUSolver solver(MPI_COMM_WORLD);
         solver.SetOperator(A_SuperLU);
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
#ifdef MFEM_USE_STRUMPACK
      case STRUMPACK:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "STRUMPACK Solver Requested" << endl;
         }
         //A1.SetOperatorOwner(false);
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         STRUMPACKRowLocMatrix A_STRUMPACK(*A1C);
         STRUMPACKSolver solver(0, NULL, MPI_COMM_WORLD);
         solver.SetPrintFactorStatistics(true);
         solver.SetPrintSolveStatistics(false);
         solver.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         solver.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
         solver.DisableMatching();
         solver.SetOperator(A_STRUMPACK);
         solver.SetFromCommandLine();
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
#ifdef MFEM_USE_MUMPS
      case DMUMPS:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MUMPS (Real) Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         MUMPSSolver dmumps;
         dmumps.SetPrintLevel(1);
         dmumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         dmumps.SetOperator(*A1C);
         dmumps.Mult(RHS, E);
         delete A1C;
      }
      break;
      case ZMUMPS:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MUMPS (Complex) Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         ComplexMUMPSSolver zmumps;
         zmumps.SetPrintLevel(1);
         zmumps.SetOperator(*A1Z);
         zmumps.Mult(RHS, E);
      }
      break;
#endif
      default:
         MFEM_ABORT("Requested solver is not available.");
         break;
   };

   tic_toc.Stop();

   e_.Distribute(E);

   faraday_.ComputeB();
   displacement_.ComputeD();

   delete BDP;
   if (pci != pcr) { delete pci; }
   delete pcr;

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

double
CPDSolverEB::GetEFieldError(const VectorCoefficient & EReCoef,
                            const VectorCoefficient & EImCoef) const
{
   ParFiniteElementSpace * fes =
      const_cast<ParFiniteElementSpace*>(e_.ParFESpace());
   ParComplexGridFunction z(fes);
   z = 0.0;

   double solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));


   double solErr = e_.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

void
CPDSolverEB::GetErrorEstimates(Vector & errors)
{
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "Estimating Error ... " << flush; }

   FiniteElementCollection *flux_fec = NULL;
   FiniteElementCollection *smooth_flux_fec = NULL;

   switch (pmesh_->SpaceDimension())
   {
      case 1:
         flux_fec = new RT_R1D_FECollection(order_ - 1, 1);
         smooth_flux_fec = new ND_R1D_FECollection(order_, 1);
         break;
      case 2:
         flux_fec = new RT_R2D_FECollection(order_ - 1, 2);
         smooth_flux_fec = new ND_R2D_FECollection(order_, 2);
         break;
      case 3:
         flux_fec = new RT_FECollection(order_ - 1, 3);
         smooth_flux_fec = new ND_FECollection(order_, 3);
         break;
   }

   // Space for the discontinuous (original) flux
   CurlCurlIntegrator flux_integrator(*muInvCoef_);
   ParFiniteElementSpace flux_fes(pmesh_, flux_fec);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   ParFiniteElementSpace smooth_flux_fes(pmesh_, smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, e_.real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);

   delete flux_fec;
   delete smooth_flux_fec;

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void CPDSolverEB::prepareScalarVisField(const ParComplexGridFunction &u,
                                        ComplexGridFunction &v)
{
   if (kReCoef_ || kImCoef_)
   {
      GridFunctionCoefficient u_r(&u.real());
      GridFunctionCoefficient u_i(&u.imag());
      ComplexPhaseCoefficient uk_r(kReCoef_, kImCoef_, &u_r, &u_i,
                                   true, false);
      ComplexPhaseCoefficient uk_i(kReCoef_, kImCoef_, &u_r, &u_i,
                                   false, false);

      ScalarR2DCoef uk_r_3D(uk_r, *pmesh_);
      ScalarR2DCoef uk_i_3D(uk_i, *pmesh_);

      v.ProjectCoefficient(uk_r_3D, uk_i_3D);
   }
   else
   {
      GridFunctionCoefficient u_r(&u.real());
      GridFunctionCoefficient u_i(&u.imag());

      ScalarR2DCoef u_r_3D(u_r, *pmesh_);
      ScalarR2DCoef u_i_3D(u_i, *pmesh_);

      v.ProjectCoefficient(u_r_3D, u_i_3D);
   }
}

void CPDSolverEB::prepareVectorVisField(const ParComplexGridFunction &u,
                                        ComplexGridFunction &v,
                                        ComplexGridFunction *vy,
                                        ComplexGridFunction *vz)
{
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());
      ComplexPhaseVectorCoefficient uk_r(kReCoef_, kImCoef_, &u_r, &u_i,
                                         true, false);
      ComplexPhaseVectorCoefficient uk_i(kReCoef_, kImCoef_, &u_r, &u_i,
                                         false, false);

      switch (pmesh_->SpaceDimension())
      {
         case 1:
         {}
         break;
         case 2:
         {
            VectorXYCoef ukxy_r(uk_r);
            VectorXYCoef ukxy_i(uk_i);
            VectorZCoef   ukz_r(uk_r);
            VectorZCoef   ukz_i(uk_i);

            v.ProjectCoefficient(ukxy_r, ukxy_i);
            if (vz) { vz->ProjectCoefficient(ukz_r, ukz_i); }
         }
         break;
         case 3:
         {}
         break;
      }
      // VectorR2DCoef uk_r_3D(uk_r, *pmesh_);
      // VectorR2DCoef uk_i_3D(uk_i, *pmesh_);

      // v.ProjectCoefficient(uk_r_3D, uk_i_3D);
   }
   else
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());

      switch (pmesh_->SpaceDimension())
      {
         case 1:
         {}
         break;
         case 2:
         {
            VectorXYCoef uxy_r(u_r);
            VectorXYCoef uxy_i(u_i);
            VectorZCoef   uz_r(u_r);
            VectorZCoef   uz_i(u_i);

            v.ProjectCoefficient(uxy_r, uxy_i);
            if (vz) { vz->ProjectCoefficient(uz_r, uz_i); }
         }
         break;
         case 3:
         {}
         break;
      }

      // VectorR2DCoef u_r_3D(u_r, *pmesh_);
      // VectorR2DCoef u_i_3D(u_i, *pmesh_);

      // v.ProjectCoefficient(u_r_3D, u_i_3D);
   }
}

void CPDSolverEB::prepareVisFields()
{
   e_v_.PrepareVisField(e_, kReCoef_, kImCoef_);
   b_v_.PrepareVisField(faraday_.GetMagneticFlux(), kReCoef_, kImCoef_);
   d_v_.PrepareVisField(displacement_.GetDisplacement(), kReCoef_, kImCoef_);
   j_v_.PrepareVisField(current_.GetVolumeCurrentDensity(),
                        kReCoef_, kImCoef_);
   k_v_.PrepareVisField(current_.GetSurfaceCurrentDensity(),
                        kReCoef_, kImCoef_);
   ue_v_.PrepareVisField(e_, *epsReCoef_, *epsImCoef_);
   ub_v_.PrepareVisField(e_, omega_, *muInvCoef_);
   u_v_.PrepareVisField(e_, omega_, *epsReCoef_, *epsImCoef_, *muInvCoef_);
   s_v_.PrepareVisField(e_, omega_, *muInvCoef_);
   g_v_.PrepareVisField(e_, omega_, *epsReCoef_, *epsImCoef_);
   eps_00_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 0, 0);
   eps_01_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 0, 1);
   eps_02_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 0, 2);
   eps_10_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 1, 0);
   eps_11_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 1, 1);
   eps_12_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 1, 2);
   eps_20_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 2, 0);
   eps_21_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 2, 1);
   eps_22_v_.PrepareVisField(*epsReCoef_, *epsImCoef_, 2, 2);

   divB_.ComputeDiv();
   divD_.ComputeDiv();

   db_v_.PrepareVisField(divB_.GetDivergence(), kReCoef_, kImCoef_);
   dd_v_.PrepareVisField(divD_.GetDivergence(), kReCoef_, kImCoef_);
   /*
   prepareScalarVisField(divB_.GetDivergence(), *db_v_);

    prepareScalarVisField(divD_.GetDivergence(), *dd_v_);
   */
   /*
    if (h_tilde_)
    {
       VectorGridFunctionCoefficient u_r(&h_->real());
       VectorGridFunctionCoefficient u_i(&h_->imag());

       VectorR2DCoef u_r_3D(u_r, *pmesh_);
       VectorR2DCoef u_i_3D(u_i, *pmesh_);

       h_tilde_->ProjectCoefficient(u_r_3D, u_i_3D);
    }
    */
   /*
   {
      VectorGridFunctionCoefficient d_r(&d_->real());
      VectorGridFunctionCoefficient d_i(&d_->real());

      DivergenceGridFunctionCoefficient div_d_r(&d_->real());
      DivergenceGridFunctionCoefficient div_d_i(&d_->imag());

      VectorR2DCoef dReCoef(d_r, *pmesh_);
      VectorR2DCoef dImCoef(d_i, *pmesh_);

      Vector ZeroVec(3); ZeroVec = 0.0;
      VectorConstantCoefficient ZeroCoef(ZeroVec);
      VectorCoefficient * kReCoef = (kReCoef_) ? kReCoef_ : &ZeroCoef;
      VectorCoefficient * kImCoef = (kImCoef_) ? kImCoef_ : &ZeroCoef;

      InnerProductCoefficient krdr(*kReCoef, dReCoef);
      InnerProductCoefficient krdi(*kReCoef, dImCoef);
      InnerProductCoefficient kidr(*kImCoef, dReCoef);
      InnerProductCoefficient kidi(*kImCoef, dImCoef);

      SumCoefficient ikd_r(krdi, kidr, -1.0, -1.0);
      SumCoefficient ikd_i(krdr, kidi,  1.0, -1.0);

      SumCoefficient div_d_r_3D(div_d_r, ikd_r);
      SumCoefficient div_d_i_3D(div_d_i, ikd_i);

      ComplexPhaseCoefficient ddk_r(kReCoef_, kImCoef_, &div_d_r_3D, &div_d_i_3D,
                                    true, false);
      ComplexPhaseCoefficient ddk_i(kReCoef_, kImCoef_, &div_d_r_3D, &div_d_i_3D,
                                    false, false);

      div_d_->ProjectCoefficient(ddk_r, ddk_i);
   }
   */
   if (BCoef_ && false)
   {
      // VectorGridFunctionCoefficient b_hat(b_hat_);
      // VectorR2DCoef b_hat_3D(b_hat, *pmesh_);
      VectorR2DCoef b_hat_3D(*BCoef_, *pmesh_);
      b_hat_v_->ProjectCoefficient(b_hat_3D);
      /*
      VectorGridFunctionCoefficient e_r(&e_v_->real());
      VectorGridFunctionCoefficient e_i(&e_v_->imag());
      InnerProductCoefficient eb_r(e_r, *BCoef_);
      InnerProductCoefficient eb_i(e_i, *BCoef_);

      e_b_v_->ProjectCoefficient(eb_r, eb_i);
      */
   }
}

void
CPDSolverEB::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   if (L2VSFESpace_ == NULL)
   {
      L2VSFESpace_ = new L2_ParFESpace(pmesh_,order_,pmesh_->Dimension(),
                                       pmesh_->SpaceDimension());
   }

   StixS_ = new ComplexGridFunction(L2FESpace_);
   StixD_ = new ComplexGridFunction(L2FESpace_);
   StixP_ = new ComplexGridFunction(L2FESpace_);

   e_v_.RegisterVisItFields(visit_dc);
   b_v_.RegisterVisItFields(visit_dc);
   d_v_.RegisterVisItFields(visit_dc);
   j_v_.RegisterVisItFields(visit_dc);
   k_v_.RegisterVisItFields(visit_dc);
   ue_v_.RegisterVisItFields(visit_dc);
   ub_v_.RegisterVisItFields(visit_dc);
   u_v_.RegisterVisItFields(visit_dc);
   s_v_.RegisterVisItFields(visit_dc);
   g_v_.RegisterVisItFields(visit_dc);
   eps_00_v_.RegisterVisItFields(visit_dc);
   eps_01_v_.RegisterVisItFields(visit_dc);
   eps_02_v_.RegisterVisItFields(visit_dc);
   eps_10_v_.RegisterVisItFields(visit_dc);
   eps_11_v_.RegisterVisItFields(visit_dc);
   eps_12_v_.RegisterVisItFields(visit_dc);
   eps_20_v_.RegisterVisItFields(visit_dc);
   eps_21_v_.RegisterVisItFields(visit_dc);
   eps_22_v_.RegisterVisItFields(visit_dc);
   db_v_.RegisterVisItFields(visit_dc);
   dd_v_.RegisterVisItFields(visit_dc);
   /*
   if ( BCoef_)
   {
      b_hat_v_ = new GridFunction(L2VFESpace3D_);

      visit_dc.RegisterField("B_hat", b_hat_v_);
   }

   if ( u_ )
   {
      visit_dc.RegisterField("U", u_);
      visit_dc.RegisterField("U_E", uE_);
      visit_dc.RegisterField("U_B", uB_);
      visit_dc.RegisterField("Re_S", &S_->real());
      visit_dc.RegisterField("Im_S", &S_->imag());
      // visit_dc.RegisterField("Im(u)", &u_->imag());
   }
   */
   // if ( m_ ) { visit_dc.RegisterField("M", m_); }
   // if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
   if ( StixS_ )
   {
      visit_dc.RegisterField("Re_StixS", &StixS_->real());
      visit_dc.RegisterField("Im_StixS", &StixS_->imag());
      visit_dc.RegisterField("Re_StixD", &StixD_->real());
      visit_dc.RegisterField("Im_StixD", &StixD_->imag());
      visit_dc.RegisterField("Re_StixP", &StixP_->real());
      visit_dc.RegisterField("Im_StixP", &StixP_->imag());
   }
}

void
CPDSolverEB::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      prepareVisFields();

      if ( StixS_ )
      {
         StixS_->ProjectCoefficient(*SReCoef_, *SImCoef_);
         StixD_->ProjectCoefficient(*DReCoef_, *DImCoef_);
         StixP_->ProjectCoefficient(*PReCoef_, *PImCoef_);
      }
      /*
      curl_->Mult(e_->real(), b_->imag());
      curl_->Mult(e_->imag(), b_->real());
      if (kImCross_)
      {
         kImCross_->AddMult(e_->real(), b_->imag(), -1.0);
         kImCross_->AddMult(e_->imag(), b_->real(), -1.0);
      }
      if (kReCross_)
      {
         kReCross_->AddMult(e_->imag(), b_->imag(), -1.0);
         kReCross_->AddMult(e_->real(), b_->real(),  1.0);
      }
      b_->real() /= omega_;
      b_->imag() /= -omega_;

      if ( BCoef_)
      {
         b_hat_->ProjectCoefficient(*BCoef_);
      }

      if ( j_ )
      {
         j_->ProjectCoefficient(*jrCoef_, *jiCoef_);
      }
      if ( u_ )
      {
         u_->ProjectCoefficient(uCoef_);
         uE_->ProjectCoefficient(uECoef_);
         uB_->ProjectCoefficient(uBCoef_);
         S_->ProjectCoefficient(SrCoef_, SiCoef_);
      }
      */
      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
CPDSolverEB::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   socks_["Er"] = new socketstream;
   socks_["Er"]->precision(8);

   socks_["Ei"] = new socketstream;
   socks_["Ei"]->precision(8);

   socks_["Dr"] = new socketstream;
   socks_["Dr"]->precision(8);

   socks_["Di"] = new socketstream;
   socks_["Di"]->precision(8);

   if (BCoef_)
   {
      socks_["EBr"] = new socketstream;
      socks_["EBr"]->precision(8);

      socks_["EBi"] = new socketstream;
      socks_["EBi"]->precision(8);
   }

   // socks_["B"] = new socketstream;
   // socks_["B"]->precision(8);

   // socks_["H"] = new socketstream;
   // socks_["H"]->precision(8);
   /*
   if ( j_v_ )
   {
      socks_["Jr"] = new socketstream;
      socks_["Jr"]->precision(8);

      socks_["Ji"] = new socketstream;
      socks_["Ji"]->precision(8);
   }
   */
   /*
   if ( u_ )
   {
      socks_["U"] = new socketstream;
      socks_["U"]->precision(8);

      socks_["U_E"] = new socketstream;
      socks_["U_E"]->precision(8);

      socks_["U_B"] = new socketstream;
      socks_["U_B"]->precision(8);

      socks_["Sr"] = new socketstream;
      socks_["Sr"]->precision(8);

      socks_["Si"] = new socketstream;
      socks_["Si"]->precision(8);
   }
   */
   /*
   if ( k_ )
   {
      socks_["K"] = new socketstream;
      socks_["K"]->precision(8);
      socks_["Psi"] = new socketstream;
      socks_["Psi"]->precision(8);
   }
   if ( m_ )
   {
      socks_["M"] = new socketstream;
      socks_["M"]->precision(8);
   }
   */
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl; }
}

void
CPDSolverEB::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   prepareVisFields();
   /*
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);

      VectorGridFunctionCoefficient d_r(&d_->real());
      VectorGridFunctionCoefficient d_i(&d_->imag());
      VectorSumCoefficient drCoef(d_r, d_i, *coskx_, *negsinkx_);
      VectorSumCoefficient diCoef(d_i, d_r, *coskx_, *sinkx_);

      d_v_->ProjectCoefficient(drCoef, diCoef);
   }
   else
   {
      e_v_ = e_;
      d_v_ = d_;
   }
   */
   /*
   ostringstream er_keys, ei_keys;
   er_keys << "aaAcppppp";
   ei_keys << "aaAcppppp";

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:
         VisualizeField(*socks_["Er"], vishost, visport,
                        e_v_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh,
                        er_keys.str().c_str());
         Wx += offx;

         VisualizeField(*socks_["Ei"], vishost, visport,
                        e_v_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh,
                        ei_keys.str().c_str());
         break;
   }
   */
   /*
   ostringstream br_keys, bi_keys;
   br_keys << "aaAcPPPP";
   bi_keys << "aaAcPPPP";

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:

         VisualizeField(*socks_["Br"], vishost, visport,
                        b_v_->real(), "Magnetic Field, Re(B)", Wx, Wy, Ww, Wh,
                        br_keys.str().c_str());
         Wx += offx;

         VisualizeField(*socks_["Bi"], vishost, visport,
                        b_v_->imag(), "Magnetic Field, Im(B)", Wx, Wy, Ww, Wh,
                        br_keys.str().c_str());

         Wx += offx;
         break;
   }
   */
   /*
   VisualizeField(*socks_["Er"], vishost, visport,
                  e_v_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Ei"], vishost, visport,
                  e_v_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh);
   */
   /*
   Wx += offx;
   VisualizeField(*socks_["Dr"], vishost, visport,
                  d_v_->real(), "Electric Flux, Re(D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Di"], vishost, visport,
                  d_v_->imag(), "Electric Flux, Im(D)", Wx, Wy, Ww, Wh);
   */
   if (BCoef_)
   {
      /*
       VectorGridFunctionCoefficient e_r(&e_v_->real());
       VectorGridFunctionCoefficient e_i(&e_v_->imag());
       InnerProductCoefficient ebrCoef(e_r, *BCoef_);
       InnerProductCoefficient ebiCoef(e_i, *BCoef_);

       e_b_->ProjectCoefficient(ebrCoef, ebiCoef);
      */
      /*
       VisualizeField(*socks_["EBr"], vishost, visport,
                      e_b_->real(), "Parallel Electric Field, Re(E.B)",
                      Wx, Wy, Ww, Wh);
       Wx += offx;

       VisualizeField(*socks_["EBi"], vishost, visport,
                      e_b_->imag(), "Parallel Electric Field, Im(E.B)",
                      Wx, Wy, Ww, Wh);
       Wx += offx;
      */
   }
   /*
   Wx += offx;
   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);
   Wx += offx;
   VisualizeField(*socks_["H"], vishost, visport,
                  *h_, "Magnetic Field (H)", Wx, Wy, Ww, Wh);
   Wx += offx;
   */
   // if ( j_v_ )
   {
      Wx = 0; Wy += offy; // next line
      /*
      j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

      if (kReCoef_ || kImCoef_)
      {
         VectorGridFunctionCoefficient j_r(&j_->real());
         VectorGridFunctionCoefficient j_i(&j_->imag());
         VectorSumCoefficient jrCoef(j_r, j_i, *coskx_, *negsinkx_);
         VectorSumCoefficient jiCoef(j_i, j_r, *coskx_, *sinkx_);

         j_v_->ProjectCoefficient(jrCoef, jiCoef);
      }
      else
      {
         j_v_ = j_;
      }

      VisualizeField(*socks_["Jr"], vishost, visport,
                     j_v_->real(), "Current Density, Re(J)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Ji"], vishost, visport,
                     j_v_->imag(), "Current Density, Im(J)", Wx, Wy, Ww, Wh);
      */
   }
   Wx = 0; Wy += offy; // next line

   // if ( u_ )
   {
      /*
       Wx = 0; Wy += offy; // next line

       u_->ProjectCoefficient(uCoef_);
       uE_->ProjectCoefficient(uECoef_);
       uB_->ProjectCoefficient(uBCoef_);
       S_->ProjectCoefficient(SrCoef_, SiCoef_);

       VisualizeField(*socks_["U"], vishost, visport,
                      *u_, "Energy Density, U", Wx, Wy, Ww, Wh);

       Wx += offx;
       VisualizeField(*socks_["U_E"], vishost, visport,
                      *uE_, "Energy Density, U_E", Wx, Wy, Ww, Wh);

       Wx += offx;
       VisualizeField(*socks_["U_B"], vishost, visport,
                      *uB_, "Energy Density, U_B", Wx, Wy, Ww, Wh);

       Wx += offx;
       VisualizeField(*socks_["Sr"], vishost, visport,
                      S_->real(), "Poynting Vector, Re(S)", Wx, Wy, Ww, Wh);
       Wx += offx;
       VisualizeField(*socks_["Si"], vishost, visport,
                      S_->imag(), "Poynting Vector, Im(S)", Wx, Wy, Ww, Wh);
      */
   }
   Wx = 0; Wy += offy; // next line
   /*
   if ( k_ )
   {
      VisualizeField(*socks_["K"], vishost, visport,
                     *k_, "Surface Current Density (K)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Psi"], vishost, visport,
                     *SurfCur_->GetPsi(),
                     "Surface Current Potential (Psi)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( m_ )
   {
      VisualizeField(*socks_["M"], vishost, visport,
                     *m_, "Magnetization (M)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   */
   if (myid_ == 0) { cout << " done." << endl; }
}

void
CPDSolverEB::DisplayAnimationToGLVis()
{
   if (myid_ == 0) { cout << "Sending animation data to GLVis ..." << flush; }
   /*
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_.real());
      VectorGridFunctionCoefficient e_i(&e_.imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      VectorGridFunctionCoefficient e_r(&e_.real());
      VectorGridFunctionCoefficient e_i(&e_.imag());

      e_v_->ProjectCoefficient(e_r, e_i);
   }

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   double norm_r = e_v_->real().ComputeMaxError(zeroCoef);
   double norm_i = e_v_->imag().ComputeMaxError(zeroCoef);

   *e_t_ = e_v_->real();

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh_ << *e_t_
            << "window_title 'Harmonic Solution (t = 0.0 T)'"
            << "valuerange 0.0 " << max(norm_r, norm_i) << "\n"
            << "autoscale off\n"
            << "keys cvvv\n"
            << "pause\n" << flush;
   if (myid_ == 0)
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
   int num_frames = 24;
   int i = 0;
   while (sol_sock)
   {
      double t = (double)(i % num_frames) / num_frames;
      ostringstream oss;
      oss << "Harmonic Solution (t = " << t << " T)";

      add( cos( 2.0 * M_PI * t), e_v_->real(),
           sin( 2.0 * M_PI * t), e_v_->imag(), *e_t_);
      sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
      sol_sock << "solution\n" << *pmesh_ << *e_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
   */
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
