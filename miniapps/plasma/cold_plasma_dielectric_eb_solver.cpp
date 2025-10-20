// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

#include <climits>

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

void ElectricFieldFromE::EvalE(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);
}

void MagneticFluxFromCurlE::EvalB(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   dErCoef_.Eval(Bi_, T, ip); Bi_ /= -omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ /=  omega_;
}

void ElectricFluxFromE::EvalD(ElementTransformation &T,
                              const IntegrationPoint &ip)
{
   epsrCoef_.Eval(eps_r_, T, ip);
   epsiCoef_.Eval(eps_i_, T, ip);

   EvalE(T, ip);

   eps_r_.Mult(Er_, Dr_);
   eps_i_.AddMult_a(-1.0, Ei_, Dr_);

   eps_i_.Mult(Er_, Di_);
   eps_r_.AddMult(Ei_, Di_);
}

void MagneticFieldFromCurlE::EvalH(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   EvalB(T, ip);

   real_t muInv = muInvCoef_.Eval(T, ip);

   Hr_.Set(muInv, Br_);
   Hi_.Set(muInv, Bi_);
}

real_t ElectricEnergyDensityReCoef::Eval(ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   EvalD(T, ip);

   real_t u = (Er_ * Dr_) + (Ei_ * Di_);

   return 0.5 * u;
}

real_t ElectricEnergyDensityImCoef::Eval(ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   EvalD(T, ip);

   real_t u = (Ei_ * Dr_) - (Er_ * Di_);

   return 0.5 * u;
}

real_t MagneticEnergyDensityReCoef::Eval(ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   EvalH(T, ip);

   real_t u = (Hr_ * Br_) + (Hi_ * Bi_);

   return 0.5 * u;
}

real_t EnergyDensityReCoef::Eval(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   EvalD(T, ip);
   EvalH(T, ip);

   real_t u = Er_ * Dr_ + Ei_ * Di_ + Hr_ * Br_ + Hi_ * Bi_;

   return 0.5 * u;
}

void PoyntingVectorReCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   EvalE(T, ip);
   EvalH(T, ip);

   S.SetSize(3);

   // S = 1/2 E x Conj(H)
   S[0] = Er_[1] * Hr_[2] - Er_[2] * Hr_[1] +
          Ei_[1] * Hi_[2] - Ei_[2] * Hi_[1] ;

   S[1] = Er_[2] * Hr_[0] - Er_[0] * Hr_[2] +
          Ei_[2] * Hi_[0] - Ei_[0] * Hi_[2] ;

   S[2] = Er_[0] * Hr_[1] - Er_[1] * Hr_[0] +
          Ei_[0] * Hi_[1] - Ei_[1] * Hi_[0] ;

   S *= 0.5;
}

void PoyntingVectorImCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   EvalE(T, ip);
   EvalH(T, ip);

   S.SetSize(3);

   // S = 1/2 E x Conj(H)
   S[0] = -(Er_[1] * Hi_[2] - Er_[2] * Hi_[1]) +
          Ei_[1] * Hr_[2] - Ei_[2] * Hr_[1];

   S[1] = -(Er_[2] * Hi_[0] - Er_[0] * Hi_[2]) +
          Ei_[2] * Hr_[0] - Ei_[0] * Hr_[2];

   S[2] = -(Er_[0] * Hi_[1] - Er_[1] * Hi_[0]) +
          Ei_[0] * Hr_[1] - Ei_[1] * Hr_[0];

   S *= 0.5;
}

void MinkowskiMomentumDensityReCoef::Eval(Vector &G, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   EvalD(T, ip);
   EvalB(T, ip);

   G.SetSize(3);

   // 1/2 D x Conj(B)
   G[0] = Dr_[1] * Br_[2] - Dr_[2] * Br_[1] +
          Di_[1] * Bi_[2] - Di_[2] * Bi_[1] ;

   G[1] = Dr_[2] * Br_[0] - Dr_[0] * Br_[2] +
          Di_[2] * Bi_[0] - Di_[0] * Bi_[2] ;

   G[2] = Dr_[0] * Br_[1] - Dr_[1] * Br_[0] +
          Di_[0] * Bi_[1] - Di_[1] * Bi_[0] ;

   G *= 0.5;
}

void MinkowskiMomentumDensityImCoef::Eval(Vector &G, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   EvalD(T, ip);
   EvalB(T, ip);

   G.SetSize(3);

   // 1/2 D x Conj(B)
   G[0] = -(Dr_[1] * Bi_[2] - Dr_[2] * Bi_[1]) +
          Di_[1] * Br_[2] - Di_[2] * Br_[1] ;

   G[1] = -(Dr_[2] * Bi_[0] - Dr_[0] * Bi_[2]) +
          Di_[2] * Br_[0] - Di_[0] * Br_[2] ;

   G[2] = -(Dr_[0] * Bi_[1] - Dr_[1] * Bi_[0]) +
          Di_[0] * Br_[1] - Di_[1] * Br_[0] ;

   G *= 0.5;
}

real_t EAlongBackgroundBCoef::Eval(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   EvalE(T, ip);
   BCoef_.Eval(B_, T, ip);

   const real_t magB = B_.Norml2();

   if (magB > 0.0)
   {
      return sqrt(pow(Er_ * B_, 2) + pow(Ei_ * B_, 2)) / magB;
   }

   return 0.0;
}

Maxwell2ndE::Maxwell2ndE(ParFiniteElementSpace & HCurlFESpace,
                         real_t omega,
                         ComplexOperator::Convention conv,
                         MatrixCoefficient & epsReCoef,
                         MatrixCoefficient & epsImCoef,
                         Coefficient & muInvCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef)
   : ParSesquilinearForm(&HCurlFESpace, conv),
     massReCoef_(omega, epsReCoef),
     massImCoef_(omega, epsImCoef),
     kmkReCoef_(kReCoef, kImCoef, &muInvCoef, true, -1.0),
     kmkImCoef_(kReCoef, kImCoef, &muInvCoef, false, -1.0),
     kmReCoef_(kReCoef, &muInvCoef, 1.0),
     kmImCoef_(kImCoef, &muInvCoef, -1.0)
{
   this->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef), NULL);
   this->AddDomainIntegrator(new VectorFEMassIntegrator(massReCoef_),
                             new VectorFEMassIntegrator(massImCoef_));

   if ( kReCoef || kImCoef )
   {
      this->AddDomainIntegrator(new VectorFEMassIntegrator(kmkReCoef_),
                                new VectorFEMassIntegrator(kmkImCoef_));
      this->AddDomainIntegrator(new MixedVectorCurlIntegrator(kmImCoef_),
                                new MixedVectorCurlIntegrator(kmReCoef_));
      this->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(kmImCoef_),
                                new MixedVectorWeakCurlIntegrator(kmReCoef_));
   }
}

void Maxwell2ndE::Assemble()
{
   this->ParSesquilinearForm::Assemble();
   this->Finalize();
}

CurrentSourceE::CurrentSourceE(ParFiniteElementSpace & HCurlFESpace,
                               ParFiniteElementSpace & HDivFESpace,
                               real_t omega,
                               ComplexOperator::Convention conv,
                               const CmplxVecCoefArray & jsrc,
                               const CmplxVecCoefArray & ksrc,
                               VectorCoefficient * kReCoef,
                               VectorCoefficient * kImCoef)
   : ParComplexLinearForm(&HCurlFESpace, conv),
     omega_(omega),
     jt_(&HDivFESpace),
     kt_(&HCurlFESpace),
     jtilde_(jsrc.Size()),
     ktilde_(ksrc.Size())
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

      this->AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(*jtilde_[i]->imag),
         new VectorFEDomainLFIntegrator(*jtilde_[i]->real),
         jtilde_[i]->attr_marker);
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

      this->AddBoundaryIntegrator(
         new VectorFEBoundaryTangentLFIntegrator(*ktilde_[i]->imag),
         new VectorFEBoundaryTangentLFIntegrator(*ktilde_[i]->real),
         ktilde_[i]->attr_marker);
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
                         real_t omega,
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

real_t
FaradaysLaw::GetBFieldError(const VectorCoefficient & BReCoef,
                            const VectorCoefficient & BImCoef) const
{
   ParFiniteElementSpace * fes =
      const_cast<ParFiniteElementSpace*>(b_.ParFESpace());
   ParComplexGridFunction z(fes);
   z = 0.0;

   real_t solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(BReCoef),
                                     const_cast<VectorCoefficient&>(BImCoef));


   real_t solErr = b_.ComputeL2Error(const_cast<VectorCoefficient&>(BReCoef),
                                     const_cast<VectorCoefficient&>(BImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

GausssLaw::GausssLaw(ParFiniteElementSpace & HDivFESpace,
                     ParFiniteElementSpace & L2FESpace,
                     VectorCoefficient * kReCoef,
                     VectorCoefficient * kImCoef)
   : HDivFESpace_(HDivFESpace),
     df_(&L2FESpace),
     div_(&HDivFESpace_, &L2FESpace),
     kReDot_(NULL),
     kImDot_(NULL),
     assembled_(false)
{
   if (kReCoef)
   {
      kReDot_ = new ParDiscreteLinearOperator(&HDivFESpace_, &L2FESpace);
      kReDot_->AddDomainInterpolator(
         new VectorInnerProductInterpolator(*kReCoef));
   }
   if (kImCoef)
   {
      kImDot_ = new ParDiscreteLinearOperator(&HDivFESpace_, &L2FESpace);
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

   assembled_ = false;
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

   assembled_ = true;
}

void GausssLaw::ComputeDiv(const ParComplexGridFunction &f,
                           ParComplexGridFunction &df)
{
   if (!assembled_) { this->Assemble(); }

   // df = Div(f)
   div_.Mult(f.real(), df.real());
   div_.Mult(f.imag(), df.imag());

   if (kReDot_)
   {
      // df += i Re(k) . f
      kReDot_->AddMult(f.real(), df.imag(),  1.0);
      kReDot_->AddMult(f.imag(), df.real(), -1.0);
   }
   if (kImDot_)
   {
      // df -= Im(k) . f
      kImDot_->AddMult(f.real(), df.real(), -1.0);
      kImDot_->AddMult(f.imag(), df.imag(), -1.0);
   }
}

Displacement::Displacement(const ParComplexGridFunction &e,
                           ParFiniteElementSpace & HDivFESpace,
                           MatrixCoefficient & epsReCoef,
                           MatrixCoefficient & epsImCoef)
   : d_(&HDivFESpace),
     dReCoef_(&epsReCoef, &epsImCoef, &e.real(), &e.imag()),
     dImCoef_(&epsReCoef, &epsImCoef, &e.real(), &e.imag()),
     d_lf_(&HDivFESpace),
     m_(&HDivFESpace)
{
   d_lf_.AddDomainIntegrator(new VectorFEDomainLFIntegrator(dReCoef_),
                             new VectorFEDomainLFIntegrator(dImCoef_));

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
   m_.Finalize();
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
/*
SheathPotential::SheathPotential(real_t omega,
                                 const Array<ComplexCoefficientByAttr*> & sbc,
                                 const ParComplexGridFunction &d,
                                 ParFiniteElementSpace & H1,
                                 ParFiniteElementSpace & HCurl,
                                 ParFiniteElementSpace & L2,
                                 bool pa)
   : pa_(pa),
     omega_(omega),
     sbc_(sbc),
     d_(d),
     fes_h1_(H1),
     fes_nd_(HCurl),
     fes_rt_(*d.ParFESpace()),
     // fes_l2_(L2),
     // fec_rtt_(d.ParFESpace()->GetElementOrder(0),
     //        H1.GetParMesh()->Dimension()),
     fec_rtt_(fes_rt_.FEColl()->GetTraceCollection()),
     fes_rtt_(H1.GetParMesh(), fec_rtt_),
     grad_(&fes_h1_, &fes_nd_),
     phi_h1_(&fes_h1_),
     // phi_l2_(&fes_l2_),
     phi_rtt_(&fes_rtt_),
     PhiReCoef_(&phi_rtt_.real()),
     PhiImCoef_(&phi_rtt_.imag()),
     zeroCoef_(0.0),
     phi_lf_(&H1),
     m_(&H1)
{
   phi_h1_ = 0.0;
   // phi_l2_ = 0.0;
   phi_rtt_ = 0.0;

   ParMesh & pmesh = *(d_.ParFESpace()->GetParMesh());
   sbc_marker_.SetSize(pmesh.bdr_attributes.Max());
   sbc_marker_ = 0;

   for (int i=0; i<sbc_.Size(); i++)
   {
      for (int j=0; j<sbc_[i]->attr.Size(); j++)
      {
         sbc_marker_[sbc_[i]->attr[j] - 1] = 1;
      }

      SheathBase *z_r = dynamic_cast<SheathBase*>(sbc_[i]->real);
      SheathBase *z_i = dynamic_cast<SheathBase*>(sbc_[i]->imag);

      if (z_r == NULL && z_i == NULL)
      {
         cout << "Sheath Impedance coefficients not of type SheathImpedance"
              << endl;
      }
      if (z_r)
      {
         z_r->SetPotential(phi_h1_);
      }
      if (z_i)
      {
         z_i->SetPotential(phi_h1_);
      }
   }

   this->UpdateDofs();
   // phi_lf_.AddBoundaryIntegrator(new DomainLFIntegrator(PhiReCoef_),
   //           new DomainLFIntegrator(PhiImCoef_),
   //           sbc_marker_);

   if (pa_) { m_.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m_.AddDomainIntegrator(new MassIntegrator(zeroCoef_)); // set sparsity pattern
   m_.AddBoundaryIntegrator(new MassIntegrator);
}

SheathPotential::~SheathPotential()
{
   delete fec_rtt_;
}

void SheathPotential::NegGrad(ParComplexGridFunction &e)
{
   grad_.Mult(phi_h1_.real(), e.real());
   grad_.Mult(phi_h1_.imag(), e.imag());

   e *= -1.0;
}

void SheathPotential::UpdateDofs()
{
   Array<int> sbc_h1_tdof;
   fes_h1_.GetEssentialTrueDofs(sbc_marker_, sbc_h1_tdof);

   Array<int> sbc_h1_tdof_marker;
   fes_h1_.ListToMarker(sbc_h1_tdof, fes_h1_.GetTrueVSize(),
                        sbc_h1_tdof_marker, 1);

  // {
  //   ofstream ofs("sbc_h1_tdof_marker.dat");
  //   sbc_h1_tdof_marker.Print(ofs);
  //   ofs.close();
  // }

   // Invert marker
   for (int i=0; i<sbc_h1_tdof_marker.Size(); i++)
   {
      sbc_h1_tdof_marker[i] = 1 - sbc_h1_tdof_marker[i];
   }

  // {
  //   ofstream ofs("non_sbc_h1_tdof_marker.dat");
  //   sbc_h1_tdof_marker.Print(ofs);
  //   ofs.close();
  // }

   fes_h1_.MarkerToList(sbc_h1_tdof_marker, non_sbc_h1_tdofs_);

   // {
   //   ofstream ofs("non_sbc_h1_tdofs.dat");
   //   non_sbc_h1_tdofs_.Print(ofs, 1);
   //   ofs.close();
   // }
   // {
   //   ofstream ofs("sbc_marker.dat");
   //   sbc_marker_.Print(ofs);
   //   ofs.close();
   // }
   // {
   //   ofstream ofs("sbc_h1_tdof.dat");
   //   sbc_h1_tdof.Print(ofs);
   //   ofs.close();
   // }
}

void SheathPotential::Update()
{
   phi_h1_.Update();
   // phi_l2_.Update();
   fes_rtt_.Update(false);
   phi_rtt_.Update();
   this->UpdateDofs();
   grad_.Update();
   m_.Update();
   phi_lf_.Update();
}

void SheathPotential::Assemble()
{
   grad_.Assemble();
   m_.Assemble(0);
   if (!pa_) { m_.Finalize(); }
}

void SheathPotential::ComputePhi()
{
   ParMesh & pmesh = *(d_.ParFESpace()->GetParMesh());

   Array<int> vdofs_rt;
   Array<int> vdofs_rtt;
   Array<int> vdofs_h1;

   Vector lvec_d_r_rt, lvec_d_i_rt;
   Vector lvec_phi0_r_rtt, lvec_phi0_i_rtt;
   Vector lvec_phi1_r_rtt, lvec_phi1_i_rtt;
   Vector lvec_phi_r_h1, lvec_phi_i_h1;

   DenseMatrix shape_rtt;
   Vector      shape_rtt_ip;

   DenseMatrix shape_h1;
   Vector      shape_h1_ip;

   Vector tmp_r, tmp_i;

   phi_lf_.real() = 0.0;
   phi_lf_.imag() = 0.0;

   minit_ = INT_MAX;
   maxit_ = 0;
   fpslv_ = 0;
   sumit_ = 0;

   for (int be=0; be<pmesh.GetNBE(); be++)
   {
      const int bdr_attr = pmesh.GetBdrAttribute(be);
      if (sbc_marker_[bdr_attr-1] == 0) { continue; }

      for (int i=0; i<sbc_.Size(); i++)
      {
         if (sbc_[i]->attr_marker[bdr_attr-1] == 0) { continue; }

         fes_rt_.GetBdrElementVDofs(be, vdofs_rt);
         fes_rtt_.GetBdrElementVDofs(be, vdofs_rtt);
         fes_h1_.GetBdrElementVDofs(be, vdofs_h1);

         d_.real().GetSubVector(vdofs_rt, lvec_d_r_rt);
         d_.imag().GetSubVector(vdofs_rt, lvec_d_i_rt);

         phi_rtt_.real().GetSubVector(vdofs_rtt, lvec_phi0_r_rtt);
         phi_rtt_.imag().GetSubVector(vdofs_rtt, lvec_phi0_i_rtt);

         lvec_phi1_r_rtt = lvec_phi0_r_rtt;
         lvec_phi1_i_rtt = lvec_phi0_i_rtt;

         // We have found a boundary element on one of the sheath boundaries
         // const FiniteElement &rt = *fes_rt_.GetBE(be);
         const FiniteElement &rtt = *fes_rtt_.GetBE(be);
         const FiniteElement &h1 = *fes_h1_.GetBE(be);
         ElementTransformation &T = *fes_rt_.GetBdrElementTransformation(be);

         // Compute average value of sheath impedance on this boundary element
         const IntegrationRule &ir = IntRules.Get(h1.GetGeomType(),
                                                  2 * h1.GetOrder() + 1);

         shape_rtt.SetSize(vdofs_rtt.Size(), ir.GetNPoints());
         shape_h1.SetSize(vdofs_h1.Size(), ir.GetNPoints());

         real_t area = 0.0;

         for (int j=0; j<ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T.SetIntPoint(&ip);

            const real_t detJ = T.Weight();
            area += ip.weight * detJ;

            shape_rtt.GetColumnReference(j, shape_rtt_ip);
            rtt.CalcPhysShape(T, shape_rtt_ip);

            shape_h1.GetColumnReference(j, shape_h1_ip);
            h1.CalcPhysShape(T, shape_h1_ip);
         }

         SheathImpedance * zCoef = dynamic_cast<SheathImpedance*>(sbc_[i]->real);

         int it = 0;
         while (it < 10)
         {
            lvec_phi1_r_rtt = lvec_phi0_r_rtt;
            lvec_phi1_i_rtt = lvec_phi0_i_rtt;

            complex<real_t> zAvg(0.0, 0.0);
            // cout << "computing zAvg" << endl;
            for (int j=0; j<ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               T.SetIntPoint(&ip);

               shape_rtt.GetColumnReference(j, shape_rtt_ip);

               complex<real_t> phi(lvec_phi1_r_rtt * shape_rtt_ip,
                                   lvec_phi1_i_rtt * shape_rtt_ip);

               complex<real_t> z = zCoef->z(phi, T, ip);
               // cout << ip.x << " phi " << phi << " -> z(phi) " << z << endl;
               const real_t detJ = T.Weight();
               zAvg += z * ip.weight * detJ;
            }
            zAvg /= area;
            // cout << "computed zAvg in " << it << " iterations " << zAvg << endl;
            // phi = i omega z D
            add(omega_ * zAvg.imag(), lvec_d_r_rt,
                -omega_ * zAvg.real(), lvec_d_i_rt, lvec_phi0_r_rtt);
            add(omega_ * zAvg.real(), lvec_d_r_rt,
                -omega_ * zAvg.imag(), lvec_d_i_rt, lvec_phi0_i_rtt);

            lvec_phi1_r_rtt -= lvec_phi0_r_rtt;
            lvec_phi1_i_rtt -= lvec_phi0_i_rtt;

            real_t norm = sqrt(lvec_phi0_r_rtt * lvec_phi0_r_rtt +
                               lvec_phi0_i_rtt * lvec_phi0_i_rtt);
            real_t diff = sqrt(lvec_phi1_r_rtt * lvec_phi1_r_rtt +
                               lvec_phi1_i_rtt * lvec_phi1_i_rtt);

            it++;

            //cout << it << " diff vs norm " << diff << " " << norm << endl;
            if (it > 1 && diff < 1e-4 * norm)
            {
               break;
            }
         }

         minit_ = std::min(it, minit_);
         maxit_ = std::max(it, maxit_);
         sumit_ += it;
         fpslv_++;

         phi_rtt_.real().SetSubVector(vdofs_rtt, lvec_phi0_r_rtt);
         phi_rtt_.imag().SetSubVector(vdofs_rtt, lvec_phi0_i_rtt);

         tmp_r.SetSize(ir.GetNPoints());
         tmp_i.SetSize(ir.GetNPoints());
         shape_rtt.MultTranspose(lvec_phi0_r_rtt, tmp_r);
         shape_rtt.MultTranspose(lvec_phi0_i_rtt, tmp_i);

         lvec_phi_r_h1.SetSize(vdofs_h1.Size());
         lvec_phi_i_h1.SetSize(vdofs_h1.Size());

         lvec_phi_r_h1 = 0.0;
         lvec_phi_i_h1 = 0.0;

         for (int j=0; j<ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T.SetIntPoint(&ip);
            const real_t detJ = T.Weight();

            shape_h1.GetColumnReference(j, shape_h1_ip);

            lvec_phi_r_h1.Add(ip.weight * detJ * tmp_r[j], shape_h1_ip);
            lvec_phi_i_h1.Add(ip.weight * detJ * tmp_i[j], shape_h1_ip);
         }

         phi_lf_.real().AddElementVector(vdofs_h1, lvec_phi_r_h1);
         phi_lf_.imag().AddElementVector(vdofs_h1, lvec_phi_i_h1);
      }
   }

   OperatorPtr M;
   m_.FormSystemMatrix(non_sbc_h1_tdofs_, M);

  // {
  //   M.As<HypreParMatrix>()->Print("M.mat");
  // }

   int tvsize = fes_h1_.TrueVSize();
   PHI_.SetSize(tvsize);
   RHS_.SetSize(tvsize);

   Operator *diag = NULL;
   Operator *pcg = NULL;
   if (pa_)
   {
      diag = new OperatorJacobiSmoother(m_, non_sbc_h1_tdofs_);
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
      cg->SetLogging(3);
      cg->SetPreconditioner(static_cast<HypreDiagScale&>(*diag));
      cg->SetTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }

   // phi_lf_.Assemble();
   // phi_lf_ = 0.0;
   phi_lf_.SyncAlias();

   phi_lf_.real().ParallelAssemble(RHS_);
   PHI_ = 0.0;
   pcg->Mult(RHS_, PHI_);
   phi_h1_.real().Distribute(PHI_);

   phi_lf_.imag().ParallelAssemble(RHS_);
   if (phi_lf_.GetConvention() == ComplexOperator::BLOCK_SYMMETRIC)
   {
      RHS_ *= -1.0;
   }
   PHI_ = 0.0;
   pcg->Mult(RHS_, PHI_);
   phi_h1_.imag().Distribute(PHI_);

   delete diag;
   delete pcg;
}

void SheathPotential::PrintStatistics() const
{
   int glbminit = INT_MAX;
   int glbmaxit = 0;
   int glbfpslv = 0;
   long int sumit = sumit_; // Avoids type warning in MPI_Allreduce
   long int glbsumit = 0;

   MPI_Allreduce(&minit_, &glbminit, 1, MPI_INTEGER, MPI_MIN,
                 fes_rt_.GetComm());
   MPI_Allreduce(&maxit_, &glbmaxit, 1, MPI_INTEGER, MPI_MAX,
                 fes_rt_.GetComm());
   MPI_Allreduce(&fpslv_, &glbfpslv, 1, MPI_INTEGER, MPI_SUM,
                 fes_rt_.GetComm());
   MPI_Allreduce(&sumit, &glbsumit, 1, MPI_LONG, MPI_SUM,
                 fes_rt_.GetComm());

   int myrank = 0;
   MPI_Comm_rank(fes_rt_.GetComm(), &myrank);
   if (myrank == 0)
   {
      cout << "Sheath Potential Statistics: ";
      cout << "iterations (min, max, avg) = ( "
           << glbminit << ", " << glbmaxit << ", " << glbsumit / glbfpslv << ")"
           << endl;
   }
}
*/
ParallelElectricFieldVisObject::ParallelElectricFieldVisObject(
   const std::string & field_name,
   VectorCoefficient & BCoef,
   shared_ptr<L2_ParFESpace> sfes,
   bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo), BCoef_(BCoef)
{}

void ParallelElectricFieldVisObject::PrepareVisField(const
                                                     ParComplexGridFunction &e,
                                                     VectorCoefficient *kr,
                                                     VectorCoefficient *ki)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());

   NormalizedVectorCoefficient BUnit(BCoef_);

   InnerProductCoefficient EBr(BUnit, Er);
   InnerProductCoefficient EBi(BUnit, Ei);

   this->ComplexScalarFieldVisObject::PrepareVisField(EBr, EBi, kr, ki);
}

ElectricEnergyDensityVisObject::ElectricEnergyDensityVisObject(
   const std::string & field_name,
   shared_ptr<L2_ParFESpace> sfes,
   bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void ElectricEnergyDensityVisObject::PrepareVisField(const
                                                     ParComplexGridFunction &e,
                                                     MatrixCoefficient &epsr,
                                                     MatrixCoefficient &epsi)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());

   ElectricEnergyDensityReCoef ur(Er, Ei, epsr, epsi);
   ElectricEnergyDensityImCoef ui(Er, Ei, epsr, epsi);

   this->PrepareVisField(ur, ui, NULL, NULL);
}
/*
ElectricEnergyDensityEDVisObject::ElectricEnergyDensityEDVisObject(
   const std::string & field_name,
   shared_ptr<L2_ParFESpace> sfes,
   bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void ElectricEnergyDensityEDVisObject::PrepareVisField(const
                                                       ParComplexGridFunction &e,
                                                       const
                                                       ParComplexGridFunction &d)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());

   VectorGridFunctionCoefficient Dr(&d.real());
   VectorGridFunctionCoefficient Di(&d.imag());

   ElectricEnergyDensityEDCoef ur(Er, Ei, Dr, Di);
   ConstantCoefficient ui(0.0);

   this->PrepareVisField(ur, ui, NULL, NULL);
}
*/
MagneticEnergyDensityVisObject::MagneticEnergyDensityVisObject(
   const std::string & field_name,
   shared_ptr<L2_ParFESpace> sfes,
   bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void MagneticEnergyDensityVisObject::PrepareVisField(const
                                                     ParComplexGridFunction &e,
                                                     real_t omega,
                                                     Coefficient &muInv)
{
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   MagneticEnergyDensityReCoef ur(omega, dEr, dEi, muInv);
   MagneticEnergyDensityImCoef ui;

   this->PrepareVisField(ur, ui, NULL, NULL);
}

EnergyDensityVisObject::EnergyDensityVisObject(const std::string & field_name,
                                               shared_ptr<L2_ParFESpace> sfes,
                                               bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void EnergyDensityVisObject::PrepareVisField(const ParComplexGridFunction &e,
                                             real_t omega,
                                             MatrixCoefficient &epsr,
                                             MatrixCoefficient &epsi,
                                             Coefficient &muInv)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());
   CurlGridFunctionCoefficient dEr(&e.real());
   CurlGridFunctionCoefficient dEi(&e.imag());

   EnergyDensityReCoef ur(omega, Er, Ei, dEr, dEi, epsr, epsi, muInv);
   EnergyDensityImCoef ui(omega, Er, Ei, dEr, dEi, epsr, epsi, muInv);

   this->PrepareVisField(ur, ui, NULL, NULL);
}

PoyntingVectorVisObject::PoyntingVectorVisObject(const std::string & field_name,
                                                 shared_ptr<L2_ParFESpace> vfes)
   : ComplexVectorFieldVisObject(field_name, vfes)
{}

void PoyntingVectorVisObject::PrepareVisField(const ParComplexGridFunction &e,
                                              real_t omega,
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
   shared_ptr<L2_ParFESpace> vfes)
   : ComplexVectorFieldVisObject(field_name, vfes)
{}

void MinkowskiMomentumDensityVisObject::PrepareVisField(
   const ParComplexGridFunction &e,
   real_t omega,
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

EAlongBackgroundBVisObject::EAlongBackgroundBVisObject(
   const std::string & field_name,
   std::shared_ptr<L2_ParFESpace> sfes,
   bool cyl, bool pseudo)
   : ScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void
EAlongBackgroundBVisObject::PrepareVisField(const ParComplexGridFunction &e,
                                            VectorCoefficient &BCoef)
{
   VectorGridFunctionCoefficient Er(&e.real());
   VectorGridFunctionCoefficient Ei(&e.imag());

   EAlongBackgroundBCoef ebCoef(Er, Ei, BCoef);

   this->PrepareVisField(ebCoef);
}

TensorCompVisObject::TensorCompVisObject(const std::string & field_name,
                                         shared_ptr<L2_ParFESpace> sfes,
                                         bool cyl, bool pseudo)
   : ComplexScalarFieldVisObject(field_name, sfes, cyl, pseudo)
{}

void TensorCompVisObject::PrepareVisField(MatrixCoefficient &mr,
                                          MatrixCoefficient &mi,
                                          int i, int j)
{
   TensorCompCoef mrCoef(mr, i, j);
   TensorCompCoef miCoef(mi, i, j);

   this->PrepareVisField(mrCoef, miCoef, NULL, NULL);
}

void CPDVisBase::set_bool_flags(unsigned int num_flags,
                                unsigned int active_flags_mask,
                                Array<bool> &flags)
{
   flags.SetSize(num_flags);
   for (int i=0; i<num_flags; i++)
   {
      flags[i] = active_flags_mask >> i & 1;
   }
}

void CPDVisBase::SetOptions(const Array<bool> &opts)
{
   for (int i=0; i<opts.Size(); i++)
   {
      if (opts[i])
      {
         this->SetVisFlag(i);
      }
      else
      {
         this->ClearVisFlag(i);
      }
   }
}

const string CPDInputVis::opt_str_[] =
{
   "bb", "background-b", "background magnetic flux",
   "vc", "volume-current", "volumetric current density",
   "id", "ion-densities", "ion species densities",
   "it", "ion-temperatures", "ion species temperatures",
   "ed", "electron-density", "electron density",
   "et", "electron-temperature", "electron temperature",
   "ss", "stix-s", "Stix S coefficient",
   "sd", "stix-d", "Stix D coefficient",
   "sp", "stix-p", "Stix P coefficient",
   "sl", "stix-l", "Stix L coefficient",
   "sr", "stix-r", "Stix R coefficient",
   "sisp", "stix-inv-sp", "Stix 1 / (S P) coefficient",
   "wl", "wavelength-l", "wavelength L",
   "wr", "wavelength-r", "wavelength R",
   "wo", "wavelength-o", "wavelength O",
   "wx", "wavelength-x", "wavelength X",
   "dl", "skin-depth-l", "skin-depth L",
   "dr", "skin-depth-r", "skin-depth R",
   "do", "skin-depth-o", "skin-depth O",
   "dx", "skin-depth-x", "skin-depth X"
};

std::array<std::string, CPDInputVis::NUM_VIS_FIELDS> CPDInputVis::optt_;
std::array<std::string, CPDInputVis::NUM_VIS_FIELDS> CPDInputVis::optlt_;
std::array<std::string, CPDInputVis::NUM_VIS_FIELDS> CPDInputVis::optf_;
std::array<std::string, CPDInputVis::NUM_VIS_FIELDS> CPDInputVis::optlf_;
std::array<std::string, CPDInputVis::NUM_VIS_FIELDS> CPDInputVis::optd_;

void CPDInputVis::AddOptions(OptionsParser &args, Array<bool> &opts)
{
   for (int i=0; i<NUM_VIS_FIELDS; i++)
   {
      optt_[i]  = "-vi" + opt_str_[3*i+0];
      optlt_[i] = "--vis-input-" + opt_str_[3*i+1];
      optf_[i]  = "-no-vi" + opt_str_[3*i+0];
      optlf_[i] = "--no-vis-input-" + opt_str_[3*i+1];
      optd_[i]  = "Visualize input " + opt_str_[3*i+2];
      args.AddOption(&opts[i],
                     optt_[i].c_str(), optlt_[i].c_str(),
                     optf_[i].c_str(), optlf_[i].c_str(),
                     optd_[i].c_str());
   }
}

CPDInputVis::CPDInputVis(StixParams &stixParams,
                         std::shared_ptr<L2_ParFESpace> l2_sfes,
                         std::shared_ptr<L2_ParFESpace> l2_vfes,
                         unsigned int vis_flag,
                         bool cyl)
   : CPDVisBase(vis_flag),
     sfes_(l2_sfes),
     vfes_(l2_vfes),
     BCoef_(stixParams.BCoef),
     B_("BackgroundB", l2_vfes),
     J_("VolumeCurrent", l2_vfes),
     //
     numIonSpec_(stixParams.specDensityCoef.GetVDim() - 1),
     ionDensityCoefs_(numIonSpec_),
     ionTempCoefs_(numIonSpec_),
     ionDensities_(numIonSpec_),
     ionTemps_(numIonSpec_),
     //
     electronDensityCoef_(numIonSpec_,stixParams.specDensityCoef),
     electronTempCoef_(numIonSpec_,stixParams.specTemperatureCoef),
     electronDensity_("ElectronDensity", l2_sfes, cyl, false),
     electronTemp_("ElectronTemperature", l2_sfes, cyl, false),
     //
     stixSReCoef_(stixParams, StixCoef::REAL_PART),
     stixSImCoef_(stixParams, StixCoef::IMAG_PART),
     stixDReCoef_(stixParams, StixCoef::REAL_PART),
     stixDImCoef_(stixParams, StixCoef::IMAG_PART),
     stixPReCoef_(stixParams, StixCoef::REAL_PART),
     stixPImCoef_(stixParams, StixCoef::IMAG_PART),
     stixLReCoef_(stixParams, StixCoef::REAL_PART),
     stixLImCoef_(stixParams, StixCoef::IMAG_PART),
     stixRReCoef_(stixParams, StixCoef::REAL_PART),
     stixRImCoef_(stixParams, StixCoef::IMAG_PART),
     stixInvSPReCoef_(stixParams, StixCoef::REAL_PART),
     stixInvSPImCoef_(stixParams, StixCoef::IMAG_PART),
     stixS_("StixS", l2_sfes, cyl, false),
     stixD_("StixD", l2_sfes, cyl, false),
     stixP_("StixP", l2_sfes, cyl, false),
     stixL_("StixL", l2_sfes, cyl, false),
     stixR_("StixR", l2_sfes, cyl, false),
     stixInvSP_("StixInvSP", l2_sfes, cyl, false),
     //
     lambdaLCoef_('L', stixParams, StixCoef::REAL_PART),
     lambdaRCoef_('R', stixParams, StixCoef::REAL_PART),
     lambdaOCoef_('O', stixParams, StixCoef::REAL_PART),
     lambdaXCoef_('X', stixParams, StixCoef::REAL_PART),
     waveLengthL_("LambdaL", l2_sfes, cyl, false),
     waveLengthR_("LambdaR", l2_sfes, cyl, false),
     waveLengthO_("LambdaO", l2_sfes, cyl, false),
     waveLengthX_("LambdaX", l2_sfes, cyl, false),
     //
     deltaLCoef_('L', stixParams, StixCoef::IMAG_PART),
     deltaRCoef_('R', stixParams, StixCoef::IMAG_PART),
     deltaOCoef_('O', stixParams, StixCoef::IMAG_PART),
     deltaXCoef_('X', stixParams, StixCoef::IMAG_PART),
     skinDepthL_("DeltaL", l2_sfes, cyl, false),
     skinDepthR_("DeltaR", l2_sfes, cyl, false),
     skinDepthO_("DeltaO", l2_sfes, cyl, false),
     skinDepthX_("DeltaX", l2_sfes, cyl, false)
{
   for (int i=0; i<numIonSpec_; i++)
   {
      ionDensityCoefs_[i] =
         new ComponentCoefficient(i,stixParams.specDensityCoef);
      ionTempCoefs_[i] =
         new ComponentCoefficient(i,stixParams.specTemperatureCoef);

      ostringstream oss_den, oss_temp;
      oss_den << "IonDensity";
      oss_temp << "IonTemperature";
      if (numIonSpec_ > 1)
      {
         oss_den << " (Species " << i+1 << ")";
         oss_temp << " (Species " << i+1 << ")";
      }
      ionDensities_[i] =
         new ScalarFieldVisObject(oss_den.str(), l2_sfes, cyl, false);
      ionTemps_[i] =
         new ScalarFieldVisObject(oss_temp.str(), l2_sfes, cyl, false);
   }
}

CPDInputVis::~CPDInputVis()
{
   for (int i=0; i<numIonSpec_; i++)
   {
      delete ionDensityCoefs_[i];
      delete ionTempCoefs_[i];
      delete ionDensities_[i];
      delete ionTemps_[i];
   }
}

void CPDInputVis::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   if (CheckVisFlag(BACKGROUND_B)) { B_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(VOLUMETRIC_CURRENT)) { J_.RegisterVisItFields(visit_dc); }

   if (CheckVisFlag(ION_DENSITIES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionDensities_[i]->RegisterVisItFields(visit_dc);
      }
   }
   if (CheckVisFlag(ION_TEMPERATURES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionTemps_[i]->RegisterVisItFields(visit_dc);
      }
   }

   if (CheckVisFlag(ELECTRON_DENSITY))
   { electronDensity_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(ELECTRON_TEMPERATURE))
   { electronTemp_.RegisterVisItFields(visit_dc); }

   if (CheckVisFlag(STIX_S)) { stixS_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(STIX_D)) { stixD_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(STIX_P)) { stixP_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(STIX_L)) { stixL_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(STIX_R)) { stixR_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(STIX_INVSP)) { stixInvSP_.RegisterVisItFields(visit_dc); }

   if (CheckVisFlag(WAVELENGTH_L))
   { waveLengthL_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(WAVELENGTH_R))
   { waveLengthR_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(WAVELENGTH_O))
   { waveLengthO_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(WAVELENGTH_X))
   { waveLengthX_.RegisterVisItFields(visit_dc); }

   if (CheckVisFlag(SKIN_DEPTH_L))
   { skinDepthL_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(SKIN_DEPTH_R))
   { skinDepthR_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(SKIN_DEPTH_O))
   { skinDepthO_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(SKIN_DEPTH_X))
   { skinDepthX_.RegisterVisItFields(visit_dc); }
}

void CPDInputVis::PrepareVisFields(const ParComplexGridFunction & j,
                                   VectorCoefficient * kReCoef,
                                   VectorCoefficient * kImCoef)
{
   if (CheckVisFlag(BACKGROUND_B)) { B_.PrepareVisField(BCoef_); }
   if (CheckVisFlag(VOLUMETRIC_CURRENT))
   {
      J_.PrepareVisField(j, kReCoef, kImCoef);
   }

   if (CheckVisFlag(ION_DENSITIES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionDensities_[i]->PrepareVisField(*ionDensityCoefs_[i]);
      }
   }
   if (CheckVisFlag(ION_TEMPERATURES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionTemps_[i]->PrepareVisField(*ionTempCoefs_[i]);
      }
   }

   if (CheckVisFlag(ELECTRON_DENSITY))
   { electronDensity_.PrepareVisField(electronDensityCoef_); }
   if (CheckVisFlag(ELECTRON_TEMPERATURE))
   { electronTemp_.PrepareVisField(electronTempCoef_); }

   if (CheckVisFlag(STIX_S))
   { stixS_.PrepareVisField(stixSReCoef_, stixSImCoef_, NULL, NULL); }
   if (CheckVisFlag(STIX_D))
   { stixD_.PrepareVisField(stixDReCoef_, stixDImCoef_, NULL, NULL); }
   if (CheckVisFlag(STIX_P))
   { stixP_.PrepareVisField(stixPReCoef_, stixPImCoef_, NULL, NULL); }
   if (CheckVisFlag(STIX_L))
   { stixL_.PrepareVisField(stixLReCoef_, stixLImCoef_, NULL, NULL); }
   if (CheckVisFlag(STIX_R))
   { stixR_.PrepareVisField(stixRReCoef_, stixRImCoef_, NULL, NULL); }
   if (CheckVisFlag(STIX_INVSP))
   {
      stixInvSP_.PrepareVisField(stixInvSPReCoef_, stixInvSPImCoef_,
                                 NULL, NULL);
   }

   if (CheckVisFlag(WAVELENGTH_L))
   { waveLengthL_.PrepareVisField(lambdaLCoef_); }
   if (CheckVisFlag(WAVELENGTH_R))
   { waveLengthR_.PrepareVisField(lambdaRCoef_); }
   if (CheckVisFlag(WAVELENGTH_O))
   { waveLengthO_.PrepareVisField(lambdaOCoef_); }
   if (CheckVisFlag(WAVELENGTH_X))
   { waveLengthX_.PrepareVisField(lambdaXCoef_); }

   if (CheckVisFlag(SKIN_DEPTH_L)) { skinDepthL_.PrepareVisField(deltaLCoef_); }
   if (CheckVisFlag(SKIN_DEPTH_R)) { skinDepthR_.PrepareVisField(deltaRCoef_); }
   if (CheckVisFlag(SKIN_DEPTH_O)) { skinDepthO_.PrepareVisField(deltaOCoef_); }
   if (CheckVisFlag(SKIN_DEPTH_X)) { skinDepthX_.PrepareVisField(deltaXCoef_); }
}

void CPDInputVis::DisplayToGLVis()
{
   if (CheckVisFlag(BACKGROUND_B)) { B_.DisplayToGLVis(); }
   if (CheckVisFlag(VOLUMETRIC_CURRENT)) { J_.DisplayToGLVis(); }

   if (CheckVisFlag(ION_DENSITIES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionDensities_[i]->DisplayToGLVis();
      }
   }
   if (CheckVisFlag(ION_TEMPERATURES))
   {
      for (int i=0; i<numIonSpec_; i++)
      {
         ionTemps_[i]->DisplayToGLVis();
      }
   }

   if (CheckVisFlag(ELECTRON_DENSITY))
   { electronDensity_.DisplayToGLVis(); }
   if (CheckVisFlag(ELECTRON_TEMPERATURE))
   { electronTemp_.DisplayToGLVis(); }

   if (CheckVisFlag(STIX_S)) { stixS_.DisplayToGLVis(); }
   if (CheckVisFlag(STIX_D)) { stixD_.DisplayToGLVis(); }
   if (CheckVisFlag(STIX_P)) { stixP_.DisplayToGLVis(); }
   if (CheckVisFlag(STIX_L)) { stixL_.DisplayToGLVis(); }
   if (CheckVisFlag(STIX_R)) { stixR_.DisplayToGLVis(); }
   if (CheckVisFlag(STIX_INVSP)) { stixInvSP_.DisplayToGLVis(); }

   if (CheckVisFlag(WAVELENGTH_L)) { waveLengthL_.DisplayToGLVis(); }
   if (CheckVisFlag(WAVELENGTH_R)) { waveLengthR_.DisplayToGLVis(); }
   if (CheckVisFlag(WAVELENGTH_O)) { waveLengthO_.DisplayToGLVis(); }
   if (CheckVisFlag(WAVELENGTH_X)) { waveLengthX_.DisplayToGLVis(); }

   if (CheckVisFlag(SKIN_DEPTH_L)) { skinDepthL_.DisplayToGLVis(); }
   if (CheckVisFlag(SKIN_DEPTH_R)) { skinDepthR_.DisplayToGLVis(); }
   if (CheckVisFlag(SKIN_DEPTH_O)) { skinDepthO_.DisplayToGLVis(); }
   if (CheckVisFlag(SKIN_DEPTH_X)) { skinDepthX_.DisplayToGLVis(); }
}

void CPDInputVis::Update()
{
   sfes_->Update();
   vfes_->Update();

   B_.Update();
   J_.Update();

   stixS_.Update();
   stixD_.Update();
   stixP_.Update();
   stixL_.Update();
   stixR_.Update();
   stixInvSP_.Update();

   waveLengthL_.Update();
   waveLengthR_.Update();
   waveLengthO_.Update();
   waveLengthX_.Update();

   skinDepthL_.Update();
   skinDepthR_.Update();
   skinDepthO_.Update();
   skinDepthX_.Update();
}

const string CPDFieldVis::opt_str_[] =
{
   "ef", "e-field", "electric field",
   "bf", "b-flux", "magnetic flux",
   "df", "d-flux", "electric flux",
   "dbf", "div-b-flux", "divergence of magnetic flux",
   "ddf", "div-d-flux", "divergence of electric flux"
};

std::array<std::string, CPDFieldVis::NUM_VIS_FIELDS> CPDFieldVis::optt_;
std::array<std::string, CPDFieldVis::NUM_VIS_FIELDS> CPDFieldVis::optlt_;
std::array<std::string, CPDFieldVis::NUM_VIS_FIELDS> CPDFieldVis::optf_;
std::array<std::string, CPDFieldVis::NUM_VIS_FIELDS> CPDFieldVis::optlf_;
std::array<std::string, CPDFieldVis::NUM_VIS_FIELDS> CPDFieldVis::optd_;

void CPDFieldVis::AddOptions(OptionsParser &args, Array<bool> &opts)
{
   for (int i=0; i<NUM_VIS_FIELDS; i++)
   {
      optt_[i]  = "-v" + opt_str_[3*i+0];
      optlt_[i] = "--vis-" + opt_str_[3*i+1];
      optf_[i]  = "-no-v" + opt_str_[3*i+0];
      optlf_[i] = "--no-vis-" + opt_str_[3*i+1];
      optd_[i]  = "Visualize " + opt_str_[3*i+2];
      args.AddOption(&opts[i],
                     optt_[i].c_str(), optlt_[i].c_str(),
                     optf_[i].c_str(), optlf_[i].c_str(),
                     optd_[i].c_str());
   }
}

CPDFieldVis::CPDFieldVis(StixParams &stixParams,
                         std::shared_ptr<L2_ParFESpace> l2_sfes,
                         std::shared_ptr<L2_ParFESpace> l2_vfes,
                         const std::string hcurl_field_name,
                         const std::string hdiv_field_name,
                         unsigned int vis_flag)
   : CPDVisBase(vis_flag),
     sfes_(l2_sfes),
     vfes_(l2_vfes),
     HCurlField_(hcurl_field_name, l2_vfes),
     HDivField_(hdiv_field_name, l2_vfes),
     DivBField_("DivB", l2_sfes, false, true),
     DivDField_("DivD", l2_sfes, false, true)
{
}

void CPDFieldVis::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   if (CheckVisFlag(ELECTRIC_FIELD))
   { HCurlField_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(MAGNETIC_FLUX))
   { HDivField_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(DIV_MAGNETIC_FLUX))
   { DivBField_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(DIV_ELECTRIC_FLUX))
   { DivDField_.RegisterVisItFields(visit_dc); }
}

void CPDFieldVis::PrepareVisFields(const ParComplexGridFunction & e,
                                   const ParComplexGridFunction & b,
                                   const ParComplexGridFunction & db,
                                   const ParComplexGridFunction & dd,
                                   VectorCoefficient * kReCoef,
                                   VectorCoefficient * kImCoef)
{
   if (CheckVisFlag(ELECTRIC_FIELD))
   { HCurlField_.PrepareVisField(e, kReCoef, kImCoef); }
   if (CheckVisFlag(MAGNETIC_FLUX))
   { HDivField_.PrepareVisField(b, kReCoef, kImCoef); }
   if (CheckVisFlag(DIV_MAGNETIC_FLUX))
   { DivBField_.PrepareVisField(db, kReCoef, kImCoef); }
   if (CheckVisFlag(DIV_ELECTRIC_FLUX))
   { DivDField_.PrepareVisField(dd, kReCoef, kImCoef); }
}

void CPDFieldVis::DisplayToGLVis()
{
   if (CheckVisFlag(ELECTRIC_FIELD)) { HCurlField_.DisplayToGLVis(); }
   if (CheckVisFlag(MAGNETIC_FLUX)) { HDivField_.DisplayToGLVis(); }
   if (CheckVisFlag(DIV_MAGNETIC_FLUX)) { DivBField_.DisplayToGLVis(); }
   if (CheckVisFlag(DIV_ELECTRIC_FLUX)) { DivDField_.DisplayToGLVis(); }
}

void CPDFieldVis::Update()
{
   sfes_->Update();
   vfes_->Update();

   HCurlField_.Update();
   HDivField_.Update();
   DivBField_.Update();
   DivDField_.Update();
}

const string CPDOutputVis::opt_str_[] =
{
   "u", "energy-density", "energy density",
   "pf", "poynting-flux", "Poynting flux",
   "md", "momentum-density", "Minkowski momentum density",
   "ue", "electric-energy-density", "electric energy density",
   "um", "magnetic-energy-density", "magnetic energy density",
   "eb", "e-dot-b", "electric field along background B"
};

std::array<std::string, CPDOutputVis::NUM_VIS_FIELDS> CPDOutputVis::optt_;
std::array<std::string, CPDOutputVis::NUM_VIS_FIELDS> CPDOutputVis::optlt_;
std::array<std::string, CPDOutputVis::NUM_VIS_FIELDS> CPDOutputVis::optf_;
std::array<std::string, CPDOutputVis::NUM_VIS_FIELDS> CPDOutputVis::optlf_;
std::array<std::string, CPDOutputVis::NUM_VIS_FIELDS> CPDOutputVis::optd_;

void CPDOutputVis::AddOptions(OptionsParser &args, Array<bool> &opts)
{
   for (int i=0; i<NUM_VIS_FIELDS; i++)
   {
      optt_[i]  = "-vo" + opt_str_[3*i+0];
      optlt_[i] = "--vis-output-" + opt_str_[3*i+1];
      optf_[i]  = "-no-vo" + opt_str_[3*i+0];
      optlf_[i] = "--no-vis-output-" + opt_str_[3*i+1];
      optd_[i]  = "Visualize output " + opt_str_[3*i+2];
      args.AddOption(&opts[i],
                     optt_[i].c_str(), optlt_[i].c_str(),
                     optf_[i].c_str(), optlf_[i].c_str(),
                     optd_[i].c_str());
   }
}

CPDOutputVis::CPDOutputVis(StixParams &stixParams,
                           std::shared_ptr<L2_ParFESpace> l2_sfes,
                           std::shared_ptr<L2_ParFESpace> l2_vfes,
                           real_t omega,
                           unsigned int vis_flag,
                           bool cyl)
   : CPDVisBase(vis_flag),
     sfes_(l2_sfes),
     vfes_(l2_vfes),
     omega_(omega),
     BCoef_(stixParams.BCoef),
     epsReCoef_(stixParams, StixCoef::REAL_PART),
     epsImCoef_(stixParams, StixCoef::IMAG_PART),
     muInvCoef_(1.0 / mu0_),
     energyDensity_("EnergyDensity", l2_sfes, cyl, true),
     poyntingFlux_("PoyntingFlux", l2_vfes),
     momentumDensity_("MomentumDensity", l2_vfes),
     edEnergyDensity_("EDEnergyDensity", l2_sfes, cyl, true),
     bhEnergyDensity_("BHEnergyDensity", l2_sfes, cyl, true),
     ealongb_("EAlongBackgroundB", l2_sfes, cyl, true)
{
}

void CPDOutputVis::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   if (CheckVisFlag(ENERGY_DENSITY))
   { energyDensity_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(POYNTING_FLUX))
   { poyntingFlux_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(MOMENTUM_DENSITY))
   { momentumDensity_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(ED_ENERGY_DENSITY))
   { edEnergyDensity_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(BH_ENERGY_DENSITY))
   { bhEnergyDensity_.RegisterVisItFields(visit_dc); }
   if (CheckVisFlag(E_BACKGROUND_B))
   { ealongb_.RegisterVisItFields(visit_dc); }
}

void CPDOutputVis::PrepareVisFields(const ParComplexGridFunction & e)
{
   if (CheckVisFlag(ENERGY_DENSITY))
   {
      energyDensity_.PrepareVisField(e, omega_, epsReCoef_,
                                     epsImCoef_, muInvCoef_);
   }
   if (CheckVisFlag(POYNTING_FLUX))
   { poyntingFlux_.PrepareVisField(e, omega_, muInvCoef_); }
   if (CheckVisFlag(MOMENTUM_DENSITY))
   { momentumDensity_.PrepareVisField(e, omega_, epsReCoef_, epsImCoef_); }
   if (CheckVisFlag(ED_ENERGY_DENSITY))
   { edEnergyDensity_.PrepareVisField(e, epsReCoef_, epsImCoef_); }
   if (CheckVisFlag(BH_ENERGY_DENSITY))
   { bhEnergyDensity_.PrepareVisField(e, omega_, muInvCoef_); }
   if (CheckVisFlag(E_BACKGROUND_B))
   { ealongb_.PrepareVisField(e, BCoef_); }
}

void CPDOutputVis::DisplayToGLVis()
{
   if (CheckVisFlag(ENERGY_DENSITY)) { energyDensity_.DisplayToGLVis(); }
   if (CheckVisFlag(POYNTING_FLUX)) { poyntingFlux_.DisplayToGLVis(); }
   if (CheckVisFlag(MOMENTUM_DENSITY)) { momentumDensity_.DisplayToGLVis(); }
   if (CheckVisFlag(ED_ENERGY_DENSITY)) { edEnergyDensity_.DisplayToGLVis(); }
   if (CheckVisFlag(BH_ENERGY_DENSITY)) { bhEnergyDensity_.DisplayToGLVis(); }
   if (CheckVisFlag(E_BACKGROUND_B)) { ealongb_.DisplayToGLVis(); }
}

void CPDOutputVis::Update()
{
   sfes_->Update();
   vfes_->Update();

   energyDensity_.Update();
   poyntingFlux_.Update();
   momentumDensity_.Update();
   edEnergyDensity_.Update();
   bhEnergyDensity_.Update();
   ealongb_.Update();
}

const string CPDFieldAnim::opt_str_[] =
{
   "efa", "e-field-anim", "electric field animation",
   "bfa", "b-flux-anim", "magnetic flux animation"
};

std::array<std::string, CPDFieldAnim::NUM_VIS_FIELDS> CPDFieldAnim::optt_;
std::array<std::string, CPDFieldAnim::NUM_VIS_FIELDS> CPDFieldAnim::optlt_;
std::array<std::string, CPDFieldAnim::NUM_VIS_FIELDS> CPDFieldAnim::optf_;
std::array<std::string, CPDFieldAnim::NUM_VIS_FIELDS> CPDFieldAnim::optlf_;
std::array<std::string, CPDFieldAnim::NUM_VIS_FIELDS> CPDFieldAnim::optd_;

void CPDFieldAnim::AddOptions(OptionsParser &args, Array<bool> &opts)
{
   for (int i=0; i<NUM_VIS_FIELDS; i++)
   {
      optt_[i]  = "-v" + opt_str_[3*i+0];
      optlt_[i] = "--vis-" + opt_str_[3*i+1];
      optf_[i]  = "-no-v" + opt_str_[3*i+0];
      optlf_[i] = "--no-vis-" + opt_str_[3*i+1];
      optd_[i]  = "Visualize " + opt_str_[3*i+2];
      args.AddOption(&opts[i],
                     optt_[i].c_str(), optlt_[i].c_str(),
                     optf_[i].c_str(), optlf_[i].c_str(),
                     optd_[i].c_str());
   }
}

CPDFieldAnim::CPDFieldAnim(StixParams &stixParams,
                           std::shared_ptr<L2_ParFESpace> l2_sfes,
                           std::shared_ptr<L2_ParFESpace> l2_vfes,
                           const std::string hcurl_field_name,
                           const std::string hdiv_field_name,
                           unsigned int vis_flag)
   : CPDVisBase(vis_flag),
     sfes_(l2_sfes),
     vfes_(l2_vfes),
     HCurlFieldAnim_(hcurl_field_name, l2_vfes),
     HDivFieldAnim_(hdiv_field_name, l2_vfes)
{
}

void CPDFieldAnim::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   // Animation not yet supported for VisIt output
}

void CPDFieldAnim::PrepareVisFields(const ParComplexGridFunction & e,
                                    const ParComplexGridFunction & b,
                                    VectorCoefficient * kReCoef,
                                    VectorCoefficient * kImCoef)
{
   if (CheckVisFlag(ELECTRIC_FIELD_ANIM))
   { HCurlFieldAnim_.PrepareVisField(e, kReCoef, kImCoef); }
   if (CheckVisFlag(MAGNETIC_FLUX_ANIM))
   { HDivFieldAnim_.PrepareVisField(b, kReCoef, kImCoef); }
}

void CPDFieldAnim::DisplayToGLVis()
{
   if (CheckVisFlag(ELECTRIC_FIELD_ANIM)) { HCurlFieldAnim_.DisplayToGLVis(); }
   if (CheckVisFlag(MAGNETIC_FLUX_ANIM)) { HDivFieldAnim_.DisplayToGLVis(); }
}

void CPDFieldAnim::Update()
{
   sfes_->Update();
   vfes_->Update();

   HCurlFieldAnim_.Update();
   HDivFieldAnim_.Update();
}

CPDSolverEB::CPDSolverEB(ParMesh & pmesh, int order,
                         CPDSolverEB::SolverType sol, SolverOptions & sOpts,
                         CPDSolverEB::PrecondType prec,
                         ComplexOperator::Convention conv,
                         StixParams &stixParams,
                         VectorCoefficient * betaReCoef,
                         VectorCoefficient * betaImCoef,
                         StixBCs & stixBCs,
                         bool cyl)
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(1),
     sol_(sol),
     solOpts_(sOpts),
     prec_(prec),
     conv_(conv),
     cyl_(cyl),
     omega_(stixParams.omega),
     pmesh_(&pmesh),
     L2FESpace_(make_shared<L2_ParFESpace>(pmesh_, order-1,
                                           pmesh_->Dimension())),
     L2FESpace2p_(make_shared<L2_ParFESpace>(pmesh_,2*order-1,
                                             pmesh_->Dimension())),
     L2VFESpace_(make_shared<L2_ParFESpace>(pmesh_, order_,
                                            pmesh_->Dimension(), 3)),
     HCurlFESpace_(MakeHCurlParFESpace(pmesh, order)),
     HDivFESpace_(MakeHDivParFESpace(pmesh, order)),
     b1_(NULL),
     e_(HCurlFESpace_.get()),
     db_(L2FESpace_.get()),
     dd_(L2FESpace_.get()),
     inputVis_(stixParams, L2FESpace_, L2VFESpace_),
     fieldVis_(stixParams, L2FESpace_, L2VFESpace_, "E", "B"),
     outputVis_(stixParams, L2FESpace2p_, L2VFESpace_, omega_, cyl_),
     fieldAnim_(stixParams, L2FESpace_, L2VFESpace_, "E", "B"),
     epsReCoef_(stixParams, StixCoef::REAL_PART),
     epsImCoef_(stixParams, StixCoef::IMAG_PART),
     epsAbsCoef_(stixParams),
     muInvCoef_(1.0 / mu0_),
     betaReCoef_(betaReCoef),
     betaImCoef_(betaImCoef),
     omegaCoef_(new ConstantCoefficient(omega_)),
     negOmegaCoef_(new ConstantCoefficient(-omega_)),
     omega2Coef_(new ConstantCoefficient(pow(omega_, 2))),
     abcReCoef_(NULL),
     abcImCoef_(NULL),
     posMassCoef_(NULL),
     dbcs_(stixBCs.GetDirichletBCs()),
     abcs_(stixBCs.GetSommerfeldBCs()),
     axis_(stixBCs.GetCylindricalAxis()),
     maxwell_(*HCurlFESpace_, omega_, conv,
              epsReCoef_, epsImCoef_, muInvCoef_,
              betaReCoef, betaImCoef),
     current_(*HCurlFESpace_, *HDivFESpace_, omega_, conv,
              stixBCs.GetCurrentSrcs(), stixBCs.GetNeumannBCs(),
              betaReCoef, betaImCoef),
     faraday_(e_, *HDivFESpace_, omega_, betaReCoef, betaImCoef),
     displacement_(e_, *HDivFESpace_, epsReCoef_, epsImCoef_),
     gauss_(*HDivFESpace_, *L2FESpace_, betaReCoef, betaImCoef),
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
                                                     epsAbsCoef_);

   // Sommerfeld (absorbing) boundary conditions
   if ( abcs_.Size() > 0 )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Absorbing BC Coefficients" << endl;
      }

      abc_bdr_marker_.SetSize(pmesh.bdr_attributes.Max());
      if ( abcs_.Size() == 1 && abcs_[0]->attr[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_bdr_marker_ = 1;

         for (int i=1; i<=pmesh.bdr_attributes.Max(); i++)
         {
            if (abcs_[0]->real != NULL)
            {
               etaInvReCoef_.UpdateCoefficient(i, *(abcs_[0]->real));
            }
            if (abcs_[0]->imag != NULL)
            {
               etaInvImCoef_.UpdateCoefficient(i, *(abcs_[0]->imag));
            }
         }
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_bdr_marker_ = 0;
         for (int i=0; i<abcs_.Size(); i++)
         {
            for (int j=0; j<abcs_[i]->attr.Size(); j++)
            {
               abc_bdr_marker_[abcs_[i]->attr[j]-1] = 1;
               if (abcs_[i]->real != NULL)
               {
                  etaInvReCoef_.UpdateCoefficient(abcs_[i]->attr[j], *(abcs_[i]->real));
               }
               if (abcs_[i]->imag != NULL)
               {
                  etaInvImCoef_.UpdateCoefficient(abcs_[i]->attr[j], *(abcs_[i]->imag));
               }
            }
         }
      }

      abcReCoef_ = new TransformedCoefficient(omegaCoef_, &etaInvImCoef_,
                                              prodFunc);
      abcImCoef_ = new TransformedCoefficient(negOmegaCoef_, &etaInvReCoef_,
                                              prodFunc);
   }

   // Bilinear Forms
   if ( abcs_.Size() > 0 )
   {
      maxwell_.AddBoundaryIntegrator(new VectorFEMassIntegrator(*abcReCoef_),
                                     new VectorFEMassIntegrator(*abcImCoef_),
                                     abc_bdr_marker_);
   }

   b1_ = new ParBilinearForm(HCurlFESpace_.get());
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));

   // Build grid functions
   e_ = 0.0;

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

CPDSolverEB::~CPDSolverEB()
{
   delete posMassCoef_;
   delete abcReCoef_;
   delete abcImCoef_;
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;

   delete b1_;
}

HYPRE_Int
CPDSolverEB::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolverEB::PrintSizes()
{
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
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
   b1_->Finalize();

   maxwell_.Assemble();
   current_.Assemble();

   if ( RequireMagneticFlux() )
   {
      faraday_.Assemble();
   }
   if ( RequireElectricFlux() )
   {
      displacement_.Assemble();
   }
   if ( RequireDivergence() )
   {
      gauss_.Assemble();
   }

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
   if (L2FESpace_) { L2FESpace_->Update(); }
   if (L2FESpace2p_) { L2FESpace2p_->Update(); }
   HCurlFESpace_->Update();
   HDivFESpace_->Update();

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
   db_.Update();
   dd_.Update();

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

   inputVis_.Update();
   fieldVis_.Update();
   outputVis_.Update();
   fieldAnim_.Update();
   // db_v_.Update();
   // dd_v_.Update();

   // Inform the bilinear forms that the space has changed.
   b1_->Update();

   // Inform the other objects that the space has changed.
   maxwell_.Update();
   current_.Update();
   faraday_.Update();
   displacement_.Update();
   gauss_.Update();

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
            ComplexPhaseVectorCoefficient re_e_dbc(betaReCoef_, betaImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, true, true);
            ComplexPhaseVectorCoefficient im_e_dbc(betaReCoef_, betaImCoef_,
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

   if (sol_ == GMRES || sol_ == FGMRES || sol_ == MINRES)
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
                               HCurlFESpace_.get());
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
         MUMPSSolver dmumps(MPI_COMM_WORLD);
         dmumps.SetPrintLevel(1);
         dmumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         dmumps.SetOperator(*A1C);
         dmumps.Mult(RHS, E);
         delete A1C;
      }
      break;
#endif
      default:
         MFEM_ABORT("Requested solver is not available.");
         break;
   };

   tic_toc.Stop();

   e_.Distribute(E);

   if ( RequireMagneticFlux() )
   {
      faraday_.ComputeB();
      if ( RequireDivergence() )
      {
         gauss_.ComputeDiv(faraday_.GetMagneticFlux(), db_);
      }
   }
   if ( RequireElectricFlux() )
   {
      displacement_.ComputeD();
      if ( RequireDivergence() )
      {
         gauss_.ComputeDiv(displacement_.GetDisplacement(), dd_);
      }
   }

   delete BDP;
   if (pci != pcr) { delete pci; }
   delete pcr;

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

real_t
CPDSolverEB::GetEFieldError(const VectorCoefficient & EReCoef,
                            const VectorCoefficient & EImCoef) const
{
   ParFiniteElementSpace * fes =
      const_cast<ParFiniteElementSpace*>(e_.ParFESpace());
   ParComplexGridFunction z(fes);
   z = 0.0;

   real_t solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));


   real_t solErr = e_.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

real_t CPDSolverEB::GetBFieldError(const VectorCoefficient & BReCoef,
                                   const VectorCoefficient & BImCoef) const
{
   if (!RequireMagneticFlux())
   {
      const_cast<FaradaysLaw&>(faraday_).ComputeB();
   }
   return faraday_.GetBFieldError(BReCoef, BImCoef);
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
   CurlCurlIntegrator flux_integrator(muInvCoef_);
   ParFiniteElementSpace flux_fes(pmesh_, flux_fec);

   // Space for the smoothed (conforming) flux
   real_t norm_p = 1;
   ParFiniteElementSpace smooth_flux_fes(pmesh_, smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, e_.real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);

   delete flux_fec;
   delete smooth_flux_fec;

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void CPDSolverEB::prepareVisFields()
{
   inputVis_.PrepareVisFields(current_.GetVolumeCurrentDensity(),
                              betaReCoef_, betaImCoef_);
   fieldVis_.PrepareVisFields(e_, faraday_.GetMagneticFlux(),
                              db_, dd_,
                              betaReCoef_, betaImCoef_);
   outputVis_.PrepareVisFields(e_);
   fieldAnim_.PrepareVisFields(e_, faraday_.GetMagneticFlux(),
                               betaReCoef_, betaImCoef_);
}

void
CPDSolverEB::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   inputVis_.RegisterVisItFields(visit_dc);
   fieldVis_.RegisterVisItFields(visit_dc);
   outputVis_.RegisterVisItFields(visit_dc);
   fieldAnim_.RegisterVisItFields(visit_dc);
}

void
CPDSolverEB::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      prepareVisFields();

      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
CPDSolverEB::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   prepareVisFields();

   inputVis_.DisplayToGLVis();
   fieldVis_.DisplayToGLVis();
   outputVis_.DisplayToGLVis();
   fieldAnim_.DisplayToGLVis();

   if (myid_ == 0) { cout << " done." << endl; }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
