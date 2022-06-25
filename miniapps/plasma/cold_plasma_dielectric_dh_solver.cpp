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

#include "cold_plasma_dielectric_dh_solver.hpp"
#include "cold_plasma_dielectric_coefs.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{
/*
double phi_r_func(const Vector &x)
{
return x[0] + 2.0 * x[1];
}
double phi_i_func(const Vector &x)
{
return 3.0 * x[0] - 2.0 * x[1];
}
*/
// Used for combining scalar coefficients
//double prodFunc(double a, double b) { return a * b; }
/*
ElectricEnergyDensityCoef::ElectricEnergyDensityCoef(VectorCoefficient &Er,
                                                     VectorCoefficient &Ei,
                                                     MatrixCoefficient &epsr,
                                                     MatrixCoefficient &epsi)
   :  ErCoef_(Er),
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

   if (T.ElementNo == 1)
   {
      cout << "eps_r" << endl;
      eps_r_.Print(std::cout, 3);
      cout << "eps_i" << endl;
      eps_i_.Print(std::cout, 3);
   }

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
*/
void nxGradIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Trans,
                                              DenseMatrix &elmat)
{
   int   test_nd = test_fe.GetDof();
   int test_vdim = test_fe.GetVDim() + 1;
   int  trial_nd = trial_fe.GetDof();
   int      sdim = Trans.GetSpaceDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector nor, nxj;
   DenseMatrix test_shape;
   DenseMatrix trial_dshape;
#endif

   nor.SetSize(3); nor = 0.0;
   nxj.SetSize(3);
   test_shape.SetSize(test_nd, test_vdim);
   trial_dshape.SetSize(trial_nd, sdim);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetIntPoint(&ip);

      nor.SetSize(sdim);
      CalcOrtho(Trans.Jacobian(), nor);
      nor.SetSize(3);

      test_fe.CalcPhysVShape(Trans, test_shape);
      trial_fe.CalcPhysDShape(Trans, trial_dshape);

      w = ip.weight;
      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }

      for (int j=0; j<trial_nd; j++)
      {
         // Compute cross product
         nxj[0] = 0.0;
         nxj[1] = nor[2] * trial_dshape(j,0);
         nxj[2] = -nor[1] * trial_dshape(j,0);
         if (sdim > 1)
         {
            nxj[0] += -nor[2] * trial_dshape(j,1);
            nxj[1] += 0.0;
            nxj[2] += nor[0] * trial_dshape(j,1);
         }
         if (sdim > 2)
         {
            nxj[0] += nor[1] * trial_dshape(j,2);
            nxj[1] += -nor[0] * trial_dshape(j,2);
            nxj[2] += 0.0;
         }


         for (int i=0; i<test_nd; i++)
         {
            // Compute inner product
            double v = 0.0;
            for (int d=0; d < test_vdim; d++)
            {
               v += nxj[d] * test_shape(i,d);
            }
            elmat(i,j) += w * v;
         }
      }
   }
}

void nxkIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                           const FiniteElement &test_fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &elmat)
{
   int   test_nd = test_fe.GetDof();
   int test_vdim = test_fe.GetVDim() + 1;
   int  trial_nd = trial_fe.GetDof();
   int      kdim = K->GetVDim();
   int      sdim = Trans.GetSpaceDim();
   double w;

   MFEM_ASSERT(kdim == 3, "");

#ifdef MFEM_THREAD_SAFE
   Vector nor, nxj, k;
   DenseMatrix test_shape;
   Vector trial_shape;
#endif

   // Allocate three entries for the normal vector and set them to zero
   nor.SetSize(3); nor = 0.0;
   nxj.SetSize(3); nxj = 0.0;
   k.SetSize(3);   k = 0.0;
   test_shape.SetSize(test_nd, test_vdim);
   trial_shape.SetSize(trial_nd);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetIntPoint(&ip);

      // Set the first sdim entries in nor
      nor.SetSize(sdim);
      CalcOrtho(Trans.Jacobian(), nor);

      // Expand nor, if necessary, so that we can access all three entries
      nor.SetSize(3);

      K->Eval(k, Trans, ip);

      // Compute cross product
      nxj[0] = nor[1] * k[2] - nor[2] * k[1];
      nxj[1] = nor[2] * k[0] - nor[0] * k[2];
      nxj[2] = nor[0] * k[1] - nor[1] * k[0];

      w = ip.weight;
      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }

      test_fe.CalcPhysVShape(Trans, test_shape);
      trial_fe.CalcPhysShape(Trans, trial_shape);

      for (int j=0; j<trial_nd; j++)
      {
         for (int i=0; i<test_nd; i++)
         {
            // Compute inner product
            double v = 0.0;
            for (int d = 0; d < test_vdim; d++)
            {
               v += nxj[d] * test_shape(i,d);
            }

            elmat(i,j) += w * v * trial_shape[j];
         }
      }

   }
}

void nDotCurlIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                                const FiniteElement &test_fe,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
   int       sdim = Trans.GetSpaceDim();
   int    test_nd = test_fe.GetDof();
   int   trial_nd = trial_fe.GetDof();
   int trial_cdim = std::max(sdim, trial_fe.GetCurlDim());
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector nor, nDotCurl;
   Vector test_shape;
   DenseMatrix trial_dshape;
#endif

   nor.SetSize(3); nor = 0.0;
   nDotCurl.SetSize(trial_nd);
   test_shape.SetSize(test_nd);
   trial_dshape.SetSize(trial_nd, trial_cdim);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetIntPoint(&ip);

      nor.SetSize(sdim);
      CalcOrtho(Trans.Jacobian(), nor);
      nor.SetSize(trial_cdim);

      test_fe.CalcPhysShape(Trans, test_shape);
      trial_fe.CalcPhysCurlShape(Trans, trial_dshape);

      w = ip.weight;
      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      trial_dshape.Mult(nor, nDotCurl);

      for (int j=0; j<trial_nd; j++)
      {
         for (int i=0; i<test_nd; i++)
         {
            elmat(i,j) += w * test_shape[i] * nDotCurl[j];
         }
      }
   }
}

void zkxIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                           const FiniteElement &test_fe,
                                           ElementTransformation &Trans,
                                           DenseMatrix &elmat)
{
   int    test_nd = test_fe.GetDof();
   int   trial_nd = trial_fe.GetDof();
   int trial_vdim = trial_fe.GetVDim() + 1;
   int       sdim = Trans.GetSpaceDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector nor, nxj, k;
   Vector test_shape;
   DenseMatrix trial_shape;
#endif

   nor.SetSize(3); nor = 0.0;
   nxj.SetSize(trial_vdim);
   k.SetSize(3);
   test_shape.SetSize(test_nd);
   trial_shape.SetSize(trial_nd, trial_vdim);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int ir_order = this->GetIntegrationOrder(trial_fe, test_fe, Trans);
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetIntPoint(&ip);

      nor.SetSize(sdim);
      CalcOrtho(Trans.Jacobian(), nor);
      nor.SetSize(3);

      K->Eval(k, Trans, ip);

      // Compute cross product
      nxj[0] = nor[1] * k[2] - nor[2] * k[1];
      nxj[1] = nor[2] * k[0] - nor[0] * k[2];
      if (trial_vdim > 2)
      {
         nxj[2] = nor[0] * k[1] - nor[1] * k[0];
      }

      w = a * ip.weight;
      if (Z)
      {
         w *= Z->Eval(Trans, ip);
      }

      test_fe.CalcPhysShape(Trans, test_shape);
      trial_fe.CalcPhysVShape(Trans, trial_shape);

      for (int j=0; j<trial_nd; j++)
      {
         for (int i=0; i<test_nd; i++)
         {
            // Compute inner product
            elmat(i,j) += w * (nxj[0] * trial_shape(j,0) +
                               nxj[1] * trial_shape(j,1) +
                               ((trial_vdim > 2) ?
                                nxj[2] * trial_shape(j,2) : 0.0)
                              ) * test_shape[i];
         }
      }

   }
}

Maxwell2ndH::Maxwell2ndH(ParFiniteElementSpace & HCurlFESpace,
                         double omega,
                         ComplexOperator::Convention conv,
                         MatrixCoefficient & epsInvReCoef,
                         MatrixCoefficient & epsInvImCoef,
                         Coefficient & muCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef,
                         bool cyl,
                         bool pa)
   : ParSesquilinearForm(&HCurlFESpace, conv),
     cyl_(cyl),
     pa_(pa),
     epsInvReCylCoef_(epsInvReCoef, 1),
     epsInvImCylCoef_(epsInvImCoef, 1),
     muCylCoef_(muCoef, 1),
     massCoef_(omega, muCoef),
     massCylCoef_(omega, muCylCoef_),
     kmkReCoef_(kReCoef, kImCoef, &epsInvReCoef, &epsInvImCoef, true, -1.0),
     kmkImCoef_(kReCoef, kImCoef, &epsInvReCoef, &epsInvImCoef, false, -1.0),
     kmReCoef_(kReCoef, kImCoef, &epsInvReCoef, &epsInvImCoef, true, 1.0),
     kmImCoef_(kReCoef, kImCoef, &epsInvReCoef, &epsInvImCoef, false, 1.0),
     mkReCoef_(&epsInvReCoef, &epsInvImCoef, kReCoef, kImCoef, true, 1.0),
     mkImCoef_(&epsInvReCoef, &epsInvImCoef, kReCoef, kImCoef, false, 1.0),
     kmkCylReCoef_(kReCoef, kImCoef, &epsInvReCylCoef_, &epsInvImCylCoef_,
                   true, -1.0),
     kmkCylImCoef_(kReCoef, kImCoef, &epsInvReCylCoef_, &epsInvImCylCoef_,
                   false, -1.0),
     kmCylReCoef_(kReCoef, kImCoef, &epsInvReCylCoef_, &epsInvImCylCoef_,
                  true, 1.0),
     kmCylImCoef_(kReCoef, kImCoef, &epsInvReCylCoef_, &epsInvImCylCoef_,
                  false, 1.0),
     mkCylReCoef_(&epsInvReCylCoef_, &epsInvImCylCoef_, kReCoef, kImCoef,
                  true, 1.0),
     mkCylImCoef_(&epsInvReCylCoef_, &epsInvImCylCoef_, kReCoef, kImCoef,
                  false, 1.0)
{
   if (pa_) { this->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   if (!cyl_)
   {
      this->AddDomainIntegrator(new CurlCurlIntegrator(epsInvReCoef),
                                new CurlCurlIntegrator(epsInvImCoef));
      this->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef_), NULL);
   }
   else
   {
      this->AddDomainIntegrator(new CurlCurlIntegrator(epsInvReCylCoef_),
                                new CurlCurlIntegrator(epsInvImCylCoef_));
      this->AddDomainIntegrator(new VectorFEMassIntegrator(massCylCoef_), NULL);
   }

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

void Maxwell2ndH::Assemble()
{
   this->ParSesquilinearForm::Assemble();
   if (!pa_) { this->Finalize(); }
}

CurrentSourceH::CurrentSourceH(ParFiniteElementSpace & HCurlFESpace,
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

CurrentSourceH::~CurrentSourceH()
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

void CurrentSourceH::Update()
{
   this->ParComplexLinearForm::Update();

   jt_.Update();
   kt_.Update();
}

void CurrentSourceH::Assemble()
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

AmperesLaw::AmperesLaw(const ParComplexGridFunction &h,
                       const ParComplexGridFunction &j,
                       double omega,
                       VectorCoefficient * kReCoef,
                       VectorCoefficient * kImCoef)
   : h_(h),
     j_(j),
     d_(const_cast<ParFiniteElementSpace*>(j.ParFESpace())),
     omega_(omega),
     curl_(const_cast<ParFiniteElementSpace*>(h.ParFESpace()),
           const_cast<ParFiniteElementSpace*>(j.ParFESpace())),
     kReCross_(NULL),
     kImCross_(NULL)
{
   ParFiniteElementSpace & HCurlFESpace =
      const_cast<ParFiniteElementSpace&>(*h_.ParFESpace());
   ParFiniteElementSpace & HDivFESpace =
      const_cast<ParFiniteElementSpace&>(*j_.ParFESpace());

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

void AmperesLaw::Update()
{
   d_.Update();

   curl_.Update();

   if (kReCross_) { kReCross_->Update(); }
   if (kImCross_) { kImCross_->Update(); }
}

void AmperesLaw::Assemble()
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

void AmperesLaw::ComputeD()
{
   // D = (Curl(H) - J) / (-i omega) = i (Curl(H) - J) / omega
   curl_.Mult(h_.imag(), d_.real()); d_.real() *= -1.0 / omega_;
   curl_.Mult(h_.real(), d_.imag()); d_.imag() *=  1.0 / omega_;

   d_.real().Add( 1.0 / omega_, j_.imag());
   d_.imag().Add(-1.0 / omega_, j_.real());

   if (kReCross_)
   {
      // D += i k x H / (-i omega) = -k x H / omega
      kReCross_->AddMult(h_.real(), d_.real(), -1.0 / omega_);
      kReCross_->AddMult(h_.imag(), d_.imag(), -1.0 / omega_);
   }

   if (kImCross_)
   {
      kImCross_->AddMult(h_.imag(), d_.real(),  1.0 / omega_);
      kImCross_->AddMult(h_.real(), d_.imag(), -1.0 / omega_);
   }
}

CPDSolverDH::CPDSolverDH(ParMesh & pmesh, int order, double omega,
                         CPDSolverDH::SolverType sol, SolverOptions & sOpts,
                         CPDSolverDH::PrecondType prec,
                         ComplexOperator::Convention conv,
                         VectorCoefficient & BCoef,
                         MatrixCoefficient & epsInvReCoef,
                         MatrixCoefficient & epsInvImCoef,
                         MatrixCoefficient & epsAbsCoef,
                         Coefficient & muCoef,
                         Coefficient * etaCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef,
                         Array<int> & abcs,
                         StixBCs & stixBCs,
                         bool vis_u, bool cyl, bool pa)
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(2),
     sol_(sol),
     solOpts_(sOpts),
     prec_(prec),
     conv_(conv),
     ownsEta_(etaCoef == NULL),
     cyl_(cyl),
     vis_u_(vis_u),
     pa_(pa),
     omega_(omega),
     pmesh_(&pmesh),
     H1FESpace_(new H1_ParFESpace(pmesh_,order,pmesh_->Dimension())),
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
     // a1_(NULL),
     // b1_(NULL),
     nxD01_(NULL),
     d21EpsInv_(NULL),
     // m2_(NULL),
     // m12EpsRe_(NULL),
     // m12EpsIm_(NULL),
     m1_(NULL),
     m21EpsInv_(NULL),
     negOneCoef_(-1.0),
     m0_(NULL),
     // n20ZRe_(NULL),
     // n20ZIm_(NULL),
     nzD12_(NULL),
     grad_(NULL),
     curl_(NULL),
     kReCross_(NULL),
     kImCross_(NULL),
     h_(HCurlFESpace_),
     h_dbc_(HCurlFESpace_),
     e_(HCurlFESpace_),
     // e_tmp_(NULL),
     // d_(NULL),
     j_(NULL),
     curlj_(NULL),
     phi_(NULL),
     prev_phi_(NULL),
     // temp_(NULL),
     // phi_tmp_(NULL),
     rectPot_(NULL),
     rhs1_(NULL),
     rhs0_(NULL),
     h_t_(NULL),
     h_v_("H", L2VSFESpace_, L2FESpace_, cyl_, false),
     d_v_("D", L2VSFESpace_, L2FESpace_, cyl_, true),
     e_v_("E", L2VSFESpace_, L2FESpace_, cyl_, false),
     h_dbc_v_("H_DBC", L2VSFESpace_, L2FESpace_, cyl_, false),
     phi_v_("Phi", L2FESpace_, cyl_, true),
     z_v_("Sheath_Impedance", L2FESpace_, cyl_, true),
     // e_b_v_(NULL),
     // h_v_(NULL),
     // h_tilde_(NULL),
     // e_v_(NULL),
     // d_v_(NULL),
     // phi_v_(NULL),
     // rectPot_v_(NULL),
     // j_v_(NULL),
     // div_d_(NULL),
     // b_hat_(NULL),
     // b_hat_v_(NULL),
     // u_(NULL),
     // uE_(NULL),
     // uB_(NULL),
     // S_(NULL),
     StixS_(NULL),
     StixD_(NULL),
     StixP_(NULL),
     EpsPara_(NULL),
     BCoef_(&BCoef),
     // epsReCoef_(&epsReCoef),
     // epsImCoef_(&epsImCoef),
     epsInvReCoef_(&epsInvReCoef),
     epsInvImCoef_(&epsInvImCoef),
     // epsAbsCoef_(&epsAbsCoef),
     muCoef_(&muCoef),
     muInvCoef_(muCoef, -1),
     etaCoef_(etaCoef),
     kReCoef_(kReCoef),
     kImCoef_(kImCoef),
     cylSymm_(cyl),
     cylMassCoef_(muCoef),
     cylStiffnessReCoef_(epsInvReCoef),
     cylStiffnessImCoef_(epsInvImCoef),
     cylSourceReCoef_(epsInvReCoef),
     cylSourceImCoef_(epsInvImCoef),
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
     // negMuInvCoef_(NULL),
     massCoef_(NULL),
     posMassCoef_(NULL),
     kekReCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, true, -1.0),
     kekImCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, false, -1.0),
     keReCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, true),
     keImCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, false, -1.0),
     ekReCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, true),
     ekImCoef_(kReCoef_, kImCoef_, epsInvReCoef_, epsInvImCoef_, false, -1.0),
     // negMuInvkxkxCoef_(NULL),
     // negMuInvkCoef_(NULL),
     jrCoef_(NULL),
     jiCoef_(NULL),
     rhsrCoef_(NULL),
     rhsiCoef_(NULL),
     // erCoef_(EReCoef),
     // eiCoef_(EImCoef),
     derCoef_(NULL),
     deiCoef_(NULL),
     // uCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_,
     //       *epsReCoef_, *epsImCoef_, *muInvCoef_),
     // uECoef_(erCoef_, eiCoef_, *epsReCoef_, *epsImCoef_),
     // uBCoef_(omega_, derCoef_, deiCoef_, *muInvCoef_),
     // SrCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     // SiCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     // j_r_src_(j_r_src),
     // j_i_src_(j_i_src),
     // e_r_bc_(e_r_bc),
     // e_i_bc_(e_i_bc),
     // dbcs_(&dbcs),
     dbcs_(stixBCs.GetDirichletBCs()),
     nbcs_(stixBCs.GetNeumannBCs()),
     sbcs_(stixBCs.GetSheathBCs()),
     axis_(stixBCs.GetCylindricalAxis()),
     maxwell_(*HCurlFESpace_, omega, conv,
              epsInvReCoef, epsInvImCoef, muCoef,
              kReCoef, kImCoef, cyl, pa),
     current_(*HCurlFESpace_, *HDivFESpace_, omega, conv,
              stixBCs.GetCurrentSrcs(), nbcs_, kReCoef, kImCoef, cyl, pa),
     ampere_(h_, current_.GetVolumeCurrentDensity(),
             omega, kReCoef, kImCoef),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Constructing CPDSolverDH ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   /*
   H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   L2FESpace_    = new L2_ParFESpace(pmesh_,order-1,pmesh_->Dimension());

   switch (pmesh_->Dimension())
   {
      case 1:
         HCurlFESpace_ = new ND_R1D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_R1D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
      case 2:
         HCurlFESpace_ = new ND_R2D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_R2D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
      case 3:
         HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
   }
   */
   if (BCoef_)
   {
      // e_b_ = new ParComplexGridFunction(L2FESpace_);
      // *e_b_ = 0.0;
      // b_hat_ = new ParGridFunction(HDivFESpace_);
      // EpsPara_ = new ParComplexGridFunction(L2FESpace_);
   }
   if (kReCoef_ || kImCoef_)
   {
      // L2VFESpace_ = new L2_ParFESpace(pmesh_,order,pmesh_->Dimension(), 3);
      h_t_ = new ParGridFunction(L2VSFESpace_);
      // h_v_ = new ParComplexGridFunction(L2VFESpace_);
      // e_v_ = new ParComplexGridFunction(L2VFESpace_);
      // d_v_ = new ParComplexGridFunction(L2VFESpace_);
      // j_v_ = new ParComplexGridFunction(L2VFESpace_);
      // if (sbcs_.Size() > 0)
      // {
      //    phi_v_ = new ParComplexGridFunction(L2FESpace_);
      // }

      // negMuInvCoef_ = new ProductCoefficient(-1.0, *muInvCoef_);
      // negMuInvkCoef_ = new ScalarVectorProductCoefficient(*negMuInvCoef_,
      //                                                    *kCoef_);
      // negMuInvkxkxCoef_ = new CrossCrossCoefficient(*muInvCoef_, *kCoef_);
   }
   else
   {
      h_t_ = new ParGridFunction(HCurlFESpace_);
   }

   // HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());
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

   this->collectBdrAttributes(dbcs_, dbc_bdr_marker_);
   this->locateTrueDBCDofs(dbc_bdr_marker_,
                           dbc_nd_tdofs_);
   if ( logging_ > 0 )
   {
      int l_dbc_nd_tdofs = dbc_nd_tdofs_.Size();
      int g_dbc_nd_tdofs = 0;
      MPI_Allreduce(&l_dbc_nd_tdofs, &g_dbc_nd_tdofs, 1,
                    MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      cout << "Number of H(Curl) Dirichlet BC Dofs: " << g_dbc_nd_tdofs
           << endl;
   }

   this->collectBdrAttributes(sbcs_, sbc_bdr_marker_);
   this->locateTrueSBCDofs(sbc_bdr_marker_, non_sbc_h1_tdofs_,
                           sbc_nd_tdofs_);

   if ( sbcs_.Size() > 0 )
   {
      if ( logging_ > 0 )
      {
         int l_non_sbc_h1_tdofs = non_sbc_h1_tdofs_.Size();
         int l_sbc_nd_tdofs = sbc_nd_tdofs_.Size();
         int g_non_sbc_h1_tdofs = 0;
         int g_sbc_nd_tdofs = 0;

         MPI_Allreduce(&l_non_sbc_h1_tdofs, &g_non_sbc_h1_tdofs, 1,
                       MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&l_sbc_nd_tdofs, &g_sbc_nd_tdofs, 1,
                       MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         if (myid_ == 0)
         {
            cout << "Number of H(Curl) Sheath BC Dofs: " << g_sbc_nd_tdofs
                 << endl;
            cout << "Number of H1 Non-Sheath BC Dofs: " << g_non_sbc_h1_tdofs
                 << endl;
            cout << "Sheath BC marker: "; sbc_bdr_marker_.Print(cout);
         }
      }

      nxD01_ = new ParMixedSesquilinearForm(H1FESpace_, HCurlFESpace_, conv_);
      nxD01_->AddBoundaryIntegrator(NULL,
                                    new nxGradIntegrator(*omegaCoef_),
                                    sbc_bdr_marker_);
      if (kReCoef_ || kImCoef_)
      {
         nxD01_->AddBoundaryIntegrator((kReCoef_) ?
                                       new nxkIntegrator(*kReCoef_,
                                                         *negOmegaCoef_) : NULL,
                                       (kImCoef_) ?
                                       new nxkIntegrator(*kImCoef_,
                                                         *negOmegaCoef_) : NULL,
                                       sbc_bdr_marker_);
      }
   }

   // Setup various coefficients
   /*
   // Vector Potential on the outer surface
   if ( a_bc_ == NULL )
   {
      Vector Zero(3);
      Zero = 0.0;
      aBCCoef_ = new VectorConstantCoefficient(Zero);
   }
   else
   {
      aBCCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                               *a_bc_);
   }
   */
   // massCoef_ = new ProductCoefficient(*negOmega2Coef_,
   //                                   *muCoef_);
   posMassCoef_ = new ProductCoefficient(*omega2Coef_,
                                         *muCoef_);
   /*
   if (cylSymm_)
   {
      cylMassCoef_.SetCoefficient(*massCoef_);
   }
   */
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

      if ( etaCoef_ == NULL )
      {
         etaCoef_ = new ConstantCoefficient(sqrt(mu0_/epsilon0_));
      }
      abcCoef_ = new TransformedCoefficient(negOmegaCoef_, etaCoef_,
                                            prodFunc);
   }
   /*
   // Volume Current Density
   if ( j_r_src_ != NULL )
   {
      jrCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_r_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jrCoef_ = new VectorConstantCoefficient(j);
   }
   if ( j_i_src_ != NULL )
   {
      jiCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_i_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jiCoef_ = new VectorConstantCoefficient(j);
   }
   //rhsrCoef_ = new ScalarVectorProductCoefficient(omega_, *jiCoef_);
   //rhsiCoef_ = new ScalarVectorProductCoefficient(-omega_, *jrCoef_);
   */
   if (nbcs_.Size() > 0)
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Processing Neumann BCs" << endl;
      }

      // nkbcs_ = new Array<ComplexVectorCoefficientByAttr*>(nbcs_->Size());
      nkbcs_.SetSize(nbcs_.Size());
      for (int i=0; i<nbcs_.Size(); i++)
      {
         nkbcs_[i] = new ComplexVectorCoefficientByAttr;
         nkbcs_[i]->attr = nbcs_[i]->attr;
         nkbcs_[i]->attr_marker.SetSize(pmesh.bdr_attributes.Max());
         nkbcs_[i]->attr_marker = 0;
         for (int j=0; j<nbcs_[i]->attr.Size(); j++)
         {
            nkbcs_[i]->attr_marker[nbcs_[i]->attr[j] - 1] = 1;
         }

         nkbcs_[i]->real =
            new ScalarVectorProductCoefficient(-omega_, *nbcs_[i]->imag);
         nkbcs_[i]->imag =
            new ScalarVectorProductCoefficient(omega_, *nbcs_[i]->real);
      }
   }
   /*
   // Magnetization
   if ( m_src_ != NULL )
   {
      mCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             m_src_);
   }
   */
   // Bilinear Forms
   /*
   a1_ = new ParSesquilinearForm(HCurlFESpace_, conv_);
   if (pa_) { a1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (!cylSymm_)
   {
      a1_->AddDomainIntegrator(new MixedCurlCurlIntegrator(*epsInvReCoef_),
                               new MixedCurlCurlIntegrator(*epsInvImCoef_));
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massCoef_), NULL);
   }
   else
   {
      a1_->AddDomainIntegrator(new MixedCurlCurlIntegrator(cylStiffnessReCoef_),
                               new MixedCurlCurlIntegrator(cylStiffnessImCoef_));
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(cylMassCoef_), NULL);
   }
   if (kReCoef_ || kImCoef_)
   {
      if (pa_)
      {
         MFEM_ABORT("kCoef_: Partial Assembly has not yet been implemented for "
                    "MixedCrossCurlIntegrator and MixedWeakCurlCrossIntegrator.");
      }
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(kekReCoef_),
                               new VectorFEMassIntegrator(kekImCoef_));
      a1_->AddDomainIntegrator(new MixedVectorCurlIntegrator(keImCoef_),
                               new MixedVectorCurlIntegrator(keReCoef_));
      a1_->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(ekImCoef_),
                               new MixedVectorWeakCurlIntegrator(ekReCoef_));

      kReCross_ = new ParDiscreteLinearOperator(HCurlFESpace_, HDivFESpace_);
      kReCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kReCoef_));

      kImCross_ = new ParDiscreteLinearOperator(HCurlFESpace_, HDivFESpace_);
      kImCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kImCoef_));
   }
   if ( abcCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("abcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      a1_->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef_),
                                 abc_bdr_marker_);
   }
   */
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
   /*
   b1_ = new ParBilinearForm(HCurlFESpace_);
   if (pa_) { b1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   // b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsAbsCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));
   //b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massImCoef_));
   */
   /*
   d21EpsInv_ = new ParMixedSesquilinearForm(HDivFESpace_, HCurlFESpace_,
                                             conv_);
   d21EpsInv_->AddDomainIntegrator(
      new MixedVectorWeakCurlIntegrator(*epsInvReCoef_),
      new MixedVectorWeakCurlIntegrator(*epsInvImCoef_));

   if (kReCoef_ || kImCoef_)
   {
      d21EpsInv_->AddDomainIntegrator(new VectorFEMassIntegrator(keImCoef_),
                                      new VectorFEMassIntegrator(keReCoef_));
   }
   */
   /*
   m2_ = new ParBilinearForm(HDivFESpace_);
   if (pa_) { m2_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m2_->AddDomainIntegrator(new VectorFEMassIntegrator);

   m12EpsRe_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
   m12EpsIm_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
   m12EpsRe_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsReCoef_));
   m12EpsIm_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsImCoef_));
   */
   if (myid_ == 0 && false)
   {
      Mesh mesh1 = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, 1.0,
                                         1.0);
      Mesh mesh2 = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, 2.0,
                                         2.0);

      H1_FECollection fec0(1, 2);
      ND_R2D_FECollection fec1(1, 2);

      FiniteElementSpace fes0m1(&mesh1, &fec0);
      FiniteElementSpace fes0m2(&mesh2, &fec0);
      FiniteElementSpace fes1m1(&mesh1, &fec1);
      FiniteElementSpace fes1m2(&mesh2, &fec1);

      BilinearForm m0m1(&fes0m1);
      BilinearForm m0m2(&fes0m2);

      BilinearForm m1m1(&fes1m1);
      BilinearForm m1m2(&fes1m2);

      MixedBilinearForm d01m1(&fes0m1, &fes1m1);
      MixedBilinearForm d01m2(&fes0m2, &fes1m2);

      MixedBilinearForm d10m1(&fes1m1, &fes0m1);
      MixedBilinearForm d10m2(&fes1m2, &fes0m2);

      m0m1.AddBoundaryIntegrator(new MassIntegrator);
      m0m2.AddBoundaryIntegrator(new MassIntegrator);
      m1m1.AddDomainIntegrator(new VectorFEMassIntegrator);
      m1m2.AddDomainIntegrator(new VectorFEMassIntegrator);

      m0m1.Assemble(); m0m1.Finalize();
      m0m2.Assemble(); m0m2.Finalize();
      m1m1.Assemble(); m1m1.Finalize();
      m1m2.Assemble(); m1m2.Finalize();

      ofstream ofs_m1m1("m1m1.mat");
      ofstream ofs_m1m2("m1m2.mat");
      m1m1.SpMat().Print(ofs_m1m1);
      m1m2.SpMat().Print(ofs_m1m2);

      cout << "m0(0,0) " << m0m1(0,0) << " vs " << m0m2(0,0) << endl;
      cout << "m1(0,0) " << m1m1(0,0) << " vs " << m1m2(0,0) << endl;

      d01m1.AddBoundaryIntegrator(new nxGradIntegrator);
      d01m2.AddBoundaryIntegrator(new nxGradIntegrator);

      d01m1.Assemble(); d01m1.Finalize();
      d01m2.Assemble(); d01m2.Finalize();

      ofstream ofs_d01m1("d01m1.mat");
      ofstream ofs_d01m2("d01m2.mat");
      d01m1.SpMat().Print(ofs_d01m1);
      d01m2.SpMat().Print(ofs_d01m2);
      cout << "d01(0,1) " << d01m1.Elem(0,1) << " vs " << d01m2.Elem(0,1) << endl;

      d10m1.AddBoundaryIntegrator(new nDotCurlIntegrator);
      d10m2.AddBoundaryIntegrator(new nDotCurlIntegrator);

      d10m1.Assemble(); d10m1.Finalize();
      d10m2.Assemble(); d10m2.Finalize();

      ofstream ofs_d10m1("d10m1.mat");
      ofstream ofs_d10m2("d10m2.mat");
      d10m1.SpMat().Print(ofs_d10m1);
      d10m2.SpMat().Print(ofs_d10m2);
      cout << "d10(0,1) " << d10m1.Elem(0,1) << " vs " << d10m2.Elem(0,1) << endl;
   }
   if (myid_ == 0 && false)
   {
      class PhiCoefficient : public Coefficient
      {
      private:
         double p00, p01, p10, p11;
         double sx, sy;
      public:
         PhiCoefficient(double p00, double p01, double p10, double p11,
                        double sx, double sy)
            : p00(p00), p01(p01), p10(p10), p11(p11), sx(sx), sy(sy) {}

         double Eval(ElementTransformation &T, const IntegrationPoint &ip)
         {
            double xPtr[2];
            Vector xVec(xPtr, 2);
            T.Transform(ip, xVec);

            double x = xPtr[0] / sx, y = xPtr[1] / sy;
            return p00 * (1.0 - x) * (1.0 - y)
                   + p01 * (1.0 - x) * y
                   + p10 * x * (1.0 - y)
                   + p11 * x * y;
         }
      };

      class GradPhiCoefficient : public VectorCoefficient
      {
      private:
         double p00, p01, p10, p11;
         double sx, sy;
      public:
         GradPhiCoefficient(double p00, double p01, double p10, double p11,
                            double sx, double sy)
            : VectorCoefficient(2),
              p00(p00), p01(p01), p10(p10), p11(p11), sx(sx), sy(sy) {}

         void Eval(Vector &dPhi,
                   ElementTransformation &T,
                   const IntegrationPoint &ip)
         {
            dPhi.SetSize(2);

            double xPtr[2];
            Vector xVec(xPtr, 2);
            T.Transform(ip, xVec);

            double x = xPtr[0] / sx, y = xPtr[1] / sy;
            dPhi[0] = ((p10 - p00) * (1.0 - y) + (p11 - p01) * y) / sx;
            dPhi[1] = ((p01 - p00) * (1.0 - x) + (p11 - p10) * x) / sy;
         }
      };
      /*
      class nxGradPhiCoefficient : public VectorCoefficient
      {
      private:
        int a;
        mutable Vector dPhi;
        GradPhiCoefficient dPhiCoef;

      public:
        nxGradPhiCoefficient(int a) : VectorCoefficient(3), a(a), dPhi(2) {}
        void Eval(Vector &nxdPhi,
       ElementTransformation &T,
       const IntegrationPoint &ip)
        {
      nxdPhi.SetSize(3); nxdPhi = 0.0;
      dPhiCoef.Eval(dPhi, T, ip);

      switch(a)
      {
      case 0:
       nxdPhi[2] = dPhi[0];
       break;
      case 1:
       nxdPhi[2] = dPhi[1];
       break;
      case 2:
       nxdPhi[2] = -dPhi[0];
       break;
      case 3:
       nxdPhi[2] = -dPhi[1];
       break;
      }
        }
      };
      */
      class JSrcCoefficient : public VectorCoefficient
      {
      private:
         double dp1, dp2, dp3, dp4;
         double sx, sy;

      public:
         JSrcCoefficient(double p00, double p01, double p10, double p11,
                         double e1, double e2, double e3, double e4,
                         double sx, double sy)
            : VectorCoefficient(3),
              dp1(e1*(p10-p00)/sx), dp2(e2*(p11-p10)/sy),
              dp3(e3*(p11-p01)/sx), dp4(e4*(p01-p00)/sy),
              sx(sx), sy(sy)
         {}

         void Eval(Vector &JSrc,
                   ElementTransformation &T,
                   const IntegrationPoint &ip)
         {
            JSrc.SetSize(3); JSrc = 0.0;

            double xPtr[2];
            Vector xVec(xPtr, 2);
            T.Transform(ip, xVec);

            double x = xPtr[0] / sx, y = xPtr[1] / sy;
            JSrc[2] = 0.5 * (-dp1 * sy * y * (2.0 - y)
                             +dp2 * sx * x * x
                             -dp3 * sy * y * y
                             +dp4 * sx * x * (2.0 - x));
         }
      };
      class HExactCoefficient : public VectorCoefficient
      {
      private:
         double dp1, dp2, dp3, dp4;
         double sx, sy;

      public:
         HExactCoefficient(double p00, double p01, double p10, double p11,
                           double e1, double e2, double e3, double e4,
                           double sx, double sy)
            : VectorCoefficient(3),
              dp1(e1*(p10-p00)/sx), dp2(e2*(p11-p10)/sy),
              dp3(e3*(p11-p01)/sx), dp4(e4*(p01-p00)/sy),
              sx(sx), sy(sy)
         {}

         void Eval(Vector &H,
                   ElementTransformation &T,
                   const IntegrationPoint &ip)
         {
            H.SetSize(3); H = 0.0;

            double xPtr[2];
            Vector xVec(xPtr, 2);
            T.Transform(ip, xVec);

            double x = xPtr[0], y = xPtr[1];
            H[2] = 0.5 * (+dp1 * (2.0 + (2.0 * sy - y) * y) / sy
                          +dp2 * (2.0 - x * x) / sx
                          -dp3 * (2.0 - y * y) / sy
                          -dp4 * (2.0 + (2.0 * sx - x) * x) / sx);
         }
      };

      double sx = 1.0, sy = 1.0;
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                        true, sx, sy);

      H1_FECollection fec0(3, 2);
      ND_R2D_FECollection fec1(3, 2);

      FiniteElementSpace fes0(&mesh, &fec0);
      FiniteElementSpace fes1(&mesh, &fec1);

      ConstantCoefficient negOneCoef(-1.0);
      BilinearForm a1(&fes1);
      a1.AddDomainIntegrator(new CurlCurlIntegrator);
      a1.AddDomainIntegrator(new VectorFEMassIntegrator(negOneCoef));
      a1.Assemble(); a1.Finalize();

      double ph00 = 0.0, ph01 = 1.0, ph10 = 2.0, ph11 = 3.0;
      double et1 = 1.0, et2 = 1.0, et3 = 1.0, et4 = 1.0;

      PhiCoefficient phiCoef(ph00, ph01, ph10, ph11, sx, sy);
      JSrcCoefficient JCoef(ph00, ph01, ph10, ph11,
                            et1, et2, et3, et4, sx, sy);
      HExactCoefficient HCoef(ph00, ph01, ph10, ph11,
                              et1, et2, et3, et4, sx, sy);

      Array<int> bdr_marker(4);
      bdr_marker = 0;
      if (et1 > 0.5) { bdr_marker[0] = 1; }
      if (et2 > 0.5) { bdr_marker[1] = 1; }
      if (et3 > 0.5) { bdr_marker[2] = 1; }
      if (et4 > 0.5) { bdr_marker[3] = 1; }

      MixedBilinearForm d01(&fes0, &fes1);
      d01.AddBoundaryIntegrator(new nxGradIntegrator, bdr_marker);
      d01.Assemble(); d01.Finalize();

      ofstream ofs_d01("d01o2.mat");
      d01.SpMat().Print(ofs_d01);

      GridFunction phi(&fes0);
      phi.ProjectCoefficient(phiCoef);

      ofstream ofsphi("phi.vec");
      phi.Print(ofsphi);

      LinearForm dh(&fes1);
      d01.Mult(phi, dh);

      ofstream ofsdh("dh.vec");
      dh.Print(ofsdh);

      LinearForm j(&fes1);
      j.AddDomainIntegrator(new VectorFEDomainLFIntegrator(JCoef));
      j.Assemble();

      ofstream ofsj("j.vec");
      j.Print(ofsj);

      //nxGradPhiCoefficient hCoef(0);
      GridFunction hExact(&fes1); hExact.ProjectCoefficient(HCoef);
      double errhe = hExact.ComputeL2Error(HCoef);
      cout << "Error in hExact: " << errhe << endl;
      ofstream ofshe("h_exact.vec");
      hExact.Print(ofshe);

      GridFunction h(&fes1); h = 0.0;

      j.Add(-1.0, dh);
      CG(a1, j, h, 1);

      double errh = h.ComputeL2Error(HCoef);
      cout << "Error in h: " << errh << endl;
      ofstream ofsh("h.vec");
      h.Print(ofsh);

      ElementTransformation *TPtr = fes1.GetElementTransformation(0);
      IntegrationPoint ip;
      ip.y = 0.0; ip.z = 0.0;

      Vector hExactVal(3);
      Vector hVal(3);

      for (int i=0; i<5; i++)
      {
         ip.x = 0.25 * i;
         TPtr->SetIntPoint(&ip);

         hExact.GetVectorValue(*TPtr, ip, hExactVal);
         h.GetVectorValue(*TPtr, ip, hVal);

         cout << ip.x << " (" << hExactVal[0] << "," << hExactVal[1] << "," <<
              hExactVal[2] << ")" << " vs " << "(" << hVal[0] << "," << hVal[1] << "," <<
              hVal[2] << ")" << endl;
      }
   }
   /*
   MFEM_ASSERT(false, "Exiting");
   */
   if (pa_)
   {
      // TODO: PA
      //m12EpsRe_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      //m12EpsIm_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Preparing Matrices for computing E from D" << endl;
   }

   m1_ = new ParSesquilinearForm(HCurlFESpace_, conv_);
   if (pa_) { m1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m1_->AddDomainIntegrator(new VectorFEMassIntegrator, NULL);

   m21EpsInv_ = new ParMixedSesquilinearForm(HDivFESpace_, HCurlFESpace_, conv_);
   m21EpsInv_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsInvReCoef_),
                                   new VectorFEMassIntegrator(*epsInvImCoef_));

   if (sbcs_.Size() > 0)
   {
      // TODO: PA does not support BoundaryIntegrator yet
      /*
      m0_ = new ParBilinearForm(H1FESpace_);

      n20ZRe_ = new ParMixedBilinearForm(HDivFESpace_, H1FESpace_);
      n20ZIm_ = new ParMixedBilinearForm(HDivFESpace_, H1FESpace_);
      */
      m0_ = new ParSesquilinearForm(H1FESpace_, conv_);
      nzD12_ = new ParMixedSesquilinearForm(HCurlFESpace_, H1FESpace_, conv_);

      for (int i=0; i<sbcs_.Size(); i++)
      {
         ComplexCoefficientByAttr & sbc = *sbcs_[i];
         /*
              m0_->AddBoundaryIntegrator(new MassIntegrator, sbc.attr_marker);

              n20ZRe_->AddBoundaryIntegrator(new MassIntegrator(*sbc.real),
                                             sbc.attr_marker);
              n20ZIm_->AddBoundaryIntegrator(new MassIntegrator(*sbc.imag),
                                             sbc.attr_marker);
         */
         m0_->AddBoundaryIntegrator(new MassIntegrator(negOneCoef_), NULL,
                                    sbc.attr_marker);
         /*
              nzD12_->AddBoundaryIntegrator(new MixedVectorCurlIntegrator(*sbc.real),
                                            new MixedVectorCurlIntegrator(*sbc.imag),
                                            sbc.attr_marker);
         */
         /*
              nzD12_->AddBoundaryIntegrator(new VectorFECurlIntegrator(*sbc.real),
                                            new VectorFECurlIntegrator(*sbc.imag),
                                            sbc.attr_marker);
         */
         nzD12_->AddBoundaryIntegrator(new nDotCurlIntegrator(*sbc.real),
                                       new nDotCurlIntegrator(*sbc.imag),
                                       sbc.attr_marker);

         if (kReCoef_)
         {
            nzD12_->AddBoundaryIntegrator(new zkxIntegrator(*sbc.imag,
                                                            *kReCoef_, -1.0),
                                          new zkxIntegrator(*sbc.real,
                                                            *kReCoef_,  1.0),
                                          sbc.attr_marker);
         }
         if (kImCoef_)
         {
            nzD12_->AddBoundaryIntegrator(new zkxIntegrator(*sbc.real,
                                                            *kImCoef_, -1.0),
                                          new zkxIntegrator(*sbc.imag,
                                                            *kImCoef_, -1.0),
                                          sbc.attr_marker);
         }
      }
   }

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Creating GridFunction Objects" << endl;
   }
   // Build grid functions
   // h_  = new ParComplexGridFunction(HCurlFESpace_);
   h_ = 0.0;

   // e_  = new ParComplexGridFunction(HCurlFESpace_);
   e_ = 0.0;

   // temp_ = new ParGridFunction(HCurlFESpace_);
   // *temp_ = 0.0;

   // d_  = new ParComplexGridFunction(HDivFESpace_);
   // *d_ = 0.0;

   j_  = new ParComplexGridFunction(HDivFESpace_);
   *j_ = 0.0;

   prev_phi_  = new ParComplexGridFunction(H1FESpace_);
   phi_  = new ParComplexGridFunction(H1FESpace_);
   *prev_phi_ = 0.0;
   *phi_ = 0.0;

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Creating LinearForm Objects" << endl;
   }
   curlj_  = new ParComplexLinearForm(HCurlFESpace_, conv_);
   curlj_->real() = 0.0;
   curlj_->imag() = 0.0;

   rhs1_ = new ParComplexLinearForm(HCurlFESpace_, conv_);
   rhs0_ = new ParComplexLinearForm(H1FESpace_, conv_);

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Creating discrete curl operator" << endl;
   }
   curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);

   if (sbcs_.Size() > 0)
   {
      /*
      PlasmaProfile::Type dpt = PlasmaProfile::GRADIENT;
      Vector dpp(6);
      dpp[0] = 2e20; dpp[1] = 0; dpp[2] = 0; dpp[3] = 0; dpp[4] = 0; dpp[5] = 100; dpp[6] = 0;
      PlasmaProfile rhoCoef(dpt, dpp);
      phi_->ProjectCoefficient(rhoCoef, rhoCoef);
       */

      rectPot_ = new ParComplexGridFunction(H1FESpace_);
      *rectPot_ = 0.0;

      for (int i=0; i<sbcs_.Size(); i++)
      {
         ComplexCoefficientByAttr & sbc = *sbcs_[i];
         SheathImpedance * z_r = dynamic_cast<SheathImpedance*>(sbc.real);
         SheathImpedance * z_i = dynamic_cast<SheathImpedance*>(sbc.imag);

         if (z_r) { z_r->SetPotential(*phi_); }
         if (z_i) { z_i->SetPotential(*phi_); }
      }

      grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
   }
   // solNorm_ = e_->ComputeL2Error(const_cast<VectorCoefficient&>(erCoef_),
   //                               const_cast<VectorCoefficient&>(eiCoef_));

   // rhs_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsrCoef_),
   //                        new VectorFEDomainLFIntegrator(*rhsiCoef_));
   if (nkbcs_ != NULL)
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Adding boundary integrators to rhs1" << endl;
      }
      for (int i=0; i<nkbcs_.Size(); i++)
      {
         rhs1_->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator
                                      (*nkbcs_[i]->real),
                                      new VectorFEBoundaryTangentLFIntegrator
                                      (*nkbcs_[i]->imag),
                                      nkbcs_[i]->attr_marker);

      }
   }

   rhs1_->real().Vector::operator=(0.0);
   rhs1_->imag().Vector::operator=(0.0);
   rhs0_->real().Vector::operator=(0.0);
   rhs0_->imag().Vector::operator=(0.0);
   /*
   if (vis_u_)
   {
      if (L2FESpace2p_ == NULL)
      {
         L2FESpace2p_ = new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension());
      }
      u_ = new ParGridFunction(L2FESpace2p_);
      uE_ = new ParGridFunction(L2FESpace2p_);
      uB_ = new ParGridFunction(L2FESpace2p_);

      HDivFESpace2p_ = new RT_ParFESpace(pmesh_,2*order,pmesh_->Dimension());
      S_ = new ParComplexGridFunction(HDivFESpace2p_);

      erCoef_.SetGridFunction(&e_->real());
      eiCoef_.SetGridFunction(&e_->imag());

      derCoef_.SetGridFunction(&e_->real());
      deiCoef_.SetGridFunction(&e_->imag());
   }
   */
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Stix Coefs" << endl;
      }
      StixCoefBase * s = dynamic_cast<StixCoefBase*>(epsInvReCoef_);

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

         // StixS_ = new ComplexGridFunction(L2FESpace3D_);
         // StixD_ = new ComplexGridFunction(L2FESpace3D_);
         // StixP_ = new ComplexGridFunction(L2FESpace3D_);
      }
   }

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

CPDSolverDH::~CPDSolverDH()
{
   // delete negMuInvkxkxCoef_;
   // delete negMuInvkCoef_;
   // delete negMuInvCoef_;
   // delete rhsrCoef_;
   // delete rhsiCoef_;
   delete jrCoef_;
   delete jiCoef_;
   // delete erCoef_;
   // delete eiCoef_;
   delete SReCoef_;
   delete SImCoef_;
   delete DReCoef_;
   delete DImCoef_;
   delete PReCoef_;
   delete PImCoef_;
   delete massCoef_;
   delete posMassCoef_;
   delete abcCoef_;
   if ( ownsEta_ ) { delete etaCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   // delete DivFreeProj_;
   // delete SurfCur_;

   // if (h_v_ != h_) { delete h_v_; }
   // if (e_v_ != e_) { delete e_v_; }
   // if (d_v_ != d_) { delete d_v_; }
   // if (j_v_ != j_) { delete j_v_; }
   // if (phi_v_ != phi_) {delete phi_v_;}
   // delete h_v_;
   // delete h_tilde_;
   // delete e_v_;
   // delete d_v_;
   // delete j_v_;
   // delete div_d_;
   // delete phi_v_;
   // delete e_b_v_;
   // delete b_hat_;
   // delete b_hat_v_;
   // delete e_r_;
   // delete e_i_;
   // delete h_;
   // delete e_;
   // delete d_;
   delete j_;
   delete curlj_;
   // delete temp_;
   delete phi_;
   delete prev_phi_;
   // delete phi_tmp_;
   delete rectPot_;
   // delete rectPot_v_;
   // delete b_;
   // delete h_;
   // delete u_;
   // delete uE_;
   // delete uB_;
   delete StixS_;
   delete StixD_;
   delete StixP_;
   delete EpsPara_;
   // delete j_r_;
   // delete j_i_;
   // delete j_;
   // delete k_;
   // delete m_;
   // delete bd_;
   delete rhs1_;
   delete rhs0_;
   delete h_t_;
   // delete jd_r_;
   // delete jd_i_;
   delete grad_;
   delete curl_;
   delete kReCross_;
   delete kImCross_;

   // delete a1_;
   delete nxD01_;
   // delete b1_;
   delete d21EpsInv_;
   // delete m2_;
   // delete m12EpsRe_;
   // delete m12EpsIm_;
   delete m1_;
   delete m21EpsInv_;
   delete m0_;
   // delete n20ZRe_;
   // delete n20ZIm_;
   delete nzD12_;
   // delete curlMuInvCurl_;
   // delete hCurlMass_;
   // delete hDivHCurlMuInv_;
   // delete weakCurlMuInv_;

   delete H1FESpace_;
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
CPDSolverDH::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolverDH::PrintSizes()
{
   HYPRE_Int size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
CPDSolverDH::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << endl; }

   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  Curl(1/eps Curl) - omega^2 mu ..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   maxwell_.Assemble();
   current_.Assemble();
   ampere_.Assemble();

   // a0_->Assemble();
   // a0_->Finalize();

   // a1_->Assemble();
   // if (!pa_) { a1_->Finalize(); }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   // if ( myid_ == 0 && logging_ > 0 )
   // { cout << "  Curl(1/mu Curl) + omega^2 |epsilon| ..." << flush; }
   // tic_toc.Clear();
   // tic_toc.Start();

   // b1_->Assemble();
   // if (!pa_) { b1_->Finalize(); }

   // tic_toc.Stop();
   // if ( myid_ == 0 && logging_ > 0 )
   //  {
   //   cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   // }
   if (nxD01_)
   {
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  n x Grad ..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      nxD01_->Assemble();

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
   }

   if (d21EpsInv_)
   {
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  Curl(1/eps) ..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      d21EpsInv_->Assemble();
      if (!pa_) { d21EpsInv_->Finalize(); }

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
   }

   if (m1_)
   {
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  H(Curl) mass matrix ..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      // m2_->Assemble();
      // if (!pa_) { m2_->Finalize(); }
      m1_->Assemble();
      if (!pa_) { m1_->Finalize(); }

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
   }
   /*
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  (epsilon u, v), u in H(Curl), v in H(Div)..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   // TODO: PA
   m12EpsRe_->Assemble();
   m12EpsRe_->Finalize();
   //if (!pa_) m12EpsRe_->Finalize();

   // TODO: PA
   m12EpsIm_->Assemble();
   m12EpsIm_->Finalize();
   //if (!pa_) m12EpsIm_->Finalize();
   */
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  (epsilon^{-1} u, v), u in H(Div), v in H(Curl)..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   // TODO: PA
   m21EpsInv_->Assemble();
   m21EpsInv_->Finalize();
   //if (!pa_) m21EpsInv_->Finalize();

   if (m0_)
   {
      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  H1 mass matrix ..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      // TODO: PA
      m0_->Assemble();
      m0_->Finalize();
      //if (!pa_) m0_->Finalize();

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
      // if ( myid_ == 0 && logging_ > 0 )
      // { cout << "  (z n.u, v), u in H(Div), v in H1..." << flush; }
      // tic_toc.Clear();
      // tic_toc.Start();

      // n20ZRe_->Assemble();
      // n20ZRe_->Finalize();
      //if (!pa_) n20ZRe_->Finalize();

      // n20ZIm_->Assemble();
      // n20ZIm_->Finalize();
      //if (!pa_) n20ZIm_->Finalize();

      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  (z n.Curl u, v), u in H(Curl), v in H1..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      nzD12_->Assemble();
      nzD12_->Finalize();

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
   }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  Source term..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   // rhs1_->Assemble();

   if ( grad_ )
   {
      grad_->Assemble();
      grad_->Finalize();
   }
   curl_->Assemble();
   curl_->Finalize();

   if ( kReCross_ )
   {
      kReCross_->Assemble();
      kReCross_->Finalize();
   }
   if ( kImCross_ )
   {
      kImCross_->Assemble();
      kImCross_->Finalize();
   }
   /*
   curlMuInvCurl_->Assemble();
   curlMuInvCurl_->Finalize();
   hDivHCurlMuInv_->Assemble();
   hDivHCurlMuInv_->Finalize();
   hCurlMass_->Assemble();
   hCurlMass_->Finalize();
   if ( weakCurlMuInv_ )
   {
      weakCurlMuInv_->Assemble();
      weakCurlMuInv_->Finalize();
   }
   */
   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolverDH::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   // H1FESpace_->Update(false);
   if (H1FESpace_) { H1FESpace_->Update(); }
   if (L2FESpace_) { L2FESpace_->Update(); }
   if (L2FESpace2p_) { L2FESpace2p_->Update(); }
   if (L2VSFESpace_) { L2VSFESpace_->Update(); }
   if (L2VSFESpace2p_) { L2VSFESpace2p_->Update(); }
   if (L2V3FESpace_) { L2V3FESpace_->Update(); }
   // H1FESpace_->Update();
   HCurlFESpace_->Update();
   HDivFESpace_->Update();
   // if (HDivFESpace2p_) { HDivFESpace2p_->Update(false); }
   /*
   if (dbcs_->Size() > 0)
   {
      HCurlFESpace_->GetEssentialTrueDofs(dbc_bdr_marker_, dbc_nd_tdofs_);
   }
   if (sbcs_->Size() > 0)
   {
      HCurlFESpace_->GetEssentialTrueDofs(sbc_bdr_marker_, sbc_nd_tdofs_);
   }
   */
   this->locateTrueDBCDofs(dbc_bdr_marker_,
                           dbc_nd_tdofs_);
   if ( logging_ > 0 )
   {
      cout << myid_
           << ": Number of H(Curl) Dirichlet BC Dofs: " << dbc_nd_tdofs_.Size()
           << endl;
   }

   this->locateTrueSBCDofs(sbc_bdr_marker_, non_sbc_h1_tdofs_,
                           sbc_nd_tdofs_);
   if ( sbcs_.Size() > 0 )
   {
      if ( logging_ > 0 )
      {
         cout << myid_
              << ": Number of H(Curl) Sheath BC Dofs: " << sbc_nd_tdofs_.Size()
              << endl;
      }
   }

   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // Inform the grid functions that the space has changed.
   h_.Update();
   h_dbc_.Update();
   e_.Update();
   // d_->Update();
   j_->Update();
   curlj_->Update();
   phi_->Update();
   prev_phi_->Update();
   // temp_->Update();
   // if (u_) { u_->Update(); }
   // if (uE_) { uE_->Update(); }
   // if (uB_) { uB_->Update(); }
   // if (S_) { S_->Update(); }
   if (h_t_) { h_t_->Update(); }
   // if (e_b_v_) { e_b_v_->Update(); }
   h_v_.Update();
   h_dbc_v_.Update();
   // if (h_tilde_) { h_tilde_->Update(); }
   // if (e_v_) { e_v_->Update(); }
   d_v_.Update();
   e_v_.Update();
   phi_v_.Update();
   z_v_.Update();
   // if (j_v_) { j_v_->Update(); }
   // if (phi_v_) { phi_v_->Update(); }
   if (rectPot_) { rectPot_->Update();}
   // if (rectPot_v_) { rectPot_v_->Update();}
   // if (b_hat_) { b_hat_->Update(); }
   // if (b_hat_v_) { b_hat_v_->Update(); }
   if (StixS_) { StixS_->Update(); }
   if (StixD_) { StixD_->Update(); }
   if (StixP_) { StixP_->Update(); }
   // e_r_->Update();
   // e_i_->Update();
   // h_->Update();
   // b_->Update();
   // bd_->Update();
   rhs1_->Update();
   rhs0_->Update();
   // jd_i_->Update();
   // if ( jr_ ) { jr_->Update(); }
   // if ( j_  ) {  j_->Update(); }
   // if ( j_r_  ) {  j_r_->Update(); }
   // if ( j_i_  ) {  j_i_->Update(); }
   // if ( k_  ) {  k_->Update(); }
   // if ( m_  ) {  m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   // a0_->Update();
   // a1_->Update();
   if (nxD01_) { nxD01_->Update(); }
   // b1_->Update();
   if (d21EpsInv_) { d21EpsInv_->Update(); }
   // m2_->Update();
   // m12EpsRe_->Update();
   // m12EpsIm_->Update();
   if (m1_) { m1_->Update(); }
   if (m21EpsInv_) { m21EpsInv_->Update(); }
   if (m0_)
   {
      m0_->Update();
      // n20ZRe_->Update();
      // n20ZIm_->Update();
      nzD12_->Update();
   }
   // curlMuInvCurl_->Update();
   // hCurlMass_->Update();
   // hDivHCurlMuInv_->Update();
   // if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   curl_->Update();
   if ( grad_ ) { grad_->Update(); }
   if ( kReCross_ ) { kReCross_->Update(); }
   if ( kImCross_ ) { kImCross_->Update(); }
   // if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
   // if ( SurfCur_     ) { SurfCur_->Update(); }
   maxwell_.Update();
   current_.Update();
   ampere_.Update();

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolverDH::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   // Apply Neumann BCs (if present) or zero the RHS
   rhs1_->Assemble();
   /*
   // Set the current density
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

   if (logging_ > 0)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVCoef(zeroVec);

      double nrmj = j_->ComputeL2Error(zeroVCoef, zeroVCoef);
      if (myid_ == 0) { cout << "norm of J: " << nrmj << endl; }
   }

   d21EpsInv_->Mult(*j_, *curlj_);
   *rhs1_ += *curlj_;
   */
   /*
   *phi_ = 0.0;

   if (nxD01_)
   {
      nxD01_->real().AddMult(phi_->imag(), rhs1_->real(), -omega_);
      nxD01_->imag().AddMult(phi_->real(), rhs1_->imag(),  omega_);
   }
   */
   h_ = 0.0;

   OperatorHandle A1;
   Vector H, RHS1;
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
            ComplexPhaseVectorCoefficient re_h_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, true, true);
            ComplexPhaseVectorCoefficient im_h_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, false, true);

            h_.ProjectBdrCoefficientTangent(re_h_dbc, im_h_dbc, attr_marker);
            h_dbc_.ProjectBdrCoefficientTangent(re_h_dbc, im_h_dbc, attr_marker);
            // h_dbc_.ProjectCoefficient(re_h_dbc, im_h_dbc);
         }
         else
         {
            h_.ProjectBdrCoefficientTangent(*dbcs_[i]->real, *dbcs_[i]->imag,
                                            attr_marker);
            h_dbc_.ProjectBdrCoefficientTangent(*dbcs_[i]->real, *dbcs_[i]->imag,
                                                attr_marker);
         }
      }
      if (logging_ > 1)
      {
         Vector zeroVec(3); zeroVec = 0.0;
         VectorConstantCoefficient zeroVCoef(zeroVec);

         double nrmh = h_.ComputeL2Error(zeroVCoef, zeroVCoef);
         if (myid_ == 0) { cout << "norm of H (w DBC): " << nrmh << endl; }
      }
      // h_dbc_ = h_;
   }

   maxwell_.FormLinearSystem(dbc_nd_tdofs_, h_, current_, A1, H, RHS1);

   ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
   HypreParMatrix * A1C = (sol_ != SolverType::ZMUMPS) ?
                          A1Z->GetSystemMatrix() : NULL;

   Solver * AInv = NULL;

#ifdef MFEM_USE_SUPERLU
   if (sol_ == SolverType::SUPERLU)
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "SuperLU Solver Requested" << endl;
      }
      SuperLURowLocMatrix A_SuperLU(*A1C);
      SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(true);
      // SuperLU Options
      // AInv.SetNumLookAheads(10);
      // AInv.SetReplaceTinyPivot(true);
      // AInv.SetEquilibriate(false);
      // AInv.SetIterativeRefine(superlu::SLU_DOUBLE);
      superlu->SetOperator(A_SuperLU);
      AInv = superlu;
   }
#endif
#ifdef MFEM_USE_MUMPS
   if (sol_ == SolverType::DMUMPS)
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "MUMPS (Real) Solver Requested" << endl;
      }
      MUMPSSolver * dmumps = new MUMPSSolver;
      dmumps->SetPrintLevel(1);
      dmumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      dmumps->SetOperator(*A1C);
      AInv = dmumps;
   }
   if (sol_ == SolverType::ZMUMPS)
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "MUMPS (Complex) Solver Requested" << endl;
      }
      ComplexMUMPSSolver * zmumps = new ComplexMUMPSSolver;
      zmumps->SetPrintLevel(1);
      zmumps->SetOperator(*A1Z);
      AInv = zmumps;
   }
#endif

   if (sbcs_.Size() > 0)
   {
      OperatorHandle B, C, D;

      nxD01_->FormRectangularSystemMatrix(non_sbc_h1_tdofs_, dbc_nd_tdofs_, B);
      m0_->FormSystemMatrix(non_sbc_h1_tdofs_, D);

      Vector diag_r;
      D.As<ComplexHypreParMatrix>()->real().GetDiag(diag_r);

      int size_h1 = diag_r.Size();
      Vector diag0(2*size_h1);
      for (int i=0; i<size_h1; i++)
      {
         diag0(i) = diag_r(i);
         diag0(i+size_h1) = diag_r(i);
      }

      OperatorJacobiSmoother prec0(diag0, non_sbc_h1_tdofs_);

      {
         if (B.As<ComplexHypreParMatrix>()->hasRealPart())
         { B.As<ComplexHypreParMatrix>()->real().Print("nxD01_Re.mat"); }
         if (B.As<ComplexHypreParMatrix>()->hasImagPart())
         { B.As<ComplexHypreParMatrix>()->imag().Print("nxD01_Im.mat"); }
         if (D.As<ComplexHypreParMatrix>()->hasRealPart())
         { D.As<ComplexHypreParMatrix>()->real().Print("m0_Re.mat"); }
         if (D.As<ComplexHypreParMatrix>()->hasImagPart())
         { D.As<ComplexHypreParMatrix>()->imag().Print("m0_Im.mat"); }
      }

      rhs0_->real() = 0.0;
      rhs0_->imag() = 0.0;

      Vector RHS0(2 * H1FESpace_->GetTrueVSize()); RHS0 = 0.0;
      Vector PHI(2 * H1FESpace_->GetTrueVSize()); PHI = 0.0;

      int H_iter = 0;
      double phi_diff = std::numeric_limits<double>::max();
      GridFunctionCoefficient prevPhiReCoef(&prev_phi_->real());
      GridFunctionCoefficient prevPhiImCoef(&prev_phi_->imag());
      while (H_iter < 15)
      {
         if ( phi_diff < 1e-3) {break;}
         nzD12_->Update();
         nzD12_->Assemble();
         nzD12_->Finalize();
         nzD12_->FormRectangularSystemMatrix(dbc_nd_tdofs_,
                                             non_sbc_h1_tdofs_, C);
         {
            if (C.As<ComplexHypreParMatrix>()->hasRealPart())
            { C.As<ComplexHypreParMatrix>()->real().Print("nzD12_Re.mat"); }
            if (C.As<ComplexHypreParMatrix>()->hasImagPart())
            { C.As<ComplexHypreParMatrix>()->imag().Print("nzD12_Im.mat"); }
         }

         SchurComplimentOperator schur(*AInv, &(*B), &(*C), *D);
         // SchurComplimentOperator schur(*AInv, NULL, &(*C), *D);

         const Vector & RHS = schur.GetRHSVector(RHS1, RHS0);

         IterativeSolver::PrintLevel gmres_print_level;
         gmres_print_level.FirstAndLast();

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetKDim(50);
         gmres.SetRelTol(1e-8);
         gmres.SetAbsTol(1e-10);
         gmres.SetMaxIter(500);
         gmres.SetPrintLevel(gmres_print_level);
         // gmres.SetPrintLevel(1);
         gmres.SetOperator(schur);
         gmres.SetPreconditioner(prec0);

         gmres.Mult(RHS, PHI);

         m0_->RecoverFEMSolution(PHI, *rhs0_, *phi_);

         schur.Solve(RHS1, PHI, H);

         double dr = phi_->real().ComputeL2Error(prevPhiReCoef);
         double di = phi_->imag().ComputeL2Error(prevPhiImCoef);
         phi_diff = sqrt(dr*dr + di*di);

         if (myid_ == 0) { cout << H_iter << '\t' << phi_diff << endl; }

         prev_phi_->Vector::operator=((Vector&)(*phi_));

         H_iter++;
      }
      if (myid_ == 0)
      {
         cout << " Outer H field calculation done in " << H_iter
              << " iteration(s)." << endl;
      }
   }
   else
   {
      AInv->Mult(RHS1, H);

      *phi_ = 0.0;
   }

   delete AInv;
   delete A1C;

   maxwell_.RecoverFEMSolution(H, current_, h_);

   if (logging_ > 0)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVCoef(zeroVec);

      double nrmh = h_.ComputeL2Error(zeroVCoef, zeroVCoef);
      if (myid_ == 0) { cout << "norm of H: " << nrmh << endl; }
   }

   // Compute D from H and J:  D = (Curl(H) - J) / (-i omega)
   ampere_.ComputeD();

   // Compute E from D: E = epsilon^{-1} D with BC given by -Grad(phi)
   this->computeE(ampere_.GetElectricFlux(), e_);

   if (myid_ == 0)
   {
      cout << " Solve done." << endl;
   }
}

void CPDSolverDH::collectBdrAttributes(const Array<AttributeArrays*> & aa,
                                       Array<int> & bdr_attr_marker)
{
   bdr_attr_marker.SetSize(pmesh_->bdr_attributes.Max());
   bdr_attr_marker = 0;
   if ( aa.Size() > 0 )
   {
      if ( aa.Size() == 1 && aa[0]->attr[0] == -1 )
      {
         bdr_attr_marker = 1;
      }
      else
      {
         for (int i=0; i<aa.Size(); i++)
         {
            for (int j=0; j<aa[i]->attr.Size(); j++)
            {
               bdr_attr_marker[aa[i]->attr[j]-1] = 1;
            }
         }
      }
   }
}

void CPDSolverDH::locateTrueDBCDofs(const Array<int> & dbc_bdr_marker,
                                    Array<int> & dbc_nd_tdofs)
{
   int mmax = dbc_bdr_marker.Max();
   int mmin = dbc_bdr_marker.Min();
   if (mmax != 0 || mmin != 0)
   {
      HCurlFESpace_->GetEssentialTrueDofs(dbc_bdr_marker, dbc_nd_tdofs);
   }
}

void CPDSolverDH::locateTrueSBCDofs(const Array<int> & sbc_bdr_marker,
                                    Array<int> & non_sbc_h1_tdofs,
                                    Array<int> & sbc_nd_tdofs)
{
   int mmax = sbc_bdr_marker.Max();
   int mmin = sbc_bdr_marker.Min();
   if (mmax != 0 || mmin != 0)
   {
      Array<int> sbc_h1_tdof;
      H1FESpace_->GetEssentialTrueDofs(sbc_bdr_marker, sbc_h1_tdof);

      Array<int> sbc_h1_tdof_marker;
      H1FESpace_->ListToMarker(sbc_h1_tdof, H1FESpace_->GetTrueVSize(),
                               sbc_h1_tdof_marker, 1);

      // Invert marker
      for (int i=0; i<sbc_h1_tdof_marker.Size(); i++)
      {
         sbc_h1_tdof_marker[i] = 1 - sbc_h1_tdof_marker[i];
      }

      H1FESpace_->MarkerToList(sbc_h1_tdof_marker, non_sbc_h1_tdofs);

      HCurlFESpace_->GetEssentialTrueDofs(sbc_bdr_marker, sbc_nd_tdofs);
   }
}

// Compute D from H and J:  D = (Curl(H) - J) / (-i omega)
void CPDSolverDH::computeD(const ParComplexGridFunction & h,
                           const ParComplexGridFunction & j,
                           ParComplexGridFunction & d)
{
   // D = -J / (-i omega) = -i J / omega
   d.real().Set( 1.0 / omega_, j.imag());
   d.imag().Set(-1.0 / omega_, j.real());

   // D += Curl(H) / (-i omega) = i Curl(H) / omega
   curl_->AddMult(h.imag(), d.real(),-1.0 / omega_);
   curl_->AddMult(h.real(), d.imag(), 1.0 / omega_);

   if (kReCoef_ || kImCoef_)
   {
      // D += i k x H / (-i omega) = -k x H / omega
      kReCross_->AddMult(h.real(), d.real(), -1.0 / omega_);
      kReCross_->AddMult(h.imag(), d.imag(), -1.0 / omega_);

      kImCross_->AddMult(h.imag(), d.real(),  1.0 / omega_);
      kImCross_->AddMult(h.real(), d.imag(), -1.0 / omega_);
   }

   if (logging_ > 1)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVCoef(zeroVec);

      double nrmd = d.ComputeL2Error(zeroVCoef, zeroVCoef);
      if (myid_ == 0) { cout << "norm of D: " << nrmd << endl; }
   }
}

// Compute E from D: E = epsilon^{-1} D with BC given by -Grad(phi)
void CPDSolverDH::computeE(const ParComplexGridFunction & d,
                           ParComplexGridFunction & e)
{
   m21EpsInv_->Mult(d, *rhs1_);
   rhs1_->SyncAlias();

   if (sbcs_.Size() > 0)
   {
      grad_->Mult(phi_->real(), e.real());
      grad_->Mult(phi_->imag(), e.imag());
   }
   else
   {
      e = 0.0;
   }

   OperatorHandle M1;
   Vector E, RHS1;

   m1_->FormLinearSystem(sbc_nd_tdofs_, e, *rhs1_, M1, E, RHS1);

   HypreDiagScale diag_r(M1.As<ComplexHypreParMatrix>()->real());
   Operator * diag_i = &diag_r;

   BlockDiagonalPreconditioner diag(blockTrueOffsets_);
   diag.SetDiagonalBlock(0, &diag_r);
   diag.SetDiagonalBlock(1, diag_i);
   diag.owns_blocks = 0;

   MINRESSolver minres(HCurlFESpace_->GetComm());
   minres.SetPreconditioner(diag);
   minres.SetOperator(*M1.Ptr());
   minres.SetRelTol(solOpts_.relTol);
   minres.SetMaxIter(solOpts_.maxIter);
   minres.SetPrintLevel(solOpts_.printLvl+1);

   minres.Mult(RHS1, E);

   m1_->RecoverFEMSolution(E, *rhs1_, e);

   if (logging_ > 0)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVCoef(zeroVec);

      double nrme = e.ComputeL2Error(zeroVCoef, zeroVCoef);
      if (myid_ == 0) { cout << "norm of E: " << nrme << endl; }
   }
}

double
CPDSolverDH::GetEFieldError(const VectorCoefficient & EReCoef,
                            const VectorCoefficient & EImCoef) const
{
   ParComplexGridFunction z(e_);
   z = 0.0;

   double solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));


   double solErr = e_.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

double
CPDSolverDH::GetHFieldError(const VectorCoefficient & HReCoef,
                            const VectorCoefficient & HImCoef) const
{
   ParFiniteElementSpace * fes =
      const_cast<ParFiniteElementSpace*>(h_.ParFESpace());
   ParComplexGridFunction z(fes);
   z = 0.0;

   double solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(HReCoef),
                                     const_cast<VectorCoefficient&>(HImCoef));


   double solErr = h_.ComputeL2Error(const_cast<VectorCoefficient&>(HReCoef),
                                     const_cast<VectorCoefficient&>(HImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

void
CPDSolverDH::GetErrorEstimates(Vector & errors, bool err_h)
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
   CurlCurlIntegrator *flux_integrator = NULL;
   if (err_h)
   {
      flux_integrator = new CurlCurlIntegrator(*epsInvReCoef_);
   }
   else
   {
      flux_integrator = new CurlCurlIntegrator(muInvCoef_);
   }
   ParFiniteElementSpace flux_fes(pmesh_, flux_fec);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   ParFiniteElementSpace smooth_flux_fes(pmesh_, smooth_flux_fec);

   Vector err_i(errors.Size());

   L2ZZErrorEstimator(*flux_integrator, (err_h) ? h_.real() : e_.real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);
   L2ZZErrorEstimator(*flux_integrator, (err_h) ? h_.imag() : e_.imag(),
                      smooth_flux_fes, flux_fes, err_i, norm_p);

   errors += err_i;

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void CPDSolverDH::prepareVectorVisField(const ParComplexGridFunction &u,
                                        ComplexGridFunction &v)
{
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());
      ComplexPhaseVectorCoefficient uk_r(kReCoef_, kImCoef_, &u_r, &u_i,
                                         true, false);
      ComplexPhaseVectorCoefficient uk_i(kReCoef_, kImCoef_, &u_r, &u_i,
                                         false, false);

      VectorR2DCoef uk_r_3D(uk_r, *pmesh_);
      VectorR2DCoef uk_i_3D(uk_i, *pmesh_);

      v.ProjectCoefficient(uk_r_3D, uk_i_3D);
   }
   else
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());

      VectorR2DCoef u_r_3D(u_r, *pmesh_);
      VectorR2DCoef u_i_3D(u_i, *pmesh_);

      v.ProjectCoefficient(u_r_3D, u_i_3D);
   }
}

void CPDSolverDH::prepareVisFields()
{
   h_v_.PrepareVisField(h_, kReCoef_, kImCoef_);
   h_dbc_v_.PrepareVisField(h_dbc_, kReCoef_, kImCoef_);
   d_v_.PrepareVisField(ampere_.GetElectricFlux(), kReCoef_, kImCoef_);
   e_v_.PrepareVisField(e_, kReCoef_, kImCoef_);
   if (phi_)
   {
      phi_v_.PrepareVisField(*phi_, kReCoef_, kImCoef_);
      /*
      if (sbcs_.Size() > 0)
      {
         for (int i = 0; i< sbcs_.Size(); i++)
         {
            ComplexCoefficientByAttr & sbc = *sbcs_[i];
            z_v_.PrepareVisField(*sbc.real, *sbc.imag, kReCoef_, kImCoef_,
                                 sbc.attr_marker);
         }
      }
      */
   }
   /*
   prepareVectorVisField(*h_, *h_v_);
   prepareVectorVisField(*e_, *e_v_);
   prepareVectorVisField(*d_, *d_v_);

   if (h_tilde_)
   {
      VectorGridFunctionCoefficient u_r(&h_->real());
      VectorGridFunctionCoefficient u_i(&h_->imag());

      VectorR2DCoef u_r_3D(u_r, *pmesh_);
      VectorR2DCoef u_i_3D(u_i, *pmesh_);

      h_tilde_->ProjectCoefficient(u_r_3D, u_i_3D);
   }
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

   if (BCoef_)
   {
      // VectorGridFunctionCoefficient b_hat(b_hat_);
      // VectorR2DCoef b_hat_3D(b_hat, *pmesh_);
      VectorR2DCoef b_hat_3D(*BCoef_, *pmesh_);
      b_hat_v_->ProjectCoefficient(b_hat_3D);

      VectorGridFunctionCoefficient e_r(&e_v_->real());
      VectorGridFunctionCoefficient e_i(&e_v_->imag());
      InnerProductCoefficient eb_r(e_r, *BCoef_);
      InnerProductCoefficient eb_i(e_i, *BCoef_);

      e_b_v_->ProjectCoefficient(eb_r, eb_i);
   }
   */
}

void
CPDSolverDH::RegisterVisItFields(VisItDataCollection & visit_dc)
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

   h_v_.RegisterVisItFields(visit_dc);
   h_dbc_v_.RegisterVisItFields(visit_dc);
   d_v_.RegisterVisItFields(visit_dc);
   e_v_.RegisterVisItFields(visit_dc);
   if (phi_) { phi_v_.RegisterVisItFields(visit_dc); }
   // if (phi_) { z_v_.RegisterVisItFields(visit_dc); }
   /*
   h_v_ = new ComplexGridFunction(HCurlFESpace3D_);
   e_v_ = new ComplexGridFunction(L2VFESpace3D_);
   d_v_ = new ComplexGridFunction(L2VFESpace3D_);

   visit_dc.RegisterField("Re_H", &h_v_->real());
   visit_dc.RegisterField("Im_H", &h_v_->imag());

   visit_dc.RegisterField("Re_H~", &h_tilde_->real());
   visit_dc.RegisterField("Im_H~", &h_tilde_->imag());

   visit_dc.RegisterField("Re_E", &e_v_->real());
   visit_dc.RegisterField("Im_E", &e_v_->imag());

   visit_dc.RegisterField("Re_D", &d_v_->real());
   visit_dc.RegisterField("Im_D", &d_v_->imag());

   visit_dc.RegisterField("Re_DivD", &div_d_->real());
   visit_dc.RegisterField("Im_DivD", &div_d_->imag());
   */
   /*
   visit_dc.RegisterField("Re_H", &h_->real());
   visit_dc.RegisterField("Im_H", &h_->imag());

   visit_dc.RegisterField("Re_E", &e_->real());
   visit_dc.RegisterField("Im_E", &e_->imag());

   visit_dc.RegisterField("Re_D", &d_->real());
   visit_dc.RegisterField("Im_D", &d_->imag());
   */
   /*
   if (sbcs_.Size() > 0)
   {
      phi_v_ = new ComplexGridFunction(L2FESpace3D_);

      visit_dc.RegisterField("Re_Phi", &phi_v_->real());
      visit_dc.RegisterField("Im_Phi", &phi_v_->imag());
   }

   if ( rectPot_ )
   {
      rectPot_v_ = new ComplexGridFunction(L2FESpace3D_);

      visit_dc.RegisterField("Rec_Re_Phi", &rectPot_v_->real());
      visit_dc.RegisterField("Rec_Im_Phi", &rectPot_v_->imag());
   }

   if ( BCoef_)
   {
      b_hat_v_ = new GridFunction(L2VFESpace3D_);
      e_b_v_   = new ComplexGridFunction(L2FESpace3D_);

      visit_dc.RegisterField("B_hat", b_hat_v_);
      visit_dc.RegisterField("Re_EB", &e_b_v_->real());
      visit_dc.RegisterField("Im_EB", &e_b_v_->imag());
      // visit_dc.RegisterField("Re_EpsPara", &EpsPara_->real());
      // visit_dc.RegisterField("Im_EpsPara", &EpsPara_->imag());
   }
   */
   // visit_dc.RegisterField("Er", e_r_);
   // visit_dc.RegisterField("Ei", e_i_);
   // visit_dc.RegisterField("B", b_);
   // visit_dc.RegisterField("H", h_);
   /*
   if ( j_ )
   {
      visit_dc.RegisterField("Re_J", &j_->real());
      visit_dc.RegisterField("Im_J", &j_->imag());
   }
   */
   /*
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
   if ( StixS_ )
   {
      visit_dc.RegisterField("Re_StixS", &StixS_->real());
      visit_dc.RegisterField("Im_StixS", &StixS_->imag());
      visit_dc.RegisterField("Re_StixD", &StixD_->real());
      visit_dc.RegisterField("Im_StixD", &StixD_->imag());
      visit_dc.RegisterField("Re_StixP", &StixP_->real());
      visit_dc.RegisterField("Im_StixP", &StixP_->imag());
   }
   // if ( j_r_ ) { visit_dc.RegisterField("Jr", j_r_); }
   // if ( j_i_ ) { visit_dc.RegisterField("Ji", j_i_); }
   // if ( k_ ) { visit_dc.RegisterField("K", k_); }
   // if ( m_ ) { visit_dc.RegisterField("M", m_); }
   // if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
}

void
CPDSolverDH::WriteVisItFields(int it)
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
      if ( rectPot_ )
      {

         ComplexCoefficientByAttr * sbc = sbcs_[0];
         SheathBase * sb = dynamic_cast<SheathBase*>(sbc->real);

         if (sb != NULL)
         {
            RectifiedSheathPotential rectPotCoefR(*sb, true);
            RectifiedSheathPotential rectPotCoefI(*sb, false);

            rectPot_->ProjectCoefficient(rectPotCoefR, rectPotCoefI);
         }

      }
      */
      if ( BCoef_)
      {
         /*
         b_hat_->ProjectCoefficient(*BCoef_);

               VectorGridFunctionCoefficient e_r(&e_->real());
               VectorGridFunctionCoefficient e_i(&e_->imag());
               InnerProductCoefficient ebrCoef(e_r, *BCoef_);
               InnerProductCoefficient ebiCoef(e_i, *BCoef_);

               e_b_->ProjectCoefficient(ebrCoef, ebiCoef);
         */
         /*
                    MatrixVectorProductCoefficient ReEpsB(*epsReCoef_, *BCoef_);
                    MatrixVectorProductCoefficient ImEpsB(*epsImCoef_, *BCoef_);
                    InnerProductCoefficient ReEpsParaCoef(*BCoef_, ReEpsB);
                    InnerProductCoefficient ImEpsParaCoef(*BCoef_, ImEpsB);

                    EpsPara_->ProjectCoefficient(ReEpsParaCoef, ImEpsParaCoef);
                    *EpsPara_ *= 1.0 / epsilon0_;
               */
      }

      //ComplexCoefficientByAttr & sbc = (*sbcs_)[0];
      //SheathBase * sb = dynamic_cast<SheathBase*>(sbc.real);

      /*
      if (sb != NULL)
      {
        RectifiedSheathPotential rectPotCoefR(*sb, true);
        RectifiedSheathPotential rectPotCoefI(*sb, false);

        rectPot_->ProjectCoefficient(rectPotCoefR, rectPotCoefI);
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
CPDSolverDH::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:
         socks_["Hr"] = new socketstream;
         socks_["Hr"]->precision(8);

         socks_["Hi"] = new socketstream;
         socks_["Hi"]->precision(8);

         socks_["Er"] = new socketstream;
         socks_["Er"]->precision(8);

         socks_["Ei"] = new socketstream;
         socks_["Ei"]->precision(8);

         socks_["Dr"] = new socketstream;
         socks_["Dr"]->precision(8);

         socks_["Di"] = new socketstream;
         socks_["Di"]->precision(8);
         break;
   }

   if (sbcs_.Size() > 0)
   {
      socks_["Phir"] = new socketstream;
      socks_["Phir"]->precision(8);

      socks_["Phii"] = new socketstream;
      socks_["Phii"]->precision(8);

      socks_["RecPhir"] = new socketstream;
      socks_["RecPhir"]->precision(8);

      socks_["RecPhii"] = new socketstream;
      socks_["RecPhii"]->precision(8);
   }

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

   if ( j_ )
   {
      switch (pmesh_->Dimension())
      {
         case 1:
            break;
         case 2:
            break;
         case 3:
            socks_["Jr"] = new socketstream;
            socks_["Jr"]->precision(8);

            socks_["Ji"] = new socketstream;
            socks_["Ji"]->precision(8);
            break;
      }
   }
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
CPDSolverDH::DisplayToGLVis()
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
      cout << "Preparing H viz" << endl;
      VectorGridFunctionCoefficient h_r(&h_->real());
      VectorGridFunctionCoefficient h_i(&h_->imag());
      VectorSumCoefficient hrCoef(h_r, h_i, *coskx_, *negsinkx_);
      VectorSumCoefficient hiCoef(h_i, h_r, *coskx_, *sinkx_);

      h_v_->ProjectCoefficient(hrCoef, hiCoef);

      cout << "Preparing E viz" << endl;
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);

      cout << "Preparing D viz" << endl;
      VectorGridFunctionCoefficient d_r(&d_->real());
      VectorGridFunctionCoefficient d_i(&d_->imag());
      VectorSumCoefficient drCoef(d_r, d_i, *coskx_, *negsinkx_);
      VectorSumCoefficient diCoef(d_i, d_r, *coskx_, *sinkx_);

      d_v_->ProjectCoefficient(drCoef, diCoef);

      if (sbcs_.Size() > 0)
      {
         GridFunctionCoefficient phi_r(&phi_->real());
         GridFunctionCoefficient phi_i(&phi_->imag());
         ProductCoefficient cosphi_r(phi_r, *coskx_);
         ProductCoefficient cosphi_i(phi_i, *coskx_);
         ProductCoefficient sinphi_r(phi_r, *sinkx_);
         ProductCoefficient negsinphi_i(phi_i, *negsinkx_);
         SumCoefficient phirCoef(cosphi_r, negsinphi_i);
         SumCoefficient phiiCoef(cosphi_i, sinphi_r);

         phi_v_->ProjectCoefficient(phirCoef, phiiCoef);
      }
   }
   else
   {
      h_v_ = h_;

      cout << "Preparing E viz" << endl;
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(e_r, e_i);

      d_v_ = d_;
      phi_v_ = phi_;
   }
   */
   /*
   ostringstream hr_keys, hi_keys;
   hr_keys << "aaAcPPPP";
   hi_keys << "aaAcPPPP";

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:

         VisualizeField(*socks_["Hr"], vishost, visport,
                        h_v_->real(), "Magnetic Field, Re(H)", Wx, Wy, Ww, Wh,
                        hr_keys.str().c_str());
         Wx += offx;

         VisualizeField(*socks_["Hi"], vishost, visport,
                        h_v_->imag(), "Magnetic Field, Im(H)", Wx, Wy, Ww, Wh,
                        hr_keys.str().c_str());

         Wx += offx;
         break;
   }

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
   Wx += offx;
   VisualizeField(*socks_["Dr"], vishost, visport,
                 d_v_->real(), "Electric Flux, Re(D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Di"], vishost, visport,
                 d_v_->imag(), "Electric Flux, Im(D)", Wx, Wy, Ww, Wh);
    */
   /*
   if (sbcs_.Size() > 0)
   {
      ostringstream pr_keys, pi_keys;
      pr_keys << "aaAc";
      pi_keys << "aaAc";

      Wx += offx;
      VisualizeField(*socks_["Phir"], vishost, visport,
                     phi_v_->real(), "Sheath Potential, Re(Phi)",
                     Wx, Wy, Ww, Wh, pr_keys.str().c_str());
      Wx += offx;

      VisualizeField(*socks_["Phii"], vishost, visport,
                     phi_v_->imag(), "Sheath Potential, Im(Phi)",
                     Wx, Wy, Ww, Wh, pi_keys.str().c_str());
   */
   /*

   Wx += offx;
   VisualizeField(*socks_["RecPhir"], vishost, visport,
                 rectPot_->real(), "Rectified Potential, Re(RecPhi)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["RecPhii"], vishost, visport,
                 rectPot_->imag(), "Rectified Potential, Im(RecPhi)", Wx, Wy, Ww, Wh);
    */

   // }

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
       ostringstream ebr_keys, ebi_keys;
       ebr_keys << "aaAc";
       ebi_keys << "aaAc";

       VisualizeField(*socks_["EBr"], vishost, visport,
                      e_b_v_->real(), "Parallel Electric Field, Re(E.B)",
                      Wx, Wy, Ww, Wh, ebr_keys.str().c_str());
       Wx += offx;

       VisualizeField(*socks_["EBi"], vishost, visport,
                      e_b_v_->imag(), "Parallel Electric Field, Im(E.B)",
                      Wx, Wy, Ww, Wh, ebi_keys.str().c_str());
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
   if ( j_ )
   {
      Wx = 0; Wy += offy; // next line

      // j_->ProjectCoefficient(*jrCoef_, *jiCoef_);
      /*
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
      */
      /*
      switch (pmesh_->Dimension())
      {
         case 1:
            break;
         case 2:
            break;
         case 3:
            VisualizeField(*socks_["Jr"], vishost, visport,
                           j_v_->real(), "Current Density, Re(J)", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(*socks_["Ji"], vishost, visport,
                           j_v_->imag(), "Current Density, Im(J)", Wx, Wy, Ww, Wh);
            break;
      }
      */
   }
   Wx = 0; Wy += offy; // next line
   /*
   if ( u_ )
   {
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
   }
   */
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
CPDSolverDH::DisplayAnimationToGLVis()
{
   if (myid_ == 0) { cout << "Sending animation data to GLVis ..." << flush; }
   /*
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      // VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      // VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);
      ComplexPhaseVectorCoefficient erCoef(kReCoef_, kImCoef_, &e_r, &e_i,
                                           true, false);
      ComplexPhaseVectorCoefficient eiCoef(kReCoef_, kImCoef_, &e_r, &e_i,
                                           false, false);

      e_v_->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());

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
            << "keys cvvvppppp\n"
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
           sin( -2.0 * M_PI * t), e_v_->imag(), *e_t_);
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
