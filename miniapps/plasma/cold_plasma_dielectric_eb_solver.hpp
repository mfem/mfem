// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER

#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "plasma.hpp"
#include "stix_bcs.hpp"
#include "vis_object.hpp"
#include "cold_plasma_dielectric_coefs.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

//using common::L2_FESpace;
using common::H1_ParFESpace;
using common::ND_R1D_ParFESpace;
using common::ND_R2D_ParFESpace;
using common::ND_ParFESpace;
using common::RT_R1D_ParFESpace;
using common::RT_R2D_ParFESpace;
using common::RT_ParFESpace;
using common::L2_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::ParDiscreteDivOperator;

namespace plasma
{

inline ParFiniteElementSpace *
MakeHCurlParFESpace(ParMesh &pmesh, int order)
{
   switch (pmesh.Dimension())
   {
      case 1:
         return new ND_R1D_ParFESpace(&pmesh, order, 1);
      case 2:
         return new ND_R2D_ParFESpace(&pmesh, order, 2);
      case 3:
         return new ND_ParFESpace(&pmesh, order, 3);
      default:
         return NULL;
   }
}

inline ParFiniteElementSpace *
MakeHDivParFESpace(ParMesh &pmesh, int order)
{
   switch (pmesh.Dimension())
   {
      case 1:
         return new RT_R1D_ParFESpace(&pmesh, order, 1);
      case 2:
         return new RT_R2D_ParFESpace(&pmesh, order, 2);
      case 3:
         return new RT_ParFESpace(&pmesh, order, 3);
      default:
         return NULL;
   }
}

/*
/// Coefficient which returns func(kr*x)*exp(-ki*x) where kr and ki are vectors
class ComplexPhaseCoefficient : public Coefficient
{
private:
 real_t(*func_)(real_t);
 VectorCoefficient * kr_;
 VectorCoefficient * ki_;
 mutable Vector krVec_;
 mutable Vector kiVec_;

public:
 ComplexPhaseCoefficient(Vector & kr, Vector & ki, real_t(&func)(real_t))
    : func_(&func), kr_(NULL), ki_(NULL), krVec_(kr), kiVec_(ki) {}

 ComplexPhaseCoefficient(VectorCoefficient & kr, VectorCoefficient & ki,
                         real_t(&func)(real_t))
    : func_(&func), kr_(&kr), ki_(&ki),
      krVec_(kr.GetVDim()), kiVec_(ki.GetVDim()) {}

 real_t Eval(ElementTransformation &T,
             const IntegrationPoint &ip)
 {
    real_t x[3];
    Vector transip(x, 3); transip = 0.0;
    T.Transform(ip, transip);
    if (kr_) { kr_->Eval(krVec_, T, ip); }
    if (ki_) { ki_->Eval(kiVec_, T, ip); }
    transip.SetSize(krVec_.Size());
    return (*func_)(krVec_ * transip)*exp(-(kiVec_ * transip));
 }
};
*/
class ElectricEnergyDensityCoef : public Coefficient
{
public:
   ElectricEnergyDensityCoef(VectorCoefficient &Er, VectorCoefficient &Ei,
                             MatrixCoefficient &epsr, MatrixCoefficient &epsi);

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable DenseMatrix eps_r_;
   mutable DenseMatrix eps_i_;
};

class ElectricEnergyDensityEDCoef : public Coefficient
{
public:
  ElectricEnergyDensityEDCoef(VectorCoefficient &Er, VectorCoefficient &Ei,
			      VectorCoefficient &Dr, VectorCoefficient &Di);

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &DrCoef_;
   VectorCoefficient &DiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
};

class MagneticEnergyDensityCoef : public Coefficient
{
public:
   MagneticEnergyDensityCoef(real_t omega,
                             VectorCoefficient &dEr, VectorCoefficient &dEi,
                             Coefficient &muInv);

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Br_;
   mutable Vector Bi_;
};

class EnergyDensityCoef : public Coefficient
{
public:
   EnergyDensityCoef(real_t omega,
                     VectorCoefficient &Er, VectorCoefficient &Ei,
                     VectorCoefficient &dEr, VectorCoefficient &dEi,
                     MatrixCoefficient &epsr, MatrixCoefficient &epsi,
                     Coefficient &muInv);

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable Vector Br_;
   mutable Vector Bi_;
   mutable DenseMatrix eps_r_;
   mutable DenseMatrix eps_i_;
};

class PoyntingVectorReCoef : public VectorCoefficient
{
public:
   PoyntingVectorReCoef(real_t omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Hr_;
   mutable Vector Hi_;
};

class PoyntingVectorImCoef : public VectorCoefficient
{
public:
   PoyntingVectorImCoef(real_t omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Hr_;
   mutable Vector Hi_;
};

class MinkowskiMomentumDensityReCoef : public VectorCoefficient
{
public:
   MinkowskiMomentumDensityReCoef(real_t omega,
                                  VectorCoefficient &Er,
                                  VectorCoefficient &Ei,
                                  VectorCoefficient &dEr,
                                  VectorCoefficient &dEi,
                                  MatrixCoefficient &epsr,
                                  MatrixCoefficient &epsi);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable Vector Br_;
   mutable Vector Bi_;
   mutable DenseMatrix epsr_;
   mutable DenseMatrix epsi_;
};

class MinkowskiMomentumDensityImCoef : public VectorCoefficient
{
public:
   MinkowskiMomentumDensityImCoef(real_t omega,
                                  VectorCoefficient &Er,
                                  VectorCoefficient &Ei,
                                  VectorCoefficient &dEr,
                                  VectorCoefficient &dEi,
                                  MatrixCoefficient &epsr,
                                  MatrixCoefficient &epsi);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   real_t omega_;

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;
   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
   mutable Vector Dr_;
   mutable Vector Di_;
   mutable Vector Br_;
   mutable Vector Bi_;
   mutable DenseMatrix epsr_;
   mutable DenseMatrix epsi_;
};

class TensorCompCoef : public Coefficient
{
public:
   TensorCompCoef(MatrixCoefficient &m, int i, int j)
      : i_(i), j_(j), m_(m) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      m_.Eval(M_, T, ip);
      return M_(i_, j_);
   }

private:
   int i_;
   int j_;

   MatrixCoefficient &m_;

   mutable DenseMatrix M_;
};

class Maxwell2ndE : public ParSesquilinearForm
{
private:

   class MassCoefficient : public MatrixCoefficient
   {
   private:
      real_t omegaSqr_;
      MatrixCoefficient & mCoef_;

   public:

      MassCoefficient(real_t omega, MatrixCoefficient & mCoef)
         : MatrixCoefficient(mCoef.GetHeight()),
           omegaSqr_(omega * omega),
           mCoef_(mCoef)
      {}

      virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                        const IntegrationPoint &ip)
      {
         mCoef_.Eval(M, T, ip);
         M *= -omegaSqr_;
      }
   };

   class kmkCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      Coefficient       * mCoef_;
      MatrixCoefficient * MCoef_;

      bool realPart_;
      real_t a_;

      mutable Vector kr;
      mutable Vector ki;
      mutable DenseMatrix M_;

      void kmk(real_t a,
               const Vector & kl, real_t m, const Vector &kr,
               DenseMatrix & M)
      {
         real_t kk = kl * kr;
         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               M(i,j) += a * m * kl(j) * kr(i);
            }
            M(i,i) -= a * m * kk;
         }
      }

      void kmk(real_t a,
               const Vector & kl, DenseMatrix & m, const Vector &kr,
               DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            int i1 = (i + 1) % 3;
            int i2 = (i + 2) % 3;
            for (int j=0; j<3; j++)
            {
               int j1 = (j + 1) % 3;
               int j2 = (j + 2) % 3;
               M(i,j) += a * kl(i1) * m(i2,j1) * kr(j2);
               M(i,j) += a * kl(i2) * m(i1,j2) * kr(j1);
               M(i,j) -= a * kl(i1) * m(i2,j2) * kr(j1);
               M(i,j) -= a * kl(i2) * m(i1,j1) * kr(j2);
            }
         }
      }

   public:
      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     Coefficient *mCoef,
                     bool realPart, real_t a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(mCoef),
           MCoef_(NULL),
           realPart_(realPart),
           a_(a), kr(3), ki(3)
      { kr = 0.0; ki = 0.0; }

      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     MatrixCoefficient *MCoef,
                     bool realPart, real_t a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3), M_(3)
      { kr = 0.0; ki = 0.0; M_ = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             mCoef_ == NULL)
         {
            return;
         }
         real_t m = 0.0;
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MCoef_) { MCoef_->Eval(M_, T, ip); }

         if (realPart_)
         {
            if (!MCoef_)
            {
               if (krCoef_) { kmk(1.0, kr, m, kr, M); }
               if (kiCoef_) { kmk(-1.0, ki, m, ki, M); }
            }
            else
            {
               if (krCoef_) { kmk(1.0, kr, M_, kr, M); }
               if (kiCoef_) { kmk(-1.0, ki, M_, ki, M); }
            }
         }
         else
         {
            if (!MCoef_)
            {
               if (krCoef_ && kiCoef_) { kmk(1.0, kr, m, ki, M); }
               if (kiCoef_ && krCoef_) { kmk(1.0, ki, m, kr, M); }
            }
            else
            {
               if (krCoef_ && kiCoef_) { kmk(1.0, kr, M_, ki, M); }
               if (kiCoef_ && krCoef_) { kmk(1.0, ki, M_, kr, M); }
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class CrossCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * kCoef_;
      Coefficient       * mCoef_;
      MatrixCoefficient * MCoef_;

      real_t a_;

      bool km_;

      mutable Vector k;
      mutable DenseMatrix M_;

   public:
      CrossCoefficient(VectorCoefficient *kCoef,
                       Coefficient *mCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(mCoef),
           MCoef_(NULL),
           a_(a), km_(true), k(3)
      { k = 0.0; }

      CrossCoefficient(VectorCoefficient *kCoef,
                       MatrixCoefficient *MCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           a_(a), km_(true), k(3), M_(3)
      { k = 0.0; M_ = 0.0; }

      CrossCoefficient(MatrixCoefficient *MCoef,
                       VectorCoefficient *kCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           a_(a), km_(false), k(3), M_(3)
      { k = 0.0; M_ = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;

         real_t m = 0.0;
         if (kCoef_) { kCoef_->Eval(k, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MCoef_) { MCoef_->Eval(M_, T, ip); }

         if (MCoef_ == NULL)
         {
            M(2,1) = a_ * m * k(0);
            M(0,2) = a_ * m * k(1);
            M(1,0) = a_ * m * k(2);

            M(1,2) = -M(2,1);
            M(2,0) = -M(0,2);
            M(0,1) = -M(1,0);
         }
         else
         {
            if (km_)
            {
               for (int i=0; i<3; i++)
               {
                  int i1 = (i + 1) % 3;
                  int i2 = (i + 2) % 3;
                  for (int j=0; j<3; j++)
                  {
                     M(i,j) += a_ * k(i1) * M_(i2,j);
                     M(i,j) -= a_ * k(i2) * M_(i1,j);
                  }
               }
            }
            else
            {
               for (int i=0; i<3; i++)
               {
                  for (int j=0; j<3; j++)
                  {
                     int j1 = (j + 1) % 3;
                     int j2 = (j + 2) % 3;
                     M(i,j) += a_ * M_(i,j1) * k(j2);
                     M(i,j) -= a_ * M_(i,j2) * k(j1);
                  }
               }
            }
         }
      }
   };

   MassCoefficient massReCoef_;  // -omega^2 Re(epsilon)
   MassCoefficient massImCoef_;  // -omega^2 Im(epsilon)

   kmkCoefficient kmkReCoef_;  // Real part of k x muInv k x
   kmkCoefficient kmkImCoef_;  // Imaginary part of k x muInv k x
   CrossCoefficient kmReCoef_; // Real part of k x muInv
   CrossCoefficient kmImCoef_; // Imaginary part of k x muInv

public:
   Maxwell2ndE(ParFiniteElementSpace & HCurlFESpace,
               real_t omega,
               ComplexOperator::Convention conv,
               MatrixCoefficient & epsReCoef,
               MatrixCoefficient & epsImCoef,
               Coefficient & muInvCoef,
               VectorCoefficient * kReCoef,
               VectorCoefficient * kImCoef);

   void Assemble();
};

class CurrentSourceE : public ParComplexLinearForm
{
private:

   real_t omega_;

   ParComplexGridFunction jt_;
   ParComplexGridFunction kt_;

   // These contain the J*exp(-i k x) and K*exp(-i k x)
   CmplxVecCoefArray jtilde_; // Volume Currents w/Phase
   CmplxVecCoefArray ktilde_; // Surface Currents w/Phase

public:
   CurrentSourceE(ParFiniteElementSpace & HCurlFESpace,
                  ParFiniteElementSpace & HDivFESpace,
                  real_t omega,
                  ComplexOperator::Convention conv,
                  const CmplxVecCoefArray & jsrc,
                  const CmplxVecCoefArray & ksrc,
                  VectorCoefficient * kReCoef,
                  VectorCoefficient * kImCoef);

   ~CurrentSourceE();

   ParComplexGridFunction & GetVolumeCurrentDensity() { return jt_; }
   ParComplexGridFunction & GetSurfaceCurrentDensity() { return kt_; }

   void Update();
   void Assemble();
};

class FaradaysLaw
{
private:

   const ParComplexGridFunction & e_; // Complex electric field (HCurl)
   ParComplexGridFunction         b_; // Complex magnetic flux (HDiv)

   const real_t omega_;

   ParDiscreteCurlOperator curl_;

   ParDiscreteLinearOperator * kReCross_;
   ParDiscreteLinearOperator * kImCross_;

public:
   FaradaysLaw(const ParComplexGridFunction &e,
               ParFiniteElementSpace & HDivFESpace,
               real_t omega,
               VectorCoefficient * kReCoef,
               VectorCoefficient * kImCoef);

   ParComplexGridFunction & GetMagneticFlux() { return b_; }

   void Update();
   void Assemble();
   void ComputeB();
};

class GausssLaw
{
private:

   const ParComplexGridFunction &  f_; // Complex pseudovector field (HDiv)
   ParComplexGridFunction         df_; // Complex divergence (L2)

   ParDiscreteDivOperator div_;

   ParDiscreteLinearOperator * kReDot_;
   ParDiscreteLinearOperator * kImDot_;

   bool assembled_;

public:
   GausssLaw(const ParComplexGridFunction &f,
             ParFiniteElementSpace & L2FESpace,
             VectorCoefficient * kReCoef,
             VectorCoefficient * kImCoef);

   ParComplexGridFunction & GetDivergence() { return df_; }

   void Update();
   void Assemble();
   void ComputeDiv();
};

class ComplexMatrixVectorProductCoef : public VectorCoefficient
{
protected:

   const GridFunction * vRe_gf_;
   const GridFunction * vIm_gf_;

   MatrixCoefficient * mReCoef_;
   MatrixCoefficient * mImCoef_;

   mutable Vector vRe_;
   mutable Vector vIm_;
   mutable DenseMatrix mRe_;
   mutable DenseMatrix mIm_;

   ComplexMatrixVectorProductCoef(MatrixCoefficient *MRe,
                                  MatrixCoefficient *MIm,
                                  const GridFunction *vRe,
                                  const GridFunction *vIm)
      : VectorCoefficient((MRe) ? MRe->GetHeight() : MIm->GetHeight()),
        vRe_gf_(vRe),
        vIm_gf_(vIm),
        mReCoef_(MRe),
        mImCoef_(MIm)
   {
      vRe_.SetSize(vdim); vRe_ = 0.0;
      vIm_.SetSize(vdim); vIm_ = 0.0;
      mRe_.SetSize(vdim); mRe_ = 0.0;
      mIm_.SetSize(vdim); mIm_ = 0.0;
   }

   void evaluateCoefs(ElementTransformation &T,
                      const IntegrationPoint &ip)
   {
      if (vRe_gf_)
      {
         vRe_gf_->GetVectorValue(T, ip, vRe_);
      }
      if (vIm_gf_)
      {
         vIm_gf_->GetVectorValue(T, ip, vIm_);
      }
      if (mReCoef_)
      {
         mReCoef_->Eval(mRe_, T, ip);
      }
      if (mImCoef_)
      {
         mImCoef_->Eval(mIm_, T, ip);
      }
   }
};

class ComplexMatrixVectorProductRealCoef : public ComplexMatrixVectorProductCoef
{
public:
   ComplexMatrixVectorProductRealCoef(MatrixCoefficient *MRe,
                                      MatrixCoefficient *MIm,
                                      const GridFunction *vRe,
                                      const GridFunction *vIm)
      : ComplexMatrixVectorProductCoef(MRe, MIm, vRe, vIm)
   {}

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      this->evaluateCoefs(T, ip);
      mRe_.Mult(vRe_, V);
      mIm_.AddMult_a(-1.0, vIm_, V);
   }
};

class ComplexMatrixVectorProductImagCoef : public ComplexMatrixVectorProductCoef
{
public:
   ComplexMatrixVectorProductImagCoef(MatrixCoefficient *MRe,
                                      MatrixCoefficient *MIm,
                                      const GridFunction *vRe,
                                      const GridFunction *vIm)
      : ComplexMatrixVectorProductCoef(MRe, MIm, vRe, vIm)
   {}

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      this->evaluateCoefs(T, ip);
      mRe_.Mult(vIm_, V);
      mIm_.AddMult(vRe_, V);
   }
};

class Displacement
{
private:
   ParComplexGridFunction d_; // Complex electric displacement (HDiv)

   ComplexMatrixVectorProductRealCoef dReCoef_;
   ComplexMatrixVectorProductImagCoef dImCoef_;

   ParComplexLinearForm d_lf_; // Dual displacement (HDiv)
   ParBilinearForm m_; // HDiv mass matrix

   mutable Vector RHS_;
   mutable Vector D_;

public:
   Displacement(const ParComplexGridFunction &e,
                ParFiniteElementSpace & HDivFESpace,
                MatrixCoefficient & epsReCoef,
                MatrixCoefficient & epsImCoef);

   ParComplexGridFunction & GetDisplacement() { return d_; }

   void Update();
   void Assemble();
   void ComputeD();
};

class ParallelElectricFieldVisObject : public ComplexScalarFieldVisObject
{
private:

   VectorCoefficient & BCoef_;

public:
   ParallelElectricFieldVisObject(const std::string & field_name,
                                  VectorCoefficient &BCoef,
                                  L2_ParFESpace *sfes,
                                  bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        VectorCoefficient *kr,
                        VectorCoefficient *ki);
};

class ElectricEnergyDensityVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   ElectricEnergyDensityVisObject(const std::string & field_name,
                                  L2_ParFESpace *sfes,
                                  bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        MatrixCoefficient &epsr,
                        MatrixCoefficient &epsi);
};

class ElectricEnergyDensityEDVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   ElectricEnergyDensityEDVisObject(const std::string & field_name,
				    L2_ParFESpace *sfes,
				    bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        const ParComplexGridFunction &d);
};

class MagneticEnergyDensityVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   MagneticEnergyDensityVisObject(const std::string & field_name,
                                  L2_ParFESpace *sfes,
                                  bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &de,
                        real_t omega,
                        Coefficient &muInv);
};

class EnergyDensityVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   EnergyDensityVisObject(const std::string & field_name,
                          L2_ParFESpace *sfes,
                          bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        real_t omega,
                        MatrixCoefficient &epsr,
                        MatrixCoefficient &epsi,
                        Coefficient &muInv);
};

class PoyntingVectorVisObject : public ComplexVectorFieldVisObject
{
private:

   using ComplexVectorFieldVisObject::PrepareVisField;

public:
   PoyntingVectorVisObject(const std::string & field_name,
                           L2_ParFESpace *vfes, L2_ParFESpace *sfes,
                           bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        real_t omega,
                        Coefficient & muInvCoef);
};

class MinkowskiMomentumDensityVisObject : public ComplexVectorFieldVisObject
{
private:

   using ComplexVectorFieldVisObject::PrepareVisField;

public:
   MinkowskiMomentumDensityVisObject(const std::string & field_name,
                                     L2_ParFESpace *vfes, L2_ParFESpace *sfes,
                                     bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        real_t omega,
                        MatrixCoefficient & epsReCoef,
                        MatrixCoefficient & epsImCoef);
};

class TensorCompVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   TensorCompVisObject(const std::string & field_name,
                       L2_ParFESpace *sfes,
                       bool cyl, bool pseudo);

   void PrepareVisField(MatrixCoefficient &mr,
                        MatrixCoefficient &mi,
                        int i, int j);
};

class CPDInputVis
{
public:
   CPDInputVis(StixParams &stixParams,
	       std::shared_ptr<L2_ParFESpace> l2_sfes,
	       std::shared_ptr<L2_ParFESpace> l2_vfes,
	       bool cyl);

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }
  
   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields();

   void Update();
  
private:

  std::shared_ptr<L2_ParFESpace> sfes_;
  std::shared_ptr<L2_ParFESpace> vfes_;
  
  // Background Magnetic Field
  VectorCoefficient &BCoef_;

  VectorFieldVisObject B_;
  
  // Stix Dielectric tensor Coefficients
  StixSCoef stixSReCoef_;
  StixSCoef stixSImCoef_;
  StixDCoef stixDReCoef_;
  StixDCoef stixDImCoef_;
  StixPCoef stixPReCoef_;
  StixPCoef stixPImCoef_;
  
  ComplexScalarFieldVisObject stixS_;
  ComplexScalarFieldVisObject stixD_;
  ComplexScalarFieldVisObject stixP_;

  // Wave Lengths of plane waves parallel or perpendicular to the magnetic field
  StixWaveLengthCoef lambdaLCoef_;
  StixWaveLengthCoef lambdaRCoef_;
  StixWaveLengthCoef lambdaOCoef_;
  StixWaveLengthCoef lambdaXCoef_;
  
  ScalarFieldVisObject waveLengthL_;
  ScalarFieldVisObject waveLengthR_;
  ScalarFieldVisObject waveLengthO_;
  ScalarFieldVisObject waveLengthX_;

  // Decay constants for plane waves parallel or perpendicular to the magnetic
  // field
  StixWaveLengthCoef deltaLCoef_;
  StixWaveLengthCoef deltaRCoef_;
  StixWaveLengthCoef deltaOCoef_;
  StixWaveLengthCoef deltaXCoef_;

  ScalarFieldVisObject skinDepthL_;
  ScalarFieldVisObject skinDepthR_;
  ScalarFieldVisObject skinDepthO_;
  ScalarFieldVisObject skinDepthX_;  
};

class CPDFieldVis
{
public:
   CPDFieldVis(StixParams &stixParams,
	       std::shared_ptr<L2_ParFESpace> l2_sfes,
	       std::shared_ptr<L2_ParFESpace> l2_vfes,
	       const std::string hcurl_field_name,
	       const std::string hdiv_field_name,
	       bool cyl);

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }
  
   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields(const ParComplexGridFunction & e,
			 const ParComplexGridFunction & b,
			 VectorCoefficient * kReCoef,
			 VectorCoefficient * kImCoef);

   void Update();
  
private:

  std::shared_ptr<L2_ParFESpace> sfes_;
  std::shared_ptr<L2_ParFESpace> vfes_;

  ComplexVectorFieldVisObject HCurlField_;
  ComplexVectorFieldVisObject HDivField_;
};

class CPDOutputVis
{
public:
   CPDOutputVis(StixParams &stixParams,
		std::shared_ptr<L2_ParFESpace> l2_sfes,
		std::shared_ptr<L2_ParFESpace> l2_vfes,
		real_t omega,
		bool cyl);

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }
  
   void RegisterVisItFields(VisItDataCollection & visit_dc);

  void PrepareVisFields(const ParComplexGridFunction & e,
			const ParComplexGridFunction & d);

   void Update();
  
private:

   std::shared_ptr<L2_ParFESpace> sfes_;
   std::shared_ptr<L2_ParFESpace> vfes_;

   real_t omega_;
  
   DielectricTensor epsReCoef_;    // Dielectric Material Coefficient
   DielectricTensor epsImCoef_;    // Dielectric Material Coefficient
   ConstantCoefficient muInvCoef_; // Dia/Paramagnetic Material Coefficient

   EnergyDensityVisObject energyDensity_;
   PoyntingVectorVisObject poyntingFlux_;
   MinkowskiMomentumDensityVisObject momentumDensity_;

   ElectricEnergyDensityEDVisObject edEnergyDensity_;
};

/// Cold Plasma Dielectric Solver
class CPDSolverEB
{
public:

   enum PrecondType
   {
      INVALID_PC  = -1,
      DIAG_SCALE  =  1,
      PARASAILS   =  2,
      EUCLID      =  3,
      AMS         =  4
   };

   enum SolverType
   {
      INVALID_SOL = -1,
      GMRES       =  1,
      FGMRES      =  2,
      MINRES      =  3,
      SUPERLU     =  4,
      STRUMPACK   =  5,
      DMUMPS      =  6
   };

   // Solver options
   struct SolverOptions
   {
      int maxIter;
      int kDim;
      int printLvl;
      real_t relTol;

      // Euclid Options
      int euLvl;
   };

   CPDSolverEB(ParMesh & pmesh, int order,
               CPDSolverEB::SolverType s, SolverOptions & sOpts,
               CPDSolverEB::PrecondType p,
               ComplexOperator::Convention conv,
	       StixParams &stixParams,
	       // Coefficient * etaInvCoef,
               VectorCoefficient * kReCoef,
               VectorCoefficient * kImCoef,
	       // Array<int> & abcs,
               StixBCs & stixBCs,
               bool vis_u = false, bool cyl = false);
   ~CPDSolverEB();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   real_t GetEFieldError(const VectorCoefficient & EReCoef,
                         const VectorCoefficient & EImCoef) const;

   void GetErrorEstimates(Vector & errors);

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   void DisplayAnimationToGLVis();

private:

   void computeH(const ParComplexGridFunction & b,
                 ParComplexGridFunction & h);

   void computeD(const ParComplexGridFunction & e,
                 ParComplexGridFunction & d);

   void prepareScalarVisField(const ParComplexGridFunction &u,
                              ComplexGridFunction &v);

   void prepareVectorVisField(const ParComplexGridFunction &u,
                              ComplexGridFunction &v,
                              ComplexGridFunction *vy,
                              ComplexGridFunction *vz);

   void prepareVisFields();

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   StixParams &stixParams_;
  
   // bool ownsEtaInv_;
   bool cyl_;
   bool vis_u_;

   real_t omega_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2FESpace2p_;
   L2_ParFESpace * L2VSFESpace_;
   L2_ParFESpace * L2VSFESpace2p_;
   L2_ParFESpace * L2V3FESpace_;
   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * HDivFESpace_;

   Array<HYPRE_Int> blockTrueOffsets_;

   ParBilinearForm * b1_;

   ParComplexGridFunction   e_;   // Complex electric field (HCurl)
   ParComplexGridFunction * e_v_; // Complex electric field (HCurl)
   ParGridFunction        * e_t_; // Time dependent Electric field
   ParComplexGridFunction * e_b_; // Complex parallel electric field (L2)

   CPDInputVis   inputVis_;
   CPDFieldVis   fieldVis_;
   CPDOutputVis outputVis_;
  
   ComplexScalarFieldVisObject db_v_; // Complex divergence of magnetic flux (L2)
   ComplexVectorFieldVisObject d_v_;
   ComplexScalarFieldVisObject dd_v_; // Complex divergence of electric flux (L2)
   ComplexVectorFieldVisObject j_v_;
   ComplexVectorFieldVisObject k_v_;

   ElectricEnergyDensityVisObject ue_v_;
   MagneticEnergyDensityVisObject ub_v_;

   ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
   GridFunction           * b_hat_v_; // Unit vector along B (L2^d)
   ParGridFunction        * uB_;  // Magnetic Energy density (L2)

   DielectricTensor epsReCoef_;     // Dielectric Material Coefficient
   DielectricTensor epsImCoef_;     // Dielectric Material Coefficient
   SPDDielectricTensor epsAbsCoef_;
   ConstantCoefficient muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   PWCoefficient       etaInvReCoef_; // Admittance Coefficient
   PWCoefficient       etaInvImCoef_; // Admittance Coefficient
   VectorCoefficient * kReCoef_;        // Wave Vector
   VectorCoefficient * kImCoef_;        // Wave Vector

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * abcReCoef_;       // -omega Re(eta^{-1})
   Coefficient * abcImCoef_;       // -omega Im(eta^{-1})
   Coefficient * sbcReCoef_;     //  omega Im(eta^{-1})
   Coefficient * sbcImCoef_;     // -omega Re(eta^{-1})
   Coefficient * sinkx_;         // sin(ky * y + kz * z)
   Coefficient * coskx_;         // cos(ky * y + kz * z)
   Coefficient * negsinkx_;      // -sin(ky * y + kz * z)

   MatrixCoefficient * posMassCoef_; // omega^2 Abs(epsilon)

   VectorGridFunctionCoefficient erCoef_;
   VectorGridFunctionCoefficient eiCoef_;

   CurlGridFunctionCoefficient derCoef_;
   CurlGridFunctionCoefficient deiCoef_;

   EnergyDensityCoef     uCoef_;
   ElectricEnergyDensityCoef uECoef_;
   MagneticEnergyDensityCoef uBCoef_;
   PoyntingVectorReCoef SrCoef_;
   PoyntingVectorImCoef SiCoef_;

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_bdr_marker_;

   const Array<ComplexVectorCoefficientByAttr*> & dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   const Array<ComplexVectorCoefficientByAttr*> & nbcs_; // Surface current BCs
   Array<ComplexVectorCoefficientByAttr*> nkbcs_; // Neumann BCs (-i*omega*K)

   const Array<ComplexCoefficientByAttr*> & abcs_; // Sommerfeld (absorbing) BCs
  
   const Array<AttributeArrays*> & axis_; // Cylindrical Axis
   Array<int> axis_tdofs_;

   Maxwell2ndE     maxwell_;
   CurrentSourceE  current_;
   FaradaysLaw     faraday_;
   GausssLaw       divB_;
   Displacement    displacement_;
   GausssLaw       divD_;

   Array<VectorCoefficient*> vCoefs_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER
