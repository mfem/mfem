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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER

#include "../../general/optparser.hpp"
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

class ElectricFieldFromE
{
protected:
   ElectricFieldFromE(VectorCoefficient &Er,
                      VectorCoefficient &Ei)
      : ErCoef_(Er), EiCoef_(Ei), Er_(3), Ei_(3)
   {}

   virtual void EvalE(ElementTransformation &T,
                      const IntegrationPoint &ip);

   VectorCoefficient &ErCoef_;
   VectorCoefficient &EiCoef_;

   mutable Vector Er_;
   mutable Vector Ei_;
};

class MagneticFluxFromCurlE
{
protected:
   MagneticFluxFromCurlE(real_t omega,
                         VectorCoefficient &dEr,
                         VectorCoefficient &dEi)
      : omega_(omega), dErCoef_(dEr), dEiCoef_(dEi), Br_(3), Bi_(3)
   {}

   virtual void EvalB(ElementTransformation &T,
                      const IntegrationPoint &ip);

   real_t omega_;

   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;

   mutable Vector Br_;
   mutable Vector Bi_;
};

class ElectricFluxFromE : public ElectricFieldFromE
{
protected:
   ElectricFluxFromE(VectorCoefficient &Er,
                     VectorCoefficient &Ei,
                     MatrixCoefficient &epsr,
                     MatrixCoefficient &epsi)
      : ElectricFieldFromE(Er, Ei),
        epsrCoef_(epsr), epsiCoef_(epsi), Dr_(3), Di_(3), eps_r_(3), eps_i_(3)
   {}

   virtual void EvalD(ElementTransformation &T,
                      const IntegrationPoint &ip);

   MatrixCoefficient &epsrCoef_;
   MatrixCoefficient &epsiCoef_;

   mutable Vector Dr_;
   mutable Vector Di_;
   mutable DenseMatrix eps_r_;
   mutable DenseMatrix eps_i_;
};

class MagneticFieldFromCurlE : public MagneticFluxFromCurlE
{
protected:
   MagneticFieldFromCurlE(real_t omega,
                          VectorCoefficient &dEr,
                          VectorCoefficient &dEi,
                          Coefficient &muInv)
      : MagneticFluxFromCurlE(omega, dEr, dEi),
        muInvCoef_(muInv), Hr_(3), Hi_(3)
   {}

   virtual void EvalH(ElementTransformation &T,
                      const IntegrationPoint &ip);

   Coefficient &muInvCoef_;

   mutable Vector Hr_;
   mutable Vector Hi_;
};

class ElectricEnergyDensityReCoef : public Coefficient,
   public ElectricFluxFromE
{
public:
   ElectricEnergyDensityReCoef(VectorCoefficient &Er,
                               VectorCoefficient &Ei,
                               MatrixCoefficient &epsr,
                               MatrixCoefficient &epsi)
      : ElectricFluxFromE(Er, Ei, epsr, epsi) {}


   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class ElectricEnergyDensityImCoef : public Coefficient,
   public ElectricFluxFromE
{
public:
   ElectricEnergyDensityImCoef(VectorCoefficient &Er,
                               VectorCoefficient &Ei,
                               MatrixCoefficient &epsr,
                               MatrixCoefficient &epsi)
      : ElectricFluxFromE(Er, Ei, epsr, epsi) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class MagneticEnergyDensityReCoef : public Coefficient,
   public MagneticFieldFromCurlE
{
public:
   MagneticEnergyDensityReCoef(real_t omega,
                               VectorCoefficient &dEr,
                               VectorCoefficient &dEi,
                               Coefficient &muInv)
      : MagneticFieldFromCurlE(omega, dEr, dEi, muInv) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class MagneticEnergyDensityImCoef : public ConstantCoefficient
{
public:
   MagneticEnergyDensityImCoef() : ConstantCoefficient(0.0) {}
};

class EnergyDensityReCoef : public Coefficient,
   public ElectricFluxFromE,
   public MagneticFieldFromCurlE
{
public:
   EnergyDensityReCoef(real_t omega,
                       VectorCoefficient &Er, VectorCoefficient &Ei,
                       VectorCoefficient &dEr, VectorCoefficient &dEi,
                       MatrixCoefficient &epsr, MatrixCoefficient &epsi,
                       Coefficient &muInv)
      : ElectricFluxFromE(Er, Ei, epsr, epsi),
        MagneticFieldFromCurlE(omega, dEr, dEi, muInv)
   {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

class EnergyDensityImCoef : public ElectricEnergyDensityImCoef
{
public:
   EnergyDensityImCoef(real_t omega,
                       VectorCoefficient &Er, VectorCoefficient &Ei,
                       VectorCoefficient &dEr, VectorCoefficient &dEi,
                       MatrixCoefficient &epsr, MatrixCoefficient &epsi,
                       Coefficient &muInv)
      : ElectricEnergyDensityImCoef(Er, Ei, epsr, epsi)
   {}
};

class PoyntingVectorReCoef : public VectorCoefficient,
   public ElectricFieldFromE,
   public MagneticFieldFromCurlE
{
public:
   PoyntingVectorReCoef(real_t omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv)
      : VectorCoefficient(3),
        ElectricFieldFromE(Er, Ei),
        MagneticFieldFromCurlE(omega, dEr, dEi, muInv)
   {}

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class PoyntingVectorImCoef : public VectorCoefficient,
   public ElectricFieldFromE,
   public MagneticFieldFromCurlE
{
public:
   PoyntingVectorImCoef(real_t omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv)
      : VectorCoefficient(3),
        ElectricFieldFromE(Er, Ei),
        MagneticFieldFromCurlE(omega, dEr, dEi, muInv)
   {}

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class MinkowskiMomentumDensityReCoef : public VectorCoefficient,
   public ElectricFluxFromE,
   public MagneticFluxFromCurlE
{
public:
   MinkowskiMomentumDensityReCoef(real_t omega,
                                  VectorCoefficient &Er,
                                  VectorCoefficient &Ei,
                                  VectorCoefficient &dEr,
                                  VectorCoefficient &dEi,
                                  MatrixCoefficient &epsr,
                                  MatrixCoefficient &epsi)
      : VectorCoefficient(3),
        ElectricFluxFromE(Er, Ei, epsr, epsi),
        MagneticFluxFromCurlE(omega, dEr, dEi)
   {}

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class MinkowskiMomentumDensityImCoef : public VectorCoefficient,
   public ElectricFluxFromE,
   public MagneticFluxFromCurlE
{
public:
   MinkowskiMomentumDensityImCoef(real_t omega,
                                  VectorCoefficient &Er,
                                  VectorCoefficient &Ei,
                                  VectorCoefficient &dEr,
                                  VectorCoefficient &dEi,
                                  MatrixCoefficient &epsr,
                                  MatrixCoefficient &epsi)
      : VectorCoefficient(3),
        ElectricFluxFromE(Er, Ei, epsr, epsi),
        MagneticFluxFromCurlE(omega, dEr, dEi)
   {}

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class EAlongBackgroundBCoef : public Coefficient,
   public ElectricFieldFromE
{
private:
   VectorCoefficient &BCoef_;

   mutable Vector B_;

public:
   EAlongBackgroundBCoef(VectorCoefficient &Er,
                         VectorCoefficient &Ei,
                         VectorCoefficient &B)
      : ElectricFieldFromE(Er, Ei), BCoef_(B), B_(3) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
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

      mutable Vector kr_;
      mutable Vector ki_;
      mutable DenseMatrix M_;

      static void kmk(real_t a,
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

      static void kmk(real_t a,
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
           a_(a), kr_(3), ki_(3)
      { kr_ = 0.0; ki_ = 0.0; }

      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     MatrixCoefficient *MCoef,
                     bool realPart, real_t a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           realPart_(realPart),
           a_(a), kr_(3), ki_(3), M_(3)
      { kr_ = 0.0; ki_ = 0.0; M_ = 0.0; }

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
         if (krCoef_) { krCoef_->Eval(kr_, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki_, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MCoef_) { MCoef_->Eval(M_, T, ip); }

         if (realPart_)
         {
            if (!MCoef_)
            {
               if (krCoef_) { kmk(1.0, kr_, m, kr_, M); }
               if (kiCoef_) { kmk(-1.0, ki_, m, ki_, M); }
            }
            else
            {
               if (krCoef_) { kmk(1.0, kr_, M_, kr_, M); }
               if (kiCoef_) { kmk(-1.0, ki_, M_, ki_, M); }
            }
         }
         else
         {
            if (!MCoef_)
            {
               if (krCoef_ && kiCoef_) { kmk(1.0, kr_, m, ki_, M); }
               if (kiCoef_ && krCoef_) { kmk(1.0, ki_, m, kr_, M); }
            }
            else
            {
               if (krCoef_ && kiCoef_) { kmk(1.0, kr_, M_, ki_, M); }
               if (kiCoef_ && krCoef_) { kmk(1.0, ki_, M_, kr_, M); }
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

      mutable Vector k_;
      mutable DenseMatrix M_;

   public:
      CrossCoefficient(VectorCoefficient *kCoef,
                       Coefficient *mCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(mCoef),
           MCoef_(NULL),
           a_(a), km_(true), k_(3)
      { k_ = 0.0; }

      CrossCoefficient(VectorCoefficient *kCoef,
                       MatrixCoefficient *MCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           a_(a), km_(true), k_(3), M_(3)
      { k_ = 0.0; M_ = 0.0; }

      CrossCoefficient(MatrixCoefficient *MCoef,
                       VectorCoefficient *kCoef,
                       real_t a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(NULL),
           MCoef_(MCoef),
           a_(a), km_(false), k_(3), M_(3)
      { k_ = 0.0; M_ = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;

         real_t m = 0.0;
         if (kCoef_) { kCoef_->Eval(k_, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MCoef_) { MCoef_->Eval(M_, T, ip); }

         if (MCoef_ == NULL)
         {
            M(2,1) = a_ * m * k_(0);
            M(0,2) = a_ * m * k_(1);
            M(1,0) = a_ * m * k_(2);

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
                     M(i,j) += a_ * k_(i1) * M_(i2,j);
                     M(i,j) -= a_ * k_(i2) * M_(i1,j);
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
                     M(i,j) += a_ * M_(i,j1) * k_(j2);
                     M(i,j) -= a_ * M_(i,j2) * k_(j1);
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

   real_t GetBFieldError(const VectorCoefficient & BReCoef,
                         const VectorCoefficient & BImCoef) const;

};

class GausssLaw
{
private:

   ParFiniteElementSpace & HDivFESpace_; // Space for source field
   ParComplexGridFunction           df_; // Complex divergence (L2)

   ParDiscreteDivOperator div_;

   ParDiscreteLinearOperator * kReDot_;
   ParDiscreteLinearOperator * kImDot_;

   bool assembled_;

public:
   GausssLaw(ParFiniteElementSpace & HDivFESpace,
             ParFiniteElementSpace & L2FESpace,
             VectorCoefficient * kReCoef,
             VectorCoefficient * kImCoef);

   ParComplexGridFunction & GetDivergence() { return df_; }

   void Update();
   void Assemble();
   void ComputeDiv(const ParComplexGridFunction &f) { ComputeDiv(f, df_); }
   void ComputeDiv(const ParComplexGridFunction &f,
                   ParComplexGridFunction &df);
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
                                  std::shared_ptr<L2_ParFESpace> sfes,
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
                                  std::shared_ptr<L2_ParFESpace> sfes,
                                  bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        MatrixCoefficient &epsr,
                        MatrixCoefficient &epsi);
};

class MagneticEnergyDensityVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   MagneticEnergyDensityVisObject(const std::string & field_name,
                                  std::shared_ptr<L2_ParFESpace> sfes,
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
                          std::shared_ptr<L2_ParFESpace> sfes,
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
                           std::shared_ptr<L2_ParFESpace> vfes);

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
                                     std::shared_ptr<L2_ParFESpace> vfes);

   void PrepareVisField(const ParComplexGridFunction &e,
                        real_t omega,
                        MatrixCoefficient & epsReCoef,
                        MatrixCoefficient & epsImCoef);
};

class EAlongBackgroundBVisObject : public ScalarFieldVisObject
{
private:

   using ScalarFieldVisObject::PrepareVisField;

public:
   EAlongBackgroundBVisObject(const std::string & field_name,
                              std::shared_ptr<L2_ParFESpace> sfes,
                              bool cyl, bool pseudo);

   void PrepareVisField(const ParComplexGridFunction &e,
                        VectorCoefficient &BCoef);
};

class TensorCompVisObject : public ComplexScalarFieldVisObject
{
private:

   using ComplexScalarFieldVisObject::PrepareVisField;

public:
   TensorCompVisObject(const std::string & field_name,
                       std::shared_ptr<L2_ParFESpace> sfes,
                       bool cyl, bool pseudo);

   void PrepareVisField(MatrixCoefficient &mr,
                        MatrixCoefficient &mi,
                        int i, int j);
};

class CPDVisBase
{
protected:
   unsigned int vis_mask_;

   CPDVisBase(unsigned int mask = 0) : vis_mask_(mask) {}

   static void set_bool_flags(unsigned int num_flags,
                              unsigned int active_flags_mask,
                              Array<bool> &flags);

public:

   static unsigned int GetNumVisFields();
   static unsigned int GetDefaultVisFlag();

   virtual void SetOptions(const Array<bool> &opts);

   inline bool CheckVisFlag(unsigned int flag) const
   { return (vis_mask_ >> flag) & 1; }

   inline unsigned int SetVisFlag(unsigned int flag)
   { return (vis_mask_ |= 1 << flag); }

   inline unsigned int ClearVisFlag(unsigned int flag)
   { return vis_mask_ &= ~(1 << flag); }

   inline unsigned int FlipVisFlag(unsigned int flag)
   { return vis_mask_ ^= 1 << flag; }

   virtual bool RequireMagneticFlux() const { return false; }
   virtual bool RequireElectricFlux() const { return false; }
   virtual bool RequireDivergence() const { return false; }
};

class CPDInputVis : public CPDVisBase
{
public:
   enum VisField {BACKGROUND_B = 0,
                  VOLUMETRIC_CURRENT,
                  ION_DENSITIES,
                  ION_TEMPERATURES,
                  ELECTRON_DENSITY,
                  ELECTRON_TEMPERATURE,
                  STIX_S,
                  STIX_D,
                  STIX_P,
                  STIX_L,
                  STIX_R,
                  STIX_INVSP,
                  WAVELENGTH_L,
                  WAVELENGTH_R,
                  WAVELENGTH_O,
                  WAVELENGTH_X,
                  SKIN_DEPTH_L,
                  SKIN_DEPTH_R,
                  SKIN_DEPTH_O,
                  SKIN_DEPTH_X,
                  NUM_VIS_FIELDS
                 };

private:
   // Visualize only BACKGROUND_B by default
   static const unsigned int default_vis_mask_ = 1;

   // Strings needed for option parser
   static const std::string opt_str_[];

   // Strings constructed for option parser arguments
   static std::array<std::string, NUM_VIS_FIELDS> optt_;
   static std::array<std::string, NUM_VIS_FIELDS> optlt_;
   static std::array<std::string, NUM_VIS_FIELDS> optf_;
   static std::array<std::string, NUM_VIS_FIELDS> optlf_;
   static std::array<std::string, NUM_VIS_FIELDS> optd_;

public:
   CPDInputVis(StixParams &stixParams,
               std::shared_ptr<L2_ParFESpace> l2_sfes,
               std::shared_ptr<L2_ParFESpace> l2_vfes,
               unsigned int vis_flag = default_vis_mask_, bool cyl = false);

   ~CPDInputVis();

   static unsigned int GetNumVisFields() { return NUM_VIS_FIELDS; }
   static unsigned int GetDefaultVisMask() { return default_vis_mask_; }
   static Array<bool> GetDefaultVisFlags()
   {
      Array<bool> flags;
      CPDVisBase::set_bool_flags(GetNumVisFields(), GetDefaultVisMask(), flags);
      return flags;
   }
   static void AddOptions(OptionsParser &args, Array<bool> &opts);

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields(const ParComplexGridFunction & j,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef);

   void DisplayToGLVis();

   void Update();

private:

   std::shared_ptr<L2_ParFESpace> sfes_;
   std::shared_ptr<L2_ParFESpace> vfes_;

   // Background Magnetic Field
   VectorCoefficient &BCoef_;

   VectorFieldVisObject B_;

   // Volumetric Current Density (J)
   ComplexVectorFieldVisObject J_;

   // Ion Species States
   int numIonSpec_;
   Array<ComponentCoefficient*> ionDensityCoefs_;
   Array<ComponentCoefficient*> ionTempCoefs_;

   Array<ScalarFieldVisObject*> ionDensities_;
   Array<ScalarFieldVisObject*> ionTemps_;

   // electron state
   ComponentCoefficient electronDensityCoef_;
   ComponentCoefficient electronTempCoef_;

   ScalarFieldVisObject electronDensity_;
   ScalarFieldVisObject electronTemp_;

   // Stix Dielectric tensor Coefficients
   StixSCoef stixSReCoef_;
   StixSCoef stixSImCoef_;
   StixDCoef stixDReCoef_;
   StixDCoef stixDImCoef_;
   StixPCoef stixPReCoef_;
   StixPCoef stixPImCoef_;
   StixLCoef stixLReCoef_;
   StixLCoef stixLImCoef_;
   StixRCoef stixRReCoef_;
   StixRCoef stixRImCoef_;
   StixInvSPCoef stixInvSPReCoef_;
   StixInvSPCoef stixInvSPImCoef_;

   ComplexScalarFieldVisObject stixS_;
   ComplexScalarFieldVisObject stixD_;
   ComplexScalarFieldVisObject stixP_;
   ComplexScalarFieldVisObject stixL_;
   ComplexScalarFieldVisObject stixR_;
   ComplexScalarFieldVisObject stixInvSP_;

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

class CPDFieldVis : public CPDVisBase
{
public:
   enum VisField {ELECTRIC_FIELD = 0,
                  MAGNETIC_FLUX,
                  ELECTRIC_FLUX,
                  DIV_MAGNETIC_FLUX,
                  DIV_ELECTRIC_FLUX,
                  NUM_VIS_FIELDS
                 };

private:
   // Visualize both E and B fields by default
   static const unsigned int default_vis_mask_ = 3;

   // Strings needed for option parser
   static const std::string opt_str_[];

   // Strings constructed for option parser arguments
   static std::array<std::string, NUM_VIS_FIELDS> optt_;
   static std::array<std::string, NUM_VIS_FIELDS> optlt_;
   static std::array<std::string, NUM_VIS_FIELDS> optf_;
   static std::array<std::string, NUM_VIS_FIELDS> optlf_;
   static std::array<std::string, NUM_VIS_FIELDS> optd_;

public:
   CPDFieldVis(StixParams &stixParams,
               std::shared_ptr<L2_ParFESpace> l2_sfes,
               std::shared_ptr<L2_ParFESpace> l2_vfes,
               const std::string hcurl_field_name,
               const std::string hdiv_field_name,
               unsigned int vis_flag = default_vis_mask_);

   static unsigned int GetNumVisFields() { return NUM_VIS_FIELDS; }
   static unsigned int GetDefaultVisMask() { return default_vis_mask_; }
   static Array<bool> GetDefaultVisFlags()
   {
      Array<bool> flags;
      CPDVisBase::set_bool_flags(GetNumVisFields(), GetDefaultVisMask(), flags);
      return flags;
   }
   static void AddOptions(OptionsParser &args, Array<bool> &opts);

   bool RequireMagneticFlux() const override
   { return CheckVisFlag(MAGNETIC_FLUX) || CheckVisFlag(DIV_MAGNETIC_FLUX); }

   bool RequireElectricFlux() const override
   { return CheckVisFlag(ELECTRIC_FLUX) || CheckVisFlag(DIV_ELECTRIC_FLUX); }

   bool RequireDivergence() const override
   {
      return CheckVisFlag(DIV_MAGNETIC_FLUX) ||
             CheckVisFlag(DIV_ELECTRIC_FLUX);
   }

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields(const ParComplexGridFunction & e,
                         const ParComplexGridFunction & b,
                         const ParComplexGridFunction & db,
                         const ParComplexGridFunction & dd,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef);

   void DisplayToGLVis();

   void Update();

private:

   std::shared_ptr<L2_ParFESpace> sfes_;
   std::shared_ptr<L2_ParFESpace> vfes_;

   ComplexVectorFieldVisObject HCurlField_;
   ComplexVectorFieldVisObject HDivField_;
   ComplexScalarFieldVisObject DivBField_;
   ComplexScalarFieldVisObject DivDField_;
};

class CPDOutputVis : public CPDVisBase
{
public:
   enum VisField {ENERGY_DENSITY = 0,
                  POYNTING_FLUX,
                  MOMENTUM_DENSITY,
                  ED_ENERGY_DENSITY,
                  BH_ENERGY_DENSITY,
                  E_BACKGROUND_B,
                  NUM_VIS_FIELDS
                 };

private:
   // Visualize nothing by default
   static const unsigned int default_vis_mask_ = 0;

   // Strings needed for option parser
   static const std::string opt_str_[];

   // Strings constructed for option parser arguments
   static std::array<std::string, NUM_VIS_FIELDS> optt_;
   static std::array<std::string, NUM_VIS_FIELDS> optlt_;
   static std::array<std::string, NUM_VIS_FIELDS> optf_;
   static std::array<std::string, NUM_VIS_FIELDS> optlf_;
   static std::array<std::string, NUM_VIS_FIELDS> optd_;

public:
   CPDOutputVis(StixParams &stixParams,
                std::shared_ptr<L2_ParFESpace> l2_sfes,
                std::shared_ptr<L2_ParFESpace> l2_vfes,
                real_t omega,
                unsigned int vis_flag = default_vis_mask_,
                bool cyl = false);

   static unsigned int GetNumVisFields() { return NUM_VIS_FIELDS; }
   static unsigned int GetDefaultVisMask() { return default_vis_mask_; }
   static Array<bool> GetDefaultVisFlags()
   {
      Array<bool> flags;
      CPDVisBase::set_bool_flags(GetNumVisFields(), GetDefaultVisMask(), flags);
      return flags;
   }
   static void AddOptions(OptionsParser &args, Array<bool> &opts);

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields(const ParComplexGridFunction & e);

   void DisplayToGLVis();

   void Update();

private:

   std::shared_ptr<L2_ParFESpace> sfes_;
   std::shared_ptr<L2_ParFESpace> vfes_;

   real_t omega_;

   VectorCoefficient &BCoef_; // Background B field

   DielectricTensor epsReCoef_;    // Dielectric Material Coefficient
   DielectricTensor epsImCoef_;    // Dielectric Material Coefficient
   ConstantCoefficient muInvCoef_; // Dia/Paramagnetic Material Coefficient

   EnergyDensityVisObject energyDensity_;
   PoyntingVectorVisObject poyntingFlux_;
   MinkowskiMomentumDensityVisObject momentumDensity_;

   ElectricEnergyDensityVisObject edEnergyDensity_;
   MagneticEnergyDensityVisObject bhEnergyDensity_;

   EAlongBackgroundBVisObject ealongb_;
};

class CPDFieldAnim : public CPDVisBase
{
public:
   enum VisField {ELECTRIC_FIELD_ANIM = 0,
                  MAGNETIC_FLUX_ANIM,
                  NUM_VIS_FIELDS
                 };

private:
   // Visualize both E and B fields by default
   static const unsigned int default_vis_mask_ = 1;

   // Strings needed for option parser
   static const std::string opt_str_[];

   // Strings constructed for option parser arguments
   static std::array<std::string, NUM_VIS_FIELDS> optt_;
   static std::array<std::string, NUM_VIS_FIELDS> optlt_;
   static std::array<std::string, NUM_VIS_FIELDS> optf_;
   static std::array<std::string, NUM_VIS_FIELDS> optlf_;
   static std::array<std::string, NUM_VIS_FIELDS> optd_;

public:
   CPDFieldAnim(StixParams &stixParams,
                std::shared_ptr<L2_ParFESpace> l2_sfes,
                std::shared_ptr<L2_ParFESpace> l2_vfes,
                const std::string hcurl_field_name,
                const std::string hdiv_field_name,
                unsigned int vis_flag = default_vis_mask_);

   static unsigned int GetNumVisFields() { return NUM_VIS_FIELDS; }
   static unsigned int GetDefaultVisMask() { return default_vis_mask_; }
   static Array<bool> GetDefaultVisFlags()
   {
      Array<bool> flags;
      CPDVisBase::set_bool_flags(GetNumVisFields(), GetDefaultVisMask(), flags);
      return flags;
   }
   static void AddOptions(OptionsParser &args, Array<bool> &opts);

   bool RequireMagneticFlux() const override
   { return CheckVisFlag(MAGNETIC_FLUX_ANIM); }

   std::shared_ptr<L2_ParFESpace> GetScalarFES() const { return sfes_; }
   std::shared_ptr<L2_ParFESpace> GetVectorFES() const { return vfes_; }

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void PrepareVisFields(const ParComplexGridFunction & e,
                         const ParComplexGridFunction & b,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef);

   void DisplayToGLVis();

   void Update();

private:

   std::shared_ptr<L2_ParFESpace> sfes_;
   std::shared_ptr<L2_ParFESpace> vfes_;

   ComplexVectorFieldAnimObject HCurlFieldAnim_;
   ComplexVectorFieldAnimObject HDivFieldAnim_;
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
               VectorCoefficient * kReCoef,
               VectorCoefficient * kImCoef,
               StixBCs & stixBCs,
               bool cyl = false);
   ~CPDSolverEB();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   ParComplexGridFunction & GetElectricField() { return e_; }
   ParComplexGridFunction & GetMagneticFlux()
   { return faraday_.GetMagneticFlux(); }

   bool RequireMagneticFlux() const
   {
      return
         inputVis_.RequireMagneticFlux() ||
         fieldVis_.RequireMagneticFlux() ||
         outputVis_.RequireMagneticFlux() ||
         fieldAnim_.RequireMagneticFlux();
   }

   bool RequireElectricFlux() const
   {
      return
         inputVis_.RequireElectricFlux() ||
         fieldVis_.RequireElectricFlux() ||
         outputVis_.RequireElectricFlux() ||
         fieldAnim_.RequireElectricFlux();
   }

   bool RequireDivergence() const
   {
      return
         inputVis_.RequireDivergence() ||
         fieldVis_.RequireDivergence() ||
         outputVis_.RequireDivergence() ||
         fieldAnim_.RequireDivergence();
   }

   real_t GetEFieldError(const VectorCoefficient & EReCoef,
                         const VectorCoefficient & EImCoef) const;

   real_t GetBFieldError(const VectorCoefficient & BReCoef,
                         const VectorCoefficient & BImCoef) const;

   void GetErrorEstimates(Vector & errors);

   const CPDInputVis & GetInputVis() const { return inputVis_; }
   const CPDFieldVis & GetFieldVis() const { return fieldVis_; }
   const CPDOutputVis & GetOutputVis() const { return outputVis_; }
   const CPDFieldAnim & GetFieldAnim() const { return fieldAnim_; }

   CPDInputVis & GetInputVis() { return inputVis_; }
   CPDFieldVis & GetFieldVis() { return fieldVis_; }
   CPDOutputVis & GetOutputVis() { return outputVis_; }
   CPDFieldAnim & GetFieldAnim() { return fieldAnim_; }

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void DisplayToGLVis();

   std::shared_ptr<L2_ParFESpace> GetScalarVisFES() const
   { return inputVis_.GetScalarFES(); }
   std::shared_ptr<L2_ParFESpace> GetVectorVisFES() const
   { return L2VFESpace_; }

private:

   void prepareVisFields();

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   bool cyl_;

   real_t omega_;

   ParMesh * pmesh_;

   std::shared_ptr<L2_ParFESpace> L2FESpace_;
   std::shared_ptr<L2_ParFESpace> L2FESpace2p_;
   std::shared_ptr<L2_ParFESpace> L2VFESpace_;
   std::shared_ptr<ParFiniteElementSpace> HCurlFESpace_;
   std::shared_ptr<ParFiniteElementSpace> HDivFESpace_;

   Array<HYPRE_Int> blockTrueOffsets_;

   ParBilinearForm * b1_;

   ParComplexGridFunction    e_;  // Complex electric field (HCurl)
   ParComplexGridFunction   db_;  // Complex divergence of magnetic flux (L2)
   ParComplexGridFunction   dd_;  // Complex divergence of electric flux (L2)

   CPDInputVis   inputVis_;
   CPDFieldVis   fieldVis_;
   CPDOutputVis outputVis_;
   CPDFieldAnim fieldAnim_;

   DielectricTensor    epsReCoef_;    // Dielectric Material Coefficient
   DielectricTensor    epsImCoef_;    // Dielectric Material Coefficient
   SPDDielectricTensor epsAbsCoef_;   // Used in preconditioner
   ConstantCoefficient muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   PWCoefficient       etaInvReCoef_; // Admittance Coefficient
   PWCoefficient       etaInvImCoef_; // Admittance Coefficient
   VectorCoefficient * betaReCoef_;   // Phase Shift Vector
   VectorCoefficient * betaImCoef_;   // Phase Shift Vector

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * abcReCoef_;     // -omega Re(eta^{-1})
   Coefficient * abcImCoef_;     // -omega Im(eta^{-1})

   MatrixCoefficient * posMassCoef_; // omega^2 Abs(epsilon)

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_bdr_marker_;

   const Array<ComplexVectorCoefficientByAttr*> & dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;

   const Array<ComplexCoefficientByAttr*> & abcs_; // Sommerfeld (absorbing) BCs

   const Array<AttributeArrays*> & axis_; // Cylindrical Axis
   Array<int> axis_tdofs_;

   Maxwell2ndE     maxwell_;
   CurrentSourceE  current_;
   FaradaysLaw     faraday_;
   Displacement    displacement_;
   GausssLaw       gauss_;

   VisItDataCollection * visit_dc_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_EB_SOLVER
