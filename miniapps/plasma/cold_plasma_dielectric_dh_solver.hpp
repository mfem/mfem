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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER

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

using common::ND_FESpace;
using common::L2_FESpace;
using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::L2_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;

namespace plasma
{
/*
// Solver options
struct SolverOptions
{
   int maxIter;
   int kDim;
   int printLvl;
   double relTol;

   // Euclid Options
   int euLvl;
};
*/
/*
struct ComplexCoefficientByAttr : public AttributeArrays
{
   Coefficient * real;
   Coefficient * imag;
};

struct ComplexVectorCoefficientByAttr
{
   Array<int> attr;
   Array<int> attr_marker;
   VectorCoefficient * real;
   VectorCoefficient * imag;
};

class ElectricEnergyDensityCoef : public Coefficient
{
public:
   ElectricEnergyDensityCoef(VectorCoefficient &Er, VectorCoefficient &Ei,
                             MatrixCoefficient &epsr, MatrixCoefficient &epsi);

   double Eval(ElementTransformation &T,
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

class MagneticEnergyDensityCoef : public Coefficient
{
public:
   MagneticEnergyDensityCoef(double omega,
                             VectorCoefficient &dEr, VectorCoefficient &dEi,
                             Coefficient &muInv);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   double omega_;

   VectorCoefficient &dErCoef_;
   VectorCoefficient &dEiCoef_;
   Coefficient &muInvCoef_;

   mutable Vector Br_;
   mutable Vector Bi_;
};

class EnergyDensityCoef : public Coefficient
{
public:
   EnergyDensityCoef(double omega,
                     VectorCoefficient &Er, VectorCoefficient &Ei,
                     VectorCoefficient &dEr, VectorCoefficient &dEi,
                     MatrixCoefficient &epsr, MatrixCoefficient &epsi,
                     Coefficient &muInv);

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip);

private:
   double omega_;

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
   PoyntingVectorReCoef(double omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   double omega_;

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
   PoyntingVectorImCoef(double omega,
                        VectorCoefficient &Er, VectorCoefficient &Ei,
                        VectorCoefficient &dEr, VectorCoefficient &dEi,
                        Coefficient &muInv);

   void Eval(Vector &S, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   double omega_;

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
*/

class nxGradIntegrator : public BilinearFormIntegrator
{
private:

   Coefficient *Q;

#ifndef MFEM_THREAD_SAFE
   Vector nor, nxj;
   DenseMatrix test_shape;
   DenseMatrix trial_dshape;
#endif

public:
   nxGradIntegrator() : Q(NULL) {}
   nxGradIntegrator(Coefficient &q) : Q(&q) {}

   int GetIntegrationOrder(const FiniteElement & trial_fe,
                           const FiniteElement & test_fe,
                           ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat);
};

class nxkIntegrator : public BilinearFormIntegrator
{
private:

   Coefficient *Q;
   VectorCoefficient *K;

#ifndef MFEM_THREAD_SAFE
   Vector nor, nxj, k;
   DenseMatrix test_shape;
   Vector trial_shape;
#endif

public:
   nxkIntegrator(VectorCoefficient &k) : Q(NULL), K(&k) {}
   nxkIntegrator(VectorCoefficient &k, Coefficient &q) : Q(&q), K(&k) {}

   int GetIntegrationOrder(const FiniteElement & trial_fe,
                           const FiniteElement & test_fe,
                           ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat);
};

class nDotCurlIntegrator : public BilinearFormIntegrator
{
private:

   Coefficient *Q;

#ifndef MFEM_THREAD_SAFE
   Vector nor, nDotCurl;
   Vector test_shape;
   DenseMatrix trial_dshape;
#endif

public:
   nDotCurlIntegrator() : Q(NULL) {}
   nDotCurlIntegrator(Coefficient &q) : Q(&q) {}

   int GetIntegrationOrder(const FiniteElement & trial_fe,
                           const FiniteElement & test_fe,
                           ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat);
};

class zkxIntegrator : public BilinearFormIntegrator
{
private:

   Coefficient *Z;
   VectorCoefficient *K;
   double a;

#ifndef MFEM_THREAD_SAFE
   Vector nor, nxj, k;
   Vector test_shape;
   DenseMatrix trial_shape;
#endif

public:
   zkxIntegrator(Coefficient &z, VectorCoefficient &k, double _a = 1.0)
      : Z(&z), K(&k), a(_a) {}

   int GetIntegrationOrder(const FiniteElement & trial_fe,
                           const FiniteElement & test_fe,
                           ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat);
};

class Maxwell2ndH : public ParSesquilinearForm
{
private:

   class ScalarMassCoefficient : public Coefficient
   {
   private:
      double omegaSqr_;
      Coefficient & mCoef_;

   public:

      ScalarMassCoefficient(double omega, Coefficient & mCoef)
         : omegaSqr_(omega * omega),
           mCoef_(mCoef)
      {}

      virtual double Eval(ElementTransformation &T,
                          const IntegrationPoint &ip)
      {
         return -omegaSqr_  * mCoef_.Eval(T, ip);
      }
   };

   class MatrixMassCoefficient : public MatrixCoefficient
   {
   private:
      double omegaSqr_;
      MatrixCoefficient & mCoef_;

   public:

      MatrixMassCoefficient(double omega, MatrixCoefficient & mCoef)
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
      MatrixCoefficient * MrCoef_;
      MatrixCoefficient * MiCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr_;
      mutable Vector ki_;
      mutable DenseMatrix Mr_;
      mutable DenseMatrix Mi_;

      static void kmk(double a,
                      const Vector & kl, double m, const Vector &kr,
                      DenseMatrix & M)
      {
         double kk = kl * kr;
         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               M(i,j) += a * m * kl(j) * kr(i);
            }
            M(i,i) -= a * m * kk;
         }
      }

      static void kmk(double a,
                      const Vector & kl,
                      const DenseMatrix & m,
                      const Vector &kr,
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
                     bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(mCoef),
           MrCoef_(NULL),
           MiCoef_(NULL),
           realPart_(realPart),
           a_(a), kr_(3), ki_(3)
      { kr_ = 0.0; ki_ = 0.0; }

      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     MatrixCoefficient *MrCoef, MatrixCoefficient *MiCoef,
                     bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(NULL),
           MrCoef_(MrCoef),
           MiCoef_(MiCoef),
           realPart_(realPart),
           a_(a), kr_(3), ki_(3), Mr_(3), Mi_(3)
      { kr_ = 0.0; ki_ = 0.0; Mr_ = 0.0; Mi_ = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             (mCoef_ == NULL && MrCoef_ == NULL && MiCoef_ == NULL))
         {
            return;
         }

         double m = 0.0;
         if (krCoef_) { krCoef_->Eval(kr_, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki_, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MrCoef_) { MrCoef_->Eval(Mr_, T, ip); }
         if (MiCoef_) { MiCoef_->Eval(Mi_, T, ip); }

         if (realPart_)
         {
            if (mCoef_)
            {
               if (krCoef_) { kmk(1.0, kr_, m, kr_, M); }
               if (kiCoef_) { kmk(-1.0, ki_, m, ki_, M); }
            }
            else
            {
               if (MrCoef_)
               {
                  if (krCoef_) { kmk(1.0, kr_, Mr_, kr_, M); }
                  if (kiCoef_) { kmk(-1.0, ki_, Mr_, ki_, M); }
               }
               if (MiCoef_)
               {
                  if (krCoef_ && kiCoef_) { kmk(-1.0, kr_, Mi_, ki_, M); }
                  if (kiCoef_ && krCoef_) { kmk(-1.0, ki_, Mi_, kr_, M); }
               }
            }
         }
         else
         {
            if (mCoef_)
            {
               if (krCoef_ && kiCoef_) { kmk(1.0, kr_, m, ki_, M); }
               if (kiCoef_ && krCoef_) { kmk(1.0, ki_, m, kr_, M); }
            }
            else
            {
               if (MrCoef_)
               {
                  if (krCoef_ && kiCoef_) { kmk(1.0, kr_, Mr_, ki_, M); }
                  if (kiCoef_ && krCoef_) { kmk(1.0, ki_, Mr_, kr_, M); }
               }
               if (MiCoef_)
               {
                  if (krCoef_) { kmk(1.0, kr_, Mi_, kr_, M); }
                  if (kiCoef_) { kmk(-1.0, ki_, Mi_, ki_, M); }
               }
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class CrossCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      Coefficient       * mCoef_;
      MatrixCoefficient * MrCoef_;
      MatrixCoefficient * MiCoef_;

      double a_;

      bool realPart_;
      bool km_;

      mutable Vector kr_;
      mutable Vector ki_;
      mutable DenseMatrix Mr_;
      mutable DenseMatrix Mi_;

      static void km(double a, const Vector & k, double m,
                     DenseMatrix & M)
      {
         M(2,1) = a * m * k(0);
         M(0,2) = a * m * k(1);
         M(1,0) = a * m * k(2);

         M(1,2) = -M(2,1);
         M(2,0) = -M(0,2);
         M(0,1) = -M(1,0);
      }

      void km(double a, const Vector & k, const DenseMatrix & m,
              DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            int i1 = (i + 1) % 3;
            int i2 = (i + 2) % 3;
            for (int j=0; j<3; j++)
            {
               M(i,j) += a * k(i1) * m(i2,j);
               M(i,j) -= a * k(i2) * m(i1,j);
            }
         }
      }

      void mk(double a, const DenseMatrix & m, const Vector & k,
              DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               int j1 = (j + 1) % 3;
               int j2 = (j + 2) % 3;
               M(i,j) += a * m(i,j1) * k(j2);
               M(i,j) -= a * m(i,j2) * k(j1);
            }
         }
      }

   public:
      CrossCoefficient(VectorCoefficient *krCoef,
                       VectorCoefficient *kiCoef,
                       Coefficient *mCoef,
                       bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef),
           kiCoef_(kiCoef),
           mCoef_(mCoef),
           MrCoef_(NULL),
           MiCoef_(NULL),
           a_(a), realPart_(realPart), km_(true), kr_(3), ki_(3)
      { kr_ = 0.0; ki_ = 0.0; }

      CrossCoefficient(VectorCoefficient *krCoef,
                       VectorCoefficient *kiCoef,
                       MatrixCoefficient *MrCoef,
                       MatrixCoefficient *MiCoef,
                       bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef),
           kiCoef_(kiCoef),
           mCoef_(NULL),
           MrCoef_(MrCoef),
           MiCoef_(MiCoef),
           a_(a), realPart_(realPart), km_(true),
           kr_(3), ki_(3), Mr_(3), Mi_(3)
      { kr_ = 0.0; ki_ = 0.0; Mr_ = 0.0; Mi_ = 0.0; }

      CrossCoefficient(MatrixCoefficient *MrCoef,
                       MatrixCoefficient *MiCoef,
                       VectorCoefficient *krCoef,
                       VectorCoefficient *kiCoef,
                       bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef),
           kiCoef_(kiCoef),
           mCoef_(NULL),
           MrCoef_(MrCoef),
           MiCoef_(MiCoef),
           a_(a), realPart_(realPart), km_(false),
           kr_(3), ki_(3), Mr_(3), Mi_(3)
      { kr_ = 0.0; ki_ = 0.0; Mr_ = 0.0; Mi_ = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             (mCoef_ == NULL && MrCoef_ == NULL && MiCoef_ == NULL))
         {
            return;
         }

         double m = 0.0;
         if (krCoef_) { krCoef_->Eval(kr_, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki_, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }
         if (MrCoef_) { MrCoef_->Eval(Mr_, T, ip); }
         if (MiCoef_) { MiCoef_->Eval(Mi_, T, ip); }

         if (realPart_)
         {
            if (mCoef_)
            {
               if (krCoef_) { km(1.0, kr_, m, M); }
            }
            else
            {
               if (km_)
               {
                  if (krCoef_ && MrCoef_) { km(1.0, kr_, Mr_, M); }
                  if (kiCoef_ && MiCoef_) { km(-1.0, ki_, Mi_, M); }
               }
               else
               {
                  if (krCoef_ && MrCoef_) { mk(1.0, Mr_, kr_, M); }
                  if (kiCoef_ && MiCoef_) { mk(-1.0, Mi_, ki_, M); }
               }
            }
         }
         else
         {
            if (mCoef_)
            {
               if (kiCoef_) { km(1.0, ki_, m, M); }
            }
            else
            {
               if (km_)
               {
                  if (krCoef_ && MiCoef_) { km(1.0, kr_, Mi_, M); }
                  if (kiCoef_ && MrCoef_) { km(1.0, ki_, Mr_, M); }
               }
               else
               {
                  if (MrCoef_ && kiCoef_) { mk(1.0, Mr_, ki_, M); }
                  if (MiCoef_ && krCoef_) { mk(1.0, Mi_, kr_, M); }
               }
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   bool cyl_;
   bool pa_;

   HCurlCylStiffnessCoefficient epsInvReCylCoef_;
   HCurlCylStiffnessCoefficient epsInvImCylCoef_;
   HCurlCylMassCoefficient muCylCoef_;

   ScalarMassCoefficient massCoef_;  // -omega^2 mu
   MatrixMassCoefficient massCylCoef_;  // -omega^2 mu

   kmkCoefficient kmkReCoef_;  // Real part of k x epsInv k x
   kmkCoefficient kmkImCoef_;  // Imaginary part of k x epsInv k x
   CrossCoefficient kmReCoef_; // Real part of k x epsInv
   CrossCoefficient kmImCoef_; // Imaginary part of k x epsInv
   CrossCoefficient mkReCoef_; // Real part of epsInv k x
   CrossCoefficient mkImCoef_; // Imaginary part of epsInv k x

   kmkCoefficient kmkCylReCoef_;  // Real part of k x epsInv k x
   kmkCoefficient kmkCylImCoef_;  // Imaginary part of k x epsInv k x
   CrossCoefficient kmCylReCoef_; // Real part of k x epsInv
   CrossCoefficient kmCylImCoef_; // Imaginary part of k x epsInv
   CrossCoefficient mkCylReCoef_; // Real part of epsInv k x
   CrossCoefficient mkCylImCoef_; // Imaginary part of epsInv k x

public:
   Maxwell2ndH(ParFiniteElementSpace & HCurlFESpace,
               double omega,
               ComplexOperator::Convention conv,
               MatrixCoefficient & epsInvReCoef,
               MatrixCoefficient & epsInvImCoef,
               Coefficient & muCoef,
               VectorCoefficient * kReCoef,
               VectorCoefficient * kImCoef,
               bool cyl,
               bool pa);

   void Assemble();
};

class CurrentSourceH : public ParComplexLinearForm
{
private:

   double omega_;

   ParComplexGridFunction jt_;
   ParComplexGridFunction kt_;

   // These contain the J*exp(-i k x) and K*exp(-i k x)
   CmplxVecCoefArray jtilde_; // Volume Currents w/Phase
   CmplxVecCoefArray ktilde_; // Surface Currents w/Phase

   HCurlCylSourceCoefficient rhoSCoef_;

   // These contain the rho*S*J*exp(-i k x) and rho*S*K*exp(-i k x)
   CmplxVecCoefArray jtildeCyl_; // Volume Currents w/Phase
   CmplxVecCoefArray ktildeCyl_; // Surface Currents w/Phase

public:
   CurrentSourceH(ParFiniteElementSpace & HCurlFESpace,
                  ParFiniteElementSpace & HDivFESpace,
                  double omega,
                  ComplexOperator::Convention conv,
                  const CmplxVecCoefArray & jsrc,
                  const CmplxVecCoefArray & ksrc,
                  VectorCoefficient * kReCoef,
                  VectorCoefficient * kImCoef,
                  bool cyl,
                  bool pa);

   ~CurrentSourceH();

   ParComplexGridFunction & GetVolumeCurrentDensity() { return jt_; }
   ParComplexGridFunction & GetSurfaceCurrentDensity() { return kt_; }

   void Update();
   void Assemble();
};

class AmperesLaw
{
private:

   const ParComplexGridFunction & h_; // Complex magnetic field (HCurl)
   const ParComplexGridFunction & j_; // Complex current density (HDiv)
   ParComplexGridFunction         d_; // Complex electric flux (HDiv)

   const double omega_;

   ParDiscreteCurlOperator curl_;

   ParDiscreteLinearOperator * kReCross_;
   ParDiscreteLinearOperator * kImCross_;

public:
   AmperesLaw(const ParComplexGridFunction &h,
              const ParComplexGridFunction &j,
              double omega,
              VectorCoefficient * kReCoef,
              VectorCoefficient * kImCoef);

   ParComplexGridFunction & GetElectricFlux() { return d_; }

   void Update();
   void Assemble();
   void ComputeD();
};

/// Cold Plasma Dielectric Solver
class CPDSolverDH
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
      DMUMPS      =  6,
      ZMUMPS      =  7
   };

   // Solver options
   struct SolverOptions
   {
      int maxIter;
      int kDim;
      int printLvl;
      double relTol;

      // Euclid Options
      int euLvl;
   };

   CPDSolverDH(ParMesh & pmesh, int order, double omega,
               CPDSolverDH::SolverType s, SolverOptions & sOpts,
               CPDSolverDH::PrecondType p,
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
               bool vis_u = false,
               bool cyl = false,
               bool pa = false);
   ~CPDSolverDH();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   double GetEFieldError(const VectorCoefficient & EReCoef,
                         const VectorCoefficient & EImCoef) const;

   double GetHFieldError(const VectorCoefficient & HReCoef,
                         const VectorCoefficient & HImCoef) const;

   void GetErrorEstimates(Vector & errors, bool err_h = true);

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   void DisplayAnimationToGLVis();

   // const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   class kekCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      MatrixCoefficient * erCoef_;
      MatrixCoefficient * eiCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr;
      mutable Vector ki;
      mutable DenseMatrix er;
      mutable DenseMatrix ei;

      void kek(double a,
               const Vector & kl, const DenseMatrix & e, const Vector &kr,
               DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            int i1 = (i+1)%3;
            int i2 = (i+2)%3;
            for (int j=0; j<3; j++)
            {
               int j1 = (j+1)%3;
               int j2 = (j+2)%3;
               M(i,j) +=
                  a * (kl(i2) * e(i1,j2) * kr(j1) -
                       kl(i2) * e(i1,j1) * kr(j2) -
                       kl(i1) * e(i2,j2) * kr(j1) +
                       kl(i1) * e(i2,j1) * kr(j2)
                      );
            }
         }
      }

   public:
      kekCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     MatrixCoefficient *erCoef, MatrixCoefficient *eiCoef,
                     bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           erCoef_(erCoef), eiCoef_(eiCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3), er(3), ei(3)
      { kr = 0.0; ki = 0.0; er = 0.0; ei = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             (erCoef_ == NULL && eiCoef_ == NULL))
         {
            return;
         }
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (erCoef_) { erCoef_->Eval(er, T, ip); }
         if (eiCoef_) { eiCoef_->Eval(ei, T, ip); }

         if (realPart_)
         {
            if (krCoef_ && erCoef_) { kek(1.0, kr, er, kr, M); }
            if (kiCoef_ && erCoef_) { kek(-1.0, ki, er, ki, M); }
            if (krCoef_ && eiCoef_ && kiCoef_) { kek(-1.0, kr, ei, ki, M); }
            if (kiCoef_ && eiCoef_ && krCoef_) { kek(-1.0, ki, ei, kr, M); }
         }
         else
         {
            if (krCoef_ && eiCoef_) { kek(1.0, kr, ei, kr, M); }
            if (kiCoef_ && eiCoef_) { kek(-1.0, ki, ei, ki, M); }
            if (krCoef_ && erCoef_ && kiCoef_) { kek(1.0, kr, er, ki, M); }
            if (kiCoef_ && erCoef_ && krCoef_) { kek(1.0, ki, er, kr, M); }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class ekCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      MatrixCoefficient * erCoef_;
      MatrixCoefficient * eiCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr;
      mutable Vector ki;
      mutable DenseMatrix er;
      mutable DenseMatrix ei;

      void ek(double a,
              const DenseMatrix & e, const Vector &k,
              DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               int j1 = (j+1)%3;
               int j2 = (j+2)%3;
               M(i,j) += a * (e(i,j1) * k(j2) - e(i,j2) * k(j1));
            }
         }
      }

   public:
      ekCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                    MatrixCoefficient *erCoef, MatrixCoefficient *eiCoef,
                    bool realPart,
                    double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           erCoef_(erCoef), eiCoef_(eiCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3), er(3), ei(3)
      { kr = 0.0; ki = 0.0; er = 0.0; ei = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             (erCoef_ == NULL && eiCoef_ == NULL))
         {
            return;
         }
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (erCoef_) { erCoef_->Eval(er, T, ip); }
         if (eiCoef_) { eiCoef_->Eval(ei, T, ip); }

         if (realPart_)
         {
            if (erCoef_ && krCoef_) { ek(1.0, er, kr, M); }
            if (eiCoef_ && kiCoef_) { ek(-1.0, ei, ki, M); }
         }
         else
         {
            if (eiCoef_ && krCoef_) { ek(1.0, ei, kr, M); }
            if (erCoef_ && kiCoef_) { ek(1.0, er, ki, M); }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class keCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      MatrixCoefficient * erCoef_;
      MatrixCoefficient * eiCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr;
      mutable Vector ki;
      mutable DenseMatrix er;
      mutable DenseMatrix ei;

      void ke(double a,
              const Vector &k, const DenseMatrix & e,
              DenseMatrix & M)
      {
         for (int i=0; i<3; i++)
         {
            int i1 = (i+1)%3;
            int i2 = (i+2)%3;
            for (int j=0; j<3; j++)
            {
               M(i,j) += a * (k(i1) * e(i2,j) - k(i2) * e(i1,j));
            }
         }
      }

   public:
      keCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                    MatrixCoefficient *erCoef, MatrixCoefficient *eiCoef,
                    bool realPart,
                    double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           erCoef_(erCoef), eiCoef_(eiCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3), er(3), ei(3)
      { kr = 0.0; ki = 0.0; er = 0.0; ei = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;
         if ((krCoef_ == NULL && kiCoef_ == NULL) ||
             (erCoef_ == NULL && eiCoef_ == NULL))
         {
            return;
         }
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (erCoef_) { erCoef_->Eval(er, T, ip); }
         if (eiCoef_) { eiCoef_->Eval(ei, T, ip); }

         if (realPart_)
         {
            if (krCoef_ && erCoef_) { ke(1.0, kr, er, M); }
            if (kiCoef_ && eiCoef_) { ke(-1.0, ki, ei, M); }
         }
         else
         {
            if (krCoef_ && eiCoef_) { ke(1.0, kr, ei, M); }
            if (kiCoef_ && erCoef_) { ke(1.0, ki, er, M); }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class CylStiffnessCoef : public MatrixCoefficient
   {
   private:
      MatrixCoefficient &A_;
      mutable Vector x_;

   public:
      CylStiffnessCoef(MatrixCoefficient &A)
         : MatrixCoefficient(3), A_(A), x_(3) {}

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         T.Transform(ip, x_);

         A_.Eval(M, T, ip);

         M(0,0) /= x_[1];
         M(1,0) /= x_[1];
         M(0,1) /= x_[1];
         M(1,1) /= x_[1];

         M(2,2) *= x_[1];
      }
   };

   class CylMassCoef : public MatrixCoefficient
   {
   private:
      Coefficient &A_;
      mutable Vector x_;

   public:
      CylMassCoef(Coefficient &A)
         : MatrixCoefficient(3), A_(A), x_(3) {}

      void SetCoefficient(Coefficient &A) { A_ = A; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         T.Transform(ip, x_);

         double a = A_.Eval(T, ip);

         M.SetSize(3);
         M = 0.0;

         M(0,0) = a * x_[1];
         M(1,1) = a * x_[1];
         M(2,2) = a / x_[1];
      }
   };

   class CylSourceCoef : public MatrixCoefficient
   {
   private:
      MatrixCoefficient &A_;
      mutable Vector x_;

   public:
      CylSourceCoef(MatrixCoefficient &A)
         : MatrixCoefficient(3), A_(A), x_(3) {}

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         T.Transform(ip, x_);

         A_.Eval(M, T, ip);

         M(2,0) *= x_[1];
         M(2,1) *= x_[1];
         M(2,2) *= x_[1];
      }
   };

   void collectBdrAttributes(const Array<AttributeArrays*> & aa,
                             Array<int> & attr_marker);

   void locateTrueDBCDofs(const Array<int> & dbc_bdr_marker,
                          Array<int> & dbc_nd_tdofs);

   void locateTrueSBCDofs(const Array<int> & sbc_bdr_marker,
                          Array<int> & non_sbc_h1_tdofs,
                          Array<int> & sbc_nd_tdofs);

   void computeD(const ParComplexGridFunction & h,
                 const ParComplexGridFunction & j,
                 ParComplexGridFunction & d);

   void computeE(const ParComplexGridFunction & d,
                 ParComplexGridFunction & e);

   void computeB(const ParComplexGridFunction & h,
                 ParComplexGridFunction & b);

   void prepareVectorVisField(const ParComplexGridFunction &u,
                              ComplexGridFunction &v);

   void prepareVisFields();

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   bool ownsEta_;
   bool cyl_;
   bool vis_u_;
   bool pa_;

   double omega_;

   ParMesh * pmesh_;

   H1_ParFESpace * H1FESpace_;
   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2FESpace2p_;
   L2_ParFESpace * L2VSFESpace_;
   L2_ParFESpace * L2VSFESpace2p_;
   L2_ParFESpace * L2V3FESpace_;
   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * HDivFESpace_;
   // ParFiniteElementSpace * HDivFESpace2p_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   // ParSesquilinearForm * a1_;
   // ParBilinearForm * b1_;
   ParMixedSesquilinearForm * nxD01_;
   ParMixedSesquilinearForm * d21EpsInv_;

   ParSesquilinearForm * m1_;
   ParMixedSesquilinearForm * m21EpsInv_;

   // ParBilinearForm * m0_;
   // ParMixedBilinearForm * n20ZRe_;
   // ParMixedBilinearForm * n20ZIm_;

   ConstantCoefficient negOneCoef_;
   ParSesquilinearForm * m0_;
   ParMixedSesquilinearForm * nzD12_;

   ParDiscreteGradOperator * grad_; // For Computing E from phi
   ParDiscreteCurlOperator * curl_; // For Computing D from H
   ParDiscreteLinearOperator * kReCross_;
   ParDiscreteLinearOperator * kImCross_;

   ParComplexGridFunction   h_;   // Complex magnetic field (HCurl)
   ParComplexGridFunction   h_dbc_;   // Complex magnetic field (HCurl)
   ParComplexGridFunction   e_;   // Complex electric field (HCurl)
   // ParComplexGridFunction * d_;   // Complex electric flux (HDiv)
   ParComplexGridFunction * j_;   // Complex current density (HDiv)
   ParComplexLinearForm   * curlj_; // Curl of current density (HCurl)
   ParComplexGridFunction * phi_; // Complex sheath potential (H1)
   ParComplexGridFunction * prev_phi_; // Complex sheath potential (H1)

   // ParComplexGridFunction * e_tmp_; // Temporary complex electric field (HCurl)
   // ParGridFunction * temp_; // Temporary grid function (HCurl)
   // ParComplexGridFunction * phi_tmp_; // Complex sheath potential temporary (H1)
   ParComplexGridFunction * rectPot_; // Complex rectified potential (H1)
   // ParComplexGridFunction * j_;   // Complex current density (HCurl)
   ParComplexLinearForm   * rhs1_; // RHS of magnetic field eqn (HCurl)
   ParComplexLinearForm   * rhs0_; // RHS of sheath potential eqn (H1)
   ParGridFunction        * h_t_; // Time dependent magnetic field
   ComplexVectorFieldVisObject h_v_;
   ComplexVectorFieldVisObject d_v_;
   ComplexVectorFieldVisObject e_v_;
   ComplexVectorFieldVisObject h_dbc_v_;
   ScalarFieldVisObject phi_v_;
   ScalarFieldBdrVisObject z_v_;
   // ComplexGridFunction    * e_b_v_; // Complex parallel electric field (L2)
   // ComplexGridFunction    * h_v_; // Complex magnetic field (L2^d)
   // ComplexGridFunction    * h_tilde_; // Complex magnetic field (HCurl3D)
   // ComplexGridFunction    * e_v_; // Complex electric field (L2^d)
   // ComplexGridFunction    * d_v_; // Complex electric flux (L2^d)
   // ComplexGridFunction    * phi_v_; // Complex sheath potential (L2)
   // ComplexGridFunction    * rectPot_v_; // Complex rectified potential (L2)
   // ComplexGridFunction    * j_v_; // Complex current density (L2^d)
   // ComplexGridFunction    * div_d_; // Complex charge density (L2)
   // ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
   // GridFunction           * b_hat_v_; // Unit vector along B (L2^d)
   // ParGridFunction        * u_;   // Energy density (L2)
   // ParGridFunction        * uE_;  // Electric Energy density (L2)
   // ParGridFunction        * uB_;  // Magnetic Energy density (L2)
   // ParComplexGridFunction * S_;  // Poynting Vector (HDiv)
   ComplexGridFunction * StixS_; // Stix S Coefficient (L2)
   ComplexGridFunction * StixD_; // Stix D Coefficient (L2)
   ComplexGridFunction * StixP_; // Stix P Coefficient (L2)
   ParComplexGridFunction * EpsPara_; // B^T eps B / |B|^2 Coefficient (L2)

   VectorCoefficient * BCoef_;        // B Field Unit Vector
   // MatrixCoefficient * epsReCoef_;    // Dielectric Material Coefficient
   // MatrixCoefficient * epsImCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsInvReCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsInvImCoef_;    // Dielectric Material Coefficient
   // MatrixCoefficient * epsAbsCoef_;   // Dielectric Material Coefficient
   Coefficient       * muCoef_;       // Dia/Paramagnetic Material Coefficient
   PowerCoefficient    muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaCoef_;      // Impedance Coefficient
   VectorCoefficient * kReCoef_;        // Wave Vector
   VectorCoefficient * kImCoef_;        // Wave Vector

   bool cylSymm_;
   CylMassCoef      cylMassCoef_;
   CylStiffnessCoef cylStiffnessReCoef_;
   CylStiffnessCoef cylStiffnessImCoef_;
   CylSourceCoef    cylSourceReCoef_;
   CylSourceCoef    cylSourceImCoef_;

   Coefficient * SReCoef_; // Stix S Coefficient
   Coefficient * SImCoef_; // Stix S Coefficient
   Coefficient * DReCoef_; // Stix D Coefficient
   Coefficient * DImCoef_; // Stix D Coefficient
   Coefficient * PReCoef_; // Stix P Coefficient
   Coefficient * PImCoef_; // Stix P Coefficient

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * abcCoef_;       // -omega eta
   // Coefficient * sbcReCoef_;     //  omega Im(eta^{-1})
   // Coefficient * sbcImCoef_;     // -omega Re(eta^{-1})
   // Coefficient * negMuInvCoef_;  // -1.0 / mu

   Coefficient * massCoef_;  // -omega^2 mu
   Coefficient * posMassCoef_; // omega^2 mu
   // MatrixCoefficient * negMuInvkxkxCoef_; // -\vec{k}\times\vec{k}\times/mu

   // VectorCoefficient * negMuInvkCoef_; // -\vec{k}/mu

   kekCoefficient kekReCoef_;
   kekCoefficient kekImCoef_;
   keCoefficient keReCoef_;
   keCoefficient keImCoef_;
   ekCoefficient ekReCoef_;
   ekCoefficient ekImCoef_;

   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsrCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsiCoef_;     // Volume Current Density Function

   VectorGridFunctionCoefficient erCoef_;
   VectorGridFunctionCoefficient eiCoef_;

   CurlGridFunctionCoefficient derCoef_;
   CurlGridFunctionCoefficient deiCoef_;

   // EnergyDensityCoef     uCoef_;
   // ElectricEnergyDensityCoef uECoef_;
   // MagneticEnergyDensityCoef uBCoef_;
   // PoyntingVectorReCoef SrCoef_;
   // PoyntingVectorImCoef SiCoef_;

   // const VectorCoefficient & erCoef_;     // Electric Field Boundary Condition
   // const VectorCoefficient & eiCoef_;     // Electric Field Boundary Condition

   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_bdr_marker_;

   const Array<ComplexVectorCoefficientByAttr*> & dbcs_;
   Array<int> dbc_bdr_marker_;
   Array<int> dbc_nd_tdofs_;
   Array<int> non_k_bdr_;

   const Array<ComplexVectorCoefficientByAttr*> & nbcs_; // Surface current BCs
   Array<ComplexVectorCoefficientByAttr*> nkbcs_; // Neumann BCs (-i*omega*K)

   const Array<ComplexCoefficientByAttr*> & sbcs_; // Sheath BCs
   Array<int> sbc_bdr_marker_;
   Array<int> non_sbc_h1_tdofs_;
   Array<int> sbc_nd_tdofs_;

   const Array<AttributeArrays*> & axis_; // Cylindrical Axis
   Array<int> axis_tdofs_;

   Maxwell2ndH    maxwell_;
   CurrentSourceH current_;
   AmperesLaw     ampere_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER
