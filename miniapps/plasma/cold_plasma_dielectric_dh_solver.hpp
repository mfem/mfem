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
#include "cold_plasma_dielectric_solver.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

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

class VectorR2DCoef : public VectorCoefficient
{
private:
   VectorCoefficient & coef_;
   ParMesh & pmesh_;
public:
   VectorR2DCoef(VectorCoefficient & coef, ParMesh & pmesh)
      : VectorCoefficient(coef.GetVDim()), coef_(coef), pmesh_(pmesh) {}

   void Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      int e = T.ElementNo;
      ElementTransformation * pT = pmesh_.GetElementTransformation(e);
      pT->SetIntPoint(&ip);
      coef_.Eval(v, *pT, ip);
   }
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
      STRUMPACK   =  5
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
               // Array<ComplexVectorCoefficientByAttr*> & dbcs,
               // Array<ComplexVectorCoefficientByAttr*> & nbcs,
               // Array<ComplexCoefficientByAttr*> & sbcs,
               void (*j_r_src)(const Vector&, Vector&),
               void (*j_i_src)(const Vector&, Vector&),
               bool vis_u = false,
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

   void GetErrorEstimates(Vector & errors);

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

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   bool ownsEta_;
   bool vis_u_;
   bool pa_;

   double omega_;

   // double solNorm_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2FESpace2p_;
   L2_ParFESpace * L2VFESpace_;
   L2_FESpace * L2FESpace3D_;
   L2_FESpace * L2VFESpace3D_;
   H1_ParFESpace * H1FESpace_;
   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * HDivFESpace_;
   ParFiniteElementSpace * HDivFESpace2p_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   ParSesquilinearForm * a1_;
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

   ParComplexGridFunction * h_;   // Complex magnetic field (HCurl)
   ParComplexGridFunction * e_;   // Complex electric field (HCurl)
   ParComplexGridFunction * d_;   // Complex electric flux (HDiv)
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
   ParGridFunction        * e_t_; // Time dependent Electric field
   ParComplexGridFunction * e_b_; // Complex parallel electric field (L2)
   ParComplexGridFunction * h_v_; // Complex magnetic field (L2^d)
   ComplexGridFunction    * e_v_; // Complex electric field (L2^d)
   ParComplexGridFunction * d_v_; // Complex electric flux (L2^d)
   ParComplexGridFunction * phi_v_; // Complex sheath potential (L2)
   ParComplexGridFunction * j_v_; // Complex current density (L2^d)
   ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
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
   Coefficient * sinkx_;         // sin(ky * y + kz * z)
   Coefficient * coskx_;         // cos(ky * y + kz * z)
   Coefficient * negsinkx_;      // -sin(ky * y + kz * z)
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

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER
