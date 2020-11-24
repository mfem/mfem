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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER

#include "../common/pfem_extras.hpp"
#include "cold_plasma_dielectric_solver.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

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
struct ComplexCoefficientByAttr
{
   Array<int> attr;
   Array<int> attr_marker;
   Coefficient * real;
   Coefficient * imag;
};
/*
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
               VectorCoefficient * kCoef,
               Array<int> & abcs,
               Array<ComplexVectorCoefficientByAttr> & dbcs,
               Array<ComplexVectorCoefficientByAttr> & nbcs,
               Array<ComplexCoefficientByAttr> & sbcs,
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
      VectorCoefficient * kCoef_;
      MatrixCoefficient * eCoef_;

      double a_;

      mutable Vector k;
      mutable DenseMatrix e;

   public:
      kekCoefficient(VectorCoefficient *kCoef, MatrixCoefficient *eCoef,
                     double a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef), eCoef_(eCoef), a_(a), k(3), e(3) {}

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         if (kCoef_ == NULL || eCoef_ == NULL)
         {
            M = 0.0;
            return;
         }
         kCoef_->Eval(k, T, ip);
         eCoef_->Eval(e, T, ip);

         for (int i=0; i<3; i++)
         {
            int i1 = (i+1)%3;
            int i2 = (i+2)%3;
            for (int j=0; j<3; j++)
            {
               int j1 = (j+1)%3;
               int j2 = (j+2)%3;
               M(i,j) =
                  k(i1) * e(i2,j2) * k(j1) -
                  k(i1) * e(i2,j1) * k(j2) -
                  k(i2) * e(i1,j2) * k(j1) +
                  k(i2) * e(i1,j1) * k(j2);
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class ekCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * kCoef_;
      MatrixCoefficient * eCoef_;

      double a_;

      mutable Vector k;
      mutable DenseMatrix e;

   public:
      ekCoefficient(VectorCoefficient *kCoef, MatrixCoefficient *eCoef,
                    double a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef), eCoef_(eCoef), a_(a), k(3), e(3) {}

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         if (kCoef_ == NULL || eCoef_ == NULL)
         {
            M = 0.0;
            return;
         }
         kCoef_->Eval(k, T, ip);
         eCoef_->Eval(e, T, ip);

         for (int i=0; i<3; i++)
         {
            for (int j=0; j<3; j++)
            {
               int j1 = (j+1)%3;
               int j2 = (j+2)%3;
               M(i,j) = e(i,j1) * k(j2) - e(i,j2) * k(j1);
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class keCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * kCoef_;
      MatrixCoefficient * eCoef_;

      double a_;

      mutable Vector k;
      mutable DenseMatrix e;

   public:
      keCoefficient(VectorCoefficient *kCoef, MatrixCoefficient *eCoef,
                    double a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef), eCoef_(eCoef), a_(a), k(3), e(3) {}

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         if (kCoef_ == NULL || eCoef_ == NULL)
         {
            M = 0.0;
            return;
         }
         kCoef_->Eval(k, T, ip);
         eCoef_->Eval(e, T, ip);

         for (int i=0; i<3; i++)
         {
            int i1 = (i+1)%3;
            int i2 = (i+2)%3;
            for (int j=0; j<3; j++)
            {
               M(i,j) = k(i2) * e(i1,j) - k(i1) * e(i2,j);
            }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

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
   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;
   RT_ParFESpace * HDivFESpace2p_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   ParSesquilinearForm * a1_;
   // ParBilinearForm * b1_;
   ParMixedSesquilinearForm * d21EpsInv_;

   ParSesquilinearForm * m1_;
   ParMixedSesquilinearForm * m21EpsInv_;


   ParBilinearForm * m0_;
   ParMixedBilinearForm * n20ZRe_;
   ParMixedBilinearForm * n20ZIm_;

   ParDiscreteGradOperator * grad_; // For Computing E from phi
   ParDiscreteCurlOperator * curl_; // For Computing D from H
   ParDiscreteLinearOperator * kCross_;

   ParComplexGridFunction * h_;   // Complex magnetic field (HCurl)
   ParComplexGridFunction * e_;   // Complex electric field (HCurl)
   ParComplexGridFunction * d_;   // Complex electric flux (HDiv)
   ParComplexGridFunction * j_;   // Complex current density (HDiv)
   ParComplexGridFunction * phi_; // Complex sheath potential (H1)

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
   ParComplexGridFunction * e_v_; // Complex electric field (L2^d)
   ParComplexGridFunction * d_v_; // Complex electric flux (L2^d)
   ParComplexGridFunction * phi_v_; // Complex sheath potential (L2)
   ParComplexGridFunction * j_v_; // Complex current density (L2^d)
   ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
   // ParGridFunction        * u_;   // Energy density (L2)
   // ParGridFunction        * uE_;  // Electric Energy density (L2)
   // ParGridFunction        * uB_;  // Magnetic Energy density (L2)
   // ParComplexGridFunction * S_;  // Poynting Vector (HDiv)
   ParComplexGridFunction * StixS_; // Stix S Coefficient (L2)
   ParComplexGridFunction * StixD_; // Stix D Coefficient (L2)
   ParComplexGridFunction * StixP_; // Stix P Coefficient (L2)
   ParComplexGridFunction * EpsPara_; // B^T eps B / |B|^2 Coefficient (L2)

   VectorCoefficient * BCoef_;        // B Field Unit Vector
   // MatrixCoefficient * epsReCoef_;    // Dielectric Material Coefficient
   // MatrixCoefficient * epsImCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsInvReCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsInvImCoef_;    // Dielectric Material Coefficient
   // MatrixCoefficient * epsAbsCoef_;   // Dielectric Material Coefficient
   Coefficient       * muCoef_;       // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaCoef_;      // Impedance Coefficient
   VectorCoefficient * kCoef_;        // Wave Vector

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
   Array<int> abc_marker_;

   // Array of 0's and 1's marking the location of sheath surfaces
   // Array<int> sbc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker_;
   // void   (*e_r_bc_)(const Vector&, Vector&);
   // void   (*e_i_bc_)(const Vector&, Vector&);

   // Array<int> * dbcs_;
   Array<ComplexVectorCoefficientByAttr> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   Array<ComplexVectorCoefficientByAttr> * nbcs_; // Surface current BCs
   Array<ComplexVectorCoefficientByAttr> * nkbcs_; // Neumann BCs (-i*omega*K)

   Array<ComplexCoefficientByAttr> * sbcs_; // Sheath BCs
   Array<int> sbc_bdr_;
   Array<int> sbc_nd_tdofs_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_DH_SOLVER
