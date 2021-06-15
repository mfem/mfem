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

#ifndef MFEM_COLD_PLASMA_DIELECTRIC_SOLVER
#define MFEM_COLD_PLASMA_DIELECTRIC_SOLVER

#include "../common/pfem_extras.hpp"
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

struct AttributeArrays
{
   Array<int> attr;
   Array<int> attr_marker;

};

struct ComplexCoefficientByAttr : public AttributeArrays
{
   Coefficient * real;
   Coefficient * imag;
};

struct ComplexVectorCoefficientByAttr : public AttributeArrays
{
   VectorCoefficient * real;
   VectorCoefficient * imag;
};

// Used for combining scalar coefficients
double prodFunc(double a, double b);

class StixBCs
{
public:
   enum BCType {DIRICHLET_BC, NEUMANN_BC, SHEATH_BC};

private:
   Array<ComplexVectorCoefficientByAttr*>  dbc; // Dirichlet BC data
   Array<ComplexVectorCoefficientByAttr*>  nbc; // Neumann BC data
   Array<ComplexCoefficientByAttr*> sbc; // Sheath BC data

   mutable Array<int>  hbc_attr; // Homogeneous Neumann BC boundary attributes
   Array<int>  dbc_attr; // Dirichlet BC boundary attributes

   std::set<int> bc_attr;
   const Array<int> & bdr_attr;

public:
   StixBCs(const Array<int> & bdr)
      : bdr_attr(bdr) {}

   ~StixBCs();

   static const char * GetBCTypeName(BCType bctype);

   // Enforce u = val on boundaries with attributes in bdr
   void AddDirichletBC(const Array<int> & bdr,
                       VectorCoefficient &real_val,
                       VectorCoefficient &imag_val);

   // Enforce du/dn = val on boundaries with attributes in bdr
   void AddNeumannBC(const Array<int> & bdr,
                     VectorCoefficient &real_val,
                     VectorCoefficient &imag_val);

   // Model a non-linear plasma sheath on boundaries with attributes in bdr
   void AddSheathBC(const Array<int> & bdr,
                    Coefficient &real_imped,
                    Coefficient &imag_imped);

   const Array<ComplexVectorCoefficientByAttr*> & GetDirichletBCs() const
   { return dbc; }
   const Array<ComplexVectorCoefficientByAttr*> & GetNeumannBCs() const
   { return nbc; }
   const Array<ComplexCoefficientByAttr*> & GetSheathBCs() const { return sbc; }

   const Array<int> & GetHomogeneousNeumannBDR() const;
   const Array<int> & GetDirichletBDR() const { return dbc_attr; }
};

/// Coefficient which returns func(kr*x)*exp(-ki*x) where kr and ki are vectors
class ComplexPhaseCoefficient : public Coefficient
{
private:
   double(*func_)(double);
   VectorCoefficient * kr_;
   VectorCoefficient * ki_;
   mutable Vector krVec_;
   mutable Vector kiVec_;

public:
   ComplexPhaseCoefficient(Vector & kr, Vector & ki, double(&func)(double))
      : func_(&func), kr_(NULL), ki_(NULL), krVec_(kr), kiVec_(ki) {}

   ComplexPhaseCoefficient(VectorCoefficient & kr, VectorCoefficient & ki,
                           double(&func)(double))
      : func_(&func), kr_(&kr), ki_(&ki),
        krVec_(kr.GetVDim()), kiVec_(ki.GetVDim()) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3); transip = 0.0;
      T.Transform(ip, transip);
      if (kr_) { kr_->Eval(krVec_, T, ip); }
      if (ki_) { ki_->Eval(kiVec_, T, ip); }
      transip.SetSize(krVec_.Size());
      return (*func_)(krVec_ * transip)*exp(-(kiVec_ * transip));
   }
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

/// Cold Plasma Dielectric Solver
class CPDSolver
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

   CPDSolver(ParMesh & pmesh, int order, double omega,
             CPDSolver::SolverType s, SolverOptions & sOpts,
             CPDSolver::PrecondType p,
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
             Array<ComplexVectorCoefficientByAttr> & dbcs,
             Array<ComplexVectorCoefficientByAttr> & nbcs,
             Array<ComplexCoefficientByAttr> & sbcs,
             void (*j_r_src)(const Vector&, Vector&),
             void (*j_i_src)(const Vector&, Vector&),
             bool vis_u = false,
             bool pa = false);
   ~CPDSolver();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   double GetError(const VectorCoefficient & EReCoef,
                   const VectorCoefficient & EImCoef) const;

   void GetErrorEstimates(Vector & errors);

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   void DisplayAnimationToGLVis();

   // const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   class kmkCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * krCoef_;
      VectorCoefficient * kiCoef_;
      Coefficient       * mCoef_;

      bool realPart_;
      double a_;

      mutable Vector kr;
      mutable Vector ki;

      void kmk(double a,
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

   public:
      kmkCoefficient(VectorCoefficient *krCoef, VectorCoefficient *kiCoef,
                     Coefficient *mCoef,
                     bool realPart, double a = 1.0)
         : MatrixCoefficient(3),
           krCoef_(krCoef), kiCoef_(kiCoef),
           mCoef_(mCoef),
           realPart_(realPart),
           a_(a), kr(3), ki(3)
      { kr = 0.0; ki = 0.0; }

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
         double m = 0.0;
         if (krCoef_) { krCoef_->Eval(kr, T, ip); }
         if (kiCoef_) { kiCoef_->Eval(ki, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }

         if (realPart_)
         {
            if (krCoef_) { kmk(1.0, kr, m, kr, M); }
            if (kiCoef_) { kmk(-1.0, ki, m, ki, M); }
         }
         else
         {
            if (krCoef_ && kiCoef_) { kmk(1.0, kr, m, ki, M); }
            if (kiCoef_ && krCoef_) { kmk(1.0, ki, m, kr, M); }
         }
         if (a_ != 1.0) { M *= a_; }
      }
   };

   class CrossCoefficient : public MatrixCoefficient
   {
   private:
      VectorCoefficient * kCoef_;
      Coefficient * mCoef_;

      double a_;

      mutable Vector k;

   public:
      CrossCoefficient(VectorCoefficient *kCoef,
                       Coefficient *mCoef,
                       double a = 1.0)
         : MatrixCoefficient(3),
           kCoef_(kCoef),
           mCoef_(mCoef),
           a_(a), k(3)
      { k = 0.0; }

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         M.SetSize(3);
         M = 0.0;

         double m = 0.0;
         if (kCoef_) { kCoef_->Eval(k, T, ip); }
         if (mCoef_) { m = mCoef_->Eval(T, ip); }

         M(2,1) = a_ * m * k(0);
         M(0,2) = a_ * m * k(1);
         M(1,0) = a_ * m * k(2);

         M(1,2) = -M(2,1);
         M(2,0) = -M(0,2);
         M(0,1) = -M(1,0);
      }
   };

   void computeB(const ParComplexGridFunction & e,
                 ParComplexGridFunction & b);

   void computeD(const ParComplexGridFunction & e,
                 ParComplexGridFunction & d);

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   SolverType sol_;
   SolverOptions & solOpts_;
   PrecondType prec_;

   ComplexOperator::Convention conv_;

   bool ownsEtaInv_;
   bool vis_u_;
   bool pa_;

   double omega_;

   // double solNorm_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2FESpace2p_;
   L2_ParFESpace * L2VFESpace_;
   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * HDivFESpace_;
   ParFiniteElementSpace * HDivFESpace2p_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   ParSesquilinearForm * a1_;
   ParBilinearForm * b1_;

   ParBilinearForm * m2_;
   ParMixedBilinearForm * m12EpsRe_;
   ParMixedBilinearForm * m12EpsIm_;

   ParDiscreteCurlOperator * curl_; // For Computing D from H
   ParDiscreteLinearOperator * kReCross_;
   ParDiscreteLinearOperator * kImCross_;

   ParComplexGridFunction * e_;   // Complex electric field (HCurl)
   ParComplexGridFunction * d_;   // Complex electric flux (HDiv)
   ParComplexGridFunction * b_;   // Complex magnetic flux (HDiv)
   ParComplexGridFunction * j_;   // Complex current density (HCurl)
   ParComplexLinearForm   * rhs_; // Dual of complex current density (HCurl)
   ParGridFunction        * e_t_; // Time dependent Electric field
   ParComplexGridFunction * e_b_; // Complex parallel electric field (L2)
   ParComplexGridFunction * e_v_; // Complex electric field (L2^d)
   ParComplexGridFunction * d_v_; // Complex electric flux (L2^d)
   ParComplexGridFunction * j_v_; // Complex current density (L2^d)
   ParGridFunction        * b_hat_; // Unit vector along B (HDiv)
   ParGridFunction        * u_;   // Energy density (L2)
   ParGridFunction        * uE_;  // Electric Energy density (L2)
   ParGridFunction        * uB_;  // Magnetic Energy density (L2)
   ParComplexGridFunction * S_;  // Poynting Vector (HDiv)

   VectorCoefficient * BCoef_;        // B Field Vector
   MatrixCoefficient * epsReCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsImCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsAbsCoef_;   // Dielectric Material Coefficient
   Coefficient       * muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaInvCoef_;   // Admittance Coefficient
   VectorCoefficient * kReCoef_;        // Wave Vector
   VectorCoefficient * kImCoef_;        // Wave Vector

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * abcCoef_;       // -omega eta^{-1}
   Coefficient * sbcReCoef_;     //  omega Im(eta^{-1})
   Coefficient * sbcImCoef_;     // -omega Re(eta^{-1})
   Coefficient * sinkx_;         // sin(ky * y + kz * z)
   Coefficient * coskx_;         // cos(ky * y + kz * z)
   Coefficient * negsinkx_;      // -sin(ky * y + kz * z)
   // Coefficient * negMuInvCoef_;  // -1.0 / mu

   MatrixCoefficient * massReCoef_;  // -omega^2 Re(epsilon)
   MatrixCoefficient * massImCoef_;  // omega^2 Im(epsilon)
   MatrixCoefficient * posMassCoef_; // omega^2 Abs(epsilon)
   // MatrixCoefficient * negMuInvkxkxCoef_; // -\vec{k}\times\vec{k}\times/mu

   kmkCoefficient kmkReCoef_;
   kmkCoefficient kmkImCoef_;
   CrossCoefficient kmReCoef_;
   CrossCoefficient kmImCoef_;

   // VectorCoefficient * negMuInvkCoef_; // -\vec{k}/mu
   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsrCoef_;     // Volume Current Density Function
   VectorCoefficient * rhsiCoef_;     // Volume Current Density Function

   VectorGridFunctionCoefficient erCoef_;
   VectorGridFunctionCoefficient eiCoef_;

   CurlGridFunctionCoefficient derCoef_;
   CurlGridFunctionCoefficient deiCoef_;

   EnergyDensityCoef     uCoef_;
   ElectricEnergyDensityCoef uECoef_;
   MagneticEnergyDensityCoef uBCoef_;
   PoyntingVectorReCoef SrCoef_;
   PoyntingVectorImCoef SiCoef_;

   // const VectorCoefficient & erCoef_;     // Electric Field Boundary Condition
   // const VectorCoefficient & eiCoef_;     // Electric Field Boundary Condition

   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_bdr_marker_;

   Array<ComplexVectorCoefficientByAttr> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   Array<ComplexVectorCoefficientByAttr> * nbcs_; // Surface current BCs
   Array<ComplexVectorCoefficientByAttr> * nkbcs_; // Neumann BCs (-i*omega*K)

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_SOLVER
