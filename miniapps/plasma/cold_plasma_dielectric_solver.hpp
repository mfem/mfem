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

using miniapps::H1_ParFESpace;
using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
using miniapps::L2_ParFESpace;
using miniapps::ParDiscreteGradOperator;
using miniapps::ParDiscreteCurlOperator;

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

struct ComplexVectorCoefficientByAttr
{
   Array<int> attr;
   VectorCoefficient * real;
   VectorCoefficient * imag;
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

class VectorEntryCoefficient : public Coefficient
{
private:
   Vector V;
   VectorCoefficient * VCoef;
   int r;

public:
   VectorEntryCoefficient(VectorCoefficient &coef, int row)
      : VCoef(&coef), r(row)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      VCoef->Eval(V, T, ip);
      return V[r];
   }
};

class MatrixEntryCoefficient : public Coefficient
{
private:
   DenseMatrix M;
   MatrixCoefficient * MCoef;
   int r, c;

public:
   MatrixEntryCoefficient(MatrixCoefficient &coef, int row, int col)
      : MCoef(&coef), r(row), c(col)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      MCoef->Eval(M, T, ip);
      return M(r, c);
   }
};

class SubVectorCoefficient : public VectorCoefficient
{
private:
   DenseMatrix M;
   Vector v;
   MatrixCoefficient * MCoef;
   VectorCoefficient * VCoef;
   int h, w, r, c;

public:
   SubVectorCoefficient(MatrixCoefficient &coef, int height, int width,
                        int row, int col)
      : VectorCoefficient(std::max(height, width)),
        MCoef(&coef), VCoef(NULL), h(height), w(width), r(row), c(col)
   {
      MFEM_ASSERT(h==1 || w==1, "Either the height or width must be 1. "
                  "Provided values are " << h << " and " << w << ".");
   }

   SubVectorCoefficient(VectorCoefficient &coef, int length, int offset)
      : VectorCoefficient(length),
        MCoef(NULL), VCoef(&coef), h(length), w(1), r(offset), c(0)
   {
      MFEM_ASSERT(h==1 || w==1, "Either the height or width must be 1. "
                  "Provided values are " << h << " and " << w << ".");
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      V.SetSize(vdim);

      if (MCoef)
      {
         MCoef->Eval(M, T, ip);

         if (h == 1)
         {
            for (int i=0; i<vdim; i++)
            {
               V[i] = M(r, c + i);
            }
         }
         else
         {
            for (int i=0; i<vdim; i++)
            {
               V[i] = M(r + i, c);
            }
         }
      }
      else if (VCoef)
      {
         VCoef->Eval(v, T, ip);

         for (int i=0; i<vdim; i++)
         {
            V[i] = v[r + i];
         }
      }
      else
      {
         V = 0.0;
      }
   }
};

class SubMatrixCoefficient : public MatrixCoefficient
{
private:
   DenseMatrix M;
   MatrixCoefficient * MCoef;
   int h, w, r, c;

public:
   SubMatrixCoefficient(MatrixCoefficient &coef, int height, int width,
                        int row, int col)
      : MatrixCoefficient(height, width),
        MCoef(&coef), r(row), c(col)
   {
   }

   void Eval(DenseMatrix &D, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      D.SetSize(height, width);

      MCoef->Eval(M, T, ip);

      for (int i=0; i<height; i++)
         for (int j=0; j<width; j++)
         {
            D(i, j) = M(r + i, c + j);
         }
   }
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
             MatrixCoefficient & epsReCoef,
             MatrixCoefficient & epsImCoef,
             MatrixCoefficient & epsAbsCoef,
             Coefficient & muInvCoef,
             Coefficient * etaInvCoef,
             Coefficient * etaInvReCoef,
             Coefficient * etaInvImCoef,
             VectorCoefficient * kCoef,
             Array<int> & abcs,
             Array<int> & sbcs,
             // Array<int> & dbcs,
             Array<ComplexVectorCoefficientByAttr> & dbcs,
             void (*j_r_src)(const Vector&, Vector&),
             void (*j_i_src)(const Vector&, Vector&),
             bool vis_u = false);
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

   double omega_;

   // double solNorm_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2VFESpace_;
   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   Array<HYPRE_Int> blockTrueOffsets_;

   Array2D<ParSesquilinearForm*>      a_;
   Array2D<ParMixedSesquilinearForm*> am_;
   Array<ParBilinearForm*>            b_;

   Array<ParComplexGridFunction*> e_;   // Complex electric field (HCurl)
   Array<ParComplexGridFunction*> j_;   // Complex current density (HCurl)
   Array<ParComplexLinearForm*> rhs_; // Dual of complex current density (HCurl)
   ParGridFunction        * e_t_; // Time dependent Electric field
   Array<ParComplexGridFunction*> e_v_; // Complex electric field (L2^d)
   ParComplexGridFunction * j_v_; // Complex current density (L2^d)
   ParGridFunction        * u_;   // Energy density (L2)
   ParComplexGridFunction * S_;  // Poynting Vector (HDiv)

   MatrixCoefficient * epsReCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsImCoef_;    // Dielectric Material Coefficient
   MatrixCoefficient * epsAbsCoef_;   // Dielectric Material Coefficient
   Coefficient       * muInvCoef_;    // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaInvCoef_;   // Admittance Coefficient
   Coefficient       * etaInvReCoef_; // Real Admittance Coefficient
   Coefficient       * etaInvImCoef_; // Imaginary Admittance Coefficient
   VectorCoefficient * kCoef_;        // Wave Vector

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
   Coefficient * negMuInvCoef_;  // -1.0 / mu

   MatrixCoefficient * massReCoef_;       // -omega^2 Re(epsilon)
   MatrixCoefficient * massImCoef_;       // omega^2 Im(epsilon)
   MatrixCoefficient * posMassCoef_;      // omega^2 Abs(epsilon)
   MatrixCoefficient * negMuInvkxkxCoef_; // -\vec{k}\times\vec{k}\times/mu

   // Coefficients for 2D system
   MatrixCoefficient * massRe2x2Coef_;     // -omega^2 Re(epsilon)
   VectorCoefficient * massRe2x1Coef_;     // -omega^2 Re(epsilon)
   VectorCoefficient * massRe1x2Coef_;     // -omega^2 Re(epsilon)
   Coefficient       * massReZZCoef_;      // -omega^2 Re(epsilon)
   MatrixCoefficient * massIm2x2Coef_;     // omega^2 Im(epsilon)
   VectorCoefficient * massIm2x1Coef_;     // omega^2 Im(epsilon)
   VectorCoefficient * massIm1x2Coef_;     // omega^2 Im(epsilon)
   Coefficient       * massImZZCoef_;      // omega^2 Im(epsilon)
   MatrixCoefficient * posMass2x2Coef_;    // omega^2 Abs(epsilon)
   Coefficient       * posMassZZCoef_;     // omega^2 Abs(epsilon)

   VectorCoefficient * negMuInvkCoef_; // -\vec{k}/mu
   VectorCoefficient * jrCoef_;        // Volume Current Density Function
   VectorCoefficient * jiCoef_;        // Volume Current Density Function
   VectorCoefficient * rhsrCoef_;      // Volume Current Density Function
   VectorCoefficient * rhsiCoef_;      // Volume Current Density Function

   VectorCoefficient * jr2x1Coef_;        // Volume Current Density Function
   VectorCoefficient * ji2x1Coef_;        // Volume Current Density Function
   Coefficient       * jrZCoef_;        // Volume Current Density Function
   Coefficient       * jiZCoef_;        // Volume Current Density Function
   VectorCoefficient * rhsr2x1Coef_;     // Volume Current Density Function
   VectorCoefficient * rhsi2x1Coef_;     // Volume Current Density Function
   Coefficient       * rhsrZCoef_;       // Volume Current Density Function
   Coefficient       * rhsiZCoef_;       // Volume Current Density Function

   VectorGridFunctionCoefficient erCoef_;
   VectorGridFunctionCoefficient eiCoef_;

   CurlGridFunctionCoefficient derCoef_;
   CurlGridFunctionCoefficient deiCoef_;

   EnergyDensityCoef     uCoef_;
   PoyntingVectorReCoef SrCoef_;
   PoyntingVectorImCoef SiCoef_;

   // const VectorCoefficient & erCoef_; // Electric Field Boundary Condition
   // const VectorCoefficient & eiCoef_; // Electric Field Boundary Condition

   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker_;

   // Array of 0's and 1's marking the location of sheath surfaces
   Array<int> sbc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker_;
   // void   (*e_r_bc_)(const Vector&, Vector&);
   // void   (*e_i_bc_)(const Vector&, Vector&);

   // Array<int> * dbcs_;
   Array<ComplexVectorCoefficientByAttr> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_nd_tdofs_;
   Array<int> ess_bdr_h1_tdofs_;
   Array<int> non_k_bdr_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_COLD_PLASMA_DIELECTRIC_SOLVER
