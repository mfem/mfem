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

#ifndef MFEM_CPD1D_SOLVER
#define MFEM_CPD1D_SOLVER

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

/// Cold Plasma Dielectric Solver for 1D geometries
class CPD1DSolver
{
public:

   enum SolverType
   {
      INVALID   = -1,
      GMRES     =  1,
      FGMRES    =  2,
      MINRES    =  3,
      SUPERLU   =  4,
      STRUMPACK =  5
   };

   CPD1DSolver(ParMesh & pmesh, int order, double omega,
               CPD1DSolver::SolverType s,
               ComplexOperator::Convention conv,
               MatrixCoefficient & epsReCoef,
               MatrixCoefficient & espImCoef,
               Coefficient & muInvCoef,
               Coefficient * etaInvCoef,
               Array<int> & abcs,
               Array<int> & dbcs,
               // void   (*e_r_bc )(const Vector&, Vector&),
               // void   (*e_i_bc )(const Vector&, Vector&),
               VectorCoefficient & EReCoef,
               VectorCoefficient & EImCoef,
               void   (*j_r_src)(const Vector&, Vector&),
               void   (*j_i_src)(const Vector&, Vector&));
   ~CPD1DSolver();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

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

   ComplexOperator::Convention conv_;

   bool ownsEtaInv_;

   double omega_;

   ParMesh * pmesh_;

   // H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   // RT_ParFESpace * HDivFESpace_;

   Array<HYPRE_Int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   ParSesquilinearForm * a1_;
   ParBilinearForm * b1_;

   ParComplexGridFunction * e_;  // Complex electric field (HCurl)
   ParComplexGridFunction * j_;  // Complex current density (HCurl)
   ParComplexLinearForm   * jd_; // Dual of complex current density (HCurl)

   MatrixCoefficient * epsReCoef_;  // Dielectric Material Coefficient
   MatrixCoefficient * epsImCoef_;  // Dielectric Material Coefficient
   Coefficient       * muInvCoef_;  // Dia/Paramagnetic Material Coefficient
   Coefficient       * etaInvCoef_; // Admittance Coefficient

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * abcCoef_;       // -omega eta^{-1}

   MatrixCoefficient * massReCoef_;  // -omega^2 Re(epsilon)
   MatrixCoefficient * massImCoef_;  // omega^2 Im(epsilon)
   MatrixCoefficient * posMassCoef_; // omega^2 Re(epsilon)

   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
   const VectorCoefficient & erCoef_;     // Electric Field Boundary Condition
   const VectorCoefficient & eiCoef_;     // Electric Field Boundary Condition

   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker_;
   // void   (*e_r_bc_)(const Vector&, Vector&);
   // void   (*e_i_bc_)(const Vector&, Vector&);

   Array<int> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_CPD1D_SOLVER
