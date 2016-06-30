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

#ifndef MFEM_TESLA_SOLVER
#define MFEM_TESLA_SOLVER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../common/pfem_extras.hpp"
#include <string>
#include <map>

namespace mfem
{

using miniapps::H1_ParFESpace;
using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
using miniapps::ParDiscreteGradOperator;
using miniapps::ParDiscreteCurlOperator;
using miniapps::DivergenceFreeProjector;

namespace electromagnetics
{

// Physical Constants
// Permeability of Free Space (units H/m)
static double mu0_ = 4.0e-7*M_PI;

class SurfaceCurrent;
class TeslaSolver
{
public:
   TeslaSolver(ParMesh & pmesh, int order, Array<int> & kbcs,
               Array<int> & vbcs, Vector & vbcv,
               Coefficient & muInvCoef,
               void   (*a_bc )(const Vector&, Vector&),
               void   (*j_src)(const Vector&, Vector&),
               void   (*m_src)(const Vector&, Vector&));
   ~TeslaSolver();

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

   const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   int myid_;
   int num_procs_;
   int order_;

   ParMesh * pmesh_;

   VisItDataCollection * visit_dc_;

   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   ParBilinearForm * curlMuInvCurl_;
   ParBilinearForm * hCurlMass_;
   ParBilinearForm * hDivMassMuInv_;
   ParMixedBilinearForm * hDivHCurlMuInv_;

   ParDiscreteGradOperator * Grad_;
   ParDiscreteCurlOperator * Curl_;

   ParGridFunction * a_;
   ParGridFunction * b_;
   ParGridFunction * h_;
   ParGridFunction * j_;
   ParGridFunction * k_;
   ParGridFunction * m_;

   DivergenceFreeProjector * DivFreeProj_;
   SurfaceCurrent          * SurfCur_;

   Coefficient       * muInvCoef_; // Dia/Paramagnetic Material Coefficient
   VectorCoefficient * aBCCoef_;   // Vector Potential BC Function
   VectorCoefficient * jCoef_;     // Volume Current Density Function
   VectorCoefficient * mCoef_;     // Magnetization Vector Function

   void   (*a_bc_ )(const Vector&, Vector&);
   void   (*j_src_)(const Vector&, Vector&);
   void   (*m_src_)(const Vector&, Vector&);

   Array<int> ess_bdr_;
   Array<int> non_k_bdr_;

   std::map<std::string,socketstream*> socks_;
};

class SurfaceCurrent
{
public:
   SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
                  ParFiniteElementSpace & HCurlFESpace,
                  ParDiscreteGradOperator & Grad,
                  Array<int> & kbcs, Array<int> & vbcs, Vector & vbcv);
   ~SurfaceCurrent();

   void ComputeSurfaceCurrent(ParGridFunction & k);

   void Update();

   ParGridFunction * GetPsi() { return psi_; }

private:
   int myid_;

   ParFiniteElementSpace   * H1FESpace_;
   ParFiniteElementSpace   * HCurlFESpace_;
   ParDiscreteGradOperator * Grad_;
   Array<int>              * kbcs_;
   Array<int>              * vbcs_;
   Vector                  * vbcv_;

   ParBilinearForm * s0_;
   ParGridFunction * psi_;

   HypreBoomerAMG  * amg_;
   HyprePCG        * pcg_;
   HypreParMatrix  * S0_;
   HypreParVector  * PSI_;
   HypreParVector  * RHS_;
   HypreParVector  * K_;

   Array<int> ess_bdr_;
   Array<int> non_k_bdr_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TESLA_SOLVER
