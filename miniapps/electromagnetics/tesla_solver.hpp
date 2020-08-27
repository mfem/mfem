// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TESLA_SOLVER
#define MFEM_TESLA_SOLVER

#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"
#include "electromagnetics.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::DivergenceFreeProjector;

namespace electromagnetics
{

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
   ParMixedBilinearForm * hDivHCurlMuInv_;
   ParMixedBilinearForm * weakCurlMuInv_;

   ParDiscreteGradOperator * grad_;
   ParDiscreteCurlOperator * curl_;

   ParGridFunction * a_;  // Vector Potential (HCurl)
   ParGridFunction * b_;  // Magnetic Flux (HDiv)
   ParGridFunction * h_;  // Magnetic Field (HCurl)
   ParGridFunction * jr_; // Raw Volumetric Current Density (HCurl)
   ParGridFunction * j_;  // Volumetric Current Density (HCurl)
   ParGridFunction * k_;  // Surface Current Density (HCurl)
   ParGridFunction * m_;  // Magnetization (HDiv)
   ParGridFunction * bd_; // Dual of B (HCurl)
   ParGridFunction * jd_; // Dual of J, the rhs vector (HCurl)

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
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   std::map<std::string,socketstream*> socks_;
};

class SurfaceCurrent
{
public:
   SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
                  ParDiscreteGradOperator & Grad,
                  Array<int> & kbcs, Array<int> & vbcs, Vector & vbcv);
   ~SurfaceCurrent();

   void InitSolver() const;

   void ComputeSurfaceCurrent(ParGridFunction & k);

   void Update();

   ParGridFunction * GetPsi() { return psi_; }

private:
   int myid_;

   ParFiniteElementSpace   * H1FESpace_;
   ParDiscreteGradOperator * grad_;
   Array<int>              * kbcs_;
   Array<int>              * vbcs_;
   Vector                  * vbcv_;

   ParBilinearForm * s0_;
   ParGridFunction * psi_;
   ParGridFunction * rhs_;

   HypreParMatrix  * S0_;
   mutable Vector Psi_;
   mutable Vector RHS_;

   mutable HypreBoomerAMG  * amg_;
   mutable HyprePCG        * pcg_;

   Array<int> ess_bdr_, ess_bdr_tdofs_;
   Array<int> non_k_bdr_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TESLA_SOLVER
