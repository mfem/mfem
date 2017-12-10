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

#ifndef MFEM_HERTZ_SOLVER
#define MFEM_HERTZ_SOLVER

#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_MPI

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
// Permittivity of Free Space (units F/m)
static double epsilon0_ = 8.8541878176e-12;
// Permeability of Free Space (units H/m)
static double mu0_ = 4.0e-7*M_PI;

  //class SurfaceCurrent;
class HertzSolver
{
public:
   HertzSolver(ParMesh & pmesh, int order, Array<int> & kbcs,
               Array<int> & vbcs, Vector & vbcv,
               Coefficient & epsCoef,
               Coefficient & muInvCoef,
               Coefficient * sigmaCoef,
               void   (*e_r_bc )(const Vector&, Vector&),
               void   (*e_i_bc )(const Vector&, Vector&),
               void   (*j_r_src)(const Vector&, Vector&),
               void   (*j_i_src)(const Vector&, Vector&));
   ~HertzSolver();

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

  // const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   int myid_;
   int num_procs_;
   int order_;
   int logging_;
  
   double freq_;
  
   ParMesh * pmesh_;

   // H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   // RT_ParFESpace * HDivFESpace_;

   ParSesquilinearForm * a1_;

   ParGridFunction * e_r_;  // Real part of electric field (HCurl)
   ParGridFunction * e_i_;  // Imaginary part of electric field (HCurl)
   ParGridFunction * e_;  // Complex electric field (HCurl)
   ParGridFunction * j_r_;  // Real part of current density (HCurl)
   ParGridFunction * j_i_;  // Imaginary part of current density (HCurl)

   ParLinearForm   * jd_r_; // Dual of real part of current density (HCurl)
   ParLinearForm   * jd_i_; // Dual of imaginary part of current density (HCurl)
  /*
   ParBilinearForm * curlMuInvCurl_;
   ParBilinearForm * hCurlMass_;
   ParMixedBilinearForm * hDivHCurlMuInv_;
   ParMixedBilinearForm * weakCurlMuInv_;
  */
  // ParDiscreteGradOperator * grad_;
  // ParDiscreteCurlOperator * curl_;
  /*
   ParGridFunction * a_;  // Vector Potential (HCurl)
   ParGridFunction * b_;  // Magnetic Flux (HDiv)
   ParGridFunction * h_;  // Magnetic Field (HCurl)
   ParGridFunction * jr_; // Raw Volumetric Current Density (HCurl)
   ParGridFunction * j_;  // Volumetric Current Density (HCurl)
   ParGridFunction * k_;  // Surface Current Density (HCurl)
   ParGridFunction * m_;  // Magnetization (HDiv)
   ParGridFunction * bd_; // Dual of B (HCurl)
   ParGridFunction * jd_; // Dual of J, the rhs vector (HCurl)
  */
  // DivergenceFreeProjector * DivFreeProj_;
  // SurfaceCurrent          * SurfCur_;

   Coefficient       * epsCoef_;   // Dielectric Material Coefficient
   Coefficient       * muInvCoef_; // Dia/Paramagnetic Material Coefficient
   Coefficient       * sigmaCoef_; // Electrical Conductivity Coefficient
  // VectorCoefficient * aBCCoef_;   // Vector Potential BC Function
   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
  // VectorCoefficient * mCoef_;     // Magnetization Vector Function

  // void   (*a_bc_ )(const Vector&, Vector&);
   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);
  // void   (*m_src_)(const Vector&, Vector&);

   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   VisItDataCollection * visit_dc_;

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

#endif // MFEM_HERTZ_SOLVER
