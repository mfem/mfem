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

#ifndef MFEM_HERTZ_SOLVER
#define MFEM_HERTZ_SOLVER

#include "../common/fem_extras.hpp"

#include <string>
#include <map>

namespace mfem
{

using common::H1_FESpace;
using common::ND_FESpace;
using common::RT_FESpace;
using common::DiscreteGradOperator;
using common::DiscreteCurlOperator;
//using common::DivergenceFreeProjector;

namespace electromagnetics
{

// Physical Constants
// Permittivity of Free Space (units F/m)
static const double epsilon0_ = 8.8541878176e-12;
// Permeability of Free Space (units H/m)
static const double mu0_ = 4.0e-7*M_PI;

//class SurfaceCurrent;
class HertzSolver
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

   HertzSolver(Mesh & mesh, int order, double freq,
               HertzSolver::SolverType s,
               ComplexOperator::Convention conv,
               Coefficient & epsCoef,
               Coefficient & muInvCoef,
               Coefficient * sigmaCoef,
               Coefficient * etaInvCoef,
               Array<int> & abcs,
               Array<int> & dbcs,
               void   (*e_r_bc )(const Vector&, Vector&),
               void   (*e_i_bc )(const Vector&, Vector&),
               void   (*j_r_src)(const Vector&, Vector&),
               void   (*j_i_src)(const Vector&, Vector&));
   ~HertzSolver();

   int GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   //void GetErrorEstimates(Vector & errors);

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   BilinearFormIntegrator * GetErrorInteg() { return err_integ_; }

   GridFunction * GetErrorField() { return &e_->real(); }

   FiniteElementSpace * GetErrorFluxFES()
   {
      return new FiniteElementSpace(mesh_, err_flux_fec_);
   }
   // const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   int order_;
   int logging_;

   SolverType sol_;

   ComplexOperator::Convention conv_;

   bool ownsEtaInv_;

   double freq_;

   Mesh * mesh_;

   // H1_ParFESpace * H1FESpace_;
   ND_FESpace * HCurlFESpace_;
   // RT_ParFESpace * HDivFESpace_;

   Array<int> blockTrueOffsets_;

   // ParSesquilinearForm * a0_;
   SesquilinearForm * a1_;
   BilinearForm * b1_;

   // ParGridFunction * e_r_;  // Real part of electric field (HCurl)
   // ParGridFunction * e_i_;  // Imaginary part of electric field (HCurl)
   ComplexGridFunction * e_;  // Complex electric field (HCurl)
   ComplexGridFunction * j_;  // Complex current density (HCurl)
   // ParGridFunction * j_i_;  // Imaginary part of current density (HCurl)

   ComplexLinearForm   * jd_; // Dual of complex current density (HCurl)
   // ParLinearForm   * jd_r_; // Dual of real part of current density (HCurl)
   // ParLinearForm   * jd_i_; // Dual of imaginary part of current density (HCurl)

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
   Coefficient       * etaInvCoef_; // Admittance Coefficient

   Coefficient * omegaCoef_;     // omega expressed as a Coefficient
   Coefficient * negOmegaCoef_;  // -omega expressed as a Coefficient
   Coefficient * omega2Coef_;    // omega^2 expressed as a Coefficient
   Coefficient * negOmega2Coef_; // -omega^2 expressed as a Coefficient
   Coefficient * massCoef_;      // -omega^2 epsilon
   Coefficient * posMassCoef_;   // omega^2 epsilon
   Coefficient * lossCoef_;      // -omega sigma
   // Coefficient * gainCoef_;    // omega sigma
   Coefficient * abcCoef_;       // -omega eta^{-1}

   // VectorCoefficient * aBCCoef_;   // Vector Potential BC Function
   VectorCoefficient * jrCoef_;     // Volume Current Density Function
   VectorCoefficient * jiCoef_;     // Volume Current Density Function
   VectorCoefficient * erCoef_;     // Electric Field Boundary Condition
   VectorCoefficient * eiCoef_;     // Electric Field Boundary Condition
   // VectorCoefficient * mCoef_;     // Magnetization Vector Function

   BilinearFormIntegrator * err_integ_;
   FiniteElementCollection * err_flux_fec_;

   // void   (*a_bc_ )(const Vector&, Vector&);
   void   (*j_r_src_)(const Vector&, Vector&);
   void   (*j_i_src_)(const Vector&, Vector&);
   // void   (*m_src_)(const Vector&, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker_;
   void   (*e_r_bc_)(const Vector&, Vector&);
   void   (*e_i_bc_)(const Vector&, Vector&);

   Array<int> * dbcs_;
   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   VisItDataCollection * visit_dc_;

   std::map<std::string,socketstream*> socks_;
};
/*
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
*/
} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_HERTZ_SOLVER
