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

#ifndef MFEM_VOLTA_SOLVER
#define MFEM_VOLTA_SOLVER

#include "../common/pfem_extras.hpp"
#include "electromagnetics.hpp"

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
using miniapps::ParDiscreteDivOperator;

namespace electromagnetics
{

class VoltaSolver
{
public:
   VoltaSolver(ParMesh & pmesh, int order,
               Array<int> & dbcs, Vector & dbcv,
               Array<int> & nbcs, Vector & nbcv,
               Coefficient & epsCoef,
               double (*phi_bc )(const Vector&),
               double (*rho_src)(const Vector&),
               void   (*p_src  )(const Vector&, Vector&),
               Vector & point_charges);
   ~VoltaSolver();

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

   const ParGridFunction & GetVectorPotential() { return *phi_; }

private:

   int myid_;      // Local processor rank
   int num_procs_; // Number of processors
   int order_;     // Basis function order

   ParMesh * pmesh_;

   Array<int> * dbcs_; // Dirichlet BC Surface Attribute IDs
   Vector     * dbcv_; // Corresponding Dirichlet Values
   Array<int> * nbcs_; // Neumann BC Surface Attribute IDs
   Vector     * nbcv_; // Corresponding Neumann Values

   VisItDataCollection * visit_dc_; // To prepare fields for VisIt viewing

   H1_ParFESpace * H1FESpace_;    // Continuous space for phi
   ND_ParFESpace * HCurlFESpace_; // Tangentially continuous space for E
   RT_ParFESpace * HDivFESpace_;  // Normally continuous space for D
   L2_ParFESpace * L2FESpace_;    // Discontinuous space for rho

   ParBilinearForm * divEpsGrad_; // Laplacian operator
   ParBilinearForm * h1Mass_;     // For Volumetric Charge Density Source
   ParBilinearForm * h1SurfMass_; // For Surface Charge Density Source
   ParBilinearForm * hDivMass_;   // For Computing D from E

   ParMixedBilinearForm * hCurlHDivEps_; // For computing D from E
   ParMixedBilinearForm * hCurlHDiv_;    // For computing D from E and P
   ParMixedBilinearForm * weakDiv_;      // For computing the source term from P

   ParLinearForm * rhod_; // Dual of Volumetric Charge Density Source

   ParLinearForm * l2_vol_int_;  // Integral of L2 field
   ParLinearForm * rt_surf_int_; // Integral of H(Div) field over boundary

   ParDiscreteGradOperator * grad_; // For Computing E from phi
   ParDiscreteDivOperator  * div_;  // For Computing rho from D

   ParGridFunction * phi_;       // Electric Scalar Potential
   ParGridFunction * rho_src_;   // Volumetric Charge Density Source
   ParGridFunction * rho_;       // Volumetric Charge Density (Div(D))
   ParGridFunction * sigma_src_; // Surface Charge Density Source
   ParGridFunction * e_;         // Electric Field
   ParGridFunction * d_;         // Electric Flux Density (aka Dielectric Flux)
   ParGridFunction * p_src_;     // Polarization Field Source

   ConstantCoefficient oneCoef_;   // Coefficient equal to 1
   Coefficient       * epsCoef_;   // Dielectric Permittivity Coefficient
   Coefficient       * phiBCCoef_; // Scalar Potential Boundary Condition
   Coefficient       * rhoCoef_;   // Charge Density Coefficient
   VectorCoefficient * pCoef_;     // Polarization Vector Field Coefficient

   // Source functions
   double (*phi_bc_func_ )(const Vector&);          // Scalar Potential BC
   double (*rho_src_func_)(const Vector&);          // Volumetric Charge Density
   void   (*p_src_func_  )(const Vector&, Vector&); // Polarization Field

   const Vector & point_charge_params_;

   std::vector<DeltaCoefficient*> point_charges_;

   std::map<std::string,socketstream*> socks_; // Visualization sockets

   Array<int> ess_bdr_, ess_bdr_tdofs_; // Essential Boundary Condition DoFs
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VOLTA_SOLVER
