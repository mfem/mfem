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
               void   (*p_src  )(const Vector&, Vector&));
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

   int myid_;
   int num_procs_;
   int order_;

   ParMesh * pmesh_;

   Array<int> * dbcs_;
   Vector     * dbcv_;
   Array<int> * nbcs_;
   Vector     * nbcv_;

   VisItDataCollection * visit_dc_;

   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   ParBilinearForm * divEpsGrad_;
   ParBilinearForm * h1Mass_;
   ParBilinearForm * h1SurfMass_;
   ParBilinearForm * hCurlMass_;
   ParBilinearForm * hDivMass_;
   ParMixedBilinearForm * hCurlHDivEps_;
   ParMixedBilinearForm * hCurlHDiv_;

   ParDiscreteGradOperator * Grad_;

   ParGridFunction * phi_;
   ParGridFunction * rho_;
   ParGridFunction * sigma_;
   ParGridFunction * e_;
   ParGridFunction * d_;
   ParGridFunction * p_;

   Coefficient       * epsCoef_;
   Coefficient       * phiBCCoef_;
   Coefficient       * rhoCoef_;
   VectorCoefficient * pCoef_;

   double (*phi_bc_ )(const Vector&);
   double (*rho_src_)(const Vector&);
   void   (*p_src_  )(const Vector&, Vector&);

   std::map<std::string,socketstream*> socks_;

   Array<int> ess_bdr_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VOLTA_SOLVER
