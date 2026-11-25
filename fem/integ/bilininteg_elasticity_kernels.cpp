// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bilininteg_elasticity_kernels.hpp"
#include "../kernel_dispatch.hpp"

namespace mfem
{

namespace internal
{

/// \cond DO_NOT_DOCUMENT

ElasticityComponentAddMultPAKernelsType
ElasticityComponentAddMultPAKernels::Fallback(int, int, int)
{
   MFEM_ABORT("Invalid configuration.");
}

/// Register the existing specializations once, on first use.
static void RegisterElasticityComponentAddMultPAKernels()
{
   struct Kernels
   {
      Kernels()
      {
         ElasticityComponentAddMultPAKernels::Specialization<2,0,0>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<2,0,1>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<2,1,0>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<2,1,1>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,0,0>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,0,1>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,0,2>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,1,0>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,1,1>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,1,2>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,2,0>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,2,1>::Add();
         ElasticityComponentAddMultPAKernels::Specialization<3,2,2>::Add();
      }
   };
   static Kernels kernels;
}

ElasticityAddMultPAKernelsType ElasticityAddMultPAKernels::Fallback(int)
{
   MFEM_ABORT("Only dimensions 2 and 3 supported.");
}

/// Register the existing specializations once, on first use.
static void RegisterElasticityAddMultPAKernels()
{
   struct Kernels
   {
      Kernels()
      {
         ElasticityAddMultPAKernels::Specialization<2>::Add();
         ElasticityAddMultPAKernels::Specialization<3>::Add();
      }
   };
   static Kernels kernels;
}

ElasticityAssembleDiagonalPAKernelsType
ElasticityAssembleDiagonalPAKernels::Fallback(int)
{
   MFEM_ABORT("Only dimensions 2 and 3 supported.");
}

/// Register the existing specializations once, on first use.
static void RegisterElasticityAssembleDiagonalPAKernels()
{
   struct Kernels
   {
      Kernels()
      {
         ElasticityAssembleDiagonalPAKernels::Specialization<2>::Add();
         ElasticityAssembleDiagonalPAKernels::Specialization<3>::Add();
      }
   };
   static Kernels kernels;
}

ElasticityAssembleEAKernelsType ElasticityAssembleEAKernels::Fallback(int)
{
   MFEM_ABORT("Only dimensions 2 and 3 supported.");
}

/// Register the existing specializations once, on first use.
static void RegisterElasticityAssembleEAKernels()
{
   struct Kernels
   {
      Kernels()
      {
         ElasticityAssembleEAKernels::Specialization<2>::Add();
         ElasticityAssembleEAKernels::Specialization<3>::Add();
      }
   };
   static Kernels kernels;
}

/// \endcond DO_NOT_DOCUMENT

void ElasticityComponentAddMultPA(const int dim, const int nDofs,
                                  const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y,
                                  const int i_block, const int j_block)
{
   RegisterElasticityComponentAddMultPAKernels();
   ElasticityComponentAddMultPAKernels::Run(
      dim, i_block, j_block,
      nDofs,fespace,lambda,mu,geom,maps,x,QVec,y);
}

void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                         const CoefficientVector &mu, const GeometricFactors &geom,
                         const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y)
{
   RegisterElasticityAddMultPAKernels();
   ElasticityAddMultPAKernels::Run(
      dim,
      nDofs,fespace,lambda,mu,geom,maps,x,QVec,y);
}

void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, const IntegrationRule &ir, Vector &diag)
{
   RegisterElasticityAssembleDiagonalPAKernels();
   ElasticityAssembleDiagonalPAKernels::Run(
      dim,
      nDofs,lambda,mu,geom,maps,ir,diag);
}

void ElasticityAssembleEA(const int dim, const int i_block, const int j_block,
                          const int nDofs, const IntegrationRule &ir,
                          const CoefficientVector &lambda,
                          const CoefficientVector &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat)
{
   RegisterElasticityAssembleEAKernels();
   ElasticityAssembleEAKernels::Run(
      dim,
      i_block,j_block,nDofs,ir,lambda,mu,geom,maps,emat);
}

} // namespace internal

} // namespace mfem
