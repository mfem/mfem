// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../bilininteg.hpp"
#include "bilininteg_elasticity_kernels.hpp"

namespace mfem
{

namespace internal
{

void ElasticitySetupPAData(const int dim, const IntegrationRule &ir,
                           const CoefficientVector &lambda,
                           const CoefficientVector &mu,
                           const GeometricFactors &geom,
                           Vector &pa_data)
{
   const int entries = dim*dim + 2;
   pa_data.SetSize(entries*lambda.Size());
   pa_data.UseDevice(true);
   if (dim == 2)
   {
      ElasticitySetupPAData_<2>(ir, lambda, mu, geom, pa_data);
   }
   else if (dim == 3)
   {
      ElasticitySetupPAData_<3>(ir, lambda, mu, geom, pa_data);
   }
   else
   {
      MFEM_ABORT("Elasticity PA is implemented only in dimensions 2 and 3.");
   }
}

void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace,
                         const DofToQuad &maps,
                         const Vector &pa_data,
                         const Vector &x,
                         QuadratureFunction &QVec,
                         Vector &y)
{
   if (dim == 2)
   {
      ElasticityAddMultPA_<2>(nDofs, fespace, maps, pa_data, x, QVec, y);
   }
   else if (dim == 3)
   {
      ElasticityAddMultPA_<3>(nDofs, fespace, maps, pa_data, x, QVec, y);
   }
   else
   {
      MFEM_ABORT("Elasticity PA is implemented only in dimensions 2 and 3.");
   }
}

void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const DofToQuad &maps,
                                  const IntegrationRule &ir,
                                  const Vector &pa_data,
                                  Vector &diag)
{
   if (dim == 2)
   {
      ElasticityAssembleDiagonalPA_<2>(nDofs, maps, ir, pa_data, diag);
   }
   else if (dim == 3)
   {
      ElasticityAssembleDiagonalPA_<3>(nDofs, maps, ir, pa_data, diag);
   }
   else
   {
      MFEM_ABORT("Elasticity PA is implemented only in dimensions 2 and 3.");
   }
}

void ElasticityAssembleEA(const int dim, const int i_block,
                          const int j_block, const int nDofs,
                          const IntegrationRule &ir,
                          const DofToQuad &maps,
                          const Vector &pa_data,
                          Vector &emat, const bool add)
{
   if (dim == 2)
   {
      ElasticityAssembleEA_<2>(i_block, j_block, nDofs, ir, maps,
                               pa_data, emat, add);
   }
   else if (dim == 3)
   {
      ElasticityAssembleEA_<3>(i_block, j_block, nDofs, ir, maps,
                               pa_data, emat, add);
   }
   else
   {
      MFEM_ABORT("Elasticity EA is implemented only in dimensions 2 and 3.");
   }
}

} // namespace internal

/// \cond DO_NOT_DOCUMENT

template<int DIM, int D1D, int Q1D>
ElasticityIntegrator::ApplyKernelType
ElasticityIntegrator::ApplyPAKernels::Kernel()
{
   if constexpr (DIM == 2)
   {
      return internal::ElasticityAddMultPATensor2D_<D1D, Q1D>;
   }
   else if constexpr (DIM == 3)
   {
      return internal::ElasticityAddMultPATensor3D_<D1D, Q1D>;
   }
   MFEM_ABORT("Elasticity tensor PA is implemented only in dimensions 2 and 3.");
   return nullptr;
}

ElasticityIntegrator::ApplyKernelType
ElasticityIntegrator::ApplyPAKernels::Fallback(int dim, int, int)
{
   MFEM_CONTRACT_VAR(dim);
   MFEM_ABORT("No specialized elasticity tensor PA kernel for this (D1D, Q1D).");
   return nullptr;
}

ElasticityIntegrator::Kernels::Kernels()
{
   // 2D: Q = P+1, P+2, P+3
   ElasticityIntegrator::AddSpecialization<2,2,2>();
   ElasticityIntegrator::AddSpecialization<2,2,3>();
   ElasticityIntegrator::AddSpecialization<2,2,4>();
   ElasticityIntegrator::AddSpecialization<2,3,3>();
   ElasticityIntegrator::AddSpecialization<2,3,4>();
   ElasticityIntegrator::AddSpecialization<2,3,5>();
   ElasticityIntegrator::AddSpecialization<2,4,4>();
   ElasticityIntegrator::AddSpecialization<2,4,5>();
   ElasticityIntegrator::AddSpecialization<2,4,6>();
   ElasticityIntegrator::AddSpecialization<2,5,5>();
   ElasticityIntegrator::AddSpecialization<2,5,6>();
   ElasticityIntegrator::AddSpecialization<2,5,7>();
   ElasticityIntegrator::AddSpecialization<2,6,6>();
   ElasticityIntegrator::AddSpecialization<2,6,7>();
   ElasticityIntegrator::AddSpecialization<2,6,8>();
   ElasticityIntegrator::AddSpecialization<2,7,7>();
   ElasticityIntegrator::AddSpecialization<2,7,8>();
   ElasticityIntegrator::AddSpecialization<2,7,9>();
   ElasticityIntegrator::AddSpecialization<2,8,8>();
   ElasticityIntegrator::AddSpecialization<2,8,9>();
   ElasticityIntegrator::AddSpecialization<2,8,10>();
   // 3D
   ElasticityIntegrator::AddSpecialization<3,2,2>();
   ElasticityIntegrator::AddSpecialization<3,2,3>();
   ElasticityIntegrator::AddSpecialization<3,2,4>();
   ElasticityIntegrator::AddSpecialization<3,3,3>();
   ElasticityIntegrator::AddSpecialization<3,3,4>();
   ElasticityIntegrator::AddSpecialization<3,3,5>();
   ElasticityIntegrator::AddSpecialization<3,4,4>();
   ElasticityIntegrator::AddSpecialization<3,4,5>();
   ElasticityIntegrator::AddSpecialization<3,4,6>();
   ElasticityIntegrator::AddSpecialization<3,5,5>();
   ElasticityIntegrator::AddSpecialization<3,5,6>();
   ElasticityIntegrator::AddSpecialization<3,5,7>();
   ElasticityIntegrator::AddSpecialization<3,6,6>();
   ElasticityIntegrator::AddSpecialization<3,6,7>();
   ElasticityIntegrator::AddSpecialization<3,7,7>();
}

template<int DIM, int I, int J>
ElasticityComponentIntegrator::ApplyKernelType
ElasticityComponentIntegrator::ApplyPAKernels::Kernel()
{
   return internal::ElasticityAddMultPA_<DIM, I, J>;
}

ElasticityComponentIntegrator::ApplyKernelType
ElasticityComponentIntegrator::ApplyPAKernels::Fallback(int dim, int i, int j)
{
   MFEM_ABORT("Invalid elasticity component block (" << dim << ", "
              << i << ", " << j << ").");
   return nullptr;
}

ElasticityComponentIntegrator::Kernels::Kernels()
{
   ElasticityComponentIntegrator::AddSpecialization<2,0,0>();
   ElasticityComponentIntegrator::AddSpecialization<2,0,1>();
   ElasticityComponentIntegrator::AddSpecialization<2,1,0>();
   ElasticityComponentIntegrator::AddSpecialization<2,1,1>();
   ElasticityComponentIntegrator::AddSpecialization<3,0,0>();
   ElasticityComponentIntegrator::AddSpecialization<3,0,1>();
   ElasticityComponentIntegrator::AddSpecialization<3,0,2>();
   ElasticityComponentIntegrator::AddSpecialization<3,1,0>();
   ElasticityComponentIntegrator::AddSpecialization<3,1,1>();
   ElasticityComponentIntegrator::AddSpecialization<3,1,2>();
   ElasticityComponentIntegrator::AddSpecialization<3,2,0>();
   ElasticityComponentIntegrator::AddSpecialization<3,2,1>();
   ElasticityComponentIntegrator::AddSpecialization<3,2,2>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
