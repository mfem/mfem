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

void ElasticityComponentAddMultPA(
   const int dim, const int nDofs, const FiniteElementSpace &fespace,
   const DofToQuad &maps, const Vector &pa_data, const Vector &x,
   QuadratureFunction &QVec, Vector &y,
   const int i_block, const int j_block)
{
   const int id = (dim << 8) | (i_block << 4) | j_block;
   switch (id)
   {
      case 0x200:
         ElasticityAddMultPA_<2,0,0>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x201:
         ElasticityAddMultPA_<2,0,1>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x210:
         ElasticityAddMultPA_<2,1,0>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x211:
         ElasticityAddMultPA_<2,1,1>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x300:
         ElasticityAddMultPA_<3,0,0>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x301:
         ElasticityAddMultPA_<3,0,1>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x302:
         ElasticityAddMultPA_<3,0,2>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x310:
         ElasticityAddMultPA_<3,1,0>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x311:
         ElasticityAddMultPA_<3,1,1>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x312:
         ElasticityAddMultPA_<3,1,2>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x320:
         ElasticityAddMultPA_<3,2,0>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x321:
         ElasticityAddMultPA_<3,2,1>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      case 0x322:
         ElasticityAddMultPA_<3,2,2>(nDofs, fespace, maps, pa_data,
                                     x, QVec, y);
         return;
      default:
         MFEM_ABORT("Invalid elasticity component block.");
   }
}

#define MFEM_ELASTICITY_TENSOR_CASE_2D(D1D, Q1D) \
   case ((D1D << 8) | Q1D): \
      ElasticityAddMultPATensor2D_<D1D, Q1D>(numEls, maps, pa_data, x, y); \
      return true

#define MFEM_ELASTICITY_TENSOR_CASE_3D(D1D, Q1D) \
   case ((D1D << 8) | Q1D): \
      ElasticityAddMultPATensor3D_<D1D, Q1D>(numEls, maps, pa_data, x, y); \
      return true

bool ElasticityAddMultPATensor(const int dim, const int numEls,
                               const DofToQuad &maps,
                               const Vector &pa_data,
                               const Vector &x, Vector &y)
{
   if (maps.mode != DofToQuad::TENSOR || maps.ndof > maps.nqpt)
   {
      return false;
   }
   const int id = (maps.ndof << 8) | maps.nqpt;

   if (dim == 2)
   {
      switch (id)
      {
         MFEM_ELASTICITY_TENSOR_CASE_2D(2,2);
         MFEM_ELASTICITY_TENSOR_CASE_2D(2,3);
         MFEM_ELASTICITY_TENSOR_CASE_2D(2,4);
         MFEM_ELASTICITY_TENSOR_CASE_2D(3,3);
         MFEM_ELASTICITY_TENSOR_CASE_2D(3,4);
         MFEM_ELASTICITY_TENSOR_CASE_2D(3,5);
         MFEM_ELASTICITY_TENSOR_CASE_2D(4,4);
         MFEM_ELASTICITY_TENSOR_CASE_2D(4,5);
         MFEM_ELASTICITY_TENSOR_CASE_2D(4,6);
         MFEM_ELASTICITY_TENSOR_CASE_2D(5,5);
         MFEM_ELASTICITY_TENSOR_CASE_2D(5,6);
         MFEM_ELASTICITY_TENSOR_CASE_2D(5,7);
         MFEM_ELASTICITY_TENSOR_CASE_2D(6,6);
         MFEM_ELASTICITY_TENSOR_CASE_2D(6,7);
         MFEM_ELASTICITY_TENSOR_CASE_2D(6,8);
         MFEM_ELASTICITY_TENSOR_CASE_2D(7,7);
         MFEM_ELASTICITY_TENSOR_CASE_2D(7,8);
         MFEM_ELASTICITY_TENSOR_CASE_2D(7,9);
         MFEM_ELASTICITY_TENSOR_CASE_2D(8,8);
         MFEM_ELASTICITY_TENSOR_CASE_2D(8,9);
         MFEM_ELASTICITY_TENSOR_CASE_2D(8,10);
         default:
            return false;
      }
   }
   if (dim == 3)
   {
      switch (id)
      {
         MFEM_ELASTICITY_TENSOR_CASE_3D(2,2);
         MFEM_ELASTICITY_TENSOR_CASE_3D(2,3);
         MFEM_ELASTICITY_TENSOR_CASE_3D(2,4);
         MFEM_ELASTICITY_TENSOR_CASE_3D(3,3);
         MFEM_ELASTICITY_TENSOR_CASE_3D(3,4);
         MFEM_ELASTICITY_TENSOR_CASE_3D(3,5);
         MFEM_ELASTICITY_TENSOR_CASE_3D(4,4);
         MFEM_ELASTICITY_TENSOR_CASE_3D(4,5);
         MFEM_ELASTICITY_TENSOR_CASE_3D(4,6);
         MFEM_ELASTICITY_TENSOR_CASE_3D(5,5);
         MFEM_ELASTICITY_TENSOR_CASE_3D(5,6);
         MFEM_ELASTICITY_TENSOR_CASE_3D(5,7);
         MFEM_ELASTICITY_TENSOR_CASE_3D(6,6);
         MFEM_ELASTICITY_TENSOR_CASE_3D(6,7);
         MFEM_ELASTICITY_TENSOR_CASE_3D(7,7);
         default:
            return false;
      }
   }
   return false;
}

#undef MFEM_ELASTICITY_TENSOR_CASE_2D
#undef MFEM_ELASTICITY_TENSOR_CASE_3D

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
} // namespace mfem
