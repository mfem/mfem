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

void ElasticityComponentAddMultPA(const int dim, const int nDofs,
                                  const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y,
                                  const int i_block, const int j_block)
{
   const int id = (dim << 8)| (i_block << 4) | j_block;
   switch (id)
   {
      case 0x200:
         ElasticityAddMultPA_<2,0,0>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x211:
         ElasticityAddMultPA_<2,1,1>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x201:
         ElasticityAddMultPA_<2,0,1>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x210:
         ElasticityAddMultPA_<2,1,0>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x300:
         ElasticityAddMultPA_<3,0,0>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x311:
         ElasticityAddMultPA_<3,1,1>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x322:
         ElasticityAddMultPA_<3,2,2>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x301:
         ElasticityAddMultPA_<3,0,1>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x302:
         ElasticityAddMultPA_<3,0,2>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x312:
         ElasticityAddMultPA_<3,1,2>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x310:
         ElasticityAddMultPA_<3,1,0>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x320:
         ElasticityAddMultPA_<3,2,0>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 0x321:
         ElasticityAddMultPA_<3,2,1>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      default:
         MFEM_ABORT("Invalid configuration.");
   }
}

void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                         const CoefficientVector &mu, const GeometricFactors &geom,
                         const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y)
{
   switch (dim)
   {
      case 2:
         ElasticityAddMultPA_<2>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      case 3:
         ElasticityAddMultPA_<3>(nDofs, fespace, lambda, mu, geom, maps, x, QVec, y);
         break;
      default:
         MFEM_ABORT("Only dimensions 2 and 3 supported.");
   }
}

void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag)
{
   switch (dim)
   {
      case 2:
         ElasticityAssembleDiagonalPA_<2>(nDofs, lambda, mu, geom, maps, QVec, diag);
         break;
      case 3:
         ElasticityAssembleDiagonalPA_<3>(nDofs, lambda, mu, geom, maps, QVec, diag);
         break;
      default:
         MFEM_ABORT("Only dimensions 2 and 3 supported.");
   }
}

void ElasticityAssembleEA(const int dim, const int i_block, const int j_block,
                          const int nDofs, const IntegrationRule &ir,
                          const CoefficientVector &lambda,
                          const CoefficientVector &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat)
{
   switch (dim)
   {
      case 2:
         ElasticityAssembleEA_<2>(i_block, j_block, nDofs, ir, lambda, mu, geom, maps,
                                  emat);
         break;
      case 3:
         ElasticityAssembleEA_<3>(i_block, j_block, nDofs, ir, lambda, mu, geom, maps,
                                  emat);
         break;
      default:
         MFEM_ABORT("Only dimensions 2 and 3 supported.");
   }
}

} // namespace internal

} // namespace mfem
