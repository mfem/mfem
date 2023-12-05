// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
void ElasticityAddMultPA(const int dim, const int nDofs,
                         const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                         const CoefficientVector &mu, const GeometricFactors &geom,
                         const DofToQuad &maps, const Vector &x, QuadratureFunction &QVec, Vector &y,
                         const int IBlock, const int JBlock)
{
   //make this dispatch cleaner. Convert -1 to F?
   if (IBlock == -1 && JBlock == -1)
   {
      switch (dim)
      {
         case 2:ElasticityAddMultPA<2>(nDofs, fespace, lambda, mu, geom, maps, x, QVec,
                                          y); break;
         case 3:ElasticityAddMultPA<3>(nDofs, fespace, lambda, mu, geom, maps, x, QVec,
                                          y); break;
         default:
            MFEM_ABORT("Only dimensions 2 and 3 supported.");
            break;
      }
   }
   else if (IBlock >= 0 && JBlock >= 0)
   {
      const int id = (dim<<8)| (IBlock << 4) | JBlock;
      switch (id)
      {
         case 0x200:ElasticityAddMultPA<2,0,0>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x211:ElasticityAddMultPA<2,1,1>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x201:ElasticityAddMultPA<2,0,1>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x210:ElasticityAddMultPA<2,1,0>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x300:ElasticityAddMultPA<3,0,0>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x311:ElasticityAddMultPA<3,1,1>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x322:ElasticityAddMultPA<3,2,2>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x301:ElasticityAddMultPA<3,0,1>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x302:ElasticityAddMultPA<3,0,2>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x312:ElasticityAddMultPA<3,1,2>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x310:ElasticityAddMultPA<3,1,0>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x320:ElasticityAddMultPA<3,2,0>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         case 0x321:ElasticityAddMultPA<3,2,1>(nDofs, fespace, lambda, mu, geom, maps, x,
                                                  QVec,y); break;
         default:
            MFEM_ABORT("Block not compiled. Add to switch if valid.");
            break;
      }
   }
   else
   {
      MFEM_ABORT("Invalid block selection.");
   }

}

void ElasticityAssembleDiagonalPA(const int dim, const int nDofs,
                                  const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                                  const CoefficientVector &mu, const GeometricFactors &geom,
                                  const DofToQuad &maps, QuadratureFunction &QVec, Vector &diag)
{
   switch (dim)
   {
      case 2:ElasticityAssembleDiagonalPA<2>(nDofs, fespace, lambda, mu, geom, maps,
                                                QVec, diag); break;
      case 3:ElasticityAssembleDiagonalPA<3>(nDofs, fespace, lambda, mu, geom, maps,
                                                QVec, diag); break;
      default:
         MFEM_ABORT("Only dimensions 2 and 3 supported.");
         break;
   }
}

void ElasticityAssembleEA(const int dim, const int IBlock, const int JBlock,
                          const int nDofs, const IntegrationRule &ir,
                          const FiniteElementSpace &fespace, const CoefficientVector &lambda,
                          const CoefficientVector &mu, const GeometricFactors &geom,
                          const DofToQuad &maps, Vector &emat)
{
   switch (dim)
   {
      case 2:ElasticityAssembleEA<2>(IBlock, JBlock, nDofs, ir, fespace, lambda, mu,
                                        geom,
                                        maps,
                                        emat); break;
      case 3:ElasticityAssembleEA<3>(IBlock, JBlock, nDofs, ir, fespace, lambda, mu,
                                        geom,
                                        maps,
                                        emat); break;
      default:
         MFEM_ABORT("Only dimensions 2 and 3 supported.");
         break;
   }
}

} // namespace internal

} // namespace mfem
