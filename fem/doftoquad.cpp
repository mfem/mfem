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

#include "../config/config.hpp"

#include "../general/okina.hpp"
#include "../linalg/kernels/vector.hpp"
#include "fem.hpp"

#include "doftoquad.hpp"
#include <map>

namespace mfem
{

// ***************************************************************************
// * DofToQuad
// ***************************************************************************
static std::map<std::string, DofToQuad* > AllDofQuadMaps;

// *****************************************************************************
DofToQuad* DofToQuad::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*fes.GetFE(0), *fes.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElementSpace& trialFES,
                          const FiniteElementSpace& testFES,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*trialFES.GetFE(0), *testFES.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElement& trialFE,
                          const FiniteElement& testFE,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return GetTensorMaps(trialFE, testFE, ir, transpose);
}

// ***************************************************************************
DofToQuad* DofToQuad::GetTensorMaps(const FiniteElement& trialFE,
                                    const FiniteElement& testFE,
                                    const IntegrationRule& ir,
                                    const bool transpose)
{
   const TensorBasisElement& trialTFE =
      dynamic_cast<const TensorBasisElement&>(trialFE);
   const TensorBasisElement& testTFE =
      dynamic_cast<const TensorBasisElement&>(testFE);
   std::stringstream ss;
   ss << "TensorMap:"
      << " O1:"  << trialFE.GetOrder()
      << " O2:"  << testFE.GetOrder()
      << " BT1:" << trialTFE.GetBasisType()
      << " BT2:" << testTFE.GetBasisType()
      << " Q:"   << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   // Otherwise, build them
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QTensorMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QTensorMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

// ***************************************************************************
DofToQuad* DofToQuad::GetD2QTensorMaps(const FiniteElement& fe,
                                       const IntegrationRule& ir,
                                       const bool transpose)
{
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());

   const int dims = fe.GetDim();
   const int order = fe.GetOrder();
   const int ND1d = order + 1;
   const int NQ1d = ir1D.GetNPoints();
   const int NQ2d = NQ1d * NQ1d;
   const int NQ3d = NQ2d * NQ1d;
   const int numQuad =
      (dims == 1) ? NQ1d :
      (dims == 2) ? NQ2d :
      (dims == 3) ? NQ3d : 0;
   assert(numQuad > 0);
   std::stringstream ss;
   ss << "D2QTensorMap:"
      << " dims:" << dims
      << " order:" << order
      << " ND1d:" << ND1d
      << " NQ1d:" << NQ1d
      << " transpose:"  << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(NQ1d*ND1d);
   maps->G.SetSize(NQ1d*ND1d);
   const int dim0 = (!transpose)?1:ND1d;
   const int dim1 = (!transpose)?NQ1d:1;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(numQuad);
   }
   mfem::Vector d2q(ND1d);
   mfem::Vector d2qD(ND1d);
   mfem::Array<double> W1d(NQ1d);
   mfem::Array<double> B1d(NQ1d*ND1d);
   mfem::Array<double> G1d(NQ1d*ND1d);
   const TensorBasisElement& tbe = dynamic_cast<const TensorBasisElement&>(fe);
   const Poly_1D::Basis& basis = tbe.GetBasis1D();
   for (int q = 0; q < NQ1d; ++q)
   {
      const IntegrationPoint& ip = ir1D.IntPoint(q);
      if (transpose)
      {
         W1d[q] = ip.weight;
      }
      basis.Eval(ip.x, d2q, d2qD);
      for (int d = 0; d < ND1d; ++d)
      {
         const double w = d2q[d];
         const double wD = d2qD[d];
         const int idx = dim0*q + dim1*d;
         B1d[idx] = w;
         G1d[idx] = wD;
      }
   }
   if (transpose)
   {
      mfem::Array<double> W(numQuad);
      for (int q = 0; q < numQuad; ++q)
      {
         const int qx = q % NQ1d;
         const int qz = q / NQ2d;
         const int qy = (q - qz*NQ2d) / NQ1d;
         double w = W1d[qx];
         if (dims > 1) { w *= W1d[qy]; }
         if (dims > 2) { w *= W1d[qz]; }
         W[q] = w;
      }
      maps->W = W;
   }
   kernels::vector::Assign(NQ1d*ND1d, B1d, maps->B);
   kernels::vector::Assign(NQ1d*ND1d, G1d, maps->G);
   return maps;
}

// ***************************************************************************
DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& fe,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   return GetSimplexMaps(fe, fe, ir, transpose);
}

// *****************************************************************************
DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& trialFE,
                                     const FiniteElement& testFE,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   std::stringstream ss;
   ss << "SimplexMap:"
      << " O1:" << trialFE.GetOrder()
      << " O2:" << testFE.GetOrder()
      << " Q:"  << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QSimplexMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QSimplexMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

// ***************************************************************************
DofToQuad* DofToQuad::GetD2QSimplexMaps(const FiniteElement& fe,
                                        const IntegrationRule& ir,
                                        const bool transpose)
{
   const int dims = fe.GetDim();
   const int ND = fe.GetDof();
   const int NQ = ir.GetNPoints();
   std::stringstream ss ;
   ss << "D2QSimplexMap:"
      << " Dim:" << dims
      << " ND:" << ND
      << " NQ:" << NQ
      << " transpose:" << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad* maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(NQ*ND);
   maps->G.SetSize(dims*NQ*ND);
   const int dim0 = (!transpose)?1:ND;
   const int dim1 = (!transpose)?NQ:1;
   const int dim0D = (!transpose)?1:NQ;
   const int dim1D = (!transpose)?dims:1;
   const int dim2D = dims*NQ;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(NQ);
   }
   mfem::Vector d2q(ND);
   mfem::DenseMatrix d2qD(ND, dims);
   mfem::Array<double> W(NQ);
   mfem::Array<double> B(NQ*ND);
   mfem::Array<double> G(dims*NQ*ND);
   for (int q = 0; q < NQ; ++q)
   {
      const IntegrationPoint& ip = ir.IntPoint(q);
      if (transpose)
      {
         W[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < ND; ++d)
      {
         const double w = d2q[d];
         const int idx = dim0*q + dim1*d;
         B[idx] = w;
         for (int dim = 0; dim < dims; ++dim)
         {
            const double wD = d2qD(d, dim);
            const int idxD = dim0D*dim + dim1D*q + dim2D*d;
            G[idxD] = wD;
         }
      }
   }
   if (transpose)
   {
      kernels::vector::Assign(NQ, W, maps->W);
   }
   kernels::vector::Assign(NQ*ND, B, maps->B);
   kernels::vector::Assign(dims*NQ*ND, G, maps->G);
   return maps;
}

} // namespace mfem
