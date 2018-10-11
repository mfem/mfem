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
// * kDofQuadMaps
// ***************************************************************************
static std::map<std::string, kDofQuadMaps* > AllDofQuadMaps;

// ***************************************************************************
kDofQuadMaps::~kDofQuadMaps() {}

// *****************************************************************************
void kDofQuadMaps::delkDofQuadMaps()
{
   for (std::map<std::string,
        kDofQuadMaps*>::iterator itr = AllDofQuadMaps.begin();
        itr != AllDofQuadMaps.end();
        itr++)
   {
      delete itr->second;
   }
}

// *****************************************************************************
kDofQuadMaps* kDofQuadMaps::Get(const FiniteElementSpace& fes,
                                const IntegrationRule& ir,
                                const bool transpose)
{
   return Get(*fes.GetFE(0), *fes.GetFE(0), ir, transpose);
}

kDofQuadMaps* kDofQuadMaps::Get(const FiniteElementSpace& trialFES,
                                const FiniteElementSpace& testFES,
                                const IntegrationRule& ir,
                                const bool transpose)
{
   return Get(*trialFES.GetFE(0), *testFES.GetFE(0), ir, transpose);
}

kDofQuadMaps* kDofQuadMaps::Get(const FiniteElement& trialFE,
                                const FiniteElement& testFE,
                                const IntegrationRule& ir,
                                const bool transpose)
{
   return GetTensorMaps(trialFE, testFE, ir, transpose);
}

// ***************************************************************************
kDofQuadMaps* kDofQuadMaps::GetTensorMaps(const FiniteElement& trialFE,
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
   kDofQuadMaps *maps = new kDofQuadMaps();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const kDofQuadMaps* trialMaps = GetD2QTensorMaps(trialFE, ir);
   const kDofQuadMaps* testMaps  = GetD2QTensorMaps(testFE, ir, true);
   maps->dofToQuad   = trialMaps->dofToQuad;
   maps->dofToQuadD  = trialMaps->dofToQuadD;
   maps->quadToDof   = testMaps->dofToQuad;
   maps->quadToDofD  = testMaps->dofToQuadD;
   maps->quadWeights = testMaps->quadWeights;
   //assert(false);
   return maps;
}

// ***************************************************************************
kDofQuadMaps* kDofQuadMaps::GetD2QTensorMaps(const FiniteElement& fe,
                                             const IntegrationRule& ir,
                                             const bool transpose)
{
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());

   const int dims = fe.GetDim();
   const int order = fe.GetOrder();
   const int numDofs = order + 1;
   const int numQuad1D = ir1D.GetNPoints();
   const int numQuad2D = numQuad1D * numQuad1D;
   const int numQuad3D = numQuad2D * numQuad1D;
   const int numQuad =
      (dims == 1) ? numQuad1D :
      (dims == 2) ? numQuad2D :
      (dims == 3) ? numQuad3D : 0;
   assert(numQuad > 0);

   std::stringstream ss;
   ss << "D2QTensorMap:"
      << " dims:" << dims
      << " order:" << order
      << " numDofs:" << numDofs
      << " numQuad1D:" << numQuad1D
      << " transpose:"  << (transpose?"true":"false");
   std::string hash = ss.str();

   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }

   kDofQuadMaps *maps = new kDofQuadMaps();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;

   maps->dofToQuad.allocate( numQuad1D, numDofs, 1, 1, transpose);
   maps->dofToQuadD.allocate(numQuad1D, numDofs, 1, 1, transpose);
   const int dim0 = maps->dofToQuad.dim()[0];
   const int dim1 = maps->dofToQuad.dim()[1];

   if (transpose) // Initialize quad weights only for transpose
   {
      maps->quadWeights.allocate(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::Vector d2qD(numDofs);
   mfem::Array<double> quadWeights1D(numQuad1D);
   mfem::Array<double> dofToQuad(numQuad1D*numDofs);
   mfem::Array<double> dofToQuadD(numQuad1D*numDofs);
   const TensorBasisElement& tbe = dynamic_cast<const TensorBasisElement&>(fe);
   const Poly_1D::Basis& basis = tbe.GetBasis1D();

   for (int q = 0; q < numQuad1D; ++q)
   {
      const IntegrationPoint& ip = ir1D.IntPoint(q);
      if (transpose)
      {
         quadWeights1D[q] = ip.weight;
      }
      basis.Eval(ip.x, d2q, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const double wD = d2qD[d];
         const int idx = dim0*q + dim1*d;
         dofToQuad[idx] = w;
         dofToQuadD[idx] = wD;
      }
   }
   if (transpose)
   {
      mfem::Array<double> quadWeights(numQuad);
      for (int q = 0; q < numQuad; ++q)
      {
         const int qx = q % numQuad1D;
         const int qz = q / numQuad2D;
         const int qy = (q - qz*numQuad2D) / numQuad1D;
         double w = quadWeights1D[qx];
         if (dims > 1) { w *= quadWeights1D[qy]; }
         if (dims > 2) { w *= quadWeights1D[qz]; }
         quadWeights[q] = w;
      }
      //maps->quadWeights = quadWeights;
      mm::Get().Push(quadWeights.GetData());
      kVectorAssign(numQuad, quadWeights.GetData(), maps->quadWeights);
   }
   //maps->dofToQuad = dofToQuad;
   mm::Get().Push(dofToQuad.GetData());
   kVectorAssign(numQuad1D*numDofs, dofToQuad.GetData(), maps->dofToQuad);
   
   //maps->dofToQuadD = dofToQuadD;
   mm::Get().Push(dofToQuadD.GetData());
   kVectorAssign(numQuad1D*numDofs, dofToQuadD.GetData(), maps->dofToQuadD);
   return maps;
}

// ***************************************************************************
kDofQuadMaps* kDofQuadMaps::GetSimplexMaps(const FiniteElement& fe,
                                           const IntegrationRule& ir,
                                           const bool transpose)
{
   return GetSimplexMaps(fe, fe, ir, transpose);
}

// *****************************************************************************
kDofQuadMaps* kDofQuadMaps::GetSimplexMaps(const FiniteElement& trialFE,
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
   kDofQuadMaps *maps = new kDofQuadMaps();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const kDofQuadMaps* trialMaps = GetD2QSimplexMaps(trialFE, ir);
   const kDofQuadMaps* testMaps  = GetD2QSimplexMaps(testFE, ir, true);
   maps->dofToQuad   = trialMaps->dofToQuad;
   maps->dofToQuadD  = trialMaps->dofToQuadD;
   maps->quadToDof   = testMaps->dofToQuad;
   maps->quadToDofD  = testMaps->dofToQuadD;
   maps->quadWeights = testMaps->quadWeights;
   return maps;
}

// ***************************************************************************
kDofQuadMaps* kDofQuadMaps::GetD2QSimplexMaps(const FiniteElement& fe,
                                              const IntegrationRule& ir,
                                              const bool transpose)
{
   const int dims = fe.GetDim();
   const int numDofs = fe.GetDof();
   const int numQuad = ir.GetNPoints();

   std::stringstream ss ;
   ss << "D2QSimplexMap:"
      << " Dim:" << dims
      << " numDofs:" << numDofs
      << " numQuad:" << numQuad
      << " transpose:" << (transpose?"true":"false");
   std::string hash = ss.str();

   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }

   kDofQuadMaps* maps = new kDofQuadMaps();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;

   maps->dofToQuad.allocate( numQuad, numDofs,       1, 1, transpose);
   maps->dofToQuadD.allocate(   dims, numQuad, numDofs, 1, transpose);
   const int dim0 = maps->dofToQuad.dim()[0];
   const int dim1 = maps->dofToQuad.dim()[1];
   const int dim0D = maps->dofToQuadD.dim()[0];
   const int dim1D = maps->dofToQuadD.dim()[1];
   const int dim2D = maps->dofToQuadD.dim()[2];

   if (transpose) // Initialize quad weights only for transpose
   {
      maps->quadWeights.allocate(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::DenseMatrix d2qD(numDofs, dims);
   mfem::Array<double> quadWeights(numQuad);
   mfem::Array<double> dofToQuad(numQuad*numDofs);
   mfem::Array<double> dofToQuadD(dims*numQuad*numDofs);

   for (int q = 0; q < numQuad; ++q)
   {
      const IntegrationPoint& ip = ir.IntPoint(q);
      if (transpose)
      {
         quadWeights[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const int idx = dim0*q + dim1*d;
         dofToQuad[idx] = w;
         for (int dim = 0; dim < dims; ++dim)
         {
            const double wD = d2qD(d, dim);
            const int idxD = dim0D*dim + dim1D*q + dim2D*d;
            dofToQuadD[idxD] = wD;
         }
      }
   }
   if (transpose)
   {
      //maps->quadWeights = quadWeights;
      mm::Get().Push(quadWeights.GetData());
      kVectorAssign(numQuad, quadWeights.GetData(), maps->quadWeights);
    }

   //maps->dofToQuad = dofToQuad;
   mm::Get().Push(dofToQuad.GetData());
   kVectorAssign(numQuad*numDofs, dofToQuad.GetData(), maps->dofToQuad);
   //assert(false);
   
   //maps->dofToQuadD = dofToQuadD;
   mm::Get().Push(dofToQuadD.GetData());
   kVectorAssign(dims*numQuad*numDofs, dofToQuadD.GetData(), maps->dofToQuadD);
   return maps;
}

} // namespace mfem
