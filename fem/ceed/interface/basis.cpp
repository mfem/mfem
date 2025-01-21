// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../gridfunc.hpp"
#include "util.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static CeedElemTopology GetCeedTopology(Geometry::Type geom)
{
   switch (geom)
   {
      case Geometry::SEGMENT:
         return CEED_TOPOLOGY_LINE;
      case Geometry::TRIANGLE:
         return CEED_TOPOLOGY_TRIANGLE;
      case Geometry::SQUARE:
         return CEED_TOPOLOGY_QUAD;
      case Geometry::TETRAHEDRON:
         return CEED_TOPOLOGY_TET;
      case Geometry::CUBE:
         return CEED_TOPOLOGY_HEX;
      case Geometry::PRISM:
         return CEED_TOPOLOGY_PRISM;
      case Geometry::PYRAMID:
         return CEED_TOPOLOGY_PYRAMID;
      default:
         MFEM_ABORT("This type of element is not supported");
         return CEED_TOPOLOGY_PRISM; // Silence warning
   }
}

static void InitNonTensorBasis(const mfem::FiniteElementSpace &fes,
                               const mfem::FiniteElement &fe,
                               const mfem::IntegrationRule &ir,
                               Ceed ceed, CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::FULL);
   mfem::Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   mfem::DenseMatrix qX(dim,nqpts);
   mfem::Vector qW(nqpts);
   for (int i = 0; i < nqpts; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX(0,i) = ip.x;
      if (dim>1) { qX(1,i) = ip.y; }
      if (dim>2) { qX(2,i) = ip.z; }
      qW(i) = ip.weight;
   }
   CeedBasisCreateH1(ceed, GetCeedTopology(fe.GetGeomType()),
                     fes.GetVDim(), ndofs, nqpts,
                     maps.Bt.GetData(), maps.Gt.GetData(),
                     qX.GetData(), qW.GetData(), basis);
}

static void InitTensorBasis(const mfem::FiniteElementSpace &fes,
                            const mfem::FiniteElement &fe,
                            const mfem::IntegrationRule &ir,
                            Ceed ceed, CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::TENSOR);
   mfem::Mesh *mesh = fes.GetMesh();
   const int ndofs = maps.ndof;
   const int nqpts = maps.nqpt;
   mfem::Vector qX(nqpts), qW(nqpts);
   // The x-coordinates of the first `nqpts` points of the integration rule are
   // the points of the corresponding 1D rule. We also scale the weights
   // accordingly.
   double w_sum = 0.0;
   for (int i = 0; i < nqpts; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX(i) = ip.x;
      qW(i) = ip.weight;
      w_sum += ip.weight;
   }
   qW *= 1.0/w_sum;
   CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes.GetVDim(), ndofs,
                           nqpts, maps.Bt.GetData(),
                           maps.Gt.GetData(), qX.GetData(),
                           qW.GetData(), basis);
}

static void InitBasisImpl(const FiniteElementSpace &fes,
                          const FiniteElement &fe,
                          const IntegrationRule &ir,
                          Ceed ceed, CeedBasis *basis)
{
   // Check for FES -> basis, restriction in hash tables
   const int P = fe.GetDof();
   const int Q = ir.GetNPoints();
   const int ncomp = fes.GetVDim();
   BasisKey basis_key(&fes, &ir, ncomp, P, Q);
   auto basis_itr = mfem::internal::ceed_basis_map.find(basis_key);
   const bool tensor = dynamic_cast<const mfem::TensorBasisElement *>
                       (&fe) != nullptr;

   // Init or retrieve key values
   if (basis_itr == mfem::internal::ceed_basis_map.end())
   {
      if ( tensor )
      {
         InitTensorBasis(fes, fe, ir, ceed, basis);
      }
      else
      {
         InitNonTensorBasis(fes, fe, ir, ceed, basis);
      }
      mfem::internal::ceed_basis_map[basis_key] = *basis;
   }
   else
   {
      *basis = basis_itr->second;
   }
}

void InitBasis(const FiniteElementSpace &fes,
               const IntegrationRule &ir,
               Ceed ceed, CeedBasis *basis)
{
   const mfem::FiniteElement &fe = *fes.GetTypicalFE();
   InitBasisImpl(fes, fe, ir, ceed, basis);
}

void InitBasisWithIndices(const FiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          int nelem,
                          const int* indices,
                          Ceed ceed, CeedBasis *basis)
{
   const mfem::FiniteElement &fe = *fes.GetFE(indices[0]);
   InitBasisImpl(fes, fe, ir, ceed, basis);
}

#endif

} // namespace ceed

} // namespace mfem
