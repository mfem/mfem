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

#include "fem.hpp"
#include "geom_ext.hpp"
#include "kernels/geom.hpp"
#include "../linalg/kernels/vector.hpp"
#include "../general/kernels/array.hpp"

namespace mfem
{

static GeometryExtension *geom = NULL;

// *****************************************************************************
static void GeomFill(const int dims,
                     const size_t elements, const size_t numDofs,
                     const int* elementMap, int* eMap,
                     const double *nodes, double *meshNodes)
{
   GET_CONST_PTR_T(elementMap,int);
   GET_PTR_T(eMap,int);
   GET_CONST_PTR(nodes);
   GET_PTR(meshNodes);
   MFEM_FORALL(e, elements,
   {
      for (size_t d = 0; d < numDofs; ++d)
      {
         const int lid = d+numDofs*e;
         const int gid = d_elementMap[lid];
         d_eMap[lid] = gid;
         for (int v = 0; v < dims; ++v)
         {
            const int moffset = v+dims*lid;
            const int xoffset = v+dims*gid;
            d_meshNodes[moffset] = d_nodes[xoffset];
         }
      }
   });
}

// *****************************************************************************
static void NodeCopyByVDim(const int elements,
                           const int numDofs,
                           const int ndofs,
                           const int dims,
                           const int* eMap,
                           const double* Sx,
                           double* nodes)
{
   MFEM_FORALL(e,elements,
   {
      for (int dof = 0; dof < numDofs; ++dof)
      {
         const int lid = dof+numDofs*e;
         const int gid = eMap[lid];
         for (int v = 0; v < dims; ++v)
         {
            const int moffset = v+dims*lid;
            const int voffset = gid+v*ndofs;
            nodes[moffset] = Sx[voffset];
         }
      }
   });
}


// *****************************************************************************
GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir,
                                          const Vector& Sx)
{
   const Mesh *mesh = fes.GetMesh();
   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fespace = nodes->FESpace();
   const FiniteElement *fe = fespace->GetFE(0);
   const int dims     = fe->GetDim();
   const int numDofs  = fe->GetDof();
   const int numQuad  = ir.GetNPoints();
   const int elements = fespace->GetNE();
   const int ndofs    = fespace->GetNDofs();
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   NodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->nodes);
   kernels::fem::Geom(dims, numDofs, numQuad, elements,
                      maps->B, maps->G,
                      geom->nodes, geom->x, geom->J, geom->invJ, geom->detJ);
   return geom;
}

// *****************************************************************************
GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
   const bool geom_to_allocate = !geom;
   if (geom_to_allocate) { geom = new GeometryExtension(); }
   mesh->EnsureNodes();
   const GridFunction *nodes = mesh->GetNodes();

   const mfem::FiniteElementSpace *fespace = nodes->FESpace();
   const mfem::FiniteElement *fe = fespace->GetFE(0);
   const int dims     = fe->GetDim();
   const int elements = fespace->GetNE();
   const int numDofs  = fe->GetDof();
   const int numQuad  = ir.GetNPoints();
   const bool orderedByNODES = (fespace->GetOrdering() == Ordering::byNODES);
   if (orderedByNODES) { ReorderByVDim(nodes); }
   const int asize = dims*numDofs*elements;
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);
   GeomFill(dims,
            elements,
            numDofs,
            elementMap,
            eMap.GetData(),
            nodes->GetData(),
            meshNodes.GetData());

   if (geom_to_allocate)
   {
      geom->nodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   kernels::vector::Assign(asize, meshNodes, geom->nodes);
   kernels::array::Assign(numDofs*elements, eMap.GetData(), geom->eMap);
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      geom->J.allocate(dims, dims, numQuad, elements);
      geom->invJ.allocate(dims, dims, numQuad, elements);
      geom->detJ.allocate(numQuad, elements);
      geom->x.allocate(dims, numQuad, elements);
   }
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   kernels::fem::Geom(dims, numDofs, numQuad, elements,
                      maps->B,
                      maps->G,
                      geom->nodes,
                      geom->x,
                      geom->J, geom->invJ, geom->detJ);
   return geom;
}

// ***************************************************************************
void GeometryExtension::ReorderByVDim(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k=0;
   for (int d = 0; d < ndofs; d++)
      for (int v = 0; v < vdim; v++)
      {
         temp[k++] = data[d+v*ndofs];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

// ***************************************************************************
void GeometryExtension::ReorderByNodes(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k = 0;
   for (int j = 0; j < ndofs; j++)
      for (int i = 0; i < vdim; i++)
      {
         temp[j+i*ndofs] = data[k++];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

} // namespace mfem
