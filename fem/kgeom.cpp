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
#include "../linalg/kvector.hpp"

#include "kgeom.hpp"
#include "fem.hpp"
#include "doftoquad.hpp"
#include "kernels/geom.hpp"

namespace mfem
{

// *****************************************************************************
static kGeometry *geom=NULL;

// ***************************************************************************
// * ~ kGeometry
// ***************************************************************************
kGeometry::~kGeometry()
{
   free(geom->meshNodes);
   free(geom->J);
   free(geom->invJ);
   free(geom->detJ);
   delete[] geom;
}

// *****************************************************************************
// * kGeometry Get: use this one to fetch nodes from vector Sx
// *****************************************************************************
kGeometry* kGeometry::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir,
                          const Vector& Sx)
{
   dbg();
   const Mesh *mesh = fes.GetMesh();
   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fespace = nodes->FESpace();
   const FiniteElement *fe = fespace->GetFE(0);
   const int dims     = fe->GetDim();
   const int numDofs  = fe->GetDof();
   const int numQuad  = ir.GetNPoints();
   const int elements = fespace->GetNE();
   const int ndofs    = fespace->GetNDofs();
   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(*fe, ir);
   rNodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->meshNodes);
   rIniGeom(dims,numDofs,numQuad,elements,
            maps->dofToQuadD,
            geom->meshNodes,
            geom->J,
            geom->invJ,
            geom->detJ);
   return geom;
}

// **************************************************************************
static void kGeomFill(const int dims,
                      const size_t elements, const size_t numDofs,
                      const int* elementMap, int* eMap,
                      const double *nodes, double *meshNodes){
   GET_CONST_ADRS_T(elementMap,int);
   GET_ADRS_T(eMap,int);
   GET_CONST_ADRS(nodes);
   GET_ADRS(meshNodes);
   forall(e, elements, {
         for (int d = 0; d < numDofs; ++d) {
            const int lid = d+numDofs*e;
            const int gid = d_elementMap[lid];
            d_eMap[lid] = gid;
            for (int v = 0; v < dims; ++v) {
               const int moffset = v+dims*lid;
               const int xoffset = v+dims*gid;
               d_meshNodes[moffset] = d_nodes[xoffset];
            }
         }
      });
}

// **************************************************************************
static void kArrayAssign(const int n, const int *src, int *dest){
   GET_CONST_ADRS_T(src,int);
   GET_ADRS_T(dest,int);
   forall(i, n, d_dest[i] = d_src[i];);
}
   
// *****************************************************************************
kGeometry* kGeometry::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir)
{
   dbg();
   Mesh& mesh = *(fes.GetMesh());
   const bool geom_to_allocate = !geom;
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate: new kGeometry");
      geom = new kGeometry();
   }
   if (!mesh.GetNodes()) {
      assert(false);
      dbg("GetNodes, SetCurvature");
      mesh.SetCurvature(1, false, -1, Ordering::byVDIM);
   }
   GridFunction& nodes = *(mesh.GetNodes());
   //dbg("nodes: %p", nodes.GetData());
   //dbg("nodes size: %d", nodes.Size());
   //GET_CONST_ADRS(nodes);
   //dbg("d_nodes: %p", d_nodes);
   //mm::Get().Rsync(nodes.GetData());
   //dbg("nodes:\n");  nodes.Print();
   
   const mfem::FiniteElementSpace& fespace = *(nodes.FESpace());
   const mfem::FiniteElement& fe = *(fespace.GetFE(0));
   const int dims     = fe.GetDim();
   const int elements = fespace.GetNE();
   const int numDofs  = fe.GetDof();
   const int numQuad  = ir.GetNPoints();
   const bool orderedByNODES = (fespace.GetOrdering() == Ordering::byNODES);
   dbg("orderedByNODES: %s", orderedByNODES?"true":"false");

   if (orderedByNODES)
   {
      dbg("orderedByNODES => ReorderByVDim");
      ReorderByVDim(nodes);
   }
   const int asize = dims*numDofs*elements;
   dbg("meshNodes(%d)",asize);
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);
   {
      dbg("kGeomFill");
      kGeomFill(dims,
                elements,
                numDofs,
                elementMap,
                eMap.GetData(),
                nodes.GetData(),
                meshNodes.GetData());
   }
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate");
      dbg("meshNodes: asize=%d", asize);
      geom->meshNodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   {
      dbg("meshNodes= & eMap=");
      kVectorAssign(asize, meshNodes.GetData(), geom->meshNodes);
      //dbg("kVectorPrint(geom->meshNodes:");
      //kVectorPrint(asize, geom->meshNodes);
      
      //geom->meshNodes = meshNodes;
      //geom->eMap = eMap;
      kArrayAssign(numDofs*elements, eMap.GetData(), geom->eMap);
   }

   // Reorder the original gf back
   if (orderedByNODES)
   {
      dbg("Reorder the original gf back");
      ReorderByNodes(nodes);
   }

   if (geom_to_allocate)
   {
      dbg("dims=%d",dims);
      dbg("numQuad=%d",numQuad);
      dbg("elements=%d",elements);
      dbg("geom_to_allocate: J, invJ & detJ: %ld", dims*dims*numQuad*elements);
      geom->J.allocate(dims, dims, numQuad, elements);
      geom->invJ.allocate(dims, dims, numQuad, elements);
      geom->detJ.allocate(numQuad, elements);
   }

   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(fe, ir);
   assert(maps);
   
   dbg("dims=%d",dims);
   dbg("numDofs=%d",numDofs);
   dbg("numQuad=%d",numQuad);
   dbg("elements=%d",elements);
   rIniGeom(dims, numDofs, numQuad, elements,
            maps->dofToQuadD,
            geom->meshNodes,
            geom->J,
            geom->invJ,
            geom->detJ);
   return geom;
}

// ***************************************************************************
void kGeometry::ReorderByVDim(GridFunction& nodes)
{
   const mfem::FiniteElementSpace *fes=nodes.FESpace();
   const int size = nodes.Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes.GetData();
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
void kGeometry::ReorderByNodes(GridFunction& nodes)
{
   const mfem::FiniteElementSpace *fes=nodes.FESpace();
   const int size = nodes.Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes.GetData();
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

