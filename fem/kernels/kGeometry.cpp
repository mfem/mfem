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

#include "../../config/config.hpp"

#include "../../general/okina.hpp"
#include "../../linalg/kernels/vector.hpp"

#include "kGeometry.hpp"
#include "../fem.hpp"
#include "../doftoquad.hpp"

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
void kGeom2D(const int,const double*,const double*,double*,double*,double*);

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
void kGeom3D(const int,const double*,const double*,double*,double*,double*);

// *****************************************************************************
typedef void (*fIniGeom)(const int,const double*,const double*,
                         double*, double*, double*);

// *****************************************************************************
void kGeom(const int DIM,
           const int NUM_DOFS,
           const int NUM_QUAD,
           const int numElements,
           const double* dofToQuadD,
           const double* nodes,
           double* J,
           double* invJ,
           double* detJ)
{
   const unsigned int dofs1D = IROOT(DIM,NUM_DOFS);
   const unsigned int quad1D = IROOT(DIM,NUM_QUAD);
   const unsigned int id = (DIM<<8)|(dofs1D-2)<<4|(quad1D-2);
   dbg("DIM=%d",DIM);
   dbg("dofs1D=%d",dofs1D);
   dbg("quad1D=%d",quad1D);
   dbg("id=%d",id);
   assert(LOG2(DIM)<=4);
   assert(LOG2(dofs1D-2)<=4);
   assert(LOG2(quad1D-2)<=4);
   static std::unordered_map<unsigned int, fIniGeom> call =
   {
      // 2D
      {0x200,&kGeom2D<2,2>},
      {0x201,&kGeom2D<2,3>},
      {0x202,&kGeom2D<2,4>},
      {0x203,&kGeom2D<2,5>},
      {0x204,&kGeom2D<2,6>},
      {0x205,&kGeom2D<2,7>},
      
      {0x211,&kGeom2D<3,3>},
      {0x222,&kGeom2D<4,4>},
/*      {0x211,&kGeom2D<3,3>},
      {0x222,&kGeom2D<4,4>},
      {0x233,&kGeom2D<5,5>},
      {0x244,&kGeom2D<6,6>},
      {0x255,&kGeom2D<7,7>},
      {0x266,&kGeom2D<8,8>},
      {0x277,&kGeom2D<9,9>},
      {0x288,&kGeom2D<10,10>},
      {0x299,&kGeom2D<11,11>},
      {0x2AA,&kGeom2D<12,12>},
      {0x2BB,&kGeom2D<13,13>},
      {0x2CC,&kGeom2D<14,14>},
      {0x2DD,&kGeom2D<15,15>},
      {0x2EE,&kGeom2D<16,16>},
      {0x2FF,&kGeom2D<17,17>},*/
      // 3D
      //{0x300,&kGeom3D<2,2>},
/*
      {0x311,&kGeom3D<3,3>},
      {0x322,&kGeom3D<4,4>},
      {0x333,&kGeom3D<5,5>},
      {0x344,&kGeom3D<6,6>},
      {0x355,&kGeom3D<7,7>},
      {0x366,&kGeom3D<8,8>},
      {0x377,&kGeom3D<9,9>},
      {0x388,&kGeom3D<10,10>},
      {0x399,&kGeom3D<11,11>},
      {0x3AA,&kGeom3D<12,12>},
      {0x3BB,&kGeom3D<13,13>},
      {0x3CC,&kGeom3D<14,14>},
      {0x3DD,&kGeom3D<15,15>},
      {0x3EE,&kGeom3D<16,16>},
      {0x3FF,&kGeom3D<17,17>},*/
   };
   if (!call[id])
   {
      printf("\n[kGeom] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   else
   {
      dbg("\n[kGeom] id \033[33m0x%X\033[m ",id);
   }
   assert(call[id]);
   GET_CONST_ADRS(dofToQuadD);
   GET_CONST_ADRS(nodes);
   GET_ADRS(J);
   GET_ADRS(invJ);
   GET_ADRS(detJ);
   call[id](numElements, d_dofToQuadD, d_nodes, d_J, d_invJ, d_detJ);
}

// *****************************************************************************
static kGeometry *geom = NULL;

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
static void kGeomFill(const int dims,
                      const size_t elements, const size_t numDofs,
                      const int* elementMap, int* eMap,
                      const double *nodes, double *meshNodes)
{
   GET_CONST_ADRS_T(elementMap,int);
   GET_ADRS_T(eMap,int);
   GET_CONST_ADRS(nodes);
   GET_ADRS(meshNodes);
   forall(e, elements,
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
static void kArrayAssign(const int n, const int *src, int *dest)
{
   GET_CONST_ADRS_T(src,int);
   GET_ADRS_T(dest,int);
   forall(i, n, d_dest[i] = d_src[i];);
}

// *****************************************************************************
static void rNodeCopyByVDim(const int elements,
                            const int numDofs,
                            const int ndofs,
                            const int dims,
                            const int* eMap,
                            const double* Sx,
                            double* nodes)
{
   forall(e,elements,
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
kGeometry* kGeometry::Get(const FiniteElementSpace& fes,
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
   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(*fe, ir);
   rNodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->meshNodes);
   kGeom(dims, numDofs, numQuad, elements,
         maps->dofToQuadD,
         geom->meshNodes, geom->J, geom->invJ, geom->detJ);
   return geom;
}

// *****************************************************************************
kGeometry* kGeometry::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
   const bool geom_to_allocate = !geom;
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate");
      geom = new kGeometry();
   }
   if (!mesh->GetNodes())
   {
      assert(false);
      dbg("\033[7mGetNodes, SetCurvature");
      mesh->SetCurvature(1, false, -1, Ordering::byVDIM);
   }
   const GridFunction *nodes = mesh->GetNodes();

   const mfem::FiniteElementSpace *fespace = nodes->FESpace();
   const mfem::FiniteElement *fe = fespace->GetFE(0);
   const int dims     = fe->GetDim();
   const int elements = fespace->GetNE();
   const int numDofs  = fe->GetDof();
   const int numQuad  = ir.GetNPoints();
   dbg("dims=%d",dims);
   dbg("elements=%d",elements);
   dbg("numDofs=%d",numDofs);
   dbg("numQuad=%d",numQuad);

   const bool orderedByNODES = (fespace->GetOrdering() == Ordering::byNODES);
   dbg("orderedByNODES: %s", orderedByNODES?"true":"false");

   if (orderedByNODES)
   {
      dbg("orderedByNODES => ReorderByVDim");
      ReorderByVDim(nodes);
   }
   const int asize = dims*numDofs*elements;
   dbg("meshNodes(%d)",asize);
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);

   dbg("kGeomFill");
   kGeomFill(dims,
             elements,
             numDofs,
             elementMap,
             eMap.GetData(),
             nodes->GetData(),
             meshNodes.GetData());

   if (geom_to_allocate)
   {
      dbg("geom_to_allocate");
      dbg("meshNodes: asize=%d", asize);
      geom->meshNodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   {
      //geom->meshNodes = meshNodes;
      kVectorAssign(asize, meshNodes.GetData(), geom->meshNodes);

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

   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(*fe, ir);
   assert(maps);

   dbg("dims=%d",dims);
   dbg("numDofs=%d",numDofs);
   dbg("numQuad=%d",numQuad);
   dbg("elements=%d",elements);
   kGeom(dims, numDofs, numQuad, elements,
         maps->dofToQuadD,
         geom->meshNodes, geom->J, geom->invJ, geom->detJ);
   return geom;
}

// ***************************************************************************
void kGeometry::ReorderByVDim(const GridFunction *nodes)
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
void kGeometry::ReorderByNodes(const GridFunction *nodes)
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

// *****************************************************************************
MFEM_NAMESPACE_END
