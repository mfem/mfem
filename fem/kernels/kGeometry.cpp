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

namespace mfem
{

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
      {0x206,&kGeom2D<2,8>},
      {0x207,&kGeom2D<2,9>},
      {0x210,&kGeom2D<3,2>},/*
      {0x208,&kGeom2D<2,10>},
      {0x209,&kGeom2D<2,11>},
      {0x20A,&kGeom2D<2,12>},
      {0x20B,&kGeom2D<2,13>},
      {0x20C,&kGeom2D<2,14>},
      {0x20D,&kGeom2D<2,15>},
      {0x20E,&kGeom2D<2,16>},
      {0x20F,&kGeom2D<2,17>},*/
      // 3D
      {0x300,&kGeom3D<2,2>},
      {0x301,&kGeom3D<2,3>},
      {0x302,&kGeom3D<2,4>},
      {0x303,&kGeom3D<2,5>},
      {0x304,&kGeom3D<2,6>},
      {0x305,&kGeom3D<2,7>},
      {0x306,&kGeom3D<2,8>},
      {0x307,&kGeom3D<2,9>},
      {0x321,&kGeom3D<4,3>},/*
      {0x308,&kGeom3D<2,10>},
      {0x309,&kGeom3D<2,11>},
      {0x30A,&kGeom3D<2,12>},
      {0x30B,&kGeom3D<2,13>},
      {0x30C,&kGeom3D<2,14>},
      {0x30D,&kGeom3D<2,15>},
      {0x30E,&kGeom3D<2,16>},
      {0x30F,&kGeom3D<2,17>},*/
   };
   if (!call[id])
   {
      printf("\n[kGeom] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
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
static void kArrayAssign(const int n, const int *src, int *dest)
{
   GET_CONST_ADRS_T(src,int);
   GET_ADRS_T(dest,int);
   MFEM_FORALL(i, n, d_dest[i] = d_src[i];);
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
   if (geom_to_allocate) { geom = new kGeometry(); }
   if (!mesh->GetNodes())
   {
      // mesh->SetCurvature(1, false, -1, Ordering::byVDIM);
   }
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
   kGeomFill(dims,
             elements,
             numDofs,
             elementMap,
             eMap.GetData(),
             nodes->GetData(),
             meshNodes.GetData());

   if (geom_to_allocate)
   {
      geom->meshNodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   kVectorAssign(asize, meshNodes.GetData(), geom->meshNodes);
   kArrayAssign(numDofs*elements, eMap.GetData(), geom->eMap);
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      geom->J.allocate(dims, dims, numQuad, elements);
      geom->invJ.allocate(dims, dims, numQuad, elements);
      geom->detJ.allocate(numQuad, elements);
   }
   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(*fe, ir);
   kGeom(dims, numDofs, numQuad, elements, maps->dofToQuadD,
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

}
