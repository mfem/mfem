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

#include "geometry.hpp"
#include "../fem.hpp"
#include "../doftoquad.hpp"

namespace mfem
{
namespace kernels
{
namespace geometry
{

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
void Geom2D(const int,const double*,const double*,double*,double*,double*);

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
void Geom3D(const int,const double*,const double*,double*,double*,double*);

// *****************************************************************************
typedef void (*fIniGeom)(const int,const double*,const double*,
                         double*, double*, double*);

// *****************************************************************************
static void Geom(const int DIM,
                 const int NUM_DOFS,
                 const int NUM_QUAD,
                 const int numElements,
                 const double* dofToQuadD,
                 const double* nodes,
                 double* J,
                 double* invJ,
                 double* detJ)
{
   static int loop = 0;
   //if (loop++==1) { assert(false); }
   const unsigned int dofs1D = IROOT(DIM,NUM_DOFS);
   const unsigned int quad1D = IROOT(DIM,NUM_QUAD);
   const unsigned int id = (DIM<<8)|(dofs1D-2)<<4|(quad1D-2);
   assert(LOG2(DIM)<=4);
   assert(LOG2(dofs1D-2)<=4);
   assert(LOG2(quad1D-2)<=4);
   static std::unordered_map<unsigned int, fIniGeom> call =
   {
      // 2D
      {0x200,&Geom2D<2,2>},
      {0x201,&Geom2D<2,3>},
      {0x202,&Geom2D<2,4>},
      {0x203,&Geom2D<2,5>},
      {0x204,&Geom2D<2,6>},
      {0x205,&Geom2D<2,7>},
      {0x206,&Geom2D<2,8>},
      {0x207,&Geom2D<2,9>},
      {0x210,&Geom2D<3,2>},/*
      {0x208,&Geom2D<2,10>},
      {0x209,&Geom2D<2,11>},
      {0x20A,&Geom2D<2,12>},
      {0x20B,&Geom2D<2,13>},
      {0x20C,&Geom2D<2,14>},
      {0x20D,&Geom2D<2,15>},
      {0x20E,&Geom2D<2,16>},
      {0x20F,&Geom2D<2,17>},*/
      // 3D
      {0x300,&Geom3D<2,2>},
      {0x301,&Geom3D<2,3>},
      {0x302,&Geom3D<2,4>},
      {0x303,&Geom3D<2,5>},
      {0x304,&Geom3D<2,6>},
      {0x305,&Geom3D<2,7>},
      {0x306,&Geom3D<2,8>},
      {0x307,&Geom3D<2,9>},
      {0x321,&Geom3D<4,3>},/*
      {0x308,&Geom3D<2,10>},
      {0x309,&Geom3D<2,11>},
      {0x30A,&Geom3D<2,12>},
      {0x30B,&Geom3D<2,13>},
      {0x30C,&Geom3D<2,14>},
      {0x30D,&Geom3D<2,15>},
      {0x30E,&Geom3D<2,16>},
      {0x30F,&Geom3D<2,17>},*/
   };
   if (!call[id])
   {
      printf("\n[Geom] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   GET_CONST_PTR(dofToQuadD);
   GET_CONST_PTR(nodes);
   GET_PTR(J);
   GET_PTR(invJ);
   GET_PTR(detJ);
   call[id](numElements, d_dofToQuadD, d_nodes, d_J, d_invJ, d_detJ);
}

// *****************************************************************************
static Geometry *geom = NULL;

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
static void ArrayAssign(const int n, const int *src, int *dest)
{
   GET_CONST_PTR_T(src,int);
   GET_PTR_T(dest,int);
   MFEM_FORALL(i, n, d_dest[i] = d_src[i];);
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
Geometry* Geometry::Get(const FiniteElementSpace& fes,
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
   NodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->meshNodes);
   Geom(dims, numDofs, numQuad, elements,
        maps->dofToQuadD,
        geom->meshNodes, geom->J, geom->invJ, geom->detJ);
   delete maps;
   return geom;
}

// *****************************************************************************
Geometry* Geometry::Get(const FiniteElementSpace& fes,
                        const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
#warning geom_to_allocate
   const bool geom_to_allocate = true;//!geom;
   if (geom_to_allocate) {
      geom = new Geometry();
   }
   if (!mesh->GetNodes())
   {
      mesh->EnsureNodes();
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
   dbg("dims=%d",dims);
   dbg("elements=%d",elements);
   dbg("numDofs=%d",numDofs);
   assert(nodes);
   assert(elementMap);
   GeomFill(dims,
            elements,
            numDofs,
            elementMap,
            eMap.GetData(),
            nodes->GetData(),
            meshNodes.GetData());
   static int loop = 0;

   if (geom_to_allocate)
   {
      dbg("geom_to_allocate: meshNodes, eMap");
      geom->meshNodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   kernels::vector::Assign(asize, meshNodes.GetData(), geom->meshNodes);
   ArrayAssign(numDofs*elements, eMap.GetData(), geom->eMap);
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate: J, invJ, detJ");
      geom->J.allocate(dims, dims, numQuad, elements);
      geom->invJ.allocate(dims, dims, numQuad, elements);
      geom->detJ.allocate(numQuad, elements);
   }
   dbg("maps...");
   const kDofQuadMaps* maps = kDofQuadMaps::GetSimplexMaps(*fe, ir);
   assert(geom->J);
   assert(geom->invJ);
   assert(geom->detJ);
   assert(geom->meshNodes);
   assert(maps);
   assert(maps->dofToQuadD);
   //dbg("while..."); if (loop==1) {while(true);}
   dbg("Geom...");
   Geom(dims, numDofs, numQuad, elements, maps->dofToQuadD,
        geom->meshNodes, geom->J, geom->invJ, geom->detJ);
   //if (loop==1) { assert(false); }
#warning no delete maps
   //delete maps;
   loop++;
   return geom;
}

// ***************************************************************************
void Geometry::ReorderByVDim(const GridFunction *nodes)
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
void Geometry::ReorderByNodes(const GridFunction *nodes)
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


} // namespace geometry
} // namespace kernels
} // namespace mfem
