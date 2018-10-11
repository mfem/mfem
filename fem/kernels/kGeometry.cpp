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
template<const int NUM_DOFS,
         const int NUM_QUAD>
void kGeom2D(const int,const double*,const double*,double*,double*,double*);

// *****************************************************************************
template<const int NUM_DOFS,
         const int NUM_QUAD>
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
           double* detJ){
   const unsigned int dofs1D = IROOT(DIM,NUM_DOFS);
   const unsigned int quad1D = IROOT(DIM,NUM_QUAD);
   const unsigned int id = (DIM<<4)|(dofs1D-2);
   dbg("DIM=%d",DIM);
   dbg("quad1D=%d",quad1D);
   dbg("dofs1D=%d",dofs1D);
   dbg("id=%d",id);
   assert(LOG2(DIM)<=4);
   assert(LOG2(dofs1D-2)<=4);
   static std::unordered_map<unsigned int, fIniGeom> call = {
      // 2D
      {0x20,&kGeom2D<2*2,(2*2-2)*(2*2-2)>},
      {0x21,&kGeom2D<3*3,(3*2-2)*(3*2-2)>},
      {0x22,&kGeom2D<4*4,(4*2-2)*(4*2-2)>},
      {0x23,&kGeom2D<5*5,(5*2-2)*(5*2-2)>},
      {0x24,&kGeom2D<6*6,(6*2-2)*(6*2-2)>},
      {0x25,&kGeom2D<7*7,(7*2-2)*(7*2-2)>},
      {0x26,&kGeom2D<8*8,(8*2-2)*(8*2-2)>},
      {0x27,&kGeom2D<9*9,(9*2-2)*(9*2-2)>},
      {0x28,&kGeom2D<10*10,(10*2-2)*(10*2-2)>},
      {0x29,&kGeom2D<11*11,(11*2-2)*(11*2-2)>},
      {0x2A,&kGeom2D<12*12,(12*2-2)*(12*2-2)>},
      {0x2B,&kGeom2D<13*13,(13*2-2)*(13*2-2)>},
      {0x2C,&kGeom2D<14*14,(14*2-2)*(14*2-2)>},
      {0x2D,&kGeom2D<15*15,(15*2-2)*(15*2-2)>},
      {0x2E,&kGeom2D<16*16,(16*2-2)*(16*2-2)>},
      {0x2F,&kGeom2D<17*17,(17*2-2)*(17*2-2)>},
      /*
      // 3D
      {0x30,&kGeom3D<2*2*2,2*2*2>},
      {0x31,&kGeom3D<3*3*3,4*4*4>},
      {0x32,&kGeom3D<4*4*4,6*6*6>},
      {0x33,&kGeom3D<5*5*5,8*8*8>},
      {0x34,&kGeom3D<6*6*6,10*10*10>},
      {0x35,&kGeom3D<7*7*7,12*12*12>},
      {0x36,&kGeom3D<8*8*8,14*14*14>},
      {0x37,&kGeom3D<9*9*9,16*16*16>},
      {0x38,&kGeom3D<10*10*10,18*18*18>},
      {0x39,&kGeom3D<11*11*11,20*20*20>},
      {0x3A,&kGeom3D<12*12*12,22*22*22>},
      {0x3B,&kGeom3D<13*13*13,24*24*24>},
      {0x3C,&kGeom3D<14*14*14,26*26*26>},
      {0x3D,&kGeom3D<15*15*15,28*28*28>},
      {0x3E,&kGeom3D<16*16*16,30*30*30>},
      {0x3F,&kGeom3D<17*17*17,32*32*32>},
      */
   };
   if (!call[id]){
      printf("\n[kGeom] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }else{
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

// *****************************************************************************
static void kArrayAssign(const int n, const int *src, int *dest){
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
                      double* nodes){
   forall(e,elements, {
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
   Mesh& mesh = *(fes.GetMesh());
   const bool geom_to_allocate = !geom;
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate: new kGeometry");
      geom = new kGeometry();
   }
   if (!mesh.GetNodes()) {
      dbg("\033[7mGetNodes, SetCurvature");
      mesh.SetCurvature(1, false, -1, Ordering::byVDIM);
   }
   GridFunction& nodes = *(mesh.GetNodes());
   //mm::Get().Push(nodes.GetData());
   //dbg("nodes:"); kVectorPrint(nodes.Size(),nodes.GetData());
   //cudaDeviceSynchronize();
   //dbg("nodes: %p", nodes.GetData());
   //dbg("nodes size: %d", nodes.Size());
   //GET_CONST_ADRS(nodes);
   //dbg("d_nodes: %p", d_nodes);
   //mm::Get().Rsync(nodes.GetData());
   //dbg("nodes:\n"); nodes.Print();
   //0 0 1 0 0.309017 0.951057 1.30902 0.951057
   //-0.809017 0.587785 -0.5 1.53884 -0.809017 -0.587785 -1.61803 0
   //0.309017 -0.951057 -0.5 -1.53884 1.30902 -0.951057 0.5 0
   //1.15451 0.475529 0.809019 0.951057 0.154508 0.475529 -0.0954915 1.24495
   //-0.654508 1.06331 -0.404508 0.293893 -1.21352 0.293893 -1.21352 -0.293892
   //-0.404508 -0.293893 -0.654508 -1.06331 -0.0954915 -1.24495 0.154508 -0.475529
   //0.809019 -0.951057 1.15451 -0.475529 0.654509 0.475529 -0.25 0.769421
   //-0.809016 0 -0.25 -0.76942 0.654509 -0.475529

   //assert(false);
   
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
   
   dbg("kGeomFill");
   kGeomFill(dims,
             elements,
             numDofs,
             elementMap,
             eMap.GetData(),
             nodes.GetData(),
             meshNodes.GetData());
   
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
   kGeom(dims, numDofs, numQuad, elements,
         maps->dofToQuadD,
         geom->meshNodes, geom->J, geom->invJ, geom->detJ);
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

// *****************************************************************************
MFEM_NAMESPACE_END
