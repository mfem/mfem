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
#include "../linalg/kernels/vector.hpp"

namespace mfem
{

// *****************************************************************************
static long sequence = -1;
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
static void Geom2D(const int NUM_DOFS,
                   const int NUM_QUAD,
                   const int ne,
                   const double* __restrict B,
                   const double* __restrict G,
                   const double* __restrict X,
                   double* __restrict x,
                   double* __restrict J,
                   double* __restrict invJ,
                   double* __restrict detJ)
{
   // number of doubles in shared memory per threads
   const int Nspt = 2*NUM_DOFS;
   MFEM_FORALL_SHARED(e, ne, Nspt,
   {
      double *s_X = __shared;
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d +=NUM_QUAD)
         {
            s_X[ijN(0,d,2)] = X[ijkNM(0,d,e,2,NUM_DOFS)];
            s_X[ijN(1,d,2)] = X[ijkNM(1,d,e,2,NUM_DOFS)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = G[ijkNM(0,q,d,2,NUM_QUAD)];
            const double wy = G[ijkNM(1,q,d,2,NUM_QUAD)];
            const double x = s_X[ijN(0,d,2)];
            const double y = s_X[ijN(1,d,2)];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
         }
         const double r_detJ = (J11 * J22)-(J12 * J21);
         assert(r_detJ!=0.0);
         J[ijklNM(0,0,q,e,2,NUM_QUAD)] = J11;
         J[ijklNM(1,0,q,e,2,NUM_QUAD)] = J12;
         J[ijklNM(0,1,q,e,2,NUM_QUAD)] = J21;
         J[ijklNM(1,1,q,e,2,NUM_QUAD)] = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0,0,q,e,2,NUM_QUAD)] =  J22 * r_idetJ;
         invJ[ijklNM(1,0,q,e,2,NUM_QUAD)] = -J12 * r_idetJ;
         invJ[ijklNM(0,1,q,e,2,NUM_QUAD)] = -J21 * r_idetJ;
         invJ[ijklNM(1,1,q,e,2,NUM_QUAD)] =  J11 * r_idetJ;
         detJ[ijN(q,e,NUM_QUAD)] = r_detJ;
      }
   });
}

// *****************************************************************************
static void Geom3D(const int NUM_DOFS,
                   const int NUM_QUAD,
                   const int ne,
                   const double* __restrict B,
                   const double* __restrict G,
                   const double* __restrict X,
                   double* __restrict x,
                   double* __restrict J,
                   double* __restrict invJ,
                   double* __restrict detJ)
{
   // number of doubles in shared memory per threads
   const int Nspt = 3 * NUM_DOFS;
   MFEM_FORALL_SHARED(e, ne, Nspt,
   {
      double *s_X = __shared;
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d += NUM_QUAD)
         {
            s_X[ijN(0,d,3)] = X[ijkNM(0,d,e,3,NUM_DOFS)];
            s_X[ijN(1,d,3)] = X[ijkNM(1,d,e,3,NUM_DOFS)];
            s_X[ijN(2,d,3)] = X[ijkNM(2,d,e,3,NUM_DOFS)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = G[ijkNM(0,q,d,3,NUM_QUAD)];
            const double wy = G[ijkNM(1,q,d,3,NUM_QUAD)];
            const double wz = G[ijkNM(2,q,d,3,NUM_QUAD)];
            const double x = s_X[ijN(0,d,3)];
            const double y = s_X[ijN(1,d,3)];
            const double z = s_X[ijN(2,d,3)];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         assert(r_detJ!=0.0);
         J[ijklNM(0,0,q,e,3,NUM_QUAD)] = J11;
         J[ijklNM(1,0,q,e,3,NUM_QUAD)] = J12;
         J[ijklNM(2,0,q,e,3,NUM_QUAD)] = J13;
         J[ijklNM(0,1,q,e,3,NUM_QUAD)] = J21;
         J[ijklNM(1,1,q,e,3,NUM_QUAD)] = J22;
         J[ijklNM(2,1,q,e,3,NUM_QUAD)] = J23;
         J[ijklNM(0,2,q,e,3,NUM_QUAD)] = J31;
         J[ijklNM(1,2,q,e,3,NUM_QUAD)] = J32;
         J[ijklNM(2,2,q,e,3,NUM_QUAD)] = J33;

         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ[ijklNM(1,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ[ijklNM(2,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ[ijklNM(0,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ[ijklNM(1,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ[ijklNM(2,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ[ijklNM(0,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ[ijklNM(1,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ[ijklNM(2,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
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
   const double *B = (double*) mm::ptr(maps->B);
   const double *G = (double*) mm::ptr(maps->G);
   const double *X = (double*) mm::ptr(geom->nodes);
   double *x = (double*) mm::ptr(geom->X);
   double *J = (double*) mm::ptr(geom->J);
   double *invJ = (double*) mm::ptr(geom->invJ);
   double *detJ = (double*) mm::ptr(geom->detJ);
   if (dims==2)
   {
      Geom2D(numDofs, numQuad, elements, B, G, X, x, J, invJ, detJ);
   }

   if (dims==3)
   {
      Geom3D(numDofs, numQuad, elements, B, G, X, x, J, invJ, detJ);
   }
   return geom;
}

// *****************************************************************************
GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
   const bool geom_to_allocate = sequence < fes.GetSequence();
   sequence = fes.GetSequence();
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
            eMap,
            nodes->GetData(),
            meshNodes);
   if (geom_to_allocate)
   {
      geom->nodes.SetSize(dims*numDofs*elements);
      geom->eMap.SetSize(numDofs*elements);
   }
   geom->nodes = meshNodes;
   geom->eMap = eMap;
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      geom->X.SetSize(dims*numQuad*elements);
      geom->J.SetSize(dims*dims*numQuad*elements);
      geom->invJ.SetSize(dims*dims*numQuad*elements);
      geom->detJ.SetSize(numQuad*elements);
   }
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   const double *B = (double*) mm::ptr(maps->B);
   const double *G = (double*) mm::ptr(maps->G);
   const double *X = (double*) mm::ptr(geom->nodes);
   double *x = (double*) mm::ptr(geom->X);
   double *J = (double*) mm::ptr(geom->J);
   double *invJ = (double*) mm::ptr(geom->invJ);
   double *detJ = (double*) mm::ptr(geom->detJ);
   if (dims==2)
   {
      Geom2D(numDofs, numQuad, elements, B, G, X, x, J, invJ, detJ);
   }

   if (dims==3)
   {
      Geom3D(numDofs, numQuad, elements, B, G, X, x, J, invJ, detJ);
   }
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
