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

#include "../general/forall.hpp"
#include "../fem/fespace.hpp"
#include "mesh_ext.hpp"
#include "../fem/bilininteg.hpp"
#include "../fem/gridfunc.hpp"
#include <cassert>

using namespace std;

namespace mfem
{

static void GeomFill(const int vdim,
                     const int NE, const int ND, const int NX,
                     const int *h_eMap, int *d_eMap,
                     const double *h_x, double *d_x)
{
   const DeviceArray d_h_eMap(h_eMap, ND*NE);
   DeviceArray d_d_eMap(d_eMap, ND*NE);
   const DeviceVector h_X(h_x, NX);
   DeviceVector d_X(d_x, vdim*ND*NE);
   MFEM_FORALL(e, NE,
   {
      for (int d = 0; d < ND; ++d)
      {
         const int lid = d+ND*e;
         const int gid = d_h_eMap[lid];
         d_d_eMap[lid] = gid;
         for (int v = 0; v < vdim; ++v)
         {
            const int moffset = v+vdim*lid;
            const int xoffset = v+vdim*gid;
            d_X[moffset] = h_X[xoffset];
         }
      }
   });
}

static void ReorderByVDim(const GridFunction *nodes)
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

static void ReorderByNodes(const GridFunction *nodes)
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

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PAGeom2D(const int NE,
                     const double* _B,
                     const double* _G,
                     const double* _X,
                     double* _Xq,
                     double* _J,
                     double* _invJ,
                     double* _detJ,
                     const int d1d = 0,
                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D;
   const int NQ = Q1D*Q1D;
   const DeviceTensor<2> B(_B, NQ, ND);
   const DeviceTensor<3> G(_G, 2,NQ, ND);
   const DeviceTensor<3> X(_X, 2,ND, NE);
   DeviceTensor<3> Xq(_Xq, 2, NQ, NE);
   DeviceTensor<4> J(_J, 2, 2, NQ, NE);
   DeviceTensor<4> invJ(_invJ, 2, 2, NQ, NE);
   DeviceMatrix detJ(_detJ, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D;
      const int NQ = Q1D*Q1D;
      double s_X[2*MAX_D1D*MAX_D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d +=NQ)
         {
            s_X[0+d*2] = X(0,d,e);
            s_X[1+d*2] = X(1,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double X0  = 0; double X1  = 0;
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double x = s_X[0+d*2];
            const double y = s_X[1+d*2];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
            X0 += b*x; X1 += b*y;
         }
         Xq(0,q,e) = X0; Xq(1,q,e) = X1;
         const double r_detJ = (J11 * J22)-(J12 * J21);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) =  J22 * r_idetJ;
         invJ(1,0,q,e) = -J12 * r_idetJ;
         invJ(0,1,q,e) = -J21 * r_idetJ;
         invJ(1,1,q,e) =  J11 * r_idetJ;
         detJ(q,e) = r_detJ;
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PAGeom3D(const int NE,
                     const double* _B,
                     const double* _G,
                     const double* _X,
                     double* _Xq,
                     double* _J,
                     double* _invJ,
                     double* _detJ,
                     const int d1d = 0,
                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D*D1D;
   const int NQ = Q1D*Q1D*Q1D;
   const DeviceTensor<2> B(_B, NQ, NE);
   const DeviceTensor<3> G(_G, 3, NQ, NE);
   const DeviceTensor<3> X(_X, 3, ND, NE);
   DeviceTensor<3> Xq(_Xq, 3, NQ, NE);
   DeviceTensor<4> J(_J, 3, 3, NQ, NE);
   DeviceTensor<4> invJ(_invJ, 3, 3, NQ, NE);
   DeviceMatrix detJ(_detJ, NQ, NE);
   MFEM_FORALL(e,NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D*D1D;
      const int NQ = Q1D*Q1D*Q1D;
      double s_nodes[3*MAX_D1D*MAX_D1D*MAX_D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d += NQ)
         {
            s_nodes[0+d*3] = X(0,d,e);
            s_nodes[1+d*3] = X(1,d,e);
            s_nodes[2+d*3] = X(2,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double wz = G(2,q,d);
            const double x = s_nodes[0+d*3];
            const double y = s_nodes[1+d*3];
            const double z = s_nodes[2+d*3];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
            Xq(0,q,e) = b*x; Xq(1,q,e) = b*y; Xq(2,q,e) = b*z;
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(2,0,q,e) = J13;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         J(2,1,q,e) = J23;
         J(0,2,q,e) = J31;
         J(1,2,q,e) = J32;
         J(2,2,q,e) = J33;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ(q,e) = r_detJ;
      }
   });
}

static void PAGeom(const int dim,
                   const int D1D,
                   const int Q1D,
                   const int NE,
                   const double *B,
                   const double *G,
                   const double *X,
                   double *Xq,
                   double *J,
                   double *invJ,
                   double *detJ)
{
   if (dim == 2)
   {
      switch ((D1D << 4) | Q1D)
      {
         case 0x22: PAGeom2D<2,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x23: PAGeom2D<2,3>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x24: PAGeom2D<2,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x25: PAGeom2D<2,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x32: PAGeom2D<3,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x34: PAGeom2D<3,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x42: PAGeom2D<4,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x44: PAGeom2D<4,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x45: PAGeom2D<4,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x46: PAGeom2D<4,6>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x58: PAGeom2D<5,8>(NE, B, G, X, Xq, J, invJ, detJ); break;
         default: PAGeom2D(NE, B, G, X, Xq, J, invJ, detJ, D1D, Q1D); break;
      }
      return;
   }
   if (dim == 3)
   {
      switch ((D1D << 4) | Q1D)
      {
         case 0x23: PAGeom3D<2,3>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x24: PAGeom3D<2,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x25: PAGeom3D<2,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x26: PAGeom3D<2,6>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x34: PAGeom3D<3,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         default: PAGeom3D(NE, B, G, X, Xq, J, invJ, detJ, D1D, Q1D); break;
      }
      return;
   }
   MFEM_ABORT("Unknown kernel.");
}

XTMesh::XTMesh(const Mesh *mesh, const IntegrationRule &ir) : sequence(-1)
{
   const GridFunction *nodes = mesh->GetNodes();
   const mfem::FiniteElementSpace *fespace = nodes->FESpace();
   const mfem::FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());
   const int vdim = fe->GetDim();
   const int NE   = fespace->GetNE();
   const int DND  = fe->GetDof();
   const int D1D  = fe->GetOrder() + 1;
   const int Q1D  = ir1D.GetNPoints();
   const int QND  = ir.GetNPoints();
   const bool byNODES = (fespace->GetOrdering() == Ordering::byNODES);
   if (byNODES) { ReorderByVDim(nodes); }
   mfem::Array<double> Enodes(vdim*DND*NE);
   const Table& e2dTable = fespace->GetElementToDofTable();
   const int *h_eMap = e2dTable.GetJ();
   mfem::Array<int> eMap(DND*NE);
   GeomFill(vdim, NE, DND, nodes->Size(), h_eMap, eMap, nodes->GetData(), Enodes);
   if (byNODES) { ReorderByNodes(nodes); }
   this->X.SetSize(vdim*QND*NE);
   this->J.SetSize(vdim*vdim*QND*NE);
   this->invJ.SetSize(vdim*vdim*QND*NE);
   this->detJ.SetSize(QND*NE);
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   PAGeom(vdim, D1D, Q1D, NE, maps->B, maps->G, Enodes,
          this->X, this->J, this->invJ, this->detJ);
   delete maps;
}

} // namespace mfem
