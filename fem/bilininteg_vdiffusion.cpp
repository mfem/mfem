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
#include "bilininteg.hpp"
#include "gridfunc.hpp"

using namespace std;

namespace mfem
{

// PA Vector Diffusion Integrator

void VectorDiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   const int eldim = el.GetDim();
   const int symmDims = (eldim * (eldim + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int NQ = ir->GetNPoints();
   dim = mesh->Dimension();
   NE = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   const int sdim = mesh->SpaceDimension();
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   D1D = maps->ndof;
   Q1D = maps->nqpt;
   pa_data.SetSize(symmDims * NQ * NE, Device::GetMemoryType());
   ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
   MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
   const double coeff = cQ->constant;

   dbg("D1D=%d, Q1D=%d, eldim=%d, sdim=%d", D1D, Q1D, dim, sdim);
   const Array<double> &w = ir->GetWeights();
   const Vector &j = geom->J;
   Vector &op = pa_data;

   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAVectorDiffusionSetup"); }
   if (dim == 2 && sdim == 2)
   { MFEM_ABORT("dim==2 && sdim==2 not supported in PAVectorDiffusionSetup"); }
   if (dim == 2 && sdim == 3)
   {
      // TODO: Same assemble than non-vector case: should use it!
      constexpr int DIM = 2;
      constexpr int VDIM = 3;
      const int NQ = Q1D*Q1D;
      auto W = w.Read();
      auto J = Reshape(j.Read(), NQ, VDIM, DIM, NE);
      auto y = Reshape(op.Write(), NQ, 3, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double wq = W[q];
            const double J11 = J(q,0,0,e);
            const double J21 = J(q,1,0,e);
            const double J31 = J(q,2,0,e);
            const double J12 = J(q,0,1,e);
            const double J22 = J(q,1,1,e);
            const double J32 = J(q,2,1,e);
            const double E = J11*J11 + J21*J21 + J31*J31;
            const double G = J12*J12 + J22*J22 + J32*J32;
            const double F = J11*J12 + J21*J22 + J31*J32;
            const double iw = 1.0 / sqrt(E*G - F*F);
            const double alpha = wq * coeff * iw;
            y(q,0,e) =  alpha * G; // 1,1
            y(q,1,e) = -alpha * F; // 1,2
            y(q,2,e) =  alpha * E; // 2,2
         }
      });
      //dbg("pa_data:\n"); pa_data.Print();
   }
   if (dim == 3)
   { MFEM_ABORT("dim==3 not supported in PAVectorDiffusionSetup"); }
}


// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAVectorDiffusionApply2D(const int NE,
                              const Array<double> &b,
                              const Array<double> &g,
                              const Array<double> &bt,
                              const Array<double> &gt,
                              const Vector &_op,
                              const Vector &_x,
                              Vector &_y,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[max_Q1D][max_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[max_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;

            const double O11 = op(q,0,e);
            const double O12 = op(q,1,e);
            const double O22 = op(q,2,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[max_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

/*static void ReorderByNodes(const int NE, const int dofs1D,
                           const int vdim, Vector &x)
{
   dbg("");
   const int size = x.Size();
   double *data = x.GetData();
   int i, j, k;
   const int ndofs = NE*dofs1D*dofs1D;
   double *temp = new double[size];
   k = 0;
   for (j = 0; j < ndofs; j++)
   {
      for (i = 0; i < vdim; i++)
      {
         const int offset = j+i*ndofs;
         temp[offset] = data[k++];
      }
   }
   for (i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}*/

/*static void ReorderByVdim(const int NE, const int dofs1D, const int vdim,
                          Vector &x)
{
   dbg("");
   const int size = x.Size();
   double *data = x.GetData();
   int i, j, k;
   const int ndofs = NE*dofs1D*dofs1D;
   double *temp = new double[size];
   k = 0;
   for (j = 0; j < ndofs; j++)
   {
      for (i = 0; i < vdim; i++)
      {
         const int offset = j+i*ndofs;
         temp[k++] = data[offset];
      }
   }
   for (i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}*/

// PA Vector Diffusion Apply kernel
void VectorDiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   static bool viewed = false;
   if (!viewed)
   {
      viewed = true;
      dbg("D1D=%d, Q1D=%d", D1D, Q1D);
      dbg("Sizes x:%d, y:%d", x.Size(), y.Size());
      dbg("B:\n"); maps->B.Print();
      dbg("G:\n"); maps->G.Print();
      dbg("Bt\n");  maps->Bt.Print();
      dbg("Gt:\n");  maps->Gt.Print();
      dbg("pa_data:\n"); pa_data.Print();
   }

   dbg("x:\n"); x.Print();
   MFEM_VERIFY(D1D==2 && Q1D==2,"");
   PAVectorDiffusionApply2D<2,2>(NE, maps->B, maps->G,
                                 maps->Bt, maps->Gt,
                                 pa_data, x, y);
   dbg("y:\n"); y.Print();
   mfem_error();
}

} // namespace mfem
