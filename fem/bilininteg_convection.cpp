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

// PA Convection Integrator

// PA Convection Assemble 2D kernel
static void PAConvectionSetup2D(const int Q1D,
                               const int NE,
                               const Array<double> &w,
                               const Vector &j,
                               const Vector &coeff,
                               const double alpha,
                               Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   // auto c = Reshape(coeff.Read(), NQ, 2, NE);
   const double c0 = coeff[0];
   const double c1 = coeff[1];
   auto y = Reshape(op.Write(), NQ, 2, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double w = alpha * W[q];
         // const double wx = w * c(q,0,e);
         // const double wy = w * c(q,1,e);
         const double wx = w * c0;
         const double wy = w * c1;
         //w*J^-T
         y(q,0,e) =  wx * J22 - wy * J21; // 1
         y(q,1,e) = -wy * J21 + wy * J11; // 2
      }
   });
}

// PA Convection Assemble 3D kernel
static void PAConvectionSetup3D(const int Q1D,
                               const int NE,
                               const Array<double> &w,
                               const Vector &j,
                               const Vector &coeff,
                               const double alpha,
                               Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   // auto c = Reshape(coeff.Read(), NQ, 2, NE);
   const double c0 = coeff[0];
   const double c1 = coeff[1];
   const double c2 = coeff[2];
   auto y = Reshape(op.Write(), NQ, 3, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         // const double detJ = J11 * (J22 * J33 - J32 * J23) -
         // /* */               J21 * (J12 * J33 - J32 * J13) +
         // /* */               J31 * (J12 * J23 - J22 * J13);
         const double w = alpha * W[q];
         // const double wx = w * c(q,0,e);
         // const double wy = w * c(q,1,e);
         // const double wz = w * c(q,2,e);
         const double wx = w * c0;
         const double wy = w * c1;
         const double wz = w * c2;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // detJ q . J^{-T} = q . adj(J)^T
         y(q,0,e) =  wx * A11 + wy * A21 + wz * A31; // 1,1
         y(q,1,e) =  wx * A12 + wy * A22 + wz * A32; // 1,1
         y(q,2,e) =  wx * A13 + wy * A23 + wz * A33; // 1,1
      }
   });
}

static void PAConvectionSetup(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &W,
                             const Vector &J,
                             const Vector &coeff,
                             const double alpha,
                             Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAConvectionSetup"); }
   if (dim == 2)
   {
      PAConvectionSetup2D(Q1D, NE, W, J, coeff, alpha, op);
   }
   if (dim == 3)
   {
      PAConvectionSetup3D(Q1D, NE, W, J, coeff, alpha, op);
   }
}

// PA Convection Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionApply2D(const int NE,
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
   auto op = Reshape(_op.Read(), Q1D, Q1D, 2, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double u[max_D1D][max_D1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
        for (int dx = 0; dx < D1D; ++dx)
        {
          u[dy][dx] = x(dx,dy,e);
        }
      }
      double Bu[max_D1D][max_Q1D];
      double Gu[max_D1D][max_Q1D];
      for (int dy = 0; dy < D1D; ++dy)
      {
        for (int qx = 0; qx < Q1D; ++qx)
        {
          Bu[dy][qx] = 0.0;
          Gu[dy][qx] = 0.0;
          for (int dx = 0; dx < D1D; ++dx)
          {
            const double bx  = B(qx,dx);
            const double gx  = G(qx,dx);
            const double x = u[dy][dx];
            Bu[dy][qx] += bx * x;
            Gu[dy][qx] += gx * x;
          }
        }
      }
      double GBu[max_Q1D][max_Q1D];
      double BGu[max_Q1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
        for (int qy = 0; qy < Q1D; ++qy)
        {
          GBu[qy][qx] = 0.0;
          BGu[qy][qx] = 0.0;
          for (int dy = 0; dy < D1D; ++dy)
          {
            const double bx  = B(qy,dy);
            const double gx  = G(qy,dy);
            GBu[qy][qx] += gx * Bu[dy][qx];
            BGu[qy][qx] += bx * Gu[dy][qx];
          }
        }
      }
      // Calculate Dxy, xDy in plane
      double DGu[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O1 = op(qx,qy,0,e);
            const double O2 = op(qx,qy,1,e);

            const double gradX = BGu[qy][qx];
            const double gradY = GBu[qy][qx];

            DGu[qy][qx] = (O1 * gradX) + (O2 * gradY);
         }
      }
      double BDGu[max_D1D][max_Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
        for (int dy = 0; dy < D1D; ++dy)
        {
          BDGu[dy][qx] = 0.0;
          for (int qy = 0; qy < Q1D; ++qy)
          {
            const double w  = Bt(dy,qy);
            BDGu[dy][qx] += w * DGu[qy][qx];
          }
        }
      }
      double BBDGu[max_D1D][max_Q1D];
      for (int dx = 0; dx < D1D; ++dx)
      {
        for (int dy = 0; dy < D1D; ++dy)
        {
          BBDGu[dy][dx] = 0.0;
          for (int qx = 0; qx < Q1D; ++qx)
          {
            const double w  = Bt(dx,qx);
            BBDGu[dy][dx] += w * BDGu[dy][qx];
          }
          y(dx,dy,e) += BBDGu[dy][dx];
        }
      }
   });
}

// PA Convection Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PAConvectionApply3D(const int NE,
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
  auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, 3, NE);
  auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
  auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
  MFEM_FORALL(e, NE,
  {
    const int D1D = T_D1D ? T_D1D : d1d;
    const int Q1D = T_Q1D ? T_Q1D : q1d;
    // the following variables are evaluated at compile time
    constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
    constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

    double u[max_D1D][max_D1D][max_D1D];
    for (int dz = 0; dz < D1D; ++dz)
    {
      for (int dy = 0; dy < D1D; ++dy)
      {
        for (int dx = 0; dx < D1D; ++dx)
        {
          u[dz][dy][dx] = x(dx,dy,dz,e);
        }
      }
    }
    double Bu[max_D1D][max_D1D][max_Q1D];
    double Gu[max_D1D][max_D1D][max_Q1D];
    for (int dz = 0; dz < D1D; ++dz)
    {
      for (int dy = 0; dy < D1D; ++dy)
      {
        for (int qx = 0; qx < Q1D; ++qx)
        {
          Bu[dz][dy][qx] = 0.0;
          Gu[dz][dy][qx] = 0.0;
          for (int dx = 0; dx < D1D; ++dx)
          {
            const double bx  = B(qx,dx);
            const double gx  = G(qx,dx);
            const double x = u[dz][dy][dx];
            Bu[dz][dy][qx] += bx * x;
            Gu[dz][dy][qx] += gx * x;
          }
        }
      }
    }
    double BBu[max_D1D][max_Q1D][max_Q1D];
    double GBu[max_D1D][max_Q1D][max_Q1D];
    double BGu[max_D1D][max_Q1D][max_Q1D];
    for (int dz = 0; dz < D1D; ++dz)
    {
      for (int qx = 0; qx < Q1D; ++qx)
      {
        for (int qy = 0; qy < Q1D; ++qy)
        {
          BBu[dz][qy][qx] = 0.0;
          GBu[dz][qy][qx] = 0.0;
          BGu[dz][qy][qx] = 0.0;
          for (int dy = 0; dy < D1D; ++dy)
          {
            const double bx  = B(qy,dy);
            const double gx  = G(qy,dy);
            BBu[dz][qy][qx] += bx * Bu[dz][dy][qx];
            GBu[dz][qy][qx] += gx * Bu[dz][dy][qx];
            BGu[dz][qy][qx] += bx * Gu[dz][dy][qx];
          }
        }
      }
    }
    double GBBu[max_D1D][max_Q1D][max_Q1D];
    double BGBu[max_D1D][max_Q1D][max_Q1D];
    double BBGu[max_D1D][max_Q1D][max_Q1D];
    for (int qx = 0; qx < Q1D; ++qx)
    {
      for (int qy = 0; qy < Q1D; ++qy)
      {
        for (int qz = 0; qz < Q1D; ++qz)
        {
          GBBu[qz][qy][qx] = 0.0;
          BGBu[qz][qy][qx] = 0.0;
          BBGu[qz][qy][qx] = 0.0;
          for (int dz = 0; dz < D1D; ++dz)
          {
            const double bx  = B(qz,dz);
            const double gx  = G(qz,dz);
            GBBu[qz][qy][qx] += gx * BBu[dz][qy][qx];
            BGBu[qz][qy][qx] += bx * GBu[dz][qy][qx];
            BBGu[qz][qy][qx] += bx * BGu[dz][qy][qx];
          }
        }
      }
    }
    // Calculate Dxy, xDy in plane
    double DGu[max_Q1D][max_Q1D][max_Q1D];
    for (int qz = 0; qz < Q1D; ++qz)
    {
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double O1 = op(qx,qy,qz,0,e);
            const double O2 = op(qx,qy,qz,1,e);
            const double O3 = op(qx,qy,qz,1,e);

            const double gradX = BBGu[qz][qy][qx];
            const double gradY = BGBu[qz][qy][qx];
            const double gradZ = GBBu[qz][qy][qx];

            DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
         }
      }
    }
    double BDGu[max_D1D][max_Q1D][max_Q1D];
    for (int qx = 0; qx < Q1D; ++qx)
    {
      for (int qy = 0; qy < Q1D; ++qy)
      {
        for (int dz = 0; dz < D1D; ++dz)
        {
          BDGu[dz][qy][qx] = 0.0;
          for (int qz = 0; qz < Q1D; ++qz)
          {
            const double w  = Bt(dz,qz);
            BDGu[dz][qy][qx] += w * DGu[qz][qy][qx];
          }
        }
      }
    }
    double BBDGu[max_D1D][max_Q1D][max_Q1D];
    for (int dz = 0; dz < D1D; ++dz)
    {
      for (int qx = 0; qx < Q1D; ++qx)
      {
        for (int dy = 0; dy < D1D; ++dy)
        {
          BBDGu[dz][dy][qx] = 0.0;
          for (int qy = 0; qy < Q1D; ++qy)
          {
            const double w  = Bt(dy,qy);
            BBDGu[dz][dy][qx] += w * BDGu[dz][qy][qx];
          }
        }
      }
    }
    double BBBDGu[max_D1D][max_Q1D][max_Q1D];
    for (int dz = 0; dz < D1D; ++dz)
    {
      for (int dy = 0; dy < D1D; ++dy)
      {
        for (int dx = 0; dx < D1D; ++dx)
        {
          BBBDGu[dz][dy][dx] = 0.0;
          for (int qx = 0; qx < Q1D; ++qx)
          {
            const double w  = Bt(dx,qx);
            BBBDGu[dz][dy][dx] += w * BBDGu[dz][dy][qx];
          }
          y(dx,dy,dz,e) += BBBDGu[dz][dy][dx];
        }
      }
    }
  });
}

void ConvectionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   const int dims = el.GetDim();
   const int symmDims = dims; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   VectorConstantCoefficient *cQ = dynamic_cast<VectorConstantCoefficient*>(Q);//TODO not constant coeff
   MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
   const Vector& coeff = cQ->GetVec();   
   PAConvectionSetup(dim, dofs1D, quad1D, ne, ir->GetWeights(), geom->J,
                    coeff, alpha, pa_data);
}

static void PAConvectionApply(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &B,
                             const Array<double> &G,
                             const Array<double> &Bt,
                             const Array<double> &Gt,
                             const Vector &op,
                             const Vector &x,
                             Vector &y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PAConvectionApply2D<2,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return PAConvectionApply2D<3,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return PAConvectionApply2D<4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return PAConvectionApply2D<5,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return PAConvectionApply2D<6,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return PAConvectionApply2D<7,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return PAConvectionApply2D<8,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return PAConvectionApply2D<9,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAConvectionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PAConvectionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return PAConvectionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return PAConvectionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return PAConvectionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return PAConvectionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return PAConvectionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return PAConvectionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PAConvectionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Convection Apply kernel
void ConvectionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PAConvectionApply(dim, dofs1D, quad1D, ne,
                    maps->B, maps->G, maps->Bt, maps->Gt,
                    pa_data, x, y);
}

} // namespace mfem
