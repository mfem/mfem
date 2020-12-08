// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "libceed/mass.hpp"

using namespace std;

namespace mfem
{

// PA Mass Integrator

// PA Mass Assemble kernel
void VectorMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T = mesh->GetElementTransformation(0);
   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(el, el, *T);
   if (DeviceCanUseCeed())
   {
      delete ceedDataPtr;
      ceedDataPtr = new CeedData;
      InitCeedCoeff(Q, *mesh, *ir, ceedDataPtr);
      return CeedPAMassAssemble(fes, *ir, *ceedDataPtr);
   }
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::COORDINATES |
                                    GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, Device::GetDeviceMemoryType());
   double coeff = 1.0;
   if (Q)
   {
      ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
      MFEM_VERIFY(cQ != NULL, "Only ConstantCoefficient is supported.");
      coeff = cQ->constant;
   }
   if (!(dim == 2 || dim == 3))
   {
      MFEM_ABORT("Dimension not supported.");
   }
   if (dim == 2)
   {
      const double constant = coeff;
      const int NE = ne;
      const int NQ = nq;
      auto w = ir->GetWeights().Read();
      auto J = Reshape(geom->J.Read(), NQ,2,2,NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(q,0,0,e);
            const double J12 = J(q,1,0,e);
            const double J21 = J(q,0,1,e);
            const double J22 = J(q,1,1,e);
            const double detJ = (J11*J22)-(J21*J12);
            v(q,e) =  w[q] * constant * detJ;
         }
      });
   }
   if (dim == 3)
   {
      const double constant = coeff;
      const int NE = ne;
      const int NQ = nq;
      auto W = ir->GetWeights().Read();
      auto J = Reshape(geom->J.Read(), NQ,3,3,NE);
      auto v = Reshape(pa_data.Write(), NQ,NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(q,0,0,e), J12 = J(q,0,1,e), J13 = J(q,0,2,e);
            const double J21 = J(q,1,0,e), J22 = J(q,1,1,e), J23 = J(q,1,2,e);
            const double J31 = J(q,2,0,e), J32 = J(q,2,1,e), J33 = J(q,2,2,e);
            const double detJ = J11 * (J22 * J33 - J32 * J23) -
            /* */               J21 * (J12 * J33 - J32 * J13) +
            /* */               J31 * (J12 * J23 - J22 * J13);
            v(q,e) = W[q] * constant * detJ;
         }
      });
   }
}

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static void PAVectorMassApply2D(const int NE,
                                const Array<double> &B_,
                                const Array<double> &_Bt,
                                const Vector &_op,
                                const Vector &_x,
                                Vector &_y,
                                const int d1d = 0,
                                const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(_Bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double sol_xy[max_Q1D][max_Q1D];
      for (int c = 0; c < VDIM; ++c)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[max_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               sol_x[qy] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B(qx,dx)* s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double d2q = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += d2q * sol_x[qx];
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] *= op(qx,qy,e);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[max_D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double s = sol_xy[qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt(dx,qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double q2d = Bt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,c,e) += q2d * sol_x[dx];
               }
            }
         }
      }
   });
}

template<const int T_D1D = 0,
         const int T_Q1D = 0>
static void PAVectorMassApply3D(const int NE,
                                const Array<double> &B_,
                                const Array<double> &_Bt,
                                const Vector &_op,
                                const Vector &_x,
                                Vector &_y,
                                const int d1d = 0,
                                const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 3;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto Bt = Reshape(_Bt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double sol_xyz[max_Q1D][max_Q1D][max_Q1D];
      for (int c = 0; c < VDIM; ++ c)
      {
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] = 0.0;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            double sol_xy[max_Q1D][max_Q1D];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] = 0.0;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               double sol_x[max_Q1D];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] = 0;
               }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double s = x(dx,dy,dz,c,e);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_x[qx] += B(qx,dx) * s;
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double wy = B(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_xy[qy][qx] += wy * sol_x[qx];
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double wz = B(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
                  }
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            double sol_xy[max_D1D][max_D1D];
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] = 0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               double sol_x[max_D1D];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double s = sol_xyz[qz][qy][qx];
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     sol_x[dx] += Bt(dx,qx) * s;
                  }
               }
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const double wy = Bt(dy,qy);
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     sol_xy[dy][dx] += wy * sol_x[dx];
                  }
               }
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double wz = Bt(dz,qz);
               for (int dy = 0; dy < D1D; ++dy)
               {
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     y(dx,dy,dz,c,e) += wz * sol_xy[dy][dx];
                  }
               }
            }
         }
      }
   });
}

static void PAVectorMassApply(const int dim,
                              const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &Bt,
                              const Vector &op,
                              const Vector &x,
                              Vector &y)
{
   if (dim == 2)
   {
      return PAVectorMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D);
   }
   if (dim == 3)
   {
      return PAVectorMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void VectorMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      CeedAddMult(ceedDataPtr, x, y);
   }
   else
   {
      PAVectorMassApply(dim, dofs1D, quad1D, ne, maps->B, maps->Bt, pa_data, x, y);
   }
}

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal2D(const int NE,
                                           const Array<double> &B_,
                                           const Array<double> &_Bt,
                                           const Vector &_op,
                                           Vector &_diag,
                                           const int d1d = 0,
                                           const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 2;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, NE);
   auto y = Reshape(_diag.ReadWrite(), D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double temp[max_Q1D][max_D1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            temp[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               temp[qx][dy] += B(qy, dy) * B(qy, dy) * op(qx, qy, e);
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            double temp1 = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               temp1 += B(qx, dx) * B(qx, dx) * temp[qx][dy];
            }
            y(dx, dy, 0, e) = temp1;
            y(dx, dy, 1, e) = temp1;
         }
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D = 0>
static void PAVectorMassAssembleDiagonal3D(const int NE,
                                           const Array<double> &B_,
                                           const Array<double> &_Bt,
                                           const Vector &_op,
                                           Vector &_diag,
                                           const int d1d = 0,
                                           const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int VDIM = 3;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(B_.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D, Q1D, Q1D, NE);
   auto y = Reshape(_diag.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double temp[max_Q1D][max_Q1D][max_D1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               temp[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  temp[qx][qy][dz] += B(qz, dz) * B(qz, dz) * op(qx, qy, qz, e);
               }
            }
         }
      }
      double temp2[max_Q1D][max_D1D][max_D1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               temp2[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  temp2[qx][dy][dz] += B(qy, dy) * B(qy, dy) * temp[qx][qy][dz];
               }
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               double temp3 = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  temp3 += B(qx, dx) * B(qx, dx)
                           * temp2[qx][dy][dz];
               }
               y(dx, dy, dz, 0, e) = temp3;
               y(dx, dy, dz, 1, e) = temp3;
               y(dx, dy, dz, 2, e) = temp3;
            }
         }
      }
   });
}

static void PAVectorMassAssembleDiagonal(const int dim,
                                         const int D1D,
                                         const int Q1D,
                                         const int NE,
                                         const Array<double> &B,
                                         const Array<double> &Bt,
                                         const Vector &op,
                                         Vector &y)
{
   if (dim == 2)
   {
      return PAVectorMassAssembleDiagonal2D(NE, B, Bt, op, y, D1D, Q1D);
   }
   else if (dim == 3)
   {
      return PAVectorMassAssembleDiagonal3D(NE, B, Bt, op, y, D1D, Q1D);
   }
   MFEM_ABORT("Dimension not implemented.");
}

void VectorMassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      CeedAssembleDiagonal(ceedDataPtr, diag);
   }
   else
   {
      PAVectorMassAssembleDiagonal(dim,
                                   dofs1D,
                                   quad1D,
                                   ne,
                                   maps->B,
                                   maps->Bt,
                                   pa_data,
                                   diag);
   }
}

} // namespace mfem
