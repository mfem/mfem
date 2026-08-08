// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_UTIL
#define MFEM_LOR_UTIL

#include "../../config/config.hpp"
#include "../../general/backends.hpp"
#include "../../general/globals.hpp"
#include "../../linalg/dtensor.hpp"

namespace mfem
{

MFEM_HOST_DEVICE inline real_t Det2D(DeviceMatrix &J)
{
   return J(0,0)*J(1,1) - J(1,0)*J(0,1);
}

MFEM_HOST_DEVICE inline real_t Det3D(DeviceMatrix &J)
{
   return J(0,0) * (J(1,1) * J(2,2) - J(2,1) * J(1,2)) -
          J(1,0) * (J(0,1) * J(2,2) - J(2,1) * J(0,2)) +
          J(2,0) * (J(0,1) * J(1,2) - J(1,1) * J(0,2));
}

template <int ORDER, int SDIM=2>
MFEM_HOST_DEVICE inline void LORVertexCoordinates2D(
   const real_t *X, int iel_ho, int kx, int ky, real_t **v)
{
   const int nd1d = ORDER + 1;
   const int nvert_per_el = nd1d*nd1d;

   const int v0 = kx + nd1d*ky;
   const int v1 = kx + 1 + nd1d*ky;
   const int v2 = kx + 1 + nd1d*(ky + 1);
   const int v3 = kx + nd1d*(ky + 1);

   const int e0 = SDIM*(v0 + nvert_per_el*iel_ho);
   const int e1 = SDIM*(v1 + nvert_per_el*iel_ho);
   const int e2 = SDIM*(v2 + nvert_per_el*iel_ho);
   const int e3 = SDIM*(v3 + nvert_per_el*iel_ho);

   // Vertex coordinates
   v[0][0] = X[e0 + 0];
   v[1][0] = X[e0 + 1];

   v[0][1] = X[e1 + 0];
   v[1][1] = X[e1 + 1];

   v[0][2] = X[e2 + 0];
   v[1][2] = X[e2 + 1];

   v[0][3] = X[e3 + 0];
   v[1][3] = X[e3 + 1];

   if (SDIM == 3)
   {
      v[2][0] = X[e0 + 2];
      v[2][1] = X[e1 + 2];
      v[2][2] = X[e2 + 2];
      v[2][3] = X[e3 + 2];
   }
}

template <int ORDER>
MFEM_HOST_DEVICE inline void LORVertexCoordinates3D(
   const real_t *X, int iel_ho, int kx, int ky, int kz,
   real_t vx[8], real_t vy[8], real_t vz[8])
{
   const int dim = 3;
   const int nd1d = ORDER + 1;
   const int nvert_per_el = nd1d*nd1d*nd1d;

   const int v0 = kx + nd1d*(ky + nd1d*kz);
   const int v1 = kx + 1 + nd1d*(ky + nd1d*kz);
   const int v2 = kx + 1 + nd1d*(ky + 1 + nd1d*kz);
   const int v3 = kx + nd1d*(ky + 1 + nd1d*kz);
   const int v4 = kx + nd1d*(ky + nd1d*(kz + 1));
   const int v5 = kx + 1 + nd1d*(ky + nd1d*(kz + 1));
   const int v6 = kx + 1 + nd1d*(ky + 1 + nd1d*(kz + 1));
   const int v7 = kx + nd1d*(ky + 1 + nd1d*(kz + 1));

   const int e0 = dim*(v0 + nvert_per_el*iel_ho);
   const int e1 = dim*(v1 + nvert_per_el*iel_ho);
   const int e2 = dim*(v2 + nvert_per_el*iel_ho);
   const int e3 = dim*(v3 + nvert_per_el*iel_ho);
   const int e4 = dim*(v4 + nvert_per_el*iel_ho);
   const int e5 = dim*(v5 + nvert_per_el*iel_ho);
   const int e6 = dim*(v6 + nvert_per_el*iel_ho);
   const int e7 = dim*(v7 + nvert_per_el*iel_ho);

   vx[0] = X[e0 + 0];
   vy[0] = X[e0 + 1];
   vz[0] = X[e0 + 2];

   vx[1] = X[e1 + 0];
   vy[1] = X[e1 + 1];
   vz[1] = X[e1 + 2];

   vx[2] = X[e2 + 0];
   vy[2] = X[e2 + 1];
   vz[2] = X[e2 + 2];

   vx[3] = X[e3 + 0];
   vy[3] = X[e3 + 1];
   vz[3] = X[e3 + 2];

   vx[4] = X[e4 + 0];
   vy[4] = X[e4 + 1];
   vz[4] = X[e4 + 2];

   vx[5] = X[e5 + 0];
   vy[5] = X[e5 + 1];
   vz[5] = X[e5 + 2];

   vx[6] = X[e6 + 0];
   vy[6] = X[e6 + 1];
   vz[6] = X[e6 + 2];

   vx[7] = X[e7 + 0];
   vy[7] = X[e7 + 1];
   vz[7] = X[e7 + 2];
}

template <int SDIM=2>
MFEM_HOST_DEVICE inline void Jacobian2D(
   const real_t x, const real_t y, real_t **v, DeviceMatrix &J);

template <> MFEM_HOST_DEVICE inline void Jacobian2D<2>(
   const real_t x, const real_t y, real_t **v, DeviceMatrix &J)
{
   J(0,0) = -(1-y)*v[0][0] + (1-y)*v[0][1] + y*v[0][2] - y*v[0][3];
   J(0,1) = -(1-x)*v[0][0] - x*v[0][1] + x*v[0][2] + (1-x)*v[0][3];

   J(1,0) = -(1-y)*v[1][0] + (1-y)*v[1][1] + y*v[1][2] - y*v[1][3];
   J(1,1) = -(1-x)*v[1][0] - x*v[1][1] + x*v[1][2] + (1-x)*v[1][3];
}

template <> MFEM_HOST_DEVICE inline void Jacobian2D<3>(
   const real_t x, const real_t y, real_t **v, DeviceMatrix &J)
{
   J(0,0) = -(1-y)*v[0][0] + (1-y)*v[0][1] + y*v[0][2] - y*v[0][3];
   J(0,1) = -(1-x)*v[0][0] - x*v[0][1] + x*v[0][2] + (1-x)*v[0][3];

   J(1,0) = -(1-y)*v[1][0] + (1-y)*v[1][1] + y*v[1][2] - y*v[1][3];
   J(1,1) = -(1-x)*v[1][0] - x*v[1][1] + x*v[1][2] + (1-x)*v[1][3];

   J(2,0) = -(1-y)*v[2][0] + (1-y)*v[2][1] + y*v[2][2] - y*v[2][3];
   J(2,1) = -(1-x)*v[2][0] - x*v[2][1] + x*v[2][2] + (1-x)*v[2][3];
}

MFEM_HOST_DEVICE inline void Get2DMatrixCoeff(
   ConstDeviceTensor<4> coeff, bool is_const, int i, int j, int e, real_t *vals)
{
   if (is_const) { i = j = e = 0; }
   const int vdim = coeff.GetShape()[0];
   if (vdim == 1)
   {
      vals[0] = vals[3] = coeff(0, i, j, e);
      vals[1] = vals[2] = 0.0;
   }
   else if (vdim == 2)
   {
      vals[0] = coeff(0, i, j, e);
      vals[3] = coeff(1, i, j, e);
      vals[1] = vals[2] = 0.0;
   }
   else
   {
      vals[0] = coeff(0, i, j, e);
      vals[1] = coeff(1, i, j, e);
      vals[2] = coeff(2, i, j, e);
      vals[3] = coeff(3, i, j, e);
   }
}

MFEM_HOST_DEVICE inline void Get2DSurfaceMatrixCoeff(
   ConstDeviceTensor<4> coeff, bool is_const, int i, int j, int e, real_t *vals)
{
   if (is_const) { i = j = e = 0; }
   const int vdim = coeff.GetShape()[0];
   if (vdim == 1)
   {
      vals[0] = vals[4] = vals[8] = coeff(0, i, j, e);
      vals[1] = vals[2] = vals[3] = vals[5] = vals[6] = vals[7] = 0.0;
   }
   else if (vdim == 3)
   {
      vals[0] = coeff(0, i, j, e);
      vals[4] = coeff(1, i, j, e);
      vals[8] = coeff(2, i, j, e);
      vals[1] = vals[2] = vals[3] = vals[5] = vals[6] = vals[7] = 0.0;
   }
   else
   {
      for (int k = 0; k < vdim; ++k) { vals[k] = coeff(k, i, j, e); }
   }
}

template <int ORDER, int SDIM, bool RT, bool ND>
MFEM_HOST_DEVICE inline void SetupLORQuadData2D(
   const real_t *X, bool const_1, ConstDeviceTensor<4> coeff_1,
   bool const_2, ConstDeviceTensor<3> coeff_2,
   int iel_ho, int kx, int ky, DeviceTensor<3> &Q)
{
   real_t vx[4], vy[4], vz[4];
   real_t *v[] = {vx, vy, vz};
   LORVertexCoordinates2D<ORDER,SDIM>(X, iel_ho, kx, ky, v);

   for (int iqy=0; iqy<2; ++iqy)
   {
      for (int iqx=0; iqx<2; ++iqx)
      {
         // c_2 is always a scalar coefficient
         const real_t c_2 = const_2 ? coeff_2(0,0,0) : coeff_2(kx+iqx, ky+iqy, iel_ho);

         const real_t x = iqx;
         const real_t y = iqy;
         const real_t w = 1.0/4.0;

         real_t J_[SDIM*2];
         DeviceTensor<2> J(J_, SDIM, 2);

         Jacobian2D<SDIM>(x, y, v, J);

         if (SDIM == 2)
         {
            // c_1 may be a 2x2 matrix coefficient
            real_t c_1[4];
            Get2DMatrixCoeff(coeff_1, const_1, kx+iqx, ky+iqy, iel_ho, c_1);
            const real_t e11 = c_1[0];
            const real_t e21 = c_1[1];
            const real_t e12 = c_1[2];
            const real_t e22 = c_1[3];

            const real_t detJ = Det2D(J);
            const real_t w_detJ = w/detJ;

            const real_t a = J(0,0);
            const real_t b = J(0,1);
            const real_t c = J(1,0);
            const real_t d = J(1,1);

            if (RT)
            {
               // Q = (w/detJ) * J^T K J
               const real_t M11 = a*e11 + c*e21;
               const real_t M12 = b*e11 + d*e21;
               const real_t M21 = a*e12 + c*e22;
               const real_t M22 = b*e12 + d*e22;

               Q(0,iqy,iqx) = w_detJ * (a*M11 + c*M21); // 1,1
               Q(1,iqy,iqx) = w_detJ * (b*M11 + d*M21); // 1,2
               Q(2,iqy,iqx) = w_detJ * (b*M12 + d*M22); // 2,2
            }
            else
            {
               // Q = (w/detJ) * adj(J) K adj(J)^T
               const real_t M11 = d*e11 - b*e21;
               const real_t M12 = d*e12 - b*e22;
               const real_t M21 = -c*e11 + a*e21;
               const real_t M22 = -c*e12 + a*e22;

               Q(0,iqy,iqx) = w_detJ * (d*M11 - b*M12);  // 1,1
               Q(1,iqy,iqx) = w_detJ * (-c*M11 + a*M12); // 1,2
               Q(2,iqy,iqx) = w_detJ * (-c*M21 + a*M22); // 2,2
            }
            Q(3,iqy,iqx) = c_2 * ((ND || RT) ? w_detJ : w*detJ);
         }
         else
         {
            // c_1 may be a 3x3 matrix coefficient. Assume symmetry.
            real_t c_1[9];
            Get2DSurfaceMatrixCoeff(coeff_1, const_1, kx+iqx, ky+iqy, iel_ho, c_1);
            const real_t e00 = c_1[0];
            const real_t e01 = c_1[3];
            const real_t e11 = c_1[4];
            const real_t e02 = c_1[6];
            const real_t e12 = c_1[7];
            const real_t e22 = c_1[8];

            const real_t E = J(0,0)*J(0,0) + J(1,0)*J(1,0) + J(2,0)*J(2,0);
            const real_t F = J(0,0)*J(0,1) + J(1,0)*J(1,1) + J(2,0)*J(2,1);
            const real_t G = J(0,1)*J(0,1) + J(1,1)*J(1,1) + J(2,1)*J(2,1);

            const real_t detg = E*G - F*F;
            const real_t detJ = sqrt(detg);

            // B = J^T eps J, B = [ B00 B01 ]
            //                    [ B10 B11 ]
            const real_t B00 =
               J(0,0)*(e00*J(0,0) + e01*J(1,0) + e02*J(2,0)) +
               J(1,0)*(e01*J(0,0) + e11*J(1,0) + e12*J(2,0)) +
               J(2,0)*(e02*J(0,0) + e12*J(1,0) + e22*J(2,0));

            const real_t B01 =
               J(0,0)*(e00*J(0,1) + e01*J(1,1) + e02*J(2,1)) +
               J(1,0)*(e01*J(0,1) + e11*J(1,1) + e12*J(2,1)) +
               J(2,0)*(e02*J(0,1) + e12*J(1,1) + e22*J(2,1));

            const real_t B11 =
               J(0,1)*(e00*J(0,1) + e01*J(1,1) + e02*J(2,1)) +
               J(1,1)*(e01*J(0,1) + e11*J(1,1) + e12*J(2,1)) +
               J(2,1)*(e02*J(0,1) + e12*J(1,1) + e22*J(2,1));

            if (RT)
            {
               const real_t s = w / detJ;
               Q(0,iqy,iqx) = s * B00; // 1,1
               Q(1,iqy,iqx) = s * B01; // 1,2
               Q(2,iqy,iqx) = s * B11; // 2,2
            }
            else
            {
               const real_t C00 = G*(G*B00 - F*B01) - F*(G*B01 - F*B11);
               const real_t C01 = G*(-F*B00 + E*B01) - F*(-F*B01 + E*B11);
               const real_t C11 = -F*(-F*B00 + E*B01) + E*(-F*B01 + E*B11);
               const real_t s = w / (detJ * detg);
               Q(0,iqy,iqx) = s * C00; // 1,1
               Q(1,iqy,iqx) = s * C01; // 1,2
               Q(2,iqy,iqx) = s * C11; // 2,2
            }

            const real_t w_detJ = w/detJ;
            Q(3,iqy,iqx) = c_2 * ((ND || RT) ? w_detJ : w*detJ);
         }
      }
   }
}

MFEM_HOST_DEVICE inline void Jacobian3D(
   const real_t x, const real_t y, const real_t z,
   const real_t vx[8], const real_t vy[8], const real_t vz[8],
   DeviceMatrix &J)
{
   // c: (1-x)(1-y)(1-z)v0[c] + x (1-y)(1-z)v1[c] + x y (1-z)v2[c] + (1-x) y (1-z)v3[c]
   //  + (1-x)(1-y) z   v4[c] + x (1-y) z   v5[c] + x y z    v6[c] + (1-x) y z    v7[c]
   J(0,0) = -(1-y)*(1-z)*vx[0]
            + (1-y)*(1-z)*vx[1] + y*(1-z)*vx[2] - y*(1-z)*vx[3]
            - (1-y)*z*vx[4] + (1-y)*z*vx[5] + y*z*vx[6] - y*z*vx[7];

   J(0,1) = -(1-x)*(1-z)*vx[0]
            - x*(1-z)*vx[1] + x*(1-z)*vx[2] + (1-x)*(1-z)*vx[3]
            - (1-x)*z*vx[4] - x*z*vx[5] + x*z*vx[6] + (1-x)*z*vx[7];

   J(0,2) = -(1-x)*(1-y)*vx[0] - x*(1-y)*vx[1]
            - x*y*vx[2] - (1-x)*y*vx[3] + (1-x)*(1-y)*vx[4]
            + x*(1-y)*vx[5] + x*y*vx[6] + (1-x)*y*vx[7];

   J(1,0) = -(1-y)*(1-z)*vy[0] + (1-y)*(1-z)*vy[1]
            + y*(1-z)*vy[2] - y*(1-z)*vy[3] - (1-y)*z*vy[4]
            + (1-y)*z*vy[5] + y*z*vy[6] - y*z*vy[7];

   J(1,1) = -(1-x)*(1-z)*vy[0] - x*(1-z)*vy[1]
            + x*(1-z)*vy[2] + (1-x)*(1-z)*vy[3]- (1-x)*z*vy[4] -
            x*z*vy[5] + x*z*vy[6] + (1-x)*z*vy[7];

   J(1,2) = -(1-x)*(1-y)*vy[0] - x*(1-y)*vy[1]
            - x*y*vy[2] - (1-x)*y*vy[3] + (1-x)*(1-y)*vy[4]
            + x*(1-y)*vy[5] + x*y*vy[6] + (1-x)*y*vy[7];

   J(2,0) = -(1-y)*(1-z)*vz[0] + (1-y)*(1-z)*vz[1]
            + y*(1-z)*vz[2] - y*(1-z)*vz[3]- (1-y)*z*vz[4] +
            (1-y)*z*vz[5] + y*z*vz[6] - y*z*vz[7];

   J(2,1) = -(1-x)*(1-z)*vz[0] - x*(1-z)*vz[1]
            + x*(1-z)*vz[2] + (1-x)*(1-z)*vz[3] - (1-x)*z*vz[4]
            - x*z*vz[5] + x*z*vz[6] + (1-x)*z*vz[7];

   J(2,2) = -(1-x)*(1-y)*vz[0] - x*(1-y)*vz[1]
            - x*y*vz[2] - (1-x)*y*vz[3] + (1-x)*(1-y)*vz[4]
            + x*(1-y)*vz[5] + x*y*vz[6] + (1-x)*y*vz[7];
}

MFEM_HOST_DEVICE inline void Adjugate3D(const DeviceMatrix &J, DeviceMatrix &A)
{
   A(0,0) = (J(1,1) * J(2,2)) - (J(1,2) * J(2,1));
   A(0,1) = (J(2,1) * J(0,2)) - (J(0,1) * J(2,2));
   A(0,2) = (J(0,1) * J(1,2)) - (J(1,1) * J(0,2));
   A(1,0) = (J(2,0) * J(1,2)) - (J(1,0) * J(2,2));
   A(1,1) = (J(0,0) * J(2,2)) - (J(0,2) * J(2,0));
   A(1,2) = (J(1,0) * J(0,2)) - (J(0,0) * J(1,2));
   A(2,0) = (J(1,0) * J(2,1)) - (J(2,0) * J(1,1));
   A(2,1) = (J(2,0) * J(0,1)) - (J(0,0) * J(2,1));
   A(2,2) = (J(0,0) * J(1,1)) - (J(0,1) * J(1,0));
}

MFEM_HOST_DEVICE inline void Get3DMatrixCoeff(
   ConstDeviceTensor<5> coeff, bool is_const,
   int i, int j, int k, int e, real_t *vals)
{
   if (is_const) { i = j = k = e = 0; }
   const int vdim = coeff.GetShape()[0];
   if (vdim == 1)
   {
      vals[0] = vals[4] = vals[8] = coeff(0, i, j, k, e);
      vals[1] = vals[2] = vals[3] = vals[5] = vals[6] = vals[7] = 0.0;
   }
   else if (vdim == 3)
   {
      vals[0] = coeff(0, i, j, k, e);
      vals[4] = coeff(1, i, j, k, e);
      vals[8] = coeff(2, i, j, k, e);
      vals[1] = vals[2] = vals[3] = vals[5] = vals[6] = vals[7] = 0.0;
   }
   else
   {
      for (int l = 0; l < vdim; ++l) { vals[l] = coeff(l, i, j, k, e); }
   }
};

// Assuming B is symmetric, compute the triple product AtBA in packed format
MFEM_HOST_DEVICE inline void FillAtBA(
   const DeviceMatrix &A, const DeviceMatrix &B, real_t *C, real_t alpha=1.0)
{
   const real_t e00 = B(0,0);
   const real_t e01 = B(0,1);
   const real_t e11 = B(1,1);
   const real_t e02 = B(0,2);
   const real_t e12 = B(1,2);
   const real_t e22 = B(2,2);

   const real_t u00 = A(0,0)*e00 + A(0,1)*e01 + A(0,2)*e02;
   const real_t u10 = A(0,0)*e01 + A(0,1)*e11 + A(0,2)*e12;
   const real_t u20 = A(0,0)*e02 + A(0,1)*e12 + A(0,2)*e22;

   const real_t u01 = A(1,0)*e00 + A(1,1)*e01 + A(1,2)*e02;
   const real_t u11 = A(1,0)*e01 + A(1,1)*e11 + A(1,2)*e12;
   const real_t u21 = A(1,0)*e02 + A(1,1)*e12 + A(1,2)*e22;

   const real_t u02 = A(2,0)*e00 + A(2,1)*e01 + A(2,2)*e02;
   const real_t u12 = A(2,0)*e01 + A(2,1)*e11 + A(2,2)*e12;
   const real_t u22 = A(2,0)*e02 + A(2,1)*e12 + A(2,2)*e22;

   C[0] = alpha*(A(0,0)*u00 + A(0,1)*u10 + A(0,2)*u20); // 1,1
   C[1] = alpha*(A(0,0)*u01 + A(0,1)*u11 + A(0,2)*u21); // 2,1
   C[2] = alpha*(A(0,0)*u02 + A(0,1)*u12 + A(0,2)*u22); // 3,1
   C[3] = alpha*(A(1,0)*u01 + A(1,1)*u11 + A(1,2)*u21); // 2,2
   C[4] = alpha*(A(1,0)*u02 + A(1,1)*u12 + A(1,2)*u22); // 3,2
   C[5] = alpha*(A(2,0)*u02 + A(2,1)*u12 + A(2,2)*u22); // 3,3
}

MFEM_HOST_DEVICE inline void Transpose3D(DeviceMatrix &A)
{
   real_t t;
   t = A(0,1); A(0,1) = A(1,0); A(1,0) = t;
   t = A(0,2); A(0,2) = A(2,0); A(2,0) = t;
   t = A(1,2); A(1,2) = A(2,1); A(2,1) = t;
}

}

#endif
