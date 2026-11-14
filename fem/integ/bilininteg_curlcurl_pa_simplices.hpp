// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG_CURLCURL_PA_SIMPLICES_HPP
#define MFEM_BILININTEG_CURLCURL_PA_SIMPLICES_HPP

namespace mfem
{

namespace internal
{

template<int T_D1D = 0, int T_Q1D = 0>
inline void PANDPosCurlCurlApply3D(
   const int NE, const bool symmetric,
   const RaggedDofToQuad &maps, const Vector &pa_data,
   const Vector &x, Vector &y, const int d1d, const int q1d)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr static int VDIM = 3;

   const int BASIS_DIM3D = (D1D-1) * (D1D+1) * (D1D+2) / 2;
   const int BASIS_DIM2D_SCALAR = (D1D-1) * D1D / 2;
   const int BASIS_DIM3D_SCALAR = (D1D-1) * (D1D) * (D1D+1) / 6;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto forward_map2d = maps.forward_map2d_diff.Read();
   const auto forward_map3d = maps.forward_map3d_diff.Read();
   const auto inverse_map2d = maps.inverse_map2d_diff.Read();
   const auto inverse_map3d = maps.inverse_map3d_diff.Read();
   const auto scalar_offsets = maps.scalar_offsets_curl.Read();
   const auto scalar_multiindices = maps.scalar_multiindices_curl.Read();
   const auto scalar_coeffs = maps.scalar_coeffs_curl.Read();
   const auto Ga1 = ConstDeviceMatrix(maps.Ga1.Read(), D1D-1, Q1D);
   const auto Ga2 = ConstDeviceMatrix(maps.Ga2.Read(), BASIS_DIM2D_SCALAR, Q1D);
   const auto Ga1t = ConstDeviceMatrix(maps.Ga1t.Read(), Q1D, D1D-1);
   const auto Ga2t = ConstDeviceMatrix(maps.Ga2t.Read(), Q1D, BASIS_DIM2D_SCALAR);
   const auto Ga3t = ConstDeviceMatrix(maps.Ga3t.Read(), Q1D, BASIS_DIM3D_SCALAR);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D,
                          symmetric ? 6 : 9, NE);
   const auto X = Reshape(x.Read(), BASIS_DIM3D, NE);
   auto Y = Reshape(y.ReadWrite(), BASIS_DIM3D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int max_D1D =
         T_D1D ? T_D1D : DofQuadLimits::MAX_D1D_SIMPLEX;
      constexpr int max_Q1D =
         T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D_SIMPLEX;

      constexpr int basis_dim2d = (max_D1D-1) * max_D1D / 2;
      constexpr int basis_dim3d =
         (max_D1D-1) * max_D1D * (max_D1D+1) / 6;
      real_t C1[3 * basis_dim2d * max_Q1D];
      real_t C2[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t C3[3 * max_Q1D * max_Q1D * max_Q1D];
      real_t F1[3 * (max_D1D-1) * max_Q1D * max_Q1D];
      real_t F2[3 * basis_dim2d * max_Q1D];
      real_t F3[3 * basis_dim3d];
      real_t Xscalar[3 * basis_dim3d];

      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const int q = 3*(i1 + Q1D*(i2 + Q1D*i3));
               C3[q] = 0.0;
               C3[1+q] = 0.0;
               C3[2+q] = 0.0;
            }
            for (int a1 = 0; a1 < D1D-1; a1++)
            {
               const int q = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               C2[q] = 0.0;
               C2[1+q] = 0.0;
               C2[2+q] = 0.0;
               F1[q] = 0.0;
               F1[1+q] = 0.0;
               F1[2+q] = 0.0;
            }
         }

         for (int a = 0; a < BASIS_DIM2D_SCALAR; a++)
         {
            const int q = 3*(a + BASIS_DIM2D_SCALAR*i3);
            C1[q] = 0.0;
            C1[1+q] = 0.0;
            C1[2+q] = 0.0;
            F2[q] = 0.0;
            F2[1+q] = 0.0;
            F2[2+q] = 0.0;
         }
      }

      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         F3[3*a] = 0.0;
         F3[1 + 3*a] = 0.0;
         F3[2 + 3*a] = 0.0;
         Xscalar[3*a] = 0.0;
         Xscalar[1 + 3*a] = 0.0;
         Xscalar[2 + 3*a] = 0.0;
      }

      // pre-processing step: map Bernstein Nedelec coefficients in X to scalar
      // Bernstein coefficients in Xscalar
      for (int i = 0; i < BASIS_DIM3D; i++)
      {
         for (int k = scalar_offsets[i]; k < scalar_offsets[i+1]; k++)
         {
            for (int d = 0; d < VDIM; d++)
            {
               const int a1 = scalar_multiindices[3*k];
               const int a2 = scalar_multiindices[1 + 3*k];
               const int a3 = scalar_multiindices[2 + 3*k];
               const int idx =
                  forward_map3d[a3 + (D1D-1)*(a2 + (D1D-1)*a1)];
               Xscalar[d + 3*idx] += X(i, e) * scalar_coeffs[d + 3*k];
            }
         }
      }

      const int p = D1D - 1;
      const real_t c01 = X(0, e);
      const real_t c02 = X(p, e);
      const real_t c03 = X(2*p, e);
      const real_t c12 = X(3*p, e);
      const real_t c13 = X(4*p, e);
      const real_t c23 = X(5*p, e);

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];

         // gather coefficients of the scalar Bernstein from the Type 2-4 basis functions
         real_t u = Xscalar[3*a];
         real_t v = Xscalar[1 + 3*a];
         real_t w = Xscalar[2 + 3*a];

         // Type 1 basis curls are just constant vectors and can be degree
         // elevated in terms of the degree p-1 scalar Bernstein basis:
         //
         //    c_ij * curl w_ij = 2*c_ij*t_ij
         //       = 2*t_ij * \sum_{|\alpha|=p-1} B^{p-1}_{\alpha}.
         //
         // The contribution of curl w_ij to (u,v,w) is thus 2*c_ij*t_ij.
         u += 2.0 * (c02 - c03 + c23);
         v += 2.0 * (-c01 + c03 - c13);
         w += 2.0 * (c01 - c02 + c12);

         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a_2d);
            const real_t Gai = Ga3t(i3,a);

            C1[a1a2i3] += u * Gai;
            C1[1 + a1a2i3] += v * Gai;
            C1[2 + a1a2i3] += w * Gai;
         }
      }

      // C2 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the second spatial dimension
      for (int a = 0; a < BASIS_DIM2D_SCALAR; a++)
      {
         const int a1 = inverse_map2d[2*a];
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a);
            const real_t C1x = C1[a1a2i3];
            const real_t C1y = C1[1 + a1a2i3];
            const real_t C1z = C1[2 + a1a2i3];
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const real_t Gai = Ga2t(i2,a);

               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               C2[a1i2i3] += C1x * Gai;
               C2[1 + a1i2i3] += C1y * Gai;
               C2[2 + a1i2i3] += C1z * Gai;
            }
         }
      }

      // C3 now contains the Bernstein polynomial uh(x) = \sum_i c_i * Phi_i(x)
      // evaluated at the Stroud quadrature nodes.
      for (int a1 = 0; a1 < D1D-1; a1++)
      {
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            for (int i2 = 0; i2 < Q1D; i2++)
            {
               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               const real_t C2x = C2[a1i2i3];
               const real_t C2y = C2[1 + a1i2i3];
               const real_t C2z = C2[2 + a1i2i3];
               for (int i1 = 0; i1 < Q1D; i1++)
               {
                  const real_t Gai = Ga1t(i1,a1);
                  const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
                  C3[i1i2i3] += C2x * Gai;
                  C3[1 + i1i2i3] += C2y * Gai;
                  C3[2 + i1i2i3] += C2z * Gai;
               }
            }
         }
      }

      // Update C3 with the action of the operator D at each quadrature point
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const real_t O11 = D(i1,i2,i3,0,e);
               const real_t O12 = D(i1,i2,i3,1,e);
               const real_t O13 = D(i1,i2,i3,2,e);
               const real_t O21 = symmetric ? O12 : D(i1,i2,i3,3,e);
               const real_t O22 = symmetric ? D(i1,i2,i3,3,e) : D(i1,i2,i3,4,e);
               const real_t O23 = symmetric ? D(i1,i2,i3,4,e) : D(i1,i2,i3,5,e);
               const real_t O31 = symmetric ? O13 : D(i1,i2,i3,6,e);
               const real_t O32 = symmetric ? O23 : D(i1,i2,i3,7,e);
               const real_t O33 = symmetric ? D(i1,i2,i3,5,e) : D(i1,i2,i3,8,e);

               const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
               real_t gX = C3[i1i2i3];
               real_t gY = C3[1 + i1i2i3];
               real_t gZ = C3[2 + i1i2i3];

               C3[i1i2i3] = O11 * gX + O12 * gY + O13 * gZ;
               C3[1 + i1i2i3] = O21 * gX + O22 * gY + O23 * gZ;
               C3[2 + i1i2i3] = O31 * gX + O32 * gY + O33 * gZ;
            }
         }
      }

      // F1 computes the Bernstein moment over the first ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int i1 = 0; i1 < Q1D; i1++)
            {
               const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
               for (int a1 = 0; a1 < D1D-1; a1++)
               {
                  const real_t Gai = Ga1(a1,i1);
                  const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
                  F1[a1i2i3] += C3[i1i2i3] * Gai;
                  F1[1 + a1i2i3] += C3[1 + i1i2i3] * Gai;
                  F1[2 + a1i2i3] += C3[2 + i1i2i3] * Gai;
               }
            }
         }
      }

      // F2 computes the Bernstein moment over the second ragged tensor dimension.
      for (int i3 = 0; i3 < Q1D; i3++)
      {
         for (int i2 = 0; i2 < Q1D; i2++)
         {
            for (int a = 0; a < BASIS_DIM2D_SCALAR; a++)
            {
               const int a1 = inverse_map2d[2*a];
               const real_t Gai = Ga2(a,i2);

               const int a1a2i3 = 3*(a + BASIS_DIM2D_SCALAR*i3);
               const int a1i2i3 = 3*(a1 + (D1D-1)*(i2 + Q1D*i3));
               F2[a1a2i3] += F1[a1i2i3] * Gai;
               F2[1 + a1a2i3] += F1[1 + a1i2i3] * Gai;
               F2[2 + a1a2i3] += F1[2 + a1i2i3] * Gai;
            }
         }
      }

      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + (D1D-1)*a1];

         real_t u = 0.0, v = 0.0, w = 0.0;
         // const int a1a2a3 = 3*(a3 + a1a2);
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            // const int idx = i3 + Q1D*a;
            const real_t Gai = Ga3t(i3,a);
            const int a1a2i3 = 3*(a_2d + BASIS_DIM2D_SCALAR*i3);
            u += F2[a1a2i3] * Gai;
            v += F2[1 + a1a2i3] * Gai;
            w += F2[2 + a1a2i3] * Gai;
         }

         F3[3*a] = u;
         F3[3*a + 1] = v;
         F3[3*a + 2] = w;
      }

      // Post-processing step: map contributions of scalar Bernstein basis
      // back to the Bernstein Nedelec basis for Type 2, 3, and 4 basis
      // functions.
      for (int i = 0; i < BASIS_DIM3D; i++)
      {
         for (int k = scalar_offsets[i]; k < scalar_offsets[i+1]; k++)
         {
            const int a1 = scalar_multiindices[3*k];
            const int a2 = scalar_multiindices[1 + 3*k];
            const int a3 = scalar_multiindices[2 + 3*k];
            const int idx =
               forward_map3d[a3 + (D1D-1)*(a2 + (D1D-1)*a1)];
            for (int d = 0; d < VDIM; d++)
            {
               Y(i,e) += F3[d + 3*idx] * scalar_coeffs[d + 3*k];
            }
         }
      }

      // Type 1 basis functions are handled separately. Since curls of the
      // Whitney edge functions are constant vectors, their moments can be
      // computed by summing over all of the order p-1 scalar moments.
      real_t F3_edge[3];
      for (int i = 0; i < 3; i++)
      {
         F3_edge[i] = 0.0;
      }
      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         for (int d = 0; d < 3; d++)
         {
            const real_t moment = F3[d + 3*a];
            F3_edge[d] += moment;
         }
      }

      // edge (i,j) contributions look like
      //    y_ij = Moment(D*u * 2 * t_ij)
      //         = 2*t_ij * \sum_{|\alpha|=p-1} Moment_{\alpha}(D*u).
      // edge (0,1) contributions
      Y(0,e) += 2.0 * (-F3_edge[1] + F3_edge[2]);

      // edge (0,2) contributions
      Y(p,e) += 2.0 * (F3_edge[0] - F3_edge[2]);

      // edge (0,3) contributions
      Y(2*p,e) += 2.0 * (-F3_edge[0] + F3_edge[1]);

      // edge (1,2) contributions
      Y(3*p,e) += 2.0 * F3_edge[2];

      // edge (1,3) contributions
      Y(4*p,e) += -2.0 * F3_edge[1];

      // edge (2,3) contributions
      Y(5*p,e) += 2.0 * F3_edge[0];
   });
}


} // namespace internal

/// \cond DO_NOT_DOCUMENT

template <int DIM, int T_D1D, int T_Q1D>
CurlCurlIntegrator::ApplySimplexKernelType
CurlCurlIntegrator::ApplySimplexPAKernels::Kernel()
{
   static_assert(DIM == 3, "NDPos simplex curl-curl kernels require dimension 3");
   return internal::PANDPosCurlCurlApply3D<T_D1D,T_Q1D>;
}

inline CurlCurlIntegrator::ApplySimplexKernelType
CurlCurlIntegrator::ApplySimplexPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 3,
               "NDPos simplex curl-curl kernels require dimension 3");
   return internal::PANDPosCurlCurlApply3D;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

#endif // MFEM_BILININTEG_CURLCURL_PA_SIMPLICES_HPP
