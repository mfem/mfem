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

#ifndef MFEM_BILININTEG_VECTORFEMASS_PA_SIMPLICES_HPP
#define MFEM_BILININTEG_VECTORFEMASS_PA_SIMPLICES_HPP

namespace mfem
{

namespace internal
{

inline void PANDPosHcurlMassApply3D(
   const int NE, const bool symmetric,
   const RaggedDofToQuad &maps, const Vector &pa_data,
   const Vector &x, Vector &y, const int d1d, const int q1d)
{
   const int D1D = d1d;
   const int Q1D = q1d;
   constexpr static int VDIM = 3;

   const int BASIS_DIM3D = (D1D-1) * (D1D+1) * (D1D+2) / 2;
   const int BASIS_DIM2D_SCALAR = D1D * (D1D+1) / 2;
   const int BASIS_DIM3D_SCALAR = D1D * (D1D+1) * (D1D+2) / 6;

   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D_SIMPLEX, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D_SIMPLEX, "");

   const auto lex_map = maps.lex_map.Read();
   const auto forward_map2d = maps.forward_map2d_mass.Read();
   const auto inverse_map2d = maps.inverse_map2d_mass.Read();
   const auto inverse_map3d = maps.inverse_map3d_mass.Read();
   const auto scalar_offsets = maps.scalar_offsets.Read();
   const auto scalar_multiindices = maps.scalar_multiindices.Read();
   const auto scalar_coeffs = maps.scalar_coeffs.Read();
   const auto Ba1 = ConstDeviceMatrix(maps.Ba1.Read(), D1D, Q1D);
   const auto Ba2 = ConstDeviceMatrix(maps.Ba2.Read(), BASIS_DIM2D_SCALAR, Q1D);
   const auto Ba1t = ConstDeviceMatrix(maps.Ba1t.Read(), Q1D, D1D);
   const auto Ba2t = ConstDeviceMatrix(maps.Ba2t.Read(), Q1D, BASIS_DIM2D_SCALAR);
   const auto Ba3t = ConstDeviceMatrix(maps.Ba3t.Read(), Q1D, BASIS_DIM3D_SCALAR);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D,
                          symmetric ? 6 : 9, NE);
   const auto X = Reshape(x.Read(), BASIS_DIM3D, NE);
   auto Y = Reshape(y.ReadWrite(), BASIS_DIM3D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = d1d;
      const int Q1D = q1d;

      constexpr int max_D1D = DofQuadLimits::MAX_D1D_SIMPLEX;
      constexpr int max_Q1D = DofQuadLimits::MAX_Q1D_SIMPLEX;

      constexpr int basis_dim2d = max_D1D * (max_D1D+1) / 2;
      constexpr int basis_dim3d =
         max_D1D * (max_D1D+1) * (max_D1D+2) / 6;
      real_t C1[3 * basis_dim2d * max_Q1D];
      real_t C2[3 * max_D1D * max_Q1D * max_Q1D];
      real_t C3[3 * max_Q1D * max_Q1D * max_Q1D];
      real_t F1[3 * max_D1D * max_Q1D * max_Q1D];
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
            for (int a1 = 0; a1 < D1D; a1++)
            {
               const int q = 3*(a1 + D1D*(i2 + Q1D*i3));
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
               int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
               Xscalar[d + 3*idx] += X(i, e) * scalar_coeffs[d + 3*k];
            }
         }
      }

      const int p = D1D - 1;
      const real_t inv_degree = 1.0 / p;
      const real_t c01 = X(0, e);
      const real_t c02 = X(p, e);
      const real_t c03 = X(2*p, e);
      const real_t c12 = X(3*p, e);
      const real_t c13 = X(4*p, e);
      const real_t c23 = X(5*p, e);
      const real_t c02_m_c12 = c02 - c12;
      const real_t c03_m_c13 = c03 - c13;
      const real_t c01_p_c12 = c01 + c12;
      const real_t c03_m_c23 = c03 - c23;
      const real_t c01_p_c13 = c01 + c13;
      const real_t c02_p_c23 = c02 + c23;

      // C1 contains the Bernstein polynomial on a triangle evaluated at the quadrature
      // point in the first spatial dimension
      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + D1D*a1];

         // gather coefficients of the scalar Bernstein from the Type 2-4 basis functions
         const int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
         real_t u = Xscalar[3*idx];
         real_t v = Xscalar[1 + 3*idx];
         real_t w = Xscalar[2 + 3*idx];

         // Type 1 basis can be degree elevated and their coefficients added to (u,v,w). E.g.
         // since \lambda_i = \sum_{|\alpha|=p+1} (\alpha_i/p) B^{p+1}\alpha, we have
         //
         //    w_ij = \sum_{|\alpha|=p+1} c_ij * (\alpha_i/p * \nabla\lambda_j - \alpha_j/p * \nabla\lambda_i).
         //
         // The contribution of w_ij to (u,v,w) is thus c_ij * (\alpha_i/p * \nabla\lambda_j -
         // \alpha_j/p * \nabla\lambda_i).
         const int alpha0 = p - a1 - a2 - a3;
         const real_t weight0 = alpha0 * inv_degree;
         const real_t weight1 = a1 * inv_degree;
         const real_t weight2 = a2 * inv_degree;
         const real_t weight3 = a3 * inv_degree;
         u += c01*(weight0 + weight1) + c02_m_c12*weight2
              + c03_m_c13*weight3;
         v += c01_p_c12*weight1 + c02*(weight0 + weight2)
              + c03_m_c23*weight3;
         w += c01_p_c13*weight1 + c02_p_c23*weight2
              + c03*(weight0 + weight3);

         for (int i3 = 0; i3 < Q1D; i3++)
         {
            const int a1a2i3 = 3*(i3 + Q1D*a_2d);
            const real_t Bai = Ba3t(i3,a);

            C1[a1a2i3] += u * Bai;
            C1[1 + a1a2i3] += v * Bai;
            C1[2 + a1a2i3] += w * Bai;
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
               const real_t Bai = Ba2t(i2,a);

               const int a1i2i3 = 3*(i2 + Q1D*(i3 + Q1D*a1));
               C2[a1i2i3] += C1x * Bai;
               C2[1 + a1i2i3] += C1y * Bai;
               C2[2 + a1i2i3] += C1z * Bai;
            }
         }
      }

      // C3 now contains the Bernstein polynomial uh(x) = \sum_i c_i * Phi_i(x)
      // evaluated at the Stroud quadrature nodes.
      for (int a1 = 0; a1 < D1D; a1++)
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
                  const real_t Bai = Ba1t(i1,a1);
                  const int i1i2i3 = 3*(i1 + Q1D*(i2 + Q1D*i3));
                  C3[i1i2i3] += C2x * Bai;
                  C3[1 + i1i2i3] += C2y * Bai;
                  C3[2 + i1i2i3] += C2z * Bai;
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
               for (int a1 = 0; a1 < D1D; a1++)
               {
                  const real_t Bai = Ba1(a1,i1);
                  const int a1i2i3 = 3*(a1 + D1D*(i2 + Q1D*i3));
                  F1[a1i2i3] += C3[i1i2i3] * Bai;
                  F1[1 + a1i2i3] += C3[1 + i1i2i3] * Bai;
                  F1[2 + a1i2i3] += C3[2 + i1i2i3] * Bai;
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
               const real_t Bai = Ba2(a,i2);

               const int a1a2i3 = 3*(a + BASIS_DIM2D_SCALAR*i3);
               const int a1i2i3 = 3*(a1 + D1D*(i2 + Q1D*i3));
               F2[a1a2i3] += F1[a1i2i3] * Bai;
               F2[1 + a1a2i3] += F1[1 + a1i2i3] * Bai;
               F2[2 + a1a2i3] += F1[2 + a1i2i3] * Bai;
            }
         }
      }

      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         const int a1 = inverse_map3d[3*a];
         const int a2 = inverse_map3d[1 + 3*a];
         const int a3 = inverse_map3d[2 + 3*a];
         const int a_2d = forward_map2d[a2 + D1D*a1];

         real_t u = 0.0, v = 0.0, w = 0.0;
         // const int a1a2a3 = 3*(a3 + a1a2);
         for (int i3 = 0; i3 < Q1D; i3++)
         {
            // const int idx = i3 + Q1D*a;
            const real_t Bai = Ba3t(i3,a);
            const int a1a2i3 = 3*(a_2d + BASIS_DIM2D_SCALAR*i3);
            u += F2[a1a2i3] * Bai;
            v += F2[1 + a1a2i3] * Bai;
            w += F2[2 + a1a2i3] * Bai;
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
            const int idx = lex_map[a3 + D1D*(a2 + D1D*a1)];
            for (int d = 0; d < VDIM; d++)
            {
               Y(i,e) += F3[d + 3*idx] * scalar_coeffs[d + 3*k];
            }
         }
      }

      // Type 1 basis functions are handled separately. The Whitney edge functions
      // are linear combinations of the barycentric coordinates, which can all be
      // expressed in terms of the scalar Bernstein basis functions of degree p+1:
      //
      //    \lambda_i = \sum_{|\alpha| = p+1} \alpha_i / p * B^p_{\alpha}.
      //
      //  Thus, we first compute the moments of \lambda_i which are just linear
      // combinations of the scalar Bernstein moments in F3.
      real_t F3_edge[12];
      for (int i = 0; i < 12; i++)
      {
         F3_edge[i] = 0.0;
      }
      for (int a = 0; a < BASIS_DIM3D_SCALAR; a++)
      {
         const int alpha1 = inverse_map3d[3*a];
         const int alpha2 = inverse_map3d[1 + 3*a];
         const int alpha3 = inverse_map3d[2 + 3*a];
         const int alpha0 = p - alpha1 - alpha2 - alpha3;
         const real_t weight0 = alpha0 * inv_degree;
         const real_t weight1 = alpha1 * inv_degree;
         const real_t weight2 = alpha2 * inv_degree;
         const real_t weight3 = alpha3 * inv_degree;
         for (int d = 0; d < 3; d++)
         {
            const real_t moment = F3[d + 3*a];
            F3_edge[d] += weight0 * moment;
            F3_edge[d + 3] += weight1 * moment;
            F3_edge[d + 6] += weight2 * moment;
            F3_edge[d + 9] += weight3 * moment;
         }
      }

      // edge (0,1) contributions look like
      //    Moment(\lambda_0) * grad(\lambda_1) - Moment(\lambda_1) * grad(\lambda_0) =
      //    Moment(\lambda_0)_x + (Moment(\lambda_1)_x + Moment(\lambda_1)_y + Moment(\lambda_1)_z)
      Y(0,e) += F3_edge[0] + F3_edge[3] + F3_edge[4] + F3_edge[5];

      // edge (0,2) contributions look like
      //    Moment(\lambda_0)_y + (Moment(\lambda_2)_x + Moment(\lambda_2)_y + Moment(\lambda_2)_z)
      Y(p,e) += F3_edge[1] + F3_edge[6] + F3_edge[7] + F3_edge[8];

      // edge (0,3) contributions look like
      //    Moment(\lambda_0)_z + (Moment(\lambda_3)_x + Moment(\lambda_3)_y + Moment(\lambda_3)_z)
      Y(2*p,e) += F3_edge[2] + F3_edge[9] + F3_edge[10] + F3_edge[11];

      // edge (1,2) contributions look like
      //    Moment(\lambda_1)_y - Moment(\lambda_2)_x
      Y(3*p,e) += F3_edge[4] - F3_edge[6];

      // edge (1,3) contributions look like
      //    Moment(\lambda_1)_z - Moment(\lambda_3)_x
      Y(4*p,e) += F3_edge[5] - F3_edge[9];

      // edge (2,3) contributions look like
      //    Moment(\lambda_2)_z - Moment(\lambda_3)_y
      Y(5*p,e) += F3_edge[8] - F3_edge[10];
   });
}

} // namespace internal

/// \cond DO_NOT_DOCUMENT

template <int DIM, int T_D1D, int T_Q1D>
VectorFEMassIntegrator::ApplySimplexKernelType
VectorFEMassIntegrator::ApplySimplexPAKernels::Kernel()
{
   static_assert(DIM == 3, "NDPos simplex mass kernels require dimension 3");
   return internal::PANDPosHcurlMassApply3D;
}

inline VectorFEMassIntegrator::ApplySimplexKernelType
VectorFEMassIntegrator::ApplySimplexPAKernels::Fallback(int dim, int, int)
{
   MFEM_VERIFY(dim == 3, "NDPos simplex mass kernels require dimension 3");
   return internal::PANDPosHcurlMassApply3D;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem

#endif // MFEM_BILININTEG_VECTORFEMASS_PA_SIMPLICES_HPP
