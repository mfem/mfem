// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
//#include "../mesh/mesh.hpp"
//#include "../linalg/vector.hpp"
//#include "../general/array.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
//#include "../general/debug.hpp"
//#include "../general/nvvp.hpp"
#define NvtxPush(...)
#define NvtxPop(...)
#define FOR_LOOP_UNROLL(N) MFEM_UNROLL(N)

namespace mfem
{

template <int order>
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              Array<int> &dof_glob2loc_,
                              Array<int> &dof_glob2loc_offsets_,
                              Array<int> &el_dof_lex_,
                              Vector &Q_,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat)
{
   NvtxPush(Assemble3DBatchedLOR_GPU_Kernels, LightCoral);
   const int nel_ho = mesh_ho.GetNE();
   const int ndof = fes_ho.GetVSize();

   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = order + 1;
   static constexpr int ndof_per_el = nd1d*nd1d*nd1d;
   static constexpr int nnz_per_row = 27;
   static constexpr int nnz_per_el = nnz_per_row * ndof_per_el;
   static constexpr int sz_grad_A = 3*3*2*2*2*2;
   static constexpr int sz_grad_B = sz_grad_A*2;
   static constexpr int sz_local_mat = 8*8;

   NvtxPush(DOFCopy, Coral);
   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto dof_glob2loc = dof_glob2loc_.Read();
   const auto K = dof_glob2loc_offsets_.Read();
   NvtxPop(DOFCopy);

   NvtxPush(IJACopy, HotPink);
   const auto I = A_mat.ReadI();
   const auto J = A_mat.ReadJ();
   auto A = A_mat.ReadWriteData();
   NvtxPop(IJACopy);

   const auto Q = Reshape(Q_.Write(), ddm2, 2,2,2, order,order,order, nel_ho);
   const auto X = mesh_lor.GetNodes()->Read();

   NvtxPush(Assembly, SeaGreen);

   MFEM_FORALL_3D(iel_ho, nel_ho, order, order, order,
   {
      // nnz_per_el = nnz_per_row(=27) * (order+1) * (order+1) * (order+1);
      MFEM_SHARED double V_[nnz_per_el];
      DeviceTensor<4> V(V_, nnz_per_row, nd1d, nd1d, nd1d);

      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      MFEM_FOREACH_THREAD(iz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(iy,y,nd1d)
         {
            MFEM_FOREACH_THREAD(ix,x,nd1d)
            {
               FOR_LOOP_UNROLL(27)
               for (int j=0; j<nnz_per_row; ++j)
               {
                  V(j,ix,iy,iz) = 0.0;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(kz,z,order)
      {
         MFEM_FOREACH_THREAD(ky,y,order)
         {
            MFEM_FOREACH_THREAD(kx,x,order)
            {
               const int v_i0 = kx + nd1d*(ky + nd1d*kz);
               const int v_i1 = kx + 1 + nd1d*(ky + nd1d*kz);
               const int v_i2 = kx + 1 + nd1d*(ky + 1 + nd1d*kz);
               const int v_i3 = kx + nd1d*(ky + 1 + nd1d*kz);

               const int v_i4 = kx + nd1d*(ky + nd1d*(kz + 1));
               const int v_i5 = kx + 1 + nd1d*(ky + nd1d*(kz + 1));
               const int v_i6 = kx + 1 + nd1d*(ky + 1 + nd1d*(kz + 1));
               const int v_i7 = kx + nd1d*(ky + 1 + nd1d*(kz + 1));

               const int e0 = el_dof_lex(v_i0, iel_ho);
               const int e1 = el_dof_lex(v_i1, iel_ho);
               const int e2 = el_dof_lex(v_i2, iel_ho);
               const int e3 = el_dof_lex(v_i3, iel_ho);
               const int e4 = el_dof_lex(v_i4, iel_ho);
               const int e5 = el_dof_lex(v_i5, iel_ho);
               const int e6 = el_dof_lex(v_i6, iel_ho);
               const int e7 = el_dof_lex(v_i7, iel_ho);

               const double v_0_x = X[3*e0 + 0];
               const double v_0_y = X[3*e0 + 1];
               const double v_0_z = X[3*e0 + 2];

               const double v_1_x = X[3*e1 + 0];
               const double v_1_y = X[3*e1 + 1];
               const double v_1_z = X[3*e1 + 2];

               const double v_2_x = X[3*e2 + 0];
               const double v_2_y = X[3*e2 + 1];
               const double v_2_z = X[3*e2 + 2];

               const double v_3_x = X[3*e3 + 0];
               const double v_3_y = X[3*e3 + 1];
               const double v_3_z = X[3*e3 + 2];

               const double v_4_x = X[3*e4 + 0];
               const double v_4_y = X[3*e4 + 1];
               const double v_4_z = X[3*e4 + 2];

               const double v_5_x = X[3*e5 + 0];
               const double v_5_y = X[3*e5 + 1];
               const double v_5_z = X[3*e5 + 2];

               const double v_6_x = X[3*e6 + 0];
               const double v_6_y = X[3*e6 + 1];
               const double v_6_z = X[3*e6 + 2];

               const double v_7_x = X[3*e7 + 0];
               const double v_7_y = X[3*e7 + 1];
               const double v_7_z = X[3*e7 + 2];

               FOR_LOOP_UNROLL(2)
               for (int iqz=0; iqz<2; ++iqz)
               {
                  FOR_LOOP_UNROLL(2)
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     FOR_LOOP_UNROLL(2)
                     for (int iqx=0; iqx<2; ++iqx)
                     {

                        const double x = iqx;
                        const double y = iqy;
                        const double z = iqz;
                        const double w = 1.0/8.0;

                        // c: (1-x)(1-y)(1-z)v0[c] + x (1-y)(1-z)v1[c] + x y (1-z)v2[c] + (1-x) y (1-z)v3[c]
                        //  + (1-x)(1-y) z   v4[c] + x (1-y) z   v5[c] + x y z    v6[c] + (1-x) y z    v7[c]
                        const double J11 = -(1-y)*(1-z)*v_0_x
                                           + (1-y)*(1-z)*v_1_x + y*(1-z)*v_2_x - y*(1-z)*v_3_x
                                           - (1-y)*z*v_4_x + (1-y)*z*v_5_x + y*z*v_6_x - y*z*v_7_x;

                        const double J12 = -(1-x)*(1-z)*v_0_x
                                           - x*(1-z)*v_1_x + x*(1-z)*v_2_x + (1-x)*(1-z)*v_3_x
                                           - (1-x)*z*v_4_x - x*z*v_5_x + x*z*v_6_x + (1-x)*z*v_7_x;

                        const double J13 = -(1-x)*(1-y)*v_0_x - x*(1-y)*v_1_x
                                           - x*y*v_2_x - (1-x)*y*v_3_x + (1-x)*(1-y)*v_4_x
                                           + x*(1-y)*v_5_x + x*y*v_6_x + (1-x)*y*v_7_x;

                        const double J21 = -(1-y)*(1-z)*v_0_y + (1-y)*(1-z)*v_1_y
                                           + y*(1-z)*v_2_y - y*(1-z)*v_3_y - (1-y)*z*v_4_y
                                           + (1-y)*z*v_5_y + y*z*v_6_y - y*z*v_7_y;

                        const double J22 = -(1-x)*(1-z)*v_0_y - x*(1-z)*v_1_y
                                           + x*(1-z)*v_2_y + (1-x)*(1-z)*v_3_y- (1-x)*z*v_4_y -
                                           x*z*v_5_y + x*z*v_6_y + (1-x)*z*v_7_y;

                        const double J23 = -(1-x)*(1-y)*v_0_y - x*(1-y)*v_1_y
                                           - x*y*v_2_y - (1-x)*y*v_3_y + (1-x)*(1-y)*v_4_y
                                           + x*(1-y)*v_5_y + x*y*v_6_y + (1-x)*y*v_7_y;

                        const double J31 = -(1-y)*(1-z)*v_0_z + (1-y)*(1-z)*v_1_z
                                           + y*(1-z)*v_2_z - y*(1-z)*v_3_z- (1-y)*z*v_4_z +
                                           (1-y)*z*v_5_z + y*z*v_6_z - y*z*v_7_z;

                        const double J32 = -(1-x)*(1-z)*v_0_z - x*(1-z)*v_1_z
                                           + x*(1-z)*v_2_z + (1-x)*(1-z)*v_3_z - (1-x)*z*v_4_z
                                           - x*z*v_5_z + x*z*v_6_z + (1-x)*z*v_7_z;

                        const double J33 = -(1-x)*(1-y)*v_0_z - x*(1-y)*v_1_z
                                           - x*y*v_2_z - (1-x)*y*v_3_z + (1-x)*(1-y)*v_4_z
                                           + x*(1-y)*v_5_z + x*y*v_6_z + (1-x)*y*v_7_z;

                        const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                            J21 * (J12 * J33 - J32 * J13) +
                                            J31 * (J12 * J23 - J22 * J13);
                        const double w_detJ = w/detJ;

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

                        // Put these in the opposite order...
                        Q(0,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
                        Q(1,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
                        Q(2,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
                        Q(3,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
                        Q(4,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
                        Q(5,iqz,iqy,iqx,kz,ky,kx,iel_ho) =
                           w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Loop over sub-elements
      MFEM_FOREACH_THREAD(kz,z,order)
      {
         MFEM_FOREACH_THREAD(ky,y,order)
         {
            MFEM_FOREACH_THREAD(kx,x,order)
            {
               double grad_A_[sz_grad_A];
               double grad_B_[sz_grad_B];
               double local_mat_[sz_local_mat];
               DeviceTensor<2> local_mat(local_mat_, 8, 8);
               DeviceTensor<6> grad_A(grad_A_, 3, 3, 2, 2, 2, 2);
               DeviceTensor<7> grad_B(grad_B_, 3, 3, 2, 2, 2, 2, 2);

               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }
               // Intermediate quantities
               // (see e.g. Mora and Demkowicz for notation).
               for (int i=0; i<sz_grad_A; ++i) { grad_A[i] = 0.0; }
               for (int i=0; i<sz_grad_B; ++i) { grad_B[i] = 0.0; }

               FOR_LOOP_UNROLL(2)
               for (int iqx=0; iqx<2; ++iqx)
               {
                  FOR_LOOP_UNROLL(2)
                  for (int jz=0; jz<2; ++jz)
                  {
                     // Note loop starts at iz=jz here, taking advantage of
                     // symmetries.
                     FOR_LOOP_UNROLL(2)
                     for (int iz=jz; iz<2; ++iz)
                     {
                        MFEM_UNROLL(2)
                        for (int iqy=0; iqy<2; ++iqy)
                        {
                           MFEM_UNROLL(2)
                           for (int iqz=0; iqz<2; ++iqz)
                           {
                              const double biz = (iz == iqz) ? 1.0 : 0.0;
                              const double giz = (iz == 0) ? -1.0 : 1.0;

                              const double bjz = (jz == iqz) ? 1.0 : 0.0;
                              const double gjz = (jz == 0) ? -1.0 : 1.0;

                              const double J11 = Q(0,iqz,iqy,iqx,kz,ky,kx,iel_ho);
                              const double J21 = Q(1,iqz,iqy,iqx,kz,ky,kx,iel_ho);
                              const double J31 = Q(2,iqz,iqy,iqx,kz,ky,kx,iel_ho);
                              const double J12 = J21;
                              const double J22 = Q(3,iqz,iqy,iqx,kz,ky,kx,iel_ho);
                              const double J32 = Q(4,iqz,iqy,iqx,kz,ky,kx,iel_ho);
                              const double J13 = J31;
                              const double J23 = J32;
                              const double J33 = Q(5,iqz,iqy,iqx,kz,ky,kx,iel_ho);

                              grad_A(0,0,iqy,iz,jz,iqx) += J11*biz*bjz;
                              grad_A(1,0,iqy,iz,jz,iqx) += J21*biz*bjz;
                              grad_A(2,0,iqy,iz,jz,iqx) += J31*giz*bjz;
                              grad_A(0,1,iqy,iz,jz,iqx) += J12*biz*bjz;
                              grad_A(1,1,iqy,iz,jz,iqx) += J22*biz*bjz;
                              grad_A(2,1,iqy,iz,jz,iqx) += J32*giz*bjz;
                              grad_A(0,2,iqy,iz,jz,iqx) += J13*biz*gjz;
                              grad_A(1,2,iqy,iz,jz,iqx) += J23*biz*gjz;
                              grad_A(2,2,iqy,iz,jz,iqx) += J33*giz*gjz;
                           }
                           MFEM_UNROLL(2)
                           for (int jy=0; jy<2; ++jy)
                           {
                              MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 const double biy = (iy == iqy) ? 1.0 : 0.0;
                                 const double giy = (iy == 0) ? -1.0 : 1.0;

                                 const double bjy = (jy == iqy) ? 1.0 : 0.0;
                                 const double gjy = (jy == 0) ? -1.0 : 1.0;

                                 grad_B(0,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,0,iqy,iz,jz,iqx);
                                 grad_B(1,0,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,0,iqy,iz,jz,iqx);
                                 grad_B(2,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,0,iqy,iz,jz,iqx);
                                 grad_B(0,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(0,1,iqy,iz,jz,iqx);
                                 grad_B(1,1,iy,jy,iz,jz,iqx) += giy*gjy*grad_A(1,1,iqy,iz,jz,iqx);
                                 grad_B(2,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(2,1,iqy,iz,jz,iqx);
                                 grad_B(0,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,2,iqy,iz,jz,iqx);
                                 grad_B(1,2,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,2,iqy,iz,jz,iqx);
                                 grad_B(2,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,2,iqy,iz,jz,iqx);
                              }
                           }
                        }
                        FOR_LOOP_UNROLL(2)
                        for (int jy=0; jy<2; ++jy)
                        {
                           FOR_LOOP_UNROLL(2)
                           for (int jx=0; jx<2; ++jx)
                           {
                              FOR_LOOP_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 FOR_LOOP_UNROLL(2)
                                 for (int ix=0; ix<2; ++ix)
                                 {
                                    const double bix = (ix == iqx) ? 1.0 : 0.0;
                                    const double gix = (ix == 0) ? -1.0 : 1.0;

                                    const double bjx = (jx == iqx) ? 1.0 : 0.0;
                                    const double gjx = (jx == 0) ? -1.0 : 1.0;

                                    int ii_loc = ix + 2*iy + 4*iz;
                                    int jj_loc = jx + 2*jy + 4*jz;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    double val = 0.0;
                                    val += gix*gjx*grad_B(0,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(1,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(2,0,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,1,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,2,iy,jy,iz,jz,iqx);

                                    local_mat(ii_loc, jj_loc) += val;
                                 }
                              }
                           }
                        }
                     }
                  }
               }
               // Assemble the local matrix into the macro-element sparse matrix
               // in a format similar to coordinate format. The (I,J) arrays
               // are implicit (not stored explicitly).
               FOR_LOOP_UNROLL(8)
               for (int ii_loc=0; ii_loc<8; ++ii_loc)
               {
                  const int ix = ii_loc%2;
                  const int iy = (ii_loc/2)%2;
                  const int iz = ii_loc/2/2;

                  for (int jj_loc=0; jj_loc<8; ++jj_loc)
                  {
                     const int jx = jj_loc%2;
                     const int jy = (jj_loc/2)%2;
                     const int jz = jj_loc/2/2;
                     const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);

                     if (jj_loc <= ii_loc)
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz), local_mat(ii_loc, jj_loc));
                     }
                     else
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz), local_mat(jj_loc, ii_loc));
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Place the macro-element sparse matrix into the global sparse matrix.
      MFEM_FOREACH_THREAD(iz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(iy,y,nd1d)
         {
            MFEM_FOREACH_THREAD(ix,x,nd1d)
            {
               double col_ptr[nnz_per_row]; // 27

               const int ii_el = ix + nd1d*(iy + nd1d*iz);
               const int ii = el_dof_lex(ii_el, iel_ho);

               // Set column pointer to avoid searching in the row
               for (int j = I[ii], end = I[ii+1]; j < end; j++)
               {
                  const int jj = J[j];
                  int jj_el = -1;
                  for (int k = K[jj], k_end = K[jj+1]; k < k_end; k += 2)
                  {
                     if (dof_glob2loc[k] == iel_ho)
                     {
                        jj_el = dof_glob2loc[k+1];
                        break;
                     }
                  }
                  if (jj_el < 0) { continue; }
                  const int jx = jj_el%nd1d;
                  const int jy = (jj_el/nd1d)%nd1d;
                  const int jz = jj_el/nd1d/nd1d;
                  const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);
                  col_ptr[jj_off] = j;
               }

               const int jx_begin = (ix > 0) ? ix - 1 : 0;
               const int jx_end = (ix < order) ? ix + 1 : order;

               const int jy_begin = (iy > 0) ? iy - 1 : 0;
               const int jy_end = (iy < order) ? iy + 1 : order;

               const int jz_begin = (iz > 0) ? iz - 1 : 0;
               const int jz_end = (iz < order) ? iz + 1 : order;

               for (int jz=jz_begin; jz<=jz_end; ++jz)
               {
                  for (int jy=jy_begin; jy<=jy_end; ++jy)
                  {
                     for (int jx=jx_begin; jx<=jx_end; ++jx)
                     {
                        const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);
                        const double Vji = V(jj_off, ix, iy, iz);
                        const int col_ptr_jj = col_ptr[jj_off];
                        if ((ix == 0 && jx == 0) || (ix == order && jx == order) ||
                            (iy == 0 && jy == 0) || (iy == order && jy == order) ||
                            (iz == 0 && jz == 0) || (iz == order && jz == order))
                        {
                           AtomicAdd(A[col_ptr_jj], Vji);
                        }
                        else
                        {
                           A[col_ptr_jj] += Vji;
                        }
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
   });

   NvtxPop(Assembly);
   NvtxPop(Assemble3DBatchedLOR_GPU_Kernels);
}

#define TEMPLATE_LOR_KERNEL(order) \
template void Assemble3DBatchedLOR_GPU<order>\
    (Mesh &,Array<int> &,Array<int> &, Array<int> &, \
     Vector &,\
     Mesh &,\
     FiniteElementSpace &,SparseMatrix &)

TEMPLATE_LOR_KERNEL(1);
TEMPLATE_LOR_KERNEL(2);
TEMPLATE_LOR_KERNEL(3);
TEMPLATE_LOR_KERNEL(4);

} // namespace mfem
