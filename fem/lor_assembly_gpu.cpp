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

#include "restriction.hpp"
#include "lor_assembly.hpp"

#include "../linalg/dtensor.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"
#include "../general/nvvp.hpp"

#include <cassert>

namespace mfem
{

template <int order> static
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              const int *map,
                              Array<int> &dof_glob2loc_,
                              Array<int> &dof_glob2loc_offsets_,
                              Array<int> &el_dof_lex_,
                              Vector &Q_,
                              Vector &el_vert_,
                              Vector &Xe,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat)
{
   NvtxPush(Assemble3DBatchedLOR_GPU_Kernels, LightCoral);
   const int nel_ho = mesh_ho.GetNE();
   const int nel_lor = mesh_lor.GetNE();
   //const int ndof = fes_ho.GetVSize();
   //dbg("nel_ho:%d nel_lor:%d",nel_ho,nel_lor);

   static constexpr int dim = 3;
   static constexpr int nv = 8;
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

   //const auto MAP = Reshape(map, nd1d,nd1d,nd1d, nel_lor);
   //assert(3*8*nel_lor == Xe.Size());
   //const auto XE = Reshape(Xe.Read(), 8, 3, nel_lor);
   const auto el_vert = Reshape(el_vert_.Read(), dim, nv, nel_lor);
   const auto Q = Reshape(Q_.Write(), ddm2, 2,2,2, order,order,order, nel_ho);

   //const auto MAP = Reshape(map, D1D,D1D,D1D, NE);
   NvtxPush(Assembly, SeaGreen);
   MFEM_FORALL_3D(iel_ho, nel_ho, order, order, order,
   {
      //dbg("iel_ho:%d/%d",iel_ho,nel_ho-1);
      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(kz,z,order)
      {
         MFEM_FOREACH_THREAD(ky,y,order)
         {
            MFEM_FOREACH_THREAD(kx,x,order)
            {
               const int k = kx + order * (ky + order * kz);
               constexpr int order_d = order*order*order;
               const int iel_lor = order_d*iel_ho + k;
               //dbg("iel_lor:%d/%d",iel_lor,nel_ho-1);

               //const int gid = MAP(kx, ky, kz, iel_lor);
               //const int idx = gid >= 0 ? gid : -1 - gid;

               //const int e = iel_lor;
               const double *v0 = &el_vert(0, 0, iel_lor);
               //dbg("v0:%.8e %.8e %.8e",v0[0],v0[1],v0[2]);
               //dbg("x0:%.8e %.8e %.8e",XE(0,0,e),XE(0,1,e),XE(0,2,e));

               const double *v1 = &el_vert(0, 1, iel_lor);
               //dbg("v1:%.8e %.8e %.8e",v1[0],v1[1],v1[2]);
               //dbg("x1:%.8e %.8e %.8e",XE(1,0,e),XE(1,1,e),XE(1,2,e));

               const double *v2 = &el_vert(0, 2, iel_lor);
               //dbg("v2:%.8e %.8e %.8e",v2[0],v2[1],v2[2]);
               //dbg("x2:%.8e %.8e %.8e",XE(2,0,e),XE(2,1,e),XE(2,2,e));

               const double *v3 = &el_vert(0, 3, iel_lor);
               //dbg("v3:%.8e %.8e %.8e",v3[0],v3[1],v3[2]);
               //dbg("x3:%.8e %.8e %.8e",XE(3,0,e),XE(3,1,e),XE(3,2,e));

               const double *v4 = &el_vert(0, 4, iel_lor);
               //dbg("v4:%.8e %.8e %.8e",v4[0],v4[1],v4[2]);
               //dbg("x4:%.8e %.8e %.8e",XE(4,0,e),XE(4,1,e),XE(4,2,e));

               const double *v5 = &el_vert(0, 5, iel_lor);
               //dbg("v5:%.8e %.8e %.8e",v5[0],v5[1],v5[2]);
               //dbg("x5:%.8e %.8e %.8e",XE(5,0,e),XE(5,1,e),XE(5,2,e));

               const double *v6 = &el_vert(0, 6, iel_lor);
               //dbg("v6:%.8e %.8e %.8e",v6[0],v6[1],v6[2]);
               //dbg("x6:%.8e %.8e %.8e",XE(6,0,e),XE(6,1,e),XE(6,2,e));

               const double *v7 = &el_vert(0, 7, iel_lor);
               //dbg("v7:%.8e %.8e %.8e",v7[0],v7[1],v7[2]);
               //dbg("x7:%.8e %.8e %.8e",XE(7,0,e),XE(7,1,e),XE(7,2,e));

               MFEM_UNROLL(2)
               for (int iqz=0; iqz<2; ++iqz)
               {
                  MFEM_UNROLL(2)
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     MFEM_UNROLL(2)
                     for (int iqx=0; iqx<2; ++iqx)
                     {
                        const double x = iqx;
                        const double y = iqy;
                        const double z = iqz;
                        const double w = 1.0/8.0;

                        //const double x0 = XE(0,e);
                        //dbg("x0:%f v0[0]:%f",x0,v0[0]);

                        // c: (1-x)(1-y)(1-z)v0[c] + x (1-y)(1-z)v1[c] + x y (1-z)v2[c] + (1-x) y (1-z)v3[c]
                        //  + (1-x)(1-y) z   v4[c] + x (1-y) z   v5[c] + x y z    v6[c] + (1-x) y z    v7[c]
                        const double J11 = -(1-y)*(1-z)*v0[0]
                        + (1-y)*(1-z)*v1[0] + y*(1-z)*v2[0] - y*(1-z)*v3[0]
                        - (1-y)*z*v4[0] + (1-y)*z*v5[0] + y*z*v6[0] - y*z*v7[0];

                        const double J12 = -(1-x)*(1-z)*v0[0]
                        - x*(1-z)*v1[0] + x*(1-z)*v2[0] + (1-x)*(1-z)*v3[0]
                        - (1-x)*z*v4[0] - x*z*v5[0] + x*z*v6[0] + (1-x)*z*v7[0];

                        const double J13 = -(1-x)*(1-y)*v0[0] - x*(1-y)*v1[0]
                        - x*y*v2[0] - (1-x)*y*v3[0] + (1-x)*(1-y)*v4[0]
                        + x*(1-y)*v5[0] + x*y*v6[0] + (1-x)*y*v7[0];

                        const double J21 = -(1-y)*(1-z)*v0[1] + (1-y)*(1-z)*v1[1]
                        + y*(1-z)*v2[1] - y*(1-z)*v3[1] - (1-y)*z*v4[1]
                        + (1-y)*z*v5[1] + y*z*v6[1] - y*z*v7[1];

                        const double J22 = -(1-x)*(1-z)*v0[1] - x*(1-z)*v1[1]
                        + x*(1-z)*v2[1] + (1-x)*(1-z)*v3[1]- (1-x)*z*v4[1] -
                        x*z*v5[1] + x*z*v6[1] + (1-x)*z*v7[1];

                        const double J23 = -(1-x)*(1-y)*v0[1] - x*(1-y)*v1[1]
                        - x*y*v2[1] - (1-x)*y*v3[1] + (1-x)*(1-y)*v4[1]
                        + x*(1-y)*v5[1] + x*y*v6[1] + (1-x)*y*v7[1];

                        const double J31 = -(1-y)*(1-z)*v0[2] + (1-y)*(1-z)*v1[2]
                        + y*(1-z)*v2[2] - y*(1-z)*v3[2]- (1-y)*z*v4[2] +
                        (1-y)*z*v5[2] + y*z*v6[2] - y*z*v7[2];

                        const double J32 = -(1-x)*(1-z)*v0[2] - x*(1-z)*v1[2]
                        + x*(1-z)*v2[2] + (1-x)*(1-z)*v3[2] - (1-x)*z*v4[2]
                        - x*z*v5[2] + x*z*v6[2] + (1-x)*z*v7[2];

                        const double J33 = -(1-x)*(1-y)*v0[2] - x*(1-y)*v1[2]
                        - x*y*v2[2] - (1-x)*y*v3[2] + (1-x)*(1-y)*v4[2]
                        + x*(1-y)*v5[2] + x*y*v6[2] + (1-x)*y*v7[2];

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
                        Q(0,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
                        Q(1,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
                        Q(2,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
                        Q(3,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
                        Q(4,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
                        Q(5,iqz,iqy,iqx,kz,ky,kx,
                          iel_ho) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      //dbg("Quit"); assert(false);

      // nnz_per_el = nnz_per_row(=27) * (order+1) * (order+1) * (order+1);
      MFEM_SHARED double V_[nnz_per_el];
      DeviceTensor<4> V(V_, nnz_per_row, nd1d, nd1d, nd1d);

      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      //
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      MFEM_FOREACH_THREAD(iz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(iy,y,nd1d)
         {
            MFEM_FOREACH_THREAD(ix,x,nd1d)
            {
               MFEM_UNROLL(27)
               for (int j=0; j<nnz_per_row; ++j)
               {
                  V(j,ix,iy,iz) = 0.0;
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

               //const int k = kx + ky*order + kz*order*order;
               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }
               // Intermediate quantities (see e.g. Mora and Demkowicz for
               // notation).
               for (int i=0; i<sz_grad_A; ++i) { grad_A[i] = 0.0; }
               for (int i=0; i<sz_grad_B; ++i) { grad_B[i] = 0.0; }

               MFEM_UNROLL(2)
               for (int iqx=0; iqx<2; ++iqx)
               {
                  MFEM_UNROLL(2)
                  for (int jz=0; jz<2; ++jz)
                  {
                     // Note loop starts at iz=jz here, taking advantage of
                     // symmetries.
                     MFEM_UNROLL(2)
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
                        MFEM_UNROLL(2)
                        for (int jy=0; jy<2; ++jy)
                        {
                           MFEM_UNROLL(2)
                           for (int jx=0; jx<2; ++jx)
                           {
                              MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 MFEM_UNROLL(2)
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
               MFEM_UNROLL(8)
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

void AssembleBatchedLOR_GPU(BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs,
                            OperatorHandle &Ah)
{
   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();
   const int ndofs = fes_ho.GetTrueVSize();

   const int nel_ho = mesh_ho.GetNE();
   const int nel_lor = mesh_lor.GetNE();
   const int ndof = fes_ho.GetVSize();
   constexpr int nv = 8;
   const int ddm2 = (dim*(dim+1))/2;
   const int nd1d = order + 1;
   const int ndof_per_el = nd1d*nd1d*nd1d;

   const bool has_to_init = Ah.Ptr() == nullptr;
   SparseMatrix *A_mat = Ah.As<SparseMatrix>();

   Vector Xe;
   const int *map = nullptr;

   static Array<int> *dof_glob2loc_ = nullptr;//(2*ndof_per_el*nel_ho);
   static Array<int> *dof_glob2loc_offsets_ = nullptr;//(ndof+1);
   static Array<int> *el_dof_lex_ = nullptr;//(ndof_per_el*nel_ho);
   static Vector *Q_ = nullptr;//(nel_ho*pow(order,dim)*nv*ddm2);
   static Vector *el_vert_ = nullptr;//(dim*nv*nel_lor);

   if (has_to_init)
   {
      dof_glob2loc_ = new Array<int>(2*ndof_per_el*nel_ho);
      dof_glob2loc_offsets_ = new Array<int>(ndof+1);
      el_dof_lex_ = new Array<int>(ndof_per_el*nel_ho);
      Q_ = new Vector(nel_ho*pow(order,dim)*nv*ddm2);
      el_vert_ = new Vector(dim*nv*nel_lor);

      mesh_lor.EnsureNodes();
      const GridFunction *nodes = mesh_lor.GetNodes();
      assert(nodes);
      const FiniteElementSpace *nfes = nodes->FESpace();
      constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      //dbg("nfes NVDofs:%d",nfes->GetNVDofs());
      const Operator *ERop = nfes->GetElementRestriction(ordering);
      const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
      map = ER ? ER->GatherMap().Read() : nullptr;
      assert(map);

      const GridFunction *X = mesh_lor.GetNodes();
      //dbg("X:%d",X->Size()); X->Print();
      Xe.SetSize(ER->Height());
      ER->Mult(*X,Xe);
      //dbg("Xe:%d",Xe.Size()); Xe.Print();

      NvtxPush(AssembleBatchedLOR_GPU, LawnGreen);
      MFEM_VERIFY(UsesTensorBasis(fes_ho),
                  "Batched LOR assembly requires tensor basis");

      // the sparsity pattern is defined from the map: element->dof
      const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();

      NvtxPush(Sparsity, PaleTurquoise);
      Table dof_dof, dof_elem;

      NvtxPush(Transpose, LightGoldenrod);
      Transpose(elem_dof, dof_elem, ndofs);
      NvtxPop(Transpose);

      NvtxPush(Mult, LightGoldenrod);
      mfem::Mult(dof_elem, elem_dof, dof_dof);
      NvtxPop();

      NvtxPush(SortRows, LightGoldenrod);
      dof_dof.SortRows();
      int *I = dof_dof.GetI();
      int *J = dof_dof.GetJ();
      NvtxPop();

      NvtxPush(A_Allocate, Cyan);
      double *data = Memory<double>(I[ndofs]);
      NvtxPop();

      NvtxPush(newSparseMatrix, PeachPuff);
      A_mat = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
      NvtxPop();

      NvtxPush(Ah=0.0, Peru);
      *A_mat = 0.0;
      NvtxPop();

      NvtxPush(LoseData, PaleTurquoise);
      dof_dof.LoseData();
      NvtxPop();

      {
         NvtxPush(BlockMapping, Olive);
         Array<int> dofs;
         const Array<int> &lex_map =
            dynamic_cast<const NodalFiniteElement&>
            (*fes_ho.GetFE(0)).GetLexicographicOrdering();
         *dof_glob2loc_offsets_ = 0;
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               (*el_dof_lex_)[i + iel_ho*ndof_per_el] = dof;
               (*dof_glob2loc_offsets_)[dof+1] += 2;
            }
         }
         dof_glob2loc_offsets_->PartialSum();
         // Sanity check
         MFEM_VERIFY((*dof_glob2loc_offsets_)[ndof] == dof_glob2loc_->Size(), "");
         Array<int> dof_ptr(ndof);
         for (int i=0; i<ndof; ++i) { dof_ptr[i] = (*dof_glob2loc_offsets_)[i]; }
         for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
         {
            fes_ho.GetElementDofs(iel_ho, dofs);
            for (int i=0; i<ndof_per_el; ++i)
            {
               const int dof = dofs[lex_map[i]];
               (*dof_glob2loc_)[dof_ptr[dof]++] = iel_ho;
               (*dof_glob2loc_)[dof_ptr[dof]++] = i;
            }
         }
         NvtxPop(BlockMapping);
      }
      {
         NvtxPush(GetVertices, IndianRed);
         for (int iel_lor=0; iel_lor<nel_lor; ++iel_lor)
         {
            Array<int> v;
            mesh_lor.GetElementVertices(iel_lor, v);
            for (int iv=0; iv<nv; ++iv)
            {
               const double *vc = mesh_lor.GetVertex(v[iv]);
               for (int d=0; d<dim; ++d)
               {
                  (*el_vert_)[d + iv*dim + iel_lor*nv*dim] = vc[d];
               }
            }
         }
         NvtxPop(GetVertices);
      }
      NvtxPop(Sparsity);
   }


   void (*Kernel)(Mesh &mesh_lor,
                  const int *map,
                  Array<int> &dof_glob2loc_,
                  Array<int> &dof_glob2loc_offsets_,
                  Array<int> &el_dof_lex_,
                  Vector &Q_,
                  Vector &el_vert_,
                  Vector &Xe,
                  Mesh &mesh_ho,
                  FiniteElementSpace &fes_ho,
                  SparseMatrix &A_mat) = nullptr;

   if (dim == 2) { MFEM_ABORT("Unsuported!"); }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Kernel = Assemble3DBatchedLOR_GPU<1>; break;
         case 2: Kernel = Assemble3DBatchedLOR_GPU<2>; break;
         case 3: Kernel = Assemble3DBatchedLOR_GPU<3>; break;
         case 4: Kernel = Assemble3DBatchedLOR_GPU<4>; break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   Kernel(mesh_lor,map,
          *dof_glob2loc_,
          *dof_glob2loc_offsets_,
          *el_dof_lex_,
          *Q_,
          *el_vert_,
          Xe,mesh_ho,fes_ho,*A_mat);

   {
      NvtxPush(Diag=0.0, DarkGoldenrod);
      const auto I_d = A_mat->ReadI();
      const auto J_d = A_mat->ReadJ();
      auto A_d = A_mat->ReadWriteData();
      const int n_ess_dofs = ess_dofs.Size();

      MFEM_FORALL(i, n_ess_dofs,
      {
         for (int j=I_d[i]; j<I_d[i+1]; ++j)
         {
            if (J_d[j] != i)
            {
               A_d[j] = 0.0;
            }
         }
      });
      NvtxPop();
   }

   if (has_to_init) { Ah.Reset(A_mat); } // A now owns A_mat
   NvtxPop(AssembleBatchedLOR_GPU);
   //dbg("Quit"); assert(false);
}

} // namespace mfem
