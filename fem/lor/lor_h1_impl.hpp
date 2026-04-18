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

#include "lor_util.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

template <int ORDER, int SDIM>
void BatchedLOR_H1::Assemble2D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 4;
   static constexpr int dim = 2;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = ORDER + 1;
   static constexpr int ndof_per_el = nd1d*nd1d;
   static constexpr int nnz_per_row = 9;
   static constexpr int sz_local_mat = nv*nv;

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1)
                   : Reshape(c1.Read(), nd1d, nd1d, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1)
                   : Reshape(c2.Read(), nd1d, nd1d, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, nd1d, nd1d, nel_ho);

   auto X = X_vert.Read();

   mfem::forall_2D(nel_ho, ORDER, ORDER, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,ix,iy) stores the jth nonzero in the row of the sparse matrix
      // corresponding to local DOF (ix, iy).
      MFEM_FOREACH_THREAD(iy,y,nd1d)
      {
         MFEM_FOREACH_THREAD(ix,x,nd1d)
         {
            for (int j=0; j<nnz_per_row; ++j)
            {
               V(j,ix,iy,iel_ho) = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(ky,y,ORDER)
      {
         MFEM_FOREACH_THREAD(kx,x,ORDER)
         {
            real_t Q_[(ddm2 + 1)*nv];
            real_t local_mat_[sz_local_mat];
            DeviceTensor<3> Q(Q_, ddm2 + 1, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, nv, nv);

            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            SetupLORQuadData2D<ORDER,SDIM,false,false>(X, iel_ho, kx, ky, Q, false);

            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  const real_t mq = const_mq ? MQ(0,0,0) : MQ(kx+iqx, ky+iqy, iel_ho);
                  const real_t dq = const_dq ? DQ(0,0,0) : DQ(kx+iqx, ky+iqy, iel_ho);
                  for (int jy=0; jy<2; ++jy)
                  {
                     const real_t bjy = (jy == iqy) ? 1.0 : 0.0;
                     const real_t gjy = (jy == 0) ? -1.0 : 1.0;
                     for (int jx=0; jx<2; ++jx)
                     {
                        const real_t bjx = (jx == iqx) ? 1.0 : 0.0;
                        const real_t gjx = (jx == 0) ? -1.0 : 1.0;

                        const real_t djx = gjx*bjy;
                        const real_t djy = bjx*gjy;

                        int jj_loc = jx + 2*jy;

                        for (int iy=0; iy<2; ++iy)
                        {
                           const real_t biy = (iy == iqy) ? 1.0 : 0.0;
                           const real_t giy = (iy == 0) ? -1.0 : 1.0;
                           for (int ix=0; ix<2; ++ix)
                           {
                              const real_t bix = (ix == iqx) ? 1.0 : 0.0;
                              const real_t gix = (ix == 0) ? -1.0 : 1.0;

                              const real_t dix = gix*biy;
                              const real_t diy = bix*giy;

                              int ii_loc = ix + 2*iy;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              real_t val = 0.0;
                              val += dix*djx*Q(0,iqy,iqx);
                              val += (dix*djy + diy*djx)*Q(1,iqy,iqx);
                              val += diy*djy*Q(2,iqy,iqx);
                              val *= dq;

                              val += mq*bix*biy*bjx*bjy*Q(3,iqy,iqx);

                              local_mat(ii_loc, jj_loc) += val;
                           }
                        }
                     }
                  }
               }
            }
            // Assemble the local matrix into the macro-element sparse matrix
            // in a format similar to coordinate format. The (I,J) arrays
            // are implicit (not stored explicitly).
            for (int ii_loc=0; ii_loc<nv; ++ii_loc)
            {
               const int ix = ii_loc%2;
               const int iy = ii_loc/2;
               for (int jj_loc=0; jj_loc<nv; ++jj_loc)
               {
                  const int jx = jj_loc%2;
                  const int jy = jj_loc/2;
                  const int jj_off = (jx-ix+1) + 3*(jy-iy+1);

                  // Symmetry
                  if (jj_loc <= ii_loc)
                  {
                     AtomicAdd(V(jj_off, ix+kx, iy+ky, iel_ho), local_mat(ii_loc, jj_loc));
                  }
                  else
                  {
                     AtomicAdd(V(jj_off, ix+kx, iy+ky, iel_ho), local_mat(jj_loc, ii_loc));
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row*ndof_per_el);
   sparse_mapping = -1;
   auto map = Reshape(sparse_mapping.HostReadWrite(), nnz_per_row, ndof_per_el);
   for (int iy=0; iy<nd1d; ++iy)
   {
      const int jy_begin = (iy > 0) ? iy - 1 : 0;
      const int jy_end = (iy < ORDER) ? iy + 1 : ORDER;
      for (int ix=0; ix<nd1d; ++ix)
      {
         const int jx_begin = (ix > 0) ? ix - 1 : 0;
         const int jx_end = (ix < ORDER) ? ix + 1 : ORDER;
         const int ii_el = ix + nd1d*iy;
         for (int jy=jy_begin; jy<=jy_end; ++jy)
         {
            for (int jx=jx_begin; jx<=jx_end; ++jx)
            {
               const int jj_off = (jx-ix+1) + 3*(jy-iy+1);
               const int jj_el = jx + nd1d*jy;
               map(jj_off, ii_el) = jj_el;
            }
         }
      }
   }
}

template <int ORDER>
void BatchedLOR_H1::Assemble3D()
{
   const int nel_ho = fes_ho.GetNE();
   static constexpr int nv = 8;
   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = ORDER + 1;
   static constexpr int ndof_per_el = nd1d*nd1d*nd1d;
   static constexpr int nnz_per_row = 27;
   static constexpr int sz_grad_A = 3*3*2*2*2*2;
   static constexpr int sz_grad_B = sz_grad_A*2;
   static constexpr int sz_mass_A = 2*2*2*2;
   static constexpr int sz_mass_B = sz_mass_A*2;
   static constexpr int sz_local_mat = nv*nv;

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1, 1)
                   : Reshape(c1.Read(), nd1d, nd1d, nd1d, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1, 1)
                   : Reshape(c2.Read(), nd1d, nd1d, nd1d, nel_ho);

   sparse_ij.SetSize(nel_ho*ndof_per_el*nnz_per_row);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, nd1d, nd1d, nd1d, nel_ho);

   auto X = X_vert.Read();

   // Last thread dimension is lowered to avoid "too many resources" error
   mfem::forall_3D(nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
                   [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      MFEM_FOREACH_THREAD(iz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(iy,y,nd1d)
         {
            MFEM_FOREACH_THREAD(ix,x,nd1d)
            {
               MFEM_UNROLL(nnz_per_row)
               for (int j=0; j<nnz_per_row; ++j)
               {
                  V(j,ix,iy,iz,iel_ho) = 0.0;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(kz,z,ORDER)
      {
         MFEM_FOREACH_THREAD(ky,y,ORDER)
         {
            MFEM_FOREACH_THREAD(kx,x,ORDER)
            {
               real_t Q_[(ddm2 + 1)*nv];
               real_t grad_A_[sz_grad_A];
               real_t grad_B_[sz_grad_B];
               real_t mass_A_[sz_mass_A];
               real_t mass_B_[sz_mass_B];
               real_t local_mat_[sz_local_mat];

               DeviceTensor<4> Q(Q_, ddm2 + 1, 2, 2, 2);
               DeviceTensor<2> local_mat(local_mat_, nv, nv);
               DeviceTensor<6> grad_A(grad_A_, 3, 3, 2, 2, 2, 2);
               DeviceTensor<7> grad_B(grad_B_, 3, 3, 2, 2, 2, 2, 2);
               DeviceTensor<4> mass_A(mass_A_, 2, 2, 2, 2);
               DeviceTensor<5> mass_B(mass_B_, 2, 2, 2, 2, 2);

               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

               // Intermediate quantities
               // (see e.g. Mora and Demkowicz for notation).
               for (int i=0; i<sz_grad_A; ++i) { grad_A[i] = 0.0; }
               for (int i=0; i<sz_grad_B; ++i) { grad_B[i] = 0.0; }

               for (int i=0; i<sz_mass_A; ++i) { mass_A[i] = 0.0; }
               for (int i=0; i<sz_mass_B; ++i) { mass_B[i] = 0.0; }

               real_t vx[8], vy[8], vz[8];
               LORVertexCoordinates3D<ORDER>(X, iel_ho, kx, ky, kz, vx, vy, vz);

               // MFEM_UNROLL(2)
               for (int iqz=0; iqz<2; ++iqz)
               {
                  // MFEM_UNROLL(2)
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     // MFEM_UNROLL(2)
                     for (int iqx=0; iqx<2; ++iqx)
                     {
                        const real_t x = iqx;
                        const real_t y = iqy;
                        const real_t z = iqz;
                        const real_t w = 1.0/8.0;

                        real_t J_[3*3];
                        DeviceTensor<2> J(J_, 3, 3);

                        Jacobian3D(x, y, z, vx, vy, vz, J);

                        const real_t detJ = Det3D(J);
                        const real_t w_detJ = w/detJ;

                        // adj(J)
                        real_t A_[3*3];
                        DeviceTensor<2> A(A_, 3, 3);
                        Adjugate3D(J, A);

                        Q(0,iqz,iqy,iqx) = w_detJ*(A(0,0)*A(0,0)+A(0,1)*A(0,1)+A(0,2)*A(0,2)); // 1,1
                        Q(1,iqz,iqy,iqx) = w_detJ*(A(0,0)*A(1,0)+A(0,1)*A(1,1)+A(0,2)*A(1,2)); // 2,1
                        Q(2,iqz,iqy,iqx) = w_detJ*(A(0,0)*A(2,0)+A(0,1)*A(2,1)+A(0,2)*A(2,2)); // 3,1
                        Q(3,iqz,iqy,iqx) = w_detJ*(A(1,0)*A(1,0)+A(1,1)*A(1,1)+A(1,2)*A(1,2)); // 2,2
                        Q(4,iqz,iqy,iqx) = w_detJ*(A(1,0)*A(2,0)+A(1,1)*A(2,1)+A(1,2)*A(2,2)); // 3,2
                        Q(5,iqz,iqy,iqx) = w_detJ*(A(2,0)*A(2,0)+A(2,1)*A(2,1)+A(2,2)*A(2,2)); // 3,3
                        Q(6,iqz,iqy,iqx) = w*detJ;
                     }
                  }
               }

               // MFEM_UNROLL(2)
               for (int iqx=0; iqx<2; ++iqx)
               {
                  // MFEM_UNROLL(2)
                  for (int jz=0; jz<2; ++jz)
                  {
                     // Note loop starts at iz=jz here, taking advantage of
                     // symmetries.
                     // MFEM_UNROLL(2)
                     for (int iz=jz; iz<2; ++iz)
                     {
                        // MFEM_UNROLL(2)
                        for (int iqy=0; iqy<2; ++iqy)
                        {
                           // MFEM_UNROLL(2)
                           for (int iqz=0; iqz<2; ++iqz)
                           {
                              const real_t mq = const_mq ? MQ(0,0,0,0) : MQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);
                              const real_t dq = const_dq ? DQ(0,0,0,0) : DQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);

                              const real_t biz = (iz == iqz) ? 1.0 : 0.0;
                              const real_t giz = (iz == 0) ? -1.0 : 1.0;

                              const real_t bjz = (jz == iqz) ? 1.0 : 0.0;
                              const real_t gjz = (jz == 0) ? -1.0 : 1.0;

                              const real_t J11 = Q(0,iqz,iqy,iqx);
                              const real_t J21 = Q(1,iqz,iqy,iqx);
                              const real_t J31 = Q(2,iqz,iqy,iqx);
                              const real_t J12 = J21;
                              const real_t J22 = Q(3,iqz,iqy,iqx);
                              const real_t J32 = Q(4,iqz,iqy,iqx);
                              const real_t J13 = J31;
                              const real_t J23 = J32;
                              const real_t J33 = Q(5,iqz,iqy,iqx);

                              grad_A(0,0,iqy,iz,jz,iqx) += dq*J11*biz*bjz;
                              grad_A(1,0,iqy,iz,jz,iqx) += dq*J21*biz*bjz;
                              grad_A(2,0,iqy,iz,jz,iqx) += dq*J31*giz*bjz;
                              grad_A(0,1,iqy,iz,jz,iqx) += dq*J12*biz*bjz;
                              grad_A(1,1,iqy,iz,jz,iqx) += dq*J22*biz*bjz;
                              grad_A(2,1,iqy,iz,jz,iqx) += dq*J32*giz*bjz;
                              grad_A(0,2,iqy,iz,jz,iqx) += dq*J13*biz*gjz;
                              grad_A(1,2,iqy,iz,jz,iqx) += dq*J23*biz*gjz;
                              grad_A(2,2,iqy,iz,jz,iqx) += dq*J33*giz*gjz;

                              real_t wdetJ = Q(6,iqz,iqy,iqx);
                              mass_A(iqy,iz,jz,iqx) += mq*wdetJ*biz*bjz;
                           }
                           // MFEM_UNROLL(2)
                           for (int jy=0; jy<2; ++jy)
                           {
                              // MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 const real_t biy = (iy == iqy) ? 1.0 : 0.0;
                                 const real_t giy = (iy == 0) ? -1.0 : 1.0;

                                 const real_t bjy = (jy == iqy) ? 1.0 : 0.0;
                                 const real_t gjy = (jy == 0) ? -1.0 : 1.0;

                                 grad_B(0,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,0,iqy,iz,jz,iqx);
                                 grad_B(1,0,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,0,iqy,iz,jz,iqx);
                                 grad_B(2,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,0,iqy,iz,jz,iqx);
                                 grad_B(0,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(0,1,iqy,iz,jz,iqx);
                                 grad_B(1,1,iy,jy,iz,jz,iqx) += giy*gjy*grad_A(1,1,iqy,iz,jz,iqx);
                                 grad_B(2,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(2,1,iqy,iz,jz,iqx);
                                 grad_B(0,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,2,iqy,iz,jz,iqx);
                                 grad_B(1,2,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,2,iqy,iz,jz,iqx);
                                 grad_B(2,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,2,iqy,iz,jz,iqx);

                                 mass_B(iy,jy,iz,jz,iqx) += biy*bjy*mass_A(iqy,iz,jz,iqx);
                              }
                           }
                        }
                        // MFEM_UNROLL(2)
                        for (int jy=0; jy<2; ++jy)
                        {
                           // MFEM_UNROLL(2)
                           for (int jx=0; jx<2; ++jx)
                           {
                              // MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 // MFEM_UNROLL(2)
                                 for (int ix=0; ix<2; ++ix)
                                 {
                                    const real_t bix = (ix == iqx) ? 1.0 : 0.0;
                                    const real_t gix = (ix == 0) ? -1.0 : 1.0;

                                    const real_t bjx = (jx == iqx) ? 1.0 : 0.0;
                                    const real_t gjx = (jx == 0) ? -1.0 : 1.0;

                                    int ii_loc = ix + 2*iy + 4*iz;
                                    int jj_loc = jx + 2*jy + 4*jz;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    real_t val = 0.0;
                                    val += gix*gjx*grad_B(0,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(1,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(2,0,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,1,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,2,iy,jy,iz,jz,iqx);

                                    val += bix*bjx*mass_B(iy,jy,iz,jz,iqx);

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
               // MFEM_UNROLL(8)
               for (int ii_loc=0; ii_loc<nv; ++ii_loc)
               {
                  const int ix = ii_loc%2;
                  const int iy = (ii_loc/2)%2;
                  const int iz = ii_loc/2/2;

                  for (int jj_loc=0; jj_loc<nv; ++jj_loc)
                  {
                     const int jx = jj_loc%2;
                     const int jy = (jj_loc/2)%2;
                     const int jz = jj_loc/2/2;
                     const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);

                     if (jj_loc <= ii_loc)
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz, iel_ho), local_mat(ii_loc, jj_loc));
                     }
                     else
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz, iel_ho), local_mat(jj_loc, ii_loc));
                     }
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row*ndof_per_el);
   sparse_mapping = -1;
   auto map = Reshape(sparse_mapping.HostReadWrite(), nnz_per_row, ndof_per_el);
   for (int iz=0; iz<nd1d; ++iz)
   {
      const int jz_begin = (iz > 0) ? iz - 1 : 0;
      const int jz_end = (iz < ORDER) ? iz + 1 : ORDER;
      for (int iy=0; iy<nd1d; ++iy)
      {
         const int jy_begin = (iy > 0) ? iy - 1 : 0;
         const int jy_end = (iy < ORDER) ? iy + 1 : ORDER;
         for (int ix=0; ix<nd1d; ++ix)
         {
            const int jx_begin = (ix > 0) ? ix - 1 : 0;
            const int jx_end = (ix < ORDER) ? ix + 1 : ORDER;

            const int ii_el = ix + nd1d*(iy + nd1d*iz);

            for (int jz=jz_begin; jz<=jz_end; ++jz)
            {
               for (int jy=jy_begin; jy<=jy_end; ++jy)
               {
                  for (int jx=jx_begin; jx<=jx_end; ++jx)
                  {
                     const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);
                     const int jj_el = jx + nd1d*(jy + nd1d*jz);
                     map(jj_off, ii_el) = jj_el;
                  }
               }
            }
         }
      }
   }
}

} // namespace mfem
