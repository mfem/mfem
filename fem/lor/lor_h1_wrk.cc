// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_h1.hpp"
#include "lor_util.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

#define MFEM_DEBUG_COLOR 87
#include "../../general/debug.hpp"

namespace mfem
{

template <int ORDER>
void BatchedLOR_H1::Assemble2D()
{
   dbg();
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

   MFEM_FORALL_2D(iel_ho, nel_ho, ORDER, ORDER, 1,
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
            double Q_[(ddm2 + 1)*nv];
            double local_mat_[sz_local_mat];
            DeviceTensor<3> Q(Q_, ddm2 + 1, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, nv, nv);

            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            double vx[4], vy[4];
            LORVertexCoordinates2D<ORDER>(X, iel_ho, kx, ky, vx, vy);

            for (int iqy=0; iqy<2; ++iqy)
            {
               for (int iqx=0; iqx<2; ++iqx)
               {
                  const double x = iqx;
                  const double y = iqy;
                  const double w = 1.0/4.0;

                  double J_[2*2];
                  DeviceTensor<2> J(J_, 2, 2);

                  Jacobian2D(x, y, vx, vy, J);

                  const double detJ = Det2D(J);
                  const double w_detJ = w/detJ;

                  Q(0,iqy,iqx) = w_detJ * (J(0,1)*J(0,1) + J(1,1)*J(1,1)); // 1,1
                  Q(1,iqy,iqx) = -w_detJ * (J(0,1)*J(0,0) + J(1,1)*J(1,0)); // 1,2
                  Q(2,iqy,iqx) = w_detJ * (J(0,0)*J(0,0) + J(1,0)*J(1,0)); // 2,2
                  Q(3,iqy,iqx) = w*detJ;
               }
            }
            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  const double mq = const_mq ? MQ(0,0,0) : MQ(kx+iqx, ky+iqy, iel_ho);
                  const double dq = const_dq ? DQ(0,0,0) : DQ(kx+iqx, ky+iqy, iel_ho);
                  for (int jy=0; jy<2; ++jy)
                  {
                     const double bjy = (jy == iqy) ? 1.0 : 0.0;
                     const double gjy = (jy == 0) ? -1.0 : 1.0;
                     for (int jx=0; jx<2; ++jx)
                     {
                        const double bjx = (jx == iqx) ? 1.0 : 0.0;
                        const double gjx = (jx == 0) ? -1.0 : 1.0;

                        const double djx = gjx*bjy;
                        const double djy = bjx*gjy;

                        int jj_loc = jx + 2*jy;

                        for (int iy=0; iy<2; ++iy)
                        {
                           const double biy = (iy == iqy) ? 1.0 : 0.0;
                           const double giy = (iy == 0) ? -1.0 : 1.0;
                           for (int ix=0; ix<2; ++ix)
                           {
                              const double bix = (ix == iqx) ? 1.0 : 0.0;
                              const double gix = (ix == 0) ? -1.0 : 1.0;

                              const double dix = gix*biy;
                              const double diy = bix*giy;

                              int ii_loc = ix + 2*iy;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              double val = 0.0;
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
   dbg();
   const int nel_ho = fes_ho.GetNE();
   static constexpr int nv = 8;
   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = ORDER + 1;
   static constexpr int ndof_per_el = nd1d*nd1d*nd1d;
   static constexpr int nnz_per_row = 27;
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
   MFEM_FORALL_3D(iel_ho, nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
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
               double Q_[(ddm2 + 1)*nv];
               double local_mat_[sz_local_mat];

               DeviceTensor<4> Q(Q_, ddm2 + 1, 2, 2, 2);
               DeviceTensor<2> local_mat(local_mat_, nv, nv);

               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

               double vx[8], vy[8], vz[8];
               LORVertexCoordinates3D<ORDER>(X, iel_ho, kx, ky, kz, vx, vy, vz);

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

                        double J_[3*3];
                        DeviceTensor<2> J(J_, 3, 3);

                        Jacobian3D(x, y, z, vx, vy, vz, J);

                        const double detJ = Det3D(J);
                        const double w_detJ = w/detJ;

                        // adj(J)
                        double A_[3*3];
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

               MFEM_UNROLL(2)
               for (int iqz=0; iqz<2; ++iqz)
               {
                  MFEM_UNROLL(2)
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     MFEM_UNROLL(2)
                     for (int iqx=0; iqx<2; ++iqx)
                     {
                        const double J11 = Q(0,iqz,iqy,iqx);
                        const double J21 = Q(1,iqz,iqy,iqx);
                        const double J31 = Q(2,iqz,iqy,iqx);
                        const double J12 = J21;
                        const double J22 = Q(3,iqz,iqy,iqx);
                        const double J32 = Q(4,iqz,iqy,iqx);
                        const double J13 = J31;
                        const double J23 = J32;
                        const double J33 = Q(5,iqz,iqy,iqx);
                        const double wdetJ = Q(6,iqz,iqy,iqx);

                        MFEM_UNROLL(2)
                        for (int jz=0; jz<2; ++jz)
                        {
                           const double bjz = (jz == iqz) ? 1.0 : 0.0;
                           const double gjz = (jz == 0) ? -1.0 : 1.0;
                           MFEM_UNROLL(2)
                           for (int jy=0; jy<2; ++jy)
                           {
                              const double bjy = (jy == iqy) ? 1.0 : 0.0;
                              const double gjy = (jy == 0) ? -1.0 : 1.0;
                              MFEM_UNROLL(2)
                              for (int jx=0; jx<2; ++jx)
                              {
                                 const double bjx = (jx == iqx) ? 1.0 : 0.0;
                                 const double gjx = (jx == 0) ? -1.0 : 1.0;

                                 const double djx = gjx*bjy*bjz;
                                 const double djy = bjx*gjy*bjz;
                                 const double djz = bjx*bjy*gjz;

                                 const int jj_loc = jx + 2*jy + 4*jz;
                                 MFEM_UNROLL(2)
                                 for (int iz=0; iz<2; ++iz)
                                 {
                                    const double biz = (iz == iqz) ? 1.0 : 0.0;
                                    const double giz = (iz == 0) ? -1.0 : 1.0;
                                    MFEM_UNROLL(2)
                                    for (int iy=0; iy<2; ++iy)
                                    {
                                       const double biy = (iy == iqy) ? 1.0 : 0.0;
                                       const double giy = (iy == 0) ? -1.0 : 1.0;

                                       MFEM_UNROLL(2)
                                       for (int ix=0; ix<2; ++ix)
                                       {
                                          const double bix = (ix == iqx) ? 1.0 : 0.0;
                                          const double gix = (ix == 0) ? -1.0 : 1.0;

                                          const double dix = gix*biy*biz;
                                          const double diy = bix*giy*biz;
                                          const double diz = bix*biy*giz;

                                          const int ii_loc = ix + 2*iy + 4*iz;

                                          // Only store the lower-triangular part of
                                          // the matrix (by symmetry).
                                          if (jj_loc > ii_loc) { continue; }

                                          double grad_grad = 0.0;
                                          grad_grad += dix*djx*J11;
                                          grad_grad += diy*djx*J12;
                                          grad_grad += diz*djx*J13;

                                          grad_grad += dix*djy*J21;
                                          grad_grad += diy*djy*J22;
                                          grad_grad += diz*djy*J23;

                                          grad_grad += dix*djz*J31;
                                          grad_grad += diy*djz*J32;
                                          grad_grad += diz*djz*J33;

                                          const double basis_basis = wdetJ*bix*biy*biz*bjx*bjy*bjz;
#warning mq/dq indexing
                                          // const double mq = const_mq ? MQ(0,0,0,0) : MQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);
                                          // const double dq = const_dq ? DQ(0,0,0,0) : DQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);
                                          const double mq = MQ(0,0,0,0);
                                          const double dq = DQ(0,0,0,0);

                                          local_mat(ii_loc, jj_loc) += dq*grad_grad + mq*basis_basis;
                                       }
                                    }
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
               //MFEM_UNROLL(8)
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

// Explicit template instantiations
template void BatchedLOR_H1::Assemble2D<1>();
template void BatchedLOR_H1::Assemble2D<2>();
template void BatchedLOR_H1::Assemble2D<3>();
template void BatchedLOR_H1::Assemble2D<4>();
template void BatchedLOR_H1::Assemble2D<5>();
template void BatchedLOR_H1::Assemble2D<6>();
template void BatchedLOR_H1::Assemble2D<7>();
template void BatchedLOR_H1::Assemble2D<8>();

template void BatchedLOR_H1::Assemble3D<1>();
template void BatchedLOR_H1::Assemble3D<2>();
template void BatchedLOR_H1::Assemble3D<3>();
template void BatchedLOR_H1::Assemble3D<4>();
template void BatchedLOR_H1::Assemble3D<5>();
template void BatchedLOR_H1::Assemble3D<6>();
template void BatchedLOR_H1::Assemble3D<7>();
template void BatchedLOR_H1::Assemble3D<8>();

BatchedLOR_H1::BatchedLOR_H1(BilinearForm &a,
                             FiniteElementSpace &fes_ho_,
                             Vector &X_vert_,
                             Vector &sparse_ij_,
                             Array<int> &sparse_mapping_)
   : BatchedLORKernel(fes_ho_, X_vert_, sparse_ij_, sparse_mapping_)
{
   dbg();
   ProjectLORCoefficient<MassIntegrator>(a, c1);
   ProjectLORCoefficient<DiffusionIntegrator>(a, c2);

   //   MassIntegrator *mass = GetIntegrator<MassIntegrator>(a);
   //   DiffusionIntegrator *diffusion = GetIntegrator<DiffusionIntegrator>(a);

   //   if (mass != nullptr)
   //   {
   //      auto *coeff = dynamic_cast<const ConstantCoefficient*>(mass->GetCoefficient());
   //      mass_coeff = coeff ? coeff->constant : 1.0;
   //   }
   //   else
   //   {
   //      mass_coeff = 0.0;
   //   }

   //   if (diffusion != nullptr)
   //   {
   //      auto *coeff = dynamic_cast<const ConstantCoefficient*>
   //                    (diffusion->GetCoefficient());
   //      diffusion_coeff = coeff ? coeff->constant : 1.0;
   //   }
   //   else
   //   {
   //      diffusion_coeff = 0.0;
   //   }
}

} // namespace mfem
