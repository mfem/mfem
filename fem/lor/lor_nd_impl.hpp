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
void BatchedLOR_ND::Assemble2D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 4;
   static constexpr int ne = 4;
   static constexpr int dim = 2;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int ngeom = ddm2 + 1;
   static constexpr int o = ORDER;
   static constexpr int op1 = ORDER + 1;
   static constexpr int ndof_per_el = dim*o*op1;
   static constexpr int nnz_per_row = 7;
   static constexpr int sz_local_mat = ne*ne;

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1)
                   : Reshape(c1.Read(), op1, op1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1)
                   : Reshape(c2.Read(), op1, op1, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*op1, dim, nel_ho);

   auto X = X_vert.Read();

   mfem::forall_2D(nel_ho, ORDER, ORDER, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,ix,iy) stores the jth nonzero in the row of the sparse matrix
      // corresponding to local DOF (ix, iy).
      MFEM_FOREACH_THREAD(iy,y,o)
      {
         MFEM_FOREACH_THREAD(ix,x,op1)
         {
            for (int c=0; c<2; ++c)
            {
               for (int j=0; j<nnz_per_row; ++j)
               {
                  V(j,ix+iy*op1,c,iel_ho) = 0.0;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Loop over the sub-elements
      MFEM_FOREACH_THREAD(ky,y,ORDER)
      {
         MFEM_FOREACH_THREAD(kx,x,ORDER)
         {
            // Compute geometric factors at quadrature points
            real_t Q_[ngeom*nv];
            real_t local_mat_[sz_local_mat];

            DeviceTensor<3> Q(Q_, ngeom, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, ne, ne);

            // local_mat is the local (dense) stiffness matrix
            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            SetupLORQuadData2D<ORDER,SDIM,false,true>(X, iel_ho, kx, ky, Q, true);

            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  const real_t mq = const_mq ? MQ(0,0,0) : MQ(kx+iqx, ky+iqy, iel_ho);
                  const real_t dq = const_dq ? DQ(0,0,0) : DQ(kx+iqx, ky+iqy, iel_ho);

                  // Loop over x,y components. c=0 => x, c=1 => y
                  for (int cj=0; cj<dim; ++cj)
                  {
                     for (int bj=0; bj<2; ++bj)
                     {
                        const real_t curl_j = ((cj == 0) ? 1 : -1)*((bj == 0) ? 1 : -1);
                        const real_t bxj = (cj == 0) ? ((bj == iqy) ? 1 : 0) : 0;
                        const real_t byj = (cj == 1) ? ((bj == iqx) ? 1 : 0) : 0;

                        const real_t jj_loc = bj + 2*cj;

                        for (int ci=0; ci<dim; ++ci)
                        {
                           for (int bi=0; bi<2; ++bi)
                           {
                              const real_t curl_i = ((ci == 0) ? 1 : -1)*((bi == 0) ? 1 : -1);
                              const real_t bxi = (ci == 0) ? ((bi == iqy) ? 1 : 0) : 0;
                              const real_t byi = (ci == 1) ? ((bi == iqx) ? 1 : 0) : 0;

                              const real_t ii_loc = bi + 2*ci;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              real_t val = 0.0;
                              val += bxi*bxj*Q(0,iqy,iqx);
                              val += byi*bxj*Q(1,iqy,iqx);
                              val += bxi*byj*Q(1,iqy,iqx);
                              val += byi*byj*Q(2,iqy,iqx);
                              val *= mq;
                              val += dq*curl_i*curl_j*Q(3,iqy,iqx);

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
            for (int ii_loc=0; ii_loc<ne; ++ii_loc)
            {
               const int ci = ii_loc/2;
               const int bi = ii_loc%2;
               const int ix = (ci == 0) ? 0 : bi;
               const int iy = (ci == 1) ? 0 : bi;

               int ii = (ci == 0) ? kx+ix + (ky+iy)*o : kx+ix + (ky+iy)*op1;

               for (int jj_loc=0; jj_loc<ne; ++jj_loc)
               {
                  const int cj = jj_loc/2;
                  const int bj = jj_loc%2;

                  const int jj_off = (ci == cj) ? (bj - bi + 1) : (3 + bj + (1-bi)*2);

                  // Symmetry
                  const real_t val = (jj_loc <= ii_loc)
                                     ? local_mat(ii_loc, jj_loc)
                                     : local_mat(jj_loc, ii_loc);
                  AtomicAdd(V(jj_off, ii, ci, iel_ho), val);
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row*ndof_per_el);
   sparse_mapping = -1;
   auto map = Reshape(sparse_mapping.HostReadWrite(), nnz_per_row, ndof_per_el);
   for (int ci=0; ci<2; ++ci)
   {
      for (int i1=0; i1<o; ++i1)
      {
         for (int i2=0; i2<op1; ++i2)
         {
            const int ii_el = (ci == 0) ? i1 + i2*o : i2 + i1*op1 + o*op1;
            for (int cj=0; cj<2; ++cj)
            {
               const int j1_begin = (ci == cj) ? i1 : ((i2 > 0) ? i2-1 : i2);
               const int j1_end = (ci == cj) ? i1 : ((i2 < o) ? i2 : i2-1);
               const int j2_begin = (ci == cj) ? ((i2 > 0) ? i2-1 : i2) : i1;
               const int j2_end = (ci == cj) ? ((i2 < o) ? i2+1 : i2) : i1+1;

               for (int j1=j1_begin; j1<=j1_end; ++j1)
               {
                  for (int j2=j2_begin; j2<=j2_end; ++j2)
                  {
                     const int jj_el = (cj == 0) ? j1 + j2*o : j2 + j1*op1 + o*op1;
                     int jj_off = (ci == cj) ? (j2-i2+1) : 3 + (j2-i1) + 2*(j1-i2+1);
                     map(jj_off, ii_el) = jj_el;
                  }
               }
            }
         }
      }
   }
}

template <int ORDER>
void BatchedLOR_ND::Assemble3D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 8; // number of vertices in hexahedron
   static constexpr int ne = 12; // number of edges in hexahedron
   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int ngeom = 2*ddm2; // number of geometric factors stored
   static constexpr int o = ORDER;
   static constexpr int op1 = ORDER + 1;
   static constexpr int ndof_per_el = dim*o*op1*op1;
   static constexpr int nnz_per_row = 33;
   static constexpr int sz_local_mat = ne*ne;

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1, 1)
                   : Reshape(c1.Read(), op1, op1, op1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1, 1)
                   : Reshape(c2.Read(), op1, op1, op1, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*op1*op1, dim, nel_ho);

   auto X = X_vert.Read();

   // Last thread dimension is lowered to avoid "too many resources" error
   mfem::forall_3D(nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
                   [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      MFEM_FOREACH_THREAD(iz,z,o)
      {
         MFEM_FOREACH_THREAD(iy,y,op1)
         {
            MFEM_FOREACH_THREAD(ix,x,op1)
            {
               for (int c=0; c<dim; ++c)
               {
                  for (int j=0; j<nnz_per_row; ++j)
                  {
                     V(j,ix+iy*op1+iz*op1*op1,c,iel_ho) = 0.0;
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Loop over the sub-elements
      MFEM_FOREACH_THREAD(kz,z,ORDER)
      {
         MFEM_FOREACH_THREAD(ky,y,ORDER)
         {
            MFEM_FOREACH_THREAD(kx,x,ORDER)
            {
               // Geometric factors at quadrature points (element vertices)
               real_t Q_[ngeom*nv];
               DeviceTensor<4> Q(Q_, ngeom, 2, 2, 2);

               real_t local_mat_[sz_local_mat];
               DeviceTensor<2> local_mat(local_mat_, ne, ne);
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

               real_t vx[8], vy[8], vz[8];
               LORVertexCoordinates3D<ORDER>(X, iel_ho, kx, ky, kz, vx, vy, vz);

               for (int iqz=0; iqz<2; ++iqz)
               {
                  for (int iqy=0; iqy<2; ++iqy)
                  {
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

                        // w J^T J / det(J)
                        Q(6,iqz,iqy,iqx)  = w_detJ*(J(0,0)*J(0,0)+J(1,0)*J(1,0)+J(2,0)*J(2,0)); // 1,1
                        Q(7,iqz,iqy,iqx)  = w_detJ*(J(0,0)*J(0,1)+J(1,0)*J(1,1)+J(2,0)*J(2,1)); // 2,1
                        Q(8,iqz,iqy,iqx)  = w_detJ*(J(0,0)*J(0,2)+J(1,0)*J(1,2)+J(2,0)*J(2,2)); // 3,1
                        Q(9,iqz,iqy,iqx)  = w_detJ*(J(0,1)*J(0,1)+J(1,1)*J(1,1)+J(2,1)*J(2,1)); // 2,2
                        Q(10,iqz,iqy,iqx) = w_detJ*(J(0,1)*J(0,2)+J(1,1)*J(1,2)+J(2,1)*J(2,2)); // 3,2
                        Q(11,iqz,iqy,iqx) = w_detJ*(J(0,2)*J(0,2)+J(1,2)*J(1,2)+J(2,2)*J(2,2)); // 3,3
                     }
                  }
               }
               for (int iqz=0; iqz<2; ++iqz)
               {
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     for (int iqx=0; iqx<2; ++iqx)
                     {
                        const real_t mq = const_mq ? MQ(0,0,0,0) : MQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);
                        const real_t dq = const_dq ? DQ(0,0,0,0) : DQ(kx+iqx, ky+iqy, kz+iqz, iel_ho);
                        // Loop over x,y,z components. 0 => x, 1 => y, 2 => z
                        for (int cj=0; cj<dim; ++cj)
                        {
                           const real_t jq1 = (cj == 0) ? iqy : ((cj == 1) ? iqz : iqx);
                           const real_t jq2 = (cj == 0) ? iqz : ((cj == 1) ? iqx : iqy);

                           const int jd_0 = cj;
                           const int jd_1 = (cj + 1)%3;
                           const int jd_2 = (cj + 2)%3;

                           for (int bj=0; bj<4; ++bj) // 4 edges in each dim
                           {
                              const int bj1 = bj%2;
                              const int bj2 = bj/2;

                              real_t curl_j[3];
                              curl_j[jd_0] = 0.0;
                              curl_j[jd_1] = ((bj1 == 0) ? jq1 - 1 : -jq1)*((bj2 == 0) ? 1 : -1);
                              curl_j[jd_2] = ((bj2 == 0) ? 1 - jq2 : jq2)*((bj1 == 0) ? 1 : -1);

                              real_t basis_j[3];
                              basis_j[jd_0] = ((bj1 == 0) ? 1 - jq1 : jq1)*((bj2 == 0) ? 1 - jq2 : jq2);
                              basis_j[jd_1] = 0.0;
                              basis_j[jd_2] = 0.0;

                              const int jj_loc = bj + 4*cj;

                              for (int ci=0; ci<dim; ++ci)
                              {
                                 const real_t iq1 = (ci == 0) ? iqy : ((ci == 1) ? iqz : iqx);
                                 const real_t iq2 = (ci == 0) ? iqz : ((ci == 1) ? iqx : iqy);

                                 const int id_0 = ci;
                                 const int id_1 = (ci + 1)%3;
                                 const int id_2 = (ci + 2)%3;

                                 for (int bi=0; bi<4; ++bi)
                                 {
                                    const int bi1 = bi%2;
                                    const int bi2 = bi/2;

                                    real_t curl_i[3];
                                    curl_i[id_0] = 0.0;
                                    curl_i[id_1] = ((bi1 == 0) ? iq1 - 1 : -iq1)*((bi2 == 0) ? 1 : -1);
                                    curl_i[id_2] = ((bi2 == 0) ? 1 - iq2 : iq2)*((bi1 == 0) ? 1 : -1);

                                    real_t basis_i[3];
                                    basis_i[id_0] = ((bi1 == 0) ? 1 - iq1 : iq1)*((bi2 == 0) ? 1 - iq2 : iq2);
                                    basis_i[id_1] = 0.0;
                                    basis_i[id_2] = 0.0;

                                    const int ii_loc = bi + 4*ci;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    real_t curl_curl = 0.0;
                                    curl_curl += Q(6,iqz,iqy,iqx)*curl_i[0]*curl_j[0];
                                    curl_curl += Q(7,iqz,iqy,iqx)*(curl_i[0]*curl_j[1] + curl_i[1]*curl_j[0]);
                                    curl_curl += Q(8,iqz,iqy,iqx)*(curl_i[0]*curl_j[2] + curl_i[2]*curl_j[0]);
                                    curl_curl += Q(9,iqz,iqy,iqx)*curl_i[1]*curl_j[1];
                                    curl_curl += Q(10,iqz,iqy,iqx)*(curl_i[1]*curl_j[2] + curl_i[2]*curl_j[1]);
                                    curl_curl += Q(11,iqz,iqy,iqx)*curl_i[2]*curl_j[2];

                                    real_t basis_basis = 0.0;
                                    basis_basis += Q(0,iqz,iqy,iqx)*basis_i[0]*basis_j[0];
                                    basis_basis += Q(1,iqz,iqy,iqx)*(basis_i[0]*basis_j[1] + basis_i[1]*basis_j[0]);
                                    basis_basis += Q(2,iqz,iqy,iqx)*(basis_i[0]*basis_j[2] + basis_i[2]*basis_j[0]);
                                    basis_basis += Q(3,iqz,iqy,iqx)*basis_i[1]*basis_j[1];
                                    basis_basis += Q(4,iqz,iqy,iqx)*(basis_i[1]*basis_j[2] + basis_i[2]*basis_j[1]);
                                    basis_basis += Q(5,iqz,iqy,iqx)*basis_i[2]*basis_j[2];

                                    const real_t val = dq*curl_curl + mq*basis_basis;

                                    local_mat(ii_loc, jj_loc) += val;
                                 }
                              }
                           }
                        }
                     }
                  }
               }
               // Assemble the local matrix into the macro-element sparse matrix
               // The nonzeros of the macro-element sparse matrix are ordered as
               // follows:
               //
               // The axes are ordered relative to the direction of the basis
               // vector, e.g. for x-vectors, the axes are (x,y,z), for
               // y-vectors the axes are (y,z,x), and for z-vectors the axes are
               // (z,x,y).
               //
               // The nonzeros are then given in "rotated lexicographic"
               // ordering, according to these axes.
               for (int ii_loc=0; ii_loc<ne; ++ii_loc)
               {
                  const int ci = ii_loc/4;
                  const int bi = ii_loc%4;

                  const int id0 = ci;
                  const int id1 = (ci+1)%3;
                  const int id2 = (ci+2)%3;

                  const int i0 = 0;
                  const int i1 = bi%2;
                  const int i2 = bi/2;

                  int ii_lex[3];
                  ii_lex[id0] = i0;
                  ii_lex[id1] = i1;
                  ii_lex[id2] = i2;

                  const int nx = (ci == 0) ? o : op1;
                  const int ny = (ci == 1) ? o : op1;

                  const int ii = kx+ii_lex[0] + (ky+ii_lex[1])*nx + (kz+ii_lex[2])*nx*ny;

                  for (int jj_loc=0; jj_loc<ne; ++jj_loc)
                  {
                     const int cj = jj_loc/4;
                     // add 3 to take modulus (rather than remainder) when
                     // (cj - ci) is negative
                     const int cj_rel = (3 + cj - ci)%3;

                     const int bj = jj_loc%4;

                     const int jd0 = cj_rel;
                     const int jd1 = (cj_rel+1)%3;
                     const int jd2 = (cj_rel+2)%3;

                     int jj_rel[3];
                     jj_rel[jd0] = 0;
                     jj_rel[jd1] = bj%2;
                     jj_rel[jd2] = bj/2;

                     const int d0 = jj_rel[0] - i0;
                     const int d1 = 1 + jj_rel[1] - i1;
                     const int d2 = 1 + jj_rel[2] - i2;
                     int jj_off;
                     if (cj_rel == 0) { jj_off = d1 + 3*d2; }
                     else if (cj_rel == 1) { jj_off = 9 + d0 + 2*d1 + 4*d2; }
                     else /* if (cj_rel == 2) */ { jj_off = 21 + d0 + 2*d1 + 6*d2; }

                     // Symmetry
                     const real_t val = (jj_loc <= ii_loc)
                                        ? local_mat(ii_loc, jj_loc)
                                        : local_mat(jj_loc, ii_loc);
                     AtomicAdd(V(jj_off, ii, ci, iel_ho), val);
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row*ndof_per_el);
   sparse_mapping = -1;
   auto map = Reshape(sparse_mapping.HostReadWrite(), nnz_per_row, ndof_per_el);
   for (int ci=0; ci<dim; ++ci)
   {
      const int i_off = ci*o*op1*op1;
      const int id0 = ci;
      const int id1 = (ci+1)%3;
      const int id2 = (ci+2)%3;

      const int nxi = (ci == 0) ? o : op1;
      const int nyi = (ci == 1) ? o : op1;

      for (int i0=0; i0<o; ++i0)
      {
         for (int i1=0; i1<op1; ++i1)
         {
            for (int i2=0; i2<op1; ++i2)
            {
               int ii_lex[3];
               ii_lex[id0] = i0;
               ii_lex[id1] = i1;
               ii_lex[id2] = i2;
               const int ii_el = i_off + ii_lex[0] + ii_lex[1]*nxi + ii_lex[2]*nxi*nyi;

               for (int cj_rel=0; cj_rel<dim; ++cj_rel)
               {
                  const int cj = (ci + cj_rel) % 3;
                  const int j_off = cj*o*op1*op1;

                  const int nxj = (cj == 0) ? o : op1;
                  const int nyj = (cj == 1) ? o : op1;

                  const int j0_begin = i0;
                  const int j0_end = (cj_rel == 0) ? i0 : i0 + 1;
                  const int j1_begin = (i1 > 0) ? i1-1 : i1;
                  const int j1_end = (cj_rel == 1)
                                     ? ((i1 < o) ? i1 : i1-1)
                                     : ((i1 < o) ? i1+1 : i1);
                  const int j2_begin = (i2 > 0) ? i2-1 : i2;
                  const int j2_end = (cj_rel == 2)
                                     ? ((i2 < o) ? i2 : i2-1)
                                     : ((i2 < o) ? i2+1 : i2);

                  for (int j0=j0_begin; j0<=j0_end; ++j0)
                  {
                     const int d0 = j0 - i0;
                     for (int j1=j1_begin; j1<=j1_end; ++j1)
                     {
                        const int d1 = j1 - i1 + 1;
                        for (int j2=j2_begin; j2<=j2_end; ++j2)
                        {
                           const int d2 = j2 - i2 + 1;
                           int jj_lex[3];
                           jj_lex[id0] = j0;
                           jj_lex[id1] = j1;
                           jj_lex[id2] = j2;
                           const int jj_el = j_off + jj_lex[0] + jj_lex[1]*nxj + jj_lex[2]*nxj*nyj;
                           int jj_off;
                           if (cj_rel == 0) { jj_off = d1 + 3*d2; }
                           else if (cj_rel == 1) { jj_off = 9 + d0 + 2*d1 + 4*d2; }
                           else /* if (cj_rel == 2) */ { jj_off = 21 + d0 + 2*d1 + 6*d2; }
                           map(jj_off, ii_el) = jj_el;
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

} // namespace mfem
