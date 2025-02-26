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
void BatchedLOR_RT::Assemble2D()
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

            SetupLORQuadData2D<ORDER,SDIM,true,false>(X, iel_ho, kx, ky, Q, true);

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
                        const real_t bxj = (cj == 0 && bj == iqx) ? 1 : 0;
                        const real_t byj = (cj == 1 && bj == iqy) ? 1 : 0;
                        const real_t div_j = (bj == 0) ? -1 : 1;

                        const real_t jj_loc = bj + 2*cj;

                        for (int ci=0; ci<dim; ++ci)
                        {
                           for (int bi=0; bi<2; ++bi)
                           {
                              const real_t bxi = (ci == 0 && bi == iqx) ? 1 : 0;
                              const real_t byi = (ci == 1 && bi == iqy) ? 1 : 0;
                              const real_t div_i = (bi == 0) ? -1 : 1;

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
                              val += dq*div_j*div_i*Q(3,iqy,iqx);

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
               const int ix = (ci == 0) ? bi : 0;
               const int iy = (ci == 1) ? bi : 0;

               int ii = kx+ix + (ky+iy)*((ci == 0) ? op1 : o);

               for (int jj_loc=0; jj_loc<ne; ++jj_loc)
               {
                  const int cj = jj_loc/2;
                  const int bj = jj_loc%2;

                  const int jj_off = (ci == cj) ? (bj - bi + 1) : (3 + 1-bi + 2*bj);

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
      const int i_off = (ci == 0) ? 0 : o*op1;
      const int id0 = ci;
      const int id1 = (ci+1)%2;

      const int nxi = (ci == 0) ? op1 : o;

      for (int i0=0; i0<op1; ++i0)
      {
         for (int i1=0; i1<o; ++i1)
         {
            int ii_lex[2];
            ii_lex[id0] = i0;
            ii_lex[id1] = i1;
            const int ii_el = i_off + ii_lex[0] + ii_lex[1]*nxi;

            for (int cj_rel=0; cj_rel<2; ++cj_rel)
            {
               const int cj = (ci + cj_rel) % 2;
               const int j_off = (cj == 0) ? 0 : o*op1;
               const int nxj = (cj == 0) ? op1 : o;

               const int j0_begin = (i0 > 0) ? i0-1 : i0;
               const int j0_end = (cj_rel == 0)
                                  ? ((i0 < o) ? i0+1 : i0)
                                  : ((i0 < o) ? i0 : i0-1);
               const int j1_begin = i1;
               const int j1_end = (cj_rel == 0) ? i1 : i1+1;

               for (int j0=j0_begin; j0<=j0_end; ++j0)
               {
                  const int d0 = 1 + j0 - i0;
                  for (int j1=j1_begin; j1<=j1_end; ++j1)
                  {
                     const int d1 = j1 - i1;
                     int jj_lex[2];
                     jj_lex[id0] = j0;
                     jj_lex[id1] = j1;
                     const int jj_el = j_off + jj_lex[0] + jj_lex[1]*nxj;
                     const int jj_off = (cj_rel == 0) ? d0 : 3 + d0 + 2*d1;
                     map(jj_off, ii_el) = jj_el;
                  }
               }
            }
         }
      }
   }
}

template <int ORDER>
void BatchedLOR_RT::Assemble3D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 8; // number of vertices in hexahedron
   static constexpr int nf = 6; // number of faces in hexahedron
   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int ngeom = ddm2 + 1; // number of geometric factors stored
   static constexpr int o = ORDER;
   static constexpr int op1 = ORDER + 1;
   static constexpr int ndof_per_el = dim*o*o*op1;
   static constexpr int nnz_per_row = 11;
   static constexpr int sz_local_mat = nf*nf;

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1, 1)
                   : Reshape(c1.Read(), op1, op1, op1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1, 1)
                   : Reshape(c2.Read(), op1, op1, op1, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*o*op1, dim, nel_ho);

   auto X = X_vert.Read();

   // Last thread dimension is lowered to avoid "too many resources" error
   mfem::forall_3D(nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
                   [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      MFEM_FOREACH_THREAD(iz,z,o)
      {
         MFEM_FOREACH_THREAD(iy,y,o)
         {
            MFEM_FOREACH_THREAD(ix,x,op1)
            {
               for (int c=0; c<dim; ++c)
               {
                  for (int j=0; j<nnz_per_row; ++j)
                  {
                     V(j,ix+iy*op1+iz*o*op1,c,iel_ho) = 0.0;
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
               DeviceTensor<2> local_mat(local_mat_, nf, nf);
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

                        Q(0,iqz,iqy,iqx) = w_detJ*(J(0,0)*J(0,0)+J(1,0)*J(1,0)+J(2,0)*J(2,0)); // 1,1
                        Q(1,iqz,iqy,iqx) = w_detJ*(J(0,1)*J(0,0)+J(1,1)*J(1,0)+J(2,1)*J(2,0)); // 2,1
                        Q(2,iqz,iqy,iqx) = w_detJ*(J(0,2)*J(0,0)+J(1,2)*J(1,0)+J(2,2)*J(2,0)); // 3,1
                        Q(3,iqz,iqy,iqx) = w_detJ*(J(0,1)*J(0,1)+J(1,1)*J(1,1)+J(2,1)*J(2,1)); // 2,2
                        Q(4,iqz,iqy,iqx) = w_detJ*(J(0,2)*J(0,1)+J(1,2)*J(1,1)+J(2,2)*J(2,1)); // 3,2
                        Q(5,iqz,iqy,iqx) = w_detJ*(J(0,2)*J(0,2)+J(1,2)*J(1,2)+J(2,2)*J(2,2)); // 3,3
                        Q(6,iqz,iqy,iqx) = w_detJ;
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
                           const real_t jq0 = (cj == 0) ? iqx : ((cj == 1) ? iqy : iqz);

                           const int jd_0 = cj;
                           const int jd_1 = (cj + 1)%3;
                           const int jd_2 = (cj + 2)%3;

                           for (int bj=0; bj<2; ++bj) // 2 faces in each dim
                           {
                              const real_t div_j = (bj == 0) ? -1 : 1;

                              real_t basis_j[3];
                              basis_j[jd_0] = (bj == jq0) ? 1 : 0;
                              basis_j[jd_1] = 0.0;
                              basis_j[jd_2] = 0.0;

                              const int jj_loc = bj + 2*cj;

                              for (int ci=0; ci<dim; ++ci)
                              {
                                 const real_t iq0 = (ci == 0) ? iqx : ((ci == 1) ? iqy : iqz);

                                 const int id_0 = ci;
                                 const int id_1 = (ci + 1)%3;
                                 const int id_2 = (ci + 2)%3;

                                 for (int bi=0; bi<2; ++bi)
                                 {
                                    const real_t div_i = (bi == 0) ? -1 : 1;

                                    real_t basis_i[3];
                                    basis_i[id_0] = (bi == iq0) ? 1 : 0;
                                    basis_i[id_1] = 0.0;
                                    basis_i[id_2] = 0.0;

                                    const int ii_loc = bi + 2*ci;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    const real_t div_div = Q(6,iqz,iqy,iqx)*div_i*div_j;

                                    real_t basis_basis = 0.0;
                                    basis_basis += Q(0,iqz,iqy,iqx)*basis_i[0]*basis_j[0];
                                    basis_basis += Q(1,iqz,iqy,iqx)*(basis_i[0]*basis_j[1] + basis_i[1]*basis_j[0]);
                                    basis_basis += Q(2,iqz,iqy,iqx)*(basis_i[0]*basis_j[2] + basis_i[2]*basis_j[0]);
                                    basis_basis += Q(3,iqz,iqy,iqx)*basis_i[1]*basis_j[1];
                                    basis_basis += Q(4,iqz,iqy,iqx)*(basis_i[1]*basis_j[2] + basis_i[2]*basis_j[1]);
                                    basis_basis += Q(5,iqz,iqy,iqx)*basis_i[2]*basis_j[2];

                                    const real_t val = dq*div_div + mq*basis_basis;
                                    // const double val = 1.0;

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
               for (int ii_loc=0; ii_loc<nf; ++ii_loc)
               {
                  const int ci = ii_loc/2;
                  const int bi = ii_loc%2;

                  const int id0 = ci;
                  const int id1 = (ci+1)%3;
                  const int id2 = (ci+2)%3;

                  const int i0 = bi;
                  const int i1 = 0;
                  const int i2 = 0;

                  int ii_lex[3];
                  ii_lex[id0] = i0;
                  ii_lex[id1] = i1;
                  ii_lex[id2] = i2;

                  const int nx = (ci == 0) ? op1 : o;
                  const int ny = (ci == 1) ? op1 : o;

                  const int ii = kx+ii_lex[0] + (ky+ii_lex[1])*nx + (kz+ii_lex[2])*nx*ny;

                  for (int jj_loc=0; jj_loc<nf; ++jj_loc)
                  {
                     const int cj = jj_loc/2;
                     // add 3 to take modulus (rather than remainder) when
                     // (cj - ci) is negative
                     const int cj_rel = (3 + cj - ci)%3;

                     const int bj = jj_loc%2;

                     const int jd0 = cj_rel;
                     const int jd1 = (cj_rel+1)%3;
                     const int jd2 = (cj_rel+2)%3;

                     int jj_rel[3];
                     jj_rel[jd0] = bj;
                     jj_rel[jd1] = 0;
                     jj_rel[jd2] = 0;

                     const int d0 = jj_rel[0] - i0 + 1;
                     const int d1 = jj_rel[1] - i1;
                     const int d2 = jj_rel[2] - i2;
                     int jj_off;
                     if (cj_rel == 0) { jj_off = d0; }
                     else if (cj_rel == 1) { jj_off = 3 + d0 + 2*d1; }
                     else /* if (cj_rel == 2) */ { jj_off = 7 + d0 + 2*d2; }

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
      const int i_off = ci*o*o*op1;
      const int id0 = ci;
      const int id1 = (ci+1)%3;
      const int id2 = (ci+2)%3;

      const int nxi = (ci == 0) ? op1 : o;
      const int nyi = (ci == 1) ? op1 : o;

      for (int i0=0; i0<op1; ++i0)
      {
         for (int i1=0; i1<o; ++i1)
         {
            for (int i2=0; i2<o; ++i2)
            {
               int ii_lex[3];
               ii_lex[id0] = i0;
               ii_lex[id1] = i1;
               ii_lex[id2] = i2;
               const int ii_el = i_off + ii_lex[0] + ii_lex[1]*nxi + ii_lex[2]*nxi*nyi;

               for (int cj_rel=0; cj_rel<dim; ++cj_rel)
               {
                  const int cj = (ci + cj_rel) % 3;
                  const int j_off = cj*o*o*op1;

                  const int nxj = (cj == 0) ? op1 : o;
                  const int nyj = (cj == 1) ? op1 : o;

                  const int j0_begin = (i0 > 0) ? i0-1 : i0;
                  const int j0_end = (cj_rel == 0)
                                     ? ((i0 < o) ? i0+1 : i0)
                                     : ((i0 < o) ? i0 : i0-1);
                  const int j1_begin = i1;
                  const int j1_end = (cj_rel == 1) ? i1+1 : i1;
                  const int j2_begin = i2;
                  const int j2_end = (cj_rel == 2) ? i2+1 : i2;

                  for (int j0=j0_begin; j0<=j0_end; ++j0)
                  {
                     const int d0 = 1 + j0 - i0;
                     for (int j1=j1_begin; j1<=j1_end; ++j1)
                     {
                        const int d1 = j1 - i1;
                        for (int j2=j2_begin; j2<=j2_end; ++j2)
                        {
                           const int d2 = j2 - i2;
                           int jj_lex[3];
                           jj_lex[id0] = j0;
                           jj_lex[id1] = j1;
                           jj_lex[id2] = j2;
                           const int jj_el = j_off + jj_lex[0] + jj_lex[1]*nxj + jj_lex[2]*nxj*nyj;
                           int jj_off;
                           if (cj_rel == 0) { jj_off = d0; }
                           else if (cj_rel == 1) { jj_off = 3 + d0 + 2*d1; }
                           else /* if (cj_rel == 2) */ { jj_off = 7 + d0 + 2*d2; }
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
