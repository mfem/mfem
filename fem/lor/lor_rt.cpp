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

#include "lor_rt.hpp"
#include "lor_util.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

template <int ORDER>
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

   const double DQ = div_div_coeff;
   const double MQ = mass_coeff;

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*op1, dim, nel_ho);

   auto X = X_vert.Read();

   MFEM_FORALL_2D(iel_ho, nel_ho, ORDER, ORDER, 1,
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
            double Q_[ngeom*nv];
            double local_mat_[sz_local_mat];

            DeviceTensor<3> Q(Q_, ngeom, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, ne, ne);

            // local_mat is the local (dense) stiffness matrix
            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            double vx[4], vy[4];
            LORVertexCoordinates2D<ORDER>(X, iel_ho, kx, ky, vx, vy);

            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  const double x = iqx;
                  const double y = iqy;
                  const double w = 1.0/4.0;

                  const double J11 = -(1-y)*vx[0] + (1-y)*vx[1] + y*vx[2] - y*vx[3];
                  const double J12 = -(1-x)*vx[0] - x*vx[1] + x*vx[2] + (1-x)*vx[3];

                  const double J21 = -(1-y)*vy[0] + (1-y)*vy[1] + y*vy[2] - y*vy[3];
                  const double J22 = -(1-x)*vy[0] - x*vy[1] + x*vy[2] + (1-x)*vy[3];

                  const double detJ = J11*J22 - J21*J12;
                  const double w_detJ = w/detJ;

                  Q(0,iqy,iqx) = w_detJ * (J11*J11 + J21*J21); // 1,1
                  Q(1,iqy,iqx) = w_detJ * (J11*J12 + J21*J22); // 1,2
                  Q(2,iqy,iqx) = w_detJ * (J12*J12 + J22*J22); // 2,2
                  Q(3,iqy,iqx) = w_detJ;
               }
            }
            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  // Loop over x,y components. c=0 => x, c=1 => y
                  for (int cj=0; cj<dim; ++cj)
                  {
                     for (int bj=0; bj<2; ++bj)
                     {
                        const double bxj = (cj == 0 && bj == iqx) ? 1 : 0;
                        const double byj = (cj == 1 && bj == iqy) ? 1 : 0;
                        const double div_j = (bj == 0) ? -1 : 1;

                        const double jj_loc = bj + 2*cj;

                        for (int ci=0; ci<dim; ++ci)
                        {
                           for (int bi=0; bi<2; ++bi)
                           {
                              const double bxi = (ci == 0 && bi == iqx) ? 1 : 0;
                              const double byi = (ci == 1 && bi == iqy) ? 1 : 0;
                              const double div_i = (bi == 0) ? -1 : 1;

                              const double ii_loc = bi + 2*ci;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              double val = 0.0;
                              val += bxi*bxj*Q(0,iqy,iqx);
                              val += byi*bxj*Q(1,iqy,iqx);
                              val += bxi*byj*Q(1,iqy,iqx);
                              val += byi*byj*Q(2,iqy,iqx);
                              val *= MQ;
                              val += DQ*div_j*div_i*Q(3,iqy,iqx);

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
                  const double val = (jj_loc <= ii_loc)
                                     ? local_mat(ii_loc, jj_loc)
                                     : local_mat(jj_loc, ii_loc);
                  AtomicAdd(V(jj_off, ii, ci, iel_ho), val);
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row, ndof_per_el);
   sparse_mapping = -1;
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
                     sparse_mapping(jj_off, ii_el) = jj_el;
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
   static constexpr int nlor_vert_per_el = op1*op1*op1;
   static constexpr int nnz_per_row = 11;
   static constexpr int sz_local_mat = nf*nf;

   const double DQ = div_div_coeff;
   const double MQ = mass_coeff;

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*o*op1, dim, nel_ho);

   auto X = X_vert.Read();

   // Last thread dimension is lowered to avoid "too many resources" error
   MFEM_FORALL_3D(iel_ho, nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
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
               double Q_[ngeom*nv];
               DeviceTensor<4> Q(Q_, ngeom, 2, 2, 2);

               double local_mat_[sz_local_mat];
               DeviceTensor<2> local_mat(local_mat_, nf, nf);
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

               // TODO: refactor this into host device function that gets
               // the vertices of the mesh and entries of the Jacobian matrix
               const int v0 = kx + op1*(ky + op1*kz);
               const int v1 = kx + 1 + op1*(ky + op1*kz);
               const int v2 = kx + 1 + op1*(ky + 1 + op1*kz);
               const int v3 = kx + op1*(ky + 1 + op1*kz);
               const int v4 = kx + op1*(ky + op1*(kz + 1));
               const int v5 = kx + 1 + op1*(ky + op1*(kz + 1));
               const int v6 = kx + 1 + op1*(ky + 1 + op1*(kz + 1));
               const int v7 = kx + op1*(ky + 1 + op1*(kz + 1));

               const int e0 = dim*(v0 + nlor_vert_per_el*iel_ho);
               const int e1 = dim*(v1 + nlor_vert_per_el*iel_ho);
               const int e2 = dim*(v2 + nlor_vert_per_el*iel_ho);
               const int e3 = dim*(v3 + nlor_vert_per_el*iel_ho);
               const int e4 = dim*(v4 + nlor_vert_per_el*iel_ho);
               const int e5 = dim*(v5 + nlor_vert_per_el*iel_ho);
               const int e6 = dim*(v6 + nlor_vert_per_el*iel_ho);
               const int e7 = dim*(v7 + nlor_vert_per_el*iel_ho);

               const double v0x = X[e0 + 0];
               const double v0y = X[e0 + 1];
               const double v0z = X[e0 + 2];

               const double v1x = X[e1 + 0];
               const double v1y = X[e1 + 1];
               const double v1z = X[e1 + 2];

               const double v2x = X[e2 + 0];
               const double v2y = X[e2 + 1];
               const double v2z = X[e2 + 2];

               const double v3x = X[e3 + 0];
               const double v3y = X[e3 + 1];
               const double v3z = X[e3 + 2];

               const double v4x = X[e4 + 0];
               const double v4y = X[e4 + 1];
               const double v4z = X[e4 + 2];

               const double v5x = X[e5 + 0];
               const double v5y = X[e5 + 1];
               const double v5z = X[e5 + 2];

               const double v6x = X[e6 + 0];
               const double v6y = X[e6 + 1];
               const double v6z = X[e6 + 2];

               const double v7x = X[e7 + 0];
               const double v7y = X[e7 + 1];
               const double v7z = X[e7 + 2];

               for (int iqz=0; iqz<2; ++iqz)
               {
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     for (int iqx=0; iqx<2; ++iqx)
                     {
                        const double x = iqx;
                        const double y = iqy;
                        const double z = iqz;
                        const double w = 1.0/8.0;

                        const double J11 = -(1-y)*(1-z)*v0x
                                           + (1-y)*(1-z)*v1x + y*(1-z)*v2x - y*(1-z)*v3x
                                           - (1-y)*z*v4x + (1-y)*z*v5x + y*z*v6x - y*z*v7x;

                        const double J12 = -(1-x)*(1-z)*v0x
                                           - x*(1-z)*v1x + x*(1-z)*v2x + (1-x)*(1-z)*v3x
                                           - (1-x)*z*v4x - x*z*v5x + x*z*v6x + (1-x)*z*v7x;

                        const double J13 = -(1-x)*(1-y)*v0x - x*(1-y)*v1x
                                           - x*y*v2x - (1-x)*y*v3x + (1-x)*(1-y)*v4x
                                           + x*(1-y)*v5x + x*y*v6x + (1-x)*y*v7x;

                        const double J21 = -(1-y)*(1-z)*v0y + (1-y)*(1-z)*v1y
                                           + y*(1-z)*v2y - y*(1-z)*v3y - (1-y)*z*v4y
                                           + (1-y)*z*v5y + y*z*v6y - y*z*v7y;

                        const double J22 = -(1-x)*(1-z)*v0y - x*(1-z)*v1y
                                           + x*(1-z)*v2y + (1-x)*(1-z)*v3y- (1-x)*z*v4y -
                                           x*z*v5y + x*z*v6y + (1-x)*z*v7y;

                        const double J23 = -(1-x)*(1-y)*v0y - x*(1-y)*v1y
                                           - x*y*v2y - (1-x)*y*v3y + (1-x)*(1-y)*v4y
                                           + x*(1-y)*v5y + x*y*v6y + (1-x)*y*v7y;

                        const double J31 = -(1-y)*(1-z)*v0z + (1-y)*(1-z)*v1z
                                           + y*(1-z)*v2z - y*(1-z)*v3z- (1-y)*z*v4z +
                                           (1-y)*z*v5z + y*z*v6z - y*z*v7z;

                        const double J32 = -(1-x)*(1-z)*v0z - x*(1-z)*v1z
                                           + x*(1-z)*v2z + (1-x)*(1-z)*v3z - (1-x)*z*v4z
                                           - x*z*v5z + x*z*v6z + (1-x)*z*v7z;

                        const double J33 = -(1-x)*(1-y)*v0z - x*(1-y)*v1z
                                           - x*y*v2z - (1-x)*y*v3z + (1-x)*(1-y)*v4z
                                           + x*(1-y)*v5z + x*y*v6z + (1-x)*y*v7z;

                        const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                            J21 * (J12 * J33 - J32 * J13) +
                                            J31 * (J12 * J23 - J22 * J13);
                        const double w_detJ = w/detJ;

                        Q(0,iqz,iqy,iqx) = w_detJ * (J11*J11 + J21*J21 + J31*J31); // 1,1
                        Q(1,iqz,iqy,iqx) = w_detJ * (J12*J11 + J22*J21 + J32*J31); // 2,1
                        Q(2,iqz,iqy,iqx) = w_detJ * (J13*J11 + J23*J21 + J33*J31); // 3,1
                        Q(3,iqz,iqy,iqx) = w_detJ * (J12*J12 + J22*J22 + J32*J32); // 2,2
                        Q(4,iqz,iqy,iqx) = w_detJ * (J13*J12 + J23*J22 + J33*J32); // 3,2
                        Q(5,iqz,iqy,iqx) = w_detJ * (J13*J13 + J23*J23 + J33*J33); // 3,3
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
                        // Loop over x,y,z components. 0 => x, 1 => y, 2 => z
                        for (int cj=0; cj<dim; ++cj)
                        {
                           const double jq0 = (cj == 0) ? iqx : ((cj == 1) ? iqy : iqz);

                           const int jd_0 = cj;
                           const int jd_1 = (cj + 1)%3;
                           const int jd_2 = (cj + 2)%3;

                           for (int bj=0; bj<2; ++bj) // 2 faces in each dim
                           {
                              const double div_j = (bj == 0) ? -1 : 1;

                              double basis_j[3];
                              basis_j[jd_0] = (bj == jq0) ? 1 : 0;
                              basis_j[jd_1] = 0.0;
                              basis_j[jd_2] = 0.0;

                              const int jj_loc = bj + 2*cj;

                              for (int ci=0; ci<dim; ++ci)
                              {
                                 const double iq0 = (ci == 0) ? iqx : ((ci == 1) ? iqy : iqz);

                                 const int id_0 = ci;
                                 const int id_1 = (ci + 1)%3;
                                 const int id_2 = (ci + 2)%3;

                                 for (int bi=0; bi<2; ++bi)
                                 {
                                    const double div_i = (bi == 0) ? -1 : 1;

                                    double basis_i[3];
                                    basis_i[id_0] = (bi == iq0) ? 1 : 0;
                                    basis_i[id_1] = 0.0;
                                    basis_i[id_2] = 0.0;

                                    const int ii_loc = bi + 2*ci;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    const double div_div = Q(6,iqz,iqy,iqx)*div_i*div_j;

                                    double basis_basis = 0.0;
                                    basis_basis += Q(0,iqz,iqy,iqx)*basis_i[0]*basis_j[0];
                                    basis_basis += Q(1,iqz,iqy,iqx)*(basis_i[0]*basis_j[1] + basis_i[1]*basis_j[0]);
                                    basis_basis += Q(2,iqz,iqy,iqx)*(basis_i[0]*basis_j[2] + basis_i[2]*basis_j[0]);
                                    basis_basis += Q(3,iqz,iqy,iqx)*basis_i[1]*basis_j[1];
                                    basis_basis += Q(4,iqz,iqy,iqx)*(basis_i[1]*basis_j[2] + basis_i[2]*basis_j[1]);
                                    basis_basis += Q(5,iqz,iqy,iqx)*basis_i[2]*basis_j[2];

                                    const double val = DQ*div_div + MQ*basis_basis;
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
                     const double val = (jj_loc <= ii_loc)
                                        ? local_mat(ii_loc, jj_loc)
                                        : local_mat(jj_loc, ii_loc);
                     AtomicAdd(V(jj_off, ii, ci, iel_ho), val);
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row, ndof_per_el);
   sparse_mapping = -1;
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
                           sparse_mapping(jj_off, ii_el) = jj_el;
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

// Explicit template instantiations
template void BatchedLOR_RT::Assemble2D<1>();
template void BatchedLOR_RT::Assemble2D<2>();
template void BatchedLOR_RT::Assemble2D<3>();
template void BatchedLOR_RT::Assemble2D<4>();
template void BatchedLOR_RT::Assemble2D<5>();
template void BatchedLOR_RT::Assemble2D<6>();
template void BatchedLOR_RT::Assemble2D<7>();
template void BatchedLOR_RT::Assemble2D<8>();

template void BatchedLOR_RT::Assemble3D<1>();
template void BatchedLOR_RT::Assemble3D<2>();
template void BatchedLOR_RT::Assemble3D<3>();
template void BatchedLOR_RT::Assemble3D<4>();
template void BatchedLOR_RT::Assemble3D<5>();
template void BatchedLOR_RT::Assemble3D<6>();
template void BatchedLOR_RT::Assemble3D<7>();
template void BatchedLOR_RT::Assemble3D<8>();

BatchedLOR_RT::BatchedLOR_RT(BilinearForm &a,
                             FiniteElementSpace &fes_ho_,
                             Vector &X_vert_,
                             Vector &sparse_ij_,
                             DenseMatrix &sparse_mapping_)
   : fes_ho(fes_ho_), X_vert(X_vert_), sparse_ij(sparse_ij_),
     sparse_mapping(sparse_mapping_)
{
   if (VectorFEMassIntegrator *mass = GetIntegrator<VectorFEMassIntegrator>(a))
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>(mass->GetCoefficient());
      mass_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      mass_coeff = 0.0;
   }

   if (DivDivIntegrator *divdiv = GetIntegrator<DivDivIntegrator>(a))
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>
                    (divdiv->GetCoefficient());
      div_div_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      div_div_coeff = 0.0;
   }
}

} // namespace mfem
