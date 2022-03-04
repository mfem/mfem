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

#include "lor_nd.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

template <int ORDER>
void BatchedLOR_ND::Assemble2D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 4;
   static constexpr int ne = 4;
   static constexpr int dim = 2;
   static constexpr int o = ORDER;
   static constexpr int op1 = ORDER + 1;
   static constexpr int ndof_per_el = 2*o*op1;
   static constexpr int nnz_per_row = 7;
   static constexpr int sz_local_mat = ne*ne;

   const double DQ = curl_curl_coeff;
   const double MQ = mass_coeff;

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*op1, 2, nel_ho);

   auto X = X_vert.Read();

   MFEM_FORALL_2D(iel_ho, nel_ho, ORDER, ORDER, 1,
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,ix,iy) stores the jth nonzero in the row of the sparse matrix
      // corresponding to local DOF (ix, iy).
      for (int c=0; c<2; ++c)
      {
         MFEM_FOREACH_THREAD(iy,y,o)
         {
            MFEM_FOREACH_THREAD(ix,x,op1)
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
            double Q_[nv];
            double local_mat_[sz_local_mat];

            DeviceTensor<3> Q(Q_, 1, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, nv, nv);

            // local_mat is the local (dense) stiffness matrix
            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            const int v0 = kx + op1*ky;
            const int v1 = kx + 1 + op1*ky;
            const int v2 = kx + 1 + op1*(ky + 1);
            const int v3 = kx + op1*(ky + 1);

            const int e0 = dim*(v0 + ndof_per_el*iel_ho);
            const int e1 = dim*(v1 + ndof_per_el*iel_ho);
            const int e2 = dim*(v2 + ndof_per_el*iel_ho);
            const int e3 = dim*(v3 + ndof_per_el*iel_ho);

            // Vertex coordinates
            const double v0x = X[e0 + 0];
            const double v0y = X[e0 + 1];

            const double v1x = X[e1 + 0];
            const double v1y = X[e1 + 1];

            const double v2x = X[e2 + 0];
            const double v2y = X[e2 + 1];

            const double v3x = X[e3 + 0];
            const double v3y = X[e3 + 1];

            for (int iqy=0; iqy<2; ++iqy)
            {
               for (int iqx=0; iqx<2; ++iqx)
               {
                  const double x = iqx;
                  const double y = iqy;
                  const double w = 1.0/4.0;

                  const double J11 = -(1-y)*v0x + (1-y)*v1x + y*v2x - y*v3x;
                  const double J12 = -(1-x)*v0x - x*v1x + x*v2x + (1-x)*v3x;

                  const double J21 = -(1-y)*v0y + (1-y)*v1y + y*v2y - y*v3y;
                  const double J22 = -(1-x)*v0y - x*v1y + x*v2y + (1-x)*v3y;

                  const double detJ = J11*J22 - J21*J12;

                  Q(0,iqy,iqx) = w*detJ;
               }
            }
            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  // Loop over x,y components. c=0 => x, c=1 => y
                  for (int cj=0; cj<2; ++cj)
                  {
                     for (int bj=0; bj<2; ++bj)
                     {
                        const double curl_j = ((cj == 0) ? 1 : -1)*((bj == 0) ? 1 : -1);
                        const double bxj = (cj == 0) ? ((bj == iqy) ? 1 : 0) : 0;
                        const double byj = (cj == 1) ? ((bj == iqx) ? 1 : 0) : 0;

                        const double jj_loc = bj + 2*cj;

                        for (int ci=0; ci<2; ++ci)
                        {
                           for (int bi=0; bi<2; ++bi)
                           {
                              const double curl_i = ((ci == 0) ? 1 : -1)*((bi == 0) ? 1 : -1);
                              const double bxi = (ci == 0) ? ((bi == iqy) ? 1 : 0) : 0;
                              const double byi = (ci == 1) ? ((bi == iqx) ? 1 : 0) : 0;

                              const double ii_loc = bi + 2*ci;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              double val = 0.0;
                              double wdetJ = Q(0,iqy,iqx);
                              val += DQ*curl_i*curl_j*wdetJ;
                              val += MQ*(bxi*bxj + byi*byj)*wdetJ;

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
                  double val = (jj_loc <= ii_loc) ? local_mat(ii_loc, jj_loc) : local_mat(jj_loc,
                                                                                          ii_loc);
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
                     sparse_mapping(jj_off, ii_el) = jj_el;
                  }
               }
            }
         }
      }
   }
}

void BatchedLOR_ND::AssemblyKernel()
{
   Mesh &mesh_ho = *fes_ho.GetMesh();
   const int dim = mesh_ho.Dimension();
   const int order = fes_ho.GetMaxElementOrder();

   if (dim == 2)
   {
      switch (order)
      {
         case 1: Assemble2D<1>(); break;
         case 2: Assemble2D<2>(); break;
         case 3: Assemble2D<3>(); break;
         case 4: Assemble2D<4>(); break;
         case 5: Assemble2D<5>(); break;
         case 6: Assemble2D<6>(); break;
         default: MFEM_ABORT("No kernel.");
      }
   }
   else if (dim == 3)
   {
      MFEM_ABORT("No kernel.");
   }
}

BatchedLOR_ND::BatchedLOR_ND(BilinearForm &a,
                             FiniteElementSpace &fes_ho_,
                             const Array<int> &ess_dofs_)
   : BatchedLORAssembly(a, fes_ho_, ess_dofs_)
{
   VectorFEMassIntegrator *mass = GetIntegrator<VectorFEMassIntegrator>(a);
   if (mass != nullptr)
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>(mass->GetCoefficient());
      mass_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      mass_coeff = 0.0;
   }

   CurlCurlIntegrator *diffusion = GetIntegrator<CurlCurlIntegrator>(a);
   if (diffusion != nullptr)
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>
                    (diffusion->GetCoefficient());
      curl_curl_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      curl_curl_coeff = 0.0;
   }
}

} // namespace mfem
