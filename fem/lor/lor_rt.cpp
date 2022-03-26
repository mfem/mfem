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
   static constexpr int nlor_vert_per_el = op1*op1;
   static constexpr int nnz_per_row = 7;
   static constexpr int sz_local_mat = ne*ne;

   const double DQ = div_div_coeff;
   const double MQ = mass_coeff;

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, o*op1, dim, nel_ho);

   auto X = X_vert.Read();

   MFEM_FORALL_2D(iel_ho, nel_ho, ORDER, ORDER, 1,
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
            double Q_[ngeom*nv];
            double local_mat_[sz_local_mat];

            DeviceTensor<3> Q(Q_, ngeom, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, ne, ne);

            // local_mat is the local (dense) stiffness matrix
            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            const int v0 = kx + op1*ky;
            const int v1 = kx + 1 + op1*ky;
            const int v2 = kx + 1 + op1*(ky + 1);
            const int v3 = kx + op1*(ky + 1);

            const int e0 = dim*(v0 + nlor_vert_per_el*iel_ho);
            const int e1 = dim*(v1 + nlor_vert_per_el*iel_ho);
            const int e2 = dim*(v2 + nlor_vert_per_el*iel_ho);
            const int e3 = dim*(v3 + nlor_vert_per_el*iel_ho);

            // Vertex coordinates
            const double v0x = X[e0 + 0];
            const double v0y = X[e0 + 1];

            const double v1x = X[e1 + 0];
            const double v1y = X[e1 + 1];

            const double v2x = X[e2 + 0];
            const double v2y = X[e2 + 1];

            const double v3x = X[e3 + 0];
            const double v3y = X[e3 + 1];

            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  const double x = iqx;
                  const double y = iqy;
                  const double w = 1.0/4.0;

                  const double J11 = -(1-y)*v0x + (1-y)*v1x + y*v2x - y*v3x;
                  const double J12 = -(1-x)*v0x - x*v1x + x*v2x + (1-x)*v3x;

                  const double J21 = -(1-y)*v0y + (1-y)*v1y + y*v2y - y*v3y;
                  const double J22 = -(1-x)*v0y - x*v1y + x*v2y + (1-x)*v3y;

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
   MFEM_ABORT("Not implemented");
}

void BatchedLOR_RT::AssemblyKernel()
{
   const int dim = fes_ho.GetMesh()->Dimension();
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
         case 7: Assemble2D<7>(); break;
         case 8: Assemble2D<8>(); break;
         default: MFEM_ABORT("No kernel order " << order << "!");
      }
   }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Assemble3D<1>(); break;
         case 2: Assemble3D<2>(); break;
         case 3: Assemble3D<3>(); break;
         case 4: Assemble3D<4>(); break;
         case 5: Assemble3D<5>(); break;
         case 6: Assemble3D<6>(); break;
         case 7: Assemble3D<7>(); break;
         case 8: Assemble3D<8>(); break;
         default: MFEM_ABORT("No kernel order " << order << "!");
      }
   }
}

BatchedLOR_RT::BatchedLOR_RT(BilinearForm &a,
                             FiniteElementSpace &fes_ho_,
                             const Array<int> &ess_dofs_)
   : BatchedLORAssembly(a, fes_ho_, ess_dofs_)
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
