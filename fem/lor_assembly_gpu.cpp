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

#include "lor_assembly.hpp"
//#include "../linalg/dtensor.hpp"
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

namespace mfem
{

template <int order> static
void Assemble3DBatchedLOR_GPU(Mesh &mesh_lor,
                              Mesh &mesh_ho,
                              FiniteElementSpace &fes_ho,
                              SparseMatrix &A_mat)
{
   const int nel_ho = mesh_ho.GetNE();
   const int nel_lor = mesh_lor.GetNE();

   static bool ini = true;
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

   // Set up element to dof mapping (in lexicographic ordering)
   Array<int> el_dof_lex_(ndof_per_el*nel_ho);
   {
      Array<int> dofs;
      const Array<int> &lex_map =
         dynamic_cast<const NodalFiniteElement&>
         (*fes_ho.GetFE(0)).GetLexicographicOrdering();
      for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
      {
         fes_ho.GetElementDofs(iel_ho, dofs);
         for (int i=0; i<ndof_per_el; ++i)
         {
            el_dof_lex_[i + iel_ho*ndof_per_el] = dofs[lex_map[i]];
         }
      }
   }

   // Compute geometric factors at quadrature points
   Array<double> Q_(nel_ho*pow(order,dim)*nv*ddm2);
   {
      IntegrationRules irs(0, Quadrature1D::GaussLobatto);
      const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
      const int nq = ir.Size();

      if (ini)
      {
         dbg("order:%d nel_lor:%d nel_ho:%d nq:%d ddm2:%d A_mat.Height():%d",
             order, nel_lor, nel_ho, nq, ddm2, A_mat.Height());
      }

      Array<int> lor2ho_(nel_lor), lor2ref_(nel_lor);
      {
         const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
         for (int iel_lor=0; iel_lor<nel_lor; ++iel_lor)
         {
            lor2ho_[iel_lor] = cf_tr.embeddings[iel_lor].parent;
            lor2ref_[iel_lor] = cf_tr.embeddings[iel_lor].matrix;
         }
      }

      const auto lor2ho = lor2ho_.Read();
      const auto lor2ref = lor2ref_.Read();

      const GeometricFactors *geom =
         mesh_lor.GetGeometricFactors(ir,GeometricFactors::JACOBIANS);

      const auto J = Reshape(geom->J.Read(), 2,2,2,3,3,nel_lor);
      const auto W = Reshape(ir.GetWeights().Read(), 2,2,2);

      auto Q = Reshape(Q_.Write(), ddm2, 2,2,2, pow(order,dim), nel_ho);

      MFEM_FORALL_3D(iel_lor, nel_lor, 2, 2, 2,
      {
         const int iel_ho = lor2ho[iel_lor];
         const int iref = lor2ref[iel_lor];

         MFEM_FOREACH_THREAD(qz,z,2)
         {
            MFEM_FOREACH_THREAD(qy,y,2)
            {
               MFEM_FOREACH_THREAD(qx,x,2)
               {
                  const double J11 = J(qx,qy,qz,0,0,iel_lor);
                  const double J21 = J(qx,qy,qz,1,0,iel_lor);
                  const double J31 = J(qx,qy,qz,2,0,iel_lor);
                  const double J12 = J(qx,qy,qz,0,1,iel_lor);
                  const double J22 = J(qx,qy,qz,1,1,iel_lor);
                  const double J32 = J(qx,qy,qz,2,1,iel_lor);
                  const double J13 = J(qx,qy,qz,0,2,iel_lor);
                  const double J23 = J(qx,qy,qz,1,2,iel_lor);
                  const double J33 = J(qx,qy,qz,2,2,iel_lor);
                  const double detJ = J11 * (J22 * J33 - J32 * J23) -
                  J21 * (J12 * J33 - J32 * J13) +
                  J31 * (J12 * J23 - J22 * J13);
                  const double w_detJ = W(qx,qy,qz) / detJ;
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
                  Q(0,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
                  Q(1,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
                  Q(2,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
                  Q(3,qx,qy,qz,iref,iel_ho) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
                  Q(4,qx,qy,qz,iref,iel_ho) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
                  Q(5,qx,qy,qz,iref,iel_ho) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
               }
            }
         }
      });
   }

   constexpr int GRID = 256;
   //const int A_height = A_mat.Height();
   //Array<int> col_ptr_grid_arrays(A_height*GRID);
   //auto col_ptr_base = col_ptr_grid_arrays.Write();

   const auto I = A_mat.ReadI();
   const auto J = A_mat.ReadJ();
   auto A = A_mat.ReadWriteData();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto Q = Reshape(Q_.Read(), ddm2, 2,2,2, pow(order,dim), nel_ho);

   MFEM_FORALL_3D_GRID(iel_ho, nel_ho, order, order, order, GRID,
   {
      // nnz_per_el = nnz_per_row(=27) * (order+1) * (order+1) * (order+1);
      MFEM_SHARED double V_[nnz_per_el];
      DeviceTensor<4> V(V_, nnz_per_row, nd1d, nd1d, nd1d);

      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      //
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      MFEM_FOREACH_THREAD(kz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(ky,y,nd1d)
         {
            MFEM_FOREACH_THREAD(kx,x,nd1d)
            {
               for (int i=0; i<nnz_per_row; ++i)
               {
                  V(i,kx,ky,kz) = 0.0;
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

               double k = kx + ky*order + kz*order*order;
               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<8; ++i)
               {
                  for (int j=0; j<8; ++j)
                  {
                     local_mat(i,j) = 0.0;
                  }
               }
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

                              const double J11 = Q(0,iqz,iqy,iqx,k,iel_ho);
                              const double J21 = Q(1,iqz,iqy,iqx,k,iel_ho);
                              const double J31 = Q(2,iqz,iqy,iqx,k,iel_ho);
                              const double J12 = J21;
                              const double J22 = Q(3,iqz,iqy,iqx,k,iel_ho);
                              const double J32 = Q(4,iqz,iqy,iqx,k,iel_ho);
                              const double J13 = J31;
                              const double J23 = J32;
                              const double J33 = Q(5,iqz,iqy,iqx,k,iel_ho);

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

      /*DeviceTensor<1,int> col_ptr(col_ptr_base + A_height*MFEM_BLOCK_ID(x),
                                  A_height);*/

      const int tidx = MFEM_THREAD_ID(x);
      const int tidy = MFEM_THREAD_ID(y);
      const int tidz = MFEM_THREAD_ID(z);
      if (tidx==0 && tidy==0 && tidz==0)
      {

         // Place the macro-element sparse matrix into the global sparse matrix.
         //MFEM_FOREACH_THREAD(iz,z,nd1d)
         for (int ii_el=0; ii_el<ndof_per_el; ++ii_el)
         {
            /* MFEM_FOREACH_THREAD(iy,y,nd1d)
             {
                MFEM_FOREACH_THREAD(ix,x,nd1d)
                {
                   const int ii_el = ix + nd1d*(iy + nd1d*iz);*/
            const int ix = ii_el%nd1d;
            const int iy = (ii_el/nd1d)%nd1d;
            const int iz = ii_el/nd1d/nd1d;
            const int ii = el_dof_lex(ii_el, iel_ho);

            // Set column pointer to avoid searching in the row
            /*for (int j = I[ii], end = I[ii+1]; j < end; j++)
            {
               col_ptr(J[j]) = j;
            }*/

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

                     const int jj_el = jx + jy*nd1d + jz*nd1d*nd1d;
                     const int jj = el_dof_lex(jj_el, iel_ho);
                     int col_ptr_jj = -1;
                     // Row search to get col_ptr_jj
                     for (int j = I[ii], end = I[ii+1]; j < end; j++)
                     {
                        if (J[j]==jj) { col_ptr_jj = j; break; }
                     }
                     //assert(col_ptr_jj>=0);
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
               //}
               //}
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
   ini = false;
}

void AssembleBatchedLOR_GPU(BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            const Array<int> &ess_dofs, OperatorHandle &A)
{
   MFEM_VERIFY(UsesTensorBasis(fes_ho),
               "Batched LOR assembly requires tensor basis");
   // ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   // const Operator *restrict = fes_ho.GetElementRestriction(ordering);

   Mesh &mesh_lor = *form_lor.FESpace()->GetMesh();
   Mesh &mesh_ho = *fes_ho.GetMesh();
   int dim = mesh_ho.Dimension();
   int order = fes_ho.GetMaxElementOrder();
   int ndofs = fes_ho.GetTrueVSize();

   const Table &elem_dof = form_lor.FESpace()->GetElementToDofTable();
   Table dof_dof;
   // the sparsity pattern is defined from the map: element->dof
   Table dof_elem;
   Transpose(elem_dof, dof_elem, ndofs);
   mfem::Mult(dof_elem, elem_dof, dof_dof);
   dof_dof.SortRows();
   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();
   double *data = Memory<double>(I[ndofs]);
   SparseMatrix *A_mat = new SparseMatrix(I,J,data,ndofs,ndofs,true,true,true);
   *A_mat = 0.0;

   dof_dof.LoseData();

   if (dim == 2) { MFEM_ABORT("Unsuported!"); }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Assemble3DBatchedLOR_GPU<1>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 2: Assemble3DBatchedLOR_GPU<2>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 3: Assemble3DBatchedLOR_GPU<3>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 4: Assemble3DBatchedLOR_GPU<4>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   auto I_d = A_mat->ReadI();
   auto J_d = A_mat->ReadJ();
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

   A.Reset(A_mat); // A now owns A_mat
}

}
