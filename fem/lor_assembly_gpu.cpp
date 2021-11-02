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
                              FiniteElementSpace &fes_lo,
                              SparseMatrix &A_mat)
{
   const int nel_lo = mesh_lor.GetNE();
   const int nel_ho = mesh_ho.GetNE();
   const int dim = mesh_ho.Dimension();
   const int ddm2 = (dim*(dim+1))/2;

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
   const int nq = ir.Size();

   constexpr int Q1D = 2;
   assert(Q1D==order);

   static bool ini = true;
   if (ini)
   {
      dbg("order:%d nel_lo:%d nel_ho:%d nq:%d ddm2:%d",
          order, nel_lo, nel_ho, nq, ddm2);
   }

   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   static Array<double> invJ_data(nel_ho*pow(order,dim)*nq*ddm2);
   auto invJ = Reshape(invJ_data.Write(), ddm2, Q1D,Q1D,Q1D, pow(order,dim),
                       nel_ho);

   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom =
      mesh_lor.GetGeometricFactors(ir,GeometricFactors::JACOBIANS, mt);

   const auto J = Reshape(geom->J.Read(), Q1D,Q1D,Q1D,3,3,nel_lo);
   const auto W = Reshape(ir.GetWeights().Read(), Q1D,Q1D,Q1D);

   MFEM_FORALL_3D(iel_lor, nel_lo, Q1D, Q1D, Q1D,
   {
      const int e = iel_lor;
      const int iel_ho = cf_tr.embeddings[e].parent;
      const int iref = cf_tr.embeddings[e].matrix;

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J33 = J(qx,qy,qz,2,2,e);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
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
               invJ(0,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
               invJ(1,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
               invJ(2,qx,qy,qz,iref,iel_ho) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
               invJ(3,qx,qy,qz,iref,iel_ho) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
               invJ(4,qx,qy,qz,iref,iel_ho) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
               invJ(5,qx,qy,qz,iref,iel_ho) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
            }
         }
      }
   });

   static constexpr int ndof_per_el = (order+1)*(order+1)*(order+1);
   static constexpr int nnz_per_row = 27;
   static constexpr int nnz_per_el =
      nnz_per_row*ndof_per_el; // <-- pessimsistic bound, doesn't distinguish vertices, edges, faces, interiors
   dbg("nnz_per_el:%d",nnz_per_el);
   Array<double> V(nnz_per_el);
   auto d_V = Reshape(V.ReadWrite(), nnz_per_el);

   const int nd1d = order + 1;
   //Array<int> dofs;
   const Array<int> &lex_map = dynamic_cast<const NodalFiniteElement&>
                               (*fes_ho.GetFE(0)).GetLexicographicOrdering();

   static constexpr int sz_grad_A = 3*3*2*2*2*2;
   static constexpr int sz_grad_B = sz_grad_A*2;
   double grad_A_[sz_grad_A];
   double grad_B_[sz_grad_B];

   auto grad_A = Reshape(grad_A_, 3, 3, 2, 2, 2, 2);
   auto grad_B = Reshape(grad_B_, 3, 3, 2, 2, 2, 2, 2);

   static constexpr int sz_local_mat = 8*8;
   double local_mat_[sz_local_mat];
   auto local_mat = Reshape(local_mat_, 8, 8);

   MFEM_FORALL_3D(iel_ho, nel_ho, Q1D, Q1D, Q1D,
   {
      for (int i=0; i<nnz_per_el; ++i) { V[i] = 0.0; }

      // Loop over sub-elements
      MFEM_FOREACH_THREAD(kz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(ky,y,Q1D)
         {
            MFEM_FOREACH_THREAD(kx,x,Q1D)
            {
               double k = kx + ky*order + kz*order*order;
               for (int i=0; i<sz_local_mat; ++i)
               {
                  local_mat[i] = 0.0;
               }
               for (int i=0; i<sz_grad_A; ++i)
               {
                  grad_A[i] = 0.0;
               }
               for (int i=0; i<sz_grad_B; ++i)
               {
                  grad_B[i] = 0.0;
               }

               MFEM_UNROLL(2)
               for (int iqx=0; iqx<2; ++iqx)
               {
                  MFEM_UNROLL(2)
                  for (int jz=0; jz<2; ++jz)
                  {
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

                              const double J11 = invJ(0,iqz,iqy,iqx,k,iel_ho);
                              const double J21 = invJ(1,iqz,iqy,iqx,k,iel_ho);
                              const double J31 = invJ(2,iqz,iqy,iqx,k,iel_ho);
                              const double J12 = J21;
                              const double J22 = invJ(3,iqz,iqy,iqx,k,iel_ho);
                              const double J32 = invJ(4,iqz,iqy,iqx,k,iel_ho);
                              const double J13 = J31;
                              const double J23 = J32;
                              const double J33 = invJ(5,iqz,iqy,iqx,k,iel_ho);

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
               for (int ii_loc=0; ii_loc<8; ++ii_loc)
               {
                  int ix = ii_loc%2;
                  int iy = (ii_loc/2)%2;
                  int iz = ii_loc/2/2;
                  int ii_el = (ix+kx) + (iy+ky)*nd1d + (iz+kz)*nd1d*nd1d;
                  for (int jj_loc=0; jj_loc<8; ++jj_loc)
                  {
                     int jx = jj_loc%2;
                     int jy = (jj_loc/2)%2;
                     int jz = jj_loc/2/2;
                     int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);

                     if (jj_loc <= ii_loc)
                     {
                        d_V[jj_off + ii_el*nnz_per_row] += local_mat(ii_loc, jj_loc);
                     }
                     else
                     {
                        d_V[jj_off + ii_el*nnz_per_row] += local_mat(jj_loc, ii_loc);
                     }
                  }
               }
            }
         }
      }

      Array<int> dofs;
      fes_ho.GetElementDofs(iel_ho, dofs);
      for (int ii_el=0; ii_el<ndof_per_el; ++ii_el)
      {
         const int ix = ii_el%nd1d;
         const int iy = (ii_el/nd1d)%nd1d;
         const int iz = ii_el/nd1d/nd1d;
         const int ii = dofs[lex_map[ii_el]];

         A_mat.SetColPtr(ii);

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
                  int jj_el = jx + jy*nd1d + jz*nd1d*nd1d;
                  int jj = dofs[lex_map[jj_el]];
                  int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);
                  A_mat._Add_(jj, V[jj_off + ii_el*nnz_per_row]);
               }
            }
         }
         A_mat.ClearColPtr();
      }
   });
   ini = false;
}

void AssembleBatchedLOR_GPU(BilinearForm &form_lor,
                            FiniteElementSpace &fes_ho,
                            FiniteElementSpace &fes_lo,
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
         //case 1: Assemble3DBatchedLOR_GPU<1>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 2: Assemble3DBatchedLOR_GPU<2>(mesh_lor, mesh_ho, fes_ho, fes_lo, *A_mat);
            break;
         //case 3: Assemble3DBatchedLOR_GPU<3>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         //case 4: Assemble3DBatchedLOR_GPU<4>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         default: MFEM_ABORT("Kernel not ready!");
      }
   }

   for (int i : ess_dofs)
   {
      A_mat->EliminateRowCol(i, Operator::DIAG_KEEP);
   }

   A_mat->Finalize();
   A.Reset(A_mat); // A now owns A_mat
}

}
