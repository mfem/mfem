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
#include "../linalg/dtensor.hpp"

namespace mfem
{

template <int order>
void Assemble2DBatchedLOR(Mesh &mesh_lor,
                          Mesh &mesh_ho,
                          FiniteElementSpace &fes_ho,
                          SparseMatrix &A_mat)
{
   int nel_ho = mesh_ho.GetNE();
   int nel_lor = mesh_lor.GetNE();
   int dim = mesh_ho.Dimension();

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
   int nq = ir.Size();

   // Use GeometricFactors to compute the Jacobian matrices.
   // Since the mesh is linear, we can also easily compute the Jacobian matrices
   // using the mesh vertices, see the commented-out code below.
   //
   // Maybe this "more specific" version could be faster than using the generic
   // GeometricFactors?
   const GeometricFactors *geom
      = mesh_lor.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const Vector &J_data = geom->J;

   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   Array<double> invJ_data(nel_ho*pow(order,dim)*nq*(dim*(dim+1))/2);
   auto invJ = Reshape(invJ_data.Write(), (dim*(dim+1))/2, nq, pow(order,dim),
                       nel_ho);
   auto J = Reshape(geom->J.Read(), nq, dim, dim, nel_lor);

   // Array<double> J_data(nq*dim*dim*nel_lor);
   // auto J = Reshape(J_data.Write(), nq, dim, dim, nel_lor);
   constexpr double btab[4] = {0.0,1.0,1.0,0.0};

   for (int iel_lor=0; iel_lor<mesh_lor.GetNE(); ++iel_lor)
   {
      // Array<int> v;
      // mesh_lor.GetElementVertices(iel_lor, v);
      // double vx0 = mesh_lor.GetVertex(v[0])[0];
      // double vx1 = mesh_lor.GetVertex(v[1])[0];
      // double vx2 = mesh_lor.GetVertex(v[2])[0];
      // double vx3 = mesh_lor.GetVertex(v[3])[0];

      // double vy0 = mesh_lor.GetVertex(v[0])[1];
      // double vy1 = mesh_lor.GetVertex(v[1])[1];
      // double vy2 = mesh_lor.GetVertex(v[2])[1];
      // double vy3 = mesh_lor.GetVertex(v[3])[1];

      int iel_ho = cf_tr.embeddings[iel_lor].parent;
      int iref = cf_tr.embeddings[iel_lor].matrix;
      for (int iq=0; iq<nq; ++iq)
      {
         // const double x = ir[iq].x;
         // const double y = ir[iq].y;
         // const double J11 = (-1 + y)*vx0 + vx1 - y*(vx1 - vx2 + vx3);
         // const double J12 = (-1 + x)*vx0 + vx3 - x*(vx1 - vx2 + vx3);
         // const double J21 = (-1 + y)*vy0 + vy1 - y*(vy1 - vy2 + vy3);
         // const double J22 = (-1 + x)*vy0 + vy3 - x*(vy1 - vy2 + vy3);

         // J(iq,0,0,iel_lor) = J11;
         // J(iq,1,0,iel_lor) = J21;
         // J(iq,0,1,iel_lor) = J12;
         // J(iq,1,1,iel_lor) = J22;

         const double J11 = J(iq,0,0,iel_lor);
         const double J21 = J(iq,1,0,iel_lor);
         const double J12 = J(iq,0,1,iel_lor);
         const double J22 = J(iq,1,1,iel_lor);

         const double wq = ir[iq].weight;
         const double w_detJ = wq/((J11*J22)-(J21*J12));

         invJ(0,iq,iref,iel_ho) =  w_detJ * (J12*J12 + J22*J22); // 1,1
         invJ(1,iq,iref,iel_ho) = -w_detJ * (J12*J11 + J22*J21); // 1,2
         invJ(2,iq,iref,iel_ho) =  w_detJ * (J11*J11 + J21*J21); // 2,2
      }
   }

   // These gradient arrays give the value of the derivative of bilinear
   // basis functions at vertices of four adjacent elements:
   // _______________
   // |*    *|*    *|
   // |      |      |
   // |*____*|*____*|
   // |*    *^*    *|
   // |      | \----+-- The basis function is associated with this vertex
   // |*____*|*____*|
   //
   // The (i,j) indices are offsets from the lower-left corner of this 2x2
   // grid, and give the derivatives of the "hat function" at the central
   // vertex evaluated at the locations indicated by the "*" (i.e. at the
   // vertices, evaluating from within each of these elements.

   // DenseMatrix grad_x({
   //    {0,  1,  1, 0},
   //    {0,  1,  1, 0},
   //    {0, -1, -1, 0},
   //    {0, -1, -1, 0}
   // });
   // DenseMatrix grad_y({
   //    {0, 0,  0,  0},
   //    {1, 1, -1, -1},
   //    {1, 1, -1, -1},
   //    {0, 0,  0,  0}
   // });

   // Rather than look up values in the tables, we can also compute them
   // using a very simple strategy.
   //
   // The one dimensional coordinates will be like this:
   //
   // 0-----1*2-----3
   //
   // where * indicates the central vertex. The values of the basis function
   // at these vertices are vals = (0, 1, 1, 0) in order. The values of the
   // gradient of the basis function are derivs = (1, 1, -1, -1) in order. We
   // can just compute the gradient at vertex (ix, iy) as
   //
   // d/dx = derivs(ix)*vals(iy), d/dy = vals(ix)*derivs(iy)

   Array<int> dofs;
   const Array<int> &lex_map = dynamic_cast<const NodalFiniteElement&>
                               (*fes_ho.GetFE(0)).GetLexicographicOrdering();

   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      fes_ho.GetElementDofs(iel_ho, dofs);

      for (int iy=0; iy<order+1; ++iy)
      {
         for (int ix=0; ix<order+1; ++ix)
         {
            int ii = dofs[lex_map[ix + iy*(order+1)]];
            for (int xshift=-1; xshift<=1; ++xshift)
            {
               int jx = ix + xshift;
               if (jx < 0 || jx >= order+1) { continue; }
               double kx_begin = std::max(std::max(ix-1,0), jx-1);
               double kx_end = std::min(std::min(ix,order-1), jx) + 1;
               for (int yshift=-1; yshift<=1; ++yshift)
               {
                  int jy = iy + yshift;
                  if (jy < 0 || jy >= order+1) { continue; }
                  double ky_begin = std::max(std::max(iy-1,0), jy-1);
                  double ky_end = std::min(std::min(iy,order-1), jy) + 1;

                  int jj = dofs[lex_map[jx + jy*(order+1)]];

                  double val = 0.0;
                  for (int ky=ky_begin; ky<ky_end; ++ky)
                  {
                     for (int kx=kx_begin; kx<kx_end; ++kx)
                     {
                        int k = kx + ky*order;
                        for (int iqy=0; iqy<2; ++iqy)
                        {
                           for (int iqx=0; iqx<2; ++iqx)
                           {
                              int offset_x_i = (kx-ix+1)*2 + iqx;
                              int offset_y_i = (ky-iy+1)*2 + iqy;

                              int offset_x_j = (kx-jx+1)*2 + iqx;
                              int offset_y_j = (ky-jy+1)*2 + iqy;

                              int iq = iqx + iqy*2;

                              double a = btab[offset_y_i]*(offset_x_i < 2 ? 1.0 : -1.0);
                              double b = btab[offset_x_i]*(offset_y_i < 2 ? 1.0 : -1.0);

                              double c = btab[offset_y_j]*(offset_x_j < 2 ? 1.0 : -1.0);
                              double d = btab[offset_x_j]*(offset_y_j < 2 ? 1.0 : -1.0);

                              val += a*c*invJ(0, iq, k, iel_ho);
                              val += (b*c + a*d)*invJ(1, iq, k, iel_ho);
                              val += b*d*invJ(2, iq, k, iel_ho);
                           }
                        }
                     }
                  }
                  A_mat.Add(ii, jj, val);
               }
            }
         }
      }
   }
}

template <int order>
void Assemble3DBatchedLOR(Mesh &mesh_lor,
                          Mesh &mesh_ho,
                          FiniteElementSpace &fes_ho,
                          SparseMatrix &A_mat)
{
   int nel_ho = mesh_ho.GetNE();
   int nel_lor = mesh_lor.GetNE();
   int dim = mesh_ho.Dimension();

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
   int nq = ir.Size();

   const GeometricFactors *geom
      = mesh_lor.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const Vector &J_data = geom->J;

   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   Array<double> invJ_data(nel_ho*pow(order,dim)*nq*(dim*(dim+1))/2);
   auto invJ = Reshape(invJ_data.Write(), (dim*(dim+1))/2, nq, pow(order,dim),
                       nel_ho);
   auto J = Reshape(geom->J.Read(), nq, dim, dim, nel_lor);

   constexpr double btab[4] = {0.0,1.0,1.0,0.0};

   for (int iel_lor=0; iel_lor<mesh_lor.GetNE(); ++iel_lor)
   {
      int iel_ho = cf_tr.embeddings[iel_lor].parent;
      int iref = cf_tr.embeddings[iel_lor].matrix;
      for (int iq=0; iq<nq; ++iq)
      {
         const double J11 = J(iq,0,0,iel_lor);
         const double J21 = J(iq,1,0,iel_lor);
         const double J31 = J(iq,2,0,iel_lor);
         const double J12 = J(iq,0,1,iel_lor);
         const double J22 = J(iq,1,1,iel_lor);
         const double J32 = J(iq,2,1,iel_lor);
         const double J13 = J(iq,0,2,iel_lor);
         const double J23 = J(iq,1,2,iel_lor);
         const double J33 = J(iq,2,2,iel_lor);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         const double w_detJ = ir[iq].weight / detJ;
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

         invJ(0,iq,iref,iel_ho) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         invJ(1,iq,iref,iel_ho) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         invJ(2,iq,iref,iel_ho) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         invJ(3,iq,iref,iel_ho) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         invJ(4,iq,iref,iel_ho) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         invJ(5,iq,iref,iel_ho) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   }

   int nd1d = order + 1;
   Array<int> dofs;
   const Array<int> &lex_map = dynamic_cast<const NodalFiniteElement&>
                               (*fes_ho.GetFE(0)).GetLexicographicOrdering();
   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      fes_ho.GetElementDofs(iel_ho, dofs);

      for (int iz=0; iz<nd1d; ++iz)
         for (int iy=0; iy<nd1d; ++iy)
            for (int ix=0; ix<nd1d; ++ix)
            {
               int ii_local = ix + iy*nd1d + iz*nd1d*nd1d;
               int ii = dofs[lex_map[ii_local]];
               for (int xshift=-1; xshift<=1; ++xshift)
               {
                  int jx = ix + xshift;
                  if (jx < 0 || jx >= nd1d) { continue; }
                  double kx_begin = std::max(std::max(ix-1,0), jx-1);
                  double kx_end = std::min(std::min(ix,order-1), jx) + 1;
                  for (int yshift=-1; yshift<=1; ++yshift)
                  {
                     int jy = iy + yshift;
                     if (jy < 0 || jy >= nd1d) { continue; }
                     double ky_begin = std::max(std::max(iy-1,0), jy-1);
                     double ky_end = std::min(std::min(iy,order-1), jy) + 1;

                     for (int zshift=-1; zshift<=1; ++zshift)
                     {
                        int jz = iz + zshift;
                        if (jz < 0 || jz >= nd1d) { continue; }
                        double kz_begin = std::max(std::max(iz-1,0), jz-1);
                        double kz_end = std::min(std::min(iz,order-1), jz) + 1;

                        int jj_local = jx + jy*nd1d + jz*nd1d*nd1d;
                        int jj = dofs[lex_map[jj_local]];

                        double val = 0.0;
                        for (int kz=kz_begin; kz<kz_end; ++kz)
                           for (int ky=ky_begin; ky<ky_end; ++ky)
                              for (int kx=kx_begin; kx<kx_end; ++kx)
                              {
                                 int k = kx + ky*order + kz*order*order;
                                 for (int iqz=0; iqz<2; ++iqz)
                                    for (int iqy=0; iqy<2; ++iqy)
                                       for (int iqx=0; iqx<2; ++iqx)
                                       {
                                          int offset_x_i = (kx-ix+1)*2 + iqx;
                                          int offset_y_i = (ky-iy+1)*2 + iqy;
                                          int offset_z_i = (kz-iz+1)*2 + iqz;

                                          int offset_x_j = (kx-jx+1)*2 + iqx;
                                          int offset_y_j = (ky-jy+1)*2 + iqy;
                                          int offset_z_j = (kz-jz+1)*2 + iqz;

                                          double gix = btab[offset_y_i]*btab[offset_z_i]*(offset_x_i < 2 ? 1.0 : -1.0);
                                          double giy = btab[offset_x_i]*btab[offset_z_i]*(offset_y_i < 2 ? 1.0 : -1.0);
                                          double giz = btab[offset_x_i]*btab[offset_y_i]*(offset_z_i < 2 ? 1.0 : -1.0);

                                          double gjx = btab[offset_y_j]*btab[offset_z_j]*(offset_x_j < 2 ? 1.0 : -1.0);
                                          double gjy = btab[offset_x_j]*btab[offset_z_j]*(offset_y_j < 2 ? 1.0 : -1.0);
                                          double gjz = btab[offset_x_j]*btab[offset_y_j]*(offset_z_j < 2 ? 1.0 : -1.0);

                                          int iq = iqx + iqy*2 + iqz*4;

                                          val += gix*gjx*invJ(0, iq, k, iel_ho);
                                          val += giy*gjx*invJ(1, iq, k, iel_ho);
                                          val += gix*gjy*invJ(1, iq, k, iel_ho);
                                          val += giz*gjx*invJ(2, iq, k, iel_ho);
                                          val += gix*gjz*invJ(2, iq, k, iel_ho);
                                          val += giy*gjy*invJ(3, iq, k, iel_ho);
                                          val += giz*gjy*invJ(4, iq, k, iel_ho);
                                          val += giy*gjz*invJ(4, iq, k, iel_ho);
                                          val += giz*gjz*invJ(5, iq, k, iel_ho);
                                       }
                              }
                        A_mat.Add(ii, jj, val);
                     }
                  }
               }
            }
   }
}

void AssembleBatchedLOR(BilinearForm &form_lor, FiniteElementSpace &fes_ho,
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
   SparseMatrix *A_mat = new SparseMatrix(ndofs, ndofs);

   if (dim == 2)
   {
      switch (order)
      {
         case 1: Assemble2DBatchedLOR<1>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 2: Assemble2DBatchedLOR<2>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 3: Assemble2DBatchedLOR<3>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 4: Assemble2DBatchedLOR<4>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 5: Assemble2DBatchedLOR<5>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 6: Assemble2DBatchedLOR<6>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 7: Assemble2DBatchedLOR<7>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 8: Assemble2DBatchedLOR<8>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 9: Assemble2DBatchedLOR<9>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 10: Assemble2DBatchedLOR<10>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 11: Assemble2DBatchedLOR<11>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 12: Assemble2DBatchedLOR<12>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 13: Assemble2DBatchedLOR<13>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 14: Assemble2DBatchedLOR<14>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 15: Assemble2DBatchedLOR<15>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 16: Assemble2DBatchedLOR<16>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
      }
   }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Assemble3DBatchedLOR<1>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 2: Assemble3DBatchedLOR<2>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 3: Assemble3DBatchedLOR<3>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 4: Assemble3DBatchedLOR<4>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 5: Assemble3DBatchedLOR<5>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 6: Assemble3DBatchedLOR<6>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 7: Assemble3DBatchedLOR<7>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 8: Assemble3DBatchedLOR<8>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 9: Assemble3DBatchedLOR<9>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 10: Assemble3DBatchedLOR<10>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 11: Assemble3DBatchedLOR<11>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 12: Assemble3DBatchedLOR<12>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 13: Assemble3DBatchedLOR<13>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 14: Assemble3DBatchedLOR<14>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 15: Assemble3DBatchedLOR<15>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 16: Assemble3DBatchedLOR<16>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
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
