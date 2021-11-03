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
#include "../general/forall.hpp"

#define MFEM_DEBUG_COLOR 123
#include "../general/debug.hpp"

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

   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   Array<double> invJ_data(nel_ho*pow(order,dim)*nq*(dim*(dim+1))/2);
   auto invJ = Reshape(invJ_data.Write(), (dim*(dim+1))/2, nq, pow(order,dim),
                       nel_ho);
   const auto J = Reshape(geom->J.Read(), nq, dim, dim, nel_lor);

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
   const int nel_ho = mesh_ho.GetNE();
   const int nel_lor = mesh_lor.GetNE();

   static constexpr int dim = 3;
   static constexpr int nv = 8;
   static constexpr int nd1d = order + 1;
   static constexpr int ndof_per_el = nd1d*nd1d*nd1d;
   static constexpr int nnz_per_row = 27;
   static constexpr int nnz_per_el = nnz_per_row*ndof_per_el;
   static constexpr int sz_grad_A = 3*3*2*2*2*2;
   static constexpr int sz_grad_B = sz_grad_A*2;
   static constexpr int sz_local_mat = 8*8;

   // Set up element to dof mapping (in lexicographic ordering)
   Array<int> el_dof_lex_(ndof_per_el*nel_ho);
   {
      Array<int> dofs;
      const Array<int> &lex_map = dynamic_cast<const NodalFiniteElement&>
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

   Array<double> Q_(nel_ho*pow(order,dim)*nv*(dim*(dim+1))/2);

   // Compute geometric factors at quadrature points
   {
      IntegrationRules irs(0, Quadrature1D::GaussLobatto);
      const IntegrationRule &ir = irs.Get(mesh_lor.GetElementGeometry(0), 1);
      const int nq = ir.Size();

      Array<int> lor2ho_(nel_lor), lor2ref_(nel_lor);
      {
         const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
         for (int iel_lor=0; iel_lor<nel_lor; ++iel_lor)
         {
            lor2ho_[iel_lor] = cf_tr.embeddings[iel_lor].parent;
            lor2ref_[iel_lor] = cf_tr.embeddings[iel_lor].matrix;
         }
      }

      Vector el_vert_(dim*nv*nel_lor);
      for (int iel_lor=0; iel_lor<nel_lor; ++iel_lor)
      {
         Array<int> v;
         mesh_lor.GetElementVertices(iel_lor, v);
         for (int iv=0; iv<nv; ++iv)
         {
            const double *vc = mesh_lor.GetVertex(v[iv]);
            for (int d=0; d<dim; ++d)
            {
               el_vert_[d + iv*dim + iel_lor*nv*dim] = vc[d];
            }
         }
      }

      const auto el_vert = Reshape(el_vert_.Read(), dim, nv, nel_lor);
      const auto lor2ho = lor2ho_.Read();
      const auto lor2ref = lor2ref_.Read();
      auto Q = Reshape(Q_.Write(),(dim*(dim+1))/2,nv,pow(order,dim),nel_ho);

      MFEM_FORALL(iel_lor, nel_lor,
      {
         const int iel_ho = lor2ho[iel_lor];
         const int iref = lor2ref[iel_lor];

         const double *v0 = &el_vert(0, 0, iel_lor);
         const double *v1 = &el_vert(0, 1, iel_lor);
         const double *v2 = &el_vert(0, 2, iel_lor);
         const double *v3 = &el_vert(0, 3, iel_lor);
         const double *v4 = &el_vert(0, 4, iel_lor);
         const double *v5 = &el_vert(0, 5, iel_lor);
         const double *v6 = &el_vert(0, 6, iel_lor);
         const double *v7 = &el_vert(0, 7, iel_lor);

         MFEM_UNROLL(8)
         for (int iq=0; iq<nq; ++iq)
         {
            const int iqx = iq%2;
            const int iqy = (iq/2)%2;
            const int iqz = (iq/2)/2;

            const double x = iqx;
            const double y = iqy;
            const double z = iqz;
            const double w = 1.0/8.0;

            // c: (1-x)(1-y)(1-z)v0[c] + x (1-y)(1-z)v1[c] + x y (1-z)v2[c] + (1-x) y (1-z)v3[c]
            //  + (1-x)(1-y) z   v4[c] + x (1-y) z   v5[c] + x y z    v6[c] + (1-x) y z    v7[c]
            const double J11 =
            -(1-y)*(1-z)*v0[0] + (1-y)*(1-z)*v1[0] + y*(1-z)*v2[0] - y*
            (1-z)*v3[0]
            - (1-y)*z*v4[0] + (1-y)*z*v5[0] + y*z*v6[0] - y*z*v7[0];
            const double J12
            = -(1-x)*(1-z)*v0[0] - x*(1-z)*v1[0] + x*(1-z)*v2[0] + (1-x)*
            (1-z)*v3[0]
            - (1-x)*z*v4[0] - x*z*v5[0] + x*z*v6[0] + (1-x)*z*v7[0];
            const double J13
            = -(1-x)*(1-y)*v0[0] - x*(1-y)*v1[0] - x*y*v2[0] -
            (1-x)*y*v3[0]
            + (1-x)*(1-y)*v4[0] + x*(1-y)*v5[0] + x*y*v6[0] + (1-x)*y*v7[0];

            const double J21
            = -(1-y)*(1-z)*v0[1] + (1-y)*(1-z)*v1[1] + y*(1-z)*v2[1] - y*
            (1-z)*v3[1]
            - (1-y)*z*v4[1] + (1-y)*z*v5[1] + y*z*v6[1] - y*z*v7[1];
            const double J22
            = -(1-x)*(1-z)*v0[1] - x*(1-z)*v1[1] + x*(1-z)*v2[1] + (1-x)*
            (1-z)*v3[1]
            - (1-x)*z*v4[1] - x*z*v5[1] + x*z*v6[1] + (1-x)*z*v7[1];
            const double J23
            = -(1-x)*(1-y)*v0[1] - x*(1-y)*v1[1] - x*y*v2[1] -
            (1-x)*y*v3[1]
            + (1-x)*(1-y)*v4[1] + x*(1-y)*v5[1] + x*y*v6[1] + (1-x)*y*v7[1];

            const double J31
            = -(1-y)*(1-z)*v0[2] + (1-y)*(1-z)*v1[2] + y*(1-z)*v2[2] - y*
            (1-z)*v3[2]
            - (1-y)*z*v4[2] + (1-y)*z*v5[2] + y*z*v6[2] - y*z*v7[2];
            const double J32
            = -(1-x)*(1-z)*v0[2] - x*(1-z)*v1[2] + x*(1-z)*v2[2] + (1-x)*
            (1-z)*v3[2]
            - (1-x)*z*v4[2] - x*z*v5[2] + x*z*v6[2] + (1-x)*z*v7[2];
            const double J33
            = -(1-x)*(1-y)*v0[2] - x*(1-y)*v1[2] - x*y*v2[2] -
            (1-x)*y*v3[2]
            + (1-x)*(1-y)*v4[2] + x*(1-y)*v5[2] + x*y*v6[2] + (1-x)*y*v7[2];

            const double detJ = J11 * (J22 * J33 - J32 * J23) -
            J21 * (J12 * J33 - J32 * J13) +
            J31 * (J12 * J23 - J22 * J13);
            const double w_detJ = w/detJ;
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
            const int iq2 = iqz + 2*iqy + 4*iqx;

            Q(0,iq2,iref,iel_ho) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
            Q(1,iq2,iref,iel_ho) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
            Q(2,iq2,iref,iel_ho) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
            Q(3,iq2,iref,iel_ho) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
            Q(4,iq2,iref,iel_ho) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
            Q(5,iq2,iref,iel_ho) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
         }
      });
   }

   constexpr int GRID = 256;
   const int A_height = A_mat.Height();
   Array<int> col_ptr_grid_arrays(A_height*GRID);
   int *col_ptr_base = col_ptr_grid_arrays.Write();

   const auto I = A_mat.ReadI();
   const auto J = A_mat.ReadJ();
   auto A = A_mat.ReadWriteData();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto Q = Reshape(Q_.Read(),(dim*(dim+1))/2,nv,pow(order,dim),nel_ho);

   MFEM_FORALL_3D_GRID(iel_ho, nel_ho, 1, 1, 1, GRID,
   {
      double V_[nnz_per_el];
      DeviceTensor<2> V(V_, nnz_per_row, ndof_per_el);

      DeviceTensor<1,int> col_ptr(col_ptr_base + A_height*MFEM_BLOCK_ID(x), A_height);

      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      //
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      for (int i=0; i<nnz_per_row; ++i)
      {
         for (int j=0; j<ndof_per_el; ++j)
         {
            V(i,j) = 0.0;
         }
      }

      // Loop over sub-elements
      //MFEM_FOREACH_THREAD(kz,z,order)
      for (int kz=0; kz<order; ++kz)
      {
         //MFEM_FOREACH_THREAD(ky,y,order)
         for (int ky=0; ky<order; ++ky)
         {
            //MFEM_FOREACH_THREAD(kx,x,order)
            for (int kx=0; kx<order; ++kx)
            {
               double grad_A_[sz_grad_A];
               double grad_B_[sz_grad_B];
               double local_mat_[sz_local_mat];
               DeviceTensor<2> local_mat(local_mat_, 8, 8);
               DeviceTensor<6> grad_A(grad_A_, 3, 3, 2, 2, 2, 2);
               DeviceTensor<7> grad_B(grad_B_, 3, 3, 2, 2, 2, 2, 2);

               const int k = kx + ky*order + kz*order*order;
               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i)
               {
                  local_mat[i] = 0.0;
               }
               // Intermediate quantities (see e.g. Mora and Demkowicz for
               // notation).
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
                              int iq = iqz + 2*iqy + 4*iqx;
                              const double biz = (iz == iqz) ? 1.0 : 0.0;
                              const double giz = (iz == 0) ? -1.0 : 1.0;

                              const double bjz = (jz == iqz) ? 1.0 : 0.0;
                              const double gjz = (jz == 0) ? -1.0 : 1.0;

                              const double J11 = Q(0,iq,k,iel_ho);
                              const double J21 = Q(1,iq,k,iel_ho);
                              const double J31 = Q(2,iq,k,iel_ho);
                              const double J12 = J21;
                              const double J22 = Q(3,iq,k,iel_ho);
                              const double J32 = Q(4,iq,k,iel_ho);
                              const double J13 = J31;
                              const double J23 = J32;
                              const double J33 = Q(5,iq,k,iel_ho);

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
                        V(jj_off, ii_el) += local_mat(ii_loc, jj_loc);
                     }
                     else
                     {
                        V(jj_off, ii_el) += local_mat(jj_loc, ii_loc);
                     }
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Place the macro-element sparse matrix into the global sparse matrix.
      for (int ii_el=0; ii_el<ndof_per_el; ++ii_el)
      {
         int ix = ii_el%nd1d;
         int iy = (ii_el/nd1d)%nd1d;
         int iz = ii_el/nd1d/nd1d;
         int ii = el_dof_lex(ii_el, iel_ho);

         // Set column pointer to avoid searching in the row
         for (int j = I[ii], end = I[ii+1]; j < end; j++)
         {
            col_ptr[J[j]] = j;
         }

         int jx_begin = (ix > 0) ? ix - 1 : 0;
         int jx_end = (ix < order) ? ix + 1 : order;

         int jy_begin = (iy > 0) ? iy - 1 : 0;
         int jy_end = (iy < order) ? iy + 1 : order;

         int jz_begin = (iz > 0) ? iz - 1 : 0;
         int jz_end = (iz < order) ? iz + 1 : order;

         for (int jz=jz_begin; jz<=jz_end; ++jz)
         {
            for (int jy=jy_begin; jy<=jy_end; ++jy)
            {
               for (int jx=jx_begin; jx<=jx_end; ++jx)
               {
                  int jj_el = jx + jy*nd1d + jz*nd1d*nd1d;
                  int jj = el_dof_lex(jj_el, iel_ho);
                  int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);

                  const int col_ptr_jj = col_ptr(jj);
                  AtomicAdd(A[col_ptr_jj], V(jj_off, ii_el));
               }
            }
         }
      }
   });
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

   if (dim == 2)
   {
      /*switch (order)
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
      }*/
   }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Assemble3DBatchedLOR<1>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 2: Assemble3DBatchedLOR<2>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 3: Assemble3DBatchedLOR<3>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
         case 4: Assemble3DBatchedLOR<4>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
            /*case 5: Assemble3DBatchedLOR<5>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;
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
            case 16: Assemble3DBatchedLOR<16>(mesh_lor, mesh_ho, fes_ho, *A_mat); break;*/
      }
   }


   auto I_d = A_mat->ReadI();
   auto J_d = A_mat->ReadJ();
   auto A_d = A_mat->ReadWriteData();
   //const auto ess_dofs_d = ess_dofs.Read();
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
