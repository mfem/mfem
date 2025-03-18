// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "lor_dg.hpp"

namespace mfem
{

template <int ORDER, int SDIM>
void BatchedLOR_DG::Assemble2D()
{
   const int nel_ho = fes_ho.GetNE();
   //const int p = ORDER;
   const int pp1 = ORDER + 1;
   const int pp2 = ORDER + 2;

   IntegrationRule ir_pp1;
   QuadratureFunctions1D::GaussLobatto(pp1, &ir_pp1);
   IntegrationRule ir_pp2;
   QuadratureFunctions1D::GaussLobatto(pp2, &ir_pp2);
   
   //vectorize integration points
   Vector vec_ir_pp1_x(pp1);
   Vector vec_ir_pp2_x(pp2); 
   for(int i=0; i < pp1; i++){
      vec_ir_pp1_x[i] = ir_pp1[i].x;
   }
   for(int j=0; j < pp2; j++){
      vec_ir_pp2_x[j] = ir_pp2[j].x;
   }
   
   static constexpr int ndof_per_el = pp1*pp1;
   static constexpr int nnz_per_row = 5;
   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1)
                   : Reshape(c1.Read(), pp1, pp1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1)
                   : Reshape(c2.Read(), pp1, pp1, nel_ho);

   const auto w_1d = ir_pp1.GetWeights().Read();
   const auto W = Reshape(ir.GetWeights().Read(), pp1, pp1);
   const auto X = Reshape(X_vert.Read(), 2, pp2, pp2, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, pp1, pp1, nel_ho);

   auto geom = fes_ho.GetMesh()->GetGeometricFactors(
                  ir, GeometricFactors::DETERMINANTS);
   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, nel_ho);

   const auto *d_vec_ir_pp1_x = vec_ir_pp1_x.Read();
   const auto *d_vec_ir_pp2_x = vec_ir_pp2_x.Read();
   const real_t d_kappa = kappa;

   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for (int iy = 0; iy < pp1; ++iy)
      {
         for (int ix = 0; ix < pp1; ++ix)
         {
            const real_t A_ref = (d_vec_ir_pp2_x[ix+1] - d_vec_ir_pp2_x[ix])
                                 * (d_vec_ir_pp2_x[iy+1] - d_vec_ir_pp2_x[iy]);
            // Shoelace formula for area of a quadrilateral
            const real_t A_el = fabs(0.5*(X(0, ix, iy, iel_ho)*X(1, ix+1, iy, iel_ho)
                                          - X(0, ix+1, iy, iel_ho)*X(1, ix, iy, iel_ho)
                                          + X(0, ix+1, iy, iel_ho)*X(1, ix+1, iy+1, iel_ho)
                                          - X(0, ix+1, iy+1, iel_ho)*X(1, ix+1, iy, iel_ho)
                                          + X(0, ix+1, iy+1, iel_ho)*X(1, ix, iy+1, iel_ho)
                                          - X(0, ix, iy+1, iel_ho)*X(1, ix+1, iy+1, iel_ho)
                                          + X(0, ix, iy+1, iel_ho)*X(1, ix, iy, iel_ho)
                                          - X(0, ix, iy, iel_ho)*X(1, ix, iy+1, iel_ho)));
            const real_t mq = const_mq ? MQ(0,0,0) : MQ(ix, iy, iel_ho);
            const real_t dq = const_dq ? DQ(0,0,0) : DQ(ix, iy, iel_ho);
         
            for (int n_idx = 0; n_idx < 2; ++n_idx)
            {
               for (int e_i = 0; e_i < 2; ++e_i)
               {
                  static const int lex_map[] = {4, 2, 1, 3};
                  const int v_idx_lex = e_i + n_idx*2;
                  const int v_idx = lex_map[v_idx_lex];

                  const int i_0 = (n_idx == 0) ? ix + e_i : ix;
                  const int j_0 = (n_idx == 1) ? iy + e_i : iy;

                  const int i_1 = (n_idx == 0) ? ix + e_i : ix + 1;
                  const int j_1 = (n_idx == 1) ? iy + e_i : iy + 1;

                  const bool bdr = (n_idx == 0) ? (i_0 == 0 || i_0 == pp1)
                                   : (j_0 == 0 || j_0 == pp1);

                  const int w_idx = (n_idx == 0) ? iy : ix;
                  const int int_idx = (n_idx == 0) ? i_0 : j_0;
                  const int el_idx = (n_idx == 0) ? j_0 : i_0;
                  const real_t el_1 = d_vec_ir_pp2_x[el_idx+1] - d_vec_ir_pp2_x[el_idx];
                  const real_t el_2 = A_ref / el_1;

                  const real_t x1 = X(0, i_0, j_0, iel_ho);
                  const real_t y1 = X(1, i_0, j_0, iel_ho);
                  const real_t x2 = X(0, i_1, j_1, iel_ho);
                  const real_t y2 = X(1, i_1, j_1, iel_ho);
                  const real_t A_face = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)); 

                  if (bdr)
                  {
                     const real_t h_recip = A_face*A_face/A_el*el_2/el_1;
                     V(v_idx, ix, iy, iel_ho) = -dq * d_kappa * w_1d[w_idx] * h_recip;

                  }
                  else
                  {
                     const int ix2 = (n_idx == 0) ? ix + (e_i == 0 ? -1 : 1) : ix;
                     const int iy2 = (n_idx == 1) ? iy + (e_i == 0 ? -1 : 1) : iy;
                     const real_t A_el_2 =
                        fabs(0.5*(X(0, ix2, iy2, iel_ho)*X(1, ix2+1, iy2, iel_ho)
                                  - X(0, ix2+1, iy2, iel_ho)*X(1, ix2, iy2, iel_ho)
                                  + X(0, ix2+1, iy2, iel_ho)*X(1, ix2+1, iy2+1, iel_ho)
                                  - X(0, ix2+1, iy2+1, iel_ho)*X(1, ix2+1, iy2, iel_ho)
                                  + X(0, ix2+1, iy2+1, iel_ho)*X(1, ix2, iy2+1, iel_ho)
                                  - X(0, ix2, iy2+1, iel_ho)*X(1, ix2+1, iy2+1, iel_ho)
                                  + X(0, ix2, iy2+1, iel_ho)*X(1, ix2, iy2, iel_ho)
                                  - X(0, ix2, iy2, iel_ho)*X(1, ix2, iy2+1, iel_ho)));
                     const real_t A_ref_1 = (n_idx == 0) ? d_vec_ir_pp2_x[i_0+1] - d_vec_ir_pp2_x[i_0] :
                                            d_vec_ir_pp2_x[j_0+1] - d_vec_ir_pp2_x[j_0];
                     const real_t A_ref_2 = (n_idx == 0) ? d_vec_ir_pp2_x[i_0] - d_vec_ir_pp2_x[i_0-1] :
                                            d_vec_ir_pp2_x[j_0] - d_vec_ir_pp2_x[j_0-1];
                     const real_t h = (0.5*A_el + 0.5*A_el_2) / A_face / (0.5 * (A_ref_1 + A_ref_2)); 

                     V(v_idx, ix, iy, iel_ho) = -dq * A_face * w_1d[w_idx] / h / el_1 /
                                                (d_vec_ir_pp1_x[int_idx] - d_vec_ir_pp1_x[int_idx-1]);
                  }
               }
            }
            V(0, ix, iy, iel_ho) = mq * detJ(ix, iy, iel_ho) * W(ix, iy);
            for (int i = 1; i < 5; ++i)
            {
               V(0, ix, iy, iel_ho) -= V(i, ix, iy, iel_ho);
            }
         }
      }
   });
}

template <int ORDER>
void BatchedLOR_DG::Assemble3D()
{
   const int nel_ho = fes_ho.GetNE();
   const int pp1 = ORDER + 1;
   const int pp2 = ORDER + 2;
   IntegrationRule ir_pp1;
   QuadratureFunctions1D::GaussLobatto(pp1, &ir_pp1);
   IntegrationRule ir_pp2;
   QuadratureFunctions1D::GaussLobatto(pp2, &ir_pp2);

   static constexpr int ndof_per_el = pp1*pp1*pp1;
   static constexpr int nnz_per_row = 7;
   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1, 1)
                   : Reshape(c1.Read(), pp1, pp1, pp1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1, 1)
                   : Reshape(c2.Read(), pp1, pp1, pp1, nel_ho);
   const auto w_1d = ir_pp1.GetWeights().Read();
   const auto W = Reshape(ir.GetWeights().Read(), pp1, pp1, pp1);
   const auto X = Reshape(X_vert.Read(), 3, pp2, pp2, pp2, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, pp1, pp1, pp1, nel_ho);

   auto geom = fes_ho.GetMesh()->GetGeometricFactors(
      ir, GeometricFactors::DETERMINANTS);
   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, pp1, nel_ho);
   
   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for(int iz = 0; iz < pp1; ++iz){
         for (int iy = 0; iy < pp1; ++iy){
            for (int ix = 0; ix < pp1; ++ix){
               const real_t mq = const_mq ? MQ(0,0,0,0) : MQ(ix, iy, iz, iel_ho);
               const real_t dq = const_dq ? DQ(0,0,0,0) : DQ(ix, iy, iz, iel_ho);  

               //compute A_el and A_ref
               //For A_el, split element into two tetrahedra, then compute volumes of each of these and add
               Vector T1_B1(3);
               Vector T1_B2(3);
               Vector T1_B3(3);
               Vector T2_B1(3);
               Vector T2_B2(3);
               Vector T2_B3(3);

               T1_B1[0] = X(0, ix, iy, iz, iel_ho) - X(0, ix+1, iy, iz+1, iel_ho); 
               T1_B2[0] = X(0, ix+1, iy, iz, iel_ho) - X(0, ix+1, iy, iz+1, iel_ho); 
               T1_B3[0] = X(0, ix, iy+1, iz, iel_ho) - X(0, ix+1, iy, iz+1, iel_ho); 
               T1_B1[1] = X(1, ix, iy, iz, iel_ho) - X(1, ix+1, iy, iz+1, iel_ho); 
               T1_B2[1] = X(1, ix+1, iy, iz, iel_ho) - X(1, ix+1, iy, iz+1, iel_ho); 
               T1_B3[1] = X(1, ix, iy+1, iz, iel_ho) - X(1, ix+1, iy, iz+1, iel_ho); 
               T1_B1[2] = X(2, ix, iy, iz, iel_ho) - X(2, ix+1, iy, iz+1, iel_ho); 
               T1_B2[2] = X(2, ix+1, iy, iz, iel_ho) - X(2, ix+1, iy, iz+1, iel_ho); 
               T1_B3[2] = X(2, ix, iy+1, iz, iel_ho) - X(2, ix+1, iy, iz+1, iel_ho); 

               T2_B1[0] = X(0, ix, iy, iz+1, iel_ho) - X(0, ix+1, iy, iz, iel_ho); 
               T2_B2[0] = X(0, ix+1, iy+1, iz+1, iel_ho) - X(0, ix, iy, iz, iel_ho); 
               T2_B3[0] = X(0, ix, iy+1, iz+1, iel_ho) - X(0, ix, iy, iz, iel_ho); 
               T2_B1[1] = X(1, ix, iy, iz+1, iel_ho) - X(1, ix+1, iy, iz, iel_ho); 
               T2_B2[1] = X(1, ix+1, iy+1, iz+1, iel_ho) - X(1, ix, iy, iz, iel_ho); 
               T2_B3[1] = X(1, ix, iy+1, iz+1, iel_ho) - X(1, ix, iy, iz, iel_ho); 
               T2_B1[2] = X(2, ix, iy, iz+1, iel_ho) - X(2, ix+1, iy, iz, iel_ho); 
               T2_B2[2] = X(2, ix+1, iy+1, iz+1, iel_ho) - X(2, ix, iy, iz, iel_ho); 
               T2_B3[2] = X(2, ix, iy+1, iz+1, iel_ho) - X(2, ix, iy, iz, iel_ho);
               
               Vector cross1(3);
               Vector cross2(3);

               T1_B2.cross3D(T1_B3, cross1);
               T2_B2.cross3D(T2_B3, cross2);

               real_t V1 = fabs(T1_B1*cross1);
               real_t V2 = fabs(T2_B1*cross2);
               real_t A_el = (V1+V2)/2;

               const real_t A_ref = (ir_pp2[ix+1].x - ir_pp2[ix].x)
                                 * (ir_pp2[iy+1].x - ir_pp2[iy].x)
                                 * (ir_pp2[iz+1].x - ir_pp2[iz].x);
               
               //std::cout << "A_el =  " << A_el << std::endl;
               //std::cout << "A_ref = " << A_ref << std::endl;

               for (int n_idx = 0; n_idx < 3; ++n_idx){
                  for (int e_i = 0; e_i < 2; ++e_i){
                     const int v_idx_lex = e_i + n_idx*2;
                     static const int lex_map[] = {5,3,2,4,1,6};
                     const int v_idx = lex_map[v_idx_lex];
                     const int i_0 = (n_idx == 0) ? ix + e_i : ix;
                     const int j_0 = (n_idx == 1) ? iy + e_i : iy;
                     const int k_0 = (n_idx == 2) ? iz + e_i : iz;

                     const int i_1 = (n_idx == 2 || n_idx == 1) ? i_0 + 1 : i_0;
                     const int j_1 = j_0;
                     const int k_1 = (n_idx == 0) ? k_0 + 1 : k_0;

                     const int i_2 = (n_idx == 0) ? ix + e_i : ix+1;
                     const int j_2 = (n_idx == 1) ? iy + e_i : iy+1;
                     const int k_2 = (n_idx == 2) ? iz + e_i : iz+1;

                     const int i_3 = i_0;
                     const int j_3 = (n_idx == 0 || n_idx == 2) ? j_0 + 1 : j_0;
                     const int k_3 = (n_idx == 1) ? k_0 + 1 : k_0;

                     int w_idx_1 = (n_idx == 0) ? iy : (n_idx == 1) ? iz : ix;
                     int w_idx_2 = (n_idx == 0) ? iz : (n_idx == 1) ? ix : iy;

                     int int_idx = (n_idx == 0) ? i_0 : (n_idx == 1) ? j_0 : k_0;

                     const real_t A_ref_face = (n_idx == 0) ? (ir_pp2[iy+1].x - ir_pp2[iy].x) * (ir_pp2[iz+1].x - ir_pp2[iz].x) :
                     (n_idx == 1) ? (ir_pp2[ix+1].x - ir_pp2[ix].x) * (ir_pp2[iz+1].x - ir_pp2[iz].x) :
                     (ir_pp2[iy+1].x - ir_pp2[iy].x)*(ir_pp2[ix+1].x - ir_pp2[ix].x);

                     const real_t A_ref_perp = A_ref/A_ref_face;

                     const real_t x0 = X(0, i_0, j_0, k_0, iel_ho); 
                     const real_t y0 = X(1, i_0, j_0, k_0, iel_ho); 
                     const real_t z0 = X(2, i_0, j_0, k_0, iel_ho); 
                     const real_t x1 = X(0, i_1, j_1, k_1, iel_ho); 
                     const real_t y1 = X(1, i_1, j_1, k_1, iel_ho); 
                     const real_t z1 = X(2, i_1, j_1, k_1, iel_ho); 
                     const real_t x2 = X(0, i_2, j_2, k_2, iel_ho); 
                     const real_t y2 = X(1, i_2, j_2, k_2, iel_ho); 
                     const real_t z2 = X(2, i_2, j_2, k_2, iel_ho); 
                     const real_t x3 = X(0, i_3, j_3, k_3, iel_ho); 
                     const real_t y3 = X(1, i_3, j_3, k_3, iel_ho); 
                     const real_t z3 = X(2, i_3, j_3, k_3, iel_ho); 

                     const real_t A_face = (n_idx == 0) ? 0.5*fabs((y0*z1 + y1*z2 + y2*z3 + y3*z0 - z0*y1 - z1*y2 - z2*y3 - z3*y0)) :
                     ((n_idx == 1) ? 0.5*fabs((x0*z1 + x1*z2 + x2*z3 + x3*z0 - z0*x1 - z1*x2 - z2*x3 - z3*x0)) : 0.5*fabs((y0*x1 + y1*x2 + y2*x3 + y3*x0 - x0*y1 - x1*y2 - x2*y3 - x3*y0)));
                  
                     bool bdr = true;
                     if (n_idx == 0){bdr = (i_0 == 0 || i_0 == pp1); w_idx_1 = iy; w_idx_2 = iz; int_idx = i_0;}
                     else if(n_idx == 1){bdr = (j_0 == 0 || j_0 == pp1); w_idx_1 = ix; w_idx_2 = iz; int_idx = j_0;}
                     else {bdr = (k_0 == 0 || k_0 == pp1); w_idx_1 = ix; w_idx_2 = iy; int_idx = k_0;}



                     if (bdr){
                        const real_t h_recip = A_face*A_ref_perp/(A_ref_face*A_el);
                        //const real_t h_recip = A_ref_perp / A_ref;
                        V(v_idx, ix, iy, iz, iel_ho) = -dq * kappa * w_1d[w_idx_1] * w_1d[w_idx_2] * A_face* h_recip;
                     }
                     else{
                        const int ix2 = (n_idx == 0) ? ix + (e_i == 0 ? -1 : 1) : ix;
                        const int iy2 = (n_idx == 1) ? iy + (e_i == 0 ? -1 : 1) : iy;
                        const int iz2 = (n_idx == 2) ? iz + (e_i == 0 ? -1 : 1) : iz;

                        //compute A_el_2
                        //For A_el_2, split element into two tetrahedra, then compute volumes of each of these and add
                        Vector adj_T1_B1(3);
                        Vector adj_T1_B2(3);
                        Vector adj_T1_B3(3);
                        Vector adj_T2_B1(3);
                        Vector adj_T2_B2(3);
                        Vector adj_T2_B3(3);

                        adj_T1_B1[0] = X(0, ix2, iy2, iz2, iel_ho) - X(0, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B2[0] = X(0, ix2+1, iy2, iz2, iel_ho) - X(0, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B3[0] = X(0, ix2, iy2+1, iz2, iel_ho) - X(0, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B1[1] = X(1, ix2, iy2, iz2, iel_ho) - X(1, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B2[1] = X(1, ix2+1, iy2, iz2, iel_ho) - X(1, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B3[1] = X(1, ix2, iy2+1, iz2, iel_ho) - X(1, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B1[2] = X(2, ix2, iy2, iz2, iel_ho) - X(2, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B2[2] = X(2, ix2+1, iy2, iz2, iel_ho) - X(2, ix2+1, iy2, iz2+1, iel_ho); 
                        adj_T1_B3[2] = X(2, ix2, iy2+1, iz2, iel_ho) - X(2, ix2+1, iy2, iz2+1, iel_ho); 

                        adj_T2_B1[0] = X(0, ix2, iy2, iz2+1, iel_ho) - X(0, ix2+1, iy2, iz2, iel_ho); 
                        adj_T2_B2[0] = X(0, ix2+1, iy2+1, iz2+1, iel_ho) - X(0, ix2, iy2, iz2, iel_ho); 
                        adj_T2_B3[0] = X(0, ix2, iy2+1, iz2+1, iel_ho) - X(0, ix2, iy2, iz2, iel_ho); 
                        adj_T2_B1[1] = X(1, ix2, iy2, iz2+1, iel_ho) - X(1, ix2+1, iy2, iz2, iel_ho); 
                        adj_T2_B2[1] = X(1, ix2+1, iy2+1, iz2+1, iel_ho) - X(1, ix2, iy2, iz2, iel_ho); 
                        adj_T2_B3[1] = X(1, ix2, iy2+1, iz2+1, iel_ho) - X(1, ix2, iy2, iz2, iel_ho); 
                        adj_T2_B1[2] = X(2, ix2, iy2, iz2+1, iel_ho) - X(2, ix2+1, iy2, iz2, iel_ho); 
                        adj_T2_B2[2] = X(2, ix2+1, iy2+1, iz2+1, iel_ho) - X(2, ix2, iy2, iz2, iel_ho); 
                        adj_T2_B3[2] = X(2, ix2, iy2+1, iz2+1, iel_ho) - X(2, ix2, iy2, iz2, iel_ho);
                        
                        Vector adj_cross1(3);
                        Vector adj_cross2(3);

                        adj_T1_B2.cross3D(adj_T1_B3, adj_cross1);
                        adj_T2_B2.cross3D(adj_T2_B3, adj_cross2);

                        real_t adj_V1 = fabs(adj_T1_B1*adj_cross1);
                        real_t adj_V2 = fabs(adj_T2_B1*adj_cross2);
                        const real_t A_el_2 = (adj_V1+adj_V2)/2;

                        const real_t A_el_avg = 0.5*(A_el + A_el_2);

                        const real_t adj_A_ref = (ir_pp2[ix2+1].x - ir_pp2[ix2].x)
                                 * (ir_pp2[iy2+1].x - ir_pp2[iy2].x)
                                 * (ir_pp2[iz2+1].x - ir_pp2[iz2].x);
                        //const real_t A_ref_avg = 0.5*(A_ref + adj_A_ref);
                        
                        const real_t adj_A_ref_perp = adj_A_ref/A_ref_face;
                        const real_t A_ref_perp_avg = 0.5*(adj_A_ref_perp + A_ref_perp);

                        //std::cout << "adj A_el =  " << A_el_2 << std::endl;
                        //std::cout << "adj A_ref = " << adj_A_ref << std::endl;



                        const real_t h = A_ref_face * A_el_avg / (A_ref_perp_avg * A_face);
                        //const real_t h = A_ref_avg/A_ref_perp_avg;
                        V(v_idx, ix, iy, iz, iel_ho) = -dq * A_face * w_1d[w_idx_1] *  w_1d[w_idx_2] / h /
                                                (ir_pp1[int_idx].x - ir_pp1[int_idx-1].x); 
                     }
                  }
               }
               V(0, ix, iy, iz, iel_ho) = mq * detJ(ix, iy, iz, iel_ho) * W(ix, iy, iz);
               for (int i = 1; i < 7; ++i)
               {
                  V(0, ix, iy, iz, iel_ho) -= V(i, ix, iy, iz, iel_ho);
               }
            }
         }
      }
   });
}

} // namespace mfem
