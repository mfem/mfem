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
   static constexpr int nd1d = pp1;
   static constexpr int ndof_per_el = nd1d*nd1d;
   static constexpr int nnz_per_row = 5;
   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1)
                   : Reshape(c1.Read(), nd1d, nd1d, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1)
                   : Reshape(c2.Read(), nd1d, nd1d, nel_ho);

   const auto w_1d = ir_pp1.GetWeights().Read();
   const auto W = Reshape(ir.GetWeights().Read(), nd1d, nd1d);
   const auto X = Reshape(X_vert.Read(), 2, pp2, pp2, nel_ho);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, nd1d, nd1d, nel_ho);

   auto geom = fes_ho.GetMesh()->GetGeometricFactors(
                  ir, GeometricFactors::DETERMINANTS);
   const auto detJ = Reshape(geom->detJ.Read(), nd1d, nd1d, nel_ho);

   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for (int iy = 0; iy < nd1d; ++iy)
      {
         for (int ix = 0; ix < nd1d; ++ix)
         {
            const real_t A_ref = (ir_pp2[ix+1].x - ir_pp2[ix].x)
                                 * (ir_pp2[iy+1].x - ir_pp2[iy].x);
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
                  const real_t el_1 = ir_pp2[el_idx+1].x - ir_pp2[el_idx].x;
                  const real_t el_2 = A_ref / el_1;

                  const real_t x1 = X(0, i_0, j_0, iel_ho);
                  const real_t y1 = X(1, i_0, j_0, iel_ho);
                  const real_t x2 = X(0, i_1, j_1, iel_ho);
                  const real_t y2 = X(1, i_1, j_1, iel_ho);
                  const real_t A_face = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));

                  if (bdr)
                  {
                     const real_t h_recip = A_face*A_face/A_el*el_2/el_1;
                     V(v_idx, ix, iy, iel_ho) = -dq * kappa * w_1d[w_idx] * h_recip;
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
                     const real_t A_ref_1 = (n_idx == 0) ? ir_pp2[i_0+1].x - ir_pp2[i_0].x :
                                            ir_pp2[j_0+1].x - ir_pp2[j_0].x;
                     const real_t A_ref_2 = (n_idx == 0) ? ir_pp2[i_0].x - ir_pp2[i_0-1].x :
                                            ir_pp2[j_0].x - ir_pp2[j_0-1].x;
                     const real_t h = (0.5*A_el + 0.5*A_el_2) / A_face / (0.5 * (A_ref_1 + A_ref_2));
                     V(v_idx, ix, iy, iel_ho) = -dq * A_face * w_1d[w_idx] / h / el_1 /
                                                (ir_pp1[int_idx].x - ir_pp1[int_idx-1].x);
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
   MFEM_ABORT("Not yet implemented.");
}

} // namespace mfem
