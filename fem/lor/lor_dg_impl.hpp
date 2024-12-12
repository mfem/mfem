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
   const int p = ORDER;
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

   //const FiniteElementCollection fes_coll = fes_ho.FEColl();

   const auto detJ = Reshape(geom->detJ.Read(), nd1d, nd1d, nel_ho);

   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for (int iy = 0; iy < nd1d; ++iy)
      {
         for (int ix = 0; ix < nd1d; ++ix)
         {
            const real_t mq = const_mq ? MQ(0,0,0) : MQ(ix, iy, iel_ho);
            const real_t dq = const_dq ? DQ(0,0,0) : DQ(ix, iy, iel_ho);
            for (int n_idx = 0; n_idx < 2; ++n_idx)
            {
               for (int e_i = 0; e_i < 2; ++e_i)
               {
                  static int lex_map[] = {4, 2, 1, 3};
                  const int v_idx_lex = e_i + n_idx*2;
                  const int v_idx = lex_map[v_idx_lex];

                  const int i_0 = (n_idx == 0) ? ix + e_i : ix;
                  const int j_0 = (n_idx == 1) ? iy + e_i : iy;

                  const int i_1 = (n_idx == 0) ? ix + e_i : ix + 1;
                  const int j_1 = (n_idx == 1) ? iy + e_i : iy + 1;

                  const bool bdr = (n_idx == 0) ? (i_0 == 0 || i_0 == pp1)
                                   : (j_0 == 0 || j_0 == pp1);

                  const real_t h_face =1.0;
                  const real_t h_el = 1.0; // from X_vert
                  const real_t h = h_face/h_el;
                  const int w_idx = (n_idx == 0) ? iy : ix;
                  const int int_idx = (n_idx == 0) ? i_0 : j_0;
                  if (bdr)
                  {
                     V(v_idx, ix, iy, iel_ho) = -dq * kappa * w_1d[w_idx] * detJ(ix, iy,iel_ho)/h;
                  }
                  else
                  {
                     V(v_idx, ix, iy, iel_ho) = -dq * w_1d[w_idx] / (ir_pp1[int_idx].x -
                                                                     ir_pp1[int_idx-1].x);
                  }
               }
            }
            /*
            const real_t h_face = 1.0;
            const real_t h_el = 1.0; // from X_vert
            const real_t h = h_face/h_el;
            if (ix == 0)
            {
               V(4, ix, iy, iel_ho) = -dq*kappa * w_1d[iy] * detJ(ix, iy,iel_ho)/h;
            }
            else
            {
               V(4, ix, iy, iel_ho) = -dq*w_1d[iy]/(ir[ix].x - ir[ix-1].x);
            }
            if (ix == ORDER)
            {
               V(2, ix, iy, iel_ho) = -dq*kappa * w_1d[iy] * detJ(ix, iy,iel_ho)/h;
            }
            else
            {
               V(2, ix, iy, iel_ho) = -dq*w_1d[iy]/(ir[ix+1].x - ir[ix].x);
            }
            if (iy == 0)
            {
               V(1, ix, iy, iel_ho) = -dq*kappa * w_1d[ix] * detJ(ix, iy,iel_ho)/h;
            }
            else
            {
               V(1, ix, iy, iel_ho) = -dq*w_1d[ix]/(ir[iy].x - ir[iy-1].x);
            }
            if (iy == ORDER)
            {
               V(3, ix, iy, iel_ho) = -dq*kappa * w_1d[ix] * detJ(ix, iy,iel_ho)/h;
            }
            else
            {
               V(3, ix, iy, iel_ho) = -dq*w_1d[ix]/(ir[iy+1].x - ir[iy].x);
            }
            */
            for (int i = 1; i < 5; ++i)
            {
               V(0, ix, iy, iel_ho) -= V(i, ix, iy, iel_ho);
            }

            V(0, ix, iy, iel_ho) += mq * detJ(ix, iy, iel_ho) * W(ix, iy);
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
