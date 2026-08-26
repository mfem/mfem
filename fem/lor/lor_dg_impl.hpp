// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#include "lor_dg.hpp"
#include "../fe/face_map_utils.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

Array<int> BatchedLOR_DG::GetFaceInfo() const
{
   Mesh &mesh = *fes_ho.GetMesh();
   const Array<int> &bdr_face_attrs = mesh.GetBdrFaceAttributes();
   const int nf = mesh.GetNumFaces();
   Array<int> face_info(nf * 6); // (e0, f0, o0, e1, f1, o1)
   auto h_face_info = Reshape(face_info.HostWrite(), 6, nf);

   int bdr_face_counter = 0;

   for (int f = 0; f < nf; ++f)
   {
      auto finfo = mesh.GetFaceInformation(f);
      h_face_info(0, f) = finfo.element[0].index;
      h_face_info(1, f) = finfo.element[0].local_face_id;
      h_face_info(2, f) = finfo.element[0].orientation;
      if (finfo.IsLocal()) // Interior, non-shared face
      {
         h_face_info(3, f) = finfo.element[1].index;
         h_face_info(4, f) = finfo.element[1].local_face_id;
         h_face_info(5, f) = finfo.element[1].orientation;
      }
      else
      {
         h_face_info(3, f) = -1;
         h_face_info(4, f) = -1;
         h_face_info(5, f) = -1;
      }

      if (finfo.IsBoundary())
      {
         // Check if Neumann boundary; skip these when adding boundary penalties
         const int bdr_attr = bdr_face_attrs[bdr_face_counter];
         if (!has_bdr_integ || (bdr_markers && !(*bdr_markers)[bdr_attr - 1]))
         {
            h_face_info(0, f) = -1;
            h_face_info(1, f) = -1;
            h_face_info(2, f) = -1;
         }
         bdr_face_counter += 1;
      }
   }
   return face_info;
}

Vector BatchedLOR_DG::GetBdrPenaltyFactor() const
{
   Mesh &mesh = *fes_ho.GetMesh();

   const int nf = mesh.GetNumFaces();
   Array<int> f_int(mesh.GetNFbyType(FaceType::Interior));
   Array<int> f_bdr(mesh.GetNFbyType(FaceType::Boundary));
   {
      int i_int = 0;
      int i_bdr = 0;
      for (int i = 0; i < nf; ++i)
      {
         const auto f = mesh.GetFaceInformation(i);
         if (f.IsBoundary())
         {
            f_bdr[i_bdr] = i;
            ++i_bdr;
         }
         else if (f.IsInterior())
         {
            f_int[i_int] = i;
            ++i_int;
         }
      }
   }

   const auto geom = fes_ho.GetMesh()->GetGeometricFactors(
                        ir, GeometricFactors::DETERMINANTS);

   const int nq = ir_face.Size();
   Vector face_Jh(nq * nf);
   for (const FaceType ft : {FaceType::Interior, FaceType::Boundary})
   {
      const int nft = mesh.GetNFbyType(ft);
      auto *geom_face = mesh.GetFaceGeometricFactors(
                           ir_face, FaceGeometricFactors::DETERMINANTS, ft);

      const L2FaceValues fv = (ft == FaceType::Interior)
                              ? L2FaceValues::DoubleValued
                              : L2FaceValues::SingleValued;
      const int m = (fv == L2FaceValues::DoubleValued) ? 2 : 1;

      auto *r = fes_ho.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC, ft, fv);
      Vector detJ_r(nq * m * nft);
      r->Mult(geom->detJ, detJ_r);

      const auto *d_i = (ft == FaceType::Interior) ? f_int.Read() : f_bdr.Read();
      const auto d_detJ_face = Reshape(geom_face->detJ.Read(), nq, nft);
      const auto d_detJ_r = Reshape(detJ_r.Read(), nq, m, nft);
      auto d_face_Jh = Reshape(face_Jh.Write(), nq, nf);

      mfem::forall(nft * nq, [=] MFEM_HOST_DEVICE (int ii)
      {
         const int i = ii % nq;
         const int f = ii / nq;
         const real_t J_el = 0.5*(d_detJ_r(i, 0, f) + d_detJ_r(i, m==2?1:0, f));
         const real_t J_f = d_detJ_face(i, f);
         d_face_Jh(i, d_i[f]) = J_f * J_f / J_el;
      });
   }
   return face_Jh;
}

void BatchedLOR_DG::AssembleFaceTerms()
{
   Mesh &mesh = *fes_ho.GetMesh();

   const int nnz_per_row = 1 + mesh.Dimension()*2;
   const int pp1 = fes_ho.GetMaxElementOrder() + 1;
   const int nel_ho = mesh.GetNE();
   const int nf = mesh.GetNumFaces();
   const int nd_face = ir_face.Size();
   const int nd = ir.Size();
   const int dim = mesh.Dimension();

   Array<int> face_info = GetFaceInfo();
   const auto d_face_info = Reshape(face_info.Read(), 6, nf);

   Vector face_Jh = GetBdrPenaltyFactor();
   const auto d_face_Jh = Reshape(face_Jh.Read(), nd_face, nf);

   const auto *w_face = ir_face.GetWeights().Read();

   // Penalty parameter (avoid capturing *this in lambda)
   const real_t d_kappa = kappa;

   // Get diffusion coefficient
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq?Reshape(c2.Read(),1,1):Reshape(c2.Read(),nd,nel_ho);

   // Sparse matrix entries
   auto V = Reshape(sparse_ij.ReadWrite(), nnz_per_row, nd, nel_ho);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
   {
      const int f_0 = d_face_info(1, f);
      const int f_1 = d_face_info(4, f);
      if (f_0 < 0) { return; } // Skip Neumann boundary faces
      const int nsides = (f_1 >= 0) ? 2 : 1;
      for (int el_i = 0; el_i < nsides; ++el_i)
      {
         const int e = d_face_info(3*el_i, f);
         const int o = d_face_info(3*el_i + 2, f);
         const int v_idx = 1 + ((el_i == 0) ? f_0 : f_1);
         for (int i = 0; i < nd_face; ++i)
         {
            const int ii = internal::FaceIdxToVolIdx(dim, i, pp1, f_0, f_1, el_i, o);
            const real_t Jh = d_face_Jh(i, f);
            const real_t dq = const_dq ? DQ(0,0) : DQ(ii, e);
            V(v_idx, ii, e) = -dq*d_kappa*Jh*w_face[i];
         }
      }
   });
}

template <int ORDER, int SDIM>
void BatchedLOR_DG::Assemble2D()
{
   MFEM_VERIFY(SDIM == 2, "Surface meshes not currently supported for LOR-DG.")

   static constexpr int pp1 = ORDER + 1;
   static constexpr int ndof_per_el = pp1*pp1;
   static constexpr int nnz_per_row = 5;
   const int nel_ho = fes_ho.GetNE();

   // Get element geometric factors; calling before AssembleFaceTerms, since
   // in AssembleFaceTerms, element Jacobian determinants are used, potentially
   // saving recomputation.
   const auto factors = GeometricFactors::DETERMINANTS |
                        GeometricFactors::JACOBIANS;
   const auto *geom = fes_ho.GetMesh()->GetGeometricFactors(ir, factors);

   // Sparse matrix entries
   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   sparse_ij.UseDevice(true);
   sparse_ij = 0.0;
   auto V = Reshape(sparse_ij.ReadWrite(), nnz_per_row, pp1, pp1, nel_ho);

   AssembleFaceTerms();

   // Populate Gauss-Lobatto quadrature rule of size (p+1)
   IntegrationRule ir_pp1;
   QuadratureFunctions1D::GaussLobatto(pp1, &ir_pp1);
   Vector glx_pp1(pp1), glw_pp1(pp1);
   for (int i = 0; i < pp1; ++i)
   {
      glx_pp1[i] = ir_pp1[i].x;
      glw_pp1[i] = ir_pp1[i].weight;
   }
   const auto *x_pp1 = glx_pp1.Read();
   const auto *w_1d = glw_pp1.Read();

   // Get coefficients for mass and diffusion
   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1)
                   : Reshape(c1.Read(), pp1, pp1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1)
                   : Reshape(c2.Read(), pp1, pp1, nel_ho);

   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, nel_ho);
   const auto J = Reshape(geom->J.Read(), pp1, pp1, 2, 2, nel_ho);
   const auto W = Reshape(ir.GetWeights().Read(), pp1, pp1);

   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for (int iy = 0; iy < pp1; ++iy)
      {
         for (int ix = 0; ix < pp1; ++ix)
         {
            const real_t mq = const_mq ? MQ(0,0,0) : MQ(ix, iy, iel_ho);
            const real_t dq = const_dq ? DQ(0,0,0) : DQ(ix, iy, iel_ho);

            for (int n_idx = 0; n_idx < 2; ++n_idx)
            {
               for (int e_i = 0; e_i < 2; ++e_i)
               {
                  const int i_0 = (n_idx == 0) ? ix + e_i : ix;
                  const int j_0 = (n_idx == 1) ? iy + e_i : iy;

                  const bool bdr = (n_idx == 0 && (i_0 == 0 || i_0 == pp1)) ||
                                   (n_idx == 1 && (j_0 == 0 || j_0 == pp1));

                  if (bdr) { continue; }

                  static constexpr int lex_map[] = {4, 2, 1, 3};
                  const int v_idx_lex = e_i + n_idx*2;
                  const int v_idx = lex_map[v_idx_lex];

                  const int w_idx = (n_idx == 0) ? iy : ix;
                  const int x_idx = (n_idx == 0) ? i_0 : j_0;

                  const real_t J1 = J(ix, iy, n_idx, !n_idx, iel_ho);
                  const real_t J2 = J(ix, iy, !n_idx, !n_idx, iel_ho);
                  const real_t Jh = (J1*J1 + J2*J2) / detJ(ix, iy, iel_ho);

                  V(v_idx, ix, iy, iel_ho) =
                     -dq * Jh * w_1d[w_idx] / (x_pp1[x_idx] - x_pp1[x_idx -1]);
               }
            }
            V(0, ix, iy, iel_ho) = mq * detJ(ix, iy, iel_ho) * W(ix, iy);
            for (int i = 1; i < nnz_per_row; ++i)
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
   static constexpr int pp1 = ORDER + 1;
   static constexpr int ndof_per_el = pp1*pp1*pp1;
   static constexpr int nnz_per_row = 7;
   const int nel_ho = fes_ho.GetNE();

   // Get element geometric factors; calling before AssembleFaceTerms, since
   // in AssembleFaceTerms, element Jacobian determinants are used, potentially
   // saving recomputation.
   const auto factors = GeometricFactors::DETERMINANTS |
                        GeometricFactors::JACOBIANS;
   const auto geom = fes_ho.GetMesh()->GetGeometricFactors(ir, factors);

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   sparse_ij.UseDevice(true);
   sparse_ij = 0.0;
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, pp1, pp1, pp1, nel_ho);

   AssembleFaceTerms();

   // Populate Gauss-Lobatto quadrature rule of size (p+1)
   IntegrationRule ir_pp1;
   QuadratureFunctions1D::GaussLobatto(pp1, &ir_pp1);
   Vector glx_pp1(pp1), glw_pp1(pp1);
   for (int i = 0; i < pp1; ++i)
   {
      glx_pp1[i] = ir_pp1[i].x;
      glw_pp1[i] = ir_pp1[i].weight;
   }
   const auto *x_pp1 = glx_pp1.Read();
   const auto *w_1d = glw_pp1.Read();

   const bool const_mq = c1.Size() == 1;
   const auto MQ = const_mq
                   ? Reshape(c1.Read(), 1, 1, 1, 1)
                   : Reshape(c1.Read(), pp1, pp1, pp1, nel_ho);
   const bool const_dq = c2.Size() == 1;
   const auto DQ = const_dq
                   ? Reshape(c2.Read(), 1, 1, 1, 1)
                   : Reshape(c2.Read(), pp1, pp1, pp1, nel_ho);
   const auto W = Reshape(ir.GetWeights().Read(), pp1, pp1, pp1);

   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, pp1, nel_ho);
   const auto J = Reshape(geom->J.Read(), pp1, pp1, pp1, 3, 3, nel_ho);

   mfem::forall(nel_ho, [=] MFEM_HOST_DEVICE (int iel_ho)
   {
      for (int iz = 0; iz < pp1; ++iz)
      {
         for (int iy = 0; iy < pp1; ++iy)
         {
            for (int ix = 0; ix < pp1; ++ix)
            {
               const real_t mq = const_mq ? MQ(0,0,0,0) : MQ(ix, iy, iz, iel_ho);
               const real_t dq = const_dq ? DQ(0,0,0,0) : DQ(ix, iy, iz, iel_ho);

               const real_t DETJ = detJ(ix, iy, iz, iel_ho);

               for (int n_idx = 0; n_idx < 3; ++n_idx)
               {
                  for (int e_i = 0; e_i < 2; ++e_i)
                  {
                     static constexpr int lex_map[] = {5,3,2,4,1,6};
                     const int v_idx_lex = e_i + n_idx*2;
                     const int v_idx = lex_map[v_idx_lex];

                     const int i_0 = (n_idx == 0) ? ix + e_i : ix;
                     const int j_0 = (n_idx == 1) ? iy + e_i : iy;
                     const int k_0 = (n_idx == 2) ? iz + e_i : iz;

                     const bool bdr =
                        (n_idx == 0 && (i_0 == 0 || i_0 == pp1)) ||
                        (n_idx == 1 && (j_0 == 0 || j_0 == pp1)) ||
                        (n_idx == 2 && (k_0 == 0 || k_0 == pp1));

                     if (bdr) { continue; }

                     int x_idx = (n_idx == 0) ? i_0 : (n_idx == 1) ? j_0 : k_0;
                     int w_idx_1 = (n_idx == 0) ? iy : (n_idx == 1) ? iz : ix;
                     int w_idx_2 = (n_idx == 0) ? iz : (n_idx == 1) ? ix : iy;

                     const real_t J00 = J(ix, iy, iz, 0, 0, iel_ho);
                     const real_t J01 = J(ix, iy, iz, 0, 1, iel_ho);
                     const real_t J02 = J(ix, iy, iz, 0, 2, iel_ho);
                     const real_t J10 = J(ix, iy, iz, 1, 0, iel_ho);
                     const real_t J11 = J(ix, iy, iz, 1, 1, iel_ho);
                     const real_t J12 = J(ix, iy, iz, 1, 2, iel_ho);
                     const real_t J20 = J(ix, iy, iz, 2, 0, iel_ho);
                     const real_t J21 = J(ix, iy, iz, 2, 1, iel_ho);
                     const real_t J22 = J(ix, iy, iz, 2, 2, iel_ho);

                     real_t JinvJinvT_diag = 0.0;
                     if (n_idx == 0)
                     {
                        JinvJinvT_diag = J02*J02*(J11*J11 + J21*J21) + (J12*J21 - J11*J22)*
                                         (J12*J21 - J11*J22) - 2*J01*J02*(J11*J12 + J21*J22) + J01*J01*
                                         (J12*J12 + J22*J22);
                     }
                     else if (n_idx == 1)
                     {
                        JinvJinvT_diag = J02*J02*(J10*J10 + J20*J20) + (J12*J20 - J10*J22)*
                                         (J12*J20 - J10*J22) - 2*J00*J02*(J10*J12 + J20*J22) + J00*J00*
                                         (J12*J12 + J22*J22);
                     }
                     else if (n_idx == 2)
                     {
                        JinvJinvT_diag = J01*J01*(J10*J10 + J20*J20) + (J11*J20 - J10*J21)*
                                         (J11*J20 - J10*J21) - 2*J00*J01*(J10*J11 + J20*J21) + J00*J00*
                                         (J11*J11 + J21*J21);
                     }

                     const real_t Jh = JinvJinvT_diag / DETJ;

                     V(v_idx, ix, iy, iz, iel_ho) = -dq * Jh * w_1d[w_idx_1] * w_1d[w_idx_2] /
                                                    (x_pp1[x_idx] - x_pp1[x_idx -1]);
                  }
               }
               V(0, ix, iy, iz, iel_ho) = mq * DETJ * W(ix, iy, iz);
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
