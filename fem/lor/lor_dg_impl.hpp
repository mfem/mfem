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
   MFEM_VERIFY(SDIM == 2, "Surface meshes not currently supported for LOR-DG.")

   static constexpr int pp1 = ORDER + 1;
   static constexpr int ndof_per_el = pp1*pp1;
   static constexpr int nnz_per_row = 5;

   const int nel_ho = fes_ho.GetNE();

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

   // Geometric factors and quadrature weights
   Mesh &mesh = *fes_ho.GetMesh();
   const auto factors = GeometricFactors::DETERMINANTS |
                        GeometricFactors::JACOBIANS;
   auto geom = mesh.GetGeometricFactors(ir, factors);
   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, nel_ho);
   const auto J = Reshape(geom->J.Read(), pp1, pp1, 2, 2, nel_ho);
   const auto W = Reshape(ir.GetWeights().Read(), pp1, pp1);

   const int nf = mesh.GetNumFaces();
   Array<int> face_info(nf * 4); // (e0, fid0, e1, fid1)
   {
      auto h_face_info = Reshape(face_info.HostWrite(), 4, nf);
      for (int f = 0; f < nf; ++f)
      {
         auto finfo = mesh.GetFaceInformation(f);
         h_face_info(0, f) = finfo.element[0].index;
         h_face_info(1, f) = finfo.element[0].local_face_id;
         if (finfo.IsInterior())
         {
            h_face_info(2, f) = finfo.element[1].index;
            h_face_info(3, f) = finfo.element[1].local_face_id;
         }
         else
         {
            h_face_info(2, f) = -1;
            h_face_info(3, f) = -1;
         }
      }
   }
   const auto d_face_info = Reshape(face_info.Read(), 4, nf);

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

   Vector face_Jh(pp1 * nf);
   auto d_face_Jh = Reshape(face_Jh.Write(), pp1, nf);
   for (const FaceType ft : {FaceType::Interior, FaceType::Boundary})
   {
      const int nft = mesh.GetNFbyType(ft);
      auto *geom_face = mesh.GetFaceGeometricFactors(
                           ir_pp1, FaceGeometricFactors::DETERMINANTS, ft);

      auto *r = fes_ho.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC, ft);
      Vector detJ_r(pp1 * 2 * nft);
      r->Mult(geom->detJ, detJ_r);

      const auto *d_i = (ft == FaceType::Interior) ? f_int.Read() : f_bdr.Read();
      const auto d_detJ_face = Reshape(geom_face->detJ.Read(), pp1, nft);
      const auto d_detJ_r = Reshape(detJ_r.Read(), pp1, 2, nft);

      mfem::forall(nft * pp1, [=] MFEM_HOST_DEVICE (int ii)
      {
         const int i = ii % pp1;
         const int f = ii / pp1;
         const real_t J_el = 0.5*(d_detJ_r(i, 0, f) + d_detJ_r(i, 0, f));
         const real_t J_f = d_detJ_face(i, f);
         d_face_Jh(i, d_i[f]) = J_f * J_f / J_el;
      });
   }

   // Penalty parameter (avoid capturing *this in lambda)
   const real_t d_kappa = kappa;

   // Sparse matrix entries
   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   sparse_ij.UseDevice(true);
   sparse_ij = 0.0;
   auto V = Reshape(sparse_ij.ReadWrite(), nnz_per_row, pp1, pp1, nel_ho);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
   {
      const int f_0 = d_face_info(1, f);
      const int f_1 = d_face_info(3, f);
      const int nsides = (f_1 >= 0) ? 2 : 1;
      for (int el_i = 0; el_i < nsides; ++el_i)
      {
         const int el = d_face_info(2*el_i, f);
         const int v_idx = 1 + ((el_i == 0) ? f_0 : f_1);
         for (int i = 0; i < pp1; ++i)
         {
            int ix, iy;
            internal::FaceIdxToVolIdx2D(i, pp1, f_0, f_1, el_i, ix, iy);
            const real_t Jh = d_face_Jh(i, f);
            const real_t dq = const_dq ? DQ(0,0,0) : DQ(ix, iy, el);
            V(v_idx, ix, iy, el) = -dq * d_kappa * Jh * w_1d[i];
         }
      }
   });

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
   static constexpr int pp2 = ORDER + 2;
   const int nel_ho = fes_ho.GetNE();

   Mesh &mesh = *fes_ho.GetMesh();

   IntegrationRule ir_pp1;
   QuadratureFunctions1D::GaussLobatto(pp1, &ir_pp1);
   IntegrationRule ir_pp2;
   QuadratureFunctions1D::GaussLobatto(pp2, &ir_pp2);
   Vector vec_ir_pp1_x(pp1);
   Vector vec_ir_pp2_x(pp2);
   for (int i=0; i < pp1; i++)
   {
      vec_ir_pp1_x[i] = ir_pp1[i].x;
   }
   for (int j=0; j < pp2; j++)
   {
      vec_ir_pp2_x[j] = ir_pp2[j].x;
   }

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

   const auto factors = GeometricFactors::DETERMINANTS |
                        GeometricFactors::JACOBIANS;
   auto geom = fes_ho.GetMesh()->GetGeometricFactors(ir, factors);
   const auto detJ = Reshape(geom->detJ.Read(), pp1, pp1, pp1, nel_ho);
   const auto J = Reshape(geom->J.Read(), pp1, pp1, pp1, 3, 3, nel_ho);

   const auto *x_pp1 = vec_ir_pp1_x.Read();
   const auto *d_vec_ir_pp2_x = vec_ir_pp2_x.Read();
   const real_t d_kappa = kappa;

   /////////////////////////////////////////////////////////

   const int nf = mesh.GetNumFaces();
   Array<int> face_info(nf * 6); // (e0, f0, o0, e1, f1, o1)
   {
      auto h_face_info = Reshape(face_info.HostWrite(), 6, nf);
      for (int f = 0; f < nf; ++f)
      {
         auto finfo = mesh.GetFaceInformation(f);
         h_face_info(0, f) = finfo.element[0].index;
         h_face_info(1, f) = finfo.element[0].local_face_id;
         h_face_info(2, f) = finfo.element[0].orientation;
         if (finfo.IsInterior())
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
      }
   }
   const auto d_face_info = Reshape(face_info.Read(), 6, nf);

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

   Vector face_Jh(pp1 * pp1 * nf);
   auto d_face_Jh = Reshape(face_Jh.Write(), pp1 * pp1, nf);
   for (const FaceType ft : {FaceType::Interior, FaceType::Boundary})
   {
      IntegrationRule ir_face = GetCollocatedFaceIntRule(fes_ho);
      const int nft = mesh.GetNFbyType(ft);
      auto *geom_face = mesh.GetFaceGeometricFactors(
                           ir_face, FaceGeometricFactors::DETERMINANTS, ft);

      auto *r = fes_ho.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC, ft);
      Vector detJ_r(pp1 * pp1 * 2 * nft);
      r->Mult(geom->detJ, detJ_r);

      const auto *d_i = (ft == FaceType::Interior) ? f_int.Read() : f_bdr.Read();
      const auto d_detJ_face = Reshape(geom_face->detJ.Read(), pp1 * pp1, nft);
      const auto d_detJ_r = Reshape(detJ_r.Read(), pp1 * pp1, 2, nft);

      mfem::forall(nft * pp1 * pp1, [=] MFEM_HOST_DEVICE (int ii)
      {
         const int i = ii % (pp1 * pp1);
         const int f = ii / (pp1 * pp1);
         const real_t J_el = 0.5*(d_detJ_r(i, 0, f) + d_detJ_r(i, 0, f));
         const real_t J_f = d_detJ_face(i, f);
         const real_t h = J_el / J_f;
         d_face_Jh(i, d_i[f]) =  J_f / h;
      });
   }

   ///////////////////////////////////////////////

   mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
   {
      const int f_0 = d_face_info(1, f);
      const int f_1 = d_face_info(4, f);
      const int nsides = (f_1 >= 0) ? 2 : 1;
      for (int el_i = 0; el_i < nsides; ++el_i)
      {
         const int e = d_face_info(3*el_i, f);
         const int o = d_face_info(3*el_i + 2, f);
         const int v_idx = 1 + ((el_i == 0) ? f_0 : f_1);
         for (int i = 0; i < pp1*pp1; ++i)
         {
            int ix, iy, iz;
            // internal::FaceIdxToVolIdx3D(i, pp1, f_0, f_1, el_i, ix, iy);
            internal::FaceIdxToVolIdx3D(i, pp1, f_0, f_1, el_i, o, ix, iy, iz);
            const real_t Jh = d_face_Jh(i, f);
            const real_t dq = const_dq ? DQ(0,0,0,0) : DQ(ix, iy, iz, e);
            V(v_idx, ix, iy, iz, e) = -dq*d_kappa*Jh*w_1d[i%pp1]*w_1d[i/pp1];
         }
      }
   });

   ///////////////////////////////////////////////


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

                     int w_idx_1 = (n_idx == 0) ? iy : (n_idx == 1) ? iz : ix;
                     int w_idx_2 = (n_idx == 0) ? iz : (n_idx == 1) ? ix : iy;

                     int x_idx = (n_idx == 0) ? i_0 : (n_idx == 1) ? j_0 : k_0;

                     bool bdr = true;
                     if (n_idx == 0) {bdr = (i_0 == 0 || i_0 == pp1); w_idx_1 = iy; w_idx_2 = iz; x_idx = i_0;}
                     else if (n_idx == 1) {bdr = (j_0 == 0 || j_0 == pp1); w_idx_1 = ix; w_idx_2 = iz; x_idx = j_0;}
                     else {bdr = (k_0 == 0 || k_0 == pp1); w_idx_1 = ix; w_idx_2 = iy; x_idx = k_0;}

                     if (bdr) { continue; }

                     const int ix2 = (n_idx == 0) ? ix + (e_i == 0 ? -1 : 1) : ix;
                     const int iy2 = (n_idx == 1) ? iy + (e_i == 0 ? -1 : 1) : iy;
                     const int iz2 = (n_idx == 2) ? iz + (e_i == 0 ? -1 : 1) : iz;

                     const real_t J00 = J(ix, iy, iz, 0, 0, iel_ho);
                     const real_t J01 = J(ix, iy, iz, 0, 1, iel_ho);
                     const real_t J02 = J(ix, iy, iz, 0, 2, iel_ho);
                     const real_t J10 = J(ix, iy, iz, 1, 0, iel_ho);
                     const real_t J11 = J(ix, iy, iz, 1, 1, iel_ho);
                     const real_t J12 = J(ix, iy, iz, 1, 2, iel_ho);
                     const real_t J20 = J(ix, iy, iz, 2, 0, iel_ho);
                     const real_t J21 = J(ix, iy, iz, 2, 1, iel_ho);
                     const real_t J22 = J(ix, iy, iz, 2, 2, iel_ho);

                     real_t J_diag = 0.0;
                     if (n_idx == 0)
                     {
                        J_diag = J02*J02*(J11*J11 + J21*J21) + (J12*J21 - J11*J22)*
                                 (J12*J21 - J11*J22) - 2*J01*J02*(J11*J12 + J21*J22) + J01*J01*
                                 (J12*J12 + J22*J22);
                     }
                     else if (n_idx == 1)
                     {
                        J_diag = J02*J02*(J10*J10 + J20*J20) + (J12*J20 - J10*J22)*
                                 (J12*J20 - J10*J22) - 2*J00*J02*(J10*J12 + J20*J22) + J00*J00*
                                 (J12*J12 + J22*J22);
                     }
                     else if (n_idx == 2)
                     {
                        J_diag = J01*J01*(J10*J10 + J20*J20) + (J11*J20 - J10*J21)*
                                 (J11*J20 - J10*J21) - 2*J00*J01*(J10*J11 + J20*J21) + J00*J00*
                                 (J11*J11 + J21*J21);
                     }

                     const real_t Jh = J_diag / detJ(ix, iy, iz, iel_ho);

                     V(v_idx, ix, iy, iz, iel_ho) =
                        -dq * Jh * w_1d[w_idx_1] * w_1d[w_idx_2] / (x_pp1[x_idx] - x_pp1[x_idx -1]);
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
