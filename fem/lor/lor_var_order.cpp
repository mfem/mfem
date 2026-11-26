// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor_var_order.hpp"

#ifdef MFEM_USE_MPI

#include "../bilininteg.hpp"
#include "../fe/fe_h1.hpp"
#include "../fe/fe_nd.hpp"
#include "../fe/fe_rt.hpp"
#include "../pgridfunc.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace mfem
{

namespace
{

static constexpr int kHexVx[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
static constexpr int kHexVy[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
static constexpr int kHexVz[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

inline void Cross3(const real_t a[3], const real_t b[3], real_t out[3])
{
   out[0] = a[1] * b[2] - a[2] * b[1];
   out[1] = a[2] * b[0] - a[0] * b[2];
   out[2] = a[0] * b[1] - a[1] * b[0];
}

enum class LORSpaceKind { H1, ND, RT, Unsupported };

LORSpaceKind ClassifySpace(const FiniteElementSpace &fes)
{
   const FiniteElementCollection *fec = fes.FEColl();
   if (const auto *h1 = dynamic_cast<const H1_FECollection*>(fec))
   {
      if (h1->GetBasisType() == BasisType::GaussLobatto)
      {
         return LORSpaceKind::H1;
      }
      return LORSpaceKind::Unsupported;
   }
   if (const auto *nd = dynamic_cast<const ND_FECollection*>(fec))
   {
      if (nd->GetClosedBasisType() == BasisType::GaussLobatto &&
          nd->GetOpenBasisType() == BasisType::IntegratedGLL)
      {
         return LORSpaceKind::ND;
      }
      return LORSpaceKind::Unsupported;
   }
   if (const auto *rt = dynamic_cast<const RT_FECollection*>(fec))
   {
      if (rt->GetClosedBasisType() == BasisType::GaussLobatto &&
          rt->GetOpenBasisType() == BasisType::IntegratedGLL)
      {
         return LORSpaceKind::RT;
      }
      return LORSpaceKind::Unsupported;
   }
   return LORSpaceKind::Unsupported;
}

bool IntegratorSupported(const BilinearFormIntegrator *bfi, LORSpaceKind kind)
{
   if (kind == LORSpaceKind::H1)
   {
      return dynamic_cast<const DiffusionIntegrator*>(bfi) != nullptr ||
             dynamic_cast<const MassIntegrator*>(bfi) != nullptr;
   }
   if (kind == LORSpaceKind::ND)
   {
      return dynamic_cast<const CurlCurlIntegrator*>(bfi) != nullptr ||
             dynamic_cast<const VectorFEMassIntegrator*>(bfi) != nullptr;
   }
   if (kind == LORSpaceKind::RT)
   {
      return dynamic_cast<const DivDivIntegrator*>(bfi) != nullptr ||
             dynamic_cast<const VectorFEMassIntegrator*>(bfi) != nullptr;
   }
   return false;
}

std::unique_ptr<HypreParMatrix> BuildParallel4NnzOp(
   ParFiniteElementSpace &row_fes,
   ParFiniteElementSpace &col_fes,
   const std::vector<uint8_t> &seen,
   const std::vector<std::array<int, 4>> &cols,
   const std::vector<std::array<real_t, 4>> &vals)
{
   const int nrow = row_fes.GetVSize();
   const int ncol = col_fes.GetVSize();

   std::vector<int> I(static_cast<std::size_t>(nrow) + 1, 0);
   std::vector<int> J;
   std::vector<real_t> A;
   J.reserve(static_cast<std::size_t>(4) * nrow);
   A.reserve(static_cast<std::size_t>(4) * nrow);

   std::array<std::pair<int, real_t>, 4> row_buf;
   for (int r = 0; r < nrow; ++r)
   {
      if (seen[r])
      {
         for (int k = 0; k < 4; ++k)
         {
            row_buf[k] = { cols[r][k], vals[r][k] };
         }
         std::sort(row_buf.begin(), row_buf.end(),
                   [](const auto &a, const auto &b) { return a.first < b.first; });
         int k = 0;
         while (k < 4)
         {
            const int c = row_buf[k].first;
            real_t v = row_buf[k].second;
            ++k;
            while (k < 4 && row_buf[k].first == c)
            {
               v += row_buf[k].second;
               ++k;
            }
            if (c >= 0 && v != 0.0)
            {
               J.push_back(c);
               A.push_back(v);
            }
         }
      }
      I[r + 1] = static_cast<int>(J.size());
   }

   SparseMatrix mat(I.data(), J.empty() ? nullptr : J.data(),
                    A.empty() ? nullptr : A.data(),
                    nrow, ncol, false, false, true);
   std::unique_ptr<SparseMatrix> RA(Mult(*row_fes.GetRestrictionMatrix(), mat));
   HypreParMatrix *P = col_fes.Dof_TrueDof_Matrix();
   std::unique_ptr<HypreParMatrix> res(
      P->LeftDiagMult(*RA, row_fes.GetTrueDofOffsets()));
   res->CopyRowStarts();
   res->CopyColStarts();
   return res;
}

} // namespace

bool VarOrderLORAssembly::FormIsSupported(BilinearForm &a_ho)
{
   auto *pa_ho = dynamic_cast<ParBilinearForm*>(&a_ho);
   if (!pa_ho) { return false; }

   ParFiniteElementSpace *fes = pa_ho->ParFESpace();
   if (!fes || fes->GetVDim() != 1) { return false; }

   ParMesh *pmesh = fes->GetParMesh();
   if (!pmesh || pmesh->Dimension() != 3 || pmesh->SpaceDimension() != 3)
   {
      return false;
   }

   Array<Geometry::Type> geoms;
   pmesh->GetGeometries(3, geoms);
   int loc_hex_only = (geoms.Size() == 0 ||
                       (geoms.Size() == 1 && geoms[0] == Geometry::CUBE)) ? 1 : 0;
   int glob_hex_only = 0;
   MPI_Allreduce(&loc_hex_only, &glob_hex_only, 1, MPI_INT, MPI_MIN,
                 pmesh->GetComm());
   if (!glob_hex_only) { return false; }

   const LORSpaceKind kind = ClassifySpace(*fes);
   if (kind == LORSpaceKind::Unsupported) { return false; }

   if (a_ho.GetBBFI()->Size() > 0 || a_ho.GetFBFI()->Size() > 0 ||
       a_ho.GetBFBFI()->Size() > 0)
   {
      return false;
   }
   const Array<BilinearFormIntegrator*> *dbfi = a_ho.GetDBFI();
   if (dbfi->Size() == 0) { return false; }
   for (int i = 0; i < dbfi->Size(); ++i)
   {
      if (!IntegratorSupported((*dbfi)[i], kind)) { return false; }
   }
   return true;
}

void VarOrderLORAssembly::Assemble(ParBilinearForm &a_ho,
                                   const Array<int> &ess_tdof_list,
                                   OperatorHandle &A,
                                   VarOrderLORTimings *timings)
{
   const double t0 = MPI_Wtime();
   ParFiniteElementSpace &fes = *a_ho.ParFESpace();
   const LORSpaceKind kind = ClassifySpace(fes);
   MFEM_VERIFY(kind != LORSpaceKind::Unsupported,
               "VarOrderLORAssembly: unsupported finite element space.");

   const Array<BilinearFormIntegrator*> &dbfi = *a_ho.GetDBFI();
   const Array<Array<int>*> &markers = *a_ho.GetDBFI_Marker();
   const int n_integ = dbfi.Size();

   // Save original integration rules so integrators are restored identically.
   std::vector<const IntegrationRule*> saved_irs(n_integ, nullptr);
   for (int i = 0; i < n_integ; ++i)
   {
      saved_irs[i] = dbfi[i]->GetIntRule();
   }

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule *ir_el = &irs.Get(Geometry::CUBE, 1);

   H1_HexahedronElement sub_geom_fe(1);
   std::unique_ptr<FiniteElement> sub_fe;
   int sub_ndof = 0;
   if (kind == LORSpaceKind::H1)
   {
      sub_fe.reset(new H1_HexahedronElement(1));
      sub_ndof = 8;
   }
   else if (kind == LORSpaceKind::ND)
   {
      sub_fe.reset(new ND_HexahedronElement(1));
      sub_ndof = 12;
   }
   else
   {
      sub_fe.reset(new RT_HexahedronElement(0));
      sub_ndof = 6;
   }
   const auto *sub_tfe = dynamic_cast<const TensorBasisElement*>(sub_fe.get());
   const Array<int> &dofmap_lor = sub_tfe->GetDofMap();

   IsoparametricTransformation sub_T;
   sub_T.SetFE(&sub_geom_fe);
   DenseMatrix &pm = sub_T.GetPointMat();
   pm.SetSize(3, 8);

   SparseMatrix A_loc(fes.GetVSize());
   Array<int> vdofs;
   Array<int> sub_vdofs(sub_ndof);
   DenseMatrix sub_elmat(sub_ndof, sub_ndof), tmp_elmat(sub_ndof, sub_ndof);
   Vector pt(3);
   IntegrationPoint ip;
   std::vector<real_t> node_coords;

   for (int e = 0; e < fes.GetNE(); ++e)
   {
      const FiniteElement *el = fes.GetFE(e);
      const auto *tfe = dynamic_cast<const TensorBasisElement*>(el);
      MFEM_VERIFY(tfe != nullptr && el->GetGeomType() == Geometry::CUBE,
                  "VarOrderLORAssembly requires 3D hexahedral tensor-product elements.");
      ElementTransformation *T = fes.GetElementTransformation(e);
      const int attr = T->Attribute;
      fes.GetElementVDofs(e, vdofs);
      const int p = el->GetOrder();

      if (p == 1)
      {
         for (int i = 0; i < n_integ; ++i)
         {
            if (markers[i] && (*markers[i])[attr - 1] == 0) { continue; }
            dbfi[i]->SetIntRule(saved_irs[i]);
            dbfi[i]->AssembleElementMatrix(*el, *T, tmp_elmat);
            A_loc.AddSubMatrix(vdofs, vdofs, tmp_elmat);
         }
         continue;
      }

      for (int i = 0; i < n_integ; ++i)
      {
         dbfi[i]->SetIntRule(ir_el);
      }

      const real_t *cp = poly1d.ClosedPoints(p, BasisType::GaussLobatto);
      const Array<int> &dofmap_ho = tfe->GetDofMap();
      const int p1 = p + 1;

      node_coords.resize(static_cast<std::size_t>(3) * p1 * p1 * p1);
      for (int k = 0; k <= p; ++k)
      {
         for (int j = 0; j <= p; ++j)
         {
            for (int i = 0; i <= p; ++i)
            {
               ip.Set3(cp[i], cp[j], cp[k]);
               T->Transform(ip, pt);
               const std::size_t base =
                  static_cast<std::size_t>(3) * (i + j * p1 + k * p1 * p1);
               node_coords[base + 0] = pt(0);
               node_coords[base + 1] = pt(1);
               node_coords[base + 2] = pt(2);
            }
         }
      }

      sub_T.Attribute = attr;
      sub_T.ElementNo = T->ElementNo;

      auto map_signed_dof = [&](int lor_lex, int ho_lex)
      {
         const int i_lor = dofmap_lor[lor_lex];
         const int s1 = (i_lor < 0) ? -1 : 1;
         const int idx_lor = (i_lor < 0) ? (-1 - i_lor) : i_lor;
         const int i_ho = dofmap_ho[ho_lex];
         const int s2 = (i_ho < 0) ? -1 : 1;
         const int idx_ho = (i_ho < 0) ? (-1 - i_ho) : i_ho;
         const int v = vdofs[idx_ho];
         const int s3 = (v < 0) ? -1 : 1;
         const int g = (v < 0) ? (-1 - v) : v;
         const int s = s1 * s2 * s3;
         sub_vdofs[idx_lor] = (s < 0) ? (-1 - g) : g;
      };

      const int nd_dof3 = p * p1 * p1;
      const int rt_dof3 = p * p * p1;

      for (int oz = 0; oz < p; ++oz)
      {
         for (int oy = 0; oy < p; ++oy)
         {
            for (int ox = 0; ox < p; ++ox)
            {
               for (int v = 0; v < 8; ++v)
               {
                  const int ix = ox + kHexVx[v];
                  const int iy = oy + kHexVy[v];
                  const int iz = oz + kHexVz[v];
                  const std::size_t base =
                     static_cast<std::size_t>(3) * (ix + iy * p1 + iz * p1 * p1);
                  pm(0, v) = node_coords[base + 0];
                  pm(1, v) = node_coords[base + 1];
                  pm(2, v) = node_coords[base + 2];
               }
               sub_T.Reset();
               sub_elmat = 0.0;
               bool any_contrib = false;
               for (int i = 0; i < n_integ; ++i)
               {
                  if (markers[i] && (*markers[i])[attr - 1] == 0) { continue; }
                  dbfi[i]->AssembleElementMatrix(*sub_fe, sub_T, tmp_elmat);
                  sub_elmat += tmp_elmat;
                  any_contrib = true;
               }
               if (!any_contrib) { continue; }

               if (kind == LORSpaceKind::H1)
               {
                  for (int vz = 0; vz < 2; ++vz)
                  {
                     for (int vy = 0; vy < 2; ++vy)
                     {
                        for (int vx = 0; vx < 2; ++vx)
                        {
                           const int idx_lor = dofmap_lor[vx + 2 * vy + 4 * vz];
                           const int idx_ho = dofmap_ho[(ox + vx) + (oy + vy) * p1 +
                                                        (oz + vz) * p1 * p1];
                           sub_vdofs[idx_lor] = vdofs[idx_ho];
                        }
                     }
                  }
               }
               else if (kind == LORSpaceKind::ND)
               {
                  for (int i2 = 0; i2 < 2; ++i2)
                     for (int i1 = 0; i1 < 2; ++i1)
                     {
                        map_signed_dof(0 + i1 + 2 * i2,
                                       ox + (oy + i1) * p + (oz + i2) * p * p1);
                        map_signed_dof(4 + i1 + 2 * i2,
                                       nd_dof3 + (ox + i1) + oy * p1 + (oz + i2) * p * p1);
                        map_signed_dof(8 + i1 + 2 * i2,
                                       2 * nd_dof3 + (ox + i1) + (oy + i2) * p1 + oz * p1 * p1);
                     }
               }
               else // RT
               {
                  for (int s = 0; s < 2; ++s)
                  {
                     map_signed_dof(0 + s, 0 * rt_dof3 + (ox + s) + (oy + oz * p) * p1);
                     map_signed_dof(2 + s, 1 * rt_dof3 + ox + ((oy + s) + oz * p1) * p);
                     map_signed_dof(4 + s, 2 * rt_dof3 + ox + (oy + (oz + s) * p) * p);
                  }
               }

               A_loc.AddSubMatrix(sub_vdofs, sub_vdofs, sub_elmat);
            }
         }
      }
   }

   for (int i = 0; i < n_integ; ++i)
   {
      dbfi[i]->SetIntRule(saved_irs[i]);
   }

   A_loc.Finalize();
   ParBilinearForm blf_lor(&fes);
   HypreParMatrix *lmat = blf_lor.ParallelAssemble(&A_loc);
   lmat->EliminateBC(ess_tdof_list, Operator::DIAG_ONE);
   A.Reset(lmat);

   if (timings)
   {
      timings->lor_A_s = MPI_Wtime() - t0;
      timings->lor_nnz = lmat->NNZ();
   }
}

std::unique_ptr<HypreParMatrix> VarOrderLORAssembly::AssembleGradientOrPi(
   ParFiniteElementSpace &h1_fes,
   ParFiniteElementSpace &nd_fes,
   int dir)
{
   const int nrow = nd_fes.GetVSize();
   const int ncol = h1_fes.GetVSize();
   std::vector<uint8_t> seen(nrow, 0);
   std::vector<int> col0(nrow, -1), col1(nrow, -1);
   std::vector<real_t> val0(nrow, 0.0), val1(nrow, 0.0);

   Array<int> nd_vdofs, h1_vdofs;
   std::vector<real_t> coords;
   IntegrationPoint ip;
   Vector pt(3);

   for (int e = 0; e < nd_fes.GetNE(); ++e)
   {
      const FiniteElement *h1_el = h1_fes.GetFE(e);
      const FiniteElement *nd_el = nd_fes.GetFE(e);
      const auto *h1_tfe = dynamic_cast<const TensorBasisElement*>(h1_el);
      const auto *nd_tfe = dynamic_cast<const TensorBasisElement*>(nd_el);
      MFEM_VERIFY(h1_tfe && nd_tfe && h1_el->GetOrder() == nd_el->GetOrder(),
                  "AssembleGradientOrPi requires matching tensor-product H1/ND hex elements.");

      const int p = h1_el->GetOrder();
      const int p1 = p + 1;
      const int dof3 = p * p1 * p1;
      const Array<int> &h1_map = h1_tfe->GetDofMap();
      const Array<int> &nd_map = nd_tfe->GetDofMap();
      nd_fes.GetElementVDofs(e, nd_vdofs);
      h1_fes.GetElementVDofs(e, h1_vdofs);

      if (dir >= 0)
      {
         ElementTransformation *T = nd_fes.GetElementTransformation(e);
         const real_t *cp = poly1d.ClosedPoints(p, BasisType::GaussLobatto);
         coords.resize(static_cast<std::size_t>(p1) * p1 * p1);
         for (int k = 0; k <= p; ++k)
         {
            for (int j = 0; j <= p; ++j)
            {
               for (int i = 0; i <= p; ++i)
               {
                  ip.Set3(cp[i], cp[j], cp[k]);
                  T->Transform(ip, pt);
                  coords[i + j * p1 + k * p1 * p1] = pt(dir);
               }
            }
         }
      }

      auto set_edge = [&](int nd_lex, int h1_lex0, int h1_lex1)
      {
         const int e_s = nd_map[nd_lex];
         const int e_loc = (e_s < 0) ? (-1 - e_s) : e_s;
         const real_t s_m = (e_s < 0) ? -1.0 : 1.0;
         const int nd_v = nd_vdofs[e_loc];
         const int r = (nd_v < 0) ? (-1 - nd_v) : nd_v;
         if (seen[r]) { return; }
         const real_t s_r = (nd_v < 0) ? -1.0 : 1.0;

         const int h0_v = h1_vdofs[h1_map[h1_lex0]];
         const int c0 = (h0_v < 0) ? (-1 - h0_v) : h0_v;
         const real_t s_c0 = (h0_v < 0) ? -1.0 : 1.0;

         const int h1_v = h1_vdofs[h1_map[h1_lex1]];
         const int c1 = (h1_v < 0) ? (-1 - h1_v) : h1_v;
         const real_t s_c1 = (h1_v < 0) ? -1.0 : 1.0;

         seen[r] = 1;
         col0[r] = c0;
         col1[r] = c1;
         if (dir < 0)
         {
            const real_t s = s_r * s_m;
            val0[r] = -s * s_c0;
            val1[r] = s * s_c1;
         }
         else
         {
            const real_t w = 0.5 * s_r * s_m * (coords[h1_lex1] - coords[h1_lex0]);
            val0[r] = w * s_c0;
            val1[r] = w * s_c1;
         }
      };

      for (int k = 0; k <= p; ++k)
         for (int j = 0; j <= p; ++j)
            for (int i = 0; i < p; ++i)
               set_edge(0 * dof3 + i + (j + k * p1) * p,
                        i + j * p1 + k * p1 * p1,
                        (i + 1) + j * p1 + k * p1 * p1);

      for (int k = 0; k <= p; ++k)
         for (int j = 0; j < p; ++j)
            for (int i = 0; i <= p; ++i)
               set_edge(1 * dof3 + i + (j + k * p) * p1,
                        i + j * p1 + k * p1 * p1,
                        i + (j + 1) * p1 + k * p1 * p1);

      for (int k = 0; k < p; ++k)
         for (int j = 0; j <= p; ++j)
            for (int i = 0; i <= p; ++i)
               set_edge(2 * dof3 + i + (j + k * p1) * p1,
                        i + j * p1 + k * p1 * p1,
                        i + j * p1 + (k + 1) * p1 * p1);
   }

   std::vector<int> I(static_cast<std::size_t>(nrow) + 1, 0);
   int nnz = 0;
   for (int r = 0; r < nrow; ++r)
   {
      if (seen[r])
      {
         if (col0[r] == col1[r])
         {
            if (val0[r] + val1[r] != 0.0) { ++nnz; }
         }
         else
         {
            if (val0[r] != 0.0) { ++nnz; }
            if (val1[r] != 0.0) { ++nnz; }
         }
      }
      I[r + 1] = nnz;
   }

   std::vector<int> J(nnz);
   std::vector<real_t> A(nnz);
   for (int r = 0; r < nrow; ++r)
   {
      if (!seen[r]) { continue; }
      int pos = I[r];
      if (col0[r] == col1[r])
      {
         const real_t v = val0[r] + val1[r];
         if (v != 0.0) { J[pos] = col0[r]; A[pos] = v; }
      }
      else if (col0[r] < col1[r])
      {
         if (val0[r] != 0.0) { J[pos] = col0[r]; A[pos] = val0[r]; ++pos; }
         if (val1[r] != 0.0) { J[pos] = col1[r]; A[pos] = val1[r]; ++pos; }
      }
      else
      {
         if (val1[r] != 0.0) { J[pos] = col1[r]; A[pos] = val1[r]; ++pos; }
         if (val0[r] != 0.0) { J[pos] = col0[r]; A[pos] = val0[r]; ++pos; }
      }
   }

   SparseMatrix mat(I.data(), J.empty() ? nullptr : J.data(),
                    A.empty() ? nullptr : A.data(),
                    nrow, ncol, false, false, true);
   std::unique_ptr<SparseMatrix> RA(Mult(*nd_fes.GetRestrictionMatrix(), mat));
   HypreParMatrix *P = h1_fes.Dof_TrueDof_Matrix();
   std::unique_ptr<HypreParMatrix> res(
      P->LeftDiagMult(*RA, nd_fes.GetTrueDofOffsets()));
   res->CopyRowStarts();
   res->CopyColStarts();
   return res;
}

std::unique_ptr<HypreParMatrix> VarOrderLORAssembly::AssembleCurl(
   ParFiniteElementSpace &nd_fes,
   ParFiniteElementSpace &rt_fes)
{
   const int nrow = rt_fes.GetVSize();
   std::vector<uint8_t> seen(nrow, 0);
   std::vector<std::array<int, 4>> cols(nrow, { -1, -1, -1, -1 });
   std::vector<std::array<real_t, 4>> vals(nrow, { 0.0, 0.0, 0.0, 0.0 });

   Array<int> rt_vdofs, nd_vdofs;

   for (int e = 0; e < rt_fes.GetNE(); ++e)
   {
      const FiniteElement *nd_el = nd_fes.GetFE(e);
      const FiniteElement *rt_el = rt_fes.GetFE(e);
      const auto *nd_tfe = dynamic_cast<const TensorBasisElement*>(nd_el);
      const auto *rt_tfe = dynamic_cast<const TensorBasisElement*>(rt_el);
      MFEM_VERIFY(nd_tfe && rt_tfe && nd_el->GetOrder() == rt_el->GetOrder(),
                  "AssembleCurl requires matching tensor-product ND/RT hex elements.");

      const int p = rt_el->GetOrder();
      const int p1 = p + 1;
      const int nd_dof3 = p * p1 * p1;
      const int rt_dof3 = p * p * p1;
      const Array<int> &nd_map = nd_tfe->GetDofMap();
      const Array<int> &rt_map = rt_tfe->GetDofMap();
      rt_fes.GetElementVDofs(e, rt_vdofs);
      nd_fes.GetElementVDofs(e, nd_vdofs);

      for (int c = 0; c < 3; ++c)
      {
         const int nx = (c == 0) ? p1 : p;
         const int ny = (c == 1) ? p1 : p;
         const int nz = (c == 2) ? p1 : p;
         const int c1 = (c + 1) % 3;
         const int c2 = (c + 2) % 3;
         const int nx_e1 = (c1 == 0) ? p : p1;
         const int ny_e1 = (c1 == 1) ? p : p1;
         const int nx_e2 = (c2 == 0) ? p : p1;
         const int ny_e2 = (c2 == 1) ? p : p1;

         for (int iz = 0; iz < nz; ++iz)
         {
            for (int iy = 0; iy < ny; ++iy)
            {
               for (int ix = 0; ix < nx; ++ix)
               {
                  const int rt_lex = c * rt_dof3 + ix + iy * nx + iz * nx * ny;
                  const int f_s = rt_map[rt_lex];
                  const int f_loc = (f_s < 0) ? (-1 - f_s) : f_s;
                  const real_t s_fm = (f_s < 0) ? -1.0 : 1.0;
                  const int f_v = rt_vdofs[f_loc];
                  const int r = (f_v < 0) ? (-1 - f_v) : f_v;
                  if (seen[r]) { continue; }
                  const real_t s_fr = (f_v < 0) ? -1.0 : 1.0;
                  const real_t s_face = s_fr * s_fm;

                  int i1[3] = { ix, iy, iz };
                  int i2[3] = { ix, iy, iz };
                  ++i1[c2];
                  ++i2[c1];

                  const int nd_lex[4] =
                  {
                     c1 * nd_dof3 + ix + iy * nx_e1 + iz * nx_e1 * ny_e1,
                     c1 * nd_dof3 + i1[0] + i1[1] * nx_e1 + i1[2] * nx_e1 * ny_e1,
                     c2 * nd_dof3 + ix + iy * nx_e2 + iz * nx_e2 * ny_e2,
                     c2 * nd_dof3 + i2[0] + i2[1] * nx_e2 + i2[2] * nx_e2 * ny_e2
                  };
                  static constexpr real_t kEdgeSign[4] = { 1.0, -1.0, -1.0, 1.0 };

                  seen[r] = 1;
                  for (int k = 0; k < 4; ++k)
                  {
                     const int e_s = nd_map[nd_lex[k]];
                     const int e_loc = (e_s < 0) ? (-1 - e_s) : e_s;
                     const real_t s_em = (e_s < 0) ? -1.0 : 1.0;
                     const int e_v = nd_vdofs[e_loc];
                     const int col = (e_v < 0) ? (-1 - e_v) : e_v;
                     const real_t s_ec = (e_v < 0) ? -1.0 : 1.0;
                     cols[r][k] = col;
                     vals[r][k] = kEdgeSign[k] * s_face * (s_ec * s_em);
                  }
               }
            }
         }
      }
   }

   return BuildParallel4NnzOp(rt_fes, nd_fes, seen, cols, vals);
}

std::unique_ptr<HypreParMatrix> VarOrderLORAssembly::AssembleRTPi(
   ParFiniteElementSpace &h1_fes,
   ParFiniteElementSpace &rt_fes,
   int dir)
{
   MFEM_VERIFY(dir >= 0 && dir < 3, "AssembleRTPi requires dir in {0, 1, 2}.");
   const int nrow = rt_fes.GetVSize();
   std::vector<uint8_t> seen(nrow, 0);
   std::vector<std::array<int, 4>> cols(nrow, { -1, -1, -1, -1 });
   std::vector<std::array<real_t, 4>> vals(nrow, { 0.0, 0.0, 0.0, 0.0 });

   Array<int> rt_vdofs, h1_vdofs;
   std::vector<real_t> node_coords;
   IntegrationPoint ip;
   Vector pt(3);

   for (int e = 0; e < rt_fes.GetNE(); ++e)
   {
      const FiniteElement *h1_el = h1_fes.GetFE(e);
      const FiniteElement *rt_el = rt_fes.GetFE(e);
      const auto *h1_tfe = dynamic_cast<const TensorBasisElement*>(h1_el);
      const auto *rt_tfe = dynamic_cast<const TensorBasisElement*>(rt_el);
      MFEM_VERIFY(h1_tfe && rt_tfe && h1_el->GetOrder() == rt_el->GetOrder(),
                  "AssembleRTPi requires matching tensor-product H1/RT hex elements.");

      const int p = rt_el->GetOrder();
      const int p1 = p + 1;
      const int rt_dof3 = p * p * p1;
      const Array<int> &h1_map = h1_tfe->GetDofMap();
      const Array<int> &rt_map = rt_tfe->GetDofMap();
      rt_fes.GetElementVDofs(e, rt_vdofs);
      h1_fes.GetElementVDofs(e, h1_vdofs);

      ElementTransformation *T = rt_fes.GetElementTransformation(e);
      const real_t *cp = poly1d.ClosedPoints(p, BasisType::GaussLobatto);
      node_coords.resize(static_cast<std::size_t>(3) * p1 * p1 * p1);
      for (int k = 0; k <= p; ++k)
      {
         for (int j = 0; j <= p; ++j)
         {
            for (int i = 0; i <= p; ++i)
            {
               ip.Set3(cp[i], cp[j], cp[k]);
               T->Transform(ip, pt);
               const std::size_t base =
                  static_cast<std::size_t>(3) * (i + j * p1 + k * p1 * p1);
               node_coords[base + 0] = pt(0);
               node_coords[base + 1] = pt(1);
               node_coords[base + 2] = pt(2);
            }
         }
      }

      auto h1_lex = [&](const int idx[3]) -> int
      {
         return idx[0] + idx[1] * p1 + idx[2] * p1 * p1;
      };

      for (int c = 0; c < 3; ++c)
      {
         const int nx = (c == 0) ? p1 : p;
         const int ny = (c == 1) ? p1 : p;
         const int nz = (c == 2) ? p1 : p;
         const int c1 = (c + 1) % 3;
         const int c2 = (c + 2) % 3;

         for (int iz = 0; iz < nz; ++iz)
         {
            for (int iy = 0; iy < ny; ++iy)
            {
               for (int ix = 0; ix < nx; ++ix)
               {
                  const int rt_lex = c * rt_dof3 + ix + iy * nx + iz * nx * ny;
                  const int f_s = rt_map[rt_lex];
                  const int f_loc = (f_s < 0) ? (-1 - f_s) : f_s;
                  const real_t s_fm = (f_s < 0) ? -1.0 : 1.0;
                  const int f_v = rt_vdofs[f_loc];
                  const int r = (f_v < 0) ? (-1 - f_v) : f_v;
                  if (seen[r]) { continue; }
                  const real_t s_fr = (f_v < 0) ? -1.0 : 1.0;
                  const real_t s_face = s_fr * s_fm;

                  int idx[4][3] =
                  {
                     { ix, iy, iz },
                     { ix, iy, iz },
                     { ix, iy, iz },
                     { ix, iy, iz }
                  };
                  ++idx[1][c1];
                  ++idx[2][c1];
                  ++idx[2][c2];
                  ++idx[3][c2];

                  int vlex[4];
                  const real_t *v[4];
                  for (int k = 0; k < 4; ++k)
                  {
                     vlex[k] = h1_lex(idx[k]);
                     v[k] = &node_coords[static_cast<std::size_t>(3) * vlex[k]];
                  }

                  real_t e01[3], e03[3], e12[3], e32[3];
                  for (int d = 0; d < 3; ++d)
                  {
                     e01[d] = v[1][d] - v[0][d];
                     e03[d] = v[3][d] - v[0][d];
                     e12[d] = v[2][d] - v[1][d];
                     e32[d] = v[2][d] - v[3][d];
                  }
                  real_t n_corner[4][3];
                  Cross3(e01, e03, n_corner[0]);
                  Cross3(e01, e12, n_corner[1]);
                  Cross3(e32, e12, n_corner[2]);
                  Cross3(e32, e03, n_corner[3]);

                  seen[r] = 1;
                  for (int k = 0; k < 4; ++k)
                  {
                     const int h_v = h1_vdofs[h1_map[vlex[k]]];
                     const int col = (h_v < 0) ? (-1 - h_v) : h_v;
                     const real_t s_c = (h_v < 0) ? -1.0 : 1.0;
                     cols[r][k] = col;
                     vals[r][k] = 0.25 * s_face * s_c * n_corner[k][dir];
                  }
               }
            }
         }
      }
   }

   return BuildParallel4NnzOp(rt_fes, h1_fes, seen, cols, vals);
}

std::unique_ptr<HypreAMS> VarOrderLORAssembly::BuildAMS(
   ParBilinearForm &a_ho,
   const Array<int> &ess_tdof_list,
   OperatorHandle &A,
   VarOrderLORTimings *timings)
{
   ParFiniteElementSpace &nd_fes = *a_ho.ParFESpace();
   ParMesh *pmesh = nd_fes.GetParMesh();

   int loc_max_p = std::max(1, nd_fes.GetMaxElementOrder());
   int max_p = 1;
   MPI_Allreduce(&loc_max_p, &max_p, 1, MPI_INT, MPI_MAX, pmesh->GetComm());

   const double t_aux0 = MPI_Wtime();
   H1_FECollection h1_fec(max_p, 3, BasisType::GaussLobatto);
   ParFiniteElementSpace h1_fes(pmesh, &h1_fec);
   if (nd_fes.IsVariableOrder())
   {
      for (int e = 0; e < pmesh->GetNE(); ++e)
      {
         h1_fes.SetElementOrder(e, nd_fes.GetElementOrder(e));
      }
      h1_fes.Update(false);
   }
   const double t_aux1 = MPI_Wtime();

   Assemble(a_ho, ess_tdof_list, A, timings);

   const double t_ops0 = MPI_Wtime();
   auto G = AssembleGradientOrPi(h1_fes, nd_fes, -1);
   auto Pix = AssembleGradientOrPi(h1_fes, nd_fes, 0);
   auto Piy = AssembleGradientOrPi(h1_fes, nd_fes, 1);
   auto Piz = AssembleGradientOrPi(h1_fes, nd_fes, 2);
   const double t_ops1 = MPI_Wtime();

   const HYPRE_BigInt g_nnz = G->NNZ();
   const HYPRE_BigInt pi_nnz = Pix->NNZ() + Piy->NNZ() + Piz->NNZ();

   const double t_hy0 = MPI_Wtime();
   std::unique_ptr<HypreAMS> ams(
      new HypreAMS(*A.As<HypreParMatrix>(),
                   G.release(), Pix.release(), Piy.release(), Piz.release()));
   ams->SetPrintLevel(0);
   const double t_hy1 = MPI_Wtime();

   if (timings)
   {
      timings->aux_space_s = t_aux1 - t_aux0;
      timings->lor_aux_ops_s = t_ops1 - t_ops0;
      timings->hypre_init_s = t_hy1 - t_hy0;
      timings->g_nnz = g_nnz;
      timings->pi_nnz = pi_nnz;
   }
   return ams;
}

std::unique_ptr<HypreADS> VarOrderLORAssembly::BuildADS(
   ParBilinearForm &a_ho,
   const Array<int> &ess_tdof_list,
   OperatorHandle &A,
   VarOrderLORTimings *timings)
{
   ParFiniteElementSpace &rt_fes = *a_ho.ParFESpace();
   ParMesh *pmesh = rt_fes.GetParMesh();

   int loc_max_p = std::max(1, rt_fes.GetMaxElementOrder());
   int max_p = 1;
   MPI_Allreduce(&loc_max_p, &max_p, 1, MPI_INT, MPI_MAX, pmesh->GetComm());

   const double t_aux0 = MPI_Wtime();
   H1_FECollection h1_fec(max_p, 3, BasisType::GaussLobatto);
   ND_FECollection nd_fec(max_p, 3, BasisType::GaussLobatto,
                          BasisType::IntegratedGLL);
   ParFiniteElementSpace h1_fes(pmesh, &h1_fec);
   ParFiniteElementSpace nd_fes(pmesh, &nd_fec);
   if (rt_fes.IsVariableOrder())
   {
      for (int e = 0; e < pmesh->GetNE(); ++e)
      {
         const int p = rt_fes.GetElementOrder(e);
         h1_fes.SetElementOrder(e, p);
         nd_fes.SetElementOrder(e, p);
      }
      h1_fes.Update(false);
      nd_fes.Update(false);
   }
   const double t_aux1 = MPI_Wtime();

   Assemble(a_ho, ess_tdof_list, A, timings);

   const double t_ops0 = MPI_Wtime();
   auto C = AssembleCurl(nd_fes, rt_fes);
   auto G = AssembleGradientOrPi(h1_fes, nd_fes, -1);
   auto ND_Pix = AssembleGradientOrPi(h1_fes, nd_fes, 0);
   auto ND_Piy = AssembleGradientOrPi(h1_fes, nd_fes, 1);
   auto ND_Piz = AssembleGradientOrPi(h1_fes, nd_fes, 2);
   auto RT_Pix = AssembleRTPi(h1_fes, rt_fes, 0);
   auto RT_Piy = AssembleRTPi(h1_fes, rt_fes, 1);
   auto RT_Piz = AssembleRTPi(h1_fes, rt_fes, 2);
   const double t_ops1 = MPI_Wtime();

   const HYPRE_BigInt c_nnz = C->NNZ();
   const HYPRE_BigInt g_nnz = G->NNZ();
   const HYPRE_BigInt pi_nnz = ND_Pix->NNZ() + ND_Piy->NNZ() + ND_Piz->NNZ() +
                               RT_Pix->NNZ() + RT_Piy->NNZ() + RT_Piz->NNZ();

   const double t_hy0 = MPI_Wtime();
   std::unique_ptr<HypreADS> ads(
      new HypreADS(*A.As<HypreParMatrix>(),
                   C.release(), G.release(),
                   ND_Pix.release(), ND_Piy.release(), ND_Piz.release(),
                   RT_Pix.release(), RT_Piy.release(), RT_Piz.release()));
   ads->SetPrintLevel(0);
   const double t_hy1 = MPI_Wtime();

   if (timings)
   {
      timings->aux_space_s = t_aux1 - t_aux0;
      timings->lor_aux_ops_s = t_ops1 - t_ops0;
      timings->hypre_init_s = t_hy1 - t_hy0;
      timings->c_nnz = c_nnz;
      timings->g_nnz = g_nnz;
      timings->pi_nnz = pi_nnz;
   }
   return ads;
}

} // namespace mfem

#endif // MFEM_USE_MPI
