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

#include "hcurl.hpp"
#include "mma.hpp"
#include "../../qfunction.hpp"
#include "../../fe/fe_nd.hpp"
#include "../../doftrans.hpp"

#include <limits>

namespace mfem
{

namespace
{

void ProjectVecFeCoeff(Coefficient *Q, DiagonalMatrixCoefficient *DQ,
                       MatrixCoefficient *MQ, CoefficientVector &coeff)
{
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }
}

/** Reference curl shapes at IR: C(q,i,c). Host FE eval, once. */
void BuildNdRefCurlShape(const FiniteElement &el, const IntegrationRule &ir,
                         Array<real_t> &C)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int cdim = el.GetCurlDim();
   C.SetSize(nq * nd * cdim);
   DenseMatrix cshape(nd, cdim);
   auto Cc = Reshape(C.HostWrite(), nq, nd, cdim);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcCurlShape(ir.IntPoint(q), cshape);
      for (int i = 0; i < nd; ++i)
         for (int c = 0; c < cdim; ++c)
         {
            Cc(q, i, c) = cshape(i, c);
         }
   }
}

} // namespace

namespace internal
{

void BuildRefVShape(const FiniteElement &el, const IntegrationRule &ir,
                    Array<real_t> &B)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int sdim = el.GetDim();
   B.SetSize(nq * nd * sdim);
   DenseMatrix vshape(nd, sdim);
   auto Bb = Reshape(B.HostWrite(), nq, nd, sdim);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcVShape(ir.IntPoint(q), vshape);
      for (int i = 0; i < nd; ++i)
         for (int c = 0; c < sdim; ++c)
         {
            Bb(q, i, c) = vshape(i, c);
         }
   }
}

void GatherNdFaceOrientations(const FiniteElementSpace &fes, Array<int> &fo)
{
   fo.SetSize(0);
   const FiniteElement &el = *fes.GetTypicalFE();
   const StatelessDofTransformation *sdt = el.GetDofTransformation();
   if (!sdt || sdt->IsIdentity()) { return; }

   const int NE = fes.GetNE();
   if (NE < 1) { return; }

   Array<int> vdofs;
   DofTransformation *dt0 = fes.GetElementVDofs(0, vdofs);
   if (!dt0) { return; }
   const Array<int> fo0 = dt0->GetFaceOrientations();
   const int nfaces = fo0.Size();
   if (nfaces < 1) { return; }

   fo.SetSize(nfaces * NE);
   fo.UseDevice(true);
   auto Fo = Reshape(fo.HostWrite(), nfaces, NE);
   for (int f = 0; f < nfaces; ++f) { Fo(f, 0) = fo0[f]; }
   for (int e = 1; e < NE; ++e)
   {
      DofTransformation *dt = fes.GetElementVDofs(e, vdofs);
      MFEM_VERIFY(dt, "ND face orientations missing on simplex element");
      const Array<int> &row = dt->GetFaceOrientations();
      MFEM_VERIFY(row.Size() == nfaces, "ND Fo size");
      for (int f = 0; f < nfaces; ++f) { Fo(f, e) = row[f]; }
   }
}

void ApplyNdDofTransEVector(NdDofTransOp op, const FiniteElementSpace &fes,
                            const Array<int> &fo, Vector &y)
{
   if (fo.Size() == 0) { return; }

   const int NE = fes.GetNE();
   const FiniteElement &el = *fes.GetTypicalFE();
   const int nd = el.GetDof();
   const int p = el.GetOrder();
   const int dim = fes.GetMesh()->Dimension();
   MFEM_VERIFY(y.Size() == nd * NE, "ApplyNdDofTransEVector size");
   MFEM_VERIFY(NE > 0 && fo.Size() % NE == 0, "ND Fo packing");
   const int nfaces = fo.Size() / NE;
   const int nedges = (dim == 2) ? 3 : 6;
   const int nedofs = p;
   const int ntdofs = p * (p - 1);
   if (ntdofs < 2) { return; }
   const int face_base = nedges * nedofs;
   const int npairs = ntdofs / 2;
   const int mode = static_cast<int>(op);

   const int nwork = nfaces * npairs;
   const auto Fo = Reshape(fo.Read(), nfaces, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);
   mfem::forall(nwork, NE, [=] MFEM_HOST_DEVICE (int w, int e)
   {
      const int f = w / npairs;
      const int i = w - f * npairs;
      const int ori = Fo(f, e);
      const int idx = face_base + f * ntdofs + 2 * i;
      real_t x0 = Y(idx, e), x1 = Y(idx + 1, e);
      NdApplyFacePair(mode, ori, x0, x1);
      Y(idx, e) = x0;
      Y(idx + 1, e) = x1;
   });
}

void TransformDualEVector(const FiniteElementSpace &fes, Vector &y)
{
   Array<int> fo;
   GatherNdFaceOrientations(fes, fo);
   ApplyNdDofTransEVector(NdDofTransOp::Dual, fes, fo, y);
}

void TransformPrimalEVector(const FiniteElementSpace &fes, Vector &y)
{
   Array<int> fo;
   GatherNdFaceOrientations(fes, fo);
   ApplyNdDofTransEVector(NdDofTransOp::Primal, fes, fo, y);
}

void InvTransformPrimalEVector(const FiniteElementSpace &fes, Vector &y)
{
   Array<int> fo;
   GatherNdFaceOrientations(fes, fo);
   ApplyNdDofTransEVector(NdDofTransOp::InvPrimal, fes, fo, y);
}

void InvTransformDualEVector(const FiniteElementSpace &fes, Vector &y)
{
   Array<int> fo;
   GatherNdFaceOrientations(fes, fo);
   ApplyNdDofTransEVector(NdDofTransOp::InvDual, fes, fo, y);
}

void BakeNdDofTransformation(const FiniteElementSpace &fes,
                             Array<real_t> &B, int nq, int nd, int ncomp)
{
   const int NE = fes.GetNE();
   MFEM_VERIFY(B.Size() == nq * nd * ncomp * NE || B.Size() == nq * nd * ncomp,
               "unexpected simplex basis size");
   if (B.Size() == nq * nd * ncomp)
   {
      const long long nB = (long long)nq * nd * ncomp * NE;
      // Array::SetSize is int; skip per-element bake when it cannot fit.
      if (nB > std::numeric_limits<int>::max()) { return; }
      Array<real_t> B0 = B;
      B.SetSize(static_cast<int>(nB));
      const auto Bin = Reshape(B0.Read(), nq, nd, ncomp);
      auto Bout = Reshape(B.Write(), nq, nd, ncomp, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int c = 0; c < ncomp; ++c)
            for (int i = 0; i < nd; ++i)
               for (int q = 0; q < nq; ++q)
               {
                  Bout(q, i, c, e) = Bin(q, i, c);
               }
      });
   }
   // TransformDual is a host DofTransformation API.
   auto Bb = Reshape(B.HostReadWrite(), nq, nd, ncomp, NE);
   Array<int> vdofs;
   Vector col(nd);
   for (int e = 0; e < NE; ++e)
   {
      DofTransformation *dt = fes.GetElementVDofs(e, vdofs);
      if (!dt) { continue; }
      for (int q = 0; q < nq; ++q)
         for (int c = 0; c < ncomp; ++c)
         {
            for (int i = 0; i < nd; ++i) { col(i) = Bb(q, i, c, e); }
            dt->TransformDual(col);
            for (int i = 0; i < nd; ++i) { Bb(q, i, c, e) = col(i); }
         }
   }
}

} // namespace internal

void VectorFEMassIntegrator::AssembleSimplexMmaHcurlPA(
   const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::CURL, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   nq = ir.GetNPoints();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = mapsOtest = mapsCtest = nullptr;
   geom = nullptr;
   trial_fetype = test_fetype = FiniteElement::CURL;
   use_simplices_mma = true;
   use_tensors_mma = false;
   simplex_fes = &fes;
   internal::GatherNdFaceOrientations(fes, simplex_nd_fo);

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;

   internal::BuildRefVShape(el, ir, simplex_B);

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   ProjectVecFeCoeff(Q, DQ, MQ, coeff);
   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim * dim);
   const int ncomp = symmetric ? (dim * (dim + 1)) / 2 : dim * dim;
   pa_data.SetSize(ncomp * nq * ne, mt);

   Vector nodes_e;
   const Array<real_t> *G = nullptr;
   int nd_n = 0;
   internal::GetSimplexSetupGeom(*mesh, ir, mt, nodes_e, G, nd_n);
   internal::PAJinvQJinvTSetupSimplexFromNodes(
      dim, coeff_dim, ne, nq, nd_n, ir.GetWeights(), *G, nodes_e, coeff, pa_data);
}

void CurlCurlIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::CURL, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   nq = ir.GetNPoints();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = nullptr;
   geom = nullptr;
   use_simplices_mma = true;
   use_tensors_mma = false;
   simplex_fes = &fes;
   internal::GatherNdFaceOrientations(fes, simplex_nd_fo);

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = el.GetCurlDim();

   BuildNdRefCurlShape(el, ir, simplex_B);

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   ProjectVecFeCoeff(Q, DQ, MQ, coeff);
   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim * dim);

   Vector nodes_e;
   const Array<real_t> *G = nullptr;
   int nd_n = 0;
   internal::GetSimplexSetupGeom(*mesh, ir, mt, nodes_e, G, nd_n);

   if (dim == 2)
   {
      pa_data.SetSize(nq * ne, mt);
      internal::PAInvDetJSetupSimplexFromNodes(
         dim, coeff_dim, ne, nq, nd_n, ir.GetWeights(), *G, nodes_e, coeff,
         pa_data);
      return;
   }

   const int ncomp = symmetric ? 6 : 9;
   pa_data.SetSize(ncomp * nq * ne, mt);
   internal::PAJTQJSetupSimplexFromNodes(
      dim, coeff_dim, ne, nq, nd_n, ir.GetWeights(), *G, nodes_e, coeff, pa_data);
}

} // namespace mfem
