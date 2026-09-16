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

namespace mfem
{

namespace
{


/** Apply TransformDual on the nd-axis of a (nq,nd,ncomp[,NE]) basis so that
    B_eff @ x_E = B_native @ InvTransformPrimal(x_E) and the dual pullback
    matches FA TransformDual without an EA Mult path. */
void BakeNdDofTransformation(const FiniteElementSpace &fes,
                             Array<real_t> &B, int nq, int nd, int ncomp)
{
   const int NE = fes.GetNE();
   MFEM_VERIFY(B.Size() == nq * nd * ncomp * NE || B.Size() == nq * nd * ncomp,
               "unexpected simplex basis size");
   if (B.Size() == nq * nd * ncomp)
   {
      Array<real_t> B0 = B;
      B.SetSize(nq * nd * ncomp * NE);
      const auto Bin = Reshape(B0.HostRead(), nq, nd, ncomp);
      auto Bout = Reshape(B.HostWrite(), nq, nd, ncomp, NE);
      for (int e = 0; e < NE; ++e)
         for (int q = 0; q < nq; ++q)
            for (int c = 0; c < ncomp; ++c)
               for (int i = 0; i < nd; ++i)
               {
                  Bout(q, i, c, e) = Bin(q, i, c);
               }
   }
   // In-place transform on host memory
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

/** Pack physical ND vector shapes at IR into B: layout (nq, nd, sdim, NE). */
void BuildNdPhysVShape(const FiniteElementSpace &fes, const IntegrationRule &ir,
                       Array<real_t> &B)
{
   const FiniteElement &el = *fes.GetTypicalFE();
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int sdim = el.GetDim();
   const int NE = fes.GetNE();
   B.SetSize(nq * nd * sdim * NE);
   DenseMatrix vshape(nd, sdim);
   auto Bb = Reshape(B.HostWrite(), nq, nd, sdim, NE);
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcVShape(T, vshape); // physical Piola
         for (int i = 0; i < nd; ++i)
            for (int c = 0; c < sdim; ++c)
            {
               Bb(q, i, c, e) = vshape(i, c);
            }
      }
   }
}

/** Physical-space mass metric D = Q * w * |J| (FA VectorFEMass weight). */
void SetupHcurlMassPaSimplex(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             Coefficient *Q,
                             DiagonalMatrixCoefficient *DQ,
                             MatrixCoefficient *MQ,
                             bool &symmetric,
                             Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   const int symmDims = (dim * (dim + 1)) / 2;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }
   else
   {
      symmetric = true;
   }
   const int ncomp = symmetric ? symmDims : dim * dim;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);

   DenseMatrix M, A(dim);
   Vector Dvec;
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t wdet = ip.weight * T.Weight();
         A = 0.0;
         if (MQ)
         {
            MQ->Eval(M, T, ip);
            A = M;
            A *= wdet;
         }
         else if (DQ)
         {
            DQ->Eval(Dvec, T, ip);
            for (int i = 0; i < dim; ++i) { A(i, i) = Dvec(i) * wdet; }
         }
         else
         {
            const real_t qv = Q ? Q->Eval(T, ip) : real_t(1.0);
            for (int i = 0; i < dim; ++i) { A(i, i) = qv * wdet; }
         }

         if (symmetric)
         {
            if (dim == 2)
            {
               D(q, 0, e) = A(0, 0);
               D(q, 1, e) = A(1, 0);
               D(q, 2, e) = A(1, 1);
            }
            else
            {
               D(q, 0, e) = A(0, 0);
               D(q, 1, e) = A(1, 0);
               D(q, 2, e) = A(2, 0);
               D(q, 3, e) = A(1, 1);
               D(q, 4, e) = A(2, 1);
               D(q, 5, e) = A(2, 2);
            }
         }
         else
         {
            for (int i = 0; i < dim; ++i)
               for (int j = 0; j < dim; ++j)
               {
                  D(q, i * dim + j, e) = A(i, j);
               }
         }
      }
   }
}

/** Build reference curl shapes at IR: C(q,i,c). */
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

void SetupCurlCurlPaSimplex(const FiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            Coefficient *Q,
                            DiagonalMatrixCoefficient *DQ,
                            MatrixCoefficient *MQ,
                            bool &symmetric,
                            Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   symmetric = true;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }

   if (dim == 2)
   {
      // D(q,e) = Q * w / |J|  so that (ref_curl)^T D (ref_curl) matches FA
      pa_data.SetSize(nq * NE, Device::GetMemoryType());
      auto D = Reshape(pa_data.HostWrite(), nq, NE);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation &T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            real_t coeff = 1.0;
            if (Q) { coeff = Q->Eval(T, ip); }
            else if (DQ)
            {
               Vector d(1);
               DQ->Eval(d, T, ip);
               coeff = d(0);
            }
            else if (MQ)
            {
               DenseMatrix M(1);
               MQ->Eval(M, T, ip);
               coeff = M(0, 0);
            }
            D(q, e) = coeff * ip.weight / T.Weight();
         }
      }
      return;
   }

   // 3D: phys_curl = J * curl_ref / |J|
   // FA: w*|J|*Q * phys·phys = w/|J| * curl_ref^T (J^T Q J) curl_ref
   const int ncomp = symmetric ? 6 : 9;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);
   DenseMatrix M, Qm(3), JJ(3), tmp(3), A(3);
   Vector Dv;
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const DenseMatrix &J = T.Jacobian();
         Qm = 0.0;
         if (MQ)
         {
            MQ->Eval(M, T, ip);
            Qm = M;
         }
         else if (DQ)
         {
            DQ->Eval(Dv, T, ip);
            for (int i = 0; i < 3; ++i) { Qm(i, i) = Dv(i); }
         }
         else if (Q)
         {
            const real_t qv = Q->Eval(T, ip);
            for (int i = 0; i < 3; ++i) { Qm(i, i) = qv; }
         }
         else
         {
            for (int i = 0; i < 3; ++i) { Qm(i, i) = 1.0; }
         }
         // A = (w/|J|) * J^T Q J
         MultAtB(J, Qm, tmp); // J^T Q
         Mult(tmp, J, A);     // J^T Q J
         A *= (ip.weight / T.Weight());
         if (symmetric)
         {
            D(q, 0, e) = A(0, 0);
            D(q, 1, e) = A(1, 0);
            D(q, 2, e) = A(2, 0);
            D(q, 3, e) = A(1, 1);
            D(q, 4, e) = A(2, 1);
            D(q, 5, e) = A(2, 2);
         }
         else
         {
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
               {
                  D(q, i * 3 + j, e) = A(i, j);
               }
         }
      }
   }
}

} // namespace


void VectorFEMassIntegrator::AssembleSimplexMmaHcurlPA(
   const FiniteElementSpace &fes)
{
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

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;

   BuildNdPhysVShape(fes, ir, simplex_B);
   if (el.GetDofTransformation() != nullptr)
   {
      BakeNdDofTransformation(fes, simplex_B, nq, simplex_nd, simplex_sdim);
   }
   SetupHcurlMassPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
}

void CurlCurlIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
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

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = el.GetCurlDim();

   BuildNdRefCurlShape(el, ir, simplex_B);
   if (el.GetDofTransformation() != nullptr)
   {
      BakeNdDofTransformation(fes, simplex_B, nq, simplex_nd, simplex_curl_dim);
   }
   SetupCurlCurlPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
}

} // namespace mfem
