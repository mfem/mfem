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

namespace mfem
{

namespace internal
{

void MmaHcurlMassApplySimplex(const int dim, const int NE, const int nd,
                              const int nq, const int sdim,
                              const bool symmetric,
                              const Array<real_t> &B,
                              const Vector &pa_data,
                              const Vector &x, Vector &y)
{
   // EA mode: B empty, pa_data is (nd, nd, NE) element matrices
   if (B.Size() == 0)
   {
      const auto Em = Reshape(pa_data.Read(), nd, nd, NE);
      const auto X = Reshape(x.Read(), nd, NE);
      auto Y = Reshape(y.ReadWrite(), nd, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int j = 0; j < nd; ++j) { s += Em(i, j, e) * X(j, e); }
            Y(i, e) += s;
         }
      });
      return;
   }

   // B: (nq, nd, sdim, NE) physical vector shapes (Piola already applied)
   // pa_data: (nq, ncomp, NE) physical mass metric Q*w*|J|
   const int ncomp = symmetric ? (sdim * (sdim + 1)) / 2 : sdim * sdim;
   const bool per_elem_B = (B.Size() == nq * nd * sdim * NE);
   const auto Bb4 = Reshape(B.Read(), nq, nd, sdim, per_elem_B ? NE : 1);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      const int be = per_elem_B ? e : 0;
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < sdim; ++c)
         {
            real_t s = 0.0;
            for (int i = 0; i < nd; ++i) { s += Bb4(q, i, c, be) * X(i, e); }
            u[c] = s;
         }

         real_t v[3] = {};
         if (sdim == 2)
         {
            const real_t O11 = D(q, 0, e);
            const real_t O21 = D(q, 1, e);
            const real_t O12 = symmetric ? O21 : D(q, 2, e);
            const real_t O22 = symmetric ? D(q, 2, e) : D(q, 3, e);
            v[0] = O11 * u[0] + O12 * u[1];
            v[1] = O21 * u[0] + O22 * u[1];
         }
         else
         {
            real_t A[3][3];
            if (symmetric)
            {
               A[0][0] = D(q, 0, e);
               A[1][0] = A[0][1] = D(q, 1, e);
               A[2][0] = A[0][2] = D(q, 2, e);
               A[1][1] = D(q, 3, e);
               A[2][1] = A[1][2] = D(q, 4, e);
               A[2][2] = D(q, 5, e);
            }
            else
            {
               for (int i = 0; i < 3; ++i)
                  for (int j = 0; j < 3; ++j)
                  {
                     A[i][j] = D(q, i * 3 + j, e);
                  }
            }
            for (int i = 0; i < 3; ++i)
            {
               v[i] = A[i][0] * u[0] + A[i][1] * u[1] + A[i][2] * u[2];
            }
         }

         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < sdim; ++c) { s += Bb4(q, i, c, be) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

void MmaCurlCurlApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int curl_dim,
                             const bool /*symmetric*/,
                             const Array<real_t> &C,
                             const Vector &pa_data,
                             const Vector &x, Vector &y)
{
   // EA mode: C empty, pa_data is (nd, nd, NE)
   if (C.Size() == 0)
   {
      const auto Em = Reshape(pa_data.Read(), nd, nd, NE);
      const auto X = Reshape(x.Read(), nd, NE);
      auto Y = Reshape(y.ReadWrite(), nd, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int j = 0; j < nd; ++j) { s += Em(i, j, e) * X(j, e); }
            Y(i, e) += s;
         }
      });
      return;
   }

   // C: (nq, nd, curl_dim) reference curl shapes
   // pa_data layout:
   //   2D: (nq, NE) scalar metric  (Q * w / |J|)
   //   3D: (nq, 6 or 9, NE) metric acting on reference curl
   const auto Cc = Reshape(C.Read(), nq, nd, curl_dim);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   if (dim == 2)
   {
      MFEM_VERIFY(curl_dim == 1, "");
      const auto D = Reshape(pa_data.Read(), nq, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
      {
         for (int q = 0; q < nq; ++q)
         {
            real_t curl = 0.0;
            for (int i = 0; i < nd; ++i) { curl += Cc(q, i, 0) * X(i, e); }
            curl *= D(q, e);
            for (int i = 0; i < nd; ++i) { Y(i, e) += Cc(q, i, 0) * curl; }
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3 && curl_dim == 3, "");
   const int ncomp = pa_data.Size() / (nq * NE);
   const auto D = Reshape(pa_data.Read(), nq, ncomp, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         real_t u[3] = {};
         for (int c = 0; c < 3; ++c)
         {
            for (int i = 0; i < nd; ++i) { u[c] += Cc(q, i, c) * X(i, e); }
         }
         real_t v[3] = {};
         if (ncomp == 6)
         {
            const real_t a00 = D(q, 0, e), a10 = D(q, 1, e), a20 = D(q, 2, e);
            const real_t a11 = D(q, 3, e), a21 = D(q, 4, e), a22 = D(q, 5, e);
            v[0] = a00 * u[0] + a10 * u[1] + a20 * u[2];
            v[1] = a10 * u[0] + a11 * u[1] + a21 * u[2];
            v[2] = a20 * u[0] + a21 * u[1] + a22 * u[2];
         }
         else
         {
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j)
               {
                  v[i] += D(q, i * 3 + j, e) * u[j];
               }
         }
         for (int i = 0; i < nd; ++i)
         {
            real_t s = 0.0;
            for (int c = 0; c < 3; ++c) { s += Cc(q, i, c) * v[c]; }
            Y(i, e) += s;
         }
      }
   });
}

} // namespace internal

namespace
{

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

   // When the ND element has a DofTransformation (tet faces), bake FA element
   // matrices (with TransformDual) into pa_data and apply as dense EA.
   // Otherwise use physical-shape PA at Q.
   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
   if (el.GetDofTransformation() != nullptr)
   {
      pa_data.SetSize(simplex_nd * simplex_nd * ne, Device::GetMemoryType());
      auto Em = Reshape(pa_data.HostWrite(), simplex_nd, simplex_nd, ne);
      DenseMatrix elmat;
      Array<int> vdofs;
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation *T = fes.GetElementTransformation(e);
         const FiniteElement &fel = *fes.GetFE(e);
         AssembleElementMatrix(fel, *T, elmat);
         DofTransformation *dof_trans = fes.GetElementVDofs(e, vdofs);
         if (dof_trans) { dof_trans->TransformDual(elmat); }
         for (int i = 0; i < simplex_nd; ++i)
            for (int j = 0; j < simplex_nd; ++j)
            {
               Em(i, j, e) = elmat(i, j);
            }
      }
      simplex_B.SetSize(0); // EA mode marker
      symmetric = true;
      return;
   }

   BuildNdPhysVShape(fes, ir, simplex_B);
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

   if (el.GetDofTransformation() != nullptr)
   {
      pa_data.SetSize(simplex_nd * simplex_nd * ne, Device::GetMemoryType());
      auto Em = Reshape(pa_data.HostWrite(), simplex_nd, simplex_nd, ne);
      DenseMatrix elmat;
      Array<int> vdofs;
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation *T = fes.GetElementTransformation(e);
         const FiniteElement &fel = *fes.GetFE(e);
         AssembleElementMatrix(fel, *T, elmat);
         DofTransformation *dof_trans = fes.GetElementVDofs(e, vdofs);
         if (dof_trans) { dof_trans->TransformDual(elmat); }
         for (int i = 0; i < simplex_nd; ++i)
            for (int j = 0; j < simplex_nd; ++j)
            {
               Em(i, j, e) = elmat(i, j);
            }
      }
      simplex_B.SetSize(0);
      symmetric = true;
      return;
   }

   BuildNdRefCurlShape(el, ir, simplex_B);
   SetupCurlCurlPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
}

} // namespace mfem
