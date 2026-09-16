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

#include "hdiv.hpp"
#include "hcurl.hpp"
#include "mma.hpp"
#include "../../qfunction.hpp"
#include "../../fe/fe_rt.hpp"

namespace mfem
{

namespace internal
{

void MmaHdivMassApplySimplex(const int dim, const int NE, const int nd,
                             const int nq, const int sdim,
                             const bool symmetric,
                             const Array<real_t> &B,
                             const Vector &pa_data,
                             const Vector &x, Vector &y)
{
   // Same dense vec-mass apply as H(curl); Piola is absorbed in pa_data.
   MmaHcurlMassApplySimplex(dim, NE, nd, nq, sdim, symmetric, B, pa_data, x, y);
}

void MmaDivDivApplySimplex(const int NE, const int nd, const int nq,
                           const Array<real_t> &Div,
                           const Vector &pa_data,
                           const Vector &x, Vector &y)
{
   const auto Dd = Reshape(Div.Read(), nq, nd);
   const auto D = Reshape(pa_data.Read(), nq, NE);
   const auto X = Reshape(x.Read(), nd, NE);
   auto Y = Reshape(y.ReadWrite(), nd, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         real_t div = 0.0;
         for (int i = 0; i < nd; ++i) { div += Dd(q, i) * X(i, e); }
         div *= D(q, e);
         for (int i = 0; i < nd; ++i) { Y(i, e) += Dd(q, i) * div; }
      }
   });
}

} // namespace internal

namespace
{

void BuildRtRefVShape(const FiniteElement &el, const IntegrationRule &ir,
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

void BuildRtRefDivShape(const FiniteElement &el, const IntegrationRule &ir,
                        Array<real_t> &Div)
{
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   Div.SetSize(nq * nd);
   Vector dshape(nd);
   auto Dd = Reshape(Div.HostWrite(), nq, nd);
   for (int q = 0; q < nq; ++q)
   {
      el.CalcDivShape(ir.IntPoint(q), dshape);
      for (int i = 0; i < nd; ++i) { Dd(q, i) = dshape(i); }
   }
}

/** H(div) mass: phys = (1/|J|) J û  → metric w/|J| * J^T Q J on ref shapes. */
void SetupHdivMassPaSimplex(const FiniteElementSpace &fes,
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
   symmetric = true;
   if (MQ && !dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      symmetric = false;
   }
   const int ncomp = symmetric ? symmDims : dim * dim;
   pa_data.SetSize(ncomp * nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, ncomp, NE);

   DenseMatrix Qm(dim), tmp(dim), A(dim), M;
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
            for (int i = 0; i < dim; ++i) { Qm(i, i) = Dv(i); }
         }
         else if (Q)
         {
            const real_t qv = Q->Eval(T, ip);
            for (int i = 0; i < dim; ++i) { Qm(i, i) = qv; }
         }
         else
         {
            for (int i = 0; i < dim; ++i) { Qm(i, i) = 1.0; }
         }
         // A = (w/|J|) * J^T Q J
         MultAtB(J, Qm, tmp);
         Mult(tmp, J, A);
         A *= (ip.weight / T.Weight());

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

void SetupDivDivPaSimplex(const FiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          Coefficient *Q,
                          Vector &pa_data)
{
   Mesh *mesh = fes.GetMesh();
   const int NE = fes.GetNE();
   const int nq = ir.GetNPoints();
   // phys_div = ref_div / |J|; FA weight*|J|*Q*phys^2 = Q*w/|J| * ref^2
   pa_data.SetSize(nq * NE, Device::GetMemoryType());
   auto D = Reshape(pa_data.HostWrite(), nq, NE);
   for (int e = 0; e < NE; ++e)
   {
      ElementTransformation &T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t coeff = Q ? Q->Eval(T, ip) : real_t(1.0);
         D(q, e) = coeff * ip.weight / T.Weight();
      }
   }
}

} // namespace

void VectorFEMassIntegrator::AssembleSimplexMmaHdivPA(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::DIV, "");
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
   trial_fetype = test_fetype = FiniteElement::DIV;
   use_simplices_mma = true;
   use_tensors_mma = false;

   BuildRtRefVShape(el, ir, simplex_B);
   SetupHdivMassPaSimplex(fes, ir, Q, DQ, MQ, symmetric, pa_data);
   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

void DivDivIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetDerivType() == FiniteElement::DIV, "");
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir_ptr = IntRule ? IntRule :
                                   &MassIntegrator::GetRule(el, el, Trans);
   const IntegrationRule &ir = *ir_ptr;
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = 0;
   mapsO = mapsC = nullptr;
   geom = nullptr;
   use_simplices_mma = true;
   use_tensors_mma = false;
   // DivDivIntegrator has no nq member — store via pa_data geometry only
   BuildRtRefDivShape(el, ir, simplex_B);
   SetupDivDivPaSimplex(fes, ir, Q, pa_data);
   simplex_nd = el.GetDof();
   simplex_nq = ir.GetNPoints();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

} // namespace mfem
