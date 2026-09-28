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
#include "mma.hpp"
#include "../../qfunction.hpp"
#include "../../fe/fe_rt.hpp"

namespace mfem
{

namespace
{

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

void ProjectVecFeCoeff(Coefficient *Q, DiagonalMatrixCoefficient *DQ,
                       MatrixCoefficient *MQ, CoefficientVector &coeff)
{
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }
}

} // namespace

void VectorFEMassIntegrator::AssembleSimplexMmaHdivPA(
   const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
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
   internal::PAJTQJSetupSimplexFromNodes(
      dim, coeff_dim, ne, nq, nd_n, ir.GetWeights(), *G, nodes_e, coeff, pa_data);

   simplex_nd = el.GetDof();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

void DivDivIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
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

   BuildRtRefDivShape(el, ir, simplex_B);

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);
   if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }
   pa_data.SetSize(ir.GetNPoints() * ne, mt);

   Vector nodes_e;
   const Array<real_t> *G = nullptr;
   int nd_n = 0;
   internal::GetSimplexSetupGeom(*mesh, ir, mt, nodes_e, G, nd_n);
   internal::PAInvDetJSetupSimplexFromNodes(
      dim, coeff.GetVDim(), ne, ir.GetNPoints(), nd_n, ir.GetWeights(), *G,
      nodes_e, coeff, pa_data);

   simplex_nd = el.GetDof();
   nq = simplex_nq = ir.GetNPoints();
   simplex_sdim = dim;
   simplex_curl_dim = 0;
}

} // namespace mfem
