// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "basis.hpp"

#include "util.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static CeedElemTopology GetCeedTopology(Geometry::Type geom)
{
   switch (geom)
   {
      case Geometry::SEGMENT:
         return CEED_TOPOLOGY_LINE;
      case Geometry::TRIANGLE:
         return CEED_TOPOLOGY_TRIANGLE;
      case Geometry::SQUARE:
         return CEED_TOPOLOGY_QUAD;
      case Geometry::TETRAHEDRON:
         return CEED_TOPOLOGY_TET;
      case Geometry::CUBE:
         return CEED_TOPOLOGY_HEX;
      case Geometry::PRISM:
         return CEED_TOPOLOGY_PRISM;
      case Geometry::PYRAMID:
         return CEED_TOPOLOGY_PYRAMID;
      default:
         MFEM_ABORT("This type of element is not supported");
         return CEED_TOPOLOGY_PRISM; // Silence warning
   }
}

static void InitNonTensorBasis(const mfem::FiniteElementSpace &fes,
                               const mfem::FiniteElement &fe,
                               const mfem::IntegrationRule &ir,
                               Ceed ceed,
                               CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::FULL);
   const int dim = fe.GetDim();
   const int ncomp = fes.GetVDim();
   const int P = maps.ndof;
   const int Q = maps.nqpt;
   mfem::DenseMatrix qX(dim, Q);
   mfem::Vector qW(Q);
   for (int i = 0; i < Q; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX(0, i) = ip.x;
      if (dim > 1) { qX(1, i) = ip.y; }
      if (dim > 2) { qX(2, i) = ip.z; }
      qW(i) = ip.weight;
   }
   if (fe.GetMapType() == mfem::FiniteElement::H_DIV)
   {
      CeedBasisCreateHdiv(ceed, GetCeedTopology(fe.GetGeomType()), ncomp, P, Q,
                          maps.Bt.GetData(), maps.Gt.GetData(),
                          qX.GetData(), qW.GetData(), basis);
   }
   else if (fe.GetMapType() == mfem::FiniteElement::H_CURL)
   {
      CeedBasisCreateHcurl(ceed, GetCeedTopology(fe.GetGeomType()), ncomp, P, Q,
                           maps.Bt.GetData(), maps.Gt.GetData(),
                           qX.GetData(), qW.GetData(), basis);
   }
   else
   {
      CeedBasisCreateH1(ceed, GetCeedTopology(fe.GetGeomType()), ncomp, P, Q,
                        maps.Bt.GetData(), maps.Gt.GetData(),
                        qX.GetData(), qW.GetData(), basis);
   }
}

static void InitTensorBasis(const mfem::FiniteElementSpace &fes,
                            const mfem::FiniteElement &fe,
                            const mfem::IntegrationRule &ir,
                            Ceed ceed,
                            CeedBasis *basis)
{
   const mfem::DofToQuad &maps = fe.GetDofToQuad(ir, mfem::DofToQuad::TENSOR);
   const int dim = fe.GetDim();
   const int ncomp = fes.GetVDim();
   const int P = maps.ndof;
   const int Q = maps.nqpt;
   mfem::Vector qX(Q), qW(Q);
   // The x-coordinates of the first `Q` points of the integration rule are
   // the points of the corresponding 1D rule. We also scale the weights
   // accordingly.
   double w_sum = 0.0;
   for (int i = 0; i < Q; i++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qX(i) = ip.x;
      qW(i) = ip.weight;
      w_sum += ip.weight;
   }
   qW *= 1.0 / w_sum;
   CeedBasisCreateTensorH1(ceed, dim, ncomp, P, Q,
                           maps.Bt.GetData(), maps.Gt.GetData(),
                           qX.GetData(), qW.GetData(), basis);
}

#if 0
static void InitCeedInterpolatorBasis(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes,
                                      const FiniteElement &trial_fe,
                                      const FiniteElement &test_fe,
                                      Ceed ceed,
                                      CeedBasis *basis)
{
   // Basis projection operator using libCEED
   CeedBasis trial_basis, test_basis;
   const int P = std::max(trial_fe.GetDof(), test_fe.GetDof()), ir_order_max = 100;
   int ir_order = std::max(trial_fe.GetOrder(), test_fe.GetOrder());
   for (; ir_order < ir_order_max; ir_order++)
   {
      if (IntRules.Get(trial_fe.GetGeomType(), ir_order).GetNPoints() >= P) { break; }
   }
   const IntegrationRule &ir = IntRules.Get(trial_fe.GetGeomType(), ir_order);


   // //XX TODO DEBUG
   // std::cout << "\nlibCEED Basis projection:\n" <<
   //              "    Q = " << ir.GetNPoints() << "\n" <<
   //              "    P = " << test_fe.GetDof() << " and " << trial_fe.GetDof() << "\n";


   InitBasis(trial_fes, trial_fe, ir, ceed, &trial_basis);
   InitBasis(test_fes, test_fe, ir, ceed, &test_basis);


   // //XX TODO TESTING
   // std::cout << "\nBASIS C:\n\n";
   // CeedBasisView(trial_basis, stdout);
   // std::cout << "\nBASIS F:\n\n";
   // CeedBasisView(test_basis, stdout);


   CeedBasisCreateProjection(trial_basis, test_basis, basis);


   // //XX TODO TESTING
   // std::cout << "\nBASIS C TO F:\n\n";
   // CeedBasisView(*basis, stdout);


}
#endif

static void InitMfemInterpolatorBasis(const FiniteElementSpace &trial_fes,
                                      const FiniteElementSpace &test_fes,
                                      const FiniteElement &trial_fe,
                                      const FiniteElement &test_fe,
                                      Ceed ceed,
                                      CeedBasis *basis)
{
   MFEM_VERIFY(trial_fes.GetVDim() == test_fes.GetVDim(),
               "libCEED discrete linear operator requires same vdim for trial "
               "and test FE spaces.");
   const int dim = trial_fe.GetDim();
   const int ncomp = trial_fes.GetVDim();
   const int trial_P = trial_fe.GetDof();
   const int test_P = test_fe.GetDof();
   mfem::DenseMatrix qX(dim, test_P), Gt(trial_P, test_P * dim), Bt;
   mfem::Vector qW(test_P);
   mfem::IsoparametricTransformation dummy;
   dummy.SetIdentityTransformation(trial_fe.GetGeomType());
   if (trial_fe.GetMapType() == test_fe.GetMapType())
   {
      // Prolongation
      test_fe.GetTransferMatrix(trial_fe, dummy, Bt);
   }
   else if (trial_fe.GetMapType() == mfem::FiniteElement::VALUE &&
            test_fe.GetMapType() == mfem::FiniteElement::H_CURL)
   {
      // Discrete gradient interpolator
      test_fe.ProjectGrad(trial_fe, dummy, Bt);
   }
   else if (trial_fe.GetMapType() == mfem::FiniteElement::H_CURL &&
            test_fe.GetMapType() == mfem::FiniteElement::H_DIV)
   {
      // Discrete curl interpolator
      test_fe.ProjectCurl(trial_fe, dummy, Bt);
   }
   else if (trial_fe.GetMapType() == mfem::FiniteElement::H_DIV &&
            test_fe.GetMapType() == mfem::FiniteElement::INTEGRAL)
   {
      // Discrete divergence interpolator
      test_fe.ProjectDiv(trial_fe, dummy, Bt);
   }
   else
   {
      MFEM_ABORT("Unsupported trial/test FE spaces for libCEED discrete "
                 "linear operator");
   }
   Bt.Transpose();
   Gt = 0.0;
   qX = 0.0;
   qW = 0.0;


   // //XX TODO TESTING
   // std::cout << "\nBASIS C TO F:\n\n";
   // Bt.Print();


   CeedBasisCreateH1(ceed, GetCeedTopology(trial_fe.GetGeomType()), ncomp,
                     trial_P, test_P, Bt.GetData(), Gt.GetData(),
                     qX.GetData(), qW.GetData(), basis);
}

void InitBasis(const FiniteElementSpace &fes,
               const FiniteElement &fe,
               const IntegrationRule &ir,
               Ceed ceed,
               CeedBasis *basis)
{
   // Check for fes -> basis in hash table
   const int ncomp = fes.GetVDim();
   const int P = fe.GetDof();
   const int Q = ir.GetNPoints();
   BasisKey basis_key(&fes, nullptr, &ir, {ncomp, P, Q});
   auto basis_itr = mfem::internal::ceed_basis_map.find(basis_key);

   // Init or retrieve key values
   if (basis_itr == mfem::internal::ceed_basis_map.end())
   {
      const bool tensor =
         dynamic_cast<const mfem::TensorBasisElement *>(&fe) != nullptr;
      const bool vector = fe.GetRangeType() == mfem::FiniteElement::VECTOR;
      if (tensor && !vector)
      {
         InitTensorBasis(fes, fe, ir, ceed, basis);
      }
      else
      {
         InitNonTensorBasis(fes, fe, ir, ceed, basis);
      }
      mfem::internal::ceed_basis_map[basis_key] = *basis;
   }
   else
   {
      *basis = basis_itr->second;
   }
}

void InitInterpolatorBasis(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes,
                           const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           Ceed ceed,
                           CeedBasis *basis)
{
   // Check for fes -> basis in hash table
   const int ncomp = trial_fes.GetVDim() + test_fes.GetVDim();
   const int P = trial_fe.GetDof();
   const int Q = test_fe.GetDof();
   BasisKey basis_key(&trial_fes, &test_fes, nullptr, {ncomp, P, Q});
   auto basis_itr = mfem::internal::ceed_basis_map.find(basis_key);

   // Init or retrieve key values
   if (basis_itr == mfem::internal::ceed_basis_map.end())
   {
#if 0
      if (trial_fe.GetMapType() == test_fe.GetMapType())
      {
         InitCeedInterpolatorBasis(trial_fes, test_fes, trial_fe, test_fe,
                                   ceed, basis);
      }
      else
#endif
      {
         InitMfemInterpolatorBasis(trial_fes, test_fes, trial_fe, test_fe,
                                   ceed, basis);
      }
      mfem::internal::ceed_basis_map[basis_key] = *basis;
   }
   else
   {
      *basis = basis_itr->second;
   }
}

#endif

} // namespace ceed

} // namespace mfem
