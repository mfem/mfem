// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "quadinterpolator.hpp"
#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const bool use_tensor_products):

   fespace(&fes),
   qspace(nullptr),
   IntRule(&ir),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(use_tensor_products)
{

   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const QuadratureSpace &qs,
                                               const bool use_tensor_products):

   fespace(&fes),
   qspace(&qs),
   IntRule(nullptr),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(use_tensor_products)
{
   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

template<>
void QuadratureInterpolator::Mult<QVectorLayout::UNSET>(
   const Vector &e_vec,
   unsigned eval_flags,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det) const
{
   dbg("%s", q_layout == QVectorLayout::byVDIM ? "byVDIM" : "byNODES");
   dbg("use_tensor_products: %s", use_tensor_products ? "true" : "false");

   if (q_layout == QVectorLayout::byVDIM)
   { Mult<QVectorLayout::byVDIM>(e_vec, eval_flags, q_val, q_der, q_det); }

   if (q_layout == QVectorLayout::byNODES)
   {
      if (use_tensor_products)
      { MultByNodesTensor(e_vec, eval_flags, q_val, q_der, q_det); }
      else
      { Mult<QVectorLayout::byNODES>(e_vec, eval_flags, q_val, q_der, q_det); }
   }
}

void QuadratureInterpolator::MultTranspose(
   unsigned eval_flags, const Vector &q_val, const Vector &q_der,
   Vector &e_vec) const
{
   MFEM_ABORT("this method is not implemented yet");
}

template<>
void QuadratureInterpolator::Values<QVectorLayout::UNSET>(
   const Vector &e_vec, Vector &q_val) const
{
   if (q_layout == QVectorLayout::byNODES)
   { Values<QVectorLayout::byNODES>(e_vec, q_val); }

   if (q_layout == QVectorLayout::byVDIM)
   { Values<QVectorLayout::byVDIM>(e_vec, q_val); }
}

template<>
void QuadratureInterpolator::Derivatives<QVectorLayout::UNSET>(
   const Vector &e_vec, Vector &q_der) const
{
   if (q_layout == QVectorLayout::byNODES)
   { Derivatives<QVectorLayout::byNODES>(e_vec, q_der); }

   if (q_layout == QVectorLayout::byVDIM)
   { Derivatives<QVectorLayout::byVDIM>(e_vec, q_der); }
}

template<>
void QuadratureInterpolator::PhysDerivatives<QVectorLayout::UNSET>(
   const Vector &e_vec, Vector &q_der) const
{
   if (q_layout == QVectorLayout::byNODES)
   { PhysDerivatives<QVectorLayout::byNODES>(e_vec, q_der); }

   if (q_layout == QVectorLayout::byVDIM)
   { PhysDerivatives<QVectorLayout::byVDIM>(e_vec, q_der); }
}

} // namespace mfem
