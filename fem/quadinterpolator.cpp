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

void QuadratureInterpolator::Mult(const Vector &e_vec,
                                  unsigned eval_flags,
                                  Vector &q_val,
                                  Vector &q_der,
                                  Vector &q_det) const
{
   if (q_layout == QVectorLayout::byNODES)
   {
      if (use_tensor_products)
      {
         if (eval_flags & VALUES)
         {
            Values<QVectorLayout::byNODES>(e_vec, q_val);
         }
         if (eval_flags & DERIVATIVES)
         {
            Derivatives<QVectorLayout::byNODES>(e_vec, q_der);
         }
         if (eval_flags & DETERMINANTS)
         {
            Determinants<QVectorLayout::byNODES>(e_vec, q_det);
         }
      }
      else
      {
         Mult<QVectorLayout::byNODES>(e_vec, eval_flags, q_val, q_der, q_det);
      }
   }

   if (q_layout == QVectorLayout::byVDIM)
   {
      if (use_tensor_products)
      {
         if (eval_flags & VALUES)
         {
            Values<QVectorLayout::byVDIM>(e_vec, q_val);
         }
         if (eval_flags & DERIVATIVES)
         {
            Derivatives<QVectorLayout::byVDIM>(e_vec, q_der);
         }
         if (eval_flags & DETERMINANTS)
         {
            Determinants<QVectorLayout::byVDIM>(e_vec, q_det);
         }
      }
      else
      {
         MFEM_ABORT("this method is not implemented yet");
      }
   }
}

void QuadratureInterpolator::MultTranspose(unsigned eval_flags,
                                           const Vector &q_val,
                                           const Vector &q_der,
                                           Vector &e_vec) const
{
   MFEM_ABORT("this method is not implemented yet");
}

void QuadratureInterpolator::Values(const Vector &e_vec,
                                    Vector &q_val) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Values<QVectorLayout::byNODES>(e_vec, q_val);
      }

      if (q_layout == QVectorLayout::byVDIM)
      {
         Values<QVectorLayout::byVDIM>(e_vec, q_val);
      }
   }
   else
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Vector empty;
         Mult<QVectorLayout::byNODES>(e_vec, VALUES, q_val, empty, empty);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         MFEM_ABORT("this method is not implemented yet");
      }
   }
}

void QuadratureInterpolator::Derivatives(const Vector &e_vec,
                                         Vector &q_der) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Derivatives<QVectorLayout::byNODES>(e_vec, q_der);
      }

      if (q_layout == QVectorLayout::byVDIM)
      {
         Derivatives<QVectorLayout::byVDIM>(e_vec, q_der);
      }
   }
   else
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Vector empty;
         Mult<QVectorLayout::byNODES>(e_vec, DERIVATIVES, empty, q_der, empty);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         MFEM_ABORT("this method is not implemented yet");
      }
   }
}

void QuadratureInterpolator::PhysDerivatives(const Vector &e_vec,
                                             Vector &q_der) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         PhysDerivatives<QVectorLayout::byNODES>(e_vec, q_der);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         PhysDerivatives<QVectorLayout::byVDIM>(e_vec, q_der);
      }
   }
   else
   {
      MFEM_ABORT("this method is not implemented yet");
   }
}

void QuadratureInterpolator::Determinants(const Vector &e_vec,
                                          Vector &q_det) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Determinants<QVectorLayout::byNODES>(e_vec, q_det);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         Determinants<QVectorLayout::byVDIM>(e_vec, q_det);
      }
   }
   else
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Vector empty;
         Mult<QVectorLayout::byNODES>(e_vec, DETERMINANTS, empty, empty, q_det);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         MFEM_ABORT("this method is not implemented yet");
      }
   }
}

} // namespace mfem
