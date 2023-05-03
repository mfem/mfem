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

#include "diagonal_preconditioner.hpp"
#include "general/forall.hpp"
#include "linalg/tensor.hpp"

using mfem::internal::tensor;

namespace mfem
{

void ElasticityDiagonalPreconditioner::SetOperator(const Operator &op)
{
   gradient_operator_ = dynamic_cast<const ElasticityGradientOperator *>(&op);
   MFEM_ASSERT(gradient_operator_ != nullptr,
               "Operator is not ElasticityGradientOperator");

   width = height = op.Height();

   gradient_operator_->AssembleGradientDiagonal(Ke_diag_, K_diag_local_,
                                                K_diag_);

   submat_height_ = gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
   num_submats_ = gradient_operator_->elasticity_op_.h1_fes_.GetTrueVSize() /
                  gradient_operator_->elasticity_op_.h1_fes_.GetVDim();
}

void ElasticityDiagonalPreconditioner::Mult(const Vector &x, Vector &y) const
{
   const int ns = num_submats_, sh = submat_height_, nsh = ns * sh;

   const auto K_diag_submats = Reshape(K_diag_.Read(), ns, sh, sh);
   const auto X = Reshape(x.Read(), ns, dim);

   auto Y = Reshape(y.Write(), ns, dim);

   if (type_ == Type::Diagonal)
   {
      // Assuming Y and X are ordered byNODES. K_diag is ordered byVDIM.
      mfem::forall(nsh, [=] MFEM_HOST_DEVICE (int si)
      {
         const int s = si / sh;
         const int i = si % sh;
         Y(s, i) = X(s, i) / K_diag_submats(s, i, i);
      });
   }
   else if (type_ == Type::BlockDiagonal)
   {
      mfem::forall(ns, [=] MFEM_HOST_DEVICE (int s)
      {
         const auto submat = make_tensor<dim, dim>(
         [&](int i, int j) { return K_diag_submats(s, i, j); });

         const auto submat_inv = inv(submat);

         const auto x_block = make_tensor<dim>([&](int i) { return X(s, i); });

         tensor<double, dim> y_block = submat_inv * x_block;

         for (int i = 0; i < dim; i++)
         {
            Y(s, i) = y_block(i);
         }
      });
   }
   else
   {
      MFEM_ABORT("Unknown ElasticityDiagonalPreconditioner::Type");
   }
}

} // namespace mfem
