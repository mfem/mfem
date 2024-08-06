// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "elasticity_operator.hpp"
#include "elasticity_gradient_operator.hpp"

namespace mfem
{

ElasticityOperator::ElasticityOperator(ParMesh &mesh, const int order)
   : Operator(), mesh_(mesh), order_(order), dim_(mesh_.SpaceDimension()),
     vdim_(mesh_.SpaceDimension()), ne_(mesh_.GetNE()), h1_fec_(order_, dim_),
     h1_fes_(&mesh_, &h1_fec_, vdim_, Ordering::byNODES)
{
   this->height = h1_fes_.GetTrueVSize();
   this->width = this->height;

   int global_tdof_size = h1_fes_.GlobalTrueVSize();
   if (mesh.GetMyRank() == 0)
   {
      out << "#dofs: " << global_tdof_size << std::endl;
   }

   h1_element_restriction_ =
      h1_fes_.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   h1_prolongation_ = h1_fes_.GetProlongationMatrix();

   ir_ = const_cast<IntegrationRule *>(
            &IntRules.Get(mfem::Element::HEXAHEDRON, 2 * h1_fes_.GetOrder(0) + 1));

   geometric_factors_ = h1_fes_.GetParMesh()->GetGeometricFactors(
                           *ir_, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   maps_ = &h1_fes_.GetTypicalFE()->GetDofToQuad(*ir_, DofToQuad::TENSOR);
   d1d_ = maps_->ndof;
   q1d_ = maps_->nqpt;

   dX_ess_.UseDevice(true);
   dX_ess_.SetSize(h1_fes_.GetTrueVSize());

   X_el_.UseDevice(true);
   X_el_.SetSize(h1_element_restriction_->Height());

   Y_el_.UseDevice(true);
   Y_el_.SetSize(h1_element_restriction_->Height());

   cstate_el_.UseDevice(true);
   cstate_el_.SetSize(h1_element_restriction_->Height());

   X_local_.UseDevice(true);
   X_local_.SetSize(h1_prolongation_->Height());

   Y_local_.UseDevice(true);
   Y_local_.SetSize(h1_prolongation_->Height());

   cstate_local_.UseDevice(true);
   cstate_local_.SetSize(h1_prolongation_->Height());

   if (use_cache_)
   {
      dsigma_cache_.SetSize(ne_ * q1d_ * q1d_ * q1d_ * dim_ * dim_ * dim_ * dim_);
   }

   gradient_ = new ElasticityGradientOperator(*this);
}

void ElasticityOperator::Mult(const Vector &X, Vector &Y) const
{
   ess_tdof_list_.Read();

   // T-vector to L-vector
   h1_prolongation_->Mult(X, X_local_);
   // L-vector to E-vector
   h1_element_restriction_->Mult(X_local_, X_el_);

   // Reset output vector
   Y_el_ = 0.0;

   // Apply operator
   element_apply_kernel_wrapper(ne_, maps_->B, maps_->G, ir_->GetWeights(),
                                geometric_factors_->J, geometric_factors_->detJ,
                                X_el_, Y_el_);

   // E-vector to L-vector
   h1_element_restriction_->MultTranspose(Y_el_, Y_local_);
   // L-vector to T-vector
   h1_prolongation_->MultTranspose(Y_local_, Y);

   // Set the residual at Dirichlet dofs on the T-vector to zero
   Y.SetSubVector(ess_tdof_list_, 0.0);
}

Operator &ElasticityOperator::GetGradient(const Vector &x) const
{
   // invalidate cache
   recompute_cache_ = true;

   h1_prolongation_->Mult(x, cstate_local_);
   h1_element_restriction_->Mult(cstate_local_, cstate_el_);
   return *gradient_;
}

void ElasticityOperator::GradientMult(const Vector &dX, Vector &Y) const
{
   ess_tdof_list_.Read();

   // Column elimination for essential dofs
   dX_ess_ = dX;
   dX_ess_.SetSubVector(ess_tdof_list_, 0.0);

   // T-vector to L-vector
   h1_prolongation_->Mult(dX_ess_, X_local_);
   // L-vector to E-vector
   h1_element_restriction_->Mult(X_local_, X_el_);

   // Reset output vector
   Y_el_ = 0.0;

   // Apply operator
   element_apply_gradient_kernel_wrapper(
      ne_, maps_->B, maps_->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, X_el_, Y_el_, cstate_el_);

   // E-vector to L-vector
   h1_element_restriction_->MultTranspose(Y_el_, Y_local_);
   // L-vector to T-vector
   h1_prolongation_->MultTranspose(Y_local_, Y);

   // Re-assign the essential degrees of freedom on the final output vector.
   {
      const auto d_dX = dX.Read();
      auto d_Y = Y.ReadWrite();
      const auto d_ess_tdof_list = ess_tdof_list_.Read();
      mfem::forall(ess_tdof_list_.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Y[d_ess_tdof_list[i]] = d_dX[d_ess_tdof_list[i]];
      });
   }

   recompute_cache_ = false;
}

void ElasticityOperator::AssembleGradientDiagonal(Vector &Ke_diag,
                                                  Vector &K_diag_local,
                                                  Vector &K_diag) const
{
   Ke_diag.SetSize(d1d_ * d1d_ * d1d_ * dim_ * ne_ * dim_);
   K_diag_local.SetSize(h1_element_restriction_->Width() * dim_);
   K_diag.SetSize(h1_prolongation_->Width() * dim_);

   element_kernel_assemble_diagonal_wrapper(
      ne_, maps_->B, maps_->G, ir_->GetWeights(), geometric_factors_->J,
      geometric_factors_->detJ, cstate_el_, Ke_diag);

   // For each dimension, the H1 element restriction and H1 prolongation
   // transpose actions are applied separately.
   for (int i = 0; i < dim_; i++)
   {
      // Scalar component E-size
      int sce_sz = d1d_ * d1d_ * d1d_ * dim_ * ne_;
      // Scalar component L-size
      int scl_sz = h1_element_restriction_->Width();

      Vector vin_local, vout_local;
      vin_local.MakeRef(Ke_diag, i * sce_sz, sce_sz);
      vout_local.MakeRef(K_diag_local, i * scl_sz, scl_sz);
      h1_element_restriction_->MultTranspose(vin_local, vout_local);
      vout_local.GetMemory().SyncAlias(K_diag_local.GetMemory(),
                                       vout_local.Size());

      // Scalar component T-size
      int sct_sz = h1_prolongation_->Width();
      Vector vout;
      vout.MakeRef(K_diag, i * sct_sz, sct_sz);
      h1_prolongation_->MultTranspose(vout_local, vout);
      vout.GetMemory().SyncAlias(K_diag.GetMemory(), vout.Size());
   }

   // Each essential dof row and column are set to zero with it's diagonal entry
   // set to 1, i.e. (Ke)_ii = 1.0.
   ess_tdof_list_.HostRead();
   int num_submats = h1_fes_.GetTrueVSize() / h1_fes_.GetVDim();
   auto K_diag_submats = Reshape(K_diag.HostWrite(), num_submats, dim_, dim_);
   for (int i = 0; i < ess_tdof_list_.Size(); i++)
   {
      int ess_idx = ess_tdof_list_[i];
      int submat = ess_idx % num_submats;
      int row = ess_idx / num_submats;
      for (int j = 0; j < dim_; j++)
      {
         if (row == j)
         {
            K_diag_submats(submat, row, j) = 1.0;
         }
         else
         {
            K_diag_submats(submat, row, j) = 0.0;
            K_diag_submats(submat, j, row) = 0.0;
         }
      }
   }
}

ElasticityOperator::~ElasticityOperator() { delete gradient_; }

} // namespace mfem
