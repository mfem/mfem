// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#include "constitutive.hpp"
#include "mfem.hpp"

namespace mfem
{
class NEML2StressDivergenceIntegrator
    : public StressDivergenceIntegrator<NonlinearFormIntegrator>
{
 public:
   /**
   * @brief Construct a new Linear Momentum Balance object
   *
   * @param fe_space Finite element space for the displacements
   * @param ir Integration rule for the quadrature
   * @param cmodel NEML2 constitutive model for the material
   */
   NEML2StressDivergenceIntegrator(std::shared_ptr<neml2::Model> cmodel,
                                   const IntegrationRule *ir = nullptr);

   using StressDivergenceIntegrator<NonlinearFormIntegrator>::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   /**
   * @brief Perform weak form evaluation
   *
   * @param x Input displacement E-vector
   * @param y Output residual E-vector
   */
   void AddMultPA(const Vector &X, Vector &R) const override;

   void AssembleGradPA(const Vector &x, const FiniteElementSpace &fes) override;
   template <int vdim> void AssembleGradDiagonalPAImpl(Vector &emat) const;
   void AssembleGradDiagonalPA(Vector &diag) const override;

   template <int vdim> void AssembleGradEAImpl(Vector &emat);
   void AssembleGradEA(const Vector &x, const FiniteElementSpace &fes,
                       Vector &emat) override;

   /**
    * Perform action of gradient (Jacobian) upon input vector \p x and put into \p y
    */
   void AddMultGradPA(const Vector &x, Vector &y) const override;

   /// The quadrature space for symmetric 2nd order tensors
   std::unique_ptr<UniformParameterSpace> _q_space_symr2;

 private:
   /// The strain storage
   mutable std::unique_ptr<ParameterFunction> _strain;
   /// The stress storage
   mutable std::unique_ptr<ParameterFunction> _stress;
   /// Whether we're ordering by nodes or by vdim
   std::optional<Ordering::Type> _ordering;

   /// Material tangent (dstress/dstrain) obtained from NEML2 evaluated at the
   /// current strain
   std::optional<neml2::Tensor> _tangent;

   /// The NEML2 constitutive model wrapped with MFEM APIs
   const ConstitutiveModel _constit_op;

   template <int vdim>
   void ComputeStrainImpl(const Vector &X, ParameterFunction &strain) const;
   void ComputeStrain(const Vector &X, ParameterFunction &strain) const;
   template <int vdim>
   void ComputeRImpl(const ParameterFunction &stress, Vector &R) const;
   void ComputeR(const ParameterFunction &stress, Vector &R) const;
};

} // namespace mfem
