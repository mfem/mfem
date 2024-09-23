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

#ifndef MFEM_ELASTICITY_OP_HPP
#define MFEM_ELASTICITY_OP_HPP

#include "../kernels/elasticity_kernels.hpp"
#include "mfem.hpp"

namespace mfem
{
class ElasticityOperator : public Operator
{
public:
   ElasticityOperator(ParMesh &mesh, const int order);

   /**
    * @brief Compute the residual Y = R(U) representing the elasticity equation
    * with a material model chosen by calling SetMaterial.
    *
    * The output vector @a Y has essential degrees of freedom applied by setting
    * them to zero. This ensures R(U)_i = 0 being satisfied for each essential
    * dof i.
    *
    * @param U U
    * @param Y Residual R(U)
    */
   virtual void Mult(const Vector &U, Vector &Y) const override;

   /**
    * @brief Get the Gradient object
    *
    * Update and cache the state vector @a U, used to compute the linearization
    * dR(U)/dU.
    *
    * @param U
    * @return Operator&
    */
   Operator &GetGradient(const Vector &U) const override;

   /**
    * @brief Multiply the linearization of the residual R(U) wrt to the current
    * state U by a perturbation @a dX.
    *
    * Y = dR(U)/dU * dX = K(U) dX
    *
    * @param dX
    * @param Y
    */
   void GradientMult(const Vector &dX, Vector &Y) const;

   /**
    * @brief Assemble the linearization of the residual K = dR(U)/dU.
    *
    * This method needs three input vectors which also act as output vectors.
    * They don't have to be the right size on the first call, but it is advised
    * that memory is kept alive during successive call. The data layout of the
    * outputs will be
    *
    * @a Ke_diag: dofs x dofs x dofs x dim x ne x dim
    *
    * @a K_diag_local: width(H1_Restriction) x dim
    *
    * @a K_diag: width(H1_Prolongation) x dim
    *
    * This data layout is needed due to the Ordering::byNODES. See method
    * implementation comments for more details. The output @a K_diag has
    * modified entries when essential boundaries are defined. Each essential dof
    * row and column are set to zero with it's diagonal entry set to 1.
    *
    * @param Ke_diag
    * @param K_diag_local
    * @param K_diag
    */
   void AssembleGradientDiagonal(Vector &Ke_diag, Vector &K_diag_local,
                                 Vector &K_diag) const;

   ~ElasticityOperator();

   ParMesh &mesh_;
   /// Polynomial order of the FE space
   const int order_;
   const int dim_;
   const int vdim_;
   /// Number of elements in the mesh (rank local)
   const int ne_;
   /// H1 finite element collection
   H1_FECollection h1_fec_;
   /// H1 finite element space
   ParFiniteElementSpace h1_fes_;
   // Integration rule
   IntegrationRule *ir_ = nullptr;
   /// Number of degrees of freedom in 1D
   int d1d_;
   /// Number of quadrature points in 1D
   int q1d_;
   const Operator *h1_element_restriction_;
   const Operator *h1_prolongation_;
   Array<int> ess_tdof_list_;
   Array<int> displaced_tdof_list_;
   Operator *gradient_;
   const GeometricFactors *geometric_factors_;
   const DofToQuad *maps_;
   /// Input state L-vector
   mutable Vector X_local_;
   /// Input state E-vector
   mutable Vector X_el_;
   /// Output state L-vector
   mutable Vector Y_local_;
   /// Output state E-Vector
   mutable Vector Y_el_;
   /// Cached current state. Used to determine the state on which to compute the
   /// linearization on during the Newton method.
   mutable Vector current_state;
   mutable Vector cstate_local_;
   mutable Vector cstate_el_;
   /// Temporary vector for the perturbation of the solution with essential
   /// boundaries eliminated. Defined as a T-vector.
   mutable Vector dX_ess_;

   /// Flag to enable caching of the gradient. If enabled, during linear
   /// iterations the operator only applies the gradient on each quadrature
   /// point rather than recompute the action.
   bool use_cache_ = true;
   mutable bool recompute_cache_ = false;
   Vector dsigma_cache_;

   /**
    * @brief Wrapper for the application of the residual R(U).
    *
    * The wrapper is used in SetMaterial to instantiate the chosen kernel and
    * erase the material type kernel. This is purely an interface design choice
    * and could be replaced by an abstract base class for the material including
    * virtual function calls.
    */
   std::function<void(const int, const Array<real_t> &, const Array<real_t> &,
                      const Array<real_t> &, const Vector &, const Vector &,
                      const Vector &, Vector &)>
   element_apply_kernel_wrapper;

   /**
    * @brief Wrapper for the application of the gradient of the residual
    *
    *  K(U) dX = dR(U)/dU dX
    */
   std::function<void(const int, const Array<real_t> &, const Array<real_t> &,
                      const Array<real_t> &, const Vector &, const Vector &,
                      const Vector &, Vector &, const Vector &)>
   element_apply_gradient_kernel_wrapper;

   /**
    * @brief Wrapper for the assembly of the gradient on each diagonal element
    *
    * Ke_ii(U) = dRe_ii(U)/dU
    */
   std::function<void(const int, const Array<real_t> &, const Array<real_t> &,
                      const Array<real_t> &, const Vector &, const Vector &,
                      const Vector &, Vector &)>
   element_kernel_assemble_diagonal_wrapper;

   /**
    * @brief Set the material type.
    *
    * This method sets the material type by instantiating the kernels with a
    * material_type object.
    *
    * @tparam material_type
    * @param[in] material
    */
   template <typename material_type>
   void SetMaterial(const material_type &material)
   {
      if (dim_ != 3)
      {
         MFEM_ABORT("dim != 3 not implemented");
      }

      element_apply_kernel_wrapper =
         [=](const int ne, const Array<real_t> &B_, const Array<real_t> &G_,
             const Array<real_t> &W_, const Vector &Jacobian_,
             const Vector &detJ_, const Vector &X_, Vector &Y_)
      {
         const int id = (d1d_ << 4) | q1d_;
         switch (id)
         {
            case 0x22:
            {
               ElasticityKernels::Apply3D<2, 2, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            }
            case 0x33:
            {
               ElasticityKernels::Apply3D<3, 3, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            }
            case 0x44:
               ElasticityKernels::Apply3D<4, 4, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            default:
               MFEM_ABORT("Not implemented: " << std::hex << id << std::dec);
         }
      };

      element_apply_gradient_kernel_wrapper =
         [=](const int ne, const Array<real_t> &B_, const Array<real_t> &G_,
             const Array<real_t> &W_, const Vector &Jacobian_,
             const Vector &detJ_, const Vector &dU_, Vector &dF_,
             const Vector &U_)
      {
         const int id = (d1d_ << 4) | q1d_;
         switch (id)
         {
            case 0x22:
               ElasticityKernels::ApplyGradient3D<2, 2, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material,
                  use_cache_, recompute_cache_, dsigma_cache_);
               break;
            case 0x33:
               ElasticityKernels::ApplyGradient3D<3, 3, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material,
                  use_cache_, recompute_cache_, dsigma_cache_);
               break;
            case 0x44:
               ElasticityKernels::ApplyGradient3D<4, 4, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, dU_, dF_, U_, material,
                  use_cache_, recompute_cache_, dsigma_cache_);
               break;
            default:
               MFEM_ABORT("Not implemented for D1D=" << d1d_ << " and Q1D=" << q1d_);
         }
      };

      element_kernel_assemble_diagonal_wrapper =
         [=](const int ne, const Array<real_t> &B_, const Array<real_t> &G_,
             const Array<real_t> &W_, const Vector &Jacobian_,
             const Vector &detJ_, const Vector &X_, Vector &Y_)
      {
         const int id = (d1d_ << 4) | q1d_;
         switch (id)
         {
            case 0x22:
               ElasticityKernels::AssembleGradientDiagonal3D<2, 2, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            case 0x33:
               ElasticityKernels::AssembleGradientDiagonal3D<3, 3, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            case 0x44:
               ElasticityKernels::AssembleGradientDiagonal3D<4, 4, material_type>(
                  ne, B_, G_, W_, Jacobian_, detJ_, X_, Y_, material);
               break;
            default:
               MFEM_ABORT("Not implemented: " << std::hex << id << std::dec);
         }
      };
   }

   /**
    * @brief Set the essential attributes which mark degrees of freedom for the
    * solving process.
    *
    * Can be either a fixed boundary or a prescribed displacement.
    *
    * @param attr
    */
   void SetEssentialAttributes(const Array<int> attr)
   {
      h1_fes_.GetEssentialTrueDofs(attr, ess_tdof_list_);
   }

   /**
    * @brief Set the attributes which mark the degrees of freedom that have a
    * fixed displacement.
    *
    * @param[in] attr
    */
   void SetPrescribedDisplacement(const Array<int> attr)
   {
      h1_fes_.GetEssentialTrueDofs(attr, displaced_tdof_list_);
   }

   /**
    * @brief Return the T-vector degrees of freedom that have been marked as
    * displaced.
    *
    * @return T-vector degrees of freedom that have been marked as displaced
    */
   const Array<int> &GetPrescribedDisplacementTDofs() { return displaced_tdof_list_; };
};

} // namespace mfem

#endif
