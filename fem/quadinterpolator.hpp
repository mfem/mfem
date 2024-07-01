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

#ifndef MFEM_QUADINTERP
#define MFEM_QUADINTERP

#include "fespace.hpp"
#include "kernel_dispatch.hpp"

namespace mfem
{

/** @brief A class that performs interpolation from an E-vector to quadrature
    point values and/or derivatives (Q-vectors). */
/** An E-vector represents the element-wise discontinuous version of the FE
    space and can be obtained, for example, from a GridFunction using the
    Operator returned by FiniteElementSpace::GetElementRestriction().

    The target quadrature points in the elements can be described either by an
    IntegrationRule (all mesh elements must be of the same type in this case) or
    by a QuadratureSpace. */
class QuadratureInterpolator
{
protected:
   friend class FiniteElementSpace; // Needs access to qspace and IntRule

   const FiniteElementSpace *fespace;  ///< Not owned
   const QuadratureSpace *qspace;      ///< Not owned
   const IntegrationRule *IntRule;     ///< Not owned
   mutable QVectorLayout q_layout;     ///< Output Q-vector layout

   mutable bool use_tensor_products;   ///< Tensor product evaluation mode
   mutable Vector d_buffer;            ///< Auxiliary device buffer

public:
   static const int MAX_NQ2D = 100;
   static const int MAX_ND2D = 100;
   static const int MAX_VDIM2D = 3;

   static const int MAX_NQ3D = 1000;
   static const int MAX_ND3D = 1000;
   static const int MAX_VDIM3D = 3;

   enum EvalFlags
   {
      VALUES       = 1 << 0,  ///< Evaluate the values at quadrature points
      DERIVATIVES  = 1 << 1,  ///< Evaluate the derivatives at quadrature points
      /** @brief Assuming the derivative at quadrature points form a matrix,
          this flag can be used to compute and store their determinants. This
          flag can only be used in Mult(). */
      DETERMINANTS = 1 << 2,
      PHYSICAL_DERIVATIVES = 1 << 3 ///< Evaluate the physical derivatives
   };

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const IntegrationRule &ir);

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const QuadratureSpace &qs);

   /** @brief Disable the use of tensor product evaluations, for tensor-product
       elements, e.g. quads and hexes. By default, tensor product evaluations
       are enabled. */
   /** @sa EnableTensorProducts(), UsesTensorProducts(). */
   void DisableTensorProducts(bool disable = true) const
   { use_tensor_products = !disable; }

   /** @brief Enable the use of tensor product evaluations, for tensor-product
       elements, e.g. quads and hexes. By default, this option is enabled. */
   /** @sa DisableTensorProducts(), UsesTensorProducts(). */
   void EnableTensorProducts() const { use_tensor_products = true; }

   /** @brief Query the current tensor product evaluation mode. */
   /** @sa DisableTensorProducts(), EnableTensorProducts(). */
   bool UsesTensorProducts() const { return use_tensor_products; }

   /** @brief Query the current output Q-vector layout. The default value is
       QVectorLayout::byNODES. */
   /** @sa SetOutputLayout(). */
   QVectorLayout GetOutputLayout() const { return q_layout; }

   /** @brief Set the desired output Q-vector layout. The default value is
       QVectorLayout::byNODES. */
   /** @sa GetOutputLayout(). */
   void SetOutputLayout(QVectorLayout layout) const { q_layout = layout; }

   /// Interpolate the E-vector @a e_vec to quadrature points.
   /** The @a eval_flags are a bitwise mask of constants from the EvalFlags
       enumeration. When the VALUES flag is set, the values at quadrature points
       are computed and stored in the Vector @a q_val. Similarly, when one of
       the flags DERIVATIVES or PHYSICAL_DERIVATIVES is set, the derivatives
       (with respect to reference or physical coordinates, respectively) are
       computed and stored in @a q_der. Only one of the flags DERIVATIVES or
       PHYSICAL_DERIVATIVES can be set in a call. When the DETERMINANTS flag is
       set, it is assumed that the derivatives (with respect to reference
       coordinates) form a matrix at each quadrature point (i.e. the associated
       FiniteElementSpace is a vector space) and their determinants are computed
       and stored in @a q_det.

       The layout of the input E-vector, @a e_vec, must be consistent with the
       evaluation mode: if tensor-product evaluations are enabled, then
       tensor-product elements, must use the ElementDofOrdering::LEXICOGRAPHIC
       layout; otherwise -- ElementDofOrdering::NATIVE layout. See
       FiniteElementSpace::GetElementRestriction(). */
   void Mult(const Vector &e_vec, unsigned eval_flags,
             Vector &q_val, Vector &q_der, Vector &q_det) const;

   /// Interpolate the values of the E-vector @a e_vec at quadrature points.
   void Values(const Vector &e_vec, Vector &q_val) const;

   /** @brief Interpolate the derivatives (with respect to reference
       coordinates) of the E-vector @a e_vec at quadrature points. */
   void Derivatives(const Vector &e_vec, Vector &q_der) const;

   /** @brief Interpolate the derivatives in physical space of the E-vector
       @a e_vec at quadrature points. */
   void PhysDerivatives(const Vector &e_vec, Vector &q_der) const;

   /** @brief Compute the determinants of the derivatives (with respect to
       reference coordinates) of the E-vector @a e_vec at quadrature points. */
   void Determinants(const Vector &e_vec, Vector &q_det) const;

   /// Perform the transpose operation of Mult(). (TODO)
   void MultTranspose(unsigned eval_flags, const Vector &q_val,
                      const Vector &q_der, Vector &e_vec) const;


   using TensorEvalKernelType = void(*)(const int, const real_t *, const real_t *,
                                        real_t *, const int, const int, const int);
   using GradKernelType = void(*)(const int, const real_t *, const real_t *,
                                  const real_t *, const real_t *, real_t *,
                                  const int, const int, const int, const int);
   using DetKernelType = void(*)(const int NE, const real_t *, const real_t *,
                                 const real_t *, real_t *, const int, const int,
                                 Vector *);
   using EvalKernelType = void(*)(const int, const int, const QVectorLayout,
                                  const GeometricFactors *, const DofToQuad &,
                                  const Vector &, Vector &, Vector &, Vector &,
                                  const int);

   MFEM_REGISTER_KERNELS(TensorEvalKernels, TensorEvalKernelType,
                         (int, QVectorLayout, int, int, int), (int));
   MFEM_REGISTER_KERNELS(GradKernels, GradKernelType,
                         (int, QVectorLayout, bool, int, int, int), (int));
   MFEM_REGISTER_KERNELS(DetKernels, DetKernelType, (int, int, int, int));
   MFEM_REGISTER_KERNELS(EvalKernels, EvalKernelType, (int, int, int, int));

   static struct Kernels { Kernels(); } kernels;
};

}

#endif
