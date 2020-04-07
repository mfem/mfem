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

#ifndef MFEM_QUADINTERP
#define MFEM_QUADINTERP

#include "fespace.hpp"

namespace mfem
{

/// Type describing possible layouts for Q-vectors.
enum class QVectorLayout
{
   byNODES,  ///< NQPT x VDIM x NE
   byVDIM    ///< VDIM x NQPT x NE
};

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

   mutable bool use_tensor_products;

   static const int MAX_NQ2D = 100;
   static const int MAX_ND2D = 100;
   static const int MAX_VDIM2D = 2;

   static const int MAX_NQ3D = 1000;
   static const int MAX_ND3D = 1000;
   static const int MAX_VDIM3D = 3;

public:
   enum EvalFlags
   {
      VALUES       = 1 << 0,  ///< Evaluate the values at quadrature points
      DERIVATIVES  = 1 << 1,  ///< Evaluate the derivatives at quadrature points
      /** @brief Assuming the derivative at quadrature points form a matrix,
          this flag can be used to compute and store their determinants. This
          flag can only be used in Mult(). */
      DETERMINANTS = 1 << 2
   };

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const IntegrationRule &ir);

   QuadratureInterpolator(const FiniteElementSpace &fes,
                          const QuadratureSpace &qs);

   /** @brief Disable the use of tensor product evaluations, for tensor-product
       elements, e.g. quads and hexes. */
   /** Currently, tensor product evaluations are not implemented and this method
       has no effect. */
   void DisableTensorProducts(bool disable = true) const
   { use_tensor_products = !disable; }

   /** @brief Query the current output Q-vector layout. The default value is
       QVectorLayout::byNODES. */
   QVectorLayout GetOutputLayout() const { return q_layout; }

   /** @brief Set the desired output Q-vector layout. The default value is
       QVectorLayout::byNODES. */
   void SetOutputLayout(QVectorLayout out_layout) const
   { q_layout = out_layout; }

   /// Interpolate the E-vector @a e_vec to quadrature points.
   /** The @a eval_flags are a bitwise mask of constants from the EvalFlags
       enumeration. When the VALUES flag is set, the values at quadrature points
       are computed and stored in the Vector @a q_val. Similarly, when the flag
       DERIVATIVES is set, the derivatives are computed and stored in @a q_der.
       When the DETERMINANTS flags is set, it is assumed that the derivatives
       form a matrix at each quadrature point (i.e. the associated
       FiniteElementSpace is a vector space) and their determinants are computed
       and stored in @a q_det. */
   void Mult(const Vector &e_vec, unsigned eval_flags,
             Vector &q_val, Vector &q_der, Vector &q_det) const;

   /// Interpolate the values of the E-vector @a e_vec at quadrature points.
   void Values(const Vector &e_vec, Vector &q_val) const;

   /** @brief Interpolate the derivatives of the E-vector @a e_vec at quadrature
       points. */
   void Derivatives(const Vector &e_vec, Vector &q_der) const;

   /** @brief Interpolate the derivatives in physical space of the E-vector
       @a e_vec at quadrature points. */
   void PhysDerivatives(const Vector &e_vec, Vector &q_der) const;

   /// Perform the transpose operation of Mult(). (TODO)
   void MultTranspose(unsigned eval_flags, const Vector &q_val,
                      const Vector &q_der, Vector &e_vec) const;

   // Compute kernels follow (cannot be private or protected with nvcc)

   /// Template compute kernel for 2D.
   template<const int T_VDIM = 0, const int T_ND = 0, const int T_NQ = 0>
   static void Eval2D(const int NE,
                      const int vdim,
                      const DofToQuad &maps,
                      const Vector &e_vec,
                      Vector &q_val,
                      Vector &q_der,
                      Vector &q_det,
                      const int eval_flags);

   /// Template compute kernel for 3D.
   template<const int T_VDIM = 0, const int T_ND = 0, const int T_NQ = 0>
   static void Eval3D(const int NE,
                      const int vdim,
                      const DofToQuad &maps,
                      const Vector &e_vec,
                      Vector &q_val,
                      Vector &q_der,
                      Vector &q_det,
                      const int eval_flags);
};

}

#endif
