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

#include <type_traits>

namespace mfem::future
{

/// @brief Base class for FieldOperators.
///
/// This class serves as a base for different FieldOperator types which can be
/// applied to fields that are used with inputs to a quadrature point function.
/// See DifferentialOperator.
template <int FIELD_ID = -1>
class FieldOperator
{
public:
   /// @brief Constructor for the FieldOperator.
   ///
   /// This constructor initializes the FieldOperator with it's size on
   /// quadrature points. The size on quadrature points has to be determined by
   /// the FieldOperator type, the dimension and the vector dimension (number
   /// of components). See the following examples
   ///
   /// Scalar FiniteElementSpace with Value FieldOperator:
   /// size = vdim x dim x 1 = 1 x dim x 1 = dim
   ///
   /// Vector FiniteElementSpace with Gradient FieldOperator:
   /// size = vdim x dim x dim = vdim x dim x dim = vdim * dim^2
   ///
   /// ParameterSpace with Identity FieldOperator:
   /// size = vdim = vdim
   constexpr FieldOperator(int size_on_qp = 0) :
      size_on_qp(size_on_qp) {};

   /// @brief Get the field id this FieldOperator is attached to.
   static constexpr int GetFieldId() { return FIELD_ID; }

   /// @brief Get the size on quadrature point for this FieldOperator.
   int size_on_qp = -1;

   /// @brief Get the dimension of the FieldOperator.
   int dim = -1;

   /// @brief Get the vector dimension (number of components)
   /// of the FieldOperator.
   int vdim = -1;
};

/// @brief Identity FieldOperator.
///
/// This FieldOperator does nothing to the field. The field (usually a
/// ParametricFunction) transfers the values to the quadrature point data and
/// Identity can be viewed as an identity operation.
template <int FIELD_ID = -1>
class Identity : public FieldOperator<FIELD_ID>
{
public:
   constexpr Identity() : FieldOperator<FIELD_ID>() {}
};

template< typename T >
struct is_identity_fop : std::false_type {};

template <int FIELD_ID>
struct is_identity_fop<Identity<FIELD_ID>> : std::true_type {};

/// @brief Weight FieldOperator.
///
/// This FieldOperator is used to signal that this field contains the quadrature
/// point weights.
class Weight : public FieldOperator<-1>
{
public:
   constexpr Weight() : FieldOperator<-1>() {};
};

template< typename T >
struct is_weight_fop : std::false_type {};

template <>
struct is_weight_fop<Weight> : std::true_type {};

/// @brief Value FieldOperator.
///
/// This FieldOperator is used to signal that the field contains the
/// interpolated values of the degrees of freedom at the quadrature points.
template <int FIELD_ID = -1>
class Value : public FieldOperator<FIELD_ID>
{
public:
   constexpr Value() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_value_fop : std::false_type {};

template <int FIELD_ID>
struct is_value_fop<Value<FIELD_ID>> : std::true_type {};

/// @brief Gradient FieldOperator.
///
/// This FieldOperator is used to signal that the field contains the
/// interpolated gradients of the degrees of freedom at the quadrature points.
template <int FIELD_ID = -1>
class Gradient : public FieldOperator<FIELD_ID>
{
public:
   constexpr Gradient() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_gradient_fop : std::false_type {};

template <int FIELD_ID>
struct is_gradient_fop<Gradient<FIELD_ID>> : std::true_type {};

/// @brief Sum FieldOperator.
///
/// This FieldOperator is commonly used to signal that an output of a quadrature
/// function should be summed.
template <int FIELD_ID = -1>
class Sum : public FieldOperator<FIELD_ID>
{
public:
   constexpr Sum() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_sum_fop : std::false_type {};

template <int FIELD_ID>
struct is_sum_fop<Sum<FIELD_ID>> : std::true_type {};

} // namespace mfem::future
