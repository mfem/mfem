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

#include "../fespace.hpp"
#include "parameterspace.hpp"

namespace mfem::future
{

/// @brief FieldDescriptor struct
///
/// This struct is used to store information about a field.
struct FieldDescriptor
{
   using data_variant_t =
      std::variant<const FiniteElementSpace *,
      const ParFiniteElementSpace *,
      const QuadratureFunction *,
      const ParameterSpace *>;

   /// Field ID
   std::size_t id;

   /// Field variant
   data_variant_t data;

   /// Default constructor
   FieldDescriptor() :
      id(SIZE_MAX), data(data_variant_t{}) {}

   /// Constructor
   template <typename T>
   FieldDescriptor(std::size_t field_id, const T* v) :
      id(field_id), data(v) {}

   bool operator==(const FieldDescriptor& other) const
   {
      return id == other.id;
   }

   bool operator<(const FieldDescriptor& other) const
   {
      return id < other.id;
   }

   friend void swap(FieldDescriptor& a, FieldDescriptor& b)
   {
      using std::swap;
      swap(a.id, b.id);
      swap(a.data, b.data);
   }
};

}
