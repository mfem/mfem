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

namespace mfem::experimental
{

template <int FIELD_ID = -1>
class FieldOperator
{
public:
   constexpr FieldOperator(int size_on_qp = 0) :
      size_on_qp(size_on_qp) {};

   static constexpr int GetFieldId() { return FIELD_ID; }

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

template <int FIELD_ID = -1>
class None : public FieldOperator<FIELD_ID>
{
public:
   constexpr None() : FieldOperator<FIELD_ID>() {}
};

template< typename T >
struct is_none_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_none_fop<None<FIELD_ID>>
{
   static const bool value = true;
};

template <typename T>
struct DisableAD
{
   T& operator()() const { return fop; }
   T fop;
};

class Weight : public FieldOperator<-1>
{
public:
   constexpr Weight() : FieldOperator<-1>() {};
};

template< typename T >
struct is_weight_fop
{
   static const bool value = false;
};

template <>
struct is_weight_fop<Weight>
{
   static const bool value = true;
};

template <int FIELD_ID = -1>
class Value : public FieldOperator<FIELD_ID>
{
public:
   constexpr Value() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_value_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_value_fop<Value<FIELD_ID>>
{
   static const bool value = true;
};

template <typename T>
struct is_value_fop<DisableAD<T>>
{
   static const bool value = is_value_fop<T>::value;
};

template <int FIELD_ID = -1>
class Gradient : public FieldOperator<FIELD_ID>
{
public:
   constexpr Gradient() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_gradient_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_gradient_fop<Gradient<FIELD_ID>>
{
   static const bool value = true;
};

template <int FIELD_ID = -1>
class One : public FieldOperator<FIELD_ID>
{
public:
   constexpr One() : FieldOperator<FIELD_ID>() {};
};

template< typename T >
struct is_one_fop
{
   static const bool value = false;
};

template <int FIELD_ID>
struct is_one_fop<One<FIELD_ID>>
{
   static const bool value = true;
};

} // namespace mfem::experimental
