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

#ifndef MFEM_TENSOR
#define MFEM_TENSOR

/// Contain compilation time functions used with tensors
#include "tensor_traits.hpp"
/// Utility functions to abstract iterating over the tensor dimensions
#include "utilities/foreach.hpp"

namespace mfem
{

/** A generic tensor class using a linear memory container storing the values,
    and a layout mapping a rank N index to a linear index corresponding to the
    values indices in the container.
    @a Container is the type of data container, they can either be statically or
       dynamically allocated,
    @a Layout is a class that represents the data layout
       There is two main sub-categories of Layout, Static and Dynamic layouts.
       Dynamic Layout have the following signature:
       template <int Rank>,
       Static Layout have the following signature:
       template <int... Sizes>,
       where Sizes... is the list of the sizes of the dimensions of the Tensor.
   */
template <typename Container,
          typename Layout>
class Tensor: public Container, public Layout
{
public:
   using T = get_tensor_value_type<Tensor>;
   using container = Container;
   using layout = Layout;

   /// Default Constructor
   /** In order to use the Tensor default constructor, both the Container and
       the Layout need to have a default constructor.
   */
   MFEM_HOST_DEVICE
   Tensor() : Container(), Layout() { }

   /// Main Constructors
   /** Construct a Tensor by providing the sizes of its different dimensions,
       both the Container and the Layout need to have similar constructors for
       this Tensor constructor to be usable.
   */
   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(int size0, Sizes... sizes)
   : Container(size0,sizes...), Layout(size0,sizes...) { }

   /** Construct a Tensor by providing a pointer to its data, and the sizes of
       its Layout. The Container needs to have a constructor using a pointer for
       this constructor to be usable.
   */
   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(T* ptr, Sizes... sizes)
   : Container(ptr), Layout(sizes...) { }

   /// Utility Constructors
   /** Construct a tensor based on a Layout, the Container needs to be default
       constructible. Note: A Tensor is a Layout (through inheritance).
   */
   MFEM_HOST_DEVICE
   Tensor(Layout index): Container(), Layout(index) { }

   /** Construct a Tensor by providing a Container object and a Layout object.
   */
   MFEM_HOST_DEVICE
   Tensor(Container data, Layout index): Container(data), Layout(index) { }

   /// Copy Constructors
   /** Copy a Tensor of the same type, the copy is deep or shallow depending on
       the Container.
   */
   MFEM_HOST_DEVICE
   Tensor(const Tensor &rhs): Container(rhs), Layout(rhs) { }

   /** Deep copy of a Tensor of a different type. */
   template <typename OtherTensor,
             std::enable_if_t<
               is_tensor<OtherTensor>,
               bool> = true > MFEM_HOST_DEVICE
   Tensor(const OtherTensor &rhs): Container(), Layout(rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
   }

   /// Accessor
   /** This operator allows to access a value inside a Tensor by providing
       indices, the number of indices must be equal to the rank of the Tensor.
   */
   template <typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(get_tensor_rank<Tensor> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Const Accessor
   /** This operator allows to access a const value inside a const Tensor by
       providing indices, the number of indices must be equal to the rank of the
       Tensor.
   */
   template <typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(get_tensor_rank<Tensor> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Initialization of a Tensor to a constant value.
   MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const T &val)
   {
      ForallDims<Tensor>::Apply(*this, [&](auto... idx)
      {
         (*this)(idx...) = val;
      });
      return *this;
   }

   /// operator=, compatible with other types of Tensors
   template <typename OtherTensor,
             std::enable_if_t<
               is_tensor<OtherTensor>,
               bool> = true > MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
      return *this;
   }

   /// operator+=, compatible with other types of Tensors
   template <typename OtherTensor,
             std::enable_if_t<
               is_tensor<OtherTensor>,
               bool> = true > MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator+=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) += rhs(idx...);
      });
      return *this;
   }

   /// operator-=, compatible with other types of Tensors
   template <typename OtherTensor,
             std::enable_if_t<
               is_tensor<OtherTensor>,
               bool> = true > MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator-=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) -= rhs(idx...);
      });
      return *this;
   }
};

} // namespace mfem

#endif // MFEM_TENSOR
