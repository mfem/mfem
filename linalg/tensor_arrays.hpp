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

#ifndef MFEM_TENSOR_ARRAYS_HPP
#define MFEM_TENSOR_ARRAYS_HPP

#include "tensor.hpp"
#include <array>       // std::array, std::size_t (indirectly)
#include <type_traits> /* std::remove_cv_t, std::remove_reference_t,
                          std::is_const_v */
#include <utility>     /* std::forward, std::index_sequence,
                          std::make_index_sequence */
#include <algorithm>   // std::min
#include <tuple>       // std::apply, std::tuple_size_v
#include <numeric>     // std::iota

namespace mfem
{

namespace future
{

template <std::size_t... Is, typename Fn>
constexpr inline void for_unrolled_simple(std::index_sequence<Is...>, Fn &&fn)
{
   (fn(Is), ...);
}


template <int... loop_sizes>
constexpr inline auto to_multiindex(std::size_t i)
{
   constexpr auto dims = sizeof...(loop_sizes);
   constexpr std::array<std::size_t,dims> sizes{loop_sizes...};
   std::array<std::size_t,dims> is{};  // value initialization with zeros
   for (std::size_t d = 0; d < dims; d++)
   {
      is[d] = i%sizes[d];
      i /= sizes[d];
   }
   return is;
}


/// lambda_t:
/// - input: const std::array<std::size_t,sizeof...(loop_sizes)> &
/// - output: void
/// Note: 0D loop executes the lambda one time with an array of dim 0.
template <int... loop_sizes, typename lambda_t>
constexpr inline void for_multiindex(lambda_t f)
{
   constexpr auto dims = sizeof...(loop_sizes);
   if constexpr (dims == 0)
   {
      f(std::array<std::size_t,0> {});
   }
   else
   {
      if constexpr (std::min({loop_sizes...}) <= 0) { return; }
      constexpr auto total_loop_size = (loop_sizes * ...);
      for_unrolled_simple(std::make_index_sequence<total_loop_size> {},
                          [&f](std::size_t i)
      {
         f(to_multiindex<loop_sizes...>(i));
      });
   }
}


/// Extend std::apply to work with 0-size arrays.
template <typename Fn, typename Tuple>
inline constexpr decltype(auto) apply(Fn&& f, Tuple&& t)
{
   if constexpr (std::tuple_size_v<std::remove_reference_t<Tuple>> == 0)
   { return f(); }
   return std::apply(std::forward<Fn>(f), std::forward<Tuple>(t));
}


/// Multi-dimensional array of tensors of the same size.
/** The array sizes are dynamic while the tensor sizes are static, i.e. template
    parameters.

    This class provides flexible global data layout where the dynamic (array)
    dimnsions and the tensor dimnsions are stored in memory using a runtime
    defined strided layout. */
template <typename scalar_t, int ndims, int... tensor_sizes>
class tensor_ndarray
{
public:
   typedef scalar_t scalar_type;
   typedef tensor<std::remove_cv_t<scalar_t>,tensor_sizes...> tensor_type;

   static constexpr std::integer_sequence<size_t, tensor_sizes...> tensor_sizes_;
   static constexpr auto tensor_dims = sizeof...(tensor_sizes);
   static constexpr auto total_dims = ndims + tensor_dims;
   static constexpr std::array<std::size_t,tensor_dims>
   tensor_sizes_array{tensor_sizes...};

private:
   scalar_t *data;  /// Not owned
   std::array<std::size_t,ndims> dyn_sizes;
   std::array<std::size_t,total_dims> strides;

public:
   /** @brief Constructor with the default, column-major or left, layout where
       the dynamic dimensions are first, on the left, and the tensor dimensions
       are second. */
   tensor_ndarray(scalar_t *ptr, std::array<std::size_t,ndims> dynamic_sizes)
      : data(ptr), dyn_sizes(dynamic_sizes)
   {
      std::array<std::size_t,total_dims> default_perm;
      std::iota(default_perm.begin(), default_perm.end(), 0); // 0, 1, 2, ...
      set_layout(default_perm);
   }

   /// Number of dynamic array dimensions.
   static constexpr std::size_t rank() { return ndims; }

   /// Array size in the @a k-th dynamic dimension.
   std::size_t size(int k = 0) const { return dyn_sizes[k]; }

   /// Returns the product of all sizes of the dynamic dimensions.
   std::size_t total_size() const
   {
      std::size_t t = 1;
      for (int d = 0; d < ndims; d++)
      {
         t *= dyn_sizes[d];
      }
      return t;
   }

   /// Number of tensor (static) dimensions.
   static constexpr std::size_t tensor_rank()
   { return sizeof...(tensor_sizes); }

   /// Tensor size in the @a k-th tensor (static) dimension.
   static constexpr std::size_t tensor_size(int k = 0)
   { return tensor_sizes_array[k]; }

   /// Returns the product of all sizes of the static (tensor) dimensions.
   static constexpr std::size_t total_tensor_size()
   { return (tensor_sizes * ...); }

   /// Set the global data layout based on the given permutation @a perm.
   /** The entries of @a perm are numbers identifying either a dynamic or a
       tensor (static) dimension. Values in the range [0,rank()) identify the
       dynamic dimensions and values in the range [rank(),rank()+tensor_rank())
       identify the tensor dymensions. The first entry in @a perm determines
       which dynamic or tensor dimension will have stride 1. The k-th entry of
       @a perm determines which dimension will use the next stride which is
       defined as the product of the sizes of all k-1 previous dimensions from
       @a perm.

       @note The default layout corresponds to the identity permutation:
       { 0, 1, ..., rank()+tensor_rank()-1 }.

       @note This method does not permute the global 1D data array. */
   void set_layout(std::array<std::size_t,rank()+tensor_rank()> perm)
   {
      std::size_t stride = 1;
      for (std::size_t d_g = 0; d_g < total_dims; d_g++)
      {
         const auto d_l = perm[d_g];
         strides[d_l] = stride;
         stride *= (d_l < ndims) ? dyn_sizes[d_l] :
                   tensor_sizes_array[d_l-ndims];
      }
   }

   /** @brief Comute the dynamic offset for a given dynamic multi-index @a is.
       The total offset in the global data array is the sum of the dynamic and
       static (tensor) offsets. */
   std::size_t get_dynamic_offset(
      const std::array<std::size_t,rank()> &is) const
   {
      std::size_t dynamic_offset = 0;
      for (std::size_t d = 0; d < ndims; d++)
      {
         dynamic_offset += is[d]*strides[d];
      }
      return dynamic_offset;
   }

   /** @brief Comute the static (tensor) offset for a given tensor multi-index
       @a js. The total offset in the global data array is the sum of the
       dynamic and static (tensor) offsets. */
   std::size_t get_static_offset(
      const std::array<std::size_t,tensor_rank()> &js) const
   {
      std::size_t static_offset = 0;
      for (std::size_t d = 0; d < tensor_dims; d++)
      {
         static_offset += js[d]*strides[ndims+d];
      }
      return static_offset;
   }

   /** @brief Return a local tensor extracted from the global data array
       corresponding to the given dynamic multi-index @a is. */
   /** @note Return a const tensor to prevent attempts to assign to the
       temporary object which is considered a mistake. */
   const tensor_type get_tensor(std::array<std::size_t,rank()> is) const
   {
      tensor_type result;
      const std::size_t dynamic_offset = get_dynamic_offset(is);
      for_multiindex<tensor_sizes...>(
         [&result, this, dynamic_offset](
            const std::array<std::size_t,tensor_rank()> &js)
      {
         ::mfem::future::apply(result, js) =
            data[dynamic_offset + get_static_offset(js)];
      });
      return result;
   }

   /** @brief Return a local tensor extracted from the global data array
       corresponding to the given dynamic indices @a is. */
   /** @note Return a const tensor to prevent attempts to assign to the
       temporary object which is considered a mistake. */
   template <typename... index_types>
   const tensor_type get_tensor(index_types... is) const
   {
      static_assert(sizeof...(is) == rank(), "invalid number of indices!");
      return get_tensor(std::array<std::size_t,rank()> {std::size_t(is)...});
   }

   /** @brief Returns one of the following depending on the type scalar_t:
       - get_tensor(std::array<std::size_t,rank()>) iff scalar_t is const,
       - get_accessor(std::array<std::size_t,rank()>) iff scalar_t is not
         const. */
   decltype(auto) operator()(std::array<std::size_t,rank()> is) const
   {
      if constexpr (std::is_const_v<scalar_t>) { return get_tensor(is); }
      else { return get_accessor(is); }
   }

   /** @brief Returns one of the following depending on the type scalar_t:
       - get_tensor(index_types...) iff scalar_t is const,
       - get_accessor(index_types...) iff scalar_t is not const. */
   template <typename... index_types>
   decltype(auto) operator()(index_types... is) const
   {
      if constexpr (std::is_const_v<scalar_t>) { return get_tensor(is...); }
      else { return get_accessor(is...); }
   }

   /** @brief Helper class facilitating the reading/writing of local tensor
       objects to the global data array of the tensor_ndarray. */
   class tensor_accessor
   {
   private:
      const tensor_ndarray &base_array;
      scalar_t *offset_data;  /// Not owned

   public:
      /** @brief Construct a tensor_accessor to @a base for the given dynamic
          multi-index @a is.

          During its life time, this object assumes that the @a base object
          remains unmodified. */
      tensor_accessor(const tensor_ndarray &base,
                      const std::array<std::size_t,rank()> &is)
         : base_array(base)
      {
         offset_data = base_array.data + base_array.get_dynamic_offset(is);
      }

      /// Read-write access to a particular entry of the referenced tensor.
      /** The returned reference points to the corresponding entry in the global
          data array of the base tensor_ndarray. */
      scalar_t &operator()(const std::array<std::size_t,tensor_rank()> &js)
      {
         return offset_data[base_array.get_static_offset(js)];
      }

      /** @brief Write a tensor to the referenced tensor in the global data
          array of the base tensor_ndarray. */
      tensor_accessor &operator=(const tensor_type &rhs)
      {
         for_multiindex<tensor_sizes...>(
            [&](const std::array<std::size_t,tensor_rank()> &js)
         {
            operator()(js) = ::mfem::future::apply(rhs, js);
         });
         return *this;
      }
   };

   /** @brief Get a tensor_accessor object referencing the tensor stored at the
       dynamic multi-index @a is. This object can be used to write tensor
       objects into the global data array of the tensor_ndarray. */
   tensor_accessor get_accessor(std::array<std::size_t,rank()> is) const
   {
      return tensor_accessor(*this, is);
   }

   /** @brief Get a tensor_accessor object referencing the tensor stored at the
       dynamic indices @a is. This object can be used to write tensor objects
       into the global data array of the tensor_ndarray. */
   template <typename... index_types>
   tensor_accessor get_accessor(index_types... is) const
   {
      static_assert(sizeof...(is) == rank(), "invalid number of indices!");
      return get_accessor(std::array<std::size_t,rank()> {std::size_t(is)...});
   }
};


/** @brief Construct a tensor_ndarray where only the tensor sizes have to be
    explicitly given as template parameters, the rest can be deduced from the
    function call arguments. */
template <int... tensor_sizes, typename scalar_t, typename... dyn_sizes_t>
decltype(auto) make_tensor_ndarray(scalar_t *ptr, dyn_sizes_t... dynamic_sizes)
{
   return tensor_ndarray<scalar_t,sizeof...(dynamic_sizes),tensor_sizes...>(
             ptr, {std::size_t(dynamic_sizes)...});
}


/// Alias for make_tensor_ndarray = make_tensor_array.
template <int... tensor_sizes, typename scalar_t, typename... dyn_sizes_t>
decltype(auto) make_tensor_array(scalar_t *ptr, dyn_sizes_t... dynamic_sizes)
{
   return tensor_ndarray<scalar_t,sizeof...(dynamic_sizes),tensor_sizes...>(
             ptr, {std::size_t(dynamic_sizes)...});
}


/// Short name for one-dimensional tensor_ndarray: tensor_array.
template <typename scalar_t, int... tensor_sizes>
using tensor_array = tensor_ndarray<scalar_t, 1, tensor_sizes...>;

} // namespace mfem::future

} // namespace mfem

#endif // MFEM_TENSOR_ARRAYS_HPP
