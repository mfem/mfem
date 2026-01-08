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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>
#include <numeric>
#include <iomanip>

#include "../../general/communication.hpp"
#include "../../general/forall.hpp"
#ifdef MFEM_USE_MPI
#include "../fe/fe_base.hpp"
#include "../fespace.hpp"
#include "../pfespace.hpp"
#include "../../mesh/mesh.hpp"
#include "../../linalg/dtensor.hpp"

#include "fieldoperator.hpp"
#include "parameterspace.hpp"
#include "tuple.hpp"

namespace mfem::future
{

template<typename... Ts>
constexpr auto to_array(const std::tuple<Ts...>& tuple)
{
   constexpr auto get_array = [](const Ts&... x) { return std::array<typename std::common_type<Ts...>::type, sizeof...(Ts)> { x... }; };
   return std::apply(get_array, tuple);
}

namespace detail
{

template <typename lambda, std::size_t... i>
constexpr void for_constexpr(lambda&& f,
                             std::integral_constant<std::size_t, i>... Is)
{
   f(Is...);
}


template <std::size_t... n, typename lambda, typename... arg_types>
constexpr void for_constexpr(lambda&& f,
                             std::integer_sequence<std::size_t, n...>,
                             arg_types... args)
{
   (detail::for_constexpr(f, args..., std::integral_constant<std::size_t,n> {}),
    ...);
}

}  // namespace detail

template <typename lambda, std::size_t... i>
constexpr void for_constexpr(lambda&& f,
                             std::integer_sequence<std::size_t, i ... >)
{
   (f(std::integral_constant<std::size_t, i> {}), ...);
}

template <typename lambda>
constexpr void for_constexpr(lambda&& f, std::integer_sequence<std::size_t>) {}

template <int... n, typename lambda>
constexpr void for_constexpr(lambda&& f)
{
   detail::for_constexpr(f, std::make_integer_sequence<std::size_t, n> {}...);
}

template <typename lambda, typename arg_t>
constexpr void for_constexpr_with_arg(lambda&& f, arg_t&& arg,
                                      std::integer_sequence<std::size_t>)
{
   // Base case - do nothing for empty sequence
}

template <typename lambda, typename arg_t, std::size_t i, std::size_t... Is>
constexpr void for_constexpr_with_arg(lambda&& f, arg_t&& arg,
                                      std::integer_sequence<std::size_t, i, Is...>)
{
   f(std::integral_constant<std::size_t, i> {}, get<i>(arg));
   for_constexpr_with_arg(f, std::forward<arg_t>(arg),
                          std::integer_sequence<std::size_t, Is...> {});
}

template <typename lambda, typename arg_t>
constexpr void for_constexpr_with_arg(lambda&& f, arg_t&& arg)
{
   using indices =
      std::make_index_sequence<tuple_size<std::remove_reference_t<arg_t>>::value>;
   for_constexpr_with_arg(std::forward<lambda>(f), std::forward<arg_t>(arg),
                          indices{});
}

template <std::size_t I, typename Tuple, std::size_t... Is>
std::array<bool, sizeof...(Is)>
make_dependency_array(const Tuple& inputs, std::index_sequence<Is...>)
{
   return { (get<I>(inputs).GetFieldId() == get<Is>(inputs).GetFieldId())... };
}

template <typename... input_ts, std::size_t... Is>
auto make_dependency_map_impl(tuple<input_ts...> inputs,
                              std::index_sequence<Is...>)
{
   constexpr std::size_t N = sizeof...(input_ts);

   if constexpr (N == 0)
      return std::unordered_map<int, std::array<bool, 0>> {};

   std::unordered_map<int, std::array<bool, N>> map;

   (void)std::initializer_list<int>
   {
      (
         map[get<Is>(inputs).GetFieldId()] =
      make_dependency_array<Is>(inputs, std::make_index_sequence<N>{}),
      0
      )...
   };

   return map;
}

// @brief Create a dependency map from a tuple of inputs.
//
// @param inputs a tuple of objects derived from FieldOperator.
// @returns an unordered_map where the keys are the field IDs and the values
// are arrays of booleans indicating which inputs depend on each field ID.
template <typename... input_ts>
auto make_dependency_map(tuple<input_ts...> inputs)
{
   return make_dependency_map_impl(inputs, std::index_sequence_for<input_ts...> {});
}

// @brief Get the type name of a template parameter T.
//
// Convenient helper function for debugging.
// Usage example
// ```c++
// mfem::out << get_type_name<int>() << std::endl;
// ```
// prints "int".
template <typename T>
constexpr auto get_type_name() -> std::string_view
{
#if defined(__clang__)
   constexpr auto prefix = std::string_view {"[T = "};
   constexpr auto suffix = "]";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
   constexpr auto prefix = std::string_view {"with T = "};
   constexpr auto suffix = "; ";
   constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
   constexpr auto prefix = std::string_view {"get_type_name<"};
   constexpr auto suffix = ">(void)";
   constexpr auto function = std::string_view{__FUNCSIG__};
#else
#error Unsupported compiler
#endif

   const auto start = function.find(prefix) + prefix.size();
   const auto end = function.find(suffix);
   const auto size = end - start;

   return function.substr(start, size);
}

template <typename Tuple, std::size_t... Is>
void print_tuple_impl(const Tuple& t, std::index_sequence<Is...>)
{
   ((out << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
}

// @brief Helper function to print a single tuple.
//
// @param t The tuple to print.
template <typename... Args>
void print_tuple(const std::tuple<Args...>& t)
{
   out << "(";
   print_tuple_impl(t, std::index_sequence_for<Args...> {});
   out << ")";
}

/// @brief Pretty print an mfem::DenseMatrix to out
///
/// Formatted s.t. the output is
/// [[v00, v01, ..., v0n],
///  [v10, v11, ..., v1n],
///             ..., vmn]]
/// which is compatible with numpy syntax.
///
/// @param out ostream to print to
/// @param A mfem::DenseMatrix to print
inline
void pretty_print(std::ostream &out, const mfem::DenseMatrix &A)
{
   // Determine the max width of any entry in scientific notation
   int max_width = 0;
   for (int i = 0; i < A.NumRows(); ++i)
   {
      for (int j = 0; j < A.NumCols(); ++j)
      {
         std::ostringstream oss;
         oss << std::scientific << std::setprecision(2) << A(i, j);
         max_width = std::max(max_width, static_cast<int>(oss.str().length()));
      }
   }

   out << "[\n";
   for (int i = 0; i < A.NumRows(); ++i)
   {
      out << "  [";
      for (int j = 0; j < A.NumCols(); ++j)
      {
         out << std::setw(max_width) << std::scientific << std::setprecision(2) <<
             A(i, j);

         if (j < A.NumCols() - 1)
         {
            out << ", ";
         }
      }
      out << "]";
      if (i < A.NumRows() - 1)
      {
         out << ",\n";
      }
      else
      {
         out << "\n";
      }
   }
   out << "]\n";
}

/// @brief Pretty print an mfem::Vector to out
///
/// Formatted s.t. the output is [v0, v1, ..., vn] which
/// is compatible with numpy syntax.
///
/// @param v vector of vectors to print
inline
void pretty_print(const mfem::Vector& v)
{
   out << "[";
   for (int i = 0; i < v.Size(); i++)
   {
      out << v(i);
      if (i < v.Size() - 1)
      {
         out << ", ";
      }
   }
   out << "]\n";
}

/// @brief Pretty print an mfem::Array to out
///
/// T has to have an overloaded operator<<
///
/// Formatted s.t. the output is [v0, v1, ..., vn] which
/// is compatible with numpy syntax.
///
/// @param v vector of vectors to print
template <typename T>
void pretty_print(const mfem::Array<T>& v)
{
   out << "[";
   for (int i = 0; i < v.Size(); i++)
   {
      out << v[i];
      if (i < v.Size() - 1)
      {
         out << ", ";
      }
   }
   out << "]\n";
}

/// @brief Pretty prints an unordered map of std::array to out
///
/// Useful for printing the output of make_dependency_map
///
/// @param map unordered map to print
/// @tparam T type of array elements
/// @tparam N size of array
template<typename K, typename T, std::size_t N>
void pretty_print(const std::unordered_map<K,std::array<T,N>>& map)
{
   out << "{";
   std::size_t count = 0;
   for (const auto& [key, value] : map)
   {
      out << key << ": [";
      for (std::size_t i = 0; i < N; i++)
      {
         out << value[i];
         if (i < N-1) { out << ", "; }
      }
      out << "]";
      if (count < map.size() - 1)
      {
         out << ", ";
      }
      count++;
   }
   out << "}\n";
}

inline
void print_mpi_root(const std::string& msg)
{
   auto myrank = Mpi::WorldRank();
   if (myrank == 0)
   {
      out << msg << std::endl;
      out.flush(); // Ensure output is flushed
   }
}

/// @brief print with MPI rank synchronization
///
/// @param msg Message to print
inline
void print_mpi_sync(const std::string& msg)
{
   auto myrank = static_cast<size_t>(Mpi::WorldRank());
   auto nranks = static_cast<size_t>(Mpi::WorldSize());

   if (nranks == 1)
   {
      // Single process case - just print directly
      out << msg << std::endl;
      return;
   }

   // First gather string lengths
   size_t msg_len = msg.length();
   std::vector<size_t> lengths(nranks);
   MPI_Gather(&msg_len, 1, MPITypeMap<size_t>::mpi_type,
              lengths.data(), 1, MPITypeMap<size_t>::mpi_type,
              0, MPI_COMM_WORLD);

   if (myrank == 0)
   {
      // Rank 0: Allocate receive buffer based on gathered lengths
      std::vector<std::string> messages(nranks);
      messages[0] = msg; // Store rank 0's message

      // Receive messages from other ranks
      for (size_t r = 1; r < nranks; r++)
      {
         std::vector<char> buffer(lengths[r] + 1);
         MPI_Recv(buffer.data(), static_cast<int>(lengths[r]), MPI_CHAR,
                  static_cast<int>(r), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         messages[r] = std::string(buffer.data(), static_cast<size_t>(lengths[r]));
      }

      // Print all messages in rank order
      for (size_t r = 0; r < nranks; r++)
      {
         out << "[Rank " << r << "] " << messages[r] << std::endl;
      }
      out.flush();
   }
   else
   {
      // Other ranks: Send message to rank 0
      MPI_Send(const_cast<char*>(msg.c_str()), static_cast<int>(msg_len), MPI_CHAR,
               0, 0, MPI_COMM_WORLD);
   }

   // Final barrier to ensure completion
   MPI_Barrier(MPI_COMM_WORLD);
}

/// @brief Pretty print an mfem::Vector with MPI rank
///
/// @param v vector to print
inline
void pretty_print_mpi(const mfem::Vector& v)
{
   std::stringstream ss;
   ss << "[";
   for (int i = 0; i < v.Size(); i++)
   {
      ss << v(i);
      if (i < v.Size() - 1) { ss << ", "; }
   }
   ss << "]";

   print_mpi_sync(ss.str());
}


template <typename ... Ts>
constexpr auto decay_types(tuple<Ts...> const &)
-> tuple<std::remove_cv_t<std::remove_reference_t<Ts>>...>;

template <typename T>
using decay_tuple = decltype(decay_types(std::declval<T>()));

template <class F> struct FunctionSignature;

template <typename output_t, typename... input_ts>
struct FunctionSignature<output_t(input_ts...)>
{
   using return_t = output_t;
   using parameter_ts = tuple<input_ts...>;
};

template <class T> struct create_function_signature;

// Specialization for member functions (lambdas)
template <typename output_t, typename T, typename... input_ts>
struct create_function_signature<output_t (T::*)(input_ts...) const>
{
   using type = FunctionSignature<output_t(input_ts...)>;
};

// Specialization for function pointers
template <typename output_t, typename... input_ts>
struct create_function_signature<output_t (*)(input_ts...)>
{
   using type = FunctionSignature<output_t(input_ts...)>;
};

template <typename T>
constexpr int GetFieldId()
{
   return T::GetFieldId();
}

template <typename Tuple, std::size_t... Is>
constexpr auto extract_field_ids_impl(Tuple&& t, std::index_sequence<Is...>)
{
   return std::array<int, sizeof...(Is)>
   {
      std::decay_t<decltype(std::get<Is>(t))>{}.GetFieldId()...
   };
}

/// @brief Extracts field IDs from a tuple of objects derived from FieldOperator.
///
/// @param t the tuple to extract field IDs from.
/// @returns an array of field IDs.
template <typename... Ts>
constexpr auto extract_field_ids(const std::tuple<Ts...>& t)
{
   return extract_field_ids_impl(t, std::index_sequence_for<Ts...> {});
}

/// @brief Helper function to check if an element is in the array.
///
/// @param arr the array to search in.
/// @param size the size of the array.
/// @param value the value to search for.
/// @returns true if the value is found, false otherwise.
constexpr bool contains(const int* arr, std::size_t size, int value)
{
   for (std::size_t i = 0; i < size; ++i)
   {
      if (arr[i] == value)
      {
         return true;
      }
   }
   return false;
}

/// @brief Function to count unique field IDs in a tuple.
///
/// @param t the tuple to count unique field IDs from.
/// @returns the number of unique field IDs.
template <typename... Ts>
constexpr std::size_t count_unique_field_ids(const std::tuple<Ts...>& t)
{
   auto ids = extract_field_ids(t);
   constexpr std::size_t size = sizeof...(Ts);

   std::array<int, size> unique_ids = {};
   std::size_t unique_count = 0;

   for (std::size_t i = 0; i < size; ++i)
   {
      if (!contains(unique_ids.data(), unique_count, ids[i]))
      {
         unique_ids[unique_count] = ids[i];
         ++unique_count;
      }
   }

   return unique_count;
}

/// @brief Get marked entries from an std::array based on a marker array.
///
/// @param a the std::array to get entries from.
/// @param marker the marker std::array indicating which entries to get.
/// @returns a std::vector containing the marked entries.
template <typename T, std::size_t N>
auto get_marked_entries(
   const std::array<T, N> &a,
   const std::array<bool, N> &marker)
{
   std::vector<T> r;
   for (int i = 0; i < N; i++)
   {
      if (marker[i])
      {
         r.push_back(a[i]);
      }
   }
   return r;
}

/// @brief Filter fields from a tuple based on their field IDs.
///
/// @param t the tuple to filter fields from.
/// @returns a tuple containing only the fields with field IDs not equal to -1.
template <typename... Ts>
constexpr auto filter_fields(const std::tuple<Ts...>& t)
{
   return std::tuple_cat(
             std::conditional_t<Ts::GetFieldId() != -1, std::tuple<Ts>, std::tuple<>> {}...);
}

/// @brief FieldDescriptor struct
///
/// This struct is used to store information about a field.
struct FieldDescriptor
{
   using data_variant_t =
      std::variant<const FiniteElementSpace *,
      const ParFiniteElementSpace *,
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
};

namespace dfem
{
template <class... T> constexpr bool always_false = false;
}

/// @brief Entity struct
///
/// This struct is used to store information about an entity type.
namespace Entity
{
struct Element;
struct BoundaryElement;
struct Face;
struct BoundaryFace;
}

/// @brief ThreadBlocks struct
///
/// This struct is used to store information about thread blocks
/// for GPU dispatch.
struct ThreadBlocks
{
   int x = 1;
   int y = 1;
   int z = 1;
};

#if defined(MFEM_USE_CUDA_OR_HIP)
template <typename func_t>
__global__ void forall_kernel_shmem(func_t f, int n)
{
   int i = blockIdx.x;
   extern __shared__ real_t shmem[];
   if (i < n)
   {
      f(i, shmem);
   }
}
#endif

template <typename func_t>
void forall(func_t f,
            const int &N,
            const ThreadBlocks &blocks,
            int num_shmem = 0,
            real_t *shmem = nullptr)
{
   if (Device::Allows(Backend::CUDA_MASK) ||
       Device::Allows(Backend::HIP_MASK))
   {
#if defined(MFEM_USE_CUDA_OR_HIP)
      // int gridsize = (N + Z - 1) / Z;
      int num_bytes = num_shmem * sizeof(decltype(shmem));
      dim3 block_size(blocks.x, blocks.y, blocks.z);
      forall_kernel_shmem<<<N, block_size, num_bytes>>>(f, N);
#if defined(MFEM_USE_CUDA)
      MFEM_GPU_CHECK(cudaGetLastError());
#elif defined(MFEM_USE_HIP)
      MFEM_GPU_CHECK(hipGetLastError());
#endif
      MFEM_DEVICE_SYNC;
#endif
   }
   else if (Device::Allows(Backend::CPU_MASK))
   {
      MFEM_ASSERT(!((bool)num_shmem != (bool)shmem),
                  "Backend::CPU needs a pre-allocated shared memory block");
      for (int i = 0; i < N; i++)
      {
         f(i, shmem);
      }
   }
   else
   {
      MFEM_ABORT("no compute backend available");
   }
}

/// @todo To be removed.
class FDJacobian : public Operator
{
public:
   FDJacobian(const Operator &op, const Vector &x, real_t fixed_eps = 0.0) :
      Operator(op.Height(), op.Width()),
      op(op),
      x(x),
      fixed_eps(fixed_eps)
   {
      f.UseDevice(x.UseDevice());
      f.SetSize(Height());

      xpev.UseDevice(x.UseDevice());
      xpev.SetSize(Width());

      op.Mult(x, f);

      const real_t xnorm_local = x.Norml2();
      MPI_Allreduce(&xnorm_local, &xnorm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    MPI_COMM_WORLD);
   }

   void Mult(const Vector &v, Vector &y) const override
   {
      // See [1] for choice of eps.
      //
      // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
      // finite difference matrix-vector products in Newton-Krylov solvers for
      // implicit climate dynamics with spectral elements. Procedia Computer
      // Science, 51, pp.2036-2045.
      real_t eps;
      if (fixed_eps > 0.0)
      {
         eps = fixed_eps;
      }
      else
      {
         const real_t vnorm_local = v.Norml2();
         real_t vnorm;
         MPI_Allreduce(&vnorm_local, &vnorm, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                       MPI_COMM_WORLD);
         eps = lambda * (lambda + xnorm / vnorm);
      }

      // x + eps * v
      {
         const auto d_v = v.Read();
         const auto d_x = x.Read();
         auto d_xpev = xpev.Write();
         mfem::forall(x.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_xpev[i] = d_x[i] + eps * d_v[i];
         });
      }

      // y = f(x + eps * v)
      op.Mult(xpev, y);

      // y = (f(x + eps * v) - f(x)) / eps
      {
         const auto d_f = f.Read();
         auto d_y = y.ReadWrite();
         mfem::forall(f.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_y[i] = (d_y[i] - d_f[i]) / eps;
         });
      }
   }

   virtual MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

private:
   const Operator &op;
   Vector x, f;
   mutable Vector xpev;
   real_t lambda = 1.0e-6;
   real_t fixed_eps;
   real_t xnorm;
};

/// @brief Find the index of a field descriptor in a vector of field descriptors.
///
/// @param id the field ID to search for.
/// @param fields the vector of field descriptors.
/// @returns the index of the field descriptor with the given ID,
/// or SIZE_MAX if not found.
inline
std::size_t FindIdx(const std::size_t& id,
                    const std::vector<FieldDescriptor>& fields)
{
   for (std::size_t i = 0; i < fields.size(); i++)
   {
      if (fields[i].id == id)
      {
         return i;
      }
   }
   return SIZE_MAX;
}

/// @brief Get the vdof size of a field descriptor.
///
/// @param f the field descriptor.
/// @returns the vdof size of the field descriptor.
inline
int GetVSize(const FieldDescriptor &f)
{
   return std::visit([](auto arg)
   {
      if (arg == nullptr)
      {
         MFEM_ABORT("FieldDescriptor data is nullptr");
      }

      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->GetVSize();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetVSize on type");
      }
      return 0; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the element vdofs of a field descriptor.
///
/// @note Can't be used with ParameterSpace.
///
/// @param f the field descriptor.
/// @param el the element index.
/// @param vdofs the array to store the element vdofs.
inline
void GetElementVDofs(const FieldDescriptor &f, int el, Array<int> &vdofs)
{
   return std::visit([&](auto arg)
   {
      if (arg == nullptr)
      {
         MFEM_ABORT("FieldDescriptor data is nullptr");
      }

      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         arg->GetElementVDofs(el, vdofs);
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         arg->GetElementVDofs(el, vdofs);
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         MFEM_ABORT("internal error");
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetElementVdofs on type");
      }
   }, f.data);
}

/// @brief Get the true dof size of a field descriptor.
///
/// @param f the field descriptor.
/// @returns the true dof size of the field descriptor.
inline
int GetTrueVSize(const FieldDescriptor &f)
{
   return std::visit([](auto arg)
   {
      if (arg == nullptr)
      {
         MFEM_ABORT("FieldDescriptor data is nullptr");
      }

      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return arg->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->GetTrueVSize();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetTrueVSize on type");
      }
      return 0; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the vdim of a field descriptor.
///
/// @param f the field descriptor.
/// @returns the vdim of the field descriptor.
inline
int GetVDim(const FieldDescriptor &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return arg->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->GetVDim();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetVDim on type");
      }
      return 0; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the spatial dimension of a field descriptor.
///
/// @param f the field descriptor.
/// @tparam entity_t the entity type (see Entity).
/// @returns the spatial dimension of the field descriptor.
template <typename entity_t>
int GetDimension(const FieldDescriptor &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         if constexpr (std::is_same_v<entity_t, Entity::Element>)
         {
            return arg->GetMesh()->Dimension();
         }
         else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
         {
            return arg->GetMesh()->Dimension() - 1;
         }
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->Dimension();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetDimension on type");
      }
      return 0; // Unreachable, but avoids compiler warning
   }, f.data);
}


/// @brief Get the prolongation operator for a field descriptor.
///
/// @param f the field descriptor.
/// @returns the prolongation operator for the field descriptor.
inline
const Operator *get_prolongation(const FieldDescriptor &f)
{
   return std::visit([](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->GetProlongationMatrix();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetProlongation on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the element restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the element dof ordering.
/// @returns the element restriction operator for the field descriptor in
/// specified ordering.
inline
const Operator *get_element_restriction(const FieldDescriptor &f,
                                        ElementDofOrdering o)
{
   return std::visit([&o](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>
                    || std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return arg->GetElementRestriction(o);
      }
      else
      {
         static_assert(dfem::always_false<T>,
                       "can't use get_element_restriction on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the face restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the face dof ordering.
/// @param ft the face type
/// @param m indicator if single or double valued
/// @returns the face restriction operator for the field descriptor in
/// specified ordering.
inline
const Operator *get_face_restriction(const FieldDescriptor &f,
                                     ElementDofOrdering o,
                                     FaceType ft,
                                     L2FaceValues m)
{
   return std::visit([&o, &ft, &m](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetFaceRestriction(o, ft, m);
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         // ParameterSpace does not support face restrictions
         MFEM_ABORT("internal error");
      }
      else
      {
         static_assert(dfem::always_false<T>,
                       "can't use get_face_restriction on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Get the restriction operator for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the element dof ordering.
/// @returns the restriction operator for the field descriptor in
/// specified ordering.
template <typename entity_t>
inline
const Operator *get_restriction(const FieldDescriptor &f,
                                const ElementDofOrdering &o)
{
   if constexpr (std::is_same_v<entity_t, Entity::Element>)
   {
      return get_element_restriction(f, o);
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      return get_face_restriction(f, o, FaceType::Boundary,
                                  L2FaceValues::SingleValued);
   }
   MFEM_ABORT("restriction not implemented for Entity");
   return nullptr;
}

/// @brief Get a transpose restriction callback for a field descriptor.
///
/// @param f the field descriptor.
/// @param o the element dof ordering.
/// @param fop the field operator.
/// @returns a tuple containing a std::function with the transpose
/// restriction callback and it's height.
template <typename entity_t, typename fop_t>
inline std::tuple<std::function<void(const Vector&, Vector&)>, int>
get_restriction_transpose(
   const FieldDescriptor &f,
   const ElementDofOrdering &o,
   const fop_t &fop)
{
   if constexpr (is_sum_fop<fop_t>::value)
   {
      auto RT = [=](const Vector &v_e, Vector &v_l)
      {
         v_l += v_e;
      };
      return std::make_tuple(RT, 1);
   }
   else
   {
      const Operator *R = get_restriction<entity_t>(f, o);
      std::function<void(const Vector&, Vector&)> RT = [=](const Vector &x, Vector &y)
      {
         R->AddMultTranspose(x, y);
      };
      return std::make_tuple(RT, R->Height());
   }
   return std::make_tuple(
             std::function<void(const Vector&, Vector&)>([](const Vector&, Vector&)
   {
      /* no-op */
   }), 0); // Never reached, but avoids compiler warning.
}

/// @brief Apply the prolongation operator to a field.
///
/// @param field the field descriptor.
/// @param x the input vector in tdofs.
/// @param field_l the output vector in vdofs.
inline
void prolongation(const FieldDescriptor field, const Vector &x, Vector &field_l)
{
   const auto P = get_prolongation(field);
   field_l.SetSize(P->Height());
   P->Mult(x, field_l);
}

/// @brief Apply the prolongation operator to a vector of fields.
///
/// x is a long vector containing the data for all fields on tdofs and
/// fields contains the information about each individual field to retrieve
/// it's corresponding prolongation.
///
/// @param fields the array of field descriptors.
/// @param x the input vector in tdofs.
/// @param fields_l the array of output vectors in vdofs.
/// @tparam N the number of fields.
/// @tparam M the number of output fields.
template <std::size_t N, std::size_t M>
void prolongation(const std::array<FieldDescriptor, N> fields,
                  const Vector &x,
                  std::array<Vector, M> &fields_l)
{
   int data_offset = 0;
   for (int i = 0; i < N; i++)
   {
      const auto P = get_prolongation(fields[i]);
      const int width = P->Width();
      // const Vector x_i(x.GetData() + data_offset, width);
      const Vector x_i(const_cast<Vector&>(x), data_offset, width);
      fields_l[i].SetSize(P->Height());

      P->Mult(x_i, fields_l[i]);
      data_offset += width;
   }
}

/// @brief Apply the prolongation operator to a vector of fields.
///
/// x is a long vector containing the data for all fields on tdofs and
/// fields contains the information about each individual field to retrieve
/// it's corresponding prolongation.
///
/// @param fields the array of field descriptors.
/// @param x the input vector in tdofs.
/// @param fields_l the array of output vectors in vdofs.
inline
void prolongation(const std::vector<FieldDescriptor> fields,
                  const Vector &x,
                  std::vector<Vector> &fields_l)
{
   int data_offset = 0;
   for (std::size_t i = 0; i < fields.size(); i++)
   {
      const auto P = get_prolongation(fields[i]);
      const int width = P->Width();
      const Vector x_i(const_cast<Vector&>(x), data_offset, width);
      fields_l[i].SetSize(P->Height());
      P->Mult(x_i, fields_l[i]);
      data_offset += width;
   }
}

inline
void get_lvectors(const std::vector<FieldDescriptor> fields,
                  const Vector &x,
                  std::vector<Vector> &fields_l)
{
   int data_offset = 0;
   for (std::size_t i = 0; i < fields.size(); i++)
   {
      const int sz = GetVSize(fields[i]);
      fields_l[i].SetSize(sz);

      const Vector x_i(const_cast<Vector&>(x), data_offset, sz);
      fields_l[i] = x_i;

      data_offset += sz;
   }
}

/// @brief Get a transpose prolongation callback for a field descriptor.
///
/// In the special case of a one field operator, the transpose prolongation
/// is a simple sum of the local vector that is reduced to the global vector.
///
/// @param f the field descriptor.
/// @param fop the field operator.
/// @param mpi_comm the MPI communicator.
/// @tparam fop_t the field operator type.
template <typename fop_t>
inline
std::function<void(const Vector&, Vector&)> get_prolongation_transpose(
   const FieldDescriptor &f,
   const fop_t &fop,
   MPI_Comm mpi_comm)
{
   if constexpr (is_sum_fop<fop_t>::value)
   {
      auto PT = [=](const Vector &r_local, Vector &y)
      {
         MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
         real_t local_sum = r_local.Sum();
         MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      };
      return PT;
   }
   else if constexpr (is_identity_fop<fop_t>::value)
   {
      auto PT = [=](const Vector &r_local, Vector &y)
      {
         y = r_local;
      };
      return PT;
   }
   const Operator *P = get_prolongation(f);
   auto PT = [=](const Vector &r_local, Vector &y)
   {
      P->MultTranspose(r_local, y);
   };
   return PT;
}

/// @brief Apply the restriction operator to a field.
///
/// @param u the field descriptor.
/// @param u_l the input vector in vdofs.
/// @param field_e the output vector in edofs.
/// @param ordering the element dof ordering.
/// @tparam entity_t the entity type (see Entity).
template <typename entity_t>
void restriction(const FieldDescriptor u,
                 const Vector &u_l,
                 Vector &field_e,
                 ElementDofOrdering ordering)
{
   const auto R = get_restriction<entity_t>(u, ordering);
   MFEM_ASSERT(R->Width() == u_l.Size(),
               "restriction not applicable to given data size");
   const int height = R->Height();
   field_e.SetSize(height);
   R->Mult(u_l, field_e);
}

/// @brief Apply the restriction operator to a vector of fields.
///
/// @param u the vector of field descriptors.
/// @param u_l the vector of input vectors in vdofs.
/// @param fields_e the vector of output vectors in edofs.
/// @param ordering the element dof ordering.
/// @param offset the array index offset to start writing in fields_e.
/// @tparam entity_t the entity type (see Entity).
template <typename entity_t>
void restriction(const std::vector<FieldDescriptor> u,
                 const std::vector<Vector> &u_l,
                 std::vector<Vector> &fields_e,
                 ElementDofOrdering ordering,
                 const int offset = 0)
{
   for (std::size_t i = 0; i < u.size(); i++)
   {
      const auto R = get_restriction<entity_t>(u[i], ordering);
      MFEM_ASSERT(R->Width() == u_l[i].Size(),
                  "restriction not applicable to given data size");
      const int height = R->Height();
      fields_e[i + offset].SetSize(height);
      R->Mult(u_l[i], fields_e[i + offset]);
   }
}

// TODO: keep this temporarily
template <std::size_t N, std::size_t M>
void element_restriction(const std::array<FieldDescriptor, N> u,
                         const std::array<Vector, N> &u_l,
                         std::array<Vector, M> &fields_e,
                         ElementDofOrdering ordering,
                         const int offset = 0)
{
   for (int i = 0; i < N; i++)
   {
      const auto R = get_element_restriction(u[i], ordering);
      MFEM_ASSERT(R->Width() == u_l[i].Size(),
                  "element restriction not applicable to given data size");
      const int height = R->Height();
      fields_e[i + offset].SetSize(height);
      R->Mult(u_l[i], fields_e[i + offset]);
   }
}

/// @brief Get the number of entities of a given type.
///
/// @param mesh the mesh.
/// @tparam entity_t the entity type (see Entity).
/// @returns the number of entities of the given type.
template <typename entity_t>
int GetNumEntities(const mfem::Mesh &mesh)
{
   if constexpr (std::is_same_v<entity_t, Entity::Element>)
   {
      return mesh.GetNE();
   }
   else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      return mesh.GetNBE();
   }
   else
   {
      static_assert(dfem::always_false<entity_t>, "can't use GetNumEntites on type");
   }
   return 0; // Unreachable, but avoids compiler warning
}

/// @brief Get the GetDofToQuad object for a given entity type.
///
/// This function retrieves the DofToQuad object for a given field descriptor
/// and integration rule.
///
/// @param f the field descriptor.
/// @param ir the integration rule.
/// @param mode the mode of the DofToQuad object.
/// @tparam entity_t the entity type (see Entity).
template <typename entity_t>
inline
const DofToQuad *GetDofToQuad(const FieldDescriptor &f,
                              const IntegrationRule &ir,
                              DofToQuad::Mode mode)
{
   return std::visit([&ir, &mode](auto&& arg) -> const DofToQuad*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>
                    || std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         if constexpr (std::is_same_v<entity_t, Entity::Element>)
         {
            return &arg->GetTypicalFE()->GetDofToQuad(ir, mode);
         }
         else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
         {
            return &arg->GetTypicalTraceElement()->GetDofToQuad(ir, mode);
         }
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         return &arg->GetDofToQuad();
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetDofToQuad on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Check the compatibility of a field operator type with a
/// FieldDescriptor.
///
/// This function checks if the field operator type is compatible with the
/// FieldDescriptor type.
///
/// @param f the field descriptor.
/// @tparam field_operator_t the field operator type.
template <typename field_operator_t>
void CheckCompatibility(const FieldDescriptor &f)
{
   std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         if constexpr (std::is_same_v<field_operator_t, Value<>>)
         {
            // Supported by all FE spaces
         }
         else if constexpr (std::is_same_v<field_operator_t, Gradient<>>)
         {
            MFEM_ASSERT(arg->GetTypicalElement()->GetMapType() ==
                        FiniteElement::MapType::VALUE,
                        "Gradient not compatible with FE");
         }
         else
         {
            static_assert(dfem::always_false<T, field_operator_t>,
                          "FieldOperator not compatible with FiniteElementSpace");
         }
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         if constexpr (std::is_same_v<field_operator_t, Identity<>>)
         {
            // Only supported field operation for ParameterSpace
         }
         else
         {
            static_assert(dfem::always_false<T, field_operator_t>,
                          "FieldOperator not compatible with ParameterSpace");
         }
      }
      else
      {
         static_assert(dfem::always_false<T, field_operator_t>,
                       "Operator not compatible with FE");
      }
   }, f.data);
}

/// @brief Get the size on quadrature point for a field operator type
/// and FieldDescriptor combination.
///
/// @tparam entity_t the entity type (see Entity).
/// @tparam field_operator_t the field operator type.
/// @param f the field descriptor.
/// @returns the size on quadrature point.
template <typename entity_t, typename field_operator_t>
int GetSizeOnQP(const field_operator_t &, const FieldDescriptor &f)
{
   // CheckCompatibility<field_operator_t>(f);

   if constexpr (is_value_fop<field_operator_t>::value)
   {
      return GetVDim(f);
   }
   else if constexpr (is_gradient_fop<field_operator_t>::value)
   {
      return GetVDim(f) * GetDimension<entity_t>(f);
   }
   else if constexpr (is_identity_fop<field_operator_t>::value)
   {
      return GetVDim(f);
   }
   else if constexpr (is_sum_fop<field_operator_t>::value)
   {
      return 1;
   }
   else
   {
      MFEM_ABORT("can't get size on quadrature point for field descriptor");
   }
   return 0; // Unreachable, but avoids compiler warning
}

/// @brief Create a map from field operator types to FieldDescriptor indices.
///
/// @param fields the vector of field descriptors.
/// @param fops the field operator types.
/// @tparam entity_t the entity type (see Entity).
/// @returns an array mapping field operator types to field descriptor indices.
template <typename entity_t, typename field_operator_ts>
std::array<size_t, tuple_size<field_operator_ts>::value>
create_descriptors_to_fields_map(
   const std::vector<FieldDescriptor> &fields,
   field_operator_ts &fops)
{
   std::array<size_t, tuple_size<field_operator_ts>::value> map;

   auto find_id = [](const std::vector<FieldDescriptor> &fields, std::size_t i)
   {
      auto it = std::find_if(begin(fields), end(fields),
                             [&](const FieldDescriptor &field)
      {
         return field.id == i;
      });

      if (it == fields.end())
      {
         return SIZE_MAX;
      }
      return static_cast<size_t>(it - fields.begin());
   };

   auto f = [&](auto &fop, auto &map)
   {
      if constexpr (std::is_same_v<std::decay_t<decltype(fop)>, Weight>)
      {
         // TODO-bug: stealing dimension from the first field
         fop.dim = GetDimension<entity_t>(fields[0]);
         fop.vdim = 1;
         fop.size_on_qp = 1;
         map = SIZE_MAX;
      }
      else
      {
         int i = find_id(fields, fop.GetFieldId());
         if (i != -1)
         {
            fop.dim = GetDimension<entity_t>(fields[i]);
            fop.vdim = GetVDim(fields[i]);
            fop.size_on_qp = GetSizeOnQP<entity_t>(fop, fields[i]);
            map = i;
         }
         else
         {
            MFEM_ABORT("can't find field for id: " << fop.GetFieldId());
         }
      }
   };

   for_constexpr<tuple_size<field_operator_ts>::value>([&](auto idx)
   {
      f(get<idx>(fops), map[idx]);
   });

   return map;
}

/// @brief Wrap input memory for a given set of inputs.
template <typename input_t, std::size_t... i>
std::array<DeviceTensor<3>, sizeof...(i)> wrap_input_memory(
   std::array<Vector, sizeof...(i)> &input_qp_mem, int num_qp, int num_entities,
   const input_t &inputs, std::index_sequence<i...>)
{
   return {DeviceTensor<3>(input_qp_mem[i].Write(), get<i>(inputs).size_on_qp, num_qp, num_entities) ...};
}

/// @brief Create input memory for a given set of inputs.
template <typename input_t, std::size_t... i>
std::array<Vector, sizeof...(i)> create_input_qp_memory(
   int num_qp,
   int num_entities,
   input_t &inputs,
   std::index_sequence<i...>)
{
   return {Vector(get<i>(inputs).size_on_qp * num_qp * num_entities)...};
}

/// @brief DofToQuadMap struct
///
/// This struct is used to store the mapping from degrees of freedom to
/// quadrature points for a given field operator type.
struct DofToQuadMap
{
   /// Enumeration for the indices of the mappings B and G.
   enum Index
   {
      QP,
      DIM,
      DOF
   };

   /// @brief Basis functions evaluated at quadrature points.
   ///
   /// This is a 3D tensor with dimensions (num_qp, dim, num_dofs).
   DeviceTensor<3, const real_t> B;

   /// @brief Gradient of the basis functions evaluated at quadrature points.
   ///
   /// This is a 3D tensor with dimensions (num_qp, dim, num_dofs).
   DeviceTensor<3, const real_t> G;

   /// Reverse mapping indicating which input this map belongs to.
   int which_input = -1;
};

/// @brief Get the size on quadrature point for a given set of inputs.
///
/// @param inputs the inputs tuple.
/// @returns a vector containing the size on quadrature point for each input.
template <typename input_t, std::size_t... i>
std::vector<int> get_input_size_on_qp(
   const input_t &inputs,
   std::index_sequence<i...>)
{
   return {get<i>(inputs).size_on_qp...};
}

struct SharedMemory
{
   enum Index
   {
      INPUT_DTQ,
      OUTPUT_DTQ,
      FIELD,
      DIRECTION,
      INPUT,
      SHADOW,
      OUTPUT,
      TEMP
   };
};

template <std::size_t num_fields, std::size_t num_inputs, std::size_t num_outputs>
struct SharedMemoryInfo
{
   int total_size;
   std::array<int, 8> offsets;
   std::array<std::array<int, 2>, num_inputs> input_dtq_sizes;
   std::array<std::array<int, 2>, num_outputs> output_dtq_sizes;
   std::array<int, num_fields> field_sizes;
   int direction_size;
   std::array<int, num_inputs> input_sizes;
   std::array<int, num_inputs> shadow_sizes;
   int residual_size;
   std::array<int, 6> temp_sizes;
};

template <typename entity_t, std::size_t num_fields, std::size_t num_inputs, std::size_t num_outputs, typename input_t>
SharedMemoryInfo<num_fields, num_inputs, num_outputs>
get_shmem_info(
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
   const std::vector<FieldDescriptor> &fields,
   const int &num_entities,
   const input_t &inputs,
   const int &num_qp,
   const std::vector<int> &input_size_on_qp,
   const int &residual_size_on_qp,
   const ElementDofOrdering &dof_ordering,
   const int &derivative_action_field_idx = -1)
{
   std::array<int, 8> offsets = {0};
   int total_size = 0;

   offsets[SharedMemory::Index::INPUT_DTQ] = total_size;
   std::array<std::array<int, 2>, num_inputs> input_dtq_sizes;
   int max_dtq_qps = 0;
   int max_dtq_dofs = 0;
   for (std::size_t i = 0; i < num_inputs; i++)
   {
      auto a = input_dtq_maps[i].B.GetShape();
      input_dtq_sizes[i][0] = a[0] * a[1] * a[2];
      auto b = input_dtq_maps[i].G.GetShape();
      input_dtq_sizes[i][1] = b[0] * b[1] * b[2];

      max_dtq_qps = std::max(max_dtq_qps, a[DofToQuadMap::Index::QP]);
      max_dtq_dofs = std::max(max_dtq_dofs, a[DofToQuadMap::Index::DOF]);

      total_size += std::accumulate(std::begin(input_dtq_sizes[i]),
                                    std::end(input_dtq_sizes[i]),
                                    0);
   }

   offsets[SharedMemory::Index::OUTPUT_DTQ] = total_size;
   std::array<std::array<int, 2>, num_outputs> output_dtq_sizes;
   for (std::size_t i = 0; i < num_outputs; i++)
   {
      auto a = output_dtq_maps[i].B.GetShape();
      output_dtq_sizes[i][0] = a[0] * a[1] * a[2];
      auto b = output_dtq_maps[i].G.GetShape();
      output_dtq_sizes[i][1] = b[0] * b[1] * b[2];

      max_dtq_qps = std::max(max_dtq_qps, a[DofToQuadMap::Index::QP]);
      max_dtq_dofs = std::max(max_dtq_dofs, a[DofToQuadMap::Index::DOF]);

      total_size += std::accumulate(std::begin(output_dtq_sizes[i]),
                                    std::end(output_dtq_sizes[i]),
                                    0);
   }

   offsets[SharedMemory::Index::FIELD] = total_size;
   std::array<int, num_fields> field_sizes;
   for (std::size_t i = 0; i < num_fields; i++)
   {
      field_sizes[i] =
         num_entities
         ? (get_restriction<entity_t>(fields[i], dof_ordering)->Height()
            / num_entities)
         : 0;
   }
   total_size += std::accumulate(
                    std::begin(field_sizes), std::end(field_sizes), 0);

   offsets[SharedMemory::Index::DIRECTION] = total_size;
   int direction_size = 0;
   if (derivative_action_field_idx != -1)
   {
      direction_size =
         num_entities ? (get_restriction<entity_t>(
                            fields[derivative_action_field_idx], dof_ordering)
                         ->Height()
                         / num_entities)
         : 0;
      total_size += direction_size;
   }

   offsets[SharedMemory::Index::INPUT] = total_size;
   std::array<int, num_inputs> input_sizes;
   for (std::size_t i = 0; i < num_inputs; i++)
   {
      input_sizes[i] = input_size_on_qp[i] * num_qp;
   }
   total_size += std::accumulate(
                    std::begin(input_sizes), std::end(input_sizes), 0);

   offsets[SharedMemory::Index::SHADOW] = total_size;
   std::array<int, num_inputs> shadow_sizes{0};
   if (derivative_action_field_idx != -1)
   {
      for (std::size_t i = 0; i < num_inputs; i++)
      {
         shadow_sizes[i] = input_size_on_qp[i] * num_qp;
      }
      total_size += std::accumulate(
                       std::begin(shadow_sizes), std::end(shadow_sizes), 0);
   }

   offsets[SharedMemory::Index::OUTPUT] = total_size;
   const int residual_size = residual_size_on_qp;
   total_size += residual_size * num_qp;

   offsets[SharedMemory::Index::TEMP] = total_size;
   constexpr int num_temp = 6;
   std::array<int, num_temp> temp_sizes = {0};
   // TODO-bug: this assumes q1d >= d1d
   const int q1d = max_dtq_qps;
   [[maybe_unused]] const int d1d = max_dtq_dofs;

   // TODO-bug: this depends on the dimension
   constexpr int hardcoded_temp_num = 6;
   for (std::size_t i = 0; i < hardcoded_temp_num; i++)
   {
      // TODO-bug: over-allocates if q1d <= d1d
      temp_sizes[i] = q1d * q1d * q1d;
   }
   total_size += std::accumulate(
                    std::begin(temp_sizes), std::end(temp_sizes), 0);

   return SharedMemoryInfo<num_fields, num_inputs, num_outputs>
   {
      total_size,
      offsets,
      input_dtq_sizes,
      output_dtq_sizes,
      field_sizes,
      direction_size,
      input_sizes,
      shadow_sizes,
      residual_size,
      temp_sizes
   };
}

template <typename shmem_info_t>
void print_shared_memory_info(shmem_info_t &shmem_info)
{
   out << "Shared Memory Info\n"
       << "total size: " << shmem_info.total_size
       << " " << "(" << shmem_info.total_size * real_t(sizeof(real_t))/1024.0 << "kb)";
   out << "\ninput dtq sizes (B G): ";
   for (auto &i : shmem_info.input_dtq_sizes)
   {
      out << "(";
      for (int j = 0; j < 2; j++)
      {
         out << i[j];
         if (j < 1)
         {
            out << " ";
         }
      }
      out << ") ";
   }
   out << "\noutput dtq sizes (B G): ";
   for (auto &i : shmem_info.output_dtq_sizes)
   {
      out << "(";
      for (int j = 0; j < 2; j++)
      {
         out << i[j];
         if (j < 1)
         {
            out << " ";
         }
      }
      out << ") ";
   }
   out << "\nfield sizes: ";
   for (auto &i : shmem_info.field_sizes)
   {
      out << i << " ";
   }
   out << "\ndirection size: ";
   out << shmem_info.direction_size << " ";
   out << "\ninput sizes: ";
   for (auto &i : shmem_info.input_sizes)
   {
      out << i << " ";
   }
   out << "\nshadow sizes: ";
   for (auto &i : shmem_info.shadow_sizes)
   {
      out << i << " ";
   }
   out << "\ntemp sizes: ";
   for (auto &i : shmem_info.temp_sizes)
   {
      out << i << " ";
   }
   out << "\noffsets: ";
   for (auto &i : shmem_info.offsets)
   {
      out << i << " ";
   }
   out << "\n\n";
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
std::array<DofToQuadMap, N> load_dtq_mem(
   void *mem,
   int offset,
   const std::array<std::array<int, 2>, N> &sizes,
   const std::array<DofToQuadMap, N> &dtq)
{
   std::array<DofToQuadMap, N> f;
   for (std::size_t i = 0; i < N; i++)
   {
      if (dtq[i].which_input != -1)
      {
         const auto [nqp_b, dim_b, ndof_b] = dtq[i].B.GetShape();
         const auto B = Reshape(&dtq[i].B[0], nqp_b, dim_b, ndof_b);
         auto mem_Bi = Reshape(reinterpret_cast<real_t *>(mem) + offset, nqp_b, dim_b,
                               ndof_b);

         MFEM_FOREACH_THREAD(q, x, nqp_b)
         {
            MFEM_FOREACH_THREAD(d, y, ndof_b)
            {
               for (int b = 0; b < dim_b; b++)
               {
                  auto v = B(q, b, d);
                  mem_Bi(q, b, d) = v;
               }
            }
         }

         offset += sizes[i][0];

         const auto [nqp_g, dim_g, ndof_g] = dtq[i].G.GetShape();
         const auto G = Reshape(&dtq[i].G[0], nqp_g, dim_g, ndof_g);
         auto mem_Gi = Reshape(reinterpret_cast<real_t *>(mem) + offset, nqp_g, dim_g,
                               ndof_g);

         MFEM_FOREACH_THREAD(q, x, nqp_g)
         {
            MFEM_FOREACH_THREAD(d, y, ndof_g)
            {
               for (int b = 0; b < dim_g; b++)
               {
                  mem_Gi(q, b, d) = G(q, b, d);
               }
            }
         }

         offset += sizes[i][1];

         f[i] = DofToQuadMap{DeviceTensor<3, const real_t>(&mem_Bi[0], nqp_b, dim_b, ndof_b),
                             DeviceTensor<3, const real_t>(&mem_Gi[0], nqp_g, dim_g, ndof_g),
                             dtq[i].which_input};
      }
      else
      {
         // When which_input is -1, just copy the original DofToQuadMap with empty data.
         f[i] = dtq[i];
      }
   }
   return f;
}

template <std::size_t num_fields>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1>, num_fields>
load_field_mem(
   void *mem,
   int offset,
   const std::array<int, num_fields> &sizes,
   const std::array<DeviceTensor<2>, num_fields> &fields_e,
   const int &entity_idx)
{
   std::array<DeviceTensor<1>, num_fields> f;

   for_constexpr<num_fields>([&](auto field_idx)
   {
      int block_size = MFEM_THREAD_SIZE(x) *
                       MFEM_THREAD_SIZE(y) *
                       MFEM_THREAD_SIZE(z);
      int tid = MFEM_THREAD_ID(x) +
                MFEM_THREAD_SIZE(x) *
                (MFEM_THREAD_ID(y) + MFEM_THREAD_SIZE(y) * MFEM_THREAD_ID(z));
      for (int k = tid; k < sizes[field_idx]; k += block_size)
      {
         reinterpret_cast<real_t *>(mem)[offset + k] =
            fields_e[field_idx](k, entity_idx);
      }

      f[field_idx] =
         DeviceTensor<1>(&reinterpret_cast<real_t *> (mem)[offset], sizes[field_idx]);

      offset += sizes[field_idx];
   });

   return f;
}

MFEM_HOST_DEVICE inline
DeviceTensor<1> load_direction_mem(
   void *mem,
   int offset,
   const int &size,
   const DeviceTensor<2> &direction,
   const int &entity_idx)
{
   int block_size = MFEM_THREAD_SIZE(x) *
                    MFEM_THREAD_SIZE(y) *
                    MFEM_THREAD_SIZE(z);
   int tid = MFEM_THREAD_ID(x) +
             MFEM_THREAD_SIZE(x) *
             (MFEM_THREAD_ID(y) + MFEM_THREAD_SIZE(y) * MFEM_THREAD_ID(z));
   for (int k = tid; k < size; k += block_size)
   {
      reinterpret_cast<real_t *>(mem)[offset + k] = direction(k, entity_idx);
   }
   MFEM_SYNC_THREAD;

   return DeviceTensor<1>(
             &reinterpret_cast<real_t *>(mem)[offset], size);
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<2>, N> load_input_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes,
   const int &num_qp)
{
   std::array<DeviceTensor<2>, N> f;
   for (std::size_t i = 0; i < N; i++)
   {
      f[i] = DeviceTensor<2>(&reinterpret_cast<real_t *>(mem)[offset],
                             sizes[i] / num_qp,
                             num_qp);
      offset += sizes[i];
   }
   return f;
}

MFEM_HOST_DEVICE inline
DeviceTensor<2> load_residual_mem(
   void *mem,
   int offset,
   const int &residual_size,
   const int &num_qp)
{
   return DeviceTensor<2>(reinterpret_cast<real_t *>(mem) + offset, residual_size,
                          num_qp);
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1>, 6> load_scratch_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes)
{
   std::array<DeviceTensor<1>, N> f;
   for (std::size_t i = 0; i < N; i++)
   {
      f[i] = DeviceTensor<1>(&reinterpret_cast<real_t *>(mem)[offset], sizes[i]);
      offset += sizes[i];
   }
   return f;
}

template <typename shared_mem_info_t, std::size_t num_inputs, std::size_t num_outputs, std::size_t num_fields>
MFEM_HOST_DEVICE inline
auto unpack_shmem(
   void *shmem,
   const shared_mem_info_t &shmem_info,
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
   const std::array<DeviceTensor<2>, num_fields> &wrapped_fields_e,
   const int &num_qp,
   const int &e)
{
   auto input_dtq_shmem =
      load_dtq_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
         shmem_info.input_dtq_sizes,
         input_dtq_maps);

   auto output_dtq_shmem =
      load_dtq_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
         shmem_info.output_dtq_sizes,
         output_dtq_maps);

   auto fields_shmem =
      load_field_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::FIELD],
         shmem_info.field_sizes,
         wrapped_fields_e,
         e);

   // These functions don't copy, they simply create a `DeviceTensor` object
   // that points to correct chunks of the shared memory pool.
   auto input_shmem =
      load_input_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::INPUT],
         shmem_info.input_sizes,
         num_qp);

   auto residual_shmem =
      load_residual_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::OUTPUT],
         shmem_info.residual_size,
         num_qp);

   auto scratch_mem =
      load_scratch_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::TEMP],
         shmem_info.temp_sizes);

   MFEM_SYNC_THREAD;

   // nvcc needs make_tuple to be fully qualified
   return mfem::future::make_tuple(
             input_dtq_shmem, output_dtq_shmem, fields_shmem,
             input_shmem, residual_shmem, scratch_mem);
}

template <typename shared_mem_info_t, std::size_t num_inputs, std::size_t num_outputs, std::size_t num_fields>
MFEM_HOST_DEVICE inline
auto unpack_shmem(
   void *shmem,
   const shared_mem_info_t &shmem_info,
   const std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
   const std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
   const std::array<DeviceTensor<2>, num_fields> &wrapped_fields_e,
   const DeviceTensor<2> &wrapped_direction_e,
   const int &num_qp,
   const int &e)
{
   auto input_dtq_shmem =
      load_dtq_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
         shmem_info.input_dtq_sizes,
         input_dtq_maps);

   auto output_dtq_shmem =
      load_dtq_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
         shmem_info.output_dtq_sizes,
         output_dtq_maps);

   auto fields_shmem =
      load_field_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::FIELD],
         shmem_info.field_sizes,
         wrapped_fields_e,
         e);

   auto direction_shmem =
      load_direction_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::DIRECTION],
         shmem_info.direction_size,
         wrapped_direction_e,
         e);

   // These methods don't copy, they simply create a `DeviceTensor` object
   // that points to correct chunks of the shared memory pool.
   auto input_shmem =
      load_input_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::INPUT],
         shmem_info.input_sizes,
         num_qp);

   auto shadow_shmem =
      load_input_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::SHADOW],
         shmem_info.input_sizes,
         num_qp);

   auto residual_shmem =
      load_residual_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::OUTPUT],
         shmem_info.residual_size,
         num_qp);

   auto scratch_mem =
      load_scratch_mem(
         shmem,
         shmem_info.offsets[SharedMemory::Index::TEMP],
         shmem_info.temp_sizes);

   MFEM_SYNC_THREAD;

   // nvcc needs make_tuple to be fully qualified
   return mfem::future::make_tuple(
             input_dtq_shmem, output_dtq_shmem, fields_shmem,
             direction_shmem, input_shmem, shadow_shmem,
             residual_shmem, scratch_mem);
}

template <std::size_t... i>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<2>, sizeof...(i)> get_local_input_qp(
   const std::array<DeviceTensor<3>, sizeof...(i)> &input_qp_global, int e,
   std::index_sequence<i...>)
{
   return
   {
      DeviceTensor<2>(
         &input_qp_global[i](0, 0, e),
         input_qp_global[i].GetShape()[0],
         input_qp_global[i].GetShape()[1]) ...
   };
}

template <std::size_t N>
MFEM_HOST_DEVICE inline
void set_zero(std::array<DeviceTensor<2>, N> &v)
{
   for (std::size_t i = 0; i < N; i++)
   {
      int size = v[i].GetShape()[0] * v[i].GetShape()[1];
      auto vi = Reshape(&v[i][0], size);
      for (int j = 0; j < size; j++)
      {
         vi[j] = 0.0;
      }
   }
}

template <std::size_t n>
MFEM_HOST_DEVICE inline
void set_zero(DeviceTensor<n> &u)
{
   int s = 1;
   for (int i = 0; i < n; i++)
   {
      s *= u.GetShape()[i];
   }
   auto ui = Reshape(&u[0], s);
   for (int j = 0; j < s; j++)
   {
      ui[j] = 0.0;
   }
}

/// @brief Copy data from DeviceTensor u to DeviceTensor v
///
/// @param u source DeviceTensor
/// @param v destination DeviceTensor
/// @tparam n DeviceTensor rank
template <int n>
MFEM_HOST_DEVICE inline
void copy(DeviceTensor<n> &u, DeviceTensor<n> &v)
{
   int s = 1;
   for (int i = 0; i < n; i++)
   {
      s *= u.GetShape()[i];
   }
   auto ui = Reshape(&u[0], s);
   auto vi = Reshape(&v[0], s);
   for (int j = 0; j < s; j++)
   {
      vi[j] = ui[j];
   }
}

/// @brief Copy data from array of DeviceTensor u to array of DeviceTensor v
///
/// @param u source DeviceTensor array
/// @param v destination DeviceTensor array
/// @tparam n DeviceTensor rank
/// @tparam m number of DeviceTensors
template <int n, std::size_t m>
MFEM_HOST_DEVICE inline
void copy(std::array<DeviceTensor<n>, m> &u,
          std::array<DeviceTensor<n>, m> &v)
{
   for (int i = 0; i < m; i++)
   {
      copy(u[i], v[i]);
   }
}

/// @brief Wraps plain data in DeviceTensors for fields
///
/// @param fields array of field data
/// @param field_sizes for each field, number of values stored for each entity
/// @param num_entities number of entities (elements, faces, etc) in mesh
/// @tparam num_fields number of fields
/// @return array of field data wrapped in DeviceTensors
template <std::size_t num_fields>
std::array<DeviceTensor<2>, num_fields> wrap_fields(
   std::vector<Vector> &fields,
   std::array<int, num_fields> &field_sizes,
   const int &num_entities)
{
   std::array<DeviceTensor<2>, num_fields> f;

   for_constexpr<num_fields>([&](auto i)
   {
      f[i] = DeviceTensor<2>(fields[i].ReadWrite(), field_sizes[i], num_entities);
   });

   return f;
}

/// @brief Accumulates the sizes of field operators on quadrature points for
/// dependent inputs
///
/// @tparam input_t Type of input field operators tuple
/// @tparam num_fields Number of fields
/// @tparam i Parameter pack indices for field operators
///
/// @param inputs Tuple of input field operators
/// @param kinput_is_dependent Array indicating which inputs are dependent
/// @param input_to_field Array mapping input indices to field indices
/// @param fields Array of field descriptors
/// @param seq Index sequence for inputs
///
/// @return Sum of sizes on quadrature points for all dependent inputs
///
/// @details
/// This function accumulates the sizes needed on quadrature points for all
/// dependent input field operators. For each dependent input, it calculates the
/// size required on quadrature points using GetSizeOnQP() and adds it to the
/// total. Non-dependent inputs contribute zero to the total size.
template <typename input_t, std::size_t num_fields, std::size_t... i>
int accumulate_sizes_on_qp(
   const input_t &inputs,
   std::array<bool, sizeof...(i)> &kinput_is_dependent,
   const std::array<int, sizeof...(i)> &input_to_field,
   const std::array<FieldDescriptor, num_fields> &fields,
   std::index_sequence<i...> seq)
{
   MFEM_CONTRACT_VAR(seq); // 'seq' is needed for doxygen
   return (... + [](auto &input, auto is_dependent, auto field)
   {
      if (!is_dependent)
      {
         return 0;
      }
      return GetSizeOnQP(input, field);
   }
   (get<i>(inputs),
    get<i>(kinput_is_dependent),
    fields[input_to_field[i]]));
}

template <
   typename entity_t,
   typename field_operator_ts,
   std::size_t N = tuple_size<field_operator_ts>::value,
   std::size_t... Is>
std::array<DofToQuadMap, N> create_dtq_maps_impl(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> &dtqs,
   const std::array<size_t, N> &field_map,
   std::index_sequence<Is...>)
{
   auto f = [&](auto fop, std::size_t idx)
   {
      [[maybe_unused]] auto g = [&](int idx)
      {
         auto dtq = dtqs[field_map[idx]];

         int value_dim = 1;
         int grad_dim = 1;

         if ((dtq->mode != DofToQuad::Mode::TENSOR) &&
             (!is_identity_fop<decltype(fop)>::value))
         {
            value_dim = dtq->FE->GetRangeDim() ? dtq->FE->GetRangeDim() : 1;
            grad_dim = dtq->FE->GetDim();
         }

         return std::tuple{dtq, value_dim, grad_dim};
      };

      if constexpr (is_value_fop<decltype(fop)>::value ||
                    is_gradient_fop<decltype(fop)>::value)
      {
         auto [dtq, value_dim, grad_dim] = g(idx);
         return DofToQuadMap
         {
            DeviceTensor<3, const real_t>(dtq->B.Read(), dtq->nqpt, value_dim, dtq->ndof),
            DeviceTensor<3, const real_t>(dtq->G.Read(), dtq->nqpt, grad_dim, dtq->ndof),
            static_cast<int>(idx)
         };
      }
      else if constexpr (std::is_same_v<decltype(fop), Weight>)
      {
         return DofToQuadMap
         {
            DeviceTensor<3, const real_t>(nullptr, 1, 1, 1),
            DeviceTensor<3, const real_t>(nullptr, 1, 1, 1),
            -1
         };
      }
      else if constexpr (is_identity_fop<decltype(fop)>::value ||
                         is_sum_fop<decltype(fop)>::value)
      {
         auto [dtq, value_dim, grad_dim] = g(idx);
         return DofToQuadMap
         {
            DeviceTensor<3, const real_t>(nullptr, dtq->nqpt, value_dim, dtq->ndof),
            DeviceTensor<3, const real_t>(nullptr, dtq->nqpt, grad_dim, dtq->ndof),
            -1
         };
      }
      else
      {
         static_assert(dfem::always_false<decltype(fop)>,
                       "field operator type is not implemented");
      }
      return DofToQuadMap
      {
         DeviceTensor<3, const real_t>(nullptr, 0, 0, 0),
         DeviceTensor<3, const real_t>(nullptr, 0, 0, 0),
         -1
      }; // Unreachable, but avoids compiler warning
   };
   return std::array<DofToQuadMap, N>
   {
      f(get<Is>(fops), Is)...
   };
}

/// @brief Create DofToQuad maps for a given set of field operators.
///
/// @param fops field operators
/// @param dtqmaps DofToQuad maps
/// @param to_field_map mapping from input indices to field indices
/// @tparam entity_t type of the entity
/// @return array of DofToQuad maps
template <
   typename entity_t,
   typename field_operator_ts,
   std::size_t num_fields>
std::array<DofToQuadMap, num_fields> create_dtq_maps(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> &dtqmaps,
   const std::array<size_t, num_fields> &to_field_map)
{
   return create_dtq_maps_impl<entity_t>(
             fops, dtqmaps,
             to_field_map,
             std::make_index_sequence<num_fields> {});
}

} // namespace mfem::future
#endif
