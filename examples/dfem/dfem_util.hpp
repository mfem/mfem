#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>
#include <mfem.hpp>
#include <general/forall.hpp>
#include <type_traits>
#include "dfem_fieldoperator.hpp"
#include "dfem_parametricspace.hpp"
#include "linalg/dtensor.hpp"
#include "tuple.hpp"
#include "mesh/mesh.hpp"
#include <linalg/tensor.hpp>

using std::size_t;

namespace mfem
{

template<typename tuple_t>
constexpr auto get_array_from_tuple(tuple_t&& tuple)
{
   constexpr auto get_array = [](auto&& ... x) { return std::array{std::forward<decltype(x)>(x) ... }; };
   return std::apply(get_array, std::forward<tuple_t>(tuple));
}

namespace detail
{

template <typename lambda, size_t... i>
constexpr void for_constexpr(lambda&& f,
                             std::integral_constant<size_t, i>... args)
{
   f(args...);
}


template <size_t... n, typename lambda, typename... arg_types>
constexpr void for_constexpr(lambda&& f, std::integer_sequence<size_t, n...>,
                             arg_types... args)
{
   (detail::for_constexpr(f, args..., std::integral_constant<size_t,n> {}), ...);
}

}  // namespace detail

template <typename lambda, size_t... i>
constexpr void for_constexpr(lambda&& f, std::integer_sequence<size_t, i ... >)
{
   f(std::integral_constant< size_t, i > {} ...);
}

template <int... n, typename lambda>
constexpr void for_constexpr(lambda&& f)
{
   detail::for_constexpr(f, std::make_integer_sequence<size_t, n> {}...);
}

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

// Helper function to print a single tuple
template <typename Tuple, std::size_t... Is>
void print_tuple_impl(const Tuple& t, std::index_sequence<Is...>)
{
   ((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(t)), ...);
}

template <typename... Args>
void print_tuple(const std::tuple<Args...>& t)
{
   std::cout << "(";
   print_tuple_impl(t, std::index_sequence_for<Args...> {});
   std::cout << ")";
}

// Helper function to print a tuple of tuples
template <typename Tuple, std::size_t... Is>
void print_tuple_of_tuples_impl(const Tuple& t, std::index_sequence<Is...>)
{
   ((std::cout << (Is == 0 ? "" : ", "), print_tuple(std::get<Is>(t))), ...);
}

template <typename... Args>
void print_tuple_of_tuples(const std::tuple<Args...>& t)
{
   std::cout << "(";
   print_tuple_of_tuples_impl(t, std::index_sequence_for<Args...> {});
   std::cout << ")";
}

void print_matrix(const mfem::DenseMatrix& m)
{
   std::cout << "[";
   for (int i = 0; i < m.NumRows(); i++)
   {
      for (int j = 0; j < m.NumCols(); j++)
      {
         std::cout << m(i, j);
         if (j < m.NumCols() - 1)
         {
            std::cout << " ";
         }
      }
      if (i < m.NumRows() - 1)
      {
         std::cout << "; ";
      }
   }
   std::cout << "]\n";
}

void print_vector(const mfem::Vector& v)
{
   std::cout << "[";
   for (int i = 0; i < v.Size(); i++)
   {
      std::cout << v(i);
      if (i < v.Size() - 1)
      {
         std::cout << " ";
      }
   }
   std::cout << "]\n";
}

template <typename T>
void print_array(const mfem::Array<T>& v)
{
   std::cout << "[";
   for (int i = 0; i < v.Size(); i++)
   {
      std::cout << v[i];
      if (i < v.Size() - 1)
      {
         std::cout << " ";
      }
   }
   std::cout << "]\n";
}

template <typename ... Ts>
constexpr auto decay_types(mfem::tuple<Ts...> const &)
-> mfem::tuple<std::remove_cv_t<std::remove_reference_t<Ts>>...>;

template <typename T>
using decay_tuple = decltype(decay_types(std::declval<T>()));

template <class F> struct FunctionSignature;

template <typename output_t, typename... input_ts>
struct FunctionSignature<output_t(input_ts...)>
{
   using return_t = output_t;
   using parameter_ts = mfem::tuple<input_ts...>;
};

template <class T> struct create_function_signature;

template <typename output_t, typename T, typename... input_ts>
struct create_function_signature<output_t (T::*)(input_ts...) const>
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

template <typename... Ts>
constexpr auto extract_field_ids(const std::tuple<Ts...>& t)
{
   return extract_field_ids_impl(t, std::index_sequence_for<Ts...> {});
}

// Helper function to check if an element is in the array
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

// Function to count unique Field IDs in a tuple
template <typename... Ts>
constexpr std::size_t count_unique_field_ids(const std::tuple<Ts...>& t)
{
   constexpr auto ids = extract_field_ids(decltype(t) {});
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

template <typename... Ts>
constexpr auto filter_fields(const std::tuple<Ts...>& t)
{
   return std::tuple_cat(
             std::conditional_t<Ts::GetFieldId() != -1, std::tuple<Ts>, std::tuple<>> {}...);
}

struct FieldDescriptor
{
   int id;
   std::variant<const FiniteElementSpace *,
       const ParFiniteElementSpace *,
       const ParametricSpace *> data;
};

template <class... T> constexpr bool always_false = false;

struct GeometricFactorMaps
{
   DeviceTensor<3, const double> normal;
};

namespace Entity
{
struct Element;
struct BoundaryElement;
struct Face;
struct BoundaryFace;
};

namespace EntityType
{
struct Hexahedron;
};

struct TensorProduct {};
struct NonTensorProduct {};

namespace AutoDiff
{
struct NativeDualNumber {};
struct EnzymeForward {};
}

#if (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
template <typename func_t>
__global__ void forall_kernel_shmem(func_t f, int n)
{
   int i = blockIdx.x;
   extern __shared__ double shmem[];
   if (i < n)
   {
      f(i, shmem);
   }
}
#endif

template <typename func_t>
void forall(func_t f,
            int N,
            int X,
            int Y,
            int Z,
            int num_shmem = 0,
            double *shmem = nullptr)
{
   if (Device::Allows(Backend::CUDA_MASK) ||
       Device::Allows(Backend::HIP_MASK))
   {
#if (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      // int gridsize = (N + Z - 1) / Z;
      int num_bytes = num_shmem * sizeof(decltype(shmem));
      dim3 block_size(X, Y, Z);
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

template <typename entity_t>
GeometricFactorMaps GetGeometricFactorMaps(Mesh &mesh,
                                           const IntegrationRule &ir)
{
   if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
   {
      const FaceGeometricFactors *fg =
         mesh.GetFaceGeometricFactors(
            ir,
            FaceGeometricFactors::FactorFlags::NORMALS,
            FaceType::Boundary);

      return GeometricFactorMaps
      {
         DeviceTensor<3, const double>(
            fg->normal.Read(), ir.GetNPoints(), mesh.SpaceDimension(), mesh.GetNBE()
         )
      };
   }

   Vector zero;
   return GeometricFactorMaps{DeviceTensor<3, const double>(zero.Read(), 0, 0, 0)};
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetTotalSize();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetVSize on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         MFEM_ABORT("internal error");
      }
      else
      {
         static_assert(always_false<T>, "can't use GetElementVdofs on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetTotalSize();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetTrueVSize on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetLocalSize();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetVDim on type");
      }
   }, f.data);
}

int GetVectorFEDim(const FieldDescriptor &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         if (arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_CURL ||
             arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_DIV)
         {
            return arg->GetFE(0)->GetDim();
         }
         else
         {
            return 1;
         }
      }
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return 1;
      }
      else
      {
         static_assert(always_false<T>, "can't use GetVectorFEDim on type");
      }
   }, f.data);
}

int GetVectorFECurlDim(const FieldDescriptor &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         if (arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_CURL)
         {
            return arg->GetFE(0)->GetCurlDim();
         }
         else
         {
            return 1;
         }
      }
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return 1;
      }
      else
      {
         static_assert(always_false<T>, "can't use GetVectorFECurlDim on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->Dimension();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetDimension on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetProlongation();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetProlongation on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetRestriction();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetElementRestriction on type");
      }
   }, f.data);
}

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
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return arg->GetRestriction();
      }
      else
      {
         static_assert(always_false<T>, "can't use get_face_restriction on type");
      }
   }, f.data);
}


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

void prolongation(const FieldDescriptor field, const Vector &x, Vector &field_l)
{
   const auto P = get_prolongation(field);
   field_l.SetSize(P->Height());
   P->Mult(x, field_l);
}

template <size_t N, size_t M>
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

void prolongation(const std::vector<FieldDescriptor> fields,
                  const Vector &x,
                  std::vector<Vector> &fields_l)
{
   int data_offset = 0;
   for (int i = 0; i < fields.size(); i++)
   {
      const auto P = get_prolongation(fields[i]);
      const int width = P->Width();
      const Vector x_i(const_cast<Vector&>(x), data_offset, width);
      fields_l[i].SetSize(P->Height());
      P->Mult(x_i, fields_l[i]);
      data_offset += width;
   }
}


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

// template <typename entity_t, size_t N, size_t M>
// void restriction(const std::array<FieldDescriptor, N> u,
//                  const std::array<Vector, N> &u_l,
//                  std::array<Vector, M> &fields_e,
//                  ElementDofOrdering ordering,
//                  const int offset = 0)
// {
//    for (int i = 0; i < N; i++)
//    {
//       const auto R = get_restriction<entity_t>(u[i], ordering);
//       MFEM_ASSERT(R->Width() == u_l[i].Size(),
//                   "restriction not applicable to given data size");
//       const int height = R->Height();
//       fields_e[i + offset].SetSize(height);
//       R->Mult(u_l[i], fields_e[i + offset]);
//    }
// }

template <typename entity_t>
void restriction(const std::vector<FieldDescriptor> u,
                 const std::vector<Vector> &u_l,
                 std::vector<Vector> &fields_e,
                 ElementDofOrdering ordering,
                 const int offset = 0)
{
   for (int i = 0; i < u.size(); i++)
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
template <size_t N, size_t M>
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
      static_assert(always_false<entity_t>, "can't use GetNumEntites on type");
   }
}

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
            return &arg->GetFE(0)->GetDofToQuad(ir, mode);
         }
         else if constexpr (std::is_same_v<entity_t, Entity::BoundaryElement>)
         {
            return &arg->GetBE(0)->GetDofToQuad(ir, mode);
         }
      }
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         return &arg->GetDofToQuad();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetDofToQuad on type");
      }
   }, f.data);
}

template <typename field_operator_t>
void CheckCompatibility(const FieldDescriptor &f)
{
   std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         // if constexpr (std::is_same_v<field_operator_t, One>)
         // {
         //    // Supported by all FE spaces
         // }
         if constexpr (std::is_same_v<field_operator_t, Value<>>)
         {
            // Supported by all FE spaces
         }
         else if constexpr (std::is_same_v<field_operator_t, Gradient<>>)
         {
            MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::VALUE,
                        "Gradient not compatible with FE");
         }
         // else if constexpr (std::is_same_v<field_operator_t, Curl>)
         // {
         //    MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_CURL,
         //                "Curl not compatible with FE");
         // }
         // else if constexpr (std::is_same_v<field_operator_t, Div>)
         // {
         //    MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_DIV,
         //                "Div not compatible with FE");
         // }
         // else if constexpr (std::is_same_v<field_operator_t, FaceValueLeft> ||
         //                    std::is_same_v<field_operator_t, FaceValueRight>)
         // {
         //    MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::VALUE,
         //                "FaceValueLeft/FaceValueRight not compatible with FE");
         // }
         else
         {
            static_assert(always_false<T, field_operator_t>,
                          "FieldOperator not compatible with FiniteElementSpace");
         }
      }
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         if constexpr (std::is_same_v<field_operator_t, None<>>)
         {
            // Only supported field operation for ParametricSpace
         }
         else
         {
            static_assert(always_false<T, field_operator_t>,
                          "FieldOperator not compatible with ParametricSpace");
         }
      }
      else
      {
         static_assert(always_false<T, field_operator_t>,
                       "Operator not compatible with FE");
      }
   }, f.data);
}

template <typename entity_t, typename field_operator_t>
int GetSizeOnQP(const field_operator_t &, const FieldDescriptor &f)
{
   // CheckCompatibility<field_operator_t>(f);

   if constexpr (is_value_fop<field_operator_t>::value)
   {
      return GetVDim(f) * GetVectorFEDim(f);
   }
   else if constexpr (is_gradient_fop<field_operator_t>::value)
   {
      return GetVDim(f) * GetDimension<entity_t>(f);
   }
   // else if constexpr (std::is_same_v<field_operator_t, Curl>)
   // {
   //    return GetVDim(f) * GetVectorFECurlDim(f);
   // }
   // else if constexpr (std::is_same_v<field_operator_t, Div>)
   // {
   //    return GetVDim(f);
   // }
   // else if constexpr (std::is_same_v<field_operator_t, None>)
   // {
   //    return GetVDim(f);
   // }
   // else if constexpr (std::is_same_v<field_operator_t, One>)
   // {
   //    return 1;
   // }
   else
   {
      MFEM_ABORT("can't get size on quadrature point for field descriptor");
   }
}

// template <size_t num_fields>
// typename std::array<FieldDescriptor, num_fields>::const_iterator find_name(
//    const std::array<FieldDescriptor, num_fields> &fields,
//    const std::string &input_name)
// {
//    auto it = std::find_if(fields.begin(),
//                           fields.end(), [&](const FieldDescriptor &field)
//    {
//       return field.field_label == input_name;
//    });

//    return it;
// }

// template <size_t num_fields>
// int find_name_idx(const std::array<FieldDescriptor, num_fields> &fields,
//                   const std::string &input_name)
// {
//    typename std::array<FieldDescriptor, num_fields>::const_iterator it
//       = find_name(fields, input_name);
//    if (it == fields.end())
//    {
//       return -1;
//    }
//    return (it - fields.begin());
// }

// typename std::vector<FieldDescriptor>::const_iterator find_name(
//    const std::vector<FieldDescriptor> &fields,
//    const std::string &input_name)
// {
//    auto it = std::find_if(fields.begin(),
//                           fields.end(), [&](const FieldDescriptor &field)
//    {
//       return field.field_label == input_name;
//    });

//    return it;
// }

// int find_name_idx(const std::vector<FieldDescriptor> &fields,
//                   const std::string &input_name)
// {
//    typename std::vector<FieldDescriptor>::const_iterator it
//       = find_name(fields, input_name);
//    if (it == fields.end())
//    {
//       return -1;
//    }
//    return (it - fields.begin());
// }

template <typename entity_t, typename field_operator_ts, std::size_t... idx>
std::array<int, mfem::tuple_size<field_operator_ts>::value>
create_descriptors_to_fields_map(
   const std::vector<FieldDescriptor> &fields,
   field_operator_ts &fops,
   std::index_sequence<idx...>)
{
   std::array<int, mfem::tuple_size<field_operator_ts>::value> map;

   auto find_id = [](const std::vector<FieldDescriptor> &fields, int i)
   {
      auto it = std::find_if(begin(fields), end(fields),
                             [&](const FieldDescriptor &field)
      {
         return field.id == i;
      });

      if (it == fields.end())
      {
         return -1;
      }
      return static_cast<int>(it - fields.begin());
   };

   auto f = [&](auto &fop, auto &map)
   {
      int i;

      if constexpr (std::is_same_v<decltype(fop), Weight&>)
      {
         // TODO: stealing dimension from the first field
         fop.dim = GetDimension<Entity::Element>(fields[0]);
         fop.vdim = 1;
         fop.size_on_qp = 1;
         map = -1;
      }
      // else if constexpr (std::is_same_v<decltype(fop), FaceNormal&>)
      // {
      //    fop.dim = GetDimension<Entity::Element>(fields[i]);
      //    fop.vdim = 1;
      //    fop.size_on_qp = fop.dim;
      //    map = -1;
      // }
      // else if ((i = find_name_idx(fields, fop.field_label)) != -1)
      // {
      //    fop.dim = GetDimension<entity_t>(fields[i]);
      //    fop.vdim = GetVDim(fields[i]);
      //    fop.size_on_qp = GetSizeOnQP<entity_t>(fop, fields[i]);
      //    map = i;
      // }
      else if ((i = find_id(fields, fop.GetFieldId())) != -1)
      {
         fop.dim = GetDimension<entity_t>(fields[i]);
         fop.vdim = GetVDim(fields[i]);
         fop.size_on_qp = GetSizeOnQP<entity_t>(fop, fields[i]);
         map = i;
      }
      else
      {
         MFEM_ABORT("can't find field for label: " << fop.GetFieldId());
      }
   };

   (f(mfem::get<idx>(fops), map[idx]), ...);

   return map;
}

template <typename input_t, std::size_t... i>
std::array<DeviceTensor<3>, sizeof...(i)> wrap_input_memory(
   std::array<Vector, sizeof...(i)> &input_qp_mem, int num_qp, int num_entities,
   const input_t &inputs, std::index_sequence<i...>)
{
   return {DeviceTensor<3>(input_qp_mem[i].Write(), mfem::get<i>(inputs).size_on_qp, num_qp, num_entities) ...};
}

template <typename input_t, std::size_t... i>
std::array<Vector, sizeof...(i)> create_input_qp_memory(
   int num_qp,
   int num_entities,
   input_t &inputs,
   std::index_sequence<i...>)
{
   return {Vector(mfem::get<i>(inputs).size_on_qp * num_qp * num_entities)...};
}

struct DofToQuadMap
{
   enum Index
   {
      QP,
      DIM,
      DOF
   };
   DeviceTensor<3, const double> B;
   DeviceTensor<3, const double> G;
   int which_input = -1;
};

template <typename input_t, std::size_t... i>
std::vector<int> get_input_size_on_qp(
   const input_t &inputs,
   std::index_sequence<i...>)
{
   return {mfem::get<i>(inputs).size_on_qp...};
}

namespace SharedMemory
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
}

template <size_t num_fields, size_t num_inputs, size_t num_outputs>
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

template <typename entity_t, size_t num_fields, size_t num_inputs, size_t num_outputs, typename input_t>
SharedMemoryInfo<num_fields, num_inputs, num_outputs>
get_shmem_info(
   std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
   std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
   const std::vector<FieldDescriptor> &fields,
   const int &num_entities,
   const input_t &inputs,
   const int &num_qp,
   const std::vector<int> &input_size_on_qp,
   const int &residual_size_on_qp,
   const int &derivative_idx = -1)
{
   std::array<int, 8> offsets = {0};
   int total_size = 0;

   offsets[SharedMemory::Index::INPUT_DTQ] = total_size;
   std::array<std::array<int, 2>, num_inputs> input_dtq_sizes;
   int max_dtq_qps = 0;
   int max_dtq_dofs = 0;
   for (int i = 0; i < num_inputs; i++)
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
   for (int i = 0; i < num_outputs; i++)
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
   for (int i = 0; i < num_fields; i++)
   {
      field_sizes[i] = get_restriction<entity_t>(
                          fields[i],
                          ElementDofOrdering::LEXICOGRAPHIC)->Height() / num_entities;
   }
   total_size += std::accumulate(
                    std::begin(field_sizes), std::end(field_sizes), 0);

   offsets[SharedMemory::Index::DIRECTION] = total_size;
   int direction_size = 0;
   if (derivative_idx != -1)
   {
      direction_size = get_restriction<entity_t>(
                          fields[derivative_idx],
                          ElementDofOrdering::LEXICOGRAPHIC)->Height() / num_entities;
      total_size += direction_size;
   }

   offsets[SharedMemory::Index::INPUT] = total_size;
   std::array<int, num_inputs> input_sizes;
   for (int i = 0; i < num_inputs; i++)
   {
      input_sizes[i] = input_size_on_qp[i] * num_qp;
   }
   total_size += std::accumulate(
                    std::begin(input_sizes), std::end(input_sizes), 0);

   offsets[SharedMemory::Index::SHADOW] = total_size;
   std::array<int, num_inputs> shadow_sizes{0};
   if (derivative_idx != -1)
   {
      for (int i = 0; i < num_inputs; i++)
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
   const int d1d = max_dtq_dofs;

   // TODO-bug: this depends on the dimension
   constexpr int hardcoded_temp_num = 6;
   for (int i = 0; i < hardcoded_temp_num; i++)
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
       << " " << "(" << shmem_info.total_size * double(sizeof(double))/1024.0 << "kb)";
   out << "\ninput dtq sizes: ";
   for (auto &i : shmem_info.input_dtq_sizes)
   {
      for (auto &j : i)
      {
         out << j << " ";
      }
   }
   out << "\noutput dtq sizes: ";
   for (auto &i : shmem_info.output_dtq_sizes)
   {
      for (auto &j : i)
      {
         out << j << " ";
      }
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

template <size_t N>
MFEM_HOST_DEVICE inline
std::array<DofToQuadMap, N> load_dtq_mem(
   void *mem,
   int offset,
   const std::array<std::array<int, 2>, N> &sizes,
   const std::array<DofToQuadMap, N> &dtq)
{
   std::array<DofToQuadMap, N> f;
   for (int i = 0; i < N; i++)
   {
      const auto [nqp_b, dim_b, ndof_b] = dtq[i].B.GetShape();

      const auto B = Reshape(&dtq[i].B[0], nqp_b, dim_b, ndof_b);
      auto mem_Bi = Reshape(reinterpret_cast<real_t *>(mem) + offset, nqp_b, dim_b,
                            ndof_b);
      if (dtq[i].which_input != -1)
      {
         MFEM_FOREACH_THREAD(q, x, nqp_b)
         {
            MFEM_FOREACH_THREAD(d, y, ndof_b)
            {
               mem_Bi(q, 0, d) = B(q, 0, d);
            }
         }
      }
      offset += sizes[i][0];

      const auto [nqp_g, dim_g, ndof_g] = dtq[i].G.GetShape();
      const auto G = Reshape(&dtq[i].G[0], nqp_g, dim_g, ndof_g);
      auto mem_Gi = Reshape(reinterpret_cast<real_t *>(mem) + offset, nqp_g, dim_g,
                            ndof_g);
      if (dtq[i].which_input != -1)
      {
         MFEM_FOREACH_THREAD(q, x, nqp_g)
         {
            MFEM_FOREACH_THREAD(d, y, ndof_g)
            {
               mem_Gi(q, 0, d) = G(q, 0, d);
            }
         }
      }
      offset += sizes[i][1];

      f[i] = DofToQuadMap{DeviceTensor<3, const double>(&mem_Bi[0], nqp_b, dim_b, ndof_b),
                          DeviceTensor<3, const double>(&mem_Gi[0], nqp_g, dim_g, ndof_g),
                          dtq[i].which_input};
   }
   return f;
}

template <size_t num_fields>
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
      // const auto fop = mfem::get<input_idx>(fops);

      // if constexpr (is_weight_fop<std::decay_t<decltype(fop)>>::value)
      // {
      //    f[input_idx] = DeviceTensor<1>(nullptr, 0);
      // }
      // else if constexpr (is_none_fop<std::decay_t<decltype(fop)>>::value)
      // {
      //    f[input_idx] =
      //       DeviceTensor<1>(reinterpret_cast<real_t *>(&fields_e[field_idx](0, entity_idx)),
      //                       sizes[field_idx]);
      //    offset += sizes[field_idx];
      // }
      // else
      // {
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
      // }
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

template <size_t N>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<2>, N> load_input_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes,
   const int &num_qp)
{
   std::array<DeviceTensor<2>, N> f;
   for (int i = 0; i < N; i++)
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
   return DeviceTensor<2>(reinterpret_cast<real_t *>(mem) + offset, residual_size, num_qp);
}

template <size_t N>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1>, 6> load_scratch_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes)
{
   std::array<DeviceTensor<1>, N> f;
   for (int i = 0; i < N; i++)
   {
      f[i] = DeviceTensor<1>(&reinterpret_cast<real_t *>(mem)[offset], sizes[i]);
      offset += sizes[i];
   }
   return f;
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

template <size_t N>
MFEM_HOST_DEVICE inline
void zero_all(std::array<DeviceTensor<2>, N> &v)
{
   for (size_t i = 0; i < N; i++)
   {
      int size = v[i].GetShape()[0] * v[i].GetShape()[1];
      auto vi = Reshape(&v[i][0], size);
      for (int j = 0; j < size; j++)
      {
         vi[j] = 0.0;
      }
   }
}

template<size_t num_fields, size_t... i>
std::array<DeviceTensor<2>, num_fields> wrap_fields_impl(
   std::vector<Vector> &fields,
   std::array<int, num_fields> &field_sizes,
   int num_entities,
   std::index_sequence<i...>)
{
   return std::array<DeviceTensor<2>, num_fields>
   {
      {
         DeviceTensor<2>(
            fields[i].ReadWrite(),
            field_sizes[i],
            num_entities)...
      }
   };
}

template <size_t num_fields>
std::array<DeviceTensor<2>, num_fields> wrap_fields(
   std::vector<Vector> &fields,
   std::array<int, num_fields> &field_sizes,
   const int &num_entities)
{
   return wrap_fields_impl(fields, field_sizes, num_entities,
                           std::make_index_sequence<num_fields> {});
}

template <typename input_t, size_t num_fields, std::size_t... i>
int accumulate_sizes_on_qp(
   const input_t &inputs,
   std::array<bool, sizeof...(i)> &kinput_is_dependent,
   const std::array<int, sizeof...(i)> &kinput_to_field,
   const std::array<FieldDescriptor, num_fields> &fields,
   std::index_sequence<i...>)
{
   return (... + [](auto &input, auto is_dependent, auto field)
   {
      if (!is_dependent)
      {
         return 0;
      }
      return GetSizeOnQP(input, field);
   }
   (mfem::get<i>(inputs),
    mfem::get<i>(kinput_is_dependent),
    fields[kinput_to_field[i]]));
}

// template <typename entity_t, typename field_operator_ts, size_t N, std::size_t... i>
// std::array<DofToQuadMap, N> create_dtq_maps_impl(
//    field_operator_ts &fops,
//    std::vector<const DofToQuad*> dtqs,
//    const std::array<int, N> &field_map,
//    std::index_sequence<i...>)
// {
//    auto f = [&](auto fop, size_t idx)
//    {
//       auto g = [&](int idx)
//       {
//          auto dtq = dtqs[field_map[idx]];

//          int value_dim = 1;
//          int grad_dim = 1;

//          if ((dtq->mode != DofToQuad::Mode::TENSOR) &&
//              (!std::is_same_v<decltype(fop), None>))
//          {
//             value_dim = dtq->FE->GetRangeDim() ?
//                         dtq->FE->GetRangeDim() :
//                         fop.vdim;

//             grad_dim = dtq->FE->GetDim();
//          }

//          return std::tuple{dtq, value_dim, grad_dim};
//       };

//       if constexpr (std::is_same_v<decltype(fop), Value> ||
//                     std::is_same_v<decltype(fop), Gradient>)
//       {
//          auto [dtq, value_dim, grad_dim] = g(idx);
//          return DofToQuadMap
//          {
//             DeviceTensor<3, const double>(dtq->B.Read(), dtq->nqpt, value_dim, dtq->ndof),
//             DeviceTensor<3, const double>(dtq->G.Read(), dtq->nqpt, grad_dim, dtq->ndof),
//             static_cast<int>(idx)
//          };
//       }
//       else if constexpr (std::is_same_v<decltype(fop), Weight>)
//       {
//          // no op
//          // this is handled at runtime by the first condition
//          // to_field_map[idx] == -1.
//          // has to exist at compile time for completeness
//          return DofToQuadMap
//          {
//             DeviceTensor<3, const double>(nullptr, 1, 1, 1),
//             DeviceTensor<3, const double>(nullptr, 1, 1, 1),
//             -1
//          };
//       }
//       else if constexpr (std::is_same_v<decltype(fop), None>)
//       {
//          auto [dtq, value_dim, grad_dim] = g(idx);
//          return DofToQuadMap
//          {
//             DeviceTensor<3, const double>(nullptr, dtq->nqpt, value_dim, dtq->ndof),
//             DeviceTensor<3, const double>(nullptr, dtq->nqpt, grad_dim, dtq->ndof),
//             -1
//          };
//       }
//       else
//       {
//          static_assert(always_false<decltype(fop)>,
//                        "field operator type is not implemented");
//       }
//    };
//    return std::array<DofToQuadMap, N>
//    {
//       f(mfem::get<i>(fops), i)...
//    };
// }

// template <typename entity_t, typename field_operator_ts, size_t N>
// std::array<DofToQuadMap, N> create_dtq_maps(
//    field_operator_ts &fops,
//    std::vector<const DofToQuad*> dtqmaps,
//    const std::array<int, N> &to_field_map)
// {
//    return create_dtq_maps_impl<entity_t>(
//              fops, dtqmaps,
//              to_field_map,
//              std::make_index_sequence<mfem::tuple_size<field_operator_ts>::value> {});
// }

template <
   typename entity_t,
   typename field_operator_ts,
   size_t N = mfem::tuple_size<field_operator_ts>::value,
   size_t... Is>
std::array<DofToQuadMap, N> create_dtq_maps_impl(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqs,
   const std::array<int, N> &field_map,
   std::index_sequence<Is...>)
{
   auto f = [&](auto fop, size_t idx)
   {
      auto g = [&](int idx)
      {
         auto dtq = dtqs[field_map[idx]];

         int value_dim = 1;
         int grad_dim = 1;

         if ((dtq->mode != DofToQuad::Mode::TENSOR) &&
             (!is_none_fop<decltype(fop)>::value))
         {
            value_dim = dtq->FE->GetRangeDim() ?
                        dtq->FE->GetRangeDim() :
                        fop.vdim;

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
            DeviceTensor<3, const double>(dtq->B.Read(), dtq->nqpt, value_dim, dtq->ndof),
            DeviceTensor<3, const double>(dtq->G.Read(), dtq->nqpt, grad_dim, dtq->ndof),
            static_cast<int>(idx)
         };
      }
      else if constexpr (std::is_same_v<decltype(fop), Weight>)
      {
         // no op
         // this is handled at runtime by the first condition
         // to_field_map[idx] == -1.
         // has to exist at compile time for completeness
         return DofToQuadMap
         {
            DeviceTensor<3, const double>(nullptr, 1, 1, 1),
            DeviceTensor<3, const double>(nullptr, 1, 1, 1),
            -1
         };
      }
      else if constexpr (is_none_fop<decltype(fop)>::value)
      {
         auto [dtq, value_dim, grad_dim] = g(idx);
         return DofToQuadMap
         {
            DeviceTensor<3, const double>(nullptr, dtq->nqpt, value_dim, dtq->ndof),
            DeviceTensor<3, const double>(nullptr, dtq->nqpt, grad_dim, dtq->ndof),
            -1
         };
      }
      else
      {
         static_assert(always_false<decltype(fop)>,
                       "field operator type is not implemented");
      }
   };
   return std::array<DofToQuadMap, N>
   {
      f(mfem::get<Is>(fops), Is)...
   };
}


template <
   typename entity_t,
   typename field_operator_ts,
   size_t N = mfem::tuple_size<field_operator_ts>::value>
std::array<DofToQuadMap, N> create_dtq_maps(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map)
{
   return create_dtq_maps_impl<entity_t>(
             fops, dtqmaps,
             to_field_map,
             std::make_index_sequence<N> {});
}

} // namespace mfem
