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
#include "tuple.hpp"
#include "mesh/mesh.hpp"
#include <linalg/tensor.hpp>
#include <enzyme/utils>
#include <enzyme/enzyme>

using std::size_t;

namespace mfem
{

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

struct FieldDescriptor
{
   using variant_t = std::variant<const FiniteElementSpace *,
         const ParFiniteElementSpace *,
         const ParametricSpace *>;

   variant_t data;

   std::string field_label;
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

struct TensorProduct;
struct NonTensorProduct;

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

template <typename func_t, typename input_t,
          typename output_t>
struct ElementOperator;

template <typename func_t, typename... input_ts,
          typename... output_ts>
struct ElementOperator<func_t, mfem::tuple<input_ts...>,
          mfem::tuple<output_ts...>>
{
   using entity_t = Entity::Element;
   func_t func;
   mfem::tuple<input_ts...> inputs;
   mfem::tuple<output_ts...> outputs;
   using kf_param_ts = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::parameter_ts;

   using kf_output_t = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::return_t;

   using kernel_inputs_t = decltype(inputs);
   using kernel_outputs_t = decltype(outputs);

   static constexpr size_t num_kinputs =
      mfem::tuple_size<kernel_inputs_t>::value;
   static constexpr size_t num_koutputs =
      mfem::tuple_size<kernel_outputs_t>::value;

   Array<int> attributes;

   ElementOperator(func_t func,
                   mfem::tuple<input_ts...> inputs,
                   mfem::tuple<output_ts...> outputs,
                   Array<int> *attr = nullptr)
      : func(func), inputs(inputs), outputs(outputs)
   {
      if (attr)
      {
         attributes = *attr;
      }
      // Properly check all parameter types of the kernel
      // std::apply([](auto&&... args)
      // {
      //    ((out << std::is_reference_v<decltype(args)> << "\n"), ...);
      // },
      // kf_param_ts);

      // Consistency checks
      if constexpr (num_koutputs > 1)
      {
         static_assert(always_false<func_t>,
                       "more than one output per kernel is not supported right now");
      }

      constexpr size_t num_kfinputs = mfem::tuple_size<kf_param_ts>::value;
      static_assert(num_kfinputs == num_kinputs,
                    "kernel function inputs and descriptor inputs have to match");

      constexpr size_t num_kfoutputs = mfem::tuple_size<kf_output_t>::value;
      static_assert(num_kfoutputs == num_koutputs,
                    "kernel function outputs and descriptor outputs have to match");
   }
};

template <typename func_t, typename... input_ts,
          typename... output_ts>
ElementOperator(func_t, mfem::tuple<input_ts...>,
                mfem::tuple<output_ts...>)
-> ElementOperator<func_t, mfem::tuple<input_ts...>,
mfem::tuple<output_ts...>>;

template <typename func_t, typename input_t, typename output_t>
struct BoundaryElementOperator : public
   ElementOperator<func_t, input_t, output_t>
{
public:
   using entity_t = Entity::BoundaryElement;
   BoundaryElementOperator(func_t func, input_t inputs, output_t outputs)
      : ElementOperator<func_t, input_t, output_t>(func, inputs, outputs) {}
};

template <typename func_t, typename input_t, typename output_t>
struct FaceOperator : public
   ElementOperator<func_t, input_t, output_t>
{
public:
   using entity_t = Entity::Face;
   FaceOperator(func_t func, input_t inputs, output_t outputs)
      : ElementOperator<func_t, input_t, output_t>(func, inputs, outputs) {}
};

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

template <typename entity_t, size_t N, size_t M>
void restriction(const std::array<FieldDescriptor, N> u,
                 const std::array<Vector, N> &u_l,
                 std::array<Vector, M> &fields_e,
                 ElementDofOrdering ordering,
                 const int offset = 0)
{
   for (int i = 0; i < N; i++)
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
int GetNumEntities(mfem::Mesh &mesh)
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
         if constexpr (std::is_same_v<field_operator_t, One>)
         {
            // Supported by all FE spaces
         }
         else if constexpr (std::is_same_v<field_operator_t, Value>)
         {
            // Supported by all FE spaces
         }
         else if constexpr (std::is_same_v<field_operator_t, Gradient>)
         {
            MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::VALUE,
                        "Gradient not compatible with FE");
         }
         else if constexpr (std::is_same_v<field_operator_t, Curl>)
         {
            MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_CURL,
                        "Curl not compatible with FE");
         }
         else if constexpr (std::is_same_v<field_operator_t, Div>)
         {
            MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::H_DIV,
                        "Div not compatible with FE");
         }
         else if constexpr (std::is_same_v<field_operator_t, FaceValueLeft> ||
                            std::is_same_v<field_operator_t, FaceValueRight>)
         {
            MFEM_ASSERT(arg->GetFE(0)->GetMapType() == FiniteElement::MapType::VALUE,
                        "FaceValueLeft/FaceValueRight not compatible with FE");
         }
         else
         {
            static_assert(always_false<T, field_operator_t>,
                          "FieldOperator not compatible with FiniteElementSpace");
         }
      }
      else if constexpr (std::is_same_v<T, const ParametricSpace *>)
      {
         if constexpr (std::is_same_v<field_operator_t, None>)
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

   if constexpr (std::is_same_v<field_operator_t, Value>)
   {
      return GetVDim(f) * GetVectorFEDim(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, Gradient>)
   {
      return GetVDim(f) * GetDimension<entity_t>(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, Curl>)
   {
      return GetVDim(f) * GetVectorFECurlDim(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, Div>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, None>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, One>)
   {
      return 1;
   }
   else
   {
      MFEM_ABORT("can't get size on quadrature point for field descriptor");
   }
}

template <size_t num_fields>
typename std::array<FieldDescriptor, num_fields>::const_iterator find_name(
   const std::array<FieldDescriptor, num_fields> &fields,
   const std::string &input_name)
{
   auto it = std::find_if(fields.begin(),
                          fields.end(), [&](const FieldDescriptor &field)
   {
      return field.field_label == input_name;
   });

   return it;
}

template <size_t num_fields>
int find_name_idx(const std::array<FieldDescriptor, num_fields> &fields,
                  const std::string &input_name)
{
   typename std::array<FieldDescriptor, num_fields>::const_iterator it
      = find_name(fields, input_name);
   if (it == fields.end())
   {
      return -1;
   }
   return (it - fields.begin());
}

template <typename entity_t, size_t num_fields, typename field_operator_ts, std::size_t... idx>
std::array<int, mfem::tuple_size<field_operator_ts>::value>
create_descriptors_to_fields_map(
   std::array<FieldDescriptor, num_fields> &fields,
   field_operator_ts &fops,
   std::index_sequence<idx...>)
{
   std::array<int, mfem::tuple_size<field_operator_ts>::value> map;

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
      else if constexpr (std::is_same_v<decltype(fop), FaceNormal&>)
      {
         fop.dim = GetDimension<Entity::Element>(fields[i]);
         fop.vdim = 1;
         fop.size_on_qp = fop.dim;
         map = -1;
      }
      else if ((i = find_name_idx(fields, fop.field_label)) != -1)
      {
         fop.dim = GetDimension<entity_t>(fields[i]);
         fop.vdim = GetVDim(fields[i]);
         fop.size_on_qp = GetSizeOnQP<entity_t>(fop, fields[i]);
         map = i;
      }
      else
      {
         MFEM_ABORT("can't find field for label: " << fop.field_label);
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

template <typename field_operator_t>
MFEM_HOST_DEVICE inline
void map_field_to_quadrature_data_tensor_product(
   DeviceTensor<2> &field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1, const double> &field_e,
   field_operator_t &input,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   auto B = dtq.B;
   auto G = dtq.G;

   if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::Value>)
   {
      auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e[0], d1d, d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, q1d, q1d, q1d);
      auto s0 = Reshape(&scratch_mem[0](0), d1d, d1d, q1d);
      auto s1 = Reshape(&scratch_mem[1](0), d1d, q1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  double acc = 0.0;
                  for (int dx = 0; dx < d1d; dx++)
                  {
                     acc += B(qx, 0, dx) * field(dx, dy, dz, vd);
                  }
                  s0(dz, dy, qx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(qx, x, q1d)
            {
               MFEM_FOREACH_THREAD(qy, y, q1d)
               {
                  double acc = 0.0;
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     acc += s0(dz, dy, qx) * B(qy, 0, dy);
                  }
                  s1(dz, qy, qx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  double acc = 0.0;
                  for (int dz = 0; dz < d1d; dz++)
                  {
                     acc += s1(dz, qy, qx) * B(qz, 0, dz);
                  }
                  fqp(vd, qx, qy, qz) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else if constexpr (
      std::is_same_v<field_operator_t, BareFieldOperator::Gradient>)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = input.vdim;
      const int dim = input.dim;
      const auto field = Reshape(&field_e[0], d1d, d1d, d1d, vdim);
      auto fqp = Reshape(&field_qp[0], vdim, dim, q1d, q1d, q1d);

      auto s0 = Reshape(&scratch_mem[0](0), d1d, d1d, q1d);
      auto s1 = Reshape(&scratch_mem[1](0), d1d, d1d, q1d);
      auto s2 = Reshape(&scratch_mem[2](0), d1d, q1d, q1d);
      auto s3 = Reshape(&scratch_mem[3](0), d1d, q1d, q1d);
      auto s4 = Reshape(&scratch_mem[4](0), d1d, q1d, q1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uv[2] = {0.0, 0.0};
                  for (int dx = 0; dx < d1d; dx++)
                  {
                     const real_t f = field(dx, dy, dz, vd);
                     uv[0] += f * B(qx, 0, dx);
                     uv[1] += f * G(qx, 0, dx);
                  }
                  s0(dz, dy, qx) = uv[0];
                  s1(dz, dy, qx) = uv[1];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int dy = 0; dy < d1d; dy++)
                  {
                     const real_t s0i = s0(dz, dy, qx);
                     uvw[0] += s1(dz, dy, qx) * B(qy, 0, dy);
                     uvw[1] += s0i * G(qy, 0, dy);
                     uvw[2] += s0i * B(qy, 0, dy);
                  }
                  s2(dz, qy, qx) = uvw[0];
                  s3(dz, qy, qx) = uvw[1];
                  s4(dz, qy, qx) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qx, x, q1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int dz = 0; dz < d1d; dz++)
                  {
                     uvw[0] += s2(dz, qy, qx) * B(qz, 0, dz);
                     uvw[1] += s3(dz, qy, qx) * B(qz, 0, dz);
                     uvw[2] += s4(dz, qy, qx) * G(qz, 0, dz);
                  }
                  fqp(vd, 0, qx, qy, qz) = uvw[0];
                  fqp(vd, 1, qx, qy, qz) = uvw[1];
                  fqp(vd, 2, qx, qy, qz) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      // TODO: eeek
      const int q1d = (int)floor(pow(num_qp, 1.0/input.dim) + 0.5);
      auto w = Reshape(&integration_weights[0], q1d, q1d, q1d);
      auto f = Reshape(&field_qp[0], q1d, q1d, q1d);
      MFEM_FOREACH_THREAD(qx, x, q1d)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(qz, z, q1d)
            {
               f(qx, qy, qz) = w(qx, qy, qz);
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   else if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::None>)
   {
      const int q1d = B.GetShape()[0];
      auto field = Reshape(&field_e[0], input.size_on_qp, q1d, q1d, q1d);
      auto fqp = Reshape(&field_qp[0], input.size_on_qp, q1d, q1d, q1d);

      for (int sq = 0; sq < input.size_on_qp; sq++)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  fqp(sq, qx, qy, qz) = field(sq, qx, qy, qz);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else
   {
      static_assert(always_false<field_operator_t>,
                    "can't map field to quadrature data");
   }
}


template <typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   const DofToQuadMap &dtq,
   const DeviceTensor<1, const double> &field_e,
   field_operator_t &input,
   DeviceTensor<1, const double> integration_weights)
{
   auto B = dtq.B;
   auto G = dtq.G;
   if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::Value>)
   {
      auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e(0), num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            double acc = 0.0;
            for (int dof = 0; dof < num_dof; dof++)
            {
               acc += B(qp, 0, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (
      std::is_same_v<field_operator_t, BareFieldOperator::Gradient>)
   {
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = input.vdim;
      const auto field = Reshape(&field_e(0), num_dof, vdim);

      auto f = Reshape(&field_qp[0], vdim, dim, num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int d = 0; d < dim; d++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += G(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
            }
         }
      }
   }
   // else if constexpr (std::is_same_v<field_operator_t, FaceNormal>)
   // {
   //    auto normal = geometric_factors.normal;
   //    auto [num_qp, dim, num_entities] = normal.GetShape();
   //    auto f = Reshape(&field_qp[0], dim, num_qp);
   //    for (int qp = 0; qp < num_qp; qp++)
   //    {
   //       for (int d = 0; d < dim; d++)
   //       {
   //          f(d, qp) = normal(qp, d, entity_idx);
   //       }
   //    }
   // }
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, BareFieldOperator::None>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const auto field = Reshape(&field_e[0], size_on_qp * num_qp);
      auto f = Reshape(&field_qp[0], size_on_qp * num_qp);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         f(i) = field(i);
      }
   }
   else
   {
      static_assert(always_false<field_operator_t>,
                    "can't map field to quadrature data");

   }
}

template <typename T = NonTensorProduct, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
MFEM_HOST_DEVICE inline
void map_fields_to_quadrature_data(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   const std::array<DeviceTensor<1, const double>, num_kinputs> &fields_e,
   const std::array<DofToQuadMap, num_kinputs> &dtqmaps,
   field_operator_tuple_t fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   std::index_sequence<i...>)
{
   if constexpr (std::is_same_v<T, TensorProduct>)
   {

      (map_field_to_quadrature_data_tensor_product(fields_qp[i],
                                                   dtqmaps[i], fields_e[i],
                                                   mfem::get<i>(fops), integration_weights,
                                                   scratch_mem),
       ...);
   }
   else
   {
      (map_field_to_quadrature_data(fields_qp[i],
                                    dtqmaps[i], fields_e[i],
                                    mfem::get<i>(fops), integration_weights),
       ...);
   }
}

template <typename T, typename field_operator_t>
MFEM_HOST_DEVICE
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> &field_qp,
   const DeviceTensor<1, const double> &field_e,
   const DofToQuadMap &dtqmap,
   field_operator_t &fop,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const bool &condition)
{
   if (condition)
   {
      if constexpr (std::is_same_v<T, TensorProduct>)
      {
         map_field_to_quadrature_data_tensor_product(field_qp, dtqmap,
                                                     field_e, fop,
                                                     integration_weights,
                                                     scratch_mem);
      }
      else
      {
         map_field_to_quadrature_data(field_qp, dtqmap, field_e, fop,
                                      integration_weights);
      }
   }
}

template <typename T = NonTensorProduct, size_t num_fields, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
MFEM_HOST_DEVICE
void map_fields_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   const std::array<DeviceTensor<1, const double>, num_fields> &fields_e,
   const std::array<DofToQuadMap, num_kinputs> &dtqmaps,
   field_operator_tuple_t fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_kinputs> &conditions,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional<T>(fields_qp[i],
                                                fields_e[i],
                                                dtqmaps[i],
                                                mfem::get<i>(fops),
                                                integration_weights,
                                                scratch_mem,
                                                conditions[i]),
    ...);
}

template <typename T = NonTensorProduct, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
MFEM_HOST_DEVICE
void map_direction_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_kinputs> &directions_qp,
   const DeviceTensor<1, const double> &direction_e,
   const std::array<DofToQuadMap, num_kinputs> &dtqmaps,
   field_operator_tuple_t fops,
   const DeviceTensor<1, const double> &integration_weights,
   const std::array<DeviceTensor<1>, 6> &scratch_mem,
   const std::array<bool, num_kinputs> &conditions,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional<T>(directions_qp[i],
                                                direction_e,
                                                dtqmaps[i],
                                                mfem::get<i>(fops),
                                                integration_weights,
                                                scratch_mem,
                                                conditions[i]),
    ...);
}

template <typename input_t, std::size_t... i>
std::array<int, sizeof...(i)> get_input_size_on_qp(
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

template <typename entity_t, typename input_t, size_t num_fields, size_t num_inputs, size_t num_outputs>
SharedMemoryInfo<num_fields, num_inputs, num_outputs>
get_shmem_info(
   std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
   std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
   const std::array<FieldDescriptor, num_fields> &fields,
   int num_entities,
   const input_t &inputs,
   int num_qp,
   const std::array<int, num_inputs> &input_size_on_qp,
   int residual_size_on_qp,
   int derivative_idx = -1)
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

template <typename shmeminfo_t>
void print_shared_memory_info(
   shmeminfo_t &shmem_info)
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

template <size_t N, size_t num_kinputs>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<1, const double>, num_kinputs>
load_field_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes,
   const std::array<int, num_kinputs> &kinput_to_field,
   const std::array<DeviceTensor<2, const double>, N> &fields_e,
   const int &entity_idx)
{
   std::array<DeviceTensor<1, const double>, num_kinputs> f;
   for (int i = 0; i < N; i++)
   {
      int block_size = MFEM_THREAD_SIZE(x) *
                       MFEM_THREAD_SIZE(y) *
                       MFEM_THREAD_SIZE(z);
      int tid = MFEM_THREAD_ID(x) +
                MFEM_THREAD_SIZE(x) *
                (MFEM_THREAD_ID(y) + MFEM_THREAD_SIZE(y) * MFEM_THREAD_ID(z));
      for (int k = tid; k < sizes[i]; k += block_size)
      {
         reinterpret_cast<real_t *>(mem)[offset + k] = fields_e[i](k, entity_idx);
      }

      for (int j = 0; j < num_kinputs; j++)
      {
         if (kinput_to_field[j] == i)
         {
            f[j] = DeviceTensor<1, const double>(&reinterpret_cast<real_t *>(mem)[offset],
                                                 sizes[i]);
         }
      }
      offset += sizes[i];
   }
   return f;
}

MFEM_HOST_DEVICE inline
DeviceTensor<1, const double> load_direction_mem(
   void *mem,
   int offset,
   const int &size,
   const DeviceTensor<2, const double> &direction,
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

   return DeviceTensor<1, const double>(
             &reinterpret_cast<real_t *>(mem)[offset], size);
}

template <size_t N>
MFEM_HOST_DEVICE inline
std::array<DeviceTensor<2>, N> load_input_mem(
   void *mem,
   int offset,
   const std::array<int, N> &sizes,
   int M)
{
   std::array<DeviceTensor<2>, N> f;
   for (int i = 0; i < N; i++)
   {
      f[i] = DeviceTensor<2>(&reinterpret_cast<real_t *>(mem)[offset], sizes[i] / M,
                             M);
      offset += sizes[i];
   }
   return f;
}

MFEM_HOST_DEVICE inline
DeviceTensor<2> load_residual_mem(
   void *mem,
   int offset,
   int residual_size,
   int num_qp)
{
   return DeviceTensor<2>(reinterpret_cast<real_t *>(mem) + offset, residual_size,
                          num_qp);
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

template<size_t N, size_t... i>
std::array<DeviceTensor<2, const double>, N> wrap_fields_impl(
   std::array<Vector, N> &fields,
   std::array<int, N> &field_sizes,
   int num_entities,
   std::index_sequence<i...>)
{
   return std::array<DeviceTensor<2, const double>, N>
   {
      {
         DeviceTensor<2, const double>(
            fields[i].Read(),
            field_sizes[i],
            num_entities)...
      }
   };
}

template <size_t N>
std::array<DeviceTensor<2, const double>, N> wrap_fields(
   std::array<Vector, N> &fields,
   std::array<int, N> &field_sizes,
   int num_entities)
{
   return wrap_fields_impl(fields, field_sizes, num_entities,
                           std::make_index_sequence<N> {});
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

MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   double &arg)
{
   arg = u(0);
}

MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<double> &arg)
{
   arg(0) = u(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<T, n> &arg)
{
   for (int i = 0; i < n; i++)
   {
      arg(i) = u(i);
   }
}

template <int n, int m>
MFEM_HOST_DEVICE
void process_kf_arg(
   const DeviceTensor<1> &u,
   internal::tensor<double, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * m) + j);
      }
   }
   // assuming col major layout. translating to row major.
   // i + N_i*j
   // arg(0, 0) = u(0);
   // arg(0, 1) = u(0 + 2 * 1);
   // arg(1, 0) = u(1 + 2 * 0);
   // arg(1, 1) = u(1 + 2 * 1);
}

template <typename arg_type>
MFEM_HOST_DEVICE
void process_kf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   // out << "qp: " << qp << "\n";
   // for (int i = 0; i < u.GetShape()[0] * u.GetShape()[1]; i++)
   // {
   //    out << (&u(0, 0))[i] << " ";
   // }
   // out << "\n";

   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   // for (int i = 0; i < u_qp.GetShape()[0]; i++)
   // {
   //    out << (&u_qp(0))[i] << " ";
   // }
   // out << "\n";

   process_kf_arg(u_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i>
MFEM_HOST_DEVICE
void process_kf_args(const std::array<DeviceTensor<2>, num_fields> &u,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (process_kf_arg(u[i], mfem::get<i>(args), qp), ...);
}

template <typename T0, typename T1> inline
Vector process_kf_result(T0, T1)
{
   static_assert(always_false<T0, T1>,
                 "process_kf_result not implemented for result type");
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const double &x)
{
   r(0) = x;
}

template <typename T>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T> &x)
{
   r(0) = x(0);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T, n> &x)
{
   for (size_t i = 0; i < n; i++)
   {
      r(i) = x(i);
   }
}

template <typename T, int n, int m>
MFEM_HOST_DEVICE inline
void process_kf_result(
   DeviceTensor<1, T> &r,
   const internal::tensor<T, n, m> &x)
{
   // out << "x: " << x << "\n";
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(i + n * j) = x(i, j);
      }
   }

   // out << "r: ";
   // for (int i = 0; i < r.GetShape()[0]; i++)
   // {
   //    out << r(i) << " ";
   // }
   // out << "\n\n";
}

template <typename kernel_func_t, typename kernel_args_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel(
   DeviceTensor<1, double> &f_qp,
   const kernel_func_t &kf,
   kernel_args_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   int qp)
{
   process_kf_args(u, args, qp,
                   std::make_index_sequence<mfem::tuple_size<kernel_args_ts>::value> {});

   process_kf_result(f_qp, mfem::get<0>(mfem::apply(kf, args)));
}

// Version for active function arguments only
//
// This is an Enzyme regression and can be removed in later versions.
template <typename kernel_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts>
inline auto fwddiff_apply_enzyme_indexed(kernel_t kernel, arg_ts &&args,
                                         arg_ts &&shadow_args,
                                         std::index_sequence<Is...>,
                                         inactive_arg_ts &&inactive_args,
                                         std::index_sequence<>)
{
   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;
   return __enzyme_fwddiff<kf_return_t>(
             +kernel, enzyme_dup, &mfem::get<Is>(args)..., enzyme_interleave,
             &mfem::get<Is>(shadow_args)...);
}

// Interleave function arguments for enzyme
template <typename kernel_t, typename arg_ts, std::size_t... Is,
          typename inactive_arg_ts, std::size_t... Js>
inline auto fwddiff_apply_enzyme_indexed(kernel_t kernel, arg_ts &&args,
                                         arg_ts &&shadow_args,
                                         std::index_sequence<Is...>,
                                         inactive_arg_ts &&inactive_args,
                                         std::index_sequence<Js...>)
{
   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;
   return __enzyme_fwddiff<kf_return_t>(
             +kernel, enzyme_dup, &std::get<Is>(args)..., enzyme_const,
             &mfem::get<Js>(inactive_args)..., enzyme_interleave,
             &mfem::get<Is>(shadow_args)...);
}

template <typename kernel_t, typename arg_ts, typename inactive_arg_ts>
inline auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args,
                                 arg_ts &&shadow_args,
                                 inactive_arg_ts &&inactive_args)
{
   auto arg_indices = std::make_index_sequence<
                      mfem::tuple_size<std::remove_reference_t<arg_ts>>::value> {};

   auto inactive_arg_indices = std::make_index_sequence<
                               mfem::tuple_size<std::remove_reference_t<inactive_arg_ts>>::value> {};

   return fwddiff_apply_enzyme_indexed(kernel, args, shadow_args, arg_indices,
                                       inactive_args, inactive_arg_indices);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args>
MFEM_HOST_DEVICE inline
void apply_kernel_fwddiff_enzyme(
   DeviceTensor<1, double> &f_qp,
   const kf_t &kf,
   kernel_arg_ts &args,
   const std::array<DeviceTensor<2>, num_args> &u,
   kernel_arg_ts &shadow_args,
   const std::array<DeviceTensor<2>, num_args> &v,
   int qp_idx)
{
   process_kf_args(u, args, qp_idx,
                   std::make_index_sequence<mfem::tuple_size<kernel_arg_ts>::value> {});

   process_kf_args(v, shadow_args, qp_idx,
                   std::make_index_sequence<mfem::tuple_size<kernel_arg_ts>::value> {});

   process_kf_result(f_qp,
                     mfem::get<0>(fwddiff_apply_enzyme(kf, args, shadow_args, mfem::tuple<> {})));
}

template <typename T> inline
void process_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    double &arg)
{
   arg = u(0);
}


template <int n, int m> inline
void process_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<double, n, m> &arg)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         arg(j, i) = u((i * m) + j);
      }
   }
}

template <typename arg_type> inline
void process_kf_arg(const DeviceTensor<2> &u, const DeviceTensor<2> &v,
                    arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   process_kf_arg(u_qp, v_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i> inline
void process_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     std::array<DeviceTensor<2>, num_fields> &v,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (process_kf_arg(u[i], v[i], mfem::get<i>(args), qp), ...);
}

template <typename output_type>
MFEM_HOST_DEVICE
void map_quadrature_data_to_fields_impl(DeviceTensor<2, double> &y,
                                        const DeviceTensor<3, double> &f,
                                        output_type output,
                                        const DofToQuadMap &dtq)
{
   auto B = dtq.B;
   auto G = dtq.G;
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<decltype(output), BareFieldOperator::Value>)
   {
      const auto [num_qp, cdim, num_dof] = B.GetShape();
      const int vdim = output.vdim > 0 ? output.vdim : cdim ;
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               acc += B(qp, 0, dof) * f(vd, 0, qp);
            }
            y(dof, vd) += acc;
         }
      }
   }
   else if constexpr (
      std::is_same_v<decltype(output), BareFieldOperator::Gradient>)
   {
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = output.vdim;
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  acc += G(qp, d, dof) * f(vd, d, qp);
               }
            }
            y(dof, vd) += acc;
         }
      }
   }
   // else if constexpr (std::is_same_v<decltype(output), One>)
   // {
   //    // This is the "integral over all quadrature points type" applying
   //    // B = 1 s.t. B^T * C \in R^1.
   //    const auto [a, b, num_qp] = B.GetShape();
   //    auto cc = Reshape(&c(0, 0, 0), num_qp);
   //    for (int i = 0; i < num_qp; i++)
   //    {
   //       y(0, 0) += cc(i);
   //    }
   // }
   else if constexpr (std::is_same_v<decltype(output), BareFieldOperator::None>)
   {
      const auto [vdim, dim, num_qp] = G.GetShape();
      auto cc = Reshape(&f(0, 0, 0), num_qp * vdim);
      auto yy = Reshape(&y(0, 0), num_qp * vdim);
      for (int i = 0; i < num_qp * vdim; i++)
      {
         yy(i) = cc(i);
      }
   }
   else
   {
      MFEM_ABORT("quadrature data mapping to field is not implemented for"
                 " this field descriptor");
   }
}

template <typename output_type>
MFEM_HOST_DEVICE
void map_quadrature_data_to_fields_tensor_impl(DeviceTensor<2, double> &y,
                                               const DeviceTensor<3, double> &f,
                                               output_type output,
                                               const DofToQuadMap &dtq,
                                               std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   auto B = dtq.B;
   auto G = dtq.G;

   if constexpr (std::is_same_v<decltype(output), BareFieldOperator::Value>)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      const int vdim = output.vdim;
      const int test_dim = output.size_on_qp / vdim;

      auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
      auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);

      auto s0 = Reshape(&scratch_mem[0](0), q1d, q1d, d1d);
      auto s1 = Reshape(&scratch_mem[1](0), q1d, d1d, d1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(qy, y, q1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  double acc = 0.0;
                  for (int qx = 0; qx < q1d; qx++)
                  {
                     acc += fqp(vd, 0, qx, qy, qz) * B(qx, 0, dx);
                  }
                  s0(qz, qy, dx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  double acc = 0.0;
                  for (int qy = 0; qy < q1d; qy++)
                  {
                     acc += s0(qz, qy, dx) * B(qy, 0, dy);
                  }
                  s1(qz, dy, dx) = acc;
               }
            }
         }
         MFEM_SYNC_THREAD;


         MFEM_FOREACH_THREAD(dy, y, d1d)
         {
            MFEM_FOREACH_THREAD(dx, x, d1d)
            {
               MFEM_FOREACH_THREAD(dz, z, d1d)
               {
                  double acc = 0.0;
                  for (int qz = 0; qz < q1d; qz++)
                  {
                     acc += s1(qz, dy, dx) * B(qz, 0, dz);
                  }
                  yd(dx, dy, dz, vd) += acc;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else if constexpr (
      std::is_same_v<decltype(output), BareFieldOperator::Gradient>)
   {
      const auto [q1d, unused, d1d] = G.GetShape();
      const int vdim = output.vdim;
      const int test_dim = output.size_on_qp / vdim;
      auto fqp = Reshape(&f(0, 0, 0), vdim, test_dim, q1d, q1d, q1d);
      auto yd = Reshape(&y(0, 0), d1d, d1d, d1d, vdim);

      auto s0 = Reshape(&scratch_mem[0](0), q1d, q1d, d1d);
      auto s1 = Reshape(&scratch_mem[1](0), q1d, q1d, d1d);
      auto s2 = Reshape(&scratch_mem[2](0), q1d, q1d, d1d);
      auto s3 = Reshape(&scratch_mem[3](0), q1d, d1d, d1d);
      auto s4 = Reshape(&scratch_mem[4](0), q1d, d1d, d1d);
      auto s5 = Reshape(&scratch_mem[5](0), q1d, d1d, d1d);

      for (int vd = 0; vd < vdim; vd++)
      {
         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int qx = 0; qx < q1d; qx++)
                  {
                     uvw[0] += fqp(vd, 0, qx, qy, qz) * G(qx, 0, dx);
                     uvw[1] += fqp(vd, 1, qx, qy, qz) * B(qx, 0, dx);
                     uvw[2] += fqp(vd, 2, qx, qy, qz) * B(qx, 0, dx);
                  }
                  s0(qz, qy, dx) = uvw[0];
                  s1(qz, qy, dx) = uvw[1];
                  s2(qz, qy, dx) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(qz, z, q1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int qy = 0; qy < q1d; qy++)
                  {
                     uvw[0] += s0(qz, qy, dx) * B(qy, 0, dy);
                     uvw[1] += s1(qz, qy, dx) * G(qy, 0, dy);
                     uvw[2] += s2(qz, qy, dx) * B(qy, 0, dy);
                  }
                  s3(qz, dy, dx) = uvw[0];
                  s4(qz, dy, dx) = uvw[1];
                  s5(qz, dy, dx) = uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz, z, d1d)
         {
            MFEM_FOREACH_THREAD(dy, y, d1d)
            {
               MFEM_FOREACH_THREAD(dx, x, d1d)
               {
                  real_t uvw[3] = {0.0, 0.0, 0.0};
                  for (int qz = 0; qz < q1d; qz++)
                  {
                     uvw[0] += s3(qz, dy, dx) * B(qz, 0, dz);
                     uvw[1] += s4(qz, dy, dx) * B(qz, 0, dz);
                     uvw[2] += s5(qz, dy, dx) * G(qz, 0, dz);
                  }
                  yd(dx, dy, dz, vd) += uvw[0] + uvw[1] + uvw[2];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else if constexpr (std::is_same_v<decltype(output), BareFieldOperator::None>)
   {
      const auto [q1d, unused, d1d] = B.GetShape();
      auto fqp = Reshape(&f(0, 0, 0), output.size_on_qp, q1d, q1d, q1d);
      auto yqp = Reshape(&y(0, 0), output.size_on_qp, q1d, q1d, q1d);

      for (int sq = 0; sq < output.size_on_qp; sq++)
      {
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  yqp(sq, qx, qy, qz) = fqp(sq, qx, qy, qz);
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   }
   else
   {
      MFEM_ABORT("quadrature data mapping to field is not implemented for"
                 " this field descriptor with sum factorization on tensor product elements");
   }
}

template <typename T = NonTensorProduct, typename output_type>
MFEM_HOST_DEVICE
void map_quadrature_data_to_fields(DeviceTensor<2, double> &y,
                                   const DeviceTensor<3, double> &f,
                                   output_type output,
                                   const DofToQuadMap &dtq,
                                   std::array<DeviceTensor<1>, 6> &scratch_mem)
{
   if constexpr (std::is_same_v<T, NonTensorProduct>)
   {
      map_quadrature_data_to_fields_impl(y, f, output, dtq);
   }
   else if constexpr (std::is_same_v<T, TensorProduct>)
   {
      map_quadrature_data_to_fields_tensor_impl(y, f, output, dtq, scratch_mem);
   }
}

template <typename entity_t, typename field_operator_ts, size_t N, std::size_t... i>
std::array<DofToQuadMap, N> create_dtq_maps_impl(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqs,
   const std::array<int, N> &field_map,
   std::index_sequence<i...>)
{
   auto f = [&](auto fop, size_t idx)
   {
      auto g = [&](int idx)
      {
         auto dtq = dtqs[field_map[idx]];

         int value_dim = 1;
         int grad_dim = 1;

         if ((dtq->mode != DofToQuad::Mode::TENSOR) &&
             (!std::is_same_v<decltype(fop), None>))
         {
            value_dim = dtq->FE->GetRangeDim() ?
                        dtq->FE->GetRangeDim() :
                        fop.vdim;

            grad_dim = dtq->FE->GetDim();
         }

         return std::tuple{dtq, value_dim, grad_dim};
      };

      if constexpr (std::is_same_v<decltype(fop), Value> ||
                    std::is_same_v<decltype(fop), Gradient>)
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
      else if constexpr (std::is_same_v<decltype(fop), None>)
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
      f(mfem::get<i>(fops), i)...
   };
}

template <typename entity_t, typename field_operator_ts, size_t N>
std::array<DofToQuadMap, N> create_dtq_maps(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map)
{
   return create_dtq_maps_impl<entity_t>(
             fops, dtqmaps,
             to_field_map,
             std::make_index_sequence<mfem::tuple_size<field_operator_ts>::value> {});
}

template <typename field_operator_ts, std::size_t... I>
auto create_bare_fops_impl(
   const field_operator_ts &fops,
   std::index_sequence<I...>)
{
   auto f = [&](auto fop, size_t idx)
   {
      if constexpr (std::is_same_v<decltype(fop), Weight>)
      {
         return BareFieldOperator::Weight(fop);
      }
      else if constexpr (std::is_same_v<decltype(fop), Value>)
      {
         return BareFieldOperator::Value(fop);
      }
      else if constexpr (std::is_same_v<decltype(fop), Gradient>)
      {
         return BareFieldOperator::Gradient(fop);
      }
      else if constexpr (std::is_same_v<decltype(fop), None>)
      {
         return BareFieldOperator::None(fop);
      }
      else
      {
         static_assert(always_false<decltype(fop)>,
                       "field operator type is not implemented");
         return BareFieldOperator::Base(fop);
      }
   };
   return mfem::make_tuple(f(mfem::get<I>(fops), I)...);
}

template <typename field_operator_ts>
auto create_bare_fops(const field_operator_ts &fops)
{
   return create_bare_fops_impl(
             fops,
             std::make_index_sequence<mfem::tuple_size<field_operator_ts>::value> {});
}
}
