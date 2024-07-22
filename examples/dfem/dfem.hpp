#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>
#include <mfem.hpp>
#include <general/forall.hpp>
#include <type_traits>
#include "dfem_fieldoperator.hpp"
#include "dfem_parametricspace.hpp"
#include "fem/qspace.hpp"
#include "fem/restriction.hpp"
#include "general/backends.hpp"
#include "linalg/operator.hpp"
#include "mesh/mesh.hpp"
#include <linalg/tensor.hpp>

#include "noisy.hpp"
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

void print_matrix(const mfem::DenseMatrix m)
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

void print_vector(const mfem::Vector v)
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

template <typename ... Ts>
constexpr auto decay_types(std::tuple<Ts...> const &)
-> std::tuple<std::remove_cv_t<std::remove_reference_t<Ts>>...>;

template <typename T>
using decay_tuple = decltype(decay_types(std::declval<T>()));

template <class F> struct FunctionSignature;

template <typename output_t, typename... input_ts>
struct FunctionSignature<output_t(input_ts...)>
{
   using return_t = output_t;
   using parameter_ts = std::tuple<input_ts...>;
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

using mult_func_t = std::function<void(Vector &)>;

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
struct ElementOperator<func_t, std::tuple<input_ts...>,
          std::tuple<output_ts...>>
{
   using entity_t = Entity::Element;
   func_t func;
   std::tuple<input_ts...> inputs;
   std::tuple<output_ts...> outputs;
   using kf_param_ts = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::parameter_ts;

   using kf_output_t = typename create_function_signature<
                       decltype(&decltype(func)::operator())>::type::return_t;

   using kernel_inputs_t = decltype(inputs);
   using kernel_outputs_t = decltype(outputs);

   static constexpr size_t num_kinputs = std::tuple_size_v<kernel_inputs_t>;
   static constexpr size_t num_koutputs = std::tuple_size_v<kernel_outputs_t>;

   Array<int> attributes;

   ElementOperator(func_t func,
                   std::tuple<input_ts...> inputs,
                   std::tuple<output_ts...> outputs,
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

      constexpr size_t num_kfinputs = std::tuple_size_v<kf_param_ts>;
      static_assert(num_kfinputs == num_kinputs,
                    "kernel function inputs and descriptor inputs have to match");

      constexpr size_t num_kfoutputs = std::tuple_size_v<kf_output_t>;
      static_assert(num_kfoutputs == num_koutputs,
                    "kernel function outputs and descriptor outputs have to match");
   }
};

template <typename func_t, typename... input_ts,
          typename... output_ts>
ElementOperator(func_t, std::tuple<input_ts...>,
                std::tuple<output_ts...>)
-> ElementOperator<func_t, std::tuple<input_ts...>,
std::tuple<output_ts...>>;

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
         return 1;
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
      const Vector x_i(x.GetData() + data_offset, width);
      fields_l[i].SetSize(P->Height());

      P->Mult(x_i, fields_l[i]);
      data_offset += width;
   }
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
std::array<int, std::tuple_size_v<field_operator_ts>>
                                                   create_descriptors_to_fields_map(
                                                      std::array<FieldDescriptor, num_fields> &fields,
                                                      field_operator_ts &fops,
                                                      std::index_sequence<idx...>)
{
   std::array<int, std::tuple_size_v<field_operator_ts>> map;

   auto f = [&](auto &fop, auto &map)
   {
      int i;

      if constexpr (std::is_same_v<decltype(fop), Weight&>)
      {
         fop.dim = 1;
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

   (f(std::get<idx>(fops), map[idx]), ...);

   return map;
}

template <typename input_t, std::size_t... i>
std::array<DeviceTensor<2>, sizeof...(i)> map_inputs_to_memory(
   std::array<Vector, sizeof...(i)> &input_qp_mem, int num_qp,
   const input_t &inputs, std::index_sequence<i...>)
{
   return {DeviceTensor<2>(input_qp_mem[i].ReadWrite(), std::get<i>(inputs).size_on_qp, num_qp) ...};
}

template <typename input_t, std::size_t... i>
std::array<Vector, sizeof...(i)> create_input_qp_memory(
   int num_qp,
   input_t &inputs,
   std::index_sequence<i...>)
{
   return {Vector(std::get<i>(inputs).size_on_qp * num_qp)...};
}

struct DofToQuadMaps
{
   static constexpr int rank = 3;
   DeviceTensor<rank, const double> B;
   const int which_input = -1;

   MFEM_HOST_DEVICE inline
   const double& operator()(int qp, int d, int dof) const
   {
      return B(qp, d, dof);
   }

   MFEM_HOST_DEVICE inline
   std::array<int, rank> GetShape() const
   {
      return B.GetShape();
   }
};

template <typename field_operator_t>
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp,
   int entity_idx,
   const DofToQuadMaps &B,
   const Vector &field_e,
   field_operator_t &input,
   DeviceTensor<1, const double> integration_weights,
   GeometricFactorMaps geometric_factors)
{
   if constexpr (std::is_same_v<field_operator_t, Value>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int entity_offset = entity_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + entity_offset, num_dof, vdim);

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
   else if constexpr (std::is_same_v<field_operator_t, Gradient>)
   {
      const auto [num_qp, dim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int entity_offset = entity_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + entity_offset, num_dof, vdim);

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
                  acc += B(qp, d, dof) * field(dof, vd);
               }
               f(vd, d, qp) = acc;
            }
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, Curl>)
   {
      const auto [num_qp, cdim, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int entity_offset = entity_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + entity_offset, num_dof, vdim,
                                 cdim);

      auto f = Reshape(&field_qp[0], vdim, cdim, num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int cd = 0; cd < cdim; cd++)
            {
               double acc = 0.0;
               for (int dof = 0; dof < num_dof; dof++)
               {
                  acc += B(qp, cd, dof) * field(dof, vd, cd);
               }
               f(vd, cd, qp) = acc;
            }
         }
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, FaceValueLeft>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int entity_offset = entity_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + entity_offset, num_dof, vdim);

      // for (int vd = 0; vd < vdim; vd++)
      // {
      //    for (int qp = 0; qp < num_qp; qp++)
      //    {
      //       double acc = 0.0;
      //       for (int dof = 0; dof < num_dof; dof++)
      //       {
      //          acc += B(qp, 0, dof) * field(dof, vd);
      //       }
      //       field_qp(vd, qp) = acc;
      //    }
      // }
   }
   else if constexpr (std::is_same_v<field_operator_t, FaceValueRight>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int entity_offset = entity_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + entity_offset, num_dof, vdim);

      // for (int vd = 0; vd < vdim; vd++)
      // {
      //    for (int qp = 0; qp < num_qp; qp++)
      //    {
      //       double acc = 0.0;
      //       for (int dof = 0; dof < num_dof; dof++)
      //       {
      //          acc += B(qp, 0, dof) * field(dof, vd);
      //       }
      //       field_qp(vd, qp) = acc;
      //    }
      // }
   }
   else if constexpr (std::is_same_v<field_operator_t, FaceNormal>)
   {
      auto normal = geometric_factors.normal;
      auto [num_qp, dim, num_entities] = normal.GetShape();
      auto f = Reshape(&field_qp[0], dim, num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int d = 0; d < dim; d++)
         {
            f(d, qp) = normal(qp, d, entity_idx);
         }
      }
   }
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<field_operator_t, Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (std::is_same_v<field_operator_t, None>)
   {
      auto [num_qp, unused, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const int entity_offset = entity_idx * size_on_qp * num_qp;
      const auto field = Reshape(field_e.Read() + entity_offset,
                                 size_on_qp * num_qp);
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

template <size_t num_fields, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
void map_fields_to_quadrature_data(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   int element_idx,
   const std::array<Vector, num_fields> fields_e,
   const std::array<int, num_kinputs> &kfinput_to_field,
   const std::vector<DofToQuadMaps> &dtqmaps,
   const DeviceTensor<1, const double> &integration_weights,
   const GeometricFactorMaps &geometric_factors,
   field_operator_tuple_t fops,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[i], fields_e[kfinput_to_field[i]],
                                 std::get<i>(fops), integration_weights, geometric_factors),
    ...);
}

template <typename input_type>
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadMaps &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights,
   GeometricFactorMaps geometric_factors,
   const bool condition)
{
   if (condition)
   {
      map_field_to_quadrature_data(field_qp, element_idx, dtqmaps, field_e, input,
                                   integration_weights, geometric_factors);
   }
}

template <size_t num_fields, size_t num_kinputs, typename field_operator_tuple_t, std::size_t... i>
void map_fields_to_quadrature_data_conditional(
   std::array<DeviceTensor<2>, num_kinputs> &fields_qp,
   int element_idx,
   const std::array<Vector, num_fields> &fields_e,
   const std::array<int, num_kinputs> &kfinput_to_field,
   std::vector<DofToQuadMaps> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   GeometricFactorMaps geometric_factors,
   std::array<bool, num_kinputs> conditions,
   field_operator_tuple_t fops,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional(fields_qp[i], element_idx,
                                             dtqmaps[i], fields_e[kfinput_to_field[i]],
                                             std::get<i>(fops), integration_weights, geometric_factors,
                                             conditions[i]),
    ...);
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
      if (!is_dependent) { return 0; }
      return GetSizeOnQP(input, field);
   }(std::get<i>(inputs),
     std::get<i>(kinput_is_dependent),
     fields[kinput_to_field[i]]));
}

void prepare_kf_arg(const DeviceTensor<1> &u, double &arg) { arg = u(0); }

template <typename T, int length>
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::tensor<T, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

template <int n, int m>
void prepare_kf_arg(const DeviceTensor<1> &u,
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
void prepare_kf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   prepare_kf_arg(u_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i>
void prepare_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (prepare_kf_arg(u[i], std::get<i>(args), qp), ...);
}

inline
Vector prepare_kf_result(double x)
{
   Vector r(1);
   r = x;
   return r;
}

inline
Vector prepare_kf_result(Vector x)
{
   return x;
}

template <typename T, int length> inline
Vector prepare_kf_result(internal::tensor<T, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i);
   }
   return r;
}

template <typename T, int n, int m> inline
Vector prepare_kf_result(internal::tensor<T, n, m> x)
{
   Vector r(n * m);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         r(j + m * i) = x(i, j);
      }
   }
   return r;
}

template <typename T, int length> inline
Vector prepare_kf_result(
   internal::tensor<internal::dual<T, T>, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i).value;
   }
   return r;
}

// template <typename T> inline
// Vector prepare_kf_result(internal::tensor<T, 2, 2> x)
// {
//    Vector r(4);
//    for (size_t i = 0; i < 2; i++)
//    {
//       for (size_t j = 0; j < 2; j++)
//       {
//          // TODO: Careful with the indices here!
//          r(j + (i * 2)) = std::get<0>(x)(j, i);
//       }
//    }
//    return r;
// }

template <typename T> inline
Vector prepare_kf_result(internal::dual<T, T> x)
{
   Vector r(1);
   r(0) = x.value;
   return r;
}

template <typename T> inline
Vector get_derivative_from_dual(internal::dual<T, T> x)
{
   Vector r(1);
   r(0) = x.gradient;
   return r;
}

template <typename T, int n, int m> inline
Vector get_derivative_from_dual(
   internal::tensor<internal::dual<T, T>, n, m> x)
{
   Vector r(n * m);
   for (size_t i = 0; i < n; i++)
   {
      for (size_t j = 0; j < m; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * m)) = x(j, i).gradient;
      }
   }
   return r;
}

template <typename T, int length> inline
Vector get_derivative_from_dual(
   internal::tensor<internal::dual<T, T>, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i).gradient;
   }
   return r;
}

template <typename T> inline
Vector prepare_kf_result(T)
{
   static_assert(always_false<T>,
                 "prepare_kf_result not implemented for result type");
}

template <size_t num_fields, typename kernel_func_t, typename kernel_args>
inline
Vector apply_kernel(const kernel_func_t &kf, kernel_args &args,
                    std::array<DeviceTensor<2>, num_fields> &u,
                    int qp)
{
   prepare_kf_args(u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_args>> {});

   return prepare_kf_result(std::get<0>(std::apply(kf, args)));
}

template <typename arg_ts, std::size_t... Is> inline
auto create_enzyme_args(arg_ts &args,
                        arg_ts &shadow_args,
                        std::index_sequence<Is...>)
{
   // (out << ... << std::get<Is>(shadow_args));
   return std::tuple_cat(std::tie(enzyme_dup, std::get<Is>(args),
                                  std::get<Is>(shadow_args))...);
}

// template <typename arg_ts, std::size_t... Is> inline
// auto create_enzyme_args(arg_ts &args,
//                         arg_ts &shadow_args,
//                         std::index_sequence<Is...>)
// {
//    return std::tuple_cat(std::tie(
//                             enzyme::Duplicated<std::remove_cv_t<std::remove_reference_t<decltype(std::get<Is>(args))>>*> {&std::get<Is>(args), &std::get<Is>(shadow_args)})...);
//    // return std::tuple_cat(std::make_tuple(
//    //                          enzyme::Duplicated<std::remove_cv_t<std::remove_reference_t<decltype(std::get<Is>(args))>>*> {&std::get<Is>(args), &std::get<Is>(shadow_args)})...);
//    // return std::tuple<enzyme::Duplicated<decltype(std::get<Is>(args))>...>
//    // {
//    //    { std::get<Is>(args), std::get<Is>(shadow_args) }...
//    // };
// }

template <typename kernel_t, typename arg_ts> inline
auto fwddiff_apply_enzyme(kernel_t kernel, arg_ts &&args, arg_ts &&shadow_args)
{
   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_ts>>> {};

   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);

   using kf_return_t = typename create_function_signature<
                       decltype(&kernel_t::operator())>::type::return_t;

   return std::apply([&](auto &&...args)
   {
      // kernel: f(double &x, vec2d &y) -> std::tuple{ double r }
      // args: std::tuple{ enzyme_dup, &x, &dx, enzyme_dup, &y, &dy, ... }
      return __enzyme_fwddiff<kf_return_t>((void*)+kernel, &args...);
      // return enzyme::get<0>
      //        (enzyme::autodiff<enzyme::Forward, enzyme::DuplicatedNoNeed<kf_return_t>>
      //         (+kernel, args...));
   }, enzyme_args);
   // return kf_return_t{};
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args> inline
auto apply_kernel_fwddiff_enzyme(const kf_t &kf,
                                 kernel_arg_ts &args,
                                 std::array<DeviceTensor<2>, num_args> &u,
                                 kernel_arg_ts &shadow_args,
                                 std::array<DeviceTensor<2>, num_args> &v,
                                 int qp)
{
   prepare_kf_args(u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   prepare_kf_args(v, shadow_args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   return prepare_kf_result(std::get<0>(fwddiff_apply_enzyme(kf, args,
                                                             shadow_args)));
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    double &arg)
{
   arg = u(0);
}

template <typename T, int dim, int vdim> inline
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::tensor<internal::dual<T, T>, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i).value = u((i * vdim) + j);
      }
   }
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u,
                    internal::dual<T, T> &arg)
{
   arg.value = u(0);
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v, T &arg)
{
   arg = u(0);
}

template <typename T> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::dual<T, T> &arg)
{
   arg.value = u(0);
   arg.gradient = v(0);
}

template <typename T, int length> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<internal::dual<T, T>, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i).value = u(i);
      arg(i).gradient = v(i);
   }
}

template <int n, int m> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
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

template <typename T, int dim, int vdim> inline
void prepare_kf_arg(const DeviceTensor<1> &u, const DeviceTensor<1> &v,
                    internal::tensor<internal::dual<T, T>, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i).value = u((i * vdim) + j);
         arg(j, i).gradient = v((i * vdim) + j);
      }
   }
}

template <typename arg_type> inline
void prepare_kf_arg(const DeviceTensor<2> &u, const DeviceTensor<2> &v,
                    arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   const auto v_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   prepare_kf_arg(u_qp, v_qp, arg);
}

template <size_t num_fields, typename kf_args, std::size_t... i> inline
void prepare_kf_args(std::array<DeviceTensor<2>, num_fields> &u,
                     std::array<DeviceTensor<2>, num_fields> &v,
                     kf_args &args, int qp, std::index_sequence<i...>)
{
   (prepare_kf_arg(u[i], v[i], std::get<i>(args), qp), ...);
}

template <typename kernel_t, typename arg_ts> inline
auto fwddiff_apply_dual(kernel_t kernel, arg_ts &&args)
{
   return std::apply([&](auto &&...args)
   {
      return kernel(args...);
   },
   args);
}

template <typename kf_t, typename kernel_arg_ts, size_t num_args> inline
auto apply_kernel_fwddiff_dual(const kf_t &kf,
                               kernel_arg_ts &args,
                               std::array<DeviceTensor<2>, num_args> &u,
                               std::array<DeviceTensor<2>, num_args> &v,
                               int qp)
{
   prepare_kf_args(u, v, args, qp,
                   std::make_index_sequence<std::tuple_size_v<kernel_arg_ts>> {});

   return get_derivative_from_dual(fwddiff_apply_dual(kf, args));
}

template <typename output_type>
void map_quadrature_data_to_fields(DeviceTensor<2, double> y,
                                   DeviceTensor<3, double> c,
                                   output_type output,
                                   DofToQuadMaps &B)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<decltype(output), Value>)
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
               acc += B(qp, 0, dof) * c(vd, 0, qp);
            }
            y(dof, vd) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), Gradient>)
   {
      const auto [num_qp, dim, num_dof] = B.GetShape();
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
                  acc += B(qp, d, dof) * c(vd, d, qp);
               }
            }
            y(dof, vd) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<decltype(output), One>)
   {
      // This is the "integral over all quadrature points type" applying
      // B = 1 s.t. B^T * C \in R^1.
      int num_qp;
      std::tie(std::ignore, std::ignore, num_qp) = c.GetShape();
      auto cc = Reshape(&c(0, 0, 0), num_qp);
      for (int i = 0; i < num_qp; i++)
      {
         y(0, 0) += cc(i);
      }
   }
   else if constexpr (std::is_same_v<decltype(output), None>)
   {
      const auto [vdim, dim, num_qp] = c.GetShape();
      auto cc = Reshape(&c(0, 0, 0), num_qp * vdim);
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

template <typename entity_t, typename field_operator_ts, size_t N, std::size_t... i>
std::vector<DofToQuadMaps> create_dtq_operators_conditional(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map,
   std::array<bool, N> active,
   std::index_sequence<i...>)
{
   std::vector<DofToQuadMaps> ops;
   auto f = [&](auto fop, size_t idx)
   {
      if (active[idx])
      {
         if (to_field_map[idx] == -1)
         {
            ops.push_back({DeviceTensor<3, const double>(nullptr, 1, 1, 1), -1});
            return;
         }
         auto dtqmap = dtqmaps[to_field_map[idx]];
         if constexpr (std::is_same_v<decltype(fop), Value>)
         {
            const int d = dtqmap->FE->GetRangeDim() ? dtqmap->FE->GetRangeDim() : 1;
            ops.push_back({DeviceTensor<3, const double>(dtqmap->B.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Gradient>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::VALUE,
                        "trying to compute gradient of non compatible FE type");
            const int d = dtqmap->FE->GetDim();
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Curl>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::H_CURL,
                        "trying to compute gradient of non compatible FE type");
            const int d = dtqmap->FE->GetCurlDim();
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, d, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr (std::is_same_v<decltype(fop), Div>)
         {
            MFEM_ASSERT(dtqmap->FE->GetMapType() == FiniteElement::MapType::H_DIV,
                        "trying to compute gradient of non compatible FE type");
            ops.push_back({DeviceTensor<3, const double>(dtqmap->G.Read(), dtqmap->nqpt, 1, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr(std::is_same_v<decltype(fop), One>)
         {
            ops.push_back({DeviceTensor<3, const double>(nullptr, 1, 1, 1), static_cast<int>(idx)});
         }
         else if constexpr(std::is_same_v<decltype(fop), None>)
         {
            ops.push_back({DeviceTensor<3, const double>(nullptr, dtqmap->nqpt, 1, dtqmap->ndof), static_cast<int>(idx)});
         }
         else if constexpr(std::is_same_v<decltype(fop), Weight>)
         {
            // no op
            // this is handled at runtime by the first condition
            // to_field_map[idx] == -1.
            // has to exist at compile time for completeness
         }
         else
         {
            static_assert(always_false<decltype(fop)>,
                          "field operator type is not implemented");
         }
      }
   };
   (f(std::get<i>(fops), i), ...);
   return ops;
}

template <typename entity_t, typename field_operator_ts, size_t N>
std::vector<DofToQuadMaps> create_dtq_operators(
   field_operator_ts &fops,
   std::vector<const DofToQuad*> dtqmaps,
   const std::array<int, N> &to_field_map)
{
   std::array<bool, N> is_dependent;
   std::fill(is_dependent.begin(), is_dependent.end(), true);
   return create_dtq_operators_conditional<entity_t>(fops, dtqmaps,
                                                     to_field_map,
                                                     is_dependent,
                                                     std::make_index_sequence<std::tuple_size_v<field_operator_ts>> {});
}

template <
   typename kernels_tuple,
   size_t num_solutions,
   size_t num_parameters,
   size_t num_fields = num_solutions + num_parameters,
   size_t num_kernels = std::tuple_size_v<kernels_tuple>
   >
class DifferentiableOperator : public Operator
{
public:
   DifferentiableOperator(DifferentiableOperator&) = delete;
   DifferentiableOperator(DifferentiableOperator&&) = delete;

   class Action : public Operator
   {
   public:
      template <typename kernel_t>
      void create_action_callback(kernel_t kernel, mult_func_t &func);

      template<std::size_t... idx>
      void materialize_callbacks(kernels_tuple &ks,
                                 std::array<mult_func_t, num_kernels>,
                                 std::index_sequence<idx...> const&)
      {
         (create_action_callback(std::get<idx>(ks), funcs[idx]), ...);
      }

      Action(DifferentiableOperator &op, kernels_tuple &ks) : op(op)
      {
         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<std::tuple_size_v<kernels_tuple>>());
      }

      void Mult(const Vector &x, Vector &y) const
      {
         prolongation(op.solutions, x, solutions_l);

         residual_e = 0.0;
         for (const auto &f : funcs)
         {
            f(residual_e);
         }

         prolongation_transpose(residual_l, y);

         y.SetSubVector(op.ess_tdof_list, 0.0);
      }

      void SetParameters(std::vector<Vector *> p) const
      {
         MFEM_ASSERT(num_parameters == p.size(),
                     "number of parameters doesn't match descriptors");
         for (int i = 0; i < num_parameters; i++)
         {
            // parameters_l[i].MakeRef(*(p[i]), 0, p[i]->Size());
            parameters_l[i] = *(p[i]);
         }
      }

      const Vector& GetResidualQpMemory() const
      {
         return residual_qp_mem;
      }

   protected:
      DifferentiableOperator &op;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> restriction_transpose;
      std::function<void(Vector &, Vector &)> prolongation_transpose;

      mutable std::array<Vector, num_solutions> solutions_l;
      mutable std::array<Vector, num_parameters> parameters_l;
      mutable Vector residual_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable Vector residual_e;
      Vector residual_qp_mem;
   };

   template <size_t derivative_idx>
   class Derivative : public Operator
   {
   public:
      template <typename kernel_t>
      void create_callback(kernel_t kernel, mult_func_t &func);

      template<std::size_t... idx>
      void materialize_callbacks(kernels_tuple &ks,
                                 std::array<mult_func_t, num_kernels>,
                                 std::index_sequence<idx...> const&)
      {
         (create_callback(std::get<idx>(ks), funcs[idx]), ...);
      }

      Derivative(
         DifferentiableOperator &op,
         std::array<Vector *, num_solutions> &solutions,
         std::array<Vector *, num_parameters> &parameters,
         kernels_tuple &ks) : op(op), ks(ks)
      {
         for (int i = 0; i < num_solutions; i++)
         {
            solutions_l[i] = *solutions[i];
         }

         for (int i = 0; i < num_parameters; i++)
         {
            parameters_l[i] = *parameters[i];
         }

         // G
         // if constexpr (std::is_same_v<OperatesOn, OperatesOnElement>)
         // {
         element_restriction(op.solutions, solutions_l, fields_e,
                             op.element_dof_ordering);
         element_restriction(op.parameters, parameters_l, fields_e,
                             op.element_dof_ordering,
                             op.solutions.size());
         // }
         // else
         // {
         //    MFEM_ABORT("restriction not implemented for OperatesOn");
         // }

         // TODO-multvar: doesn't work for multiple solution variables
         directions[0] = op.fields[derivative_idx];

         size_t derivative_action_l_size = 0;
         for (auto &s : op.solutions)
         {
            derivative_action_l_size += GetVSize(s);
            this->width += GetTrueVSize(s);
         }
         this->height = derivative_action_l_size;
         derivative_action_l.SetSize(derivative_action_l_size);

         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<num_kernels>());
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         current_directions_t = x;
         current_directions_t.SetSubVector(op.ess_tdof_list, 0.0);

         prolongation(directions, current_directions_t, directions_l);

         derivative_action_e = 0.0;
         for (const auto &f : funcs)
         {
            f(derivative_action_e);
         }

         prolongation_transpose(derivative_action_l, y);

         y.SetSubVector(op.ess_tdof_list, 0.0);
      }

      template <typename kernel_t>
      void assemble_vector_impl(kernel_t kernel, Vector &v);

      template<std::size_t... idx>
      void assemble_vector(
         kernels_tuple &ks,
         Vector &v,
         std::index_sequence<idx...> const&)
      {
         (assemble_vector_impl(std::get<idx>(ks), v), ...);
      }

      void Assemble(Vector &v)
      {
         assemble_vector(ks, v, std::make_index_sequence<num_kernels>());
      }

      template <typename kernel_t>
      void assemble_hypreparmatrix_impl(kernel_t kernel, HypreParMatrix &A);

      template<std::size_t... idx>
      void assemble_hypreparmatrix(
         kernels_tuple &ks,
         HypreParMatrix &A,
         std::index_sequence<idx...> const&)
      {
         (assemble_hypreparmatrix_impl(std::get<idx>(ks), A), ...);
      }

      void Assemble(HypreParMatrix &A)
      {
         assemble_hypreparmatrix(ks, A, std::make_index_sequence<num_kernels>());
      }

      void AssembleDiagonal(Vector &d) const override {}

      const Vector& GetResidualQpMemory() const
      {
         return da_qp_mem;
      }

   protected:
      DifferentiableOperator &op;
      kernels_tuple &ks;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> restriction_transpose;
      std::function<void(Vector &, Vector &)> prolongation_transpose;

      // TODO-multvar: doesn't work for multiple solution variables
      std::array<FieldDescriptor, 1> directions;

      std::array<Vector, num_solutions> solutions_l;
      std::array<Vector, num_parameters> parameters_l;
      mutable std::array<Vector, 1> directions_l;
      mutable Vector derivative_action_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable std::array<Vector, num_fields> directions_e;
      mutable Vector derivative_action_e;

      mutable Vector current_directions_t;

      Vector da_qp_mem;
   };

   DifferentiableOperator(std::array<FieldDescriptor, num_solutions> s,
                          std::array<FieldDescriptor, num_parameters> p,
                          kernels_tuple ks,
                          ParMesh &m,
                          const IntegrationRule &integration_rule) :
      kernels(ks),
      mesh(m),
      dim(mesh.Dimension()),
      integration_rule(integration_rule),
      solutions(s),
      parameters(p)
   {
      for (int i = 0; i < num_solutions; i++)
      {
         fields[i] = solutions[i];
      }

      for (int i = 0; i < num_parameters; i++)
      {
         fields[i + num_solutions] = parameters[i];
      }

      residual.reset(new Action(*this, kernels));
   }

   void SetParameters(std::vector<Vector *> p) const
   {
      residual->SetParameters(p);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      residual->Mult(x, y);
   }

   template <int derivative_idx>
   std::shared_ptr<Derivative<derivative_idx>>
                                            GetDerivativeWrt(std::array<Vector *, num_solutions> solutions,
                                                             std::array<Vector *, num_parameters> parameters)
   {
      return std::shared_ptr<Derivative<derivative_idx>>(
                new Derivative<derivative_idx>(*this, solutions, parameters, kernels));
   }

   // This function returns a Vector holding the memory right after the kernel function
   // has been executed. This means the output transformation has not been applied yet,
   // which is useful for testing or intricate, advanced usage. It is not recommended to
   // use or rely on this function.
   const Vector& GetResidualQpMemory() const
   {
      return residual->GetResidualQpMemory();
   }

   void SetEssentialTrueDofs(const Array<int> &l) { l.Copy(ess_tdof_list); }

   kernels_tuple kernels;
   ParMesh &mesh;
   const int dim;
   const IntegrationRule &integration_rule;

   std::array<FieldDescriptor, num_solutions> solutions;
   std::array<FieldDescriptor, num_parameters> parameters;
   // solutions and parameters
   std::array<FieldDescriptor, num_fields> fields;

   int residual_lsize = 0;

   mutable std::array<Vector, num_solutions> current_state_l;
   mutable Vector direction_l;

   mutable Vector current_direction_t;

   Array<int> ess_tdof_list;

   // static constexpr ElementDofOrdering element_dof_ordering =
   //    ElementDofOrdering::LEXICOGRAPHIC;

   // static constexpr DofToQuad::Mode doftoquad_mode =
   //    DofToQuad::Mode::LEXICOGRAPHIC_FULL;

   static constexpr ElementDofOrdering element_dof_ordering =
      ElementDofOrdering::NATIVE;

   static constexpr DofToQuad::Mode doftoquad_mode =
      DofToQuad::Mode::FULL;

   std::shared_ptr<Action> residual;
};

#include "dfem_action_callback.icc"
#include "dfem_derivative_callback.icc"
#include "dfem_assemble_vector.icc"
#include "dfem_assemble_hypreparmatrix.icc"

}
