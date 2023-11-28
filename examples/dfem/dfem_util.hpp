#pragma once

#include <tuple>
#include <mfem.hpp>
#include <general/forall.hpp>
#include "dfem_fieldoperator.hpp"
#include "general/error.hpp"

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

void print_matrix(mfem::DenseMatrix m)
{
   std::cout << "{";
   for (int i = 0; i < m.NumRows(); i++)
   {
      std::cout << "{";
      for (int j = 0; j < m.NumCols(); j++)
      {
         std::cout << m(i, j);
         if (j < m.NumCols() - 1)
         {
            std::cout << ", ";
         }
      }
      if (i < m.NumRows() - 1)
      {
         std::cout << "}, ";
      }
      else
      {
         std::cout << "}";
      }
   }
   std::cout << "}\n";
}

void print_vector(mfem::Vector v)
{
   std::cout << "{";
   for (int i = 0; i < v.Size(); i++)
   {
      std::cout << v(i);
      if (i < v.Size() - 1)
      {
         std::cout << ", ";
      }
   }
   std::cout << "}\n";
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

namespace mfem
{

struct FieldDescriptor
{
   std::variant<const FiniteElementSpace *, const ParFiniteElementSpace *> data;
   std::string label;
};

using mult_func_t = std::function<void(Vector &)>;

struct DofToQuadMaps
{
   DeviceTensor<2, const double> B;
   DeviceTensor<3, const double> G;
};

template <typename func_t, typename input_t,
          typename output_t>
struct ElementOperator;

template <typename func_t, typename... input_ts,
          typename... output_ts>
struct ElementOperator<func_t, std::tuple<input_ts...>,
          std::tuple<output_ts...>>
{
   func_t func;
   std::tuple<input_ts...> inputs;
   std::tuple<output_ts...> outputs;
   constexpr ElementOperator(func_t func,
                             std::tuple<input_ts...> inputs,
                             std::tuple<output_ts...> outputs)
      : func(func), inputs(inputs), outputs(outputs) {}
};

template <typename func_t, typename... input_ts,
          typename... output_ts>
ElementOperator(func_t, std::tuple<input_ts...>,
                std::tuple<output_ts...>)
-> ElementOperator<func_t, std::tuple<input_ts...>,
std::tuple<output_ts...>>;

template <class... T> constexpr bool always_false = false;

int GetVSize(const FieldDescriptor &f)
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
         return arg->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetVSize();
      }
      else
      {
         static_assert(always_false<T>, "can't use GetVSize on type");
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
      else
      {
         static_assert(always_false<T>, "can't use GetVDim on type");
      }
   }, f.data);
}

int GetDimension(const FieldDescriptor &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return arg->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const QuadratureSpace *>)
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
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return arg->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const QuadratureSpace *>)
      {
         return nullptr;
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
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return arg->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return arg->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const QuadratureSpace *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false<T>, "can't use GetElementRestriction on type");
      }
   }, f.data);
}

const DofToQuad *GetDofToQuad(const FieldDescriptor &f,
                              const IntegrationRule &ir,
                              DofToQuad::Mode mode)
{
   return std::visit([&ir, &mode](auto&& arg) -> const DofToQuad*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *>)
      {
         return &arg->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         return &arg->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const QuadratureSpace *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false<T>, "can't use GetDofToQuad on type");
      }
   }, f.data);
}

template <typename field_operator_t>
int GetSizeOnQP(const field_operator_t &fd, const FieldDescriptor &f)
{
   if constexpr (std::is_same_v<field_operator_t, Value>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_operator_t, Gradient>)
   {
      return GetVDim(f) * GetDimension(f);
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

}
