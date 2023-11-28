#include <algorithm>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>

#include <general/forall.hpp>
#include <linalg/tensor.hpp>
#include <mfem.hpp>

#include <enzyme/enzyme>

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

using namespace mfem;

void print_matrix(DenseMatrix m)
{
   out << "{";
   for (int i = 0; i < m.NumRows(); i++)
   {
      out << "{";
      for (int j = 0; j < m.NumCols(); j++)
      {
         out << m(i, j);
         if (j < m.NumCols() - 1)
         {
            out << ", ";
         }
      }
      if (i < m.NumRows() - 1)
      {
         out << "}, ";
      }
      else
      {
         out << "}";
      }
   }
   out << "} ";
}

void print_vector(Vector v)
{
   out << "{";
   for (int i = 0; i < v.Size(); i++)
   {
      out << v(i);
      if (i < v.Size() - 1)
      {
         out << ", ";
      }
   }
   out << "}";
   out << "\n";
}

using mfem::internal::tensor;
using mfem::internal::dual;

namespace AD
{
struct None {};
struct Enzyme {};
struct DualType {};
};

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

template <typename T>
struct always_false
{
   static constexpr bool value = false;
};

template<class>
inline constexpr bool always_false_v = false;

struct Independent {};
struct Dependent {};

struct DofToQuadTensors
{
   DeviceTensor<2, const double> B;
   DeviceTensor<3, const double> G;
};

struct Field
{
   std::variant<
   const QuadratureFunction *,
         const GridFunction *,
         const ParGridFunction *
         > data;

   std::string label;
};

struct DependentField
{
   std::string label;
};

const Vector &GetFieldData(Field &f)
{
   return *std::visit([](auto&& f) -> const Vector*
   {
      return static_cast<const Vector *>(f);
   }, f.data);
}

const Operator *GetProlongationFromField(const Field &f)
{
   return std::visit([](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetProlongationMatrix();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetProlongation on type");
      }
   }, f.data);
}

const Operator *GetElementRestriction(const Field &f, ElementDofOrdering o)
{
   return std::visit([&o](auto&& arg) -> const Operator*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetElementRestriction(o);
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetElementRestriction on type");
      }
   }, f.data);
}

const DofToQuad *GetDofToQuad(const Field &f, const IntegrationRule &ir,
                              DofToQuad::Mode mode)
{
   return std::visit([&ir, &mode](auto&& arg) -> const DofToQuad*
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return &arg->FESpace()->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return &arg->FESpace()->GetFE(0)->GetDofToQuad(ir, mode);
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return nullptr;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetDofToQuad on type");
      }
   }, f.data);
}

int GetVSize(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetVSize();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->Size();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetVSize on type");
      }
   }, f.data);
}

int GetTrueVSize(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetTrueVSize();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->Size();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetTrueVSize on type");
      }
   }, f.data);
}

int GetVDim(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetVDim();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return arg->GetVDim();
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetVDim on type");
      }
   }, f.data);
}

int GetDimension(const Field &f)
{
   return std::visit([](auto && arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const GridFunction *>)
      {
         return arg->FESpace()->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const ParGridFunction *>)
      {
         return arg->ParFESpace()->GetMesh()->Dimension();
      }
      else if constexpr (std::is_same_v<T, const QuadratureFunction *>)
      {
         return 1;
      }
      else
      {
         static_assert(always_false_v<T>, "can't use GetDimension on type");
      }
   }, f.data);
}

class FieldDescriptor
{
public:
   FieldDescriptor(std::string name) : name(name) {};

   std::string name;

   int size_on_qp = -1;

   int dim = -1;

   int vdim = -1;
};

class None : public FieldDescriptor
{
public:
   None(std::string name) : FieldDescriptor(name) {}
};

class Weight : public FieldDescriptor
{
public:
   Weight(std::string name) : FieldDescriptor(name) {};
};

class Value : public FieldDescriptor
{
public:
   Value(std::string name) : FieldDescriptor(name) {};
};

class Gradient : public FieldDescriptor
{
public:
   Gradient(std::string name) : FieldDescriptor(name) {};
};

class One : public FieldDescriptor
{
public:
   One(std::string name) : FieldDescriptor(name) {};
};

template <typename field_descriptor_type>
int GetSizeOnQP(const field_descriptor_type &fd, const Field &f)
{
   if constexpr (std::is_same_v<field_descriptor_type, Value>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_descriptor_type, Gradient>)
   {
      return GetVDim(f) * GetDimension(f);
   }
   else if constexpr (std::is_same_v<field_descriptor_type, None>)
   {
      return GetVDim(f);
   }
   else if constexpr (std::is_same_v<field_descriptor_type, One>)
   {
      return 1;
   }
   else
   {
      MFEM_ABORT("can't get size on quadrature point for field descriptor");
   }
}

template <typename quadrature_function_type, typename input_type,
          typename output_type>
struct ElementOperator;

template <typename quadrature_function_type, typename... input_types,
          typename... output_types>
struct ElementOperator<quadrature_function_type, std::tuple<input_types...>,
          std::tuple<output_types...>>
{
   quadrature_function_type func;
   std::tuple<input_types...> inputs;
   std::tuple<output_types...> outputs;
   constexpr ElementOperator(quadrature_function_type func,
                             std::tuple<input_types...> inputs,
                             std::tuple<output_types...> outputs)
      : func(func), inputs(inputs), outputs(outputs) {}
};

template <typename quadrature_function_type, typename... input_types,
          typename... output_types>
ElementOperator(quadrature_function_type, std::tuple<input_types...>,
                std::tuple<output_types...>)
-> ElementOperator<quadrature_function_type, std::tuple<input_types...>,
   std::tuple<output_types...>>;

void prolongation(std::vector<Field> fields, const Vector &x,
                  std::vector<Vector> &fields_local)
{
   int offset = 0;
   for (int i = 0; i < fields.size(); i++)
   {
      const auto P = GetProlongationFromField(fields[i]);
      if (P != nullptr)
      {
         const int width = P->Width();
         const Vector x_i(x.GetData() + offset, width);
         fields_local[i].SetSize(P->Height());

         P->Mult(x_i, fields_local[i]);
         offset += width;
      }
      else
      {
         const int width = GetTrueVSize(fields[i]);
         fields_local[i].SetSize(width);
         const Vector x_i(x.GetData() + offset, width);
         fields_local[i] = x_i;
         offset += width;
      }
   }
}

void element_restriction(std::vector<Field> fields,
                         const std::vector<Vector> &fields_local,
                         std::vector<Vector> &fields_e)
{
   for (int i = 0; i < fields.size(); i++)
   {
      const auto R = GetElementRestriction(fields[i], ElementDofOrdering::NATIVE);
      if (R != nullptr)
      {
         const int height = R->Height();
         fields_e[i].SetSize(height);
         R->Mult(fields_local[i], fields_e[i]);
      }
      else
      {
         const int height = GetTrueVSize(fields[i]);
         fields_e[i].SetSize(height);
         fields_e[i] = fields_local[i];
      }
   }
}

void element_restriction(std::vector<Field> fields,
                         std::vector<Vector> &fields_e,
                         int field_offset)
{
   for (int i = 0; i < fields.size(); i++)
   {
      const auto R = GetElementRestriction(fields[i], ElementDofOrdering::NATIVE);
      if (R != nullptr)
      {
         const int height = R->Height();
         fields_e[i + field_offset].SetSize(height);
         R->Mult(GetFieldData(fields[i]), fields_e[i + field_offset]);
      }
      else
      {
         fields_e[i + field_offset] = GetFieldData(fields[i]);
      }
   }
}

template <class F> struct FunctionSignature;

template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)>
{
   using return_type = output_type;
   using parameter_types = std::tuple<input_types...>;
};

template <class T> struct create_function_signature;

template <typename output_type, typename T, typename... input_types>
struct create_function_signature<output_type (T::*)(input_types...) const>
{
   using type = FunctionSignature<output_type(input_types...)>;
};

template <typename qf_args_t, std::size_t... Is>
auto create_enzyme_args(qf_args_t qf_args, qf_args_t &qf_shadow_args,
                        std::index_sequence<Is...>)
{
   // return std::tuple_cat(std::tie(enzyme_dup, std::get<Is>(qf_args),
   //                                std::get<Is>(qf_shadow_args))...);
   return std::tuple{enzyme::Duplicated<decltype(std::get<Is>(qf_args))>
                     (std::get<Is>(qf_args), std::get<Is>(qf_shadow_args))...};
}

template <typename qf_type, typename arg_type>
auto fwddiff_apply_enzyme(qf_type qf, arg_type &&args, arg_type &&shadow_args)
{
   auto arg_indices =
      std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<arg_type>>> {};
   auto enzyme_args = create_enzyme_args(args, shadow_args, arg_indices);

   return std::apply([&](auto &&...args)
   {
      using qf_return_type = typename create_function_signature<
                             decltype(&qf_type::operator())>::type::return_type;
#ifdef MFEM_USE_ENZYME
      return enzyme::get<0>
             (enzyme::autodiff<enzyme::Forward, enzyme::DuplicatedNoNeed<qf_return_type>>
              (+qf, args...));
#else
      return 0;
#endif
   },
   enzyme_args);
}

template <typename T1, typename T2>
void allocate_qf_arg(const T1&, T2&)
{
   static_assert(always_false<T1>::value,
                 "allocate_qf_arg not implemented for requested type combination");
}

template <typename interpolation_type>
void allocate_qf_arg(const interpolation_type &, double &)
{
   // no op
}

void allocate_qf_arg(const Value &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp);
}

void allocate_qf_arg(const Gradient &input, Vector &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

void allocate_qf_arg(const Gradient &input, DenseMatrix &arg)
{
   arg.SetSize(input.size_on_qp / input.vdim);
}

template <typename interpolation_type, typename T, int d1>
void allocate_qf_arg(const interpolation_type &, tensor<T, d1> &)
{
   // no op
}

template <typename interpolation_type, typename T, int d1, int d2>
void allocate_qf_arg(const interpolation_type &, tensor<T, d1, d2> &)
{
   // no op
}

template <typename qf_args, typename input_type, std::size_t... i>
void allocate_qf_args_impl(qf_args &args, input_type inputs,
                           std::index_sequence<i...>)
{
   (allocate_qf_arg(std::get<i>(inputs), std::get<i>(args)), ...);
}

template <typename qf_args, typename input_type>
void allocate_qf_args(qf_args &args, input_type inputs)
{
   allocate_qf_args_impl(args, inputs,
                         std::make_index_sequence<std::tuple_size_v<qf_args>> {});
}

// This can also be implemented by a user who wants exotic types in their
// quadrature functions
void prepare_qf_arg(const DeviceTensor<1> &u, double &arg) { arg = u(0); }

void prepare_qf_arg(const DeviceTensor<1> &u, Vector &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg[i] = u(i);
   }
}

void prepare_qf_arg(const DeviceTensor<1> &u, DenseMatrix &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg.Data()[i] = u(i);
   }
}

template <typename T, int length>
void prepare_qf_arg(const DeviceTensor<1> &u, tensor<T, length> &arg)
{
   for (int i = 0; i < u.GetShape()[0]; i++)
   {
      arg(i) = u(i);
   }
}

template <typename T, int dim, int vdim>
void prepare_qf_arg(const DeviceTensor<1> &u, tensor<T, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i) = u((i * vdim) + j);
      }
   }
}

template <typename arg_type>
void prepare_qf_arg(const DeviceTensor<2> &u, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&u(0, qp), u.GetShape()[0]);
   prepare_qf_arg(u_qp, arg);
}

template <typename qf_type, typename qf_args, std::size_t... i>
void prepare_qf_args(const qf_type &qf, std::vector<DeviceTensor<2>> &u,
                     qf_args &args, int qp, std::index_sequence<i...>)
{
   // we have several options here
   // - reinterpret_cast
   // - memcpy (copy data of u -> arg with overloading operator= for example)
   (prepare_qf_arg(u[i], std::get<i>(args), qp), ...);
}

Vector prepare_qf_result(double x)
{
   Vector r(1);
   r = x;
   return r;
}

Vector prepare_qf_result(Vector x) { return x; }

template <int length>
Vector prepare_qf_result(tensor<double, length> x)
{
   Vector r(length);
   for (size_t i = 0; i < length; i++)
   {
      r(i) = x(i);
   }
   return r;
}

Vector prepare_qf_result(tensor<double, 2, 2> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = x(j, i);
      }
   }
   return r;
}

Vector prepare_qf_result(tensor<dual<double, double>, 2, 2> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = get_value(x(j, i));
      }
   }
   return r;
}

template <typename T>
Vector prepare_qf_result(T)
{
   static_assert(always_false<T>::value,
                 "prepare_qf_result not implemented for result type");
}

template <typename T>
Vector prepare_qf_result_dualtype_gradient(T)
{
   static_assert(always_false<T>::value,
                 "prepare_qf_result_dualtype_gradient not implemented for result type");
}

Vector prepare_qf_result_dualtype_gradient(tensor<dual<double, double>, 2, 2> x)
{
   Vector r(4);
   for (size_t i = 0; i < 2; i++)
   {
      for (size_t j = 0; j < 2; j++)
      {
         // TODO: Careful with the indices here!
         r(j + (i * 2)) = get_gradient(x(j, i));
      }
   }
   return r;
}

template <typename qf_type, typename qf_args>
auto apply_qf(const qf_type &qf, qf_args &args, std::vector<DeviceTensor<2>> &u,
              int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result(std::apply(qf, args));
}

template <typename qf_type, typename qf_args>
auto apply_qf_fwddiff_enzyme(const qf_type &qf,
                             qf_args &args,
                             std::vector<DeviceTensor<2>> &u,
                             qf_args &shadow_args,
                             std::vector<DeviceTensor<2>> &v,
                             int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   prepare_qf_args(qf, v, shadow_args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result(fwddiff_apply_enzyme(qf, args, shadow_args));
}

template <typename T>
void prepare_qf_arg_dual(const DeviceTensor<1> &, T &)
{
   // noop
}

template <int dim, int vdim>
void prepare_qf_arg_dual(const DeviceTensor<1> &v,
                         tensor<dual<double, double>, dim, vdim> &arg)
{
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         arg(j, i).gradient = v((i * vdim) + j);
      }
   }
}

template <typename arg_type>
void prepare_qf_arg_dual(const DeviceTensor<2> &v, arg_type &arg, int qp)
{
   const auto u_qp = Reshape(&v(0, qp), v.GetShape()[0]);
   prepare_qf_arg_dual(u_qp, arg);
}

template <typename qf_type, typename qf_args, std::size_t... i>
void prepare_qf_args_dual(const qf_type &qf, std::vector<DeviceTensor<2>> &v,
                          qf_args &args, int qp, std::index_sequence<i...>)
{
   // we have several options here
   // - reinterpret_cast
   // - memcpy (copy data of u -> arg with overloading operator= for example)
   (prepare_qf_arg_dual(v[i], std::get<i>(args), qp), ...);
}

template <typename qf_type, typename qf_args>
auto apply_qf_fwddiff_dualtype(const qf_type &qf,
                               qf_args &args,
                               std::vector<DeviceTensor<2>> &u,
                               std::vector<DeviceTensor<2>> &v,
                               int qp)
{
   prepare_qf_args(qf, u, args, qp,
                   std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   prepare_qf_args_dual(qf, v, args, qp,
                        std::make_index_sequence<std::tuple_size_v<qf_args>> {});

   return prepare_qf_result_dualtype_gradient(std::apply(qf, args));
}

template <typename input_type>
void map_field_to_quadrature_data(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadTensors &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights)
{
   if constexpr (std::is_same_v<
                 typename std::remove_reference<decltype(input)>::type,
                 Value>)
   {
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

      for (int vd = 0; vd < vdim; vd++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            double acc = 0.0;
            for (int dof = 0; dof < num_dof; dof++)
            {
               acc += B(qp, dof) * field(dof, vd);
            }
            field_qp(vd, qp) = acc;
         }
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      Gradient>)
   {
      const auto G(dtqmaps.G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = input.vdim;
      const int element_offset = element_idx * num_dof * vdim;
      const auto field = Reshape(field_e.Read() + element_offset, num_dof, vdim);

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
   // TODO: Create separate function for clarity
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      Weight>)
   {
      const int num_qp = integration_weights.GetShape()[0];
      auto f = Reshape(&field_qp[0], num_qp);
      for (int qp = 0; qp < num_qp; qp++)
      {
         f(qp) = integration_weights(qp);
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(input)>::type,
                      None>)
   {
      const auto B(dtqmaps.B);
      auto [num_qp, num_dof] = B.GetShape();
      const int size_on_qp = input.size_on_qp;
      const int element_offset = element_idx * size_on_qp * num_qp;
      const auto field = Reshape(field_e.Read() + element_offset,
                                 size_on_qp * num_qp);
      auto f = Reshape(&field_qp[0], size_on_qp * num_qp);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         f(i) = field(i);
      }
   }
}

template <int num_qfinputs, typename input_type, std::size_t... i>
void map_fields_to_quadrature_data(
   std::vector<DeviceTensor<2>> &fields_qp, int element_idx,
   const std::vector<Vector> &fields_e,
   std::array<int, num_qfinputs> qfinput_to_field,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   input_type &qfinputs,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data(fields_qp[i], element_idx,
                                 dtqmaps[qfinput_to_field[i]], fields_e[qfinput_to_field[i]],
                                 std::get<i>(qfinputs), integration_weights),
    ...);
}

template <typename input_type>
void map_field_to_quadrature_data_conditional(
   DeviceTensor<2> field_qp, int element_idx, DofToQuadTensors &dtqmaps,
   const Vector &field_e, input_type &input,
   DeviceTensor<1, const double> integration_weights,
   const bool condition)
{
   if (condition)
   {
      map_field_to_quadrature_data(field_qp, element_idx, dtqmaps, field_e, input,
                                   integration_weights);
   }
}

template <int num_qfinputs, typename input_type, std::size_t... i>
void map_fields_to_quadrature_data_conditional(
   std::vector<DeviceTensor<2>> &fields_qp, int element_idx,
   const std::vector<Vector> &fields_e,
   const int field_idx,
   std::vector<DofToQuadTensors> &dtqmaps,
   DeviceTensor<1, const double> integration_weights,
   std::array<bool, num_qfinputs> conditions,
   input_type &qfinputs,
   std::index_sequence<i...>)
{
   (map_field_to_quadrature_data_conditional(fields_qp[i], element_idx,
                                             dtqmaps[field_idx], fields_e[field_idx],
                                             std::get<i>(qfinputs), integration_weights, conditions[i]),
    ...);
}

template <typename output_type>
void map_quadrature_data_to_fields(Vector &y_e, int element_idx, int num_el,
                                   DeviceTensor<3, double> residual_qp,
                                   output_type output,
                                   std::vector<DofToQuadTensors> &dtqmaps,
                                   int test_space_field_idx)
{
   // assuming the quadrature point residual has to "play nice with
   // the test function"
   if constexpr (std::is_same_v<
                 typename std::remove_reference<decltype(output)>::type,
                 Value>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int vdim = output.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), vdim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               // |JxW| is assumed to be in C
               acc += B(qp, dof) * C(vd, qp);
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(output)>::type, Gradient>)
   {
      const auto G(dtqmaps[test_space_field_idx].G);
      const auto [num_qp, dim, num_dof] = G.GetShape();
      const int vdim = output.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), vdim, dim, num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_dof, vdim, num_el);
      for (int dof = 0; dof < num_dof; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double acc = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  acc += G(qp, d, dof) * C(vd, d, qp);
               }
            }
            y(dof, vd, element_idx) += acc;
         }
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(output)>::type, None>)
   {
      const auto B(dtqmaps[test_space_field_idx].B);
      const auto [num_qp, num_dof] = B.GetShape();
      const int size_on_qp = output.vdim;
      auto C = Reshape(&residual_qp(0, 0, element_idx), size_on_qp * num_qp);
      auto y = Reshape(y_e.ReadWrite(), size_on_qp * num_qp, num_el);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         y(i, element_idx) += C(i);
      }
   }
   else if constexpr (std::is_same_v<typename std::remove_reference<
                      decltype(output)>::type, One>)
   {
      // This is the "integral over all quadrature points type" applying
      // B = 1 s.t. B^T * C \in R^1.
      auto [size_on_qp, num_qp, num_el] = residual_qp.GetShape();
      auto C = Reshape(&residual_qp(0, 0, element_idx), size_on_qp * num_qp);
      auto y = Reshape(y_e.ReadWrite(), num_el);
      for (int i = 0; i < size_on_qp * num_qp; i++)
      {
         y(element_idx) += C(i);
      }
   }
   else
   {
      MFEM_ABORT("quadrature data mapping to field is not implemented for"
                 " this field descriptor");
   }
}

std::vector<Field>::const_iterator find_name(const std::vector<Field> &fields,
                                             const std::string &input_name)
{
   auto it = std::find_if(fields.begin(), fields.end(), [&](const Field &field)
   {
      return field.label == input_name;
   });

   return it;
}

int find_name_idx(const std::vector<Field> &fields,
                  const std::string &input_name)
{
   std::vector<Field>::const_iterator it = find_name(fields, input_name);
   if (it == fields.end())
   {
      return -1;
   }
   return (it - fields.begin());
}

template <size_t num_qfinputs, typename input_type, std::size_t... i>
void map_qfinput_to_field(std::vector<Field> &fields,
                          std::array<int, num_qfinputs> &map, input_type &inputs,
                          std::index_sequence<i...>)
{
   auto f = [&](auto &input, auto &map)
   {
      int idx;
      if constexpr (std::is_same_v<
                    typename std::remove_reference<decltype(input)>::type,
                    Weight>)
      {
         input.dim = 1;
         input.vdim = 1;
         input.size_on_qp = 1;
         map = -1;
      }
      else if ((idx = find_name_idx(fields, input.name)) != -1)
      {
         input.dim = GetDimension(fields[idx]);
         input.vdim = GetVDim(fields[idx]);
         input.size_on_qp = GetSizeOnQP(input, fields[idx]);
         map = idx;
      }
      else
      {
         MFEM_ABORT("can't find field for " << input.name);
      }
   };

   (f(std::get<i>(inputs), map[i]), ...);
}

std::vector<DofToQuadTensors> map_dtqmaps(std::vector<const DofToQuad*>
                                          dtqmaps, int dim, int num_qp)
{
   std::vector<DofToQuadTensors> dtqmaps_tensor;
   for (auto m : dtqmaps)
   {
      if (m != nullptr)
      {
         dtqmaps_tensor.push_back(
         {
            Reshape(m->B.Read(), m->nqpt, m->ndof),
            Reshape(m->G.Read(), m->nqpt, dim, m->ndof)
         });
      }
      else
      {
         // If there is no DofToQuad map available, we assume that the "map"
         // is identity and therefore maps 1 to 1 from #qp to #qp.
         DeviceTensor<2, const double> B(nullptr, num_qp, num_qp);
         DeviceTensor<3, const double> G(nullptr, num_qp, dim, num_qp);
         dtqmaps_tensor.push_back({B, G});
      }
   }
   return dtqmaps_tensor;
}

template <typename input_type, std::size_t... i>
void map_inputs_to_memory(std::vector<DeviceTensor<2>> &fields_qp,
                          std::vector<Vector> &fields_qp_mem, int num_qp,
                          input_type &inputs, std::index_sequence<i...>)
{
   auto f = [&](auto &input, auto &field_qp_mem)
   {
      fields_qp.emplace_back(
         DeviceTensor<2>(field_qp_mem.ReadWrite(), input.size_on_qp, num_qp));
   };

   (f(std::get<i>(inputs), fields_qp_mem[i]), ...);
}

class DifferentiableForm : public Operator
{
public:
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(DifferentiableForm &op) : Operator(op.Height()), op(op) {}

      void Mult(const Vector &x, Vector &y) const override
      {
         op.JacobianMult(x, y);
      }

   protected:
      DifferentiableForm &op;
   };

   DifferentiableForm(std::vector<Field> solutions,
                      std::vector<Field> parameters,
                      std::vector<Field> dependent_fields,
                      ParMesh &mesh) :
      solutions(solutions),
      parameters(parameters),
      dependent_fields(dependent_fields),
      mesh(mesh)
   {
      dim = mesh.Dimension();
      if (solutions.size() > 1)
      {
         MFEM_ABORT("only one trial space allowed at the moment, tried with " <<
                    solutions.size());
      }
      fields.insert(fields.end(), solutions.begin(), solutions.end());
      fields.insert(fields.end(), parameters.begin(), parameters.end());

      solutions_local.resize(solutions.size());
      parameters_local.resize(parameters.size());
      directions_local.resize(solutions.size());
      fields_e.resize(solutions.size() + parameters.size());
      directions_e.resize(solutions.size());

      // TODO: Currently this only supports one dependent variable. Could be used for second derivatives?
      //
      // This index is the field that the residual is differentiated wrt.
      if (dependent_fields.empty())
      {

      }
      else if ((dependent_field_idx = find_name_idx(fields,
                                                    dependent_fields[0].label)) != -1)
      {
         out << "dependent input is field idx " << dependent_field_idx << " -> " <<
             fields[dependent_field_idx].label << "\n";
      }
   }

   template <
      typename ad_method = AD::Enzyme,
      typename qf_type,
      typename input_type,
      typename output_type
      >
   void AddElementOperator(ElementOperator<qf_type, input_type, output_type> &qf,
                           const IntegrationRule &ir)
   {
#ifndef MFEM_USE_ENZYME
      static_assert(!std::is_same<ad_method, AD::Enzyme>(),
                    "AD configuration needs MFEM to be compiled with LLVM/Enzyme");
#endif
      constexpr ElementDofOrdering element_dof_ordering = ElementDofOrdering::NATIVE;
      constexpr size_t num_qfinputs = std::tuple_size_v<input_type>;
      constexpr size_t num_qfoutputs = std::tuple_size_v<output_type>;

      if (num_qfoutputs != 1)
      {
         MFEM_ABORT("only one test space allowed at the moment");
      }

      // bool contains_dependent_input = std::apply([&](auto ...input)
      // {
      //    auto check_dependency = [&](auto &input)
      //    {
      //       return solutions.size() &&
      //              (input.name == solutions[dependent_field_idx].label);
      //    };
      //    return (check_dependency(input) || ...);
      // },
      // qf.inputs);


      const int num_el = mesh.GetNE();
      const int num_qp = ir.GetNPoints();

      std::cout << "adding element operator with " << num_qfinputs << " inputs and "
                << num_qfoutputs << " outputs" << "\n";

      int test_space_field_idx;
      int residual_size_on_qp;

      auto output_fd = std::get<0>(qf.outputs);
      if ((test_space_field_idx = find_name_idx(fields, output_fd.name)) != -1)
      {
         residual_size_on_qp =
            GetSizeOnQP(output_fd, fields[test_space_field_idx]);
         output_fd.dim = GetDimension(fields[test_space_field_idx]);
         output_fd.vdim = GetVDim(fields[test_space_field_idx]);
         output_fd.size_on_qp = residual_size_on_qp;
      }
      else if constexpr (std::is_same_v<decltype(output_fd), One>)
      {
         residual_size_on_qp = 1;
         output_fd.dim = 1;
         output_fd.vdim = 1;
         output_fd.size_on_qp = 1;
      }
      else
      {
         MFEM_ABORT("can't figure out residual size on quadrature point level");
      }

      std::array<int, num_qfinputs> qfinput_to_field;
      map_qfinput_to_field(fields, qfinput_to_field, qf.inputs,
                           std::make_index_sequence<num_qfinputs> {});

      // All solutions T-vector sizes make up the height of the operator, since
      // they are explicitly provided in Mult() for example.
      this->height = 0;
      int residual_local_size = 0;
      for (auto &f : solutions)
      {
         this->height += GetTrueVSize(f);
         residual_local_size += GetVSize(f);
      }
      residual_local.SetSize(residual_local_size);

      // TODO: Only works for one test space
      if constexpr (std::is_same_v<decltype(output_fd), None>)
      {
         this->width = residual_size_on_qp * num_qp * num_el;
      }
      else if constexpr (std::is_same_v<decltype(output_fd), One>)
      {
         this->width = 1;
      }
      else
      {
         this->width = this->height;
      }

      // Creating this here allows to call GradientMult even if there are no
      // dependent ElementOperators.
      jacobian_op.reset(new JacobianOperator(*this));

      // Allocate memory for fields on quadrature points
      std::vector<Vector> fields_qp_mem;
      std::apply(
         [&](auto &&...input)
      {
         (fields_qp_mem.emplace_back(
             Vector(input.size_on_qp * num_qp * num_el)),
          ...);
      },
      qf.inputs);

      using qf_args = typename create_function_signature<
                      decltype(&qf_type::operator())>::type::parameter_types;

      // This tuple contains objects of every ElementOperator::func function
      // parameter which might have to be resized.
      qf_args args{};
      allocate_qf_args(args, qf.inputs);

      Array<double> integration_weights_mem = ir.GetWeights();

      // Duplicate B/G and assume only a single element type for now
      std::vector<const DofToQuad*> dtqmaps;
      for (const auto &field : fields)
      {
         dtqmaps.emplace_back(GetDofToQuad(field, ir, DofToQuad::FULL));
      }

      Vector residual_qp_mem(residual_size_on_qp * num_qp * num_el);

      // TODO: Replace with GetVSize?
      if (test_space_field_idx != -1 && (dtqmaps[test_space_field_idx] != nullptr))
      {
         // FieldDescriptor::Value, Gradient etc.
         residual_e.SetSize(dtqmaps[test_space_field_idx]->ndof * GetVDim(
                               fields[test_space_field_idx]) * num_el);
      }
      else if (test_space_field_idx != -1)
      {
         // FieldDescriptor::None
         residual_e.SetSize(ir.GetNPoints() * GetVDim(fields[test_space_field_idx]) *
                            num_el);
      }
      else
      {
         // FieldDescriptor::One
         residual_e.SetSize(num_el);
      }

      residual_integrators.emplace_back(
         [&, args, qfinput_to_field, fields_qp_mem, residual_qp_mem,
             residual_size_on_qp, test_space_field_idx, output_fd, dtqmaps,
             integration_weights_mem, num_qp, num_el](Vector &y_e) mutable
      {
         auto dtqmaps_tensor = map_dtqmaps(dtqmaps, dim, num_qp);

         const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                          residual_size_on_qp, num_qp, num_el);

         // Fields interpolated to the quadrature points in the order of
         // quadrature function arguments
         std::vector<DeviceTensor<2>> fields_qp;
         map_inputs_to_memory(
         fields_qp, fields_qp_mem, num_qp,qf.inputs, std::make_index_sequence<num_qfinputs>{});

         DeviceTensor<1, const double> integration_weights(
            integration_weights_mem.Read(), num_qp);

         for (int el = 0; el < num_el; el++)
         {
            // B
            // prepare fields on quadrature points
            map_fields_to_quadrature_data<num_qfinputs>(
               fields_qp, el, fields_e,
               qfinput_to_field, dtqmaps_tensor,
            integration_weights, qf.inputs, std::make_index_sequence<num_qfinputs> {});

            for (int qp = 0; qp < num_qp; qp++)
            {
               auto f_qp = apply_qf(qf.func, args, fields_qp, qp);

               auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);
               for (int i = 0; i < residual_size_on_qp; i++)
               {
                  r_qp(i) = f_qp(i);
               }
            }

            // B^T
            map_quadrature_data_to_fields(y_e, el, num_el, residual_qp,
                                          output_fd, dtqmaps_tensor,
                                          test_space_field_idx);
         }
      });

      if (test_space_field_idx != -1)
      {
         auto R = GetElementRestriction(fields[test_space_field_idx],
                                        element_dof_ordering);
         if (R == nullptr)
         {
            out << "G^T = Identity" << "\n";
            element_restriction_transpose = [](Vector &r_e, Vector &y)
            {
               y = r_e;
            };
         }
         else
         {
            element_restriction_transpose = [R](Vector &r_e, Vector &y)
            {
               R->MultTranspose(r_e, y);
            };
         }

         auto P = GetProlongationFromField(fields[test_space_field_idx]);
         if (P == nullptr)
         {
            out << "P^T = Identity" << "\n";
            prolongation_transpose = [](Vector &r_local, Vector &y)
            {
               y = r_local;
            };
         }
         else
         {
            prolongation_transpose = [P](Vector &r_local, Vector &y)
            {
               P->MultTranspose(r_local, y);
            };
         }
      }
      else
      {
         element_restriction_transpose = [](Vector &r_e, Vector &y)
         {
            y = r_e;
         };

         prolongation_transpose = [&](Vector &r_local, Vector &y)
         {
            double local_sum = r_local.Sum();
            MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
            MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
         };
      }

      if constexpr (std::is_same<ad_method, AD::None> {})
      {
         return;
      }
      else
      {
         // Allocate memory for directions on quadrature points
         std::vector<Vector> directions_qp_mem;
         std::apply(
            [&](auto &&...input)
         {
            (directions_qp_mem.emplace_back(
                Vector(input.size_on_qp * num_qp * num_el)),
             ...);
         },
         qf.inputs);

         for (auto &v : directions_qp_mem)
         {
            v = 0.0;
         }

         qf_args shadow_args{};
         allocate_qf_args(shadow_args, qf.inputs);

         jvp_integrators.emplace_back(
            [&, args, shadow_args, qfinput_to_field,
                fields_qp_mem,
                directions_qp_mem, residual_qp_mem,
                residual_size_on_qp,
                test_space_field_idx, output_fd, dtqmaps,
                integration_weights_mem, num_qp,
                num_el](Vector &y_e) mutable
         {
            // Check which qf inputs are dependent on the dependent variable
            std::array<bool, num_qfinputs> qfinput_is_dependent;
            bool no_qfinput_is_dependent = true;
            for (int i = 0; i < qfinput_is_dependent.size(); i++)
            {
               if (qfinput_to_field[i] == dependent_field_idx)
               {
                  no_qfinput_is_dependent = false;
                  qfinput_is_dependent[i] = true;
                  // out << "function input " << i << " is dependent on "
                  //     << fields[qfinput_to_field[i]].label << "\n";
               }
               else
               {
                  qfinput_is_dependent[i] = false;
               }
            }

            if (no_qfinput_is_dependent)
            {
               return;
            }

            auto dtqmaps_tensor = map_dtqmaps(dtqmaps, dim, num_qp);

            const auto residual_qp = Reshape(residual_qp_mem.ReadWrite(),
                                             residual_size_on_qp, num_qp, num_el);

            // Fields interpolated to the quadrature points in the order of quadrature
            // function arguments
            std::vector<DeviceTensor<2>> fields_qp;
            map_inputs_to_memory(fields_qp, fields_qp_mem, num_qp, qf.inputs,
                                 std::make_index_sequence<num_qfinputs>{});

            std::vector<DeviceTensor<2>> directions_qp;
            map_inputs_to_memory(directions_qp, directions_qp_mem, num_qp, qf.inputs,
                                 std::make_index_sequence<num_qfinputs>{});

            DeviceTensor<1, const double> integration_weights(
               integration_weights_mem.Read(), num_qp);

            for (int el = 0; el < num_el; el++)
            {
               // B
               // prepare fields on quadrature points
               map_fields_to_quadrature_data<num_qfinputs>(
                  fields_qp, el, fields_e, qfinput_to_field, dtqmaps_tensor,
                  integration_weights, qf.inputs,
                  std::make_index_sequence<num_qfinputs> {});

               // prepare directions (shadow memory)
               map_fields_to_quadrature_data_conditional<num_qfinputs>(
                  directions_qp, el,
                  directions_e, dependent_field_idx,
                  dtqmaps_tensor,
                  integration_weights,
                  qfinput_is_dependent,
                  qf.inputs,
                  std::make_index_sequence<num_qfinputs> {});

               // D -> D
               for (int qp = 0; qp < num_qp; qp++)
               {
                  Vector f_qp;
                  auto r_qp = Reshape(&residual_qp(0, qp, el), residual_size_on_qp);
                  if constexpr (std::is_same<ad_method, AD::Enzyme> {})
                  {
                     f_qp = apply_qf_fwddiff_enzyme(qf.func, args, fields_qp, shadow_args,
                                                    directions_qp, qp);
                  }
                  else if constexpr (std::is_same<ad_method, AD::DualType> {})
                  {
                     f_qp = apply_qf_fwddiff_dualtype(qf.func, args, fields_qp,
                                                      directions_qp, qp);
                  }
                  for (int i = 0; i < residual_size_on_qp; i++)
                  {
                     r_qp(i) = f_qp(i);
                  }
               }

               // B^T
               map_quadrature_data_to_fields(y_e, el, num_el, residual_qp, output_fd,
                                             dtqmaps_tensor, test_space_field_idx);
            }
         });
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(residual_integrators.size(), "form does not contain any operators");

      // P
      prolongation(solutions, x, solutions_local);

      // G
      element_restriction(solutions, solutions_local, fields_e);
      element_restriction(parameters, fields_e, solutions.size());

      // BEGIN GPU
      // B^T D B x
      residual_e = 0.0;
      for (int i = 0; i < residual_integrators.size(); i++)
      {
         residual_integrators[i](residual_e);
      }
      // END GPU

      // G^T
      element_restriction_transpose(residual_e, residual_local);

      // P^T
      prolongation_transpose(residual_local, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   }

   void SetEssentialTrueDofs(const Array<int> &l) { l.Copy(ess_tdof_list); }

   void JacobianMult(const Vector &x, Vector &y) const
   {
      // apply essential bcs
      current_direction_t = x;
      current_direction_t.SetSubVector(ess_tdof_list, 0.0);

      prolongation(solutions, current_direction_t, directions_local);
      prolongation(solutions, current_state_t, solutions_local);

      element_restriction(solutions, directions_local, directions_e);
      element_restriction(solutions, solutions_local, fields_e);
      element_restriction(parameters, fields_e, solutions.size());

      // BEGIN GPU
      // B^T D B x
      Vector &jvp_e = residual_e;

      jvp_e = 0.0;
      for (int i = 0; i < jvp_integrators.size(); i++)
      {
         jvp_integrators[i](jvp_e);
      }
      // END GPU

      // G^T
      element_restriction_transpose(jvp_e, residual_local);

      // P^T
      prolongation_transpose(residual_local, y);

      // re-assign the essential degrees of freedom on the final output vector.
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         y[ess_tdof_list[i]] = x[ess_tdof_list[i]];
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      current_state_t = x;
      return *jacobian_op;
   }

   std::vector<Field> solutions;
   std::vector<Field> parameters;
   std::vector<Field> dependent_fields;
   ParMesh &mesh;
   int dim;

   // solutions and parameters
   std::vector<Field> fields;

   std::vector<std::function<void(Vector &)>> residual_integrators;
   std::vector<std::function<void(Vector &)>> jvp_integrators;

   std::function<void(Vector &, Vector &)> element_restriction_transpose;
   std::function<void(Vector &, Vector &)> prolongation_transpose;

   mutable std::vector<Vector> solutions_local;
   mutable std::vector<Vector> parameters_local;
   mutable std::vector<Vector> directions_local;
   mutable Vector residual_local;

   mutable std::vector<Vector> fields_e;
   mutable std::vector<Vector> directions_e;
   mutable Vector residual_e;

   std::shared_ptr<JacobianOperator> jacobian_op;
   mutable Vector current_direction_t, current_state_t;

   Array<int> ess_tdof_list;

   int dependent_field_idx = -1;
};
