#pragma once

#include <string_view>

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
# error Unsupported compiler
#endif

   const auto start = function.find(prefix) + prefix.size();
   const auto end = function.find(suffix);
   const auto size = end - start;

   return function.substr(start, size);
}

#include <mfem.hpp>
#include <linalg/dtensor.hpp>
#include "linalg/tensor.hpp"
#include <linalg/kernels.hpp>

#include "eigen-3.4.0/Eigen/Eigen"

using namespace mfem;

template <
   typename type,
   int rank = 1 >
class NDArray : public Array<type>
{
};

// template < typename return_type, typename ... parameter_types, typename ... arg_types >
// auto forall(std::function< return_type(parameter_types ...) > f, arg_types ... args) {
//   uint64_t leading_dimensions[] = {args.shape[0] ...};
//   int n = leading_dimensions[0];
//   auto output = make_ndarray(n, return_type{});
//   for (int i = 0; i < n; i++) {
//     reinterpret_cast< return_type * >(output.begin())[i] = f(
//      reinterpret_cast< parameter_types * >(args.begin())[i] ...
//     );
//   };
//   return output;
// }


// template <
//    typename return_type,
//    typename ... parameter_types,
//    typename ... arg_types >
// auto forall(std::function<return_type(parameter_types ...)> f,
//             const int n, arg_types ... args)
// {
//    // int leading_dimensions[] = {args.Size() ...};
//    // const int n = leading_dimensions[0];

//    Vector output(n);
//    for (int i = 0; i < n; i++)
//    {
//       reinterpret_cast< return_type * >(output.begin())[i] =
//          f(reinterpret_cast< parameter_types * >(args.begin())[i]...);
//    };
//    return output;
// }

template <class F>
struct FunctionSignatureOld;

template <typename output_type, typename T, typename... input_types>
struct FunctionSignatureOld<output_type (T::*)(input_types...) const>
{
   using input_t = std::tuple<input_types...>;
   using output_t = output_type;
};

template <class F>
struct FunctionSignature;

template < typename output_type, typename ... input_types >
struct FunctionSignature< output_type(input_types ... ) >
{
   using return_type = output_type;
   using parameter_types = std::tuple< input_types ... >;
};

////////////////
template <class T>
struct create_function_signature;

template <
   typename output_type,
   typename T,
   typename... input_types >
struct create_function_signature<output_type (T::*)(input_types...) const>
{
   using type = FunctionSignature<output_type( input_types ... )>;
};

template <
   typename function_type,
   typename ... input_types,
   typename output_type,
   typename ... arg_types >
auto forall_impl(FunctionSignature<output_type( input_types ... )>,
                 const function_type& f,
                 const int n,
                 arg_types& ... args)
{
   Array<output_type> output(n);
   for (int i = 0; i < n; i++)
      // MFEM_FORALL(....)
   {
      reinterpret_cast< output_type * >(output.begin())[i] =
         f(reinterpret_cast< input_types * >(args.begin())[i]...);
   };
   return output;
}

template <
   typename function_type,
   typename ... arg_types >
auto forall(const function_type& f, const int n, arg_types& ... args)
{
   using function_signature_type = typename
                                   create_function_signature<decltype(&function_type::operator())>::type;
   return forall_impl(function_signature_type{}, f, n, args...);
}

// template <
//    typename output_type,
//    typename ... input_types >
// auto fwddiff(output_type (*f)(input_types ...))
// {
//    return [f](input_types ...
//               args /*, input_types ... args for the duals is missing here */)
//    {
//       return __enzyme_fwddiff< output_type > (f, enzyme_dup, args...);
//    };
// }

// template <
//    typename output_type,
//    typename input_type >
// auto fwddiff(output_type (*f)(input_type &))
// {
//    return [f](input_type arg1, input_type arg2)
//    {
//       return __enzyme_fwddiff< output_type > (f, enzyme_dup, &arg1, &arg2);
//    };
// }

template <
   typename output_type,
   typename ... input_types >
auto fwddiff(output_type (*f)(input_types...))
{
   return [f](input_types... args, input_types... args2)
   {
      auto input_types_tuple = std::tuple<input_types...>(args...);
      auto shadow_input_types_tuple = std::tuple<input_types...>(args2...);

      static_assert(
         std::is_same_v<decltype(input_types_tuple), decltype(shadow_input_types_tuple)>,
         "input and shadow not equal");

      // auto concatenated_types_tuple = std::tuple_cat(input_types_tuple,
      //                                                shadow_input_types_tuple);

      // std::cout << get_type_name<decltype(concatenated_types_tuple)>() << std::endl;

      return __enzyme_fwddiff< output_type > (f, &args..., &args2...);
   };
}

// template <
//    typename output_type,
//    typename ... parameter_types,
//    typename ... arg_types >
// auto forall(std::function<output_type(parameter_types ...)> f, const int n,
//             arg_types& ... args)
// {
//    Array<double> output(n * sizeof(output_type) / sizeof(double));
//    for (int i = 0; i < n; i++)
//    {
//       reinterpret_cast< output_type * >(output.begin())[i] =
//          f(reinterpret_cast< parameter_types * >(args.begin())[i] ...);
//    };
//    return output;
// }
////////////////

/// @return u_qp qp x vdim x elements
Vector interpolate(const GridFunction& u, const IntegrationRule& ir)
{
   auto fes = u.FESpace();
   auto B = fes->GetQuadratureInterpolator(ir);
   B->SetOutputLayout(QVectorLayout::byVDIM);
   B->DisableTensorProducts();

   auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);
   Vector u_el(R->Height());
   R->Mult(u, u_el);

   Vector u_qp(
      fes->GetVDim() *
      fes->GetMesh()->SpaceDimension() *
      fes->GetMesh()->GetNE() *
      ir.GetNPoints());

   B->Values(u_el, u_qp);

   return u_qp;
}

Vector gradient_wrt_x(const GridFunction& u, const IntegrationRule& ir)
{
   auto fes = u.FESpace();
   auto B = fes->GetQuadratureInterpolator(ir);
   B->SetOutputLayout(QVectorLayout::byVDIM);
   B->DisableTensorProducts();

   auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);
   Vector u_el(R->Height());
   R->Mult(u, u_el);

   Vector grad_u_qp(
      fes->GetVDim() *
      fes->GetMesh()->SpaceDimension() *
      fes->GetMesh()->GetNE() *
      ir.GetNPoints());

   B->PhysDerivatives(u_el, grad_u_qp);

   return grad_u_qp;
}

Vector integrate(Vector& s_qp, const FiniteElementSpace &fes,
                 const IntegrationRule& ir)
{
   auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);

   auto mesh = fes.GetMesh();
   const int dim = mesh->SpaceDimension();
   const int num_el = mesh->GetNE();
   const int vdim = fes.GetVDim();
   const int num_qp = ir.GetNPoints();
   const int num_vdofs = R->Height() / num_el;
   const int num_dofs = num_vdofs / vdim;

   const GeometricFactors *geom = mesh->GetGeometricFactors(
                                     ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   Vector yi_el(R->Height());
   auto Yi = Reshape(yi_el.Write(), num_dofs, vdim, num_el);

   auto C = Reshape(s_qp.ReadWrite(), vdim, num_qp, num_el);
   auto detJ = Reshape(geom->detJ.Read(), num_qp, num_el);

   for (int e = 0; e < num_el; e++)
   {
      const DofToQuad &maps = fes.GetFE(e)->GetDofToQuad(ir, DofToQuad::FULL);
      const auto Bt = Reshape(maps.Bt.Read(), num_dofs, num_qp);

      for (int dof = 0; dof < num_dofs; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double s = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               s += Bt(dof, qp) * C(vd, qp, e) * detJ(qp, e) * ir.GetWeights()[qp];
            }
            Yi(dof, vd, e) = s;
         }
      }
   }

   Vector yi(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
   return yi;
}

Vector integrate_basis(Vector& s_qp, const FiniteElementSpace &fes,
                       const IntegrationRule& ir)
{
   auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);

   auto mesh = fes.GetMesh();
   const int dim = mesh->SpaceDimension();
   const int num_el = mesh->GetNE();
   const int vdim = fes.GetVDim();
   const int num_qp = ir.GetNPoints();
   const int num_vdofs = R->Height() / num_el;
   const int num_dofs = num_vdofs / vdim;

   const GeometricFactors *geom = mesh->GetGeometricFactors(
                                     ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   Vector yi_el(R->Height());
   auto Yi = Reshape(yi_el.Write(), num_dofs, vdim, num_el);

   auto C = Reshape(s_qp.ReadWrite(), vdim, num_qp, num_el);
   auto detJ = Reshape(geom->detJ.Read(), num_qp, num_el);
   for (int e = 0; e < num_el; e++)
   {
      const DofToQuad &maps = fes.GetFE(e)->GetDofToQuad(ir, DofToQuad::FULL);
      const auto Bt = Reshape(maps.Bt.Read(), num_dofs, num_qp);

      for (int dof = 0; dof < num_dofs; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double s = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               s += Bt(dof, qp) * C(vd, qp, e) * detJ(qp, e) * ir.GetWeights()[qp];
            }
            Yi(dof, vd, e) = s;
         }
      }
   }

   Vector yi(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
   return yi;
}

Vector integrate_basis_gradient(Vector& s_qp, const FiniteElementSpace &fes,
                                const IntegrationRule& ir)
{
   auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);

   auto mesh = fes.GetMesh();
   const int dim = mesh->SpaceDimension();
   const int num_el = mesh->GetNE();
   const int vdim = fes.GetVDim();
   const int num_qp = ir.GetNPoints();
   const int num_vdofs = R->Height() / num_el;
   const int num_dofs = num_vdofs / vdim;

   const GeometricFactors *geom = mesh->GetGeometricFactors(
                                     ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   Vector yi_el(R->Height());
   auto Yi = Reshape(yi_el.Write(), num_dofs, vdim, num_el);

   auto C = Reshape(s_qp.ReadWrite(), vdim, dim, num_qp, num_el);
   auto detJ = Reshape(geom->detJ.Read(), num_qp, num_el);
   auto J = Reshape(geom->J.Read(), num_qp, dim, dim, num_el);
   DenseMatrix Jqp(dim, dim), JqpInv(dim, dim);
   for (int e = 0; e < num_el; e++)
   {
      const DofToQuad &maps = fes.GetFE(e)->GetDofToQuad(ir, DofToQuad::FULL);
      const auto Gt = Reshape(maps.Gt.Read(), num_dofs, num_qp, dim);

      for (int dof = 0; dof < num_dofs; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double s = 0.0;
            for (int d = 0; d < dim; d++)
            {
               for (int qp = 0; qp < num_qp; qp++)
               {
                  for (int i = 0; i < dim; i++)
                  {
                     for (int j = 0; j < dim; j++)
                     {
                        Jqp(i, j) = J(qp, i, j, e);
                     }
                  }

                  CalcInverse(Jqp, JqpInv);

                  double C_Jinv = 0;
                  for (int k = 0; k < dim; k++)
                  {
                     C_Jinv += JqpInv(d, k) * C(vd, k, qp, e) * detJ(qp, e) * ir.GetWeights()[qp];
                  }
                  s += Gt(dof, qp, d) * C_Jinv;
               }
            }
            Yi(dof, vd, e) = s;
         }
      }
   }
   Vector yi(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
   return yi;
}