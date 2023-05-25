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
#error Unsupported compiler
#endif

   const auto start = function.find(prefix) + prefix.size();
   const auto end = function.find(suffix);
   const auto size = end - start;

   return function.substr(start, size);
}

#include <type_traits>

#include <mfem.hpp>
#include <linalg/dtensor.hpp>
#include "linalg/tensor.hpp"
#include <linalg/kernels.hpp>

using namespace mfem;

template <
   typename type,
   int rank = 1>
class NDArray : public Array<type>
{
};

template <class F>
struct FunctionSignature;

template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)>
{
   using return_type = output_type;
   using parameter_types = std::tuple<input_types...>;
};

template <class T>
struct create_function_signature;

template <typename output_type, typename T, typename... input_types>
struct create_function_signature<output_type (T::*)(input_types...) const>
{
   using type = FunctionSignature<output_type(input_types...)>;
};

template <typename function_type, typename... input_types, typename output_type,
          typename... arg_types>
void forall_impl(FunctionSignature<output_type(input_types...)>,
                 const function_type &f, const int n, arg_types &...args)
{
   for (int i = 0; i < n; i++)
   {
      f(((typename std::remove_reference<input_types>::type
          *)(args.begin()))[i]...);
   };
}

template <typename function_type, typename... arg_types>
void forall(const function_type &f, const int n, arg_types &...args)
{
   using function_signature_type = typename create_function_signature<
                                   decltype(&function_type::operator())>::type;
   forall_impl(function_signature_type{}, f, n, args...);
}

template <typename... arg_types, typename... input_types>
void forall(void (*f)(arg_types &...), const int n, input_types &...args)
{
   for (int i = 0; i < n; i++)
   {
      f(((typename std::remove_reference<arg_types>::type*)(args.begin()))[i]...);
   };
}

template <
   typename output_type,
   typename... input_types>
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

      return __enzyme_fwddiff<output_type>(f, &args..., &args2...);
   };
}

/// @return u_qp qp x vdim x elements
void interpolate(const GridFunction &u, const IntegrationRule &ir, Vector &u_qp)
{
   auto fes = u.FESpace();
   auto B = fes->GetQuadratureInterpolator(ir);
   B->SetOutputLayout(QVectorLayout::byVDIM);
   B->DisableTensorProducts();

   auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);
   Vector u_el(R->Height());
   R->Mult(u, u_el);

   u_qp.SetSize(fes->GetVDim() *
                fes->GetMesh()->GetNE() *
                ir.GetNPoints());

   B->Values(u_el, u_qp);
}

void gradient_wrt_x(const GridFunction &u, const IntegrationRule &ir,
                    Vector &grad_u_qp)
{
   auto fes = u.FESpace();
   auto B = fes->GetQuadratureInterpolator(ir);
   B->SetOutputLayout(QVectorLayout::byVDIM);
   B->DisableTensorProducts();

   auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);
   Vector u_el(R->Height());
   R->Mult(u, u_el);

   grad_u_qp.SetSize(
      fes->GetVDim() *
      fes->GetMesh()->Dimension() *
      fes->GetMesh()->GetNE() *
      ir.GetNPoints());

   B->PhysDerivatives(u_el, grad_u_qp);

   if (fes->GetVDim() > 1)
   {
      forall([&](const mfem::internal::tensor<double, 2, 2> &dudx,
                 mfem::internal::tensor<double, 2, 2> &dudx_transpose)
      {
         dudx_transpose = transpose(dudx);
      }, ir.GetNPoints() * fes->GetMesh()->GetNE(), grad_u_qp, grad_u_qp);
   }
}

void integrate_basis(Vector &s_qp, const FiniteElementSpace &fes,
                     const IntegrationRule &ir, Vector &yi)
{
   auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);

   auto mesh = fes.GetMesh();
   // const int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();
   const int vdim = fes.GetVDim();
   const int num_qp = ir.GetNPoints();
   const int num_vdofs = R->Height() / num_el;
   const int num_dofs = num_vdofs / vdim;

   if constexpr(false)
   {
      out << "#el: " << num_el << " vdim: " << vdim << " #qp: " << num_qp
          << " #vdofs: " << num_vdofs << " #dofs: " << num_dofs << "\n";
   }

   const GeometricFactors *geom = mesh->GetGeometricFactors(
                                     ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   Vector yi_el(R->Height());
   yi_el = 0.0;
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

   yi.SetSize(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
}

void integrate_basis_gradient(Vector &s_qp, const FiniteElementSpace &fes,
                              const IntegrationRule &ir, Vector &yi,
                              const Vector &element_jacobian_inverse)
{
   auto R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);

   auto mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
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
   auto JqpInv = Reshape(element_jacobian_inverse.Read(), num_qp, dim, dim,
                         num_el);

   CALI_CXX_MARK_LOOP_BEGIN(element_loop, "element_loop");
   for (int e = 0; e < num_el; e++)
   {
      CALI_CXX_MARK_LOOP_ITERATION(element_loop, e);
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
                  const double JxW = detJ(qp, e) * ir.GetWeights()[qp];
                  for (int k = 0; k < dim; k++)
                  {
                     s += Gt(dof, qp, d) * JqpInv(qp, d, k, e) * C(vd, k, qp, e) * JxW;
                  }
               }
            }
            Yi(dof, vd, e) = s;
         }
      }

      // for (int qp = 0; qp < num_qp; qp++)
      // {
      //    const double JxW = detJ(qp, e) * ir.GetWeights()[qp];

      //    // Pullback Gradient into physical space
      //    Vector dphidx_data(num_dofs * dim);
      //    dphidx_data = 0.0;
      //    auto dphidx = Reshape(dphidx_data.ReadWrite(), num_dofs, dim);
      //    for (int d = 0; d < dim; d++)
      //    {
      //       for (int dof = 0; dof < num_dofs; dof++)
      //       {
      //          for (int k = 0; k < dim; k++)
      //          {
      //             dphidx(dof, d) += G(qp, k, dof) * JqpInv(qp, k, d, e);
      //          }
      //       }
      //    }

      //    for (int vd = 0; vd < vdim; vd++)
      //    {
      //       for (int dof = 0; dof < num_dofs; dof++)
      //       {
      //          double s = 0;
      //          for (int d = 0; d < dim; d++)
      //          {
      //             s += dphidx(dof, d) * C(vd, d, qp, e) * JxW;
      //          }
      //          Yi(dof, vd, e) = s;
      //       }
      //    }
      // }
   }
   CALI_CXX_MARK_LOOP_END(element_loop);
   yi.SetSize(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
}

void interpolate_boundary(const GridFunction &u, const IntegrationRule &ir_face,
                          Vector &u_qp)
{
   auto fes = u.FESpace();
   auto B = fes->GetFaceQuadratureInterpolator(ir_face, FaceType::Boundary);
   B->SetOutputLayout(QVectorLayout::byVDIM);
   B->DisableTensorProducts();

   auto R = fes->GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,
                                    FaceType::Boundary);
   Vector u_el(R->Height());
   R->Mult(u, u_el);

   u_qp.SetSize(
      fes->GetVDim() *
      fes->GetMesh()->GetNBE() *
      ir_face.GetNPoints());

   B->Values(u_el, u_qp);
}

void integrate_basis_boundary(Vector &s_qp,
                              const FiniteElementSpace &fes,
                              const IntegrationRule &ir_face, Vector &yi)
{
   const auto fe = fes.GetFaceElement(0);
   const auto tfe = dynamic_cast<const TensorBasisElement *>(fe);
   MFEM_VERIFY(tfe != nullptr, "FE not a TensorBasisElement");

   auto R = fes.GetFaceRestriction(ElementDofOrdering::LEXICOGRAPHIC,
                                   FaceType::Boundary);

   auto mesh = fes.GetMesh();
   // const int dim = mesh->Dimension();
   const int num_fel = mesh->GetNBE();
   // const int num_fel = mesh->GetNFaces();
   const int vdim = fes.GetVDim();
   const int num_qp = ir_face.GetNPoints();
   const int num_vdofs = R->Height() / num_fel;
   const int num_dofs = num_vdofs / vdim;

   const FaceGeometricFactors *geom = mesh->GetFaceGeometricFactors(
                                         ir_face, FaceGeometricFactors::DETERMINANTS,
                                         FaceType::Boundary, s_qp.GetMemory().GetMemoryType());
   Vector yi_el(R->Height());
   yi_el = 0.0;
   auto Yi = Reshape(yi_el.Write(), num_dofs, vdim, num_fel);
   auto C = Reshape(s_qp.ReadWrite(), vdim, num_qp, num_fel);
   auto detJ = Reshape(geom->detJ.Read(), num_qp, num_fel);

   const DofToQuad &maps = fe->GetDofToQuad(ir_face, DofToQuad::FULL);
   auto lex_to_native = tfe->GetDofMap();

   for (int e = 0; e < num_fel; e++)
   {
      // const DofToQuad &maps = fes.GetBE(e)->GetDofToQuad(ir_face, DofToQuad::FULL);
      const auto Bt = Reshape(maps.Bt.Read(), num_dofs, num_qp);

      for (int dof = 0; dof < num_dofs; dof++)
      {
         for (int vd = 0; vd < vdim; vd++)
         {
            double s = 0.0;
            for (int qp = 0; qp < num_qp; qp++)
            {
               s += Bt(lex_to_native[dof], qp) * C(vd, qp, e)
                    * detJ(qp, e) * ir_face.GetWeights()[qp];
            }
            Yi(dof, vd, e) = s;
         }
      }
   }

   yi.SetSize(fes.GetVSize());
   R->MultTranspose(yi_el, yi);
}
