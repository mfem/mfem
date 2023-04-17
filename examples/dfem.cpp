#include <cassert>
#include <functional>
#include <iostream>
#include <variant>
#include <vector>

#include "dfem.hpp"

#include "eigen-3.4.0/Eigen/Eigen"

using namespace mfem;
using mfem::internal::tensor;
using mfem::internal::make_tensor;
using mfem::internal::dual;
using mfem::internal::make_dual;

using qfunction = std::function<void(double **in, double **out)>;

template<class> inline constexpr bool always_false_v = false;

enum InterpolationMode
{
   None, /* will this be used for syntax clarity? */
   Value,
   Gradient,
   Curl,
   Jump
};

struct Argument
{
   int field_id;
   InterpolationMode mode;
   // Cache variables
};

Argument value(int i)
{
   return
   {
      .field_id = i, .mode = InterpolationMode::Value
   };
}
Argument grad(int i)
{
   return
   {
      .field_id = i, .mode = InterpolationMode::Gradient
   };
}

Argument derivative_wrt(int i)
{
   return
   {
      .field_id = i, .mode = InterpolationMode::None
   };
}


// template <class F>
// struct FunctionSignature;

// template <typename output_type, typename T, typename... input_types>
// struct FunctionSignature<output_type (T::*)(input_types...) const>
// {
//    using input_t = std::tuple<input_types...>;
//    using output_t = output_type;
// };

struct Kernel
{
   template <typename qf_t>
   Kernel(qf_t qf, Mesh &mesh, const IntegrationRule *ir) :
      mesh(mesh),
      ir(ir)
   {
      const int dim = mesh.Dimension();
      const int num_el = mesh.GetNE();
      const int num_qp = ir->GetNPoints();

      const GeometricFactors *geom = mesh.GetGeometricFactors(
                                        *ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);

      this->f = [=](const std::vector<Vector> &qp_cache_in,
                    std::vector<Vector> &qp_cache_out,
                    std::vector<int> qp_cache_in_sizes,
                    std::vector<int> qp_cache_out_sizes)
      {
         typename
         FunctionSignatureOld<decltype(&decltype(qf)::operator())>::input_t in_qp;

         typename
         FunctionSignatureOld<decltype(&decltype(qf)::operator())>::output_t out_qp;

         // Q -> Q
         for (int e = 0; e < num_el; e++)
         {
            for (int qp = 0; qp < num_qp; qp++)
            {
               // @TODO
               double *in_ptr = (double *)(&in_qp);
               for (int arg_i = 0; arg_i < args_in.size(); arg_i++)
               {
                  auto C = Reshape(qp_cache_in[arg_i].Read(), num_qp, qp_cache_in_sizes[arg_i],
                                   num_el);
                  // @TODO These are tiny loops, necessary to move to contiguous
                  // memory layout? Double check with the GPU guys.
                  for (int i = 0; i < qp_cache_in_sizes[arg_i]; i++)
                  {
                     std::memcpy(in_ptr++, &C(qp, i, e), sizeof(double));
                  }
               }

               // Q_i -> Q_i
               out_qp = std::apply(qf, in_qp);

               double *out_ptr = (double *)(&out_qp);
               for (int arg_i = 0; arg_i < args_out.size(); arg_i++)
               {
                  auto C = Reshape(qp_cache_out[arg_i].ReadWrite(), num_qp,
                                   qp_cache_out_sizes[arg_i],
                                   num_el);
                  auto detJ = Reshape(geom->detJ.Read(), num_qp, num_el);
                  for (int i = 0; i < qp_cache_out_sizes[arg_i]; i++)
                  {
                     // move integration related operations outside
                     C(qp, i, e) += *out_ptr * detJ(qp, e) * ir->GetWeights()[qp];
                     // C(qp, i, e) += *out_ptr;
                     out_ptr++;
                  }
               }
            }
         }
      };
   }

   std::function<void(const std::vector<Vector> &, std::vector<Vector> &, std::vector<int>,
                      std::vector<int> )> f;
   std::vector<Argument> args_in;
   std::vector<Argument> args_out;
   Mesh &mesh;
   const IntegrationRule *ir;
};

class ParametricFunction
{
public:
   ParametricFunction(int lsize, int qsize) :
      lsize(lsize),
      qsize(qsize)
   {

   }

   // The "T"ransformation
   // For a GridFunction this is G * B
   // For a Constant this is the identity
   virtual void T(const Vector &u, Vector &Tu)
   {
      Tu = u(0);
   };

   // Transpose of the Transformation
   // For a GridFunction this is G^t * B^t
   virtual void Tt(const Vector &u, Vector &Ttu)
   {
      Ttu = u.Sum();
   };

   // Action of the derivative of the transformation. Used to provide all
   // components to the chain rule.
   void dTxV() {};

   // "Tvector" size
   int lsize;
   int qsize;
};

struct QuadratureF
{
   int lsize;
   int qsize;
};

using FieldVariant =
   std::variant<GridFunction *, ParametricFunction *, QuadratureF *>;

std::ostream& operator<<(std::ostream& os, const FieldVariant& v)
{
   std::visit([&os](auto&& arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, GridFunction *>)
      {
         os << "GridFunction";
      }
      else if constexpr (std::is_same_v<T, ParametricFunction *>)
      {
         os << "ParametricFunction";
      }
      else if constexpr (std::is_same_v<T, QuadratureF *>)
      {
         os << "QuadratureF";
      }
      else
      {
         static_assert(always_false_v<T>, "unknown field type");
      }
   }, v);
   return os;
}

std::ostream& operator<<(std::ostream& os, const InterpolationMode& m)
{
   switch (m)
   {
      case InterpolationMode::None:
         os << "None";
         break;
      case InterpolationMode::Value:
         os << "Value";
         break;
      case InterpolationMode::Gradient:
         os << "Gradient";
         break;
      case InterpolationMode::Curl:
         os << "Curl";
         break;
      case InterpolationMode::Jump:
         os << "Jump";
         break;
      default:
         assert(false);
   }
   return os;
}

class Integral
{
public:
   inline
   int get_qp_size(GridFunction *gf, InterpolationMode op)
   {
      switch (op)
      {
         case InterpolationMode::Value:
            return gf->FESpace()->GetVDim();
         case InterpolationMode::Gradient:
            return gf->FESpace()->GetVDim() * gf->FESpace()->GetMesh()->Dimension();
         default:
            assert(false);
      }
      return -1;
   }

   inline
   int get_qp_size(ParametricFunction *pf, InterpolationMode op)
   {
      switch (op)
      {
         case InterpolationMode::Value:
            return pf->qsize;
         case InterpolationMode::Gradient:
            assert(false);
         default:
            assert(false);
      }
      return -1;
   }

   inline
   int get_lsize(GridFunction *gf, InterpolationMode op)
   {
      switch (op)
      {
         case InterpolationMode::Value:
            return gf->FESpace()->GetNDofs();
         case InterpolationMode::Gradient:
            return gf->FESpace()->GetNDofs() * gf->FESpace()->GetVDim();
         default:
            assert(false);
      }
      return -1;
   }


   Integral(std::initializer_list<FieldVariant> in,
            std::initializer_list<FieldVariant> out)
      : inputs(in), outputs(out), input_offsets(inputs.size()),
        output_offsets(outputs.size())
   {
      ComputeOffsets(inputs, input_offsets);
      ComputeOffsets(outputs, output_offsets);
   }

   template <typename qf_t>
   void AddKernel(std::string kernel_name, qf_t f,
                  std::vector<Argument> args_in,
                  std::vector<Argument> args_out, Mesh &mesh, const IntegrationRule *ir)
   {
      // MFEM_ASSERT(input_offsets.size() == args_in.size(),
      //             "expecting " << input_offsets.size() <<
      //             " inputs as defined in the Integral constructor");
      // MFEM_ASSERT(output_offsets.size() == args_out.size(),
      //             "expecting " << output_offsets.size() <<
      //             " outputs as defined in the Integral constructor");
      kernels[kernel_name] = new Kernel(f, mesh, ir);
      kernels[kernel_name]->args_in = args_in;
      kernels[kernel_name]->args_out = args_out;
   }

   inline
   void gridfunction_callback(FiniteElementSpace *fes, InterpolationMode mode,
                              int dim,
                              int arg_i,
                              int field_id,
                              int num_el,
                              int num_qp,
                              const Vector &x,
                              const IntegrationRule *ir,
                              const GeometricFactors *geom)
   {
      auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);

      // Extract the memory from the input vector
      Vector u(x.GetData() + input_offsets[field_id], fes->GetVSize());

      // L -> E
      Vector u_el(R->Height()); // TODO: cache this
      R->Mult(u, u_el);

      const int vdim = fes->GetVDim();
      const int num_vdofs = R->Height() / num_el;
      const int num_dofs = num_vdofs / vdim;

      const int interpolation_dim = (mode ==
                                     InterpolationMode::Gradient ? dim : 1);
      qp_cache_in_sizes[arg_i] = vdim * interpolation_dim;
      qp_cache_in[arg_i] = Vector(num_el * num_qp * qp_cache_in_sizes[arg_i]);

      // interpolate values and/or grad of field
      const auto U = Reshape(u_el.Read(), num_dofs, vdim, num_el);
      if (mode == InterpolationMode::Value)
      {
         auto C = Reshape(qp_cache_in[arg_i].ReadWrite(), num_qp, vdim, num_el);
         for (int e = 0; e < num_el; e++)
         {
            const DofToQuad &maps = fes->GetFE(e)->GetDofToQuad(*ir,
                                                                DofToQuad::FULL);
            const auto B = Reshape(maps.B.Read(), num_qp, num_dofs);

            for (int qp = 0; qp < num_qp; qp++)
            {
               for (int vd = 0; vd < vdim; vd++)
               {
                  double s = 0.0;
                  for (int dof = 0; dof < num_dofs; dof++)
                  {
                     s += B(qp, dof) * U(dof, vd, e);
                  }
                  C(qp, vd, e) = s;
               }
            }
         }
      }
      else if (mode == InterpolationMode::Gradient)
      {
         qp_cache_in[arg_i] = 0.0;
         auto C = Reshape(qp_cache_in[arg_i].ReadWrite(), num_qp, vdim, dim, num_el);
         auto J = Reshape(geom->J.Read(), num_qp, dim, dim, num_el);
         DenseMatrix Jqp(dim, dim), JqpInv(dim, dim), grad_hat(vdim, dim);
         for (int e = 0; e < num_el; e++)
         {
            const DofToQuad &maps = fes->GetFE(e)->GetDofToQuad(*ir,
                                                                DofToQuad::FULL);
            const auto G = Reshape(maps.G.Read(), num_qp, dim, num_dofs);
            for (int qp = 0; qp < num_qp; qp++)
            {
               for (int vd = 0; vd < vdim; vd++)
               {
                  for (int d = 0; d < dim; d++)
                  {
                     double s = 0.0;
                     for (int dof = 0; dof < num_dofs; dof++)
                     {
                        s += G(qp, d, dof) * U(dof, vd, e);
                     }
                     grad_hat(vd, d) = s;
                  }
               }

               for (int i = 0; i < dim; i++)
               {
                  for (int j = 0; j < dim; j++)
                  {
                     Jqp(i, j) = J(qp, i, j, e);
                  }
               }

               // @TODO double check
               CalcInverse(Jqp, JqpInv);

               for (int i = 0; i < vdim; i++)
               {
                  for (int k = 0; k < dim; k++)
                  {
                     for (int j = 0; j < dim; j++)
                     {
                        C(qp, i, j, e) += grad_hat(i, k) * JqpInv(k, j);
                     }
                  }
               }

               // for (int i = 0; i < vdim; i++)
               // {
               //    for (int j = 0; j < dim; j++)
               //    {
               //       out << C(qp, i, j, e) << " ";
               //    }
               //    out << "\n";
               // }
               // out << "\n";
            }
         }
         // qp_cache_in[arg_i].Print(out, qp_cache_in[arg_i].Size());
      }
   }

   inline
   void parametricfunction_callback(ParametricFunction *pf, int arg_i,
                                    int num_el, int num_qp, const Vector &x)
   {
      Vector u(x.GetData() + input_offsets[arg_i], pf->lsize);
      Vector y(pf->qsize);

      qp_cache_in_sizes[arg_i] = pf->qsize;
      qp_cache_in[arg_i].SetSize(num_qp * pf->qsize * num_el);

      auto C = Reshape(qp_cache_in[arg_i].Write(), num_qp, pf->qsize, num_el);

      for (int e = 0; e < num_el; e++)
      {
         for (int qp = 0; qp < num_qp; qp++)
         {
            pf->T(u, y);
            for (int i = 0; i < pf->qsize; i++)
            {
               C(qp, i, e) = y(i);
            }
         }
      }
   }

   // x is a "block" vector if multiple spaces are involved
   void CallKernel(std::string kernel_name, const Vector &x, Vector &y, Mesh &mesh,
                   std::initializer_list<Argument> derivative_idx = {})
   {
      MFEM_ASSERT(kernels.find(kernel_name) != kernels.end(),
                  "kernel " << kernel_name << " not found");

      auto kernel = kernels[kernel_name];

      // some constants
      const int dim = mesh.Dimension();
      const int num_el = mesh.GetNE();
      const int num_qp = kernel->ir->GetNPoints();

      const GeometricFactors *geom = mesh.GetGeometricFactors(
                                        *kernel->ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);

      // reset residual
      y *= 0;

      // Only transfer fields that are used in the kernel to L and E vectors (in
      // loop below). Also, only needed if we deal with an FESpace.

      double **in_qp = new double *[kernel->args_in.size()];
      double **out_qp = new double *[kernel->args_out.size()];

      qp_cache_in.resize(kernel->args_in.size());
      qp_cache_in_sizes.resize(kernel->args_in.size());

      qp_cache_out.resize(kernel->args_out.size());
      qp_cache_out_sizes.resize(kernel->args_out.size());

      for (int arg_i = 0; arg_i < kernel->args_in.size(); arg_i++)
      {
         int field_id = kernel->args_in[arg_i].field_id;

         std::visit([=](auto&& arg)
         {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, GridFunction *>)
            {
               gridfunction_callback(arg->FESpace(), kernel->args_in[arg_i].mode, dim, arg_i,
                                     field_id, num_el,
                                     num_qp, x,
                                     kernel->ir,
                                     geom);
               // in_qp[arg_i] = new double[get_qp_size(arg, kernel->args_in[arg_i].mode)];
            }
            else if constexpr (std::is_same_v<T, ParametricFunction *>)
            {
               parametricfunction_callback(arg, arg_i, num_el, num_qp, x);
               // in_qp[arg_i] = new double[get_qp_size(arg, kernel->args_in[arg_i].mode)];
            }
            else if constexpr (std::is_same_v<T, QuadratureF *>)
            {
               MFEM_ABORT("not implemented");
            }
            else
            {
               static_assert(always_false_v<T>, "unknown field type");
            }
         }, inputs[field_id]);
      }

      for (int arg_i = 0; arg_i < kernel->args_out.size(); arg_i++)
      {
         int field_id = kernel->args_out[arg_i].field_id;
         std::visit([=](auto&& arg)
         {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, GridFunction *>)
            {
               if (kernel->args_out[arg_i].mode == InterpolationMode::Value)
               {
                  qp_cache_out_sizes[arg_i] = arg->FESpace()->GetVDim();
               }
               else if (kernel->args_out[arg_i].mode == InterpolationMode::Gradient)
               {
                  qp_cache_out_sizes[arg_i] = arg->FESpace()->GetVDim() *
                                              arg->FESpace()->GetMesh()->Dimension();
               }
               else
               {
                  MFEM_ABORT("unknown InterpolationMode");
               }
            }
            else if constexpr (std::is_same_v<T, ParametricFunction *>)
            {
               qp_cache_out_sizes[arg_i] = arg->qsize;
            }
            else if constexpr (std::is_same_v<T, QuadratureF *>)
            {
               MFEM_ABORT("not implemented yet");
            }
            else
            {
               static_assert(always_false_v<T>, "unknown field type");
            }
         }, outputs[field_id]);

         qp_cache_out[arg_i].SetSize(num_qp * qp_cache_out_sizes[arg_i] * num_el);
         qp_cache_out[arg_i] = 0.0;
         out_qp[arg_i] = new double[qp_cache_out_sizes[arg_i]];
      }

      if (false)
      {
         mfem::out << "Total number of inputs " << inputs.size() << "\n"
                   << "Total number of outputs " << outputs.size()
                   << std::endl;

         mfem::out << "Inputs\n";
         for (int arg_i = 0; arg_i < kernel->args_in.size(); arg_i++)
         {
            mfem::out << "Field " << arg_i << "\n"
                      << " Type: " << inputs[kernel->args_in[arg_i].field_id] << "\n"
                      << " Interpolation: " << kernel->args_in[arg_i].mode << "\n"
                      << " Size on qp: " << qp_cache_in_sizes[arg_i] << "\n";
         }
         mfem::out << std::endl;

         mfem::out << "Outputs\n";
         for (int arg_i = 0; arg_i < kernel->args_out.size(); arg_i++)
         {
            mfem::out << "Field " << arg_i << "\n"
                      << " Type: " << outputs[kernel->args_out[arg_i].field_id] << "\n"
                      << " Interpolation: " << kernel->args_out[arg_i].mode << "\n"
                      << " Size on qp: " << qp_cache_out_sizes[arg_i] << "\n";
         }
         mfem::out << std::endl;
      }

      // Q -> Q
      kernel->f(qp_cache_in, qp_cache_out, qp_cache_in_sizes, qp_cache_out_sizes);

      // qp_cache_out[0].Print(out, qp_cache_out[0].Size());

      for (int arg_i = 0; arg_i < kernel->args_out.size(); arg_i++)
      {
         int field_id = kernel->args_out[arg_i].field_id;
         std::visit([&](auto&& arg)
         {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, GridFunction *>)
            {
               auto fes = arg->FESpace();
               auto R = fes->GetElementRestriction(ElementDofOrdering::NATIVE);

               const int vdim = fes->GetVDim();
               const int num_vdofs = R->Height() / num_el;
               const int num_dofs = num_vdofs / vdim;

               Vector yi_el(R->Height());
               auto Yi = Reshape(yi_el.Write(), num_dofs, vdim, num_el);

               if (kernel->args_out[arg_i].mode == InterpolationMode::Value)
               {
                  auto C = Reshape(qp_cache_out[arg_i].Read(), num_qp, vdim, num_el);
                  for (int e = 0; e < num_el; e++)
                  {
                     const DofToQuad &maps = fes->GetFE(e)->GetDofToQuad(*kernel->ir,
                                                                         DofToQuad::FULL);
                     const auto Bt = Reshape(maps.Bt.Read(), num_dofs, num_qp);

                     for (int dof = 0; dof < num_dofs; dof++)
                     {
                        for (int vd = 0; vd < vdim; vd++)
                        {
                           double s = 0.0;
                           for (int qp = 0; qp < num_qp; qp++)
                           {
                              s += Bt(dof, qp) * C(qp, vd, e);
                           }
                           Yi(dof, vd, e) = s;
                        }
                     }
                  }
               }
               else if (kernel->args_out[arg_i].mode == InterpolationMode::Gradient)
               {
                  auto C = Reshape(qp_cache_out[arg_i].Read(), num_qp, vdim, dim, num_el);
                  auto J = Reshape(geom->J.Read(), num_qp, dim, dim, num_el);
                  DenseMatrix Jqp(dim, dim), JqpInv(dim, dim);
                  for (int e = 0; e < num_el; e++)
                  {
                     const DofToQuad &maps = fes->GetFE(e)->GetDofToQuad(*kernel->ir,
                                                                         DofToQuad::FULL);
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
                                    C_Jinv += JqpInv(d, k) * C(qp, vd, k, e);
                                 }
                                 s += Gt(dof, qp, d) * C_Jinv;
                              }
                           }
                           Yi(dof, vd, e) = s;
                        }
                     }
                  }
               }
               else
               {
                  MFEM_ABORT("unknown InterpolationMode");
               }
               Vector yi(y.GetData() + output_offsets[field_id], fes->GetVSize());
               Vector yi_tmp(yi.Size());
               R->MultTranspose(yi_el, yi_tmp);
               yi += yi_tmp;
            }
            else if constexpr (std::is_same_v<T, ParametricFunction *>)
            {
               Vector Ttu(arg->lsize);
               arg->Tt(qp_cache_out[arg_i], Ttu);

               auto Y = Reshape(y.Write() + output_offsets[arg_i], arg->lsize);
               for (int i = 0; i < arg->lsize; i++)
               {
                  Y(i) = Ttu(i);
               }
            }
            else if constexpr (std::is_same_v<T, QuadratureF *>)
            {
               MFEM_ABORT("not implemented yet");
            }
            else
            {
               static_assert(always_false_v<T>, "unknown field type");
            }
         }, outputs[field_id]);
      }

      // for (int arg_i = 0; arg_i < inputs.size(); arg_i++)
      // {
      //    delete[] in_qp[arg_i];
      // }
      // delete[] in_qp;

      // for (int arg_i = 0; arg_i < outputs.size(); arg_i++)
      // {
      //    delete[] out_qp[arg_i];
      // }
      // delete[] out_qp;
   }

private:
   void ComputeOffsets(std::vector<FieldVariant> fields,
                       std::vector<int> &offsets)
   {
      offsets[0] = 0;
      for (int i = 1; i < fields.size(); i++)
      {
         int i_size = 0;
         std::visit([&](auto&& arg)
         {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, GridFunction *>)
            {
               i_size = arg->FESpace()->GetVSize();
            }
            else if constexpr (std::is_same_v<T, ParametricFunction *>)
            {
               i_size = arg->lsize;
            }
            else if constexpr (std::is_same_v<T, QuadratureF *>)
            {
               MFEM_ABORT("not implemented yet");
            }
            else
            {
               static_assert(always_false_v<T>, "unknown field type");
            }
            offsets[i] = offsets[i - 1] + i_size;
         }, fields[i - 1]);
      }
   }

   // const int dimension = 0;
   std::vector<FieldVariant> inputs, outputs;
   std::vector<int> input_offsets, output_offsets;
   std::unordered_map<std::string, Kernel *> kernels;
   // #arg_in_kernel
   std::vector<Vector> qp_cache_in;
   std::vector<int> qp_cache_in_sizes;

   std::vector<Vector> qp_cache_out;
   std::vector<int> qp_cache_out_sizes;
};

void run_problem6();
void run_problem7();

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int problem_type = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&problem_type, "-p", "--problem",
                  "Problem to run");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(mfem::out);
   }

   // // Mesh volume computation (local/global)
   // if (problem_type == 0)
   // {
   //    const int dim = 2;

   //    Mesh mesh = Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL, false,
   //                                      2.0*M_PI,
   //                                      2.0*M_PI);
   //    mesh.EnsureNodes();

   //    ParMesh pmesh(MPI_COMM_WORLD, mesh);

   //    IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
   //                                                    2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

   //    ParametricFunction scalar_one(1, 1);
   //    Integral integral({&scalar_one}, {&scalar_one});

   //    // enum InputFieldID { ScalarOne };
   //    // enum OutputFieldID { ScalarOne };

   //    int Input_ScalarOne = 0,
   //        Output_ScalarOne = 0;

   //    auto area = [](double **in, double **out)
   //    {
   //       // double *scalar_one = reinterpret_cast<double *>(in[0]);
   //       out[0][0] = 1.0;
   //    };

   //    // 1^T detJ * w 1 u

   //    // 1^T   Added from output spec value(OutputFieldID::ScalarOne)
   //    // w     Added automatically, determined by IntegrationRule
   //    // detJ  Added automatically, determined by the mesh
   //    // 1     Added from input spec value(InputFieldID::ScalarOne)
   //    integral.AddKernel("area", area, {value(Input_ScalarOne)},
   //    {value(Output_ScalarOne)}, &integration_rule);

   //    Vector in(1);
   //    in = 1.0;
   //    Vector area_out(1);
   //    area_out = 0.0;
   //    integral.CallKernel("area", in, area_out, pmesh);
   //    // area_out now contains the processor local area

   //    mfem::out << "area = " << area_out(0) << std::endl;
   // }
   // else if (problem_type == 1)
   // {
   //    const int dim = 2;

   //    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
   //                                      2.0*M_PI,
   //                                      2.0*M_PI);
   //    mesh.EnsureNodes();

   //    ParMesh pmesh(MPI_COMM_WORLD, mesh);

   //    IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
   //                                                    2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

   //    H1_FECollection h1fec(1);
   //    ParFiniteElementSpace h1fes(&pmesh, &h1fec);

   //    ParGridFunction u(&h1fes);

   //    Integral integral({pmesh.GetNodes()}, {&u});

   //    auto lf = [](double **in, double **out)
   //    {
   //       double *x = in[0];
   //       out[0][0] = x[0] + x[1];
   //    };

   //    integral.AddKernel("lf", lf, {value(0)}, {value(0)}, &integration_rule);

   //    u = 0.0;
   //    integral.CallKernel("lf", *pmesh.GetNodes(), u, pmesh);
   //    u.Print(mfem::out, u.Size());

   //    // Comparison against existing implementation
   //    {
   //       FunctionCoefficient f([](const Vector &x)
   //       {
   //          return x[0] + x[1];
   //       });

   //       LinearForm lf(&h1fes);
   //       auto integrator = new DomainLFIntegrator(f);
   //       integrator->SetIntRule(&integration_rule);
   //       lf.AddDomainIntegrator(integrator);
   //       lf.Assemble();
   //       lf.Print(mfem::out, lf.Size());
   //    }
   // }
   if (problem_type == 2)
   {
      const int dim = 2;

      Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                        2.0*M_PI,
                                        2.0*M_PI);
      mesh.EnsureNodes();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
                                                      2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(&pmesh, &h1fec);

      ParGridFunction u(&h1fes);
      FunctionCoefficient init([](const Vector &x)
      {
         return x[0] * x[1];
      });
      u.ProjectCoefficient(init);

      Vector x_qp = interpolate(*pmesh.GetNodes(), integration_rule);
      Vector grad_u_qp = gradient_wrt_x(u, integration_rule);

      auto convection_qf = [](tensor<double, 2> x,
                              tensor<dual<double, double>, 2> grad_u)
      {
         tensor<double, 2> vel{x[1], x[0]};
         auto r = dot(vel, grad_u);
         return r;
      };

      // auto convection_qp = forall(convection_qf,
      //                             mesh.GetNE() * integration_rule.GetNPoints(), x_qp, grad_u_qp);

      auto grad_u_dual_qp = forall(std::function<dual<double, double>(double)>
                                   (make_dual),
                                   mesh.GetNE() * integration_rule.GetNPoints(),
                                   grad_u_qp);

      auto convection_gradu_qp = forall(convection_qf,
                                        mesh.GetNE() * integration_rule.GetNPoints(), x_qp, grad_u_dual_qp);

      // Vector s = integrate_basis(convection_gradu_qp, mesh, integration_rule, h1fes);

      // s.Print(out, s.Size());
      // printf("\n\n");

      Integral integral({pmesh.GetNodes(), &u}, {&u});

      auto convection = [](tensor<double, 2> x, tensor<double, 2> grad_u)
      {
         tensor<double, 2> vel{x[1], x[0]};
         double r = dot(vel, grad_u);
         // printf("%.5f ", r);
         // out << grad_u << std::endl;
         return r;
      };

      integral.AddKernel("convection", convection, {value(0), grad(1)}, {value(0)},
                         mesh, &integration_rule);

      Array<int> block_offsets(3);
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + pmesh.GetNodes()->Size();
      block_offsets[2] = block_offsets[1] + u.Size();
      BlockVector in(block_offsets);

      in.GetBlock(0) = *pmesh.GetNodes();
      in.GetBlock(1) = u;

      integral.CallKernel("convection", in, u, pmesh);

      printf("\n\n");
      u.Print(out, u.Size());

      // Comparison against existing implementation
      {
         VectorFunctionCoefficient f(dim, [](const Vector &x, Vector &u)
         {
            u(0) = x[1];
            u(1) = x[0];
         });

         ParBilinearForm K(&h1fes);
         auto integrator = new ConvectionIntegrator(f);
         integrator->SetIntRule(&integration_rule);
         K.AddDomainIntegrator(integrator);
         K.Assemble();
         K.Finalize();
         auto Kmat = K.ParallelAssemble();

         ParGridFunction u(&h1fes);
         u.ProjectCoefficient(init);
         auto u_tdofs = u.GetTrueDofs();
         Vector y_tdofs(u_tdofs->Size());
         Kmat->Mult(*u_tdofs, y_tdofs);
         u.SetFromTrueDofs(y_tdofs);
         u.Print(mfem::out, u.Size());
      }
   }
   // else if (problem_type == 3)
   // {
   //    const int dim = 2;

   //    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
   //                                      2.0*M_PI,
   //                                      2.0*M_PI);
   //    mesh.EnsureNodes();

   //    ParMesh pmesh(MPI_COMM_WORLD, mesh);

   //    IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
   //                                                    2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

   //    H1_FECollection h1fec(1);
   //    ParFiniteElementSpace h1fes(&pmesh, &h1fec);

   //    ParGridFunction u(&h1fes);
   //    FunctionCoefficient init([](const Vector &x)
   //    {
   //       return x[0] * x[1];
   //    });
   //    u.ProjectCoefficient(init);

   //    Integral integral({&u}, {&u});

   //    int InputID_Potential = 0,
   //        OutputID_Potential = 0;

   //    // A_Q
   //    auto diffusion = [&](double **in, double **out)
   //    {
   //       Eigen::Map<Eigen::Vector<double, 1>> u(in[0]);
   //       Eigen::Map<Eigen::Vector2d> grad_u(in[1]);

   //       Eigen::Map<Eigen::Vector<double, 1>> v(out[0]);
   //       Eigen::Map<Eigen::Vector2d> grad_v(out[1]);

   //       v = u;
   //       grad_v = grad_u;
   //    };

   //    integral.AddKernel("diffusion", diffusion,
   //    {value(InputID_Potential), grad(InputID_Potential)},
   //    {value(OutputID_Potential), grad(OutputID_Potential)},
   //    &integration_rule);

   //    Vector out(u.Size());
   //    out = 0.0;

   //    integral.CallKernel("diffusion", u, out, pmesh);
   //    out.Print(mfem::out, out.Size());

   //    // integral.CallKernelDerivative("diffusion", u, out, pmesh, derivative_wrt(0));

   //    // Comparison against existing implementation
   //    {
   //       ParBilinearForm K(&h1fes);

   //       auto mass_integrator = new MassIntegrator;
   //       mass_integrator->SetIntRule(&integration_rule);
   //       K.AddDomainIntegrator(mass_integrator);

   //       auto diffusion_integrator = new DiffusionIntegrator;
   //       diffusion_integrator->SetIntRule(&integration_rule);
   //       K.AddDomainIntegrator(diffusion_integrator);

   //       K.Assemble();
   //       K.Finalize();
   //       auto Kmat = K.ParallelAssemble();

   //       ParGridFunction u(&h1fes);
   //       u.ProjectCoefficient(init);
   //       auto u_tdofs = u.GetTrueDofs();
   //       Vector y_tdofs(u_tdofs->Size());
   //       Kmat->Mult(*u_tdofs, y_tdofs);
   //       u.SetFromTrueDofs(y_tdofs);
   //       u.Print(mfem::out, u.Size());
   //    }
   // }
   if (problem_type == 4)
   {
      const int dim = 2;

      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                        2.0*M_PI,
                                        2.0*M_PI);
      mesh.EnsureNodes();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
                                                      2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(&pmesh, &h1fec, pmesh.Dimension());

      ParGridFunction u(&h1fes);
      VectorFunctionCoefficient init(dim, [](const Vector &x, Vector &u)
      {
         u(0) = x[0] * x[1];
         u(1) = x[0] * x[1];
      });
      u.ProjectCoefficient(init);

      {
         auto grad_u_qp = gradient_wrt_x(u, integration_rule);
         auto vector_diffusion = integrate_basis_gradient(grad_u_qp, h1fes,
                                                          integration_rule);
         vector_diffusion.Print();
      }

      Integral integral({&u}, {&u});

      // A_Q
      auto vector_diffusion = [](tensor<double, 2, 2> grad_u)
      {
         return grad_u;
      };

      integral.AddKernel("vector_diffusion", vector_diffusion, {grad(0)}, {grad(0)},
                         mesh, &integration_rule);

      Vector out(u.Size());
      out = 0.0;
      integral.CallKernel("vector_diffusion", u, out, pmesh);
      out.Print(mfem::out, out.Size());

      // Comparison against existing implementation
      {
         ParBilinearForm K(&h1fes);
         auto integrator = new VectorDiffusionIntegrator;
         integrator->SetIntRule(&integration_rule);
         K.AddDomainIntegrator(integrator);
         K.Assemble();
         K.Finalize();
         auto Kmat = K.ParallelAssemble();

         ParGridFunction u(&h1fes);
         u.ProjectCoefficient(init);
         auto u_tdofs = u.GetTrueDofs();
         Vector y_tdofs(u_tdofs->Size());
         Kmat->Mult(*u_tdofs, y_tdofs);
         u.SetFromTrueDofs(y_tdofs);

         u -= out;
         printf("||u|| = %.5E\n", u.Norml2());
      }
   }
   // else if (problem_type == 5)
   // {
   //    const int dim = 2;

   //    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
   //                                      2.0*M_PI,
   //                                      2.0*M_PI);
   //    mesh.EnsureNodes();

   //    ParMesh pmesh(MPI_COMM_WORLD, mesh);

   //    IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
   //                                                    2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

   //    H1_FECollection h1fec(1);
   //    ParFiniteElementSpace h1fes(&pmesh, &h1fec);
   //    ParFiniteElementSpace h1vfes(&pmesh, &h1fec, pmesh.Dimension());

   //    ParGridFunction u(&h1vfes);
   //    VectorFunctionCoefficient init_u(dim, [](const Vector &x, Vector &u)
   //    {
   //       // u(0) = x[0] + x[1];
   //       // u(1) = x[0] - x[1];
   //       u(0) = 1.0;
   //       u(1) = 2.0;
   //    });
   //    u.ProjectCoefficient(init_u);

   //    ParGridFunction p(&h1fes);
   //    FunctionCoefficient init_p([](const Vector &x)
   //    {
   //       return x[0] + x[1];
   //    });
   //    p.ProjectCoefficient(init_p);

   //    Integral integral({&u, &p}, {&u, &p});

   //    int InputID_Velocity{0},
   //        InputID_Pressure{1},
   //        OutputID_Velocity{0},
   //        OutputID_Pressure{1};

   //    // A_Q
   //    auto stokes = [](double **in, double **out)
   //    {
   //       Eigen::Map<Eigen::Matrix2d> grad_u(in[0]);
   //       Eigen::Map<Eigen::Vector2d> grad_p(in[1]);

   //       Eigen::Map<Eigen::Vector2d> v(out[0]);
   //       Eigen::Map<Eigen::Matrix2d> grad_v(out[1]);
   //       double *q = out[2];

   //       v = grad_p; // (v, \nabla p)
   //       // v *= 0.0;
   //       grad_v = grad_u; // (\nabla v, \nabla u)
   //       *q = grad_u.diagonal().sum(); // (q, \nabla \cdot u)
   //    };

   //    integral.AddKernel("stokes", stokes,
   //    {grad(InputID_Velocity), grad(InputID_Pressure)},
   //    {value(OutputID_Velocity), grad(OutputID_Velocity), value(OutputID_Pressure)},
   //    &integration_rule);

   //    Array<int> block_offsets(3);
   //    block_offsets[0] = 0;
   //    block_offsets[1] = block_offsets[0] + u.Size();
   //    block_offsets[2] = block_offsets[1] + p.Size();

   //    BlockVector in(block_offsets);
   //    in.GetBlock(0) = u;
   //    in.GetBlock(1) = p;

   //    BlockVector out(block_offsets);
   //    out = 0.0;

   //    integral.CallKernel("stokes", in, out, pmesh);

   //    // integral.CallKernelDerivative("stokes", in, out, pmesh, derivative_wrt(InputID_Velocity));

   //    u.SetVector(out.GetBlock(0), 0);
   //    u.Print(mfem::out, u.Size());

   //    p.SetVector(out.GetBlock(1), 0);
   //    ConstantCoefficient zero(0.0);
   //    double p_err = p.ComputeL2Error(zero);
   //    printf("||div(u) - div(u)_ex|| = %.5E\n", p_err);
   // }
   if (problem_type == 6)
   {
      run_problem6();
   }
   if (problem_type == 7)
   {
      run_problem7();
   }

   return 0;
}

class OperatorWithGradient : public Operator
{
public:
   OperatorWithGradient(int size) : Operator(size) {}
   virtual void GradientMult(const Vector &dX, Vector &Y) const = 0;
};

class PLaplacianGradientOperator : public Operator
{
public:
   PLaplacianGradientOperator(OperatorWithGradient &op) :
      Operator(op.Height()), op(op) {}

   void Mult(const Vector &x, Vector &y) const override
   {
      op.GradientMult(x, y);
   }

   OperatorWithGradient &op;
};

class PLaplacianOperator : public OperatorWithGradient
{
public:
   PLaplacianOperator(ParFiniteElementSpace &fes, ParGridFunction &u) :
      OperatorWithGradient(fes.GetTrueVSize()),
      mesh(fes.GetParMesh()),
      fes(fes),
      u(u),
      integration_rule(const_cast<IntegrationRule &>(IntRules.Get(
                                                        Element::QUADRILATERAL,
                                                        2 * mesh->GetNodes()->FESpace()->GetOrder(0) + 1)))
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      ConstantCoefficient one(1.0);
      u.ProjectBdrCoefficient(one, ess_bdr);

      du.SetSpace(u.FESpace());
      x_lvec.SetSize(fes.GetProlongationMatrix()->Height());
      x_lvec = 0.0;
      y_lvec.SetSize(fes.GetProlongationMatrix()->Height());
      du_tvec.SetSize(fes.GetProlongationMatrix()->Width());

      gradient = new PLaplacianGradientOperator(*this);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // T -> L
      fes.GetProlongationMatrix()->Mult(x, u);

      // L -> Q

      // [vdim, dim, num_qp, num_el]
      auto grad_u_qp = gradient_wrt_x(u, integration_rule);

      // Q -> Q
      // auto plap = [](Eigen::Matrix2<double> grad_u)
      auto plap = [](tensor<double, 2> grad_u)
      {
         // grad_u := [dudx dudy]
         const int p = 4;
         return (pow(norm(grad_u), 0.5 * (p - 2)) * grad_u);
      };

      auto f_grad_u_qp = forall(plap, integration_rule.GetNPoints() * mesh->GetNE(),
                                grad_u_qp);

      // Layout of f_grad_u_qp_flat has to be [vdim, dim, num_qp, num_el]
      Vector f_grad_u_qp_flat((double *)f_grad_u_qp.GetData(),
                              1 * 2 * integration_rule.GetNPoints() *
                              mesh->GetNE());

      Vector f_grad_u_grad_phi = integrate_basis_gradient(f_grad_u_qp_flat,
                                                          fes,
                                                          integration_rule);
      // L -> T
      fes.GetProlongationMatrix()->MultTranspose(f_grad_u_grad_phi, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      // T -> L
      fes.GetProlongationMatrix()->Mult(x, state_lvec);

      return *gradient;
   }

   // dX: current iterate
   // Y: dR/dU * dX
   void GradientMult(const Vector &X, Vector &Y) const override
   {
      // apply essential bcs
      du_tvec = X;
      du_tvec.SetSubVector(ess_tdof_list, 0.0);
      du.SetFromTrueDofs(du_tvec);
      u = state_lvec;

      // u.Print(out, u.Size());
      // du_tvec.Print(out, du_tvec.Size());

      auto grad_u_qp = gradient_wrt_x(u, integration_rule);
      auto grad_du_qp = gradient_wrt_x(du, integration_rule);

      const int N = mesh->GetNE() * integration_rule.GetNPoints();

      // auto plap = [](tensor<dual<double, double>, 2> grad_u)
      // {
      //    const int p = 4;
      //    return grad_u * grad_u[0];
      // };

      // auto flux_qp = forall([&](tensor<double,2> grad_u, tensor<double,2> grad_du)
      // {

      //    return make_tensor<2>([&](int i) { return r[i].gradient; });
      // }, N, grad_u_qp, grad_du_qp);

      ////////////
      auto plap2 = [](tensor<double, 2> &grad_u)
      {
         // grad_u := [dudx dudy]
         const int p = 4;
         return (pow(norm(grad_u), 0.5 * (p - 2)) * grad_u);
      };

      auto flux_qp = forall(fwddiff(+plap2), N, grad_u_qp, grad_du_qp);
      // auto plap2_diff = fwddiff(+plap2);

      // auto flux_qp = forall([&, plap2](tensor<double,2> grad_u,
      //                                  tensor<double,2> grad_du)
      // {
      //    auto r = __enzyme_fwddiff<tensor<double, 2>>(+plap2, enzyme_dup, &grad_u,
      //                                                 &grad_du);
      //    return r;
      // }, N, grad_u_qp, grad_du_qp);
      ///////////////

      ///////////////
      // auto flux_qp2 = forall_gradient_action([&, plap2](tensor<double,2> grad_u,
      //                                                   tensor<double,2> grad_du)
      // {
      //    tensor<double, 2> (*f)(tensor<double, 2>&) = plap2;
      //    auto r = __enzyme_fwddiff<tensor<double, 2>>(f, enzyme_dup, &grad_u,
      //                                                 &grad_du);
      //    return r;
      // }, N, grad_u_qp, grad_du_qp);
      ///////////////

      // has to be [vdim, dim, num_qp, num_el]
      Vector flux_qp_flat((double *)flux_qp.GetData(), 1 * 2 * N);

      Vector y = integrate_basis_gradient(flux_qp_flat, fes, integration_rule);

      // L-vector to T-vector
      fes.GetProlongationMatrix()->MultTranspose(y, Y);

      // Re-assign the essential degrees of freedom on the final output vector.
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         Y[ess_tdof_list[i]] = X[ess_tdof_list[i]];
      }
   }

   ~PLaplacianOperator()
   {
      delete integral;
   }

   ParMesh *mesh;
   ParFiniteElementSpace &fes;
   ParGridFunction &u;
   mutable ParGridFunction du;
   IntegrationRule &integration_rule;
   Integral *integral;
   Array<int> ess_tdof_list;
   mutable Vector x_lvec, y_lvec, state_lvec, du_tvec;

   PLaplacianGradientOperator *gradient = nullptr;
};

void run_problem6()
{
   const int dim = 2;

   Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL, false,
                                     2.0*M_PI,
                                     2.0*M_PI);
   mesh.EnsureNodes();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   IntegrationRule integration_rule = IntRules.Get(Element::QUADRILATERAL,
                                                   2 * pmesh.GetNodes()->FESpace()->GetOrder(0) + 1);

   H1_FECollection h1fec(2);
   ParFiniteElementSpace h1fes(&pmesh, &h1fec);

   ParGridFunction u(&h1fes);

   PLaplacianOperator plap(h1fes, u);

   CGSolver cg(MPI_COMM_WORLD);
   cg.iterative_mode = false;
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(2);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetPreconditioner(cg);
   newton.SetOperator(plap);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(1000);
   newton.SetPrintLevel(1);

   Vector zero;
   Vector *u_tdof = u.GetTrueDofs();
   u_tdof->Randomize(1245);
   newton.Mult(zero, *u_tdof);
}

class ElasticityOperator : public Operator
{
public:
   ElasticityOperator(ParFiniteElementSpace &fes, ParGridFunction &u) :
      Operator(fes.GetTrueVSize()),
      mesh(fes.GetParMesh()),
      fes(fes),
      u(u),
      integration_rule(&IntRules.Get(Element::QUADRILATERAL,
                                     2 * mesh->GetNodes()->FESpace()->GetOrder(0) + 1))
   {
      MFEM_ASSERT(integration_rule != nullptr, "IntegrationRule failed");

      integral = new Integral({&u}, {&u});

      int InputID_Displacement = 0,
          OutputID_Displacement = 0;

      // A_Q
      // auto elasticity = [](double *grad_u, double *grad_v)
      // auto elasticity = [](Inputs, double *grad_u, Outputs, double *grad_v)
      // auto elasticity = [](Input(double *) grad_u, Output(double *) grad_v)
      auto elasticity = [](tensor<double, 2, 2> grad_u)
      {
         constexpr int dim = 2;
         auto I = mfem::internal::Identity<2>();

         double lambda = 100.0, mu = 50.0;

         auto epsilon = 0.5 * (grad_u + transpose(grad_u));
         auto sigma = lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
         return sigma; // (grad(v), sigma)
      };

      // std::tuple
      // {
      //    grad(InputID_Displacement),
      //    auto elasticity_alt = [](Eigen::Matrix2d grad_u)
      //    {
      //       constexpr int dim = 2;
      //       auto I = Eigen::Matrix2d::Identity(dim, dim);

      //       double lambda = 100.0, mu = 50.0;

      //       auto epsilon = 0.5 * (grad_u + grad_u.transpose());
      //       auto sigma = lambda * epsilon.trace() * I + 2.0 * mu * epsilon;

      //       return sigma;
      //    }
      // };

      integral->AddKernel("elasticity", elasticity,
      {grad(InputID_Displacement)},
      {grad(OutputID_Displacement)},
      *mesh,
      integration_rule);

      Array<int> ess_bdr(mesh->bdr_attributes.Max());

      ess_bdr = 0;
      ess_bdr[1] = 1;
      VectorFunctionCoefficient bdr(mesh->Dimension(), [](const Vector &x,
                                                          Vector &u)
      {
         u(0) = 0.0;
         u(1) = -1.0;
      });
      u.ProjectBdrCoefficient(bdr, ess_bdr);

      ess_bdr[3] = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      x_lvec.SetSize(fes.GetProlongationMatrix()->Height());
      x_lvec = 0.0;
      y_lvec.SetSize(fes.GetProlongationMatrix()->Height());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // T -> L
      fes.GetProlongationMatrix()->Mult(x, x_lvec);

      // L -> Q <-> Q -> L
      y_lvec = 0.0;
      integral->CallKernel("elasticity", x_lvec, y_lvec, *mesh);

      // L -> T
      fes.GetProlongationMatrix()->MultTranspose(y_lvec, y);

      y.SetSubVector(ess_tdof_list, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      return const_cast<ElasticityOperator &>(*this);
   }

   ~ElasticityOperator()
   {
      delete integral;
   }

   ParMesh *mesh;
   ParFiniteElementSpace &fes;
   ParGridFunction &u;
   const IntegrationRule *integration_rule;
   Integral *integral;
   Array<int> ess_tdof_list;
   mutable Vector x_lvec, y_lvec;
};

void run_problem7()
{
   const int dim = 2;

   Mesh mesh = Mesh::MakeCartesian2D(6, 2, Element::QUADRILATERAL, false,
                                     6.0, 1.0);
   mesh.EnsureNodes();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   H1_FECollection h1fec(2);
   ParFiniteElementSpace h1fes(&pmesh, &h1fec, dim);

   ParGridFunction u(&h1fes);
   u = 0.0;

   ElasticityOperator dop(h1fes, u);

   CGSolver cg(MPI_COMM_WORLD);
   cg.iterative_mode = false;
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(2);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetPreconditioner(cg);
   newton.SetOperator(dop);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(1000);
   newton.SetPrintLevel(1);

   Vector zero;
   Vector *u_tdof = u.GetTrueDofs();
   newton.Mult(zero, *u_tdof);

   u.SetFromTrueDofs(*u_tdof);

   char vishost[] = "128.15.198.77";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   // sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << u << std::flush;
}