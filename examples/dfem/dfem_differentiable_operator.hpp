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
#include <type_traits>
#include "dfem_fieldoperator.hpp"
#include "dfem_parametricspace.hpp"
#include "tuple.hpp"
#include <linalg/tensor.hpp>
#include <enzyme/utils>
#include <enzyme/enzyme>
#include "dfem_util.hpp"

namespace mfem
{

using mult_func_t = std::function<void(Vector &)>;

template <
   typename kernels_tuple,
   size_t num_solutions,
   size_t num_parameters,
   size_t num_fields = num_solutions + num_parameters,
   size_t num_kernels = mfem::tuple_size<kernels_tuple>::value
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
         (create_action_callback(mfem::get<idx>(ks), funcs[idx]), ...);
      }

      Action(DifferentiableOperator &op, kernels_tuple &ks) : op(op)
      {
         materialize_callbacks(ks, funcs,
                               std::make_index_sequence<mfem::tuple_size<kernels_tuple>::value>());
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
            p[i]->Read();
            parameters_l[i] = *p[i];
            // parameters_l[i].MakeRef(p[i], 0, p[i]->Size());
         }
      }

   protected:
      DifferentiableOperator &op;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> prolongation_transpose;

      mutable std::array<Vector, num_solutions> solutions_l;
      mutable std::array<Vector, num_parameters> parameters_l;
      mutable Vector residual_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable Vector residual_e;
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
         (create_callback(mfem::get<idx>(ks), funcs[idx]), ...);
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
         direction = op.fields[derivative_idx];

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
         current_direction_t = x;
         current_direction_t.SetSubVector(op.ess_tdof_list, 0.0);

         prolongation(direction, current_direction_t, direction_l);

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
         (assemble_vector_impl(mfem::get<idx>(ks), v), ...);
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
         (assemble_hypreparmatrix_impl(mfem::get<idx>(ks), A), ...);
      }

      void Assemble(HypreParMatrix &A)
      {
         assemble_hypreparmatrix(ks, A, std::make_index_sequence<num_kernels>());
      }

      void AssembleDiagonal(Vector &d) const override {}

   protected:
      DifferentiableOperator &op;
      kernels_tuple &ks;
      std::array<mult_func_t, num_kernels> funcs;

      std::function<void(Vector &, Vector &)> prolongation_transpose;

      FieldDescriptor direction;

      std::array<Vector, num_solutions> solutions_l;
      std::array<Vector, num_parameters> parameters_l;
      mutable Vector direction_l;
      mutable Vector derivative_action_l;

      mutable std::array<Vector, num_fields> fields_e;
      mutable Vector direction_e;
      mutable Vector derivative_action_e;

      mutable Vector current_direction_t;
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

   void SetEssentialTrueDofs(const Array<int> &l)
   {
      l.Copy(ess_tdof_list);
   }

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

   static constexpr ElementDofOrdering element_dof_ordering =
      ElementDofOrdering::LEXICOGRAPHIC;

   static constexpr DofToQuad::Mode doftoquad_mode =
      DofToQuad::Mode::TENSOR;

   // static constexpr ElementDofOrdering element_dof_ordering =
   //    ElementDofOrdering::NATIVE;

   // static constexpr DofToQuad::Mode doftoquad_mode =
   //    DofToQuad::Mode::FULL;

   std::shared_ptr<Action> residual;
};

template <
   typename kernels_tuple,
   size_t num_solutions,
   size_t num_parameters,
   size_t num_fields,
   size_t num_kernels
   >
template <
   typename kernel_t
   >
void DifferentiableOperator<kernels_tuple,
     num_solutions,
     num_parameters,
     num_fields,
     num_kernels>::Action::create_action_callback(
        kernel_t kernel,
        mult_func_t &func)
{
   using entity_t = typename kernel_t::entity_t;

   auto kinput_to_field = create_descriptors_to_fields_map<entity_t>(op.fields,
                                                                     kernel.inputs, std::make_index_sequence<kernel.num_kinputs> {});

   auto koutput_to_field = create_descriptors_to_fields_map<entity_t>(op.fields,
                                                                      kernel.outputs, std::make_index_sequence<kernel.num_koutputs> {});

   constexpr int hardcoded_output_idx = 0;
   const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

   const Operator *R = get_restriction<entity_t>(op.fields[test_space_field_idx],
                                                 element_dof_ordering);

   auto output_fop = mfem::get<hardcoded_output_idx>(kernel.outputs);

   const int num_elements = GetNumEntities<Entity::Element>(op.mesh);
   const int num_entities = GetNumEntities<entity_t>(op.mesh);
   const int num_qp = op.integration_rule.GetNPoints();

   // All solutions T-vector sizes make up the width of the operator, since
   // they are explicitly provided in Mult() for example.

   op.width = GetTrueVSize(op.fields[test_space_field_idx]);
   op.residual_lsize = GetVSize(op.fields[test_space_field_idx]);

   if constexpr (std::is_same_v<decltype(output_fop), One>)
   {
      op.height = 1;
   }
   else
   {
      op.height = op.residual_lsize;
   }

   residual_l.SetSize(op.residual_lsize);

   // assume only a single element type for now
   std::vector<const DofToQuad*> dtq;
   for (const auto &field : op.fields)
   {
      dtq.emplace_back(GetDofToQuad<entity_t>(field, op.integration_rule,
                                              doftoquad_mode));
   }
   const int q1d = (int)floor(pow(num_qp, 1.0/op.mesh.Dimension()) + 0.5);

   residual_e.SetSize(R->Height());

   const int residual_size_on_qp = GetSizeOnQP<entity_t>(
                                      mfem::get<hardcoded_output_idx>(kernel.outputs),
                                      op.fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(kernel.inputs, dtq,
                                                   kinput_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(kernel.outputs, dtq,
                                                    koutput_to_field);

   auto input_fops = create_bare_fops(kernel.inputs);
   auto output_fops = create_bare_fops(kernel.outputs);

   const int test_vdim = mfem::get<hardcoded_output_idx>(output_fops).vdim;
   const int test_op_dim =
      mfem::get<hardcoded_output_idx>(output_fops).size_on_qp /
      mfem::get<hardcoded_output_idx>(output_fops).vdim;
   const int num_test_dof = R->Height() /
                            mfem::get<hardcoded_output_idx>(output_fops).vdim /
                            num_entities;

   auto ir_weights = Reshape(this->op.integration_rule.GetWeights().Read(),
                             num_qp);

   auto input_size_on_qp = get_input_size_on_qp(kernel.inputs,
                                                std::make_index_sequence<kernel.num_kinputs> {});

   auto shmem_info = get_shmem_info<entity_t>(input_dtq_maps,
                                              output_dtq_maps,
                                              op.fields,
                                              num_entities,
                                              kernel.inputs,
                                              num_qp,
                                              input_size_on_qp,
                                              residual_size_on_qp);

   Vector shmem_cache(shmem_info.total_size);

   print_shared_memory_info(shmem_info);

   func = [=](Vector &ye_mem) mutable
   {
      restriction<entity_t>(op.solutions, solutions_l, this->fields_e,
                            op.element_dof_ordering);
      restriction<entity_t>(op.parameters, parameters_l, this->fields_e,
                            op.element_dof_ordering,
                            op.solutions.size());

      auto ye = Reshape(ye_mem.ReadWrite(), test_vdim, num_test_dof, num_entities);
      auto wrapped_fields_e = wrap_fields(this->fields_e, shmem_info.field_sizes, num_entities);

      forall([=] MFEM_HOST_DEVICE (int e, void *shmem)
      {
         // printf("\ne: %d\n", e);
         // tic();
         auto input_dtq_shmem = load_dtq_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
            shmem_info.input_dtq_sizes,
            input_dtq_maps);

         auto output_dtq_shmem = load_dtq_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
            shmem_info.output_dtq_sizes,
            output_dtq_maps);

         auto fields_shmem = load_field_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::FIELD],
            shmem_info.field_sizes,
            kinput_to_field,
            wrapped_fields_e,
            e);

         // These methods don't copy, they simply create a `DeviceTensor` object
         // that points to correct chunks of the shared memory pool.
         auto input_shmem = load_input_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::INPUT],
            shmem_info.input_sizes,
            num_qp);

         auto residual_shmem = load_residual_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::OUTPUT],
            shmem_info.residual_size,
            num_qp);

         auto scratch_mem = load_scratch_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::TEMP],
            shmem_info.temp_sizes);

         MFEM_SYNC_THREAD;
         // printf("shmem load elapsed: %.1fus\n", toc() * 1e6);

         // tic();
         map_fields_to_quadrature_data<TensorProduct>(
            input_shmem, fields_shmem, input_dtq_shmem, input_fops, ir_weights, scratch_mem,
         std::make_index_sequence<kernel.num_kinputs> {});
         // printf("interpolate elapsed: %.1fus\n", toc() * 1e6);

         // tic();
         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);
                  auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};
                  auto r = Reshape(&residual_shmem(0, q), residual_size_on_qp);
                  apply_kernel(r, kernel.func, kernel_args, input_shmem, q);
               }
            }
         }
         MFEM_SYNC_THREAD;
         // printf("qf elapsed: %.1fus\n", toc() * 1e6);

         // tic();
         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields<TensorProduct>(y, fhat,
                                                      mfem::get<0>(output_fops),
                                                      output_dtq_shmem[hardcoded_output_idx],
                                                      scratch_mem);
         // printf("integrate elapsed: %.1fus\n", toc() * 1e6);

      }, num_entities, q1d, q1d, q1d, shmem_info.total_size, shmem_cache.ReadWrite());

      if constexpr (std::is_same_v<decltype(output_fop), None>)
      {
         residual_l = ye_mem;
      }
      else
      {
         R->MultTranspose(ye_mem, residual_l);
      }
   };

   if constexpr (std::is_same_v<decltype(output_fop), None>)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         y = r_local;
      };
   }
   else if constexpr (std::is_same_v<decltype(output_fop), One>)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         double local_sum = r_local.Sum();
         MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM,
                       op.mesh.GetComm());
         MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
      };
   }
   else
   {
      auto P = get_prolongation(op.fields[test_space_field_idx]);
      prolongation_transpose = [P](const Vector &r_local, Vector &y)
      {
         P->MultTranspose(r_local, y);
      };
   }
}

template <
   typename kernels_tuple,
   size_t num_solutions,
   size_t num_parameters,
   size_t num_fields,
   size_t num_kernels
   >
template <
   size_t derivative_idx
   >
template <
   typename kernel_t
   >
void DifferentiableOperator<kernels_tuple,
     num_solutions,
     num_parameters,
     num_fields,
     num_kernels>::Derivative<derivative_idx>::create_callback(kernel_t kernel,
                                                               mult_func_t &func)
{
   using entity_t = typename kernel_t::entity_t;

   auto kinput_to_field = create_descriptors_to_fields_map<entity_t>(op.fields,
                                                                     kernel.inputs, std::make_index_sequence<kernel.num_kinputs> {});

   auto koutput_to_field = create_descriptors_to_fields_map<entity_t>(op.fields,
                                                                      kernel.outputs, std::make_index_sequence<kernel.num_koutputs> {});

   constexpr int hardcoded_output_idx = 0;
   const int test_space_field_idx = koutput_to_field[hardcoded_output_idx];

   const Operator *R = get_restriction<entity_t>(op.fields[test_space_field_idx],
                                                 element_dof_ordering);

   auto output_fop = mfem::get<hardcoded_output_idx>(kernel.outputs);

   const int num_elements = GetNumEntities<Entity::Element>(op.mesh);
   const int num_entities = GetNumEntities<entity_t>(op.mesh);
   const int num_qp = op.integration_rule.GetNPoints();

   // assume only a single element type for now
   std::vector<const DofToQuad*> dtq;
   for (const auto &field : op.fields)
   {
      dtq.emplace_back(GetDofToQuad<entity_t>(field, op.integration_rule,
                                              doftoquad_mode));
   }
   const int q1d = dtq[0]->nqpt;

   derivative_action_e.SetSize(R->Height());

   const int da_size_on_qp = GetSizeOnQP<entity_t>(
                                mfem::get<hardcoded_output_idx>(kernel.outputs),
                                op.fields[test_space_field_idx]);

   auto input_dtq_maps = create_dtq_maps<entity_t>(kernel.inputs, dtq,
                                                   kinput_to_field);
   auto output_dtq_maps = create_dtq_maps<entity_t>(kernel.outputs, dtq,
                                                    koutput_to_field);

   auto input_fops = create_bare_fops(kernel.inputs);
   auto output_fops = create_bare_fops(kernel.outputs);

   const int test_vdim = mfem::get<hardcoded_output_idx>(output_fops).vdim;
   const int test_op_dim =
      mfem::get<hardcoded_output_idx>(output_fops).size_on_qp /
      mfem::get<hardcoded_output_idx>(output_fops).vdim;
   const int num_test_dof = R->Height() /
                            mfem::get<hardcoded_output_idx>(output_fops).vdim /
                            num_entities;

   auto ir_weights = Reshape(this->op.integration_rule.GetWeights().Read(),
                             num_qp);

   auto input_size_on_qp = get_input_size_on_qp(kernel.inputs,
                                                std::make_index_sequence<kernel.num_kinputs> {});

   // Check which qf inputs are dependent on the dependent variable
   std::array<bool, kernel.num_kinputs> kinput_is_dependent;
   bool no_kinput_is_dependent = true;
   for (int i = 0; i < kinput_is_dependent.size(); i++)
   {
      if (kinput_to_field[i] == derivative_idx)
      {
         no_kinput_is_dependent = false;
         kinput_is_dependent[i] = true;
         // out << "function input " << i << " is dependent on "
         //     << op.fields[kinput_to_field[i]].field_label << "\n";
      }
      else
      {
         kinput_is_dependent[i] = false;
      }
   }

   bool with_derivatives = true;
   auto shmem_info = get_shmem_info<entity_t>(input_dtq_maps,
                                              output_dtq_maps,
                                              op.fields,
                                              num_entities,
                                              kernel.inputs,
                                              num_qp,
                                              input_size_on_qp,
                                              da_size_on_qp,
                                              derivative_idx);

   Vector shmem_cache(shmem_info.total_size);

   print_shared_memory_info(shmem_info);

   func = [=](Vector &ye_mem) mutable
   {
      if (no_kinput_is_dependent)
      {
         return;
      }

      restriction<entity_t>(direction, direction_l, direction_e,
                            op.element_dof_ordering);

      auto ye = Reshape(ye_mem.ReadWrite(), num_test_dof, test_vdim, num_entities);
      auto wrapped_fields_e = wrap_fields(this->fields_e, shmem_info.field_sizes, num_entities);
      auto wrapped_direction_e = Reshape(direction_e.Read(), shmem_info.direction_size, num_entities);

      forall([=] MFEM_HOST_DEVICE (int e, double *shmem)
      {
         auto input_dtq_shmem = load_dtq_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::INPUT_DTQ],
            shmem_info.input_dtq_sizes,
            input_dtq_maps);

         auto output_dtq_shmem = load_dtq_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::OUTPUT_DTQ],
            shmem_info.output_dtq_sizes,
            output_dtq_maps);

         auto fields_shmem = load_field_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::FIELD],
            shmem_info.field_sizes,
            kinput_to_field,
            wrapped_fields_e,
            e);

         auto direction_shmem = load_direction_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::DIRECTION],
            shmem_info.direction_size,
            wrapped_direction_e,
            e);

         // These methods don't copy, they simply create a `DeviceTensor` object
         // that points to correct chunks of the shared memory pool.
         auto input_shmem = load_input_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::INPUT],
            shmem_info.input_sizes,
            num_qp);

         auto shadow_shmem = load_input_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::SHADOW],
            shmem_info.input_sizes,
            num_qp);

         auto residual_shmem = load_residual_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::OUTPUT],
            shmem_info.residual_size,
            num_qp);

         auto scratch_mem = load_scratch_mem(
            shmem,
            shmem_info.offsets[SharedMemory::Index::TEMP],
            shmem_info.temp_sizes);

         map_fields_to_quadrature_data<TensorProduct>(
            input_shmem, fields_shmem, input_dtq_shmem, input_fops, ir_weights, scratch_mem,
         std::make_index_sequence<kernel.num_kinputs> {});

         zero_all(shadow_shmem);
         map_direction_to_quadrature_data_conditional<TensorProduct>(
            shadow_shmem, direction_shmem, input_dtq_shmem, input_fops, ir_weights,
            scratch_mem, kinput_is_dependent,
            std::make_index_sequence<kernel.num_kinputs> {});

         MFEM_FOREACH_THREAD(qx, x, q1d)
         {
            MFEM_FOREACH_THREAD(qy, y, q1d)
            {
               MFEM_FOREACH_THREAD(qz, z, q1d)
               {
                  const int q = qx + q1d * (qy + q1d * qz);

                  auto kernel_args = decay_tuple<typename kernel_t::kf_param_ts> {};
                  auto kernel_shadow_args = decay_tuple<typename kernel_t::kf_param_ts> {};

                  auto r = Reshape(&residual_shmem(0, q), da_size_on_qp);
                  apply_kernel_fwddiff_enzyme(
                     r,
                     kernel.func,
                     kernel_args,
                     input_shmem,
                     kernel_shadow_args,
                     shadow_shmem,
                     q);
                  // printf(">>>>> WARNING: AD DISABLED\n");
               }
            }
         }
         MFEM_SYNC_THREAD;

         auto fhat = Reshape(&residual_shmem(0, 0), test_vdim, test_op_dim, num_qp);
         auto y = Reshape(&ye(0, 0, e), num_test_dof, test_vdim);
         map_quadrature_data_to_fields<TensorProduct>(y, fhat,
                                                      mfem::get<0>(output_fops),
                                                      output_dtq_shmem[hardcoded_output_idx],
                                                      scratch_mem);
      }, num_entities, q1d, q1d, 1, shmem_info.total_size, shmem_cache.ReadWrite());

      R->MultTranspose(ye_mem, derivative_action_l);
   };

   if constexpr (std::is_same_v<decltype(output_fop), One>)
   {
      prolongation_transpose = [&](Vector &r_local, Vector &y)
      {
         double local_sum = r_local.Sum();
         MPI_Allreduce(&local_sum, y.GetData(), 1, MPI_DOUBLE, MPI_SUM,
                       op.mesh.GetComm());
         MFEM_ASSERT(y.Size() == 1, "output size doesn't match kernel description");
      };
   }
   else
   {
      auto P = get_prolongation(op.fields[test_space_field_idx]);
      prolongation_transpose = [P](const Vector &r_local, Vector &y)
      {
         P->MultTranspose(r_local, y);
      };
   }
}

// #include "dfem_assemble_vector.icc"
// #include "dfem_assemble_hypreparmatrix.icc"

}
