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

#include <cstddef>

#include "../linalg/hypre.hpp"

#include "interpolate.hpp"
#include "integrate.hpp"
#include "qfunction.hpp"
#include "util.hpp"

namespace mfem
{

///////////////////////////////////////////////////////////////////////////////
template <class T>
inline std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 15.0 * std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0) { return neg < eps; }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

///////////////////////////////////////////////////////////////////////////////
using da_hypre_parmatrix_callback_t =
   std::function<void(std::vector<Vector> &, HypreParMatrix &)>;

using da_callback_t =
   std::map<size_t, std::vector<da_hypre_parmatrix_callback_t>>;

///////////////////////////////////////////////////////////////////////////////
template<size_t num_fields,
         size_t num_inputs,
         size_t num_outputs,
         typename qfunc_t,
         typename... input_ts,
         typename... output_ts,
         typename derivative_ids_t>
void callback_derivatives_assembly(qfunc_t &qfunc,
                                   mfem::tuple<input_ts...> &inputs,
                                   mfem::tuple<output_ts...> &outputs,
                                   std::vector<FieldDescriptor> &fields,
                                   const std::array<int, num_inputs> &input_to_field,
                                   const std::array<int, num_outputs> &output_to_field,
                                   std::array<DofToQuadMap, num_inputs> &input_dtq_maps,
                                   std::array<DofToQuadMap, num_outputs> &output_dtq_maps,
                                   const bool use_sum_factorization,
                                   const int num_entities,
                                   const int num_elements,
                                   const int num_qp,
                                   const int test_vdim,
                                   const int test_op_dim,
                                   const int num_test_dof,
                                   const int dimension,
                                   const int q1d,
                                   const std::vector<int> &input_size_on_qp,
                                   const int residual_size_on_qp,
                                   const ElementDofOrdering element_dof_ordering,
                                   const std::unordered_map<int, std::array<bool, num_inputs>> &dependency_map,
                                   const std::vector<int> &inputs_vdim,
                                   const size_t test_space_field_idx,
                                   const DeviceTensor<1, const double> &ir_weights,
                                   const derivative_ids_t derivative_ids,
                                   da_callback_t &assemble_derivative_hypreparmatrix_callbacks)
{
   using entity_t = Entity::Element;
   using qf_param_ts =
      typename create_function_signature<decltype(&qfunc_t::operator())>::type::parameter_ts;
   const auto output_fop = mfem::get<0>(outputs);

   for_constexpr([&](auto derivative_id)
   {
      dbg("[derivative][assembly] id: {}", derivative_id.value);
      // Field index of the derivative
      const size_t d_field_idx = FindIdx(derivative_id, fields);

      // First Input index of the derivative
      const size_t d_input_idx = [d_field_idx, &input_to_field]
      {
         for (size_t i = 0; i < input_to_field.size(); i++)
         {
            if (input_to_field[i] == d_field_idx)
            {
               return i;
            }
         }
         return size_t(SIZE_MAX);
      }();
      dbg("[derivative][assembly] d_field_idx:{} d_input_idx:{}",
          d_field_idx, d_input_idx);

      auto shmem_info =
         get_shmem_info<entity_t, num_fields, num_inputs, num_outputs>
         (input_dtq_maps, output_dtq_maps, fields, num_entities, inputs, num_qp,
          input_size_on_qp, residual_size_on_qp, element_dof_ordering, d_field_idx);

      Vector shmem_cache(shmem_info.total_size);

      const auto input_is_dependent = dependency_map.at(derivative_id);

      const int trial_vdim = GetVDim(fields[d_field_idx]);

      const int num_trial_dof_1d =
         input_dtq_maps[d_input_idx].B.GetShape()[DofToQuadMap::Index::DOF];

      const int num_trial_dof =
         get_restriction<entity_t>(fields[d_field_idx], element_dof_ordering)->Height() /
         inputs_vdim[d_input_idx] / num_entities;
      dbg("[derivative][assembly] trial_vdim:{} num_trial_dof_1d:{} num_trial_dof:{}",
          trial_vdim, num_trial_dof_1d, num_trial_dof);

      int total_trial_op_dim = 0;
      for_constexpr<num_inputs>([&](auto s)
      {
         if (input_is_dependent[s] == false) { return; }
         total_trial_op_dim += input_size_on_qp[s] / mfem::get<s>(inputs).vdim;
      });

      const int da_size_on_qp =
         GetSizeOnQP<entity_t>(output_fop, fields[test_space_field_idx]);
      dbg("[derivative][assembly] da_size_on_qp:{}", da_size_on_qp);

      assemble_derivative_hypreparmatrix_callbacks[derivative_id].push_back(
         [=, fields = fields] (std::vector<Vector> &fields_e,
                               HypreParMatrix &A) mutable
      {
         dbg("\x1b[35m[derivative][assembly][callback] id:{}", derivative_id.value);
         Vector direction_e(get_restriction<entity_t>(fields[d_field_idx],
                                                      element_dof_ordering)->Height());

         auto shmem = shmem_cache.ReadWrite();
         auto wrapped_fields_e = wrap_fields(fields_e, shmem_info.field_sizes,
                                             num_entities);
         auto wrapped_direction_e = Reshape(direction_e.ReadWrite(),
                                            shmem_info.direction_size, num_entities);

         Vector a_qp_mem(test_vdim * test_op_dim * trial_vdim * total_trial_op_dim *
                         num_qp * num_elements);
         auto a_qp = Reshape(a_qp_mem.ReadWrite(), test_vdim, test_op_dim,
                             trial_vdim, total_trial_op_dim, num_qp, num_elements);

         Vector Ae_mem(num_test_dof * test_vdim * num_trial_dof * trial_vdim * num_elements);
         Ae_mem = 0.0;

         auto A_e = Reshape(Ae_mem.ReadWrite(), num_test_dof, test_vdim, num_trial_dof,
                            trial_vdim, num_elements);

         dbg("\x1b[35m[derivative][assembly][callback] For E-loop:{}", num_elements);
         for (int e = 0; e < num_elements; e++)
         {
            auto [input_dtq_shmem, output_dtq_shmem, fields_shmem, direction_shmem,
                                   input_shmem, shadow_shmem, residual_shmem, scratch_shmem] =
            unpack_shmem(shmem, shmem_info, input_dtq_maps,
                         output_dtq_maps, wrapped_fields_e, wrapped_direction_e, num_qp, e);

            // interpolate
            map_fields_to_quadrature_data(input_shmem,
                                          fields_shmem,
                                          input_dtq_shmem,
                                          input_to_field,
                                          inputs,
                                          ir_weights,
                                          scratch_shmem,
                                          dimension,
                                          use_sum_factorization);

            set_zero(shadow_shmem);

            for (int q = 0; q < num_qp; q++)
            {
               for (int j = 0; j < trial_vdim; j++)
               {
                  size_t m_offset = 0;
                  for_constexpr_with_arg([&](auto s, auto&& input_fop)
                  {
                     if (input_is_dependent[s] == false) { return; }

                     auto trial_op_dim = input_size_on_qp[s] / mfem::get<s>(inputs).vdim;

                     auto d_qp = Reshape(&(shadow_shmem[s])[0], trial_vdim, trial_op_dim, num_qp);
                     for (int m = 0; m < trial_op_dim; m++)
                     {
                        d_qp(j, m, q) = 1.0;

                        auto r = Reshape(&residual_shmem(0, q), da_size_on_qp);
                        auto qf_args = decay_tuple<qf_param_ts> {};
#ifdef MFEM_USE_ENZYME
                        auto qf_shadow_args = decay_tuple<qf_param_ts> {};
                        apply_kernel_fwddiff_enzyme(r, qfunc, qf_args, qf_shadow_args, input_shmem,
                                                    shadow_shmem, q);
#else
                        apply_kernel_native_dual(r, qfunc, qf_args, input_shmem, shadow_shmem, q);
#endif
                        d_qp(j, m, q) = 0.0;

                        auto f = Reshape(&r(0), test_vdim, test_op_dim);
                        for (int i = 0; i < test_vdim; i++)
                        {
                           for (int k = 0; k < test_op_dim; k++)
                           {
                              a_qp(i, k, j, m + m_offset, q, e) = f(i, k);
                           }
                        }
                     }

                     m_offset += trial_op_dim;
                  }, inputs);
               }
            }

            Vector fhat_mem(test_vdim * test_op_dim * num_qp);
            auto fhat = Reshape(fhat_mem.ReadWrite(), test_vdim, test_op_dim, num_qp);
            if (use_sum_factorization)
            {
               if (dimension == 2)
               {
                  for (int Jx = 0; Jx < num_trial_dof_1d; Jx++)
                  {
                     for (int Jy = 0; Jy < num_trial_dof_1d; Jy++)
                     {
                        const int J = Jy + Jx * num_trial_dof_1d;

                        for (int j = 0; j < trial_vdim; j++)
                        {
                           fhat_mem = 0.0;
                           int m_offset = 0;
                           for_constexpr_with_arg([&](auto s, auto&& input_fop)
                           {
                              if (input_is_dependent[s] == false) { return; }

                              int trial_op_dim = input_size_on_qp[s] / mfem::get<s>(inputs).vdim;

                              auto &B = input_dtq_maps[s].B;
                              auto &G = input_dtq_maps[s].G;

                              if constexpr (is_value_fop<std::decay_t<decltype(input_fop)>>::value)
                              {
                                 for (int qx = 0; qx < q1d; qx++)
                                 {
                                    for (int qy = 0; qy < q1d; qy++)
                                    {
                                       const int q = qy + qx * q1d;
                                       for (int m = 0; m < trial_op_dim; m++)
                                       {
                                          for (int i = 0; i < test_vdim; i++)
                                          {
                                             for (int k = 0; k < test_op_dim; k++)
                                             {
                                                const real_t f = a_qp(i, k, j, m + m_offset, q, e);
                                                fhat(i, k, q) += f * B(qx, 0, Jx) * B(qy, 0, Jy);
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                              else if constexpr (is_gradient_fop<std::decay_t<decltype(input_fop)>>::value)
                              {
                                 for (int qx = 0; qx < q1d; qx++)
                                 {
                                    for (int qy = 0; qy < q1d; qy++)
                                    {
                                       const int q = qy + qx * q1d;
                                       for (int m = 0; m < trial_op_dim; m++)
                                       {
                                          for (int i = 0; i < test_vdim; i++)
                                          {
                                             for (int k = 0; k < test_op_dim; k++)
                                             {
                                                const real_t f = a_qp(i, k, j, m + m_offset, q, e);
                                                if (m == 0)
                                                {
                                                   fhat(i, k, q) += f * B(qx, 0, Jx) * G(qy, 0, Jy);
                                                }
                                                else
                                                {
                                                   fhat(i, k, q) += f * G(qx, 0, Jx) * B(qy, 0, Jy);
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                              else
                              {
                                 MFEM_ABORT("sum factorized sparse matrix assemble routine "
                                            "not implemented for field operator");
                              }
                              m_offset += trial_op_dim;
                           }, inputs);

                           auto bvtfhat = Reshape(&A_e(0, 0, J, j, e), num_test_dof, test_vdim);
                           // integrate
                           map_quadrature_data_to_fields(bvtfhat,
                                                         fhat,
                                                         output_fop,
                                                         output_dtq_shmem[0],
                                                         scratch_shmem,
                                                         dimension,
                                                         use_sum_factorization);
                        }
                     }
                  }
               }
               else
               {
                  MFEM_ABORT("sum factorized sparse matrix assemble routine "
                             "not implemented for 3D");
               }
            }
            else // use_sum_factorization
            {
               assert(false && "❌❌ Sum factorization required ❌❌");
            }
         }

         bool same_test_and_trial = false;
         for (int s = 0; s < num_inputs; s++)
         {
            if (input_is_dependent[s])
            {
               if (output_to_field[0] == input_to_field[s])
               {
                  same_test_and_trial = true;
                  break;
               }
            }
         }

         FieldDescriptor *trial_field = nullptr;
         for (int s = 0; s < num_inputs; s++)
         {
            if (input_is_dependent[s])
            {
               trial_field = &fields[input_to_field[s]];
            }
         }

         auto trial_fes = *std::get_if<const ParFiniteElementSpace *>(&trial_field->data);
         auto test_fes = *std::get_if<const ParFiniteElementSpace *>(&fields[output_to_field[0]].data);
         assert(trial_fes && test_fes);
         SparseMatrix mat(test_fes->GetVSize(), trial_fes->GetVSize());

         //  if (test_fes == nullptr) { MFEM_ABORT("internal error"); }
         MFEM_VERIFY(test_fes, "internal error");

         // if (same_test_and_trial && use_sum_factorization)
         // {
         //    const ElementRestriction &rest =
         //       static_cast<const ElementRestriction&>(
         //          *test_fes->GetElementRestriction(element_dof_ordering));
         //    rest.FillSparseMatrix(Ae_mem, mat);
         // }
         // else
         {
            for (int e = 0; e < num_elements; e++)
            {
               auto tmp = Reshape(Ae_mem.ReadWrite(), num_test_dof * test_vdim,
                                  num_trial_dof * trial_vdim, num_elements);
               DenseMatrix Ae(&tmp(0, 0, e), num_test_dof * test_vdim,
                              num_trial_dof * trial_vdim);

               Array<int> test_vdofs, trial_vdofs;
               test_fes->GetElementVDofs(e, test_vdofs);
               GetElementVDofs(*trial_field, e, trial_vdofs);

               if (use_sum_factorization)
               {
                  Array<int> test_vdofs_mapped(test_vdofs.Size()),
                  trial_vdofs_mapped(trial_vdofs.Size());

                  const Array<int> &test_dofmap =
                  dynamic_cast<const TensorBasisElement&>(*test_fes->GetFE(0)).GetDofMap();

                  if (test_dofmap.Size() == 0)
                  {
                     test_vdofs_mapped = test_vdofs;
                  }
                  else
                  {
                     MFEM_VERIFY(test_dofmap.Size() == num_test_dof,
                                 "internal error: dof map of the test space does not "
                                 "match previously determined number of test space dofs");

                     for (int vd = 0; vd < test_vdim; vd++)
                     {
                        for (int i = 0; i < num_test_dof; i++)
                        {
                           test_vdofs_mapped[i + vd * num_test_dof] =
                              test_vdofs[test_dofmap[i] + vd * num_test_dof];
                        }
                     }
                  }

                  const Array<int> &trial_dofmap =
                     dynamic_cast<const TensorBasisElement&>(*trial_fes->GetFE(0)).GetDofMap();

                  if (trial_dofmap.Size() == 0)
                  {
                     trial_vdofs_mapped = trial_vdofs;
                  }
                  else
                  {
                     MFEM_VERIFY(trial_dofmap.Size() == num_trial_dof,
                                 "internal error: dof map of the test space does not "
                                 "match previously determined number of test space dofs");

                     for (int vd = 0; vd < trial_vdim; vd++)
                     {
                        for (int i = 0; i < num_trial_dof; i++)
                        {
                           trial_vdofs_mapped[i + vd * num_trial_dof] =
                              trial_vdofs[trial_dofmap[i] + vd * num_trial_dof];
                        }
                     }
                  }

                  mat.AddSubMatrix(test_vdofs_mapped, trial_vdofs_mapped, Ae, 1);
               }
               else
               {
                  assert(false);
                  //   mat.AddSubMatrix(test_vdofs, trial_vdofs, Ae, 1);
               }
            }
         }
         mat.Finalize();

         if (same_test_and_trial)
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               &mat);
            A = *RAP(&tmp, test_fes->Dof_TrueDof_Matrix());
         }
         else
         {
            HypreParMatrix tmp(test_fes->GetComm(),
                               test_fes->GlobalVSize(),
                               trial_fes->GlobalVSize(),
                               test_fes->GetDofOffsets(),
                               trial_fes->GetDofOffsets(),
                               &mat);
            A = *RAP(test_fes->Dof_TrueDof_Matrix(), &tmp, trial_fes->Dof_TrueDof_Matrix());
         }
      });
   }, derivative_ids);
}

} // namespace mfem