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

#include "util.hpp"

namespace mfem::future
{

/// @brief Assemble element matrix for three dimensional data.
///
/// Note: In the below layouts, total_trial_op_dim is > 1 if
/// there are more than one inputs dependent on the derivative variable.
///
/// @param A Memory for one element matrix with layout
/// [test_ndof, test_vdim, trial_ndof, trial_vdim].
/// @param fhat Memory to hold the residual computation with layout
/// [test_vdim, test_op_dim, nqp].
/// @param qpdc The quadrature point data cache with data layout
/// [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, nqp].
/// @param itod Input Trial Operator Dimension array. If the trial
/// operator is not dependent, the dimension is 0 to indicate that.
/// @param inputs The input field operator types.
/// @param output The output field operator types.
/// @param input_dtqmaps The input DofToQuad maps.
/// @param output_dtqmap The output DofToQuad maps.
/// @param scratch_shmem Scratch shared memory for computations.
/// @param q1d The number of quadrature points in one dimension.
/// @param td1d The number of trial dofs in one dimension.
template <typename input_fop_ts, size_t num_inputs, typename output_fop_t>
MFEM_HOST_DEVICE void assemble_element_mat_t3d(
   const DeviceTensor<4, real_t>& A,
   const DeviceTensor<3, real_t>& fhat,
   const DeviceTensor<5, const real_t>& qpdc,
   const DeviceTensor<1, const real_t>& itod,
   const input_fop_ts& inputs,
   const output_fop_t& output,
   const std::array<DofToQuadMap, num_inputs>& input_dtqmaps,
   const DofToQuadMap& output_dtqmap,
   std::array<DeviceTensor<1>, 6>& scratch_shmem,
   const int& q1d,
   const int& td1d)
{
   constexpr int dimension = 3;

   // [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, num_qp]
   const int test_vdim = qpdc.GetShape()[0];
   const int test_op_dim = qpdc.GetShape()[1];
   const int trial_vdim = qpdc.GetShape()[2];

   // [num_test_dof, ...]
   const auto num_test_dof = A.GetShape()[0];

   for (int Jx = 0; Jx < td1d; Jx++)
   {
      for (int Jy = 0; Jy < td1d; Jy++)
      {
         for (int Jz = 0; Jz < td1d; Jz++)
         {
            const int J = Jx + td1d * (Jy + td1d * Jz);

            for (int j = 0; j < trial_vdim; j++)
            {
               for (int tv = 0; tv < test_vdim; tv++)
               {
                  for (int tod = 0; tod < test_op_dim; tod++)
                  {
                     MFEM_FOREACH_THREAD(qx, x, q1d)
                     {
                        MFEM_FOREACH_THREAD(qy, y, q1d)
                        {
                           MFEM_FOREACH_THREAD(qz, z, q1d)
                           {
                              const int q = qx + q1d * (qy + q1d * qz);
                              fhat(tv, tod, q) = 0.0;
                           }
                        }
                     }
                  }
               }

               // MSVC lambda capture workaround
               [[maybe_unused]] const auto& inputs_ref = inputs;

               int m_offset = 0;
               for_constexpr<num_inputs>([&](auto s)
               {
                  using fop_t = std::decay_t<decltype(get<s>(inputs_ref))>;

                  const int trial_op_dim = static_cast<int>(itod(static_cast<int>(s)));
                  if (trial_op_dim == 0)
                  {
                     // This is inside a lambda so we have to return
                     // instead of idiomatic 'continue'.
                     return;
                  }

                  auto& B = input_dtqmaps[s].B;
                  auto& G = input_dtqmaps[s].G;

                  if constexpr (is_value_fop<fop_t>::value)
                  {
                     MFEM_FOREACH_THREAD(qx, x, q1d)
                     {
                        MFEM_FOREACH_THREAD(qy, y, q1d)
                        {
                           MFEM_FOREACH_THREAD(qz, z, q1d)
                           {
                              const int q = qx + q1d * (qy + q1d * qz);

                              for (int m = 0; m < trial_op_dim; m++)
                              {
                                 for (int i = 0; i < test_vdim; i++)
                                 {
                                    for (int k = 0; k < test_op_dim; k++)
                                    {
                                       const real_t f = qpdc(i, k, j, m + m_offset, q);
                                       fhat(i, k, q) += f * B(qx, 0, Jx) * B(qy, 0, Jy) * B(qz, 0, Jz);
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
                  else if constexpr (is_gradient_fop<fop_t>::value)
                  {
                     MFEM_FOREACH_THREAD(qx, x, q1d)
                     {
                        MFEM_FOREACH_THREAD(qy, y, q1d)
                        {
                           MFEM_FOREACH_THREAD(qz, z, q1d)
                           {
                              const int q = qx + q1d * (qy + q1d * qz);
                              for (int m = 0; m < trial_op_dim; m++)
                              {
                                 for (int i = 0; i < test_vdim; i++)
                                 {
                                    for (int k = 0; k < test_op_dim; k++)
                                    {
                                       const real_t f = qpdc(i, k, j, m + m_offset, q);
                                       if (m == 0)
                                       {
                                          fhat(i, k, q) += f * G(qx, 0, Jx) * B(qy, 0, Jy) * B(qz, 0, Jz);
                                       }
                                       else if (m == 1)
                                       {
                                          fhat(i, k, q) += f * B(qx, 0, Jx) * G(qy, 0, Jy) * B(qz, 0, Jz);
                                       }
                                       else if (m == 2)
                                       {
                                          fhat(i, k, q) += f * B(qx, 0, Jx) * B(qy, 0, Jy) * G(qz, 0, Jz);
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
                  else
                  {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
                     MFEM_ABORT("sum factorized sparse matrix assemble routine "
                                "not implemented for field operator");
#endif
                  }
                  MFEM_SYNC_THREAD;
                  m_offset += trial_op_dim;
               });

               auto bvtfhat = Reshape(&A(0, 0, J, j), num_test_dof, test_vdim);
               map_quadrature_data_to_fields(bvtfhat, fhat, output, output_dtqmap,
                                             scratch_shmem, dimension, true);
            }
         }
      }
   }
}

/// @brief Assemble element matrix for two dimensional data.
///
/// Note: In the below layouts, total_trial_op_dim is > 1 if
/// there are more than one inputs dependent on the derivative variable.
///
/// @param A Memory for one element matrix with layout
/// [test_ndof, test_vdim, trial_ndof, trial_vdim].
/// @param fhat Memory to hold the residual computation with layout
/// [test_vdim, test_op_dim, nqp].
/// @param qpdc The quadrature point data cache with data layout
/// [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, nqp].
/// @param itod Input Trial Operator Dimension array. If the trial
/// operator is not dependent, the dimension is 0 to indicate that.
/// @param inputs The input field operator types.
/// @param output The output field operator types.
/// @param input_dtqmaps The input DofToQuad maps.
/// @param output_dtqmap The output DofToQuad maps.
/// @param scratch_shmem Scratch shared memory for computations.
/// @param q1d The number of quadrature points in one dimension.
/// @param td1d The number of trial dofs in one dimension.
template <typename input_fop_ts, size_t num_inputs, typename output_fop_t>
MFEM_HOST_DEVICE void assemble_element_mat_t2d(
   const DeviceTensor<4, real_t>& A,
   const DeviceTensor<3, real_t>& fhat,
   const DeviceTensor<5, const real_t>& qpdc,
   const DeviceTensor<1, const real_t>& itod,
   const input_fop_ts& inputs,
   const output_fop_t& output,
   const std::array<DofToQuadMap, num_inputs>& input_dtqmaps,
   const DofToQuadMap& output_dtqmap,
   std::array<DeviceTensor<1>, 6>& scratch_shmem,
   const int& q1d,
   const int& td1d)
{
   constexpr int dimension = 2;

   // [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, num_qp]
   const int test_vdim = qpdc.GetShape()[0];
   const int test_op_dim = qpdc.GetShape()[1];
   const int trial_vdim = qpdc.GetShape()[2];

   // [num_test_dof, ...]
   const auto num_test_dof = A.GetShape()[0];

   for (int Jx = 0; Jx < td1d; Jx++)
   {
      for (int Jy = 0; Jy < td1d; Jy++)
      {
         const int J = Jy + Jx * td1d;

         for (int j = 0; j < trial_vdim; j++)
         {
            for (int tv = 0; tv < test_vdim; tv++)
            {
               for (int tod = 0; tod < test_op_dim; tod++)
               {
                  MFEM_FOREACH_THREAD(qx, x, q1d)
                  {
                     MFEM_FOREACH_THREAD(qy, y, q1d)
                     {
                        const int q = qy + qx * q1d;
                        fhat(tv, tod, q) = 0.0;
                     }
                  }
               }
            }

            // MSVC lambda capture workaround
            [[maybe_unused]] const auto& inputs_ref = inputs;

            int m_offset = 0;
            for_constexpr<num_inputs>([&](auto s)
            {
               using fop_t = std::decay_t<decltype(get<s>(inputs_ref))>;

               const int trial_op_dim = static_cast<int>(itod(static_cast<int>(s)));
               if (trial_op_dim == 0)
               {
                  // This is inside a lambda so we have to return
                  // instead of idiomatic 'continue'.
                  return;
               }

               auto& B = input_dtqmaps[s].B;
               auto& G = input_dtqmaps[s].G;

               if constexpr (is_value_fop<fop_t>::value)
               {
                  MFEM_FOREACH_THREAD(qx, x, q1d)
                  {
                     MFEM_FOREACH_THREAD(qy, y, q1d)
                     {
                        const int q = qy + qx * q1d;
                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           for (int i = 0; i < test_vdim; i++)
                           {
                              for (int k = 0; k < test_op_dim; k++)
                              {
                                 const real_t f = qpdc(i, k, j, m + m_offset, q);
                                 fhat(i, k, q) += f * B(qx, 0, Jx) * B(qy, 0, Jy);
                              }
                           }
                        }
                     }
                  }
               }
               else if constexpr (is_gradient_fop<fop_t>::value)
               {
                  MFEM_FOREACH_THREAD(qx, x, q1d)
                  {
                     MFEM_FOREACH_THREAD(qy, y, q1d)
                     {
                        const int q = qy + qx * q1d;
                        for (int m = 0; m < trial_op_dim; m++)
                        {
                           for (int i = 0; i < test_vdim; i++)
                           {
                              for (int k = 0; k < test_op_dim; k++)
                              {
                                 const real_t f = qpdc(i, k, j, m + m_offset, q);
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
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
                  MFEM_ABORT("sum factorized sparse matrix assemble routine "
                             "not implemented for field operator");
#endif
               }
               MFEM_SYNC_THREAD;
               m_offset += trial_op_dim;
            });

            auto bvtfhat = Reshape(&A(0, 0, J, j), num_test_dof, test_vdim);
            map_quadrature_data_to_fields(bvtfhat, fhat, output, output_dtqmap,
                                          scratch_shmem, dimension, true);
         }
      }
   }
}

/// @brief Assemble element matrix for two or three dimensional data.
///
/// Note: In the below layouts, total_trial_op_dim is > 1 if
/// there are more than one inputs dependent on the derivative variable.
///
/// @param A Memory for one element matrix with layout
/// [test_ndof, test_vdim, trial_ndof, trial_vdim].
/// @param fhat Memory to hold the residual computation with layout
/// [test_vdim, test_op_dim, nqp].
/// @param qpdc The quadrature point data cache with data layout
/// [test_vdim, test_op_dim, trial_vdim, total_trial_op_dim, nqp].
/// @param itod Input Trial Operator Dimension array. If the trial
/// operator is not dependent, the dimension is 0 to indicate that.
/// @param inputs The input field operator types.
/// @param output The output field operator types.
/// @param input_dtqmaps The input DofToQuad maps.
/// @param output_dtqmap The output DofToQuad maps.
/// @param scratch_shmem Scratch shared memory for computations.
/// @param dimension The spatial dimension.
/// @param q1d The number of quadrature points in one dimension.
/// @param td1d The number of trial dofs in one dimension.
/// @param use_sum_factorization Indicator if sum factorization is used.
template <typename input_fop_ts, size_t num_inputs, typename output_fop_t>
MFEM_HOST_DEVICE void assemble_element_mat_naive(
   const DeviceTensor<4, real_t>& A,
   const DeviceTensor<3, real_t>& fhat,
   const DeviceTensor<5, const real_t>& qpdc,
   const DeviceTensor<1, const real_t>& itod,
   const input_fop_ts& inputs,
   const output_fop_t& output,
   const std::array<DofToQuadMap, num_inputs>& input_dtqmaps,
   const DofToQuadMap& output_dtqmap,
   std::array<DeviceTensor<1>, 6>& scratch_shmem,
   const int& dimension,
   const int& q1d,
   const int& td1d,
   const bool& use_sum_factorization)
{
   if (use_sum_factorization)
   {
      if (dimension == 2)
      {
         assemble_element_mat_t2d(A, fhat, qpdc, itod, inputs, output,
                                  input_dtqmaps, output_dtqmap, scratch_shmem, q1d, td1d);
      }
      else if (dimension == 3)
      {
         assemble_element_mat_t3d(A, fhat, qpdc, itod, inputs, output,
                                  input_dtqmaps, output_dtqmap, scratch_shmem, q1d, td1d);
      }
   }
   else
   {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      MFEM_ABORT("element matrix assemble not implemented for non tensor "
                 "product basis");
#endif
   }
}

} // namespace mfem::future
