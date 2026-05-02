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

#include "linalg/tensor.hpp"

namespace mfem::future
{

///////////////////////////////////////////////////////////////////////////////
/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1>` */
template<typename T, int n1> inline
MFEM_HOST_DEVICE const tensor<T, n1>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1>*>(ptr));
}

template<typename T, int n1> inline
MFEM_HOST_DEVICE tensor<T, n1>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2>` */
template<typename T, int n1, int n2> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2>*>(ptr));
}

template<typename T, int n1, int n2> inline
MFEM_HOST_DEVICE tensor<T, n1, n2>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2, n3>` */
template<typename T, int n1, int n2, int n3> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3>*>(ptr));
}

template<typename T, int n1, int n2, int n3> inline
MFEM_HOST_DEVICE tensor<T, n1, n2, n3>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3>*>(ptr));
}

/** @brief Zero-copy view of a contiguous block as a `tensor<T, n1, n2, n3, n4>` */
template<typename T, int n1, int n2, int n3, int n4> inline
MFEM_HOST_DEVICE const tensor<T, n1, n2, n3, n4>& as_tensor(const T* ptr)
{
   return *std::launder(reinterpret_cast<const tensor<T, n1, n2, n3, n4>*>(ptr));
}

template<typename T, int n1, int n2, int n3, int n4> inline
MFEM_HOST_DEVICE tensor<T, n1, n2, n3, n4>& as_tensor(T* ptr)
{
   return *std::launder(reinterpret_cast<tensor<T, n1, n2, n3, n4>*>(ptr));
}

///////////////////////////////////////////////////////////////////////////////
namespace qf
{

template <int T_Q1D,
          size_t num_args,
          typename reg_t,
          typename qfunc_t,
          typename args_ts>
MFEM_HOST_DEVICE inline
void apply_kernel(reg_t &res /*output*/,
                  reg_t &reg,
                  const real_t *rd,
                  const int qx, const int qy, const int qz,
                  const qfunc_t &qfunc, args_ts &args)
{
   if constexpr (num_args == 2) // PAApply
   {
      // ∇u
      tensor<real_t, 3> &arg_0 = get<0>(args);
      arg_0[0] = reg[qz][qy][qx][0];
      arg_0[1] = reg[qz][qy][qx][1];
      arg_0[2] = reg[qz][qy][qx][2];

      // D (PA data)
      tensor<real_t, 3, 3> &arg_1 = get<1>(args);

      if constexpr (T_Q1D > 0)
      {
         const auto *D = (const real_t (*)[T_Q1D][T_Q1D][3][3]) rd;
         for (int k = 0; k < 3; k++)
         {
            for (int j = 0; j < 3; j++)
            {
               arg_1[k][j] = D[qx][qy][qz][k][j];
            }
         }
      }
      else
      {
         static_assert(false);
         // const auto D = Reshape(r2, 3, 3, Q1D, Q1D, Q1D);
         // for (int j = 0; j < 3; j++)
         // {
         //    for (int k = 0; k < 3; k++)
         //    {
         //       arg_1[k][j] = D(j, k, qz, qy, qx);
         //    }
         // }
      }
   }
   else
   {
      // MFApply comes here
      static_assert(false, "Only 2 args are supported for now");
   }

   const auto r = get<0>(apply(qfunc, args));

   if constexpr (decltype(r)::ndim == 1)
   {
      // process_qf_result_from_reg(r0, qx, qy, qz, r);
      as_tensor<real_t, 3>(&res[qz][qy][qx][0]) = r;
   }
   else
   {
      static_assert(false);
   }
}

} // namespace qf

} // namespace mfem::future
