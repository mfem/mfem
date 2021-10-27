// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_DET
#define MFEM_TENSOR_DET

#include "../tensor.hpp"

namespace mfem
{

// Determinant
template <typename Tensor,
          std::enable_if_t<
             is_dynamic_matrix<Tensor> &&
             !is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   using T = get_tensor_value_type<Tensor>;
   MFEM_ASSERT_KERNEL(
      J.template Size<0>()== J.template Size<1>(),
      "The matrix must be square.");
   if (J.template Size<0>()==3 && J.template Size<1>()==3)
   {
      return J(0,0)*J(1,1)*J(2,2)-J(0,2)*J(1,1)*J(2,0)
            +J(0,1)*J(1,2)*J(2,0)-J(0,1)*J(1,0)*J(2,2)
            +J(0,2)*J(1,0)*J(2,1)-J(0,0)*J(1,2)*J(2,1);
   }
   else if (J.template Size<0>()==2 && J.template Size<1>()==2)
   {
      return J(0,0)*J(1,1)-J(0,1)*J(1,0);
   }
   else if (J.template Size<0>()==1 && J.template Size<1>()==1)
   {
      return J(0,0);
   }
   else
   {
      MFEM_ABORT_KERNEL("Not yet supported.");
      // T det{};
      // Foreach<0>(J,[&](int row){
      //    det += pow(-1,row) * J(row,0) * Determinant(Sub<0>(row,Sub<1>(0,J)));
      // });
      // return det;
      return T{};
   }
}

template <typename Tensor,
          std::enable_if_t<
             is_dynamic_matrix<Tensor> &&
             is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   using T = get_tensor_value_type<Tensor>;
   MFEM_ASSERT_KERNEL(
      J.template Size<0>()== J.template Size<1>(),
      "The matrix must be square.");
   if (J.template Size<0>()==1 && J.template Size<1>()==1)
   {
      return J(0,0);
   }
   else
   {
      MFEM_ABORT_KERNEL("Not yet supported.");
      // Is this correct?
      // MFEM_SHARED T det{};
      // Foreach<0>(J,[&](int row){
      //    det += pow(-1,row) * J(row,0) * Determinant(Sub<0>(row,Sub<1>(0,J))); // TODO: Use AtomicAdd
      // });
      // MFEM_SYNC_THREAD;
      // return det;
      return T{};
   }
}

template <typename Tensor,
          std::enable_if_t<
             is_static_tensor<Tensor> &&
             get_tensor_rank<Tensor> == 2 &&
             get_tensor_size<0,Tensor> == get_tensor_size<1,Tensor> &&
             (get_tensor_size<0,Tensor> > 1) &&
             !is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   MFEM_ABORT_KERNEL("Not yet supported.");
   using T = get_tensor_value_type<Tensor>;
   // MFEM_SHARED T shared_mem[get_tensor_batchsize];
   // auto batch_id = 0; // <-- MFEM_THREAD_ID(z); ?
   // T &det = shared_mem[get_tensor_batchsize];
   // constexpr int Row = 0;
   // Foreach<Row>(J,[&](int row){
   //    AtomicAdd(det,
   //              pow(-1,row) * J(row,0) * Determinant(
   //                                          Sub<0>(row,
   //                                             Sub<1>(0, J ))));
   // });
   // MFEM_SYNC_THREAD;
   // return det;
   return T{};
}

template <typename Tensor,
          std::enable_if_t<
             is_static_tensor<Tensor> &&
             get_tensor_rank<Tensor> == 2 &&
             get_tensor_size<0,Tensor> == get_tensor_size<1,Tensor> &&
             (get_tensor_size<0,Tensor> > 3) &&
             is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   MFEM_ABORT_KERNEL("Not yet supported.");
   using T = get_tensor_value_type<Tensor>;
   // T det{};
   // constexpr int Row = 0;
   // Foreach<Row>(J,[&](int row){
   //    det += pow(-1,row) * J(row,0) * Determinant(Sub<0>(row,Sub<1>(0,J)));
   // });
   // return det;
   return T{};
}

template <typename Tensor,
          std::enable_if_t<
             is_static_matrix<3,3,Tensor> &&
             !is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   return J(0,0)*J(1,1)*J(2,2)-J(0,2)*J(1,1)*J(2,0)
         +J(0,1)*J(1,2)*J(2,0)-J(0,1)*J(1,0)*J(2,2)
         +J(0,2)*J(1,0)*J(2,1)-J(0,0)*J(1,2)*J(2,1);
}

template <typename Tensor,
          std::enable_if_t<
             is_static_matrix<2,2,Tensor> &&
             !is_threaded_tensor_dim<Tensor,0>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   return J(0,0)*J(1,1)-J(0,1)*J(1,0);
}

template <typename Tensor,
          std::enable_if_t<
             is_static_matrix<1,1,Tensor>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   return J(0,0);
}

} // namespace mfem

#endif // MFEM_TENSOR_DET
