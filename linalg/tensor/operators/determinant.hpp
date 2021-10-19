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
             is_dynamic_matrix<Tensor>,
          bool> = true>
MFEM_HOST_DEVICE inline
auto Determinant(const Tensor &J)
{
   if (J.Size<0>()==3 && J.Size<1>()==3)
   {
      return J(0,0)*J(1,1)*J(2,2)-J(0,2)*J(1,1)*J(2,0)
            +J(0,1)*J(1,2)*J(2,0)-J(0,1)*J(1,0)*J(2,2)
            +J(0,2)*J(1,0)*J(2,1)-J(0,0)*J(1,2)*J(2,1);
   }
   else if (J.Size<0>()==2 && J.Size<1>()==2)
   {
      return J(0,0)*J(1,1)-J(0,1)*J(1,0);
   }
   else if (J.Size<0>()==1 && J.Size<1>()==1)
   {
      return J(0,0);
   }
   else
   {
      // TODO abort
      return 0;
   }
}

template <typename Tensor,
          std::enable_if_t<
             is_static_matrix<3,3,Tensor>,
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
             is_static_matrix<2,2,Tensor>,
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
