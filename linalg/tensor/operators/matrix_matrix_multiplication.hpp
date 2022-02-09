// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more inforAion and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_MATRIX_MATRIX_MULT
#define MFEM_TENSOR_MATRIX_MATRIX_MULT

#include "../../../general/forall.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename MatLHS,
          typename MatRHS,
          std::enable_if_t<
             get_tensor_rank<MatLHS> == 2 &&
             get_tensor_rank<MatRHS> == 2 &&
             is_serial_tensor<MatLHS> &&
             is_serial_tensor<MatRHS> &&
             std::is_same<
                get_tensor_type<MatLHS>,
                get_tensor_type<MatRHS>
             >::value,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const MatLHS &lhs, const MatRHS &rhs)
{
   constexpr int Rows = 0;
   constexpr int Cols = 1;
   using Scalar = get_tensor_type<MatLHS>;
   constexpr int NRowsLHS = get_tensor_size<Rows,MatLHS>;
   constexpr int NColsLHS = get_tensor_size<Cols,MatLHS>;
   constexpr int NRowsRHS = get_tensor_size<Rows,MatRHS>;
   constexpr int NColsRHS = get_tensor_size<Cols,MatRHS>;
   static_assert(
      NColsLHS == Dynamic || NRowsRHS == Dynamic || NColsLHS == NRowsRHS,
      "Invalid dimensions for matrix matrix multiplication"
   );
   const int NRowsLHS_r = lhs.template Size<Rows>();
   const int NColsRHS_r = rhs.template Size<Cols>();
   // TODO some dynamic asserts on NColsLHS and NRowsRHS?
   using Res = ResultTensor<MatLHS,NRowsLHS,NColsRHS>;
   Res v(NRowsLHS_r, NColsRHS_r);
   Foreach<Rows>(lhs,[&](int row){
      Foreach<Cols>(rhs,[&](int col){
         Scalar res = 0;
         Foreach<Cols>(lhs, [&](int i){
            res += lhs(row,i) * rhs(i,col);
         });
         v(row, col) = res;
      });
   });
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_MATRIX_MULT
