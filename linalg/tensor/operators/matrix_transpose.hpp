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

#ifndef MFEM_TENSOR_MATRIX_TRANSPOSE
#define MFEM_TENSOR_MATRIX_TRANSPOSE

#include "../../../general/forall.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename Matrix,
          std::enable_if_t<
             get_tensor_rank<Matrix> == 2 &&
             is_serial_tensor<Matrix>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto transpose(const Matrix &M)
{
   constexpr int Rows = 0;
   constexpr int Cols = 1;
   constexpr int NRows = get_tensor_size<Rows,Matrix>;
   constexpr int NCols = get_tensor_size<Cols,Matrix>;
   using Res = ResultTensor<Matrix,NCols,NRows>;
   const int NRows_r = M.template Size<Rows>();
   const int NCols_r = M.template Size<Cols>();
   Res Mt(NCols_r,NRows_r);
   Foreach<Rows>(M,[&](int row){
      Foreach<Cols>(M, [&](int col){
         Mt(col,row) = M(row,col);
      });
   });
   return Mt;
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_TRANSPOSE
