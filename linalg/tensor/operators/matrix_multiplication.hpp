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

#ifndef MFEM_TENSOR_MATRIX_MULT
#define MFEM_TENSOR_MATRIX_MULT

#include "../../../general/forall.hpp"
#include "../tensor_traits.hpp"
#include "../utilities/foreach.hpp"

namespace mfem
{

template <typename Matrix,
          typename Vector,
          std::enable_if_t<
             get_tensor_rank<Matrix> == 2 &&
             get_tensor_rank<Vector> == 1 &&
             is_serial_tensor<Matrix> &&
             is_serial_tensor<Vector>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Matrix &M, const Vector &u)
{
   using Scalar = get_tensor_type<Vector>;
   constexpr int NCols = get_tensor_size<1,Matrix>;
   using Res = ResultTensor<Vector,NCols>;
   constexpr int Rows = 0;
   constexpr int Cols = 1;
   const int NCols_r = M.template Size<1>();
   Res v(NCols_r);
   Foreach<Rows>(M,[&](int row){
      Scalar res = 0;
      Foreach<Cols>(M, [&](int col){
          res += M(row,col) * u(col);
      });
      v(row) = res;
   });
}

} // namespace mfem

#endif // MFEM_TENSOR_MATRIX_MULT
