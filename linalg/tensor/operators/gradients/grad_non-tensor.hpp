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

#ifndef MFEM_TENSOR_GRAD_NON_TENSOR
#define MFEM_TENSOR_GRAD_NON_TENSOR

#include "grad_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::NonTensor,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e); // TODO: Add a diff dim of 1?
   return G * u;
}

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::NonTensor,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   return Gt * u;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD_NON_TENSOR
