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

#ifndef MFEM_TENSOR_INTERP_TENSOR
#define MFEM_TENSOR_INTERP_TENSOR

#include "interp_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e);
   return ContractX(B,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   return ContractY(B,Bu);
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto BBu = ContractY(B,Bu);
   return ContractZ(B,BBu);
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   return ContractX(Bt,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   auto Bu = ContractY(Bt,u);
   return ContractX(Bt,Bu);
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   auto Bu = ContractZ(Bt,u);
   auto BBu = ContractY(Bt,Bu);
   return ContractX(Bt,BBu);
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP_TENSOR
