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

#ifndef MFEM_TENSOR_GRAD
#define MFEM_TENSOR_GRAD

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include "basis.hpp"
#include "contraction.hpp"
#include "concatenate.hpp"

namespace mfem
{

// Non-tensor
// template <int Dim, int D, int Q, typename Dofs>
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_non_tensor_basis<Basis>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int Q = get_basis_quads<Basis>;
   ResultTensor<Basis,Q> u(u_e); // TODO: Add a diff dim of 1?
   return G * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int Q = get_basis_quads<Basis>;
   ResultTensor<Basis,Q> u(u_e);
   return ContractX(G,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int Q = get_basis_quads<Basis>;
   ResultTensor<Basis,Q,Q> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto GBu = ContractY(G,Bu);
   auto BGu = ContractY(B,Gu);

   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,2> Grad_u(Q_r,Q_r,2);
   Grad_u.template Get<2>(0) = BGu;
   Grad_u.template Get<2>(1) = GBu;
   return Grad_u;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int Q = get_basis_quads<Basis>;
   ResultTensor<Basis,Q,Q,Q> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto BBu = ContractY(B,Bu);
   auto BGu = ContractY(B,Gu);
   auto GBu = ContractY(G,Bu);
   auto BBGu = ContractZ(B,BGu);
   auto BGBu = ContractZ(B,GBu);
   auto GBBu = ContractZ(G,BBu);

   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q,3> Grad_u(Q_r,Q_r,Q_r,3);
   Grad_u.template Get<2>(0) = BBGu;
   Grad_u.template Get<2>(1) = BGBu;
   Grad_u.template Get<2>(2) = GBBu;
   return Grad_u;
}

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_non_tensor_basis<Basis>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   return Gt * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   return ContractX(Gt,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   constexpr int Comp = Rank-1;
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   auto ux = u.template Get<Comp>(0);
   auto Gux = ContractX(Gt,ux);
   auto v = ContractY(Bt,Gux);
   auto uy = u.template Get<Comp>(1);
   auto Buy = ContractX(Bt,uy);
   v += ContractY(Gt,Buy);
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   constexpr int Comp = Rank-1;
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   auto ux = u.template Get<Comp>(0);
   auto Gux = ContractX(Gt,ux);
   auto BGux = ContractY(Bt,Gux);
   auto v = ContractZ(Bt,BGux);
   auto uy = u.template Get<Comp>(1);
   auto Buy = ContractX(Bt,uy);
   auto GBuy = ContractY(Gt,Buy);
   v += ContractZ(Bt,GBuy);
   auto uz = u.template Get<Comp>(2);
   auto Buz = ContractX(Bt,uz);
   auto BBuz = ContractY(Bt,Buz);
   v += ContractZ(Gt,BBuz);
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD
