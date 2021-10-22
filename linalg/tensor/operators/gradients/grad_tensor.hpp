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

#ifndef MFEM_TENSOR_GRAD_TENSOR
#define MFEM_TENSOR_GRAD_TENSOR

#include "grad_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e);
   return ContractX(G,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int Dim = 2;
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto GBu = ContractY(G,Bu);
   auto BGu = ContractY(B,Gu);

   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Dim> Grad_u(Q_r,Q_r,Dim);
   constexpr int Comp = 2;
   Grad_u.template Get<Comp>(0) = BGu;
   Grad_u.template Get<Comp>(1) = GBu;
   return Grad_u;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto BBu = ContractY(B,Bu);
   auto BGu = ContractY(B,Gu);
   auto GBu = ContractY(G,Bu);
   auto BBGu = ContractZ(B,BGu);
   auto BGBu = ContractZ(B,GBu);
   auto GBBu = ContractZ(G,BBu);

   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q,Dim> Grad_u(Q_r,Q_r,Q_r,Dim);
   constexpr int Comp = 3;
   Grad_u.template Get<Comp>(0) = BBGu;
   Grad_u.template Get<Comp>(1) = BGBu;
   Grad_u.template Get<Comp>(2) = GBBu;
   return Grad_u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 1,
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
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 3,
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
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 4,
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

#endif // MFEM_TENSOR_GRAD_TENSOR
