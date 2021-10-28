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

#ifndef MFEM_TENSOR_DIV_TENSOR
#define MFEM_TENSOR_DIV_TENSOR

#include "div_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u_x(u_e);

   return ContractX(G,u_x);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int VDim = 2;
   // constexpr int CDim = 3; // FIXME? Generalize to VDim + CDim
   auto u_x = Get<VDim>(0, u_e);
   auto u_y = Get<VDim>(1, u_e);
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q> divu_q(Q_r,Q_r);

   auto Gu_x = ContractX(G,u_x);
   auto BGu_x = ContractY(B,Gu_x);
   divu_q = BGu_x;

   auto Bu_y = ContractX(B,u_y);
   auto GBu_y = ContractY(G,Bu_y);
   divu_q += GBu_y;

   return divu_q;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int VDim = 3;
   // constexpr int CDim = 4; // FIXME? Generalize to VDim + CDim
   auto u_x = Get<VDim>(0, u_e);
   auto u_y = Get<VDim>(1, u_e);
   auto u_z = Get<VDim>(2, u_e);
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q> divu_q(Q_r,Q_r,Q_r);

   auto Gu_x = ContractX(G,u_x);
   auto BGu_x = ContractY(B,Gu_x);
   auto BBGu_x = ContractZ(B,BGu_x);
   divu_q = BBGu_x;

   auto Bu_y = ContractX(B,u_y);
   auto GBu_y = ContractY(G,Bu_y);
   auto BGBu_y = ContractZ(B,GBu_y);
   divu_q += BGBu_y;

   auto Bu_z = ContractX(B,u_z);
   auto BBu_z = ContractY(B,Bu_z);
   auto GBBu_z = ContractZ(G,BBu_z);
   divu_q += GBBu_z;

   return divu_q;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> v = ContractX(Gt,u);
   return v;
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   constexpr int D = get_basis_dofs<Basis>;

   auto Bu_x = ContractY(Bt,u);
   auto div_u = ContractX(Gt,Bu_x);
   auto Bu_y = ContractY(Gt,u);
   div_u += ContractX(Bt,Bu_y);

   ResultTensor<Basis,D,D> v(div_u);
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Tensor &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   constexpr int D = get_basis_dofs<Basis>;

   auto Bu_x = ContractZ(Bt,u);
   auto BBu_x = ContractY(Bt,Bu_x);
   auto div_u = ContractX(Gt,Bu_x);
   auto Bu_y = ContractZ(Bt,u);
   auto GBu_y = ContractY(Gt,Bu_y);
   div_u += ContractX(Bt,GBu_y);
   auto Gu_z = ContractZ(Gt,u);
   auto BGu_z = ContractY(Bt,Gu_z);
   div_u += ContractX(Bt,BGu_z);

   ResultTensor<Basis,D,D> v(div_u);
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_DIV_TENSOR
