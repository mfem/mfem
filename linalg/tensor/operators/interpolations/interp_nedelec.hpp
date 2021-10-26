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

#ifndef MFEM_TENSOR_INTERP_NEDELEC
#define MFEM_TENSOR_INTERP_NEDELEC

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
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto B_c = basis.GetCloseB(s_B_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c> u_x(u_e.x);

   return ContractX(B_c,u_x);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto B_o = basis.GetOpenB(s_B_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto B_c = basis.GetCloseB(s_B_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c,D_o> u_x(u_e.x);
   ResultTensor<Basis,D_o,D_c> u_y(u_e.y);
   constexpr int Dim = 2;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Dim> u_q(Q_r,Q_r,Dim);

   constexpr int Comp = 2;
   auto Bu_x = ContractX(B_c,u_x);
   auto BBu_x = ContractY(B_o,Bu_x);
   Get<Comp>(0, u_q) = BBu_x;
   auto Bu_y = ContractX(B_o,u_y);
   auto BBu_y = ContractY(B_c,Bu_y);
   Get<Comp>(1, u_q) = BBu_y;
   return u_q;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto B_o = basis.GetOpenB(s_B_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto B_c = basis.GetCloseB(s_B_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c,D_o,D_o> u_x(u_e.x);
   ResultTensor<Basis,D_o,D_c,D_o> u_y(u_e.y);
   ResultTensor<Basis,D_o,D_o,D_c> u_z(u_e.z);
   constexpr int Dim = 3;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q,Dim> u_q(Q_r,Q_r,Q_r,Dim);

   constexpr int Comp = 3;
   auto Bu_x = ContractX(B_c,u_x);
   auto BBu_x = ContractY(B_o,Bu_x);
   auto BBBu_x = ContractZ(B_o,BBu_x);
   Get<Comp>(0, u_q) = BBBu_x;
   auto Bu_y = ContractX(B_o,u_y);
   auto BBu_y = ContractY(B_c,Bu_y);
   auto BBBu_y = ContractZ(B_o,BBu_y);
   Get<Comp>(1, u_q) = BBBu_y;
   auto Bu_z = ContractX(B_o,u_z);
   auto BBu_z = ContractY(B_o,Bu_z);
   auto BBBu_z = ContractZ(B_c,BBu_z);
   Get<Comp>(2, u_q) = BBBu_z;
   return u_q;
}

template <int Dim, int CloseDofs, int OpenDofs>
class NedelecResult; // TODO

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto Bt_c = basis.GetCloseBt(s_B_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   NedelecElementDofs<ResultTensor<Basis,D_c>> v = { ContractX(Bt_c,u) };
   return v;
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto Bt_o = basis.GetOpenBt(s_B_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto Bt_c = basis.GetCloseBt(s_B_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   constexpr int Comp = 2;
   auto u_x = Get<Comp>(0, u);
   auto Bu_x = ContractY(Bt_o,u_x);
   auto BBu_x = ContractX(Bt_c,Bu_x);
   auto u_y = Get<Comp>(1, u);
   auto Bu_y = ContractY(Bt_c,u_y);
   auto BBu_y = ContractX(Bt_o,Bu_y);
   NedelecElementDofs<ResultTensor<Basis,D_c,D_o>,
                      ResultTensor<Basis,D_o,D_c>> v = { BBu_x, BBu_y };
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto Bt_o = basis.GetOpenBt(s_B_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto Bt_c = basis.GetCloseBt(s_B_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   constexpr int Comp = 3;
   auto u_x = Get<Comp>(0, u);
   auto Bu_x = ContractZ(Bt_o,u_x);
   auto BBu_x = ContractY(Bt_o,Bu_x);
   auto BBBu_x = ContractX(Bt_c,BBu_x);
   auto u_y = Get<Comp>(1, u);
   auto Bu_y = ContractZ(Bt_o,u_y);
   auto BBu_y = ContractY(Bt_c,Bu_y);
   auto BBBu_y = ContractX(Bt_o,BBu_y);
   auto u_z = Get<Comp>(2, u);
   auto Bu_z = ContractZ(Bt_c,u_z);
   auto BBu_z = ContractY(Bt_o,Bu_z);
   auto BBBu_z = ContractX(Bt_o,BBu_z);
   NedelecElementDofs<
      ResultTensor<Basis,D_c,D_o,D_o>,
      ResultTensor<Basis,D_o,D_c,D_o>,
      ResultTensor<Basis,D_o,D_o,D_c>> v = { BBBu_x, BBBu_y, BBBu_z};
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP_NEDELEC
