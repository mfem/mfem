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

#ifndef MFEM_TENSOR_DIV_NEDELEC
#define MFEM_TENSOR_DIV_NEDELEC

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
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_G_c[close_basis_size];
   auto G_c = basis.GetCloseG(s_G_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c> u_x(u_e.x);

   return ContractX(G_c,u_x);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto B_o = basis.GetOpenB(s_B_o);
   MFEM_SHARED double s_G_o[open_basis_size];
   auto G_o = basis.GetOpenG(s_G_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto B_c = basis.GetCloseB(s_B_c);
   MFEM_SHARED double s_G_c[close_basis_size];
   auto G_c = basis.GetCloseG(s_G_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c,D_o> u_x(u_e.x);
   ResultTensor<Basis,D_o,D_c> u_y(u_e.y);
   constexpr int Dim = 2;
   constexpr int VDim = 2;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q> divu_q(Q_r,Q_r);

   auto Gu_x = ContractX(G_c,u_x);
   auto BGu_x = ContractY(B_o,Gu_x);
   divu_q = BGu_x;

   auto Bu_y = ContractX(B_o,u_y);
   auto GBu_y = ContractY(G_c,Bu_y);
   divu_q += GBu_y;

   return divu_q;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Div<Basis> &basis, const Dofs &u_e)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto B_o = basis.GetOpenB(s_B_o);
   MFEM_SHARED double s_G_o[open_basis_size];
   auto G_o = basis.GetOpenG(s_G_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto B_c = basis.GetCloseB(s_B_c);
   MFEM_SHARED double s_G_c[close_basis_size];
   auto G_c = basis.GetCloseG(s_G_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   ResultTensor<Basis,D_c,D_o,D_o> u_x(u_e.x);
   ResultTensor<Basis,D_o,D_c,D_o> u_y(u_e.y);
   ResultTensor<Basis,D_o,D_o,D_c> u_z(u_e.z);
   constexpr int Dim = 3;
   constexpr int VDim = 3;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q> divu_q(Q_r,Q_r,Q_r);

   auto Gu_x = ContractX(G_c,u_x);
   auto BGu_x = ContractY(B_o,Gu_x);
   auto BBGu_x = ContractZ(B_o,BGu_x);
   divu_q = BBGu_x;

   auto Bu_y = ContractX(B_o,u_y);
   auto GBu_y = ContractY(G_c,Bu_y);
   auto BGBu_y = ContractZ(B_o,GBu_y);
   divu_q += BGBu_y;

   auto Bu_z = ContractX(B_o,u_z);
   auto BBu_z = ContractY(B_o,Bu_z);
   auto GBBu_z = ContractZ(G_c,BBu_z);
   divu_q += GBBu_z;

   return divu_q;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_G_c[close_basis_size];
   auto Gt_c = basis.GetCloseGt(s_G_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   NedelecElementDofs<ResultTensor<Basis,D_c>> v = { ContractX(Gt_c,u) };
   return v;
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto Bt_o = basis.GetOpenBt(s_B_o);
   MFEM_SHARED double s_G_o[open_basis_size];
   auto Gt_o = basis.GetOpenGt(s_G_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto Bt_c = basis.GetCloseBt(s_B_c);
   MFEM_SHARED double s_G_c[close_basis_size];
   auto Gt_c = basis.GetCloseGt(s_G_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   constexpr int VDim = 2;

   auto u_x = Get<VDim>(0, u);
   auto u_y = Get<VDim>(1, u);

   auto Bu_xx = ContractY(Bt_o,u_x);
   auto divu_x = ContractX(Gt_c,Bu_xx);
   auto Bu_yx = ContractY(Gt_o,u_y);
   divu_x += ContractX(Bt_c,Bu_yx);

   auto Bu_xy = ContractY(Bt_c,u_x);
   auto divu_y = ContractX(Gt_o,Bu_xy);
   auto Bu_yy = ContractY(Gt_c,u_y);
   divu_y += ContractX(Bt_o,Bu_yy);

   NedelecElementDofs<ResultTensor<Basis,D_c,D_o>,
                      ResultTensor<Basis,D_o,D_c>
   > v = { divu_x, divu_y };
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_div_algo<Basis,Dofs> == DivAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Div<Basis>> &basis, const Dofs &u)
{
   constexpr int open_basis_size = get_open_basis_capacity<Basis>;
   MFEM_SHARED double s_B_o[open_basis_size];
   auto Bt_o = basis.GetOpenBt(s_B_o);
   MFEM_SHARED double s_G_o[open_basis_size];
   auto Gt_o = basis.GetOpenGt(s_G_o);
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_B_c[close_basis_size];
   auto Bt_c = basis.GetCloseBt(s_B_c);
   MFEM_SHARED double s_G_c[close_basis_size];
   auto Gt_c = basis.GetCloseGt(s_G_c);

   constexpr int D_o = get_open_basis_dofs<Basis>;
   constexpr int D_c = get_close_basis_dofs<Basis>;
   constexpr int Comp = 2;

   auto u_x = Get<Comp>(0, u);
   auto u_y = Get<Comp>(1, u);
   auto u_z = Get<Comp>(2, u);

   auto Bu_xx = ContractZ(Bt_o,u_x);
   auto BBu_xx = ContractY(Bt_o,Bu_xx);
   auto divu_x = ContractX(Gt_c,BBu_xx);
   auto Bu_yx = ContractZ(Bt_o,u_y);
   auto GBu_yx = ContractY(Gt_o,Bu_yx);
   divu_x += ContractX(Bt_c,GBu_yx);
   auto Gu_zx = ContractZ(Gt_o,u_z);
   auto GBu_zx = ContractY(Bt_o,Gu_zx);
   divu_x += ContractX(Bt_c,GBu_zx);

   auto Bu_xy = ContractZ(Bt_o,u_x);
   auto BBu_xy = ContractY(Bt_c,Bu_xy);
   auto divu_y = ContractX(Gt_o,BBu_xy);
   auto Bu_yy = ContractZ(Bt_o,u_y);
   auto GBu_yy = ContractY(Gt_c,Bu_yy);
   divu_y += ContractX(Bt_o,GBu_yy);
   auto Gu_zy = ContractZ(Gt_o,u_z);
   auto GBu_zy = ContractY(Bt_c,Gu_zy);
   divu_y += ContractX(Bt_o,GBu_zy);

   auto Bu_xz = ContractZ(Bt_c,u_x);
   auto BBu_xz = ContractY(Bt_o,Bu_xz);
   auto divu_z = ContractX(Gt_o,BBu_xz);
   auto Bu_yz = ContractZ(Bt_c,u_y);
   auto GBu_yz = ContractY(Gt_o,Bu_yz);
   divu_z += ContractX(Bt_o,GBu_yz);
   auto Gu_zz = ContractZ(Gt_c,u_z);
   auto GBu_zz = ContractY(Bt_o,Gu_zz);
   divu_z += ContractX(Bt_o,GBu_zz);

   NedelecElementDofs<ResultTensor<Basis,D_c,D_o,D_o>,
                      ResultTensor<Basis,D_o,D_c,D_o>,
                      ResultTensor<Basis,D_o,D_o,D_c>
   > v = { divu_x, divu_y, divu_z };
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_DIV_NEDELEC
