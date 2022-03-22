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

#ifndef MFEM_TENSOR_GRAD_NEDELEC
#define MFEM_TENSOR_GRAD_NEDELEC

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
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_G_c[close_basis_size];
   auto G_c = basis.GetCloseG(s_G_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   BasisResultTensor<Basis,D_c> u_x(u_e.x);

   return ContractX(G_c,u_x);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
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
   BasisResultTensor<Basis,D_c,D_o> u_x(u_e.x);
   BasisResultTensor<Basis,D_o,D_c> u_y(u_e.y);
   constexpr int Dim = 2;
   constexpr int VDim = 2;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   BasisResultTensor<Basis,Q,Q,Dim,VDim> Gu_q(Q_r,Q_r,Dim,VDim);

   constexpr int Comp = 2;
   constexpr int VDimDim = 3;
   auto Gu_qx = Get<VDimDim>(0, Gu_q);
   auto Bu_x = ContractX(B_c,u_x);
   auto Gu_x = ContractX(G_c,u_x);
   auto GBu_x = ContractY(G_o,Bu_x);
   auto BGu_x = ContractY(B_o,Gu_x);
   Get<Comp>(0, Gu_qx) = BGu_x;
   Get<Comp>(1, Gu_qx) = GBu_x;

   auto Gu_qy = Get<VDimDim>(1, Gu_q);
   auto Bu_y = ContractX(B_o,u_y);
   auto Gu_y = ContractX(G_o,u_y);
   auto GBu_y = ContractY(G_c,Bu_y);
   auto BGu_y = ContractY(B_c,Gu_y);
   Get<Comp>(0, Gu_qy) = BGu_y;
   Get<Comp>(1, Gu_qy) = GBu_y;

   return Gu_q;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
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
   BasisResultTensor<Basis,D_c,D_o,D_o> u_x(u_e.x);
   BasisResultTensor<Basis,D_o,D_c,D_o> u_y(u_e.y);
   BasisResultTensor<Basis,D_o,D_o,D_c> u_z(u_e.z);
   constexpr int Dim = 3;
   constexpr int VDim = 3;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   BasisResultTensor<Basis,Q,Q,Q,Dim,VDim> Gu_q(Q_r,Q_r,Q_r,Dim,VDim);

   constexpr int Comp = 2;
   constexpr int VDimDim = 3;
   auto Gu_qx = Get<VDimDim>(0, Gu_q);
   auto Bu_x = ContractX(B_c,u_x);
   auto Gu_x = ContractX(G_c,u_x);
   auto BBu_x = ContractY(B_o,Bu_x);
   auto GBu_x = ContractY(G_o,Bu_x);
   auto BGu_x = ContractY(B_o,Gu_x);
   auto GBBu_x = ContractZ(G_o,BBu_x);
   auto BGBu_x = ContractZ(B_o,GBu_x);
   auto BBGu_x = ContractZ(B_o,BGu_x);
   Get<Comp>(0, Gu_qx) = BBGu_x;
   Get<Comp>(1, Gu_qx) = BGBu_x;
   Get<Comp>(2, Gu_qx) = GBBu_x;

   auto Gu_qy = Get<VDimDim>(1, Gu_q);
   auto Bu_y = ContractX(B_o,u_y);
   auto Gu_y = ContractX(G_o,u_y);
   auto BBu_y = ContractY(B_c,Bu_y);
   auto GBu_y = ContractY(G_c,Bu_y);
   auto BGu_y = ContractY(B_c,Gu_y);
   auto GBBu_y = ContractZ(G_o,BBu_y);
   auto BGBu_y = ContractZ(B_o,GBu_y);
   auto BBGu_y = ContractZ(B_o,BGu_y);
   Get<Comp>(0, Gu_qy) = BBGu_y;
   Get<Comp>(1, Gu_qy) = BGBu_y;
   Get<Comp>(2, Gu_qy) = GBBu_y;

   auto Gu_qz = Get<VDimDim>(2, Gu_q);
   auto Bu_z = ContractX(B_o,u_z);
   auto Gu_z = ContractX(G_o,u_z);
   auto BBu_z = ContractY(B_o,Bu_z);
   auto GBu_z = ContractY(G_o,Bu_z);
   auto BGu_z = ContractY(B_o,Gu_z);
   auto GBBu_z = ContractZ(G_c,BBu_z);
   auto BGBu_z = ContractZ(B_c,GBu_z);
   auto BBGu_z = ContractZ(B_c,BGu_z);
   Get<Comp>(0, Gu_qz) = BBGu_z;
   Get<Comp>(1, Gu_qz) = BGBu_z;
   Get<Comp>(2, Gu_qz) = GBBu_z;

   return Gu_q;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int close_basis_size = get_close_basis_capacity<Basis>;
   MFEM_SHARED double s_G_c[close_basis_size];
   auto Gt_c = basis.GetCloseGt(s_G_c);

   constexpr int D_c = get_close_basis_dofs<Basis>;
   NedelecElementDofs<BasisResultTensor<Basis,D_c>> v = { ContractX(Gt_c,u) };
   return v;
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
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
   auto Bu_x = ContractY(Bt_o,u_x);
   auto Gu_x = ContractY(Gt_o,u_x);
   auto Gu_qx = ContractX(Gt_c,Bu_x);
   Gu_qx += ContractX(Bt_c,Gu_x);

   auto u_y = Get<Comp>(1, u);
   auto Bu_y = ContractY(Bt_c,u_y);
   auto Gu_y = ContractY(Gt_c,u_y);
   auto Gu_qy = ContractX(Gt_o,Bu_y);
   Gu_qy += ContractX(Bt_o,Gu_y);

   NedelecElementDofs<BasisResultTensor<Basis,D_c,D_o>,
                      BasisResultTensor<Basis,D_o,D_c>> v = { Gu_qx, Gu_qy };
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Nedelec &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
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
   auto Bu_x = ContractZ(Bt_o,u_x);
   auto Gu_x = ContractZ(Gt_o,u_x);
   auto BBu_x = ContractY(Bt_o,Bu_x);
   auto BGu_x = ContractY(Gt_o,Bu_x);
   auto GBu_x = ContractY(Bt_o,Gu_x);
   auto Gu_qx = ContractX(Gt_c,BBu_x);
   Gu_qx += ContractX(Bt_c,BGu_x);
   Gu_qx += ContractX(Bt_c,GBu_x);

   auto u_y = Get<Comp>(1, u);
   auto Bu_y = ContractZ(Bt_o,u_y);
   auto Gu_y = ContractZ(Gt_o,u_y);
   auto BBu_y = ContractY(Bt_c,Bu_y);
   auto BGu_y = ContractY(Gt_c,Bu_y);
   auto GBu_y = ContractY(Bt_c,Gu_y);
   auto Gu_qy = ContractX(Gt_o,BBu_y);
   Gu_qy += ContractX(Bt_o,BGu_y);
   Gu_qy += ContractX(Bt_o,GBu_y);

   auto u_z = Get<Comp>(2, u);
   auto Bu_z = ContractZ(Bt_c,u_z);
   auto Gu_z = ContractZ(Gt_c,u_z);
   auto BBu_z = ContractY(Bt_o,Bu_z);
   auto BGu_z = ContractY(Gt_o,Bu_z);
   auto GBu_z = ContractY(Bt_o,Gu_z);
   auto Gu_qz = ContractX(Gt_o,BBu_z);
   Gu_qz += ContractX(Bt_o,BGu_z);
   Gu_qz += ContractX(Bt_o,GBu_z);

   NedelecElementDofs<BasisResultTensor<Basis,D_c,D_o,D_o>,
                      BasisResultTensor<Basis,D_o,D_c,D_o>,
                      BasisResultTensor<Basis,D_o,D_o,D_c>> v = { Gu_qx, Gu_qy, Gu_qz };
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD_NEDELEC
