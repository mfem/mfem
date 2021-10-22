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

#ifndef MFEM_TENSOR_GRAD_TENSOR_VDIM
#define MFEM_TENSOR_GRAD_TENSOR_VDIM

#include "grad_tensor.hpp"

namespace mfem
{

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,VD> v(Q_r,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim>(vd, v) = basis * Get<VDim>(vd, u);
   });
   return v;
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
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = get_basis_dim<Basis>;
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,Q,Dim,VD> v(Q_r,Q_r,Dim,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim+1>(vd, v) = basis * Get<VDim>(vd, u);
   });
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
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = get_basis_dim<Basis>;
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,Q,Q,Dim,VD> v(Q_r,Q_r,Q_r,Dim,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim+1>(vd, v) = basis * Get<VDim>(vd, u);
   });
   return v;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,VD> v(D_r,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim>(vd, v) = basis * Get<VDim>(vd, u);
   });
   return v;
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,D,VD> v(D_r,D_r,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim-1>(vd, v) = basis * Get<VDim>(vd, u);
   });
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 5,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,D,D,VD> v(D_r,D_r,D_r,VD);
   Foreach<VDim>(u,[&](int vd)
   {
      Get<VDim-1>(vd, v) = basis * Get<VDim>(vd, u);
   });
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD_TENSOR_VDIM
