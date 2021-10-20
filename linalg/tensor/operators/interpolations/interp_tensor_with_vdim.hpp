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

#ifndef MFEM_TENSOR_INTERP_TENSOR_VDIM
#define MFEM_TENSOR_INTERP_TENSOR_VDIM

#include "interp_tensor.hpp"

namespace mfem
{

// 1D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,VD> v(Q_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}

// 2D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,Q,VD> v(Q_r,Q_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}

// 3D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();

   ResultTensor<Basis,Q,Q,Q,VD> v(Q_r,Q_r,Q_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}


// 1D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 1 &&
             get_tensor_rank<Dofs> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,VD> v(D_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}

// 2D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 2 &&
             get_tensor_rank<Dofs> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,D,VD> v(D_r,D_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}

// 3D Tensor with VDim
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_interp_algo<Basis,Dofs> == InterpAlgo::Tensor &&
             get_basis_dim<Basis> == 3 &&
             get_tensor_rank<Dofs> == 4,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u_e)
{
   constexpr int VDim = get_tensor_rank<Dofs> - 1;
   constexpr int VD = get_tensor_size<VDim,Dofs>;
   constexpr int D = get_basis_dofs<Basis>;
   const int D_r = basis.GetDofs();

   ResultTensor<Basis,D,D,D,VD> v(D_r,D_r,D_r,VD);
   Foreach<VDim>(u_e,[&](int vd)
   {
      v.template Get<VDim>(vd) = basis * u_e.template Get<VDim>(vd);
   });
   return v;
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP_TENSOR_VDIM
