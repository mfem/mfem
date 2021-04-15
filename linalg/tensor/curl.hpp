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

#ifndef MFEM_TENSOR_CURL
#define MFEM_TENSOR_CURL

#include "tensor.hpp"
#include "../../general/backends.hpp"
#include "basis.hpp"
#include "contraction.hpp"
#include "concatenate.hpp"

namespace mfem
{

// 1D Non-Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 1 &&
            !is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 1,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   // TODO
   auto G = basis.GetG();
   return G * u;
}

// 2D Non-Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 2 &&
            !is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 2,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   // TODO
   auto G = basis.GetG();
   auto Gx = G.template Get<2>(0);
   auto Gy = G.template Get<2>(1);
   auto ux = u.template Get<1>(0);
   auto uy = u.template Get<1>(1);
   return Gx * uy - Gy * ux;
}

// 3D Non-Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 3 &&
            !is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 2,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   auto G = basis.GetG();
   auto Gx = G.template Get<2>(0);
   auto Gy = G.template Get<2>(1);
   auto Gz = G.template Get<2>(2);
   auto ux = u.template Get<1>(0);
   auto uy = u.template Get<1>(1);
   auto uz = u.template Get<1>(2);
   return Concatenate( Gy * uz - Gz * uy, Gz * ux - Gx * uz, Gx * uy - Gy * ux );
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 1 &&
            is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 1,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   // TODO
   auto G = basis.GetG();
   return ContractX(G,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 2 &&
            is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 3,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   // TODO
   constexpr int Rank = get_tensor_rank<Dofs>;
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto ux  = u.template Get<Rank-1>(0);
   auto uy  = u.template Get<Rank-1>(1);
   auto div = ContractY(B, ContractX(G,ux) );
   div += ContractY(G, ContractX(B,uy) );
   return div;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 3 &&
            is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 4,
            bool> = true >
auto Curl(const Basis &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto ux   = u.template Get<Rank-1>(0);
   auto uy   = u.template Get<Rank-1>(1);
   auto uz   = u.template Get<Rank-1>(2);
   auto Gyuz  = ContractZ(B, ContractY(G, ContractX(B,uz) ) );
   auto Gzuy  = ContractZ(G, ContractY(B, ContractX(B,uy) ) );
   auto rotx  = Gyuz - Gzuy;
   auto Gzux  = ContractZ(G, ContractY(B, ContractX(B,ux) ) );
   auto Gxuz  = ContractZ(B, ContractY(B, ContractX(G,uz) ) );
   auto roty  = Gzux - Gxuz;
   auto Gxuy  = ContractZ(B, ContractY(B, ContractX(G,uy) ) );
   auto Gyux  = ContractZ(B, ContractY(G, ContractX(B,ux) ) );
   auto rotz  = Gxuy - Gyux;
   return Concatenate(rotx, roty, rotz);
}

} // namespace mfem

#endif // MFEM_TENSOR_CURL
