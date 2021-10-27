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

#ifndef MFEM_TENSOR_DIV
#define MFEM_TENSOR_DIV

#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"
#include "../concatenate.hpp"

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
auto Div(const Basis &basis, const Dofs &u)
{
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
auto Div(const Basis &basis, const Dofs &u)
{
   constexpr int Comp = get_tensor_rank<Dofs> - 1;
   auto G = basis.GetG();
   auto Gx = Get<2>(0, G);
   auto Gy = Get<2>(1, G);
   auto ux = Get<Comp>(0, u);
   auto uy = Get<Comp>(1, u);
   return Gx * ux + Gy * uy;
}

// 3D Non-Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 3 &&
            !is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 2,
            bool> = true >
auto Div(const Basis &basis, const Dofs &u)
{
   constexpr int Comp = get_tensor_rank<Dofs> - 1;
   auto G = basis.GetG();
   auto Gx = Get<2>(0, G);
   auto Gy = Get<2>(1, G);
   auto Gz = Get<2>(2, G);
   auto ux = Get<Comp>(0, u);
   auto uy = Get<Comp>(1, u);
   auto uz = Get<Comp>(2, u);
   return Gx * ux + Gy * uy + Gz * uz;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
            get_basis_dim<Basis> == 1 &&
            is_tensor_basis<Basis> &&
            get_tensor_rank<Dofs> == 1,
            bool> = true >
auto Div(const Basis &basis, const Dofs &u)
{
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
auto Div(const Basis &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto ux  = Get<Rank-1>(0, u);
   auto uy  = Get<Rank-1>(1, u);
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
auto Div(const Basis &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   auto B = basis.GetB();
   auto G = basis.GetG();
   auto ux   = Get<Rank-1>(0, u);
   auto uy   = Get<Rank-1>(1, u);
   auto uz   = Get<Rank-1>(2, u);
   auto div  = ContractZ(B, ContractY(B, ContractX(G,ux) ) );
   div += ContractZ(B, ContractY(G, ContractX(B,uy) ) );
   div += ContractZ(G, ContractY(B, ContractX(B,uz) ) );
   return div;
}

} // namespace mfem

#endif // MFEM_TENSOR_DIV
