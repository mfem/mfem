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

#ifndef MFEM_TENSOR_INTERP
#define MFEM_TENSOR_INTERP

#include "tensor.hpp"
#include "contraction.hpp"
#include "../../general/backends.hpp"
#include "../dtensor.hpp"
#include <utility>
#include "basis.hpp"
#include "contraction.hpp"

namespace mfem
{

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_non_tensor_basis<Basis>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e);
   return B * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e);
   return ContractX(B,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2 &&
             !is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   return ContractY(B,Bu);
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3 &&
             !is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto BBu = ContractY(B,Bu);
   return ContractZ(B,BBu);
}

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_non_tensor_basis<Basis>,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   return Bt * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   return ContractX(Bt,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2 &&
             !is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   auto Bu = ContractY(Bt,u);
   return ContractX(Bt,Bu);
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3 &&
             !is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_Bt[basis_size];
   auto Bt = basis.GetBt(s_Bt);

   auto Bu = ContractZ(Bt,u);
   auto BBu = ContractY(Bt,Bu);
   return ContractX(Bt,BBu);
}

// 2D threaded version where each thread computes one value.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2 &&
             is_device, // TODO: get_interp_algo<Basis,Dofs> == InterpAlgorithm::Untensor
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bqx[D1D], Bqy[D1D];
   ResultTensor<Basis,Q1D,Q1D> Bu;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_UNROLL(D1D)
         for (int d = 0; d < D1D; d++)
         {
            Bqx[d] = B(qx,d);
            Bqy[d] = B(qy,d);
         }
         double res = 0.0;
         MFEM_UNROLL(D1D)
         for (int dy = 0; dy < D1D; dy++)
         {
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; dx++)
            {
               res += Bqx[dx] * Bqy[dy] * u(dx,dy);
            }
         }
         Bu(qx,qy) = res;
      }
   }
   return Bu;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 2 &&
             is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bdx[Q1D], Bdy[Q1D];
   ResultTensor<Basis,Q1D,Q1D> Btu;
   // Load u into shared memory
   MFEM_SHARED double shared_mem[Q1D*Q1D];
   StaticPointerDTensor<Q1D,Q1D> s_u(shared_mem);
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         s_u(qx,qy) = u(qx,qy);
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_UNROLL(Q1D)
         for (int q = 0; q < Q1D; q++)
         {
            Bdx[q] = Bt(dx,q);
            Bdy[q] = Bt(dy,q);
         }
         double res = 0.0;
         MFEM_UNROLL(Q1D)
         for (int qy = 0; qy < Q1D; qy++)
         {
            MFEM_UNROLL(Q1D)
            for (int qx = 0; qx < Q1D; qx++)
            {
               res += Bdx[qx] * Bdy[qy] * s_u(qx,qy);
            }
         }
         Btu(dx,dy) = res;
      }
   }
   return Btu;
}

// 3D threaded version where each thread computes one value.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3 &&
             is_device, // TODO: get_interp_algo<Basis,Dofs> == InterpAlgorithm::Untensor
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Basis &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bqx[D1D], Bqy[D1D], Bqz[D1D];
   ResultTensor<Basis,Q1D,Q1D,Q1D> Bu;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_UNROLL(D1D)
            for (int d = 0; d < D1D; d++)
            {
               Bqx[d] = B(qx,d);
               Bqy[d] = B(qy,d);
               Bqz[d] = B(qz,d);
            }
            double res = 0.0;
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; dy++)
               {
                  const double Bqyqz = Bqy[dy] * Bqz[dz];
                  MFEM_UNROLL(D1D)
                  for (int dx = 0; dx < D1D; dx++)
                  {
                     res += Bqx[dx] * Bqyqz * u(dx,dy,dz);
                  }
               }
            }
            Bu(qx,qy,qz) = res;
         }
      }
   }
   return Bu;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             is_tensor_basis<Basis> &&
             get_basis_dim<Basis> == 3 &&
             is_device,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Basis> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bdx[Q1D], Bdy[Q1D], Bdz[Q1D];
   ResultTensor<Basis,Q1D,Q1D,Q1D> Btu;
   // Load u into shared memory
   MFEM_SHARED double shared_mem[Q1D*Q1D*Q1D];
   StaticPointerDTensor<Q1D,Q1D,Q1D> s_u(shared_mem);
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            s_u(qx,qy,qz) = u(qx,qy,qz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(dx,x,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_UNROLL(Q1D)
            for (int q = 0; q < Q1D; q++)
            {
               Bdx[q] = Bt(dx,q);
               Bdy[q] = Bt(dy,q);
               Bdz[q] = Bt(dz,q);
            }
            double res = 0.0;
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; qy++)
               {
                  double Bdydz = Bdy[qy] * Bdz[qz];
                  MFEM_UNROLL(Q1D)
                  for (int qx = 0; qx < Q1D; qx++)
                  {
                     res += Bdx[qx] * Bdydz * s_u(qx,qy,qz);
                  }
               }
            }
            Btu(dx,dy,dz) = res;
         }
      }
   }
   return Btu;
}

} // namespace mfem

#endif // MFEM_TENSOR_INTERP
