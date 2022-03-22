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

#ifndef MFEM_TENSOR_GRAD_UNTENSOR
#define MFEM_TENSOR_GRAD_UNTENSOR

#include "grad_traits.hpp"
#include "../../tensor.hpp"
#include "../../factories/basis/basis.hpp"
#include "../contractions/contractions.hpp"

namespace mfem
{

// 2D threaded version where each thread computes one value.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Untensorized &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = 2;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   MFEM_SHARED double s_G[basis_size];
   const auto B = basis.GetB(s_B);
   const auto G = basis.GetG(s_G);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bqx[D1D];//, Bqy[D1D];
   double Gqx[D1D];//, Gqy[D1D];
   BasisResultTensor<Basis,Q1D,Q1D,Dim> Gu;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_UNROLL(D1D)
         for (int d = 0; d < D1D; d++)
         {
            Bqx[d] = B(qx,d);
            // Bqy[d] = B(qy,d);
            Gqx[d] = G(qx,d);
            // Gqy[d] = G(qy,d);
         }
         double du_dx = 0.0;
         double du_dy = 0.0;
         MFEM_UNROLL(D1D)
         for (int dy = 0; dy < D1D; dy++)
         {
            const double Bqydy = B(qy,dy);
            const double Gqydy = G(qy,dy);
            MFEM_UNROLL(D1D)
            for (int dx = 0; dx < D1D; dx++)
            {
               const double val = u(dx,dy);
               du_dx += Gqx[dx] * Bqydy * val;
               du_dy += Bqx[dx] * Gqydy * val;
            }
         }
         Gu(qx,qy,0) = du_dx;
         Gu(qx,qy,1) = du_dy;
      }
   }
   return Gu;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Untensorized &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Dim = 2;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   MFEM_SHARED double s_G[basis_size];
   auto Bt = basis.GetBt(s_B);
   auto Gt = basis.GetGt(s_G);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bdx[Q1D];//, Bdy[Q1D];
   double Gdx[Q1D];//, Gdy[Q1D];
   BasisResultTensor<Basis,D1D,D1D> Gtu;
   // Load u into shared memory
   MFEM_SHARED double shared_mem[Q1D*Q1D*Dim];
   StaticPointerDTensor<Q1D,Q1D,Dim> s_u(shared_mem);
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_UNROLL(Dim)
         for (int d = 0; d < Dim; d++)
         {
            s_u(qx,qy,d) = u(qx,qy,d);
         }
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
            // Bdy[q] = Bt(dy,q);
            Gdx[q] = Gt(dx,q);
            // Gdy[q] = Gt(dy,q);
         }
         double res = 0.0;
         MFEM_UNROLL(Q1D)
         for (int qy = 0; qy < Q1D; qy++)
         {
            const double Bdyqy = Bt(dy,qy);
            const double Gdyqy = Gt(dy,qy);
            MFEM_UNROLL(Q1D)
            for (int qx = 0; qx < Q1D; qx++)
            {
               const double val0 = s_u(qx,qy,0);
               res += Gdx[qx] * Bdyqy * val0;
               const double val1 = s_u(qx,qy,1);
               res += Bdx[qx] * Gdyqy * val1;
            }
         }
         Gtu(dx,dy) = res;
      }
   }
   return Gtu;
}

// 3D threaded version where each thread computes one value.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Untensorized &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   MFEM_SHARED double s_G[basis_size];
   const auto B = basis.GetB(s_B);
   const auto G = basis.GetG(s_G);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bqx[D1D];//, Bqy[D1D], Bqz[D1D];
   double Gqx[D1D];//, Gqy[D1D], Gqz[D1D];
   BasisResultTensor<Basis,Q1D,Q1D,Q1D,Dim> Gu;
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
               // Bqy[d] = B(qy,d);
               // Bqz[d] = B(qz,d);
               Gqx[d] = G(qx,d);
               // Gqy[d] = G(qy,d);
               // Gqz[d] = G(qz,d);
            }
            double du_dx = 0.0;
            double du_dy = 0.0;
            double du_dz = 0.0;
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               const double Bqzdz = B(qz,dz);
               const double Gqzdz = G(qz,dz);
               MFEM_UNROLL(D1D)
               for (int dy = 0; dy < D1D; dy++)
               {
                  const double Bqydy = B(qy,dy);
                  const double Gqydy = G(qy,dy);
                  MFEM_UNROLL(D1D)
                  for (int dx = 0; dx < D1D; dx++)
                  {
                     const double val = u(dx,dy,dz);
                     du_dx += Gqx[dx] * Bqydy * Bqzdz * val;
                     du_dy += Bqx[dx] * Gqydy * Bqzdz * val;
                     du_dz += Bqx[dx] * Bqydy * Gqzdz * val;
                  }
               }
            }
            Gu(qx,qy,qz,0) = du_dx;
            Gu(qx,qy,qz,1) = du_dy;
            Gu(qx,qy,qz,2) = du_dz;
         }
      }
   }
   return Gu;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Untensorized &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   MFEM_SHARED double s_G[basis_size];
   auto Bt = basis.GetBt(s_B);
   auto Gt = basis.GetGt(s_G);
   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   double Bdx[Q1D];//, Bdy[Q1D], Bdz[Q1D];
   double Gdx[Q1D];//, Gdy[Q1D], Gdz[Q1D];
   BasisResultTensor<Basis,D1D,D1D,D1D> Gtu;
   // Load u into shared memory
   MFEM_SHARED double shared_mem[Q1D*Q1D*Q1D*Dim];
   StaticPointerDTensor<Q1D,Q1D,Q1D,Dim> s_u(shared_mem);
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            for (int d = 0; d < Dim; d++)
            {
               s_u(qx,qy,qz,d) = u(qx,qy,qz,d);
            }
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
               // Bdy[q] = Bt(dy,q);
               // Bdz[q] = Bt(dz,q);
               Gdx[q] = Gt(dx,q);
               // Gdy[q] = Gt(dy,q);
               // Gdz[q] = Gt(dz,q);
            }
            double res = 0.0;
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               const double Bdz = Bt(dz,qz);
               const double Gdz = Gt(dz,qz);
               MFEM_UNROLL(Q1D)
               for (int qy = 0; qy < Q1D; qy++)
               {
                  const double Bdy = Bt(dy,qy);
                  const double Gdy = Gt(dy,qy);
                  MFEM_UNROLL(Q1D)
                  for (int qx = 0; qx < Q1D; qx++)
                  {
                     const double val0 = s_u(qx,qy,qz,0);
                     res += Gdx[qx] * Bdy * Bdz * val0;
                     const double val1 = s_u(qx,qy,qz,1);
                     res += Bdx[qx] * Gdy * Bdz * val1;
                     const double val2 = s_u(qx,qy,qz,2);
                     res += Bdx[qx] * Bdy * Gdz * val2;
                  }
               }
            }
            Gtu(dx,dy,dz) = res;
         }
      }
   }
   return Gtu;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD_UNTENSOR
