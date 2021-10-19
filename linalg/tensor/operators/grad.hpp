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

#ifndef MFEM_TENSOR_GRAD
#define MFEM_TENSOR_GRAD

#include "../tensor.hpp"
#include "../factories/basis.hpp"
#include "contractions/contraction.hpp"

namespace mfem
{

enum class GradAlgo { NonTensor, Tensor, Untensorized, Legacy, NA };

template <typename Basis, typename Dofs, typename Enable = void>
struct get_grad_algo_v
{
   static constexpr GradAlgo value = GradAlgo::NA;
};

template <typename Basis, typename Dofs>
struct get_grad_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_non_tensor_basis<Basis>
   > >
{
   static constexpr GradAlgo value = GradAlgo::NonTensor;
};

template <typename Basis, typename Dofs>
struct get_grad_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_tensor_basis<Basis> &&
      !(get_basis_dim<Basis> == 3 &&
      is_device)
   > >
{
   static constexpr GradAlgo value = GradAlgo::Tensor;
};

template <typename Basis, typename Dofs>
struct get_grad_algo_v<Basis, Dofs,
   std::enable_if_t<
      is_tensor_basis<Basis> &&
      (get_basis_dim<Basis> == 3 &&
      is_device)
   > >
{
   static constexpr GradAlgo value = GradAlgo::Legacy;
};

template <typename Basis, typename Dofs>
constexpr GradAlgo get_grad_algo = get_grad_algo_v<Basis, Dofs>::value;

// Non-tensor
// template <int Dim, int D, int Q, typename Dofs>
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::NonTensor,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e); // TODO: Add a diff dim of 1?
   return G * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D> u(u_e);
   return ContractX(G,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int Dim = 2;
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto GBu = ContractY(G,Bu);
   auto BGu = ContractY(B,Gu);

   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Dim> Grad_u(Q_r,Q_r,Dim);
   constexpr int Comp = 2;
   Grad_u.template Get<Comp>(0) = BGu;
   Grad_u.template Get<Comp>(1) = GBu;
   return Grad_u;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u_e)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Grad<Basis>>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D = get_basis_dofs<Basis>;
   ResultTensor<Basis,D,D,D> u(u_e);
   auto Bu = ContractX(B,u);
   auto Gu = ContractX(G,u);
   auto BBu = ContractY(B,Bu);
   auto BGu = ContractY(B,Gu);
   auto GBu = ContractY(G,Bu);
   auto BBGu = ContractZ(B,BGu);
   auto BGBu = ContractZ(B,GBu);
   auto GBBu = ContractZ(G,BBu);

   constexpr int Q = get_basis_quads<Basis>;
   const int Q_r = basis.GetQuads();
   ResultTensor<Basis,Q,Q,Q,Dim> Grad_u(Q_r,Q_r,Q_r,Dim);
   constexpr int Comp = 3;
   Grad_u.template Get<Comp>(0) = BBGu;
   Grad_u.template Get<Comp>(1) = BGBu;
   Grad_u.template Get<Comp>(2) = GBBu;
   return Grad_u;
}

// Non-tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::NonTensor,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   return Gt * u;
}

// 1D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 1,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   return ContractX(Gt,u);
}

// 2D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 2,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   constexpr int Comp = Rank-1;
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   auto ux = u.template Get<Comp>(0);
   auto Gux = ContractX(Gt,ux);
   auto v = ContractY(Bt,Gux);
   auto uy = u.template Get<Comp>(1);
   auto Buy = ContractX(Bt,uy);
   v += ContractY(Gt,Buy);
   return v;
}

// 3D Tensor
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Tensor &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Rank = get_tensor_rank<Dofs>;
   constexpr int Comp = Rank-1;
   constexpr int basis_size = get_basis_capacity<Trans<Grad<Basis>>>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   auto ux = u.template Get<Comp>(0);
   auto Gux = ContractX(Gt,ux);
   auto BGux = ContractY(Bt,Gux);
   auto v = ContractZ(Bt,BGux);
   auto uy = u.template Get<Comp>(1);
   auto Buy = ContractX(Bt,uy);
   auto GBuy = ContractY(Gt,Buy);
   v += ContractZ(Bt,GBuy);
   auto uz = u.template Get<Comp>(2);
   auto Buz = ContractX(Bt,uz);
   auto BBuz = ContractY(Bt,Buz);
   v += ContractZ(Gt,BBuz);
   return v;
}

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
   ResultTensor<Basis,Q1D,Q1D,Dim> Gu;
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
   ResultTensor<Basis,D1D,D1D> Gtu;
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
   ResultTensor<Basis,Q1D,Q1D,Q1D,Dim> Gu;
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
   ResultTensor<Basis,D1D,D1D,D1D> Gtu;
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

// 3D 2dthreaded version extracted from: SmemPADiffusionApply3D.
template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Legacy &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Grad<Basis> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto B = basis.GetB(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto G = basis.GetG(s_G);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[Dim*MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[Dim*MaxDQ*MaxDQ*MaxDQ];

   // Load dofs in shared memory
   StaticPointerDTensor<D1D,D1D,D1D> X(sm0);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            X(dx,dy,dz) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
   // X Contraction
   StaticPointerDTensor<D1D,D1D,Q1D,2> DDQ(sm1);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D], v[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dx = 0; dx < D1D; ++dx)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double coords = X(dx,dy,dz);
               u[dz] += coords * B(dx,qx);
               v[dz] += coords * G(dx,qx);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            DDQ(qx,dy,dz,0) = u[dz];
            DDQ(qx,dy,dz,1) = v[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<D1D,Q1D,Q1D,Dim> DQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[D1D], v[D1D], w[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = w[dz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dy = 0; dy < D1D; ++dy)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ(qx,dy,dz,1) * B(dy,qy);
               v[dz] += DDQ(qx,dy,dz,0) * G(dy,qy);
               w[dz] += DDQ(qx,dy,dz,0) * B(dy,qy);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ(qx,qy,dz,0) = u[dz];
            DQQ(qx,qy,dz,1) = v[dz];
            DQQ(qx,qy,dz,2) = w[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Z Contraction
   constexpr int batchsize = 1;
   Static2dThreadDTensor<batchsize,Q1D,Q1D,Q1D,Dim> QQQ;
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ(qx,qy,dz,0) * B(dz,qz);
               v[qz] += DQQ(qx,qy,dz,1) * B(dz,qz);
               w[qz] += DQQ(qx,qy,dz,2) * G(dz,qz);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; qz++)
         {
            QQQ(qx,qy,qz,0) = u[qz];
            QQQ(qx,qy,qz,1) = v[qz];
            QQQ(qx,qy,qz,2) = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   return QQQ;
}

template <typename Basis,
          typename Dofs,
          std::enable_if_t<
             get_grad_algo<Basis,Dofs> == GradAlgo::Legacy &&
             get_basis_dim<Basis> == 3,
             bool> = true >
MFEM_HOST_DEVICE inline
auto operator*(const Trans<Grad<Basis>> &basis, const Dofs &u)
{
   constexpr int Dim = 3;
   constexpr int basis_size = get_basis_capacity<Basis>;
   MFEM_SHARED double s_B[basis_size];
   auto Bt = basis.GetBt(s_B);
   MFEM_SHARED double s_G[basis_size];
   auto Gt = basis.GetGt(s_G);

   constexpr int D1D = get_basis_dofs<Basis>;
   constexpr int Q1D = get_basis_quads<Basis>;
   constexpr int MaxDQ = (Q1D > D1D) ? Q1D : D1D;
   // shared memory for temporary/intermediary result tensors.
   MFEM_SHARED double sm0[Dim*MaxDQ*MaxDQ*MaxDQ];
   MFEM_SHARED double sm1[Dim*MaxDQ*MaxDQ*MaxDQ];

   // Load dofs in shared memory
   StaticPointerDTensor<Q1D,Q1D,Q1D,Dim> QQQ(sm0);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_UNROLL(D1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(Dim)
            for (int d = 0; d < Dim; d++)
            {
               QQQ(qx,qy,qz,d) = u(qx,qy,qz,d);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   // X Contraction
   StaticPointerDTensor<Q1D,Q1D,D1D,Dim> QQD(sm1);
   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ(qx,qy,qz,0) * Gt(qx,dx);
               v[qz] += QQQ(qx,qy,qz,1) * Bt(qx,dx);
               w[qz] += QQQ(qx,qy,qz,2) * Bt(qx,dx);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD(dx,qy,qz,0) = u[qz];
            QQD(dx,qy,qz,1) = v[qz];
            QQD(dx,qy,qz,2) = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Y Contraction
   StaticPointerDTensor<Q1D,D1D,D1D,Dim> QDD(sm0);
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qy = 0; qy < Q1D; ++qy)
         {
            MFEM_UNROLL(Q1D)
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD(dx,qy,qz,0) * Bt(qy,dy);
               v[qz] += QQD(dx,qy,qz,1) * Gt(qy,dy);
               w[qz] += QQD(dx,qy,qz,2) * Bt(qy,dy);
            }
         }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD(dx,dy,qz,0) = u[qz];
            QDD(dx,dy,qz,1) = v[qz];
            QDD(dx,dy,qz,2) = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Z Contraction
   constexpr int batchsize = 1;
   Static2dThreadDTensor<batchsize,D1D,D1D,D1D> y;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u[D1D], v[D1D], w[D1D];
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
         MFEM_UNROLL(Q1D)
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_UNROLL(D1D)
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD(dx,dy,qz,0) * Bt(qz,dz);
               v[dz] += QDD(dx,dy,qz,1) * Bt(qz,dz);
               w[dz] += QDD(dx,dy,qz,2) * Gt(qz,dz);
            }
         }
         MFEM_UNROLL(D1D)
         for (int dz = 0; dz < D1D; ++dz)
         {
            y(dx,dy,dz) = (u[dz] + v[dz] + w[dz]);
         }
      }
   }
   return y;
}

} // namespace mfem

#endif // MFEM_TENSOR_GRAD
