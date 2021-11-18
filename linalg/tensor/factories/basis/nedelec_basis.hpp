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

#ifndef MFEM_NEDELEC_BASIS
#define MFEM_NEDELEC_BASIS

#include "basis_impl.hpp"

namespace mfem
{

template <int OpenDofs, int CloseDofs, typename KernelConfig>
struct NedelecBasis
{
   using Config = KernelConfig;

   Config config;
   const int open_dofs;
   const int close_dofs;
   const int quads;
   // Pointers containing open/close basis values at quadrature points.
   const double *B_open;
   const double *Bt_open;
   const double *B_close;
   const double *Bt_close;
   const double *G_open;
   const double *Gt_open;
   const double *G_close;
   const double *Gt_close;

   NedelecBasis(const Config &config,
                const int open_dofs,
                const int close_dofs,
                const int quads,
                const double *b_open,
                const double *bt_open,
                const double *b_close,
                const double *bt_close,
                const double *g_open = nullptr,
                const double *gt_open = nullptr,
                const double *g_close = nullptr,
                const double *gt_close = nullptr)
   : config(config),
     open_dofs(open_dofs),
     close_dofs(close_dofs),
     quads(quads),
     B_open(b_open), Bt_open(bt_open),
     B_close(b_close), Bt_close(bt_close),
     G_open(g_open), Gt_open(gt_open),
     G_close(g_close), Gt_close(gt_close)
   { }

   MFEM_HOST_DEVICE
   int GetQuads() const
   {
      return config.quads;
   }

   MFEM_HOST_DEVICE
   int GetOpenDofs() const
   {
      return open_dofs;
   }

   MFEM_HOST_DEVICE
   int GetCloseDofs() const
   {
      return close_dofs;
   }

   MFEM_HOST_DEVICE inline
   auto GetOpenB(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,Q,OpenDofs> s_B(shared_mem,quads,open_dofs);
      load_with_2dthreads(B_open,quads,open_dofs,s_B);
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetOpenBt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,OpenDofs,Q> s_Bt(shared_mem,open_dofs,quads);
      load_with_2dthreads(Bt_open,open_dofs,quads,s_Bt);
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetCloseB(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,Q,CloseDofs> s_B(shared_mem,quads,close_dofs);
      load_with_2dthreads(B_close,quads,close_dofs,s_B);
      return s_B;
   }

   MFEM_HOST_DEVICE inline
   auto GetCloseBt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,CloseDofs,Q> s_Bt(shared_mem,close_dofs,quads);
      load_with_2dthreads(Bt_close,close_dofs,quads,s_Bt);
      return s_Bt;
   }

   MFEM_HOST_DEVICE inline
   auto GetOpenG(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,Q,OpenDofs> s_G(shared_mem,quads,open_dofs);
      load_with_2dthreads(G_open,quads,open_dofs,s_G);
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetOpenGt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,OpenDofs,Q> s_Gt(shared_mem,open_dofs,quads);
      load_with_2dthreads(Gt_open,open_dofs,quads,s_Gt);
      return s_Gt;
   }

   MFEM_HOST_DEVICE inline
   auto GetCloseG(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,Q,CloseDofs> s_G(shared_mem,quads,close_dofs);
      load_with_2dthreads(G_close,quads,close_dofs,s_G);
      return s_G;
   }

   MFEM_HOST_DEVICE inline
   auto GetCloseGt(double* shared_mem) const
   {
      constexpr int Dim = get_config_dim<Config>;
      constexpr int Q = get_config_quads<Config>;
      StaticSharedBasisTensor<Dim,CloseDofs,Q> s_Gt(shared_mem,close_dofs,quads);
      load_with_2dthreads(Gt_close,close_dofs,quads,s_Gt);
      return s_Gt;
   }
};

template <int OpenDofs, int CloseDofs, typename Config>
auto MakeNedelecBasis(Config &config,
                      const int open_dofs,
                      const int close_dofs,
                      const int quads,
                      const double *b_open,
                      const double *bt_open,
                      const double *b_close,
                      const double *bt_close,
                      const double *g_open = nullptr,
                      const double *gt_open = nullptr,
                      const double *g_close = nullptr,
                      const double *gt_close = nullptr)
{
   return NedelecBasis<OpenDofs,CloseDofs,Config>(config,
                                                  open_dofs,close_dofs,quads,
                                                  b_open,bt_open,
                                                  b_close,bt_close,
                                                  g_open,gt_open,
                                                  g_close,gt_close);
}

///////////////////////
// Nedelec Basis Traits

// is_basis
template <int OpenDofs, int CloseDofs, typename Config>
struct is_basis_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr bool value = true;
};

// is_nedelec_basis
template <typename Basis>
struct is_nedelec_basis_v
{
   static constexpr bool value = false;
};

template <int OpenDofs, int CloseDofs, typename Config>
struct is_nedelec_basis_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr bool value = true;
};

template <typename Basis>
constexpr bool is_nedelec_basis = is_nedelec_basis_v<Basis>::value;

// get_basis_dim
template <int OpenDofs, int CloseDofs, typename Config>
struct get_basis_dim_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = get_config_dim<Config>;
};

// get_open_basis_dofs
template <typename Basis, typename Enable = void>
struct get_open_basis_dofs_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int OpenDofs, int CloseDofs, typename Config>
struct get_open_basis_dofs_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = OpenDofs;
};

template <typename Basis>
constexpr int get_open_basis_dofs = get_open_basis_dofs_v<Basis>::value;

// get_close_basis_dofs
template <typename Basis, typename Enable = void>
struct get_close_basis_dofs_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int OpenDofs, int CloseDofs, typename Config>
struct get_close_basis_dofs_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = CloseDofs;
};

template <typename Basis>
constexpr int get_close_basis_dofs = get_close_basis_dofs_v<Basis>::value;

// get_basis_quads
template <int OpenDofs, int CloseDofs, typename Config>
struct get_basis_quads_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = get_config_quads<Config>;
};

// get_open_basis_capacity
template <typename Basis, typename Enable = void>
struct get_open_basis_capacity_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int OpenDofs, int CloseDofs, typename Config>
struct get_open_basis_capacity_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = OpenDofs * get_config_quads<Config>;
};

template <typename Config>
struct get_open_basis_capacity_v<NedelecBasis<Dynamic,Dynamic,Config>>
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <typename Basis>
constexpr int get_open_basis_capacity = get_open_basis_capacity_v<Basis>::value;

// get_close_basis_capacity
template <typename Basis, typename Enable = void>
struct get_close_basis_capacity_v
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <int OpenDofs, int CloseDofs, typename Config>
struct get_close_basis_capacity_v<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   static constexpr int value = CloseDofs * get_config_quads<Config>;
};

template <typename Config>
struct get_close_basis_capacity_v<NedelecBasis<Dynamic,Dynamic,Config>>
{
   static constexpr int value = DynamicMaxSize*DynamicMaxSize;
};

template <typename Basis>
constexpr int get_close_basis_capacity = get_close_basis_capacity_v<Basis>::value;

// ResultTensor
template <int OpenDofs, int CloseDofs, typename Config>
struct basis_result_tensor<NedelecBasis<OpenDofs,CloseDofs,Config>>
{
   template <int... Sizes>
   using type = typename config_result_tensor<Config>
                   ::template type<Sizes...>;
};

} // mfem namespace

#endif // MFEM_NEDELEC_BASIS
