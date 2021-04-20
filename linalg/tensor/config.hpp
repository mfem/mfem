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

#ifndef MFEM_TENSOR_CONFIG
#define MFEM_TENSOR_CONFIG

#include "util.hpp"

namespace mfem
{

template <int Dim,
          bool IsTensor=false,
          int Dofs=Dynamic,
          int Quads=Dynamic,
          int BatchSize=1>
struct KernelConfig
{
   const int dofs;
   const int quads;

   KernelConfig(int dofs, int quads): dofs(Dofs), quads(Quads)
   {
      // TODO check that if Dofs!= 0 then dofs==Dofs
      // TODO check that if Quads!= 0 then quads==Quads
   }
};

template <int Dim,
          bool IsTensor,
          int BatchSize>
struct KernelConfig<Dim,IsTensor,Dynamic,Dynamic,BatchSize>
{
   const int dofs;
   const int quads;

   KernelConfig(int dofs, int quads): dofs(dofs), quads(quads)
   {
      // TODO check that if Dofs!= 0 then dofs==Dofs
      // TODO check that if Quads!= 0 then quads==Quads
   }
};

template <int Dim, bool IsTensor=false, int Dofs=0, int Quads=0, int BatchSize=1>
auto MakeConfig(int dofs, int quads)
{
   return KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>(dofs,quads);
}

/// KernelConfig Traits

// get_config_dim
template <typename Config>
struct get_config_dim_v;

template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
struct get_config_dim_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
{
   static constexpr int value = Dim;
};

template <typename Config>
constexpr int get_config_dim = get_config_dim_v<Config>::value;

// is_tensor_config
template <typename Config>
struct is_tensor_config_v;

template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
struct is_tensor_config_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
{
   static constexpr bool value = IsTensor;
};

template <typename Config>
constexpr bool is_tensor_config = is_tensor_config_v<Config>::value;

// get_config_dofs
template <typename Config>
struct get_config_dofs_v;

template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
struct get_config_dofs_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
{
   static constexpr int value = Dofs;
};

template <typename Config>
constexpr int get_config_dofs = get_config_dofs_v<Config>::value;

// get_config_quads
template <typename Config>
struct get_config_quads_v;

template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
struct get_config_quads_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
{
   static constexpr int value = Quads;
};

template <typename Config>
constexpr int get_config_quads = get_config_quads_v<Config>::value;

// get_config_batchsize
template <typename Config>
struct get_config_batchsize_v;

template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
struct get_config_batchsize_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
{
   static constexpr int value = BatchSize;
};

template <typename Config>
constexpr int get_config_batchsize = get_config_batchsize_v<Config>::value;

} // mfem namespace

#endif // MFEM_TENSOR_CONFIG
