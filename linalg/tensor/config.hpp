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
#include "tensor.hpp"

namespace mfem
{

struct DefaultKernelConfig
{
   static constexpr bool IsTensor = false;
   static constexpr int Dim = Dynamic;
   static constexpr int Dofs = Dynamic; // FIXME: Not sure this should be here
   static constexpr int Quads = Dynamic;
   static constexpr int BatchSize = 1;

   template <typename T, int BatchSize, int... Sizes>
   using StaticHostTensor = StaticCPUTensor<T, BatchSize, Sizes...>;

   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = StaticDeviceTensor<T, BatchSize, Sizes...>;
};

// class to define a use of the default policy values
// avoids ambiguities if we derive from DefaultKernelConfig more than once.
struct DefaultKernelConfigArgs : virtual public DefaultKernelConfig { };

template <bool enable>
struct config_is_tensor : virtual public DefaultKernelConfig
{
   static constexpr bool IsTensor = enable;
};

template <int dim>
struct config_dim_is : virtual public DefaultKernelConfig
{
   static constexpr int Dim = dim;
};

template <int dofs>
struct config_dofs_is : virtual public DefaultKernelConfig
{
   static constexpr int Dofs = dofs;
};

template <int quads>
struct config_quads_is : virtual public DefaultKernelConfig
{
   static constexpr int Quads = quads;
};

template <int bsize>
struct config_batchsize_is : virtual public DefaultKernelConfig
{
   static constexpr int BatchSize = bsize;
};

template <template <typename T, int BatchSize, int... Sizes> class HostTensor>
struct config_static_host_tensor_is : virtual public DefaultKernelConfig
{
   template <typename T, int BatchSize, int... Sizes>
   using StaticHostTensor = HostTensor<T, BatchSize, Sizes...>;
};

template <template <typename T, int BatchSize, int... Sizes> class DeviceTensor>
struct config_static_device_tensor_is : virtual public DefaultKernelConfig
{
   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = DeviceTensor<T, BatchSize, Sizes...>;
};

// Discriminator<> allows having even the same base class more than once
template <typename Base, int D>
class Discriminator : public Base { };

// ConfigSelector<A,B,C,D> creates A,B,C,D as base classes
template <typename Setter1, typename Setter2, typename Setter3,
          typename Setter4, typename Setter5, typename Setter6,
          typename Setter7, typename Setter8>
class ConfigSelector
: public Discriminator<Setter1,1>,
  public Discriminator<Setter2,2>,
  public Discriminator<Setter3,3>,
  public Discriminator<Setter4,4>,
  public Discriminator<Setter5,5>,
  public Discriminator<Setter6,6>,
  public Discriminator<Setter7,7>,
  public Discriminator<Setter8,8>
{ };

template <typename ConfigSetter1 = DefaultKernelConfigArgs,
          typename ConfigSetter2 = DefaultKernelConfigArgs,
          typename ConfigSetter3 = DefaultKernelConfigArgs,
          typename ConfigSetter4 = DefaultKernelConfigArgs,
          typename ConfigSetter5 = DefaultKernelConfigArgs,
          typename ConfigSetter6 = DefaultKernelConfigArgs,
          typename ConfigSetter7 = DefaultKernelConfigArgs,
          typename ConfigSetter8 = DefaultKernelConfigArgs>
struct config
{
   const int dofs;
   const int quads;

   using configs = ConfigSelector<ConfigSetter1, ConfigSetter2, ConfigSetter3,
                                  ConfigSetter4, ConfigSetter5, ConfigSetter6,
                                  ConfigSetter7, ConfigSetter8>;

   config(int dofs, int quads) : dofs(dofs), quads(quads)
   {
      // TODO check that if Dofs!= 0 then dofs==Dofs
      // TODO check that if Quads!= 0 then quads==Quads
   }
};

template <typename... ConfigSetters>
auto MakeConfig(int dofs, int quads, ConfigSetters... args)
{
   return config<ConfigSetters...>(dofs, quads);
}

// template <int Dim,
//           bool IsTensor=false,
//           int Dofs=Dynamic,
//           int Quads=Dynamic,
//           int BatchSize=1>
// struct KernelConfig
// {
//    const int dofs;
//    const int quads;

//    KernelConfig(int dofs, int quads): dofs(Dofs), quads(Quads)
//    {
//       // TODO check that if Dofs!= 0 then dofs==Dofs
//       // TODO check that if Quads!= 0 then quads==Quads
//    }
// };

// template <int Dim,
//           bool IsTensor,
//           int BatchSize>
// struct KernelConfig<Dim,IsTensor,Dynamic,Dynamic,BatchSize>
// {
//    const int dofs;
//    const int quads;

//    KernelConfig(int dofs, int quads): dofs(dofs), quads(quads)
//    {
//       // TODO check that if Dofs!= 0 then dofs==Dofs
//       // TODO check that if Quads!= 0 then quads==Quads
//    }
// };

// template <int Dim, bool IsTensor=false, int Dofs=0, int Quads=0, int BatchSize=1>
// auto MakeConfig(int dofs, int quads)
// {
//    return KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>(dofs,quads);
// }

/// KernelConfig Traits

// get_config_dim
template <typename Config>
struct get_config_dim_v;

// template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// struct get_config_dim_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
// {
//    static constexpr int value = Dim;
// };

template <typename... Configs>
struct get_config_dim_v<config<Configs...>>
{
   static constexpr int value = config<Configs...>::configs::Dim;
};

template <typename Config>
constexpr int get_config_dim = get_config_dim_v<Config>::value;

// is_tensor_config
template <typename Config>
struct is_tensor_config_v;

// template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// struct is_tensor_config_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
// {
//    static constexpr bool value = IsTensor;
// };

template <typename... Configs>
struct is_tensor_config_v<config<Configs...>>
{
   static constexpr bool value = config<Configs...>::configs::IsTensor;
};

template <typename Config>
constexpr bool is_tensor_config = is_tensor_config_v<Config>::value;

// get_config_dofs
template <typename Config>
struct get_config_dofs_v;

// template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// struct get_config_dofs_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
// {
//    static constexpr int value = Dofs;
// };

template <typename... Configs>
struct get_config_dofs_v<config<Configs...>>
{
   static constexpr int value = config<Configs...>::configs::Dofs;
};

template <typename Config>
constexpr int get_config_dofs = get_config_dofs_v<Config>::value;

// get_config_quads
template <typename Config>
struct get_config_quads_v;

// template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// struct get_config_quads_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
// {
//    static constexpr int value = Quads;
// };

template <typename... Configs>
struct get_config_quads_v<config<Configs...>>
{
   static constexpr int value = config<Configs...>::configs::Quads;
};

template <typename Config>
constexpr int get_config_quads = get_config_quads_v<Config>::value;

// get_config_batchsize
template <typename Config>
struct get_config_batchsize_v;

// template <int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// struct get_config_batchsize_v<KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>>
// {
//    static constexpr int value = BatchSize;
// };

template <typename... Configs>
struct get_config_batchsize_v<config<Configs...>>
{
   static constexpr int value = config<Configs...>::configs::BatchSize;
};

template <typename Config>
constexpr int get_config_batchsize = get_config_batchsize_v<Config>::value;

// config_use_1dthreads
template <typename Config>
struct config_use_1dthreads_v
{
   static constexpr bool value = false; // TODO
};

template <typename Config>
constexpr bool config_use_1dthreads = config_use_1dthreads_v<Config>::value;

// config_use_2dthreads
template <typename Config>
struct config_use_2dthreads_v
{
   static constexpr bool value = true; // TODO
};

template <typename Config>
constexpr bool config_use_2dthreads = config_use_2dthreads_v<Config>::value;

// config_use_3dthreads
template <typename Config>
struct config_use_3dthreads_v
{
   static constexpr bool value = false; // TODO
};

template <typename Config>
constexpr bool config_use_3dthreads = config_use_3dthreads_v<Config>::value;

// config_use_xthreads
template <typename Config>
struct config_use_xthreads_v
{
   static constexpr bool value = true; // TODO
};

template <typename Config>
constexpr bool config_use_xthreads = config_use_xthreads_v<Config>::value;

// config_use_ythreads
template <typename Config>
struct config_use_ythreads_v
{
   static constexpr bool value = true; // TODO
};

template <typename Config>
constexpr bool config_use_ythreads = config_use_ythreads_v<Config>::value;

// config_use_zthreads
template <typename Config>
struct config_use_zthreads_v
{
   static constexpr bool value = false; // TODO
};

template <typename Config>
constexpr bool config_use_zthreads = config_use_zthreads_v<Config>::value;

template <typename... Configs>
std::ostream& operator<<(std::ostream &os, const config<Configs...> &c)
{
   using C = config<Configs...>;
   os << "Kernel configuration:" << std::endl;
   os << "   Dim is: "
      << get_config_dim<C>
      << std::endl;
   os << "   Tensor: "
      << (is_tensor_config<C> ? "Yes":"No")
      << std::endl;
   os << "   Dofs is: "
      << ( get_config_dofs<C> == 0 ?
         "Dynamic" : std::to_string(get_config_dofs<C>) )
      << std::endl;
   os << "   Quads is: "
      << ( get_config_quads<C> == 0 ?
         "Dynamic" : std::to_string(get_config_quads<C>) )
      << std::endl;
   os << "   BatchSize is: "
      << get_config_batchsize<C>
      << std::endl;
   return os;
}

} // mfem namespace

#endif // MFEM_TENSOR_CONFIG
