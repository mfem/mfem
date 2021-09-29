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

   /// Defines the dynamic type of Tensor used for computation on CPU.
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicCPUTensor = DynamicTensor<Rank, T, MaxSize>;

   /// Defines the static type of Tensor used for computation on CPU.
   template <typename T, int BatchSize, int... Sizes>
   using StaticCPUTensor = StaticTensor<T, Sizes...>;

   /// Defines the dynamic type of Tensor used for computation on CUDA.
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicCUDATensor = DynamicBlockTensor<Rank, T, BatchSize, MaxSize>;

   /// Defines the static type of Tensor used for computation on CUDA.
   template <typename T, int BatchSize, int... Sizes>
   using StaticCUDATensor = StaticBlockTensor<T, BatchSize, Sizes...>;

   /// Defines the dynamic type of Tensor used for computation on Hip.
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicHipTensor = DynamicBlockTensor<Rank, T, BatchSize, MaxSize>;

   /// Defines the static type of Tensor used for computation on Hip.
   template <typename T, int BatchSize, int... Sizes>
   using StaticHipTensor = StaticBlockTensor<T, BatchSize, Sizes...>;

#if defined(__CUDA_ARCH__)
   // CUDA types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicDeviceTensor = DynamicCUDATensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = StaticCUDATensor<T,BatchSize,Sizes...>;
#elif defined(__HIP_DEVICE_COMPILE__)
   // Hip types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicDeviceTensor = DynamicHipTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = StaticHipTensor<T,BatchSize,Sizes...>;
#elif defined(FUGAKU_ARCH) // extension exemple
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicDeviceTensor = DynamicCPUTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = StaticCPUTensor<T,BatchSize,Sizes...>;
#else
   // CPU types
   template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicDeviceTensor = DynamicCPUTensor<Rank,T,BatchSize,MaxSize>;

   template <typename T, int BatchSize, int... Sizes>
   using StaticDeviceTensor = StaticCPUTensor<T,BatchSize,Sizes...>;
#endif

   /// Defines the dynamic Tensor type for the compiling architecture
   // template <int Rank, typename T, int BatchSize, int MaxSize = pow(16,Rank)>
   // using DynamicDeviceTensor = DeviceTensorType::dynamic_type<Rank,T,BatchSize,MaxSize>;

   template <int Rank, int BatchSize, int MaxSize = pow(16,Rank)>
   using DynamicDeviceDTensor = DynamicDeviceTensor<Rank,double,BatchSize,MaxSize>;

   /// Defines the static Tensor type for the compiling architecture
   // template <typename T, int BatchSize, int... Sizes>
   // using StaticDeviceTensor = DeviceTensorType::static_type<T,BatchSize,Sizes...>;

   template <int BatchSize, int... Sizes>
   using StaticDeviceDTensor = StaticDeviceTensor<double,BatchSize,Sizes...>;
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
struct KernelConfig
{
   const int dofs;
   const int quads;

   using configs = ConfigSelector<ConfigSetter1, ConfigSetter2, ConfigSetter3,
                                  ConfigSetter4, ConfigSetter5, ConfigSetter6,
                                  ConfigSetter7, ConfigSetter8>;

   KernelConfig(int dofs, int quads) : dofs(dofs), quads(quads)
   {
      // TODO check that if Dofs!= 0 then dofs==Dofs
      // TODO check that if Quads!= 0 then quads==Quads
   }
};

template <typename... ConfigSetters>
auto MakeConfig(int dofs, int quads, ConfigSetters... args)
{
   return KernelConfig<ConfigSetters...>(dofs, quads);
}

/// KernelConfig Traits

// is_config
template <typename Config>
struct is_config_v
{
   static constexpr bool value = false;
};

template <typename... Configs>
struct is_config_v<KernelConfig<Configs...>>
{
   static constexpr bool value = true;
};

template <typename Config>
constexpr bool is_config = is_config_v<Config>::value;

// get_config_dim
template <typename Config>
struct get_config_dim_v;

template <typename... Configs>
struct get_config_dim_v<KernelConfig<Configs...>>
{
   static constexpr int value = KernelConfig<Configs...>::configs::Dim;
};

template <typename Config>
constexpr int get_config_dim = get_config_dim_v<Config>::value;

// is_tensor_config
template <typename Config>
struct is_tensor_config_v;

template <typename... Configs>
struct is_tensor_config_v<KernelConfig<Configs...>>
{
   static constexpr bool value = KernelConfig<Configs...>::configs::IsTensor;
};

template <typename Config>
constexpr bool is_tensor_config = is_tensor_config_v<Config>::value;

// get_config_dofs
template <typename Config>
struct get_config_dofs_v;

template <typename... Configs>
struct get_config_dofs_v<KernelConfig<Configs...>>
{
   static constexpr int value = KernelConfig<Configs...>::configs::Dofs;
};

template <typename Config>
constexpr int get_config_dofs = get_config_dofs_v<Config>::value;

// get_config_quads
template <typename Config>
struct get_config_quads_v;

template <typename... Configs>
struct get_config_quads_v<KernelConfig<Configs...>>
{
   static constexpr int value = KernelConfig<Configs...>::configs::Quads;
};

template <typename Config>
constexpr int get_config_quads = get_config_quads_v<Config>::value;

// get_config_batchsize
template <typename Config>
struct get_config_batchsize_v;

template <typename... Configs>
struct get_config_batchsize_v<KernelConfig<Configs...>>
{
   static constexpr int value = KernelConfig<Configs...>::configs::BatchSize;
};

template <typename Config>
constexpr int get_config_batchsize = get_config_batchsize_v<Config>::value;

// ResultTensor
template <typename Config, typename Enable = std::enable_if_t<is_config<Config>> >
struct config_result_tensor
{
   template <int... Sizes>
   using type = typename Config::configs::template DynamicDeviceDTensor<
                   sizeof...(Sizes),
                   get_config_batchsize<Config> >;
};

template <typename Config>
struct config_result_tensor<Config,
std::enable_if_t<
   is_config<Config> &&
   get_config_dofs<Config> != Dynamic &&
   get_config_quads<Config> != Dynamic
> >
{
   template <int... Sizes>
   using type = typename Config::configs::template StaticDeviceDTensor<
                   get_config_batchsize<Config>,
                   Sizes...>;
};


template <typename Config, int... Sizes>
using ConfigResultTensor = typename config_result_tensor<Config>
                              ::template type<Sizes...>;

// config_use_xthreads
template <typename Config>
constexpr bool config_use_xthreads = is_threaded_layout_dim<
                                        typename ConfigResultTensor<Config,1>
                                           ::layout,0>;

// config_use_ythreads
template <typename Config>
constexpr bool config_use_ythreads = is_threaded_layout_dim<
                                        typename ConfigResultTensor<Config,1>
                                           ::layout,1>;

// config_use_zthreads
template <typename Config>
constexpr bool config_use_zthreads = is_threaded_layout_dim<
                                        typename ConfigResultTensor<Config,1>
                                           ::layout,2>;

// Print function
template <typename... Configs>
std::ostream& operator<<(std::ostream &os,
                         const KernelConfig<Configs...> &config)
{
   using C = KernelConfig<Configs...>;
   os << "Kernel configuration:" << std::endl;
   os << "   Dim is: "
      << get_config_dim<C>
      << std::endl;
   os << "   Tensor: "
      << (is_tensor_config<C> ? "Yes":"No")
      << std::endl;
   os << "   Compilation Dofs is: "
      << ( get_config_dofs<C> == 0 ?
         "Dynamic" : std::to_string(get_config_dofs<C>) )
      << std::endl;
   os << "   Runtime Dofs is: "
      << config.dofs
      << std::endl;
   os << "   Compilation Quads is: "
      << ( get_config_quads<C> == 0 ?
         "Dynamic" : std::to_string(get_config_quads<C>) )
      << std::endl;
   os << "   Runtime Quads is: "
      << config.quads
      << std::endl;
   os << "   BatchSize is: "
      << get_config_batchsize<C>
      << std::endl;
   return os;
}

} // mfem namespace

#endif // MFEM_TENSOR_CONFIG
