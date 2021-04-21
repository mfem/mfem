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

#ifndef MFEM_TENSOR_DOF
#define MFEM_TENSOR_DOF

#include "util.hpp"
#include "tensor.hpp"
#include "config.hpp"

namespace mfem
{

// template <int Dim, int VDim>
// class DofUtil
// {
// public:
//    template <typename T>
//    static T initTensor(int dofs, int ne)
//    {
//       return InitTensor<T,VDim>::makeGlobal(dofs,ne);
//    }

//    template <typename T>
//    static T initTensor(int dofs)
//    {
//       return InitTensor<T,VDim>::makeLocal(dofs);
//    }

//    template <typename T>
//    static T initNonTensor(int dofs, int ne)
//    {
//       return InitNonTensor<T,VDim>::makeGlobal(dofs,ne);
//    }

//    template <typename T>
//    static T initNonTensor(int dofs)
//    {
//       return InitNonTensor<T,VDim>::makeLocal(dofs);
//    }

// protected:
//    /// A structure to linearize the input sizes to create an object of type T.
//    template <typename T, int NComp, int NDim = 0>
//    struct InitTensor
//    {
//       template <typename... Sizes>
//       static T makeGlobal(int dofs, int ne, Sizes... sizes)
//       {
//          return InitTensor<T,NComp,NDim+1>::makeGlobal(dofs,ne,dofs,sizes...);
//       }
   
//       template <typename... Sizes>
//       static T makeLocal(int dofs, Sizes... sizes)
//       {
//          return InitTensor<T,NComp,NDim+1>::makeLocal(dofs,dofs,sizes...);
//       }
//    };

//    template <typename T, int NComp>
//    struct InitTensor<T,NComp,Dim>
//    {
//       template <typename... Sizes>
//       static T makeGlobal(int dofs, int ne, Sizes... sizes)
//       {
//          return T(sizes...,NComp,ne);
//       }

//       template <typename... Sizes>
//       static T makeLocal(int dofs, Sizes... sizes)
//       {
//          return T(sizes...,NComp);
//       }
//    };

//    template <typename T>
//    struct InitTensor<T,0,Dim>
//    {
//       template <typename... Sizes>
//       static T makeGlobal(int dofs, int ne, Sizes... sizes)
//       {
//          return T(sizes...,ne);
//       }

//       template <typename... Sizes>
//       static T makeLocal(int dofs, Sizes... sizes)
//       {
//          return T(sizes...);
//       }
//    };

//    /// A structure to linearize the input sizes to create an object of type T.
//    template <typename T, int NComp>
//    struct InitNonTensor
//    {
//       template <typename... Sizes>
//       static T makeGlobal(int dofs, int ne)
//       {
//          return T(dofs,NComp,ne);
//       }
   
//       template <typename... Sizes>
//       static T makeLocal(int dofs)
//       {
//          return T(dofs,NComp);
//       }
//    };

//    template <typename T>
//    struct InitNonTensor<T,0>
//    {
//       static T makeGlobal(int dofs, int ne)
//       {
//          return T(dofs,ne);
//       }

//       template <typename... Sizes>
//       static T makeLocal(int dofs)
//       {
//          return T(dofs);
//       }
//    };
// };

// template <typename T, int Dim, int VDim, int N=0>
// struct InitLayout
// {
//    template <typename... Sizes>
//    static T make(int dofs, int ne, Sizes... sizes)
//    {
//       return InitLayout<T,Dim,VDim,N+1>::make(dofs,ne,dofs,sizes...);
//    }
// };

// template <typename T, int Dim, int VDim>
// struct InitLayout<T,Dim,VDim,Dim>
// {
//    template <typename... Sizes>
//    static T make(int dofs, int ne, Sizes... sizes)
//    {
//       return T(sizes...,VDim,ne);
//    }
// };

// template <typename T, int Dim>
// struct InitLayout<T,Dim,0,Dim>
// {
//    template <typename... Sizes>
//    static T make(int dofs, int ne, Sizes... sizes)
//    {
//       return T(sizes...,ne);
//    }
// };

/// A class to encapsulate degrees of freedom in a Tensor.
template <typename DofTensor>
// template <int Dim,
//           bool IsTensor,
//           int Dofs,
//           int VDim,
//           typename DofTensor>
class DegreesOfFreedom
: public DofTensor
{
public:
   // // TODO implement a more standard constructor with Sizes...
   // template <typename Config>
   // DegreesOfFreedom(double *x, Config &config, int ne)
   // : DofTensor(
   //      x,//ne*(IsTensor?pow(config.dofs,Dim):config.dofs)*(VDim>0?VDim:1)),
   //      InitLayout<typename DofTensor::layout,Dim,VDim>::make(config.dofs,ne) )
   // {
   //    // TODO static asserts Config values
   // }

   // template <typename Config, typename... Sizes>
   // DegreesOfFreedom(double *x, Config &config, Sizes... sizes)
   // : DofTensor(x, sizes...)
   // {
   //    // TODO static asserts Config values
   // }

   template <typename... Sizes>
   DegreesOfFreedom(double *x, Sizes... sizes)
   : DofTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e);
   }

   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e);
   }
};

// /// Tensor DoFs
// template <int Dim,
//           int Dofs,
//           int VDim,
//           typename Container,
//           typename Layout>
// class DegreesOfFreedom<Dim,true,Dofs,VDim,Container,Layout>
// : public Tensor< get_layout_rank<Layout>, Container, Layout>
// {
// public:
//    using InTensor = Tensor<get_layout_rank<Layout>,Container,Layout>;

//    template <typename Config>
//    DegreesOfFreedom(double *x, Config &config, int ne)
//    : InTensor(
//         Container(x,ne*pow(config.dofs,Dim)*(VDim>0?VDim:1)),
//         DofUtil<Dim,VDim>::template initTensor<Layout>(config.dofs,ne) )
//    {
//       // TODO static asserts Config values
//    }

//    /// Returns a Tensor corresponding to the DoFs of element e
//    auto operator()(int e) const
//    {
//       return this->template Get<Dim+(VDim>0)>(e);
//    }

//    auto operator()(int e)
//    {
//       return this->template Get<Dim+(VDim>0)>(e);
//    }
// };

// /// Non-Tensor DoFs
// template <int Dim,
//           int Dofs,
//           int VDim,
//           typename Container,
//           typename Layout>
// class DegreesOfFreedom<Dim,false,VDim,Dofs,InTensor>
// : public InTensor<1+VDim+1>
// {
// public:
//    using Container = typename InTensor<1+VDim+1>::container;
//    using Layout = typename InTensor<1+VDim+1>::layout;

//    template <typename Config>
//    DegreesOfFreedom(const double *x, Config &config, int ne)
//    : InTensor<Dim+VDim+1>(
//         Container(x,ne*config.dofs*pow(Dim,VDim)),
//         DofUtil<Dim,VDim>::template initNonTensor<Layout>(config.dofs,ne))
//    {
//       // TODO static asserts Config values 
//    }

//    /// Returns a Tensor corresponding to the DoFs of element e
//    auto operator()(int e) const
//    {
//       return this->template Get<1+VDim>(e);
//    }

//    auto operator()(int e)
//    {
//       return this->template Get<1+VDim>(e);
//    }
// };

// /// A structure to choose the right Tensor type for DoFs according to the Config.
// template <typename Config, int VDim, int DDim>
// struct DofTensorType;

// // Static tensor DoFs
// template <int Dim, int Dofs, int Quads, int BatchSize, int VDim, int DDim>
// struct DofTensorType<KernelConfig<Dim,true,Dofs,Quads,BatchSize>,VDim,DDim>
// {
//    using type = instantiate<
//                   StaticDeviceDTensor,
//                   append<
//                      int_list<BatchSize>,
//                      append<
//                         int_repeat<Dofs,Dim>,
//                         int_repeat<Dim,VDim>
//                      >
//                   >
//                 >;
// };

// // Dynamic tensor DoFs
// template <int Dim, int BatchSize, int VDim, int DDim>
// struct DofTensorType<KernelConfig<Dim,true,Dynamic,Dynamic,BatchSize>,VDim,DDim>
// {
//    using type = DynamicDeviceDTensor<Dim+(VDim>0)+(DDim>1),BatchSize>;
// };

// // Static non-tensor DoFs
// template <int Dim, int Dofs, int Quads, int BatchSize, int VDim, int DDim>
// struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,VDim,DDim>
// {
//    using type = StaticDeviceDTensor<BatchSize,Dofs,VDim,DDim>;
// };
// // VDim == 1
// template <int Dim, int Dofs, int Quads, int BatchSize, int DDim>
// struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,1,DDim>
// {
//    using type = StaticDeviceDTensor<BatchSize,Dofs,DDim>;
// };
// // DDim == 1
// template <int Dim, int Dofs, int Quads, int BatchSize, int VDim>
// struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,VDim,1>
// {
//    using type = StaticDeviceDTensor<BatchSize,Dofs,VDim>;
// };
// // VDim == 1 and DDim == 1
// template <int Dim, int Dofs, int Quads, int BatchSize>
// struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,1,1>
// {
//    using type = StaticDeviceDTensor<BatchSize,Dofs>;
// };

// // Dynamic non-tensor DoFs
// template <int Dim, int BatchSize, int VDim, int DDim>
// struct DofTensorType<KernelConfig<Dim,false,Dynamic,Dynamic,BatchSize>,VDim,DDim>
// {
//    using type = DynamicDeviceDTensor<1+(VDim>0)+(DDim>1),BatchSize>;
// };

// template <typename Config, int VDim, int DDim = 1>
// using DofTensor = typename DofTensorType<Config,VDim,DDim>::type;

// template <int Dim,
//           bool IsTensor,
//           int Dofs,
//           int VDim>
// struct DoFsMaker
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             DeviceContainer<double>,
//                             DynamicLayout<Rank> >;
//    static auto make()
//    {
//       return DegreesOfFreedom<Dim,
//                               IsTensor,
//                               Dofs,
//                               VDim,
//                               DofTensor
//                               >(x, config, ne);
//    }
// };

/// A structure to call the constructor of T with the right sizes...
template <typename T, bool IsTensor, int Dim, int VDim, int N=1>
struct Init;

// Tensor
template <typename T, int Dim, int VDim, int N>
struct Init<T,true,Dim,VDim,N>
{
   template <typename... Sizes>
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return Init<T,true,Dim,VDim,N+1>::make(x, dofs, ne, dofs, sizes...);
   }
};

template <typename T, int Dim, int VDim>
struct Init<T,true,Dim,VDim,Dim>
{
   template <typename... Sizes>
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return T(x, sizes..., VDim, ne);
   }
};

template <typename T, int Dim>
struct Init<T,Dim,0,Dim>
{
   template <typename... Sizes>
   static T make(double *x, int dofs, int ne, Sizes... sizes)
   {
      return T(x, sizes..., ne);
   }
};

// Non-Tensor
template <typename T, int Dim>
struct Init<T,false,Dim,0,1>
{
   static T make(double *x, int dofs, int ne)
   {
      return T(x, dofs, ne);
   }
};

template <typename T, int Dim, int VDim>
struct Init<T,false,Dim,VDim,1>
{
   static T make(double *x, int dofs, int ne)
   {
      return T(x, dofs, VDim, ne);
   }
};

/// get_dof_layout
template <bool IsTensor, int Dofs, int Dim, int VDim>
struct get_dof_layout_t;
// {
//    static constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using type = DynamicLayout<Rank>;
// };

// Tensor Dynamic no-VDim
template <int Dim>
struct get_dof_layout_t<true, Dynamic, Dim, 0>
{
   static constexpr int Rank = Dim+1;
   using type = DynamicLayout<Rank>;
};

// Tensor Dynamic VDim
template <int Dim, int VDim>
struct get_dof_layout_t<true, Dynamic, Dim, VDim>
{
   static constexpr int Rank = Dim+1+1;
   using type = DynamicLayout<Rank>;
};

// Non-Tensor Dynamic no-VDim
template <int Dim>
struct get_dof_layout_t<false, Dynamic, Dim, 0>
{
   static constexpr int Rank = 1+1;
   using type = DynamicLayout<Rank>;
};

// Non-Tensor Dynamic VDim
template <int Dim, int VDim>
struct get_dof_layout_t<false, Dynamic, Dim, VDim>
{
   static constexpr int Rank = 1+1+1;
   using type = DynamicLayout<Rank>;
};

// Tensor Static no-VDim
template <int Dofs, int Dim>
struct get_dof_layout_t<true, Dofs, Dim, 0>
{
   using sizes = int_repeat<Dofs,Dim>;
   using type = instantiate< StaticELayout, sizes >;
};

// Tensor Static VDim
template <int Dofs, int Dim, int VDim>
struct get_dof_layout_t<true, Dofs, Dim, VDim>
{
   using sizes = append< int_repeat<Dofs,Dim>, int_list<VDim> >;
   using type = instantiate< StaticELayout, sizes >;
};

// Non-Tensor Static no-VDim
template <int Dofs, int Dim>
struct get_dof_layout_t<false, Dofs, Dim, 0>
{
   using type = StaticELayout<Dofs>;
};

// Non-Tensor Static VDim
template <int Dofs, int Dim, int VDim>
struct get_dof_layout_t<false, Dofs, Dim, VDim>
{
   using type = StaticELayout<Dofs,VDim>;
};

template <bool IsTensor, int Dofs, int Dim, int VDim>
using get_dof_layout = typename get_dof_layout_t<IsTensor,Dofs,Dim,VDim>::type;

/// Functor to represent degrees of freedom
template <int VDim, typename Config>
auto MakeDoFs(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<Rank,
                            double,
                            DeviceContainer<double>,
                            Layout >;
   using Dof = DegreesOfFreedom<DofTensor>;
   return Init<Dof,IsTensor,Dim,VDim>::make(x,config.dofs,ne);
}

template <int VDim, typename Config>
auto MakeDoFs(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<Rank,
                            double,
                            ReadContainer<double>,
                            Layout >;
   using Dof = DegreesOfFreedom<DofTensor>;
   return Init<Dof,IsTensor,Dim,VDim>::make(const_cast<double*>(x),config.dofs,ne);
}

// /// Functor to represent degrees of freedom
// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> == Dynamic,
//              bool> = true >
// auto MakeDoFs(Config &config, double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             DeviceContainer<double>,
//                             DynamicLayout<Rank> >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(x, config, ne);
// }

// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> != Dynamic &&
//              VDim == 0,
//              bool> = true >
// auto MakeDoFs(Config &config, double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using Layout = instantiate< StaticELayout, int_repeat<Dofs,Dim> >;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             DeviceContainer<double>,
//                             Layout >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(x, config, ne);
// }

// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> != Dynamic &&
//              (VDim > 0),
//              bool> = true >
// auto MakeDoFs(Config &config, double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using Layout = instantiate< 
//                      StaticELayout,
//                      append< int_repeat<Dofs,Dim>, int_list<VDim> >
//                   >;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             DeviceContainer<double>,
//                             Layout >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(x, config, ne);
// }

// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> == Dynamic,
//              bool> = true >
// auto MakeDoFs(Config &config, const double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             ReadContainer<double>,
//                             DynamicLayout<Rank> >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(const_cast<double*>(x), config, ne);
// }

// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> != Dynamic &&
//              VDim == 0,
//              bool> = true >
// auto MakeDoFs(Config &config, const double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using Layout = instantiate< StaticELayout, int_repeat<Dofs,Dim> >;
//    // using Layout = DynamicLayout<Rank>;
//    // using Layout = instantiate< 
//    //                   StaticLayout,
//    //                   append< int_repeat<Dofs,Dim>, int_list<Dynamic> >
//    //                >;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             ReadContainer<double>,
//                             Layout >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(const_cast<double*>(x), config, ne);
// }

// template <int VDim, typename Config,
//           std::enable_if_t<
//              get_config_dofs<Config> != Dynamic &&
//              (VDim > 0),
//              bool> = true >
// auto MakeDoFs(Config &config, const double *x, int ne)
// {
//    constexpr int Dim = get_config_dim<Config>;
//    constexpr bool IsTensor = is_tensor_config<Config>;
//    constexpr int Dofs = get_config_dofs<Config>;
//    constexpr int Rank = (IsTensor?Dim:1)+(VDim>0)+1;
//    using Layout = instantiate< 
//                      StaticELayout,
//                      append< int_repeat<Dofs,Dim>, int_list<VDim> >
//                   >;
//    // using Layout = DynamicLayout<Rank>;
//    using DofTensor = Tensor<Rank,
//                             double,
//                             ReadContainer<double>,
//                             Layout >;
//    return DegreesOfFreedom<Dim,
//                            IsTensor,
//                            Dofs,
//                            VDim,
//                            DofTensor
//                           >(const_cast<double*>(x), config, ne);
// }

} // mfem namespace

#endif // MFEM_TENSOR_DOF
