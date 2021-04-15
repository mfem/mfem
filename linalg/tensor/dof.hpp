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

template <int Dim, int VDim>
class DofUtil
{
public:
   template <typename T>
   static T initTensor(int dofs, int ne)
   {
      return InitTensor<T,VDim>::makeGlobal(dofs,ne);
   }

   template <typename T>
   static T initTensor(int dofs)
   {
      return InitTensor<T,VDim>::makeLocal(dofs);
   }

   template <typename T>
   static T initNonTensor(int dofs, int ne)
   {
      return InitNonTensor<T,VDim>::makeGlobal(dofs,ne);
   }

   template <typename T>
   static T initNonTensor(int dofs)
   {
      return InitNonTensor<T,VDim>::makeLocal(dofs);
   }

protected:
   /// A structure to linearize the input sizes to create an object of type T.
   template <typename T, int NComp, int NDim = 0>
   struct InitTensor
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int ne, Sizes... sizes)
      {
         return InitTensor<T,NComp,NDim+1>::makeGlobal(dofs,ne,dofs,sizes...);
      }
   
      template <typename... Sizes>
      static T makeLocal(int dofs, Sizes... sizes)
      {
         return InitTensor<T,NComp,NDim+1>::makeLocal(dofs,dofs,sizes...);
      }
   };

   template <typename T, int NComp>
   struct InitTensor<T,NComp,Dim>
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int ne, Sizes... sizes)
      {
         return T(sizes...,NComp,ne);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs, Sizes... sizes)
      {
         return T(sizes...,NComp);
      }
   };

   template <typename T>
   struct InitTensor<T,0,Dim>
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int ne, Sizes... sizes)
      {
         return T(sizes...,ne);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs, Sizes... sizes)
      {
         return T(sizes...);
      }
   };

   /// A structure to linearize the input sizes to create an object of type T.
   template <typename T, int NComp>
   struct InitNonTensor
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int ne)
      {
         return T(dofs,NComp,ne);
      }
   
      template <typename... Sizes>
      static T makeLocal(int dofs)
      {
         return T(dofs,NComp);
      }
   };

   template <typename T>
   struct InitNonTensor<T,0>
   {
      static T makeGlobal(int dofs, int ne)
      {
         return T(dofs,ne);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs)
      {
         return T(dofs);
      }
   };
};

/// A class to encapsulate degrees of freedom in a Tensor.
template <int Dim,
          bool IsTensor,
          int Dofs,
          int VDim,
          typename OutTensor,
          template <int> class InTensor = DeviceDTensor>
class DegreesOfFreedom;

/// Tensor DoFs
template <int Dim,
          int Dofs,
          int VDim,
          typename OutTensor,
          template <int> class InTensor>
class DegreesOfFreedom<Dim,true,Dofs,VDim,OutTensor,InTensor>
: public InTensor<Dim+(VDim>0)+1>
{
public:
   using Container = typename InTensor<Dim+(VDim>0)+1>::container;
   using Layout = typename InTensor<Dim+(VDim>0)+1>::layout;

   template <typename Config>
   DegreesOfFreedom(double *x, Config &config, int ne)
   : InTensor<Dim+(VDim>0)+1>(
        Container(x,ne*pow(config.dofs,Dim)*VDim),
        DofUtil<Dim,VDim>::template initTensor<Layout>(config.dofs,ne) )
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      OutTensor u_e(this->template Get<Dim+(VDim>0)>(e));
      return u_e;
   }

   auto operator()(int e)
   {
      return this->template Get<Dim+(VDim>0)>(e);
   }
};

/// Non-Tensor DoFs
template <int Dim,
          int Dofs,
          int DimComp,
          typename OutTensor,
          template <int> class InTensor>
class DegreesOfFreedom<Dim,false,DimComp,Dofs,OutTensor,InTensor>
: public InTensor<1+DimComp+1>
{
public:
   using Container = typename InTensor<1+DimComp+1>::container;
   using Layout = typename InTensor<1+DimComp+1>::layout;

   template <typename Config>
   DegreesOfFreedom(const double *x, Config &config, int ne)
   : InTensor<Dim+DimComp+1>(
        Container(x,ne*config.dofs*pow(Dim,DimComp)),
        DofUtil<Dim,DimComp>::template initNonTensor<Layout>(config.dofs,ne))
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      auto u_e = DofUtil<Dim,DimComp>::template initNonTensor<OutTensor>(this->template Size<0>());
      u_e = this->template Get<1+DimComp>(e);
      return u_e;
   }

   auto operator()(int e)
   {
      return this->template Get<1+DimComp>(e);
   }
};

/// A structure to choose the right Tensor type for DoFs according to the Config.
template <typename Config, int VDim, int DDim>
struct DofTensorType;

// Static tensor DoFs
template <int Dim, int Dofs, int Quads, int BatchSize, int VDim, int DDim>
struct DofTensorType<KernelConfig<Dim,true,Dofs,Quads,BatchSize>,VDim,DDim>
{
   using type = instantiate<
                  StaticDeviceDTensor,
                  append<
                     int_list<BatchSize>,
                     append<
                        int_repeat<Dofs,Dim>,
                        int_repeat<Dim,VDim>
                     >
                  >
                >;
};

// Dynamic tensor DoFs
template <int Dim, int BatchSize, int VDim, int DDim>
struct DofTensorType<KernelConfig<Dim,true,Dynamic,Dynamic,BatchSize>,VDim,DDim>
{
   using type = DynamicDeviceDTensor<Dim+(VDim>0)+(DDim>1),BatchSize>;
};

// Static non-tensor DoFs
template <int Dim, int Dofs, int Quads, int BatchSize, int VDim, int DDim>
struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,VDim,DDim>
{
   using type = StaticDeviceDTensor<BatchSize,Dofs,VDim,DDim>;
};
// VDim == 1
template <int Dim, int Dofs, int Quads, int BatchSize, int DDim>
struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,1,DDim>
{
   using type = StaticDeviceDTensor<BatchSize,Dofs,DDim>;
};
// DDim == 1
template <int Dim, int Dofs, int Quads, int BatchSize, int VDim>
struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,VDim,1>
{
   using type = StaticDeviceDTensor<BatchSize,Dofs,VDim>;
};
// VDim == 1 and DDim == 1
template <int Dim, int Dofs, int Quads, int BatchSize>
struct DofTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,1,1>
{
   using type = StaticDeviceDTensor<BatchSize,Dofs>;
};

// Dynamic non-tensor DoFs
template <int Dim, int BatchSize, int VDim, int DDim>
struct DofTensorType<KernelConfig<Dim,false,Dynamic,Dynamic,BatchSize>,VDim,DDim>
{
   using type = DynamicDeviceDTensor<1+(VDim>0)+(DDim>1),BatchSize>;
};

template <typename Config, int VDim, int DDim = 1>
using DofTensor = typename DofTensorType<Config,VDim,DDim>::type;

/// Functor to represent degrees of freedom
template <int VDim, typename Config>
auto MakeDoFs(Config &config, double *x, int ne)
{
   return DegreesOfFreedom<get_config_dim<Config>,
                           is_tensor_config<Config>,
                           get_config_dofs<Config>,
                           VDim,
                           DofTensor<Config,VDim>,
                           DeviceDTensor>(x, config, ne);
}

template <int VDim, typename Config>
auto MakeDoFs(Config &config, const double *x, int ne)
{
   return DegreesOfFreedom<get_config_dim<Config>,
                           is_tensor_config<Config>,
                           get_config_dofs<Config>,
                           VDim,
                           DofTensor<Config,VDim>,
                           ReadDTensor>(const_cast<double*>(x), config, ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_DOF
