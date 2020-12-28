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

template <int Dim, int DimComp>
class DofUtil
{
public:
   template <typename T>
   static T initTensor(int dofs, int dim, int ne)
   {
      return InitTensor<T>::makeGlobal(dofs,dim,ne);
   }

   template <typename T>
   static T initTensor(int dofs, int dim)
   {
      return InitTensor<T>::makeLocal(dofs,dim);
   }

   template <typename T>
   static T initNonTensor(int dofs, int dim, int ne)
   {
      return InitNonTensor<T>::makeGlobal(dofs,dim,ne);
   }

   template <typename T>
   static T initNonTensor(int dofs, int dim)
   {
      return InitNonTensor<T>::makeLocal(dofs,dim);
   }

protected:
   /// A structure to linearize the input sizes to create an object of type T.
   template <typename T, int NDim = 0, int NComp = 0>
   struct InitTensor
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int dim, int ne, Sizes... sizes)
      {
         return InitTensor<T,NDim,NComp+1>::makeGlobal(dofs,dim,ne,dim,sizes...);
      }
   
      template <typename... Sizes>
      static T makeLocal(int dofs, int dim, Sizes... sizes)
      {
         return InitTensor<T,NDim,NComp+1>::makeLocal(dofs,dim,dim,sizes...);
      }
   };

   template <typename T, int NDim>
   struct InitTensor<T,NDim,DimComp>
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int dim, int ne, Sizes... sizes)
      {
         return InitTensor<T,NDim+1,DimComp>::makeGlobal(dofs,dim,ne,dofs,sizes...);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs, int dim, Sizes... sizes)
      {
         return InitTensor<T,NDim+1,DimComp>::makeLocal(dofs,dim,dofs,sizes...);
      }
   };

   template <typename T>
   struct InitTensor<T,Dim,DimComp>
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int dim, int ne, Sizes... sizes)
      {
         return T(sizes...,ne);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs, int dim, Sizes... sizes)
      {
         return T(sizes...);
      }
   };

   /// A structure to linearize the input sizes to create an object of type T.
   template <typename T, int NComp = 0>
   struct InitNonTensor
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int dim, int ne, Sizes... sizes)
      {
         return InitNonTensor<T,NComp+1>::makeGlobal(dofs,dim,ne,dim,sizes...);
      }
   
      template <typename... Sizes>
      static T makeLocal(int dofs, int dim, Sizes... sizes)
      {
         return InitNonTensor<T,NComp+1>::makeLocal(dofs,dim,dim,sizes...);
      }
   };

   template <typename T>
   struct InitNonTensor<T,DimComp>
   {
      template <typename... Sizes>
      static T makeGlobal(int dofs, int dim, int ne, Sizes... sizes)
      {
         return T(dofs,sizes...,ne);
      }

      template <typename... Sizes>
      static T makeLocal(int dofs, int dim, Sizes... sizes)
      {
         return T(dofs,sizes...);
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
: public InTensor<Dim+1+1>
{
public:
   using Container = typename InTensor<Dim+1+1>::container;
   using Layout = typename InTensor<Dim+1+1>::layout;

   template <typename Config>
   DegreesOfFreedom(double *x, Config &config, int ne)
   : InTensor<Dim+(VDim>1)+1>(
        Container(x,ne*pow(config.dofs,Dim)*VDim),
                  DofUtil<Dim,VDim>::template initTensor<Layout>(config.dofs,VDim,ne) ) // TODO DofUtil needs to change
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      // TODO VDim instead of Dim^DimComp
      auto u_e = DofUtil<Dim,DimComp>::template initTensor<OutTensor>(this->template Size<0>(),Dim);
      u_e = this->template Get<Dim+DimComp>(e);
      return u_e;
   }

   auto operator()(int e)
   {
      return this->template Get<Dim+DimComp>(e);
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
        DofUtil<Dim,DimComp>::template initNonTensor<Layout>(config.dofs,Dim,ne))
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      auto u_e = DofUtil<Dim,DimComp>::template initNonTensor<OutTensor>(this->template Size<0>(),Dim);
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
   // TODO This is not correct currently
   using type = typename rerepeat<Dofs,Dim,Dim,VDim,BatchSize,StaticDeviceDTensor>::type;
   // TODO using type = typename rererepeat<Dofs,Dim,VDim,1,DDim,1,BatchSize,StaticDeviceDTensor>::type;
};

// Dynamic tensor DoFs
template <int Dim, int BatchSize, int VDim, int DDim>
struct DofTensorType<KernelConfig<Dim,true,Dynamic,Dynamic,BatchSize>,VDim,DDim>
{
   using type = DynamicDeviceDTensor<Dim+(VDim>1)+(DDim>1),BatchSize>;
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
   using type = DynamicDeviceDTensor<1+(VDim>1)+(DDim>1),BatchSize>;
};

template <typename Config, int VDim, int DDim = 1>
using DofTensor = typename DofTensorType<Config,VDim,DDim>::type;

/// Functor to represent degrees of freedom
template <int VDim, typename Config>
auto MakeDoFs(Config &config, double *x, int ne)
{
   return DegreesOfFreedom<Config::dim,
                           Config::is_tensor,
                           Config::Dofs,
                           VDim,
                           DofTensor<Config,VDim>,
                           DeviceDTensor>(x, config, ne);
}

template <int VDim, typename Config>
auto MakeDoFs(Config &config, const double *x, int ne)
{
   return DegreesOfFreedom<Config::dim,
                           Config::is_tensor,
                           Config::Dofs,
                           VDim,
                           DofTensor<Config,VDim>,
                           ReadDTensor>(const_cast<double*>(x), config, ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_DOF
