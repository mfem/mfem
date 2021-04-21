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

#ifndef MFEM_TENSOR_QDATA
#define MFEM_TENSOR_QDATA

#include "util.hpp"
#include "tensor.hpp"
#include "diagonal_tensor.hpp"
#include "config.hpp"

namespace mfem
{

template <int Dim, int DimComp>
class QuadUtil
{
public:
   template <typename T>
   T initTensor(int dofs, int dim, int ne)
   {
      return InitTensor<T>::makeGlobal(dofs,dim,ne);
   }

   template <typename T>
   T initTensor(int dofs, int dim)
   {
      return InitTensor<T>::makeLocal(dofs,dim);
   }

   template <typename T>
   T initNonTensor(int dofs, int dim, int ne)
   {
      return InitNonTensor<T>::makeGlobal(dofs,dim,ne);
   }

   template <typename T>
   T initNonTensor(int dofs, int dim)
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

/// A class to encapsulate quadrature data in a Diagonal Tensor.
template <int Dim,
          bool IsTensor,
          int Quads,
          int DimComp,
          template <int> class InTensor = DeviceDTensor>
class QData;

/// Tensor DoFs
template <int Dim,
          int Quads,
          int DimComp,
          template <int> class InTensor>
class QData<Dim,true,Quads,DimComp,InTensor>
: private QuadUtil<Dim,DimComp>, public InTensor<Dim+DimComp+1>
{
public:
   using Container = typename InTensor<Dim+DimComp+1>::container;
   using Layout = typename InTensor<Dim+DimComp+1>::layout;

   template <typename Config>
   QData(double *x, Config &config, int ne)
   : InTensor<Dim+DimComp+1>(
        x,
        this->template initTensor<Layout>(config.quads,Dim,ne) )
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the QData of element e
   auto operator()(int e) const
   {
      return makeDiagonalTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }

   auto operator()(int e)
   {
      return makeDiagonalTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }
};

/// Non-Tensor QData
template <int Dim,
          int Quads,
          int DimComp,
          template <int> class InTensor>
class QData<Dim,false,DimComp,Quads,InTensor>
: private QuadUtil<Dim,DimComp>, public InTensor<1+DimComp+1>
{
public:
   using Container = typename InTensor<1+DimComp+1>::container;
   using Layout = typename InTensor<1+DimComp+1>::layout;

   template <typename Config>
   QData(const double *x, Config &config, int ne)
   : InTensor<Dim+DimComp+1>(
        x,
        this->template initNonTensor<Layout>(config.quads,Dim,ne))
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      return makeDiagonalTensor<Dim>(this->template Get<1+DimComp>(e));
   }

   auto operator()(int e)
   {
      return makeDiagonalTensor<Dim>(this->template Get<1+DimComp>(e));
   }
};

/// A structure to choose the right Tensor type for QData according to the Config.
template <typename Config, int DimComp>
struct QuadTensorType;

template <int Dim, int Dofs, int Quads, int BatchSize, int DimComp>
struct QuadTensorType<KernelConfig<Dim,true,Dofs,Quads,BatchSize>,DimComp>
{
   using Tensor = typename rerepeat<Quads,Dim,Dim,DimComp,StaticBlockDTensor>::type;
};

template <int Dim, int BatchSize, int DimComp>
struct QuadTensorType<KernelConfig<Dim,true,Dynamic,Dynamic,BatchSize>,DimComp>
{
   using Tensor = DynamicBlockDTensor<Dim+DimComp,BatchSize>;
};

template <int Dim, int Dofs, int Quads, int BatchSize, int DimComp>
struct QuadTensorType<KernelConfig<Dim,false,Dofs,Quads,BatchSize>,DimComp>
{
   // TODO repeat is not what we need
   // using Tensor = typename repeat<Dim,DimComp,StaticBlockDTensor<Dofs>>::type;
   using Tensor = typename repeat<Dim,DimComp,StaticBlockDTensor>::type;
};

template <int Dim, int BatchSize, int DimComp>
struct QuadTensorType<KernelConfig<Dim,false,Dynamic,Dynamic,BatchSize>,DimComp>
{
   using Tensor = DynamicBlockDTensor<1+DimComp,BatchSize>;
};

template <typename Config, int DimComp>
using QuadTensor = typename QuadTensorType<Config,DimComp>::Tensor;

/// Functor to represent data at quadrature points
template <int DimComp, typename Config>
auto MakeQData(Config &config, double *x, int ne)
{
   return QData<get_config_dim<Config>,
                is_tensor_config<Config>,
                get_config_quads<Config>,
                DimComp,
                DeviceDTensor>(x, config, ne);
}

template <int DimComp, typename Config>
auto MakeQData(Config &config, const double *x, int ne)
{
   return QData<get_config_dim<Config>,
                is_tensor_config<Config>,
                get_config_quads<Config>,
                DimComp,
                ReadDTensor>(const_cast<double*>(x), config, ne);
}

/// A class to encapsulate quadrature data in a Diagonal Symmetric Tensor.
template <int Dim,
          bool IsTensor,
          int Quads,
          int DimComp,
          template <int> class InTensor = DeviceDTensor>
class SymmQData;

/// Tensor DoFs
template <int Dim,
          int Quads,
          int DimComp,
          template <int> class InTensor>
class SymmQData<Dim,true,Quads,DimComp,InTensor>
: private QuadUtil<Dim,DimComp>, public InTensor<Dim+DimComp+1>
{
public:
   using Container = typename InTensor<Dim+DimComp+1>::container;
   using Layout = typename InTensor<Dim+DimComp+1>::layout;

   template <typename Config>
   SymmQData(double *x, Config &config, int ne)
   : InTensor<Dim+DimComp+1>(
        x,
        this->template initTensor<Layout>(config.quads,Dim*(Dim+1)/2,ne) )
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      return makeDiagonalSymmetricTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }

   auto operator()(int e)
   {
      return makeDiagonalSymmetricTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }
};

/// Non-Tensor DoFs
template <int Dim,
          int Quads,
          int DimComp,
          template <int> class InTensor>
class SymmQData<Dim,false,DimComp,Quads,InTensor>
: private QuadUtil<Dim,DimComp>, public InTensor<1+DimComp+1>
{
public:
   using Container = typename InTensor<1+DimComp+1>::container;
   using Layout = typename InTensor<1+DimComp+1>::layout;

   template <typename Config>
   SymmQData(const double *x, Config &config, int ne)
   : InTensor<Dim+DimComp+1>(
        x,
        this->template initNonTensor<Layout>(config.quads,Dim*(Dim+1)/2,ne))
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      return makeDiagonalSymmetricTensor<Dim>(this->template Get<1+DimComp>(e));
   }

   auto operator()(int e)
   {
      return makeDiagonalSymmetricTensor<Dim>(this->template Get<1+DimComp>(e));
   }
};

/// Functor to represent symmetric data at quadrature points
template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, double *x, int ne)
{
   return SymmQData<get_config_dim<Config>,
                    is_tensor_config<Config>,
                    get_config_quads<Config>,
                    DimComp,
                    DeviceDTensor>(x, config, ne);
}

template <int DimComp, typename Config>
auto MakeSymmQData(Config &config, const double *x, int ne)
{
   return SymmQData<get_config_dim<Config>,
                    is_tensor_config<Config>,
                    get_config_quads<Config>,
                    DimComp,
                    ReadDTensor>(const_cast<double*>(x), config, ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_QDATA
