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
// /// A class to encapsulate quadrature data in a Tensor.
// template <int Dim, int DimComp = 0, bool IsTensor = false, int Quads = 0>
// class QData;

// /// Tensor DoFs with size known
// template <int Dim, int DimComp, int Quads>
// class QData<Dim,DimComp,true,Quads> : public Tensor<Dim+DimComp+1> // TODO
// {
// private:
//    static constexpr int dim = Dim;
//    static constexpr int dim_comp = DimComp;
//    // Declare a StaticBlockDTensor type with Dim dimensions of size Quads,
//    // and DimComp dimensions of size Dim.
//    using OutTensor = typename rerepeat<Quads,Dim,Dim,DimComp,StaticBlockDTensor>::type;

// public:
//    /// Returns a Tensor corresponding to the DoFs of element e
//    const auto operator()(int e) const
//    {
//       // return this->template Get<Dim+DimComp>(e);
//       OutTensor u_e;
//       u_e = this->template Get<Dim+DimComp>(e);
//       return u_e;
//    }
// };

// /// Tensor DoFs with size unknown
// template <int Dim, int DimComp>
// class QData<Dim,DimComp,true,0> : public Tensor<Dim+DimComp+1>
// {
// private:
//    static constexpr int dim = Dim;
//    using OutTensor = DynamicBlockDTensor<Dim+DimComp>; // Change type according to device

// public:
//    /// Returns a Tensor corresponding to the DoFs of element e
//    auto operator()(int e) const
//    {
//       auto u_e = InitOutTensor(this->template Size<0>(),dim);
//       u_e = this->template Get<Dim+DimComp>(e);
//       return u_e;
//    }

// private:
//    OutTensor InitOutTensor(int dofs, int dim)
//    {
//       return InitTensor<0,0>::make(dofs,dim);
//    }

//    /// A structure to linearize the input sizes to create a Tensor.
//    template <int NDim, int NComp>
//    struct InitTensor
//    {
//       template <typename... Sizes>
//       OutTensor make(int D, int dim, Sizes... sizes)
//       {
//          return InitTensor<NDim,NComp+1>::make(D,dim,dim,sizes...);
//       }
//    };

//    template <int NDim>
//    struct InitTensor<NDim,DimComp>
//    {
//       template <typename... Sizes>
//       OutTensor make(int D, int dim, Sizes... sizes)
//       {
//          return InitTensor<NDim+1,DimComp>(D,dim,D,sizes...);
//       }
//    };

//    template <>
//    struct InitTensor<Dim,DimComp>
//    {
//       template <typename... Sizes>
//       OutTensor make(int D, int dim, Sizes... sizes)
//       {
//          return OutTensor(sizes...);
//       }
//    };
// };

// /// Non-Tensor DoFs with size known
// template <int Dim, int DimComp, int Quads>
// class QData<Dim,DimComp,false,Quads> : public Tensor<DimComp+1>
// {
// private:
//    static constexpr int dim = Dim;
//    static constexpr int dcomp = DimComp;
//    using OutTensor = typename rerepeat<Quads,1,Dim,DimComp,StaticSharedDTensor>::type;

// public:
//    /// Returns a Tensor corresponding to the DoFs of element e
//    auto operator()(int e) const
//    {
//       OutTensor u_e;
//       u_e = this->template Get<Dim+DimComp>(e);
//       return u_e;
//    }
// };

// /// Non-Tensor DoFs with size unknown
// template <int Dim, int DimComp>
// class QData<Dim,DimComp,false,0> : public Tensor<1+DimComp+1>
// {
// private:
//    static constexpr int dim = Dim;
//    using OutTensor = DynamicSharedDTensor<1+DimComp>;

// public:
//    /// Returns a Tensor corresponding to the DoFs of element e
//    auto operator()(int e) const
//    {
//       OutTensor u_e;
//       u_e = this->template Get<DimComp>(e);
//       return u_e;
//    }
// };

// template <int Dim, int DimComp, bool IsTensor, int Quads>
// class MutQData : QData<Dim,DimComp,IsTensor,Quads>
// {
// public:
//    auto operator()(int e)
//    {
//       return this->template Get<Dim+DimComp>(e);
//    }
// };

// /// Functor to represent degrees of freedom
// template <int DimComp, int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// auto MakeQData(KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize> &config,
//                double *x,
//                int ne)
// {
//    return MutQData<Dim,DimComp,IsTensor,Quads>(x, config.dofs, ne);
// }

// template <int DimComp, int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
// auto MakeQData(KernelConfig<Dim,IsTensor,Dofs,Dofs,BatchSize> &config,
//                const double *x,
//                int ne)
// {
//    return QData<Dim,DimComp,IsTensor,Quads>(const_cast<double*>(x), config.dofs, ne);
// }

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

/// A class to encapsulate degrees of freedom in a Tensor.
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
        Container(x,ne*pow(config.quads,Dim)*pow(Dim,DimComp)),
        this->template initTensor<Layout>(config.quads,Dim,ne) )
   {
      // TODO static asserts Config values 
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      return makeDiagonalTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }

   auto operator()(int e)
   {
      return makeDiagonalTensor<Dim>(this->template Get<Dim+DimComp>(e));
   }
};

/// Non-Tensor DoFs
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
        Container(x,ne*config.quads*pow(Dim,DimComp)),
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

/// A structure to choose the right Tensor type for DoFs according to the Config.
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

/// Functor to represent degrees of freedom
template <int DimComp, typename Config>
auto MakeQData(Config &config, double *x, int ne)
{
   return QData<Config::dim,
                Config::is_tensor,
                Config::Quads,
                DimComp,
                DeviceDTensor>(x, config, ne);
}

template <int DimComp, typename Config>
auto MakeQData(Config &config, const double *x, int ne)
{
   return QData<Config::dim,
                Config::is_tensor,
                Config::Quads,
                DimComp,
                ReadDTensor>(const_cast<double*>(x), config, ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_QDATA
