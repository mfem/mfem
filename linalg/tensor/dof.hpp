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
/// A class to encapsulate degrees of freedom in a Tensor.
template <int Dim, int DimComp = 0, bool IsTensor = false, int Dofs = 0>
class DegreesOfFreedom;

/// Tensor DoFs with size known
template <int Dim, int DimComp, int Dofs>
class DegreesOfFreedom<Dim,DimComp,true,Dofs> : public Tensor<Dim+DimComp+1> // TODO
{
private:
   static constexpr int dim = Dim;
   static constexpr int dim_comp = DimComp;
   // Declare a BlockDTensor type with Dim dimensions of size Dofs,
   // and DimComp dimensions of size Dim.
   using OutTensor = typename rerepeat<Dofs,Dim,Dim,DimComp,BlockDTensor>::type;

public:
   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      OutTensor u_e;
      u_e = this->template Get<Dim+DimComp>(e);
      return u_e;
   }
};

/// Tensor DoFs with size unknown
template <int Dim, int DimComp>
class DegreesOfFreedom<Dim,DimComp,true,0> : public Tensor<Dim+DimComp+1>
{
private:
   static constexpr int dim = Dim;
   using OutTensor = DynamicBlockDTensor<Dim+DimComp>; // Change type according to device

public:
   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      auto u_e = InitOutTensor(this->template Size<0>(),dim);
      u_e = this->template Get<Dim+DimComp>(e);
      return u_e;
   }

private:
   OutTensor InitOutTensor(int dofs, int dim)
   {
      return InitTensor<0,0>::make(dofs,dim);
   }

   /// A structure to linearize the input sizes to create a Tensor.
   template <int NDim, int NComp>
   struct InitTensor
   {
      template <typename... Sizes>
      OutTensor make(int D, int dim, Sizes... sizes)
      {
         return InitTensor<NDim,NComp+1>::make(D,dim,dim,sizes...);
      }
   };

   template <int NDim>
   struct InitTensor<NDim,DimComp>
   {
      template <typename... Sizes>
      OutTensor make(int D, int dim, Sizes... sizes)
      {
         return InitTensor<NDim+1,DimComp>(D,dim,D,sizes...);
      }
   };

   template <>
   struct InitTensor<Dim,DimComp>
   {
      template <typename... Sizes>
      OutTensor make(int D, int dim, Sizes... sizes)
      {
         return OutTensor(sizes...);
      }
   };
};

/// Non-Tensor DoFs with size known
template <int Dim, int DimComp, int Dofs>
class DegreesOfFreedom<Dim,DimComp,false,Dofs> : public Tensor<DimComp+1>
{
private:
   static constexpr int dim = Dim;
   static constexpr int dcomp = DimComp;
   using OutTensor = typename rerepeat<Dofs,1,Dim,DimComp,StaticSharedDTensor>::type;

public:
   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      OutTensor u_e;
      u_e = this->template Get<Dim+DimComp>(e);
      return u_e;
   }
};

/// Non-Tensor DoFs with size unknown
template <int Dim, int DimComp>
class DegreesOfFreedom<Dim,DimComp,false,0> : public Tensor<1+DimComp+1>
{
private:
   static constexpr int dim = Dim;
   using OutTensor = SharedDTensor<1+DimComp>;

public:
   /// Returns a Tensor corresponding to the DoFs of element e
   auto operator()(int e) const
   {
      OutTensor u_e;
      u_e = this->template Get<DimComp>(e);
      return u_e;
   }
};

template <int Dim, int DimComp, bool IsTensor, int Dofs>
class MutDegreesOfFreedom : DegreesOfFreedom<Dim,DimComp,IsTensor,Dofs>
{
public:
   auto operator()(int e)
   {
      return this->template Get<Dim+DimComp>(e);
   }
};

/// Functor to represent degrees of freedom
template <int DimComp, int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
auto MakeDoFs(KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize> &config,
              double *x,
              int ne)
{
   return MutDegreesOfFreedom<Dim,DimComp,IsTensor,Dofs>(x, config.dofs, ne);
}

template <int DimComp, int Dim, bool IsTensor, int Dofs, int Quads, int BatchSize>
auto MakeDoFs(KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize> &config,
              const double *x,
              int ne)
{
   return DegreesOfFreedom<Dim,DimComp,IsTensor,Dofs>(const_cast<double*>(x), config.dofs, ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_DOF
