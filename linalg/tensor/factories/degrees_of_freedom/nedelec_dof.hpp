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

#ifndef MFEM_TENSOR_DOF_ND
#define MFEM_TENSOR_DOF_ND

#include "dof_impl.hpp"

namespace mfem
{

/// A structure to describe Nedelec degrees of freedom in one element.
template <typename... DofTensors>
struct NedelecElementDofs;

template <typename DofTensorX, typename DofTensorY, typename DofTensorZ>
struct NedelecElementDofs<DofTensorX,DofTensorY,DofTensorZ>
{
   DofTensorX x;
   DofTensorY y;
   DofTensorZ z;

   template <typename DofTensorXRHS,
             typename DofTensorYRHS,
             typename DofTensorZRHS>
   NedelecElementDofs<DofTensorX,DofTensorY,DofTensorZ>& MFEM_HOST_DEVICE
   operator=(const NedelecElementDofs<DofTensorXRHS,DofTensorYRHS,DofTensorZRHS>& rhs)
   {
      x = rhs.x;
      y = rhs.y;
      z = rhs.z;
      return (*this);
   }

   template <typename DofTensorXRHS,
             typename DofTensorYRHS,
             typename DofTensorZRHS>
   NedelecElementDofs<DofTensorX,DofTensorY,DofTensorZ>& MFEM_HOST_DEVICE
   operator+=(const NedelecElementDofs<DofTensorXRHS,DofTensorYRHS,DofTensorZRHS>& rhs)
   {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return (*this);
   }
};

template <typename DofTensorX, typename DofTensorY>
struct NedelecElementDofs<DofTensorX,DofTensorY>
{
   DofTensorX x;
   DofTensorY y;

   template <typename DofTensorXRHS,
             typename DofTensorYRHS>
   NedelecElementDofs<DofTensorX,DofTensorY>& MFEM_HOST_DEVICE
   operator=(const NedelecElementDofs<DofTensorXRHS,DofTensorYRHS>& rhs)
   {
      x = rhs.x;
      y = rhs.y;
      return (*this);
   }

   template <typename DofTensorXRHS,
             typename DofTensorYRHS>
   NedelecElementDofs<DofTensorX,DofTensorY>& MFEM_HOST_DEVICE
   operator+=(const NedelecElementDofs<DofTensorXRHS,DofTensorYRHS>& rhs)
   {
      x += rhs.x;
      y += rhs.y;
      return (*this);
   }
};

template <typename DofTensorX>
struct NedelecElementDofs<DofTensorX>
{
   DofTensorX x;

   template <typename DofTensorXRHS>
   NedelecElementDofs<DofTensorX>& MFEM_HOST_DEVICE
   operator=(const NedelecElementDofs<DofTensorXRHS>& rhs)
   {
      x = rhs.x;
      return (*this);
   }

   template <typename DofTensorXRHS>
   NedelecElementDofs<DofTensorX>& MFEM_HOST_DEVICE
   operator+=(const NedelecElementDofs<DofTensorXRHS>& rhs)
   {
      x += rhs.x;
      return (*this);
   }
};

/// A class to encapsulate Nedelec degrees of freedom in with Tensors.
template <int Dim>
class NedelecDegreesOfFreedom;

// 3D
template <>
class NedelecDegreesOfFreedom<3>
{
private:
   static constexpr int Dim = 3;
   using T = double;
   using XLayout = DynamicLayout<Dim>;
   using YLayout = DynamicLayout<Dim>;
   using ZLayout = DynamicLayout<Dim>;
   T* x;
   int dofs_open;
   int dofs_close;
   int ne;

   template <typename Container> MFEM_HOST_DEVICE
   auto build(int e) const
   {
      MFEM_ASSERT_KERNEL(
         e<ne,
         "Element index (%d) is superior to the number of elements (%d). ",
         e, ne);
      constexpr int dim = Dim;
      const int comp_size = dofs_close * dofs_open * dofs_open;
      const int elem_size = dim * comp_size;
      Container X_dofs(x + e*elem_size );
      Container Y_dofs(x + e*elem_size + comp_size);
      Container Z_dofs(x + e*elem_size + 2*comp_size);
      XLayout X_layout(dofs_close, dofs_open, dofs_open);
      YLayout Y_layout(dofs_open, dofs_close, dofs_open);
      ZLayout Z_layout(dofs_open, dofs_open, dofs_close);
      using DofX = Tensor<Container, XLayout>;
      using DofY = Tensor<Container, YLayout>;
      using DofZ = Tensor<Container, ZLayout>;
      NedelecElementDofs<DofX,DofY,DofZ> res = { DofX(X_dofs,X_layout),
                                                 DofY(Y_dofs,Y_layout),
                                                 DofZ(Z_dofs,Z_layout) };
      return res;
   }

public:
   NedelecDegreesOfFreedom(T *x, int dofs_open, int dofs_close, int ne)
   : x(x), dofs_open(dofs_open), dofs_close(dofs_close), ne(ne)
   { }

   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      using Container = ReadContainer<T>;
      return build<Container>(e);
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      using Container = DeviceContainer<T>;
      return build<Container>(e);
   }
};

/// 2D
template <>
class NedelecDegreesOfFreedom<2>
{
private:
   static constexpr int Dim = 2;
   using T = double;
   using XLayout = DynamicLayout<Dim>;
   using YLayout = DynamicLayout<Dim>;
   T* x;
   int dofs_open;
   int dofs_close;
   int ne;

   template <typename Container> MFEM_HOST_DEVICE
   auto build(int e) const
   {
      MFEM_ASSERT_KERNEL(
         e<ne,
         "Element index (%d) is superior to the number of elements (%d). ",
         e, ne);
      constexpr int dim = Dim;
      const int comp_size = dofs_close * dofs_open;
      const int elem_size = dim * comp_size;
      Container X_dofs(x + e*elem_size );
      Container Y_dofs(x + e*elem_size + comp_size);
      XLayout X_layout(dofs_close, dofs_open, dofs_open);
      YLayout Y_layout(dofs_open, dofs_close, dofs_open);
      using DofX = Tensor<Container, XLayout>;
      using DofY = Tensor<Container, YLayout>;
      NedelecElementDofs<DofX,DofY> res = { DofX(X_dofs,X_layout),
                                            DofY(Y_dofs,Y_layout) };
      return res;
   }

public:
   NedelecDegreesOfFreedom(T *x, int dofs_open, int dofs_close, int ne)
   : x(x), dofs_open(dofs_open), dofs_close(dofs_close), ne(ne)
   { }

   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      using Container = ReadContainer<T>;
      return build<Container>(e);
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      using Container = DeviceContainer<T>;
      return build<Container>(e);
   }
};

/// 1D
template <>
class NedelecDegreesOfFreedom<1>
{
private:
   static constexpr int Dim = 1;
   using T = double;
   using XLayout = DynamicLayout<Dim>;
   T* x;
   int dofs_open;
   int dofs_close;
   int ne;

   template <typename Container> MFEM_HOST_DEVICE
   auto build(int e) const
   {
      MFEM_ASSERT_KERNEL(
         e<ne,
         "Element index (%d) is superior to the number of elements (%d). ",
         e, ne);
      constexpr int dim = Dim;
      const int comp_size = dofs_close;
      const int elem_size = dim * comp_size;
      Container X_dofs(x + e*elem_size );
      XLayout X_layout(dofs_close, dofs_open, dofs_open);
      using DofX = Tensor<Container, XLayout>;
      NedelecElementDofs<DofX> res = { DofX(X_dofs,X_layout) };
      return res;
   }

public:
   NedelecDegreesOfFreedom(T *x, int dofs_open, int dofs_close, int ne)
   : x(x), dofs_open(dofs_open), dofs_close(dofs_close), ne(ne)
   { }

   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      using Container = ReadContainer<T>;
      return build<Container>(e);
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      using Container = DeviceContainer<T>;
      return build<Container>(e);
   }
};

template <typename Config>
auto MakeNedelecDoFs(Config &config, int dofs_open, int dofs_close,
                     double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   return NedelecDegreesOfFreedom<Dim>(x,dofs_open,dofs_close,ne);
};

template <typename Config>
auto MakeNedelecDoFs(Config &config, int dofs_open, int dofs_close,
                     const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   // FIXME remove const_cast
   return NedelecDegreesOfFreedom<Dim>(const_cast<double*>(x),dofs_open,dofs_close,ne);
};

// is_nedelec_dof
template <typename Dofs>
struct is_nedelec_dof_v
{
   static constexpr bool value = false;
};

template <int Dim>
struct is_nedelec_dof_v<NedelecDegreesOfFreedom<Dim>>
{
   static constexpr bool value = true;
};

template <typename... DofTensors>
struct is_nedelec_dof_v<NedelecElementDofs<DofTensors...>>
{
   static constexpr bool value = true;
};

template <typename Dofs>
constexpr bool is_nedelec_dof = is_nedelec_dof_v<Dofs>::value;

} // mfem namespace

#endif // MFEM_TENSOR_DOF_ND
