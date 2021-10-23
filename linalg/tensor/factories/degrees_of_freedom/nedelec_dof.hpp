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

/// A class to encapsulate Nedelec degrees of freedom in with Tensors.
template <typename... DofTensors>
class NedelecDegreesOfFreedom;

template <typename DofTensorX, typename DofTensorY, typename DofTensorZ>
class NedelecDegreesOfFreedom<DofTensorX,DofTensorY,DofTensorZ>
{
private:
   mutable int element;
   DofTensorX x_dofs;
   DofTensorY y_dofs;
   DofTensorZ z_dofs;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   NedelecDegreesOfFreedom(double *x, double *y, double *z,
                           int sizeX0, int sizeX1, int sizeX2,
                           int sizeY0, int sizeY1, int sizeY2,
                           int sizeZ0, int sizeZ1, int sizeZ2)
   : DofTensorX(x, sizeX0, sizeX1, sizeX2),
     DofTensorY(y, sizeY0, sizeY1, sizeY2),
     DofTensorZ(z, sizeZ0, sizeZ1, sizeZ2)
   {
      // TODO static asserts Config values
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      element = e;
      return (*this);
   }

   /// Returns a Tensor corresponding to the X DoFs of element e
   MFEM_HOST_DEVICE inline
   auto getX() const
   {
      constexpr int Rank = get_tensor_rank<DofTensorX>;
      return Get<Rank-1>(element, x_dofs); // TODO batchsize so +tidz or something?
   }

   MFEM_HOST_DEVICE inline
   auto getX()
   {
      constexpr int Rank = get_tensor_rank<DofTensorX>;
      return Get<Rank-1>(element, x_dofs);
   }

   /// Returns a Tensor corresponding to the Y DoFs of element e
   MFEM_HOST_DEVICE inline
   auto getY() const
   {
      constexpr int Rank = get_tensor_rank<DofTensorY>;
      return Get<Rank-1>(element, y_dofs); // TODO batchsize so +tidz or something?
   }

   MFEM_HOST_DEVICE inline
   auto getY()
   {
      constexpr int Rank = get_tensor_rank<DofTensorY>;
      return Get<Rank-1>(element, y_dofs);
   }

   /// Returns a Tensor corresponding to the Z DoFs of element e
   MFEM_HOST_DEVICE inline
   auto getZ() const
   {
      constexpr int Rank = get_tensor_rank<DofTensorZ>;
      return Get<Rank-1>(element, z_dofs); // TODO batchsize so +tidz or something?
   }

   MFEM_HOST_DEVICE inline
   auto getZ()
   {
      constexpr int Rank = get_tensor_rank<DofTensorZ>;
      return Get<Rank-1>(element, z_dofs);
   }
};

// is_nedelec_dof
template <typename Dofs>
struct is_nedelec_dof_v
{
   static constexpr bool value = false;
};

template <typename... DofTensors>
struct is_nedelec_dof_v<NedelecDegreesOfFreedom<DofTensors...>>
{
   static constexpr bool value = true;
};

template <typename Dofs>
constexpr bool is_nedelec_dof = is_nedelec_dof_v<Dofs>::value;

} // mfem namespace

#endif // MFEM_TENSOR_DOF_ND
