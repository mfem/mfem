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

#ifndef MFEM_TENSOR_READWRITE
#define MFEM_TENSOR_READWRITE

#include "../tensor.hpp"

namespace mfem
{

/// Generate a Tensor that be read on device
template <typename C, typename L>
auto Read(const Tensor<C,L> &t)
{
   return Tensor<ReadContainer<T>,Layout>(t.ReadData(),t);
}

/// Generate a Tensor that be writen on device (read is unsafe)
template <typename C, typename L>
auto Write(Tensor<C,L> &t)
{
   return Tensor<DeviceContainer<T>,Layout>(t.WriteData(),t);
}

/// Generate a Tensor that be read and writen on device
template <typename C, typename L>
auto ReadWrite(Tensor<C,L> &t)
{
   return Tensor<DeviceContainer<T>,Layout>(t.ReadWriteData(),t);
}

} // mfem namespace

#endif // MFEM_TENSOR_READWRITE
