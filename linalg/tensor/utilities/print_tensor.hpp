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

#ifndef MFEM_TENSOR_PRINT
#define MFEM_TENSOR_PRINT

#include "../tensor.hpp"
#include <iostream>

namespace mfem
{

/// Function to print tensors
template <typename C, typename L>
std::ostream& operator<<(std::ostream &os, const Tensor<C,L> &t)
{
   ForallDims<Tensor>::Apply(t,[&](auto... idx)
   {
      os << "value(" << idx... << ")= " << t(idx...) << ", ";
   });
   os << std::endl;
   return os;
}

} // mfem namespace

#endif // MFEM_TENSOR_PRINT
