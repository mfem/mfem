// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

#include "tensor.hpp"

namespace mfem
{
namespace internal
{

template <typename value_type, typename gradient_type, int n>
MFEM_HOST_DEVICE auto get_value(const
                                tensor<dual<value_type, gradient_type>, n>& arg)
{
   tensor<double, n> output{};
   for (int i = 0; i < n; i++)
   {
      output[i] = arg[i].value;
   }
   return output;
}

template <typename value_type, typename gradient_type, int m, int n>
MFEM_HOST_DEVICE auto get_value(const
                                tensor<dual<value_type, gradient_type>, m, n>& arg)
{
   tensor<double, m, n> output{};
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         output[i][j] = arg[i][j].value;
      }
   }
   return output;
}

}
}