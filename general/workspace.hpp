// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_WORKSPACE_HPP
#define MFEM_WORKSPACE_HPP

#include "../linalg/vector.hpp"
#include <forward_list>

namespace mfem
{

class Workspace
{
public:
   static Vector NewVector(int n)
   {
      Vector vec(n, MemoryType::DEVICE_WORKSPACE);
      vec.UseDevice(true);
      return vec;
   }
};

} // namespace mfem

#endif
