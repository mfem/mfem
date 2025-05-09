// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

// The Memory Manager checker singleton class
class MemoryManagerCheck
{
   MemoryManagerCheck(const char *argv0);
   ~MemoryManagerCheck();

   static MemoryManagerCheck& Singleton(const char *arg0)
   {
      static MemoryManagerCheck instance(arg0);
      return instance;
   }

public:
   static void Init(const char *arg0) { Singleton(arg0); }

   MemoryManagerCheck(const MemoryManagerCheck&) = delete;
   MemoryManagerCheck& operator=(const MemoryManagerCheck&) = delete;
};

} // namespace mfem