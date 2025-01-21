// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <iostream>
#include <mfem.hpp>

using namespace std;
using namespace mfem;

void mem_stack()
{
   real_t v_data[Geometry::MaxDim];
   Vector v(v_data, 3);

   // ...
}

void mem_host()
{
   // Vector v(3, MemoryType::HOST); // everything is optimized out!
   Vector v(3);

   // ...
}

void mem_host_64()
{
   Vector v(3, MemoryType::HOST_64);

   // ...
}

void mem_host_arena()
{
   Vector v(3, MemoryType::HOST_ARENA);

   // ...
}

void bench(void (*f)(), const char *name)
{
   for (int j = 0; j < 3; j++)
   {
      StopWatch sw;
      sw.Start();
      for (int i = 0; i < 20000000; i++)
      {
         f();
      }
      auto t = sw.RealTime();
      cout << "test " << name << " time: " << 1e3*t << " ms\n";
   }
}

int main()
{
   bench(mem_stack,      "mem_stack");
   bench(mem_host,       "mem_host");
   bench(mem_host_64,    "mem_host_64");
   bench(mem_host_arena, "mem_host_arena");

   return 0;
}
