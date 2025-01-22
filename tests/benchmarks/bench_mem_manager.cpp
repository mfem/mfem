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

constexpr int stack_size = 3;
int size = 3;
double val = 0_r;

inline void vec_op(Vector &v)
{
   // v = val;
   // std::fill(v.begin(), v.end(), 0_r); // can be slower for small 'size'!
   std::fill(v.begin(), v.end(), val);
}

void mem_stack()
{
   real_t v_data[stack_size];
   Vector v(v_data, size);

   vec_op(v);
}

void mem_host()
{
   // Vector v(size, MemoryType::HOST); // everything may be optimized out!
   Vector v(size);

   vec_op(v);
}

void mem_host_64()
{
   Vector v(size, MemoryType::HOST_64);

   vec_op(v);
}

void mem_host_arena()
{
   Vector v(size, MemoryType::HOST_ARENA);

   vec_op(v);
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
   bench(mem_host_64,    "mem_host_64");
   bench(mem_stack,      "mem_stack");
   bench(mem_host,       "mem_host");
   bench(mem_host_arena, "mem_host_arena");

   return 0;
}
