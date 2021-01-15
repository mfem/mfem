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

#include "mfem.hpp"
#include "unit_tests.hpp"

#ifndef _WIN32
#include <unistd.h>

using namespace mfem;

struct NullBuf: public std::streambuf { int overflow(int c) { return c; }};

static void TestMemoryTypes(MemoryType mt, bool use_dev, int N = 1024)
{
   Memory<double> mem(N, mt);
   REQUIRE(mem.Capacity() == N);
   Vector y;
   y.NewMemoryAndSize(mem, N, true);
   y.UseDevice(use_dev);
   y = 0.0;
   y.HostWrite();
   y[0] = -1.0;
   y.Write();
   y = 1.0;
   y.HostReadWrite();
   y[0] = 0.0;
   REQUIRE(y*y == MFEM_Approx(N-1));
   y.Destroy();
}

static void ScanMemoryTypes()
{
   const MemoryType h_mt = mm.GetHostMemoryType();
   const MemoryType d_mt = mm.GetDeviceMemoryType();
   TestMemoryTypes(h_mt, true);
   TestMemoryTypes(d_mt, true);
   TestMemoryTypes(h_mt, false);
   TestMemoryTypes(d_mt, false);
}

static void MmuCatch(const int N = 1024)
{
   Vector Y(N);
   // double *h_Y = (double*)Y;
   Y.UseDevice(true);
   Y = 0.0;
   // in debug device, should raise a SIGSEGV
   // but it can't be caught by this version of Catch
   // h_Y[0] = 0.0;
}

void Aliases(const int N = 0x1234)
{
   Vector S(2*3*N + N);
   S.UseDevice(true);
   S = -1.0;
   GridFunction X,V,E;
   const int Xsz = 3*N;
   const int Vsz = 3*N;
   const int Esz = N;
   X.NewMemoryAndSize(Memory<double>(S.GetMemory(), 0, Xsz), Xsz, true);
   V.NewMemoryAndSize(Memory<double>(S.GetMemory(), Xsz, Vsz), Vsz, true);
   E.NewMemoryAndSize(Memory<double>(S.GetMemory(), Xsz + Vsz, Esz), Esz, true);
   X = 1.0;
   X.SyncAliasMemory(S);
   S.HostWrite();
   S = -1.0;
   X.Write();
   X = 1.0;
   S.HostRead();
   REQUIRE(S*S == MFEM_Approx(7.0*N));
   V = 2.0;
   V.SyncAliasMemory(S);
   REQUIRE(S*S == MFEM_Approx(16.0*N));
   E = 3.0;
   E.SyncAliasMemory(S);
   REQUIRE(S*S == MFEM_Approx(24.0*N));
}

TEST_CASE("MemoryManager", "[MemoryManager]")
{
   SECTION("Debug")
   {
      NullBuf null_buffer;
      std::ostream dev_null(&null_buffer);
      // If MFEM_MEMORY is set, we start with some non-empty maps
      const int n_ptr = mm.PrintPtrs(dev_null);
      const int n_alias = mm.PrintAliases(dev_null);
      const long pagesize = sysconf(_SC_PAGE_SIZE);
      REQUIRE(pagesize > 0);
      Device device("debug");
      for (int n = 1; n < 2*pagesize; n+=7)
      {
         Aliases(n);
         REQUIRE(mm.PrintPtrs(dev_null) == n_ptr);
         REQUIRE(mm.PrintAliases(dev_null) == n_alias);
      }
      MmuCatch();
      ScanMemoryTypes();
      REQUIRE(mm.PrintPtrs(dev_null) == n_ptr);
      REQUIRE(mm.PrintAliases(dev_null) == n_alias);
   }
}

#endif // _WIN32
