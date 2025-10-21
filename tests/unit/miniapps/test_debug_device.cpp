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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
using namespace mfem;

#define CATCH_CONFIG_RUNNER
#include "run_unit_tests.hpp"

using namespace mfem;

#ifndef _WIN32 // Debug device specific tests, not supported on Windows
#include <unistd.h>

struct NullBuf: public std::streambuf { int overflow(int c) override { return c; }};

#include <iosfwd>
#include <csetjmp>
#include <csignal>

static void TestMemoryTypes(MemoryType mt, bool use_dev, int N = 1024)
{
   Memory<real_t> mem(N, mt);
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
   const auto h_mt = mm.GetHostMemoryType(), d_mt = mm.GetDeviceMemoryType();
   TestMemoryTypes(h_mt, true), TestMemoryTypes(d_mt, true);
   TestMemoryTypes(h_mt, false), TestMemoryTypes(d_mt, false);
}

static void MmuCatch(const int N = 1024)
{
   Vector Y(N);
   Y.UseDevice(true);
   static real_t *h_Y = Y.GetData(); // store host address
   Y = 0.0; // use Y on the device
   // using h_Y raises an MFEM abort that needs to be caught with a new handler
   static jmp_buf env;
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   static volatile bool caught_illegal_memory_access = false;
   sa.sa_sigaction = [](int, siginfo_t *si, void*)
   {
      REQUIRE(si->si_addr == h_Y);
      caught_illegal_memory_access = true;
      mfem::out << "Illegal memory access caught at " << si->si_addr << std::endl;
      std::longjmp(env, EXIT_FAILURE); // noreturn, setjmp returns EXIT_FAILURE
   };
   // set the new handlers
   REQUIRE(sigaction(SIGBUS, &sa, nullptr) != -1); // macOS
   REQUIRE(sigaction(SIGSEGV, &sa, nullptr) != -1); // Linux

   if (setjmp(env) == EXIT_SUCCESS) // save the execution context to env
   {
      h_Y[0] = 0.0; // raises a SIGBUS, handler, longjmp
      REQUIRE(false); // should not be here
   }
   REQUIRE(caught_illegal_memory_access); // rewinding to env through setjmp
}

static void Aliases(const int N = 0x1234)
{
   Vector S(2*3*N + N);
   S.UseDevice(true);
   S = -1.0;
   GridFunction X,V,E;
   const int Xsz = 3*N, Vsz = 3*N, Esz = N;
   X.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), 0, Xsz), Xsz, true);
   V.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), Xsz, Vsz), Vsz, true);
   E.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), Xsz + Vsz, Esz), Esz, true);
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

TEST_CASE("Array::MakeRef", "[DebugDevice]")
{
   Array<int> x(1), y;
   y.MakeRef(x);
   x.Read();
   REQUIRE_NOTHROW(y.Read());
}

TEST_CASE("MemoryManager/DebugDevice", "[DebugDevice]")
{
   // If MFEM_MEMORY is set, we can start with some non-empty maps,
   // we need to use the number of pointers and aliases there already are
   // present in the maps
   struct NullBuffer: public std::streambuf
   {
      int overflow(int c) override { return c; }
   } null_buffer;
   std::ostream dev_null(&null_buffer);
   const auto n_ptr = mm.PrintPtrs(dev_null);
   const auto n_alias = mm.PrintAliases(dev_null);
   const auto pagesize = sysconf(_SC_PAGE_SIZE);
   REQUIRE(pagesize > 0);

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

#endif // _WIN32

int main(int argc, char *argv[])
{
   Device device("debug");
   return RunCatchSession(argc, argv, {"[DebugDevice]"});
}
