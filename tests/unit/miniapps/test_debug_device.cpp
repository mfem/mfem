// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

TEST_CASE("Alias a per-field view out of its owner, never out of a raw wrap",
          "[DebugDevice]")
{
   // **This case previously asserted the opposite and the assertion was
   // wrong.** It was committed as "A non-owning wrap must not de-register its
   // base", on the reading -- meq's and then mine -- that Wrap() takes
   // GetHostMemoryType() regardless of ownership, so Memory<T>::Delete()
   // computes std_delete == false and forwards to MemoryManager::Delete_.
   // It does forward, and Delete_ opens
   //
   //    if (!mm.exists || !registered) { return; }
   //
   // so h_mt decides nothing and an unregistered wrap is inert. That much was
   // right. The conclusion drawn from it -- that the reported eviction cannot
   // happen here -- was not, and the case as written could not tell, because
   // it constructed a view and destroyed it untouched. That is the one shape
   // that is safe.
   //
   // What registers the view is Memory<T>::MakeAlias(), which registers an
   // unregistered BASE whenever the device memory type is a device type, and
   // Memory<T>::{Read,Write,ReadWrite}, which do the same at any non-HOST
   // MemoryClass. Register_ sets Registered|OWNS_INTERNAL on the view;
   // MemoryManager::Insert() emplaces, so an address already registered keeps
   // its existing entry silently; and the view's destructor then takes
   // Delete_'s Known branch and erases the pointer. A view that starts at the
   // owner's base pointer therefore erases the OWNER's entry, and the owner's
   // next device-registered write aborts in Write_ with "host pointer is not
   // registered".
   //
   // Five arms, measured, each in its own process, Device("debug") against
   // Device("cpu"):
   //
   //    wrap, host-only touch                       survives   survives
   //    wrap, device-class touch                    ABORTS     survives
   //    wrap, MakeRef taken out of it               ABORTS     survives
   //    wrap at an interior offset, + MakeRef       survives   survives
   //    wrap claiming MemoryType::HOST, + MakeRef   ABORTS     survives
   //
   // The interior-offset arm is why the reported failure was a single abort
   // rather than chaos, and the last arm is why "make Wrap() claim HOST" is
   // not the fix.
   //
   // The routine this came from is DarcyHybridization::ReconstructTotalFlux()
   // in its neq > 1 form, which is not on this branch; the pin that fails
   // without the fix lives beside it. What is pinned here is the idiom.
   const int n = 32, nsub = 8;

   Vector base(n);
   base = 1.0;
   const real_t *b_ptr = base.GetData();
   REQUIRE(mm.IsKnown(b_ptr));

   // The idiom to use: alias out of the OWNER, at whatever offset. A registered
   // alias is also what syncs, where a raw GetData() is a host read of a
   // buffer whose live copy may be on the device.
   {
      Vector view;
      view.MakeRef(base, 0, nsub);
      REQUIRE(view.Size() == nsub);
      view = 4.0;
   }
   REQUIRE(mm.IsKnown(b_ptr));

   {
      Vector view;
      view.MakeRef(base, nsub, nsub);
      REQUIRE(view.Size() == nsub);
      view = 5.0;
   }
   REQUIRE(mm.IsKnown(b_ptr));

   // Reached only if the entry survived both: base is registered, so this is
   // the call that aborts once an entry has been erased.
   base = 2.0;
   REQUIRE(base(0) == MFEM_Approx(2.0));
   REQUIRE(base(n-1) == MFEM_Approx(2.0));

   // A raw wrap touched on the host alone stays unregistered and so stays
   // inert. Kept because it is the arm that made the withdrawn claim look
   // verified, and because it is what licenses the two DenseMatrix reshapes
   // that remain in that routine.
   {
      Vector wrap(base.GetData(), nsub);
      wrap = 3.0;
   }
   REQUIRE(mm.IsKnown(b_ptr));
   base = 6.0;
   REQUIRE(base(0) == MFEM_Approx(6.0));
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
