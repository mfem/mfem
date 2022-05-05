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

#include "config/config.hpp"

#ifdef MFEM_USE_JIT

#ifndef MFEM_TEST_MJIT_EXCLUDE_CODE // exclude the catch code JIT compilation

#include "general/forall.hpp" // for MFEM_JIT

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("Just-In-Time-Compilation", "[JIT]")
{
   auto System = [](const char *arg) { REQUIRE(std::system(arg) == 0); };

   SECTION("System check")
   {
      CAPTURE("hostname");
      System("hostname");
   }

   SECTION("MFEM Compiler check")
   {
      CAPTURE(MFEM_JIT_CXX);
      System(MFEM_JIT_CXX " --version");
   }

   SECTION("mjit")
   {
      CAPTURE("mjit executable");
      System(MFEM_SOURCE_DIR "/mjit -h"); // help

      System(MFEM_SOURCE_DIR "/mjit -o test_mjit.cc " // generate source file
             MFEM_SOURCE_DIR "/tests/unit/general/test_mjit.cpp");

      std::remove("test_mjit"); // cleanup existing executable
      std::remove("libmjit.a");
      std::remove("libmjit.so");
      System(MFEM_JIT_CXX " " MFEM_JIT_BUILD_FLAGS // compilation
             " -DMFEM_TEST_MJIT_EXCLUDE_CODE"
             " -DMFEM_TEST_MJIT_INCLUDE_MAIN" // embed the main below
             " -I" MFEM_SOURCE_DIR " -o test_mjit test_mjit.cc"
             " -L" MFEM_SOURCE_DIR " -lmfem"); // for jit.hpp functions
      std::remove("test_mjit.cc");
      System("./test_mjit"); // will rebuild the libmjit libraries
      System("./test_mjit"); // should reuse the libraries
      std::remove("libmjit.so");
      System("./test_mjit"); // will rebuild only the shared library
      std::remove("test_mjit");
   }
}
#else // MFEM_TEST_MJIT_EXCLUDE_CODE
#include <cstddef>
#include <unordered_map>
#define MFEM_DEVICE_HPP
#define MFEM_MEM_MANAGER_HPP  // will pull HYPRE_config.h
#define MFEM_JIT_COMPILATION

struct Backend
{
   enum: unsigned long
   {
      CPU=1<<0, CUDA=1<<2, HIP=1<<3, DEBUG_DEVICE=1<<14
   };
   enum { DEVICE_MASK = 0 };
};
namespace mfem
{
struct Device
{
   static constexpr unsigned long backends = 0;
   static inline bool Allows(unsigned long b_mask)
   { return Device::backends & b_mask; }
};
} // namespace mfem

// avoid mfem_cuda_error inside MFEM_GPU_CHECK
#define MFEM_GPU_CHECK(...)

#include "general/forall.hpp" // for MFEM_JIT

using namespace mfem;
using namespace std;
#endif // MFEM_TEST_MJIT_EXCLUDE_CODE

// Bailey–Borwein–Plouffe formula to compute the nth base-16 digit of π
MFEM_JIT template<int T_DEPTH = 0, int T_N = 0>
void bbps( const size_t q, double *result, size_t depth = 0, size_t n = 0)
{
   const size_t D = T_DEPTH ? T_DEPTH : depth;
   const size_t N = T_N ? T_N : n;

   const size_t b = 16, p = 8, M = N + D;
   double s = 0.0,  h = 1.0 / b;
   for (size_t i = 0, k = q; i <= N; i++, k += p)
   {
      double a = b, m = 1.0;
      for (size_t r = N - i; r > 0; r >>= 1)
      {
         auto dmod = [](double x, double y) { return x-((size_t)(x/y))*y;};
         m = (r&1) ? dmod(m*a,k) : m;
         a = dmod(a*a,k);
      }
      s += m / k;
      s -= (size_t) s;
   }
   for (size_t i = N + 1; i < M; i++, h /= b) { s += h / (p*i+q); }
   *result = s;
}

size_t pi(size_t n, const size_t D = 100)
{
   auto p = [&](int k) { double r; bbps(k, &r, D, n-1); return r;};
   return pow(16,8)*fmod(4.0*p(1) - 2.0*p(4) - p(5) - p(6), 1.0);
}

#ifdef MFEM_TEST_MJIT_INCLUDE_MAIN
int main(int argc, char* argv[])
{
   double a = 0.0, b = 0.0;
   bbps<64,17>(1,&a);
   bbps(1,&b,64,17);
   const size_t ax = (size_t)(pow(16,8)*a);
   const size_t bx = (size_t)(pow(16,8)*b);
   //printf("\033[33m[0x%lx:0x%lx]\033[m",ax,bx);
   if (ax != bx) { return EXIT_FAILURE; }
   if (pi(10) != 0x5A308D31ul) { return EXIT_FAILURE; }
   return EXIT_SUCCESS;
}
#endif // MFEM_TEST_MJIT_INCLUDE_MAIN

MFEM_JIT template<int T_Q> void jit1(int s, int q = 0)
{
   MFEM_CONTRACT_VAR(s);
   MFEM_CONTRACT_VAR(q);
}

MFEM_JIT template<int T_Q> void jit2(int q = 0)
{
   /*1*/
   {
      // 2
      {
         /*3*/
         MFEM_CONTRACT_VAR(q);
         // ~3
      }
      /*~2*/
   }
   // ~1
}

MFEM_JIT template<int T_Q> void jit3(int q = 0)
{
   /*1*/
   MFEM_CONTRACT_VAR(q);
   const int N = 1024;
   {
      /*1*/
      {
         /*2*/
         {
            /*3*/
            {
               /*4*/
               MFEM_FORALL_3D(/**/i/**/,
                                  /**/N/**/,
                                  /**/4/**/,
                                  /**/4/**/,
                                  /**/4/**/,
                                  /**/
               {
                  //
                  /**/{//
                     i++;
                  }/**/
               } /**/);
               //MFEM_FORALL_3D(i, N, 4,4,4, i++;); not supported
            } /*~4*/
         } /*~3*/
      } /*~2*/
   } /*~1*/
}

MFEM_JIT template<int T_R> void jit4(/**/ int //
                                          q/**/, //
                                          //
                                          /**/int /**/ r /**/ = /**/
                                             //
                                             0
                                    )/**/
//
/**/
{
   /**/
   //
   MFEM_CONTRACT_VAR(q);
   MFEM_CONTRACT_VAR(r);
}

MFEM_JIT template<int T_Q> void jit5
(int/**/ //
 //
 /**/q/**/ = 0) // single line error?
{
   MFEM_CONTRACT_VAR(q);
}

MFEM_JIT/**/template/**/</**/int/**/T_Q/**/>//
/**/void/**/jit_6/**/(/**/int/**/q/**/=/**/0/**/)/**/
{/**/MFEM_CONTRACT_VAR/**/(/**/q/**/)/**/;/**/}

MFEM_JIT//
template//
<//
   int//
   T_Q//
   >//
void//
jit_7//
(//
   int//
   q//
   =//
      0//
)//
{
   //
   MFEM_CONTRACT_VAR//
   (//
      q//
   )//
   ;//
}

#endif // MFEM_USE_JIT
