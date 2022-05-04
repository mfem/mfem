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

// skip the catch block when trying to JIT compile
#ifndef MFEM_TEST_MJIT_SKIP_CXX

#include "mfem.hpp"

using namespace mfem;

#include "unit_tests.hpp"

#include "general/forall.hpp" // for MFEM_JIT

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

void System(const char *arg) { REQUIRE(std::system(arg) == 0); }

TEST_CASE("Just-In-Time-Compilation", "[JIT]")
{
   SECTION("System check")
   {
      CAPTURE("hostname");
      System("hostname");
   }

   SECTION("Compiler check")
   {
      CAPTURE(MFEM_JIT_CXX);
      System(MFEM_JIT_CXX " --version");
   }

   SECTION("mjit")
   {
      CAPTURE("mjit");
      System(MFEM_SOURCE_DIR "/mjit -h"); // help
      System(MFEM_SOURCE_DIR "/mjit -o test_mjit.cc " // transform
             MFEM_SOURCE_DIR "/tests/unit/general/test_mjit.cpp");

      System(MFEM_JIT_CXX " " MFEM_JIT_BUILD_FLAGS // compile
             " -DMFEM_TEST_MJIT_SKIP_CXX"
             " -DMFEM_TEST_MJIT_MAIN_CXX" // embed the main below
             " -I" MFEM_SOURCE_DIR " -o test_mjit test_mjit.cc"
             " -L" MFEM_SOURCE_DIR " -lmfem -dl"); // for jit.hpp functions

      //std::remove("./libmjit.a");
      //std::remove("./libmjit.so");

      System("./test_mjit"); // test our product

      std::remove("test_mjit.cc");
      std::remove("test_mjit.o");
   }
}

#endif // MFEM_TEST_MJIT_SKIP_CXX

#define MFEM_JIT_COMPILATION
#include "general/forall.hpp"
using namespace mfem;

MFEM_JIT template<int T_Q> void jit_1(int s, int q = 0)
{
   MFEM_CONTRACT_VAR(s);
   MFEM_CONTRACT_VAR(q);
}

MFEM_JIT template<int T_Q> void jit_2(int q = 0)
{
   /*1*/
   {
      // 2
      {
         /*3*/
         MFEM_CONTRACT_VAR(q);
      } // 3
   } /*2*/
}

MFEM_JIT template<int T_Q> void jit_3(int q = 0)
{
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
               MFEM_FORALL_3D(i, N, 4,4,4, i++;);
               //MFEM_FORALL_3D(i, N, 4,4,4, i++;); not supported
            } /*!4*/
         } /*!3*/
      } /*!2*/
   } /*!1*/
}

MFEM_JIT template<int T_R> void jit_4(/**/ int //
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

MFEM_JIT template<int T_Q> void jit_5
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

MFEM_JIT template<int T_Q = 0> void jit_main(int *n, int q = 0)
{
   *n = 40 + q;
}

MFEM_JIT template<int T_DEPTH = 0, int T_N = 0>
void bbps( const size_t q, double *result, size_t depth = 0, size_t n = 0)
{
   const size_t b = 16;
   const size_t p = 8;
   auto myfmod = [](double x, double y) { return x-((size_t)(x/y))*y;};

   const size_t N = n + depth;
   double s = 0.0;
   double h = 1.0/b;
   for (size_t i=0, k=q; i<=n; i++, k+=p)
   {
      double m = 1.0;
      double a = b;
      assert(n>=i);
      for (size_t r=n-i; r>0; r>>=1)
      {
         m = (r&1)?myfmod(m*a,k):m;
         a = myfmod(a*a,k);
      }
      s+=m/k;
      s-=(size_t)s;
   }
   for (size_t i=n+1; i<N; i++, h/=b) { s+=h/(p*i+q); }
   *result = s;
}

size_t pi(size_t n, const size_t D = 100)
{
   double a,b,c,d;
   bbps(1, &a, D, n-1);
   bbps(4, &b, D, n-1);
   bbps(5, &c, D, n-1);
   bbps(6, &d, D, n-1);
   const double pi = fmod(4.0*a-2.0*b-c-d,1.0);
   const size_t pi_hex = (size_t)(pow(16,8)*pi);
   //printf("\033[33m{%ld, 0x%08lX}\033[m\n", n, pi_hex);
   return pi_hex;
}

#ifdef MFEM_TEST_MJIT_MAIN_CXX
int main(int argc, char* argv[])
{
   int n = 0, q = 2;
   jit_main(&n, q);
   if (n != 42) { return EXIT_FAILURE; }
   if (pi(10) != 0x5A308D31ul) { return EXIT_FAILURE; }
   return EXIT_SUCCESS;
}
#endif // MFEM_TEST_MJIT_MAIN_CXX

#endif // MFEM_USE_JIT
