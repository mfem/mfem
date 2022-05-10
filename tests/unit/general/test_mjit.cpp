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

#ifndef MFEM_JIT_COMPILATION
#include "mfem.hpp"
#endif

#if defined(MFEM_USE_JIT) || defined(MFEM_JIT_COMPILATION)

#include <cmath> // pow
#include <cstddef> // size_t
#include <cassert>

#include "general/jit/jit.hpp"// for MFEM_JIT
#include "general/forall.hpp" // for MFEM_FORALL

using namespace mfem;

#ifndef MFEM_JIT_COMPILATION // exclude this catch code from JIT compilation

#include "unit_tests.hpp"

TEST_CASE("Just-In-Time-Compilation", "[JIT]")
{
   auto System = [](const char *cmd) { REQUIRE(std::system(cmd) == 0); };

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

      //#warning cleanup caches, but forces each unit_tests to rebuild it
      //std::remove("libmjit.a"); std::remove("libmjit.so");

      System(MFEM_JIT_CXX " " MFEM_JIT_BUILD_FLAGS // compilation
             " -DMFEM_JIT_COMPILATION -o test_mjit test_mjit.cc"
             " -I../.. -I" MFEM_INSTALL_DIR "/include/mfem"
             " -L../.. -L" MFEM_INSTALL_DIR "/lib -lmfem -ldl");
      std::remove("test_mjit.cc");

      System("./test_mjit"); // will rebuild the libmjit libraries

      System("./test_mjit"); // should reuse the libraries

      std::remove("libmjit.so");
      System("./test_mjit"); // will rebuild only the shared library

      std::remove("test_mjit");
   }
}
#endif // MFEM_JIT_COMPILATION

// Bailey–Borwein–Plouffe formula to compute the nth base-16 digit of π
MFEM_JIT template<int T_DEPTH = 0, int T_N = 0>
void bbps( const size_t q, double *result, int depth = 0, int n = 0)
{
   const size_t D = T_DEPTH ? T_DEPTH : (size_t) depth;
   const size_t N = T_N ? T_N : (size_t) n;

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

MFEM_JIT
template<int T_D1D = 0, int T_Q1D = 0>
void SmemPADiffusionApply3D(const int NE,
                            const double *b_,
                            const double *g_,
                            const double *x_,
                            int d1d = 0,
                            int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int M1Q = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int M1D = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= M1D, "");
   MFEM_VERIFY(Q1D <= M1Q, "");
   auto b = Reshape(b_, Q1D, D1D);
   auto g = Reshape(g_, Q1D, D1D);
   auto x = Reshape(x_, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               B[qx][dy] = b(qx,dy);
               G[qx][dy] = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0, v = 0.0;
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double coords = X[dz][dy][dx];
                  u += coords * B[qx][dx];
                  v += coords * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
   // after forall
   assert(NE > 0);
}

MFEM_JIT
template<int T_A = 0, int T_B = 0>
void ToUseOrNotToUse(int *ab, int a = 0, int b = 0)
{
   MFEM_CONTRACT_VAR(a);
   MFEM_CONTRACT_VAR(b);
   //*ab = a + b; // ERROR: a & b won't be used when JITed and instantiated
   *ab = T_A + T_B; // T_A, T_B will always be set
}

#ifdef MFEM_JIT_COMPILATION
int main(int argc, char* argv[])
{
   int ab = 0, tab = 0;
   ToUseOrNotToUse(&ab,1,2);
   ToUseOrNotToUse<1,2>(&tab);
   if (ab != tab) { return EXIT_FAILURE; }

   double a = 0.0;
   bbps<64,17>(1,&a);
   const size_t ax = (size_t)(pow(16,8)*a);

   double b = 0.0;
   bbps(1,&b,64,17);
   const size_t bx = (size_t)(pow(16,8)*b);

   //printf("\033[33m[0x%lx:0x%lx]\033[m",ax,bx);
   if (ax != bx) { return EXIT_FAILURE; }

   if (pi(10) != 0x5A308D31ul) { return EXIT_FAILURE; }
   return EXIT_SUCCESS;
}
#endif // MFEM_JIT_COMPILATION

#endif // MFEM_USE_JIT
