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

#include "mfem.hpp"

#ifdef MFEM_USE_JIT

#include <cmath> // pow
#include <cstddef> // size_t

#include "catch.hpp"

#include "general/jit/jit.hpp"// for MFEM_JIT
#include "general/forall.hpp" // for MFEM_FORALL

using namespace mfem;

namespace mjit_tests
{

MFEM_JIT template<int T_Q> void parser1(int s, int q = 0)
{
   MFEM_CONTRACT_VAR(s);
   MFEM_CONTRACT_VAR(q);
}

MFEM_JIT template<int T_Q> void parser2(int q = 0)
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

MFEM_JIT template<int T_Q> void parser3(int q = 0)
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

MFEM_JIT template<int T_R> void parser4(/**/ int //
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

MFEM_JIT template<int T_Q> void parser5
(int/**/ //
 //
 /**/q/**/ = 0) // single line error?
{
   MFEM_CONTRACT_VAR(q);
}

MFEM_JIT/**/template/**/</**/int/**/T_Q/**/>//
/**/void/**/parser6/**/(/**/int/**/q/**/=/**/0/**/)/**/
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
      MFEM_CONTRACT_VAR(e);
      MFEM_CONTRACT_VAR(b);
      MFEM_CONTRACT_VAR(g);
      MFEM_CONTRACT_VAR(x);
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

TEST_CASE("Parser", "[JIT]")
{
   SECTION("ToUseOrNotToUse")
   {
      int ab = 0, tab = 0;
      ToUseOrNotToUse(&ab,1,2);
      ToUseOrNotToUse<1,2>(&tab);
      REQUIRE(ab == tab);
   }
}

} // mjit_tests

#endif // MFEM_USE_JIT
