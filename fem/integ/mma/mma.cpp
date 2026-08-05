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

#include "mma.hpp"

#include "../../../general/globals.hpp"

#include <cstring>

namespace mfem
{

namespace
{
bool force_mma = false;
}

bool ForceMMA(bool enable)
{
   const bool previous = force_mma;
   force_mma = enable;
   return previous;
}

bool GetForceMMA()
{
   if (force_mma) { return true; }

   // Cached env lookup: MFEM_USE_MMA set (and not "0") forces MMA
   static int env_mma = -1; // -1 unset, 0 no, 1 yes
   if (env_mma < 0)
   {
      const char *e = GetEnv("MFEM_USE_MMA");
      env_mma = (e && std::strcmp(e, "0") != 0) ? 1 : 0;
   }
   return env_mma == 1;
}

namespace internal
{

void PADetJSetupSimplexFromNodes(const int dim,
                                 const int NE,
                                 const int NQ,
                                 const int ND,
                                 const bool by_val,
                                 const Array<real_t> &w,
                                 const Array<real_t> &g,
                                 const Vector &nodes_e,
                                 const Vector &c,
                                 Vector &d)
{
   const bool const_c = c.Size() == 1;
   const auto W = Reshape(w.Read(), NQ);
   // DofToQuad::FULL G layout: (nq x dim x ndof), matches QI Eval*.
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), 1, 1)
                  : Reshape(c.Read(), NQ, NE);
   auto D = Reshape(d.Write(), NQ, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ;
         const int q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t detJ = J11 * J22 - J21 * J12;
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PADetJSetupSimplexFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ;
      const int q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND, J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      const real_t coeff = const_c ? C(0, 0) : C(q, e);
      D(q, e) = W(q) * coeff * (by_val ? detJ : real_t(1) / detJ);
   });
}

} // namespace internal

} // namespace mfem
