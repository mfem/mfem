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

} // namespace mfem

namespace mfem::internal
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

namespace
{

template <typename CAcc>
MFEM_HOST_DEVICE inline real_t CoeffAt(CAcc C, const bool const_c,
                                       const int i, const int q, const int e)
{
   return const_c ? C(i, 0, 0) : C(i, q, e);
}

} // namespace

void PAInvDetJSetupSimplexFromNodes(const int dim,
                                    const int coeffDim,
                                    const int NE,
                                    const int NQ,
                                    const int ND,
                                    const Array<real_t> &w,
                                    const Array<real_t> &g,
                                    const Vector &nodes_e,
                                    const Vector &c,
                                    Vector &d)
{
   const bool const_c = c.Size() == coeffDim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1)
                  : Reshape(c.Read(), coeffDim, NQ, NE);
   auto D = Reshape(d.Write(), NQ, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ, q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t detJ = J11 * J22 - J21 * J12;
         D(q, e) = W(q) * CoeffAt(C, const_c, 0, q, e) / detJ;
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PAInvDetJSetupSimplexFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ, q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND, J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      D(q, e) = W(q) * CoeffAt(C, const_c, 0, q, e) / detJ;
   });
}

void PAJTQJSetupSimplexFromNodes(const int dim,
                                 const int coeffDim,
                                 const int NE,
                                 const int NQ,
                                 const int ND,
                                 const Array<real_t> &w,
                                 const Array<real_t> &g,
                                 const Vector &nodes_e,
                                 const Vector &c,
                                 Vector &d)
{
   const bool symmetric = (coeffDim != dim * dim);
   const bool const_c = c.Size() == coeffDim;
   const int pa_size = symmetric ? (dim * (dim + 1)) / 2 : dim * dim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1)
                  : Reshape(c.Read(), coeffDim, NQ, NE);
   auto D = Reshape(d.Write(), NQ, pa_size, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ, q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t w_det = W(q) / (J11 * J22 - J21 * J12);
         real_t A00, A01, A10, A11;
         if (coeffDim == 3 || coeffDim == 4)
         {
            const real_t C00 = CoeffAt(C, const_c, 0, q, e);
            const real_t C01 = CoeffAt(C, const_c, 1, q, e);
            const real_t C10 = symmetric ? C01 : CoeffAt(C, const_c, 2, q, e);
            const real_t C11 = symmetric ? CoeffAt(C, const_c, 2, q, e)
                               : CoeffAt(C, const_c, 3, q, e);
            const real_t R00 = C00 * J11 + C01 * J21;
            const real_t R10 = C10 * J11 + C11 * J21;
            const real_t R01 = C00 * J12 + C01 * J22;
            const real_t R11 = C10 * J12 + C11 * J22;
            A00 = w_det * (J11 * R00 + J21 * R10);
            A01 = w_det * (J11 * R01 + J21 * R11);
            A10 = w_det * (J12 * R00 + J22 * R10);
            A11 = w_det * (J12 * R01 + J22 * R11);
         }
         else
         {
            const real_t C0 = CoeffAt(C, const_c, 0, q, e);
            const real_t C1 = coeffDim == 2 ? CoeffAt(C, const_c, 1, q, e) : C0;
            A00 = w_det * (J11 * C0 * J11 + J21 * C1 * J21);
            A01 = w_det * (J11 * C0 * J12 + J21 * C1 * J22);
            A10 = A01;
            A11 = w_det * (J12 * C0 * J12 + J22 * C1 * J22);
         }
         D(q, 0, e) = A00;
         if (symmetric)
         {
            D(q, 1, e) = A10;
            D(q, 2, e) = A11;
         }
         else
         {
            D(q, 1, e) = A01;
            D(q, 2, e) = A10;
            D(q, 3, e) = A11;
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "PAJTQJSetupSimplexFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ, q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND, J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      const real_t w_det = W(q) / detJ;
      real_t A[3][3];
      if (coeffDim == 6 || coeffDim == 9)
      {
         const real_t M00 = CoeffAt(C, const_c, 0, q, e);
         const real_t M01 = CoeffAt(C, const_c, 1, q, e);
         const real_t M02 = CoeffAt(C, const_c, 2, q, e);
         const real_t M10 = symmetric ? M01 : CoeffAt(C, const_c, 3, q, e);
         const real_t M11 = symmetric ? CoeffAt(C, const_c, 3, q, e)
                            : CoeffAt(C, const_c, 4, q, e);
         const real_t M12 = symmetric ? CoeffAt(C, const_c, 4, q, e)
                            : CoeffAt(C, const_c, 5, q, e);
         const real_t M20 = symmetric ? M02 : CoeffAt(C, const_c, 6, q, e);
         const real_t M21 = symmetric ? M12 : CoeffAt(C, const_c, 7, q, e);
         const real_t M22 = symmetric ? CoeffAt(C, const_c, 5, q, e)
                            : CoeffAt(C, const_c, 8, q, e);
         const real_t R00 = M00 * J11 + M01 * J21 + M02 * J31;
         const real_t R01 = M00 * J12 + M01 * J22 + M02 * J32;
         const real_t R02 = M00 * J13 + M01 * J23 + M02 * J33;
         const real_t R10 = M10 * J11 + M11 * J21 + M12 * J31;
         const real_t R11 = M10 * J12 + M11 * J22 + M12 * J32;
         const real_t R12 = M10 * J13 + M11 * J23 + M12 * J33;
         const real_t R20 = M20 * J11 + M21 * J21 + M22 * J31;
         const real_t R21 = M20 * J12 + M21 * J22 + M22 * J32;
         const real_t R22 = M20 * J13 + M21 * J23 + M22 * J33;
         A[0][0] = w_det * (J11 * R00 + J21 * R10 + J31 * R20);
         A[0][1] = w_det * (J11 * R01 + J21 * R11 + J31 * R21);
         A[0][2] = w_det * (J11 * R02 + J21 * R12 + J31 * R22);
         A[1][0] = w_det * (J12 * R00 + J22 * R10 + J32 * R20);
         A[1][1] = w_det * (J12 * R01 + J22 * R11 + J32 * R21);
         A[1][2] = w_det * (J12 * R02 + J22 * R12 + J32 * R22);
         A[2][0] = w_det * (J13 * R00 + J23 * R10 + J33 * R20);
         A[2][1] = w_det * (J13 * R01 + J23 * R11 + J33 * R21);
         A[2][2] = w_det * (J13 * R02 + J23 * R12 + J33 * R22);
      }
      else
      {
         const real_t C0 = CoeffAt(C, const_c, 0, q, e);
         const real_t C1 = coeffDim == 3 ? CoeffAt(C, const_c, 1, q, e) : C0;
         const real_t C2 = coeffDim == 3 ? CoeffAt(C, const_c, 2, q, e) : C0;
         A[0][0] = w_det * (C0 * J11 * J11 + C1 * J21 * J21 + C2 * J31 * J31);
         A[0][1] = w_det * (C0 * J11 * J12 + C1 * J21 * J22 + C2 * J31 * J32);
         A[0][2] = w_det * (C0 * J11 * J13 + C1 * J21 * J23 + C2 * J31 * J33);
         A[1][0] = A[0][1];
         A[1][1] = w_det * (C0 * J12 * J12 + C1 * J22 * J22 + C2 * J32 * J32);
         A[1][2] = w_det * (C0 * J12 * J13 + C1 * J22 * J23 + C2 * J32 * J33);
         A[2][0] = A[0][2];
         A[2][1] = A[1][2];
         A[2][2] = w_det * (C0 * J13 * J13 + C1 * J23 * J23 + C2 * J33 * J33);
      }
      if (symmetric)
      {
         D(q, 0, e) = A[0][0];
         D(q, 1, e) = A[1][0];
         D(q, 2, e) = A[2][0];
         D(q, 3, e) = A[1][1];
         D(q, 4, e) = A[2][1];
         D(q, 5, e) = A[2][2];
      }
      else
      {
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
               D(q, i * 3 + j, e) = A[i][j];
            }
      }
   });
}

void PAJinvQJinvTSetupSimplexFromNodes(const int dim,
                                       const int coeffDim,
                                       const int NE,
                                       const int NQ,
                                       const int ND,
                                       const Array<real_t> &w,
                                       const Array<real_t> &g,
                                       const Vector &nodes_e,
                                       const Vector &c,
                                       Vector &d)
{
   const bool symmetric = (coeffDim != dim * dim);
   const bool const_c = c.Size() == coeffDim;
   const int pa_size = symmetric ? (dim * (dim + 1)) / 2 : dim * dim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), coeffDim, 1, 1)
                  : Reshape(c.Read(), coeffDim, NQ, NE);
   auto D = Reshape(d.Write(), NQ, pa_size, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ, q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t detJ = J11 * J22 - J21 * J12;
         const real_t idet = real_t(1) / detJ;
         const real_t i11 = J22 * idet, i12 = -J12 * idet;
         const real_t i21 = -J21 * idet, i22 = J11 * idet;
         real_t M00, M01, M10, M11;
         if (coeffDim == 3 || coeffDim == 4)
         {
            const real_t C00 = CoeffAt(C, const_c, 0, q, e);
            const real_t C01 = CoeffAt(C, const_c, 1, q, e);
            const real_t C10 = symmetric ? C01 : CoeffAt(C, const_c, 2, q, e);
            const real_t C11 = symmetric ? CoeffAt(C, const_c, 2, q, e)
                               : CoeffAt(C, const_c, 3, q, e);
            const real_t R00 = C00 * i11 + C01 * i12;
            const real_t R01 = C00 * i21 + C01 * i22;
            const real_t R10 = C10 * i11 + C11 * i12;
            const real_t R11 = C10 * i21 + C11 * i22;
            M00 = i11 * R00 + i12 * R10;
            M01 = i11 * R01 + i12 * R11;
            M10 = i21 * R00 + i22 * R10;
            M11 = i21 * R01 + i22 * R11;
         }
         else
         {
            const real_t C0 = CoeffAt(C, const_c, 0, q, e);
            const real_t C1 = coeffDim == 2 ? CoeffAt(C, const_c, 1, q, e) : C0;
            M00 = C0 * (i11 * i11) + C1 * (i12 * i12);
            M01 = C0 * (i11 * i21) + C1 * (i12 * i22);
            M10 = M01;
            M11 = C0 * (i21 * i21) + C1 * (i22 * i22);
         }
         const real_t wdet = W(q) * detJ;
         D(q, 0, e) = wdet * M00;
         if (symmetric)
         {
            D(q, 1, e) = wdet * M10;
            D(q, 2, e) = wdet * M11;
         }
         else
         {
            D(q, 1, e) = wdet * M01;
            D(q, 2, e) = wdet * M10;
            D(q, 3, e) = wdet * M11;
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3,
               "PAJinvQJinvTSetupSimplexFromNodes only supports dim 2/3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ, q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND, J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      real_t Cf11, Cf12, Cf13, Cf21, Cf22, Cf23, Cf31, Cf32, Cf33;
      CofactorsJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33,
                  Cf11, Cf12, Cf13, Cf21, Cf22, Cf23, Cf31, Cf32, Cf33);
      const real_t idet = real_t(1) / detJ;
      // CofactorsJ3 returns adj(J); J^{-1} = adj / det
      const real_t i11 = Cf11 * idet, i12 = Cf12 * idet, i13 = Cf13 * idet;
      const real_t i21 = Cf21 * idet, i22 = Cf22 * idet, i23 = Cf23 * idet;
      const real_t i31 = Cf31 * idet, i32 = Cf32 * idet, i33 = Cf33 * idet;
      real_t M[3][3];
      if (coeffDim == 6 || coeffDim == 9)
      {
         const real_t Q00 = CoeffAt(C, const_c, 0, q, e);
         const real_t Q01 = CoeffAt(C, const_c, 1, q, e);
         const real_t Q02 = CoeffAt(C, const_c, 2, q, e);
         const real_t Q10 = symmetric ? Q01 : CoeffAt(C, const_c, 3, q, e);
         const real_t Q11 = symmetric ? CoeffAt(C, const_c, 3, q, e)
                            : CoeffAt(C, const_c, 4, q, e);
         const real_t Q12 = symmetric ? CoeffAt(C, const_c, 4, q, e)
                            : CoeffAt(C, const_c, 5, q, e);
         const real_t Q20 = symmetric ? Q02 : CoeffAt(C, const_c, 6, q, e);
         const real_t Q21 = symmetric ? Q12 : CoeffAt(C, const_c, 7, q, e);
         const real_t Q22 = symmetric ? CoeffAt(C, const_c, 5, q, e)
                            : CoeffAt(C, const_c, 8, q, e);
         const real_t R00 = Q00 * i11 + Q01 * i12 + Q02 * i13;
         const real_t R01 = Q00 * i21 + Q01 * i22 + Q02 * i23;
         const real_t R02 = Q00 * i31 + Q01 * i32 + Q02 * i33;
         const real_t R10 = Q10 * i11 + Q11 * i12 + Q12 * i13;
         const real_t R11 = Q10 * i21 + Q11 * i22 + Q12 * i23;
         const real_t R12 = Q10 * i31 + Q11 * i32 + Q12 * i33;
         const real_t R20 = Q20 * i11 + Q21 * i12 + Q22 * i13;
         const real_t R21 = Q20 * i21 + Q21 * i22 + Q22 * i23;
         const real_t R22 = Q20 * i31 + Q21 * i32 + Q22 * i33;
         M[0][0] = i11 * R00 + i12 * R10 + i13 * R20;
         M[0][1] = i11 * R01 + i12 * R11 + i13 * R21;
         M[0][2] = i11 * R02 + i12 * R12 + i13 * R22;
         M[1][0] = i21 * R00 + i22 * R10 + i23 * R20;
         M[1][1] = i21 * R01 + i22 * R11 + i23 * R21;
         M[1][2] = i21 * R02 + i22 * R12 + i23 * R22;
         M[2][0] = i31 * R00 + i32 * R10 + i33 * R20;
         M[2][1] = i31 * R01 + i32 * R11 + i33 * R21;
         M[2][2] = i31 * R02 + i32 * R12 + i33 * R22;
      }
      else
      {
         const real_t C0 = CoeffAt(C, const_c, 0, q, e);
         const real_t C1 = coeffDim == 3 ? CoeffAt(C, const_c, 1, q, e) : C0;
         const real_t C2 = coeffDim == 3 ? CoeffAt(C, const_c, 2, q, e) : C0;
         M[0][0] = C0 * i11 * i11 + C1 * i12 * i12 + C2 * i13 * i13;
         M[0][1] = C0 * i11 * i21 + C1 * i12 * i22 + C2 * i13 * i23;
         M[0][2] = C0 * i11 * i31 + C1 * i12 * i32 + C2 * i13 * i33;
         M[1][0] = M[0][1];
         M[1][1] = C0 * i21 * i21 + C1 * i22 * i22 + C2 * i23 * i23;
         M[1][2] = C0 * i21 * i31 + C1 * i22 * i32 + C2 * i23 * i33;
         M[2][0] = M[0][2];
         M[2][1] = M[1][2];
         M[2][2] = C0 * i31 * i31 + C1 * i32 * i32 + C2 * i33 * i33;
      }
      const real_t wdet = W(q) * detJ;
      if (symmetric)
      {
         D(q, 0, e) = wdet * M[0][0];
         D(q, 1, e) = wdet * M[1][0];
         D(q, 2, e) = wdet * M[2][0];
         D(q, 3, e) = wdet * M[1][1];
         D(q, 4, e) = wdet * M[2][1];
         D(q, 5, e) = wdet * M[2][2];
      }
      else
      {
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
               D(q, i * 3 + j, e) = wdet * M[i][j];
            }
      }
   });
}

} // namespace mfem::internal
