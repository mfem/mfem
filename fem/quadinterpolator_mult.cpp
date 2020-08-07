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

#include "quadinterpolator.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Mult2D(const int NE,
                                    const int vdim,
                                    const DofToQuad &maps,
                                    const Vector &e_vec,
                                    Vector &q_val,
                                    Vector &q_der,
                                    Vector &q_det,
                                    const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND2D, "");
   MFEM_VERIFY(NQ <= MAX_NQ2D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 2, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 2, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND2D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      double s_E[max_VDIM*max_ND];
      for (int d = 0; d < ND; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,e) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES) || (eval_flags & DETERMINANTS))
         {
            // use MAX_VDIM2D to avoid "subscript out of range" warnings
            double D[MAX_VDIM2D*2];
            for (int i = 0; i < 2*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
               }
            }
            if (VDIM == 2 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 2) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 2).
               det(q,e) = kernels::Det<2>(D);
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Mult3D(const int NE,
                                    const int vdim,
                                    const DofToQuad &maps,
                                    const Vector &e_vec,
                                    Vector &q_val,
                                    Vector &q_der,
                                    Vector &q_det,
                                    const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND = T_ND ? T_ND : nd;
   const int NQ = T_NQ ? T_NQ : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND3D, "");
   MFEM_VERIFY(NQ <= MAX_NQ3D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ, ND);
   auto G = Reshape(maps.G.Read(), NQ, 3, ND);
   auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = Reshape(q_val.Write(), NQ, VDIM, NE);
   auto der = Reshape(q_der.Write(), NQ, VDIM, 3, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL(e, NE,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND3D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      double s_E[max_VDIM*max_ND];
      for (int d = 0; d < ND; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         if (eval_flags & VALUES)
         {
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*s_E[c+d*VDIM]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,e) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES) || (eval_flags & DETERMINANTS))
         {
            // use MAX_VDIM3D to avoid "subscript out of range" warnings
            double D[MAX_VDIM3D*3];
            for (int i = 0; i < 3*VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND; ++d)
            {
               const double wx = G(q,0,d);
               const double wy = G(q,1,d);
               const double wz = G(q,2,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = s_E[c+d*VDIM];
                  D[c+VDIM*0] += s_e * wx;
                  D[c+VDIM*1] += s_e * wy;
                  D[c+VDIM*2] += s_e * wz;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  der(q,c,0,e) = D[c+VDIM*0];
                  der(q,c,1,e) = D[c+VDIM*1];
                  der(q,c,2,e) = D[c+VDIM*2];
               }
            }
            if (VDIM == 3 && (eval_flags & DETERMINANTS))
            {
               // The check (VDIM == 3) should eliminate this block when VDIM is
               // known at compile time and (VDIM != 3).
               det(q,e) = kernels::Det<3>(D);
            }
         }
      }
   });
}

template<>
void QuadratureInterpolator::Mult<QVectorLayout::byNODES>(const Vector &e_vec,
                                                          unsigned eval_flags,
                                                          Vector &q_val,
                                                          Vector &q_der,
                                                          Vector &q_det) const
{
   MFEM_VERIFY(!use_tensor_products, "");
   const int ne = fespace->GetNE();
   if (ne == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::FULL);
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   void (*mult)(const int NE,
                const int vdim,
                const DofToQuad &maps,
                const Vector &e_vec,
                Vector &q_val,
                Vector &q_der,
                Vector &q_det,
                const int eval_flags) = NULL;

   if (vdim == 1)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q0
            case 101: mult = &Mult2D<1,1,1>; break;
            case 104: mult = &Mult2D<1,1,4>; break;
            // Q1
            case 404: mult = &Mult2D<1,4,4>; break;
            case 409: mult = &Mult2D<1,4,9>; break;
            // Q2
            case 909: mult = &Mult2D<1,9,9>; break;
            case 916: mult = &Mult2D<1,9,16>; break;
            // Q3
            case 1616: mult = &Mult2D<1,16,16>; break;
            case 1625: mult = &Mult2D<1,16,25>; break;
            case 1636: mult = &Mult2D<1,16,36>; break;
            // Q4
            case 2525: mult = &Mult2D<1,25,25>; break;
            case 2536: mult = &Mult2D<1,25,36>; break;
            case 2549: mult = &Mult2D<1,25,49>; break;
            case 2564: mult = &Mult2D<1,25,64>; break;
         }
         if (nq >= 100 || !mult)
         {
            mult = &Mult2D<1>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q0
            case 1001: mult = &Mult3D<1,1,1>; break;
            case 1008: mult = &Mult3D<1,1,8>; break;
            // Q1
            case 8008: mult = &Mult3D<1,8,8>; break;
            case 8027: mult = &Mult3D<1,8,27>; break;
            // Q2
            case 27027: mult = &Mult3D<1,27,27>; break;
            case 27064: mult = &Mult3D<1,27,64>; break;
            // Q3
            case 64064: mult = &Mult3D<1,64,64>; break;
            case 64125: mult = &Mult3D<1,64,125>; break;
            case 64216: mult = &Mult3D<1,64,216>; break;
            // Q4
            case 125125: mult = &Mult3D<1,125,125>; break;
            case 125216: mult = &Mult3D<1,125,216>; break;
         }
         if (nq >= 1000 || !mult)
         {
            mult = &Mult3D<1>;
         }
      }
   }
   else if (vdim == 3 && dim == 2)
   {
      switch (100*nd + nq)
      {
         // Q0
         case 101: mult = &Mult2D<3,1,1>; break;
         case 104: mult = &Mult2D<3,1,4>; break;
         // Q1
         case 404: mult = &Mult2D<3,4,4>; break;
         case 409: mult = &Mult2D<3,4,9>; break;
         // Q2
         case 904: mult = &Mult2D<3,9,4>; break;
         case 909: mult = &Mult2D<3,9,9>; break;
         case 916: mult = &Mult2D<3,9,16>; break;
         case 925: mult = &Mult2D<3,9,25>; break;
         // Q3
         case 1616: mult = &Mult2D<3,16,16>; break;
         case 1625: mult = &Mult2D<3,16,25>; break;
         case 1636: mult = &Mult2D<3,16,36>; break;
         // Q4
         case 2525: mult = &Mult2D<3,25,25>; break;
         case 2536: mult = &Mult2D<3,25,36>; break;
         case 2549: mult = &Mult2D<3,25,49>; break;
         case 2564: mult = &Mult2D<3,25,64>; break;
         default:   mult = &Mult2D<3>;
      }
   }
   else if (vdim == dim)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q1
            case 404: mult = &Mult2D<2,4,4>; break;
            case 409: mult = &Mult2D<2,4,9>; break;
            // Q2
            case 909: mult = &Mult2D<2,9,9>; break;
            case 916: mult = &Mult2D<2,9,16>; break;
            // Q3
            case 1616: mult = &Mult2D<2,16,16>; break;
            case 1625: mult = &Mult2D<2,16,25>; break;
            case 1636: mult = &Mult2D<2,16,36>; break;
            // Q4
            case 2525: mult = &Mult2D<2,25,25>; break;
            case 2536: mult = &Mult2D<2,25,36>; break;
            case 2549: mult = &Mult2D<2,25,49>; break;
            case 2564: mult = &Mult2D<2,25,64>; break;
         }
         if (nq >= 100 || !mult) { mult = &Mult2D<2>; }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q1
            case 8008: mult = &Mult3D<3,8,8>; break;
            case 8027: mult = &Mult3D<3,8,27>; break;
            // Q2
            case 27027: mult = &Mult3D<3,27,27>; break;
            case 27064: mult = &Mult3D<3,27,64>; break;
            case 27125: mult = &Mult3D<3,27,125>; break;
            // Q3
            case 64064: mult = &Mult3D<3,64,64>; break;
            case 64125: mult = &Mult3D<3,64,125>; break;
            case 64216: mult = &Mult3D<3,64,216>; break;
            // Q4
            case 125125: mult = &Mult3D<3,125,125>; break;
            case 125216: mult = &Mult3D<3,125,216>; break;
         }
         if (nq >= 1000 || !mult) {  mult = &Mult3D<3>; }
      }
   }
   if (mult) { mult(ne, vdim, maps, e_vec, q_val, q_der, q_det, eval_flags); }
   else { MFEM_ABORT("case not supported yet"); }
}

} // namespace mfem
