// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir):

   fespace(&fes),
   qspace(nullptr),
   IntRule(&ir),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(UsesTensorBasis(fes))
{
   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const QuadratureSpace &qs):

   fespace(&fes),
   qspace(&qs),
   IntRule(nullptr),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(UsesTensorBasis(fes))
{
   if (fespace->GetNE() == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(dynamic_cast<const ScalarFiniteElement*>(fe) != NULL,
               "Only scalar finite elements are supported");
}

template<const int T_VDIM, const int T_ND, const int T_NQ>
void QuadratureInterpolator::Eval2D(const int NE,
                                    const int vdim,
                                    const QVectorLayout q_layout,
                                    const GeometricFactors *geom,
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
   const int NMAX = NQ > ND ? NQ : ND;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND2D, "");
   MFEM_VERIFY(NQ <= MAX_NQ2D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   MFEM_VERIFY((!geom) || (eval_flags & PHYSICAL_DERIVATIVES), "");
   const auto B = Reshape(maps.B.Read(), NQ, ND);
   const auto G = Reshape(maps.G.Read(), NQ, 2, ND);
   const auto J = Reshape(geom ? geom->J.Read() : nullptr, NQ, 2, 2, NE);
   const auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = q_layout == QVectorLayout:: byNODES ?
              Reshape(q_val.Write(), NQ, VDIM, NE):
              Reshape(q_val.Write(), VDIM, NQ, NE);
   auto der = q_layout == QVectorLayout:: byNODES ?
              Reshape(q_der.Write(), NQ, VDIM, 2, NE):
              Reshape(q_der.Write(), VDIM, 2, NQ, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL_2D(e, NE, NMAX, 1, 1,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND2D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      MFEM_SHARED double s_E[max_VDIM*max_ND];
      MFEM_FOREACH_THREAD(d, x, ND)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(q, x, NQ)
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
            for (int c = 0; c < VDIM; c++)
            {
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,e) = ed[c]; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,e) = ed[c]; }
            }
         }
         if ((eval_flags & DERIVATIVES) ||
             (eval_flags & PHYSICAL_DERIVATIVES) ||
             (eval_flags & DETERMINANTS))
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
            if ((eval_flags & DERIVATIVES) ||
                (eval_flags & PHYSICAL_DERIVATIVES))
            {
               for (int c = 0; c < VDIM; c++)
               {
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = D[c+VDIM*0];
                     der(c,1,q,e) = D[c+VDIM*1];
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = D[c+VDIM*0];
                     der(q,c,1,e) = D[c+VDIM*1];
                  }
               }
            }
            if (eval_flags & PHYSICAL_DERIVATIVES)
            {
               double Jloc[4], Jinv[4];
               Jloc[0] = J(q,0,0,e);
               Jloc[1] = J(q,1,0,e);
               Jloc[2] = J(q,0,1,e);
               Jloc[3] = J(q,1,1,e);
               kernels::CalcInverse<2>(Jloc, Jinv);
               for (int c = 0; c < VDIM; c++)
               {
                  const double u = D[c+VDIM*0];
                  const double v = D[c+VDIM*1];
                  const double JiU = Jinv[0]*u + Jinv[1]*v;
                  const double JiV = Jinv[2]*u + Jinv[3]*v;
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = JiU;
                     der(c,1,q,e) = JiV;
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = JiU;
                     der(q,c,1,e) = JiV;
                  }
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
void QuadratureInterpolator::Eval3D(const int NE,
                                    const int vdim,
                                    const QVectorLayout q_layout,
                                    const GeometricFactors *geom,
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
   const int NMAX = NQ > ND ? NQ : ND;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND <= MAX_ND3D, "");
   MFEM_VERIFY(NQ <= MAX_NQ3D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   MFEM_VERIFY((!geom) || (eval_flags & PHYSICAL_DERIVATIVES), "");
   const auto B = Reshape(maps.B.Read(), NQ, ND);
   const auto G = Reshape(maps.G.Read(), NQ, 3, ND);
   const auto J = Reshape(geom ? geom->J.Read() : nullptr, NQ, 3, 3, NE);
   const auto E = Reshape(e_vec.Read(), ND, VDIM, NE);
   auto val = q_layout == QVectorLayout:: byNODES ?
              Reshape(q_val.Write(), NQ, VDIM, NE):
              Reshape(q_val.Write(), VDIM, NQ, NE);
   auto der = q_layout == QVectorLayout:: byNODES ?
              Reshape(q_der.Write(), NQ, VDIM, 3, NE):
              Reshape(q_der.Write(), VDIM, 3, NQ, NE);
   auto det = Reshape(q_det.Write(), NQ, NE);
   MFEM_FORALL_2D(e, NE, NMAX, 1, 1,
   {
      const int ND = T_ND ? T_ND : nd;
      const int NQ = T_NQ ? T_NQ : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND = T_ND ? T_ND : MAX_ND3D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      MFEM_SHARED double s_E[max_VDIM*max_ND];
      MFEM_FOREACH_THREAD(d, x, ND)
      {
         for (int c = 0; c < VDIM; c++)
         {
            s_E[c+d*VDIM] = E(d,c,e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(q, x, NQ)
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
            for (int c = 0; c < VDIM; c++)
            {
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,e) = ed[c]; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,e) = ed[c]; }
            }
         }
         if ((eval_flags & DERIVATIVES) ||
             (eval_flags & PHYSICAL_DERIVATIVES) ||
             (eval_flags & DETERMINANTS))
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
            if ((eval_flags & DERIVATIVES) ||
                (eval_flags & PHYSICAL_DERIVATIVES))
            {
               for (int c = 0; c < VDIM; c++)
               {
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = D[c+VDIM*0];
                     der(c,1,q,e) = D[c+VDIM*1];
                     der(c,2,q,e) = D[c+VDIM*2];
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = D[c+VDIM*0];
                     der(q,c,1,e) = D[c+VDIM*1];
                     der(q,c,2,e) = D[c+VDIM*2];
                  }
               }
            }
            if (eval_flags & PHYSICAL_DERIVATIVES)
            {
               double Jloc[9], Jinv[9];
               for (int col = 0; col < 3; col++)
               {
                  for (int row = 0; row < 3; row++)
                  {
                     Jloc[row+3*col] = J(q,row,col,e);
                  }
               }
               kernels::CalcInverse<3>(Jloc, Jinv);
               for (int c = 0; c < VDIM; c++)
               {
                  const double u = D[c+VDIM*0];
                  const double v = D[c+VDIM*1];
                  const double w = D[c+VDIM*2];
                  const double JiU = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                  const double JiV = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                  const double JiW = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,0,q,e) = JiU;
                     der(c,1,q,e) = JiV;
                     der(c,2,q,e) = JiW;
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     der(q,c,0,e) = JiU;
                     der(q,c,1,e) = JiV;
                     der(q,c,2,e) = JiW;
                  }
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

void QuadratureInterpolator::Mult(const Vector &e_vec,
                                  unsigned eval_flags,
                                  Vector &q_val,
                                  Vector &q_der,
                                  Vector &q_det) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         if (eval_flags & VALUES)
         {
            Values<QVectorLayout::byNODES>(e_vec, q_val);
         }
         if (eval_flags & DERIVATIVES)
         {
            Derivatives<QVectorLayout::byNODES>(e_vec, q_der);
         }
         if (eval_flags & DETERMINANTS)
         {
            Determinants(e_vec, q_det);
         }
         if (eval_flags & PHYSICAL_DERIVATIVES)
         {
            PhysDerivatives<QVectorLayout::byNODES>(e_vec, q_der);
         }
      }

      if (q_layout == QVectorLayout::byVDIM)
      {
         if (eval_flags & VALUES)
         {
            Values<QVectorLayout::byVDIM>(e_vec, q_val);
         }
         if (eval_flags & DERIVATIVES)
         {
            Derivatives<QVectorLayout::byVDIM>(e_vec, q_der);
         }
         if (eval_flags & DETERMINANTS)
         {
            Determinants(e_vec, q_det);
         }
         if (eval_flags & PHYSICAL_DERIVATIVES)
         {
            PhysDerivatives<QVectorLayout::byVDIM>(e_vec, q_der);
         }
      }
   }
   else
   {
      const int ne = fespace->GetNE();
      if (ne == 0) { return; }
      const int vdim = fespace->GetVDim();
      const int dim = fespace->GetMesh()->Dimension();
      const FiniteElement *fe = fespace->GetFE(0);
      const IntegrationRule *ir =
         IntRule ? IntRule : &qspace->GetElementIntRule(0);
      constexpr DofToQuad::Mode mode = DofToQuad::FULL;
      const DofToQuad &maps = fe->GetDofToQuad(*ir, mode);
      const GeometricFactors *geom = nullptr;
      if (eval_flags & PHYSICAL_DERIVATIVES)
      {
         const int jacobians = GeometricFactors::JACOBIANS;
         geom = fespace->GetMesh()->GetGeometricFactors(*ir, jacobians);
      }
      const int nd = maps.ndof;
      const int nq = maps.nqpt;

      void (*mult)(const int NE,
                   const int vdim,
                   const QVectorLayout q_layout,
                   const GeometricFactors *geom,
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
               case 101: mult = &Eval2D<1,1,1>; break;
               case 104: mult = &Eval2D<1,1,4>; break;
               // Q1
               case 404: mult = &Eval2D<1,4,4>; break;
               case 409: mult = &Eval2D<1,4,9>; break;
               // Q2
               case 909: mult = &Eval2D<1,9,9>; break;
               case 916: mult = &Eval2D<1,9,16>; break;
               // Q3
               case 1616: mult = &Eval2D<1,16,16>; break;
               case 1625: mult = &Eval2D<1,16,25>; break;
               case 1636: mult = &Eval2D<1,16,36>; break;
               // Q4
               case 2525: mult = &Eval2D<1,25,25>; break;
               case 2536: mult = &Eval2D<1,25,36>; break;
               case 2549: mult = &Eval2D<1,25,49>; break;
               case 2564: mult = &Eval2D<1,25,64>; break;
            }
            if (nq >= 100 || !mult)
            {
               mult = &Eval2D<1>;
            }
         }
         else if (dim == 3)
         {
            switch (1000*nd + nq)
            {
               // Q0
               case 1001: mult = &Eval3D<1,1,1>; break;
               case 1008: mult = &Eval3D<1,1,8>; break;
               // Q1
               case 8008: mult = &Eval3D<1,8,8>; break;
               case 8027: mult = &Eval3D<1,8,27>; break;
               // Q2
               case 27027: mult = &Eval3D<1,27,27>; break;
               case 27064: mult = &Eval3D<1,27,64>; break;
               // Q3
               case 64064: mult = &Eval3D<1,64,64>; break;
               case 64125: mult = &Eval3D<1,64,125>; break;
               case 64216: mult = &Eval3D<1,64,216>; break;
               // Q4
               case 125125: mult = &Eval3D<1,125,125>; break;
               case 125216: mult = &Eval3D<1,125,216>; break;
            }
            if (nq >= 1000 || !mult)
            {
               mult = &Eval3D<1>;
            }
         }
      }
      else if (vdim == 3 && dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q0
            case 101: mult = &Eval2D<3,1,1>; break;
            case 104: mult = &Eval2D<3,1,4>; break;
            // Q1
            case 404: mult = &Eval2D<3,4,4>; break;
            case 409: mult = &Eval2D<3,4,9>; break;
            // Q2
            case 904: mult = &Eval2D<3,9,4>; break;
            case 909: mult = &Eval2D<3,9,9>; break;
            case 916: mult = &Eval2D<3,9,16>; break;
            case 925: mult = &Eval2D<3,9,25>; break;
            // Q3
            case 1616: mult = &Eval2D<3,16,16>; break;
            case 1625: mult = &Eval2D<3,16,25>; break;
            case 1636: mult = &Eval2D<3,16,36>; break;
            // Q4
            case 2525: mult = &Eval2D<3,25,25>; break;
            case 2536: mult = &Eval2D<3,25,36>; break;
            case 2549: mult = &Eval2D<3,25,49>; break;
            case 2564: mult = &Eval2D<3,25,64>; break;
            default:   mult = &Eval2D<3>;
         }
      }
      else if (vdim == dim)
      {
         if (dim == 2)
         {
            switch (100*nd + nq)
            {
               // Q1
               case 404: mult = &Eval2D<2,4,4>; break;
               case 409: mult = &Eval2D<2,4,9>; break;
               // Q2
               case 909: mult = &Eval2D<2,9,9>; break;
               case 916: mult = &Eval2D<2,9,16>; break;
               // Q3
               case 1616: mult = &Eval2D<2,16,16>; break;
               case 1625: mult = &Eval2D<2,16,25>; break;
               case 1636: mult = &Eval2D<2,16,36>; break;
               // Q4
               case 2525: mult = &Eval2D<2,25,25>; break;
               case 2536: mult = &Eval2D<2,25,36>; break;
               case 2549: mult = &Eval2D<2,25,49>; break;
               case 2564: mult = &Eval2D<2,25,64>; break;
            }
            if (nq >= 100 || !mult) { mult = &Eval2D<2>; }
         }
         else if (dim == 3)
         {
            switch (1000*nd + nq)
            {
               // Q1
               case 8008: mult = &Eval3D<3,8,8>; break;
               case 8027: mult = &Eval3D<3,8,27>; break;
               // Q2
               case 27027: mult = &Eval3D<3,27,27>; break;
               case 27064: mult = &Eval3D<3,27,64>; break;
               case 27125: mult = &Eval3D<3,27,125>; break;
               // Q3
               case 64064: mult = &Eval3D<3,64,64>; break;
               case 64125: mult = &Eval3D<3,64,125>; break;
               case 64216: mult = &Eval3D<3,64,216>; break;
               // Q4
               case 125125: mult = &Eval3D<3,125,125>; break;
               case 125216: mult = &Eval3D<3,125,216>; break;
            }
            if (nq >= 1000 || !mult) {  mult = &Eval3D<3>; }
         }
      }
      if (mult)
      {
         mult(ne,vdim,q_layout,geom,maps,e_vec,q_val,q_der,q_det,eval_flags);
      }
      else { MFEM_ABORT("case not supported yet"); }
   }
}

void QuadratureInterpolator::MultTranspose(unsigned eval_flags,
                                           const Vector &q_val,
                                           const Vector &q_der,
                                           Vector &e_vec) const
{
   MFEM_ABORT("this method is not implemented yet");
}

void QuadratureInterpolator::Values(const Vector &e_vec,
                                    Vector &q_val) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Values<QVectorLayout::byNODES>(e_vec, q_val);
      }

      if (q_layout == QVectorLayout::byVDIM)
      {
         Values<QVectorLayout::byVDIM>(e_vec, q_val);
      }
   }
   else
   {
      Vector empty;
      Mult(e_vec, VALUES, q_val, empty, empty);
   }
}

void QuadratureInterpolator::Derivatives(const Vector &e_vec,
                                         Vector &q_der) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         Derivatives<QVectorLayout::byNODES>(e_vec, q_der);
      }

      if (q_layout == QVectorLayout::byVDIM)
      {
         Derivatives<QVectorLayout::byVDIM>(e_vec, q_der);
      }
   }
   else
   {
      Vector empty;
      Mult(e_vec, DERIVATIVES, empty, q_der, empty);
   }
}

void QuadratureInterpolator::PhysDerivatives(const Vector &e_vec,
                                             Vector &q_der) const
{
   if (use_tensor_products)
   {
      if (q_layout == QVectorLayout::byNODES)
      {
         PhysDerivatives<QVectorLayout::byNODES>(e_vec, q_der);
      }
      if (q_layout == QVectorLayout::byVDIM)
      {
         PhysDerivatives<QVectorLayout::byVDIM>(e_vec, q_der);
      }
   }
   else
   {
      Vector empty;
      Mult(e_vec, PHYSICAL_DERIVATIVES, empty, q_der, empty);
   }
}

} // namespace mfem
