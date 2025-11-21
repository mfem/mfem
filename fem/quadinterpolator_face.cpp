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

#include "quadinterpolator_face.hpp"
#include "../general/annotation.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

/// Return the sign to apply to the normals on each face to point from e1 to e2.
static void GetSigns(const FiniteElementSpace &fes, const FaceType type,
                     Array<bool> &signs)
{
   const Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.SpaceDimension();
   int face_id;
   int f_ind = 0;
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      Mesh::FaceInformation face = mesh.GetFaceInformation(f);
      face_id = face.element[0].local_face_id;
      if (face.IsNonconformingCoarse())
      {
         // We skip nonconforming coarse-fine faces as they are treated
         // by the corresponding nonconforming fine-coarse faces.
         continue;
      }
      else if ( face.IsOfFaceType(type) )
      {
         if (dim==2)
         {
            if (face_id==2 || face_id==3)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         else if (dim==3)
         {
            if (face_id==0 || face_id==3 || face_id==4)
            {
               signs[f_ind] = true;
            }
            else
            {
               signs[f_ind] = false;
            }
         }
         f_ind++;
      }
   }
}

FaceQuadratureInterpolator::FaceQuadratureInterpolator(
   const FiniteElementSpace &fes,
   const IntegrationRule &ir, FaceType type_)
   : type(type_), nf(fes.GetNFbyType(type)), signs(nf),
     q_layout(QVectorLayout::byNODES)
{
   fespace = &fes;
   IntRule = &ir;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   GetSigns(*fespace, type, signs);
   MFEM_VERIFY(SupportsFESpace(fes), "Unsupported finite element space");
}

bool FaceQuadratureInterpolator::SupportsFESpace(const FiniteElementSpace &fes)
{
   const FiniteElement *fe = fes.GetTypicalFE();
   const auto *sfe = dynamic_cast<const ScalarFiniteElement*>(fe);
   const auto *tfe = dynamic_cast<const TensorBasisElement*>(fe);
   return sfe != nullptr && tfe != nullptr && (
             tfe->GetBasisType() == BasisType::GaussLobatto ||
             tfe->GetBasisType() == BasisType::Positive);
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::Eval2D(
   const int NF,
   const int vdim,
   const QVectorLayout q_layout,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &f_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd1d = maps.ndof;
   const int nq1d = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd1d;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(f_vec.Read(), ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), NQ1D, VDIM, NF):
              Reshape(q_val.Write(), VDIM, NQ1D, NF);
   auto der = q_layout == QVectorLayout::byNODES ? // only tangential der
              Reshape(q_der.Write(), NQ1D, VDIM, NF):
              Reshape(q_der.Write(), VDIM, NQ1D, NF);
   auto det = Reshape(q_det.Write(), NQ1D, NF);
   auto n   = q_layout == QVectorLayout::byNODES ?
              Reshape(q_nor.Write(), NQ1D, 2, NF):
              Reshape(q_nor.Write(), 2, NQ1D, NF);
   // If Gauss-Lobatto
   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      const int ND1D = T_ND1D ? T_ND1D : nd1d;
      const int NQ1D = T_NQ1D ? T_NQ1D : nq1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      real_t r_F[max_ND1D][max_VDIM];
      for (int d = 0; d < ND1D; d++)
      {
         for (int c = 0; c < VDIM; c++)
         {
            r_F[d][c] = F(d,c,f);
         }
      }
      for (int q = 0; q < NQ1D; ++q)
      {
         if (eval_flags & VALUES)
         {
            real_t ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const real_t b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*r_F[d][c]; }
            }
            for (int c = 0; c < VDIM; c++)
            {
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,f) = ed[c]; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,f) = ed[c]; }
            }
         }
         if ((eval_flags & DERIVATIVES)
             || (eval_flags & DETERMINANTS)
             || (eval_flags & NORMALS))
         {
            real_t D[max_VDIM];
            for (int i = 0; i < VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const real_t w = G(q,d);
               for (int c = 0; c < VDIM; c++)
               {
                  real_t s_e = r_F[d][c];
                  D[c] += s_e * w;
               }
            }
            if (eval_flags & DERIVATIVES)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     der(c,q,f) =  D[c];
                  }
                  else // q_layout == QVectorLayout::byNODES
                  {
                     der(q,c,f) =  D[c];
                  }
               }
            }
            if (VDIM == 2 &&
                ((eval_flags & NORMALS)
                 || (eval_flags & DETERMINANTS)))
            {
               const real_t norm = sqrt(D[0]*D[0]+D[1]*D[1]);
               if (eval_flags & DETERMINANTS)
               {
                  det(q,f) = norm;
               }
               if (eval_flags & NORMALS)
               {
                  const real_t s = sign[f] ? -1.0 : 1.0;
                  if (q_layout == QVectorLayout::byVDIM)
                  {
                     n(0,q,f) =  s*D[1]/norm;
                     n(1,q,f) = -s*D[0]/norm;
                  }
                  if (q_layout == QVectorLayout::byNODES)
                  {
                     n(q,0,f) =  s*D[1]/norm;
                     n(q,1,f) = -s*D[0]/norm;
                  }
               }
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::Eval3D(
   const int NF,
   const int vdim,
   const QVectorLayout q_layout,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd1d = maps.ndof;
   const int nq1d = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd1d;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(e_vec.Read(), ND1D, ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), NQ1D, NQ1D, VDIM, NF):
              Reshape(q_val.Write(), VDIM, NQ1D, NQ1D, NF);
   auto der = q_layout == QVectorLayout::byNODES ?
              Reshape(q_der.Write(), NQ1D, NQ1D, VDIM, 2, NF):
              Reshape(q_der.Write(), VDIM, 2, NQ1D, NQ1D, NF);
   auto det = Reshape(q_det.Write(), NQ1D, NQ1D, NF);
   auto nor = q_layout == QVectorLayout::byNODES ?
              Reshape(q_nor.Write(), NQ1D, NQ1D, 3, NF):
              Reshape(q_nor.Write(), 3, NQ1D, NQ1D, NF);
   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_NQ1D = T_NQ1D ? T_NQ1D : MAX_NQ1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      real_t r_F[max_ND1D][max_ND1D][max_VDIM];
      for (int d1 = 0; d1 < ND1D; d1++)
      {
         for (int d2 = 0; d2 < ND1D; d2++)
         {
            for (int c = 0; c < VDIM; c++)
            {
               r_F[d1][d2][c] = F(d1,d2,c,f);
            }
         }
      }
      if (eval_flags & VALUES)
      {
         real_t Bu[max_NQ1D][max_ND1D][max_VDIM];
         for (int d2 = 0; d2 < ND1D; ++d2)
         {
            for (int q = 0; q < NQ1D; ++q)
            {
               for (int c = 0; c < VDIM; c++) { Bu[q][d2][c] = 0.0; }
               for (int d1 = 0; d1 < ND1D; ++d1)
               {
                  const real_t b = B(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     Bu[q][d2][c] += b*r_F[d1][d2][c];
                  }
               }
            }
         }
         real_t BBu[max_NQ1D][max_NQ1D][max_VDIM];
         for (int q2 = 0; q2 < NQ1D; ++q2)
         {
            for (int q1 = 0; q1 < NQ1D; ++q1)
            {
               for (int c = 0; c < VDIM; c++) { BBu[q2][q1][c] = 0.0; }
               for (int d2 = 0; d2 < ND1D; ++d2)
               {
                  const real_t b = B(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BBu[q2][q1][c] += b*Bu[q1][d2][c];
                  }
               }
               for (int c = 0; c < VDIM; c++)
               {
                  const real_t v = BBu[q2][q1][c];
                  if (q_layout == QVectorLayout::byVDIM)  { val(c,q1,q2,f) = v; }
                  if (q_layout == QVectorLayout::byNODES) { val(q1,q2,c,f) = v; }
               }
            }
         }
      }
      if ((eval_flags & DERIVATIVES)
          || (eval_flags & DETERMINANTS)
          || (eval_flags & NORMALS))
      {
         // We only compute the tangential derivatives
         real_t Gu[max_NQ1D][max_ND1D][max_VDIM];
         real_t Bu[max_NQ1D][max_ND1D][max_VDIM];
         for (int d2 = 0; d2 < ND1D; ++d2)
         {
            for (int q = 0; q < NQ1D; ++q)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  Gu[q][d2][c] = 0.0;
                  Bu[q][d2][c] = 0.0;
               }
               for (int d1 = 0; d1 < ND1D; ++d1)
               {
                  const real_t b = B(q,d1);
                  const real_t g = G(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     const real_t u = r_F[d1][d2][c];
                     Gu[q][d2][c] += g*u;
                     Bu[q][d2][c] += b*u;
                  }
               }
            }
         }
         real_t BGu[max_NQ1D][max_NQ1D][max_VDIM];
         real_t GBu[max_NQ1D][max_NQ1D][max_VDIM];
         for (int q2 = 0; q2 < NQ1D; ++q2)
         {
            for (int q1 = 0; q1 < NQ1D; ++q1)
            {
               for (int c = 0; c < VDIM; c++)
               {
                  BGu[q2][q1][c] = 0.0;
                  GBu[q2][q1][c] = 0.0;
               }
               for (int d2 = 0; d2 < ND1D; ++d2)
               {
                  const real_t b = B(q2,d2);
                  const real_t g = G(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BGu[q2][q1][c] += b*Gu[q1][d2][c];
                     GBu[q2][q1][c] += g*Bu[q1][d2][c];
                  }
               }
            }
         }
         if (eval_flags & DERIVATIVES)
         {
            for (int c = 0; c < VDIM; c++)
            {
               for (int q2 = 0; q2 < NQ1D; ++q2)
               {
                  for (int q1 = 0; q1 < NQ1D; ++q1)
                  {
                     if (q_layout == QVectorLayout::byVDIM)
                     {
                        der(c,0,q1,q2,f) = BGu[q2][q1][c];
                        der(c,1,q1,q2,f) = GBu[q2][q1][c];
                     }
                     else // q_layout == QVectorLayout::byNODES
                     {
                        der(q1,q2,c,0,f) = BGu[q2][q1][c];
                        der(q1,q2,c,1,f) = GBu[q2][q1][c];
                     }
                  }
               }
            }
         }
         if (VDIM == 3 && ((eval_flags & NORMALS) ||
                           (eval_flags & DETERMINANTS)))
         {
            real_t n[3];
            for (int q2 = 0; q2 < NQ1D; ++q2)
            {
               for (int q1 = 0; q1 < NQ1D; ++q1)
               {
                  const real_t s = sign[f] ? -1.0 : 1.0;
                  n[0] = s*( BGu[q2][q1][1]*GBu[q2][q1][2]-GBu[q2][q1][1]*
                             BGu[q2][q1][2] );
                  n[1] = s*(-BGu[q2][q1][0]*GBu[q2][q1][2]+GBu[q2][q1][0]*
                            BGu[q2][q1][2] );
                  n[2] = s*( BGu[q2][q1][0]*GBu[q2][q1][1]-GBu[q2][q1][0]*
                             BGu[q2][q1][1] );
                  const real_t norm = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
                  if (eval_flags & DETERMINANTS) { det(q1,q2,f) = norm; }
                  if (eval_flags & NORMALS)
                  {
                     if (q_layout == QVectorLayout::byVDIM)
                     {
                        nor(0,q1,q2,f) = n[0]/norm;
                        nor(1,q1,q2,f) = n[1]/norm;
                        nor(2,q1,q2,f) = n[2]/norm;
                     }
                     if (q_layout == QVectorLayout::byNODES)
                     {
                        nor(q1,q2,0,f) = n[0]/norm;
                        nor(q1,q2,1,f) = n[1]/norm;
                        nor(q1,q2,2,f) = n[2]/norm;
                     }
                  }
               }
            }
         }
      }
   });
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::SmemEval3D(
   const int NF,
   const int vdim,
   const QVectorLayout q_layout,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   MFEM_PERF_SCOPE("FaceQuadInterpolator::SmemEval3D");
   const int nd1d = maps.ndof;
   const int nq1d = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd1d;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(e_vec.Read(), ND1D, ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), NQ1D, NQ1D, VDIM, NF):
              Reshape(q_val.Write(), VDIM, NQ1D, NQ1D, NF);
   auto der = q_layout == QVectorLayout::byNODES ?
              Reshape(q_der.Write(), NQ1D, NQ1D, VDIM, 2, NF):
              Reshape(q_der.Write(), VDIM, 2, NQ1D, NQ1D, NF);
   auto det = Reshape(q_det.Write(), NQ1D, NQ1D, NF);
   auto nor = q_layout == QVectorLayout::byNODES ?
              Reshape(q_nor.Write(), NQ1D, NQ1D, 3, NF):
              Reshape(q_nor.Write(), 3, NQ1D, NQ1D, NF);

   mfem::forall_3D(NF, NQ1D, NQ1D, VDIM, [=] MFEM_HOST_DEVICE (int f)
   {
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_NQ1D = T_NQ1D ? T_NQ1D : MAX_NQ1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;

      MFEM_SHARED real_t sm1[max_NQ1D*max_NQ1D*max_VDIM];
      MFEM_SHARED real_t sm2[max_NQ1D*max_ND1D*max_VDIM];

      auto s_F = (real_t(*)[max_ND1D][max_VDIM])sm1;
      MFEM_FOREACH_THREAD(d1,x,ND1D)
      {
         MFEM_FOREACH_THREAD(d2,y,ND1D)
         {
            MFEM_FOREACH_THREAD(c,z,VDIM)
            {
               s_F[d1][d2][c] = F(d1,d2,c,f);
            }
         }
      }
      MFEM_SYNC_THREAD;

      if (eval_flags & VALUES)
      {
         auto Bu = (real_t (*)[max_ND1D][max_VDIM])sm2;
         MFEM_FOREACH_THREAD(d2,x,ND1D)
         {
            MFEM_FOREACH_THREAD(q1,y,NQ1D)
            {
               MFEM_FOREACH_THREAD(c,z,VDIM)
               {
                  real_t thrdBu = 0.0;
                  for (int d1 = 0; d1 < ND1D; ++d1)
                  {
                     thrdBu += B(q1,d1)*s_F[d1][d2][c];
                  }
                  Bu[q1][d2][c] = thrdBu;
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(q2,x,NQ1D)
         {
            MFEM_FOREACH_THREAD(q1,y,NQ1D)
            {
               MFEM_FOREACH_THREAD(c,z,VDIM)
               {
                  real_t v = 0.0;
                  for (int d2 = 0; d2 < ND1D; ++d2)
                  {
                     v += B(q2,d2)*Bu[q1][d2][c];
                  }
                  if (q_layout == QVectorLayout::byVDIM)  { val(c,q1,q2,f) = v; }
                  if (q_layout == QVectorLayout::byNODES) { val(q1,q2,c,f) = v; }
               }
            }
         }
      }

      if ((eval_flags & DERIVATIVES)
          || (eval_flags & DETERMINANTS)
          || (eval_flags & NORMALS))
      {
         // We only compute the tangential derivatives
         auto Gu = (real_t (*)[max_ND1D][max_VDIM])sm2;
         MFEM_SHARED real_t Bu[max_NQ1D][max_ND1D][max_VDIM];
         MFEM_FOREACH_THREAD(d2,x,ND1D)
         {
            MFEM_FOREACH_THREAD(q1,y,NQ1D)
            {
               MFEM_FOREACH_THREAD(c,z,VDIM)
               {
                  real_t thrdGu = 0;
                  real_t thrdBu = 0;
                  for (int d1 = 0; d1 < ND1D; ++d1)
                  {
                     const real_t u = s_F[d1][d2][c];
                     thrdBu += B(q1,d1)*u;
                     thrdGu += G(q1,d1)*u;
                  }
                  Gu[q1][d2][c] = thrdGu;
                  Bu[q1][d2][c] = thrdBu;
               }
            }
         }
         MFEM_SYNC_THREAD;

         auto BGu = (real_t (*)[max_NQ1D][max_VDIM])sm1;
         MFEM_SHARED real_t GBu[max_NQ1D][max_NQ1D][max_VDIM];
         MFEM_FOREACH_THREAD(q2,x,NQ1D)
         {
            MFEM_FOREACH_THREAD(q1,y,NQ1D)
            {
               MFEM_FOREACH_THREAD(c,z,VDIM)
               {
                  real_t thrdBGu = 0.0;
                  real_t thrdGBu = 0.0;
                  for (int d2 = 0; d2 < ND1D; ++d2)
                  {
                     thrdBGu += B(q2,d2)*Gu[q1][d2][c];
                     thrdGBu += G(q2,d2)*Bu[q1][d2][c];
                  }
                  BGu[q2][q1][c] = thrdBGu;
                  GBu[q2][q1][c] = thrdGBu;
               }
            }
         }
         MFEM_SYNC_THREAD;

         if (eval_flags & DERIVATIVES)
         {
            MFEM_FOREACH_THREAD(q2,x,NQ1D)
            {
               MFEM_FOREACH_THREAD(q1,y,NQ1D)
               {
                  MFEM_FOREACH_THREAD(c,z,VDIM)
                  {
                     if (q_layout == QVectorLayout::byVDIM)
                     {
                        der(c,0,q1,q2,f) = BGu[q2][q1][c];
                        der(c,1,q1,q2,f) = GBu[q2][q1][c];
                     }
                     else // q_layout == QVectorLayout::byNODES
                     {
                        der(q1,q2,c,0,f) = BGu[q2][q1][c];
                        der(q1,q2,c,1,f) = GBu[q2][q1][c];
                     }
                  }
               }
            }
         }

         if (VDIM == 3 && ((eval_flags & NORMALS) ||
                           (eval_flags & DETERMINANTS)))
         {
            real_t n[3];
            MFEM_FOREACH_THREAD(q2,x,NQ1D)
            {
               MFEM_FOREACH_THREAD(q1,y,NQ1D)
               {
                  if (MFEM_THREAD_ID(z) == 0)
                  {
                     const real_t s = sign[f] ? -1.0 : 1.0;
                     n[0] = s*( BGu[q2][q1][1]*GBu[q2][q1][2]-GBu[q2][q1][1]*
                                BGu[q2][q1][2] );
                     n[1] = s*(-BGu[q2][q1][0]*GBu[q2][q1][2]+GBu[q2][q1][0]*
                               BGu[q2][q1][2] );
                     n[2] = s*( BGu[q2][q1][0]*GBu[q2][q1][1]-GBu[q2][q1][0]*
                                BGu[q2][q1][1] );

                     const real_t norm = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);

                     if (eval_flags & DETERMINANTS) { det(q1,q2,f) = norm; }

                     if (eval_flags & NORMALS)
                     {
                        if (q_layout == QVectorLayout::byVDIM)
                        {
                           nor(0,q1,q2,f) = n[0]/norm;
                           nor(1,q1,q2,f) = n[1]/norm;
                           nor(2,q1,q2,f) = n[2]/norm;
                        }
                        if (q_layout == QVectorLayout::byNODES)
                        {
                           nor(q1,q2,0,f) = n[0]/norm;
                           nor(q1,q2,1,f) = n[1]/norm;
                           nor(q1,q2,2,f) = n[2]/norm;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

void FaceQuadratureInterpolator::Mult(
   const Vector &e_vec, unsigned eval_flags,
   Vector &q_val, Vector &q_der, Vector &q_det, Vector &q_nor) const
{
   if (nf == 0) { return; }
   const int vdim = fespace->GetVDim();
   const int dim = fespace->GetMesh()->Dimension();
   const FiniteElement *fe = fespace->GetTypicalTraceElement();
   const IntegrationRule *ir = IntRule;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int nd1d = maps.ndof;
   const int nq1d = maps.nqpt;
   void (*eval_func)(
      const int NF,
      const int vdim,
      const QVectorLayout q_layout,
      const DofToQuad &maps,
      const Array<bool> &signs,
      const Vector &e_vec,
      Vector &q_val,
      Vector &q_der,
      Vector &q_det,
      Vector &q_nor,
      const int eval_flags) = NULL;
   if (vdim == 1)
   {
      if (dim == 2)
      {
         switch (10*nd1d + nq1d)
         {
            // Q0
            case 11: eval_func = &Eval2D<1,1,1>; break;
            case 12: eval_func = &Eval2D<1,1,2>; break;
            // Q1
            case 22: eval_func = &Eval2D<1,2,2>; break;
            case 23: eval_func = &Eval2D<1,2,3>; break;
            // Q2
            case 33: eval_func = &Eval2D<1,3,3>; break;
            case 34: eval_func = &Eval2D<1,3,4>; break;
            // Q3
            case 44: eval_func = &Eval2D<1,4,4>; break;
            case 45: eval_func = &Eval2D<1,4,5>; break;
            case 46: eval_func = &Eval2D<1,4,6>; break;
            // Q4
            case 55: eval_func = &Eval2D<1,5,5>; break;
            case 56: eval_func = &Eval2D<1,5,6>; break;
            case 57: eval_func = &Eval2D<1,5,7>; break;
            case 58: eval_func = &Eval2D<1,5,8>; break;
         }
         if (nq1d >= 10 || !eval_func)
         {
            eval_func = &Eval2D<1>;
         }
      }
      else if (dim == 3)
      {
         switch (10*nd1d + nq1d)
         {
            // Q0
            case 11: eval_func = &SmemEval3D<1,1,1>; break;
            case 12: eval_func = &SmemEval3D<1,1,2>; break;
            // Q1
            case 22: eval_func = &SmemEval3D<1,2,2>; break;
            case 23: eval_func = &SmemEval3D<1,2,3>; break;
            case 24: eval_func = &SmemEval3D<1,2,4>; break;
            // Q2
            case 33: eval_func = &SmemEval3D<1,3,3>; break;
            case 34: eval_func = &SmemEval3D<1,3,4>; break;
            // Q3
            case 44: eval_func = &SmemEval3D<1,4,4>; break;
            case 45: eval_func = &SmemEval3D<1,4,5>; break;
            case 46: eval_func = &SmemEval3D<1,4,6>; break;
            // Q4
            case 55: eval_func = &SmemEval3D<1,5,5>; break;
            case 56: eval_func = &SmemEval3D<1,5,6>; break;
         }
         if (nq1d >= 10 || !eval_func)
         {
            eval_func = &Eval3D<1>;
         }
      }
   }
   else if (vdim == dim)
   {
      if (dim == 2)
      {
         switch (10*nd1d + nq1d)
         {
            // Q1
            case 22: eval_func = &Eval2D<2,2,2>; break;
            case 23: eval_func = &Eval2D<2,2,3>; break;
            // Q2
            case 33: eval_func = &Eval2D<2,3,3>; break;
            case 34: eval_func = &Eval2D<2,3,4>; break;
            // Q3
            case 44: eval_func = &Eval2D<2,4,4>; break;
            case 45: eval_func = &Eval2D<2,4,5>; break;
            case 46: eval_func = &Eval2D<2,4,6>; break;
            // Q4
            case 55: eval_func = &Eval2D<2,5,5>; break;
            case 56: eval_func = &Eval2D<2,5,6>; break;
            case 57: eval_func = &Eval2D<2,5,7>; break;
            case 58: eval_func = &Eval2D<2,5,8>; break;
         }
         if (nq1d >= 10 || !eval_func)
         {
            eval_func = &Eval2D<2>;
         }
      }
      else if (dim == 3)
      {
         switch (10*nd1d + nq1d)
         {
            // Q1
            case 22: eval_func = &SmemEval3D<3,2,2>; break;
            case 23: eval_func = &SmemEval3D<3,2,3>; break;
            case 24: eval_func = &SmemEval3D<3,2,4>; break;
            // Q2
            case 33: eval_func = &SmemEval3D<3,3,3>; break;
            case 34: eval_func = &SmemEval3D<3,3,4>; break;
            // Q3
            case 44: eval_func = &SmemEval3D<3,4,4>; break;
            case 45: eval_func = &SmemEval3D<3,4,5>; break;
            case 46: eval_func = &SmemEval3D<3,4,6>; break;
            // Q4
            case 55: eval_func = &SmemEval3D<3,5,5>; break;
            case 56: eval_func = &SmemEval3D<3,5,6>; break;
         }
         if (nq1d >= 10 || !eval_func)
         {
            eval_func = &Eval3D<3>;
         }
      }
   }
   if (eval_func)
   {
      eval_func(nf, vdim, q_layout, maps, signs, e_vec,
                q_val, q_der, q_det, q_nor, eval_flags);
   }
   else
   {
      MFEM_ABORT("case not supported yet");
   }
}

void FaceQuadratureInterpolator::Values(
   const Vector &e_vec, Vector &q_val) const
{
   Vector q_der, q_det, q_nor;
   Mult(e_vec, VALUES, q_val, q_der, q_det, q_nor);
}

} // namespace mfem
