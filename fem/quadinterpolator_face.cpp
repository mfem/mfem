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

#include "quadinterpolator_face.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

/// Return the sign to apply to the normals on each face to point from e1 to e2.
static void GetSigns(const FiniteElementSpace &fes, const FaceType type,
                     Array<bool> &signs)
{
   const int dim = fes.GetMesh()->SpaceDimension();
   int e1, e2;
   int inf1, inf2;
   int face_id;
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      face_id = inf1 / 64;
      if ( (type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
           (type==FaceType::Boundary && e2<0 && inf2<0) )
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
   : type(type_), nf(fes.GetNFbyType(type)), signs(nf)
{
   fespace = &fes;
   IntRule = &ir;
   use_tensor_products = true; // not implemented yet (not used)

   if (fespace->GetNE() == 0) { return; }
   GetSigns(*fespace, type, signs);
   const FiniteElement *fe = fespace->GetFE(0);
   const ScalarFiniteElement *sfe =
      dynamic_cast<const ScalarFiniteElement*>(fe);
   const TensorBasisElement *tfe =
      dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_VERIFY(sfe != NULL, "Only scalar finite elements are supported");
   MFEM_VERIFY(tfe != NULL &&
               (tfe->GetBasisType()==BasisType::GaussLobatto ||
                tfe->GetBasisType()==BasisType::Positive),
               "Only Gauss-Lobatto and Bernstein basis are supported in "
               "FaceQuadratureInterpolator.");
}

template<const int T_VDIM, const int T_ND1D, const int T_NQ1D>
void FaceQuadratureInterpolator::Eval2D(
   const int NF,
   const int vdim,
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &f_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 2 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(f_vec.Read(), ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = Reshape(q_val.Write(), NQ1D, VDIM, NF);
   // auto der = Reshape(q_der.Write(), NQ1D, VDIM, NF); // only tangential der
   auto det = Reshape(q_det.Write(), NQ1D, NF);
   auto n   = Reshape(q_nor.Write(), NQ1D, VDIM, NF);
   MFEM_VERIFY(eval_flags | DERIVATIVES,
               "Derivatives on the faces are not yet supported.");
   // If Gauss-Lobatto
   MFEM_FORALL(f, NF,
   {
      const int ND1D = T_ND1D ? T_ND1D : nd;
      const int NQ1D = T_NQ1D ? T_NQ1D : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM2D;
      double r_F[max_ND1D][max_VDIM];
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
            double ed[max_VDIM];
            for (int c = 0; c < VDIM; c++) { ed[c] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const double b = B(q,d);
               for (int c = 0; c < VDIM; c++) { ed[c] += b*r_F[d][c]; }
            }
            for (int c = 0; c < VDIM; c++) { val(q,c,f) = ed[c]; }
         }
         if ((eval_flags & DERIVATIVES)
             || (eval_flags & DETERMINANTS)
             || (eval_flags & NORMALS))
         {
            double D[max_VDIM];
            for (int i = 0; i < VDIM; i++) { D[i] = 0.0; }
            for (int d = 0; d < ND1D; ++d)
            {
               const double w = G(q,d);
               for (int c = 0; c < VDIM; c++)
               {
                  double s_e = r_F[d][c];
                  D[c] += s_e * w;
               }
            }
            if (VDIM == 2 &&
                ((eval_flags & NORMALS)
                 || (eval_flags & DETERMINANTS)))
            {
               const double norm = sqrt(D[0]*D[0]+D[1]*D[1]);
               if (eval_flags & DETERMINANTS)
               {
                  det(q,f) = norm;
               }
               if (eval_flags & NORMALS)
               {
                  const double s = sign[f] ? -1.0 : 1.0;
                  n(q,0,f) =  s*D[1]/norm;
                  n(q,1,f) = -s*D[0]/norm;
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
   const DofToQuad &maps,
   const Array<bool> &signs,
   const Vector &e_vec,
   Vector &q_val,
   Vector &q_der,
   Vector &q_det,
   Vector &q_nor,
   const int eval_flags)
{
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const int ND1D = T_ND1D ? T_ND1D : nd;
   const int NQ1D = T_NQ1D ? T_NQ1D : nq;
   const int VDIM = T_VDIM ? T_VDIM : vdim;
   MFEM_VERIFY(ND1D <= MAX_ND1D, "");
   MFEM_VERIFY(NQ1D <= MAX_NQ1D, "");
   MFEM_VERIFY(VDIM == 3 || !(eval_flags & DETERMINANTS), "");
   auto B = Reshape(maps.B.Read(), NQ1D, ND1D);
   auto G = Reshape(maps.G.Read(), NQ1D, ND1D);
   auto F = Reshape(e_vec.Read(), ND1D, ND1D, VDIM, NF);
   auto sign = signs.Read();
   auto val = Reshape(q_val.Write(), NQ1D, NQ1D, VDIM, NF);
   // auto der = Reshape(q_der.Write(), NQ1D, VDIM, 3, NF);
   auto det = Reshape(q_det.Write(), NQ1D, NQ1D, NF);
   auto nor = Reshape(q_nor.Write(), NQ1D, NQ1D, 3, NF);
   MFEM_VERIFY(eval_flags | DERIVATIVES,
               "Derivatives on the faces are not yet supported.");
   MFEM_FORALL(f, NF,
   {
      const int ND1D = T_ND1D ? T_ND1D : nd;
      const int NQ1D = T_NQ1D ? T_NQ1D : nq;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int max_ND1D = T_ND1D ? T_ND1D : MAX_ND1D;
      constexpr int max_NQ1D = T_NQ1D ? T_NQ1D : MAX_NQ1D;
      constexpr int max_VDIM = T_VDIM ? T_VDIM : MAX_VDIM3D;
      double r_F[max_ND1D][max_ND1D][max_VDIM];
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
         double Bu[max_NQ1D][max_ND1D][VDIM];
         for (int d2 = 0; d2 < ND1D; ++d2)
         {
            for (int q = 0; q < NQ1D; ++q)
            {
               for (int c = 0; c < VDIM; c++) { Bu[q][d2][c] = 0.0; }
               for (int d1 = 0; d1 < ND1D; ++d1)
               {
                  const double b = B(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     Bu[q][d2][c] += b*r_F[d1][d2][c];
                  }
               }
            }
         }
         double BBu[max_NQ1D][max_NQ1D][VDIM];
         for (int q2 = 0; q2 < NQ1D; ++q2)
         {
            for (int q1 = 0; q1 < NQ1D; ++q1)
            {
               for (int c = 0; c < VDIM; c++) { BBu[q2][q1][c] = 0.0; }
               for (int d2 = 0; d2 < ND1D; ++d2)
               {
                  const double b = B(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BBu[q2][q1][c] += b*Bu[q1][d2][c];
                  }
               }
               for (int c = 0; c < VDIM; c++)
               {
                  val(q1,q2,c,f) = BBu[q2][q1][c];
               }
            }
         }
      }
      if ((eval_flags & DERIVATIVES)
          || (eval_flags & DETERMINANTS)
          || (eval_flags & NORMALS))
      {
         // We only compute the tangential derivatives
         double Gu[max_NQ1D][max_ND1D][VDIM];
         double Bu[max_NQ1D][max_ND1D][VDIM];
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
                  const double b = B(q,d1);
                  const double g = G(q,d1);
                  for (int c = 0; c < VDIM; c++)
                  {
                     const double u = r_F[d1][d2][c];
                     Gu[q][d2][c] += g*u;
                     Bu[q][d2][c] += b*u;
                  }
               }
            }
         }
         double BGu[max_NQ1D][max_NQ1D][VDIM];
         double GBu[max_NQ1D][max_NQ1D][VDIM];
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
                  const double b = B(q2,d2);
                  const double g = G(q2,d2);
                  for (int c = 0; c < VDIM; c++)
                  {
                     BGu[q2][q1][c] += b*Gu[q1][d2][c];
                     GBu[q2][q1][c] += g*Bu[q1][d2][c];
                  }
               }
            }
         }
         if (VDIM == 3 && ((eval_flags & NORMALS) ||
                           (eval_flags & DETERMINANTS)))
         {
            double n[3];
            for (int q2 = 0; q2 < NQ1D; ++q2)
            {
               for (int q1 = 0; q1 < NQ1D; ++q1)
               {
                  const double s = sign[f] ? -1.0 : 1.0;
                  n[0] = s*( BGu[q2][q1][1]*GBu[q2][q1][2]-GBu[q2][q1][1]*
                             BGu[q2][q1][2] );
                  n[1] = s*(-BGu[q2][q1][0]*GBu[q2][q1][2]+GBu[q2][q1][0]*
                            BGu[q2][q1][2] );
                  n[2] = s*( BGu[q2][q1][0]*GBu[q2][q1][1]-GBu[q2][q1][0]*
                             BGu[q2][q1][1] );
                  const double norm = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
                  if (eval_flags & DETERMINANTS) { det(q1,q2,f) = norm; }
                  if (eval_flags & NORMALS)
                  {
                     nor(q1,q2,0,f) = n[0]/norm;
                     nor(q1,q2,1,f) = n[1]/norm;
                     nor(q1,q2,2,f) = n[2]/norm;
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
   const FiniteElement *fe =
      fespace->GetTraceElement(0, fespace->GetMesh()->GetFaceBaseGeometry(0));
   const IntegrationRule *ir = IntRule;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   void (*eval_func)(
      const int NF,
      const int vdim,
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
         switch (100*nd + nq)
         {
            // Q0
            case 101: eval_func = &Eval2D<1,1,1>; break;
            case 104: eval_func = &Eval2D<1,1,4>; break;
            // Q1
            case 404: eval_func = &Eval2D<1,4,4>; break;
            case 409: eval_func = &Eval2D<1,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<1,9,9>; break;
            case 916: eval_func = &Eval2D<1,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<1,16,16>; break;
            case 1625: eval_func = &Eval2D<1,16,25>; break;
            case 1636: eval_func = &Eval2D<1,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<1,25,25>; break;
            case 2536: eval_func = &Eval2D<1,25,36>; break;
            case 2549: eval_func = &Eval2D<1,25,49>; break;
            case 2564: eval_func = &Eval2D<1,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<1>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q0
            case 1001: eval_func = &Eval3D<1,1,1>; break;
            case 1008: eval_func = &Eval3D<1,1,8>; break;
            // Q1
            case 8008: eval_func = &Eval3D<1,8,8>; break;
            case 8027: eval_func = &Eval3D<1,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<1,27,27>; break;
            case 27064: eval_func = &Eval3D<1,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<1,64,64>; break;
            case 64125: eval_func = &Eval3D<1,64,125>; break;
            case 64216: eval_func = &Eval3D<1,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<1,125,125>; break;
            case 125216: eval_func = &Eval3D<1,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<1>;
         }
      }
   }
   else if (vdim == dim)
   {
      if (dim == 2)
      {
         switch (100*nd + nq)
         {
            // Q1
            case 404: eval_func = &Eval2D<2,4,4>; break;
            case 409: eval_func = &Eval2D<2,4,9>; break;
            // Q2
            case 909: eval_func = &Eval2D<2,9,9>; break;
            case 916: eval_func = &Eval2D<2,9,16>; break;
            // Q3
            case 1616: eval_func = &Eval2D<2,16,16>; break;
            case 1625: eval_func = &Eval2D<2,16,25>; break;
            case 1636: eval_func = &Eval2D<2,16,36>; break;
            // Q4
            case 2525: eval_func = &Eval2D<2,25,25>; break;
            case 2536: eval_func = &Eval2D<2,25,36>; break;
            case 2549: eval_func = &Eval2D<2,25,49>; break;
            case 2564: eval_func = &Eval2D<2,25,64>; break;
         }
         if (nq >= 100 || !eval_func)
         {
            eval_func = &Eval2D<2>;
         }
      }
      else if (dim == 3)
      {
         switch (1000*nd + nq)
         {
            // Q1
            case 8008: eval_func = &Eval3D<3,8,8>; break;
            case 8027: eval_func = &Eval3D<3,8,27>; break;
            // Q2
            case 27027: eval_func = &Eval3D<3,27,27>; break;
            case 27064: eval_func = &Eval3D<3,27,64>; break;
            // Q3
            case 64064: eval_func = &Eval3D<3,64,64>; break;
            case 64125: eval_func = &Eval3D<3,64,125>; break;
            case 64216: eval_func = &Eval3D<3,64,216>; break;
            // Q4
            case 125125: eval_func = &Eval3D<3,125,125>; break;
            case 125216: eval_func = &Eval3D<3,125,216>; break;
         }
         if (nq >= 1000 || !eval_func)
         {
            eval_func = &Eval3D<3>;
         }
      }
   }
   if (eval_func)
   {
      eval_func(nf, vdim, maps, signs, e_vec,
                q_val, q_der, q_det, q_nor, eval_flags);
   }
   else
   {
      MFEM_ABORT("case not supported yet");
   }
}

} // namespace mfem
