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

#include "compute_bdr_normal.hpp"
#include "../../general/forall.hpp"

using namespace mfem;
using namespace navier;

BoundaryNormalEvaluator::BoundaryNormalEvaluator(
   ParFiniteElementSpace &vfes_,
   ParFiniteElementSpace &pfes_,
   const IntegrationRule &ir_)
   : vfes(vfes_), pfes(pfes_), ir(ir_), g_gf(&vfes), y_gf(&pfes)
{
   ParMesh &mesh = *vfes.GetParMesh();
   mesh.ExchangeFaceNbrData();
   geom = mesh.GetFaceGeometricFactors(
             ir,
             FaceGeometricFactors::DETERMINANTS | FaceGeometricFactors::NORMALS,
             FaceType::Boundary);
   const FiniteElement &el =
      *vfes.GetTraceElement(0, mesh.GetFaceBaseGeometry(0));
   maps = &el.GetDofToQuad(ir, DofToQuad::TENSOR);
}

void BoundaryNormalEvaluator::Mult(const Vector &g, Vector &y) const
{
   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *v_face_restr =
      vfes.GetFaceRestriction(ordering, FaceType::Boundary);
   const Operator *p_face_restr =
      pfes.GetFaceRestriction(ordering, FaceType::Boundary);

   g_face.SetSize(v_face_restr->Height());
   y_face.SetSize(p_face_restr->Height());

   g_gf.Distribute(g);
   v_face_restr->Mult(g_gf, g_face);

   int dim = vfes.GetMesh()->Dimension();
   const int nf = vfes.GetNFbyType(FaceType::Boundary);

   // Special case to work around bug in restriction when there are no faces
   // (MultTranspose does not work properly)
   if (nf == 0)
   {
      y = 0.0;
      return;
   }

   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;

   auto B = Reshape(maps->B.Read(), Q1D, D1D);
   auto Bt = Reshape(maps->Bt.Read(), D1D, Q1D);

   if (dim == 2)
   {
      const auto d_det = Reshape(geom->detJ.Read(), Q1D, nf);
      const auto d_n = Reshape(geom->normal.Read(), Q1D, 2, nf);
      const auto d_w = Reshape(ir.GetWeights().Read(), Q1D);

      const auto d_g = Reshape(g_face.Read(), D1D, 2, nf);
      const auto d_y = Reshape(y_face.Write(), D1D, nf);

      MFEM_FORALL(i, nf,
      {
         double g_q[MAX_Q1D];
         // Get g.n at quads
         for (int q = 0; q < Q1D; ++q)
         {
            g_q[q] = 0.0;
         }
         for (int q = 0; q < Q1D; ++q)
         {
            double det_w = d_det(q, i)*d_w(q);
            double n0 = det_w*d_n(q, 0, i);
            double n1 = det_w*d_n(q, 1, i);
            for (int d = 0; d < D1D; ++d)
            {
               double b = B(q, d);
               g_q[q] += b*d_g(d, 0, i)*n0 + b*d_g(d, 1, i)*n1;
            }
         }
         // Get y at dofs
         for (int d = 0; d < D1D; ++d)
         {
            d_y(d, i) = 0.0;
         }
         for (int q = 0; q < Q1D; ++q)
         {
            for (int d = 0; d < D1D; ++d)
            {
               d_y(d, i) += Bt(d, q)*g_q[q];
            }
         }
      });
   }
   else if (dim == 3)
   {
      const auto d_det = Reshape(geom->detJ.Read(), Q1D, Q1D, nf);
      const auto d_n = Reshape(geom->normal.Read(), Q1D, Q1D, 3, nf);
      const auto d_w = Reshape(ir.GetWeights().Read(), Q1D, Q1D);

      const auto d_g = Reshape(g_face.Read(), D1D, D1D, 3, nf);
      const auto d_y = Reshape(y_face.Write(), D1D, D1D, nf);

      MFEM_FORALL(i, nf,
      {
         double g_q[MAX_Q1D][MAX_Q1D];
         // Get g.n at quads
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               g_q[qy][qx] = 0.0;
            }
         }

         for (int dy = 0; dy < D1D; ++dy)
         {
            double g_qx[MAX_Q1D][3];
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int d = 0; d < 3; ++ d)
               {
                  g_qx[qy][d] = 0.0;
               }
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double g_0 = d_g(dx, dy, 0, i);
               const double g_1 = d_g(dx, dy, 1, i);
               const double g_2 = d_g(dx, dy, 2, i);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  double b = B(qx, dx);
                  g_qx[qx][0] += b*g_0;
                  g_qx[qx][1] += b*g_1;
                  g_qx[qx][2] += b*g_2;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double b = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  double det_w = d_det(qx,qy,i)*d_w(qx,qy);
                  double n0 = det_w*d_n(qx,qy,0,i);
                  double n1 = det_w*d_n(qx,qy,1,i);
                  double n2 = det_w*d_n(qx,qy,2,i);

                  g_q[qy][qx] += b*(n0*g_qx[qx][0] + n1*g_qx[qx][1] + n2*g_qx[qx][2]);
               }
            }
         }

         // Get y at dofs
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               d_y(dx, dy, i) = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double y_x[MAX_D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               y_x[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double g_qx_qy = g_q[qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y_x[dx] += Bt(dx,qx)*g_qx_qy;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double b = Bt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  d_y(dx,dy,i) += b * y_x[dx];
               }
            }
         }
      });
   }

   p_face_restr->MultTranspose(y_face, y_gf);
   y_gf.ParallelAssemble(y);
}
