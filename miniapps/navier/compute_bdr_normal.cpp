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

   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;

   if (dim == 2)
   {
      auto B = Reshape(maps->B.Read(), Q1D, D1D);
      auto Bt = Reshape(maps->Bt.Read(), D1D, Q1D);

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
      MFEM_ABORT("Not implemented");
   }

   p_face_restr->MultTranspose(y_face, y_gf);
   y_gf.ParallelAssemble(y);
}
