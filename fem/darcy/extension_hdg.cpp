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

#include "extension_hdg.hpp"
#include "../geom.hpp"

namespace mfem
{

void TransferPath::Endpoint(FaceElementTransformations &FTr,
                            const IntegrationPoint &ip, Vector &xbar) const
{
   const int sdim = FTr.GetSpaceDim();
   Vector x(sdim), nor(sdim);

   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   if (sdim == 1)
   {
      nor(0) = (ip.x > 0.5)?(1.):(-1.);
   }
   else
   {
      CalcOrtho(FTr.Jacobian(), nor);
      nor /= nor.Norml2();
   }

   Endpoint(x, nor, xbar);
}

VectorPositionFunction ClosestPointPath::Sphere(const Vector &c, real_t R)
{
   const Vector cc(c);
   return [cc, R](const Vector &x, Vector &xbar)
   {
      const int dim = x.Size();
      MFEM_ASSERT(cc.Size() == dim, "dimension mismatch");
      xbar.SetSize(dim);

      real_t r = 0.;
      for (int d = 0; d < dim; d++)
      {
         const real_t s = x(d) - cc(d);
         r += s * s;
      }
      r = sqrt(r);
      MFEM_ASSERT(r > 0., "the centre of the sphere has no closest point on it");

      for (int d = 0; d < dim; d++)
      {
         xbar(d) = cc(d) + R * (x(d) - cc(d)) / r;
      }
   };
}

void LevelSetPath::Endpoint(const Vector &x, const Vector &n,
                            Vector &xbar) const
{
   const int dim = x.Size();
   xbar.SetSize(dim);

   Vector y(dim);
   auto phi_at = [&](real_t t) -> real_t
   {
      for (int d = 0; d < dim; d++) { y(d) = x(d) + t * n(d); }
      return phi(y);
   };

   MFEM_VERIFY(phi_at(0.) <= 0., "the path starts outside the domain");

   // Bracket the crossing of the level set.
   real_t ta = 0., tb = -1.;
   const real_t dt = search_length / search_steps;
   for (int i = 1; i <= search_steps; i++)
   {
      const real_t t = i * dt;
      if (phi_at(t) > 0.) { tb = t; break; }
      ta = t;
   }
   MFEM_VERIFY(tb > 0., "no crossing of the level set within the search length "
               << search_length << ": either it is too short, or the outward "
               "normal does not leave the domain");

   // Bisect it.
   const real_t ttol = tol * search_length;
   for (int it = 0; it < max_iter && tb - ta > ttol; it++)
   {
      const real_t tm = 0.5 * (ta + tb);
      if (phi_at(tm) > 0.) { tb = tm; }
      else { ta = tm; }
   }

   const real_t t = 0.5 * (ta + tb);
   for (int d = 0; d < dim; d++) { xbar(d) = x(d) + t * n(d); }
}

ElementExtension::ElementExtension()
{
   // The default #NewtonElementProject solver projects every iterate back into
   // the reference element, which is precisely what must not happen here.
   inv_tr.SetSolverType(InverseElementTransformation::Newton);
   inv_tr.SetInitialGuessType(InverseElementTransformation::Center);
}

bool ElementExtension::TransformBack(const Vector &y,
                                     IntegrationPoint &ip) const
{
   // The reference point is written before it is classified, so an #Outside
   // return still carries the out-of-element coordinates that are wanted.
   const int res = inv_tr.Transform(y, ip);
   return res != InverseElementTransformation::Unknown;
}

real_t PathIntegral(const VectorPositionFunction &Cu, const Vector &x,
                    const Vector &xbar, const IntegrationRule &line_ir)
{
   const int dim = x.Size();
   Vector m(dim), y(dim), v;

   // The unit tangent is m/|m| and ds = |m| dt, so the length cancels.
   subtract(xbar, x, m);

   real_t val = 0.;
   for (int q = 0; q < line_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &qip = line_ir.IntPoint(q);
      for (int d = 0; d < dim; d++) { y(d) = x(d) + qip.x * m(d); }
      Cu(y, v);
      val += qip.weight * (v * m);
   }

   return val;
}

int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker, int extra_refine)
{
   const int NE = mesh.GetNE();
   const int sdim = mesh.SpaceDimension();

   marker.SetSize(NE);
   marker = 0;

   Array<int> vert;
   Vector x(sdim);
   IsoparametricTransformation Tr;
   int count = 0;

   for (int i = 0; i < NE; i++)
   {
      bool inside = true;

      mesh.GetElementVertices(i, vert);
      for (int v = 0; v < vert.Size() && inside; v++)
      {
         const real_t *c = mesh.GetVertex(vert[v]);
         for (int d = 0; d < sdim; d++) { x(d) = c[d]; }
         if (phi(x) > -offset) { inside = false; }
      }

      if (inside && extra_refine >= 1)
      {
         RefinedGeometry *RefG =
            GlobGeometryRefiner.Refine(mesh.GetElementGeometry(i), extra_refine);
         mesh.GetElementTransformation(i, &Tr);
         for (int q = 0; q < RefG->RefPts.GetNPoints() && inside; q++)
         {
            Tr.Transform(RefG->RefPts.IntPoint(q), x);
            if (phi(x) > -offset) { inside = false; }
         }
      }

      if (inside) { marker[i] = 1; count++; }
   }

   return count;
}

} // namespace mfem
