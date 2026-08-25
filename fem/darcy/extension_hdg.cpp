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
   Vector m(dim), y(dim), v(dim);

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

real_t PathTraceCoefficient::Eval(ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   FaceElementTransformations *FTr =
      dynamic_cast<FaceElementTransformations *>(&T);
   MFEM_VERIFY(FTr, "PathTraceCoefficient must be evaluated on a face: the "
               "path family may need the outward normal of Gamma_h");

   path.Endpoint(*FTr, ip, xbar);
   return g(xbar);
}

void HDGExtensionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   MFEM_VERIFY(Trans.Elem2No < 0,
               "the extension term lives on a boundary face only");

   const int dof = el1.GetDof();
   const int dim = Trans.GetSpaceDim();

   elmat.SetSize(dof * dim);
   elmat = 0.;

   shape.SetSize(dof);
   shape_ext.SetSize(dof);
   nor.SetSize(dim);
   m.SetSize(dim);
   y.SetSize(dim);
   CTm.SetSize(dim);
   L.SetSize(dof, dim);
   if (MC) { Cmat.SetSize(dim); }

   const int order = (line_order >= 0) ? line_order : (2 * el1.GetOrder() + 2);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &IntRules.Get(Trans.GetGeometryType(), 2 * el1.GetOrder() + 2);
   }
   const IntegrationRule &lir = IntRules.Get(Geometry::SEGMENT, order);

   ext.SetElement(*Trans.Elem1);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Trans.SetAllIntPoints(&ip);

      // The scaled normal already carries the face Jacobian, so the face
      // weight is the quadrature weight alone.
      if (dim == 1)
      {
         nor(0) = 2 * Trans.GetElement1IntPoint().x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcPhysShape(*Trans.Elem1, shape);
      Trans.Transform(ip, x);

      // Endpoint() resets the transformations, which is why the shape and the
      // normal above are taken into arrays of our own first.
      path.Endpoint(Trans, ip, xbar);
      subtract(xbar, x, m);

      // The lifting of every basis function of the element, at this point of
      // the face.  The unit tangent is m/|m| and ds = |m| dt, so the length
      // of the path cancels and never appears.
      L = 0.;
      for (int t = 0; t < lir.GetNPoints(); t++)
      {
         const IntegrationPoint &tip = lir.IntPoint(t);
         for (int d = 0; d < dim; d++) { y(d) = x(d) + tip.x * m(d); }

         IntegrationPoint eip;
         MFEM_VERIFY(ext.TransformBack(y, eip),
                     "the inverse element transformation did not converge on "
                     "the extension of the element beyond Gamma_h");
         Trans.Elem1->SetIntPoint(&eip);
         el1.CalcPhysShape(*Trans.Elem1, shape_ext);

         // (C phi_j) . m = phi_j . (C^T m), and phi_j = shape_j e_d.
         if (MC)
         {
            MC->Eval(Cmat, *Trans.Elem1, eip);
            Cmat.MultTranspose(m, CTm);
         }
         else
         {
            CTm = m;
            CTm *= C->Eval(*Trans.Elem1, eip);
         }

         for (int d = 0; d < dim; d++)
         {
            const real_t wd = tip.weight * CTm(d);
            for (int j = 0; j < dof; j++)
            {
               L(j, d) += wd * shape_ext(j);
            }
         }
      }

      const real_t w = sign * ip.weight;
      for (int di = 0; di < dim; di++)
         for (int i = 0; i < dof; i++)
         {
            const real_t ti = w * nor(di) * shape(i);
            for (int dj = 0; dj < dim; dj++)
               for (int j = 0; j < dof; j++)
               {
                  elmat(dof * di + i, dof * dj + j) += ti * L(j, dj);
               }
         }
   }
}

void ExtensionRegionQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir, const IntegrationRule &line_ir,
   const std::function<void(const ExtensionPoint &)> &visit, real_t fd_step)
{
   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;

   Vector x(sdim), xbar(sdim), m(sdim), y(sdim);
   Vector xbar_p(sdim), xbar_m(sdim), da(sdim);
   DenseMatrix J(sdim), dxdxi(sdim, std::max(fdim, 1));
   DenseMatrix dadxi(sdim, std::max(fdim, 1));

   for (int q = 0; q < face_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = face_ir.IntPoint(q);

      path.Endpoint(FTr, ip, xbar);
      FTr.SetAllIntPoints(&ip);
      FTr.Transform(ip, x);
      subtract(xbar, x, m);

      // The face's own Jacobian, and the derivative of the path's endpoint
      // along the face.  A point face -- the boundary of a one-dimensional
      // mesh -- has neither, and the region is the path itself.
      if (fdim > 0) { dxdxi = FTr.Jacobian(); }

      for (int i = 0; i < fdim; i++)
      {
         IntegrationPoint ip_p = ip, ip_m = ip;
         real_t *cp = (i == 0) ? &ip_p.x : ((i == 1) ? &ip_p.y : &ip_p.z);
         real_t *cm = (i == 0) ? &ip_m.x : ((i == 1) ? &ip_m.y : &ip_m.z);
         *cp += fd_step;
         *cm -= fd_step;

         path.Endpoint(FTr, ip_p, xbar_p);
         path.Endpoint(FTr, ip_m, xbar_m);
         subtract(xbar_p, xbar_m, da);
         da /= 2. * fd_step;
         for (int d = 0; d < sdim; d++) { dadxi(d, i) = da(d); }
      }
      // Endpoint() moved the transformations; put them back.
      FTr.SetAllIntPoints(&ip);

      for (int t = 0; t < line_ir.GetNPoints(); t++)
      {
         const IntegrationPoint &tip = line_ir.IntPoint(t);

         for (int d = 0; d < sdim; d++) { y(d) = x(d) + tip.x * m(d); }

         // dy/dxi_i = (1-t) dx/dxi_i + t da/dxi_i, and dy/dt = m.
         for (int i = 0; i < fdim; i++)
            for (int d = 0; d < sdim; d++)
            {
               J(d, i) = (1. - tip.x) * dxdxi(d, i) + tip.x * dadxi(d, i);
            }
         for (int d = 0; d < sdim; d++) { J(d, sdim - 1) = m(d); }

         const ExtensionPoint pt{y, xbar, ip, tip.x,
                                 ip.weight * tip.weight * std::abs(J.Det())};
         visit(pt);
      }
   }
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
