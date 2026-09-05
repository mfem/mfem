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

#include <limits>
#include "../geom.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <vector>

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
      // The face is a point, whose own reference coordinate is always zero;
      // which end of the element it is has to be read from the element's.
      nor(0) = 2. * FTr.GetElement1IntPoint().x - 1.;
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

namespace
{

/** @brief March from @a x along the unit direction @a dir and bisect the first
    crossing of the zero level set, returning its parameter in @a t.

    Returns false when there is no crossing within @a length -- which is a real
    outcome and not an error: a ray can leave the domain without ever meeting
    the boundary, and on a shape with a thin feature many of them do. */
bool FirstCrossing(const PositionFunction &phi, const Vector &x,
                   const Vector &dir, real_t length, int steps, real_t tol,
                   int max_iter, real_t &t)
{
   const int dim = x.Size();
   Vector y(dim);
   auto phi_at = [&](real_t u) -> real_t
   {
      for (int d = 0; d < dim; d++) { y(d) = x(d) + u * dir(d); }
      return phi(y);
   };

   if (phi_at(0.) > 0.) { return false; }

   real_t ta = 0., tb = -1.;
   const real_t du = length / steps;
   for (int i = 1; i <= steps; i++)
   {
      const real_t u = i * du;
      if (phi_at(u) > 0.) { tb = u; break; }
      ta = u;
   }
   if (tb < 0.) { return false; }

   const real_t ttol = tol * length;
   for (int it = 0; it < max_iter && tb - ta > ttol; it++)
   {
      const real_t tm = 0.5 * (ta + tb);
      if (phi_at(tm) > 0.) { tb = tm; }
      else { ta = tm; }
   }

   t = 0.5 * (ta + tb);
   return true;
}

} // namespace

std::function<void(const Vector &, DenseMatrix &)>
ClosestPointPath::SphereJacobian(const Vector &c, real_t R)
{
   const Vector cc(c);
   return [cc, R](const Vector &x, DenseMatrix &J)
   {
      const int dim = x.Size();
      MFEM_ASSERT(cc.Size() == dim, "dimension mismatch");
      J.SetSize(dim);

      Vector u(dim);
      real_t r = 0.;
      for (int d = 0; d < dim; d++)
      {
         u(d) = x(d) - cc(d);
         r += u(d) * u(d);
      }
      r = std::sqrt(r);
      MFEM_VERIFY(r > 0., "the closest point to a sphere's own centre is not "
                  "defined, so neither is the Jacobian there");
      u /= r;

      // d a/d x = ( R/|x-c| )( I - u u^T ).  The projector annihilates the
      // radial direction because every point of a ray shares a closest point.
      const real_t s = R / r;
      for (int i = 0; i < dim; i++)
         for (int j = 0; j < dim; j++)
         {
            J(i, j) = s * ((i == j ? 1. : 0.) - u(i) * u(j));
         }
   };
}

bool ClosestPointPath::EndpointJacobian(FaceElementTransformations &FTr,
                                        const IntegrationPoint &ip,
                                        DenseMatrix &dadxi) const
{
   if (!dcp) { return false; }

   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;
   if (fdim < 1) { return false; }

   Vector x(sdim);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   // The chain rule: d a/d xi = ( d a/d x )( d x/d xi ), the second factor
   // being the face's own Jacobian.  Exact, where the difference is not.
   DenseMatrix dadx(sdim);
   dcp(x, dadx);

   dadxi.SetSize(sdim, fdim);
   Mult(dadx, FTr.Jacobian(), dadxi);
   return true;
}

void LevelSetPath::Endpoint(const Vector &x, const Vector &n,
                            Vector &xbar) const
{
   const int dim = x.Size();
   xbar.SetSize(dim);

   real_t t;
   MFEM_VERIFY(FirstCrossing(phi, x, n, search_length, search_steps, tol,
                             max_iter, t),
               "the ray along the outward normal never meets the level set "
               "within the search length " << search_length << ": either it is "
               "too short, the normal does not leave the domain, or the "
               "boundary has a feature the normal passes outside -- for which "
               "VertexConePath is the family to use");

   for (int d = 0; d < dim; d++) { xbar(d) = x(d) + t * n(d); }
}

VertexConePath::VertexConePath(const Mesh &mesh_, int gamma_h_attr,
                               PositionFunction phi_, real_t search_length_,
                               int n_rays_, int n_keep_, int search_steps_,
                               real_t tol_, int max_iter_)
   : mesh(&mesh_), phi(std::move(phi_)), search_length(search_length_),
     n_rays(n_rays_), n_keep(n_keep_), search_steps(search_steps_),
     max_iter(max_iter_), tol(tol_)
{
   MFEM_VERIFY(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2,
               "VertexConePath is a two-dimensional construction");
   MFEM_VERIFY(n_rays >= 2 && n_keep >= 1, "invalid search parameters");

   Mesh &m = const_cast<Mesh &>(*mesh);
   const int nfaces = m.GetNumFaces();

   tang.SetSize(4 * nfaces); tang = 0.;
   has_tangent.SetSize(nfaces); has_tangent = 0;

   // The outward normals of the faces of Gamma_h meeting at each vertex.
   std::map<int, Array<real_t>> vertex_normals;
   Array<int> vs;
   Vector nor(2);

   for (int be = 0; be < m.GetNBE(); be++)
   {
      if (m.GetBdrAttribute(be) != gamma_h_attr) { continue; }

      FaceElementTransformations *FTr = m.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }

      IntegrationPoint mid;
      mid.Set1w(0.5, 1.0);
      FTr->SetAllIntPoints(&mid);
      CalcOrtho(FTr->Jacobian(), nor);
      nor /= nor.Norml2();

      m.GetFaceVertices(m.GetBdrElementFaceIndex(be), vs);
      for (int k = 0; k < vs.Size(); k++)
      {
         Array<real_t> &ns = vertex_normals[vs[k]];
         ns.Append(nor(0));
         ns.Append(nor(1));
      }
   }

   // A direction at each of those vertices.
   std::map<int, std::array<real_t, 2>> vertex_tangent;
   Vector x(2), t(2);
   for (auto &kv : vertex_normals)
   {
      const real_t *c = m.GetVertex(kv.first);
      x(0) = c[0]; x(1) = c[1];
      MFEM_VERIFY(VertexDirection(x, kv.second, t),
                  "no ray from the vertex (" << x(0) << ", " << x(1)
                  << ") of Gamma_h reaches Gamma within the search length "
                  << search_length);
      vertex_tangent[kv.first] = {{t(0), t(1)}};
   }

   // Per face, those two tangents in the order its reference coordinate visits
   // the vertices. Which vertex sits at xi = 0 is read off the transformation
   // rather than assumed to follow the mesh's own ordering.
   IntegrationPoint zero;
   zero.Set1w(0.0, 1.0);
   for (int be = 0; be < m.GetNBE(); be++)
   {
      if (m.GetBdrAttribute(be) != gamma_h_attr) { continue; }

      FaceElementTransformations *FTr = m.GetBdrFaceTransformations(be);
      if (!FTr) { continue; }

      const int f = m.GetBdrElementFaceIndex(be);
      m.GetFaceVertices(f, vs);
      MFEM_VERIFY(vs.Size() == 2, "a face of a two-dimensional mesh has two "
                  "vertices");

      FTr->SetAllIntPoints(&zero);
      FTr->Transform(zero, x);

      const real_t *c0 = m.GetVertex(vs[0]);
      const real_t d0 = (x(0) - c0[0]) * (x(0) - c0[0])
                        + (x(1) - c0[1]) * (x(1) - c0[1]);
      const real_t *c1 = m.GetVertex(vs[1]);
      const real_t d1 = (x(0) - c1[0]) * (x(0) - c1[0])
                        + (x(1) - c1[1]) * (x(1) - c1[1]);

      const int at0 = (d0 <= d1) ? vs[0] : vs[1];
      const int at1 = (d0 <= d1) ? vs[1] : vs[0];

      tang[4 * f + 0] = vertex_tangent[at0][0];
      tang[4 * f + 1] = vertex_tangent[at0][1];
      tang[4 * f + 2] = vertex_tangent[at1][0];
      tang[4 * f + 3] = vertex_tangent[at1][1];
      has_tangent[f] = 1;
   }
}

bool VertexConePath::VertexDirection(const Vector &x,
                                     const Array<real_t> &normals, Vector &t)
{
   const int m = normals.Size() / 2;
   MFEM_VERIFY(m >= 1, "a vertex of Gamma_h with no face");

   // The admissible directions are those leaving D_h through every face
   // meeting here. In two dimensions each such condition is a half-circle, so
   // their intersection is the arc about the mean normal whose half-width is
   // pi/2 less the largest angle between the mean and any one of them.
   Vector mean(2);
   mean = 0.;
   for (int i = 0; i < m; i++)
   {
      mean(0) += normals[2 * i];
      mean(1) += normals[2 * i + 1];
   }
   const real_t mn = mean.Norml2();
   real_t centre_angle, half_width;
   if (mn > 1e-12)
   {
      mean /= mn;
      centre_angle = atan2(mean(1), mean(0));
      real_t worst = 0.;
      for (int i = 0; i < m; i++)
      {
         const real_t dot = mean(0) * normals[2 * i] + mean(1) * normals[2 * i + 1];
         worst = std::max(worst, acos(std::min(std::max(dot, -1.), 1.)));
      }
      half_width = M_PI / 2. - worst;
   }
   else
   {
      centre_angle = atan2(normals[1], normals[0]);
      half_width = -1.;
   }

   // Search the admissible fan, then the half-circle about the mean normal,
   // then everything. The widenings leave Assumption P.1 behind and are
   // counted so that a driver can say so.
   const real_t widths[3] = { half_width, M_PI / 2., M_PI };
   for (int pass = 0; pass < 3; pass++)
   {
      const real_t w = widths[pass];
      if (w <= 0.) { continue; }
      const real_t use = (pass == 0) ? 0.98 * w : 0.98 * w;

      // Endpoints found, sorted by distance, keeping the nearest few.
      std::vector<std::pair<real_t, std::array<real_t, 2>>> hits;
      Vector d(2);
      for (int i = 0; i < n_rays; i++)
      {
         const real_t a = centre_angle
                          + ((n_rays == 1) ? 0.
                             : (-use + 2. * use * i / (n_rays - 1)));
         d(0) = cos(a); d(1) = sin(a);
         real_t s;
         if (!FirstCrossing(phi, x, d, search_length, search_steps, tol,
                            max_iter, s)) { continue; }
         hits.push_back({s, {{x(0) + s * d(0), x(1) + s * d(1)}}});
      }
      if (hits.empty()) { continue; }

      if (pass > 0) { n_widened++; }

      std::sort(hits.begin(), hits.end(),
                [](const std::pair<real_t, std::array<real_t, 2>> &a,
                   const std::pair<real_t, std::array<real_t, 2>> &b)
      { return a.first < b.first; });

      const int keep = std::min<int>(n_keep, hits.size());
      real_t mx = 0., my = 0.;
      for (int i = 0; i < keep; i++)
      {
         mx += hits[i].second[0];
         my += hits[i].second[1];
      }
      mx /= keep; my /= keep;

      t.SetSize(2);
      t(0) = mx - x(0); t(1) = my - x(1);
      const real_t len = t.Norml2();
      if (len <= 0.) { continue; }
      t /= len;

      // One more shot along the mean, so that the endpoint lands on Gamma and
      // not merely near it.
      real_t s;
      if (!FirstCrossing(phi, x, t, search_length, search_steps, tol, max_iter,
                         s)) { continue; }
      return true;
   }

   return false;
}

void VertexConePath::Endpoint(const Vector &x, const Vector &n,
                              Vector &xbar) const
{
   MFEM_ABORT("VertexConePath is defined face by face: use the "
              "FaceElementTransformations overload. Falling back to the "
              "normal would reintroduce the two failures this family repairs.");
}

void VertexConePath::Endpoint(FaceElementTransformations &FTr,
                              const IntegrationPoint &ip, Vector &xbar) const
{
   const int f = (FTr.ElementType == ElementTransformation::BDR_FACE)
                 ? mesh->GetBdrElementFaceIndex(FTr.ElementNo)
                 : FTr.ElementNo;
   MFEM_VERIFY(f >= 0 && f < has_tangent.Size() && has_tangent[f],
               "no path was built for this face: it is not on the boundary "
               "the family was constructed for");

   const real_t th = ip.x;
   Vector t(2);
   t(0) = (1. - th) * tang[4 * f + 0] + th * tang[4 * f + 2];
   t(1) = (1. - th) * tang[4 * f + 1] + th * tang[4 * f + 3];
   const real_t len = t.Norml2();
   MFEM_VERIFY(len > 1e-12, "the paths of the two vertices of a face oppose "
               "each other, so no direction can be interpolated between them");
   t /= len;

   Vector x(2);
   FTr.SetAllIntPoints(&ip);
   FTr.Transform(ip, x);

   real_t s;
   MFEM_VERIFY(FirstCrossing(phi, x, t, search_length, search_steps, tol,
                             max_iter, s),
               "the interpolated path from a point of Gamma_h never meets "
               "Gamma within the search length " << search_length);

   xbar.SetSize(2);
   xbar(0) = x(0) + s * t(0);
   xbar(1) = x(1) + s * t(1);
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

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &IntRules.Get(Trans.GetGeometryType(), 2 * el1.GetOrder() + 2);
   }
   const IntegrationRule &lir = LineRule(el1);

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
      // the face.
      LiftBasis(el1, *Trans.Elem1, x, m, lir, L);

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

void HDGExtensionIntegrator::LiftBasis(
   const FiniteElement &el1, ElementTransformation &elem_tr, const Vector &x,
   const Vector &m, const IntegrationRule &lir, DenseMatrix &Lmat)
{
   const int dof = el1.GetDof();
   const int dim = m.Size();

   shape_ext.SetSize(dof);
   y.SetSize(dim);
   CTm.SetSize(dim);
   if (MC) { Cmat.SetSize(dim); }

   // The unit tangent is m/|m| and ds = |m| dt, so the length of the path
   // cancels and never appears.
   Lmat.SetSize(dof, dim);
   Lmat = 0.;
   for (int t = 0; t < lir.GetNPoints(); t++)
   {
      const IntegrationPoint &tip = lir.IntPoint(t);
      for (int d = 0; d < dim; d++) { y(d) = x(d) + tip.x * m(d); }

      IntegrationPoint eip;
      MFEM_VERIFY(ext.TransformBack(y, eip),
                  "the inverse element transformation did not converge on "
                  "the extension of the element beyond Gamma_h");
      elem_tr.SetIntPoint(&eip);
      el1.CalcPhysShape(elem_tr, shape_ext);

      // (C phi_j) . m = phi_j . (C^T m), and phi_j = shape_j e_d.
      if (MC)
      {
         MC->Eval(Cmat, elem_tr, eip);
         Cmat.MultTranspose(m, CTm);
      }
      else
      {
         CTm = m;
         CTm *= C->Eval(elem_tr, eip);
      }

      for (int d = 0; d < dim; d++)
      {
         const real_t wd = tip.weight * CTm(d);
         for (int j = 0; j < dof; j++)
         {
            Lmat(j, d) += wd * shape_ext(j);
         }
      }
   }
}

real_t HDGExtensionIntegrator::ComputeLift(
   const FiniteElement &el1, FaceElementTransformations &Trans,
   const IntegrationPoint &ip, const Vector &elfun)
{
   MFEM_VERIFY(Trans.Elem2No < 0,
               "the extension term lives on a boundary face only");

   const int dof = el1.GetDof();
   const int dim = Trans.GetSpaceDim();

   MFEM_VERIFY(elfun.Size() == dof * dim,
               "the flux dofs of the element owning the face are expected, "
               "with vdim equal to the space dimension");

   m.SetSize(dim);

   Trans.SetAllIntPoints(&ip);
   Trans.Transform(ip, x);

   // Endpoint() resets the transformations, so nothing read from them before
   // this point may be relied on afterwards -- which is why the element is
   // handed to the extension only below.
   path.Endpoint(Trans, ip, xbar);
   subtract(xbar, x, m);

   ext.SetElement(*Trans.Elem1);
   LiftBasis(el1, *Trans.Elem1, x, m, LineRule(el1), L);

   real_t lift = 0.;
   for (int d = 0; d < dim; d++)
      for (int j = 0; j < dof; j++)
      {
         lift += L(j, d) * elfun(dof * d + j);
      }
   return lift;
}

real_t PathLiftCoefficient::Eval(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   FaceElementTransformations *FTr =
      dynamic_cast<FaceElementTransformations *>(&T);
   MFEM_VERIFY(FTr, "PathLiftCoefficient must be evaluated on a face, as the "
               "lifting is defined along a path issuing from one");

   const FiniteElementSpace &fes = *u.FESpace();
   fes.GetElementVDofs(FTr->Elem1No, vdofs);
   u.GetSubVector(vdofs, elfun);

   return integ.ComputeLift(*fes.GetFE(FTr->Elem1No), *FTr, ip, elfun);
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

void ExtensionBoundaryQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir,
   const std::function<void(const ExtensionBoundaryPoint &)> &visit,
   real_t fd_step)
{
   const int sdim = FTr.GetSpaceDim();
   const int fdim = sdim - 1;

   Vector x(sdim), xbar(sdim), m(sdim), nu(sdim);
   Vector xbar_p(sdim), xbar_m(sdim), da(sdim);
   DenseMatrix dadxi(sdim, std::max(fdim, 1));

   for (int q = 0; q < face_ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = face_ir.IntPoint(q);

      path.Endpoint(FTr, ip, xbar);
      FTr.SetAllIntPoints(&ip);
      FTr.Transform(ip, x);
      subtract(xbar, x, m);

      // The derivative of the path's endpoint along the face, which at t = 1
      // is the WHOLE Jacobian of the map: the face's own dx/dxi is multiplied
      // by (1-t) and has gone.
      //
      // ASK THE FAMILY FIRST.  A family with a closed form gives an exact
      // Jacobian and saves 2(d-1) Endpoint() calls per point; one without says
      // so and the central difference below is taken instead, which is the
      // reference behaviour rather than a fallback in the pejorative sense.
      // Note the difference's own floor: O(fd_step^2) truncation against
      // O(eps/fd_step) round-off puts it near 1e-10 at the default step, and a
      // closed form removes that entirely.
      const bool analytic = path.EndpointJacobian(FTr, ip, dadxi);

      for (int i = 0; i < fdim && !analytic; i++)
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
      // Endpoint() -- or EndpointJacobian() -- moved the transformations; put
      // them back.
      FTr.SetAllIntPoints(&ip);

      // A point face -- the boundary of a one-dimensional mesh -- has no
      // tangent, and its image is a point of unit weight.  CalcOrtho() is not
      // defined for that shape, so it is taken separately rather than guarded
      // inside the loop above.
      real_t measure = 1.;
      if (fdim > 0)
      {
         CalcOrtho(dadxi, nu);
         measure = nu.Norml2();

         // A FACE WHOSE IMAGE IS A POINT CONTRIBUTES NOTHING, AND THAT IS
         // ORDINARY RATHER THAN AN ERROR.  A closest-point family collapses
         // every point of a face lying along a ray to the same foot -- for
         // ClosestPointPath::Sphere, a face of the staircase Gamma_h that
         // happens to be radial, which near the poles of the circle is every
         // second face.  Then da/dxi vanishes, Gamma has no normal to report,
         // and the right contribution is none: the face covers zero arc, its
         // two endpoint feet coincide, and its neighbours still cover Gamma
         // between them.
         //
         // THE TEST IS RELATIVE TO THE FACE, AND AN EXACT-ZERO TEST IS NOT
         // ENOUGH -- measured, not reasoned.  With ClosestPointPath's analytic
         // Jacobian the radial direction is annihilated by (I - uu^T), which is
         // exactly zero only in exact arithmetic; in floating point a radial
         // face gives |da/dxi| ~ 1e-16 pointing ANYWHERE.  An exact-zero test
         // passes that through, and the normal it yields is garbage -- measured
         // at 90 degrees from the true one, |nu - exact| = sqrt(2).  The
         // WEIGHT is ~1e-16 too, so a quadrature sum is unharmed and the fault
         // hides; a caller reading nu on its own is not so lucky.
         //
         // Comparing against the face's own |dx/dxi| is the defensible test:
         // both are lengths per unit reference coordinate, so the ratio is
         // dimensionless, and an image twelve orders shorter than the face that
         // produced it is degenerate on any reading.  A map that merely
         // compresses -- a distant sphere of small radius, where |da/dxi| is
         // R/|x-c| times |dx/dxi| -- is nowhere near this and is kept.
         // AND THE THRESHOLD HAS TO MATCH HOW THE JACOBIAN WAS OBTAINED, which
         // is the second half of the same lesson.  A closed form is accurate to
         // rounding, so a residue of order eps|dx/dxi| is degeneracy.  A central
         // difference is not: it divides by fd_step, so its own noise floor is
         // eps|a|/fd_step -- about 1e-10 at the default step, SIX ORDERS above
         // eps.  Measured, a radial face under the difference gives |da/dxi| ~
         // 5e-11 against a face scale of 0.1, sails through a 1e-12 test, and
         // hands back the same garbage normal at 90 degrees.
         //
         // So the two branches get their own floors, each a hundred times its
         // own noise.  A legitimate face here measures ~0.1, five orders clear
         // of the looser of them, so nothing real is at risk of being skipped.
         real_t face_scale = 0.;
         for (int i = 0; i < fdim; i++)
            for (int d = 0; d < sdim; d++)
            {
               face_scale += FTr.Jacobian()(d, i) * FTr.Jacobian()(d, i);
            }
         face_scale = std::sqrt(face_scale);

         const real_t eps = std::numeric_limits<real_t>::epsilon();
         const real_t floor = analytic
                              ? 1e-12 * face_scale
                              : 100. * (eps / fd_step) * xbar.Norml2();

         if (measure <= floor) { continue; }
         nu /= measure;
      }
      else
      {
         // In 1D the outward direction is the path's own.
         nu = m;
         const real_t length = nu.Norml2();
         MFEM_VERIFY(length > 0., "the path has zero length");
         nu /= length;
      }

      // THE ORIENTATION, AND IT IS THE PATHS THAT FIX IT.  CalcOrtho()'s sign
      // follows the ordering of dadxi's columns, which is the face's
      // parametrisation and carries no information about which side D_h is on.
      // The paths do: they run outward, so a normal agreeing with them is the
      // outward normal of Gamma.
      //
      // AND THE SAME TEST GIVES THE WEIGHT ITS SIGN, WHICH IS NOT COSMETIC.
      // The map xi -> a(x(xi)) is not required to be monotone along Gamma, and
      // for a staircase Gamma_h it is not: measured on a circle cut from a
      // diagonally split Cartesian mesh, most faces traverse their arc once but
      // a short "pinch" face runs forward, back and forward again, traversing
      // 0.0163 of arc length to cover 0.0060 of Gamma.  An UNSIGNED weight
      // integrates the traversed length, so those faces are counted two and
      // three times over and the sweep overcounts |Gamma| by O(h) -- which is
      // the accuracy the transfer technique exists to buy, thrown away in the
      // quadrature.
      //
      // A SIGNED weight cancels the backtracking exactly: where the map
      // reverses, dadxi reverses, CalcOrtho()'s normal turns inward, this test
      // fires, and the segment is subtracted.  What survives is the net
      // multiplicity, which is one.  So a caller integrating f over Gamma gets
      // the integral over Gamma however the family wanders, and the tiling
      // check becomes a statement about coverage rather than about
      // monotonicity.
      //
      // The normal handed to the visitor is ALWAYS the outward one; only the
      // weight carries the sign.  A visitor that ignores the sign -- summing
      // std::abs(weight), say -- measures the traversed length instead, which
      // is a different and occasionally useful quantity, but it is not a
      // quadrature over Gamma.
      real_t signed_weight = ip.weight * measure;
      if (nu * m < 0.) { nu.Neg(); signed_weight = -signed_weight; }

      const ExtensionBoundaryPoint pt{xbar, nu, ip, signed_weight};
      visit(pt);
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
