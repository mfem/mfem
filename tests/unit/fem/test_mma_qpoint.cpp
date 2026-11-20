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

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "fem/integ/mma/qpoint.hpp"

using namespace mfem;

namespace simplex_qpoint
{

/// Comparison tolerance, in double or single precision.
constexpr real_t Tol()
{
   return std::is_same<real_t, float>::value ? 1e-4 : 1e-10;
}

/// |a - b|_inf small relative to max(1, |b|_inf).
void RequireClose(const Vector &a, const Vector &b)
{
   REQUIRE(a.Size() == b.Size());
   Vector diff(a.Size());
   const real_t *ha = a.HostRead(), *hb = b.HostRead();
   for (int i = 0; i < a.Size(); ++i) { diff(i) = ha[i] - hb[i]; }
   const real_t scale = std::max(real_t(1), b.Normlinf());
   REQUIRE(diff.Normlinf() <= Tol() * scale);
}

/// Affine but irregular elements: a smooth map of the vertices only.
Mesh MakeAffineMesh(const char *filename)
{
   Mesh mesh(filename);
   mesh.Transform([](const Vector &x, Vector &y)
   {
      y = x;
      y(0) += 0.1 * std::sin(2.0 * x(1)) + 0.2 * x(1);
      y(1) += 0.1 * std::cos(1.5 * x(0));
      if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
   });
   REQUIRE(mesh.GetNodes() == nullptr);
   return mesh;
}

/// NATIVE E-vector of a random grid function, (ND, VDIM, NE).
void RandomEVector(FiniteElementSpace &fes, GridFunction &x, Vector &xe)
{
   x.Randomize(0x5eed);
   const Operator *R = fes.GetElementRestriction(ElementDofOrdering::NATIVE);
   xe.SetSize(R->Height());
   R->Mult(x, xe);
}

/// u, du and Lap u, and the geometric factors, against the host
/// CalcPhysShape, CalcPhysDShape, CalcPhysLaplacian and the element
/// transformation.
void test_qpoint_interpolation(const char *filename, int p, int vdim)
{
   CAPTURE(filename, p, vdim);
   Mesh mesh = MakeAffineMesh(filename);
   const int dim = mesh.Dimension(), ne = mesh.GetNE();
   H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec, vdim);
   const IntegrationRule &ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * p + 1);
   SimplexQPoint qp(fes, ir);
   const int nd = qp.NumDofs(), nq = qp.NumQuad();
   REQUIRE(qp.NumElements() == ne);

   GridFunction x(&fes);
   Vector xe, u, du, lap;
   RandomEVector(fes, x, xe);
   qp.Mult(xe, SimplexQPoint::VALUES | SimplexQPoint::GRADIENTS |
           SimplexQPoint::LAPLACIANS, u, du, lap);

   Vector u_ref(nq * vdim * ne), du_ref(nq * vdim * dim * ne);
   Vector lap_ref(nq * vdim * ne), detJ_ref(nq * ne), wdetJ_ref(nq * ne);
   Vector J_ref(nq * dim * dim * ne), invJ_ref(nq * dim * dim * ne);
   Vector shape(nd), lshape(nd), xl;
   DenseMatrix dshape(nd, dim);
   Array<int> vdofs;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &el = *fes.GetFE(e);
      fes.GetElementVDofs(e, vdofs);
      x.GetSubVector(vdofs, xl);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         el.CalcPhysShape(T, shape);
         el.CalcPhysDShape(T, dshape);
         el.CalcPhysLaplacian(T, lshape);
         for (int c = 0; c < vdim; ++c)
         {
            const Vector xc(xl.GetData() + nd * c, nd);
            u_ref(q + nq * (c + vdim * e)) = shape * xc;
            lap_ref(q + nq * (c + vdim * e)) = lshape * xc;
            for (int k = 0; k < dim; ++k)
            {
               real_t s = 0.0;
               for (int i = 0; i < nd; ++i) { s += dshape(i, k) * xc(i); }
               du_ref(q + nq * (c + vdim * (k + dim * e))) = s;
            }
         }
         for (int j = 0; j < dim; ++j)
         {
            for (int i = 0; i < dim; ++i)
            {
               const int idx = q + nq * (i + dim * (j + dim * e));
               J_ref(idx) = T.Jacobian()(i, j);
               invJ_ref(idx) = T.InverseJacobian()(i, j);
            }
         }
         detJ_ref(q + nq * e) = T.Weight();
         wdetJ_ref(q + nq * e) = ip.weight * T.Weight();
      }
   }
   RequireClose(u, u_ref);
   RequireClose(du, du_ref);
   RequireClose(lap, lap_ref);
   REQUIRE(lap_ref.Normlinf() > (p > 1 ? 1e-3 : -1.0)); // P1: Lap u = 0
   RequireClose(qp.GetJ(), J_ref);
   RequireClose(qp.GetInvJ(), invJ_ref);
   RequireClose(qp.GetDetJ(), detJ_ref);
   RequireClose(qp.GetWeightDetJ(), wdetJ_ref);

   // Each flag alone gives the same output as the combined call.
   Vector u1, du1, lap1, none;
   qp.Mult(xe, SimplexQPoint::VALUES, u1, none, none);
   qp.Mult(xe, SimplexQPoint::GRADIENTS, none, du1, none);
   qp.Mult(xe, SimplexQPoint::LAPLACIANS, none, none, lap1);
   REQUIRE(none.Size() == 0);
   RequireClose(u1, u);
   RequireClose(du1, du);
   RequireClose(lap1, lap);
}

/** Contractions and diagonals against host element matrices of the
    operator A_c = (1 + c) M + C + (2 + c) K for component c, with M the
    mass, C the convection (beta . grad u, v) and K the diffusion matrix:
    - y = A x is B^T dv + G^T dg with dv = w|J| ((1 + c) u + beta . du) and
      dg = w|J| (2 + c) du;
    - diag(A) is the diagonal of B^T dvv B + B^T dvg G + G^T dgg G with
      dvv = w|J| (1 + c), dvg = w|J| beta and dgg = w|J| (2 + c) I. */
void test_qpoint_contractions(const char *filename, int p, int vdim)
{
   CAPTURE(filename, p, vdim);
   Mesh mesh = MakeAffineMesh(filename);
   const int dim = mesh.Dimension(), ne = mesh.GetNE();
   H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec, vdim);
   const IntegrationRule &ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * p + 1);
   SimplexQPoint qp(fes, ir);
   const int nd = qp.NumDofs(), nq = qp.NumQuad();

   GridFunction x(&fes);
   Vector xe, u, du, none;
   RandomEVector(fes, x, xe);
   qp.Mult(xe, SimplexQPoint::VALUES | SimplexQPoint::GRADIENTS,
           u, du, none);

   Vector beta(dim);
   for (int k = 0; k < dim; ++k) { beta(k) = 0.5 - 0.3 * k; }
   const int nqv = nq * vdim;
   const real_t *wd = qp.GetWeightDetJ().HostRead();
   const real_t *hu = u.HostRead(), *hdu = du.HostRead();
   Vector dv(nqv * ne), dg(nqv * dim * ne), dvv(nqv * ne);
   Vector dvg(nqv * dim * ne), dgg(nqv * dim * dim * ne);
   dgg = 0.0;
   for (int e = 0; e < ne; ++e)
   {
      for (int c = 0; c < vdim; ++c)
      {
         for (int q = 0; q < nq; ++q)
         {
            const int qce = q + nq * (c + vdim * e);
            const real_t w = wd[q + nq * e];
            real_t bdu = 0.0;
            for (int k = 0; k < dim; ++k)
            {
               const int qcke = q + nq * (c + vdim * (k + dim * e));
               bdu += beta(k) * hdu[qcke];
               dg(qcke) = w * (2 + c) * hdu[qcke];
               dvg(qcke) = w * beta(k);
               dgg(q + nq * (c + vdim * (k + dim * (k + dim * e)))) =
                  w * (2 + c);
            }
            dv(qce) = w * ((1 + c) * hu[qce] + bdu);
            dvv(qce) = w * (1 + c);
         }
      }
   }

   Vector y(nd * vdim * ne), diag(nd * vdim * ne);
   y = 0.0;
   diag = 0.0;
   qp.AddMultTranspose(dv, dg, y);
   qp.AddDiagonal(dvv, dvg, dgg, diag);

   Vector y_ref(y.Size()), diag_ref(diag.Size()), xl, yc(nd);
   VectorConstantCoefficient beta_coeff(beta);
   MassIntegrator mass;
   ConvectionIntegrator conv(beta_coeff);
   DiffusionIntegrator diff;
   mass.SetIntRule(&ir);
   conv.SetIntRule(&ir);
   diff.SetIntRule(&ir);
   DenseMatrix Me, Ce, Ke, Ae;
   Array<int> vdofs;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &el = *fes.GetFE(e);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      mass.AssembleElementMatrix(el, T, Me);
      conv.AssembleElementMatrix(el, T, Ce);
      diff.AssembleElementMatrix(el, T, Ke);
      fes.GetElementVDofs(e, vdofs);
      x.GetSubVector(vdofs, xl);
      for (int c = 0; c < vdim; ++c)
      {
         Ae = Ce;
         Ae.Add(1 + c, Me);
         Ae.Add(2 + c, Ke);
         const Vector xc(xl.GetData() + nd * c, nd);
         Ae.Mult(xc, yc);
         for (int i = 0; i < nd; ++i)
         {
            y_ref(i + nd * (c + vdim * e)) = yc(i);
            diag_ref(i + nd * (c + vdim * e)) = Ae(i, i);
         }
      }
   }
   RequireClose(y, y_ref);
   RequireClose(diag, diag_ref);

   // Empty data vectors are skipped: the parts add up to the whole.
   Vector y_parts(y.Size()), diag_parts(diag.Size());
   y_parts = 0.0;
   diag_parts = 0.0;
   qp.AddMultTranspose(dv, none, y_parts);
   qp.AddMultTranspose(none, dg, y_parts);
   qp.AddDiagonal(dvv, none, none, diag_parts);
   qp.AddDiagonal(none, dvg, none, diag_parts);
   qp.AddDiagonal(none, none, dgg, diag_parts);
   RequireClose(y_parts, y);
   RequireClose(diag_parts, diag);
}

/** One generic simplex, its vertices given in every order: the metric
    G = (dxi/dx)^T K (dxi/dx), K = c (I + 1 1^T), is the same for all the
    orders and matches the host formula, while (dxi/dx)^T (dxi/dx) alone
    is not invariant. Inverted orders (det J < 0) are kept on purpose. */
void test_qpoint_metric_permutations(const int dim)
{
   CAPTURE(dim);
   const real_t X[4][3] = {{0.1, 0.0, 0.05}, {1.2, 0.2, -0.1},
      {0.3, 0.9, 0.2}, {0.25, 0.15, 1.1}
   };
   const real_t c = (dim == 2) ? 1.0 / std::sqrt(3.0) : std::cbrt(2.0) / 2.0;
   DenseMatrix K(dim), KJi(dim), G_ref(dim), JtJ(dim);
   K = c;
   for (int k = 0; k < dim; ++k) { K(k, k) = 2.0 * c; }

   std::array<int, 4> perm = {0, 1, 2, 3};
   Vector G0, JtJ0;
   real_t JtJ_spread = 0.0;
   do
   {
      Mesh mesh(dim, dim + 1, 1, 0, dim);
      for (int v = 0; v <= dim; ++v) { mesh.AddVertex(X[v]); }
      if (dim == 2)
      {
         mesh.AddTriangle(perm.data());
         mesh.FinalizeTriMesh(0, 0, false);
      }
      else
      {
         mesh.AddTet(perm.data());
         mesh.FinalizeTetMesh(0, 0, false);
      }
      H1_FECollection fec(1, dim);
      FiniteElementSpace fes(&mesh, &fec);
      const IntegrationRule &ir =
         IntRules.Get(mesh.GetTypicalElementGeometry(), 2);
      SimplexQPoint qp(fes, ir);
      Vector G;
      qp.Metric(G);
      const int nq = qp.NumQuad();
      const auto Gq = Reshape(G.HostRead(), nq, dim, dim, 1);

      ElementTransformation &T = *mesh.GetElementTransformation(0);
      T.SetIntPoint(&ir.IntPoint(0));
      Mult(K, T.InverseJacobian(), KJi);
      MultAtB(T.InverseJacobian(), KJi, G_ref);
      MultAtB(T.InverseJacobian(), T.InverseJacobian(), JtJ);
      Vector Gv(dim * dim), JtJv(JtJ.GetData(), dim * dim);
      for (int q = 0; q < nq; ++q)
      {
         for (int l = 0; l < dim; ++l)
         {
            for (int k = 0; k < dim; ++k) { Gv(k + dim * l) = Gq(q, k, l, 0); }
         }
         RequireClose(Gv, Vector(G_ref.GetData(), dim * dim));
      }
      if (G0.Size() == 0) { G0 = Gv; JtJ0 = JtJv; }
      RequireClose(Gv, G0);
      Vector spread(JtJv);
      spread -= JtJ0;
      JtJ_spread = std::max(JtJ_spread, spread.Normlinf());
   }
   while (std::next_permutation(perm.begin(), perm.begin() + dim + 1));
   REQUIRE(JtJ_spread > 0.1); // the plain metric depends on the numbering
}

TEST_CASE("Simplex QPoint interpolation", "[MMA][GPU]")
{
   const auto p = GENERATE(1, 2, 3);
   const auto vdim = GENERATE(1, 3);
   SECTION("2D")
   {
      test_qpoint_interpolation("../../data/inline-tri.mesh", p, vdim);
   }
   SECTION("3D")
   {
      test_qpoint_interpolation("../../data/inline-tet.mesh", p, vdim);
      test_qpoint_interpolation("../../data/beam-tet.mesh", p, vdim);
   }
}

// With 64 components, one P3 tetrahedron needs about 170 KB of shared memory
// for the value, gradient and Hessian planes, more than a CUDA block may
// use: Mult() then takes the SimplexInterp (dense forall) fallback instead
// of the batched GEMM kernel. The host paths are the same as above.
TEST_CASE("Simplex QPoint shared-memory fallback", "[MMA][GPU]")
{
   test_qpoint_interpolation("../../data/inline-tet.mesh", 3, 64);
}

TEST_CASE("Simplex QPoint contractions and diagonals", "[MMA][GPU]")
{
   const auto p = GENERATE(1, 2, 3);
   const auto vdim = GENERATE(1, 3);
   SECTION("2D")
   {
      test_qpoint_contractions("../../data/inline-tri.mesh", p, vdim);
   }
   SECTION("3D")
   {
      test_qpoint_contractions("../../data/inline-tet.mesh", p, vdim);
      test_qpoint_contractions("../../data/beam-tet.mesh", p, vdim);
   }
}

/** A SimplexQPoint of a pressure space built from the one of a velocity
    space shares its geometric factors and matches a standalone one. */
void test_qpoint_shared_factors(const char *filename)
{
   CAPTURE(filename);
   Mesh mesh = MakeAffineMesh(filename);
   const int dim = mesh.Dimension();
   H1_FECollection fec_v(2, dim), fec_p(1, dim);
   FiniteElementSpace fes_v(&mesh, &fec_v, dim), fes_p(&mesh, &fec_p);
   const IntegrationRule &ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 5);
   SimplexQPoint vel(fes_v, ir), pres(fes_p, vel), ref(fes_p, ir);
   REQUIRE(pres.NumQuad() == ref.NumQuad());
   REQUIRE(pres.GetInvJ().GetData() == vel.GetInvJ().GetData());
   REQUIRE(pres.GetWeightDetJ().GetData() == vel.GetWeightDetJ().GetData());
   RequireClose(pres.GetJ(), ref.GetJ());
   RequireClose(pres.GetDetJ(), ref.GetDetJ());

   GridFunction x(&fes_p);
   Vector xe, u, du, lap, u_ref, du_ref, lap_ref;
   RandomEVector(fes_p, x, xe);
   const unsigned all = SimplexQPoint::VALUES | SimplexQPoint::GRADIENTS |
                        SimplexQPoint::LAPLACIANS;
   pres.Mult(xe, all, u, du, lap);
   ref.Mult(xe, all, u_ref, du_ref, lap_ref);
   RequireClose(u, u_ref);
   RequireClose(du, du_ref);
   RequireClose(lap, lap_ref);
}

TEST_CASE("Simplex QPoint shared geometric factors", "[MMA][GPU]")
{
   test_qpoint_shared_factors("../../data/inline-tri.mesh");
   test_qpoint_shared_factors("../../data/inline-tet.mesh");
}

TEST_CASE("Simplex QPoint metric vertex permutations", "[MMA][GPU]")
{
   test_qpoint_metric_permutations(2);
   test_qpoint_metric_permutations(3);
}

/** SimplexQPoint of a P2 space with 3 components on the same straight
    elements, described by the vertices of @a lin and by the high-order nodes
    of @a ho: same geometric factors, metric and interpolation. */
void RequireSameQPoint(Mesh &lin, Mesh &ho)
{
   const int p = 2, vdim = 3, dim = lin.Dimension();
   H1_FECollection fec(p, dim);
   FiniteElementSpace fes_lin(&lin, &fec, vdim), fes_ho(&ho, &fec, vdim);
   const IntegrationRule &ir =
      IntRules.Get(lin.GetTypicalElementGeometry(), 2 * p + 1);
   SimplexQPoint qa(fes_lin, ir), qb(fes_ho, ir);
   RequireClose(qb.GetJ(), qa.GetJ());
   RequireClose(qb.GetInvJ(), qa.GetInvJ());
   RequireClose(qb.GetDetJ(), qa.GetDetJ());
   RequireClose(qb.GetWeightDetJ(), qa.GetWeightDetJ());
   Vector Ga, Gb;
   qa.Metric(Ga);
   qb.Metric(Gb);
   RequireClose(Gb, Ga);

   const unsigned all = SimplexQPoint::VALUES | SimplexQPoint::GRADIENTS |
                        SimplexQPoint::LAPLACIANS;
   Vector xe(qa.NumDofs() * vdim * qa.NumElements());
   xe.Randomize(0x5eed);
   Vector ua, dua, lapa, ub, dub, lapb;
   qa.Mult(xe, all, ua, dua, lapa);
   qb.Mult(xe, all, ub, dub, lapb);
   RequireClose(ub, ua);
   RequireClose(dub, dua);
   RequireClose(lapb, lapa);
   Vector ya(xe.Size()), yb(xe.Size());
   ya = 0.0;
   yb = 0.0;
   qa.AddMultTranspose(ua, dua, ya);
   qb.AddMultTranspose(ua, dua, yb);
   RequireClose(yb, ya);
}

TEST_CASE("Simplex QPoint affine meshes only", "[MMA][GPU]")
{
   const char *filename =
      GENERATE("../../data/inline-tri.mesh", "../../data/inline-tet.mesh");
   const auto order = GENERATE(2, 3);
   CAPTURE(filename, order);
   Mesh lin = MakeAffineMesh(filename);
   const int dim = lin.Dimension();
   REQUIRE(SimplexQPoint::IsAffine(lin));

   SECTION("SetCurvature, then a curving map")
   {
      Mesh ho(lin);
      ho.SetCurvature(1);
      REQUIRE(SimplexQPoint::IsAffine(ho));
      ho.SetCurvature(order); // nodes on the straight elements
      REQUIRE(SimplexQPoint::IsAffine(ho));
      RequireSameQPoint(lin, ho);

      ho.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         y(0) += 0.05 * std::sin(M_PI * x(1));
         y(1) += 0.05 * x(0) * x(0);
      });
      REQUIRE_FALSE(SimplexQPoint::IsAffine(ho));
#ifdef MFEM_USE_EXCEPTIONS
      H1_FECollection fec(2, dim);
      FiniteElementSpace fes(&ho, &fec);
      const IntegrationRule &ir =
         IntRules.Get(ho.GetTypicalElementGeometry(), 4);
      REQUIRE_THROWS(SimplexQPoint(fes, ir));
#endif
   }

   SECTION("SetNodalFESpace, byNODES vector space")
   {
      Mesh ho(lin);
      H1_FECollection nfec(order, dim);
      FiniteElementSpace vfes(&ho, &nfec, dim, Ordering::byNODES);
      ho.SetNodalFESpace(&vfes);
      REQUIRE(ho.GetNodes()->FESpace()->GetOrdering() == Ordering::byNODES);
      REQUIRE(SimplexQPoint::IsAffine(ho));
      RequireSameQPoint(lin, ho);
   }
}

} // namespace simplex_qpoint
