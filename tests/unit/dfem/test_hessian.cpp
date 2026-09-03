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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

// ────────────────────────────────────────────────────────────────────────────
// Reference Hessian of a scalar H1 field, and the pullback to physical space.
//
// The Hessian FieldOperator delivers Href_ab = d^2 u / dxi_a dxi_b, so the
// mapping is the q-function's job, exactly as it is for Gradient. With
// J_ca = dx_c / dxi_a and K = inv(J),
//
//   d^2 u / dx_i dx_j = K_ai K_bj
//      (Href_ab - sum_c g_c d^2 x_c / dxi_a dxi_b)
//
// where g_i = du/dx_i = K_ai du/dxi_a is the physical gradient. The sum over c
// contracts g with the Hessian of the coordinate map. This mesh term vanishes
// on affine elements.
// ────────────────────────────────────────────────────────────────────────────

/// Physical Hessian assuming an affine mapping
/// Sanity check, it should work without correction in hessian pullback
// H = J^{-T} Href J^{-1}
template <int DIM>
struct affine_hessian_qf
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<real_t, DIM, DIM> &dduxi,
      const tensor<real_t, DIM, DIM> &J,
      tensor<real_t, DIM, DIM> &out) const
   {
      const auto K = inv(J);
      out = transpose(K) * dduxi * K;
   }
};

/// Physical Hessian with the full non-affine correction.
///
/// dduxi is Href, duxi is du/dxi, and ddx is the Hessian of the coordinate
/// map. The correction is M_ab = Href_ab - sum_c g_c d^2 x_c/dxi_a dxi_b.
template <int DIM>
struct hessian_qf
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<real_t, DIM, DIM> &dduxi,
      const tensor<real_t, DIM> &duxi,
      const tensor<real_t, DIM, DIM> &J,
      const tensor<real_t, DIM, DIM, DIM> &ddx,
      tensor<real_t, DIM, DIM> &out) const
   {
      const auto K = inv(J);
      const auto Kt = transpose(K);
      const auto g = Kt * duxi;
      // dot contracts the first index: (g . ddx)_ab = sum_c g_c ddx_cab.
      const auto M = dduxi - dot(g, ddx);
      out = Kt * M * K;
   }
};

// ────────────────────────────────────────────────────────────────────────────
/// Map from a packed CalcPhysHessian column to the (i,j) entries it fills.
/// The packing is xx, xy, xz, yy, yz, zz, see FiniteElement::CalcPhysHessian.
inline void SymIndex(int dim, int k, int &i, int &j)
{
   if (dim == 2)
   {
      static const int I[3] = {0, 0, 1}, J[3] = {0, 1, 1};
      i = I[k];
      j = J[k];
   }
   else
   {
      static const int I[6] = {0, 0, 0, 1, 1, 2}, J[6] = {0, 1, 2, 1, 2, 2};
      i = I[k];
      j = J[k];
   }
}

/// @brief Host reference: the physical Hessian of @a u at every quadrature
/// point of every element, via FiniteElement::CalcPhysHessian.
///
/// Returned as (dim*dim, nqp, ne) with the (i,j) entry at i + dim*j. This works
/// in the element's own dof ordering and never touches the E-vector layout the
/// dFEM backend uses, so it is an independent check of the whole chain.
inline Vector ReferencePhysHessian(ParGridFunction &u,
                                   const IntegrationRule &ir)
{
   ParFiniteElementSpace &fes = *u.ParFESpace();
   ParMesh &mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   const int nqp = ir.GetNPoints();
   const int ne = mesh.GetNE();
   const int nsym = (dim * (dim + 1)) / 2;

   Vector out(dim * dim * nqp * ne);
   out = 0.0;
   auto H = Reshape(out.HostWrite(), dim * dim, nqp, ne);

   Array<int> dofs;
   Vector elvec;
   for (int e = 0; e < ne; e++)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &T = *fes.GetElementTransformation(e);
      fes.GetElementDofs(e, dofs);
      u.GetSubVector(dofs, elvec);
      const int ndof = fe.GetDof();

      DenseMatrix hess(ndof, nsym);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcPhysHessian(T, hess);
         for (int k = 0; k < nsym; k++)
         {
            real_t s = 0.0;
            for (int d = 0; d < ndof; d++) { s += elvec(d) * hess(d, k); }
            int i, j;
            SymIndex(dim, k, i, j);
            H(i + dim * j, q, e) = s;
            H(j + dim * i, q, e) = s;
         }
      }
   }
   return out;
}

inline real_t MaxAbsDiff(const Vector &a, const Vector &b, MPI_Comm comm)
{
   MFEM_VERIFY(a.Size() == b.Size(), "size mismatch");
   real_t local = 0.0;
   const real_t *pa = a.HostRead(), *pb = b.HostRead();
   for (int i = 0; i < a.Size(); i++)
   {
      local = std::max(local, std::abs(pa[i] - pb[i]));
   }
   real_t global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
}

// ────────────────────────────────────────────────────────────────────────────
/// @brief Compare the dFEM Hessian pullback against CalcPhysHessian.
///
/// @param affine_qf when true, run the q-function that drops the mesh term.
/// On a curved mesh that must *not* reproduce the reference: a missing
/// correction term still converges under refinement, so the only way to know it
/// is being exercised is to check that removing it breaks the answer.
template <int DIM>
real_t hessian_error(const char *filename, int p, bool affine_qf)
{
   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   p = std::max(p, mfes->GetMaxElementOrder());

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);
   const int nqp = ir->GetNPoints();
   const int ne = pmesh.GetNE();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   ParGridFunction u(&pfes);
   {
      Vector tv(pfes.GetTrueVSize());
      tv.Randomize(1);
      u.SetFromTrueDofs(tv);
   }

   static constexpr int U = 0, Coords = 1, QData = 2;

   QuadratureSpace qspace(pmesh, *ir);
   VectorQuadratureSpace qspace_vec(qspace, DIM * DIM);
   QuadratureFunction qd(qspace_vec);
   qd = 0.0;

   Vector utv, nodestv;
   u.GetTrueDofs(utv);
   nodes->GetTrueDofs(nodestv);

   if (affine_qf)
   {
      const std::vector<FieldDescriptor> input_fields = {{U, &pfes}, {Coords, mfes}};
      const std::vector<FieldDescriptor> output_fields = {{QData, &qspace_vec}};
      const auto input_fieldops = Inputs<Hessian<U>, Gradient<Coords>> {};
      const auto output_fieldops = Outputs<Identity<QData>> {};

      DifferentiableOperator dop(input_fields, output_fields, pmesh);
      affine_hessian_qf<DIM> qf;
      dop.AddDomainIntegrator<LocalQFBackend>(
         qf,
         input_fieldops,
         output_fieldops,
         *ir, all_domain_attr);
      MultiVector X{utv, nodestv};
      MultiVector Y{qd};
      dop.Mult(X, Y);
   }
   else
   {
      // We also need Gradient<U> and Hessian<Coords> to compute the mesh term.
      const std::vector<FieldDescriptor> input_fields = {{U, &pfes}, {Coords, mfes}};
      const std::vector<FieldDescriptor> output_fields = {{QData, &qspace_vec}};
      const auto input_fieldops =
         Inputs<Hessian<U>, Gradient<U>, Gradient<Coords>, Hessian<Coords>> {};
      const auto output_fieldops = Outputs<Identity<QData>> {};

      DifferentiableOperator dop(input_fields, output_fields, pmesh);
      hessian_qf<DIM> qf;
      dop.AddDomainIntegrator<LocalQFBackend>(
         qf,
         input_fieldops,
         output_fieldops,
         *ir, all_domain_attr);
      MultiVector X{utv, nodestv};
      MultiVector Y{qd};
      dop.Mult(X, Y);
   }

   const Vector ref = ReferencePhysHessian(u, *ir);
   MFEM_VERIFY(qd.Size() == DIM * DIM * nqp * ne, "quadrature size mismatch");
   return MaxAbsDiff(qd, ref, pmesh.GetComm());
}


///////////////////////////////////////////////////////////////////////////////
///-----         TESTS
///////////////////////////////////////////////////////////////////////////////

// ────────────────────────────────────────────────────────────────────────────
// 1. 1D second derivative map (independent of dFEM).
// ────────────────────────────────────────────────────────────────────────────

/// Check the 1D second derivative map using central differences.
TEST_CASE("dFEM Hessian DofToQuad 1D map", "[Parallel][dFEM][Hessian]")
{
   // H1 requires a closed basis type.
   const int btype = GENERATE(BasisType::GaussLobatto,
                              BasisType::ClosedGL,
                              BasisType::ClosedUniform);
   const int p = GENERATE(1, 2);
   CAPTURE(btype, p);

   H1_FECollection fec(p, 2, btype);
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   FiniteElementSpace fes(&mesh, &fec);

   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * p);
   const DofToQuad &d2q = fes.GetTypicalFE()->GetDofToQuad(
                             ir, DofToQuad::TENSOR);

   REQUIRE(d2q.H.Size() == d2q.nqpt * d2q.ndof);

   // Central difference of the 1D basis derivative.
   // Sanity check: we verify that Poly_1D::Basis::Barycentric
   // is flagged as having second derivatives.
   const Poly_1D::Basis &basis = poly1d.GetBasis(p, btype);
   REQUIRE(basis.HasSecondDerivatives());

   const real_t h = 1e-5;
   Vector up(p + 1), dp(p + 1), um(p + 1), dm(p + 1);

   real_t max_err = 0.0;
   int compared = 0;
   for (int i = 0; i < d2q.nqpt; i++)
   {
      const real_t x = ir.IntPoint(i).x;
      // The stencil has to stay inside [0,1].
      if (x < h || x > 1.0 - h) { continue; }
      basis.Eval(x + h, up, dp);
      basis.Eval(x - h, um, dm);
      compared++;
      for (int j = 0; j < d2q.ndof; j++)
      {
         const real_t fd = (dp(j) - dm(j)) / (2 * h);
         max_err = std::max(max_err,
                            std::abs(fd - d2q.H[i + d2q.nqpt * j]));
      }
   }
   REQUIRE(compared > 0);
   // Central differences at h = 1e-5 on an O(1)-scaled basis; the truncation
   // error grows with p because the third derivative does.
   REQUIRE(max_err < 1e-4 * std::pow(2.0, p));

   // Ht is the transpose of H.
   for (int i = 0; i < d2q.nqpt; i++)
   {
      for (int j = 0; j < d2q.ndof; j++)
      {
         REQUIRE(d2q.H[i + d2q.nqpt * j] == d2q.Ht[j + d2q.ndof * i]);
      }
   }
}

/// Check unsupported basis
TEST_CASE("dFEM Hessian unsupported basis", "[Parallel][dFEM][Hessian]")
{
   // Bernstein and integrated GLL have no second derivative evaluation,
   // in the library we keep H empty for those, just check it as a sanity check.
   const int p = 3;
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * p);

   H1_FECollection pos(p, 2, BasisType::Positive);
   FiniteElementSpace pos_fes(&mesh, &pos);
   const DofToQuad &pd2q =
      pos_fes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   REQUIRE(pd2q.H.Size() == 0);

   H1_FECollection gll(p, 2, BasisType::GaussLobatto);
   FiniteElementSpace gll_fes(&mesh, &gll);
   const DofToQuad &gd2q =
      gll_fes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   REQUIRE(gd2q.H.Size() > 0);
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Physical Hessian (dFEM) against CalcPhysHessian.
// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM Hessian 2D", "[Parallel][dFEM][Hessian]")
{
   // Affine mesh tested with/without the correction term,
   // In this case the correction term should vanish so both
   // tests should pass with the same result.
   SECTION("affine mesh, no mesh term")
   {
      const int p = GENERATE(1, 2);
      CAPTURE(p);
      REQUIRE(hessian_error<2>("../../data/inline-quad.mesh", p, true) ==
              MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("affine mesh, with mesh term")
   {
      const int p = GENERATE(1,2);
      CAPTURE(p);
      REQUIRE(hessian_error<2>("../../data/inline-quad.mesh", p, false) ==
              MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("curved mesh, non affine mesh, LO kernel path")
   {
      const int p = GENERATE(3, 4);
      CAPTURE(p);
      REQUIRE(hessian_error<2>("../../data/star-q3.mesh", p, false) ==
              MFEM_Approx(0.0, 1e-8, 1e-8));
   }

   SECTION("curved mesh, no correction (should fail)")
   {
      // Guards the correction term itself: without this the affine sections
      // would pass with a Hessian that is wrong on every curved element.
      REQUIRE(hessian_error<2>("../../data/star-q3.mesh", 2, true) > 1e-3);
   }

   SECTION("curved mesh, non affine mesh, HO kernel path")
   {
      // d1d = p + 1 > 8 selects LocalQFHOBackend rather than LocalQFLOBackend.
      REQUIRE(hessian_error<2>("../../data/star-q3.mesh", 8, false) ==
              MFEM_Approx(0.0, 1e-8, 1e-8));
   }
}



TEST_CASE("dFEM Hessian 3D", "[Parallel][dFEM][Hessian]")
{
   // Affine mesh tested with/without the correction term,
   // In this case the correction term should vanish so both
   // tests should pass with the same result.
   SECTION("affine mesh, no mesh term")
   {
      const int p = GENERATE(1, 2);
      CAPTURE(p);
      REQUIRE(hessian_error<3>("../../data/inline-hex.mesh", p, true) ==
              MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("affine mesh, with mesh term")
   {
      const int p = GENERATE(1, 2);
      CAPTURE(p);
      REQUIRE(hessian_error<3>("../../data/inline-hex.mesh", p, false) ==
              MFEM_Approx(0.0, 1e-10, 1e-10));
   }

   SECTION("curved mesh, non affine mesh, LO kernel path")
   {
      const int p = GENERATE(3, 4);
      CAPTURE(p);
      REQUIRE(hessian_error<3>("../../data/fichera-q3.mesh", p, false) ==
              MFEM_Approx(0.0, 1e-8, 1e-8));
   }

   SECTION("curved mesh, no correction (should fail)")
   {
      // Guards the correction term itself: without this the affine sections
      // would pass with a Hessian that is wrong on every curved element.
      REQUIRE(hessian_error<3>("../../data/fichera-q3.mesh", 2, true) > 1e-3);
   }

   SECTION("curved mesh, non affine mesh, HO kernel path")
   {
      // d1d = p + 1 > 8 selects LocalQFHOBackend rather than LocalQFLOBackend.
      REQUIRE(hessian_error<3>("../../data/fichera-q3.mesh", 8, false) ==
              MFEM_Approx(0.0, 1e-8, 1e-8));
   }
}

#endif // MFEM_USE_MPI
