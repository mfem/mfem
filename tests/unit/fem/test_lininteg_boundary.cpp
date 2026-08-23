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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace lininteg_boundary
{

// Boundary attributes of Mesh::MakeCartesian2D: 1 = bottom, 2 = right,
// 3 = top, 4 = left. The right-hand side is the useful one here because its
// outward normal is the constant (1,0), which lets a vector coefficient
// contracted with the normal be written as an ordinary scalar coefficient --
// and that is what makes the new vector-coefficient overloads comparable
// against the scalar ones that predate them.
constexpr int RIGHT = 2;

Array<int> OnlyRight(const Mesh &mesh)
{
   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[RIGHT - 1] = 1;
   return marker;
}

} // namespace lininteg_boundary

TEST_CASE("VectorBoundaryFluxLFIntegrator: vector overload matches the scalar",
          "[LinearFormIntegrator]")
{
   using namespace lininteg_boundary;

   // The vector-coefficient form writes vdim_f * dim blocks, one group per
   // component of the coefficient, and each group must reproduce exactly what
   // the pre-existing scalar form produces for that component.
   const int order = GENERATE(1, 2);
   const real_t sign = GENERATE(1.0, -1.0);

   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();
   CAPTURE(order, sign);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes_scalar(&mesh, &fec);
   const int nd = fes_scalar.GetNDofs();

   const int vdim_f = 2;
   Vector fv(vdim_f);
   fv(0) = 1.7;
   fv(1) = -0.9;
   VectorConstantCoefficient vf(fv);

   FiniteElementSpace fes_vec(&mesh, &fec, vdim_f * dim);
   LinearForm b_vec(&fes_vec);
   Array<int> marker = OnlyRight(mesh);
   b_vec.AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(vf, sign),
                              marker);
   b_vec.Assemble();

   for (int v = 0; v < vdim_f; v++)
   {
      ConstantCoefficient fs(fv(v));
      FiniteElementSpace fes_one(&mesh, &fec, dim);
      LinearForm b_one(&fes_one);
      Array<int> m = OnlyRight(mesh);
      b_one.AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(fs, sign),
                                 m);
      b_one.Assemble();

      REQUIRE(b_one.Size() == nd * dim);
      for (int i = 0; i < nd * dim; i++)
      {
         CAPTURE(v, i);
         REQUIRE(b_vec[v * nd * dim + i] == MFEM_Approx(b_one[i], 1e-12, 1e-11));
      }
   }
}

TEST_CASE("VectorFEBoundaryFluxLFIntegrator: vector overload matches the scalar",
          "[LinearFormIntegrator]")
{
   using namespace lininteg_boundary;

   // Same identity, on an RT test space: block v of the vector form is the
   // scalar form driven by component v.
   const int order = GENERATE(0, 1, 2);

   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();
   CAPTURE(order);

   RT_FECollection fec(order, dim);

   const int vdim_f = 2;
   Vector fv(vdim_f);
   fv(0) = 1.7;
   fv(1) = -0.9;
   VectorConstantCoefficient vf(fv);

   FiniteElementSpace fes_vec(&mesh, &fec, vdim_f);
   LinearForm b_vec(&fes_vec);
   Array<int> marker = OnlyRight(mesh);
   b_vec.AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(vf), marker);
   b_vec.Assemble();

   FiniteElementSpace fes_one(&mesh, &fec);
   const int nd = fes_one.GetNDofs();

   for (int v = 0; v < vdim_f; v++)
   {
      ConstantCoefficient fs(fv(v));
      LinearForm b_one(&fes_one);
      Array<int> m = OnlyRight(mesh);
      b_one.AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(fs), m);
      b_one.Assemble();

      for (int i = 0; i < nd; i++)
      {
         CAPTURE(v, i);
         REQUIRE(b_vec[v * nd + i] == MFEM_Approx(b_one[i], 1e-12, 1e-11));
      }
   }
}

TEST_CASE("BoundaryFlowIntegrator: vector overload matches the scalar",
          "[LinearFormIntegrator]")
{
   using namespace lininteg_boundary;

   // The vector form computes f = (vf . n) / |u . n| internally. On the right
   // face of the unit square the normal is constant, so that quotient is an
   // ordinary constant and the scalar form must produce the same vector.
   const int order = GENERATE(1, 2);
   const real_t alpha = 2.0;
   const real_t beta = GENERATE(0.0, 0.5, 1.25);

   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();
   CAPTURE(order, beta);

   L2_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Vector uv(dim);
   uv(0) = 1.4;
   uv(1) = 0.0;              // no flow through the top or bottom
   VectorConstantCoefficient u(uv);

   Vector fvv(dim);
   fvv(0) = 0.6;
   fvv(1) = -2.1;
   VectorConstantCoefficient vf(fvv);

   // On the right face: n = (1,0), so (vf.n)/|u.n| = fvv(0)/|uv(0)|.
   ConstantCoefficient fs(fvv(0) / std::abs(uv(0)));

   LinearForm b_vec(&fes), b_scalar(&fes);
   Array<int> m1 = OnlyRight(mesh), m2 = OnlyRight(mesh);
   b_vec.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(vf, u, alpha, beta),
                              m1);
   b_scalar.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(fs, u, alpha, beta),
                                 m2);
   b_vec.Assemble();
   b_scalar.Assemble();

   REQUIRE(b_vec.Norml2() > 0.0);   // the comparison must not be vacuous
   for (int i = 0; i < b_vec.Size(); i++)
   {
      CAPTURE(i);
      REQUIRE(b_vec[i] == MFEM_Approx(b_scalar[i], 1e-12, 1e-11));
   }
}

TEST_CASE("BoundaryNormalFlowIntegrator against a hand-computed integral",
          "[LinearFormIntegrator]")
{
   using namespace lininteg_boundary;

   // The form is  alpha/2 <(u.n) f, w.n> - beta <|u.n| f, w.n>  over the whole
   // boundary, with w in a scalar space of vdim = dim. Take f = 1 and
   // u = (1,0) on the unit square, and w the interpolant of (x, 0):
   //
   //   right (x=1, n=( 1,0)):  u.n = 1,  |u.n| = 1,  w.n = 1  -> alpha/2 - beta
   //   left  (x=0, n=(-1,0)):  u.n = -1,             w.n = 0  -> 0
   //   top and bottom:         u.n = 0,  |u.n| = 0            -> 0
   //
   // so the functional evaluates to alpha/2 - beta exactly.
   const int order = GENERATE(1, 2);
   const real_t alpha = GENERATE(2.0, -1.0);
   const real_t beta = GENERATE(0.0, 0.5);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();
   CAPTURE(order, alpha, beta);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   ConstantCoefficient f(1.0);
   Vector uv(dim);
   uv(0) = 1.0;
   uv(1) = 0.0;
   VectorConstantCoefficient u(uv);

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(new BoundaryNormalFlowIntegrator(f, u, alpha, beta));
   b.Assemble();

   // w = (x, 0), which any order >= 1 space represents exactly.
   VectorFunctionCoefficient wcoeff(dim, [](const Vector &x, Vector &w)
   {
      w = 0.0;
      w(0) = x(0);
   });
   GridFunction w(&fes);
   w.ProjectCoefficient(wcoeff);

   const real_t value = b * w;
   REQUIRE(value == MFEM_Approx(0.5 * alpha - beta, 1e-11, 1e-10));
}

TEST_CASE("DGBdrDisplacementLFIntegrator block structure and values",
          "[LinearFormIntegrator]")
{
   using namespace lininteg_boundary;

   // The integrator writes 1 + dim(dim+1)/2 blocks: a scalar block carrying
   // (vf . n), then the symmetric tensor blocks, diagonal and off-diagonal in
   // the order (0,0), (0,1), ..., (1,1), ...
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   const real_t sign = GENERATE(1.0, -1.0);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);
   CAPTURE(dim, order, sign);

   const int dim_lame = 1 + dim * (dim + 1) / 2;

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes_scalar(&mesh, &fec);
   FiniteElementSpace fes(&mesh, &fec, dim_lame);
   const int nd = fes_scalar.GetNDofs();

   Vector fvv(dim);
   for (int d = 0; d < dim; d++) { fvv(d) = 1.0 + 0.5 * d; }
   VectorConstantCoefficient vf(fvv);

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(new DGBdrDisplacementLFIntegrator(vf, sign));
   b.Assemble();

   REQUIRE(b.Size() == nd * dim_lame);

   // The scalar block is the trace of the tensor blocks: (vf . n) is the sum
   // over d of vf(d) n(d), and the diagonal blocks carry exactly those terms.
   // The diagonal block for direction d sits at offset 1 + d*dim - d*(d-1)/2.
   for (int i = 0; i < nd; i++)
   {
      real_t trace = 0.0;
      int doff = 1;
      for (int di = 0; di < dim; di++)
      {
         trace += b[doff * nd + i];
         doff += dim - di;   // skip this diagonal and its off-diagonals
      }
      CAPTURE(i);
      REQUIRE(b[i] == MFEM_Approx(trace, 1e-11, 1e-10));
   }

   // Over a closed boundary with a constant coefficient, the scalar block
   // integrated against the constant function is the divergence-theorem
   // integral of a constant field, which vanishes. This is the check that the
   // outward normals are consistently oriented.
   GridFunction one(&fes);
   one = 0.0;
   for (int i = 0; i < nd; i++) { one(i) = 1.0; }   // constant in block 0 only

   // A partition of unity in H1 has coefficients all equal to one at the
   // nodes, so 'one' is the constant function in the scalar block.
   const real_t closed = b * one;
   REQUIRE(std::abs(closed) < 1e-11);

   // Flipping the sign flips the whole vector.
   LinearForm b_neg(&fes);
   b_neg.AddBdrFaceIntegrator(new DGBdrDisplacementLFIntegrator(vf, -sign));
   b_neg.Assemble();
   for (int i = 0; i < b.Size(); i++)
   {
      CAPTURE(i);
      REQUIRE(b_neg[i] == MFEM_Approx(-b[i], 1e-11, 1e-10));
   }
}
