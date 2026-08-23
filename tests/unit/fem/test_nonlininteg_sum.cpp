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

namespace nonlininteg_sum
{

typedef NonlinearFormIntegrator NLFI;

void FillVarying(Vector &v, real_t shift)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = std::sin(1.7 * i + shift) + 0.5 * std::cos(0.3 * i);
   }
}

void RequireEqual(const Vector &a, const Vector &b, real_t tol = 1e-11)
{
   REQUIRE(a.Size() == b.Size());
   for (int i = 0; i < a.Size(); i++)
   {
      CAPTURE(i, a(i), b(i));
      REQUIRE(a(i) == MFEM_Approx(b(i), tol, 10 * tol));
   }
}

void RequireEqual(const DenseMatrix &a, const DenseMatrix &b,
                  real_t tol = 1e-11)
{
   REQUIRE(a.Height() == b.Height());
   REQUIRE(a.Width() == b.Width());
   for (int i = 0; i < a.Height(); i++)
      for (int j = 0; j < a.Width(); j++)
      {
         CAPTURE(i, j, a(i, j), b(i, j));
         REQUIRE(a(i, j) == MFEM_Approx(b(i, j), tol, 10 * tol));
      }
}

} // namespace nonlininteg_sum

TEST_CASE("SumNLFIntegrator adds element contributions", "[SumNLFIntegrator]")
{
   using namespace nonlininteg_sum;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);

   Vector elfun(el.GetDof());
   FillVarying(elfun, 0.0);

   ConstantCoefficient m(2.5), d(0.75);

   // The sum owns its members; the standalone integrators for comparison are
   // separate instances.
   SumNLFIntegrator sum;
   sum.AddIntegrator(new MassIntegrator(m));
   sum.AddIntegrator(new DiffusionIntegrator(d));

   MassIntegrator mass(m);
   DiffusionIntegrator diff(d);

   SECTION("AssembleElementVector")
   {
      Vector v_sum, v_a, v_b;
      sum.AssembleElementVector(el, Tr, elfun, v_sum);
      mass.AssembleElementVector(el, Tr, elfun, v_a);
      diff.AssembleElementVector(el, Tr, elfun, v_b);

      v_a += v_b;
      RequireEqual(v_sum, v_a);
   }

   SECTION("AssembleElementGrad")
   {
      DenseMatrix g_sum, g_a, g_b;
      sum.AssembleElementGrad(el, Tr, elfun, g_sum);
      mass.AssembleElementGrad(el, Tr, elfun, g_a);
      diff.AssembleElementGrad(el, Tr, elfun, g_b);

      g_a += g_b;
      RequireEqual(g_sum, g_a);
   }

   SECTION("a sum of one is that one")
   {
      SumNLFIntegrator single;
      single.AddIntegrator(new MassIntegrator(m));

      Vector v_single, v_mass;
      single.AssembleElementVector(el, Tr, elfun, v_single);
      mass.AssembleElementVector(el, Tr, elfun, v_mass);
      RequireEqual(v_single, v_mass);
   }
}

TEST_CASE("SumNLFIntegrator adds HDG face contributions", "[SumNLFIntegrator]")
{
   using namespace nonlininteg_sum;

   // This is the combination the HDG assembly actually uses: a diffusion
   // stabilization and a convective one on the same face.
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1, 2);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);

   L2_FECollection el_coll(order, dim);
   DG_Interface_FECollection tr_coll(order, dim);
   FiniteElementSpace fes_el(&mesh, &el_coll), fes_tr(&mesh, &tr_coll);

   int f = -1;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      if (mesh.FaceIsInterior(i)) { f = i; break; }
   }
   REQUIRE(f >= 0);

   FaceElementTransformations *Tr = mesh.GetFaceElementTransformations(f);
   const FiniteElement &tr_fe = *fes_tr.GetFaceElement(f);
   const FiniteElement &el_fe = *fes_el.GetFE(Tr->Elem1No);

   Vector elfun(el_fe.GetDof()), trfun(tr_fe.GetDof());
   FillVarying(elfun, 0.0);
   FillVarying(trfun, 1.1);

   ConstantCoefficient q(2.5);
   Vector vel(dim);
   vel = 0.0;
   vel(0) = 1.3;
   if (dim > 1) { vel(1) = -0.7; }
   VectorConstantCoefficient vcoeff(vel);

   SumNLFIntegrator sum;
   sum.AddIntegrator(new HDGDiffusionIntegrator(q));
   sum.AddIntegrator(new HDGConvectionUpwindedIntegrator(vcoeff));

   HDGDiffusionIntegrator a(q);
   HDGConvectionUpwindedIntegrator b(vcoeff);

   const int masks[5] =
   {
      NLFI::ELEM,
      NLFI::TRACE,
      NLFI::CONSTR | NLFI::FACE,
      NLFI::ELEM | NLFI::TRACE,
      NLFI::ELEM | NLFI::TRACE | NLFI::CONSTR | NLFI::FACE
   };

   for (int side = 0; side < 2; side++)
   {
      for (int k = 0; k < 5; k++)
      {
         const int type = masks[k] | (side & 1);
         CAPTURE(side, type);

         Vector v_sum, v_a, v_b;
         sum.AssembleHDGFaceVector(type, tr_fe, el_fe, *Tr, trfun, elfun, v_sum);
         a.AssembleHDGFaceVector(type, tr_fe, el_fe, *Tr, trfun, elfun, v_a);
         b.AssembleHDGFaceVector(type, tr_fe, el_fe, *Tr, trfun, elfun, v_b);
         v_a += v_b;
         RequireEqual(v_sum, v_a);

         DenseMatrix g_sum, g_a, g_b;
         sum.AssembleHDGFaceGrad(type, tr_fe, el_fe, *Tr, trfun, elfun, g_sum);
         a.AssembleHDGFaceGrad(type, tr_fe, el_fe, *Tr, trfun, elfun, g_a);
         b.AssembleHDGFaceGrad(type, tr_fe, el_fe, *Tr, trfun, elfun, g_b);
         g_a += g_b;
         RequireEqual(g_sum, g_a);
      }
   }
}

TEST_CASE("SumNLFIntegrator ownership", "[SumNLFIntegrator]")
{
   using namespace nonlininteg_sum;

   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   H1_FECollection fec(1, 2);
   FiniteElementSpace fes(&mesh, &fec);
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);

   Vector elfun(el.GetDof());
   FillVarying(elfun, 0.0);
   ConstantCoefficient m(1.0);

   // own_integs = 0: the members outlive the sum, and are ours to delete.
   MassIntegrator *mass = new MassIntegrator(m);
   {
      SumNLFIntegrator borrowed(0);
      borrowed.AddIntegrator(mass);
      Vector v;
      borrowed.AssembleElementVector(el, Tr, elfun, v);
      REQUIRE(v.Size() == el.GetDof());
   }
   // Still alive: using it here would crash if the sum had deleted it.
   Vector v_after;
   mass->AssembleElementVector(el, Tr, elfun, v_after);
   REQUIRE(v_after.Size() == el.GetDof());
   delete mass;

   // own_integs = 1 (the default): the sum deletes what it is given, so
   // nothing is deleted here.
   {
      SumNLFIntegrator owning;
      owning.AddIntegrator(new MassIntegrator(m));
   }
}

TEST_CASE("SumBlockNLFIntegrator adds block contributions",
          "[SumNLFIntegrator][BlockNonlinearForm]")
{
   using namespace nonlininteg_sum;

   const int dim = GENERATE(2, 3);
   const int order = GENERATE(0, 1);
   CAPTURE(dim, order);

   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);

   RT_FECollection u_coll(order, dim);
   L2_FECollection p_coll(order, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll), fes_p(&mesh, &p_coll);

   const FiniteElement *fe_u = fes_u.GetFE(0);
   const FiniteElement *fe_p = fes_p.GetFE(0);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);

   Array<const FiniteElement *> el(2);
   el[0] = fe_u;
   el[1] = fe_p;

   Vector eu(fe_u->GetDof()), ep(fe_p->GetDof());
   FillVarying(eu, 0.3);
   FillVarying(ep, 2.1);
   Array<const Vector *> elfun(2);
   elfun[0] = &eu;
   elfun[1] = &ep;

   ConstantCoefficient k1(1.5), k2(0.4);
   LinearDiffusionFlux flux1(dim, k1), flux2(dim, k2);

   SumBlockNLFIntegrator sum;
   sum.AddIntegrator(new MixedConductionNLFIntegrator(flux1));
   sum.AddIntegrator(new MixedConductionNLFIntegrator(flux2));

   MixedConductionNLFIntegrator a(flux1), b(flux2);

   auto MakeBlocks = [&](Array<Vector *> &blocks, Vector &b0, Vector &b1)
   {
      b0.SetSize(fe_u->GetDof());
      b1.SetSize(fe_p->GetDof());
      b0 = 0.0;
      b1 = 0.0;
      blocks.SetSize(2);
      blocks[0] = &b0;
      blocks[1] = &b1;
   };

   Vector s0, s1, a0, a1, b0, b1;
   Array<Vector *> vs, va, vb;
   MakeBlocks(vs, s0, s1);
   MakeBlocks(va, a0, a1);
   MakeBlocks(vb, b0, b1);

   sum.AssembleElementVector(el, Tr, elfun, vs);
   a.AssembleElementVector(el, Tr, elfun, va);
   b.AssembleElementVector(el, Tr, elfun, vb);

   a0 += b0;
   a1 += b1;
   RequireEqual(s0, a0);
   RequireEqual(s1, a1);
}
