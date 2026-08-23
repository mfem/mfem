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

TEST_CASE("NonlinearForm integrator accessors", "[NonlinearForm]")
{
   // The accessors exist so that a wrapper -- DarcyForm among them -- can walk
   // the integrators a caller added. The contract worth pinning is the one the
   // documentation states and that is easy to get wrong: an integrator added
   // without a marker reports a null marker, not an all-ones array, and the
   // markers line up index for index with the integrators.
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   NonlinearForm nf(&fes);

   REQUIRE(nf.GetInteriorFaceIntegrators().Size() == 0);
   REQUIRE(nf.GetBdrFaceIntegrators().Size() == 0);
   REQUIRE(nf.GetBdrFaceIntegratorsMarkers().Size() == 0);

   ConstantCoefficient one(1.0);
   NonlinearFormIntegrator *f0 = new MassIntegrator(one);
   NonlinearFormIntegrator *b0 = new MassIntegrator(one);
   NonlinearFormIntegrator *b1 = new MassIntegrator(one);

   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[0] = 1;

   nf.AddInteriorFaceIntegrator(f0);
   nf.AddBdrFaceIntegrator(b0);            // no marker
   nf.AddBdrFaceIntegrator(b1, marker);    // with a marker

   REQUIRE(nf.GetInteriorFaceIntegrators().Size() == 1);
   REQUIRE(nf.GetInteriorFaceIntegrators()[0] == f0);

   REQUIRE(nf.GetBdrFaceIntegrators().Size() == 2);
   REQUIRE(nf.GetBdrFaceIntegrators()[0] == b0);
   REQUIRE(nf.GetBdrFaceIntegrators()[1] == b1);

   REQUIRE(nf.GetBdrFaceIntegratorsMarkers().Size() == 2);
   REQUIRE(nf.GetBdrFaceIntegratorsMarkers()[0] == nullptr);
   REQUIRE(nf.GetBdrFaceIntegratorsMarkers()[1] == &marker);
}

TEST_CASE("NonlinearForm::Update follows its space", "[NonlinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   H1_FECollection fec(2, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   NonlinearForm nf(&fes);
   REQUIRE(nf.Height() == fes.GetTrueVSize());

   const int before = nf.Height();

   mesh.UniformRefinement();
   fes.Update();
   nf.Update();

   REQUIRE(nf.Height() == fes.GetTrueVSize());
   REQUIRE(nf.Height() > before);
   REQUIRE(nf.Width() == nf.Height());
}

TEST_CASE("BlockNonlinearForm integrator accessors",
          "[NonlinearForm][BlockNonlinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int dim = mesh.Dimension();

   RT_FECollection u_coll(0, dim);
   L2_FECollection p_coll(0, dim);
   FiniteElementSpace fes_u(&mesh, &u_coll), fes_p(&mesh, &p_coll);

   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &fes_u;
   spaces[1] = &fes_p;

   BlockNonlinearForm bnf(spaces);

   REQUIRE(bnf.GetDomainIntegrators().Size() == 0);
   REQUIRE(bnf.GetBoundaryIntegrators().Size() == 0);
   REQUIRE(bnf.GetInteriorFaceIntegrators().Size() == 0);
   REQUIRE(bnf.GetBdrFaceIntegrators().Size() == 0);

   ConstantCoefficient k(1.0);
   LinearDiffusionFlux flux(dim, k);

   BlockNonlinearFormIntegrator *d0 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *d1 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *n0 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *n1 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *f0 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *g0 = new MixedConductionNLFIntegrator(flux);
   BlockNonlinearFormIntegrator *g1 = new MixedConductionNLFIntegrator(flux);

   Array<int> dmarker(mesh.attributes.Max());
   dmarker = 1;
   Array<int> bmarker(mesh.bdr_attributes.Max());
   bmarker = 0;
   bmarker[0] = 1;
   Array<int> fmarker(mesh.bdr_attributes.Max());
   fmarker = 0;
   fmarker[1] = 1;

   bnf.AddDomainIntegrator(d0);
   bnf.AddDomainIntegrator(d1, dmarker);
   bnf.AddBoundaryIntegrator(n0);
   bnf.AddBoundaryIntegrator(n1, bmarker);
   bnf.AddInteriorFaceIntegrator(f0);
   bnf.AddBdrFaceIntegrator(g0);
   bnf.AddBdrFaceIntegrator(g1, fmarker);

   REQUIRE(bnf.GetDomainIntegrators().Size() == 2);
   REQUIRE(bnf.GetDomainIntegrators()[0] == d0);
   REQUIRE(bnf.GetDomainIntegrators()[1] == d1);
   REQUIRE(bnf.GetDomainIntegratorsMarkers().Size() == 2);
   REQUIRE(bnf.GetDomainIntegratorsMarkers()[0] == nullptr);
   REQUIRE(bnf.GetDomainIntegratorsMarkers()[1] == &dmarker);

   REQUIRE(bnf.GetBoundaryIntegrators().Size() == 2);
   REQUIRE(bnf.GetBoundaryIntegrators()[0] == n0);
   REQUIRE(bnf.GetBoundaryIntegrators()[1] == n1);
   REQUIRE(bnf.GetBoundaryIntegratorsMarkers()[0] == nullptr);
   REQUIRE(bnf.GetBoundaryIntegratorsMarkers()[1] == &bmarker);

   REQUIRE(bnf.GetInteriorFaceIntegrators().Size() == 1);
   REQUIRE(bnf.GetInteriorFaceIntegrators()[0] == f0);

   REQUIRE(bnf.GetBdrFaceIntegrators().Size() == 2);
   REQUIRE(bnf.GetBdrFaceIntegrators()[0] == g0);
   REQUIRE(bnf.GetBdrFaceIntegrators()[1] == g1);
   REQUIRE(bnf.GetBdrFaceIntegratorsMarkers()[0] == nullptr);
   REQUIRE(bnf.GetBdrFaceIntegratorsMarkers()[1] == &fmarker);
}
