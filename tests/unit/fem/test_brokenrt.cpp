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

#include <string>

using namespace mfem;

namespace brokenrt
{

Mesh MakeMesh(int dim, Element::Type type)
{
   return (dim == 2)
          ? Mesh::MakeCartesian2D(3, 2, type, false, 1.0, 1.0)
          : Mesh::MakeCartesian3D(2, 2, 2, type, 1.0, 1.0, 1.0);
}

} // namespace brokenrt

TEST_CASE("BrokenRT_FECollection is RT with nothing shared",
          "[FiniteElementCollection][BrokenRT]")
{
   using namespace brokenrt;

   const int dim = GENERATE(2, 3);
   const int p = GENERATE(0, 1, 2);
   const Element::Type type = (dim == 2) ? Element::QUADRILATERAL
                              : Element::HEXAHEDRON;

   CAPTURE(dim, p);

   Mesh mesh = MakeMesh(dim, type);

   RT_FECollection       rt(p, dim);
   BrokenRT_FECollection brt(p, dim);

   // The whole point of the collection: the same element, no continuity.
   REQUIRE(rt.GetContType() == FiniteElementCollection::NORMAL);
   REQUIRE(brt.GetContType() == FiniteElementCollection::DISCONTINUOUS);

   FiniteElementSpace fes_rt(&mesh, &rt);
   FiniteElementSpace fes_brt(&mesh, &brt);

   const int ndof_el = fes_rt.GetFE(0)->GetDof();
   REQUIRE(fes_brt.GetFE(0)->GetDof() == ndof_el);

   // Broken: every DOF belongs to exactly one element.
   REQUIRE(fes_brt.GetVSize() == mesh.GetNE() * ndof_el);

   // Unbroken: the normal components on interior faces are shared, so the
   // space is strictly smaller. (At p = 0 on this mesh there are still
   // interior faces, so the inequality is strict for every order here.)
   REQUIRE(fes_rt.GetVSize() < fes_brt.GetVSize());

   // The difference must be exactly the DOFs on interior faces.
   int shared = 0;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      if (mesh.FaceIsInterior(f))
      {
         Array<int> fdofs;
         fes_rt.GetFaceDofs(f, fdofs);
         shared += fdofs.Size();
      }
   }
   REQUIRE(fes_brt.GetVSize() - fes_rt.GetVSize() == shared);
}

TEST_CASE("BrokenRT_FECollection round-trips through its name",
          "[FiniteElementCollection][BrokenRT]")
{
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(0, 1, 3);

   CAPTURE(dim, p);

   SECTION("default bases")
   {
      BrokenRT_FECollection brt(p, dim);
      const std::string name(brt.Name());
      REQUIRE(name == "BRT_" + std::to_string(dim) + "D_P" + std::to_string(p));

      FiniteElementCollection *back = FiniteElementCollection::New(name.c_str());
      REQUIRE(back != nullptr);
      REQUIRE(std::string(back->Name()) == name);
      REQUIRE(back->GetContType() == FiniteElementCollection::DISCONTINUOUS);
      delete back;
   }

   SECTION("non-default bases")
   {
      // Exercises the other half of the name format, and with it the other
      // offset arithmetic in FiniteElementCollection::New().
      BrokenRT_FECollection brt(p, dim, BasisType::GaussLobatto,
                                BasisType::GaussLobatto);
      const std::string name(brt.Name());
      REQUIRE(name.rfind("BRT@", 0) == 0);

      FiniteElementCollection *back = FiniteElementCollection::New(name.c_str());
      REQUIRE(back != nullptr);
      REQUIRE(std::string(back->Name()) == name);
      delete back;
   }
}

TEST_CASE("BrokenRT_FECollection supplies a trace collection",
          "[FiniteElementCollection][BrokenRT]")
{
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(0, 1, 2);

   CAPTURE(dim, p);

   BrokenRT_FECollection brt(p, dim);
   FiniteElementCollection *trace = brt.GetTraceCollection();
   REQUIRE(trace != nullptr);

   // A trace of a broken space still has to carry one normal component per
   // face DOF of the unbroken space, or hybridization has nothing to constrain.
   RT_FECollection rt(p, dim);
   FiniteElementCollection *rt_trace = rt.GetTraceCollection();
   REQUIRE(rt_trace != nullptr);

   const Geometry::Type face_geom = (dim == 2) ? Geometry::SEGMENT
                                    : Geometry::SQUARE;
   REQUIRE(trace->DofForGeometry(face_geom) ==
           rt_trace->DofForGeometry(face_geom));

   delete rt_trace;
   delete trace;
}

TEST_CASE("BrokenRT_FECollection registers everything on the element",
          "[FiniteElementCollection][BrokenRT]")
{
   // BrokenRT derives from the protected RT_FECollection constructor that
   // registers no face DOFs, then reports the whole element instead. That
   // constructor is not public, so what is checked here is the observable
   // consequence: nothing on the faces, everything on the element.
   const int dim = GENERATE(2, 3);
   const int p = GENERATE(0, 1, 2);

   CAPTURE(dim, p);

   RT_FECollection rt(p, dim);
   BrokenRT_FECollection broken(p, dim);

   const Geometry::Type face_geom = (dim == 2) ? Geometry::SEGMENT
                                    : Geometry::SQUARE;
   const Geometry::Type el_geom = (dim == 2) ? Geometry::SQUARE
                                  : Geometry::CUBE;

   REQUIRE(rt.DofForGeometry(face_geom) > 0);
   REQUIRE(broken.DofForGeometry(face_geom) == 0);

   // Everything the RT element has, counted on the element itself: the
   // interior DOFs plus the face DOFs the unbroken space would share.
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);
   FiniteElementSpace fes_rt(&mesh, &rt);
   REQUIRE(broken.DofForGeometry(el_geom) == fes_rt.GetFE(0)->GetDof());
   REQUIRE(broken.DofForGeometry(el_geom) > rt.DofForGeometry(el_geom));
}
