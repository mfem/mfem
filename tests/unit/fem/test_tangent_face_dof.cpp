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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <cmath>
#include <limits>

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace tangent_face_dof
{

enum class MeshType { TET, PRISM, MIXED };

Mesh MakeMesh(MeshType type)
{
   Mesh mesh;
   switch (type)
   {
      case MeshType::TET:
         mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, false);
         break;
      case MeshType::PRISM:
         mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::WEDGE,
                                      1.0, 1.0, 1.0, false);
         break;
      case MeshType::MIXED:
         // Tetrahedra, prisms and hexahedra
         mesh = Mesh("../../data/fichera-mixed.mesh");
         mesh.UniformRefinement();
         break;
   }

   // Make every interior face a boundary element, so that the interior DOFs of
   // the shared faces are projected.
   const int attr = mesh.bdr_attributes.Max() + 1;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      if (mesh.FaceIsInterior(f))
      {
         Element *be = mesh.GetFace(f)->Duplicate(&mesh);
         be->SetAttribute(attr);
         mesh.AddBdrElement(be);
      }
   }
   mesh.FinalizeMesh();
   mesh.SetAttributes();
   return mesh;
}

// A nonlinear field, which the ND spaces of order 2 and 3 do not contain.
void Field(const Vector &x, Vector &v)
{
   v(0) = std::sin(1.3 * x(1)) * x(2) + 0.5;
   v(1) = x(0) * x(0) - std::cos(0.7 * x(2));
   v(2) = std::exp(0.5 * x(0)) * x(1);
}

} // namespace tangent_face_dof

TEST_CASE("ProjectBdrCoefficientTangent face DOF orientation",
          "[Parallel][ND][DofTransformation]")
{
   using namespace tangent_face_dof;

   const int num_procs = Mpi::WorldSize();
   const auto type = GENERATE(MeshType::TET, MeshType::PRISM, MeshType::MIXED);
   const int order = GENERATE(2, 3);
   CAPTURE(int(type), order, num_procs);

   Mesh smesh = MakeMesh(type);
   Array<int> bdr(smesh.bdr_attributes.Max());
   bdr = 1;
   VectorFunctionCoefficient coeff(3, Field);

   ND_FECollection fec(order, 3);
   FiniteElementSpace sfes(&smesh, &fec);
   GridFunction sgf(&sfes);
   sgf = 0.0;
   sgf.ProjectBdrCoefficientTangent(coeff, bdr);

   // Contiguous blocks of elements, so that the parts share triangular faces
   // on every mesh.
   Array<int> partitioning(smesh.GetNE());
   for (int e = 0; e < smesh.GetNE(); e++)
   {
      partitioning[e] = (e * num_procs) / smesh.GetNE();
   }
   ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   REQUIRE(pfes.SharedNDTriangleDofs() == (num_procs > 1));

   ParGridFunction pgf(&pfes);
   pgf = 0.0;
   pgf.ProjectBdrCoefficientTangent(coeff, bdr);

   ParGridFunction expected(&pmesh, &sgf, partitioning);
   pgf -= expected;
   const real_t error = GlobalLpNorm(infinity(), pgf.Normlinf(),
                                     MPI_COMM_WORLD);
   const real_t tol = 100 * std::numeric_limits<real_t>::epsilon();
   REQUIRE(error == MFEM_Approx(0.0, tol, tol));
}

#endif
