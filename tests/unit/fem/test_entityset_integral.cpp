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

using namespace mfem;

namespace
{

int AddElementSet(Mesh &mesh, const std::string &name, bool all_elements)
{
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   const int set = sets.AddSet(EntitySets::ELEMENT, name);
   const int count = all_elements ? mesh.GetNE() : 1;
   for (int i = 0; i < count; i++)
   {
      sets.AddEntity(EntitySets::ELEMENT, set, i);
   }
   return set;
}

int FindVertex(const Mesh &mesh, real_t x, real_t y, real_t z = 0.0)
{
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      const real_t *point = mesh.GetVertex(vertex);
      if (point[0] == x && point[1] == y &&
          (mesh.SpaceDimension() < 3 || point[2] == z))
      {
         return vertex;
      }
   }
   return -1;
}

} // namespace

TEST_CASE("L2 element-set integral", "[EntitySets][L2_FECollection]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false,
                                     1.0, 1.0);
   const int set = AddElementSet(mesh, "Domain", true);
   const EntitySets &sets = *mesh.ent_sets;

   L2_FECollection fec(2, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   FunctionCoefficient value([](const Vector &x) { return 1.0 + x[0] + x[1]; });
   field.ProjectCoefficient(value);

   L2ElementSetIntegral by_index(field, sets, set);
   L2ElementSetIntegral by_name(field, sets, "Domain");
   REQUIRE(by_index.Eval() == MFEM_Approx(2.0));
   REQUIRE(by_name.Eval() == MFEM_Approx(2.0));
}

TEST_CASE("L2 element-set integral selects only set members",
          "[EntitySets][L2_FECollection]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   const int set = AddElementSet(mesh, "Left", false);
   const EntitySets &sets = *mesh.ent_sets;

   L2_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   ConstantCoefficient value(3.0);
   field.ProjectCoefficient(value);

   L2ElementSetIntegral integral(field, sets, set);
   REQUIRE(integral.Eval() == MFEM_Approx(3.0));
}

TEST_CASE("HCurl edge-path integral", "[EntitySets][ND_FECollection]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL, false,
                                     2.0, 1.0);
   ND_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   VectorFunctionCoefficient value(2, [](const Vector &, Vector &v)
   {
      v.SetSize(2);
      v[0] = 1.0;
      v[1] = 0.0;
   });
   field.ProjectCoefficient(value);

   const int left = FindVertex(mesh, 0.0, 0.0);
   const int right = FindVertex(mesh, 2.0, 0.0);
   REQUIRE(left >= 0);
   REQUIRE(right >= 0);
   EdgePath forward(mesh, left, right);
   EdgePath reverse(mesh, right, left);

   REQUIRE(HCurlEdgePathIntegral(field, forward).Eval() == MFEM_Approx(2.0));
   REQUIRE(HCurlEdgePathIntegral(field, reverse).Eval() == MFEM_Approx(-2.0));
}

TEST_CASE("HDiv oriented-face-set integral", "[EntitySets][RT_FECollection]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                                     2.0, 1.0, 1.0);
   EntitySets &sets = mesh.GetOrCreateEntitySets();
   const int left = sets.AddElementSet("Left", Array<int>({0}));
   const int right = sets.AddElementSet("Right", Array<int>({1}));
   OrientedFaceSet interface(mesh, sets, left, right);

   RT_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction field(&fes);
   VectorFunctionCoefficient value(3, [](const Vector &, Vector &v)
   {
      v.SetSize(3);
      v[0] = 1.0;
      v[1] = 0.0;
      v[2] = 0.0;
   });
   field.ProjectCoefficient(value);

   REQUIRE(interface.Size() == 1);
   REQUIRE(HDivFaceSetIntegral(field, interface).Eval() == MFEM_Approx(1.0));
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel L2 element-set integral",
          "[EntitySets][L2_FECollection][Parallel]")
{
   Mesh serial_mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                            false, 2.0, 2.0);
   AddElementSet(serial_mesh, "Domain", true);
   int nranks;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);
   Array<int> partitioning(serial_mesh.GetNE());
   for (int i = 0; i < partitioning.Size(); i++)
   {
      partitioning[i] = i % nranks;
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   REQUIRE(mesh.ent_sets != nullptr);

   L2_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   ConstantCoefficient value(2.0);
   field.ProjectCoefficient(value);

   const EntitySets &sets = *mesh.ent_sets;
   L2ElementSetIntegral integral(field, sets, "Domain");
   REQUIRE(integral.Eval() == MFEM_Approx(8.0));
}

TEST_CASE("Parallel HCurl edge-path integral",
          "[EntitySets][ND_FECollection][Parallel]")
{
   int nranks, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Mesh serial_mesh = Mesh::MakeCartesian2D(nranks, 1, Element::QUADRILATERAL,
                                            false, real_t(nranks), 1.0);
   Array<int> partitioning(serial_mesh.GetNE());
   for (int element = 0; element < partitioning.Size(); element++)
   {
      partitioning[element] = element;
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   ND_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   VectorFunctionCoefficient value(2, [](const Vector &, Vector &v)
   {
      v.SetSize(2);
      v[0] = 1.0;
      v[1] = 1.0;
   });
   field.ProjectCoefficient(value);

   EdgePath path;
   real_t expected = 0.0;
   if (rank == 0)
   {
      Array<int> edges({0}), orientations({1}), vertices;
      mesh.GetEdgeVertices(edges[0], vertices);
      const real_t *start = mesh.GetVertex(vertices[0]);
      const real_t *end = mesh.GetVertex(vertices[1]);
      expected = end[0] - start[0] + end[1] - start[1];
      path.SetEdges(mesh, edges, orientations);
   }
   else
   {
      path.SetWaypoints(mesh, Array<int>({0}));
   }

   REQUIRE(HCurlEdgePathIntegral(field, path).Eval() == MFEM_Approx(expected));
}

TEST_CASE("Parallel HDiv oriented-face-set integral",
          "[EntitySets][RT_FECollection][Parallel]")
{
   int nranks, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Mesh serial_mesh = Mesh::MakeCartesian3D(nranks, 1, 1,
                                            Element::HEXAHEDRON,
                                            real_t(nranks), 1.0, 1.0);
   Array<int> partitioning(serial_mesh.GetNE());
   for (int element = 0; element < partitioning.Size(); element++)
   {
      partitioning[element] = element;
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   RT_FECollection fec(0, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);
   ParGridFunction field(&fes);
   VectorFunctionCoefficient value(3, [](const Vector &, Vector &v)
   {
      v.SetSize(3);
      v[0] = 1.0;
      v[1] = 0.0;
      v[2] = 0.0;
   });
   field.ProjectCoefficient(value);

   Array<int> faces, signs;
   if (rank == 0)
   {
      for (int face = 0; face < mesh.GetNumFaces(); face++)
      {
         int element1, element2;
         mesh.GetFaceElements(face, &element1, &element2);
         if (element2 >= 0) { continue; }

         ElementTransformation *trans = mesh.GetFaceTransformation(face);
         const IntegrationPoint &center = Geometries.GetCenter(
                                             mesh.GetFaceGeometry(face));
         trans->SetIntPoint(&center);
         Vector normal(mesh.SpaceDimension());
         CalcOrtho(trans->Jacobian(), normal);
         if (normal[0] == 0.0) { continue; }
         faces.Append(face);
         signs.Append(normal[0] > 0.0 ? 1 : -1);
         break;
      }
   }
   OrientedFaceSet face_set(mesh, faces, signs);

   REQUIRE(HDivFaceSetIntegral(field, face_set).Eval() == MFEM_Approx(1.0));
}

#endif
