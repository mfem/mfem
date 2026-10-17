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

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel L2 element-set integral", "[EntitySets][L2_FECollection][Parallel]")
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

#endif
