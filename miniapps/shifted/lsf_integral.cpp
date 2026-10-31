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
//

#include "mfem.hpp"
#include <cmath>
#include <iomanip>
#include <memory>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   int rank = 0, ranks = 1;
#ifdef MFEM_CUT_INTEGRAL_PARALLEL
   Mpi::Init(argc, argv);
   rank = Mpi::WorldRank();
   ranks = Mpi::WorldSize();
#endif
   const char *mesh_file = "";
   const char *algorithm = "algoim";
   int dimension = 2, cells = 2, refinements = 0, degree = 2, order = 10, shape = 1;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Optional quadrilateral/hexahedral mesh.");
   args.AddOption(&algorithm, "-cq", "--cut-quadrature",
                  "algoim, algoim-general, or algoim-polynomial.");
   args.AddOption(&dimension, "-dim", "--dimension", "Generated unit box dimension.");
   args.AddOption(&cells, "-n", "--cells", "Generated cells per coordinate axis.");
   args.AddOption(&refinements, "-rs", "--refine-serial", "Serial uniform refinements.");
   args.AddOption(&degree, "-o", "--order", "Polynomial level-set degree.");
   args.AddOption(&order, "-ao", "--aorder", "Nominal cut quadrature order.");
   args.AddOption(&shape, "-ls", "--level-set",
                  "1: sphere, 2: plane x = 0.37, 3: two-plane volume.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Display the projected level set.");
   args.Parse();
   if (!args.Good() || (dimension != 2 && dimension != 3) || cells < 1 ||
       degree < 1 || refinements < 0 || shape < 1 || shape > 3)
   {
      if (rank == 0) { args.PrintUsage(cout); }
      return 1;
   }
   auto backend = CreateCutQuadratureBackend(algorithm);
   if (!backend)
   {
      if (rank == 0) { cerr << "Requested cut quadrature backend is unavailable.\n"; }
      return MFEM_SKIP_RETURN_VALUE;
   }
   Mesh serial = mesh_file[0] ? Mesh(mesh_file, 1, 1) :
                 (dimension == 2 ?
                  Mesh::MakeCartesian2D(cells, cells, Element::QUADRILATERAL) :
                  Mesh::MakeCartesian3D(cells, cells, cells, Element::HEXAHEDRON));
   dimension = serial.Dimension();
   if (dimension != 2 && dimension != 3) { return 1; }
   for (int level = 0; level < refinements; ++level) { serial.UniformRefinement(); }
   Mesh *mesh = &serial;
#ifdef MFEM_CUT_INTEGRAL_PARALLEL
   Array<int> partition(serial.GetNE());
   for (int element = 0; element < partition.Size(); ++element)
   { partition[element] = element % ranks; }
   ParMesh parallel(MPI_COMM_WORLD, serial, partition.GetData());
   mesh = &parallel;
#endif
   FunctionCoefficient coefficient([shape](const Vector &point)
   {
      if (shape == 2) { return real_t(0.37) - point(0); }
      if (shape == 3) { return point(0) - real_t(0.3); }
      real_t value = 0.32 * 0.32;
      for (int axis = 0; axis < point.Size(); ++axis)
      { value -= (point(axis) - 0.5) * (point(axis) - 0.5); }
      return value;
   });
   FunctionCoefficient second_coefficient([](const Vector &point)
   { return point(1) - real_t(0.4); });
   L2_FECollection collection(degree, dimension);
   CutQuadratureGenerator generator(*backend);
   CutMappingOptions mapping;
   mapping.weight_type = CutWeightType::Physical;
   real_t measures[2] = {0, 0};
   int failed = 0;
   string message;
   for (int element = 0; element < mesh->GetNE() && !failed; ++element)
   {
      auto &transformation = *mesh->GetElementTransformation(element);
      const auto *finite_element = collection.FiniteElementForGeometry(
                                      transformation.GetGeometryType());
      PolynomialLevelSet level_set, second_level_set;
      auto report = level_set.Project(coefficient, transformation, *finite_element);
      if (shape == 3 && report.Succeeded())
      { report = second_level_set.Project(second_coefficient, transformation, *finite_element); }
      Array<const ReferenceLevelSet *> level_sets(shape == 3 ? 2 : 1);
      level_sets[0] = &level_set;
      if (shape == 3) { level_sets[1] = &second_level_set; }
      for (int surface = 0; surface < (shape == 3 ? 1 : 2) && report.Succeeded(); ++surface)
      {
         CutQuadratureRequest request;
         request.order = order;
         request.domain = surface ? CutDomain::Surface() : CutDomain::Positive();
         if (shape == 3) { request.domain.Add(1, LevelSetSign::Positive); }
         MappedCutQuadrature quadrature;
         report = generator.Generate(transformation, level_sets, request,
                                     mapping, quadrature);
         for (int index = 0; index < quadrature.rule.GetNPoints(); ++index)
         { measures[surface] += quadrature.rule.IntPoint(index).weight; }
      }
      if (!report.Succeeded()) { failed = 1; message = report.message; }
   }
   if (failed) { cerr << "Rank " << rank << ": " << message << '\n'; }
#ifdef MFEM_CUT_INTEGRAL_PARALLEL
   MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, measures, 2, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
#endif
   if (failed) { return 2; }
   if (!mesh_file[0] && shape == 3 && std::abs(measures[0] - 0.42) > 1e-10)
   {
      if (rank == 0) { cerr << "Two-plane volume check failed.\n"; }
      return 3;
   }
   if (!mesh_file[0] && shape == 2 &&
       (std::abs(measures[0] - 0.37) > 1e-10 ||
        std::abs(measures[1] - 1.0) > 1e-10))
   {
      if (rank == 0) { cerr << "Unit-box plane integral check failed.\n"; }
      return 3;
   }
   if (rank == 0)
   {
      cout << setprecision(16) << "Ranks: " << ranks << '\n'
           << "Backend: " << backend->Name() << '\n'
           << "Volume: " << measures[0] << '\n';
      if (shape != 3) { cout << "Surface: " << measures[1] << '\n'; }
   }
   if (visualization)
   {
      FiniteElementSpace space(mesh, &collection);
      GridFunction field(&space);
      field.ProjectCoefficient(coefficient);
      socketstream socket("localhost", 19916);
      if (ranks > 1) { socket << "parallel " << ranks << ' ' << rank << '\n'; }
      socket << "solution\n" << *mesh << field << flush;
   }
   return 0;
}
