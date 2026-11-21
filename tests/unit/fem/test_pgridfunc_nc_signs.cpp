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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>

using namespace mfem;

#ifdef MFEM_USE_MPI

// ND and RT spaces flip the sign of some DOFs on faces shared between
// processors. On nonconforming meshes these signs are not part of the local
// DOFs, unlike on conforming meshes. These tests use a row of hexahedra, one per
// processor, where the two processors see each shared face with different
// orientations, and linear fields, which the spaces contain. So every result
// must be exact, up to rounding.
namespace pgridfunc_nc_signs
{

enum class MeshType { CONFORMING, NONCONFORMING, REFINED };

// With interior_bdr, every interior face is also a boundary element. REFINED
// refines all elements, then one element on the even processors, so the
// processors share conforming faces and hanging faces.
std::unique_ptr<ParMesh> MakeMesh(MeshType type, bool interior_bdr)
{
   const int num_procs = Mpi::WorldSize();
   Mesh mesh = Mesh::MakeCartesian3D(num_procs, 1, 1, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0, false);
   if (interior_bdr)
   {
      const int attr = mesh.bdr_attributes.Max() + 1;
      Array<int> v(4);
      for (int i = 0; i < mesh.GetNumFaces(); i++)
      {
         if (mesh.FaceIsInterior(i))
         {
            mesh.GetFaceVertices(i, v);
            mesh.AddBdrQuad(v, attr);
         }
      }
      mesh.FinalizeMesh();
      mesh.SetAttributes();
   }
   if (type != MeshType::CONFORMING) { mesh.EnsureNCMesh(true); }

   Array<int> partitioning(num_procs);
   for (int i = 0; i < num_procs; i++) { partitioning[i] = i; }
   auto pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh, partitioning);
   if (type == MeshType::REFINED)
   {
      pmesh->UniformRefinement();
      Array<int> refs;
      if (Mpi::WorldRank() % 2 == 0) { refs.Append(0); }
      pmesh->GeneralRefinement(refs, 1);
   }
   return pmesh;
}

void LinearField(const Vector &x, Vector &v)
{
   v(0) = 0.3 + x(1) - 0.5 * x(2);
   v(1) = x(0) + 0.2 * x(2);
   v(2) = 0.1 - x(0) + 0.7 * x(1);
}

// The curl of LinearField
void LinearFieldCurl(const Vector &, Vector &v)
{
   v(0) = 0.5;
   v(1) = 0.5;
   v(2) = 0.0;
}

// Global max of |a(i) - b(i)| over the local DOFs, or the marked ones
real_t MaxDiff(const ParGridFunction &a, const ParGridFunction &b,
               const Array<int> *marker = nullptr)
{
   real_t diff = 0.0;
   for (int i = 0; i < a.Size(); i++)
   {
      if (!marker || (*marker)[i])
      {
         diff = std::max(diff, std::abs(a(i) - b(i)));
      }
   }
   real_t glob_diff = 0.0;
   MPI_Allreduce(&diff, &glob_diff, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);
   return glob_diff;
}

const real_t tol = 100 * std::numeric_limits<real_t>::epsilon();

} // namespace pgridfunc_nc_signs

using namespace pgridfunc_nc_signs;

TEST_CASE("ProjectBdrCoefficientTangent with DOF signs", "[Parallel][NCMesh]")
{
   const auto type = GENERATE(MeshType::CONFORMING, MeshType::NONCONFORMING,
                              MeshType::REFINED);
   const int order = GENERATE(2, 3);
   CAPTURE(int(type), order, Mpi::WorldSize());

   auto pmesh = MakeMesh(type, true);
   ND_FECollection fec(order, 3);
   ParFiniteElementSpace fes(pmesh.get(), &fec);
   Array<int> bdr(pmesh->bdr_attributes.Max());
   bdr = 1;
   VectorFunctionCoefficient coeff(3, LinearField);

   ParGridFunction gf(&fes), exact(&fes);
   gf = 0.0;
   gf.ProjectBdrCoefficientTangent(coeff, bdr);
   exact.ProjectCoefficient(coeff);

   Array<int> ess_vdofs;
   fes.GetEssentialVDofs(bdr, ess_vdofs);
   REQUIRE(MaxDiff(gf, exact, &ess_vdofs) == MFEM_Approx(0.0, tol, tol));
}

TEST_CASE("ProjectDiscCoefficient with DOF signs", "[Parallel][NCMesh]")
{
   const auto type = GENERATE(MeshType::CONFORMING, MeshType::NONCONFORMING,
                              MeshType::REFINED);
   const bool nd = GENERATE(true, false);
   CAPTURE(int(type), nd, Mpi::WorldSize());

   auto pmesh = MakeMesh(type, false);
   std::unique_ptr<FiniteElementCollection> fec;
   if (nd) { fec.reset(new ND_FECollection(2, 3)); }
   else { fec.reset(new RT_FECollection(1, 3)); }
   ParFiniteElementSpace fes(pmesh.get(), fec.get());
   VectorFunctionCoefficient coeff(3, LinearField);

   ParGridFunction gf(&fes), exact(&fes);
   gf.ProjectDiscCoefficient(coeff, GridFunction::ARITHMETIC);
   exact.ProjectCoefficient(coeff);
   REQUIRE(MaxDiff(gf, exact) == MFEM_Approx(0.0, tol, tol));
}

TEST_CASE("ProjectCoefficient ELEMENT_L2 with DOF signs", "[Parallel][NCMesh]")
{
   const auto type = GENERATE(MeshType::CONFORMING, MeshType::NONCONFORMING,
                              MeshType::REFINED);
   CAPTURE(int(type), Mpi::WorldSize());

   auto pmesh = MakeMesh(type, false);
   ND_FECollection fec(2, 3);
   ParFiniteElementSpace fes(pmesh.get(), &fec);
   VectorFunctionCoefficient coeff(3, LinearField);

   ParGridFunction gf(&fes), exact(&fes);
   gf.ProjectCoefficient(coeff, ProjectType::ELEMENT_L2);
   exact.ProjectCoefficient(coeff);
   REQUIRE(MaxDiff(gf, exact) == MFEM_Approx(0.0, tol, tol));
}

TEST_CASE("ComputeFlux with DOF signs", "[Parallel][NCMesh]")
{
   const auto type = GENERATE(MeshType::CONFORMING, MeshType::NONCONFORMING,
                              MeshType::REFINED);
   CAPTURE(int(type), Mpi::WorldSize());

   auto pmesh = MakeMesh(type, false);
   ND_FECollection nd_fec(2, 3);
   ParFiniteElementSpace nd_fes(pmesh.get(), &nd_fec);
   RT_FECollection rt_fec(0, 3);
   ParFiniteElementSpace rt_fes(pmesh.get(), &rt_fec);

   VectorFunctionCoefficient coeff(3, LinearField);
   ParGridFunction u(&nd_fes);
   u.ProjectCoefficient(coeff);

   CurlCurlIntegrator curlcurl;
   ParGridFunction flux(&rt_fes), exact(&rt_fes);
   u.ComputeFlux(curlcurl, flux, false);
   VectorFunctionCoefficient curl(3, LinearFieldCurl);
   exact.ProjectCoefficient(curl);
   REQUIRE(MaxDiff(flux, exact) == MFEM_Approx(0.0, tol, tol));
}

TEST_CASE("Save and SaveAsOne with DOF signs", "[Parallel][NCMesh]")
{
   const auto type = GENERATE(MeshType::CONFORMING, MeshType::NONCONFORMING,
                              MeshType::REFINED);
   CAPTURE(int(type), Mpi::WorldSize());

   auto pmesh = MakeMesh(type, false);
   ND_FECollection fec(2, 3);
   ParFiniteElementSpace fes(pmesh.get(), &fec);
   VectorFunctionCoefficient coeff(3, LinearField);
   ParGridFunction gf(&fes);
   gf.ProjectCoefficient(coeff);

   // Read the files back as serial meshes and grid functions, as GLVis does.
   const int precision = std::numeric_limits<real_t>::max_digits10;
   std::stringstream mesh_str, gf_str;
   mesh_str.precision(precision);
   gf_str.precision(precision);
   pmesh->Print(mesh_str);
   gf.Save(gf_str);
   Mesh local_mesh(mesh_str, 1, 1);
   GridFunction local_gf(&local_mesh, gf_str);
   real_t error = local_gf.ComputeL2Error(coeff), glob_error = 0.0;
   MPI_Allreduce(&error, &glob_error, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);
   CHECK(glob_error == MFEM_Approx(0.0, tol, tol));

   std::ostringstream mesh_one, gf_one;
   mesh_one.precision(precision);
   gf_one.precision(precision);
   pmesh->PrintAsOne(mesh_one);
   gf.SaveAsOne(gf_one);
   error = 0.0;
   if (Mpi::Root())
   {
      std::istringstream mesh_in(mesh_one.str()), gf_in(gf_one.str());
      Mesh serial_mesh(mesh_in, 1, 1);
      GridFunction serial_gf(&serial_mesh, gf_in);
      error = serial_gf.ComputeL2Error(coeff);
   }
   MPI_Bcast(&error, 1, MFEM_MPI_REAL_T, 0, MPI_COMM_WORLD);
   CHECK(error == MFEM_Approx(0.0, tol, tol));
}

#endif // MFEM_USE_MPI
