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
#include "../linalg/test_same_matrices.hpp"
#include "../../fem/lor/lor_ads.hpp"
#include "../../fem/lor/lor_ams.hpp"
#include <memory>
#include <unordered_map>

using namespace mfem;

#ifndef MFEM_USE_MPI
#define HYPRE_BigInt int
#endif // MFEM_USE_MPI

namespace lor_batched
{

template <typename FE_COLL>
FE_COLL *NewLOR_FE_Collection(int order, int dim)
{
   return new FE_COLL(order, dim);
}

template <>
ND_FECollection *NewLOR_FE_Collection<ND_FECollection>(int order, int dim)
{
   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   return new ND_FECollection(order, dim, b1, b2);
}

template <>
RT_FECollection *NewLOR_FE_Collection<RT_FECollection>(int order, int dim)
{
   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   return new RT_FECollection(order-1, dim, b1, b2);
}

std::function<real_t(const Vector &)> coeff_fn(int dim)
{
   if (dim == 2)
   {
      return [](const Vector &x)
      {
         return sin(8.0 * M_PI * x[0]) * cos(6.0 * M_PI * x[1]) + 2.0;
      };
   }
   else if (dim == 3)
   {
      return [](const Vector &x)
      {
         return sin(8.0 * M_PI * x[0]) * cos(6.0 * M_PI * x[1]) *
                sin(4.0 * M_PI * x[2]) + 2.0;
      };
   }
   MFEM_ABORT("Unsupported dimension.");
}

std::function<void(const Vector &, Vector &)> vector_fn(int dim)
{
   if (dim == 2)
   {
      return [](const Vector &x, Vector &f)
      {
         f[0] = sin(M_PI * x[1]);
         f[1] = sin(2.5 * M_PI * x[0]);
      };
   }
   else if (dim == 3)
   {
      return [](const Vector &x, Vector &f)
      {
         f[0] = sin(M_PI * x[1]);
         f[1] = sin(2.5 * M_PI * x[0]);
         f[2] = sin(6.1 * M_PI * x[2]);
      };
   }
   MFEM_ABORT("Unsupported dimension.");
}

std::function<void(const Vector &, DenseMatrix &)> matrix_fn(int dim)
{
   if (dim == 2)
   {
      return [](const Vector &x, DenseMatrix &f)
      {
         f = 0.0;
         f(0,0) = 1.1 + sin(M_PI * x[1]);  //
         f(0,1) = f(1,0) = cos(2.5 * M_PI * x[0]);
         f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);
      };
   }
   else if (dim == 3)
   {
      return [](const Vector &x, DenseMatrix &f)
      {
         f = 0.0;
         f(0,0) = sin(M_PI * x[1]);
         f(0,1) = f(1,0) = cos(2.5 * M_PI * x[0]);
         f(0,2) = f(2,0) = sin(4.9 * M_PI * x[2]);
         f(1,1) = sin(6.1 * M_PI * x[1]);
         f(1,2) = f(2,1) = cos(6.1 * M_PI * x[2]);
         f(2,2) = sin(6.1 * M_PI * x[2]);
      };
   }
   MFEM_ABORT("Unsupported dimension.");
}

std::unique_ptr<Coefficient> make_scalar_coeff(int dim)
{
   return std::make_unique<FunctionCoefficient>(coeff_fn(dim));
}

std::unique_ptr<VectorCoefficient> make_vector_coeff(int dim)
{
   return std::make_unique<VectorFunctionCoefficient>(dim, vector_fn(dim));
}

std::unique_ptr<MatrixCoefficient> make_matrix_coeff(int dim)
{
   return std::make_unique<MatrixFunctionCoefficient>(dim, matrix_fn(dim));
}

template <typename FE_COLL, typename INTEG_1, typename INTEG_2,
          typename MK_COEFF_1, typename MK_COEFF_2>
void TestBatchedLOR(MK_COEFF_1 mk_coeff_1, MK_COEFF_2 mk_coeff_2)
{
   const int order = 5;
   const auto mesh_fname = GENERATE(
                              "../../data/star-surf.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera-q3.mesh"
                           );
   CAPTURE(mesh_fname);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);

   std::unique_ptr<FE_COLL> fec(
      NewLOR_FE_Collection<FE_COLL>(order, mesh.Dimension()));
   FiniteElementSpace fespace(&mesh, fec.get());

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   auto coeff_1 = mk_coeff_1(mesh.SpaceDimension());
   auto coeff_2 = mk_coeff_2(mesh.SpaceDimension());

   // curl-curl can accept matrix coefficients in 3D only
   if (std::is_same_v<INTEG_2,CurlCurlIntegrator> &&
       mesh.Dimension() == 2 &&
       !std::is_base_of_v<Coefficient, typename std::pointer_traits<decltype(coeff_2)>::element_type>)
   {
      return; // skip
   }

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new INTEG_1(*coeff_1));
   a.AddDomainIntegrator(new INTEG_2(*coeff_2));

   LORDiscretization lor(fespace);

   // Sanity check that the LOR mesh is valid
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh.GetTypicalElementGeometry(), 1);
   const GeometricFactors::FactorFlags dets = GeometricFactors::DETERMINANTS;
   if (mesh.Dimension() == mesh.SpaceDimension())
   {
      REQUIRE(
         lor.GetFESpace().GetMesh()->GetGeometricFactors(ir, dets)->detJ.Min() > 0.0);
   }

   lor.LegacyAssembleSystem(a, ess_dofs);
   SparseMatrix A1 = lor.GetAssembledMatrix(); // deep copy
   lor.AssembleSystem(a, ess_dofs);
   SparseMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}

TEST_CASE("LOR Batched H1", "[LOR][BatchedLOR][GPU]")
{
   TestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>(
      make_scalar_coeff, make_scalar_coeff);
   TestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>(
      make_scalar_coeff, make_vector_coeff);
   TestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>(
      make_scalar_coeff, make_matrix_coeff);
}

TEST_CASE("LOR Batched ND", "[LOR][BatchedLOR][GPU]")
{
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_scalar_coeff, make_scalar_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_vector_coeff, make_scalar_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_matrix_coeff, make_scalar_coeff);

   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_scalar_coeff, make_vector_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_vector_coeff, make_vector_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_matrix_coeff, make_vector_coeff);

   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_scalar_coeff, make_matrix_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_vector_coeff, make_matrix_coeff);
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>(
      make_matrix_coeff, make_matrix_coeff);
}

TEST_CASE("LOR Batched RT", "[LOR][BatchedLOR][GPU]")
{
   TestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>(
      make_scalar_coeff, make_scalar_coeff);
   TestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>(
      make_vector_coeff, make_scalar_coeff);
   TestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>(
      make_matrix_coeff, make_scalar_coeff);
}

#ifdef MFEM_USE_MPI

template <typename FE_COLL, typename INTEG_1, typename INTEG_2>
void ParTestBatchedLOR()
{
   const bool all_tests = launch_all_non_regression_tests;
   const int order = !all_tests ? 5 : GENERATE(1,3,5);
   const auto mesh_fname = GENERATE(
                              "../../data/star-surf.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera-q3.mesh"
                           );

   CAPTURE(order, mesh_fname);

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   std::unique_ptr<FE_COLL> fec(NewLOR_FE_Collection<FE_COLL>(order,
                                                              mesh.Dimension()));
   ParFiniteElementSpace fespace(&mesh, fec.get());

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   H1_FECollection h1fec(2, mesh.Dimension());
   FiniteElementSpace h1fes(&mesh, &h1fec);
   GridFunction gf1(&h1fes), gf2(&h1fes);
   gf1.Randomize(1);
   gf2.Randomize(2);

   GridFunctionCoefficient mass_coeff(&gf1);
   GridFunctionCoefficient diff_coeff(&gf2);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new INTEG_1(diff_coeff));
   a.AddDomainIntegrator(new INTEG_2(mass_coeff));
   ParLORDiscretization lor(fespace);

   lor.LegacyAssembleSystem(a, ess_dofs);
   HypreParMatrix A1 = lor.GetAssembledMatrix(); // deep copy
   lor.AssembleSystem(a, ess_dofs);
   HypreParMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}

TEST_CASE("Parallel LOR Batched H1", "[LOR][BatchedLOR][Parallel][GPU]")
{
   ParTestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>();
}

TEST_CASE("Parallel LOR Batched ND", "[LOR][BatchedLOR][Parallel][GPU]")
{
   ParTestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>();
}

TEST_CASE("Parallel LOR Batched RT", "[LOR][BatchedLOR][Parallel][GPU]")
{
   ParTestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>();
}

TEST_CASE("LOR AMS", "[LOR][BatchedLOR][AMS][Parallel][GPU]")
{
   enum SpaceType { ND, RT };
   auto space_type = GENERATE(ND, RT);
   auto mesh_fname = GENERATE(
                        "../../data/star-surf.mesh",
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   // Only test RT spaces in 2D
   if (space_type == RT && dim == 3) { return; }

   std::unique_ptr<FiniteElementCollection> fec;
   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   if (space_type == ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }

   ParFiniteElementSpace fespace(&mesh, fec.get());

   ParLORDiscretization lor(fespace);
   ParFiniteElementSpace &edge_fespace = lor.GetParFESpace();

   H1_FECollection vert_fec(1, dim);
   ParFiniteElementSpace vert_fespace(edge_fespace.GetParMesh(), &vert_fec);

   ParDiscreteLinearOperator grad(&vert_fespace, &edge_fespace);
   grad.AddDomainInterpolator(new GradientInterpolator);
   grad.Assemble();
   grad.Finalize();
   std::unique_ptr<HypreParMatrix> G(grad.ParallelAssemble());

   Vector X_vert;
   BatchedLORAssembly::FormLORVertexCoordinates(fespace, X_vert);
   BatchedLOR_AMS batched_lor(fespace, X_vert);

   TestSameMatrices(*G, *batched_lor.GetGradientMatrix());

   ParGridFunction x_coord(&vert_fespace);
   ParGridFunction y_coord(&vert_fespace);
   ParGridFunction z_coord(&vert_fespace);
   for (int i = 0; i < edge_fespace.GetMesh()->GetNV(); i++)
   {
      const real_t *coord = edge_fespace.GetMesh()->GetVertex(i);
      x_coord(i) = coord[0];
      y_coord(i) = coord[1];
      if (sdim == 3) { z_coord(i) = coord[2]; }
   }
   std::unique_ptr<HypreParVector> x(x_coord.ParallelProject());
   std::unique_ptr<HypreParVector> y(y_coord.ParallelProject());
   std::unique_ptr<HypreParVector> z;
   if (sdim == 3) { z.reset(z_coord.ParallelProject()); }

   *x -= *batched_lor.GetXCoordinate();
   REQUIRE(x->Normlinf() == MFEM_Approx(0.0));
   *y -= *batched_lor.GetYCoordinate();
   REQUIRE(y->Normlinf() == MFEM_Approx(0.0));
   if (sdim == 3)
   {
      *z -= *batched_lor.GetZCoordinate();
      REQUIRE(z->Normlinf() == MFEM_Approx(0.0));
   }
}

TEST_CASE("LOR ADS", "[LOR][BatchedLOR][ADS][Parallel][GPU]")
{
   // Only need to test ADS in 3D
   auto mesh_fname = GENERATE("../../data/fichera-q3.mesh");
   const int order = 5;

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   const int dim = mesh.Dimension();

   RT_FECollection fec(order-1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fespace(&mesh, &fec);

   ND_FECollection fec_nd(order, dim, BasisType::GaussLobatto,
                          BasisType::IntegratedGLL);
   ParFiniteElementSpace fespace_nd(&mesh, &fec_nd);

   // Note: the LOR fespaces include the DOF permutations built into R and P
   ParLORDiscretization lor_face(fespace);
   ParFiniteElementSpace &face_fespace = lor_face.GetParFESpace();

   ParLORDiscretization lor_edge(fespace_nd);
   ParFiniteElementSpace &edge_fespace = lor_edge.GetParFESpace();

   ParDiscreteLinearOperator curl(&edge_fespace, &face_fespace);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();
   std::unique_ptr<HypreParMatrix> C(curl.ParallelAssemble());

   Vector X_vert;
   BatchedLORAssembly::FormLORVertexCoordinates(fespace, X_vert);
   BatchedLOR_ADS batched_lor(fespace, X_vert);

   TestSameMatrices(*C, *batched_lor.GetCurlMatrix());
}

#endif // MFEM_USE_MPI

} // namespace lor_batched
