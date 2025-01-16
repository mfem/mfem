// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

void TestSameMatrices(SparseMatrix &A1, const SparseMatrix &A2,
                      HYPRE_BigInt *cmap1=nullptr,
                      std::unordered_map<HYPRE_BigInt,int> *cmap2inv=nullptr)
{
   REQUIRE(A1.Height() == A2.Height());
   int n = A1.Height();

   const int *I1 = A1.HostReadI();
   const int *J1 = A1.HostReadJ();
   const real_t *V1 = A1.HostReadData();

   A2.HostReadI();
   A2.HostReadJ();
   A2.HostReadData();

   real_t error = 0.0;

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         if (cmap1)
         {
            if (cmap2inv->count(cmap1[j]) > 0)
            {
               j = (*cmap2inv)[cmap1[j]];
            }
            else
            {
               error = std::max(error, std::fabs(V1[jj]));
               continue;
            }
         }
         error = std::max(error, std::fabs(V1[jj] - A2(i,j)));
      }
   }

   REQUIRE(error == MFEM_Approx(0.0, 1e-10));
}

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

template <typename FE_COLL, typename INTEG_1, typename INTEG_2>
void TestBatchedLOR()
{
   const int order = 5;
   const auto mesh_fname = GENERATE(
                              "../../data/star-surf.mesh",
                              "../../data/star-q3.mesh",
                              "../../data/fichera-q3.mesh"
                           );

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);

   std::unique_ptr<FE_COLL> fec(
      NewLOR_FE_Collection<FE_COLL>(order, mesh.Dimension()));
   FiniteElementSpace fespace(&mesh, fec.get());

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   // Test variable coefficients using grid functions defined on a H1 space
   H1_FECollection h1fec(2, mesh.Dimension());
   FiniteElementSpace h1fes(&mesh, &h1fec);
   GridFunction gf1(&h1fes), gf2(&h1fes);
   gf1.Randomize(1);
   gf2.Randomize(2);

   GridFunctionCoefficient mass_coeff(&gf1);
   GridFunctionCoefficient diff_coeff(&gf2);

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new INTEG_1(mass_coeff));
   a.AddDomainIntegrator(new INTEG_2(diff_coeff));

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

TEST_CASE("LOR Batched H1", "[LOR][BatchedLOR][CUDA]")
{
   TestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>();
}

TEST_CASE("LOR Batched ND", "[LOR][BatchedLOR][CUDA]")
{
   TestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>();
}

TEST_CASE("LOR Batched RT", "[LOR][BatchedLOR][CUDA]")
{
   TestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>();
}

#ifdef MFEM_USE_MPI

void TestSameMatrices(HypreParMatrix &A1, const HypreParMatrix &A2)
{
   HYPRE_BigInt *cmap1, *cmap2;
   SparseMatrix diag1, offd1, diag2, offd2;

   A1.GetDiag(diag1);
   A2.GetDiag(diag2);
   A1.GetOffd(offd1, cmap1);
   A2.GetOffd(offd2, cmap2);

   TestSameMatrices(diag1, diag2);

   if (cmap1)
   {
      std::unordered_map<HYPRE_BigInt,int> cmap2inv;
      for (int i=0; i<offd2.Width(); ++i) { cmap2inv[cmap2[i]] = i; }
      TestSameMatrices(offd1, offd2, cmap1, &cmap2inv);
   }
   else
   {
      TestSameMatrices(offd1, offd2);
   }
}

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

TEST_CASE("Parallel LOR Batched H1", "[LOR][BatchedLOR][Parallel][CUDA]")
{
   ParTestBatchedLOR<H1_FECollection,MassIntegrator,DiffusionIntegrator>();
}

TEST_CASE("Parallel LOR Batched ND", "[LOR][BatchedLOR][Parallel][CUDA]")
{
   ParTestBatchedLOR<ND_FECollection,VectorFEMassIntegrator,CurlCurlIntegrator>();
}

TEST_CASE("Parallel LOR Batched RT", "[LOR][BatchedLOR][Parallel][CUDA]")
{
   ParTestBatchedLOR<RT_FECollection,VectorFEMassIntegrator,DivDivIntegrator>();
}

TEST_CASE("LOR AMS", "[LOR][BatchedLOR][AMS][Parallel][CUDA]")
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

TEST_CASE("LOR ADS", "[LOR][BatchedLOR][ADS][Parallel][CUDA]")
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
