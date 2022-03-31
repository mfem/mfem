// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
   const double *V1 = A1.HostReadData();

   A2.HostReadI();
   A2.HostReadJ();
   A2.HostReadData();

   double error = 0.0;

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

TEST_CASE("LOR Batched H1", "[LOR][BatchedLOR][CUDA]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const int order = !all_tests ? 5 : GENERATE(1,3,5);

   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   ConstantCoefficient diff_coeff(M_PI);
   ConstantCoefficient mass_coeff(1.0/M_PI);

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   a.AddDomainIntegrator(new MassIntegrator(mass_coeff));

   LORDiscretization lor(fespace);

   // Sanity check that the LOR mesh is valid
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(mesh.GetElementGeometry(0), 1);
   const GeometricFactors::FactorFlags dets = GeometricFactors::DETERMINANTS;
   REQUIRE(lor.GetFESpace().GetMesh()->GetGeometricFactors(ir, dets)->detJ.Min()
           > 0.0);

   lor.LegacyAssembleSystem(a, ess_dofs);
   SparseMatrix A1 = lor.GetAssembledMatrix(); // deep copy
   lor.AssembleSystem(a, ess_dofs);
   SparseMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
}

TEST_CASE("LOR Batched ND", "[LOR][BatchedLOR][CUDA]")
{
   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh mesh = Mesh::LoadFromFile(mesh_fname);
   ND_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   ConstantCoefficient diff_coeff(M_PI);
   ConstantCoefficient mass_coeff(1.0/M_PI);

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(diff_coeff));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   LORDiscretization lor(fespace);

   lor.LegacyAssembleSystem(a, ess_dofs);
   SparseMatrix A1 = lor.GetAssembledMatrix(); // deep copy
   lor.AssembleSystem(a, ess_dofs);
   SparseMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);
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

TEST_CASE("Parallel LOR Batched H1", "[LOR][BatchedLOR][Parallel][CUDA]")
{
   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   ConstantCoefficient diff_coeff(M_PI);
   ConstantCoefficient mass_coeff(1.0/M_PI);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   a.AddDomainIntegrator(new MassIntegrator(mass_coeff));
   ParLORDiscretization lor(fespace);

   lor.LegacyAssembleSystem(a, ess_dofs);
   HypreParMatrix A1 = lor.GetAssembledMatrix(); // deep copy
   lor.AssembleSystem(a, ess_dofs);
   HypreParMatrix &A2 = lor.GetAssembledMatrix();

   TestSameMatrices(A1, A2);
   TestSameMatrices(A2, A1);

   lor.GetAssembledSystem().Clear();
}

TEST_CASE("LOR AMS", "[LOR][BatchedLOR][AMS][Parallel][CUDA]")
{
   auto mesh_fname = GENERATE(
                        "../../data/star-q3.mesh",
                        "../../data/fichera-q3.mesh"
                     );
   const int order = 5;

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   const int dim = mesh.Dimension();

   ND_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   ParLORDiscretization lor(fespace);
   ParFiniteElementSpace &edge_fespace = lor.GetParFESpace();

   H1_FECollection vert_fec(1, dim);
   ParFiniteElementSpace vert_fespace(edge_fespace.GetParMesh(), &vert_fec);

   ParDiscreteLinearOperator grad(&vert_fespace, &edge_fespace);
   grad.AddDomainInterpolator(new GradientInterpolator);
   grad.Assemble();
   grad.Finalize();
   std::unique_ptr<HypreParMatrix> G(grad.ParallelAssemble());

   ConstantCoefficient diff_coeff(M_PI);
   ConstantCoefficient mass_coeff(1.0/M_PI);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(diff_coeff));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   BatchedLOR_AMS batched_lor(a, fespace, ess_dofs);

   TestSameMatrices(*G, *batched_lor.GetGradientMatrix());

   ParGridFunction x_coord(&vert_fespace);
   ParGridFunction y_coord(&vert_fespace);
   ParGridFunction z_coord(&vert_fespace);
   for (int i = 0; i < edge_fespace.GetMesh()->GetNV(); i++)
   {
      const double *coord = edge_fespace.GetMesh()->GetVertex(i);
      x_coord(i) = coord[0];
      y_coord(i) = coord[1];
      if (dim == 3) { z_coord(i) = coord[2]; }
   }
   std::unique_ptr<HypreParVector> x(x_coord.ParallelProject());
   std::unique_ptr<HypreParVector> y(y_coord.ParallelProject());
   std::unique_ptr<HypreParVector> z;
   if (dim == 3) { z.reset(z_coord.ParallelProject()); }

   *x -= *batched_lor.GetXCoordinate();
   REQUIRE(x->Normlinf() == MFEM_Approx(0.0));
   *y -= *batched_lor.GetYCoordinate();
   REQUIRE(y->Normlinf() == MFEM_Approx(0.0));
   if (dim == 3)
   {
      *z -= *batched_lor.GetZCoordinate();
      REQUIRE(z->Normlinf() == MFEM_Approx(0.0));
   }
}

#endif // MFEM_USE_MPI

} // namespace lor_batched
