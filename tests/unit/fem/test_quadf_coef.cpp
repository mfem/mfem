// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "catch.hpp"

using namespace mfem;

namespace qf_coeff
{

TEST_CASE("Quadrature Function Coefficients",
          "[Quadrature Function Coefficients]")
{
   int order_h1 = 2, n = 4, dim = 3;
   double tol = 1e-9;

   Mesh mesh(n, n, n, Element::HEXAHEDRON, false, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order_h1);

   int intOrder = 2 * order_h1 + 1;

   QuadratureSpace qspace(&mesh, intOrder);
   QuadratureFunction quadf_coeff(&qspace, 1);
   QuadratureFunction quadf_vcoeff(&qspace, dim);

   const IntegrationRule ir = qspace.GetElementIntRule(0);

   const GeometricFactors *geom_facts = mesh.GetGeometricFactors(ir,
                                                                 GeometricFactors::COORDINATES);

   {
      int nelems = quadf_coeff.Size() / quadf_coeff.GetVDim() / ir.GetNPoints();
      int vdim = ir.GetNPoints();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            //X has dims nqpts x sdim x ne
            quadf_coeff((i * vdim) + j) = geom_facts->X((i * vdim * dim) + (vdim * 2) + j );
         }
      }
   }

   {
      int nqpts = ir.GetNPoints();
      int nelems = quadf_vcoeff.Size() / quadf_vcoeff.GetVDim() / nqpts;
      int vdim = quadf_vcoeff.GetVDim();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            for (int k = 0; k < nqpts; k++)
            {
               //X has dims nqpts x sdim x ne
               quadf_vcoeff((i * nqpts * vdim) + (k * vdim ) + j) = geom_facts->X((
                                                                                     i * nqpts * vdim) + (j * nqpts) + k );
            }
         }
      }
   }

   QuadratureFunctionCoefficient qfc(&quadf_coeff);
   VectorQuadratureFunctionCoefficient qfvc(&quadf_vcoeff);

   H1_FECollection    fec_h1(order_h1, dim);
   FiniteElementSpace fespace_hv1(&mesh, &fec_h1, 1);
   FieldInterpolant fi(&fespace_hv1);
   fi.SetupCG();

   SECTION("Operators on VecQuadFuncCoeff")
   {
      std::cout << "Testing VecQuadFuncCoeff: " << std::endl;
#ifdef MFEM_USE_EXCEPTIONS
      std::cout << " Setting Component" << std::endl;
      REQUIRE_THROWS(qfvc.SetComponent(3, 1));
      REQUIRE_THROWS(qfvc.SetComponent(-1, 1));
      REQUIRE_NOTHROW(qfvc.SetComponent(1, 2));
      REQUIRE_THROWS(qfvc.SetComponent(0, 4));
      REQUIRE_THROWS(qfvc.SetComponent(1, 3));
      REQUIRE_NOTHROW(qfvc.SetComponent(0, 2));
      REQUIRE_THROWS(qfvc.SetComponent(0, 0));
#endif
      qfvc.SetComponent(0, 3);

      SECTION("Gridfunction L2 tests")
      {
         std::cout << "  Testing GridFunc L2 projection" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);

         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         {
            int ne = mesh.GetNE();

            GridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            int vdim = quadf_vcoeff.GetVDim();

            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               fespace_l2.GetElementVDofs(i, vdofs);
               gtrue.SetSubVector(vdofs, el_x.HostReadWrite());
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfvc, fespace_h1, fespace_l2);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }

      SECTION("Gridfunction L2 tests byVDIM")
      {
         std::cout << "  Testing GridFunc L2 projection byVDIM" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim, Ordering::byVDIM);
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim, Ordering::byVDIM);
         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         {
            int ne = mesh.GetNE();

            GridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            int vdim = quadf_vcoeff.GetVDim();

            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               fespace_l2.GetElementVDofs(i, vdofs);
               gtrue.SetSubVector(vdofs, el_x.HostReadWrite());
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfvc, fespace_h1, fespace_l2);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }

      SECTION("Gridfunction H1 tests")
      {
         std::cout << "  Testing GridFunc H1 projection" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);
         GridFunction g0(&fespace_h1);
         GridFunction gtrue(&fespace_h1);

         {
            GridFunction nodes(&fespace_h1);
            mesh.GetNodes(nodes);
            gtrue = nodes;
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfvc, fespace_h1);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }
      SECTION("Gridfunction H1 tests byVDIM")
      {
         std::cout << "  Testing GridFunc H1 projection byVDIM" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim, Ordering::byVDIM);
         GridFunction g0(&fespace_h1);
         GridFunction gtrue(&fespace_h1);

         {
            GridFunction nodes(&fespace_h1);
            mesh.GetNodes(nodes);
            gtrue = nodes;
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfvc, fespace_h1);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }
   }

   SECTION("Operators on QuadFuncCoeff")
   {
      SECTION("Gridfunction L2 tests")
      {
         std::cout << "Testing QuadFuncCoeff:";
         std::cout << "  Testing GridFunc L2 projection" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2, 1);
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);
         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         fi.SetupDiscReset();

         // When using an L2 FE space of the same order as the mesh, the below highlights
         // that the ProjectDiscCoeff method is just taking the quadrature point values and making them node values.
         {
            int ne = mesh.GetNE();

            GridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               int enodes = el_x.Size() / dim;
               for (int j = 0; j < enodes; j++)
               {
                  gtrue(j + i * enodes) = el_x(enodes * 2 + j);
               }
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfc, fespace_h1, fespace_l2);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }

      SECTION("Gridfunction H1 tests")
      {
         std::cout << "  Testing GridFunc H1 projection" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, 1);
         GridFunction g0(&fespace_h1);
         GridFunction gtrue(&fespace_h1);

         fi.FullReset();

         {
            int nnodes = gtrue.Size();
            int vdim = 1;

            Vector nodes;
            mesh.GetNodes(nodes);
            for (int i = 0; i < nnodes; i++)
            {
               gtrue(i) = nodes(i * dim + 2);
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfc, fespace_h1);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel Quadrature Function Coefficients",
          "[Parallel] , [Parallel Quadrature Function Coefficients]")
{
   int order_h1 = 2, n = 4, dim = 3;
   double tol = 1e-9;

   Mesh *tmesh = new Mesh(n, n, n, Element::HEXAHEDRON, false, 1.0, 1.0, 1.0);
   tmesh->SetCurvature(order_h1);
   ParMesh mesh(MPI_COMM_WORLD, *tmesh);

   delete tmesh;

   int intOrder = 2 * order_h1 + 1;

   QuadratureSpace qspace(&mesh, intOrder);
   QuadratureFunction quadf_coeff(&qspace, 1);
   QuadratureFunction quadf_vcoeff(&qspace, dim);

   const IntegrationRule ir = qspace.GetElementIntRule(0);

   const GeometricFactors *geom_facts = mesh.GetGeometricFactors(ir,
                                                                 GeometricFactors::COORDINATES);

   {
      int nelems = quadf_coeff.Size() / quadf_coeff.GetVDim() / ir.GetNPoints();
      int vdim = ir.GetNPoints();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            //X has dims nqpts x sdim x ne
            quadf_coeff((i * vdim) + j) = geom_facts->X((i * vdim * dim) + (vdim * 2) + j );
         }
      }
   }

   {
      int nqpts = ir.GetNPoints();
      int nelems = quadf_vcoeff.Size() / quadf_vcoeff.GetVDim() / nqpts;
      int vdim = quadf_vcoeff.GetVDim();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            for (int k = 0; k < nqpts; k++)
            {
               //X has dims nqpts x sdim x ne
               quadf_vcoeff((i * nqpts * vdim) + (k * vdim ) + j) = geom_facts->X((
                                                                                     i * nqpts * vdim) + (j * nqpts) + k );
            }
         }
      }
   }

   QuadratureFunctionCoefficient qfc(&quadf_coeff);
   VectorQuadratureFunctionCoefficient qfvc(&quadf_vcoeff);

   H1_FECollection    fec_h1(order_h1, dim);
   ParFiniteElementSpace fespace_hv1(&mesh, &fec_h1, 1);
   ParFieldInterpolant fi(&fespace_hv1);
   fi.SetupCG(MPI_COMM_WORLD);

   SECTION("Operators on VecQuadFuncCoeff")
   {
      std::cout << "Testing VecQuadFuncCoeff: " << std::endl;
#ifdef MFEM_USE_EXCEPTIONS
      std::cout << " Setting Component" << std::endl;
      REQUIRE_THROWS(qfvc.SetComponent(3, 1));
      REQUIRE_THROWS(qfvc.SetComponent(-1, 1));
      REQUIRE_NOTHROW(qfvc.SetComponent(1, 2));
      REQUIRE_THROWS(qfvc.SetComponent(0, 4));
      REQUIRE_THROWS(qfvc.SetComponent(1, 3));
      REQUIRE_NOTHROW(qfvc.SetComponent(0, 2));
      REQUIRE_THROWS(qfvc.SetComponent(0, 0));
#endif
      qfvc.SetComponent(0, 3);

      SECTION("Gridfunction L2 tests")
      {
         std::cout << "  Testing GridFunc L2 projection" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         ParFiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);
         ParGridFunction g0(&fespace_l2);
         ParGridFunction gtrue(&fespace_l2);

         {
            int ne = mesh.GetNE();

            ParGridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            int vdim = quadf_vcoeff.GetVDim();

            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               fespace_l2.GetElementVDofs(i, vdofs);
               gtrue.SetSubVector(vdofs, el_x.HostReadWrite());
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfvc, fespace_h1, fespace_l2);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }

      SECTION("Gridfunction L2 tests byVDIM")
      {
         std::cout << "  Testing GridFunc L2 projection byVDIM" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         ParFiniteElementSpace fespace_l2(&mesh, &fec_l2, dim, Ordering::byVDIM);
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, dim, Ordering::byVDIM);
         ParGridFunction g0(&fespace_l2);
         ParGridFunction gtrue(&fespace_l2);

         {
            int ne = mesh.GetNE();

            ParGridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            int vdim = quadf_vcoeff.GetVDim();

            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               fespace_l2.GetElementVDofs(i, vdofs);
               gtrue.SetSubVector(vdofs, el_x.HostReadWrite());
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfvc, fespace_h1, fespace_l2);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }

      SECTION("Gridfunction H1 tests")
      {
         std::cout << "  Testing GridFunc H1 projection" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);
         ParGridFunction g0(&fespace_h1);
         ParGridFunction gtrue(&fespace_h1);

         {
            ParGridFunction nodes(&fespace_h1);
            mesh.GetNodes(nodes);
            gtrue = nodes;
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfvc, fespace_h1);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }

      SECTION("Gridfunction H1 tests byVDIM")
      {
         std::cout << "  Testing GridFunc H1 projection byVDIM" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, dim, Ordering::byVDIM);
         ParGridFunction g0(&fespace_h1);
         ParGridFunction gtrue(&fespace_h1);

         fi.FullReset();

         {
            ParGridFunction nodes(&fespace_h1);
            mesh.GetNodes(nodes);
            gtrue = nodes;
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfvc, fespace_h1);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }
   }

   SECTION("Operators on QuadFuncCoeff")
   {
      SECTION("Gridfunction L2 tests")
      {
         std::cout << "Testing QuadFuncCoeff:";
         std::cout << "  Testing GridFunc L2 projection" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         ParFiniteElementSpace fespace_l2(&mesh, &fec_l2, 1);
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);
         ParGridFunction g0(&fespace_l2);
         ParGridFunction gtrue(&fespace_l2);

         fi.SetupDiscReset();

         // When using an L2 FE space of the same order as the mesh, the below highlights
         // that the ProjectDiscCoeff method is just taking the quadrature point values and making them node values.
         {
            int ne = mesh.GetNE();

            ParGridFunction nodes(&fespace_h1);
            Vector el_x;
            Array<int> vdofs;
            mesh.GetNodes(nodes);
            for (int i = 0; i < ne; i++)
            {
               fespace_h1.GetElementVDofs(i, vdofs);
               nodes.GetSubVector(vdofs, el_x);
               int enodes = el_x.Size() / dim;
               for (int j = 0; j < enodes; j++)
               {
                  gtrue(j + i * enodes) = el_x(enodes * 2 + j);
               }
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureDiscCoefficient(g0, qfc, fespace_h1, fespace_l2);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }

      SECTION("Gridfunction H1 tests")
      {
         std::cout << "  Testing GridFunc H1 projection" << std::endl;
         H1_FECollection    fec_h1(order_h1, dim);
         ParFiniteElementSpace fespace_h1(&mesh, &fec_h1, 1);
         ParGridFunction g0(&fespace_h1);
         ParGridFunction gtrue(&fespace_h1);

         {
            int nnodes = gtrue.Size();
            int vdim = 1;

            Vector nodes;
            mesh.GetNodes(nodes);
            for (int i = 0; i < nnodes; i++)
            {
               gtrue(i) = nodes(i * dim + 2);
            }
         }

         g0 = 0.0;
         fi.ProjectQuadratureCoefficient(g0, qfc, fespace_h1);
         gtrue -= g0;

         double lerr = gtrue.Norml2();
         double error = 0;

         MPI_Allreduce(&lerr, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         REQUIRE(error < tol);
      }
   }
}
#endif
} // namespace qf_coeff

