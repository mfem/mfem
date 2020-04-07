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
   int order_h1 = 1, n = 3, dim = 3;
   double tol = 1e-9;

   Mesh mesh(n, n, n, Element::HEXAHEDRON, false, 1.0, 1.0, 1.0);

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
      //More like nelems * nqpts
      int nelems = quadf_vcoeff.Size() / quadf_vcoeff.GetVDim();
      int vdim = quadf_vcoeff.GetVDim();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            quadf_vcoeff((i * vdim) + j) = j;
         }
      }
   }

   QuadratureFunctionCoefficient qfc(&quadf_coeff);
   VectorQuadratureFunctionCoefficient qfvc(&quadf_vcoeff);

   SECTION("Operators on VecQuadFuncCoeff")
   {
      std::cout << "Testing VecQuadFuncCoeff: " << std::endl;
#ifdef MFEM_USE_EXCEPTIONS
      std::cout << " Setting Index" << std::endl;
      REQUIRE_THROWS(qfvc.SetIndex(3));
      REQUIRE_THROWS(qfvc.SetIndex(-1));
      REQUIRE_NOTHROW(qfvc.SetIndex(1));
      qfvc.SetIndex(0);
      std::cout << " Setting Length" << std::endl;
      REQUIRE_THROWS(qfvc.SetLength(4));
      qfvc.SetIndex(1);
      REQUIRE_THROWS(qfvc.SetLength(3));
      REQUIRE_NOTHROW(qfvc.SetLength(2));
      REQUIRE_THROWS(qfvc.SetLength(0));
#endif
      qfvc.SetIndex(0);
      qfvc.SetLength(3);

      SECTION("Gridfunction L2 tests")
      {
         std::cout << "  Testing GridFunc L2 projection" << std::endl;
         L2_FECollection    fec_l2(order_h1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2, dim);
         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         {
            int nnodes = gtrue.Size() / dim;
            int vdim = dim;

            for (int i = 0; i < vdim; i++)
            {
               for (int j = 0; j < nnodes; j++)
               {
                  gtrue((i * nnodes) + j) = i;
               }
            }
         }

         g0 = 0.0;
         g0.ProjectDiscCoefficient(qfvc, GridFunction::ARITHMETIC);
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
            int nnodes = gtrue.Size() / dim;
            int vdim = dim;

            for (int i = 0; i < vdim; i++)
            {
               for (int j = 0; j < nnodes; j++)
               {
                  gtrue((i * nnodes) + j) = i;
               }
            }
         }

         g0 = 0.0;
         g0.ProjectCoefficient(qfvc);
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
         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         // When using an L2 FE space of the same order as the mesh, the below highlights
         // that the ProjectDiscCoeff method is just taking the quadrature point values and making them node values.
         {
            int ne = mesh.GetNE();
            int int_points = ir.GetNPoints();

            for (int i = 0; i < ne; i++)
            {
               for (int j = 0; j < int_points; j++)
               {
                  gtrue((i * int_points) + j) = geom_facts->X((i * int_points * dim) +
                                                              (2 * int_points) + j);
               }
            }
         }

         g0 = 0.0;
         g0.ProjectDiscCoefficient(qfc, GridFunction::ARITHMETIC);
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
         //If this was actually doing something akin to an L2 projection these values would be fairly close.
         g0 = 0.0;
         g0.ProjectCoefficient(qfc);
         gtrue -= g0;
         //This currently fails...
         REQUIRE(gtrue.Norml2() < tol);
      }
   }
}

} // namespace qf_coeff

