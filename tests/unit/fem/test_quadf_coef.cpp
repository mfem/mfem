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

#ifdef MFEM_USE_EXCEPTIONS

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

   {
      int nelems = quadf_coeff.Size() / quadf_coeff.GetVDim();
      int vdim = quadf_coeff.GetVDim();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            quadf_coeff((i * vdim) + j) = 1.0;
         }
      }
   }

   {
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
   QuadratureVectorFunctionCoefficient qfvc(&quadf_vcoeff);

   SECTION("Operators on QuadVecFuncCoeff")
   {
      std::cout << "Testing QuadVecFuncCoeff: " << std::endl;
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
      qfvc.SetIndex(0);
      qfvc.SetLength(3);

      SECTION("Gridfunction L2 tests")
      {
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
         L2_FECollection    fec_h1(order_h1, dim);
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
         L2_FECollection    fec_l2(order_h1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2, 1);
         GridFunction g0(&fespace_l2);
         GridFunction gtrue(&fespace_l2);

         {
            int nnodes = gtrue.Size();
            int vdim = 1;

            for (int i = 0; i < vdim; i++)
            {
               for (int j = 0; j < nnodes; j++)
               {
                  gtrue((i * nnodes) + j) = 1.0;
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
         L2_FECollection    fec_h1(order_h1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1, 1);
         GridFunction g0(&fespace_h1);
         GridFunction gtrue(&fespace_h1);

         {
            int nnodes = gtrue.Size();
            int vdim = 1;

            for (int i = 0; i < vdim; i++)
            {
               for (int j = 0; j < nnodes; j++)
               {
                  gtrue((i * nnodes) + j) = 1.0;
               }
            }
         }

         g0 = 0.0;
         g0.ProjectCoefficient(qfc);
         gtrue -= g0;
         REQUIRE(gtrue.Norml2() < tol);
      }
   }
}

} // namespace qf_coeff

#endif  // MFEM_USE_EXCEPTIONS
