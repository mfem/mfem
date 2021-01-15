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
#include "unit_tests.hpp"

using namespace mfem;

namespace qf_coeff
{

TEST_CASE("Quadrature Function Coefficients",
          "[Quadrature Function Coefficients]")
{
   int order_h1 = 2, n = 4, dim = 3;
   double tol = 1e-14;

   Mesh mesh(n, n, n, Element::HEXAHEDRON, false, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order_h1);

   int intOrder = 2 * order_h1 + 1;

   QuadratureSpace qspace(&mesh, intOrder);
   QuadratureFunction quadf_coeff(&qspace, 1);
   QuadratureFunction quadf_vcoeff(&qspace, dim);

   const IntegrationRule ir = qspace.GetElementIntRule(0);

   const GeometricFactors *geom_facts =
      mesh.GetGeometricFactors(ir, GeometricFactors::COORDINATES);

   {
      int nelems = quadf_coeff.Size() / quadf_coeff.GetVDim() / ir.GetNPoints();
      int vdim = ir.GetNPoints();

      for (int i = 0; i < nelems; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            //X has dims nqpts x sdim x ne
            quadf_coeff((i * vdim) + j) =
               geom_facts->X((i * vdim * dim) + (vdim * 2) + j );
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
               quadf_vcoeff((i * nqpts * vdim) + (k * vdim ) + j) =
                  geom_facts->X((i * nqpts * vdim) + (j * nqpts) + k);
            }
         }
      }
   }

   QuadratureFunctionCoefficient qfc(quadf_coeff);
   VectorQuadratureFunctionCoefficient qfvc(quadf_vcoeff);

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
   }

   SECTION("Operators on VectorQuadratureLFIntegrator")
   {
      std::cout << "Testing VectorQuadratureLFIntegrator: " << std::endl;
      H1_FECollection    fec_h1(order_h1, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1, dim);

      GridFunction nodes(&fespace_h1);
      mesh.GetNodes(nodes);

      Vector output(nodes.Size());
      output = 0.0;

      LinearForm lf(&fespace_h1);
      lf.AddDomainIntegrator(new VectorQuadratureLFIntegrator(qfvc, NULL));

      lf.Assemble();

      BilinearForm L2(&fespace_h1);

      L2.AddDomainIntegrator(new VectorMassIntegrator());
      L2.Assemble();

      SparseMatrix mat = L2.SpMat();

      mat.Mult(nodes, output);

      output -= lf;

      REQUIRE(output.Norml2() < tol);
   }

   SECTION("Operators on QuadratureLFIntegrator")
   {
      std::cout << "Testing QuadratureLFIntegrator: " << std::endl;
      H1_FECollection    fec_h1(order_h1, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1, 1);
      FiniteElementSpace fespace_h3(&mesh, &fec_h1, 3);

      GridFunction nodes(&fespace_h3);
      mesh.GetNodes(nodes);

      Vector output(nodes.Size() / dim);
      Vector nz(nodes.Size() / dim);
      output = 0.0;

      nz.MakeRef(nodes, nz.Size() * 2);

      LinearForm lf(&fespace_h1);
      lf.AddDomainIntegrator(new QuadratureLFIntegrator(qfc, NULL));

      lf.Assemble();

      BilinearForm L2(&fespace_h1);

      L2.AddDomainIntegrator(new MassIntegrator(&ir));
      L2.Assemble();

      SparseMatrix mat = L2.SpMat();

      mat.Mult(nz, output);

      output -= lf;

      REQUIRE(output.Norml2() < tol);
   }

}

} // namespace qf_coeff
