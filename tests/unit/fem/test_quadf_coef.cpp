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

using namespace mfem;


TEST_CASE("Quadrature Function Coefficients",
          "[Coefficient][QuadratureFunction][QuadratureFunctionCoefficient]")
{
   int order_h1 = 2, n = 4, dim = 3;
   double tol = 1e-14;

   Mesh mesh = Mesh::MakeCartesian3D(
                  n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order_h1);

   int intOrder = 2 * order_h1 + 1;

   QuadratureSpace qspace(&mesh, intOrder);
   QuadratureFunction quadf_coeff(&qspace, 1);
   QuadratureFunction quadf_vcoeff(&qspace, dim);

   REQUIRE(quadf_coeff.UseDevice());

   const IntegrationRule &ir = qspace.GetElementIntRule(0);

   const GeometricFactors *geom_facts =
      mesh.GetGeometricFactors(ir, GeometricFactors::COORDINATES);

   {
      int nelems = quadf_coeff.Size() / quadf_coeff.GetVDim() / ir.GetNPoints();
      int vdim = ir.GetNPoints();
      geom_facts->X.HostRead();

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
#ifdef MFEM_USE_EXCEPTIONS
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

TEST_CASE("Quadrature Function Integration", "[QuadratureFunction][GPU]")
{
   auto fname = GENERATE(
                   "../../data/star.mesh",
                   "../../data/star-q3.mesh",
                   "../../data/fichera.mesh",
                   "../../data/fichera-q3.mesh"
                );
   const int order = GENERATE(1, 2, 3);

   CAPTURE(fname, order);

   Mesh mesh = Mesh::LoadFromFile(fname);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   int int_order = 2*order + 1;

   SECTION("QuadratureSpace")
   {
      QuadratureSpace qs(&mesh, int_order);

      // Make sure invalidating the cached weights works properly
      qs.GetWeights();
      mesh.Transform([](const Vector &xold, Vector &xnew)
      {
         xnew = xold;
         xnew *= 1.1;
      });

      const IntegrationRule &ir = qs.GetIntRule(0);

      QuadratureFunction qf(qs);
      qf.Randomize(1);
      QuadratureFunctionCoefficient qf_coeff(qf);

      LinearForm lf(&fes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(qf_coeff, &ir));
      lf.Assemble();
      const double integ_1 = lf.Sum();
      const double integ_2 = qf.Integrate();
      const double integ_3 = qs.Integrate(qf_coeff);

      REQUIRE(integ_1 == MFEM_Approx(integ_2));
      REQUIRE(integ_1 == MFEM_Approx(integ_3));
   }

   SECTION("Vector-valued")
   {
      const int vdim = 3;
      const int ordering = Ordering::byNODES;
      FiniteElementSpace fes_vec(&mesh, &fec, vdim, ordering);

      QuadratureSpace qs(&mesh, int_order);
      const IntegrationRule &ir = qs.GetIntRule(0);

      QuadratureFunction qf(qs, vdim);
      qf.Randomize(1);
      VectorQuadratureFunctionCoefficient qf_coeff(qf);

      LinearForm lf(&fes_vec);
      auto *integrator = new VectorDomainLFIntegrator(qf_coeff);
      integrator->SetIntRule(&ir);
      lf.AddDomainIntegrator(integrator);
      lf.Assemble();

      Vector integrals_1(vdim);
      Vector integrals_2(vdim);

      qf.Integrate(integrals_1);
      qs.Integrate(qf_coeff, integrals_2);

      const int ndof = fes.GetNDofs();
      for (int vd = 0; vd < vdim; ++vd)
      {
         double integ = 0.0;
         for (int i = 0; i < ndof; ++i)
         {
            integ += lf[i + vd*ndof];
         }
         REQUIRE(integ == MFEM_Approx(integrals_1[vd]));
         REQUIRE(integ == MFEM_Approx(integrals_2[vd]));
      }
   }

   SECTION("FaceQuadratureSpace")
   {
      FaceQuadratureSpace qs(mesh, int_order, FaceType::Boundary);
      const IntegrationRule &ir = qs.GetIntRule(0);

      QuadratureFunction qf(qs);
      qf.Randomize(1);
      QuadratureFunctionCoefficient qf_coeff(qf);

      LinearForm lf(&fes);
      auto *integ = new BoundaryLFIntegrator(qf_coeff);
      integ->SetIntRule(&ir);
      lf.AddBoundaryIntegrator(integ);
      lf.Assemble();
      const double integ_1 = lf.Sum();
      const double integ_2 = qf.Integrate();
      REQUIRE(integ_1 == MFEM_Approx(integ_2));
   }
}

namespace lin_interp
{
double f3(const Vector &x);
void F3(const Vector &x, Vector &v);
}

TEST_CASE("Face Quadrature Function Coefficients", "[Coefficient]")
{
   auto ftype = GENERATE(FaceType::Interior, FaceType::Boundary);
   auto int_order = GENERATE(3, 5);
   int n = 4, dim = 3;

   Mesh mesh = Mesh::MakeCartesian3D(
                  n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);

   FunctionCoefficient f_coeff(lin_interp::f3);
   VectorFunctionCoefficient vf_coeff(dim, lin_interp::F3);

   FaceQuadratureSpace qspace(mesh, int_order, ftype);

   QuadratureFunction qf(qspace);
   QuadratureFunction vqf(&qspace, dim);

   f_coeff.Project(qf);
   vf_coeff.Project(vqf);

   QuadratureFunctionCoefficient qf_coeff(qf);
   VectorQuadratureFunctionCoefficient vqf_coeff(vqf);

   for (int i = 0; i < qspace.GetNE(); ++i)
   {
      const IntegrationRule &ir = qspace.GetIntRule(i);
      ElementTransformation &T = *qspace.GetTransformation(i);
      for (int iq = 0; iq < ir.Size(); ++iq)
      {
         const IntegrationPoint &ip = ir[iq];
         REQUIRE(f_coeff.Eval(T, ip) == qf_coeff.Eval(T, ip));
      }
   }
}
