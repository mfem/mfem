// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include <functional>
#include <ctime>

using namespace mfem;

static double f(const Vector &xvec)
{
   const int dim = xvec.Size();
   double val = 2*xvec[0];
   if (dim >= 2)
   {
      val += 3*xvec[1]*xvec[0];
   }
   if (dim >= 3)
   {
      val += 0.25*xvec[2]*xvec[1];
   }
   return val;
}

static void fvec_dim(const Vector &xvec, Vector &v)
{
   v.SetSize(xvec.Size());
   for (int d = 0; d < xvec.Size(); ++d)
   {
      v[d] = f(xvec) / double(d + 1);
   }
}

enum QuadratureType { Lobatto, Legendre, LegendreUnderIntegration };

struct LinearFormExtTest
{
   enum { DLFEval, QLFEval, DLFGrad, // scalar domain
          VDLFEval, VQLFEval, VDLFGrad, // vector domain
          BLFEval, BNLFEval // boundary
        };
   const double abs_tol = 1e-14, rel_tol = 1e-14;
   const char *mesh_filename;
   Mesh mesh;
   const int dim, vdim, ordering, problem, p, q, SEED = 0x100001b3;
   H1_FECollection fec;
   FiniteElementSpace vfes;
   const Geometry::Type geom_type;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL, *ir;
   Array<int> elem_marker;
   Vector vdim_vec, vdim_dim_vec;
   FunctionCoefficient fn_coeff;
   VectorFunctionCoefficient dim_fn_coeff;
   VectorConstantCoefficient vdim_cst_coeff, vdim_dim_cst_coeff;
   QuadratureSpace qspace;
   QuadratureFunction q_function, q_vdim_function;
   QuadratureFunctionCoefficient qfc;
   VectorQuadratureFunctionCoefficient qfvc;
   LinearForm lf_dev, lf_std;
   LinearFormIntegrator *lfi_dev, *lfi_std;

   static int GetQOrder(QuadratureType qtype, int p)
   {
      switch (qtype)
      {
         case Lobatto: return 2*p - 1;
         case Legendre: return 2*p + 3;
         case LegendreUnderIntegration: return 2*p - 1;
      }
      MFEM_ABORT("Unknown QuadratureType.");
      return 0;
   }

   LinearFormExtTest(const char *mesh_filename,
                     int vdim, int ordering, QuadratureType qtype, int problem,
                     int p):
      mesh_filename(mesh_filename),
      mesh(Mesh::LoadFromFile(mesh_filename)),
      dim(mesh.Dimension()), vdim(vdim), ordering(ordering),
      problem(problem), p(p), q(GetQOrder(qtype, p)), fec(p, dim),
      vfes(&mesh, &fec, vdim, ordering),
      geom_type(vfes.GetFE(0)->GetGeomType()),
      IntRulesGLL(0, Quadrature1D::GaussLobatto),
      irGLL(&IntRulesGLL.Get(geom_type, q)), ir(&IntRules.Get(geom_type, q)),
      elem_marker(), vdim_vec(vdim), vdim_dim_vec(vdim*dim),
      fn_coeff(f), dim_fn_coeff(dim, fvec_dim),
      vdim_cst_coeff((vdim_vec.Randomize(SEED), vdim_vec)),
      vdim_dim_cst_coeff((vdim_dim_vec.Randomize(SEED), vdim_dim_vec)),
      qspace(&mesh, q),
      q_function(&qspace, 1),
      q_vdim_function(&qspace, vdim),
      qfc((q_function.Randomize(SEED), q_function)),
      qfvc((q_vdim_function.Randomize(SEED), q_vdim_function)),
      lf_dev(&vfes), lf_std(&vfes), lfi_dev(nullptr), lfi_std(nullptr)
   {
      for (int e = 0; e < mesh.GetNE(); e++) { mesh.SetAttribute(e, e%2?1:2); }
      mesh.SetAttributes();
      MFEM_VERIFY(mesh.attributes.Size() == 2, "mesh attributes size error!");
      elem_marker.SetSize(2);
      elem_marker[0] = 0;
      elem_marker[1] = 1;

      if (problem == DLFEval)
      {
         lfi_dev = new DomainLFIntegrator(fn_coeff);
         lfi_std = new DomainLFIntegrator(fn_coeff);
      }
      else if (problem == QLFEval)
      {
         lfi_dev = new QuadratureLFIntegrator(qfc,NULL);
         lfi_std = new QuadratureLFIntegrator(qfc,NULL);
      }
      else if (problem == DLFGrad)
      {
         lfi_dev = new DomainLFGradIntegrator(dim_fn_coeff);
         lfi_std = new DomainLFGradIntegrator(dim_fn_coeff);
      }
      else if (problem == VDLFEval)
      {
         lfi_dev = new VectorDomainLFIntegrator(vdim_cst_coeff);
         lfi_std = new VectorDomainLFIntegrator(vdim_cst_coeff);
      }
      else if (problem == VQLFEval)
      {
         lfi_dev = new VectorQuadratureLFIntegrator(qfvc,NULL);
         lfi_std = new VectorQuadratureLFIntegrator(qfvc,NULL);
      }
      else if (problem == VDLFGrad)
      {
         lfi_dev = new VectorDomainLFGradIntegrator(vdim_dim_cst_coeff);
         lfi_std = new VectorDomainLFGradIntegrator(vdim_dim_cst_coeff);
      }
      else if (problem == BLFEval)
      {
         lfi_dev = new BoundaryLFIntegrator(fn_coeff);
         lfi_std = new BoundaryLFIntegrator(fn_coeff);
      }
      else if (problem == BNLFEval)
      {
         lfi_dev = new BoundaryNormalLFIntegrator(dim_fn_coeff);
         lfi_std = new BoundaryNormalLFIntegrator(dim_fn_coeff);
      }
      else { REQUIRE(false); }

      if (problem != QLFEval && problem != VQLFEval && problem != BLFEval &&
          problem != BNLFEval)
      {
         lfi_dev->SetIntRule((qtype == Lobatto) ? irGLL : ir);
         lfi_std->SetIntRule((qtype == Lobatto) ? irGLL : ir);
      }

      if (problem != BLFEval && problem != BNLFEval)
      {
         lf_dev.AddDomainIntegrator(lfi_dev, elem_marker);
         lf_std.AddDomainIntegrator(lfi_std, elem_marker);
      }
      else
      {
         lf_dev.AddBoundaryIntegrator(lfi_dev);
         lf_std.AddBoundaryIntegrator(lfi_std);
      }

      // Test accumulation of integrators
      if (vdim == 1)
      {
         lf_dev.AddDomainIntegrator(new DomainLFIntegrator(fn_coeff));
         lf_std.AddDomainIntegrator(new DomainLFIntegrator(fn_coeff));
      }
   }

   void Run()
   {
      const bool scalar = problem == LinearFormExtTest::DLFEval ||
                          problem == LinearFormExtTest::QLFEval ||
                          problem == LinearFormExtTest::DLFGrad;
      REQUIRE((!scalar || vdim == 1));

      const bool grad = problem == LinearFormExtTest::DLFGrad ||
                        problem == LinearFormExtTest::VDLFGrad;

      CAPTURE(mesh_filename, dim, p, q, ordering, vdim, scalar, grad);

      lf_dev.UseFastAssembly(true);
      lf_dev.Assemble();

      lf_std.UseFastAssembly(false);
      lf_std.Assemble();

      lf_std -= lf_dev;
      REQUIRE(0.0 == MFEM_Approx(lf_std*lf_std, abs_tol, rel_tol));
   }
};

TEST_CASE("Linear Form Extension", "[LinearFormExtension], [CUDA]")
{
   const bool all = launch_all_non_regression_tests;

   const auto mesh_file =
      all ? GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                     "../../data/fichera.mesh", "../../data/fichera-q3.mesh") :
      GENERATE("../../data/star-q3.mesh", "../../data/fichera-q3.mesh");
   const auto p = all ? GENERATE(1,2,3,4,5,6) : GENERATE(1,3);

   SECTION("Scalar")
   {
      const auto qtype = GENERATE(Lobatto, Legendre, LegendreUnderIntegration);
      const auto problem = GENERATE(LinearFormExtTest::DLFEval,
                                    LinearFormExtTest::QLFEval,
                                    LinearFormExtTest::DLFGrad,
                                    LinearFormExtTest::BLFEval,
                                    LinearFormExtTest::BNLFEval);
      LinearFormExtTest(mesh_file, 1, Ordering::byNODES, qtype, problem, p).Run();
   }

   SECTION("Vector")
   {
      const auto qtype = GENERATE(Lobatto, Legendre, LegendreUnderIntegration);
      const auto vdim = all ? GENERATE(1,5,7) : GENERATE(1,5);
      const auto ordering = GENERATE(Ordering::byVDIM, Ordering::byNODES);
      const auto problem = GENERATE(LinearFormExtTest::VDLFEval,
                                    LinearFormExtTest::VQLFEval,
                                    LinearFormExtTest::VDLFGrad);
      LinearFormExtTest(mesh_file, vdim, ordering, qtype, problem, p).Run();
   }

   SECTION("SetIntPoint")
   {
      Mesh mesh(mesh_file);
      const int dim = mesh.Dimension();
      CAPTURE(mesh_file, dim, p);

      H1_FECollection H1(p, dim);
      FiniteElementSpace H1fes(&mesh, &H1);

      FunctionCoefficient f([](const Vector& x)
      { return std::sin(M_PI*x(0)) * std::sin(M_PI*x(1)); });

      GridFunction x(&H1fes);
      x.ProjectCoefficient(f);
      GradientGridFunctionCoefficient grad_x(&x);
      InnerProductCoefficient norm2_grad_x(grad_x,grad_x);

      L2_FECollection L2(p-1, dim);
      FiniteElementSpace L2fes(&mesh, &L2);

      LinearForm d1(&L2fes);
      d1.AddDomainIntegrator(new DomainLFIntegrator(norm2_grad_x));
      d1.UseFastAssembly(true);
      d1.Assemble();

      LinearForm d2(&L2fes);
      d2.AddDomainIntegrator(new DomainLFIntegrator(norm2_grad_x));
      d2.UseFastAssembly(false);
      d2.Assemble();

      d1 -= d2;

      REQUIRE(d1.Norml2() == MFEM_Approx(0.0));
   }

   SECTION("L2 MapType")
   {
      auto map_type = GENERATE(FiniteElement::VALUE, FiniteElement::INTEGRAL);
      Mesh mesh(mesh_file);
      const int dim = mesh.Dimension();

      CAPTURE(mesh_file, dim, p, map_type);

      L2_FECollection fec(p, dim, BasisType::GaussLegendre, map_type);
      FiniteElementSpace fes(&mesh, &fec);

      ConstantCoefficient coeff(1.0);

      LinearForm d1(&fes);
      d1.AddDomainIntegrator(new DomainLFIntegrator(coeff));
      d1.UseFastAssembly(true);
      d1.Assemble();

      LinearForm d2(&fes);
      d2.AddDomainIntegrator(new DomainLFIntegrator(coeff));
      d2.UseFastAssembly(false);
      d2.Assemble();

      CAPTURE(d1.Norml2(), d2.Norml2());

      d1 -= d2;

      REQUIRE(d1.Norml2() == MFEM_Approx(0.0));
   }

   SECTION("VectorFE")
   {
      Mesh mesh(mesh_file);
      const int dim = mesh.Dimension();

      CAPTURE(mesh_file, dim, p);

      RT_FECollection fec(p-1, dim);
      FiniteElementSpace fes(&mesh, &fec);

      FunctionCoefficient coeff(f);

      LinearForm d1(&fes);
      d1.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(coeff));
      d1.UseFastAssembly(true);
      d1.Assemble();

      LinearForm d2(&fes);
      d2.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(coeff));
      d2.UseFastAssembly(false);
      d2.Assemble();

      CAPTURE(d1.Norml2(), d2.Norml2());

      d1 -= d2;

      REQUIRE(d1.Norml2() == MFEM_Approx(0.0));
   }
}

TEST_CASE("H(div) Linear Form Extension", "[LinearFormExtension], [CUDA]")
{
   const bool all = launch_all_non_regression_tests;

   const auto mesh_file =
      all ? GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                     "../../data/fichera.mesh", "../../data/fichera-q3.mesh") :
      GENERATE("../../data/star-q3.mesh", "../../data/fichera-q3.mesh");
   const auto p = all ? GENERATE(1,2,3,4,5,6) : GENERATE(1,3);

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   CAPTURE(mesh_file, dim, p);

   RT_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec);

   VectorFunctionCoefficient coeff(dim, fvec_dim);

   LinearForm d1(&fes);
   d1.AddDomainIntegrator(new VectorFEDomainLFIntegrator(coeff));
   d1.UseFastAssembly(true);
   d1.Assemble();

   LinearForm d2(&fes);
   d2.AddDomainIntegrator(new VectorFEDomainLFIntegrator(coeff));
   d2.UseFastAssembly(false);
   d2.Assemble();

   CAPTURE(d1.Norml2(), d2.Norml2());
   d1 -= d2;
   REQUIRE(d1.Norml2() == MFEM_Approx(0.0));
}
