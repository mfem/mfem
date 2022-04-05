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
#include <functional>

using namespace mfem;

struct LinearFormExtTest
{
   enum
   {
      DomainLF,
      DomainLFGrad,
      VectorDomainLF,
      VectorDomainLFGrad
   };
   const double abs_tol = 1e-14, rel_tol = 1e-14;
   const char *mesh_filename;
   Mesh mesh;
   const int dim, vdim, ordering, gll, problem, p, q, SEED = 0x100001b3;
   H1_FECollection fec;
   FiniteElementSpace vfes, mfes;
   const Geometry::Type geom_type;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL, *ir;
   Array<int> elem_marker;
   Vector one_vec, dim_vec, vdim_vec, vdim_dim_vec;
   ConstantCoefficient cst_coeff;
   VectorConstantCoefficient dim_cst_coeff, vdim_cst_coeff, vdim_dim_cst_coeff;
   std::function<void(const Vector&, Vector&)>
   vdim_vector_function = [&](const Vector&, Vector &y)
   { y.SetSize(vdim); y.Randomize(SEED); };
   std::function<void(const Vector&, Vector&)> vector_fct;
   VectorFunctionCoefficient vdim_function_coeff;
   LinearForm lf_dev, lf_std;
   LinearFormIntegrator *lfi_dev, *lfi_std;

   LinearFormExtTest(const char *mesh_filename,
                     int vdim, int ordering, int gll, int problem, int p):
      mesh_filename(mesh_filename),
      mesh(Mesh::LoadFromFile(mesh_filename)),
      dim(mesh.Dimension()),
      vdim(vdim),
      ordering(ordering),
      gll(gll),
      problem(problem),
      p(p),
      q(2*p + (gll?-1:3)),
      fec(p, dim),
      vfes(&mesh, &fec, vdim, ordering),
      mfes(&mesh, &fec, dim),
      geom_type(vfes.GetFE(0)->GetGeomType()),
      IntRulesGLL(0, Quadrature1D::GaussLobatto),
      irGLL(&IntRulesGLL.Get(geom_type, q)),
      ir(&IntRules.Get(geom_type, q)),
      elem_marker(),
      one_vec(1),
      dim_vec(dim),
      vdim_vec(vdim),
      vdim_dim_vec(vdim*dim),
      cst_coeff(M_PI),
      dim_cst_coeff((dim_vec.Randomize(SEED), dim_vec)),
      vdim_cst_coeff((vdim_vec.Randomize(SEED), vdim_vec)),
      vdim_dim_cst_coeff((vdim_dim_vec.Randomize(SEED), vdim_dim_vec)),
      vector_fct(vdim_vector_function),
      vdim_function_coeff(vdim, vector_fct),
      lf_dev(&vfes),
      lf_std(&vfes),
      lfi_dev(nullptr),
      lfi_std(nullptr)
   {
      REQUIRE(mesh.attributes.Size() == 1);
      for (int e = 0; e < mesh.GetNE(); e++) { mesh.SetAttribute(e, e%2?1:2); }
      mesh.SetAttributes();
      REQUIRE(mesh.attributes.Size() == 2);
      elem_marker.SetSize(2);
      elem_marker[0] = 0;
      elem_marker[1] = 1;

      if (problem == DomainLF)
      {
         lfi_dev = new DomainLFIntegrator(cst_coeff);
         lfi_std = new DomainLFIntegrator(cst_coeff);
      }
      else if (problem == DomainLFGrad)
      {
         lfi_dev = new DomainLFGradIntegrator(dim_cst_coeff);
         lfi_std = new DomainLFGradIntegrator(dim_cst_coeff);
      }
      else if (problem == VectorDomainLF)
      {
         lfi_dev = new VectorDomainLFIntegrator(vdim_function_coeff);
         lfi_std = new VectorDomainLFIntegrator(vdim_function_coeff);
      }
      else if (problem == VectorDomainLFGrad)
      {
         lfi_dev = new VectorDomainLFGradIntegrator(vdim_dim_cst_coeff);
         lfi_std = new VectorDomainLFGradIntegrator(vdim_dim_cst_coeff);
      }
      else { REQUIRE(false); }

      lfi_dev->SetIntRule(gll ? irGLL : ir);
      lfi_std->SetIntRule(gll ? irGLL : ir);

      lf_dev.AddDomainIntegrator(lfi_dev, elem_marker);
      lf_std.AddDomainIntegrator(lfi_std, elem_marker);
   }

   void Run()
   {
      const bool scalar = problem == LinearFormExtTest::DomainLF ||
                          problem == LinearFormExtTest::DomainLFGrad;
      REQUIRE((!scalar || vdim == 1));

      const bool grad = problem == LinearFormExtTest::DomainLFGrad ||
                        problem == LinearFormExtTest::VectorDomainLFGrad;

      mfem::out << "[LinearFormExt] " << dim << "D " << mesh_filename
                << " p=" << p << " q=" << q
                << (ordering == Ordering::byNODES ? " byNODES " : " byVDIM  ")
                << vdim << "-"
                << (scalar ? "Scalar" : "Vector") << (grad ? "Grad" : "Eval")
                << std::endl;

      const bool use_device = true;
      lf_dev.Assemble(use_device);

      const bool dont_use_device = false;
      lf_std.Assemble(dont_use_device);

      REQUIRE(lf_dev*lf_dev == MFEM_Approx(lf_std*lf_std, abs_tol, rel_tol));

      lf_std -= lf_dev;
      REQUIRE(0.0 == MFEM_Approx(lf_std*lf_std, abs_tol, rel_tol));
   }
};

TEST_CASE("Linear Form Extension", "[LinearformExt], [CUDA]")
{
   const bool all = launch_all_non_regression_tests;

   const auto mesh =
      all ? GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                     "../../data/fichera.mesh", "../../data/fichera-q3.mesh") :
      GENERATE("../../data/star-q3.mesh", "../../data/fichera-q3.mesh") ;
   const auto p = all ? GENERATE(1,2,3,4,5,6/*,7*/) : GENERATE(1,3,6);
   const auto gll = GENERATE(0,1);

   SECTION("Scalar")
   {
      const auto problem = GENERATE(LinearFormExtTest::DomainLF,
                                    LinearFormExtTest::DomainLFGrad);
      LinearFormExtTest(mesh, 1, Ordering::byNODES, gll, problem, p).Run();
   }

   SECTION("Vector")
   {
      const auto vdim = all ? GENERATE(1,5,7,24) : GENERATE(1,5);
      const auto ordering = GENERATE(Ordering::byVDIM, Ordering::byNODES);
      const auto problem = GENERATE(LinearFormExtTest::VectorDomainLF,
                                    LinearFormExtTest::VectorDomainLFGrad);
      LinearFormExtTest(mesh, vdim, ordering, gll, problem, p).Run();
   }
}

