// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"

#include "mfem.hpp"
#include <functional>

using namespace mfem;

namespace linearform_ext_tests
{

constexpr int SEED = 0x100001b3;

struct LinearFormExtensionTest
{
   const bool gll;
   const int dim, vdim=0;
   std::function<void(const Vector&, Vector&)>
   vdim_vector_function = [&](const Vector&, Vector &y)
   {
      assert(vdim>0);
      y.SetSize(vdim);
      y.Randomize(SEED);
   };
   const int problem, N, p, q;
   const Element::Type type;
   Mesh mesh;
   H1_FECollection fec;
   FiniteElementSpace vfes;
   FiniteElementSpace mfes;
   GridFunction x;
   const Geometry::Type geom_type;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL;
   const IntegrationRule *ir;
   ConstantCoefficient one;

   Vector one_vec, dim_vec, vdim_vec;
   ConstantCoefficient constant_coeff;
   VectorConstantCoefficient dim_constant_coeff;
   VectorConstantCoefficient vdim_constant_coeff;
   std::function<void(const Vector&, Vector&)> vector_f;
   VectorFunctionCoefficient vector_function_coeff;

   LinearForm lf_legacy, lf_full;

   LinearFormExtensionTest(int dim, int vdim, bool gll, int problem, int order):
      gll(gll),
      dim(dim),
      vdim(vdim),
      problem(problem),
      N(Device::IsEnabled() ? 32 : 4),
      p(order),
      q(2*p + (gll?-1:3)),
      type(dim==3 ?
           Element::HEXAHEDRON :
           Element::QUADRILATERAL),
      mesh(dim==2 ?
           Mesh::MakeCartesian2D(N,N,type):
           Mesh::MakeCartesian3D(N,N,N,type)),
      fec(p, dim),
      vfes(&mesh, &fec, vdim),
      mfes(&mesh, &fec, dim),
      x(&mfes),
      geom_type(vfes.GetFE(0)->GetGeomType()),
      IntRulesGLL(0, Quadrature1D::GaussLobatto),
      irGLL(&IntRulesGLL.Get(geom_type, q)),
      ir(&IntRules.Get(geom_type, q)),
      one(1.0),
      one_vec(1),
      dim_vec(dim),
      vdim_vec(vdim),
      constant_coeff((one_vec.Randomize(SEED), one_vec(0))),
      dim_constant_coeff((dim_vec.Randomize(SEED), dim_vec)),
      vdim_constant_coeff((vdim_vec.Randomize(SEED), vdim_vec)),
      vector_f(vdim_vector_function),
      vector_function_coeff(vdim, vector_f),
      lf_legacy(&vfes),
      lf_full(&vfes)
   {
      Description();
      SetupRandomMesh();

      LinearFormIntegrator *lf0 = nullptr;
      LinearFormIntegrator *lf1 = nullptr;

      switch (problem)
      {
         case 1: // DomainLFIntegrator
         {
            lf0 = new DomainLFIntegrator(constant_coeff);
            lf1 = new DomainLFIntegrator(constant_coeff);
            break;
         }
         case 2: // VectorDomainLFIntegrator
         {
            lf0 = new VectorDomainLFIntegrator(vector_function_coeff);
            lf1 = new VectorDomainLFIntegrator(vector_function_coeff);
            break;
         }
         case 3: // DomainLFGradIntegrator
         {
            lf0 = new DomainLFGradIntegrator(dim_constant_coeff);
            lf1 = new DomainLFGradIntegrator(dim_constant_coeff);
            break;
         }
         case 4: // VectorDomainLFGradIntegrator
         {
            lf0 = new VectorDomainLFGradIntegrator(vdim_constant_coeff);
            lf1 = new VectorDomainLFGradIntegrator(vdim_constant_coeff);
            break;
         }
         default: { MFEM_ABORT("Unknown Problem!"); }
      }

      lf0->SetIntRule(gll ? irGLL : ir);
      lf1->SetIntRule(gll ? irGLL : ir);

      lf_legacy.AddDomainIntegrator(lf0);
      lf_full.AddDomainIntegrator(lf1);

      lf_legacy.SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
      lf_full.SetAssemblyLevel(LinearAssemblyLevel::FULL);

      lf_legacy.Assemble();
      lf_full.Assemble();

      const double fxf = lf_full * lf_full;
      const double lxl = lf_legacy * lf_legacy;
      REQUIRE(fxf == MFEM_Approx(lxl));
   }

   void Description()
   {
      mfem::out << "[LinearFormExt]"
                << " p=" << p
                << " q=" << q
                << " "<< dim << "D"
                << " "<< vdim << "-"
                << (problem%2?"Scalar":"Vector")
                << (problem>2?"Grad":"")
                //<< (gll ? "GLL" : "GL")
                << std::endl;
   }

   void SetupRandomMesh()
   {
      mesh.SetNodalFESpace(&mfes);
      mesh.SetNodalGridFunction(&x);
      const double jitter = 1./(M_PI*M_PI);
      const double h0 = mesh.GetElementSize(0);
      GridFunction rdm(&mfes);
      rdm.Randomize(SEED);
      rdm -= 0.5; // Shift to random values in [-0.5,0.5]
      rdm *= jitter * h0; // Scale the random values to be of same order
      x -= rdm;
   }
};

TEST_CASE("Linearform Extension", "[Linearform]")
{
   const auto dim = GENERATE(2,3);
   const auto order = GENERATE(1,2,3);
   const auto gll = GENERATE(false,true); // q=p+2, q=p+1

   SECTION("Scalar")
   {
      const auto vdim = 1;
      // DomainLFIntegrator
      // DomainLFGradIntegrator
      const auto problem = GENERATE(1,3);
      LinearFormExtensionTest(dim, vdim, gll, problem, order);
   }

   SECTION("Vector")
   {
      const auto vdim = GENERATE(1,2,7);
      // VectorDomainLFIntegrator
      // VectorDomainLFGradIntegrator
      const auto problem = GENERATE(2,4);
      LinearFormExtensionTest(dim, vdim, gll, problem, order);
   }

} // test case

} // namespace linearform_ext_tests
