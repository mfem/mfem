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

#include "mfem.hpp"
#include <functional>

namespace mfem
{

namespace linearform_ext_tests
{

constexpr int SEED = 0x100001b3;

struct LinearFormExtTest
{
   enum
   {
      DomainLF = 1,
      DomainLFGrad = 2,
      VectorDomainLF = 3,
      VectorDomainLFGrad =4
   };

   const int dim, vdim;
   const bool gll, test;
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
   std::function<void(const Vector&, Vector&)>
   vdim_vector_function = [&](const Vector&, Vector &y)
   {
      y.SetSize(vdim);
      y.Randomize(SEED);
   };
   std::function<void(const Vector&, Vector&)> vector_f;
   VectorFunctionCoefficient vector_function_coeff;

   LinearForm lf_full, lf_legacy;

   const int dofs;
   double mdofs;

   LinearFormExtTest(int N, int dim, int VDIM, bool gll,
                     int problem, int order,
                     bool test):
      dim(dim),
      vdim(VDIM),
      gll(gll),
      test(test),
      problem(problem),
      N(N),
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
      lf_full(&vfes),
      lf_legacy(&vfes),
      dofs(vfes.GetTrueVSize()),
      mdofs(0.0)
   {
      SetupRandomMesh();

      LinearFormIntegrator *integ_full = nullptr;
      LinearFormIntegrator *integ_legacy = nullptr;

      switch (problem)
      {
         case DomainLF:
         {
            integ_full = new DomainLFIntegrator(constant_coeff);
            integ_legacy = new DomainLFIntegrator(constant_coeff);
            break;
         }
         case DomainLFGrad:
         {
            integ_full = new DomainLFGradIntegrator(dim_constant_coeff);
            integ_legacy = new DomainLFGradIntegrator(dim_constant_coeff);
            break;
         }
         case VectorDomainLF:
         {
            if (test)
            {
               integ_full = new VectorDomainLFIntegrator(vector_function_coeff);
               integ_legacy = new VectorDomainLFIntegrator(vector_function_coeff);
            }
            else // !test => bench, we don't want to spend time building coeff
            {
               integ_full = new VectorDomainLFIntegrator(vdim_constant_coeff);
               integ_legacy = new VectorDomainLFIntegrator(vdim_constant_coeff);
            }
            break;
         }
         case VectorDomainLFGrad:
         {
            integ_full = new VectorDomainLFGradIntegrator(vdim_constant_coeff);
            integ_legacy = new VectorDomainLFGradIntegrator(vdim_constant_coeff);
            break;
         }
         default: { MFEM_ABORT("Unknown Problem!"); }
      }

      integ_full->SetIntRule(gll ? irGLL : ir);
      integ_legacy->SetIntRule(gll ? irGLL : ir);

      lf_full.AddDomainIntegrator(integ_full);
      lf_legacy.AddDomainIntegrator(integ_legacy);

      lf_full.SetAssemblyLevel(LinearAssemblyLevel::FULL);
      lf_legacy.SetAssemblyLevel(LinearAssemblyLevel::LEGACY);
   }

   void AssembleBoth()
   {
      lf_full.Assemble();
      lf_legacy.Assemble();
   }

   virtual void Run();

   virtual void Description();

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

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

} // namespace linearform_ext_tests

} // namespace mfem

