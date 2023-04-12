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

#include "catch.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

namespace ceed_test
{

#ifdef MFEM_USE_CEED

enum class CeedCoeffType
{ Const, Grid, Quad, VecConst, VecGrid, VecQuad, MatConst, MatQuad };

double coeff_function(const Vector &x)
{
   return 1.0 + x[0]*x[0];
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   const double w = 1.0 + x[0]*x[0];
   switch (dim)
   {
      case 1: v(0) = w; break;
      case 2: v(0) = w*sqrt(2./3.); v(1) = w*sqrt(1./3.); break;
      case 3: v(0) = w*sqrt(3./6.); v(1) = w*sqrt(2./6.); v(2) = w*sqrt(1./6.); break;
   }
}

// Matrix-valued velocity coefficient
void matrix_velocity_function(const Vector &x, DenseMatrix &m)
{
   int dim = x.Size();
   Vector v(dim);
   velocity_function(x, v);
   m.Diag(v.GetData(), dim);
}

// Vector valued quantity to convect
void quantity(const Vector &x, Vector &u)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: u(0) = x[0]*x[0]; break;
      case 2: u(0) = x[0]*x[0]; u(1) = x[1]*x[1]; break;
      case 3: u(0) = x[0]*x[0]; u(1) = x[1]*x[1]; u(2) = x[2]*x[2]; break;
   }
}

// Quantity after explicit convect
// (u \cdot \nabla) v
void convected_quantity(const Vector &x, Vector &u)
{
   double a, b, c;
   int dim = x.Size();
   switch (dim)
   {
      case 1:
         u(0) = 2.*x[0]*(x[0]*x[0]+1.0);
         break;
      case 2:
         a = sqrt(2./3.);
         b = sqrt(1./3.);
         u(0) = 2.*a*x[0]*(x[0]*x[0]+1.0);
         u(1) = 2.*b*x[1]*(x[0]*x[0]+1.0);
         break;
      case 3:
         a = sqrt(3./6.);
         b = sqrt(2./6.);
         c = sqrt(1./6.);
         u(0) = 2.*a*x[0]*(x[0]*x[0]+1.0);
         u(1) = 2.*b*x[1]*(x[0]*x[0]+1.0);
         u(2) = 2.*c*x[2]*(x[0]*x[0]+1.0);
   }
}

std::string getString(AssemblyLevel assembly)
{
   switch (assembly)
   {
      case AssemblyLevel::NONE:
         return "NONE";
         break;
      case AssemblyLevel::PARTIAL:
         return "PARTIAL";
         break;
      case AssemblyLevel::ELEMENT:
         return "ELEMENT";
         break;
      case AssemblyLevel::FULL:
         return "FULL";
         break;
      case AssemblyLevel::LEGACY:
         return "LEGACY";
         break;
   }
   MFEM_ABORT("Unknown AssemblyLevel.");
   return "";
}

std::string getString(CeedCoeffType coeff_type)
{
   switch (coeff_type)
   {
      case CeedCoeffType::Const:
         return "Const";
         break;
      case CeedCoeffType::Grid:
         return "Grid";
         break;
      case CeedCoeffType::Quad:
         return "Quad";
         break;
      case CeedCoeffType::VecConst:
         return "VecConst";
         break;
      case CeedCoeffType::VecGrid:
         return "VecGrid";
         break;
      case CeedCoeffType::VecQuad:
         return "VecQuad";
         break;
      case CeedCoeffType::MatConst:
         return "MatConst";
         break;
      case CeedCoeffType::MatQuad:
         return "MatQuad";
         break;
   }
   MFEM_ABORT("Unknown CeedCoeffType.");
   return "";
}

enum class Problem { Mass,
                     Convection,
                     Diffusion,
                     VectorMass,
                     VectorDiffusion,
                     MassDiffusion
                   };

std::string getString(Problem pb)
{
   switch (pb)
   {
      case Problem::Mass:
         return "Mass";
         break;
      case Problem::Convection:
         return "Convection";
         break;
      case Problem::Diffusion:
         return "Diffusion";
         break;
      case Problem::VectorMass:
         return "VectorMass";
         break;
      case Problem::VectorDiffusion:
         return "VectorDiffusion";
         break;
      case Problem::MassDiffusion:
         return "MassDiffusion";
         break;
   }
   MFEM_ABORT("Unknown Problem.");
   return "";
}

enum class NLProblem {Convection};

std::string getString(NLProblem pb)
{
   switch (pb)
   {
      case NLProblem::Convection:
         return "Convection";
         break;
   }
   MFEM_ABORT("Unknown Problem.");
   return "";
}

void InitCoeff(Mesh &mesh, FiniteElementCollection &fec, const int dim,
               const CeedCoeffType coeff_type, GridFunction *&gf,
               FiniteElementSpace *& coeff_fes,
               Coefficient *&coeff, VectorCoefficient *&vcoeff,
               MatrixCoefficient *&mcoeff)
{
   switch (coeff_type)
   {
      case CeedCoeffType::Const:
         coeff = new ConstantCoefficient(1.0);
         break;
      case CeedCoeffType::Grid:
      {
         FunctionCoefficient f_coeff(coeff_function);
         coeff_fes = new FiniteElementSpace(&mesh, &fec);
         gf = new GridFunction(coeff_fes);
         gf->ProjectCoefficient(f_coeff);
         coeff = new GridFunctionCoefficient(gf);
         break;
      }
      case CeedCoeffType::Quad:
         coeff = new FunctionCoefficient(coeff_function);
         break;
      case CeedCoeffType::VecConst:
      {
         Vector val(dim);
         for (int i = 0; i < dim; i++)
         {
            val(i) = 1.0 + i;
         }
         vcoeff = new VectorConstantCoefficient(val);
         break;
      }
      case CeedCoeffType::VecGrid:
      {
         VectorFunctionCoefficient f_vcoeff(dim, velocity_function);
         coeff_fes = new FiniteElementSpace(&mesh, &fec, dim);
         gf = new GridFunction(coeff_fes);
         gf->ProjectCoefficient(f_vcoeff);
         vcoeff = new VectorGridFunctionCoefficient(gf);
         break;
      }
      case CeedCoeffType::VecQuad:
         vcoeff = new VectorFunctionCoefficient(dim, velocity_function);
         break;
      case CeedCoeffType::MatConst:
      {
         DenseMatrix val(dim);
         val = 0.0;
         for (int i = 0; i < dim; i++)
         {
            val(i, i) = 1.0 + i;
         }
         mcoeff = new MatrixConstantCoefficient(val);
         break;
      }
      case CeedCoeffType::MatQuad:
         mcoeff = new MatrixFunctionCoefficient(dim, matrix_velocity_function);
         break;
   }
}

void test_ceed_operator(const char *input, int order,
                        const CeedCoeffType coeff_type, const Problem pb,
                        const AssemblyLevel assembly, bool mixed_p, bool bdr_integ)
{
   std::string section = "assembly: " + getString(assembly) + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         (mixed_p ? "mixed_p: true\n" : "") +
                         (bdr_integ ? "bdr_integ: true\n" : "") +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   if (mixed_p) { mesh.EnsureNCMesh(); }
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   MatrixCoefficient *mcoeff = nullptr;
   InitCoeff(mesh, fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff, mcoeff);

   // Build the BilinearForm
   bool vecOp = pb == Problem::VectorMass || pb == Problem::VectorDiffusion;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);
   if (mixed_p)
   {
      fes.SetElementOrder(0, order+1);
      fes.SetElementOrder(fes.GetNE() - 1, order+1);
      fes.Update(false);
   }

   auto AddIntegrator = [&bdr_integ](BilinearForm &k, BilinearFormIntegrator *blfi)
   {
      if (bdr_integ)
      {
         k.AddBoundaryIntegrator(blfi);
      }
      else
      {
         k.AddDomainIntegrator(blfi);
      }
   };
   BilinearForm k_test(&fes);
   BilinearForm k_ref(&fes);
   switch (pb)
   {
      case Problem::Mass:
         AddIntegrator(k_ref, new MassIntegrator(*coeff));
         AddIntegrator(k_test, new MassIntegrator(*coeff));
         break;
      case Problem::Convection:
         AddIntegrator(k_ref, new ConvectionIntegrator(*vcoeff, -1));
         AddIntegrator(k_test, new ConvectionIntegrator(*vcoeff, -1));
         break;
      case Problem::Diffusion:
         if (coeff)
         {
            AddIntegrator(k_ref, new DiffusionIntegrator(*coeff));
            AddIntegrator(k_test, new DiffusionIntegrator(*coeff));
         }
         else if (vcoeff)
         {
            AddIntegrator(k_ref, new DiffusionIntegrator(*vcoeff));
            AddIntegrator(k_test, new DiffusionIntegrator(*vcoeff));
         }
         else if (mcoeff)
         {
            AddIntegrator(k_ref, new DiffusionIntegrator(*mcoeff));
            AddIntegrator(k_test, new DiffusionIntegrator(*mcoeff));
         }
         break;
      case Problem::VectorMass:
         AddIntegrator(k_ref, new VectorMassIntegrator(*coeff));
         AddIntegrator(k_test, new VectorMassIntegrator(*coeff));
         break;
      case Problem::VectorDiffusion:
         AddIntegrator(k_ref, new VectorDiffusionIntegrator(*coeff));
         AddIntegrator(k_test, new VectorDiffusionIntegrator(*coeff));
         break;
      case Problem::MassDiffusion:
         AddIntegrator(k_ref, new MassIntegrator(*coeff));
         AddIntegrator(k_test, new MassIntegrator(*coeff));
         AddIntegrator(k_ref, new DiffusionIntegrator(*coeff));
         AddIntegrator(k_test, new DiffusionIntegrator(*coeff));
         break;
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
   delete mcoeff;
}

void test_ceed_nloperator(const char *input, int order,
                          const CeedCoeffType coeff_type,
                          const NLProblem pb, const AssemblyLevel assembly)
{
   std::string section = "assembly: " + getString(assembly) + "\n" +
                         "coeff_type: " + getString(coeff_type) + "\n" +
                         "pb: " + getString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   MatrixCoefficient *mcoeff = nullptr;
   InitCoeff(mesh, fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff, mcoeff);
   MFEM_VERIFY(!mcoeff, "Matrix coefficient tests should use test_ceed_operator.");

   // Build the NonlinearForm
   bool vecOp = pb == NLProblem::Convection;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   NonlinearForm k_test(&fes);
   NonlinearForm k_ref(&fes);
   switch (pb)
   {
      case NLProblem::Convection:
         k_ref.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
         k_test.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
         break;
   }

   k_test.SetAssemblyLevel(assembly);
   k_test.Setup();
   k_ref.Setup();

   // Compare ceed with mfem.
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
   delete mcoeff;
}

// This function specifically tests convection of a vector valued quantity and
// using a custom integration rule. The integration rule is chosen s.t. in
// combination with an appropriate order, it can represent the analytical
// polynomial functions correctly.
void test_ceed_convection(const char *input, int order,
                          const AssemblyLevel assembly)
{
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   VectorFunctionCoefficient velocity_coeff(dim, velocity_function);

   FiniteElementSpace fes(&mesh, &fec, 1);
   FiniteElementSpace vfes(&mesh, &fec, dim);
   BilinearForm conv_op(&fes);

   IntegrationRules rules(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = rules.Get(fes.GetFE(0)->GetGeomType(),
                                         2 * order - 1);

   ConvectionIntegrator *conv_integ = new ConvectionIntegrator(velocity_coeff, 1);
   conv_integ->SetIntRule(&ir);
   conv_op.AddDomainIntegrator(conv_integ);
   conv_op.SetAssemblyLevel(assembly);
   conv_op.Assemble();

   GridFunction q(&vfes), r(&vfes), ex(&vfes);

   VectorFunctionCoefficient quantity_coeff(dim, quantity);
   q.ProjectCoefficient(quantity_coeff);

   VectorFunctionCoefficient convected_quantity_coeff(dim, convected_quantity);
   ex.ProjectCoefficient(convected_quantity_coeff);

   r = 0.0;
   for (int i = 0; i < dim; i++)
   {
      GridFunction qi, ri;
      qi.MakeRef(&fes, q, i * fes.GetVSize());
      ri.MakeRef(&fes, r, i * fes.GetVSize());
      conv_op.Mult(qi, ri);
   }

   LinearForm f(&vfes);
   VectorDomainLFIntegrator *vlf_integ = new VectorDomainLFIntegrator(
      convected_quantity_coeff);
   vlf_integ->SetIntRule(&ir);
   f.AddDomainIntegrator(vlf_integ);
   f.Assemble();

   r -= f;

   REQUIRE(r.Norml2() < 1e-12);
}

TEST_CASE("CEED mass & diffusion", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,
                              CeedCoeffType::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,Problem::MassDiffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1);
   auto bdr_integ = GENERATE(false,true);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   bool mixed_p = false;
   test_ceed_operator(mesh, order, coeff_type, pb, assembly, mixed_p, bdr_integ);
} // test case

TEST_CASE("CEED vector and matrix coefficients", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Quad,
                              CeedCoeffType::VecConst,CeedCoeffType::VecQuad,
                              CeedCoeffType::MatConst,CeedCoeffType::MatQuad);
   auto pb = GENERATE(Problem::Diffusion);
   auto order = GENERATE(1);
   auto bdr_integ = GENERATE(false,true);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   bool mixed_p = false;
   test_ceed_operator(mesh, order, coeff_type, pb, assembly, mixed_p, bdr_integ);
} // test case

TEST_CASE("CEED p-adaptivity", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,
                              CeedCoeffType::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,Problem::MassDiffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/square-mixed.mesh");
   bool mixed_p = true;
   bool bdr_integ = false;
   test_ceed_operator(mesh, order, coeff_type, pb, assembly, mixed_p, bdr_integ);
} // test case

TEST_CASE("CEED convection low", "[CEED], [Convection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::VecConst,CeedCoeffType::VecGrid,
                              CeedCoeffType::VecQuad);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   Problem pb = Problem::Convection;
   int low_order = 1;
   bool mixed_p = false;
   bool bdr_integ = false;
   test_ceed_operator(mesh, low_order, coeff_type, pb, assembly, mixed_p,
                      bdr_integ);
} // test case

TEST_CASE("CEED convection high", "[CEED], [Convection]")
{
   // Apply the CEED convection integrator applied to a vector quantity, check
   // that we get the exact answer (with sufficiently high polynomial degree)
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh");
   int high_order = 4;
   test_ceed_convection(mesh, high_order, assembly);
} // test case

TEST_CASE("CEED non-linear convection", "[CEED], [NLConvection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,
                              CeedCoeffType::Quad);
   auto pb = GENERATE(NLProblem::Convection);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   test_ceed_nloperator(mesh, order, coeff_type, pb, assembly);
} // test case

#endif

} // namespace ceed_test
