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

enum class CeedCoeffType { Const,
                           Grid,
                           Quad,
                           VecConst,
                           VecGrid,
                           VecQuad,
                           MatConst,
                           MatQuad
                         };

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
   m.SetSize(dim);
   m = 0.5;
   for (int i = 0; i < dim; i++)
   {
      m(i, i) = 1.0 + v(i);
   }
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

std::string GetString(AssemblyLevel assembly)
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

std::string GetString(CeedCoeffType coeff_type)
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
                     MassDiffusion,
                     HDivMass,
                     HCurlMass,
                     DivDiv,
                     CurlCurl,
                     MixedVectorGradient,
                     MixedVectorCurl
                   };

std::string GetString(Problem pb)
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
      case Problem::HDivMass:
         return "HDivMass";
         break;
      case Problem::HCurlMass:
         return "HCurlMass";
         break;
      case Problem::DivDiv:
         return "DivDiv";
         break;
      case Problem::CurlCurl:
         return "CurlCurl";
         break;
      case Problem::MixedVectorGradient:
         return "MixedVectorGradient";
         break;
      case Problem::MixedVectorCurl:
         return "MixedVectorCurl";
         break;
   }
   MFEM_ABORT("Unknown Problem.");
   return "";
}

enum class NLProblem {Convection};

std::string GetString(NLProblem pb)
{
   switch (pb)
   {
      case NLProblem::Convection:
         return "Convection";
         break;
   }
   MFEM_ABORT("Unknown NLProblem.");
   return "";
}

void InitCoeff(Mesh &mesh, FiniteElementCollection &fec, const int dim,
               const CeedCoeffType coeff_type, GridFunction *&gf,
               FiniteElementSpace *&coeff_fes,
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
         val = 0.5;
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
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "coeff_type: " + GetString(coeff_type) + "\n" +
                         "pb: " + GetString(pb) + "\n" +
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
   MFEM_VERIFY(!mcoeff,
               "Unexpected matrix-valued coefficient in test_ceed_operator.");

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

   BilinearForm k_ref(&fes);
   BilinearForm k_test(&fes);
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
         AddIntegrator(k_ref, new DiffusionIntegrator(*coeff));
         AddIntegrator(k_test, new DiffusionIntegrator(*coeff));
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
      default:
         MFEM_ABORT("Unexpected problem type.");
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   // Compare ceed with mfem
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);
   Vector d_ref(fes.GetTrueVSize()), d_test(fes.GetTrueVSize());

   x.Randomize(1);

   k_ref.Mult(x, y_ref);
   k_test.Mult(x, y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12 * std::max(y_ref.Norml2(), 1.0));

   if (mesh.Nonconforming())
   {
      k_ref.ConformingAssemble();
   }
   k_ref.AssembleDiagonal(d_ref);
   k_test.AssembleDiagonal(d_test);

   d_test -= d_ref;

   // // TODO: Debug
   // if (mesh.Nonconforming() &&
   //    d_test.Norml2() > 0.1 * d_ref.Norml2())
   // {
   //    out << "\nDIAGONAL ASSEMBLY DELTA\n\n";
   //    d_test.Print();
   //    out << "\nDIAGONAL ASSEMBLY REF\n\n";
   //    d_ref.Print();
   //    // Vector temp(d_test);
   //    // temp += d_ref;
   //    // out << "\nDIAGONAL ASSEMBLY TEST\n\n";
   //    // temp.Print();
   // }

   REQUIRE(d_test.Norml2() <
           (mesh.Nonconforming() ? 1.0 : 1.e-12) * std::max(d_ref.Norml2(), 1.0));
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
   delete mcoeff;
}

void test_ceed_vectorfe_operator(const char *input, int order,
                                 const CeedCoeffType coeff_type, const Problem pb,
                                 const AssemblyLevel assembly, bool bdr_integ)
{
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "coeff_type: " + GetString(coeff_type) + "\n" +
                         "pb: " + GetString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         (bdr_integ ? "bdr_integ: true\n" : "") +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   FiniteElementCollection *fec = nullptr;
   if ((pb == Problem::HDivMass || pb == Problem::DivDiv) && bdr_integ)
   {
      // Boundary RT elements in 2D and 3D are actually L2
      return;
   }
   if (pb == Problem::CurlCurl && dim - bdr_integ < 2)
   {
      // No 1D ND curl shape
      return;
   }
   switch (pb)
   {
      case Problem::Mass:
      case Problem::Diffusion:
         fec = new H1_FECollection(order, dim);
         break;
      case Problem::HDivMass:
      case Problem::DivDiv:
         fec = new RT_FECollection(order-1, dim);
         break;
      case Problem::HCurlMass:
      case Problem::CurlCurl:
         fec = new ND_FECollection(order, dim);
         break;
      default:
         MFEM_ABORT("Unexpected problem type.");
   }

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   MatrixCoefficient *mcoeff = nullptr;
   InitCoeff(mesh, *fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff, mcoeff);
   if (!coeff && (pb == Problem::Mass || pb == Problem::DivDiv ||
                  (pb == Problem::CurlCurl && dim - bdr_integ < 3)))
   {
      delete gf;
      delete coeff_fes;
      delete coeff;
      delete vcoeff;
      delete mcoeff;
      delete fec;
      return;
   }

   // Build the BilinearForm
   FiniteElementSpace fes(&mesh, fec);

   BilinearForm k_ref(&fes);
   BilinearForm k_test(&fes);
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
   switch (pb)
   {
      case Problem::Mass:
         AddIntegrator(k_ref, new MassIntegrator(*coeff));
         AddIntegrator(k_test, new MassIntegrator(*coeff));
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
      case Problem::HDivMass:
      case Problem::HCurlMass:
         if (coeff)
         {
            AddIntegrator(k_ref, new VectorFEMassIntegrator(*coeff));
            AddIntegrator(k_test, new VectorFEMassIntegrator(*coeff));
         }
         else if (vcoeff)
         {
            AddIntegrator(k_ref, new VectorFEMassIntegrator(*vcoeff));
            AddIntegrator(k_test, new VectorFEMassIntegrator(*vcoeff));
         }
         else if (mcoeff)
         {
            AddIntegrator(k_ref, new VectorFEMassIntegrator(*mcoeff));
            AddIntegrator(k_test, new VectorFEMassIntegrator(*mcoeff));
         }
         break;
      case Problem::DivDiv:
         AddIntegrator(k_ref, new DivDivIntegrator(*coeff));
         AddIntegrator(k_test, new DivDivIntegrator(*coeff));
         break;
      case Problem::CurlCurl:
         if (coeff)
         {
            AddIntegrator(k_ref, new CurlCurlIntegrator(*coeff));
            AddIntegrator(k_test, new CurlCurlIntegrator(*coeff));
         }
         else if (vcoeff)
         {
            AddIntegrator(k_ref, new CurlCurlIntegrator(*vcoeff));
            AddIntegrator(k_test, new CurlCurlIntegrator(*vcoeff));
         }
         else if (mcoeff)
         {
            AddIntegrator(k_ref, new CurlCurlIntegrator(*mcoeff));
            AddIntegrator(k_test, new CurlCurlIntegrator(*mcoeff));
         }
         break;
      default:
         MFEM_ABORT("Unexpected problem type.");
   }

   // Timer for profiling
   const int trials = 1;
   const bool debug = false;
   StopWatch chrono_setup_ref, chrono_setup_test;
   StopWatch chrono_apply_ref, chrono_apply_test;
   chrono_setup_ref.Clear();
   chrono_setup_ref.Start();

   k_ref.Assemble();
   k_ref.Finalize();

   chrono_setup_ref.Stop();
   chrono_setup_test.Clear();
   chrono_setup_test.Start();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   chrono_setup_test.Stop();

   // Compare ceed with mfem
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);
   Vector d_ref(fes.GetTrueVSize()), d_test(fes.GetTrueVSize());

   x.Randomize(1);

   chrono_apply_ref.Clear();
   chrono_apply_ref.Start();

   for (int trial = 0; trial < trials; trial++)
   {
      k_ref.Mult(x, y_ref);
   }

   chrono_apply_ref.Stop();
   chrono_apply_test.Clear();
   chrono_apply_test.Start();

   for (int trial = 0; trial < trials; trial++)
   {
      k_test.Mult(x, y_test);
   }

   chrono_apply_test.Stop();

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12 * std::max(y_ref.Norml2(), 1.0));

   if (mesh.Nonconforming())
   {
      k_ref.ConformingAssemble();
   }
   k_ref.AssembleDiagonal(d_ref);
   k_test.AssembleDiagonal(d_test);

   d_test -= d_ref;

   // // TODO: Debug
   // if (!UsesTensorBasis(fes) && order > 1 &&
   //     (pb == Problem::HCurlMass || pb == Problem::CurlCurl) &&
   //    d_test.Norml2() > 0.1 * d_ref.Norml2())
   // {
   //    out << "\nH(CURL) DIAGONAL ASSEMBLY DELTA\n\n";
   //    d_test.Print();
   //    out << "\nH(CURL) DIAGONAL ASSEMBLY REF\n\n";
   //    d_ref.Print();
   //    // Vector temp(d_test);
   //    // temp += d_ref;
   //    // out << "\nH(CURL) DIAGONAL ASSEMBLY TEST\n\n";
   //    // temp.Print();
   // }

   REQUIRE(d_test.Norml2() <
           (mesh.Nonconforming() ||
            (!UsesTensorBasis(fes) && order > 1 &&
             (pb == Problem::HCurlMass || pb == Problem::CurlCurl)) ?
            1.0 : 1.e-12) * std::max(d_ref.Norml2(), 1.0));

   if (debug)
   {
      // Estimates only for !bdr_integ, non-mixed meshes
      std::size_t mem_test = 0;
      if (!bdr_integ && mesh.GetNumGeometries(dim) == 1)
      {
         const FiniteElement &fe = *fes.GetFE(0);
         ElementTransformation &T = *mesh.GetElementTransformation(0);
         const int Q = (*k_ref.GetDBFI())[0]->GetRule(fe, T).GetNPoints();
         const int P = fe.GetDof();
         switch (pb)
         {
            case Problem::Mass:
               mem_test = Q * 1 * 8;
               mem_test += P * 4;
            case Problem::Diffusion:
               mem_test = Q * (dim * (dim + 1)) / 2 * 8;
               mem_test += P * 4;
               break;
            case Problem::HDivMass:
               mem_test = Q * (dim * (dim + 1)) / 2 * 8;
               mem_test += P * 4;
            case Problem::DivDiv:
               mem_test = Q * 1 * 8;
               mem_test += P * 4;
               break;
            case Problem::HCurlMass:
               mem_test = Q * (dim * (dim + 1)) / 2 * 8;
               mem_test += P * 3 * 4;  // Tri-diagonal curl orientations
            case Problem::CurlCurl:
               mem_test = Q * (dim - bdr_integ < 3 ? 1 : dim * (dim + 1) / 2) * 8;
               mem_test += P * 3 * 4;
               break;
            default:
               MFEM_ABORT("Unexpected problem type.");
         }
         mem_test *= mesh.GetNE();  // Estimate for QFunction memory
      }
      std::size_t mem_ref = k_ref.SpMat().NumNonZeroElems() * (8 + 4) +
                            k_ref.Height() * 4;

      out << "\n" << section << "\n";
      out << "benchmark (" << fes.GetTrueVSize() << " unknowns)\n"
          << "    setup: ref = "
          << chrono_setup_ref.RealTime() * 1e3 << " ms\n"
          << "           test = "
          << chrono_setup_test.RealTime() * 1e3 << " ms\n"
          << "    apply: ref = "
          << chrono_apply_ref.RealTime() * 1e3 / trials << " ms\n"
          << "           test = "
          << chrono_apply_test.RealTime() * 1e3 / trials << " ms\n"
          << "    mem usage: ref = " << mem_ref / 1e6 << " MB\n"
          << "               test = " << mem_test / 1e6 << " MB\n";
   }
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
   delete mcoeff;
   delete fec;
}

void test_ceed_mixed_operator(const char *input, int order,
                              const CeedCoeffType coeff_type, const Problem pb,
                              const AssemblyLevel assembly, bool bdr_integ)
{
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "coeff_type: " + GetString(coeff_type) + "\n" +
                         "pb: " + GetString(pb) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         (bdr_integ ? "bdr_integ: true\n" : "") +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   FiniteElementCollection *trial_fec = nullptr, *test_fec = nullptr;
   if (pb == Problem::MixedVectorGradient && dim - bdr_integ < 2)
   {
      // MixedVectorGradient is only supported in 2D or 3D
      return;
   }
   if (pb == Problem::MixedVectorCurl && dim - bdr_integ < 3)
   {
      // MixedVectorCurl is only supported in 3D
      return;
   }
   switch (pb)
   {
      case Problem::MixedVectorGradient:
         trial_fec = new H1_FECollection(order, dim);
         test_fec = new ND_FECollection(order, dim);
         break;
      case Problem::MixedVectorCurl:
         trial_fec = new ND_FECollection(order, dim);
         test_fec = new RT_FECollection(order - 1, dim);
         break;
      default:
         MFEM_ABORT("Unexpected problem type.");
   }

   // Coefficient Initialization
   GridFunction *gf = nullptr;
   FiniteElementSpace *coeff_fes = nullptr;
   Coefficient *coeff = nullptr;
   VectorCoefficient *vcoeff = nullptr;
   MatrixCoefficient *mcoeff = nullptr;
   InitCoeff(mesh, *trial_fec, dim, coeff_type, gf, coeff_fes, coeff, vcoeff,
             mcoeff);

   // Build the BilinearForm
   FiniteElementSpace trial_fes(&mesh, trial_fec);
   FiniteElementSpace test_fes(&mesh, test_fec);

   MixedBilinearForm k_ref(&trial_fes, &test_fes);
   MixedBilinearForm k_test(&trial_fes, &test_fes);
   MixedBilinearForm k_test_t(&test_fes, &trial_fes);
   auto AddIntegrator = [&bdr_integ](MixedBilinearForm &k,
                                     BilinearFormIntegrator *blfi)
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
   switch (pb)
   {
      case Problem::MixedVectorGradient:
         if (coeff)
         {
            AddIntegrator(k_ref, new MixedVectorGradientIntegrator(*coeff));
            AddIntegrator(k_test, new MixedVectorGradientIntegrator(*coeff));
            AddIntegrator(k_test_t, new MixedVectorWeakDivergenceIntegrator(*coeff));
         }
         else if (vcoeff)
         {
            AddIntegrator(k_ref, new MixedVectorGradientIntegrator(*vcoeff));
            AddIntegrator(k_test, new MixedVectorGradientIntegrator(*vcoeff));
            AddIntegrator(k_test_t, new MixedVectorWeakDivergenceIntegrator(*vcoeff));
         }
         else if (mcoeff)
         {
            AddIntegrator(k_ref, new MixedVectorGradientIntegrator(*mcoeff));
            AddIntegrator(k_test, new MixedVectorGradientIntegrator(*mcoeff));
            AddIntegrator(k_test_t, new MixedVectorWeakDivergenceIntegrator(*mcoeff));
         }
         break;
      case Problem::MixedVectorCurl:
         if (coeff)
         {
            AddIntegrator(k_ref, new MixedVectorCurlIntegrator(*coeff));
            AddIntegrator(k_test, new MixedVectorCurlIntegrator(*coeff));
            AddIntegrator(k_test_t, new MixedVectorWeakCurlIntegrator(*coeff));
         }
         else if (vcoeff)
         {
            AddIntegrator(k_ref, new MixedVectorCurlIntegrator(*vcoeff));
            AddIntegrator(k_test, new MixedVectorCurlIntegrator(*vcoeff));
            AddIntegrator(k_test_t, new MixedVectorWeakCurlIntegrator(*vcoeff));
         }
         else if (mcoeff)
         {
            AddIntegrator(k_ref, new MixedVectorCurlIntegrator(*mcoeff));
            AddIntegrator(k_test, new MixedVectorCurlIntegrator(*mcoeff));
            AddIntegrator(k_test_t, new MixedVectorWeakCurlIntegrator(*mcoeff));
         }
         break;
      default:
         MFEM_ABORT("Unexpected problem type.");
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   k_test_t.SetAssemblyLevel(assembly);
   k_test_t.Assemble();

   // Compare ceed with mfem
   GridFunction x(&trial_fes), y_ref(&test_fes), y_test(&test_fes);
   GridFunction x_t(&test_fes), y_t_ref(&trial_fes), y_t_test(&trial_fes);

   x.Randomize(1);

   k_ref.Mult(x, y_ref);
   k_test.Mult(x, y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12 * std::max(y_ref.Norml2(), 1.0));

   x_t.Randomize(1);

   k_ref.MultTranspose(x_t, y_t_ref);
   k_test_t.Mult(x_t, y_t_test);

   y_t_test.Add((pb == Problem::MixedVectorCurl) ? -1.0 : 1.0, y_t_ref);

   REQUIRE(y_t_test.Norml2() < 1.e-12 * std::max(y_t_ref.Norml2(), 1.0));
   delete gf;
   delete coeff_fes;
   delete coeff;
   delete vcoeff;
   delete mcoeff;
   delete trial_fec;
   delete test_fec;
}

void test_ceed_nloperator(const char *input, int order,
                          const CeedCoeffType coeff_type,
                          const NLProblem pb, const AssemblyLevel assembly)
{
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "coeff_type: " + GetString(coeff_type) + "\n" +
                         "pb: " + GetString(pb) + "\n" +
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
   MFEM_VERIFY(!vcoeff && !mcoeff,
               "Unexpected vector- or matrix-valued coefficient in test_ceed_nloperator.");

   // Build the NonlinearForm
   bool vecOp = pb == NLProblem::Convection;
   const int vdim = vecOp ? dim : 1;
   FiniteElementSpace fes(&mesh, &fec, vdim);

   NonlinearForm k_ref(&fes);
   NonlinearForm k_test(&fes);
   switch (pb)
   {
      case NLProblem::Convection:
         k_ref.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
         k_test.AddDomainIntegrator(new VectorConvectionNLFIntegrator(*coeff));
         break;
   }

   k_ref.Setup();
   k_test.SetAssemblyLevel(assembly);
   k_test.Setup();

   // Compare ceed with mfem
   GridFunction x(&fes), y_ref(&fes), y_test(&fes);

   x.Randomize(1);

   k_ref.Mult(x, y_ref);
   k_test.Mult(x, y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12 * std::max(y_ref.Norml2(), 1.0));
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
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
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

   REQUIRE(r.Norml2() < 1.e-12 * std::max(f.Norml2(), 1.0));
}

void test_ceed_full_assembly(const char *input, int order,
                             const AssemblyLevel assembly)
{
   std::string section = "assembly: " + GetString(assembly) + "\n" +
                         "order: " + std::to_string(order) + "\n" +
                         "mesh: " + input;
   INFO(section);
   Mesh mesh(input, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);

   DenseMatrix val(dim);
   val = 0.0;
   for (int i = 0; i < dim; i++)
   {
      val(i, i) = 1.0 + i;
   }
   MatrixConstantCoefficient diff_coeff(val);
   ConstantCoefficient mass_coeff(1.0);

   FiniteElementSpace fes(&mesh, &fec, 1);
   BilinearForm k_test(&fes);
   BilinearForm k_ref(&fes);

   k_ref.AddDomainIntegrator(new MassIntegrator(mass_coeff));
   k_test.AddDomainIntegrator(new MassIntegrator(mass_coeff));
   k_ref.AddBoundaryIntegrator(new MassIntegrator(mass_coeff));
   k_test.AddBoundaryIntegrator(new MassIntegrator(mass_coeff));
   k_ref.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   k_test.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   SparseMatrix *mat_ref = &k_ref.SpMat();
   SparseMatrix *mat_test = ceed::CeedOperatorFullAssemble(k_test);
   SparseMatrix *mat_diff = Add(1.0, *mat_ref, -1.0, *mat_test);

   REQUIRE(mat_diff->MaxNorm() < 1.e-12 * std::max(mat_ref->MaxNorm(), 1.0));
   delete mat_diff;
   delete mat_test;
}

TEST_CASE("CEED mass & diffusion", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,
                              CeedCoeffType::Quad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,Problem::MassDiffusion,
                      Problem::VectorMass,Problem::VectorDiffusion);
   auto order = GENERATE(1,2);
   auto bdr_integ = GENERATE(false,true);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/inline-tri.mesh",
                        "../../data/inline-tet.mesh",
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

TEST_CASE("CEED vector and matrix coefficients and vector FE operators",
          "[CEED], [VectorFE]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Quad,
                              CeedCoeffType::VecConst,CeedCoeffType::VecQuad,
                              CeedCoeffType::MatConst,CeedCoeffType::MatQuad);
   auto pb = GENERATE(Problem::Mass,Problem::Diffusion,
                      Problem::HDivMass,Problem::DivDiv,
                      Problem::HCurlMass,Problem::CurlCurl);
   auto order = GENERATE(1,3);
   auto bdr_integ = GENERATE(false,true);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/inline-tri.mesh",
                        "../../data/inline-tet.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   test_ceed_vectorfe_operator(mesh, order, coeff_type, pb, assembly, bdr_integ);
} // test case

TEST_CASE("CEED mixed integrators",
          "[CEED], [MixedVectorIntegrator], [VectorFE]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Quad,
                              CeedCoeffType::VecConst,CeedCoeffType::VecQuad,
                              CeedCoeffType::MatConst,CeedCoeffType::MatQuad);
   auto pb = GENERATE(Problem::MixedVectorGradient,Problem::MixedVectorCurl);
   auto order = GENERATE(2);
   auto bdr_integ = GENERATE(false,true);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/inline-tri.mesh",
                        "../../data/inline-tet.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   test_ceed_mixed_operator(mesh, order, coeff_type, pb, assembly, bdr_integ);
} // test case

TEST_CASE("CEED convection low", "[CEED], [Convection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::VecConst,CeedCoeffType::VecGrid,
                              CeedCoeffType::VecQuad);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/inline-tri.mesh",
                        "../../data/inline-tet.mesh",
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
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh");
   int high_order = 4;
   test_ceed_convection(mesh, high_order, assembly);
} // test case

TEST_CASE("CEED nonlinear convection", "[CEED], [NLConvection]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto coeff_type = GENERATE(CeedCoeffType::Const,CeedCoeffType::Grid,
                              CeedCoeffType::Quad);
   auto pb = GENERATE(NLProblem::Convection);
   auto order = GENERATE(1);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/inline-tri.mesh",
                        "../../data/inline-tet.mesh",
                        "../../data/periodic-square.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   test_ceed_nloperator(mesh, order, coeff_type, pb, assembly);
} // test case

TEST_CASE("CEED full assembly", "[CEED]")
{
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,AssemblyLevel::NONE);
   auto mesh = GENERATE("../../data/inline-quad.mesh",
                        "../../data/inline-hex.mesh",
                        "../../data/star-q2.mesh",
                        "../../data/fichera-q2.mesh",
                        "../../data/amr-quad.mesh",
                        "../../data/fichera-amr.mesh",
                        "../../data/square-mixed.mesh",
                        "../../data/fichera-mixed.mesh");
   int order = 1;
   test_ceed_full_assembly(mesh, order, assembly);
} // test case

#endif

} // namespace ceed_test
