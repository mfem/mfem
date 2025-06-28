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

#include "unit_tests.hpp"
#include "mfem.hpp"
#include "linalg/dtensor.hpp"
#include <math.h> // M_PI
#include <fstream>
#include <iostream>

using namespace mfem;

namespace assembly_levels
{

enum class Problem { Mass,
                     Convection,
                     Diffusion
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
   }
   MFEM_ABORT("Unknown Problem.");
   return "";
}

std::string getString(AssemblyLevel assembly)
{
   switch (assembly)
   {
      case AssemblyLevel::NONE:
         return "None";
         break;
      case AssemblyLevel::PARTIAL:
         return "Partial";
         break;
      case AssemblyLevel::ELEMENT:
         return "Element";
         break;
      case AssemblyLevel::FULL:
         return "Full";
         break;
      case AssemblyLevel::LEGACY:
         return "Legacy";
         break;
   }
   MFEM_ABORT("Unknown assembly level.");
   return "";
}

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = x(1); v(1) = -x(0); break;
      case 3: v(0) = x(1); v(1) = -x(0); v(2) = x(0); break;
   }
}

void AddConvectionIntegrators(BilinearForm &k, VectorCoefficient &velocity,
                              bool dg)
{
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));

   if (dg)
   {
      k.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
      k.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   }
}

void test_assembly_level(const char *meshname,
                         int order, int q_order_inc, bool dg,
                         const Problem pb, const AssemblyLevel assembly)
{
   const int q_order = 2*order + q_order_inc;

   INFO("mesh=" << meshname
        << ", order=" << order << ", q_order=" << q_order << ", DG=" << dg
        << ", pb=" << getString(pb) << ", assembly=" << getString(assembly));
   Mesh mesh(meshname, 1, 1);
   mesh.RemoveInternalBoundaries();
   mesh.EnsureNodes();
   const int dim = mesh.Dimension();

   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.SetAttribute(e, 1 + (e % 2));
   }
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      mesh.SetBdrAttribute(be, 1 + (be % 2));
   }
   mesh.SetAttributes();

   Array<int> elem_marker({1, 0}), bdr_marker({1, 0});
   // Periodic meshes = no boundary attributes, don't use markers
   if (mesh.bdr_attributes.Size() == 0) { bdr_marker.DeleteAll(); }

   std::unique_ptr<FiniteElementCollection> fec;
   if (dg)
   {
      fec.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }

   FiniteElementSpace fespace(&mesh, fec.get());

   BilinearForm k_test(&fespace);
   BilinearForm k_ref(&fespace);

   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient vel_coeff(dim, velocity_function);

   // Don't use a special integration rule if q_order_inc == 0
   const bool use_ir = q_order_inc > 0;
   const IntegrationRule *ir =
      use_ir ? &IntRules.Get(mesh.GetTypicalElementGeometry(), q_order) : nullptr;

   const IntegrationRule &ir_face =
      IntRules.Get(mesh.GetTypicalFaceGeometry(), q_order);

   switch (pb)
   {
      case Problem::Mass:
         k_ref.AddDomainIntegrator(new MassIntegrator(one,ir), elem_marker);
         k_test.AddDomainIntegrator(new MassIntegrator(one,ir), elem_marker);
         if (!dg && mesh.Conforming() && assembly != AssemblyLevel::FULL)
         {
            k_ref.AddBoundaryIntegrator(new MassIntegrator(one, &ir_face), bdr_marker);
            k_test.AddBoundaryIntegrator(new MassIntegrator(one, &ir_face), bdr_marker);
         }
         break;
      case Problem::Convection:
         AddConvectionIntegrators(k_ref, vel_coeff, dg);
         AddConvectionIntegrators(k_test, vel_coeff, dg);
         break;
      case Problem::Diffusion:
         k_ref.AddDomainIntegrator(new DiffusionIntegrator(one,ir));
         k_test.AddDomainIntegrator(new DiffusionIntegrator(one,ir));
         break;
   }

   k_ref.Assemble();
   k_ref.Finalize();

   k_test.SetAssemblyLevel(assembly);
   k_test.Assemble();

   GridFunction x(&fespace), y_ref(&fespace), y_test(&fespace);

   x.Randomize(1);

   // Test Mult
   k_ref.Mult(x,y_ref);
   k_test.Mult(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);

   // Test MultTranspose
   k_ref.MultTranspose(x,y_ref);
   k_test.MultTranspose(x,y_test);

   y_test -= y_ref;

   REQUIRE(y_test.Norml2() < 1.e-12);
}

TEST_CASE("H1 Assembly Levels", "[AssemblyLevel], [PartialAssembly], [GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const bool dg = false;
   auto pb = GENERATE(Problem::Mass, Problem::Convection, Problem::Diffusion);
   auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                            AssemblyLevel::ELEMENT,
                            AssemblyLevel::FULL);
   // '0' will use the default integration rule
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   SECTION("Conforming")
   {
      SECTION("2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-square.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/periodic-hexagon.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/star-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }

      SECTION("3D")
      {
         auto order = !all_tests ? GENERATE(2) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-cube.mesh",
                             order, q_order_inc, dg, pb, assembly);
         if ( !Device::Allows(~Backend::CPU_MASK) )
         {
            test_assembly_level("../../data/fichera-q3.mesh",
                                order, q_order_inc, dg, pb, assembly);
         }
      }
   }

   SECTION("Nonconforming")
   {
      // Test AMR cases
      SECTION("AMR 2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/amr-quad.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
      SECTION("AMR 3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/fichera-amr.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }
} // H1 Assembly Levels test case

TEST_CASE("H(div) Element Assembly", "[AssemblyLevel][GPU]")
{
   const auto fname = GENERATE(
                         "../../data/inline-quad.mesh",
                         "../../data/star-q3.mesh",
                         "../../data/inline-hex.mesh",
                         "../../data/fichera-q2.mesh"
                      );
   const auto order = GENERATE(1, 2);
   const auto problem = GENERATE(Problem::Mass, Problem::Diffusion);

   CAPTURE(fname, order, getString(problem));

   Mesh mesh(fname);
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   RT_FECollection fec(order - 1, dim);
   FiniteElementSpace fes(&mesh, &fec);

   std::unique_ptr<BilinearFormIntegrator> integ;
   if (problem == Problem::Mass) { integ.reset(new VectorFEMassIntegrator); }
   else if (problem == Problem::Diffusion) { integ.reset(new DivDivIntegrator); }

   const FiniteElement &fe = *fes.GetFE(0);
   {
      ElementTransformation &T = *mesh.GetElementTransformation(0);
      integ->SetIntegrationRule(MassIntegrator::GetRule(fe, fe, T));
   }

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement*>(&fe);
   MFEM_VERIFY(tbe, "");
   const int ndof = fes.GetFE(0)->GetDof();
   const Array<int> &dof_map = tbe->GetDofMap();

   Vector ea_data(ne*ndof*ndof);
   integ->AssembleEA(fes, ea_data, false);
   const auto ea_mats = Reshape(ea_data.HostRead(), ndof, ndof, ne);

   DenseMatrix elmat;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &el = *fes.GetFE(e);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      integ->AssembleElementMatrix(el, T, elmat);

      for (int i = 0; i < ndof; ++i)
      {
         const int ii_s = dof_map[i];
         const int ii = ii_s >= 0 ? ii_s : -1 - ii_s;
         const int s_i = ii_s >= 0 ? 1 : -1;
         for (int j = 0; j < ndof; ++j)
         {
            const int jj_s = dof_map[j];
            const int jj = jj_s >= 0 ? jj_s : -1 - jj_s;
            const int s_j = jj_s >= 0 ? 1 : -1;
            elmat(ii, jj) -= s_i*s_j*ea_mats(i, j, e);
         }
      }

      REQUIRE(elmat.MaxMaxNorm() == MFEM_Approx(0.0, 1e-10));
   }
}

TEST_CASE("NormalTraceJumpIntegrator Element Assembly", "[AssemblyLevel][GPU]")
{
   const auto fname = GENERATE(
                         "../../data/inline-quad.mesh",
                         "../../data/star-q3.mesh",
                         "../../data/inline-hex.mesh",
                         "../../data/fichera-q3.mesh"
                      );
   const int order = GENERATE(1, 2, 3);

   CAPTURE(fname, order);

   Mesh mesh(fname);
   const int dim = mesh.Dimension();

   RT_FECollection fec(order - 1, dim);
   FiniteElementSpace fes(&mesh, &fec);

   DG_Interface_FECollection hfec(order - 1, dim);
   FiniteElementSpace hfes(&mesh, &hfec);

   NormalTraceJumpIntegrator integ;

   const int nf = mesh.GetNFbyType(FaceType::Interior);
   const int ndof_trial = hfes.GetFaceElement(0)->GetDof();
   const int ndof_test = fes.GetFE(0)->GetDof();
   Vector emat(ndof_trial*ndof_test*2*nf);
   integ.AssembleEAInteriorFaces(hfes, fes, emat, false);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement*>(fes.GetFE(0));
   MFEM_VERIFY(tbe, "");
   const Array<int> &dof_map = tbe->GetDofMap();

   const auto e_mat = Reshape(emat.HostRead(), ndof_test, ndof_trial, 2, nf);

   int fidx = 0;
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      const Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (!info.IsInterior()) { continue; }

      const int el1 = info.element[0].index;
      const int el2 = info.element[1].index;

      FaceElementTransformations *FTr = mesh.GetInteriorFaceTransformations(f);

      DenseMatrix elmat;
      integ.AssembleFaceMatrix(*hfes.GetFaceElement(f),
                               *fes.GetFE(el1),
                               *fes.GetFE(el2),
                               *FTr, elmat);
      elmat.Threshold(1e-12 * elmat.MaxMaxNorm());
      for (int ie = 0; ie < 2; ++ie)
      {
         for (int i_lex = 0; i_lex < ndof_test; ++i_lex)
         {
            const int i_s = dof_map[i_lex];
            const int i = (i_s >= 0) ? i_s : -1 - i_s;
            for (int j = 0; j < ndof_trial; ++j)
            {
               elmat(i + ie*ndof_test, j) -= e_mat(i_lex, j, ie, fidx);
            }
         }
      }
      REQUIRE(elmat.MaxMaxNorm() == MFEM_Approx(0.0));

      fidx++;
   }
}

TEST_CASE("L2 Assembly Levels", "[AssemblyLevel], [PartialAssembly], [GPU]")
{
   const bool dg = true;
   auto pb = GENERATE(Problem::Mass, Problem::Convection);
   const bool all_tests = launch_all_non_regression_tests;
   // '0' will use the default integration rule
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   SECTION("Conforming")
   {
      auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                               AssemblyLevel::ELEMENT,
                               AssemblyLevel::FULL);

      SECTION("2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-square.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/periodic-hexagon.mesh",
                             order, q_order_inc, dg, pb, assembly);
         test_assembly_level("../../data/star-q3.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }

      SECTION("3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/periodic-cube.mesh",
                             order, q_order_inc, dg, pb, assembly);
         if ( !Device::Allows(~Backend::CPU_MASK) )
         {
            test_assembly_level("../../data/fichera-q3.mesh",
                                order, q_order_inc, dg, pb, assembly);
         }
      }
   }

   SECTION("Nonconforming")
   {
      auto assembly = GENERATE(AssemblyLevel::PARTIAL,
                               AssemblyLevel::ELEMENT,
                               AssemblyLevel::FULL);

      SECTION("AMR 2D")
      {
         auto order = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
         test_assembly_level("../../data/amr-quad.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
      SECTION("AMR 3D")
      {
         auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
         test_assembly_level("../../data/fichera-amr.mesh",
                             order, q_order_inc, dg, pb, assembly);
      }
   }
} // L2 Assembly Levels test case

#ifndef MFEM_USE_MPI
#define HYPRE_BigInt int
#endif // MFEM_USE_MPI

void CompareMatricesNonZeros(SparseMatrix &A1, const SparseMatrix &A2,
                             HYPRE_BigInt *cmap1=nullptr,
                             std::unordered_map<HYPRE_BigInt,int> *cmap2inv=nullptr)
{
   bool A1_Heigh_equals_A2_Height = A1.Height() == A2.Height();
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized() && !Mpi::IsFinalized())
   {
      const bool in = A1_Heigh_equals_A2_Height;
      MPI_Allreduce(&in, &A1_Heigh_equals_A2_Height, 1, MPI_C_BOOL, MPI_LAND,
                    MPI_COMM_WORLD);
   }
#endif
   REQUIRE(A1_Heigh_equals_A2_Height);
   int n = A1.Height();

   const int *I1 = A1.HostReadI();
   const int *J1 = A1.HostReadJ();
   const real_t *V1 = A1.HostReadData();

   A2.HostReadI();
   A2.HostReadJ();
   A2.HostReadData();

   real_t error = 0.0;

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         if (cmap1)
         {
            if (cmap2inv->count(cmap1[j]) > 0)
            {
               j = (*cmap2inv)[cmap1[j]];
            }
            else
            {
               error = std::max(error, std::fabs(V1[jj]));
               continue;
            }
         }
         error = std::max(error, std::fabs(V1[jj] - A2(i,j)));
      }
   }

#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized() && !Mpi::IsFinalized())
   {
      const real_t in = error;
      MPI_Allreduce(&in, &error, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    MPI_COMM_WORLD);
   }
#endif
   REQUIRE(error == MFEM_Approx(0.0, 1e-10));
}

void TestSameSparseMatrices(OperatorHandle &A1, OperatorHandle &A2)
{
   SparseMatrix *M1 = A1.Is<SparseMatrix>();
   SparseMatrix *M2 = A2.Is<SparseMatrix>();

   REQUIRE(M1 != NULL);
   REQUIRE(M2 != NULL);

   CompareMatricesNonZeros(*M1, *M2);
   CompareMatricesNonZeros(*M2, *M1);
}

void TestH1FullAssembly(Mesh &mesh, int order)
{
   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_tdof_list;
   fespace.GetBoundaryTrueDofs(ess_tdof_list);

   BilinearForm a_fa(&fespace);
   BilinearForm a_legacy(&fespace);

   a_fa.SetAssemblyLevel(AssemblyLevel::FULL);
   a_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);

   a_fa.AddDomainIntegrator(new DiffusionIntegrator);
   a_legacy.AddDomainIntegrator(new DiffusionIntegrator);

   a_fa.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_fa.Assemble();

   a_legacy.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_legacy.Assemble();
   a_legacy.Finalize();

   OperatorHandle A_fa, A_legacy;
   // Test that FormSystemMatrix gives the same result
   a_fa.FormSystemMatrix(ess_tdof_list, A_fa);
   a_legacy.FormSystemMatrix(ess_tdof_list, A_legacy);

   TestSameSparseMatrices(A_fa, A_legacy);

   // Test that FormLinearSystem gives the same result
   GridFunction x1(&fespace);
   LinearForm b1(&fespace);

   x1.Randomize(1);
   b1.Randomize(2);

   Vector x2(x1);
   Vector b2(b1);

   Vector X1, X2, B1, B2;

   a_fa.Assemble();

   a_fa.FormLinearSystem(ess_tdof_list, x1, b1, A_fa, X1, B1);
   a_legacy.FormLinearSystem(ess_tdof_list, x2, b2, A_legacy, X2, B2);

   TestSameSparseMatrices(A_fa, A_legacy);

   B1 -= B2;
   REQUIRE(B1.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("Serial H1 Full Assembly", "[AssemblyLevel], [GPU]")
{
   auto order = GENERATE(1, 2, 3);
   auto mesh_fname = GENERATE(
                        "../../data/star.mesh",
                        "../../data/fichera.mesh"
                     );
   Mesh mesh(mesh_fname);
   TestH1FullAssembly(mesh, order);
}

TEST_CASE("Full Assembly Connectivity", "[AssemblyLevel], [GPU]")
{
   const int order = GENERATE(1, 2, 3);
   const int ne = GENERATE(4, 8, 16, 32);

   // Create a "star-shaped" quad mesh, where all elements share one vertex at
   // the origin, and the other vertices are distributed radially in a zig-zag
   // pattern.
   //
   // The valence of the center vertex is equal to the number of elements in the
   // mesh.
   const int nv = 2*ne + 1;
   Mesh mesh(2, nv, ne, 0);
   mesh.AddVertex(0.0, 0.0);
   for (int i = 0; i < 2*ne; ++i)
   {
      const real_t theta = 2*M_PI*i / real_t(2*ne);
      const real_t r = (i%2 == 0) ? 1.0 : 0.75;
      mesh.AddVertex(r*cos(theta), r*sin(theta));
   }
   for (int i = 0; i < ne; ++i)
   {
      const int base = 2 * i;
      mesh.AddQuad(0, base + 2, base + 1, i == 0 ? 2*ne : base);
   }
   mesh.FinalizeMesh();

   TestH1FullAssembly(mesh, order);
}

#ifdef MFEM_USE_MPI

void CompareMatricesNonZeros(HypreParMatrix &A1, const HypreParMatrix &A2)
{
   HYPRE_BigInt *cmap1, *cmap2;
   SparseMatrix diag1, offd1, diag2, offd2;

   A1.GetDiag(diag1);
   A2.GetDiag(diag2);
   A1.GetOffd(offd1, cmap1);
   A2.GetOffd(offd2, cmap2);

   CompareMatricesNonZeros(diag1, diag2);

   if (cmap1)
   {
      std::unordered_map<HYPRE_BigInt,int> cmap2inv;
      for (int i=0; i<offd2.Width(); ++i) { cmap2inv[cmap2[i]] = i; }
      CompareMatricesNonZeros(offd1, offd2, cmap1, &cmap2inv);
   }
   else
   {
      CompareMatricesNonZeros(offd1, offd2);
   }
}

void TestSameHypreMatrices(OperatorHandle &A1, OperatorHandle &A2)
{
   HypreParMatrix *M1 = A1.Is<HypreParMatrix>();
   HypreParMatrix *M2 = A2.Is<HypreParMatrix>();

   REQUIRE(M1 != NULL);
   REQUIRE(M2 != NULL);

   CompareMatricesNonZeros(*M1, *M2);
   CompareMatricesNonZeros(*M2, *M1);
}

TEST_CASE("Parallel H1 Full Assembly", "[AssemblyLevel], [Parallel], [GPU]")
{
   auto order = GENERATE(1, 2, 3);
   auto mesh_fname = GENERATE(
                        "../../data/star.mesh",
                        "../../data/fichera.mesh"
                     );

   // CAPTURE(order, mesh_fname);

   Mesh serial_mesh(mesh_fname);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_tdof_list;
   fespace.GetBoundaryTrueDofs(ess_tdof_list);

   ParBilinearForm a_fa(&fespace);
   ParBilinearForm a_legacy(&fespace);

   a_fa.SetAssemblyLevel(AssemblyLevel::FULL);
   a_fa.SetDiagonalPolicy(Operator::DIAG_ONE);
   a_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);
   a_legacy.SetDiagonalPolicy(Operator::DIAG_ONE);

   a_fa.AddDomainIntegrator(new DiffusionIntegrator);
   a_legacy.AddDomainIntegrator(new DiffusionIntegrator);

   a_fa.Assemble();
   a_legacy.Assemble();
   a_legacy.Finalize();

   OperatorHandle A_fa, A_legacy;

   DYNAMIC_SECTION("[order: " << order << ", dim: " << dim
                   << "]: (1) ParallelAssemble")
   {
      // Test that ParallelAssemble gives the same result
      A_fa.Reset(a_fa.ParallelAssemble());
      A_legacy.Reset(a_legacy.ParallelAssemble());

      TestSameHypreMatrices(A_fa, A_legacy);
   }

   DYNAMIC_SECTION("[order: " << order << ", dim: " << dim
                   << "]: (2) FormSystemMatrix")
   {
      // Test that FormSystemMatrix gives the same result
      a_fa.FormSystemMatrix(ess_tdof_list, A_fa);
      a_legacy.FormSystemMatrix(ess_tdof_list, A_legacy);

      TestSameHypreMatrices(A_fa, A_legacy);
   }

   // Test that FormLinearSystem gives the same result
   ParGridFunction x1(&fespace);
   ParLinearForm b1(&fespace);

   x1.Randomize(1);
   b1.Randomize(2);

   Vector x2(x1);
   Vector b2(b1);

   Vector X1, X2, B1, B2;

   a_fa.Assemble();

   DYNAMIC_SECTION("[order: " << order << ", dim: " << dim
                   << "]: (3) FormLinearSystem")
   {
      a_fa.FormLinearSystem(ess_tdof_list, x1, b1, A_fa, X1, B1);
      a_legacy.FormLinearSystem(ess_tdof_list, x2, b2, A_legacy, X2, B2);

      TestSameHypreMatrices(A_fa, A_legacy);
   }

   DYNAMIC_SECTION("[order: " << order << ", dim: " << dim
                   << "]: (4) FormLinearSystem - RHS")
   {
      B1 -= B2;
      const real_t B_err = GlobalLpNorm(infinity(), B1.Normlinf(),
                                        MPI_COMM_WORLD);
      REQUIRE(B_err == MFEM_Approx(0.0));
   }
}

#endif

} // namespace assembly_levels
