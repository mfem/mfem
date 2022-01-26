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
#include "unit_tests.hpp"
#include "common_get_mesh.hpp"

#include <iostream>

using namespace mfem;
using namespace mfem_test_fem;

namespace bilinearform
{

static double a_ = 5.0;
static double b_ = 3.0;
static double c_ = 2.0;

TEST_CASE("Test order of boundary integrators",
          "[BilinearForm]")
{
   // Create a simple mesh
   int dim = 2, nx = 2, ny = 2, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, e_type);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   SECTION("Order of restricted boundary integrators")
   {
      ConstantCoefficient one(1.0);
      ConstantCoefficient two(2.0);
      ConstantCoefficient three(3.0);
      ConstantCoefficient four(4.0);

      Array<int> bdr1(4); bdr1 = 0; bdr1[0] = 1;
      Array<int> bdr2(4); bdr2 = 0; bdr2[1] = 1;
      Array<int> bdr3(4); bdr3 = 0; bdr3[2] = 1;
      Array<int> bdr4(4); bdr4 = 0; bdr4[3] = 1;

      BilinearForm a1234(&fes);
      a1234.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a1234.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a1234.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a1234.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a1234.Assemble(0);
      a1234.Finalize(0);

      BilinearForm a4321(&fes);
      a4321.AddBoundaryIntegrator(new MassIntegrator(four), bdr4);
      a4321.AddBoundaryIntegrator(new MassIntegrator(three), bdr3);
      a4321.AddBoundaryIntegrator(new MassIntegrator(two), bdr2);
      a4321.AddBoundaryIntegrator(new MassIntegrator(one), bdr1);
      a4321.Assemble(0);
      a4321.Finalize(0);

      const SparseMatrix &A1234 = a1234.SpMat();
      const SparseMatrix &A4321 = a4321.SpMat();

      SparseMatrix *D = Add(1.0, A1234, -1.0, A4321);

      REQUIRE(D->MaxNorm() == MFEM_Approx(0.0));

      delete D;
   }
}


TEST_CASE("FormLinearSystem/SolutionScope",
          "[BilinearForm]"
          "[CUDA]")
{
   // Create a simple mesh and FE space
   int dim = 2, nx = 2, ny = 2, order = 2;
   Element::Type e_type = Element::QUADRILATERAL;
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, e_type);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   int bdr_dof;

   // Solve a PDE on the conforming mesh and FE space defined above, storing the
   // result in 'sol'.
   auto SolvePDE = [&](AssemblyLevel al, GridFunction &sol)
   {
      // Linear form: rhs
      ConstantCoefficient f(1.0);
      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(f));
      b.Assemble();
      // Bilinear form: matrix
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.SetAssemblyLevel(al);
      a.Assemble();
      // Setup b.c.
      Array<int> ess_tdof_list;
      REQUIRE(mesh.bdr_attributes.Max() > 0);
      Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
      bdr_attr_is_ess = 1;
      fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
      REQUIRE(ess_tdof_list.Size() > 0);
      // Setup (on host) solution initial guess satisfying the desired b.c.
      ConstantCoefficient zero(0.0);
      sol.ProjectCoefficient(zero); // performed on host
      // Setup the linear system
      Vector B, X;
      OperatorPtr A;
      const bool copy_interior = true; // interior(sol) --> interior(X)
      a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B, copy_interior);
      // Solve the system
      CGSolver cg;
      cg.SetMaxIter(2000);
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.SetPrintLevel(0);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      // Recover the solution
      a.RecoverFEMSolution(X, b, sol);
      // Initialize the bdr_dof to be checked
      ess_tdof_list.HostRead();
      bdr_dof = AsConst(ess_tdof_list)[0]; // here, L-dof is the same T-dof
   };

   // Legacy full assembly
   {
      GridFunction sol(&fes);
      SolvePDE(AssemblyLevel::LEGACYFULL, sol);
      // Make sure the solution is still accessible after 'X' is destoyed
      sol.HostRead();
      REQUIRE(AsConst(sol)(bdr_dof) == 0.0);
   }

   // Partial assembly
   {
      GridFunction sol(&fes);
      SolvePDE(AssemblyLevel::PARTIAL, sol);
      // Make sure the solution is still accessible after 'X' is destoyed
      sol.HostRead();
      REQUIRE(AsConst(sol)(bdr_dof) == 0.0);
   }
}

enum FEType
{
   H1_FEC = 0,
   ND_FEC,
   RT_FEC,
   L2V_FEC,
   L2I_FEC,
};

TEST_CASE("BilinearForm Full Ops",
          "[BilinearForm]")
{
   int order = 2;
   double alpha = M_E;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      mesh->UniformRefinement();

      Vector oneVec(dim); oneVec = 1.0;

      ConstantCoefficient oneCoef(1.0);
      VectorConstantCoefficient oneVecCoef(oneVec);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         // if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         // { continue; }
         bool vec = (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC);

         if (dim == 1 && vec) { continue; }
         if (vec && (mt == (int)MeshType::WEDGE2 ||
                     mt == (int)MeshType::WEDGE4 ||
                     mt == (int)MeshType::MIXED3D6 ||
                     mt == (int)MeshType::MIXED3D8))
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec = new H1_FECollection(order, dim);
                  break;
               case FEType::ND_FEC:
                  fec = new ND_FECollection(order, dim);
                  break;
               case FEType::RT_FEC:
                  fec = new RT_FECollection(order-1, dim);
                  break;
               case FEType::L2V_FEC:
                  fec = new L2_FECollection(order-1, dim);
                  break;
               case FEType::L2I_FEC:
                  fec = new L2_FECollection(order, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace(mesh, fec);

            Array<int> ess_tdof_list;
            if (mesh->bdr_attributes.Size())
            {
               Array<int> ess_bdr(mesh->bdr_attributes.Max());
               ess_bdr = 1;
               fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
            }

            GridFunction u(&fespace);
            if (!vec)
            {
               u.ProjectCoefficient(oneCoef);
            }
            else
            {
               u.ProjectCoefficient(oneVecCoef);
            }

            BilinearForm a(&fespace);
            if (!vec)
            {
               a.AddDomainIntegrator(new MassIntegrator(oneCoef));
            }
            else
            {
               a.AddDomainIntegrator(new VectorFEMassIntegrator(oneCoef));
            }
            a.Assemble();

            LinearForm Au(&fespace);
            LinearForm ATu(&fespace);
            LinearForm aAu(&fespace);
            LinearForm aATu(&fespace);
            LinearForm b(&fespace);

            a.Mult(u, Au);
            a.MultTranspose(u, ATu);

            aAu = Au;
            aATu = ATu;

            a.AddMult(u, aAu, alpha);
            a.AddMultTranspose(u, aATu, alpha);

            // Modify the Bilinear Form
            OperatorPtr A;
            a.FormSystemMatrix(ess_tdof_list, A);

            a.FullMult(u, b);
            b -= Au;

            REQUIRE(b.Norml2() == MFEM_Approx( 0.0));

            a.FullMultTranspose(u, b);
            b -= ATu;

            REQUIRE(b.Norml2() == MFEM_Approx( 0.0));

            b = Au;
            a.FullAddMult(u, b, alpha);
            b -= aAu;

            REQUIRE(b.Norml2() == MFEM_Approx( 0.0));

            b = ATu;
            a.FullAddMultTranspose(u, b, alpha);
            b -= aATu;

            REQUIRE(b.Norml2() == MFEM_Approx( 0.0));

            delete fec;
         }
      }

      delete mesh;
   }
}

TEST_CASE("MixedBilinearform Full Ops",
          "[MixedBilinearForm]")
{
   int order = 2;
   double alpha = M_E;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      mesh->UniformRefinement();

      Vector oneVec(dim); oneVec = 1.0;

      ConstantCoefficient oneCoef(1.0);
      VectorConstantCoefficient oneVecCoef(oneVec);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         bool vec = (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC);

         if (dim == 1 && vec) { continue; }
         if (vec && (mt == (int)MeshType::WEDGE2 ||
                     mt == (int)MeshType::WEDGE4 ||
                     mt == (int)MeshType::MIXED3D6 ||
                     mt == (int)MeshType::MIXED3D8))
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec_dom = NULL;
            FiniteElementCollection *fec_ran = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec_dom = new H1_FECollection(order, dim);
                  fec_ran = new H1_FECollection(order-1, dim);
                  break;
               case FEType::ND_FEC:
                  fec_dom = new ND_FECollection(order, dim);
                  fec_ran = new RT_FECollection(order-1, dim);
                  break;
               case FEType::RT_FEC:
                  fec_dom = new RT_FECollection(order-1, dim);
                  fec_ran = new ND_FECollection(order, dim);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace_dom(mesh, fec_dom);
            FiniteElementSpace fespace_ran(mesh, fec_ran);

            Array<int> ess_tdof_list_dom;
            Array<int> ess_tdof_list_ran;
            if (mesh->bdr_attributes.Size())
            {
               Array<int> ess_bdr(mesh->bdr_attributes.Max());
               ess_bdr = 1;
               fespace_dom.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_dom);
               fespace_ran.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_ran);
            }

            GridFunction u_dom(&fespace_dom);
            GridFunction u_ran(&fespace_ran);
            if (!vec)
            {
               u_dom.ProjectCoefficient(oneCoef);
               u_ran.ProjectCoefficient(oneCoef);
            }
            else
            {
               u_dom.ProjectCoefficient(oneVecCoef);
               u_ran.ProjectCoefficient(oneVecCoef);
            }

            MixedBilinearForm a(&fespace_dom, &fespace_ran);
            if (!vec)
            {
               a.AddDomainIntegrator(new MassIntegrator(oneCoef));
            }
            else
            {
               a.AddDomainIntegrator(new VectorFEMassIntegrator(oneCoef));
            }
            a.Assemble();

            LinearForm Au(&fespace_ran);
            LinearForm ATu(&fespace_dom);
            LinearForm aAu(&fespace_ran);
            LinearForm aATu(&fespace_dom);
            LinearForm b_ran(&fespace_ran);
            LinearForm b_dom(&fespace_dom);

            a.Mult(u_dom, Au);
            a.MultTranspose(u_ran, ATu);

            aAu = Au;
            aATu = ATu;

            a.AddMult(u_dom, aAu, alpha);
            a.AddMultTranspose(u_ran, aATu, alpha);

            // Modify the Bilinear Form
            OperatorPtr A;
            a.FormRectangularSystemMatrix(ess_tdof_list_dom,
                                          ess_tdof_list_ran, A);

            a.FullMult(u_dom, b_ran);
            b_ran -= Au;

            REQUIRE(b_ran.Norml2() == MFEM_Approx( 0.0));

            a.FullMultTranspose(u_ran, b_dom);
            b_dom -= ATu;

            REQUIRE(b_dom.Norml2() == MFEM_Approx( 0.0));

            b_ran = Au;
            a.FullAddMult(u_dom, b_ran, alpha);
            b_ran -= aAu;

            REQUIRE(b_ran.Norml2() == MFEM_Approx( 0.0));

            b_dom = ATu;
            a.FullAddMultTranspose(u_ran, b_dom, alpha);
            b_dom -= aATu;

            REQUIRE(b_dom.Norml2() == MFEM_Approx( 0.0));

            delete fec_dom;
            delete fec_ran;
         }
      }

      delete mesh;
   }
}

} // namespace bilinearform
