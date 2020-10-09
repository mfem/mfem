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

#include "catch.hpp"
#include "mfem.hpp"

namespace mfem
{

constexpr double EPS = 1.e-12;

TEST_CASE("FormLinearSystem", "[FormLinearSystem]")
{
   for (int dim = 2; dim <=3; ++dim)
   {
      for (int ne = 1; ne <= 4; ++ne)
      {
         std::cout << "Testing " << dim << "D partial assembly: "
                   << std::pow(ne, dim) << " elements." << std::endl;
         for (int order = 1; order <= 3; ++order)
         {
            Mesh * mesh;
            if (dim == 2)
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection *fec = new H1_FECollection(order, dim);
            FiniteElementSpace fes(mesh, fec);

            Array<int> ess_tdof_list;
            Array<int> ess_bdr(mesh->bdr_attributes.Max());
            ess_bdr = 1;
            fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

            ConstantCoefficient one(1.0);
            GridFunction x0(&fes), x1(&fes), b(&fes);
            Vector B[2], X[2];

            OperatorPtr A_pa, A_fa;
            BilinearForm pa(&fes), fa(&fes);

            x0 = 0.0;
            b = 1.0;
            pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            pa.AddDomainIntegrator(new DiffusionIntegrator(one));
            pa.Assemble();
            pa.FormLinearSystem(ess_tdof_list, x0, b, A_pa, X[0], B[0]);
            OperatorJacobiSmoother M_pa(pa, ess_tdof_list);
            PCG(*A_pa, M_pa, B[0], X[0], 0, 1000, EPS*EPS, 0.0);
            pa.RecoverFEMSolution(X[0], b, x0);

            x1 = 0.0;
            b = 1.0;
            fa.SetAssemblyLevel(AssemblyLevel::FULL);
            fa.AddDomainIntegrator(new DiffusionIntegrator(one));
            fa.Assemble();
            fa.FormLinearSystem(ess_tdof_list, x1, b, A_fa, X[1], B[1]);
            GSSmoother M_fa((SparseMatrix&)(*A_fa));
            PCG(*A_fa, M_fa, B[1], X[1], 0, 1000, EPS*EPS, 0.0);
            fa.RecoverFEMSolution(X[1], b, x1);

            x0 -= x1;
            double error = x0.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(x0.Norml2() == Approx(EPS));

            delete mesh;
            delete fec;
         }
      }
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParallelFormLinearSystem", "[Parallel], [ParallelFormLinearSystem]")
{
   for (int dim = 2; dim <= 3; ++dim)
   {
      for (int ne = 4; ne <= 5; ++ne)
      {
         std::cout << "Testing " << dim << "D partial assembly: "
                   << std::pow(ne, dim) << " elements." << std::endl;
         for (int order = 1; order <= 3; ++order)
         {
            Mesh * mesh;
            if (dim == 2)
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
            ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
            delete mesh;

            FiniteElementCollection *fec = new H1_FECollection(order, dim);
            ParFiniteElementSpace fes(pmesh, fec);

            Array<int> ess_tdof_list;
            Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr = 1;
            fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

            ConstantCoefficient one(1.0);
            ParGridFunction x0(&fes), x1(&fes), b(&fes);
            Vector B[2], X[2];

            OperatorPtr A_pa, A_fa;
            ParBilinearForm pa(&fes), fa(&fes);

            x0 = 0.0;
            b = 1.0;
            pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            pa.AddDomainIntegrator(new DiffusionIntegrator(one));
            pa.Assemble();
            pa.FormLinearSystem(ess_tdof_list, x0, b, A_pa, X[0], B[0]);
            Solver *M_pa = new OperatorJacobiSmoother(pa, ess_tdof_list);
            CGSolver cg_pa(MPI_COMM_WORLD);
            cg_pa.SetRelTol(EPS);
            cg_pa.SetMaxIter(1000);
            cg_pa.SetPrintLevel(0);
            cg_pa.SetPreconditioner(*M_pa);
            cg_pa.SetOperator(*A_pa);
            cg_pa.Mult(B[0], X[0]);
            delete M_pa;
            pa.RecoverFEMSolution(X[0], b, x0);

            x1 = 0.0;
            b = 1.0;
            fa.SetAssemblyLevel(AssemblyLevel::FULL);
            fa.AddDomainIntegrator(new DiffusionIntegrator(one));
            fa.Assemble();
            fa.FormLinearSystem(ess_tdof_list, x1, b, A_fa, X[1], B[1]);
            HypreBoomerAMG *M_fa = new HypreBoomerAMG();
            CGSolver cg_fa(MPI_COMM_WORLD);
            cg_fa.SetRelTol(EPS);
            cg_fa.SetMaxIter(1000);
            cg_fa.SetPrintLevel(0);
            M_fa->SetPrintLevel(0);
            cg_fa.SetPreconditioner(*M_fa);
            cg_fa.SetOperator(*A_fa);
            cg_fa.Mult(B[1], X[1]);
            delete M_fa;
            fa.RecoverFEMSolution(X[1], b, x1);

            x0 -= x1;
            double error = x0.Norml2();
            std::cout << "    order: " << order << ", error norm: " << error << std::endl;
            REQUIRE(x0.Norml2() == Approx(EPS));

            delete pmesh;
            delete fec;
         }
      }
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
