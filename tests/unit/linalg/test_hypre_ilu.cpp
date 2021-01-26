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

#include "unit_tests.hpp"
#include "mfem.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI
#if MFEM_HYPRE_VERSION >= 21900


TEST_CASE("HypreILU and HypreFGMRES wrappers",
          "[Parallel], [HypreILU], [HypreFGMRES]")
{
   // Build a small diffusion problem to test the solver and preconditioner
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int ne = 40;
   Mesh mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FiniteElementCollection *fec = new H1_FECollection(1, 2);
   ParFiniteElementSpace fespace(&pmesh, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   ParGridFunction x(&fespace);
   x = 0.0;

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreSolver *ilu = new HypreILU();
   HYPRE_ILUSetLevelOfFill(*ilu, 4); // fill level of 4

   HypreFGMRES fgmres(MPI_COMM_WORLD);
   const double tol = 1e-10;
   fgmres.SetTol(tol);
   fgmres.SetMaxIter(100);
   fgmres.SetPrintLevel(0);
   fgmres.SetKDim(100);
   fgmres.SetPreconditioner(*ilu);
   fgmres.SetOperator(*A);
   fgmres.Mult(B, X);

   HYPRE_Int converged;
   HYPRE_FlexGMRESGetConverged(fgmres, &converged);
   REQUIRE(converged == 1);

   HYPRE_Real rel_resid;
   HYPRE_FlexGMRESGetFinalRelativeResidualNorm(fgmres, &rel_resid);
   REQUIRE(rel_resid < tol);

   delete ilu;
   delete fec;
}

#endif // MFEM_USE_MPI
#endif // MFEM_HYPRE_VERSION

} // namespace mfem
