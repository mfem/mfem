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

#include "mfem.hpp"
#include "unit_tests.hpp"

namespace mfem
{

static real_t exact_sln(const Vector &p);
static void TestSolve(FiniteElementSpace &fespace);
static void TestSolveVec(FiniteElementSpace &fespace);
#ifdef MFEM_USE_MPI
static void TestSolvePar(ParFiniteElementSpace &fespace);
static void TestSolveParVec(ParFiniteElementSpace &fespace);
static void TestRandomPRefinement(Mesh & mesh);
#endif

namespace var_order_test { enum class SpaceType {RT, ND}; }

Mesh MakeCartesianMesh(int nx, int dim)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true);
   }
   else
   {
      return Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   }
}

// Check basic functioning of variable order spaces, hp interpolation and
// some corner cases.
TEST_CASE("Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh]")
{
   SECTION("Quad mesh")
   {
      // 2-element quad mesh
      Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 6);
      REQUIRE(fespace.GetNConformingDofs() == 6);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 11);
      REQUIRE(fespace.GetNConformingDofs() == 10);

      // h-refine first element in the y axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 2));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 11);

      // relax the master edge to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      // 15 quadratic + 16 cubic DOFs - 2 shared vertices:
      REQUIRE(fespace.GetNDofs() == 29);
      // 3 constrained DOFs on slave side, inexact interpolation
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      // new quadratic DOF on master edge:
      REQUIRE(fespace.GetNDofs() == 30);
      // 3 constrained DOFs on slave side, 2 on master side:
      REQUIRE(fespace.GetNConformingDofs() == 25);

      TestSolve(fespace);

      // refine
      mesh.UniformRefinement();
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 93);
      REQUIRE(fespace.GetNConformingDofs() == 83);

      TestSolve(fespace);
   }

   SECTION("Quad/hex mesh projection")
   {
      for (int dim=2; dim<=3; ++dim)
      {
         // 2-element mesh
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL) :
                     Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();

         // h-refine element 1
         Array<Refinement> refinements;
         refinements.Append(Refinement(1));

         int nonconformity_limit = 0; // 0 meaning allow unlimited ratio
         mesh.GeneralRefinement(refinements, 1, nonconformity_limit);  // h-refinement

         // standard H1 space with order 2 elements
         H1_FECollection fec(2, mesh.Dimension());
         FiniteElementSpace fespace(&mesh, &fec);

         GridFunction x(&fespace);

         // p-refine element 0
         fespace.SetElementOrder(0, 3);

         fespace.Update(false);
         x.SetSpace(&fespace);

         // Test projection of the coefficient
         FunctionCoefficient exsol(exact_sln);
         x.ProjectCoefficient(exsol);

         // Enforce space constraints on locally interpolated GridFunction x
         const SparseMatrix *R = fespace.GetHpRestrictionMatrix();
         const SparseMatrix *P = fespace.GetConformingProlongation();
         Vector y(fespace.GetTrueVSize());
         R->Mult(x, y);
         P->Mult(y, x);

         const real_t error = x.ComputeL2Error(exsol);
         REQUIRE(error == MFEM_Approx(0.0));
      }
   }

   SECTION("Hex mesh")
   {
      // 2-element hex mesh
      Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
      mesh.EnsureNCMesh();

      // standard H1 space with order 1 elements
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 12);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 31);
      REQUIRE(fespace.GetNConformingDofs() == 26);

      // h-refine first element in the z axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 28);

      // relax the master face to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 35);
      REQUIRE(fespace.GetNConformingDofs() == 31);

      // increase order
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 105);
      REQUIRE(fespace.GetNConformingDofs() == 92);

      // relaxed off
      fespace.SetRelaxedHpConformity(false);

      REQUIRE(fespace.GetNDofs() == 108);
      REQUIRE(fespace.GetNConformingDofs() == 87);

      // refine one of the small elements into four
      refs[0].SetType(3);
      mesh.GeneralRefinement(refs);
      fespace.Update();

      REQUIRE(fespace.GetNDofs() == 162);
      REQUIRE(fespace.GetNConformingDofs() == 115);

      TestSolve(fespace);

      // lower the order of one of the four new elements to 1 - this minimum
      // order will propagate through two master faces and severely constrain
      // the space (since relaxed hp is off)
      fespace.SetElementOrder(0, 1);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 152);
      REQUIRE(fespace.GetNConformingDofs() == 92);
   }

   SECTION("Prism mesh")
   {
      // 2-element prism mesh
      Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::WEDGE);
      mesh.EnsureNCMesh();

      // standard H1 space with order 2 elements
      H1_FECollection fec(2, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 27);
      REQUIRE(fespace.GetNConformingDofs() == 27);

      // convert to variable order space: p-refine first element
      fespace.SetElementOrder(0, 3);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 54);
      REQUIRE(fespace.GetNConformingDofs() == 42);

      // refine to form an edge-face constraint similar to
      // https://github.com/mfem/mfem/pull/713#issuecomment-495786362
      Array<Refinement> refs;
      refs.Append(Refinement(1, 3));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      refs[0].SetType(4);
      refs.Append(Refinement(2, 4));
      mesh.GeneralRefinement(refs);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 113);
      REQUIRE(fespace.GetNConformingDofs() == 67);

      TestSolve(fespace);
   }

   SECTION("Quad/hex mesh ND/RT")
   {
      using namespace var_order_test;

      const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
      const int dim = GENERATE(2, 3);

      Mesh mesh = MakeCartesianMesh(dim == 2 ? 4 : 2, dim);
      mesh.EnsureNCMesh();

      int ndof0, ncdof1, ncdof2, ndof1;
      std::unique_ptr<FiniteElementCollection> fec;
      if (space_type == SpaceType::RT)
      {
         // Standard RT space with order 0 elements
         fec.reset(new RT_FECollection(0, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ndof1 = 62;
            ncdof1 = 56;
            ncdof2 = 312;
         }
         else
         {
            ndof0 = 36;
            ndof1 = 141;
            ncdof1 = 114;
            ncdof2 = 756;
         }
      }
      else
      {
         // Standard ND space with order 1 elements
         fec.reset(new ND_FECollection(1, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ndof1 = 50;
            ncdof1 = 46;
            ncdof2 = 144;
         }
         else
         {
            ndof0 = 54;
            ndof1 = 105;
            ncdof1 = 75;
            ncdof2 = 300;
         }
      }

      FiniteElementSpace fespace(&mesh, fec.get());

      REQUIRE(fespace.GetNDofs() == ndof0);
      REQUIRE(fespace.GetNConformingDofs() == ndof0);

      // Convert to variable order space: p-refine first element
      fespace.SetElementOrder(0, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == ndof1);
      REQUIRE(fespace.GetNConformingDofs() == ncdof1);

      // p-refine all elements
      for (int i = 1; i < mesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GetNConformingDofs() == ncdof2);

      TestSolveVec(fespace);
   }
}

#ifdef MFEM_USE_MPI
TEST_CASE("Parallel Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{
   SECTION("Quad mesh")
   {
      // 2-by-2 element quad mesh
      Mesh mesh = MakeCartesianMesh(2, 2);
      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      // Standard H1 space with order 1 elements
      H1_FECollection fe_coll(1, pmesh.Dimension());
      ParFiniteElementSpace pfes(&pmesh, &fe_coll);

      REQUIRE(pfes.GlobalTrueVSize() == 9);

      // Convert to variable order space by p-refinement
      // Increase order on all elements
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pfes.SetElementOrder(i, pfes.GetElementOrder(i) + 1);
      }
      pfes.Update(false);

      // DOFs for vertices + edges + elements = 9 + 12 + 4 = 25
      REQUIRE(pfes.GlobalTrueVSize() == 25);

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0) { pfes.SetElementOrder(0, 4); }
      pfes.Update(false);

      Array<Refinement> refs;
      if (rank == 0) { refs.Append(Refinement(0)); }
      pmesh.GeneralRefinement(refs);
      pfes.Update(false);

      TestSolvePar(pfes);
   }

   SECTION("Hex mesh")
   {
      // 2^3 element hex mesh
      Mesh mesh = MakeCartesianMesh(2, 3);
      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      // Standard H1 space with order 1 elements
      H1_FECollection fe_coll(1, pmesh.Dimension());
      ParFiniteElementSpace pfes(&pmesh, &fe_coll);

      REQUIRE(pfes.GlobalTrueVSize() == 27);  // 3^3

      // Convert to variable order space by p-refinement
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         pfes.SetElementOrder(i, pfes.GetElementOrder(i) + 1);
      }
      pfes.Update(false);

      // DOFs for vertices + edges + faces + elements = 27 + 54 + 36 + 8 = 125
      REQUIRE(pfes.GlobalTrueVSize() == 125);  // 5^3

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank == 0) { pfes.SetElementOrder(0, 4); }
      pfes.Update(false);

      Array<Refinement> refs;
      if (rank == 0) { refs.Append(Refinement(0)); }
      pmesh.GeneralRefinement(refs);
      pfes.Update(false);

      TestSolvePar(pfes);
   }

   SECTION("Hex mesh with intermediate orders")
   {
      // Test ParFiniteElementSpace::MarkIntermediateEntityDofs
      // This test is designed for 2 MPI ranks. If more than 2 ranks are used,
      // the test is run on only the first 2 ranks via a split communicator.
      int numprocs, rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

      MPI_Comm comm2;
      MPI_Comm_split(MPI_COMM_WORLD, rank < 2 ? 0 : 1, rank, &comm2);

      if (rank < 2)
      {
         // 2x1x1 element hex mesh
         Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON);
         mesh.EnsureNCMesh();

         Array<int> partition(2);
         partition = 0;
         if (numprocs > 1) { partition[1] = 1; }

         ParMesh pmesh(comm2, mesh, partition.GetData());
         mesh.Clear();

         // Standard H1 space with order 1 elements
         H1_FECollection fe_coll(1, pmesh.Dimension());
         ParFiniteElementSpace fespace(&pmesh, &fe_coll);

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(0)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(4)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         if (rank == 1)
         {
            for (int elem=0; elem<pmesh.GetNE(); ++elem)
            {
               const int p_elem = fespace.GetElementOrder(elem);
               fespace.SetElementOrder(elem, p_elem + 1);
            }
         }
         fespace.Update(false);

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(6)); }
            if (rank == 0) { refs.Append(Refinement(0)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         {
            Array<Refinement> refs;
            if (rank == 1) { refs.Append(Refinement(10)); }
            pmesh.GeneralRefinement(refs);
            fespace.Update(false);
         }

         if (rank == 1) { fespace.SetElementOrder(3, 3); }
         fespace.Update(false);

         // Set at least order 2 everywhere
         for (int elem=0; elem<pmesh.GetNE(); ++elem)
         {
            const int p_elem = fespace.GetElementOrder(elem);
            if (p_elem < 2) { fespace.SetElementOrder(elem, 2); }
         }
         fespace.Update(false);

         TestSolvePar(fespace);
      }
   }

   SECTION("Quad/hex mesh ND/RT")
   {
      using namespace var_order_test;

      const auto space_type = GENERATE(SpaceType::RT, SpaceType::ND);
      const int dim = GENERATE(2, 3);

      Mesh mesh = MakeCartesianMesh(dim == 2 ? 4 : 2, dim);

      mesh.EnsureNCMesh();

      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      int ndof0, ncdof2;

      std::unique_ptr<FiniteElementCollection> fec;
      if (space_type == SpaceType::RT)
      {
         // Standard RT space with order 0 elements
         fec.reset(new RT_FECollection(0, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ncdof2 = 312;
         }
         else
         {
            ndof0 = 36;
            ncdof2 = 756;
         }
      }
      else
      {
         // Standard ND space with order 1 elements
         fec.reset(new ND_FECollection(1, dim));

         if (dim == 2)
         {
            ndof0 = 40;
            ncdof2 = 144;
         }
         else
         {
            ndof0 = 54;
            ncdof2 = 300;
         }
      }

      ParFiniteElementSpace fespace(&pmesh, fec.get());

      REQUIRE(fespace.GlobalTrueVSize() == ndof0);

      // Convert to variable order space by p-refinement
      // Increase order on all elements
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         fespace.SetElementOrder(i, fespace.GetElementOrder(i) + 1);
      }
      fespace.Update(false);

      REQUIRE(fespace.GlobalTrueVSize() == ncdof2);

      TestSolveParVec(fespace);
   }
}

TEST_CASE("Serial-parallel Comparison for Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh][Parallel]")
{

   int dimension = GENERATE(2, 3);
   Mesh mesh = MakeCartesianMesh(4, dimension);
   TestRandomPRefinement(mesh);
}
#endif  // MFEM_USE_MPI

// Exact solution: x^2 + y^2 + z^2
static real_t exact_sln(const Vector &p)
{
   real_t x = p(0), y = p(1);
   if (p.Size() == 3)
   {
      real_t z = p(2);
      return x*x + y*y + z*z;
   }
   else
   {
      return x*x + y*y;
   }
}

static real_t exact_rhs(const Vector &p)
{
   return (p.Size() == 3) ? -6.0 : -4.0;
}

static void TestSolve(FiniteElementSpace &fespace)
{
   Mesh *mesh = fespace.GetMesh();

   // exact solution and RHS for the problem -\Delta u = 1
   FunctionCoefficient exsol(exact_sln);
   FunctionCoefficient rhs(exact_rhs);

   // set up Dirichlet BC on the boundary
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 1;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // assemble the linear form
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   lf.Assemble();

   // assemble the bilinear form.
   BilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // solve
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 0, 500, 1e-30, 0.0);

   bf.RecoverFEMSolution(X, lf, x);

   // compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));

   // visualize
#ifdef MFEM_UNIT_DEBUG_VISUALIZE
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   std::unique_ptr<GridFunction> vis_x = x.ProlongToMaxOrder();
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << *vis_x;
#endif
}

// Quadratic exact solution for vector-valued spaces
void exact_sln_vec(const Vector &x, Vector &f)
{
   if (f.Size() == 3)
   {
      f(0) = x(1)*x(2);
      f(1) = x(0)*x(2);
      f(2) = x(0)*x(1);
   }
   else
   {
      f(0) = x(0)*x(1);
      f(1) = x(0)*x(1);
   }
}

static void TestSolveVec(FiniteElementSpace &fespace)
{
   Mesh *mesh = fespace.GetMesh();
   const int sdim = mesh->SpaceDimension();

   // Exact solution and RHS for the mass-matrix problem E = f
   VectorFunctionCoefficient exsol(sdim, exact_sln_vec);

   // No boundary conditions
   Array<int> ess_attr(mesh->bdr_attributes.Max());
   ess_attr = 0;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   GridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // Assemble the linear form
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(exsol));
   lf.Assemble();

   // Assemble the bilinear form
   BilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new VectorFEMassIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // Solve
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 0, 500, 1e-30, 0.0);

   bf.RecoverFEMSolution(X, lf, x);

   // Compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);

   REQUIRE(error == MFEM_Approx(0.0));
}

#ifdef MFEM_USE_MPI
static void TestSolvePar(ParFiniteElementSpace &pfes)
{
   ParMesh *pmesh = pfes.GetParMesh();

   // exact solution and RHS for the problem -\Delta u = 1
   FunctionCoefficient exsol(exact_sln);
   FunctionCoefficient rhs(exact_rhs);

   // set up Dirichlet BC on the boundary
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;

   Array<int> ess_tdof_list;
   pfes.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   ParGridFunction x(&pfes);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // assemble the linear form
   ParLinearForm lf(&pfes);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   lf.Assemble();

   // assemble the bilinear form.
   ParBilinearForm bf(&pfes);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // solve
   HypreBoomerAMG prec;
   CGSolver cg(pfes.GetComm());
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);

   // compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));
}

void TestSolveSerial1(const Mesh & mesh, GridFunction & x)
{
   FiniteElementSpace *fespace = x.FESpace();

   Array<int> ess_attr(mesh.bdr_attributes.Max());
   ess_attr = 1;  // Dirichlet BC everywhere

   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   // assemble the linear form
   LinearForm lf(fespace);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   // assemble the bilinear form.
   BilinearForm bf(fespace);
   bf.SetDiagonalPolicy(Operator::DIAG_ONE);

   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 10, 500, 1e-30, 0.0);
   std::cout << std::flush;

   bf.RecoverFEMSolution(X, lf, x);
}

void TestSolveParallel1(ParMesh &pmesh, ParGridFunction &x)
{
   ParFiniteElementSpace *pfes = x.ParFESpace();

   Array<int> ess_attr(pmesh.bdr_attributes.Max());
   ess_attr = 1;  // Dirichlet BC

   Array<int> ess_tdof_list;
   pfes->GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   // assemble the linear form
   ParLinearForm lf(pfes);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   // assemble the bilinear form.
   ParBilinearForm bf(pfes);
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   HypreBoomerAMG prec;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(10);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);
}

GridFunction *TestRandomPRefinement_serial(Mesh & mesh)
{
   // standard H1 space with order 1 elements
   auto *fec = new H1_FECollection(1, mesh.Dimension());
   auto *fespace = new FiniteElementSpace(&mesh, fec);

   for (int i=0; i<mesh.GetNE(); ++i)
   {
      const int p = mesh.GetAttribute(i);
      if (p > 1) { fespace->SetElementOrder(i, p); }
   }

   fespace->Update(false);

   auto *sol = new GridFunction(fespace);
   sol->MakeOwner(fec);
   *sol = 0.0;  // Essential DOF value
   TestSolveSerial1(mesh, *sol);
   return sol;
}

ParGridFunction *TestRandomPRefinement_parallel(Mesh & mesh)
{
   // standard H1 space with order 1 elements

   auto *pmsh = new ParMesh(MPI_COMM_WORLD, mesh);
   auto *pfec = new H1_FECollection(1, mesh.Dimension());
   auto *pfes = new ParFiniteElementSpace(pmsh, pfec);

   for (int i=0; i<pmsh->GetNE(); ++i)
   {
      const int p = pmsh->GetAttribute(i);
      if (p > 1) { pfes->SetElementOrder(i, p); }
   }

   pfes->Update(false);

   auto *sol = new ParGridFunction(pfes);
   sol->MakeOwner(pfec);
   *sol = 0.0;  // Essential DOF value
   TestSolveParallel1(*pmsh, *sol);
   return sol;
}

// This function is based on the assumption that each element has attribute
// equal to its index in the serial mesh. This assumption enables easily
// identifying serial and parallel elements, for element-wise comparisons.
real_t ErrorSerialParallel(const GridFunction & xser,
                           const ParGridFunction & xpar)
{
   const FiniteElementSpace *fespace = xser.FESpace();
   const ParFiniteElementSpace *pfespace = xpar.ParFESpace();

   Mesh *mesh = fespace->GetMesh();
   ParMesh *pmesh = pfespace->GetParMesh();

   const int npe = pmesh->GetNE();

   int numprocs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Array<int> allnpe(numprocs);
   MPI_Allgather(&npe, 1, MPI_INT, allnpe.GetData(), 1, MPI_INT, MPI_COMM_WORLD);

   int eos = 0;
   for (int i=0; i<rank; ++i)
   {
      eos += allnpe[i];
   }

   bool elemsMatch = true;
   real_t error = 0.0;

   xser.HostRead();
   xpar.HostRead();

   // Loop over only the local elements in the parallel mesh.
   for (int e=0; e<pmesh->GetNE(); ++e)
   {
      if (pmesh->GetAttribute(e) != mesh->GetAttribute(eos + e))
      {
         elemsMatch = false;
      }

      Array<int> sdofs, pdofs;

      fespace->GetElementDofs(eos + e, sdofs);
      pfespace->GetElementDofs(e, pdofs);

      if (sdofs.Size() != pdofs.Size())
      {
         elemsMatch = false;
      }

      for (int i=0; i<sdofs.Size(); ++i)
      {
         const real_t d = xser[sdofs[i]] - xpar[pdofs[i]];
         error += d * d;
      }
   }

   REQUIRE(elemsMatch);

   MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   return error;
}

real_t CheckH1Continuity(ParGridFunction & x)
{
   x.ExchangeFaceNbrData();

   const ParFiniteElementSpace *fes = x.ParFESpace();
   ParMesh *mesh = fes->GetParMesh();

   const int dim = mesh->Dimension();

   // Following the example of KellyErrorEstimator::ComputeEstimates(),
   // we loop over interior faces and then shared faces.

   // Compute error contribution from local interior faces
   real_t errorMax = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      if (mesh->FaceIsInterior(f))
      {
         int Inf1, Inf2, NCFace;
         mesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

         auto FT = mesh->GetFaceElementTransformations(f);

         const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) :
                               fes->GetEdgeOrder(f);
         auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
         const auto nip = int_rule.GetNPoints();

         // Convention
         // * Conforming face: Face side with smaller element id handles
         // the integration
         // * Non-conforming face: The slave handles the integration.
         // See FaceInfo documentation for details.
         bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
         bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
         if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
         {
            for (int i = 0; i < nip; i++)
            {
               const auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;

               FT->Loc1.Transform(fip, ip);
               const real_t v1 = x.GetValue(FT->Elem1No, ip);

               FT->Loc2.Transform(fip, ip);
               const real_t v2 = x.GetValue(FT->Elem2No, ip);

               const real_t err_i = std::abs(v1 - v2);
               errorMax = std::max(errorMax, err_i);
            }
         }
      }
   }

   // Compute error contribution from shared interior faces
   for (int sf = 0; sf < mesh->GetNSharedFaces(); sf++)
   {
      const int f = mesh->GetSharedFace(sf);
      const bool trueInterior = mesh->FaceIsTrueInterior(f);
      if (!trueInterior) { continue; }

      auto FT = mesh->GetSharedFaceTransformations(sf, true);
      const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) : fes->GetEdgeOrder(f);
      const auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
      const auto nip = int_rule.GetNPoints();

      for (int i = 0; i < nip; i++)
      {
         const auto &fip = int_rule.IntPoint(i);
         IntegrationPoint ip;

         FT->Loc1.Transform(fip, ip);
         const real_t v1 = x.GetValue(FT->Elem1No, ip);

         FT->Loc2.Transform(fip, ip);
         const real_t v2 = x.GetValue(FT->Elem2No, ip);

         const real_t err_i = std::abs(v1 - v2);
         errorMax = std::max(errorMax, err_i);
      }
   }

   return errorMax;
}

static void TestRandomPRefinement(Mesh & mesh)
{
   for (int i=0; i<mesh.GetNE(); ++i)
   {
      mesh.SetAttribute(i, 1 + (i % 3));   // Order is 1, 2, or 3
   }
   mesh.EnsureNCMesh();

   GridFunction *solSerial = TestRandomPRefinement_serial(mesh);
   ParGridFunction *solParallel = TestRandomPRefinement_parallel(mesh);
   const real_t error = ErrorSerialParallel(*solSerial, *solParallel);
   REQUIRE(error == MFEM_Approx(0.0));

   // Check H1 continuity for the parallel solution.
   const real_t discontinuity = CheckH1Continuity(*solParallel);
   REQUIRE(discontinuity == MFEM_Approx(0.0));

   delete solParallel->ParFESpace()->GetParMesh();
   delete solSerial;
   delete solParallel;
}

static void TestSolveParVec(ParFiniteElementSpace &fespace)
{
   ParMesh *pmesh = fespace.GetParMesh();
   const int sdim = pmesh->SpaceDimension();

   // Exact solution and RHS for the mass-matrix problem E = f
   VectorFunctionCoefficient exsol(sdim, exact_sln_vec);

   // No boundary conditions
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 0;

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

   ParGridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exsol, ess_attr);

   // Assemble the linear form
   ParLinearForm lf(&fespace);
   lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(exsol));
   lf.Assemble();

   // Assemble the bilinear form
   ParBilinearForm bf(&fespace);
   bf.AddDomainIntegrator(new VectorFEMassIntegrator());
   bf.Assemble();

   OperatorPtr A;
   Vector B, X;
   bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B);

   // Solve
   HypreBoomerAMG prec;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-30);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   bf.RecoverFEMSolution(X, lf, x);

   // Compute L2 error from the exact solution
   const real_t error = x.ComputeL2Error(exsol);
   REQUIRE(error == MFEM_Approx(0.0));
}

#endif  // MFEM_USE_MPI

}
