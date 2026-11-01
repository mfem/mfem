// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#ifdef MFEM_USE_MPI

TEST_CASE("HypreParMatrixWrapConstructors-SyncChecks", "[Parallel], [GPU]")
{
   const int dim = 2;
   const int n1d = 6;
   const int p = 2;
   Mesh smesh = Mesh::MakeCartesian2D(n1d, n1d, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   SECTION("SquareBlockDiagWrapConstructor")
   {
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fespace(&mesh, &fec);
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new MassIntegrator);
      a.Assemble();
      a.Finalize();
      SparseMatrix &spmat = a.SpMat();
      const int height = spmat.Height();
      const int nnz = spmat.NumNonZeroElems();

      // Create a square block diagonal HypreParMatrix with blocks corresponding
      // to the local sparse matrices, spmat. The constructed HypreParMatrix
      // reuses the I, J and data arrays of spmat (with some exceptions).
      // The constructor will also permute the entries of its J and data arrays
      // to ensure that the diagonal entry is first in every row.
      HypreParMatrix hpmat(mesh.GetComm(),
                           fespace.GlobalVSize(),
                           fespace.GetDofOffsets(),
                           &spmat);

      // Verify that spmat's arrays are not out of sync:
      REQUIRE(spmat.GetMemoryI().CompareHostAndDevice(height+1) == 0);
      REQUIRE(spmat.GetMemoryJ().CompareHostAndDevice(nnz) == 0);
      REQUIRE(spmat.GetMemoryData().CompareHostAndDevice(nnz) == 0);
   }

   SECTION("RectangularBlockDiagWrapConstructor")
   {
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fespace(&mesh, &fec);
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new MassIntegrator);
      a.Assemble();
      a.Finalize();
      SparseMatrix &spmat = a.SpMat();
      const int height = spmat.Height();
      const int nnz = spmat.NumNonZeroElems();

      // Create a rectangular block diagonal HypreParMatrix with blocks
      // corresponding to the local sparse matrices, spmat. The constructed
      // HypreParMatrix reuses the I, J and data arrays of spmat (with some
      // exceptions).
      // When the row and column offsets are the same pointer, the constructor
      // will also permute the entries of its J and data arrays to ensure that
      // the diagonal entry is first in every row.
      HypreParMatrix hpmat(mesh.GetComm(),
                           fespace.GlobalVSize(), // num rows
                           fespace.GlobalVSize(), // num cols
                           fespace.GetDofOffsets(), // row offsets
                           fespace.GetDofOffsets(), // col offsets
                           &spmat);

      // Verify that spmat's arrays are not out of sync:
      REQUIRE(spmat.GetMemoryI().CompareHostAndDevice(height+1) == 0);
      REQUIRE(spmat.GetMemoryJ().CompareHostAndDevice(nnz) == 0);
      REQUIRE(spmat.GetMemoryData().CompareHostAndDevice(nnz) == 0);
   }

   SECTION("RectangularDiagOffdWrapConstructor")
   {
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fespace(&mesh, &fec);
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new MassIntegrator);
      a.Assemble();
      a.Finalize();
      SparseMatrix &diag = a.SpMat();
      const int height = diag.Height();
      const int nnz = diag.NumNonZeroElems();

      SparseMatrix offd(height, 0, 0); // height x 0 matrix
      HYPRE_BigInt cmap = 0;

      // Create a rectangular HypreParMatrix with diagonal blocks corresponding
      // to the local sparse matrices, diag, and zero off-diagonal block, offd.
      // The constructed HypreParMatrix reuses the I, J and data arrays of diag
      // and offd (with some exceptions).
      // When the row and column offsets are the same pointer, the constructor
      // will also permute the entries of its block diagonal's J and data arrays
      // to ensure that the diagonal entry is first in every row.
      HypreParMatrix hpmat(mesh.GetComm(),
                           fespace.GlobalVSize(), // num rows
                           fespace.GlobalVSize(), // num cols
                           fespace.GetDofOffsets(), // row offsets
                           fespace.GetDofOffsets(), // col offsets
                           &diag,
                           &offd,
                           &cmap,
                           false);

      // Verify that diag's arrays are not out of sync:
      REQUIRE(diag.GetMemoryI().CompareHostAndDevice(height+1) == 0);
      REQUIRE(diag.GetMemoryJ().CompareHostAndDevice(nnz) == 0);
      REQUIRE(diag.GetMemoryData().CompareHostAndDevice(nnz) == 0);
   }

   SECTION("BooleanRectangularBlockDiagWrapConstructor")
   {
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fespace(&mesh, &fec);
      const Table &el_dof = fespace.GetElementToDofTable();
      Table el_dof_t;
      Transpose(el_dof, el_dof_t, fespace.GetNDofs());
      Table dof_dof;
      Mult(el_dof_t, el_dof, dof_dof);
      const int height = dof_dof.Size();
      const int nnz = dof_dof.Size_of_connections();

      // Create a Boolean rectangular block diagonal HypreParMatrix with blocks
      // corresponding to the local Table dof_dof. The constructed
      // HypreParMatrix reuses the I and J arrays of dof_dof (with some
      // exceptions).
      // When the row and column offsets are the same pointer, the constructor
      // will also permute the entries of its J and data arrays to ensure that
      // the diagonal entry is first in every row.
      HypreParMatrix hpm(mesh.GetComm(),
                         fespace.GlobalVSize(), // num rows
                         fespace.GlobalVSize(), // num cols
                         fespace.GetDofOffsets(), // row offsets
                         fespace.GetDofOffsets(), // col offsets
                         &dof_dof);

      // Verify that dof_dof's arrays are not out of sync:
      REQUIRE(dof_dof.GetIMemory().CompareHostAndDevice(height+1) == 0);
      REQUIRE(dof_dof.GetJMemory().CompareHostAndDevice(nnz) == 0);
   }
}

TEST_CASE("HypreParMatrixAbsMult",  "[Parallel], [HypreParMatrixAbsMult]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh mesh = Mesh::MakeCartesian2D(
                     ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();
      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
      ParFiniteElementSpace R_space(pmesh, hdiv_coll);
      ParFiniteElementSpace W_space(pmesh, l2_coll);

      int n = R_space.GetTrueVSize();
      int m = W_space.GetTrueVSize();
      ParMixedBilinearForm a(&R_space, &W_space);
      a.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      a.Assemble();
      a.Finalize();

      HypreParMatrix *A = a.ParallelAssemble();
      HypreParMatrix *Aabs = new HypreParMatrix(*A);

      hypre_ParCSRMatrix * AparCSR = *Aabs;
      Aabs->HypreReadWrite();

      int nnzd = AparCSR->diag->num_nonzeros;
      real_t *d_diag_data = AparCSR->diag->data;
      mfem::hypre_forall(nnzd, [=] MFEM_HOST_DEVICE (int i)
      {
         d_diag_data[i] = fabs(d_diag_data[i]);
      });

      int nnzoffd = AparCSR->offd->num_nonzeros;
      real_t *d_offd_data = AparCSR->offd->data;
      mfem::hypre_forall(nnzoffd, [=] MFEM_HOST_DEVICE (int i)
      {
         d_offd_data[i] = fabs(d_offd_data[i]);
      });

      Vector X0(n), X1(n);
      Vector Y0(m), Y1(m);

      X0.Randomize();
      Y0.Randomize(1);
      Y1.Randomize(1);
      A->AbsMult(3.4,X0,-2.3,Y0);
      Aabs->Mult(3.4,X0,-2.3,Y1);

      Y1 -= Y0;
      double error = Y1.Norml2();

      mfem::out << "Testing AbsMult:   order: " << order
                << ", error norm on rank "
                << rank << ": " << error << std::endl;

      REQUIRE(error == MFEM_Approx(0.0));

      MPI_Barrier(MPI_COMM_WORLD);

      Y0.Randomize();
      X0.Randomize(1);
      X1.Randomize(1);
      A->AbsMultTranspose(3.4,Y0,-2.3,X0);
      Aabs->MultTranspose(3.4,Y0,-2.3,X1);
      X1 -= X0;

      error = X1.Norml1();
      mfem::out << "Testing AbsMultT:  order: " << order
                << ", error norm on rank "
                << rank << ": " << error << std::endl;

      REQUIRE(error == MFEM_Approx(0.0));

      delete A;
      delete Aabs;
      delete hdiv_coll;
      delete l2_coll;
      delete pmesh;
   }
}

TEST_CASE("HypreParMatrixCopyAndMove",
          "[Parallel], [HypreParMatrixCopyAndMove]")
{
   Mesh smesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   H1_FECollection fec(1, 2);
   ParFiniteElementSpace fespace(&mesh, &fec);
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();
   a.Finalize();

   std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());

   Vector x(A->Width()), y(A->Height()), y2(A->Height());
   x = 1.0;
   A->Mult(x, y);
   const real_t ynorm = y.Norml2();

   SECTION("CopyAssignmentIsDeep")
   {
      HypreParMatrix B;
      B = *A;
      B.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
      // B has to own its own data, so it still works once A is gone
      A.reset();
      B.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
   }

   SECTION("CopyAssignmentReplacesTheOldData")
   {
      // both sides own data, so the assignment frees the old target
      HypreParMatrix B(*A), C(*A);
      B = C;
      B.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
   }

   SECTION("CopyOfAnEmptyMatrixIsEmpty")
   {
      HypreParMatrix E;
      HypreParMatrix F(E);
      REQUIRE(F.Height() == 0);
      REQUIRE(F.Width() == 0);
   }

   SECTION("AssigningAnEmptyMatrixClearsTheTarget")
   {
      HypreParMatrix B(*A), E;
      B = E;
      REQUIRE(B.Height() == 0);
      // the cleared matrix is still usable
      B = *A;
      B.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
   }

   SECTION("SelfAssignment")
   {
      HypreParMatrix *self = A.get();
      *self = *self;
      self->Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
   }

   SECTION("MoveTakesOverTheHandles")
   {
      HypreParMatrix B;
      B = std::move(*A);
      B.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
      HypreParMatrix C(std::move(B));
      C.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
      HypreParMatrix D(C); // non-empty target for the move assignment
      D = std::move(C);
      D.Mult(x, y2);
      REQUIRE(y2.Norml2() == MFEM_Approx(ynorm));
      // the sources are left empty but valid
      REQUIRE(B.Height() == 0);
      REQUIRE(C.Height() == 0);
   }

   SECTION("AssigningAnEmptyMatrixToAnEmptyOne")
   {
      HypreParMatrix E, F;
      E = F;
      E = std::move(F);
      REQUIRE(E.Height() == 0);
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
