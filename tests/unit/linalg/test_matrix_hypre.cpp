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

#ifdef MFEM_USE_MPI

TEST_CASE("HypreParMatrixAbsMult",  "[Parallel], [HypreParMatrixAbsMult]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh * mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      ConstantCoefficient one(1.0);
      FiniteElementCollection *fec = new H1_FECollection(order, dim);
      ParFiniteElementSpace fes(pmesh, fec);
      int n = fes.GetTrueVSize();
      ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();
      a.Finalize();

      HypreParMatrix *A = a.ParallelAssemble();

      HypreParMatrix *Aabs = new HypreParMatrix(*A);

      hypre_ParCSRMatrix * AparCSR =
         (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Aabs);

      int nnzd = AparCSR->diag->num_nonzeros;
      for (int j = 0; j < nnzd; j++)
      {
         AparCSR->diag->data[j] = fabs(AparCSR->diag->data[j]);
      }

      int nnzoffd = AparCSR->offd->num_nonzeros;
      for (int j = 0; j < nnzoffd; j++)
      {
         AparCSR->offd->data[j] = fabs(AparCSR->offd->data[j]);
      }

      Vector X(n); X.Randomize(1);
      Vector Y(n); Y.Randomize(1);
      Vector B(n); B.Randomize();
      A->AbsMult(3.4,B,-2.3,X);
      Aabs->Mult(3.4,B,-2.3,Y);

      Y -=X;
      double error = Y.Norml2();

      std::cout << "  order: " << order
                << ", error norm on rank "
                << rank << ": " << error << std::endl;

      REQUIRE(error == Approx(EPS));

      delete A;
      delete Aabs;
      delete fec;
      delete pmesh;
   }
}

TEST_CASE("HypreParMatrixAbsMultT",  "[Parallel], [HypreParMatrixAbsMultT]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int dim = 2;
   int ne = 4;
   for (int order = 1; order <= 3; ++order)
   {
      Mesh * mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;
      ConstantCoefficient one(1.0);
      FiniteElementCollection *fec = new H1_FECollection(order, dim);
      ParFiniteElementSpace fes(pmesh, fec);
      int n = fes.GetTrueVSize();
      ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      a.AddDomainIntegrator(new MassIntegrator(one));
      a.Assemble();
      a.Finalize();

      HypreParMatrix *A = a.ParallelAssemble();

      HypreParMatrix *Aabs = new HypreParMatrix(*A);

      hypre_ParCSRMatrix * AparCSR =
         (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Aabs);

      int nnzd = AparCSR->diag->num_nonzeros;
      for (int j = 0; j < nnzd; j++)
      {
         AparCSR->diag->data[j] = fabs(AparCSR->diag->data[j]);
      }

      int nnzoffd = AparCSR->offd->num_nonzeros;
      for (int j = 0; j < nnzoffd; j++)
      {
         AparCSR->offd->data[j] = fabs(AparCSR->offd->data[j]);
      }

      Vector X(n); X.Randomize(1);
      Vector Y(n); Y.Randomize(1);
      Vector B(n); B.Randomize();
      A->AbsMultTranspose(3.4,B,-2.3,X);
      Aabs->MultTranspose(3.4,B,-2.3,Y);

      Y -=X;
      double error = Y.Norml2();

      std::cout << "  order: " << order
                << ", error norm on rank "
                << rank << ": " << error << std::endl;

      REQUIRE(error == Approx(EPS));

      delete A;
      delete Aabs;
      delete fec;
      delete pmesh;
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
