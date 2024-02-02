// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "general/forall.hpp"
#include "unit_tests.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

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

      int nnzd = AparCSR->diag->num_nonzeros;
#if !defined(HYPRE_USING_GPU)
      for (int j = 0; j < nnzd; j++)
      {
         AparCSR->diag->data[j] = fabs(AparCSR->diag->data[j]);
      }
#else
      Aabs->HypreReadWrite();
      double *d_diag_data = AparCSR->diag->data;
      MFEM_GPU_FORALL(i, nnzd,
      {
         d_diag_data[i] = fabs(d_diag_data[i]);
      });
#endif

      int nnzoffd = AparCSR->offd->num_nonzeros;
#if !defined(HYPRE_USING_GPU)
      for (int j = 0; j < nnzoffd; j++)
      {
         AparCSR->offd->data[j] = fabs(AparCSR->offd->data[j]);
      }
#else
      double *d_offd_data = AparCSR->offd->data;
      MFEM_GPU_FORALL(i, nnzoffd,
      {
         d_offd_data[i] = fabs(d_offd_data[i]);
      });
#endif

      Vector X0(n), X1(n);
      Vector Y0(m), Y1(m);

      X0.Randomize();
      Y0.Randomize(1);
      Y1.Randomize(1);
      A->AbsMult(3.4,X0,-2.3,Y0);
      Aabs->Mult(3.4,X0,-2.3,Y1);

      Y1 -=Y0;
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
      X1 -=X0;

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

#endif // MFEM_USE_MPI

} // namespace mfem
