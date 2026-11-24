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

using namespace mfem;

#ifdef MFEM_USE_MPI

namespace
{

/// Stop the run if the ranks of @a comm do not all arrive here within @a
/// seconds: a rank kept in a collective call that the others did not make would
/// otherwise hang the test. @a comm must not be used by the code before the call.
void AbortUnlessAllArrive(MPI_Comm comm, double seconds)
{
   MPI_Request request;
   MPI_Ibarrier(comm, &request);
   const double start = MPI_Wtime();
   int arrived = 0;
   while (!arrived)
   {
      MPI_Test(&request, &arrived, MPI_STATUS_IGNORE);
      if (!arrived && MPI_Wtime() - start > seconds)
      {
         mfem::err << "Not all the ranks arrived within " << seconds
                   << " s: aborting." << std::endl;
         MPI_Abort(comm, 1);
      }
   }
}

} // namespace

TEST_CASE("ParGridFunction Update with a cached true vector",
          "[ParGridFunction], [Parallel]")
{
   // A true vector cached on all the ranks, rank 0 holding all the elements.
   // After a conforming refinement, the update of the grid function refreshes
   // the cache on rank 0 only, which must not build the new prolongation, a
   // collective operation, alone.
   Mesh serial = Mesh::MakeCartesian2D(3, 3, Element::TRIANGLE);
   Array<int> p0(serial.GetNE());
   p0 = 0;
   ParMesh pmesh(MPI_COMM_WORLD, serial, p0.GetData());
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&pmesh, &fec);
   ParGridFunction x(&fes);
   FunctionCoefficient f([](const Vector &p)
   { return 1.0 + p(0) + 2.0*p(1)*p(1); });
   x.ProjectCoefficient(f);
   x.SetTrueVector();
   MPI_Comm watch;
   MPI_Comm_dup(MPI_COMM_WORLD, &watch);
   for (int local = 0; local < 2; local++)
   {
      if (local)
      {
         Array<int> marked;
         for (int e = 0; e < pmesh.GetNE(); e += 2) { marked.Append(e); }
         pmesh.GeneralRefinement(marked, 0);
      }
      else { pmesh.UniformRefinement(); }
      fes.Update();
      x.Update();
      AbortUnlessAllArrive(watch, 60.0);
      ParGridFunction y(&fes);
      y.ProjectCoefficient(f);
      y -= x;
      real_t diff = y.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      CHECK(diff < 1e-12);
      if (fes.GetTrueVSize() > 0)
      {
         // The cache kept on rank 0 is that of the new space.
         Vector t;
         x.GetTrueDofs(t);
         const ParGridFunction &cx = x;
         t -= cx.GetTrueVector();
         CHECK(t.Normlinf() == 0.0);
      }
   }
   MPI_Comm_free(&watch);
}

#endif // MFEM_USE_MPI
