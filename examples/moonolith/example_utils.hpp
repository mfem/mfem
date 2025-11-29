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

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif // MFEM_USE_MPI

#include "mfem.hpp"

inline void check_options(mfem::OptionsParser &args)
{
   using namespace std;
   using namespace mfem;

   int rank = 0;
#ifdef MFEM_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // MFEM_USE_MPI

   if (!args.Good())
   {
      if (rank == 0)
      {
         args.PrintUsage(cout);
      }

#ifdef MFEM_USE_MPI
      MPI_Finalize();
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      abort();
#endif // MFEM_USE_MPI
   }

   if (rank == 0)
   {
      args.PrintOptions(cout);
   }
}

inline void make_fun(mfem::FiniteElementSpace &fe, mfem::Coefficient &c,
                     mfem::GridFunction &f)
{
   using namespace std;
   using namespace mfem;

   f.SetSpace(&fe);
   f.ProjectCoefficient(c);
   f.Update();
}

inline double example_fun(const mfem::Vector &x)
{
   using namespace std;
   using namespace mfem;

   const int n = x.Size();
   double ret = 0;
   for (int k = 0; k < n; ++k)
   {
      ret += x(k) * x(k);
   }

   return sqrt(ret);
}

void vector_fun(const mfem::Vector &x, mfem::Vector &f)
{
   const double n = x.Norml2();
   f.SetSize(x.Size());
   f = n;
}

inline void plot(mfem::Mesh &mesh, mfem::GridFunction &x, std::string title)
{
   using namespace std;
   using namespace mfem;

   int num_procs = 1, rank = 0;

#ifdef MFEM_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif // MFEM_USE_MPI

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << rank << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << x
            << "window_title '"<< title << "'\n" << flush;
   sol_sock << flush;
}
