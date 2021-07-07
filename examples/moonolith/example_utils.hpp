#include <algorithm>
#include <assert.h>
#include <memory>
#include <mpi.h>

#include "mfem.hpp"

inline void check_options(mfem::OptionsParser &args)
{
   using namespace std;
   using namespace mfem;

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (!args.Good())
   {
      if (rank == 0)
      {
         args.PrintUsage(cout);
      }

      MPI_Finalize();
      MPI_Abort(MPI_COMM_WORLD, 1);
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

inline void plot(mfem::Mesh &mesh, mfem::GridFunction &x)
{
   using namespace std;
   using namespace mfem;

   int num_procs, rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << rank << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << x << flush;
   sol_sock << flush;
}
