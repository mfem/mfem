#ifndef TESTSTRUMPACK_HPP
#define TESTSTRUMPACK_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;


void TestStrumpackConstructor()
{
   int num_procs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const int num_loc_rows = 100;
   const int first_loc_row = num_loc_rows * rank;
   const int glob_nrows = num_loc_rows * num_procs;
   const int glob_ncols = glob_nrows;

   int *opI = new int[num_loc_rows+1];

   for (int i=0; i<num_loc_rows+1; ++i)
   {
      opI[i] = 0;
   }

   for (int i=0; i<num_loc_rows; ++i)
   {
      int nnz_i = 3;

      if ((first_loc_row + i) == 0 ||
          (first_loc_row + i) == glob_nrows-1)  // if first or last row
      {
         nnz_i = 2;
      }

      opI[i+1] = opI[i] + nnz_i;
   }

   const int nnz = opI[num_loc_rows];

   int *opJ = new int[nnz];
   double *data = new double[nnz];

   int cnt = 0;
   for (int i=0; i<num_loc_rows; ++i)
   {
      const int globalRow = first_loc_row + i;

      // Diagonal entry

      opJ[cnt] = first_loc_row + i;

      if (globalRow == 0 || globalRow == glob_nrows-1)
      {
         data[cnt] = 2.0;
      }
      else
      {
         data[cnt] = 3.0;
      }

      cnt++;

      // Upper diagonal
      if (globalRow < glob_nrows-1)
      {
         opJ[cnt] = first_loc_row + i + 1;
         data[cnt] = -1.0;
         cnt++;
      }

      // Lower diagonal
      if (globalRow > 0)
      {
         opJ[cnt] = first_loc_row + i - 1;
         data[cnt] = -1.0;
         cnt++;
      }
   }

   Operator *op = new STRUMPACKRowLocMatrix(MPI_COMM_WORLD, num_loc_rows,
                                            first_loc_row, glob_nrows, glob_ncols, opI, opJ, data);

   STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, MPI_COMM_WORLD);
   strumpack->SetPrintFactorStatistics(true);
   strumpack->SetPrintSolveStatistics(false);
   strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
   strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
   strumpack->SetOperator(*op);
   strumpack->SetFromCommandLine();

   Vector x(num_loc_rows);
   Vector y(num_loc_rows);

   x = 1.0;
   strumpack->Mult(x, y);

   delete opI;
   delete opJ;
   delete data;
}

#endif // TESTSTRUMPACK_HPP
