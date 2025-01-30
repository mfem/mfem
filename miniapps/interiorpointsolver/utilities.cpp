#include "mfem.hpp"
#include "utilities.hpp"

using namespace mfem;

HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * colOffsetsloc, HYPRE_BigInt * rowOffsetsloc, SparseMatrix * Asparse)
{
  int ncols_loc = colOffsetsloc[1] - colOffsetsloc[0];
  int nrows_loc = rowOffsetsloc[1] - rowOffsetsloc[0];
  HYPRE_BigInt ncols_glb, nrows_glb;
  MPI_Allreduce(&nrows_loc, &nrows_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ncols_loc, &ncols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int * AI          = Asparse->GetI();
  HYPRE_BigInt * AJ = Asparse->GetJ();
  double * Adata    = Asparse->GetData();

  HypreParMatrix * Ahypre = nullptr;
  Ahypre = new HypreParMatrix(MPI_COMM_WORLD, nrows_loc, nrows_glb, ncols_glb, AI, AJ, Adata, rowOffsetsloc, colOffsetsloc);
  return Ahypre;
}


HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		Vector & diag)
{
   int n_loc = offsetsloc[1] - offsetsloc[0];
   int n_glb = 0;
   MPI_Allreduce(&n_loc, &n_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   SparseMatrix * Dsparse = new SparseMatrix(n_loc, n_glb);
   Array<int> cols;
   Vector entries;
   cols.SetSize(1);
   entries.SetSize(1);
   for(int j = 0; j < n_loc; j++)
   {
     cols[0] = offsetsloc[0] + j;
     entries(0) = diag(j);
     Dsparse->SetRow(j, cols, entries);
   }   
   Dsparse->Finalize();
   HypreParMatrix * Dhypre = nullptr;
   Dhypre = mfem::GenerateHypreParMatrixFromSparseMatrix(offsetsloc, offsetsloc, Dsparse);
   delete Dsparse;
   return Dhypre;   
}

HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, HYPRE_Int * mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  SparseMatrix * Psparse = new SparseMatrix(n_rows_loc, n_cols_glb);
  Array<int> cols;
  Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask[j] == 1)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  HypreParMatrix * Phypre = nullptr;
  Phypre = mfem::GenerateHypreParMatrixFromSparseMatrix(offsets, reduced_offsets, Psparse);
  delete Psparse;
  return Phypre;
}

HypreParMatrix * GenerateProjector(HYPRE_BigInt * offsets, HYPRE_BigInt * reduced_offsets, const HypreParVector & mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  SparseMatrix * Psparse = new SparseMatrix(n_rows_loc, n_cols_glb);
  Array<int> cols;
  Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask(j) > 0.5)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  HypreParMatrix * Phypre = nullptr;
  Phypre = mfem::GenerateHypreParMatrixFromSparseMatrix(offsets, reduced_offsets, Psparse);
  delete Psparse;
  return Phypre;
}




HYPRE_BigInt * offsetsFromLocalSizes(int n)
{
  HYPRE_BigInt * offsets = new HYPRE_BigInt[2];
  
  int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();
  
  if (myrank == 0)
  {
    offsets[0] = 0;
    offsets[1] = n;
  }
  else
  {
    offsets[0] = 0;
    offsets[1] = 0;
  }
  
  // receive then send
  
  // Receive local size info from processes with rank less than myrank 
  // Populate that as entries of helper
  HYPRE_BigInt * helper;
  if (myrank > 0)
  {
    helper = new HYPRE_BigInt[myrank];
  }
  int tag;
  for (int i = 0; i < myrank; i++)
  {
    tag = myrank + i * nprocs;
    MPI_Recv (&(helper[i]), 1, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    offsets[0] += helper[i];
  }

  if (myrank > 0)
  {
    delete[] helper;
  }
  offsets[1] = offsets[0] + n;
  
  // Send local size info to all processes with rank greater than myrank
  for (int i = myrank + 1; i < nprocs; i++)
  {
    tag = i + myrank * nprocs;
    MPI_Send (&n, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
  }
  return offsets;
}


void HypreToMfemOffsets(HYPRE_BigInt * offsets)
{
  if (offsets[1] < offsets[0])
  {
    offsets[1] = offsets[0];
  }
  else
  {
    offsets[1] = offsets[1] + 1;
  }
}

