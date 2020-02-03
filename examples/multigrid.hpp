#pragma once

#include "mfem.hpp"

using namespace std;

namespace mfem {

  STRUMPACKSolver* CreateStrumpackSolver(Operator *Arow, MPI_Comm comm)
  {
    //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, comm);
    STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, comm);
    strumpack->SetPrintFactorStatistics(true);
    strumpack->SetPrintSolveStatistics(false);
    strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
    strumpack->SetOperator(*Arow);
    strumpack->SetFromCommandLine();
    return strumpack;
  }

  hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat)
{
  // First cast the parameter to a hypre_ParCSRMatrix
  hypre_ParCSRMatrix * parcsr_op =
    (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat);

  MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

  // Create the CSRMatrixMPI A_ by borrowing the internal data from a hypre_CSRMatrix.
  return hypre_MergeDiagAndOffd(parcsr_op);
}

// Row and column offsets are assumed to be the same, for each process.
// Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
					       Array2D<SparseMatrix*> const& blocksSp,
					       Array2D<double> const& coefficient,
					       std::vector<std::vector<int> > const& blockProcOffsets,
					       std::vector<std::vector<int> > const& all_block_num_loc_rows)
{
  const int numBlocks = offsets.Size() - 1;
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> all_num_loc_rows(nprocs);
  std::vector<int> procOffsets(nprocs);
  std::vector<std::vector<int> > procBlockOffsets(nprocs);

  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows.data(), 1, MPI_INT, comm);

  int first_loc_row = 0;
  int glob_nrows = 0;
  procOffsets[0] = 0;
  for (int i=0; i<nprocs; ++i)
    {
      glob_nrows += all_num_loc_rows[i];
      if (i < rank)
	first_loc_row += all_num_loc_rows[i];

      if (i < nprocs-1)
	procOffsets[i+1] = procOffsets[i] + all_num_loc_rows[i];

      if (numBlocks > 0)
	{
	  procBlockOffsets[i].resize(numBlocks);
	  procBlockOffsets[i][0] = 0;
	}
      
      for (int j=1; j<numBlocks; ++j)
	procBlockOffsets[i][j] = procBlockOffsets[i][j-1] + all_block_num_loc_rows[j-1][i];
    }
    
  const int glob_ncols = glob_nrows;

  std::vector<int> opI(num_loc_rows+1);
  std::vector<int> cnt(num_loc_rows);

  for (int i=0; i<num_loc_rows; ++i)
    {
      opI[i] = 0;
      cnt[i] = 0;
    }

  opI[num_loc_rows] = 0;
  
  Array2D<hypre_CSRMatrix*> csr_blocks(numBlocks, numBlocks);
  
  // Loop over all blocks, to determine nnz for each row.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (blocks(i, j) == NULL)
	    {
	      csr_blocks(i, j) = NULL;

	      if (blocksSp(i, j) != NULL)
		{
		  const int nrows = blocksSp(i, j)->Height();
		  for (int k=0; k<nrows; ++k)
		    {
		      const int rowg = offsets[i] + k;
		      opI[rowg + 1] += blocksSp(i, j)->GetI()[k+1] - blocksSp(i, j)->GetI()[k];
		    }
		}
	    }
	  else
	    {
	      MFEM_VERIFY(blocksSp(i, j) == NULL, "");
	      
	      csr_blocks(i, j) = GetHypreParMatrixData(*(blocks(i, j)));

	      const int nrows = csr_blocks(i, j)->num_rows;

	      for (int k=0; k<nrows; ++k)
		{
		  const int rowg = offsets[i] + k;
		  //(*(leftInjection(i, j)))[k]
		  opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		}
	    }
	}
    }

  // Now opI[i] is nnz for row i-1. Do a partial sum to get offsets.
  for (int i=0; i<num_loc_rows; ++i)
    opI[i+1] += opI[i];

  const int nnz = opI[num_loc_rows];

  std::vector<HYPRE_Int> opJ(nnz);
  std::vector<double> data(nnz);

  // Loop over all blocks, to set matrix data.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (csr_blocks(i, j) != NULL || blocksSp(i, j) != NULL)
	    {
	      const bool useCSR = (csr_blocks(i, j) != NULL);
	      
	      const int nrows = useCSR ? csr_blocks(i, j)->num_rows : blocksSp(i, j)->Height();
	      const double coef = coefficient(i, j);

	      int *Iarray = useCSR ? csr_blocks(i, j)->i : blocksSp(i, j)->GetI();
	      
	      //const bool failure = (nrows != offsets[i+1] - offsets[i]);
	      
	      MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
	      for (int k=0; k<nrows; ++k)
		{
		  const int rowg = offsets[i] + k;  // process-local row
		  const int nnz_k = Iarray[k+1] - Iarray[k];
		  const int osk = Iarray[k];
		  
		  for (int l=0; l<nnz_k; ++l)
		    {
		      // Find the column process offset for the block.
		      const int bcol = useCSR ? csr_blocks(i, j)->j[osk + l] : blocksSp(i, j)->GetJ()[osk + l];
		      int bcolproc = 0;

		      for (int p=1; p<nprocs; ++p)
			{
			  if (blockProcOffsets[j][p] > bcol)
			    {
			      bcolproc = p-1;
			      break;
			    }
			}

		      if (blockProcOffsets[j][nprocs - 1] <= bcol)
			bcolproc = nprocs - 1;

		      const int colg = procOffsets[bcolproc] + procBlockOffsets[bcolproc][j] + (bcol - blockProcOffsets[j][bcolproc]);

		      if (colg < 0)
			cout << "BUG, negative global column index" << endl;
		      
		      opJ[opI[rowg] + cnt[rowg]] = colg;
		      data[opI[rowg] + cnt[rowg]] = useCSR ? coef * csr_blocks(i, j)->data[osk + l] : coef * blocksSp(i, j)->GetData()[osk + l];
		      cnt[rowg]++;
		    }
		}
	    }
	}
    }

  bool cntCheck = true;
  for (int i=0; i<num_loc_rows; ++i)
    {
      if (cnt[i] != opI[i+1] - opI[i])
	cntCheck = false;
    }

  MFEM_VERIFY(cntCheck, "");

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (csr_blocks(i, j) != NULL)
	    {
	      hypre_CSRMatrixDestroy(csr_blocks(i, j));
	    }
	}
    }
  
  std::vector<HYPRE_Int> rowStarts2(2);
  rowStarts2[0] = first_loc_row;
  rowStarts2[1] = first_loc_row + all_num_loc_rows[rank];

  if (nnz > 0)
    {
      HYPRE_Int minJ = opJ[0];
      HYPRE_Int maxJ = opJ[0];
      for (int i=0; i<nnz; ++i)
	{
	  minJ = std::min(minJ, opJ[i]);
	  maxJ = std::max(maxJ, opJ[i]);

	  if (opJ[i] >= glob_ncols)
	    cout << "Column indices out of range" << endl;
	}
    }
  
  HypreParMatrix *hmat = new HypreParMatrix(comm, num_loc_rows, glob_nrows, glob_ncols, (int*) opI.data(), (HYPRE_Int*) opJ.data(), (double*) data.data(),
					    (HYPRE_Int*) rowStarts2.data(), (HYPRE_Int*) rowStarts2.data());
  
  return hmat;
}

class BlockMGSolver : public Solver
{
private:
   /// The linear system matrix
   Array2D<HypreParMatrix *>& Af;
   Array2D<double>& Acoef;
   vector<Array<int>>Aoffsets;
   vector<Array<int>>Poffsets_i;
   vector<Array<int>>Poffsets_j;
   std::vector<Array2D<HypreParMatrix *>> A;
   std::vector<HypreParMatrix *>& P;
   std::vector<BlockOperator *> BlkP;
   std::vector<BlockOperator *> BlkA;
   std::vector<BlockOperator *> S;
   HypreParMatrix * Ac;
   int numGrids, numBlocks;
   STRUMPACKSolver *invAc = nullptr;
   double theta = 0.5;

public:
   BlockMGSolver(const int height, const int width, Array2D<HypreParMatrix *>& Af_,
		 Array2D<double>& Acoef_, std::vector<HypreParMatrix *>& P_);

   virtual void SetOperator(const Operator &op) {}

   virtual void SetTheta(const double a) { theta = a; }

   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~BlockMGSolver();
};

BlockMGSolver::BlockMGSolver(const int height, const int width,
			     Array2D<HypreParMatrix *>& Af_, Array2D<double>& Acoef_,
			     std::vector<HypreParMatrix *>& P_)
  : Solver(height, width), Af(Af_), Acoef(Acoef_), P(P_)
{
   numBlocks = Af.NumRows();
   MFEM_VERIFY(Af.NumCols() == numBlocks, "");
   numGrids = P.size();
   BlkP.resize(numGrids);
   BlkA.resize(numGrids+1);
   S.resize(numGrids);
   A.resize(numGrids + 1);
   A[numGrids] = Af;
   Aoffsets.resize(numGrids+1);
   Poffsets_i.resize(numGrids);
   Poffsets_j.resize(numGrids);
   // Construct Bilinear form Matrices on each level
   for (int k = numGrids ; k > 0; k--)
   {
      A[k - 1].SetSize(numBlocks,numBlocks);
      Aoffsets[k].SetSize(numBlocks+1); Aoffsets[k][0] = 0;
      for (int i=0; i<numBlocks; i++)
	Aoffsets[k][i+1] = A[k](i,i)->Height();

      Aoffsets[k].PartialSum();
      BlkA[k] = new BlockOperator(Aoffsets[k]);
      S[k-1] = new BlockOperator(Aoffsets[k]);  // Smoother

      for (int i=0; i<numBlocks; i++)
      {
         for (int j=0; j<numBlocks; j++)
         {
	    if (A[k](i,j) == NULL)
	      A[k - 1](i,j) = NULL;
	    else
	      {
		A[k - 1](i,j) = RAP(A[k](i,j), P[k - 1]);
		BlkA[k]->SetBlock(i, j, A[k](i,j), Acoef(i,j));
	      }
         }

	 HypreSmoother *S_i = new HypreSmoother;
	 S_i->SetType(HypreSmoother::Jacobi);
	 S_i->SetOperator(*(A[k](i,i)));
	 
	 S[k - 1]->SetBlock(i,i,S_i);
      }

      Poffsets_i[k-1].SetSize(numBlocks+1); Poffsets_i[k-1][0] = 0;
      Poffsets_j[k-1].SetSize(numBlocks+1); Poffsets_j[k-1][0] = 0;
      for (int i=0; i<numBlocks; i++)
	{
	  Poffsets_i[k-1][i+1] = P[k-1]->Height();
	  Poffsets_j[k-1][i+1] = P[k-1]->Width();
	}
      Poffsets_i[k-1].PartialSum();
      Poffsets_j[k-1].PartialSum();

      BlkP[k-1] = new BlockOperator(Poffsets_i[k-1],Poffsets_j[k-1]);
      for (int i=0; i<numBlocks; i++)
	BlkP[k-1]->SetBlock(i,i,P[k-1]);
   }
   // Set up coarse solve operator
   // Convert the coarse grid blockmatrix to a HypreParMatrix
   Array<int> offsets(numBlocks+1);
   offsets[0]=0;
   for (int i=0; i<numBlocks; i++)
     offsets[i+1]=A[0](i,i)->Height();

   offsets.PartialSum();

   BlkA[0] = new BlockOperator(offsets);

   Array2D<SparseMatrix*> Asp;
   //Array2D<double> Acoef;
   Asp.SetSize(numBlocks,numBlocks);
   //Acoef.SetSize(numBlocks,numBlocks);
   for (int i=0; i<numBlocks; i++)
   {
      for (int j=0; j<numBlocks; j++)
      {
	 if (A[0](i,j) != NULL)
	   BlkA[0]->SetBlock(i, j, A[0](i,j), Acoef(i,j));
	 
	 Asp(i,j) = NULL;
	 //Acoef(i,j) = 1.0;
      }
   }

   // Convert to PetscParMatrix
   HypreParMatrix * Ac;

   std::vector<std::vector<int> > blockProcOffsets(numBlocks);
   std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);

   {
     int nprocs, rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

     std::vector<int> allnumrows(nprocs);
     const int blockNumRows = A[0](0,0)->Height();
     MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, MPI_COMM_WORLD);

     for (int b=0; b<numBlocks; ++b)
       {
	 blockProcOffsets[b].resize(nprocs);
	 all_block_num_loc_rows[b].resize(nprocs);
       }
     
     blockProcOffsets[0][0] = 0;
     for (int i=0; i<nprocs-1; ++i)
       blockProcOffsets[0][i+1] = blockProcOffsets[0][i] + allnumrows[i];

     for (int i=0; i<nprocs; ++i)
       {
	 for (int b=0; b<numBlocks; ++b)
	   all_block_num_loc_rows[b][i] = allnumrows[i];

	 for (int b=1; b<numBlocks; ++b)
	   blockProcOffsets[b][i] = blockProcOffsets[0][i];
       }
   }
   
   Ac = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD, offsets, A[0], Asp,
				       Acoef, blockProcOffsets, all_block_num_loc_rows);

   invAc = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*Ac), MPI_COMM_WORLD);

   delete Ac;
}

 void BlockMGSolver::Mult(const Vector &r, Vector &z) const
{
   // Residual vectors
   std::vector<Vector> rv(numGrids + 1);
   // correction vectors
   std::vector<Vector> zv(numGrids + 1);
   // allocation
   for (int i = 0; i <= numGrids ; i++)
   {
      int n = (i==0) ? invAc->Height(): BlkA[i]->Width();
      rv[i].SetSize(n);
      zv[i].SetSize(n);
   }
   // Initial residual
   rv[numGrids] = r;

   // smooth and update residuals down to the coarsest level
   for (int i = numGrids; i > 0 ; i--)
   {
      // Pre smooth
      S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
      // compute residual
      int n = BlkA[i]->Width();
      Vector w(n);
      BlkA[i]->Mult(zv[i], w);
      rv[i] -= w;
      // Restrict
      BlkP[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   }

   // Coarse grid Solve
   invAc->Mult(rv[0], zv[0]);
   //
   for (int i = 1; i <= numGrids ; i++)
   {
      // Prolong correction
      Vector u(BlkP[i - 1]->Height());
      BlkP[i - 1]->Mult(zv[i - 1], u);
      // Update correction
      zv[i] += u;
      // Update residual
      Vector v(BlkA[i]->Height());
      BlkA[i]->Mult(u, v); rv[i] -= v;
      // Post smooth
      S[i - 1]->Mult(rv[i], v); v *= theta;
      // Update correction
      zv[i] += v;
   }
   z = zv[numGrids];
}

BlockMGSolver::~BlockMGSolver() 
{
   for (int i = numGrids - 1; i >= 0 ; i--)
   {
      delete S[i];
      delete BlkP[i];
      delete BlkA[i];
      for (int j=0; j<numBlocks; j++)
      {
         for (int k=0; k<numBlocks; k++)
         {
            delete A[i](j,k);
         }
      }
      A[i].DeleteAll();
   }
   delete BlkA[numGrids];
   delete invAc;
   A.clear();
}
  
} // namespace mfem
