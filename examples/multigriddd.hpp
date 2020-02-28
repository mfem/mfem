#ifndef BGMULTIGRID
#define BGMULTIGRID

#include "mfem.hpp"

using namespace std;

//namespace mfem {
namespace blockgmg {
    
  // Row and column offsets are assumed to be the same, for each process.
  // Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
HypreParMatrix* CreateHypreParMatrixFromBlocks2(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
						Array2D<SparseMatrix*> const& blocksSp,
						Array2D<double> const& coefficient,
						std::vector<std::vector<int> > const& blockProcOffsets,
						std::vector<std::vector<int> > const& all_block_num_loc_rows);

class BlockMGSolver : public Solver
{
private:
  /// The linear system matrix
  Array2D<HypreParMatrix *>& Af;  // TODO: remove this, as it is used only in the constructor
  Array2D<double>& Acoef;  // TODO: remove this, as it is used only in the constructor
  vector<Array<int>> Aoffsets;
  vector<Array<int>> Poffsets_i;
  vector<Array<int>> Poffsets_j;
  std::vector<Array2D<HypreParMatrix *>> A;
  std::vector<HypreParMatrix *> const& P;
  std::vector<BlockOperator *> BlkP;
  std::vector<BlockOperator *> BlkA;
  std::vector<BlockOperator *> S;
  HypreParMatrix * Ac;
  int numGrids, numBlocks;
  STRUMPACKSolver *invAc = nullptr;
  double theta = 0.5;

public:
  BlockMGSolver(const int height, const int width, Array2D<HypreParMatrix *>& Af_,
		Array2D<double>& Acoef_, std::vector<HypreParMatrix *> const& P_)
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

    // Convert to HypreParMatrix
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
   
    Ac = CreateHypreParMatrixFromBlocks2(MPI_COMM_WORLD, offsets, A[0], Asp,
					 Acoef, blockProcOffsets, all_block_num_loc_rows);

    invAc = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*Ac), MPI_COMM_WORLD);

    delete Ac;
  }
    
  virtual void SetOperator(const Operator &op) {}

  virtual void SetTheta(const double a) { theta = a; }

  virtual void Mult(const Vector &r, Vector &z) const
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
     
  virtual ~BlockMGSolver()
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


};

}  
//} // namespace mfem

#endif // BGMULTIGRID
