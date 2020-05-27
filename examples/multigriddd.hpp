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
  SparseMatrix AcSp;
  int numGrids, numBlocks;
  //STRUMPACKSolver *invAc = nullptr;
  Operator *invAc = nullptr;
  double theta = 0.5;

public:
  BlockMGSolver(MPI_Comm comm, const int height, const int width, Array2D<HypreParMatrix *>& Af_,
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
    //HypreParMatrix * Ac;

    std::vector<std::vector<int> > blockProcOffsets(numBlocks);
    std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);

    {
      int nprocs, rank;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &nprocs);

      std::vector<int> allnumrows(nprocs);
      const int blockNumRows = A[0](0,0)->Height();    // TODO: Not valid if blocks are of different size
      MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, comm);

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
   
    Ac = CreateHypreParMatrixFromBlocks2(comm, offsets, A[0], Asp,
					 Acoef, blockProcOffsets, all_block_num_loc_rows);

#ifdef MFEM_USE_STRUMPACK
    invAc = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*Ac), comm);
    delete Ac;
#else
    Ac->GetDiag(AcSp);  // AcSp does not own the data
    UMFPackSolver *umf_solver = new UMFPackSolver();
    umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_solver->SetOperator(AcSp);
    invAc = umf_solver;
#endif
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

#ifdef MFEM_USE_STRUMPACK
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
#endif

};

//#define SWTIMING
#define SERIAL_PROLONGATION

class BlockMGPASolver : public Solver
{
private:
  /// The linear system matrix
  Array2D<Operator*>& Af;  // TODO: remove this, as it is used only in the constructor
  Array2D<double>& Acoef;  // TODO: remove this, as it is used only in the constructor
  vector<Array<int>> Aoffsets;
  vector<Array<int>> Poffsets_i;
  vector<Array<int>> Poffsets_j;
  std::vector<Array2D<Operator*>> A;
#ifdef SERIAL_PROLONGATION
  std::vector<SparseMatrix *> P;
#else
  std::vector<HypreParMatrix *> const& P;
#endif
  std::vector<Vector*> const& diag;
  std::vector<Operator*> Pt;
  std::vector<BlockOperator *> BlkP;
  std::vector<BlockOperator *> BlkA;
  std::vector<BlockOperator *> S;
  std::vector<OperatorJacobiSmoother*> Jacobi;

  HypreParMatrix * Ac;
  SparseMatrix AcSp;
  int numGrids, numBlocks;
  //STRUMPACKSolver *invAc = nullptr;
  Operator *invAc = nullptr;
  double theta = 0.5;

public:
  BlockMGPASolver(MPI_Comm comm, const int height, const int width, Array2D<Operator*>& Af_, Array2D<double>& Acoef_, Array2D<HypreParMatrix*> const& BlkAc,
		  std::vector<HypreParMatrix *> const& P_, std::vector<Vector*> const& diag_, Array<int>& ess_tdof_list)
#ifdef SERIAL_PROLONGATION
    : Solver(height, width), Af(Af_), Acoef(Acoef_), diag(diag_)
#else
    : Solver(height, width), Af(Af_), Acoef(Acoef_), P(P_), diag(diag_)
#endif
  {
    timeMult = 0.0;
    timeMultAc = 0.0;
    timeMultPresmooth = 0.0;
    timeMultResidual = 0.0;
    timeMultRestrict = 0.0;

#ifdef SERIAL_PROLONGATION
    P.resize(P_.size());
    for (int i=0; i<P_.size(); ++i)
      {
	P[i] = new SparseMatrix();
	P_[i]->GetDiag(*P[i]);
	P[i]->BuildTranspose();
      }
#endif

    numBlocks = Af.NumRows();
    MFEM_VERIFY(Af.NumCols() == numBlocks, "");
    MFEM_VERIFY(BlkAc.NumCols() == numBlocks && BlkAc.NumRows() == numBlocks, "");
    numGrids = P.size();
    MFEM_VERIFY(diag.size() == numBlocks, "");
    BlkP.resize(numGrids);
    BlkA.resize(numGrids+1);
    Pt.resize(numGrids);
    S.resize(numGrids);
    A.resize(numGrids + 1);
    A[numGrids] = Af;
    Aoffsets.resize(numGrids+1);
    Poffsets_i.resize(numGrids);
    Poffsets_j.resize(numGrids);

    Jacobi.resize(numBlocks);
    for (int i=0; i<numBlocks; ++i)
      Jacobi[i] = new OperatorJacobiSmoother(*(diag[i]), ess_tdof_list, 1.0);
    
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
	Pt[k - 1] = new TransposeOperator(P[k - 1]);

	for (int i=0; i<numBlocks; i++)
	  {
	    for (int j=0; j<numBlocks; j++)
	      {
		if (A[k](i,j) == NULL)
		  A[k - 1](i,j) = NULL;
		else
		  {
		    //A[k - 1](i,j) = RAP(A[k](i,j), P[k - 1]);
		    A[k - 1](i,j) = new TripleProductOperator(Pt[k - 1], A[k](i,j), P[k - 1], false, false, false);
		    BlkA[k]->SetBlock(i, j, A[k](i,j), Acoef(i,j));
		  }
	      }

	    if (k == numGrids)  // finest level
	      S[k - 1]->SetBlock(i,i,Jacobi[i]);
	    else
	      S[k - 1]->SetBlock(i,i, new TripleProductOperator(Pt[k-1], S[k], P[k-1], false, false, false));  // TODO: get a diagonal on each level
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
      {
	offsets[i+1]=A[0](i,i)->Height();
	MFEM_VERIFY(BlkAc(i,i)->Height() == A[0](i,i)->Height(), "");
	MFEM_VERIFY(BlkAc(i,i)->Width() == A[0](i,i)->Width(), "");
      }
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
	    MFEM_VERIFY((A[0](i,j) == NULL) == (BlkAc(i,j) == NULL), "");
	    
	    if (BlkAc(i,j) != NULL)
	      BlkA[0]->SetBlock(i, j, BlkAc(i,j), Acoef(i,j));
	 
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
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &nprocs);

      std::vector<int> allnumrows(nprocs);
      const int blockNumRows = BlkAc(0,0)->Height();  // TODO: Not valid if blocks are of different size
      MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, comm);

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
   
    Ac = CreateHypreParMatrixFromBlocks2(comm, offsets, BlkAc, Asp,
					 Acoef, blockProcOffsets, all_block_num_loc_rows);

#ifdef MFEM_USE_STRUMPACK
    invAc = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*Ac), comm);
    delete Ac;
#else
    Ac->GetDiag(AcSp);  // AcSp does not own the data
    UMFPackSolver *umf_solver = new UMFPackSolver();
    umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_solver->SetOperator(AcSp);
    invAc = umf_solver;
#endif
  }
    
  virtual void SetOperator(const Operator &op) {}

  virtual void SetTheta(const double a) { theta = a; }

  virtual void Mult(const Vector &r, Vector &z) const
  {
#ifdef SWTIMING
    StopWatch sw;
    sw.Clear();
    sw.Start();
#endif

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
	rv[i].UseDevice(true);
	zv[i].UseDevice(true);
      }
    // Initial residual
    rv[numGrids] = r;

    // smooth and update residuals down to the coarsest level
    for (int i = numGrids; i > 0 ; i--)
      {
	// Pre smooth
#ifdef SWTIMING
	StopWatch sws;
	sws.Clear();
	sws.Start();
#endif

	S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;

#ifdef SWTIMING
	sws.Stop();
	timeMultPresmooth += sws.RealTime();
#endif

	// compute residual
	int n = BlkA[i]->Width();
	Vector w(n);
	w.UseDevice(true);
#ifdef SWTIMING
	StopWatch swop;
	swop.Clear();
	swop.Start();
#endif
	BlkA[i]->Mult(zv[i], w);
	rv[i] -= w;
#ifdef SWTIMING
	swop.Stop();
	timeMultResidual += swop.RealTime();
#endif
	// Restrict
#ifdef SWTIMING
	StopWatch swr;
	swr.Clear();
	swr.Start();
#endif
	BlkP[i - 1]->MultTranspose(rv[i], rv[i - 1]);
#ifdef SWTIMING
	swr.Stop();
	timeMultRestrict += swr.RealTime();
#endif
      }

#ifdef SWTIMING
    StopWatch swAc;
    swAc.Clear();
    swAc.Start();
#endif

    // Coarse grid Solve
    invAc->Mult(rv[0], zv[0]);
    //

#ifdef SWTIMING
    swAc.Stop();
    timeMultAc += swAc.RealTime();
#endif

    for (int i = 1; i <= numGrids ; i++)
      {
	// Prolong correction
	Vector u(BlkP[i - 1]->Height());
	u.UseDevice(true);

	BlkP[i - 1]->Mult(zv[i - 1], u);
	// Update correction
	zv[i] += u;
	// Update residual
	Vector v(BlkA[i]->Height());
	v.UseDevice(true);

	BlkA[i]->Mult(u, v); rv[i] -= v;
	// Post smooth
	S[i - 1]->Mult(rv[i], v); v *= theta;
	// Update correction
	zv[i] += v;
      }
    z = zv[numGrids];

#ifdef SWTIMING
    sw.Stop();
    timeMult += sw.RealTime();
#endif
  }
     
  virtual ~BlockMGPASolver()
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

#ifdef MFEM_USE_STRUMPACK
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
#endif

  mutable double timeMult, timeMultAc, timeMultPresmooth, timeMultResidual, timeMultRestrict;
};

}  
//} // namespace mfem

#endif // BGMULTIGRID
