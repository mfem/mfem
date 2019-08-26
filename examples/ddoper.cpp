#include "ddmesh.hpp"
#include "ddoper.hpp"

InjectionOperator::InjectionOperator(MPI_Comm comm, ParFiniteElementSpace *subdomainSpace, ParFiniteElementSpace *interfaceSpace, int *a,
				     std::vector<int> const& gdofmap) : id(a)
{
  m_comm = comm;
  
  MPI_Comm_size(comm, &m_nprocs);
  MPI_Comm_rank(comm, &m_rank);
    
  const HYPRE_Int sdtos = (subdomainSpace == NULL) ? 0 : subdomainSpace->GetMyTDofOffset();
  const int int_sdtos = sdtos; // TODO: what is HYPRE_Int as an MPI type?

  const int sdtsize = (subdomainSpace == NULL) ? 0 : subdomainSpace->GetTrueVSize();
  const int iftsize = (interfaceSpace == NULL) ? 0 : interfaceSpace->GetTrueVSize();

  height = sdtsize;
  width = iftsize;
  //MFEM_VERIFY(height >= width, "InjectionOperator constructor");

  std::vector<int> allsdtos, alliftos;
  allsdtos.assign(m_nprocs, 0);
  alliftos.assign(m_nprocs, 0);
  m_alliftsize.assign(m_nprocs, 0);
    
  MPI_Allgather(&int_sdtos, 1, MPI_INT, allsdtos.data(), 1, MPI_INT, comm);
  MPI_Allgather(&iftsize, 1, MPI_INT, m_alliftsize.data(), 1, MPI_INT, comm);

  for (int ifp=1; ifp<m_nprocs; ++ifp)
    alliftos[ifp] = alliftos[ifp-1] + m_alliftsize[ifp-1];
  
  int ifgsize = 0;
  MPI_Allreduce(&iftsize, &ifgsize, 1, MPI_INT, MPI_SUM, comm);
    
  m_numTrueSD.assign(m_nprocs, 0);
  m_numLocalTrueSDmappedFromProc.assign(m_nprocs, 0);

  m_iftToSDrank.assign(iftsize, -1);
    
  for (int ifp=0; ifp<m_nprocs; ++ifp)
    {
      for (int i=0; i<m_alliftsize[ifp]; ++i)
	{
	  const int gsd = gdofmap[alliftos[ifp] + i];
	
	  // Find the process owning global SD DOF gsd.
	  int p = 0;
	  for (p=m_nprocs-1; p >= 0; --p)
	    {
	      if (gsd >= allsdtos[p])
		break;
	    }
	
	  m_numTrueSD[p]++;

	  if (p == m_rank)
	    m_numLocalTrueSDmappedFromProc[ifp]++;
	}
    }

  {
    int cnt = 0;
    for (int ifp=0; ifp<m_nprocs; ++ifp)
      cnt += m_numLocalTrueSDmappedFromProc[ifp];
    
    MFEM_VERIFY(cnt == m_numTrueSD[m_rank], "");
  }

  // Set m_alltrueSD to contain local true SD DOF's identified with other processes' local true interface DOF's, ordered according to gdofmap.
  m_alltrueSD.assign(m_numTrueSD[m_rank], 0);

  std::vector<int> os;
  os.assign(m_nprocs, 0);
  for (int ifp=1; ifp<m_nprocs; ++ifp)
    os[ifp] = os[ifp-1] + m_numLocalTrueSDmappedFromProc[ifp-1];

  m_scnt.assign(m_nprocs, 0);
  m_rcnt.assign(m_nprocs, 0);
    
  m_numLocalTrueSDmappedFromProc.assign(m_nprocs, 0);  // reset in the following loop

  for (int ifp=0; ifp<m_nprocs; ++ifp)
    {
      for (int i=0; i<m_alliftsize[ifp]; ++i)
	{
	  const int gsd = gdofmap[alliftos[ifp] + i];
	
	  // Find the process owning global SD DOF gsd.
	  int p = 0;
	  for (p=m_nprocs-1; p >= 0; --p)
	    {
	      if (gsd >= allsdtos[p])
		break;
	    }
	
	  if (p == m_rank)
	    {
	      m_alltrueSD[os[ifp] + m_numLocalTrueSDmappedFromProc[ifp]] = gsd - allsdtos[p];
	      m_numLocalTrueSDmappedFromProc[ifp]++;

	      m_rcnt[ifp]++;
	    }

	  if (ifp == m_rank)
	    {
	      m_scnt[p]++;
	      m_iftToSDrank[i] = p;
	    }
	}
    }

  /*
  //MFEM_VERIFY(sizeof(unsigned int) == sizeof(std::size_t), "");
  int i = sizeof(unsigned int);
  cout << "size of ui " << i << endl;
  i = sizeof(std::size_t);
  cout << "size of st " << i << endl;
  i = sizeof(int);
  cout << "size of int " << i << endl;
  */
    
  m_send.assign(iftsize, 0.0);
  m_recv.assign(m_alltrueSD.size(), 0.0);

  m_sdspl.assign(m_nprocs, 0);
  m_rdspl.assign(m_nprocs, 0);

  for (int i=1; i<m_nprocs; ++i)
    {
      m_sdspl[i] = m_sdspl[i-1] + m_scnt[i-1];
      m_rdspl[i] = m_rdspl[i-1] + m_rcnt[i-1];
    }

  MFEM_VERIFY(m_scnt[m_nprocs-1] + m_sdspl[m_nprocs-1] == iftsize, "");
    
  /*
    trueSD.assign(numTrueSD, -1);
    numTrueSD = 0;
     
    for (auto gsd : gdofmap)
    {
    if (gsd >= sdtos && gsd - sdtos < sdtsize)  // If a local true SD DOF
    {
    trueSD[numTrueSD] = gsd - sdtos;
    numTrueSD++;
    }
    }

    MFEM_VERIFY(numTrueSD == trueSDsize(), "");

    
    std::vector<int> scnt, sdspl, rcnt, rdspl;
    std::vector<std::size_t> sendTrueSD;

    sendTrueSD.assign(, 0);

    scnt.assign(nprocs, 0);
    sdspl.assign(nprocs, 0);
    rcnt.assign(nprocs, 0);
    rdspl.assign(nprocs, 0);

    MPI_Alltoall(scnt.data(), MPI_INT, MPI_INT, MPI_COMM_WORLD);
      
    MPI_Alltoallv(sendTrueSD.data(), scnt.data(), sdsp.data(), MPI_UNSIGNED, m_alltrueSD.data(),
    rcnt.data(), rdspl.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
  */
}


void test1_E_exact(const Vector &x, Vector &E)
{
  const double kappa = M_PI;
  E(0) = sin(kappa * x(1));
  E(1) = sin(kappa * x(2));
  E(2) = sin(kappa * x(0));
}

void test1_f_exact_0(const Vector &x, Vector &f)
{
  const double kappa = M_PI;
  const double c = -kappa * cos(kappa * x(1));
  f(0) = c;
  f(1) = 0.0;
  f(2) = -c;
}

void test1_f_exact_1(const Vector &x, Vector &f)
{
  const double kappa = M_PI;
  const double c = kappa * cos(kappa * x(1));
  f(0) = c;
  f(1) = 0.0;
  f(2) = -c;
}

void test2_E_exact(const Vector &x, Vector &E)
{
  const double pi = M_PI;
  E(0) = sin(pi * x(1)) * sin(pi * x(2));
  E(1) = sin(pi * x(2)) * sin(pi * x(0));
  E(2) = sin(pi * x(0)) * sin(pi * x(1));

  /*  
  E(0) = x(1) * (1.0 - x(1)) * x(2) * (1.0 - x(2));
  E(1) = x(0) * (1.0 - x(0)) * x(2) * (1.0 - x(2));
  E(2) = x(1) * (1.0 - x(1)) * x(0) * (1.0 - x(0));
  */
  /*  
  E(0) = x(0) * x(1) * (1.0 - x(1)) * x(2) * (1.0 - x(2));
  E(1) = x(1) * x(0) * (1.0 - x(0)) * x(2) * (1.0 - x(2));
  E(2) = x(2) * x(1) * (1.0 - x(1)) * x(0) * (1.0 - x(0));
  */
}

void test2_f_exact_0(const Vector &x, Vector &f)
{
  const double pi = M_PI;
  f(0) = pi * sin(pi * x(2)) * (cos(pi * x(0)) - cos(pi * x(1)));
  f(1) = 0.0;
  f(2) = -pi * sin(pi * x(0)) * (cos(pi * x(1)) - cos(pi * x(2)));
}

void test2_f_exact_1(const Vector &x, Vector &f)
{
  const double pi = M_PI;
  f(0) = -pi * sin(pi * x(2)) * (cos(pi * x(0)) - cos(pi * x(1)));
  f(1) = 0.0;
  f(2) = pi * sin(pi * x(0)) * (cos(pi * x(1)) - cos(pi * x(2)));
}

// Note that test2_rho is zero on the exterior boundary.
double test2_rho_exact_0(const Vector &x)
{
  const double pi = M_PI;

  return -pi * pi * ((sin(pi * x(2)) * sin(pi * x(0))) + (sin(pi * x(0)) * sin(pi * x(2))));
}

double test2_rho_exact_1(const Vector &x)
{
  const double pi = M_PI;

  return pi * pi * ((sin(pi * x(2)) * sin(pi * x(0))) + (sin(pi * x(0)) * sin(pi * x(2))));
}

SparseMatrix* GetSparseMatrixFromOperator(Operator *op)
{
  const int n = op->Height();
  MFEM_VERIFY(n == op->Width(), "");

  SparseMatrix *S = new SparseMatrix(n);
  Vector x(n);
  Vector y(n);

  for (int j=0; j<n; ++j)
    {
      x = 0.0;
      x[j] = 1.0;
      op->Mult(x, y);

      for (int i=0; i<n; ++i)
	{
	  //if (y[i] != 0.0)
	  if (fabs(y[i]) > 1.0e-15)
	    {
	      S->Set(i, j, y[i]);
	    }
	}
    }
      
  S->Finalize();

  return S;
}

void PrintNonzerosOfInterfaceOperator(Operator const& op, const int tsize0, const int tsize1, set<int> const& tdofsbdry0,
				      set<int> const& tdofsbdry1, const int nprocs, const int rank)
{
  const int n0 = op.Height();
  const int n1 = op.Width();

  const int tbdrysize0 = tdofsbdry0.size();
  const int tbdrysize1 = tdofsbdry1.size();

  std::vector<int> tdofsbdryvec0(tbdrysize0);
  std::vector<int> tdofsbdryvec1(tbdrysize1);

  {
    int i = 0;
    for (std::set<int>::const_iterator it = tdofsbdry0.begin(); it != tdofsbdry0.end(); ++it, ++i)
      tdofsbdryvec0[i] = *it;

    MFEM_VERIFY(i == tbdrysize0, "");
    
    i = 0;
    for (std::set<int>::const_iterator it = tdofsbdry1.begin(); it != tdofsbdry1.end(); ++it, ++i)
      tdofsbdryvec1[i] = *it;

    MFEM_VERIFY(i == tbdrysize1, "");
  }
  
  const int nf0 = n0 + tsize0 - tbdrysize0;
  const int nf1 = n1 + tsize1 - tbdrysize1;
  
  std::vector<int> alln0(nprocs);
  MPI_Allgather(&n0, 1, MPI_INT, alln0.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> alln1(nprocs);
  MPI_Allgather(&n1, 1, MPI_INT, alln1.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> allnf0(nprocs);
  MPI_Allgather(&nf0, 1, MPI_INT, allnf0.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> allnf1(nprocs);
  MPI_Allgather(&nf1, 1, MPI_INT, allnf1.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> alltsize1(nprocs);
  MPI_Allgather(&tsize1, 1, MPI_INT, alltsize1.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> alltbdrysize1(nprocs);
  MPI_Allgather(&tbdrysize1, 1, MPI_INT, alltbdrysize1.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int myos0 = 0;
  int myos1 = 0;

  int myosf0 = 0;
  int myosf1 = 0;

  int sumtbdrysize1 = 0;

  std::vector<int> rdspl(nprocs);
  rdspl[0] = 0;
  
  for (int i=0; i<nprocs; ++i)
    {
      if (i < rank)
	{
	  myosf0 += allnf0[i];
	  myosf1 += allnf1[i];
	}

      sumtbdrysize1 += alltbdrysize1[i];

      if (i > 0)
	rdspl[i] = rdspl[i-1] + alltbdrysize1[i-1];
    }

  std::vector<int> alltdofsbdryvec1(sumtbdrysize1);
  
  MPI_Allgatherv(tdofsbdryvec1.data(), tbdrysize1, MPI_INT, alltdofsbdryvec1.data(), alltbdrysize1.data(), rdspl.data(), MPI_INT, MPI_COMM_WORLD);

  //return;
  
  //MFEM_VERIFY(cnt == ng, "");
  
  Vector ej(n1);
  Vector Aej(n0);

  char buffer[80];
  //sprintf(buffer, "tmpppppppp.%05d", rank);
  sprintf(buffer, "ifopPar.%05d", rank);
  //sprintf(buffer, "ifopSer.%05d", rank);
  ofstream file(buffer);
  //ofstream file("ifopPar" + std::to_string(rank));

  file << myosf0 << " " << myosf0 + nf0 - 1 << " " << myosf1 << " " << myosf1 + nf1 - 1 << endl; 

  int posf1 = 0;
  for (int p=0; p<nprocs; ++p)
    {
      for (int j=0; j<alln1[p]; ++j)
	{
	  ej = 0.0;

	  if (p == rank)  // if true dof is on this process
	    ej[j] = 1.0;

	  int fullcol = posf1;
	  if (j < alltbdrysize1[p])
	    fullcol += alltdofsbdryvec1[rdspl[p] + j];
	  else
	    fullcol += j + alltsize1[p] - alltbdrysize1[p];
	      
	  //Aej = 0.0;  // results should be the same with or without this line
	  op.Mult(ej, Aej);

	  // Aej is now the local rows (on proc myid) for column j of proc p.

	  for (int i=0; i<n0; ++i)
	    {
	      if (fabs(Aej[i]) > 1.0e-15)
		{
		  int fullrow = myosf0;
		  if (i < tbdrysize0)
		    fullrow += tdofsbdryvec0[i];
		  else
		    fullrow += i + tsize0 - tbdrysize0;

		  file << fullrow << " " << fullcol << " " << setprecision(15) << Aej[i] << endl;
		}
	    }
	}
      
      posf1 += allnf1[p];
    }
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

SparseMatrix* GatherHypreParMatrix(HypreParMatrix *A)
{
  int nprocs, rank;
  MPI_Comm_rank(A->GetComm(), &rank);
  MPI_Comm_size(A->GetComm(), &nprocs);

  hypre_CSRMatrix* csr = GetHypreParMatrixData(*A);

  const HYPRE_Int globalNumRows = A->GetGlobalNumRows();
  const HYPRE_Int globalNumCols = A->GetGlobalNumCols();
  
  SparseMatrix *S = NULL;
  
  // Gather to rank 0
  const int nrows = csr->num_rows;

  int *allnrows = (nprocs > 0) ? new int[nprocs] : NULL;
  int *dspl = (nprocs > 0) ? new int[nprocs] : NULL;
  dspl[0] = 0;
  
  MPI_Allgather(&nrows, 1, MPI_INT, allnrows, 1, MPI_INT, A->GetComm());

  int sum = 0;
  for (int i=0; i<nprocs; ++i)
    {
      sum += allnrows[i];

      if (i<nprocs-1)
	dspl[i+1] = dspl[i] + allnrows[i];
    }
  
  MFEM_VERIFY(sum == globalNumRows, "");
  
  int *nnz = (nrows > 0) ? new int[nrows] : NULL;
  int *allnnz = (rank == 0) ? new int[globalNumRows + 1] : NULL;

  int totalnnz = 0;
  for (int k=0; k<nrows; ++k)
    {
      const int osk = csr->i[k];
      const int nnz_k = csr->i[k+1] - csr->i[k];
      /*
      for (int l=0; l<nnz_k; ++l)
	{
	  if (csr->j[osk + l] >= nrows)
	    {
	      cout << "(" << i << ", " << j << ") row " << k;
	      cout << ", col " << csr->j[osk + l] << endl;
	    }
	}
      */
      
      nnz[k] = nnz_k;
      totalnnz += nnz_k;
    }

  int *alltotalnnz = (nprocs > 0) ? new int[nprocs] : NULL;

  MPI_Gather(&totalnnz, 1, MPI_INT, alltotalnnz, 1, MPI_INT, 0, A->GetComm());
  MPI_Gatherv(nnz, nrows, MPI_INT, allnnz, allnrows, dspl, MPI_INT, 0, A->GetComm());

  int globaltotalnnz = 0;
  int *allj = NULL;
  double *alldata = NULL;
  if (rank == 0)
    {
      for (int i=globalNumRows-1; i>=0; --i)
	allnnz[i+1] = allnnz[i];

      allnnz[0] = 0;
      
      // Now allnnz[i] is nnz for row i-1. Do a partial sum to get offsets.
      for (int i=0; i<globalNumRows; ++i)
	allnnz[i+1] += allnnz[i];
      
      for (int i=0; i<nprocs; ++i)
	{
	  globaltotalnnz += alltotalnnz[i];
	  if (i<nprocs-1)
	    dspl[i+1] = dspl[i] + alltotalnnz[i];
	}

      MFEM_VERIFY(allnnz[globalNumRows] == globaltotalnnz, "");
      
      allj = new int[globaltotalnnz];
      alldata = new double[globaltotalnnz];

      S = new SparseMatrix(globalNumRows, globalNumCols);
    }

  MPI_Gatherv(csr->j, totalnnz, MPI_INT, allj, alltotalnnz, dspl, MPI_INT, 0, A->GetComm());
  MPI_Gatherv(csr->data, totalnnz, MPI_DOUBLE, alldata, alltotalnnz, dspl, MPI_DOUBLE, 0, A->GetComm());

  if (rank == 0)
    {
      for (int i=0; i<globalNumRows; ++i)
	{
	  for (int k=allnnz[i]; k<allnnz[i+1]; ++k)
	    S->Set(i, allj[k], alldata[k]);
	  
	  //S->Add(i, j, 0.0);
	}
      
      S->Finalize();

      bool nnzmatch = true;
      for (int i=0; i<globalNumRows; ++i)
	{
	  if (S->GetI()[i+1] != allnnz[i+1])
	    nnzmatch = false;
	}

      // Note that alldata may have some zeros, so S may be more compressed than alldata.
      
      //MFEM_VERIFY(nnzmatch, "");
      /*
      std::ofstream sfile("spmat.txt", std::ofstream::out);
      S->PrintMatlab(sfile);
      sfile.close();
      */
    }

  if (allnrows)
    delete allnrows;

  if (nnz)
    delete nnz;

  if (dspl)
    delete dspl;
  
  if (allnnz)
    delete allnnz;	      

  if (alltotalnnz)
    delete alltotalnnz;
  
  if (allj)
    delete allj;

  if (alldata)
    delete alldata;

  return S;
}

SparseMatrix* ReceiveSparseMatrix(MPI_Comm comm, const int source)
{
  int sizes[3];  // {matrix height, matrix width, number of nonzeros}

  MPI_Recv(sizes, 3, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);

  MFEM_VERIFY(sizes[0] > 0 && sizes[1] > 0 && sizes[2] > 0, "");
  
  int *I = new int[sizes[0]+1];
  int *J = new int[sizes[2]];
  double *data = new double[sizes[2]];

  MPI_Recv(I, sizes[0]+1, MPI_INT, source, 1, comm, MPI_STATUS_IGNORE);
  MPI_Recv(J, sizes[2], MPI_INT, source, 2, comm, MPI_STATUS_IGNORE);
  MPI_Recv(data, sizes[2], MPI_DOUBLE, source, 3, comm, MPI_STATUS_IGNORE);

  MFEM_VERIFY(sizes[2] == I[sizes[0]], "");

  SparseMatrix *S = new SparseMatrix(sizes[0], sizes[1]);
  
  for (int k=0; k<sizes[0]; ++k)  // loop over rows
    {
      for (int l=I[k]; l<I[k+1]; ++l)  // loop over nonzeros
	{
	  S->Set(k, J[l], data[l]);
	}
    }

  delete[] I;
  delete[] J;
  delete[] data;

  S->Finalize();

  return S;
}

void SendSparseMatrix(MPI_Comm comm, SparseMatrix *S, const int dest)
{
  int sizes[3];

  sizes[0] = S->Height();
  sizes[1] = S->Width();
  sizes[2] = S->GetI()[sizes[0]];  // number of nonzeros
  
  MPI_Send(sizes, 3, MPI_INT, dest, 0, comm);

  MPI_Send(S->GetI(), sizes[0]+1, MPI_INT, dest, 1, comm);
  MPI_Send(S->GetJ(), sizes[2], MPI_INT, dest, 2, comm);
  MPI_Send(S->GetData(), sizes[2], MPI_DOUBLE, dest, 3, comm);
}

void GetStarts(MPI_Comm comm, const int rank, const int nprocs, const int n, std::vector<HYPRE_Int> & starts, int& g)
{
  std::vector<int> alln(nprocs);
  MPI_Allgather(&n, 1, MPI_INT, alln.data(), 1, MPI_INT, comm);
	      
  int first_loc = 0;
  g = 0;
  for (int i=0; i<nprocs; ++i)
    {
      g += alln[i];
      if (i < rank)
	first_loc += alln[i];
    }
	      
  starts[0] = first_loc;
  starts[1] = first_loc + alln[rank];
}

// We assume here that the sparsity structure of the interface matrix I is contained in that of the subdomain matrix A.
// We simply add entries from I to existing entries of A. The constant cI times the interface identity matrix is added.
HypreParMatrix* AddSubdomainMatrixAndInterfaceMatrix(MPI_Comm ifcomm, HypreParMatrix *A, HypreParMatrix *I, std::vector<int> & inj,
						     std::vector<int> & ginj,
						     ParFiniteElementSpace *ifespace, ParFiniteElementSpace *ifespace2,
						     const bool adding, const double cI, const double cA, const bool injRows)
{
  // inj maps from full local interface DOF's to local true DOF's in the subdomain. 

  // Gather the entire global interface matrix to one root process, namely rank 0 in the ifespace communicator.

  SparseMatrix *globalI = NULL;
  if (ifespace != NULL)
    {
      globalI = GatherHypreParMatrix(I);

      if (globalI != NULL && cI != 0.0)
	{
	  for (int i=0; i<globalI->Size(); ++i)
	    {
	      globalI->Add(i, i, cI);
	    }
	}
    }

  // Find the unique rank in ifcomm of the ifespace rank 0 process that stores globalI. 

  int nprocs, rank;
  MPI_Comm_size(ifcomm, &nprocs);
  MPI_Comm_rank(ifcomm, &rank);
  
  /*
  int A_rank;
  MPI_Comm_rank(A->GetComm(), &A_rank);
  */
  
  int *allg = new int[nprocs];
  
  const int haveGlobalI = (globalI != NULL) ? 1 : 0;
  
  MPI_Allgather(&haveGlobalI, 1, MPI_INT, allg, 1, MPI_INT, ifcomm);

  int count = 0;
  int ownerGlobalI = -1;
  for (int i=0; i<nprocs; ++i)
    {
      if (allg[i] == 1)
	{
	  ownerGlobalI = i;
	  count++;
	}
    }

  cout << rank << ": ownerGlobalI " << ownerGlobalI << ", count " << count << ", nprocs " << nprocs << endl;
  
  MFEM_VERIFY(ownerGlobalI >= 0 && count == 1, "");
  
  const int haveA = (A != NULL) ? 1 : 0;
  
  MPI_Allgather(&haveA, 1, MPI_INT, allg, 1, MPI_INT, ifcomm);

  count = 0;
  for (int i=0; i<nprocs; ++i)
    {
      if (allg[i] == 1)
	{
	  count++;
	}
    }

  MFEM_VERIFY(count > 0, "");

  // Send entire global interface matrix to all processes in the subdomain communicator.
  // Note that the processes owning the interface matrix I may be different from those in the subdomain communicator. 

  if (haveGlobalI)
    {
      for (int i=0; i<nprocs; ++i)
	{
	  if (i != rank) //  && allg[i] == 1)
	    SendSparseMatrix(ifcomm, globalI, i);
	}
    }
  else // if (haveA)
    {
      globalI = ReceiveSparseMatrix(ifcomm, ownerGlobalI);

      /*
      std::string filename = "spmat_r" + std::to_string(rank) + ".txt";
      std::ofstream sfile(filename, std::ofstream::out);
      
      globalI->PrintMatlab(sfile);
      sfile.close();
      */
    }
  
  MPI_Barrier(ifcomm);
  
  // First cast the matrix A to a hypre_ParCSRMatrix
  //hypre_ParCSRMatrix * A_parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*A);

  HypreParMatrix *Acopy = NULL;
  hypre_CSRMatrix * csr_op = NULL;
  
  if (adding && haveA)
    {
      Acopy = A;  // TODO: make a copy of A?
      //HypreParMatrix Acopy(A_parcsr_op);
  
      hypre_ParCSRMatrix * parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Acopy);
  
      MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

      // Create the CSRMatrixMPI A_ by borrowing the internal data from a hypre_CSRMatrix.
      csr_op = hypre_MergeDiagAndOffd(parcsr_op);

      cout << rank << ": num rows " << csr_op->num_rows << ", RP0 " << A->RowPart()[0] << ", RP1 " << A->RowPart()[1] << endl;

      if (A != NULL)
	MFEM_VERIFY(csr_op->num_rows == A->RowPart()[1] - A->RowPart()[0], "");
    }
  
  // Create map from local true interface DOF's to local full interface DOF's. Composing inj with that map takes indices
  // from the local rows of the global interface matrix to local true DOF's in the subdomain. Gathering the result over
  // all processes yields a map from all global interface DOF's to global true subdomain DOF's.
  
  const int iftsize = (ifespace == NULL) ? 0 : ifespace->GetTrueVSize();
  const int iftsize2 = (ifespace2 == NULL) ? 0 : ifespace2->GetTrueVSize();
  const int ifullsize = (ifespace == NULL) ? 0 : ifespace->GetVSize();
  const int ifullsize2 = (ifespace2 == NULL) ? 0 : ifespace2->GetVSize();

  int *ifullsdt = (ifullsize > 0) ? new int[ifullsize] : NULL;  // interface full to subdomain true DOF map, well-defined as the max of inj over all processes.

  // Create map from interface full DOF's to global subdomain true DOF's. First, verify that global true DOF's go from 0 to the global matrix size minus 1.
  bool gtdofInRange = true;
  bool maxGlobalTDofFound = false;
  const int igsize = (I == NULL) ? 0 : I->GetGlobalNumRows();
  const int maxgtdof = igsize - 1;

  int alligsize = -1;
  MPI_Allreduce(&igsize, &alligsize, 1, MPI_INT, MPI_MAX, ifcomm);

  MFEM_VERIFY(alligsize > 0, "");

  if (globalI != NULL)
    {
      MFEM_VERIFY(alligsize == globalI->Size(), "");
    }
  
  int *ifgtsdt = new int[alligsize];  // interface global true to subdomain true DOF's.
  int *maxifgtsdt = new int[alligsize];  // interface global true to subdomain true DOF's.
  for (int i=0; i<alligsize; ++i)
    {
      ifgtsdt[i] = -1;
      maxifgtsdt[i] = -1;
    }

  const int ossdt = (A != NULL) ? A->RowPart()[0] : 0;
  const int ossdtNext = (A != NULL) ? A->RowPart()[1] : 0;

  if (haveA)
    MFEM_VERIFY(A->Height() == A->RowPart()[1] - A->RowPart()[0], "");
  
  for (int i=0; i<ifullsize; ++i)
    {
      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
      if (gtdof == maxgtdof)
	maxGlobalTDofFound = true;

      if (gtdof < 0 || gtdof > maxgtdof)
	gtdofInRange = false;

      ifgtsdt[gtdof] = (inj[i] >= 0) ? inj[i] + ossdt : inj[i];  // could be -1 if not defined
      //ifullsdt[i] = inj[i];  // could be -1 if not defined
    }

  bool allMaxGlobalTDofFound = false;

  MPI_Allreduce(&maxGlobalTDofFound, &allMaxGlobalTDofFound, 1, MPI_C_BOOL, MPI_LOR, ifcomm);
  
  MFEM_VERIFY(gtdofInRange && allMaxGlobalTDofFound, "");

  MPI_Allreduce(ifgtsdt, maxifgtsdt, alligsize, MPI_INT, MPI_MAX, ifcomm);  // TODO: can this be done in place?
  
  int *iftfull = (iftsize > 0) ? new int[iftsize] : NULL;
  int *iftfull2 = (iftsize2 > 0) ? new int[iftsize2] : NULL;
  int *iftsdt = (iftsize > 0) ? new int[iftsize] : NULL;  // interface true to subdomain true DOF map

  for (int i=0; i<iftsize; ++i)
    {
      iftfull[i] = -1;
    }
  
  for (int i=0; i<iftsize2; ++i)
    {
      iftfull2[i] = -1;
    }
  
  count = 0;
  for (int i=0; i<ifullsize; ++i)
    {
      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
      ifullsdt[i] = maxifgtsdt[gtdof];
      
      const int ltdof = ifespace->GetLocalTDofNumber(i);
      if (ltdof >= 0)
	{
	  iftfull[ltdof] = i;
	  count++;
	}
    }

  MFEM_VERIFY(count == iftsize, "");

  count = 0;
  for (int i=0; i<ifullsize2; ++i)
    {
      const int ltdof = ifespace2->GetLocalTDofNumber(i);
      if (ltdof >= 0)
	{
	  iftfull2[ltdof] = i;
	  count++;
	}
    }

  MFEM_VERIFY(count == iftsize2, "");

  for (int i=0; i<iftsize; ++i)
    {
      MFEM_VERIFY(iftfull[i] >= 0, "");

      // TODO: is this ever used?
      iftsdt[i] = ifullsdt[iftfull[i]];  // well-defined
    }

  MPI_Barrier(ifcomm);

  if (globalI != NULL && A != NULL && adding) // Add values from globalI to the local rows of csr_op
    {
      MFEM_VERIFY(globalI->Size() == alligsize, "");

      if (cA != 1.0) // Scale A entries in csr_op by cA
	{
	  for (int iA=0; iA<A->Height(); ++iA)
	    {
	      for (int l=csr_op->i[iA]; l<csr_op->i[iA + 1]; ++l)
		csr_op->data[l] *= cA;
	    }
	}
      
      bool allEntriesFound = true;

      for (int i=0; i<alligsize; ++i)
	{
	  //if (ifgtsdt[i] >= ossdt && ifgtsdt[i] < ossdtNext)  // if subdomain true DOF is on this process
	  if (ginj[i] >= ossdt && ginj[i] < ossdtNext)  // if subdomain true DOF is on this process
	    {
	      //const int ltsd = ifgtsdt[i] - ossdt;  // local true subdomain DOF
	      const int ltsd = ginj[i] - ossdt;  // local true subdomain DOF

	      // Add values in row i of globalI to values in row ltsd of csr_op

	      for (int k=globalI->GetI()[i]; k<globalI->GetI()[i+1]; ++k)
		{
		  const int igcol = globalI->GetJ()[k];
		  const double d = globalI->GetData()[k];

		  //const int sdcol = maxifgtsdt[igcol];
		  const int sdcol = ginj[igcol];
		  
		  // Find column index in csr_op->j
		  int colid = -1;
		  for (int l=csr_op->i[ltsd]; l<csr_op->i[ltsd+1]; ++l)
		    {
		      if (sdcol == csr_op->j[l])
			colid = l;
		    }

		  if (colid >= 0)
		    csr_op->data[colid] += d;
		  else
		    allEntriesFound = false;
		}
	    }
	}

      MFEM_VERIFY(allEntriesFound, "");
    }

  MPI_Barrier(ifcomm);

  HypreParMatrix *S = NULL;

  //if (A != NULL)
    {
      if (adding && A != NULL)
	{
	  const int size = A->GetGlobalNumRows();
	  S = new HypreParMatrix(A->GetComm(), csr_op->num_rows, size, size, csr_op->i, csr_op->j, csr_op->data, Acopy->RowPart(), Acopy->ColPart());
	}
      else if (!adding && globalI != NULL)
	{
	  MFEM_VERIFY(globalI->Height() == alligsize, "");

	  const bool mixed = (ifespace2 != NULL);  // interface matrix I is in the mixed space (ifespace, ifespace2)

#ifdef MIXED_MATRIX_NO_TRANSPOSE
	  // Create new sparse matrix of size nrows(A) x ncols(I) if injRows, else of size nrows(I) x ncols(A)

	  const int numLocRows = injRows ? ((A == NULL) ? 0 : A->Height()) : ((I == NULL) ? 0 : I->Height());

	  //MFEM_VERIFY(numLocRows > 0, "");

	  std::vector<std::size_t> rowCount;
	  //if (numLocRows > 0)
	  rowCount.assign(numLocRows, 0);
	  
	  int *SI = new int[numLocRows + 1];
	  for (auto i=0; i<numLocRows+1; ++i)
	    SI[i] = 0;

	  if (injRows) // Inject the rows from interface DOF's to subdomain DOF's.
	    {
	      /*
	      // Loop over interface full DOF's, finding only those that correspond to SD local true DOF's. The corresponding rows are injected.
	      for (int i=0; i<ifullsize; ++i)
		{
		  if (ifullsdt[i] >= 0 && ifullsdt[i] >= ossdt && ifullsdt[i] < ossdtNext)
		    {
		      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
		      SI[ifullsdt[i] - ossdt + 1] += globalI->GetI()[gtdof+1] - globalI->GetI()[gtdof];
		    }
		}
	      */

	      // Loop over all global interface DOF's, finding only those that correspond to SD local true DOF's. The corresponding rows are injected.
	      for (int i=0; i<alligsize; ++i)
		{
		  if (ginj[i] >= ossdt && ginj[i] < ossdtNext)  // if subdomain true DOF is on this process
		    {
		      // Inject row i of globalI to row ltsd of csr_op
		      const int ltsd = ginj[i] - ossdt;  // local true subdomain DOF
		      SI[ltsd + 1] = globalI->GetI()[i+1] - globalI->GetI()[i];
		    }
		}
	    }
	  else
	    {
	      //MFEM_VERIFY(iftsize == I->Height(), "");

	      for (int i=0; i<iftsize; ++i)
		{
		  const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(iftfull[i]);
		  SI[i+1] += globalI->GetI()[gtdof+1] - globalI->GetI()[gtdof];
		}
	    }
	  
	  // Partial sum
	  for (auto i=0; i<numLocRows; ++i)
	    SI[i+1] += SI[i];
	  
	  const HYPRE_Int nnz = SI[numLocRows];
	  HYPRE_Int *SJ = new HYPRE_Int[nnz];
	  double *Sdata = new double[nnz];

	  HYPRE_Int cnt = 0;

	  if (injRows) // Inject the rows from interface DOF's to subdomain DOF's.
	    {
	      /*
	      // Loop over interface full DOF's, finding only those that correspond to SD local true DOF's. The corresponding rows are injected.
	      for (int i=0; i<ifullsize; ++i)  
		{
		  if (ifullsdt[i] >= 0 && ifullsdt[i] >= ossdt && ifullsdt[i] < ossdtNext)
		    {
		      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
		      const std::size_t row = ifullsdt[i] - ossdt;
		      for (int k=globalI->GetI()[gtdof]; k<globalI->GetI()[gtdof+1]; ++k, ++cnt, rowCount[row]++)
			{ // Copy values in row gtdof of globalI to local row ifullsdt[i] of S.
			  SJ[SI[row]+rowCount[row]] = globalI->GetJ()[k];  // Column index in globalI is already global in ifespace or ifespace2.
			  Sdata[SI[row]+rowCount[row]] = globalI->GetData()[k];
			}
		    }
		}
	      */

	      // Loop over all global interface DOF's, finding only those that correspond to SD local true DOF's. The corresponding rows are injected.
	      for (int i=0; i<alligsize; ++i)
		{
		  if (ginj[i] >= ossdt && ginj[i] < ossdtNext)  // if subdomain true DOF is on this process
		    {
		      // Inject row i of globalI to row ltsd of csr_op
		      const int ltsd = ginj[i] - ossdt;  // local true subdomain DOF

		      for (int k=globalI->GetI()[i]; k<globalI->GetI()[i+1]; ++k, ++cnt, rowCount[ltsd]++)
			{ // Copy values in row i of globalI to local row ltsd of S.
			  SJ[SI[ltsd]+rowCount[ltsd]] = globalI->GetJ()[k];  // Column index in globalI is already global in ifespace or ifespace2.
			  Sdata[SI[ltsd]+rowCount[ltsd]] = globalI->GetData()[k];
			}
		    }
		}
	    }
	  else
	    {
	      MFEM_VERIFY(!mixed, "");

	      // Get iftsize on all processes, to map from global ifespace indices (columns of globalI) to local indices.
	      /*
	      int ifnprocs, ifrank;
	      MPI_Comm_size(ifcomm, &ifnprocs);
	      MPI_Comm_rank(ifcomm, &ifrank);
	      
	      std::vector<int> alliftsize(ifnprocs);
	      MPI_Allgather(&iftsize, 1, MPI_INT, alliftsize.data(), 1, MPI_INT, ifcomm);

	      int ifrankos[2];

	      int sumSizes = 0;
	      for (int i=0; i<ifnprocs; ++i)
		{
		  if (i == ifrank)
		    ifrankos[0] = sumSizes;
		  
		  sumSizes += alliftsize[i];
		}

	      MFEM_VERIFY(globalI->Width() == sumSizes, "");

	      ifrankos[1] = ifrankos[0] + alliftsize[ifrank];
	      */
	      
	      for (int i=0; i<iftsize; ++i)
		{
		  const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(iftfull[i]);

		  for (int k=globalI->GetI()[gtdof]; k<globalI->GetI()[gtdof+1]; ++k, ++cnt, rowCount[i]++)
		    { // Copy values in row gtdof of globalI to local row i of S.
		      /*
		      if (maxifgtsdt[globalI->GetJ()[k]] < 0)
			cout << "BUG" << endl;
		      
		      SJ[SI[i]+rowCount[i]] = maxifgtsdt[globalI->GetJ()[k]];  // Column index in globalI is already global in ifespace or ifespace2. Map to global injected DOF.
		      */

		      SJ[SI[i]+rowCount[i]] = ginj[globalI->GetJ()[k]];  // Column index in globalI is already global in ifespace or ifespace2. Map to global injected DOF.
		      
		      Sdata[SI[i]+rowCount[i]] = globalI->GetData()[k];
		    }
		}
	    }
	  
	  std::size_t sumCnt{0};
	  for (auto i : rowCount)
	    sumCnt += i;
	  
	  MFEM_VERIFY(nnz == cnt && sumCnt == cnt, "");

	  MPI_Barrier(ifcomm);
	  
	  if (injRows)
	    {
	      const int num_loc_rows = (A == NULL) ? 0 : A->Height();
	      const int num_loc_cols = (I == NULL) ? 0 : I->Width();

	      int glob_nrows = 0;
	      int glob_ncols = 0;

	      std::vector<HYPRE_Int> rowStarts(2);
	      std::vector<HYPRE_Int> colStarts(2);

	      GetStarts(ifcomm, rank, nprocs, num_loc_rows, rowStarts, glob_nrows);
	      GetStarts(ifcomm, rank, nprocs, num_loc_cols, colStarts, glob_ncols);
	      
	      //S = new HypreParMatrix(ifcomm, numLocRows, A->GetGlobalNumRows(), globalI->Width(), SI, SJ, Sdata, A->RowPart(), I->ColPart());
	      S = new HypreParMatrix(ifcomm, numLocRows, glob_nrows, globalI->Width(), SI, SJ, Sdata, (HYPRE_Int*) rowStarts.data(), (HYPRE_Int*) colStarts.data());
	    }
	  else
	    {
	      const int num_loc_rows = (I == NULL) ? 0 : I->Height();
	      const int num_loc_cols = (A == NULL) ? 0 : A->Width();

	      int glob_nrows = 0;
	      int glob_ncols = 0;
	      std::vector<HYPRE_Int> rowStarts(2);
	      std::vector<HYPRE_Int> colStarts(2);

	      GetStarts(ifcomm, rank, nprocs, num_loc_rows, rowStarts, glob_nrows);
	      GetStarts(ifcomm, rank, nprocs, num_loc_cols, colStarts, glob_ncols);
	      
	      //S = new HypreParMatrix(ifcomm, numLocRows, globalI->Height(), A->GetGlobalNumCols(), SI, SJ, Sdata, I->RowPart(), A->ColPart());
	      S = new HypreParMatrix(ifcomm, numLocRows, globalI->Height(), glob_ncols, SI, SJ, Sdata, (HYPRE_Int*) rowStarts.data(), (HYPRE_Int*) colStarts.data());
	    }
#else
	  // Create new sparse matrix of size ncols(I) x nrows(A)

	  const int numLocRows = mixed ? iftsize2 : iftsize;

	  SparseMatrix *globalI_tr = mixed ? Transpose(*globalI) : NULL;  // not the most efficient implementation, to avoid premature optimization.
	  SparseMatrix *globalImat = mixed ? globalI_tr : globalI;  // In the non-mixed case, globalI is assumed to be symmetric, so no transpose is needed.
	  
	  MFEM_VERIFY(numLocRows > 0, "");
	  
	  int *SI = new int[numLocRows + 1];
	  SI[0] = 0;
	  
	  for (int i=0; i<numLocRows; ++i)  // Loop over interface true DOF's, which correspond to local rows of S.
	    {
	      const HYPRE_Int gtdof = mixed ? ifespace2->GetGlobalTDofNumber(iftfull2[i]) : ifespace->GetGlobalTDofNumber(iftfull[i]);
	      SI[i+1] = SI[i] + globalImat->GetI()[gtdof+1] - globalImat->GetI()[gtdof];  // local offsets to rows
	    }

	  const HYPRE_Int nnz = SI[numLocRows];
	  HYPRE_Int *SJ = new HYPRE_Int[nnz];
	  double *Sdata = new double[nnz];

	  bool gciValid = true;
	  
	  HYPRE_Int cnt = 0;
	  for (int i=0; i<numLocRows; ++i)  // Loop over interface true DOF's, which correspond to local rows of S.
	    {
	      const HYPRE_Int gtdof = mixed ? ifespace2->GetGlobalTDofNumber(iftfull2[i]) : ifespace->GetGlobalTDofNumber(iftfull[i]);
	      for (int k=globalImat->GetI()[gtdof]; k<globalImat->GetI()[gtdof+1]; ++k, ++cnt)  // Copy values in row gtdof of globalImat to values in local row i of S.
		{
		  // Whether mixed is true or not, globalImat->GetJ() has global column indices in ifespace.
		  SJ[cnt] = maxifgtsdt[globalImat->GetJ()[k]];

		  if (SJ[cnt] < 0)
		    gciValid = false;
		  
		  Sdata[cnt] = globalImat->GetData()[k];
		}
	    }

	  MFEM_VERIFY(nnz == cnt && gciValid, "");

	  if (globalI_tr)
	    delete globalI_tr;
	  
	  MPI_Barrier(ifcomm);
	  S = new HypreParMatrix(ifcomm, numLocRows, globalI->Width(), A->GetGlobalNumRows(), SI, SJ, Sdata, I->ColPart(), A->RowPart());
#endif
	  
	  delete[] SI;
	  delete[] SJ;
	  delete[] Sdata;
	}
    }

  MPI_Barrier(ifcomm);
  
  if (iftfull)
    delete[] iftfull;

  if (iftsdt)
    delete[] iftsdt;

  if (ifgtsdt)
    delete[] ifgtsdt;

  if (maxifgtsdt)
    delete[] maxifgtsdt;

  if (ifullsdt)
    delete[] ifullsdt;

  if (globalI)
    delete globalI;

  delete[] allg;
  
  return S;
}

// Row and column offsets are assumed to be the same.
// Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
					       Array2D<double> const& coefficient)
{
  const int numBlocks = offsets.Size() - 1;
  
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> all_num_loc_rows(nprocs);
  std::vector<int> procOffsets(nprocs);
  std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
  std::vector<std::vector<int> > blockProcOffsets(numBlocks);
  std::vector<std::vector<int> > procBlockOffsets(nprocs);
  
  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows.data(), 1, MPI_INT, comm);

  for (int j=0; j<numBlocks; ++j)
    {
      all_block_num_loc_rows[j].resize(nprocs);
      blockProcOffsets[j].resize(nprocs);
      
      const int blockNumRows = offsets[j+1] - offsets[j];
      MPI_Allgather(&blockNumRows, 1, MPI_INT, all_block_num_loc_rows[j].data(), 1, MPI_INT, comm);

      blockProcOffsets[j][0] = 0;
      for (int i=0; i<nprocs-1; ++i)
	blockProcOffsets[j][i+1] = blockProcOffsets[j][i] + all_block_num_loc_rows[j][i];
    }
  
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

      procBlockOffsets[i].resize(numBlocks);
      procBlockOffsets[i][0] = 0;
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
	    }
	  else
	    {
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
	  if (csr_blocks(i, j) != NULL)
	    {
	      const int nrows = csr_blocks(i, j)->num_rows;
	      const double coef = coefficient(i, j);

	      //const bool failure = (nrows != offsets[i+1] - offsets[i]);
	      
	      MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
	      for (int k=0; k<nrows; ++k)
		{
		  const int rowg = offsets[i] + k;  // process-local row
		  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		  const int osk = csr_blocks(i, j)->i[k];
		  
		  for (int l=0; l<nnz_k; ++l)
		    {
		      // Find the column process offset for the block.
		      const int bcol = csr_blocks(i, j)->j[osk + l];
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

		      /*
		      if (rowg == 52 && cnt[rowg] == 15)
			cout << "Setting colg to " << colg << endl;
		      */
		      /*
		      if (rowg == 570)
			cout << rank << ": row 570 col " << colg << " val " << coef * csr_blocks(i, j)->data[osk + l] << endl;
		      */
		      
		      opJ[opI[rowg] + cnt[rowg]] = colg;
		      data[opI[rowg] + cnt[rowg]] = coef * csr_blocks(i, j)->data[osk + l];
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

  HYPRE_Int minJ = opJ[0];
  HYPRE_Int maxJ = opJ[0];
  for (int i=0; i<nnz; ++i)
    {
      minJ = std::min(minJ, opJ[i]);
      maxJ = std::max(maxJ, opJ[i]);

      if (opJ[i] >= glob_ncols)
	cout << "Column indices out of range" << endl;
    }
  
  HypreParMatrix *hmat = new HypreParMatrix(comm, num_loc_rows, glob_nrows, glob_ncols, (int*) opI.data(), (HYPRE_Int*) opJ.data(), (double*) data.data(),
					    (HYPRE_Int*) rowStarts2.data(), (HYPRE_Int*) rowStarts2.data());
  
  return hmat;
}

Operator* CreateStrumpackMatrixFromHypreBlocks(MPI_Comm comm, const Array<int> & offsets, const Array2D<HypreParMatrix*> & blocks,
					       const Array2D<std::vector<int>*> & leftInjection,
					       const Array2D<std::vector<int>*> & rightInjection,
					       const Array2D<double> & coefficient)
{
  const int numBlocks = offsets.Size() - 1;
  
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  int *all_num_loc_rows = new int[nprocs];

  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, comm);

  int first_loc_row = 0;
  int glob_nrows = 0;
  for (int i=0; i<nprocs; ++i)
    {
      glob_nrows += all_num_loc_rows[i];
      if (i < rank)
	first_loc_row += all_num_loc_rows[i];
    }

  delete[] all_num_loc_rows;
    
  const int glob_ncols = glob_nrows;

  int *opI = new int[num_loc_rows+1];
  int *cnt = new int[num_loc_rows];

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
	    }
	  else
	    {
	      csr_blocks(i, j) = GetHypreParMatrixData(*(blocks(i, j)));

	      const int nrows = csr_blocks(i, j)->num_rows;

	      if (leftInjection(i, j) == NULL)
		{
		  if (nrows != offsets[i+1] - offsets[i])
		    cout << "ERROR" << endl;
		      
		  MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
		  for (int k=0; k<nrows; ++k)
		    {
		      if (rank == 0)
			{
			  const int osk = csr_blocks(i, j)->i[k];
			  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
			  for (int l=0; l<nnz_k; ++l)
			    {
			      if (csr_blocks(i, j)->j[osk + l] >= nrows)
				{
				  cout << "(" << i << ", " << j << ") row " << k;
				  cout << ", col " << csr_blocks(i, j)->j[osk + l] << endl;
				}
			    }
			}
			  
		      const int rowg = offsets[i] + k;
		      opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		    }
		}
	      else
		{
		  // The dimension of the range of leftInjection(i, j) is not available here.
		      
		  for (int k=0; k<nrows; ++k)
		    {
		      const int rowg = offsets[i] + k;
		      //(*(leftInjection(i, j)))[k]
		      opI[rowg + 1] += csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		    }
		}
	    }
	}
    }

  // Now opI[i] is nnz for row i-1. Do a partial sum to get offsets.
  for (int i=0; i<num_loc_rows; ++i)
    opI[i+1] += opI[i];

  const int nnz = opI[num_loc_rows];

  int *opJ = new int[nnz];
  double *data = new double[nnz];

  // Loop over all blocks, to set matrix data.
  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  if (csr_blocks(i, j) != NULL)
	    {
	      const int nrows = csr_blocks(i, j)->num_rows;
	      const double coef = coefficient(i, j);
	      
	      MFEM_VERIFY(nrows == offsets[i+1] - offsets[i], "");
	      
	      for (int k=0; k<nrows; ++k)
		{
		  const int rowg = offsets[i] + k;
		  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		  const int osk = csr_blocks(i, j)->i[k];
		  
		  for (int l=0; l<nnz_k; ++l)
		    {
		      const int colg = offsets[j] + csr_blocks(i, j)->j[osk + l];

		      opJ[opI[rowg] + cnt[rowg]] = colg;
		      data[opI[rowg] + cnt[rowg]] = coef * csr_blocks(i, j)->data[osk + l];
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

  delete[] cnt;
  
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
  
  Operator *op = new STRUMPACKRowLocMatrix(comm, num_loc_rows, first_loc_row, glob_nrows, glob_ncols, opI, opJ, data);
  
  delete[] opI;
  delete[] opJ;
  delete[] data;
  
  return op;
}

// Should this be in ParFiniteElementSpace?
void FindBoundaryTrueDOFs(ParFiniteElementSpace *pfespace, set<int>& tdofsBdry)
{
  const ParMesh *pmesh = pfespace->GetParMesh();

  for (int be = 0; be < pmesh->GetNBE(); ++be)
    {
      // const int face = pmesh->GetBdrElementEdgeIndex(be);  // face index of boundary element i
      Array<int> dofs;
      pfespace->GetBdrElementDofs(be, dofs);
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int ldof = pfespace->GetLocalTDofNumber(dof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.
	  if (ldof >= 0)
	    tdofsBdry.insert(ldof);
	}
    }
}

// This function is applicable only to convex faces, as it simply compares the vertices as sets.
bool FacesCoincideGeometrically(ParMesh *volumeMesh, const int face, ParMesh *surfaceMesh, const int elem)
{
  Array<int> faceVert;
  volumeMesh->GetFaceVertices(face, faceVert);

  Array<int> elemVert;
  surfaceMesh->GetElementVertices(elem, elemVert);

  if (faceVert.Size() != elemVert.Size())
    return false;

  for (int i=0; i<faceVert.Size(); ++i)
    {
      double *vi = volumeMesh->GetVertex(faceVert[i]);
      bool vertexFound = false;
      
      for (int j=0; j<faceVert.Size(); ++j)
	{
	  double *vj = surfaceMesh->GetVertex(elemVert[j]);

	  bool verticesEqual = true;
	  for (int k=0; k<3; ++k)
	    {
	      if (fabs(vi[k] - vj[k]) > 1.0e-12)
		verticesEqual = false;
	    }

	  if (verticesEqual)
	    vertexFound = true;
	}

      if (!vertexFound)
	return false;
    }

  return true;
}

void PrintMeshBoundingBox(ParMesh *mesh)
{
  double vmin[3];
  double vmax[3];
  
  for (int i=0; i<mesh->GetNV(); ++i)
    {
      double *v = mesh->GetVertex(i);

      if (i == 0)
	{
	  for (int j=0; j<3; ++j)
	    {
	      vmin[j] = v[j];
	      vmax[j] = v[j];
	    }
	}
      else
	{
	  for (int j=0; j<3; ++j)
	    {
	      vmin[j] = std::min(v[j], vmin[j]);
	      vmax[j] = std::max(v[j], vmax[j]);
	    }
	}
    }

  cout << mesh->GetMyRank() << ": sd box [" << vmin[0] << ", " << vmin[1] << ", " << vmin[2] << "] to [" << vmax[0] << ", " << vmax[1] << ", " << vmax[2] << "]" << endl;
}

// For InterfaceToSurfaceInjection, we need a map from DOF's on the interfaces to the corresponding DOF's on the surfaces of the subdomains.
// These maps could be created efficiently by maintaining maps between subdomain meshes and the original mesh, as well as maps between the
// interface meshes and the original mesh. The ParMesh constructor appears to keep the same ordering of elements, but it reorders the vertices.
// For interface meshes, the elements are faces, which are stored in order by the std::set<int> SubdomainInterface::faces. Therefore, creating
// these maps efficiently seems to require element maps between the original mesh and the subdomain and interface meshes. The
// InterfaceToSurfaceInjection will work by mapping interface faces to the original mesh neighboring elements, followed by mapping those
// elements to subdomain elements, determining which face of each subdomain element is on the interface geometrically, and then determining
// the DOF correspondence on each face geometrically by using GetVertexDofs, GetEdgeDofs, and GetFaceDofs (since the ordering may be different
// on the subdomain faces and interface elements). 

// For subdomain operators A^{**}, the only suboperators that use injection operators are A^{S\rho} and A^{FS}. If A^{SF} were nonzero, it
// would also use injection. The first block is for u on the entire subdomain including the interior and the surface, so injection to the
// S-rows is really injection into the true DOF's of the entire ND subdomain space. The transpose of injection is used for A^{FS}, again from
// the entire ND subdomain space to the interface. 

// For interface operators C^{**}, the S-rows are just the true DOF's of the subdomain ND space on the entire subdomain boundary. Thus we can
// use the same injection operator as for the subdomain operators. However, we must map from those ordered true DOF's to their indices within
// the set, using an std::map<int, int>. 

// The true DOF issue is complicated, because interface operators are defined on interface spaces, which may have DOF's that are not true
// DOF's in the interface space but correspond to true DOF's on the surfaces of the subdomain spaces. In the extreme case, an interface space
// may have zero true DOF's on a process, although the same process may have many true DOF's in the subdomain space on that interface. As a
// result, the subdomain would not receive the contributions from the interface operator, if it acted only on true DOF's. Instead, we must
// inject from full DOF's in the interface spaces to true DOF's in the subdomain spaces. For the transpose of injection, we also need a map
// from full ifespace DOF's to full fespace DOF's. The use of full or true DOF's in the interface and subdomain spaces is handled in
// InjectionOperator via a ParGridFunction. Whether a DOF is true is determined by fespace.GetLocalTDofNumber().

// Therefore, dofmap is defined by SetInterfaceToSurfaceDOFMap() to be of full ifespace DOF size, mapping from full ifespace DOF's to true
// subdomain DOF's in fespace; fdofmap is also of full ifespace DOF size, mapping from full ifespace DOF's to full subdomain DOF's in fespace.

// For full generality of parallel partitioning, we must allow for ifespace and fespace to be NULL (possibly even both), as this function
// needs to be called by all processes in ifsdcomm. There may be processes that touch an interface but not a subdomain, and vice versa.
// Therefore, we cannot limit calls to this function only to processes touching an interface or subdomain.
void SetInterfaceToSurfaceDOFMap(MPI_Comm ifsdcomm, ParFiniteElementSpace *ifespace, ParFiniteElementSpace *fespace, ParFiniteElementSpace *fespaceGlobal,
				 ParMesh *pmesh, const int sdAttribute, const std::set<int>& pmeshFacesInInterface,
				 const std::set<int>& pmeshEdgesInInterface, const std::set<int>& pmeshVerticesInInterface, 
				 const FiniteElementCollection *fec, std::vector<int>& dofmap, //std::vector<int>& fdofmap,
				 std::vector<int>& gdofmap)
{
  const int ifSize = (ifespace == NULL) ? 0 : ifespace->GetVSize();  // Full DOF size
  const int iftSize = (ifespace == NULL) ? 0 : ifespace->GetTrueVSize();  // True DOF size, used for global gdofmap

  const HYPRE_Int iftos = (ifespace == NULL) ? 0 : ifespace->GetMyTDofOffset();
  const HYPRE_Int sdtos = (fespace == NULL) ? 0 : fespace->GetMyTDofOffset();
  
  int ifgSize = 0;

  // TODO: in general, this may need to be MPI_COMM_WORLD, since a process may see the subdomain but not the interface.
  //MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, ifespace->GetComm());
  //MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, ifsdcomm);

  std::vector<int> fdofmap; // TODO: remove
  
  dofmap.assign(ifSize, -1);
  fdofmap.assign(ifSize, -1);
  gdofmap.assign(ifgSize, -1);

  std::vector<int> ifpedge, maxifpedge;
  ifpedge.assign(ifgSize, -1);
  maxifpedge.assign(ifgSize, -1);
  
  const double vertexTol = 1.0e-12;
  
  ParMesh *ifMesh = (ifespace == NULL) ? NULL : ifespace->GetParMesh();  // Interface mesh
  ParMesh *sdMesh = (fespace == NULL) ? NULL : fespace->GetParMesh();  // Subdomain mesh

  //PrintMeshBoundingBox(sdMesh);
  
  // Create map from face indices in pmeshFacesInInterface to pmesh elements containing those faces.
  std::map<int, int> pmeshFaceToElem;
  std::set<int> pmeshElemsByInterface;
  //std::set<int> pmeshEdgesOnInterface;

  for (int elId=0; elId<pmesh->GetNE(); ++elId)
    {
      if (pmesh->GetAttribute(elId) == sdAttribute)
	{
	  Array<int> elFaces, ori;
	  pmesh->GetElementFaces(elId, elFaces, ori);
	  for (int j=0; j<elFaces.Size(); ++j)
	    {
	      std::set<int>::const_iterator it = pmeshFacesInInterface.find(elFaces[j]);
	      if (it != pmeshFacesInInterface.end())
		{
		  std::map<int, int>::iterator itf = pmeshFaceToElem.find(elFaces[j]);
		  if (itf != pmeshFaceToElem.end())
		    cout << "Already found " << elFaces[j] << endl;
		  
		  MFEM_VERIFY(itf == pmeshFaceToElem.end(), "");
		  
		  pmeshFaceToElem[elFaces[j]] = elId;

		  pmeshElemsByInterface.insert(elId);
		}

	      Array<int> edges, eori;
	      pmesh->GetFaceEdges(elFaces[j], edges, eori);

	      for (int k=0; k<edges.Size(); ++k)
		{
		  Array<int> vert;
		  pmesh->GetEdgeVertices(edges[k], vert);

		  MFEM_VERIFY(vert.Size() == 2, "");

		  bool onInterface = true;
		  bool touchingInterface = false;
		  
		  for (int l=0; l<2; ++l)
		    {
		      std::set<int>::const_iterator itv = pmeshVerticesInInterface.find(vert[l]);
			
		      if (itv == pmeshVerticesInInterface.end())
			onInterface = false;
		      else
			touchingInterface = true;
		    }

		  //if (onInterface)
		  // pmeshEdgesOnInterface.insert(edges[k]);

		  if (touchingInterface)
		    pmeshElemsByInterface.insert(elId);		    
		}
	    }
	}
    }

  // Set a map pmeshElemToSDmesh from pmesh element indices to the corresponding sdMesh element indices, only for elements neighboring the interface.
  // Also set a map pmeshToSDMeshIFEdge from pmesh edges in the interface to sdMesh edges, by searching only on sdMesh elements touching the interface.
  std::map<int, int> pmeshElemToSDmesh;
  std::map<int, int> pmeshToSDMeshIFEdge;
  std::map<int, int> pmeshToSDMeshIFVertex;
  std::map<int, int> pmeshToIFMeshIFVertex;

  /*
  { // debugging
	int edges[2];
	edges[0] = 1086;
	edges[1] = 1097;

	for (int j=0; j<2; ++j)
	  {
	    Array<int> sdMeshEdgeVert;
	    sdMesh->GetEdgeVertices(edges[j], sdMeshEdgeVert);

	    for (int k=0; k<2; ++k)
	      {
		cout << "Edge " << edges[j] << " has vertex " << sdMeshEdgeVert[k] << " with crd ";
		for (int l=0; l<3; ++l)
		  cout << sdMesh->GetVertex(sdMeshEdgeVert[k])[l] << " ";

		cout << endl;
	      }
	  }

  }
  */

  const int sdNE = (sdMesh == NULL) ? 0 : sdMesh->GetNE();
  
  for (int elId=0; elId<sdNE; ++elId)
    {
      {
	Array<int> sdEdge, sdOri;
	sdMesh->GetElementEdges(elId, sdEdge, sdOri);

	for (int j=0; j<sdEdge.Size(); ++j)
	  {
	    Array<int> sddofs;
	    fespace->GetEdgeDofs(sdEdge[j], sddofs);

	    /*
	    if (sddofs[0] == 1086 || sddofs[0] == 1097)
	      cout << "sd el " << elId << " has edge dof " << sddofs[0] << endl;
	    */
	  }

      }
      
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int pmeshElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      std::set<int>::const_iterator it = pmeshElemsByInterface.find(pmeshElemId);
      if (it != pmeshElemsByInterface.end())  // if pmeshElemId neighbors the interface
	{
	  pmeshElemToSDmesh[pmeshElemId] = elId;

	  // Set map from pmesh vertices in the interface to sdMesh vertices, by comparing coordinates on the two elements that have been identified.

	  Array<int> pVertex, sdVertex;
	  pmesh->GetElementVertices(pmeshElemId, pVertex);
	  sdMesh->GetElementVertices(elId, sdVertex);

	  MFEM_VERIFY(pVertex.Size() == sdVertex.Size(), "");

	  for (int j=0; j<pVertex.Size(); ++j)
	    {
	      // First check whether vertex is in the interface
	      std::set<int>::const_iterator itpv = pmeshVerticesInInterface.find(pVertex[j]);
	      if (itpv == pmeshVerticesInInterface.end())
		continue;

	      for (int k=0; k<sdVertex.Size(); ++k)
		{
		  bool vertexMatch = true;
		  for (int l=0; l<3; ++l)
		    {
		      if (fabs(pmesh->GetVertex(pVertex[j])[l] - sdMesh->GetVertex(sdVertex[k])[l]) > vertexTol)
			{
			  vertexMatch = false;
			}
		    }

		  if (vertexMatch)
		    {
		      std::map<int, int>::iterator itchk = pmeshToSDMeshIFVertex.find(pVertex[j]);

		      if (itchk != pmeshToSDMeshIFVertex.end())
			{
			  MFEM_VERIFY(itchk->first == pVertex[j] && itchk->second == sdVertex[k], "");
			}
		      else
			pmeshToSDMeshIFVertex[pVertex[j]] = sdVertex[k];
		    }
		}
	    }
	      
	  // Set map from pmesh edges in the interface to sdMesh edges, by comparing edge midpoints on the two elements that have been identified.

	  Array<int> pEdge, sdEdge, pOri, sdOri;
	  pmesh->GetElementEdges(pmeshElemId, pEdge, pOri);
	  sdMesh->GetElementEdges(elId, sdEdge, sdOri);

	  MFEM_VERIFY(pEdge.Size() == sdEdge.Size(), "");

	  for (int j=0; j<pEdge.Size(); ++j)
	    {
	      // First check whether edge is in the interface
	      std::set<int>::const_iterator itpe = pmeshEdgesInInterface.find(pEdge[j]);
	      if (itpe == pmeshEdgesInInterface.end())
		continue;
	      
	      double mp[3];
	      Array<int> pmeshEdgeVert;
	      pmesh->GetEdgeVertices(pEdge[j], pmeshEdgeVert);

	      MFEM_VERIFY(pmeshEdgeVert.Size() == 2, "");

	      for (int l=0; l<3; ++l)
		mp[l] = 0.5 * (pmesh->GetVertex(pmeshEdgeVert[0])[l] + pmesh->GetVertex(pmeshEdgeVert[1])[l]);
					 
	      for (int k=0; k<sdEdge.Size(); ++k)
		{
		  Array<int> sdMeshEdgeVert;
		  sdMesh->GetEdgeVertices(sdEdge[k], sdMeshEdgeVert);

		  MFEM_VERIFY(sdMeshEdgeVert.Size() == 2, "");

		  bool kjmatch = true;
		  
		  for (int l=0; l<3; ++l)
		    {
		      if (fabs(mp[l] - (0.5 * (sdMesh->GetVertex(sdMeshEdgeVert[0])[l] + sdMesh->GetVertex(sdMeshEdgeVert[1])[l]))) > vertexTol)
			kjmatch = false;
		    }

		  if (kjmatch)
		    {
		      std::map<int, int>::iterator itchk = pmeshToSDMeshIFEdge.find(pEdge[j]);

		      if (itchk != pmeshToSDMeshIFEdge.end())
			{
			  MFEM_VERIFY(itchk->first == pEdge[j] && itchk->second == sdEdge[k], "");
			}
		      else
			pmeshToSDMeshIFEdge[pEdge[j]] = sdEdge[k];
		    }
		}
	    }
	}
    }
  
  // The strategy in the following code is to loop over interface faces, identifying each with ifMesh face i and sdMesh face sdMeshFace,
  // and then mapping DOF's in ifMesh face i to DOF's in sdMesh face sdMeshFace. This is insufficient, since a process may own a 
  // portion of sdMesh that intersects the interface only on an edge or at a vertex, in which case no sdMesh DOF's will be found.
  // Although this strategy is insufficent to find all edge and vertex sdMesh DOF's in general, it is necessary in order to map face DOF's. 
  
  // Another strategy is to loop over interface edges, identifying each with an ifMesh edge and a sdMesh edge. In case there are vertex DOF's,
  // it is also necessary to loop over interface vertices, identifying each with an ifMesh vertex and a sdMesh vertex. This necessitates
  // the data pmeshEdgesInInterface and pmeshVerticesInInterface.

  // Therefore, in the following we loop over (1) faces, (2) edges, and (3) vertices in the interface.
  
  // (1) Loop over interface faces.
  std::map<int, int> pmeshEdgeToAnyFaceInInterface;
  std::map<int, int> pmeshFaceToIFFace;
  int i = 0;
  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it, ++i)
    {
      const int pmeshFace = *it;

      pmeshFaceToIFFace[pmeshFace] = i;
      
      // Face pmeshFace of pmesh coincides with face i of ifMesh on this process (the same face may also exist on a different process in the same ifMesh,
      // as there can be redundant overlapping faces in parallel, for communication).

      { // For each pmesh edge of this face, map to pmeshFace.
	Array<int> edges, ori;
	pmesh->GetFaceEdges(pmeshFace, edges, ori);

	for (int j=0; j<edges.Size(); ++j)
	  pmeshEdgeToAnyFaceInInterface[edges[j]] = pmeshFace;
      }
      
      // Find the neighboring pmesh element.
      std::map<int, int>::iterator ite = pmeshFaceToElem.find(pmeshFace);

      //MFEM_VERIFY(ite != pmeshFaceToElem.end(), "");

      if (ite == pmeshFaceToElem.end())  // This process does not have an element in this subdomain neighboring the face.
	continue;
      
      MFEM_VERIFY(ite->first == pmeshFace, "");

      const int pmeshElem = ite->second;

      // Find the neighboring sdMesh element, which coincides with pmeshElem in pmesh.
      std::map<int, int>::const_iterator itse = pmeshElemToSDmesh.find(pmeshElem);
      MFEM_VERIFY(itse != pmeshElemToSDmesh.end(), "");
      MFEM_VERIFY(itse->first == pmeshElem, "");

      const int sdMeshElem = itse->second;

      // Find the face of element sdMeshElem in sdMesh that coincides geometrically with the current interface face.
      Array<int> elFaces, ori;

      sdMesh->GetElementFaces(sdMeshElem, elFaces, ori);
      int sdMeshFace = -1;
      for (int j=0; j<elFaces.Size(); ++j)
	{
	  if (FacesCoincideGeometrically(sdMesh, elFaces[j], ifMesh, i))
	    sdMeshFace = elFaces[j];
	}

      MFEM_VERIFY(sdMeshFace >= 0, "");

      // Map vertex DOF's on ifMesh face i to vertex DOF's on sdMesh face sdMeshFace.
      // TODO: is this necessary, since FiniteElementSpace::GetEdgeDofs claims to return vertex DOF's as well?
      const int nv = fec->DofForGeometry(Geometry::POINT);
      if (nv > 0)
	{
	  Array<int> ifVert, sdVert;
	  ifMesh->GetFaceVertices(i, ifVert);
	  sdMesh->GetFaceVertices(sdMeshFace, sdVert);

	  MFEM_VERIFY(ifVert.Size() == sdVert.Size(), "");
	  
	  for (int j=0; j<ifVert.Size(); ++j)
	    {
	      double *ifv = ifMesh->GetVertex(ifVert[j]);

	      bool vertexFound = false;
	      
	      for (int k=0; k<sdVert.Size(); ++k)
		{
		  double *sdv = sdMesh->GetVertex(sdVert[k]);

		  bool verticesEqual = true;
		  for (int l=0; l<3; ++l)
		    {
		      if (fabs(ifv[l] - sdv[l]) > vertexTol)
			verticesEqual = false;
		    }

		  if (verticesEqual)
		    {
		      vertexFound = true;
		      Array<int> ifdofs, sddofs;
		      ifespace->GetVertexDofs(ifVert[j], ifdofs);
		      fespace->GetVertexDofs(sdVert[k], sddofs);

		      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
		      for (int d=0; d<ifdofs.Size(); ++d)
			{
			  MFEM_VERIFY(sddofs[d] >= 0, "");
			  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);

			  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
			  fdofmap[ifdofs[d]] = sddofs[d];
			  
			  if (sdtdof >= 0)  // if this is a true DOF of fespace.
			    {
			      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
			      dofmap[ifdofs[d]] = sdtdof;
			    }
			}
		    }
		}
	      
	      MFEM_VERIFY(vertexFound, "");
	    }
	}
      
      // Map edge DOF's on ifMesh face i to edge DOF's on sdMesh face sdMeshFace.
      const int ne = fec->DofForGeometry(Geometry::SEGMENT);
      if (ne > 0)
	{
	  // TODO: could there be multiple DOF's on an edge with different orderings (depending on orientation) in ifespace and fespace?
	  // TODO: Check orientation for ND_HexahedronElement? Does ND_TetrahedronElement have orientation?

	  Array<int> ifEdge, sdEdge, ifOri, sdOri;
	  ifMesh->GetElementEdges(i, ifEdge, ifOri);
	  sdMesh->GetFaceEdges(sdMeshFace, sdEdge, sdOri);

	  MFEM_VERIFY(ifEdge.Size() == sdEdge.Size(), "");
	  
	  for (int j=0; j<ifEdge.Size(); ++j)
	    {
	      Array<int> ifVert;
	      ifMesh->GetEdgeVertices(ifEdge[j], ifVert);

	      MFEM_VERIFY(ifVert.Size() == 2, "");

	      int sd_k = -1;
	      
	      for (int k=0; k<sdEdge.Size(); ++k)
		{
		  Array<int> sdVert;
		  sdMesh->GetEdgeVertices(sdEdge[k], sdVert);

		  MFEM_VERIFY(sdVert.Size() == 2, "");

		  bool edgesMatch = true;
		  for (int v=0; v<2; ++v)
		    {
		      double *ifv = ifMesh->GetVertex(ifVert[v]);
		      double *sdv = sdMesh->GetVertex(sdVert[v]);

		      bool verticesEqual = true;
		      for (int l=0; l<3; ++l)
			{
			  if (fabs(ifv[l] - sdv[l]) > vertexTol)
			    verticesEqual = false;
			}

		      if (!verticesEqual)
			edgesMatch = false;
		    }

		  if (edgesMatch)
		    {
		      MFEM_VERIFY(sd_k == -1, "");
		      sd_k = k;
		    }
		}

	      MFEM_VERIFY(sd_k >= 0, "");

	      Array<int> ifdofs, sddofs;
	      ifespace->GetEdgeDofs(ifEdge[j], ifdofs);
	      fespace->GetEdgeDofs(sdEdge[sd_k], sddofs);

	      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	      for (int d=0; d<ifdofs.Size(); ++d)
		{
		  MFEM_VERIFY(sddofs[d] >= 0 && ifdofs[d] >= 0, "");
		  
		  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);

		  /*
		  if (ifdofs[d] == 276)
		    cout << "276 found on " << endl;
		  */

		  const int ifdof_d = ifdofs[d];
		  
		  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
		  fdofmap[ifdofs[d]] = sddofs[d];
		  
		  if (sdtdof >= 0)  // if this is a true DOF of fespace.
		    {
		      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		      dofmap[ifdofs[d]] = sdtdof;
		    }
		}
	    }
	}

      // Map face DOF's on ifMesh face i to face DOF's on sdMesh face sdMeshFace.
      const int nf = fec->DofForGeometry(sdMesh->GetFaceGeometryType(0));
      if (nf > 0)
	{
	  //MFEM_VERIFY(false, "TODO: it may be necessary to set face DOF's in gdofmap.");
	  
	  Array<int> ifdofs, sddofs;
	  ifespace->GetElementDofs(i, ifdofs);
	  fespace->GetFaceDofs(sdMeshFace, sddofs);

	  MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	  for (int d=0; d<ifdofs.Size(); ++d)
	    {
	      const int sddof_d = sddofs[d] >= 0 ? sddofs[d] : -1 - sddofs[d];
	      const int ifdof_d = ifdofs[d] >= 0 ? ifdofs[d] : -1 - ifdofs[d];
	      
	      //const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
	      const int sdtdof = fespace->GetLocalTDofNumber(sddof_d);

	      /*
	      MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
	      fdofmap[ifdofs[d]] = sddofs[d];
			  
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		  dofmap[ifdofs[d]] = sdtdof;
		}
	      */

	      MFEM_VERIFY(fdofmap[ifdof_d] == sddof_d || fdofmap[ifdof_d] == -1, "");
	      fdofmap[ifdof_d] = sddof_d;
			  
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdof_d] == sdtdof || dofmap[ifdof_d] == -1, "");
		  dofmap[ifdof_d] = sdtdof;
		}
	    }
	}
    }

  // (2) Loop over interface edges.

  // (2a) Loop over interface edges, only setting helper maps locally and globally (via some communication).
  std::vector<int> pToIFedge, pedgeToIFGDOF;
  pToIFedge.assign(pmeshEdgesInInterface.size(), -1);
  pedgeToIFGDOF.assign(pmeshEdgesInInterface.size(), -1);

  std::map<HYPRE_Int, int> pgdofToSet;
  
  i = 0;
  for (std::set<int>::const_iterator it = pmeshEdgesInInterface.begin(); it != pmeshEdgesInInterface.end(); ++it, ++i)
    {
      const int pmeshEdge = *it;

      { // Set pgdofToSet, even if this edge is not in ifMesh
	Array<int> pdofs;
	fespaceGlobal->GetEdgeDofs(pmeshEdge, pdofs);

	const HYPRE_Int gtdof = fespaceGlobal->GetGlobalTDofNumber(pdofs[0]);  // Just use the first DOF on the edge, as a global identification for the edge.
	pgdofToSet[gtdof] = i;
      }
      
      // Find the edge of ifMesh that coincides with pmesh edge pmeshEdge.
      int ifMeshEdge = -1;

      {
	std::map<int, int>::iterator ite = pmeshEdgeToAnyFaceInInterface.find(pmeshEdge);

	if (ite == pmeshEdgeToAnyFaceInInterface.end())
	  {
	    // If the edge is not part of a local face in the interface, then it should have no local DOF's in ifespace.
	    // We will not set dofmap for any ifespace local DOF's, so we just continue to the next edge.
	    continue;  
	  }
	
	MFEM_VERIFY(ite != pmeshEdgeToAnyFaceInInterface.end(), "");
	MFEM_VERIFY(ite->first == pmeshEdge, "");

	const int pmeshFace = ite->second;

	std::map<int, int>::iterator itf = pmeshFaceToIFFace.find(pmeshFace);
	MFEM_VERIFY(itf != pmeshFaceToIFFace.end(), "");
	MFEM_VERIFY(itf->first == pmeshFace, "");
	
	const int ifMeshFace = itf->second;

	// Loop over edges of ifMeshFace to find a geometric match.
	Array<int> ifEdge, ifOri;
	ifMesh->GetElementEdges(ifMeshFace, ifEdge, ifOri);

	Array<int> pmeshEdgeVert;
	pmesh->GetEdgeVertices(pmeshEdge, pmeshEdgeVert);

	MFEM_VERIFY(pmeshEdgeVert.Size() == 2, "");
	
	double pmeshEdgeMidpoint[3];

	for (int k=0; k<3; ++k)
	  pmeshEdgeMidpoint[k] = 0.5 * (pmesh->GetVertex(pmeshEdgeVert[0])[k] + pmesh->GetVertex(pmeshEdgeVert[1])[k]);

	for (int j=0; j<ifEdge.Size(); ++j)
	  {
	    Array<int> ifmeshEdgeVert;
	    ifMesh->GetEdgeVertices(ifEdge[j], ifmeshEdgeVert);

	    MFEM_VERIFY(ifmeshEdgeVert.Size() == 2, "");

	    bool pointsEqual = true;
	    for (int k=0; k<3; ++k)
	      {
		const double ifMeshEdgeMidpoint_k = 0.5 * (ifMesh->GetVertex(ifmeshEdgeVert[0])[k] + ifMesh->GetVertex(ifmeshEdgeVert[1])[k]);
		if (fabs(ifMeshEdgeMidpoint_k - pmeshEdgeMidpoint[k]) > vertexTol)
		  pointsEqual = false;
	      }

	    if (pointsEqual)
	      {
		MFEM_VERIFY(ifMeshEdge == -1, "");
		ifMeshEdge = ifEdge[j];

		// Set the map from pmesh vertices to ifMesh vertices
		for (int k=0; k<2; ++k)
		  {
		    bool matchFound = false;
		    for (int m=0; m<2; ++m)
		      {
			bool vertexMatch = true;
			for (int l=0; l<3; ++l)
			  {
			    if (fabs(pmesh->GetVertex(pmeshEdgeVert[k])[l] - ifMesh->GetVertex(ifmeshEdgeVert[m])[l]) > vertexTol)
			      vertexMatch = false;
			  }

			if (vertexMatch)
			  {
			    std::map<int, int>::iterator itv = pmeshToIFMeshIFVertex.find(pmeshEdgeVert[k]);

			    if (itv == pmeshToIFMeshIFVertex.end())
			      pmeshToIFMeshIFVertex[pmeshEdgeVert[k]] = ifmeshEdgeVert[m];
			    else
			      {
				MFEM_VERIFY(itv->first == pmeshEdgeVert[k] && itv->second == ifmeshEdgeVert[m], "");
			      }
			    
			    matchFound = true;
			  }
		      }

		    MFEM_VERIFY(matchFound, "");
		  }
	      }
	  }

	MFEM_VERIFY(ifMeshEdge >= 0, "");
	pToIFedge[i] = ifMeshEdge;
	
	// Even if sdMeshEdge is not defined on this process, we set the map from pmesh edges in the interface to ifMesh edges, to be gathered by all processes.
	{
	  Array<int> pdofs, ifdofs;
	  fespaceGlobal->GetEdgeDofs(pmeshEdge, pdofs);
	  ifespace->GetEdgeDofs(ifMeshEdge, ifdofs);

	  const HYPRE_Int gtdof = fespaceGlobal->GetGlobalTDofNumber(pdofs[0]);  // Just use the first DOF on the edge, as a global identification for the edge.
	  const HYPRE_Int ifgtdof = ifespace->GetGlobalTDofNumber(ifdofs[0]);  // Just use the first DOF on the edge, as a global identification for the edge.

	  { // verification of a property assumed in the following code
	    bool continuousEdgeDOFs = true;
	    HYPRE_Int ifgtdofprev = ifgtdof;
	    for (int l=1; l<ifdofs.Size(); ++l)
	      {
		const HYPRE_Int ifgtdof_l = ifespace->GetGlobalTDofNumber(ifdofs[l]);
		if (ifgtdof_l != ifgtdofprev + 1)
		  continuousEdgeDOFs = false;
		
		ifgtdofprev = ifgtdof_l;
		
		if (ifdofs[l] != ifdofs[l-1]+1)
		  continuousEdgeDOFs = false;
	      }
	    
	    MFEM_VERIFY(continuousEdgeDOFs, "continuousEdgeDOFs");
	  }
	  
	  ifpedge[ifgtdof] = gtdof;
	}
      }
    }

  //MPI_Allreduce((int*) ifpedge.data(), (int*) maxifpedge.data(), ifgSize, MPI_INT, MPI_MAX, ifespace->GetComm());
  MPI_Allreduce((int*) ifpedge.data(), (int*) maxifpedge.data(), ifgSize, MPI_INT, MPI_MAX, ifsdcomm);

  for (i=0; i<ifgSize; ++i)
    {
      const HYPRE_Int pgtdof = maxifpedge[i]; // fespaceGlobal->GetGlobalTDofNumber() for the first DOF on its edge.
      if (pgtdof >= 0)
	{
	  // Check whether pgtdof is on one of the pmesh edges in the interface on this process.
	  std::map<HYPRE_Int, int>::iterator it = pgdofToSet.find(pgtdof);
	  if (it != pgdofToSet.end())
	    {
	      MFEM_VERIFY(it->first == pgtdof, "");
	      const int pmei = it->second;  // index in pmeshEdgesInInterface
	      MFEM_VERIFY(pedgeToIFGDOF[pmei] == -1 || pedgeToIFGDOF[pmei] == i, "");
	      pedgeToIFGDOF[pmei] = i;
	    }
	}
    }
  
  // (2b) Loop over interface edges, to set dofmap.

  i = 0;
  for (std::set<int>::const_iterator it = pmeshEdgesInInterface.begin(); it != pmeshEdgesInInterface.end(); ++it, ++i)
    {
      const int pmeshEdge = *it;

      MFEM_VERIFY(pedgeToIFGDOF[i] >= 0, "");

      std::map<int, int>::iterator itsde = pmeshToSDMeshIFEdge.find(pmeshEdge);

      if (itsde == pmeshToSDMeshIFEdge.end())
	{
	  continue;  // there are no SD DOF's to which to map
	}
      
      MFEM_VERIFY(itsde != pmeshToSDMeshIFEdge.end(), "");
      MFEM_VERIFY(itsde->first == pmeshEdge, "");
      
      const int sdMeshEdge = itsde->second;
      
      const int ifMeshEdge = pToIFedge[i];  // The local edge of ifMesh that coincides with pmesh edge pmeshEdge.

      // Map vertex DOF's on ifMesh edge ifMeshEdge to vertex DOF's on sdMesh edge sdMeshEdge.
      // TODO: is this necessary, since FiniteElementSpace::GetEdgeDofs claims to return vertex DOF's as well?
      const int nv = fec->DofForGeometry(Geometry::POINT);
      if (nv > 0 && ifMeshEdge < 0)
	{
	  MFEM_VERIFY(false, "TODO");
	}
      else if (nv > 0)
	{
	  Array<int> ifVert, sdVert;
	  ifMesh->GetEdgeVertices(i, ifVert);
	  sdMesh->GetEdgeVertices(sdMeshEdge, sdVert);

	  MFEM_VERIFY(ifVert.Size() == sdVert.Size(), "");
	  
	  for (int j=0; j<ifVert.Size(); ++j)
	    {
	      double *ifv = ifMesh->GetVertex(ifVert[j]);

	      bool vertexFound = false;
	      
	      for (int k=0; k<sdVert.Size(); ++k)
		{
		  double *sdv = sdMesh->GetVertex(sdVert[k]);

		  bool verticesEqual = true;
		  for (int l=0; l<3; ++l)
		    {
		      if (fabs(ifv[l] - sdv[l]) > vertexTol)
			verticesEqual = false;
		    }

		  if (verticesEqual)
		    {
		      vertexFound = true;
		      Array<int> ifdofs, sddofs;
		      ifespace->GetVertexDofs(ifVert[j], ifdofs);
		      fespace->GetVertexDofs(sdVert[k], sddofs);

		      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
		      for (int d=0; d<ifdofs.Size(); ++d)
			{
			  MFEM_VERIFY(sddofs[d] >= 0 && ifdofs[d] >= 0, "");
			  
			  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);

			  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
			  fdofmap[ifdofs[d]] = sddofs[d];
			  
			  if (sdtdof >= 0)  // if this is a true DOF of fespace.
			    {
			      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
			      dofmap[ifdofs[d]] = sdtdof;
			    }
			}
		    }
		}
	      
	      MFEM_VERIFY(vertexFound, "");
	    }
	}
      
      // Map edge DOF's on ifMesh edge ifMeshEdge to edge DOF's on sdMesh edge sdMeshEdge.
      const int ne = fec->DofForGeometry(Geometry::SEGMENT);
      if (ne > 0)
	{
	  // TODO: could there be multiple DOF's on an edge with different orderings (depending on orientation) in ifespace and fespace?
	  // TODO: Check orientation for ND_HexahedronElement? Does ND_TetrahedronElement have orientation?

	  Array<int> ifdofs, sddofs;
	  fespace->GetEdgeDofs(sdMeshEdge, sddofs);

	  if (ifMeshEdge >= 0)  // if there are local ifespace DOF's to map from in dofmap
	    {
	      ifespace->GetEdgeDofs(ifMeshEdge, ifdofs);
	      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	      MFEM_VERIFY(ifdofs[0] >= 0, "");
	      
	      const int ltdof = ifespace->GetLocalTDofNumber(ifdofs[0]);
	      if (ltdof >= 0)
		{
		  const bool check = (pedgeToIFGDOF[i] == iftos + ltdof);

		  if (!check)
		    cout << "Failure on " << ltdof << endl;
	      
		  MFEM_VERIFY(pedgeToIFGDOF[i] == iftos + ltdof, "");
		}
	    }
	  
	  for (int d=0; d<sddofs.Size(); ++d)
	    {
	      MFEM_VERIFY(sddofs[d] >= 0, "");

	      const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);

	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  if (ifMeshEdge >= 0)
		    { // Set local dofmap
		      MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		      dofmap[ifdofs[d]] = sdtdof;
		    }
		  else
		    { // Set global gdofmap
		      gdofmap[pedgeToIFGDOF[i] + d] = sdtdof + sdtos;
		    }
		}

	      if (ifMeshEdge >= 0)
		{ // Set local fdofmap
		  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
		  fdofmap[ifdofs[d]] = sddofs[d];
		}
	    }
	}
    }

  MPI_Barrier(ifsdcomm);
  
  // (3) Loop over interface vertices.
  for (std::set<int>::const_iterator it = pmeshVerticesInInterface.begin(); it != pmeshVerticesInInterface.end(); ++it)
    {
      const int pmeshVertex = *it;

      // Find the vertex of ifMesh that coincides with pmesh vertex pmeshVertex.

      std::map<int, int>::iterator itsdv = pmeshToIFMeshIFVertex.find(pmeshVertex);

      if (itsdv == pmeshToIFMeshIFVertex.end())
	continue;
      
      MFEM_VERIFY(itsdv != pmeshToIFMeshIFVertex.end(), "");
      MFEM_VERIFY(itsdv->first == pmeshVertex, "");

      const int ifMeshVertex = itsdv->second;
      
      MFEM_VERIFY(ifMeshVertex >= 0, "");

      itsdv = pmeshToSDMeshIFVertex.find(pmeshVertex);

      if (itsdv == pmeshToSDMeshIFVertex.end())
	continue;  // there are no SD DOF's to map to
      
      MFEM_VERIFY(itsdv != pmeshToSDMeshIFVertex.end(), "");
      MFEM_VERIFY(itsdv->first == pmeshVertex, "");
      
      const int sdMeshVertex = itsdv->second;
      
      // Map vertex DOF's in ifMesh space to vertex DOF's in sdMesh space.
      // TODO: is this necessary, since FiniteElementSpace::GetEdgeDofs claims to return vertex DOF's as well?
      const int nv = fec->DofForGeometry(Geometry::POINT);
      if (nv > 0)
	{
	  Array<int> ifdofs, sddofs;
	  ifespace->GetVertexDofs(ifMeshVertex, ifdofs);
	  fespace->GetVertexDofs(sdMeshVertex, sddofs);

	  MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	  for (int d=0; d<ifdofs.Size(); ++d)
	    {
	      MFEM_VERIFY(sddofs[d] >= 0 && ifdofs[d] >= 0, "");
	      
	      const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);

	      MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
	      fdofmap[ifdofs[d]] = sddofs[d];
	      
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		  dofmap[ifdofs[d]] = sdtdof;
		}
	    }
	}
    }

  // Note that some entries of dofmap may be undefined, if the corresponding subdomain DOF's in fespace are not true DOF's.
  // In contrast, all entries of gdofmap should be defined.
  
  { // Reduce gdofmap, using ifpedge as an auxiliary array.
    // First, copy local values in dofmap to gdofmap
    for (i=0; i<ifSize; ++i)
      {
	/*
	const int tdof = ifespace->GetLocalTDofNumber(i);
	if (tdof >= 0 && dofmap[i] >= 0)
	  {
	    MFEM_VERIFY(gdofmap[iftos + tdof] == -1 || gdofmap[iftos + tdof] == dofmap[i] + sdtos, "");
	    gdofmap[iftos + tdof] = dofmap[i] + sdtos;
	  }
	*/

	//MFEM_VERIFY(fdofmap[i] >= 0, "undefined dofmap");
	
	if (dofmap[i] >= 0)
	  {
	    const int gtdof = ifespace->GetGlobalTDofNumber(i);
	    MFEM_VERIFY(gdofmap[gtdof] == -1 || gdofmap[gtdof] == dofmap[i] + sdtos, "");
	    gdofmap[gtdof] = dofmap[i] + sdtos;
	  }
      }
    
    ifpedge = gdofmap;
    //MPI_Allreduce((int*) ifpedge.data(), (int*) gdofmap.data(), ifgSize, MPI_INT, MPI_MAX, ifespace->GetComm());
    MPI_Allreduce((int*) ifpedge.data(), (int*) gdofmap.data(), ifgSize, MPI_INT, MPI_MAX, ifsdcomm);
      
    for (i=0; i<ifgSize; ++i)
      {
	if (gdofmap[i] < 0)
	  MFEM_VERIFY(gdofmap[i] >= 0, "");
      }
  }

  /*
  bool mapDefined = true;
  for (i=0; i<ifSize; ++i)
    {
      if (dofmap[i] < 0)
	mapDefined = false;
    }
  
  MFEM_VERIFY(mapDefined, "");
  */
}

void SetDomainDofsFromSubdomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & ssd, Vector & s)
{
  MFEM_VERIFY(ssd.Size() == fespaceSD->GetTrueVSize(), "");
  MFEM_VERIFY(s.Size() == fespaceDomain->GetTrueVSize(), "");

  ParMesh *sdMesh = fespaceSD->GetParMesh();  // Subdomain mesh

  ParGridFunction ssd_gf(fespaceSD);
  ssd_gf.SetFromTrueDofs(ssd);

  MFEM_VERIFY(ssd_gf.Size() == fespaceSD->GetVSize(), "");

  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int domainElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      
      Array<int> sddofs;
      Array<int> dofs;
      fespaceDomain->GetElementDofs(domainElemId, dofs);
      fespaceSD->GetElementDofs(elId, sddofs);

      MFEM_VERIFY(dofs.Size() == sddofs.Size(), "");
      
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int sddof_i = sddofs[i] >= 0 ? sddofs[i] : -1 - sddofs[i];
	  const int ldof = fespaceDomain->GetLocalTDofNumber(dof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.

	  if (ldof >= 0)
	    {
	      s[ldof] = ssd_gf[sddof_i];
	    }
	}
    }
}

void SetSubdomainDofsFromDomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & s, Vector & ssd)
{
  MFEM_VERIFY(ssd.Size() == fespaceSD->GetTrueVSize(), "");
  MFEM_VERIFY(s.Size() == fespaceDomain->GetTrueVSize(), "");

  ParMesh *sdMesh = fespaceSD->GetParMesh();  // Subdomain mesh

  ParGridFunction s_gf(fespaceDomain);
  s_gf.SetFromTrueDofs(s);

  MFEM_VERIFY(s_gf.Size() == fespaceDomain->GetVSize(), "");

  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int domainElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      
      Array<int> sddofs;
      Array<int> dofs;
      fespaceDomain->GetElementDofs(domainElemId, dofs);
      fespaceSD->GetElementDofs(elId, sddofs);

      MFEM_VERIFY(dofs.Size() == sddofs.Size(), "");
      
      for (int i=0; i<dofs.Size(); ++i)
	{
	  const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
	  const int sddof_i = sddofs[i] >= 0 ? sddofs[i] : -1 - sddofs[i];
	  const int lsddof = fespaceSD->GetLocalTDofNumber(sddof_i);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.

	  if (lsddof >= 0)
	    {
	      const double flip = ((dofs[i] >= 0 && sddofs[i] < 0) || (dofs[i] < 0 && sddofs[i] >= 0)) ? -1.0 : 1.0;
	      ssd[lsddof] = s_gf[dof_i] * flip;
	    }
	}
    }
}

DDMInterfaceOperator::DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_,
					   ParFiniteElementSpace *fespaceGlobal, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
					   const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
					   std::vector<int> *interfaceLocalIndex_, const double k2_) :
  numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), pmeshSD(pmeshSD_), pmeshIF(pmeshIF_), fec(orderND, spaceDim),
  fecbdry(orderND, spaceDim-1), fecbdryH1(orderND, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
  subdomainLocalInterfaces(numSubdomains_), pmeshGlobal(pmesh_),
  k2(k2_), realPart(true),
  alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
{
  int num_procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  m_rank = rank;
    
  std::cout << "DDM using k2 " << k2 << std::endl;

  MFEM_VERIFY(numSubdomains > 0, "");
  MFEM_VERIFY(interfaceLocalIndex->size() == numInterfaces, "");

  SetParameters();
    
  preconditionerMode = false;

#ifdef DDMCOMPLEX
  AsdRe = new Operator*[numSubdomains];
  AsdIm = new Operator*[numSubdomains];
  AsdP = new Operator*[numSubdomains];
  AsdComplex = new BlockOperator*[numSubdomains];
  precAsdComplex = new BlockDiagonalPreconditioner*[numSubdomains];
  invAsdComplex = new Operator*[numSubdomains];
  injComplexSD = new BlockOperator*[numSubdomains];
#ifdef SPARSE_ASDCOMPLEX
  SpAsdComplex = new SparseMatrix*[numSubdomains];
  HypreAsdComplex = new HypreParMatrix*[numSubdomains];
  SpAsdComplexRowSizes = new HYPRE_Int*[numSubdomains];
#ifdef HYPRE_PARALLEL_ASDCOMPLEX    
  AsdRe_HypreBlocks.resize(numSubdomains);
  AsdIm_HypreBlocks.resize(numSubdomains);
  AsdRe_HypreBlockCoef.resize(numSubdomains);
  AsdIm_HypreBlockCoef.resize(numSubdomains);
  sd_com.resize(numSubdomains);
  sd_nonempty.resize(numSubdomains);
#endif
#endif
#else
  alphaIm = 0.0;
  betaIm = 0.0;
  gammaIm = 0.0;
    
  alphaRe = alpha;
  betaRe = beta;
  gammaRe = gamma;
    
  SetToRealParameters();
    
  Asd = new Operator*[numSubdomains];
#endif
    
  fespace = new ParFiniteElementSpace*[numSubdomains];
  ASPsd = new Operator*[numSubdomains];
  invAsd = new Operator*[numSubdomains];
  injSD = new BlockOperator*[numSubdomains];
  precAsd = new Solver*[numSubdomains];
  sdND = new HypreParMatrix*[numSubdomains];
  sdNDcopy = new HypreParMatrix*[numSubdomains];
  A_SS = new HypreParMatrix*[numSubdomains];
  sdNDinv = new Operator*[numSubdomains];
  sdNDPen = new HypreParMatrix*[numSubdomains];
  sdNDPlusPen = new HypreParMatrix*[numSubdomains];
  sdNDPenSp = new SparseMatrix*[numSubdomains];
  bf_sdND = new ParBilinearForm*[numSubdomains];
  ySD = new Vector*[numSubdomains];

  ifespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;
  iH1fespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;

  ifNDmass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifH1mass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifNDmassInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
  ifH1massInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
  ifNDcurlcurl = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifNDH1grad = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifNDH1gradT = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifND_FS = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  
  numLocalInterfaces = localInterfaces->size();
  globalInterfaceIndex.resize(numLocalInterfaces);
  globalInterfaceIndex.assign(numLocalInterfaces, -1);

  // QUESTION: is numLocalInterfaces == numInterfaces in general?
  MFEM_VERIFY(numLocalInterfaces == numInterfaces, "");

  ifNDtrue.assign(numInterfaces, 0);
  ifH1true.assign(numInterfaces, 0);
  
  for (int i=0; i<numInterfaces; ++i)
    {
      if (pmeshIF[i] == NULL)
	{
	  ifespace[i] = NULL;
	  iH1fespace[i] = NULL;

	  ifNDmass[i] = NULL;
	  ifNDcurlcurl[i] = NULL;
	  ifNDH1grad[i] = NULL;
	  ifNDH1gradT[i] = NULL;
	  ifH1mass[i] = NULL;
	  ifND_FS[i] = NULL;
	}
      else
	{
	  ifespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdry);  // Nedelec space for f_{m,j} when interface i is the j-th interface of subdomain m. 
	  iH1fespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdryH1);  // H^1 space \rho_{m,j} when interface i is the j-th interface of subdomain m.

	  ifNDtrue[i] = ifespace[i]->GetTrueVSize();
	  ifH1true[i] = iH1fespace[i]->GetTrueVSize();

	  cout << rank << ": Interface " << i << " number of bdry attributes: " << pmeshIF[i]->bdr_attributes.Size()
	       << ", NE " << pmeshIF[i]->GetNE() << ", NBE " << pmeshIF[i]->GetNBE() << endl;

	  /*
	    for (int j=0; j<pmeshIF[i]->GetNBE(); ++j)
	    cout << rank << " Interface " << i << ", be " << j << ", bdrAttribute " << pmeshIF[i]->GetBdrAttribute(j) << endl;
	  */
	    
	  CreateInterfaceMatrices(i);
	}

      const int ifli = (*interfaceLocalIndex)[i];

      //MFEM_VERIFY((ifli >= 0) == (pmeshIF[i] != NULL), "");
	
      if (ifli >= 0)
	{
	  subdomainLocalInterfaces[(*localInterfaces)[ifli].FirstSubdomain()].push_back(i);
	  subdomainLocalInterfaces[(*localInterfaces)[ifli].SecondSubdomain()].push_back(i);

	  MFEM_VERIFY(globalInterfaceIndex[ifli] == i || globalInterfaceIndex[ifli] == -1, "");

	  globalInterfaceIndex[ifli] = i;
	}
    }
    
  // For each subdomain parallel finite element space, determine all the true DOF's on the entire boundary. Also for each interface parallel finite element space, determine the number of true DOF's. Note that a true DOF on the boundary of a subdomain may coincide with an interface DOF that is not necessarily a true DOF on the corresponding interface mesh. The size of DDMInterfaceOperator will be the sum of the numbers of true DOF's on the subdomain mesh boundaries and interfaces.
    
  block_trueOffsets.SetSize(numSubdomains + 1); // number of blocks + 1
  block_trueOffsets = 0;

  int size = 0;

  InterfaceToSurfaceInjection.resize(numSubdomains);
  InterfaceToSurfaceInjectionData.resize(numSubdomains);
  //InterfaceToSurfaceInjectionFullData.resize(numSubdomains);
  InterfaceToSurfaceInjectionGlobalData.resize(numSubdomains);
    
  for (int m=0; m<numSubdomains; ++m)
    {
      InterfaceToSurfaceInjection[m].resize(subdomainLocalInterfaces[m].size());
      InterfaceToSurfaceInjectionData[m].resize(subdomainLocalInterfaces[m].size());
      //InterfaceToSurfaceInjectionFullData[m].resize(subdomainLocalInterfaces[m].size());
      InterfaceToSurfaceInjectionGlobalData[m].resize(subdomainLocalInterfaces[m].size());

      cout << rank << ": setup for subdomain " << m << endl;
	  
      if (pmeshSD[m] == NULL)
	{
	  fespace[m] = NULL;
	}
      else
	{
	  fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m
	    
	  /*
	    { // debugging
	    for (int i=0; i<fespace[m]->GetVSize(); ++i)
	    {
	    const HYPRE_Int gtdof = fespace[m]->GetGlobalTDofNumber(i);
      
	    if (m == 1 && gtdof == 188)
	    cout << "m 1 " << gtdof << endl;
		  
	    const int ltdof = fespace[m]->GetLocalTDofNumber(i);
	    if (ltdof >= 0)
	    {
	    if (m == 1 && ltdof == 188)
	    cout << "rank 1 " << gtdof << endl;
	    }
	    }


	    if (m == 1)
	    {
	    for (int i=0; i<pmeshSD[m]->GetNE(); ++i)
	    {
	    Array<int> dofs;
	    fespace[m]->GetElementDofs(i, dofs);
	    bool has1597 = false;
	    bool has188 = false;
		  
	    for (int j=0; j<dofs.Size(); ++j)
	    {
	    const int dofj = dofs[j] >= 0 ? dofs[j] : -1 - dofs[j];
	    const HYPRE_Int gtdof = fespace[m]->GetGlobalTDofNumber(dofj);

	    if (gtdof == 1597)
	    has1597 = true;

	    if (gtdof == 188)
	    has188 = true;
	    }

	    if (has188 && has1597)
	    {
	    cout << "Element " << i << " has both" << endl;
	    }
	    }
	    }
	    }
	  */

	  if (m == 0)
	    cout << rank << ": sd 0 ND space true size " << fespace[m]->GetTrueVSize() << ", full size " << fespace[m]->GetVSize() << endl;

	}

      {
	// Define a minimal communicator for each subdomain and its interfaces.
	int locSizeSD = (fespace[m] == NULL) ? 0 : fespace[m]->GetTrueVSize();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    locSizeSD += ifNDtrue[interfaceIndex];
	  }

	sd_nonempty[m] = (locSizeSD > 0);
	
	int color = (locSizeSD == 0);
	const int status = MPI_Comm_split(MPI_COMM_WORLD, color, m_rank, &(sd_com[m]));
	MFEM_VERIFY(status == MPI_SUCCESS, "Construction of comm failed");
      }
      
      for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	{
	  const int interfaceIndex = subdomainLocalInterfaces[m][i];
	  
	  MFEM_VERIFY((ifespace[interfaceIndex] == NULL) == (iH1fespace[interfaceIndex] == NULL), "");

	  /*
	  if (ifespace[interfaceIndex] == NULL)
	    {
	      InterfaceToSurfaceInjection[m][i] = NULL;
	      continue;
	    }
	  */
	  
	  size += ifNDtrue[interfaceIndex];  // ifespace[interfaceIndex]->GetTrueVSize();
	  size += ifH1true[interfaceIndex]; // iH1fespace[interfaceIndex]->GetTrueVSize();

	  block_trueOffsets[m+1] += ifNDtrue[interfaceIndex]; // ifespace[interfaceIndex]->GetTrueVSize();
	  block_trueOffsets[m+1] += ifH1true[interfaceIndex]; // iH1fespace[interfaceIndex]->GetTrueVSize();

	  const int ifli = (*interfaceLocalIndex)[interfaceIndex];
	  MFEM_VERIFY(ifli >= 0, "");
	  
	  if (sd_nonempty[m])
	    {
	      SetInterfaceToSurfaceDOFMap(sd_com[m], ifespace[interfaceIndex], fespace[m], fespaceGlobal, pmeshGlobal, m+1, (*localInterfaces)[ifli].faces, 
					  (*localInterfaces)[ifli].edges, (*localInterfaces)[ifli].vertices, &fecbdry,
					  InterfaceToSurfaceInjectionData[m][i], //InterfaceToSurfaceInjectionFullData[m][i],
					  InterfaceToSurfaceInjectionGlobalData[m][i]);
	  
	      // InterfaceToSurfaceInjection[m][i] = new InjectionOperator(MPI_COMM_WORLD, fespace[m], ifespace[interfaceIndex],
	      InterfaceToSurfaceInjection[m][i] = new InjectionOperator(sd_com[m], fespace[m], ifespace[interfaceIndex],
									&(InterfaceToSurfaceInjectionData[m][i][0]),
									InterfaceToSurfaceInjectionGlobalData[m][i]);
	    }
	  else
	    {
	      InterfaceToSurfaceInjection[m][i] = NULL;
	    }
	  
	  /*
	  if (fespace[m] != NULL)
	    {
	  */
	      /*
		if (m == 0)
		{ // debugging
		for (int j=0; j<pmeshIF[interfaceIndex]->GetNEdges(); ++j)
		{
		Array<int> dofs;

		ifespace[interfaceIndex]->GetEdgeDofs(j, dofs);
		MFEM_VERIFY(dofs.Size() == 1, ""); // only implemented for testing order 1 edge elements

		Array<int> edgeVert;
		pmeshIF[interfaceIndex]->GetEdgeVertices(j, edgeVert);
		MFEM_VERIFY(edgeVert.Size() == 2, "");
	    
		const int tdof = ifespace[interfaceIndex]->GetLocalTDofNumber(dofs[0]);
		//if (tdof >= 0)
		{
		bool foundEdge = true;
		const double tol = 1.0e-8;

		for (int l=0; l<3; ++l)
		{
		const double mp_l = 0.5 * (pmeshIF[interfaceIndex]->GetVertex(edgeVert[0])[l] + pmeshIF[interfaceIndex]->GetVertex(edgeVert[1])[l]);
		if (l == 0 && fabs(mp_l - 0.1875) > tol)
		foundEdge = false;
		if (l == 1 && fabs(mp_l - 0.5) > tol)
		foundEdge = false;
		if (l == 2 && fabs(mp_l - 0.25) > tol)
		foundEdge = false;
		}

		if (foundEdge)
		cout << m_rank << ": edge " << j << " has dof " << dofs[0] << " tdof " << tdof << " and midpoint [0.1875, 0.5, 0.25]" << endl;
		}
		}
		}
	      */
	  /*			    
	      InterfaceToSurfaceInjection[m][i] = new InjectionOperator(fespace[m], ifespace[interfaceIndex],
									&(InterfaceToSurfaceInjectionData[m][i][0]),
									InterfaceToSurfaceInjectionGlobalData[m][i]);
	    }
	  else
	    {
	      InterfaceToSurfaceInjection[m][i] = NULL;
	    }
	  */
	}
    }

  tdofsBdry.resize(numSubdomains);
  trueOffsetsSD.resize(numSubdomains);

  tdofsBdryInjection = new SetInjectionOperator*[numSubdomains];
  tdofsBdryInjectionTranspose = new Operator*[numSubdomains];
    
  for (int m=0; m<numSubdomains; ++m)
    {
      A_SS[m] = NULL;
      sdNDPen[m] = NULL;
      sdNDPlusPen[m] = NULL;
	
      if (pmeshSD[m] == NULL)
	{
	  sdND[m] = NULL;
	  sdNDcopy[m] = NULL;
	    
	  /*
	    Asd[m] = NULL;
	    invAsd[m] = NULL;
	    precAsd[m] = NULL;
	  */
	  tdofsBdryInjection[m] = NULL;
	  tdofsBdryInjectionTranspose[m] = NULL;
	}
      else
	{
	  FindBoundaryTrueDOFs(fespace[m], tdofsBdry[m]);  // Determine all true DOF's of fespace[m] on the boundary of pmeshSD[m], representing u_m^s.
	  size += tdofsBdry[m].size();
	  block_trueOffsets[m+1] += tdofsBdry[m].size();

	  tdofsBdryInjection[m] = new SetInjectionOperator(fespace[m]->GetTrueVSize(), &(tdofsBdry[m]));
	  tdofsBdryInjectionTranspose[m] = new TransposeOperator(tdofsBdryInjection[m]);
	    
	  CreateSubdomainMatrices(m);

	  /*
	    { // debugging
	    hypre_CSRMatrix* csr = GetHypreParMatrixData(*(sdND[m]));
	    cout << "csr nr " << csr->num_rows << endl;
	    }
	  */
	}
	
      SetOffsetsSD(m);

      //GMRESSolver *gmres = new GMRESSolver(fespace[m]->GetComm());  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?
      GMRESSolver *gmres = new GMRESSolver(MPI_COMM_WORLD);  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?

#ifdef DDMCOMPLEX
      // Real part
      SetToRealParameters();

      AsdRe[m] = CreateSubdomainOperator(m);

#ifdef HYPRE_PARALLEL_ASDCOMPLEX
      /*
      {
	// Define a minimal communicator for each subdomain and its interfaces.
	
	const int locSizeSD = trueOffsetsSD[m][trueOffsetsSD[m].Size()-1];

	sd_nonempty[m] = (locSizeSD > 0);
	
	int color = (locSizeSD == 0);
	const int status = MPI_Comm_split(MPI_COMM_WORLD, color, m_rank, &(sd_com[m]));
	MFEM_VERIFY(status == MPI_SUCCESS, "Construction of comm failed");
      }
      */
      /*
      for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	{
	  InterfaceToSurfaceInjection[m][i]->SetComm(sd_com[m]);
	}      
      */
      
      CreateSubdomainHypreBlocks(m, AsdRe_HypreBlocks[m], AsdRe_HypreBlockCoef[m]);
#endif
	
      // Imaginary part
      SetToImaginaryParameters();
	
      AsdIm[m] = CreateSubdomainOperator(m);

#ifdef HYPRE_PARALLEL_ASDCOMPLEX
      CreateSubdomainHypreBlocks(m, AsdIm_HypreBlocks[m], AsdIm_HypreBlockCoef[m]);
#endif

      // Set real-valued subdomain operator for preconditioning purposes only.
      SetToPreconditionerParameters();

      preconditionerMode = true;  // TODO!
      AsdP[m] = CreateSubdomainOperator(m);
      preconditionerMode = false;

      SetParameters();
      SetToImaginaryParameters();

      //gmres->SetOperator(*(AsdP[m]));  // TODO: maybe try an SPD version of Asd with a CG solver instead of GMRES.
#else
      Asd[m] = CreateSubdomainOperator(m);

      //gmres->SetOperator(*(Asd[m]));
#endif
	
      //ASPsd[m] = CreateSubdomainOperatorStrumpack(m);

      precAsd[m] = CreateSubdomainPreconditionerStrumpack(m);

      gmres->SetRelTol(1e-12);
      gmres->SetMaxIter(100);  // 3333
      gmres->SetPrintLevel(0);

      gmres->SetPreconditioner(*(precAsd[m]));
      gmres->SetName("invAsd" + std::to_string(m));
      gmres->iterative_mode = false;
	
      invAsd[m] = gmres;
    }
    
  block_trueOffsets.PartialSum();
  MFEM_VERIFY(block_trueOffsets.Last() == size, "");

#ifdef DDMCOMPLEX
  height = 2*size;
  width = 2*size;
    
  globalInterfaceOpRe = new BlockOperator(block_trueOffsets);
  globalInterfaceOpIm = new BlockOperator(block_trueOffsets);

  block_trueOffsets2.SetSize(numSubdomains + 1);
  for (int i=0; i<numSubdomains + 1; ++i)
    block_trueOffsets2[i] = 2*block_trueOffsets[i];
    
  globalInterfaceOp = new BlockOperator(block_trueOffsets2);

  rowTrueOffsetsComplexIF.resize(numLocalInterfaces);
  colTrueOffsetsComplexIF.resize(numLocalInterfaces);
#else
  height = size;
  width = size;

  globalInterfaceOp = new BlockOperator(block_trueOffsets);
#endif
    
  rowTrueOffsetsIF.resize(numLocalInterfaces);
  colTrueOffsetsIF.resize(numLocalInterfaces);
  
  rowTrueOffsetsIFR.resize(numLocalInterfaces);
  colTrueOffsetsIFR.resize(numLocalInterfaces);

  rowTrueOffsetsIFL.resize(numLocalInterfaces);
  colTrueOffsetsIFL.resize(numLocalInterfaces);
    
  rowTrueOffsetsIFBR.resize(numLocalInterfaces);
  colTrueOffsetsIFBR.resize(numLocalInterfaces);

  rowTrueOffsetsIFBL.resize(numLocalInterfaces);
  colTrueOffsetsIFBL.resize(numLocalInterfaces);

#ifdef EQUATE_REDUNDANT_VARS
  rowTrueOffsetsIFRR.resize(numLocalInterfaces);
  colTrueOffsetsIFRR.resize(numLocalInterfaces);
#endif
    
  for (int ili=0; ili<numLocalInterfaces; ++ili)
    {
      const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
      const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

      cout << rank << ": Interface " << ili << " sd " << sd0 << ", " << sd1 << endl;
	
      MFEM_VERIFY(sd0 < sd1, "");

      // Create operators for interface between subdomains sd0 and sd1, namely C_{sd0,sd1} R_{sd1}^T and the other.
      //if (ifNDtrue[ili] > 0)  // TODO: this is wrong, as it excludes the possibility that this process owns S but not F or rho.
      if (sd_nonempty[sd0] || sd_nonempty[sd1])
	{
#ifdef DDMCOMPLEX
	  // Real part
	  SetToRealParameters();
	
	  globalInterfaceOpRe->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
#ifndef ELIMINATE_REDUNDANT_VARS
	  globalInterfaceOpRe->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
#endif
	
	  // Imaginary part
	  SetToImaginaryParameters();
	
	  globalInterfaceOpIm->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
#ifndef ELIMINATE_REDUNDANT_VARS
	  globalInterfaceOpIm->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
#endif
	}
      
      // Set complex blocks
      rowTrueOffsetsComplexIF[ili].SetSize(3);
      colTrueOffsetsComplexIF[ili].SetSize(3);

      rowTrueOffsetsComplexIF[ili] = 0;
      colTrueOffsetsComplexIF[ili] = 0;

      rowTrueOffsetsComplexIF[ili][1] = block_trueOffsets[sd0+1] - block_trueOffsets[sd0];
      rowTrueOffsetsComplexIF[ili][2] = block_trueOffsets[sd0+1] - block_trueOffsets[sd0];
	
      colTrueOffsetsComplexIF[ili][1] = block_trueOffsets[sd1+1] - block_trueOffsets[sd1];
      colTrueOffsetsComplexIF[ili][2] = block_trueOffsets[sd1+1] - block_trueOffsets[sd1];

      rowTrueOffsetsComplexIF[ili].PartialSum();
      colTrueOffsetsComplexIF[ili].PartialSum();

      /*
	PrintNonzerosOfInterfaceOperator(globalInterfaceOpRe->GetBlock(sd0, sd1), fespace[sd0]->GetTrueVSize(), fespace[sd1]->GetTrueVSize(),
	tdofsBdry[sd0], tdofsBdry[sd1], num_procs, rank);
      */
      
      //if (ifNDtrue[ili] > 0)  // TODO: this is wrong, as it excludes the possibility that this process owns S but not F or rho.
      if (sd_nonempty[sd0] || sd_nonempty[sd1])
	{
	  BlockOperator *complexBlock01 = new BlockOperator(rowTrueOffsetsComplexIF[ili], colTrueOffsetsComplexIF[ili]);
	  complexBlock01->SetBlock(0, 0, &(globalInterfaceOpRe->GetBlock(sd0, sd1)));
	  complexBlock01->SetBlock(0, 1, &(globalInterfaceOpIm->GetBlock(sd0, sd1)), -1.0);
	  complexBlock01->SetBlock(1, 0, &(globalInterfaceOpIm->GetBlock(sd0, sd1)));
	  complexBlock01->SetBlock(1, 1, &(globalInterfaceOpRe->GetBlock(sd0, sd1)));
	  
#ifndef ELIMINATE_REDUNDANT_VARS
	  BlockOperator *complexBlock10 = new BlockOperator(colTrueOffsetsComplexIF[ili], rowTrueOffsetsComplexIF[ili]);
	  complexBlock10->SetBlock(0, 0, &(globalInterfaceOpRe->GetBlock(sd1, sd0)));
	  complexBlock10->SetBlock(0, 1, &(globalInterfaceOpIm->GetBlock(sd1, sd0)), -1.0);
	  complexBlock10->SetBlock(1, 0, &(globalInterfaceOpIm->GetBlock(sd1, sd0)));
	  complexBlock10->SetBlock(1, 1, &(globalInterfaceOpRe->GetBlock(sd1, sd0)));
#endif
	
	  globalInterfaceOp->SetBlock(sd0, sd1, complexBlock01);
#ifndef ELIMINATE_REDUNDANT_VARS
	  globalInterfaceOp->SetBlock(sd1, sd0, complexBlock10);
#endif
	}
#else // if DDMCOMPLEX is not defined
      if (ifNDtrue[ili] > 0)
	{
	  globalInterfaceOp->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
	  globalInterfaceOp->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
	}
#endif
    }
    
#ifdef DDMCOMPLEX
  // Create block diagonal operator with entries R_{sd0} A_{sd0}^{-1} R_{sd0}^T

  /*
    BlockOperator *globalSubdomainOpRe = new BlockOperator(block_trueOffsets);
    BlockOperator *globalSubdomainOpIm = new BlockOperator(block_trueOffsets);
  */
    
  rowTrueOffsetsComplexSD.resize(numSubdomains);
  colTrueOffsetsComplexSD.resize(numSubdomains);

  block_ComplexOffsetsSD.resize(numSubdomains);

  //BlockOperator *globalSubdomainOp = new BlockOperator(block_trueOffsets2);
  globalSubdomainOp = new BlockOperator(block_trueOffsets2);
#else
  // Create block diagonal operator with entries R_{sd0} A_{sd0}^{-1} R_{sd0}^T
  //BlockOperator *globalSubdomainOp = new BlockOperator(block_trueOffsets);
  globalSubdomainOp = new BlockOperator(block_trueOffsets);
#endif
    
  rowTrueOffsetsSD.resize(numSubdomains);
  colTrueOffsetsSD.resize(numSubdomains);
    
  rowSROffsetsSD.SetSize(numSubdomains + 1); // number of blocks + 1
  colSROffsetsSD.SetSize(numSubdomains + 1); // number of blocks + 1
  rowSROffsetsSD = 0;
  colSROffsetsSD = 0;

  for (int m=0; m<numSubdomains; ++m)
    {
      //if (Asd[m] != NULL)
      {
	// Create block injection operator R_{sd0}^T from (u^s, f_i, \rho_i) space to (u, f_i, \rho_i) space.

	rowTrueOffsetsSD[m].SetSize(2 + 1);  // Number of blocks + 1
	colTrueOffsetsSD[m].SetSize(2 + 1);  // Number of blocks + 1

	rowTrueOffsetsSD[m] = 0;
	rowTrueOffsetsSD[m][1] = (fespace[m] != NULL) ? fespace[m]->GetTrueVSize() : 0;

	int ifsize = 0;
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    /*
	    MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
	    MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
	    */
	    
	    //ifsize += ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
	    ifsize += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	  }

	rowTrueOffsetsSD[m][2] = ifsize;

	cout << rank << ": SD " << m << " ifsize " << ifsize << endl;
	    
	colTrueOffsetsSD[m] = rowTrueOffsetsSD[m];
	colTrueOffsetsSD[m][1] = tdofsBdry[m].size();

	rowTrueOffsetsSD[m].PartialSum();
	colTrueOffsetsSD[m].PartialSum();

	injSD[m] = new BlockOperator(rowTrueOffsetsSD[m], colTrueOffsetsSD[m]);

	if (tdofsBdryInjection[m] != NULL)
	  {
	    injSD[m]->SetBlock(0, 0, tdofsBdryInjection[m]);
	  }
	    
	injSD[m]->SetBlock(1, 1, new IdentityOperator(ifsize));

#ifdef DDMCOMPLEX
	rowTrueOffsetsComplexSD[m].SetSize(2 + 1);  // Number of blocks + 1
	colTrueOffsetsComplexSD[m].SetSize(2 + 1);  // Number of blocks + 1

	rowTrueOffsetsComplexSD[m] = 0;
	colTrueOffsetsComplexSD[m] = 0;
	    
	rowTrueOffsetsComplexSD[m][1] = rowTrueOffsetsSD[m][2];
	rowTrueOffsetsComplexSD[m][2] = rowTrueOffsetsSD[m][2];
	    
	colTrueOffsetsComplexSD[m][1] = colTrueOffsetsSD[m][2];
	colTrueOffsetsComplexSD[m][2] = colTrueOffsetsSD[m][2];

	rowTrueOffsetsComplexSD[m].PartialSum();
	colTrueOffsetsComplexSD[m].PartialSum();

	injComplexSD[m] = new BlockOperator(rowTrueOffsetsComplexSD[m], colTrueOffsetsComplexSD[m]);
	injComplexSD[m]->SetBlock(0, 0, injSD[m]);
	injComplexSD[m]->SetBlock(1, 1, injSD[m]);
	    
	block_ComplexOffsetsSD[m].SetSize(3);  // number of blocks plus 1
	block_ComplexOffsetsSD[m] = 0;
	block_ComplexOffsetsSD[m][1] = rowTrueOffsetsSD[m][2];
	block_ComplexOffsetsSD[m][2] = rowTrueOffsetsSD[m][2];
	block_ComplexOffsetsSD[m].PartialSum();

	if (sd_nonempty[m])
	  {
	    AsdComplex[m] = new BlockOperator(block_ComplexOffsetsSD[m]);
	    AsdComplex[m]->SetBlock(0, 0, AsdRe[m]);
	    AsdComplex[m]->SetBlock(0, 1, AsdIm[m], -1.0);
	    AsdComplex[m]->SetBlock(1, 0, AsdIm[m]);
	    AsdComplex[m]->SetBlock(1, 1, AsdRe[m]);

	    precAsdComplex[m] = new BlockDiagonalPreconditioner(block_ComplexOffsetsSD[m]);

	    //precAsdComplex[m]->SetDiagonalBlock(0, invAsd[m]);
	    //precAsdComplex[m]->SetDiagonalBlock(1, invAsd[m]);
	  }
	else
	  {
	    AsdComplex[m] = NULL;
	    precAsdComplex[m] = NULL;
	  }
	
	MPI_Barrier(MPI_COMM_WORLD);
  
#ifdef SPARSE_ASDCOMPLEX
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
	const int locSizeSD = trueOffsetsSD[m][trueOffsetsSD[m].Size()-1];

	/*
	  { // debugging
	  //hypre_CSRMatrix* csrsdND2 = GetHypreParMatrixData(*(sdND[0]));
	  hypre_CSRMatrix* csrsdND2 = GetHypreParMatrixData(*(AsdRe_HypreBlocks[0](0, 0)));
	  cout << csrsdND2->num_rows << endl;
	  }
	*/
	    
	if (locSizeSD > 0)
	  {
	    HypreParMatrix *HypreAsdComplexRe = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], AsdRe_HypreBlocks[m], AsdRe_HypreBlockCoef[m]);
	    HypreParMatrix *HypreAsdComplexIm = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], AsdIm_HypreBlocks[m], AsdIm_HypreBlockCoef[m]);

	    ComplexHypreParMatrix tmpComplex(HypreAsdComplexRe, HypreAsdComplexIm, false, false);
	    HypreAsdComplex[m] = tmpComplex.GetSystemMatrix();

	    /*
	      { // This should do the same as ComplexHypreParMatrix::GetSystemMatrix().
	      Array<int> complexOS(3);  // number of blocks + 1
	      complexOS = 0;
	      for (int i=1; i<3; ++i)
	      complexOS[i] = HypreAsdComplexRe->Height();
		  
	      complexOS.PartialSum();

	      Array2D<double> complexCoef;
	      Array2D<HypreParMatrix*> complexBlock;

	      complexCoef.SetSize(2, 2);
	      complexBlock.SetSize(2, 2);

	      complexBlock(0, 0) = HypreAsdComplexRe;
	      complexCoef(0, 0) = 1.0;

	      complexBlock(0, 1) = HypreAsdComplexIm;
	      complexCoef(0, 1) = -1.0;

	      complexBlock(1, 0) = HypreAsdComplexIm;
	      complexCoef(1, 0) = 1.0;

	      complexBlock(1, 1) = HypreAsdComplexRe;
	      complexCoef(1, 1) = 1.0;
		  
	      HypreAsdComplex[m] = CreateHypreParMatrixFromBlocks(sd_com[m], complexOS, complexBlock, complexCoef);
	      }
	    */
		
	    //if (m == 0) HypreAsdComplexRe->Print("HypreAsdComplexRe0_Par5");
	    //if (m == 0) HypreAsdComplexRe->Print("HypreAsdComplexRe0_Serial");
	    //if (m == 1) HypreAsdComplexRe->Print("HypreAsdComplexRe1_Par6");
	    //if (m == 1) HypreAsdComplexRe->Print("HypreAsdComplexRe1_Serial");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0_Serial");
	    //if (m == 1) HypreAsdComplexIm->Print("HypreAsdComplexIm1_Serial");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0_Par5");
	    //if (m == 1) HypreAsdComplexIm->Print("HypreAsdComplexIm1_Par6");
	    //if (m == 1) HypreAsdComplex[m]->Print("HypreAsdComplex_Par7");
	    //if (m == 1) HypreAsdComplex[m]->Print("HypreAsdComplex_Serial");
		
	    //invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), MPI_COMM_WORLD);
	    invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), sd_com[m]);
	  }
	else
	  {
	    HypreAsdComplex[m] = NULL;
	    invAsdComplex[m] = NULL;
	  }
#else // if not HYPRE_PARALLEL_ASDCOMPLEX
	SpAsdComplex[m] = GetSparseMatrixFromOperator(AsdComplex[m]);
	SpAsdComplexRowSizes[m] = new HYPRE_Int[2];
	SpAsdComplexRowSizes[m][0] = 0;
	SpAsdComplexRowSizes[m][1] = SpAsdComplex[m]->Size();

	/*
	  {
	  //std::string filename = 
	  ofstream file("SpAsdComplex" + std::to_string(m));
	  SpAsdComplex[m]->PrintMatlab(file);
	  file.close();
	  }
	*/

	HypreAsdComplex[m] = new HypreParMatrix(MPI_COMM_WORLD, SpAsdComplex[m]->Size(), SpAsdComplexRowSizes[m], SpAsdComplex[m]);  // constructor with 4 arguments, v1
	    
	invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), MPI_COMM_WORLD);
#endif // HYPRE_PARALLEL_ASDCOMPLEX
	//if (m == 0) HypreAsdComplex[m]->Print("HypreAsdComplex");
#else // if not SPARSE_ASDCOMPLEX
	//GMRESSolver *gmres = new GMRESSolver(fespace[m]->GetComm());  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?
	GMRESSolver *gmres = new GMRESSolver(MPI_COMM_WORLD);  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?

	gmres->SetOperator(*(AsdComplex[m]));

	gmres->SetRelTol(1e-12);
	gmres->SetMaxIter(100);  // 3333
	gmres->SetPrintLevel(1);

	gmres->SetPreconditioner(*(precAsdComplex[m]));
	gmres->SetName("invAsdComplex" + std::to_string(m));
	gmres->iterative_mode = false;
	    
	invAsdComplex[m] = gmres;
#endif // SPARSE_ASDCOMPLEX

	if (invAsdComplex[m] != NULL)
	  {
	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injComplexSD[m]), invAsdComplex[m], injComplexSD[m],
									false, false, false));
	  }
#else // if not DDMCOMPLEX
	globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injSD[m]), invAsd[m], injSD[m], false, false, false));
#endif // DDMCOMPLEX

	if (fespace[m] != NULL)
	  {
	    rowSROffsetsSD[m+1] = fespace[m]->GetTrueVSize();
	    colSROffsetsSD[m+1] = fespace[m]->GetTrueVSize();
	  }
      }
    }

  // Create operators R_{sd0} A_{sd0}^{-1} C_{sd0,sd1} R_{sd1}^T by multiplying globalInterfaceOp on the left by globalSubdomainOp. Then add identity.

#ifdef DDMCOMPLEX
  // The complex system Au = y is rewritten in terms of real and imaginary parts as
  // [ A^{Re} -A^{Im} ] [ u^{Re} ] = [ y^{Re} ]
  // [ A^{Im}  A^{Re} ] [ u^{Im} ] = [ y^{Im} ]
  // The inverse of the block matrix here is
  // [ A^{Re}^{-1} - A^{Re}^{-1} A^{Im} C^{-1} A^{Im} A^{Re}^{-1}   A^{Re}^{-1} A^{Im} C^{-1} ]
  // [ -C^{-1} A^{Im} A^{Re}^{-1}                                               C^{-1}        ]
  // where C = A^{Re} + A^{Im} A^{Re}^{-1} A^{Im}
  // Instead of inverting directly, which has many applications of inverses, we will just use an iterative solver for the complex system
  // with block diagonal preconditioner that solves the diagonal blocks A^{Re}. 

  // The system [  I  A_1^{-1} C_{12} ] [ u_1 ] = [ y_1 ]   (omitting restriction operators R_{sd} and their transposes for simplicity)
  //            [  A_2^{-1} C_{21}  I ] [ u_2 ] = [ y_2 ]
  //
  // for complex matrices A_m and C_{mn} and complex vectors u_m and y_m is rewritten in terms of real and imaginary parts as
  //   [  I  A_1^{-1} C_{12} ] [ u_1^{Re} ] = [ y_1^{Re} ]
  //   [        ...          ] [ u_1^{Im} ] = [ y_1^{Im} ]
  //   [  A_2^{-1} C_{21}  I ] [ u_2^{Re} ] = [ y_2^{Re} ]
  //   [        ...          ] [ u_2^{Im} ] = [ y_2^{Im} ]
  // with A_m and C_{mn} written as 2x2 blocks with real and imaginary parts. That is, the splitting into real and imaginary parts is done
  // on the subdomain block level. 
#endif
    
  globalOp = new SumOperator(new ProductOperator(globalSubdomainOp, globalInterfaceOp, false, false), new IdentityOperator(height), false, false, 1.0, 1.0);

  // Create source reduction operator.
  { // TODO: is this used?
    rowSROffsetsSD.PartialSum();
    colSROffsetsSD.PartialSum();
    /*
      NDinv = new BlockOperator(rowSROffsetsSD, colSROffsetsSD);

      for (int m=0; m<numSubdomains; ++m)
      {
      if (pmeshSD[m] != NULL)
      {
      NDinv->SetBlock(m, m, sdNDinv[m]);
      }
      }
    */
  }
    
  testing = false;
}

void DDMInterfaceOperator::GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobalRe, Vector & sourceGlobalIm,
					    Vector & sourceReduced) const
{
  Vector sourceSD, wSD, vSD;

  int nprocs, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#ifdef DDMCOMPLEX
  MFEM_VERIFY(sourceReduced.Size() == block_trueOffsets2[numSubdomains], "");
#else
  MFEM_VERIFY(sourceReduced.Size() == block_trueOffsets[numSubdomains], "");
#endif
  MFEM_VERIFY(sourceReduced.Size() == this->Height(), "");
    
  sourceReduced = 0.0;

  for (int m=0; m<numSubdomains; ++m)
    {
      ySD[m] = NULL;
	
      //if (pmeshSD[m] != NULL)
      if (sd_nonempty[m])
	{
	  const int sdsize = (fespace[m] == NULL) ? 0 : fespace[m]->GetTrueVSize();
	  //sourceSD.SetSize(fespace[m]->GetTrueVSize());
	  sourceSD.SetSize(sdsize);

	  // Map from the global u to [u_m f_m \rho_m], with blocks corresponding to subdomains, and f_m = 0, \rho_m = 0.
#ifdef DDMCOMPLEX
	  // In the complex case, we map from the global u^{Re} and u^{Im} to [u_m^{Re} f_m^{Re} \rho_m^{Re} u_m^{Im} f_m^{Im} \rho_m^{Im}].

	  MFEM_VERIFY(AsdComplex[m]->Height() == block_ComplexOffsetsSD[m][2], "");
	  MFEM_VERIFY(AsdComplex[m]->Height() == 2*block_ComplexOffsetsSD[m][1], "");

	  cout << rank << ": Setting real subdomain DOFs, sd " << m << endl;
	    
	  ySD[m] = new Vector(AsdComplex[m]->Height());  // Size of [u_m f_m \rho_m], real and imaginary parts

	  wSD.SetSize(AsdComplex[m]->Height());
	  wSD = 0.0;

	  if (sdsize > 0)
	    {
	      sourceSD = -1.0e7;
	      SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalRe, sourceSD);

	      for (int i=0; i<sourceSD.Size(); ++i)
		{
		  if (fabs(sourceSD[i]) > 1.0e5 || sourceSD[i] < -1.0e5)
		    cout << "Entry not set!!!!!!!!!!!!!!" << endl;
		}
	    }
	  
	  const bool explicitRHS = false;
	  /*
	    if (explicitRHS)
	    {
	    ParGridFunction x(fespace[m]);
	    VectorFunctionCoefficient f2(3, test2_RHS_exact);

	    //x.SetFromTrueDofs(sourceSD);
	    x.ProjectCoefficient(f2);

	    x.GetTrueDofs(sourceSD);
	    }
	  */
	  
	  /*
//////////////////////
DataCollection *dc = NULL;
const bool visit = false;
if (visit && m == 0)
{
ParGridFunction x(fespace[m]);
VectorFunctionCoefficient f2(3, test2_RHS_exact);

//x.SetFromTrueDofs(sourceSD);
x.ProjectCoefficient(f2);

{
dc = new VisItDataCollection("fsd0Re", pmeshSD[m]);
dc->SetPrecision(8);
// To save the mesh using MFEM's parallel mesh format:
// dc->SetFormat(DataCollection::PARALLEL_FORMAT);
}
dc->RegisterField("solution", &x);
dc->SetCycle(0);
dc->SetTime(0.0);
dc->Save();
}
/////////////
*/

  (*(ySD[m])) = 0.0;
  for (int i=0; i<sourceSD.Size(); ++i)
    (*(ySD[m]))[i] = sourceSD[i];  // Set the u_m block of ySD, real part

  cout << rank << ": Setting imaginary subdomain DOFs, sd " << m << endl;
  if (sdsize > 0)
    {
      SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalIm, sourceSD);
    }
  
    cout << rank << ": Done setting imaginary subdomain DOFs, sd " << m << endl;

    if (explicitRHS)
      sourceSD = 0.0;

    for (int i=0; i<sourceSD.Size(); ++i)
      (*(ySD[m]))[block_ComplexOffsetsSD[m][1] + i] = sourceSD[i];  // Set the u_m block of ySD, imaginary part
	} // if pmeshSD[m] != null

      //MPI_Barrier(MPI_COMM_WORLD);  // TODO: remove this barrier. The subdomain computations should be parallel.
      
      if (sd_nonempty[m])
      {
	cout << rank << ": ySD norm: " << ySD[m]->Norml2() << ", wSD norm " << wSD.Norml2() << " (sd " << m << ")" << endl;
	    
	cout << rank << ": Applying invAsdComplex[" << m << "]" << endl;

	//if (m == 0) HypreAsdComplex[m]->Print("HypreAsdComplexCheck");

	if (invAsdComplex[m] != NULL)
	  invAsdComplex[m]->Mult(*(ySD[m]), wSD);
	//AsdComplex[m]->Mult(*(ySD[m]), wSD);

	cout << rank << ": Done applying invAsdComplex[" << m << "], norm of wSD: " << wSD.Norml2() << endl;

	// Extract only the [u_m^s, f_m, \rho_m] entries, real and imaginary parts.
	vSD.SetSize(block_trueOffsets2[m+1] - block_trueOffsets2[m]);
	injComplexSD[m]->MultTranspose(wSD, vSD);

	for (int i=0; i<vSD.Size(); ++i)
	  sourceReduced[block_trueOffsets2[m] + i] = vSD[i];
#else
	SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalRe, sourceSD);

	ySD[m] = new Vector(Asd[m]->Height());  // Size of [u_m f_m \rho_m]
	wSD.SetSize(Asd[m]->Height());  // Size of [u_m f_m \rho_m]
	    
	(*(ySD[m])) = 0.0;
	for (int i=0; i<sourceSD.Size(); ++i)
	  (*(ySD[m]))[i] = sourceSD[i];  // Set the u_m block of ySD

	cout << "Applying invAsd[" << m << "]" << endl;
	    
	invAsd[m]->Mult(*(ySD[m]), wSD);

	cout << "Done applying invAsd[" << m << "]" << endl;

	// Extract only the [u_m^s, f_m, \rho_m] entries.
	vSD.SetSize(block_trueOffsets[m+1] - block_trueOffsets[m]);
	injSD[m]->MultTranspose(wSD, vSD);

	for (int i=0; i<vSD.Size(); ++i)
	  sourceReduced[block_trueOffsets[m] + i] = vSD[i];
#endif
      }
      /*
	else
	{
	ySD[m] = NULL;
	}
      */

      //MPI_Barrier(MPI_COMM_WORLD);
}  // loop (m) over subdomains
}

void DDMInterfaceOperator::RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced, Vector & solDomain)
{
  MFEM_VERIFY(solReduced.Size() == this->Height(), "");
  Vector w(this->Height());
  Vector v, u, uSD, wSD;

  MFEM_VERIFY(this->Height() == block_trueOffsets2[numSubdomains], "");

  // Assuming GetReducedSource has been called, ySD stores y_m (with full components [u_m f_m \rho_m]) for each subdomain m.

  // globalInterfaceOp represents the off-diagonal blocks C_{ij} R_j^T
  globalInterfaceOp->Mult(solReduced, w);

  // Now w = \sum_{j\neq i} C_{ij} R_j^T \overbar{u}_j

  Vector domainError(3), eSD(3);
  domainError = 0.0;
  
  for (int m=0; m<numSubdomains; ++m)
    {
      if (ySD[m] != NULL)
	{
#ifdef DDMCOMPLEX
	  //MFEM_VERIFY(ySD[m]->Size() == block_trueOffsets2[m+1] - block_trueOffsets2[m], "");  wrong
	  //MFEM_VERIFY(ySD[m]->Size() > block_trueOffsets2[m+1] - block_trueOffsets2[m], "");
	  MFEM_VERIFY(ySD[m]->Size() == AsdComplex[m]->Height(), "");

	  // Put the [u_m^s, f_m, \rho_m] entries of w (real and imaginary parts) into wSD.
	  wSD.SetSize(block_trueOffsets2[m+1] - block_trueOffsets2[m]);
	  uSD.SetSize(AsdComplex[m]->Height());

	  for (int i=0; i<block_trueOffsets2[m+1] - block_trueOffsets2[m]; ++i)
	    wSD[i] = w[block_trueOffsets2[m] + i];

	  {
	    const int imsize = wSD.Size() / 2;
	    Vector wIm(imsize);

	    for (int i=0; i<imsize; ++i)
	      wIm[i] = wSD[imsize + i];

	    const double normIm = wIm.Norml2();
	    const double normIm2 = normIm*normIm;

	    const double normW = wSD.Norml2();
	    const double norm2 = normW*normW;

	    for (int i=0; i<imsize; ++i)
	      wIm[i] = solReduced[block_trueOffsets2[m] + i];

	    const double normSolRe = wIm.Norml2();
	    const double normSolRe2 = normSolRe*normSolRe;

	    for (int i=0; i<imsize; ++i)
	      wIm[i] = solReduced[block_trueOffsets2[m] + imsize + i];

	    const double normSolIm = wIm.Norml2();
	    const double normSolIm2 = normSolIm*normSolIm;

	    double sumNormIm2 = 0.0;
	    double sumNormSolRe2 = 0.0;
	    double sumNormSolIm2 = 0.0;
	    double sumNorm2 = 0.0;
	    
	    MPI_Allreduce(&norm2, &sumNorm2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);
	    MPI_Allreduce(&normIm2, &sumNormIm2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);

	    MPI_Allreduce(&normSolRe2, &sumNormSolRe2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);
	    MPI_Allreduce(&normSolIm2, &sumNormSolIm2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);

	    //if (m_rank == 0)
	      {
		cout << "Subdomain " << m << " global Norml2 of imaginary part of C_ij times reduced solution: " << sqrt(sumNormIm2) << ", full reduced " << sqrt(sumNorm2) << endl;
		cout << "Subdomain " << m << " global Norml2 of real part of reduced solution: " << sqrt(sumNormSolRe2) << ", imaginary " << sqrt(sumNormSolIm2) << endl;
	      }
	  }
	  
	  injComplexSD[m]->Mult(wSD, uSD);

	  *(ySD[m]) -= uSD;
	  uSD = 0.0;
	  if (invAsdComplex[m] != NULL)
	    invAsdComplex[m]->Mult(*(ySD[m]), uSD);

	  PrintSubdomainError(m, uSD, eSD);
	  if (m_rank == 0)
	    {
	      for (int i=0; i<3; ++i)
		domainError[i] += eSD[i] * eSD[i];
	    }
#else
	  MFEM_VERIFY(ySD[m]->Size() == block_trueOffsets[m+1] - block_trueOffsets[m], "");

	  v.SetSize(ySD[m]->Size());
	  u.SetSize(ySD[m]->Size());
	    
	  for (int i=0; i<block_trueOffsets[m+1] - block_trueOffsets[m]; ++i)
	    v[i] = (*(ySD[m]))[i] - w[block_trueOffsets[m] + i];

	  invAsd[m]->Mult(v, u);

	  uSD.SetSize(fespace[m]->GetTrueVSize());

	  for (int i=0; i<uSD.Size(); ++i)
	    uSD[i] = u[i];
	    
	  SetDomainDofsFromSubdomainDofs(fespace[m], fespaceGlobal, uSD, solDomain);
#endif
	}
    } // loop (m) over subdomains

  //if (m_rank == 0)
    {
      const double errRe = sqrt(domainError[0]);
      const double errIm = sqrt(domainError[1]);
      const double errTotal = sqrt(domainError[0] + domainError[1]);
      const double normE = sqrt(domainError[2]);
      
      cout << m_rank << ": global domain || E_h - E ||_{L^2} Re = " << errRe << endl;
      cout << m_rank << ": global domain || E_h - E ||_{L^2} Im = " << errIm << endl;
      cout << m_rank << ": global domain || E ||_{L^2} Re = " << normE << endl;
      cout << m_rank << ": global domain rel err Re " << errRe / normE << endl;
      cout << m_rank << ": global domain rel err tot " << errTotal / normE << endl;
    }
}

void DDMInterfaceOperator::CreateInterfaceMatrices(const int interfaceIndex)
{
  int num_procs, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  ConstantCoefficient one(1.0);
  Array<int> H1_ess_tdof_list;
  Array<int> ND_ess_tdof_list;
    
  Array<int> H1_ess_bdr;
  Array<int> ND_ess_bdr;
    
  // Nedelec interface operators

  ParBilinearForm *NDmass = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  NDmass->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  NDmass->Assemble();

#ifdef ZERO_IFND_BC
  {
    ND_ess_bdr.SetSize(pmeshIF[interfaceIndex]->bdr_attributes.Max());
    ND_ess_bdr = 1;

    ifespace[interfaceIndex]->GetEssentialTrueDofs(ND_ess_bdr, ND_ess_tdof_list);
  }
#endif
    
  ParBilinearForm *NDcurlcurl = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  NDcurlcurl->AddDomainIntegrator(new CurlCurlIntegrator(one));
  NDcurlcurl->Assemble();
    
  ParBilinearForm *ND_FS = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ND_FS->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  ND_FS->AddDomainIntegrator(new CurlCurlIntegrator(one));
  ND_FS->Assemble();

  ifNDmass[interfaceIndex] = new HypreParMatrix();
  ifNDcurlcurl[interfaceIndex] = new HypreParMatrix();
  ifND_FS[interfaceIndex] = new HypreParMatrix();

  NDmass->FormSystemMatrix(ND_ess_tdof_list, *(ifNDmass[interfaceIndex]));
  NDcurlcurl->FormSystemMatrix(ND_ess_tdof_list, *(ifNDcurlcurl[interfaceIndex]));
  ND_FS->FormSystemMatrix(ND_ess_tdof_list, *(ifND_FS[interfaceIndex]));

  cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize() << ", mass height "
       << ifNDmass[interfaceIndex]->Height() << ", width " << ifNDmass[interfaceIndex]->Width() << ", ND V size "
       << ifespace[interfaceIndex]->GetVSize() << endl;

  Operator *ifNDmassSP = new STRUMPACKRowLocMatrix(*(ifNDmass[interfaceIndex]));
  ifNDmassInv[interfaceIndex] = CreateStrumpackSolver(ifNDmassSP, ifespace[interfaceIndex]->GetComm());

  // H^1 interface operators

  ParBilinearForm *H1mass = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  H1mass->AddDomainIntegrator(new MassIntegrator(one));
  H1mass->Assemble();

  ifH1mass[interfaceIndex] = new HypreParMatrix();

#ifdef ZERO_RHO_BC
  {
    H1_ess_bdr.SetSize(pmeshIF[interfaceIndex]->bdr_attributes.Max());
    H1_ess_bdr = 1;

    iH1fespace[interfaceIndex]->GetEssentialTrueDofs(H1_ess_bdr, H1_ess_tdof_list);
  }    
#endif
    
  H1mass->FormSystemMatrix(H1_ess_tdof_list, *(ifH1mass[interfaceIndex]));

  {
    /*
      Operator *ifH1massSP = new STRUMPACKRowLocMatrix(*(ifH1mass[interfaceIndex]));
      ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massSP, iH1fespace[interfaceIndex]->GetComm());
    */
      
    ParBilinearForm *H1stiff = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
    H1stiff->AddDomainIntegrator(new MassIntegrator(one));
    H1stiff->AddDomainIntegrator(new DiffusionIntegrator(one));
    H1stiff->Assemble();

    HypreParMatrix *ifH1stiff = new HypreParMatrix();
    
    H1stiff->FormSystemMatrix(H1_ess_tdof_list, *ifH1stiff);
    Operator *ifH1massSP = new STRUMPACKRowLocMatrix(*ifH1stiff);
    ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massSP, iH1fespace[interfaceIndex]->GetComm());
  }
    
  // Mixed interface operator
  ParMixedBilinearForm *NDH1grad = new ParMixedBilinearForm(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  NDH1grad->AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
  NDH1grad->Assemble();

  /*
#ifdef ZERO_RHO_BC
  {
  // Note that this does not seem to be necessary.
    Array<int> ess_bdr(pmeshIF[interfaceIndex]->bdr_attributes.Max());
    cout << myid << ": ess_bdr size " << pmeshIF[interfaceIndex]->bdr_attributes.Max() << endl;
    ess_bdr = 1;
      
    Vector testDummy(ifespace[interfaceIndex]->GetVSize());  // TODO: is this necessary?
    Vector trialDummy(iH1fespace[interfaceIndex]->GetVSize());

    testDummy = 0.0;
    trialDummy = 0.0;
      
    NDH1grad->EliminateTrialDofs(ess_bdr, trialDummy, testDummy);
  }
#endif
  */
  
  NDH1grad->Finalize();
    
  //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
  //NDH1grad->FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
  ifNDH1grad[interfaceIndex] = NDH1grad->ParallelAssemble();
  ifNDH1gradT[interfaceIndex] = ifNDH1grad[interfaceIndex]->Transpose();
    
  cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize()
       << ", H1 true size " << iH1fespace[interfaceIndex]->GetTrueVSize()
       << ", NDH1 height " << ifNDH1grad[interfaceIndex]->Height() << ", width " << ifNDH1grad[interfaceIndex]->Width() << endl;
}
  
Operator* DDMInterfaceOperator::CreateCij(const int localInterfaceIndex, const int orientation)
{
  const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
  const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

  const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];

  /* REMOVE THIS
  // Nedelec interface operators
  ParBilinearForm *a = new ParBilinearForm(ifespace[interfaceIndex]);

  ConstantCoefficient one(1.0);

  a->AddDomainIntegrator(new CurlCurlIntegrator(one));
  a->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  a->Assemble();
    
  HypreParMatrix A;
  Array<int> ess_tdof_list;  // empty
  a->FormSystemMatrix(ess_tdof_list, A);
  */
    
  if (orientation == 0)
    {
      rowTrueOffsetsIF[localInterfaceIndex].resize(2);
      colTrueOffsetsIF[localInterfaceIndex].resize(2);
    }
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(3);  // Number of blocks + 1
  colTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(4);  // Number of blocks + 1
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;
  colTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;

  /*
  // This is larger than it needs to be for this interface, because the solution space has DOF's on the entire subdomain boundaries.
  rowTrueOffsetsIF[localInterfaceIndex][1] = tdofsBdry[sd0].size();
  colTrueOffsetsIF[localInterfaceIndex][1] = tdofsBdry[sd1].size();
  */
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];

  rowTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];

  /*
  rowTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();

  rowTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
  */
  
  //rowTrueOffsetsIF[localInterfaceIndex][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIF[localInterfaceIndex][orientation][3] = ifH1true[interfaceIndex];
  //colTrueOffsetsIF[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
  colTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *op = new BlockOperator(rowTrueOffsetsIF[localInterfaceIndex][orientation], colTrueOffsetsIF[localInterfaceIndex][orientation]);

  // In PengLee2012 notation, (sd0,sd1) = (m,n).
    
  // In PengLee2012 C_{mn}^{SS} corresponds to
  // -alpha <\pi_{mn}(v_m), [[u]]_{mn}>_{S_{mn}} +
  // -beta <curl_\tau \pi_{mn}(v_m), curl_\tau [[u]]_{mn}>_{S_{mn}}
  // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the C_{mn}^{SS} block is the part
  // alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} +
  // beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
  // This is an interface mass plus curl-curl stiffness matrix.
#ifdef EQUATE_REDUNDANT_VARS
  if (realPart && orientation == 1)
    {
      op->SetBlock(0, 0, new SumOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, alpha, beta),
					 new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
					 false, false, 1.0, -PENALTY_U_S));
    }
  else
#endif
    {
      op->SetBlock(0, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, alpha, beta));
    }
    
  // In PengLee2012 C_{mn}^{SF} corresponds to
  // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
  // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the C_{mn}^{SF} block is the part
  // -<\pi_{mn}(v_m), \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
  // This is an interface mass matrix.

#ifdef DDMCOMPLEX
  if (realPart)
#endif
    {
      op->SetBlock(0, 1, ifNDmass[interfaceIndex], -1.0);
    }

  // In PengLee2012 C_{mn}^{S\rho} corresponds to
  // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
  // Since <<\rho>>_{mn} = \rho_m + \rho_n, the C_{mn}^{S\rho} block is the part
  // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_n >_{S_{mn}}
  // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
  op->SetBlock(0, 2, ifNDH1grad[interfaceIndex], -gamma);

  // In PengLee2012 C_{mn}^{FS} corresponds to
  // <w_m, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
  // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the C_{mn}^{FS} block is the part
  // -<w_m, \pi_{nm}(u_n)>_{S_{mn}} - beta/alpha <curl_\tau w_m, curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
  // This is an interface mass plus curl-curl stiffness matrix.

#ifdef DDMCOMPLEX
#ifdef EQUATE_REDUNDANT_VARS
  if (orientation == 0)
#endif
    {
      if (realPart)
	{
	  op->SetBlock(1, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -1.0, -betaOverAlpha));
	}
      else
	{
	  op->SetBlock(1, 0, ifNDcurlcurl[interfaceIndex], -betaOverAlpha);
	}
    }
#else
  op->SetBlock(1, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -1.0, -beta / alpha));
#endif

  // In PengLee2012 C_{mn}^{FF} corresponds to
  // alpha^{-1} <w_m, -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
  // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the C_{mn}^{FF} block is the part
  // alpha^{-1} <w_m, \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
  // This is an interface mass matrix.

#ifdef EQUATE_REDUNDANT_VARS
  if (realPart && orientation == 1)
    {
      op->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));
    }
  else
#endif
    {
      op->SetBlock(1, 1, ifNDmass[interfaceIndex], alphaInverse);
    }
    
  // In PengLee2012 C_{mn}^{F\rho} corresponds to
  // gamma / alpha <w_m, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
  // Since <<\rho>>_{mn} = \rho_m + \rho_n, the C_{mn}^{F\rho} block is the part
  // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
  // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
#ifdef EQUATE_REDUNDANT_VARS
  if (orientation == 0)
#endif
    {
      op->SetBlock(1, 2, ifNDH1grad[interfaceIndex], gammaOverAlpha);
    }
    
  // Row 2 is just zeros.
    
  return op;
}

Operator* DDMInterfaceOperator::CreateInterfaceOperator(const int localInterfaceIndex, const int orientation)
{
  const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
  const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

  const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];

  /*
  MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
  MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
  */
  
  // Find interface indices with respect to subdomains sd0 and sd1.
  int sd0if = -1;
  int sd1if = -1;

  int sd0os = 0;
  int sd1os = 0;
    
  int sd0osComp = 0;
  int sd1osComp = 0;
    
  for (int i=0; i<subdomainLocalInterfaces[sd0].size(); ++i)
    {
      if (subdomainLocalInterfaces[sd0][i] == interfaceIndex)
	{
	  MFEM_VERIFY(sd0if == -1, "");
	  sd0if = i;
	}

      if (sd0if == -1)
	{
	  sd0os += ifNDtrue[subdomainLocalInterfaces[sd0][i]] + ifH1true[subdomainLocalInterfaces[sd0][i]];
	  //sd0os += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
	}
      else
	{
	  sd0osComp += ifNDtrue[subdomainLocalInterfaces[sd0][i]] + ifH1true[subdomainLocalInterfaces[sd0][i]];
	  //sd0osComp += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
	}
    }

  for (int i=0; i<subdomainLocalInterfaces[sd1].size(); ++i)
    {
      if (subdomainLocalInterfaces[sd1][i] == interfaceIndex)
	{
	  MFEM_VERIFY(sd1if == -1, "");
	  sd1if = i;
	}

      if (sd1if == -1)
	{
	  sd1os += ifNDtrue[subdomainLocalInterfaces[sd1][i]] + ifH1true[subdomainLocalInterfaces[sd1][i]];
	  //sd1os += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
	}
      else
	{
	  sd1osComp += ifNDtrue[subdomainLocalInterfaces[sd1][i]] + ifH1true[subdomainLocalInterfaces[sd1][i]];
	  //sd1osComp += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
	}
    }
    
  MFEM_VERIFY(sd0if >= 0, "");
  MFEM_VERIFY(sd1if >= 0, "");

  sd0osComp -= ifNDtrue[interfaceIndex];
  sd1osComp -= ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  sd0osComp -= ifespace[interfaceIndex]->GetTrueVSize();
  sd1osComp -= ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */
  
  Operator *Cij = (ifNDtrue[interfaceIndex] > 0) ? CreateCij(localInterfaceIndex, orientation) : new EmptyOperator();
  
  // Cij is in the local interface space only, mapping from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

  // Compose Cij on the left and right with injection operators between the subdomain surfaces and the interface.

  // Create right injection operator for sd1.

  if (orientation == 0 && realPart)
    {
      rowTrueOffsetsIFR[localInterfaceIndex].resize(2);
      colTrueOffsetsIFR[localInterfaceIndex].resize(2);
      rowTrueOffsetsIFL[localInterfaceIndex].resize(2);
      colTrueOffsetsIFL[localInterfaceIndex].resize(2);

      rowTrueOffsetsIFBR[localInterfaceIndex].resize(2);
      colTrueOffsetsIFBR[localInterfaceIndex].resize(2);
      rowTrueOffsetsIFBL[localInterfaceIndex].resize(2);
      colTrueOffsetsIFBL[localInterfaceIndex].resize(2);
    }
    
  const int numBlocks = 2;  // 1 for the subdomain surface, 1 for the interface (f_{mn} and \rho_{mn}).
  rowTrueOffsetsIFR[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
  colTrueOffsetsIFR[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

  rowTrueOffsetsIFR[localInterfaceIndex][orientation] = 0;
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */
      
  rowTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFR[localInterfaceIndex][orientation] = 0;
  colTrueOffsetsIFR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  colTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  colTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */
  
  colTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *rightInjection = new BlockOperator(rowTrueOffsetsIFR[localInterfaceIndex][orientation], colTrueOffsetsIFR[localInterfaceIndex][orientation]);

  if (tdofsBdryInjection[sd1] != NULL)
    {
      rightInjection->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd1][sd1if]),
							 tdofsBdryInjection[sd1], false, false));
    }
  rightInjection->SetBlock(1, 1, new IdentityOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));
  //rightInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

  // Create left injection operator for sd0.

  rowTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
  colTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

  rowTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
  rowTrueOffsetsIFL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex]; // + iH1fespace[interfaceIndex]->GetTrueVSize();
  //rowTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
  //rowTrueOffsetsIFL[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
  rowTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
  colTrueOffsetsIFL[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  /*
  colTrueOffsetsIFL[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
  */ 
  colTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *leftInjection = new BlockOperator(rowTrueOffsetsIFL[localInterfaceIndex][orientation], colTrueOffsetsIFL[localInterfaceIndex][orientation]);

  if (tdofsBdryInjectionTranspose[sd0] != NULL)
    {
      leftInjection->SetBlock(0, 0, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][sd0if], false, false));
    }
  
  leftInjection->SetBlock(1, 1, new IdentityOperator(ifNDtrue[interfaceIndex]));
  //leftInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

  TripleProductOperator *CijS = new TripleProductOperator(leftInjection, Cij, rightInjection, false, false, false);

  // CijS maps from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

  // Create block injection operator from (u^s, f_i) to (u^s, f_i, \rho_i) on sd0, where the range is over all sd0 interfaces.
    
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1
  colTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = sd0os;
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];
  //rowTrueOffsetsIFBL[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize();
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation][4] = sd0osComp;
    
  rowTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
  colTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  colTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  //colTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
    
  colTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *blockInjectionLeft = new BlockOperator(rowTrueOffsetsIFBL[localInterfaceIndex][orientation], colTrueOffsetsIFBL[localInterfaceIndex][orientation]);

  blockInjectionLeft->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd0].size()));
  blockInjectionLeft->SetBlock(2, 1, new IdentityOperator(ifNDtrue[interfaceIndex]));
  //blockInjectionLeft->SetBlock(2, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

  // Create block injection operator from (u^s, f_i, \rho_i) to (u^s, f_i, \rho_i) on sd1, where the domain is over all sd1 interfaces
  // and the range is only this one interface.

  rowTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1
  colTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
  rowTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  rowTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  //rowTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
  rowTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
  colTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  colTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = sd1os;
  //colTrueOffsetsIFBR[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIFBR[localInterfaceIndex][orientation][3] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  colTrueOffsetsIFBR[localInterfaceIndex][orientation][4] = sd1osComp;
    
  colTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *blockInjectionRight = new BlockOperator(rowTrueOffsetsIFBR[localInterfaceIndex][orientation], colTrueOffsetsIFBR[localInterfaceIndex][orientation]);

  blockInjectionRight->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd1].size()));
  blockInjectionRight->SetBlock(1, 2, new IdentityOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));
  //blockInjectionRight->SetBlock(1, 2, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

#ifdef EQUATE_REDUNDANT_VARS
  if (orientation == 1 && realPart)
    {
      rowTrueOffsetsIFRR[localInterfaceIndex].SetSize(3 + 1);
      colTrueOffsetsIFRR[localInterfaceIndex].SetSize(3 + 1);
	
      rowTrueOffsetsIFRR[localInterfaceIndex] = 0;
      rowTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd0].size() + sd0os + ifNDtrue[interfaceIndex];
      rowTrueOffsetsIFRR[localInterfaceIndex][2] = ifH1true[interfaceIndex];
      rowTrueOffsetsIFRR[localInterfaceIndex][3] = sd0osComp - ifH1true[interfaceIndex];
      /*
      rowTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd0].size() + sd0os + ifespace[interfaceIndex]->GetTrueVSize();
      rowTrueOffsetsIFRR[localInterfaceIndex][2] = iH1fespace[interfaceIndex]->GetTrueVSize();
      rowTrueOffsetsIFRR[localInterfaceIndex][3] = sd0osComp - iH1fespace[interfaceIndex]->GetTrueVSize();
      */
      
      rowTrueOffsetsIFRR[localInterfaceIndex].PartialSum();

      colTrueOffsetsIFRR[localInterfaceIndex] = 0;
      colTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd1].size() + sd1os + ifNDtrue[interfaceIndex];
      colTrueOffsetsIFRR[localInterfaceIndex][2] = ifH1true[interfaceIndex];
      colTrueOffsetsIFRR[localInterfaceIndex][3] = sd1osComp;
      /*
      colTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd1].size() + sd1os + ifespace[interfaceIndex]->GetTrueVSize();
      colTrueOffsetsIFRR[localInterfaceIndex][2] = iH1fespace[interfaceIndex]->GetTrueVSize();
      colTrueOffsetsIFRR[localInterfaceIndex][3] = sd1osComp;
      */
      
      colTrueOffsetsIFRR[localInterfaceIndex].PartialSum();

      BlockOperator *Irhorho = new BlockOperator(rowTrueOffsetsIFRR[localInterfaceIndex], colTrueOffsetsIFRR[localInterfaceIndex]);
      //Irhorho->SetBlock(1, 1, new IdentityOperator(iH1fespace[interfaceIndex]->GetTrueVSize()));
      Irhorho->SetBlock(1, 1, new IdentityOperator(ifH1true[interfaceIndex]));
	
      return new SumOperator(new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false), Irhorho, false, false, 1.0, 1.0);
    }
  else
    return new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false);
#else
  return new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false);
#endif
}

void DDMInterfaceOperator::CreateSubdomainMatrices(const int subdomain)
{
  ConstantCoefficient one(1.0);
  ConstantCoefficient minusk2(-k2);

  //fespace[subdomain]->GetComm()
  bf_sdND[subdomain] = new ParBilinearForm(fespace[subdomain]);  // TODO: make this a class member and delete at the end.
  bf_sdND[subdomain]->AddDomainIntegrator(new CurlCurlIntegrator(one));
  bf_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(minusk2));

  bf_sdND[subdomain]->Assemble();

  sdND[subdomain] = new HypreParMatrix();
  sdNDcopy[subdomain] = new HypreParMatrix();

  Array<int> ess_tdof_list;  // empty

  // Eliminate essential BC on exterior boundary.
  //MFEM_VERIFY(m_rank == 0, "");  // TODO: this code will not work in parallel, in general. 

  {
    const HYPRE_Int sdtos = fespace[subdomain]->GetMyTDofOffset();
    const int sdtsize = fespace[subdomain]->GetTrueVSize();
      
    Array<int> true_ess_dofs(sdtsize);
    for (int i=0; i<fespace[subdomain]->GetTrueVSize(); ++i)
      true_ess_dofs[i] = 0;

    for (set<int>::const_iterator it = tdofsBdry[subdomain].begin(); it != tdofsBdry[subdomain].end(); ++it)
      true_ess_dofs[*it] = 1;

    MFEM_VERIFY(InterfaceToSurfaceInjectionData[subdomain].size() == subdomainLocalInterfaces[subdomain].size(), "");
    MFEM_VERIFY(InterfaceToSurfaceInjectionGlobalData[subdomain].size() == subdomainLocalInterfaces[subdomain].size(), "");

    //set<int> interfaceTDofs;  // True DOF's of fespace[subdomain] which lie on an interface.

    for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];
	const int ifli = (*interfaceLocalIndex)[interfaceIndex];
	MFEM_VERIFY(ifli >= 0, "");

	for (int j=0; j<InterfaceToSurfaceInjectionGlobalData[subdomain][i].size(); ++j)
	  {
	    if (InterfaceToSurfaceInjectionGlobalData[subdomain][i][j] >= sdtos &&
		InterfaceToSurfaceInjectionGlobalData[subdomain][i][j] - sdtos < sdtsize)
	      true_ess_dofs[InterfaceToSurfaceInjectionGlobalData[subdomain][i][j] - sdtos] = 0;
	  }
	  
	/*
	  for (int j=0; j<InterfaceToSurfaceInjectionData[subdomain][i].size(); ++j)
	  {
	  //interfaceTDofs.insert(InterfaceToSurfaceInjectionData[subdomain][i][j]);
	  //if (InterfaceToSurfaceInjectionData[subdomain][i][j] < 0)
	  //cout << "BUG" << endl;
	      
	  if (InterfaceToSurfaceInjectionData[subdomain][i][j] >= 0)
	  true_ess_dofs[InterfaceToSurfaceInjectionData[subdomain][i][j]] = 0;
	  }
	*/
      }

    // At this point, true_ess_dofs[i] == 1 if and only if true DOF i is on the boundary minus the interface.

    const int numBdryFaces = pmeshSD[subdomain]->GetNBE();
    Array<int> essBdryFace(numBdryFaces);

    for (int i=0; i<numBdryFaces; ++i)
      {
	essBdryFace[i] = 0;

	Array<int> dofs;
	fespace[subdomain]->GetBdrElementDofs(i, dofs);

	for (int d=0; d<dofs.Size(); ++d)
	  {
	    const int dof_d = dofs[d] >= 0 ? dofs[d] : -1 - dofs[d];
	    const int ldof = fespace[subdomain]->GetLocalTDofNumber(dof_d);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.
	    if (ldof >= 0)
	      {
		if (true_ess_dofs[ldof] == 1)
		  essBdryFace[i] = 1;
	      }
	  }
      }

    // Now let true_ess_dofs be 1 for DOF's on the exterior boundary, including the boundary of the interface but not the interior 
    // of the interface.

    for (int i=0; i<numBdryFaces; ++i)
      {
	if (essBdryFace[i] == 1)
	  {
	    Array<int> dofs;
	    fespace[subdomain]->GetBdrElementDofs(i, dofs);

	    for (int d=0; d<dofs.Size(); ++d)
	      {
		const int dof_d = dofs[d] >= 0 ? dofs[d] : -1 - dofs[d];
		const int ldof = fespace[subdomain]->GetLocalTDofNumber(dof_d);  // If the DOF is owned by the current processor, return its local tdof number, otherwise -1.
		if (ldof >= 0)
		  {
		    true_ess_dofs[ldof] = 1;
		  }
	      }
	  }
      }

    fespace[subdomain]->MarkerToList(true_ess_dofs, ess_tdof_list);
  }

  bf_sdND[subdomain]->FormSystemMatrix(ess_tdof_list, *(sdND[subdomain]));
  bf_sdND[subdomain]->FormSystemMatrix(ess_tdof_list, *(sdNDcopy[subdomain]));  // TODO: is there a way to avoid making a copy of this matrix?

  /*
    { // debugging

    hypre_CSRMatrix* csrsdND = GetHypreParMatrixData(*(sdND[subdomain]));
    cout << csrsdND->num_rows << endl;
    }
  */
  /*
    {
    Vector zero(3);
    zero = 0.0;
    VectorConstantCoefficient vcc(zero);
    ParLinearForm *b = new ParLinearForm(fespace[subdomain]);
    b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vcc));
    b->Assemble();
      
    ParGridFunction xsd(fespace[subdomain]);
    xsd.ProjectCoefficient(vcc);
      
    Vector sdB, sdX;
    a.FormLinearSystem(ess_tdof_list, xsd, *b, *(sdND[subdomain]), sdX, sdB);
    delete b;
    }
  */
    
  // Add sum over all interfaces of 
  // -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
    
  //MFEM_VERIFY(false, "TODO: add boundary terms");
}

Solver* DDMInterfaceOperator::CreateSubdomainPreconditionerStrumpack(const int subdomain)
{
  const bool sdNull = (fespace[subdomain] == NULL);

  if (sdNull)
    {
      sdNDinv[subdomain] = NULL;
    }
  else
    {
      //Operator *A_subdomain = new STRUMPACKRowLocMatrix(*(sdND[subdomain]));
      Operator *A_subdomain = new STRUMPACKRowLocMatrix(*(A_SS[subdomain]));
      sdNDinv[subdomain] = CreateStrumpackSolver(A_subdomain, fespace[subdomain]->GetComm());
    }
    
  BlockDiagonalPreconditioner *op = new BlockDiagonalPreconditioner(trueOffsetsSD[subdomain]);

  if (!sdNull)
    {
      /*
	#ifdef EQUATE_REDUNDANT_VARS
	if (subdomain == 1)
	{
	Operator *Apen_subdomain = new STRUMPACKRowLocMatrix(*(sdNDPlusPen[subdomain]));
	op->SetDiagonalBlock(0, CreateStrumpackSolver(Apen_subdomain, fespace[subdomain]->GetComm()));
	}
	else
	#endif
      */
      {
	op->SetDiagonalBlock(0, sdNDinv[subdomain]);
      }
    }

  
  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      if (ifNDtrue[interfaceIndex] == 0)
	continue;

      // Diagonal blocks

      // Inverse of A_m^{FF}, which corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.

      /*
	Operator *A_FF_scaled = new STRUMPACKRowLocMatrix(*(ifNDmass[interfaceIndex]));  // Factor 1/alpha is inverted separately as a scalar multiple. 
	Operator *A_FF_scaled_solver = CreateStrumpackSolver(A_FF_scaled, ifespace[interfaceIndex]->GetComm());
	ScaledOperator *A_FF_solver = new ScaledOperator(A_FF_scaled_solver, alpha);
      */

#ifdef EQUATE_REDUNDANT_VARS
      if (subdomain == 1)
	{
	  op->SetDiagonalBlock((2*i) + 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));
	}
      else
#endif
	{
	  ScaledOperator *A_FF_solver = new ScaledOperator(ifNDmassInv[interfaceIndex], alpha);
	
	  op->SetDiagonalBlock((2*i) + 1, A_FF_solver);
	}
	
      // Inverse of A_m^{\rho\rho}, which corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

      /*
	Operator *A_rr = new STRUMPACKRowLocMatrix(*(ifH1mass[interfaceIndex]));
	Operator *A_rr_solver = CreateStrumpackSolver(A_rr, iH1fespace[interfaceIndex]->GetComm());

	op->SetDiagonalBlock((2*i) + 2, A_rr_solver);
      */

#ifdef EQUATE_REDUNDANT_VARS
      if (subdomain == 1)
	{
	  op->SetDiagonalBlock((2*i) + 2, new IdentityOperator(iH1fespace[interfaceIndex]->GetTrueVSize()));
	}
      else
#endif	
	{
	  op->SetDiagonalBlock((2*i) + 2, ifH1massInv[interfaceIndex]);
	}
    }

  return op;
}

void DDMInterfaceOperator::SetOffsetsSD(const int subdomain)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
  trueOffsetsSD[subdomain].SetSize(numBlocks + 1);  // Number of blocks + 1
  //for (int i=0; i<numBlocks + 1; ++i)
  //trueOffsetsSD[subdomain][i] = 0;

  const bool sdNull = (fespace[subdomain] == NULL);
    
  trueOffsetsSD[subdomain] = 0;
  trueOffsetsSD[subdomain][1] = (!sdNull) ? fespace[subdomain]->GetTrueVSize() : 0;
    
  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      /*
      MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
      MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
      */
      
      //trueOffsetsSD[subdomain][(2*i) + 2] += ifespace[interfaceIndex]->GetTrueVSize();
      //trueOffsetsSD[subdomain][(2*i) + 3] += iH1fespace[interfaceIndex]->GetTrueVSize();
      trueOffsetsSD[subdomain][(2*i) + 2] += ifNDtrue[interfaceIndex];
      trueOffsetsSD[subdomain][(2*i) + 3] += ifH1true[interfaceIndex];
    }
    
  trueOffsetsSD[subdomain].PartialSum();
    
  const bool debugging = false;
  if (debugging && subdomain == 1)
    {
      const int numTrueDofs = trueOffsetsSD[subdomain][numBlocks];

      std::vector<double> dofcrd(3*numTrueDofs);

      const double largeOffset = 1.0e7;
      const double vertexTol = 1.0e-8;
	
      for (int j=0; j<dofcrd.size(); ++j)
	dofcrd[j] = -largeOffset;
	
      ParMesh *sdMesh = fespace[subdomain]->GetParMesh();

      bool uniqueness = true;

      // Set coordinates of edge DOF's in subdomain
      for (int j=0; j<sdMesh->GetNEdges(); ++j)
	{
	  Array<int> dofs;

	  fespace[subdomain]->GetEdgeDofs(j, dofs);
	  MFEM_VERIFY(dofs.Size() == 1, ""); // only implemented for testing order 1 edge elements

	  Array<int> sdMeshEdgeVert;
	  sdMesh->GetEdgeVertices(j, sdMeshEdgeVert);
	  MFEM_VERIFY(sdMeshEdgeVert.Size() == 2, "");

	  MFEM_VERIFY(dofs[0] >= 0, "");
	    
	  const int sdtdof = fespace[subdomain]->GetLocalTDofNumber(dofs[0]);
	  if (sdtdof >= 0)
	    {
	      const int os = (3*sdtdof);
	      for (int l=0; l<3; ++l)
		{
		  const double mp_l = 0.5 * (sdMesh->GetVertex(sdMeshEdgeVert[0])[l] + sdMesh->GetVertex(sdMeshEdgeVert[1])[l]);
		  if (dofcrd[os + l] > -largeOffset + 1.0 && fabs(dofcrd[os + l] - mp_l) > vertexTol)
		    uniqueness = false;
		    
		  dofcrd[os + l] = mp_l;
		}
	    }
	}

      // Set coordinates of edge DOF's in interfaces, with a shift to distinguish them from subdomain edge DOF's.

      const double ifshift = 1000.0;

      for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
	{
	  const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];
	
	  ParMesh *ifMesh = ifespace[interfaceIndex]->GetParMesh();

	  for (int j=0; j<ifMesh->GetNEdges(); ++j)
	    {
	      Array<int> dofs;

	      ifespace[interfaceIndex]->GetEdgeDofs(j, dofs);
	      MFEM_VERIFY(dofs.Size() == 1, ""); // only implemented for testing order 1 edge elements

	      Array<int> edgeVert;
	      ifMesh->GetEdgeVertices(j, edgeVert);
	      MFEM_VERIFY(edgeVert.Size() == 2, "");

	      MFEM_VERIFY(dofs[0] >= 0, "");

	      const int iftdof = ifespace[interfaceIndex]->GetLocalTDofNumber(dofs[0]);
	      if (iftdof >= 0)
		{
		  const int os = 3 * (trueOffsetsSD[subdomain][(2*i) + 1] + iftdof);
		    
		  for (int l=0; l<3; ++l)
		    {
		      const double mp_l = (0.5 * (ifMesh->GetVertex(edgeVert[0])[l] + ifMesh->GetVertex(edgeVert[1])[l])) + ifshift;
		      if (dofcrd[os + l] > -largeOffset + 1.0 && fabs(dofcrd[os + l] - mp_l) > vertexTol)
			uniqueness = false;
		    
		      dofcrd[os + l] = mp_l;
		    }
		}
	    }

	  for (int j=0; j<ifMesh->GetNV(); ++j)
	    {
	      Array<int> dofs;

	      iH1fespace[interfaceIndex]->GetVertexDofs(j, dofs);
	      MFEM_VERIFY(dofs.Size() == 1, ""); // only implemented for testing order 1 nodal elements

	      MFEM_VERIFY(dofs[0] >= 0, "");

	      const int iftdof = iH1fespace[interfaceIndex]->GetLocalTDofNumber(dofs[0]);
	      if (iftdof >= 0)
		{
		  const int os = 3 * (trueOffsetsSD[subdomain][(2*i) + 2] + iftdof);
		    
		  for (int l=0; l<3; ++l)
		    {
		      if (dofcrd[os + l] > -largeOffset + 1.0 && fabs(dofcrd[os + l] - ifMesh->GetVertex(j)[l]) > vertexTol)
			uniqueness = false;
		    
		      dofcrd[os + l] = ifMesh->GetVertex(j)[l];
		    }
		}
	    }	    
	}

      for (int j=0; j<dofcrd.size(); ++j)
	{
	  if (dofcrd[j] < -largeOffset + 1.0)
	    uniqueness = false;
	}
	
      MFEM_VERIFY(uniqueness, "");
      /*
      // File output
      //ofstream file("dofcrd" + std::to_string(subdomain) + "Ser");
      ofstream file("dofcrd" + std::to_string(subdomain) + "Par" + std::to_string(m_rank));

      for (int j=0; j<numTrueDofs; ++j)
      {
      for (int k=0; k<3; ++k)
      file << dofcrd[(3*j) + k] << " ";
	    
      file << endl;
      }

      file.close();
      */
    }
}

Operator* DDMInterfaceOperator::CreateSubdomainOperator(const int subdomain)
{
  BlockOperator *op = new BlockOperator(trueOffsetsSD[subdomain]);

  {
    const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
    MFEM_VERIFY(numBlocks == trueOffsetsSD[subdomain].Size()-1, "");
  }
  
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)
    return NULL;  // Nothing to do, and the communicator is different for this process than that for processes with data.
  
  const double sd1pen = (subdomain == 1) ? PENALTY_U_S : 0.0;
    
#ifdef DDMCOMPLEX
  if (realPart)
#endif
    {
      if (sdND[subdomain] != NULL)
	{
#ifdef EQUATE_REDUNDANT_VARS_notusingthis
	  // TODO: this is a hack, assuming exactly one interface. In general, the penalties should be summed for all interfaces. 
	  MFEM_VERIFY(subdomainLocalInterfaces[subdomain].size() == 1, "");
	  if (subdomain == -1)
	    {
	      const int interfaceIndex = subdomainLocalInterfaces[subdomain][0];

	      if (sdNDPen[subdomain] == NULL)
		{
		  HYPRE_Int size = sdND[subdomain]->GetGlobalNumRows();

		  int nsdprocs, sdrank;
		  MPI_Comm_size(fespace[subdomain]->GetComm(), &nsdprocs);
		  MPI_Comm_rank(fespace[subdomain]->GetComm(), &sdrank);

		  int num_loc_rows = fespace[subdomain]->GetTrueVSize();
		    
		  int *all_num_loc_rows = new int[nsdprocs];
		    
		  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, fespace[subdomain]->GetComm());

		  int sumLocalSizes = 0;

		  for (int i=0; i<nsdprocs; ++i)
		    sumLocalSizes += all_num_loc_rows[i];

		  MFEM_VERIFY(size == sumLocalSizes, "");

		  Vector diag(fespace[subdomain]->GetTrueVSize());
		  diag = 0.0;

		  Vector ones(ifespace[interfaceIndex]->GetTrueVSize());
		  ones = PENALTY_U_S;
		    
		  InterfaceToSurfaceInjection[subdomain][0]->Mult(ones, diag);
		    
		  HYPRE_Int *rowStarts = new HYPRE_Int[nsdprocs+1];
		  rowStarts[0] = 0;
		  for (int i=0; i<nsdprocs; ++i)
		    rowStarts[i+1] = rowStarts[i] + all_num_loc_rows[i];

		  const int osj = rowStarts[sdrank];
		    
		  /*
		    sdNDPenSp[subdomain] = new SparseMatrix(diag);
		    sdNDPenSp[subdomain]->Finalize();
		    
		    sdNDPen[subdomain] = new HypreParMatrix(fespace[subdomain]->GetComm(), size, rowStarts, sdNDPenSp[subdomain]);
		  */
		  /*		    
				    SparseMatrix diagWrap, spdiag;
				    sdND[subdomain]->GetDiag(diagWrap);
				    spdiag = diagWrap;  // Deep copy

				    sdNDPen[subdomain] = new HypreParMatrix(fespace[subdomain]->GetComm(), size, rowStarts, &spdiag);
		  */

		  int *I_nnz = new int[num_loc_rows + 1];
		  HYPRE_Int *J_col = new HYPRE_Int[num_loc_rows];
		    
		  for (int i=0; i<num_loc_rows + 1; ++i)
		    I_nnz[i] = i;

		  for (int i=0; i<num_loc_rows; ++i)
		    J_col[i] = osj + i;
		    
		  //sdNDPen[subdomain] = new HypreParMatrix(fespace[subdomain]->GetComm(), num_loc_rows, size, size, I_nnz, J_col, diag.GetData(), rowStarts, rowStarts);
		    
		  //sdNDPlusPen[subdomain] = ParAdd(sdND[subdomain], sdNDPen[subdomain]);
		  sdNDPlusPen[subdomain] = sdND[subdomain];

		  delete I_nnz;
		  delete J_col;
		    
		  delete rowStarts;
		  delete all_num_loc_rows;
		}

	      op->SetBlock(0, 0, sdNDPlusPen[subdomain]);

	      /*
		op->SetBlock(0, 0, new SumOperator(sdND[subdomain], new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][0],
		new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
		new TransposeOperator(InterfaceToSurfaceInjection[subdomain][0]),
		false, false, false), false, false, 1.0, PENALTY_U_S));
	      */
	    }
	  else
#endif
	    {
	      // In PengLee2012 A_m^{SS} corresponds to a(v, u) - alpha <\pi_{mn}(v_m), [[u]]_{mn}>_{S_{mn}}
	      // -beta <curl_\tau \pi_{mn}(v_m), curl_\tau [[u]]_{mn}>_{S_{mn}}
	      // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{SS} block is the part
	      // a(v, u) - alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
	      // The last two terms are an interface mass plus curl-curl stiffness matrix. These must be added in a loop over interfaces.

	      op->SetBlock(0, 0, sdND[subdomain]);
	    }
	}
    }

  Operator *A_SS_op = NULL;
    
  if (preconditionerMode)
    MFEM_VERIFY(A_SS[subdomain] == NULL, "");
    
  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A_m^{SS}.
#ifdef DDMCOMPLEX
      if (realPart)
	{
	  if (preconditionerMode)
	    {
	      // Create new HypreParMatrix for the sum
	      HypreParMatrix *sumMassCC = NULL;
	      if (ifNDtrue[interfaceIndex] > 0)
		{
		  (*(ifNDmass[interfaceIndex])) *= -alpha;
		  (*(ifNDcurlcurl[interfaceIndex])) *= -beta;

		  sumMassCC = ParAdd(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex]);
		    
		  (*(ifNDmass[interfaceIndex])) *= -1.0 / alpha;
		  (*(ifNDcurlcurl[interfaceIndex])) *= -1.0 / beta;
		}

	      if (A_SS[subdomain] == NULL)
		{
		  A_SS[subdomain] = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdNDcopy[subdomain], sumMassCC, InterfaceToSurfaceInjectionData[subdomain][i],
									 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									 ifespace[interfaceIndex], NULL, true, PENALTY_U_S);
		}
	      else
		{
		  // TODO: add another interface operator to A_SS
		  //MFEM_VERIFY(false, "TODO");

		  HypreParMatrix *previousSum = A_SS[subdomain];
		  
		  A_SS[subdomain] = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], previousSum, sumMassCC, InterfaceToSurfaceInjectionData[subdomain][i],
									 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									 ifespace[interfaceIndex], NULL, true, PENALTY_U_S);

		  delete previousSum;
		}

	      delete sumMassCC;

	      if (A_SS[subdomain] != NULL)
		op->SetBlock(0, 0, A_SS[subdomain]);
	    }
	  else
	    {
	      // Sum operators abstractly, without adding matrices into a new HypreParMatrix.
	      if (A_SS_op == NULL)
		{
		  if (ifNDtrue[interfaceIndex] == 0)
		    A_SS_op = sdND[subdomain];
		  else
		    {
		      Operator *ifop = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][i],
#ifdef EQUATE_REDUNDANT_VARS
								 new SumOperator(new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
										 new SumOperator(ifNDmass[interfaceIndex],
												 ifNDcurlcurl[interfaceIndex],
												 false, false, -alpha, -beta),
										 false, false, sd1pen, 1.0),
#else
								 new SumOperator(ifNDmass[interfaceIndex],
										 ifNDcurlcurl[interfaceIndex],
										 false, false, -alpha, -beta),
#endif
								 new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]),
								 false, false, false);

		      A_SS_op = (sdND[subdomain] == NULL) ? ifop : new SumOperator(sdND[subdomain], ifop, false, false, 1.0, 1.0);
		    }
		}
	      else
		{
		  // TODO: add another interface operator to A_SS_op
		  // MFEM_VERIFY(false, "TODO");
		  // TODO: this may be a memory leak, but its magnitude seems tolerable (each SumOperator just stores 2 vectors).
		  if (ifNDtrue[interfaceIndex] > 0)
		    {
		      A_SS_op = new SumOperator(A_SS_op,
						new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][i],
#ifdef EQUATE_REDUNDANT_VARS
									  new SumOperator(new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
											  new SumOperator(ifNDmass[interfaceIndex],
													  ifNDcurlcurl[interfaceIndex],
													  false, false, -alpha, -beta),
											  false, false, sd1pen, 1.0),
#else
									  new SumOperator(ifNDmass[interfaceIndex],
											  ifNDcurlcurl[interfaceIndex],
											  false, false, -alpha, -beta),
#endif
									  new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]),
									  false, false, false),
						false, false, 1.0, 1.0);
		    }
		}

	      if (A_SS_op != NULL)
		op->SetBlock(0, 0, A_SS_op);
	    }
	}
      else  // if not real part
	{
	  // Sum operators abstractly, without adding matrices into a new HypreParMatrix.
	  if (A_SS_op == NULL)
	    {
	      if (ifNDtrue[interfaceIndex] > 0)
		{
		  A_SS_op = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][i],
						      new SumOperator(ifNDmass[interfaceIndex],
								      ifNDcurlcurl[interfaceIndex],
								      false, false, -alpha, -beta),
						      new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]),
						      false, false, false);
		}
	    }
	  else
	    {
	      // TODO: add another interface operator to A_SS_op
	      //MFEM_VERIFY(false, "TODO");

	      // TODO: this may be a memory leak, but its magnitude seems tolerable (each SumOperator just stores 2 vectors).
	      if (ifNDtrue[interfaceIndex] > 0)
		{
		  A_SS_op = new SumOperator(A_SS_op, new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][i],
									       new SumOperator(ifNDmass[interfaceIndex],
											       ifNDcurlcurl[interfaceIndex],
											       false, false, -alpha, -beta),
									       new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]),
									       false, false, false),
					    false, false, 1.0, 1.0);
		}
	    }
	  
	  if (A_SS_op != NULL)
	    op->SetBlock(0, 0, A_SS_op);
	}
#else
      // TODO: implement A_SS in the real case. 
#endif

      // In PengLee2012 A_m^{SF} corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0.  TODO: verify this. The paper does not say this block is 0.
	
      // op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);

      // In PengLee2012 A_m^{S\rho} corresponds to
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

      if (ifNDtrue[interfaceIndex] == 0)
	continue;

#ifdef SCHURCOMPSD
      // Modify the A_m^{SF} block by subtracting A_m^{S\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{S\rho} block
      op->SetBlock(0, (2*i) + 1, new TripleProductOperator(new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex],
									       false, false),
							   ifH1massInv[interfaceIndex], ifNDH1gradT[interfaceIndex], false, false, false), gamma);
#else
      op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex], false, false), -gamma);
#endif
	
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

#ifdef SCHURCOMPSD
      // Modify the A_m^{FF} block by subtracting A_m^{F\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{F\rho} block
#else
#ifdef ELIMINATE_REDUNDANT_VARS
      if (subdomain == 0)
#endif
	{
	  op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gammaOverAlpha);
	}
#endif
	
      // In PengLee2012 A_m^{FS} corresponds to
      // <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
      // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
      // <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
      // This is an interface mass plus curl-curl stiffness matrix.

#ifdef ELIMINATE_REDUNDANT_VARS
      if (subdomain == 0)
#endif
	{
#ifdef DDMCOMPLEX
	  if (realPart)
#endif
	    {
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, betaOverAlpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false));
	    }
#ifdef DDMCOMPLEX
	  else
	    {
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(ifNDcurlcurl[interfaceIndex], new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false),  betaOverAlpha);
	    }
#endif
	}

      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      //op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!
#ifdef ELIMINATE_REDUNDANT_VARS
      if (subdomain == 0)
#endif	  
	{
#ifdef DDMCOMPLEX
	  if (realPart)
#endif
	    {
	      op->SetBlock((2*i) + 2, (2*i) + 1, ifNDH1gradT[interfaceIndex]);
	    }
	}

      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.
#ifdef SCHURCOMPSD
      // Modify the A_m^{FF} block by subtracting A_m^{F\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{F\rho} block
      op->SetBlock((2*i) + 1, (2*i) + 1, new SumOperator(ifNDmass[interfaceIndex],
							 new TripleProductOperator(ifNDH1grad[interfaceIndex], 
										   ifH1massInv[interfaceIndex],
										   ifNDH1gradT[interfaceIndex],
										   false, false, false),
							 false, false, 1.0 / alpha, -gamma / alpha));
#else
#if defined ELIMINATE_REDUNDANT_VARS || defined EQUATE_REDUNDANT_VARS
      if (realPart && subdomain == 1)
	{
	  op->SetBlock((2*i) + 1, (2*i) + 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));
	}
	
      if (subdomain == 0)
#endif
	{
	  op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], alphaInverse);
	}
#endif

      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#if defined ELIMINATE_REDUNDANT_VARS || defined EQUATE_REDUNDANT_VARS
	  if (subdomain == 1)
	    {
	      op->SetBlock((2*i) + 2, (2*i) + 2, new IdentityOperator(iH1fespace[interfaceIndex]->GetTrueVSize()));
	    }
	  else
#endif
	    {
	      op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
	    }
	}

      // TODO: should we equate redundant corner DOF's for f and \rho?
    }
    
  return op;
}

void DDMInterfaceOperator::CreateSubdomainHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block, Array2D<double>& blockCoefficient)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).

  block.SetSize(numBlocks, numBlocks);
  blockCoefficient.SetSize(numBlocks, numBlocks);

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  block(i, j) = NULL;
	  blockCoefficient(i, j) = 1.0;
	}
    }

  const int locSizeSD = trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1];
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)
    return;  // Nothing to do, and the communicator is different for this process than that for processes with data. 
  
  // Set blocks

#ifdef DDMCOMPLEX
  if (realPart)
#endif
    {
      if (sdND[subdomain] != NULL)
	{
	  block(0, 0) = sdND[subdomain];
	}
    }

  HypreParMatrix *A_SS_op = NULL;

  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A_m^{SS}.
#ifdef DDMCOMPLEX
      HypreParMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(-alpha, *(ifNDmass[interfaceIndex]), -beta, *(ifNDcurlcurl[interfaceIndex]));

      // TODO: the following are wrong, just for testing!
      //HypreParMatrix *ifSum = ParAdd(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex]);
      //HypreParMatrix *ifSum = Add(1.0, *(ifNDmass[interfaceIndex]), 1.0, *(ifNDcurlcurl[interfaceIndex]));
      {
	//ifNDmass[interfaceIndex]->Print("ifNDmass");
	//ifNDcurlcurl[interfaceIndex]->Print("ifNDcc");
      }
	
      /*
	{
	std::string fname("ifsumAdd");
	ifSum->Print(fname.c_str());
	}
      */
	
      const double sdNDcoef = realPart ? 1.0 : 0.0;
	
      if (A_SS_op == NULL)
	{
	  // TODO: try with sdND, to see if sdNDcopy is unnecessary.
	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdNDcopy[subdomain], ifSum, InterfaceToSurfaceInjectionData[subdomain][i],
							 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 ifespace[interfaceIndex], NULL, true, 0.0, sdNDcoef);
	}
      else
	{
	  // TODO: add another interface operator to A_SS_op
	  //MFEM_VERIFY(false, "TODO");
	  
	  HypreParMatrix *previousSum = A_SS_op;

	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], previousSum, ifSum, InterfaceToSurfaceInjectionData[subdomain][i],
							 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 ifespace[interfaceIndex], NULL, true, 0.0, sdNDcoef);

	  delete previousSum;
	}

      if (A_SS_op != NULL)
	block(0, 0) = A_SS_op;

      if (ifSum != NULL)
	delete ifSum;
#else
      // TODO: implement A_SS in the real case. 
#endif

      // In PengLee2012 A_m^{SF} corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0.  TODO: verify this. The paper does not say this block is 0.
	
      // op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);

      // In PengLee2012 A_m^{S\rho} corresponds to
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

      HypreParMatrix* injectedTr = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdNDcopy[subdomain], ifNDH1grad[interfaceIndex],
									InterfaceToSurfaceInjectionData[subdomain][i],
									InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									ifespace[interfaceIndex], iH1fespace[interfaceIndex], false);
									  
      /*
	{
	// Test injectedTr
	Vector t1(iH1fespace[interfaceIndex]->GetTrueVSize());
	Vector t2(fespace[subdomain]->GetTrueVSize());

	Vector t3(iH1fespace[interfaceIndex]->GetTrueVSize());
	Vector t4(ifespace[interfaceIndex]->GetTrueVSize());

	for (auto j=0; j<ifespace[interfaceIndex]->GetVSize(); j++)  // loop over interface full DOF's
	{
	const int sdtdof = InterfaceToSurfaceInjectionData[subdomain][i][j];
	const int ltdof = ifespace[interfaceIndex]->GetLocalTDofNumber(j);
  
	// This test may be valid only in serial.
	      
	if (sdtdof >= 0 && ltdof >= 0)
	{
	t1 = 0.0;
	t2 = 0.0;
		  
	t2[sdtdof] = 1.0;

	injectedTr->Mult(t2, t1);

	t3 = 0.0;
	t4 = 0.0;
		  
	t4[ltdof] = 1.0;
	ifNDH1grad[interfaceIndex]->MultTranspose(t4, t3);

	const double t3norm = t3.Norml2();
	t3 -= t1;

	cout << "Test ifdof " << j << ": diff " << t3.Norml2() << " relative to " << t3norm << endl;
	}
	}
	}
      */

      if (injectedTr != NULL)
	{
#ifdef MIXED_MATRIX_NO_TRANSPOSE
	  block(0, (2*i) + 2) = injectedTr;
#else
	  block(0, (2*i) + 2) = injectedTr->Transpose();  // not the most efficient implementation, to avoid premature optimization.
	  delete injectedTr;
#endif

	  blockCoefficient(0, (2*i) + 2) = -gamma;
	}
	
	
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      if (ifNDH1grad[interfaceIndex] != NULL)
	{
	  block((2*i) + 1, (2*i) + 2) = ifNDH1grad[interfaceIndex];
	  blockCoefficient((2*i) + 1, (2*i) + 2) = gammaOverAlpha;
	}
      
      // In PengLee2012 A_m^{FS} corresponds to
      // <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
      // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
      // <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
      // This is an interface mass plus curl-curl stiffness matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
	  
	  ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(1.0, *(ifNDmass[interfaceIndex]), betaOverAlpha, *(ifNDcurlcurl[interfaceIndex]));
	  block((2*i) + 1, 0) = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdNDcopy[subdomain], ifSum,
								     InterfaceToSurfaceInjectionData[subdomain][i],
								     InterfaceToSurfaceInjectionGlobalData[subdomain][i],
								     ifespace[interfaceIndex], NULL, false,
								     0.0, 1.0, false);
	  
	  if (ifSum != NULL)
	    delete ifSum;
	}
#ifdef DDMCOMPLEX
      else
	{
	  block((2*i) + 1, 0) = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdNDcopy[subdomain], ifNDcurlcurl[interfaceIndex],
								     InterfaceToSurfaceInjectionData[subdomain][i],
								     InterfaceToSurfaceInjectionGlobalData[subdomain][i],
								     ifespace[interfaceIndex], NULL, false,
								     0.0, 1.0, false);
	  blockCoefficient((2*i) + 1, 0) = betaOverAlpha;
	}
#endif

      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      //op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!

      if (ifNDtrue[interfaceIndex] == 0)
	continue;
      
#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
	  block((2*i) + 2, (2*i) + 1) = ifNDH1gradT[interfaceIndex];
	}

      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.
      {
	block((2*i) + 1, (2*i) + 1) = ifNDmass[interfaceIndex];
	blockCoefficient((2*i) + 1, (2*i) + 1) = alphaInverse;
      }

      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
	  block((2*i) + 2, (2*i) + 2) = ifH1mass[interfaceIndex];
	}
    }    
}

Operator* DDMInterfaceOperator::CreateSubdomainOperatorStrumpack(const int subdomain)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).

  /*
    const int num_loc_rows = trueOffsetsSD[subdomain][numBlocks];

    int nprocs, rank;
    MPI_Comm_rank(fespace[subdomain]->GetComm(), &rank);
    MPI_Comm_size(fespace[subdomain]->GetComm(), &nprocs);

    int *all_num_loc_rows = new int[nprocs];

    MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, fespace[subdomain]->GetComm());

    int first_loc_row = 0;
    int glob_nrows = 0;
    for (int i=0; i<nprocs; ++i)
    {
    glob_nrows += all_num_loc_rows[i];
    if (i < rank)
    first_loc_row += all_num_loc_rows[i];
    }

    delete all_num_loc_rows;
    
    const int glob_ncols = glob_nrows;
  */
    
  Array2D<HypreParMatrix*> blocks(numBlocks, numBlocks);
  Array2D<std::vector<int>*> blockLeftInjection(numBlocks, numBlocks);
  Array2D<std::vector<int>*> blockRightInjection(numBlocks, numBlocks);
  Array2D<double> blockCoefficient(numBlocks, numBlocks);

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  blocks(i, j) = NULL;
	  blockLeftInjection(i, j) = NULL;
	  blockRightInjection(i, j) = NULL;
	  blockCoefficient(i, j) = 1.0;
	}
    }

  // Set blocks
  blocks(0, 0) = sdND[subdomain];

  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      // In PengLee2012 A_m^{SF} corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0.  TODO: verify this. The paper does not say this block is 0.
	
      // op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);

      // In PengLee2012 A_m^{S\rho} corresponds to
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
      // op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDH1grad[interfaceIndex], false, false), -gamma);
      //blocks(0, (2*i) + 2) = sdifNDH1grad[interfaceIndex];
      blockCoefficient(0, (2*i) + 2) = -gamma;
      //blockLeftInjection(0, (2*i) + 2) = &(InterfaceToSurfaceInjectionData[subdomain][i]);

      // TODO: set factor and permutation and transpose

      //MFEM_VERIFY(false, "TODO");

	
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

      // op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gamma / alpha);
      blocks((2*i) + 1, (2*i) + 2) = ifNDH1grad[interfaceIndex];
      blockCoefficient((2*i) + 1, (2*i) + 2) = gamma / alpha;
	
      // In PengLee2012 A_m^{FS} corresponds to
      // <w_m^s, [[u]]_{mn}>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau [[u]]_{mn}>_{S_{mn}}
      // Since [[u]]_{mn} = \pi_{mn}(u_m) - \pi_{nm}(u_n), the A_m^{FS} block is the part
      // <w_m, \pi_{mn}(u_m)>_{S_{mn}} + beta/alpha <curl_\tau w_m, curl_\tau \pi_{mn}(u_m)>_{S_{mn}}
      // This is an interface mass plus curl-curl stiffness matrix.

      // op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, beta / alpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]), false, false));

      blocks((2*i) + 1, 0) = ifND_FS[interfaceIndex];

      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      // op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));
	
      blocks((2*i) + 2, (2*i) + 1) = ifNDH1gradT[interfaceIndex];
	
      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.

      //op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], 1.0 / alpha);
      blocks((2*i) + 1, (2*i) + 1) = ifNDmass[interfaceIndex];
      blockCoefficient((2*i) + 1, (2*i) + 1) = 1.0 / alpha;
	
      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

      //op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
      blocks((2*i) + 2, (2*i) + 2) = ifH1mass[interfaceIndex];
	
      // TODO: should we equate redundant corner DOF's for f and \rho?
    }

  return CreateStrumpackMatrixFromHypreBlocks(fespace[subdomain]->GetComm(), trueOffsetsSD[subdomain],
					      blocks, blockLeftInjection, blockRightInjection, blockCoefficient);
}
