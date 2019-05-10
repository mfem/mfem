#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

#define ZERO_RHO_BC
#define ZERO_IFND_BC

//#define ELIMINATE_REDUNDANT_VARS
//#define EQUATE_REDUNDANT_VARS

#define SPARSE_ASDCOMPLEX

#define PENALTY_U_S 0.0

void test1_E_exact(const Vector &x, Vector &E)
{
  const double kappa = M_PI;
  E(0) = sin(kappa * x(1));
  E(1) = sin(kappa * x(2));
  E(2) = sin(kappa * x(0));
}

void test1_RHS_exact(const Vector &x, Vector &f)
{
  const double kappa = M_PI;
  const double sigma = -2.0;
  f(0) = (sigma + kappa * kappa) * sin(kappa * x(1));
  f(1) = (sigma + kappa * kappa) * sin(kappa * x(2));
  f(2) = (sigma + kappa * kappa) * sin(kappa * x(0));
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
}

void test2_RHS_exact(const Vector &x, Vector &f)
{
  const double pi = M_PI;
  const double sigma = -2.0;
  const double c = (2.0 * pi * pi) + sigma;

  f(0) = c * sin(pi * x(1)) * sin(pi * x(2));
  f(1) = c * sin(pi * x(2)) * sin(pi * x(0));
  f(2) = c * sin(pi * x(0)) * sin(pi * x(1));
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
	  if (y[i] != 0.0)
	    {
	      S->Set(i, j, y[i]);
	    }
	}
    }
      
  S->Finalize();

  return S;
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

  const HYPRE_Int globalSize = A->GetGlobalNumRows();
  
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
  
  MFEM_VERIFY(sum == globalSize, "");
  
  int *nnz = (nrows > 0) ? new int[nrows] : NULL;
  int *allnnz = (rank == 0) ? new int[globalSize+1] : NULL;

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
      for (int i=globalSize-1; i>=0; --i)
	allnnz[i+1] = allnnz[i];

      allnnz[0] = 0;
      
      // Now allnnz[i] is nnz for row i-1. Do a partial sum to get offsets.
      for (int i=0; i<globalSize; ++i)
	allnnz[i+1] += allnnz[i];
      
      for (int i=0; i<nprocs; ++i)
	{
	  globaltotalnnz += alltotalnnz[i];
	  if (i<nprocs-1)
	    dspl[i+1] = dspl[i] + alltotalnnz[i];
	}

      MFEM_VERIFY(allnnz[globalSize] == globaltotalnnz, "");
      
      allj = new int[globaltotalnnz];
      alldata = new double[globaltotalnnz];

      S = new SparseMatrix(globalSize, globalSize);
    }

  MPI_Gatherv(csr->j, totalnnz, MPI_INT, allj, alltotalnnz, dspl, MPI_INT, 0, A->GetComm());
  MPI_Gatherv(csr->data, totalnnz, MPI_DOUBLE, alldata, alltotalnnz, dspl, MPI_DOUBLE, 0, A->GetComm());

  //int * = (rank == 0) ? new int[globalSize+1] : NULL;
  if (rank == 0)
    {
      for (int i=0; i<globalSize; ++i)
	{
	  for (int k=allnnz[i]; k<allnnz[i+1]; ++k)
	    S->Set(i, allj[k], alldata[k]);
	  
	  //S->Add(i, j, 0.0);
	}
      
      S->Finalize();

      bool nnzmatch = true;
      for (int i=0; i<globalSize; ++i)
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

SparseMatrix* ReceiveSparseMatrix(const int source)
{
  int sizes[2];  // {size of matrix, number of nonzeros}

  MPI_Recv(sizes, 2, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MFEM_VERIFY(sizes[0] > 0 && sizes[1] > 0, "");
  
  int *I = new int[sizes[0]+1];
  int *J = new int[sizes[1]];
  double *data = new double[sizes[1]];

  MPI_Recv(I, sizes[0]+1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(J, sizes[1], MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(data, sizes[1], MPI_DOUBLE, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MFEM_VERIFY(sizes[1] == I[sizes[0]], "");

  SparseMatrix *S = new SparseMatrix(sizes[0], sizes[0]);
  
  for (int k=0; k<sizes[0]; ++k)  // loop over rows
    {
      for (int l=I[k]; l<I[k+1]; ++l)  // loop over nonzeros
	{
	  S->Set(k, J[l], data[l]);
	}
    }

  delete I;
  delete J;
  delete data;

  S->Finalize();

  return S;
}

void SendSparseMatrix(SparseMatrix *S, const int dest)
{
  int sizes[2];

  sizes[0] = S->Size();  // size of matrix
  sizes[1] = S->GetI()[sizes[0]];  // number of nonzeros
  
  MPI_Send(sizes, 2, MPI_INT, dest, 0, MPI_COMM_WORLD);

  MPI_Send(S->GetI(), sizes[0]+1, MPI_INT, dest, 1, MPI_COMM_WORLD);
  MPI_Send(S->GetJ(), sizes[1], MPI_INT, dest, 2, MPI_COMM_WORLD);
  MPI_Send(S->GetData(), sizes[1], MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
}

// We assume here that the sparsity structure of the interface matrix I is contained in that of the subdomain matrix A.
// We simply add entries from I to existing entries of A. The constant cI times the interface identity matrix is added.
HypreParMatrix* AddSubdomainMatrixAndInterfaceMatrix(HypreParMatrix *A, HypreParMatrix *I, std::vector<int> & inj,
						     ParFiniteElementSpace *ifespace, const double cI)
{
  // inj maps from full local interface DOF's to local true DOF's in the subdomain. 

  // Gather the entire global interface matrix to one root process, namely rank 0 in the ifespace communicator.

  SparseMatrix *globalI = NULL;
  if (ifespace != NULL)
    {
      globalI = GatherHypreParMatrix(I);

      if (globalI != NULL)
	{
	  for (int i=0; i<globalI->Size(); ++i)
	    {
	      globalI->Add(i, i, cI);
	    }
	}
    }

  // Find the unique rank in MPI_COMM_WORLD of the ifespace rank 0 process that stores globalI. 

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int A_rank;
  MPI_Comm_rank(A->GetComm(), &A_rank);
  
  int *allg = new int[nprocs];
  
  const int haveGlobalI = (globalI != NULL) ? 1 : 0;
  
  MPI_Allgather(&haveGlobalI, 1, MPI_INT, allg, 1, MPI_INT, MPI_COMM_WORLD);

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

  MFEM_VERIFY(ownerGlobalI >= 0 && count == 1, "");
  
  const int haveA = (A != NULL) ? 1 : 0;
  
  MPI_Allgather(&haveA, 1, MPI_INT, allg, 1, MPI_INT, MPI_COMM_WORLD);

  count = 0;
  for (int i=0; i<nprocs; ++i)
    {
      if (allg[i] == 1)
	{
	  count++;
	}
    }

  MFEM_VERIFY(count > 0, "");

  // Send entire global interface matrix to all processes in the subdomain communicator given by A.
  // Note that the processes owing the interface matrix I may be different from those in the subdomain communicator. 

  if (haveGlobalI)
    {
      for (int i=0; i<nprocs; ++i)
	{
	  if (i != rank && allg[i] == 1)
	    SendSparseMatrix(globalI, i);
	}
    }
  else if (haveA)
    {
      globalI = ReceiveSparseMatrix(ownerGlobalI);

      /*
      std::string filename = "spmat_r" + std::to_string(rank) + ".txt";
      std::ofstream sfile(filename, std::ofstream::out);
      
      globalI->PrintMatlab(sfile);
      sfile.close();
      */
    }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  // First cast the matrix A to a hypre_ParCSRMatrix
  //hypre_ParCSRMatrix * A_parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*A);

  HypreParMatrix *Acopy = A;  // TODO: make a copy of A
  //HypreParMatrix Acopy(A_parcsr_op);
  
  hypre_ParCSRMatrix * parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*Acopy);
  
  MFEM_ASSERT(parcsr_op != NULL,"STRUMPACK: const_cast failed in SetOperator");

  // Create the CSRMatrixMPI A_ by borrowing the internal data from a hypre_CSRMatrix.
  hypre_CSRMatrix * csr_op = hypre_MergeDiagAndOffd(parcsr_op);

  cout << rank << ": num rows " << csr_op->num_rows << ", RP0 " << A->RowPart()[0] << ", RP1 " << A->RowPart()[1] << endl;

  if (A != NULL)
    MFEM_VERIFY(csr_op->num_rows == A->RowPart()[1] - A->RowPart()[0], "");
  
  // Create map from local true interface DOF's to local full interface DOF's. Composing inj with that map takes indices
  // from the local rows of the global interface matrix to local true DOF's in the subdomain. Gathering the result over
  // all processes yields a map from all global interface DOF's to global true subdomain DOF's.
  
  const int iftsize = (ifespace == NULL) ? 0 : ifespace->GetTrueVSize();
  const int ifullsize = (ifespace == NULL) ? 0 : ifespace->GetVSize();

  int *ifullsdt = (ifullsize > 0) ? new int[ifullsize] : NULL;  // interface full to subdomain true DOF map, well-defined as the max of inj over all processes.

  // Create map from interface full DOF's to global subdomain true DOF's. First, verify that global true DOF's go from 0 to the global matrix size minus 1.
  bool gtdofInRange = true;
  bool maxGlobalTDofFound = false;
  const int igsize = I->GetGlobalNumRows();
  const int maxgtdof = igsize - 1;

  int alligsize = -1;
  MPI_Allreduce(&igsize, &alligsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  MFEM_VERIFY(alligsize > 0, "");

  if (globalI != NULL)
    MFEM_VERIFY(igsize == globalI->Size(), "");
  
  int *ifgtsdt = new int[alligsize];  // interface global true to subdomain true DOF's.
  int *maxifgtsdt = new int[alligsize];  // interface global true to subdomain true DOF's.
  for (int i=0; i<alligsize; ++i)
    {
      ifgtsdt[i] = -1;
      maxifgtsdt[i] = -1;
    }

  const int ossdt = (A != NULL) ? A->RowPart()[0] : 0;
  const int ossdtNext = (A != NULL) ? A->RowPart()[1] : 0;
  
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

  MPI_Allreduce(&maxGlobalTDofFound, &allMaxGlobalTDofFound, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
  
  MFEM_VERIFY(gtdofInRange && allMaxGlobalTDofFound, "");

  MPI_Allreduce(ifgtsdt, maxifgtsdt, alligsize, MPI_INT, MPI_MAX, MPI_COMM_WORLD);  // TODO: can this be done in place?
  
  int *iftfull = (iftsize > 0) ? new int[iftsize] : NULL;
  int *iftsdt = (iftsize > 0) ? new int[iftsize] : NULL;  // interface true to subdomain true DOF map

  for (int i=0; i<iftsize; ++i)
    {
      iftfull[i] = -1;
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

  for (int i=0; i<iftsize; ++i)
    {
      MFEM_VERIFY(iftfull[i] >= 0, "");

      iftsdt[i] = ifullsdt[iftfull[i]];  // well-defined
    }

  if (globalI != NULL && A != NULL) // Add values from globalI to the local rows of csr_op
    {
      MFEM_VERIFY(globalI->Size() == igsize, "");

      bool allEntriesFound = true;
      
      for (int i=0; i<igsize; ++i)
	{
	  if (ifgtsdt[i] >= ossdt && ifgtsdt[i] < ossdtNext)  // if subdomain true DOF is on this process
	    {
	      const int ltsd = ifgtsdt[i] - ossdt;  // local true subdomain DOF
	      // Add values in row i of globalI to values in row ltsd of csr_op

	      for (int k=globalI->GetI()[i]; k<globalI->GetI()[i+1]; ++k)
		{
		  const int igcol = globalI->GetJ()[k];
		  const double d = globalI->GetData()[k];

		  const int sdcol = maxifgtsdt[igcol];
		  
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

  const int size = A->GetGlobalNumRows();
  HypreParMatrix *S = new HypreParMatrix(A->GetComm(), csr_op->num_rows, size, size, csr_op->i, csr_op->j, csr_op->data, Acopy->RowPart(), Acopy->ColPart());
  
  if (iftfull)
    delete iftfull;

  if (iftsdt)
    delete iftsdt;

  if (ifgtsdt)
    delete ifgtsdt;

  if (maxifgtsdt)
    delete maxifgtsdt;

  if (ifullsdt)
    delete ifullsdt;

  if (globalI)
    delete globalI;

  delete allg;
  
  return S;
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

  delete all_num_loc_rows;
    
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

  delete cnt;
  
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
  
  delete opI;
  delete opJ;
  delete data;
  
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
// inject from full DOF's in the interface spaces to true DOF's in the subdomain spaces. This is also valid for the transpose of injection.
// The use of full DOF's in the interface spaces is done in InjectionOperator. Whether a DOF is true is determined by
// fespace.GetLocalTDofNumber().

// Therefore, dofmap is defined by SetInterfaceToSurfaceDOFMap() to be of full ifespace DOF size, mapping from the full ifespace DOF's to
// true subdomain DOF's in fespace.
void SetInterfaceToSurfaceDOFMap(ParFiniteElementSpace *ifespace, ParFiniteElementSpace *fespace, ParMesh *pmesh, const int sdAttribute,
				 const std::set<int>& pmeshFacesInInterface, const FiniteElementCollection *fec, std::vector<int>& dofmap)
{
  const int ifSize = ifespace->GetVSize();  // Full DOF size

  dofmap.assign(ifSize, -1);

  const double vertexTol = 1.0e-12;
  
  ParMesh *ifMesh = ifespace->GetParMesh();  // Interface mesh
  ParMesh *sdMesh = fespace->GetParMesh();  // Subdomain mesh

  // Create map from face indices in pmeshFacesInInterface to pmesh elements containing those faces.
  std::map<int, int> pmeshFaceToElem;
  std::set<int> pmeshElemsByInterface;

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
		  MFEM_VERIFY(itf == pmeshFaceToElem.end(), "");
		  
		  pmeshFaceToElem[elFaces[j]] = elId;

		  pmeshElemsByInterface.insert(elId);
		}
	    }
	}
    }

  // Set a map pmeshElemToSDmesh from pmesh element indices to the corresponding sdMesh element indices, only for elements neighboring the interface.
  std::map<int, int> pmeshElemToSDmesh;
  for (int elId=0; elId<sdMesh->GetNE(); ++elId)
    {
      // The sdMesh element attribute is set as the local index of the corresponding pmesh element, which is unique since SD elements do not overlap.
      const int pmeshElemId = sdMesh->GetAttribute(elId) - 1;  // 1 was added to ensure a positive attribute.
      std::set<int>::const_iterator it = pmeshElemsByInterface.find(pmeshElemId);
      if (it != pmeshElemsByInterface.end())  // if pmeshElemId neighbors the interface
	{
	  pmeshElemToSDmesh[pmeshElemId] = elId;
	}
    }
  
  // Loop over interface faces.
  int i = 0;
  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it, ++i)
    {
      const int pmeshFace = *it;

      // Face pmeshFace of pmesh coincides with face i of ifMesh on this process (the same face may also exist on a different process in the same ifMesh,
      // as there can be redundant overlapping faces in parallel, for communication).

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
			  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
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
		  const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
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
	  Array<int> ifdofs, sddofs;
	  ifespace->GetFaceDofs(i, ifdofs);
	  fespace->GetFaceDofs(sdMeshFace, sddofs);

	  MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	  for (int d=0; d<ifdofs.Size(); ++d)
	    {
	      const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
			  
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		  dofmap[ifdofs[d]] = sdtdof;
		}
	    }
	}
    }

  // Note that some entries of dofmap may be undefined, if the corresponding subdomain DOF's in fespace are not true DOF's. 
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

// TODO: combine SetInjectionOperator and InjectionOperator as one class?
class SetInjectionOperator : public Operator
{
private:
  std::set<int> *id;
  
public:
  SetInjectionOperator(const int height, std::set<int> *a) : Operator(height, a->size()), id(a)
  {
    MFEM_VERIFY(height >= width, "SetInjectionOperator constructor");
  }

  ~SetInjectionOperator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    y = 0.0;

    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[*it] = x[i];
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y[i] = x[*it];
  }
};

class InjectionOperator : public Operator
{
private:
  int *id;  // Size should be fullWidth.
  mutable ParGridFunction gf;
  int fullWidth;
  
public:
  InjectionOperator(const int height, ParFiniteElementSpace *interfaceSpace, int *a) : Operator(height, interfaceSpace->GetTrueVSize()),
										       fullWidth(interfaceSpace->GetVSize()), id(a), gf(interfaceSpace)
  {
    MFEM_VERIFY(height >= width, "InjectionOperator constructor");
  }
  
  ~InjectionOperator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    gf.SetFromTrueDofs(x);
    
    y = 0.0;
    for (int i=0; i<fullWidth; ++i)
      {
	if (id[i] >= 0)
	  y[id[i]] = gf[i];
      }
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    for (int i=0; i<fullWidth; ++i)
      {
      	if (id[i] >= 0)
	  gf[i] = x[id[i]];
      }
    
    gf.GetTrueDofs(y);
  }
};

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
	      ssd[lsddof] = s_gf[dof_i];
	    }
	}
    }
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

#define DDMCOMPLEX

class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_, ParMesh **pmeshSD_, ParMesh **pmeshIF_,
		       const int orderND, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
		       std::vector<int> *interfaceLocalIndex_) :
    numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), pmeshSD(pmeshSD_), pmeshIF(pmeshIF_), fec(orderND, spaceDim),
    fecbdry(orderND, spaceDim-1), fecbdryH1(orderND, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
    subdomainLocalInterfaces(numSubdomains_), pmeshGlobal(pmesh_),
    k2(2.0), realPart(true),
    alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
  {
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    m_rank = rank;
    
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
    SpAsdComplexRowSizes = new HYPRE_Int*[numSubdomains];;
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

	    cout << rank << ": Interface " << i << " number of bdry attributes: " << pmeshIF[i]->bdr_attributes.Size()
		 << ", NE " << pmeshIF[i]->GetNE() << ", NBE " << pmeshIF[i]->GetNBE() << endl;

	    /*
	    for (int j=0; j<pmeshIF[i]->GetNBE(); ++j)
	      cout << rank << " Interface " << i << ", be " << j << ", bdrAttribute " << pmeshIF[i]->GetBdrAttribute(j) << endl;
	    */
	    
	    CreateInterfaceMatrices(i);
	  }

	const int ifli = (*interfaceLocalIndex)[i];

	MFEM_VERIFY((ifli >= 0) == (pmeshIF[i] != NULL), "");
	
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
    
    for (int m=0; m<numSubdomains; ++m)
      {
	InterfaceToSurfaceInjection[m].resize(subdomainLocalInterfaces[m].size());
	InterfaceToSurfaceInjectionData[m].resize(subdomainLocalInterfaces[m].size());

	if (pmeshSD[m] == NULL)
	  {
	    fespace[m] = NULL;
	  }
	else
	  {
	    fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m

	    /*
	    if (m == 0)
	      cout << rank << ": sd 0 ND space true size " << fespace[m]->GetTrueVSize() << ", full size " << fespace[m]->GetVSize() << endl;
	    */
	  }
	
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    size += ifespace[interfaceIndex]->GetTrueVSize();
	    size += iH1fespace[interfaceIndex]->GetTrueVSize();

	    block_trueOffsets[m+1] += ifespace[interfaceIndex]->GetTrueVSize();
	    block_trueOffsets[m+1] += iH1fespace[interfaceIndex]->GetTrueVSize();

	    if (fespace[m] != NULL)
	      {
		const int ifli = (*interfaceLocalIndex)[interfaceIndex];
		MFEM_VERIFY(ifli >= 0, "");
		
		SetInterfaceToSurfaceDOFMap(ifespace[interfaceIndex], fespace[m], pmeshGlobal, m+1, (*localInterfaces)[ifli].faces, &fecbdry,
					    InterfaceToSurfaceInjectionData[m][i]);
	    
		InterfaceToSurfaceInjection[m][i] = new InjectionOperator(fespace[m]->GetTrueVSize(), ifespace[interfaceIndex],
									  &(InterfaceToSurfaceInjectionData[m][i][0]));
	      }
	    else
	      {
		InterfaceToSurfaceInjection[m][i] = NULL;
	      }
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
	  }
	
	SetOffsetsSD(m);

	//GMRESSolver *gmres = new GMRESSolver(fespace[m]->GetComm());  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?
	GMRESSolver *gmres = new GMRESSolver(MPI_COMM_WORLD);  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?

#ifdef DDMCOMPLEX
	// Real part
	SetToRealParameters();
	
	AsdRe[m] = CreateSubdomainOperator(m);
	
	// Imaginary part
	SetToImaginaryParameters();
	
	AsdIm[m] = CreateSubdomainOperator(m);

	// Set real-valued subdomain operator for preconditioning purposes only.
	SetToPreconditionerParameters();

	preconditionerMode = true;  // TODO!
	AsdP[m] = CreateSubdomainOperator(m);
	preconditionerMode = false;

	gmres->SetOperator(*(AsdP[m]));  // TODO: maybe try an SPD version of Asd with a CG solver instead of GMRES.
#else
	Asd[m] = CreateSubdomainOperator(m);

	gmres->SetOperator(*(Asd[m]));
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

#else
	globalInterfaceOp->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
	globalInterfaceOp->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
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

    BlockOperator *globalSubdomainOp = new BlockOperator(block_trueOffsets2);
#else
    // Create block diagonal operator with entries R_{sd0} A_{sd0}^{-1} R_{sd0}^T
    BlockOperator *globalSubdomainOp = new BlockOperator(block_trueOffsets);
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
	
		MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
		MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");

		ifsize += ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
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

	    AsdComplex[m] = new BlockOperator(block_ComplexOffsetsSD[m]);
	    AsdComplex[m]->SetBlock(0, 0, AsdRe[m]);
	    AsdComplex[m]->SetBlock(0, 1, AsdIm[m], -1.0);
	    AsdComplex[m]->SetBlock(1, 0, AsdIm[m]);
	    AsdComplex[m]->SetBlock(1, 1, AsdRe[m]);

	    precAsdComplex[m] = new BlockDiagonalPreconditioner(block_ComplexOffsetsSD[m]);

	    precAsdComplex[m]->SetDiagonalBlock(0, invAsd[m]);
	    precAsdComplex[m]->SetDiagonalBlock(1, invAsd[m]);

	    MPI_Barrier(MPI_COMM_WORLD);

#ifdef SPARSE_ASDCOMPLEX
	    SpAsdComplex[m] = GetSparseMatrixFromOperator(AsdComplex[m]);
	    SpAsdComplexRowSizes[m] = new HYPRE_Int[2];
	    SpAsdComplexRowSizes[m][0] = 0;
	    SpAsdComplexRowSizes[m][1] = SpAsdComplex[m]->Size();

	    HypreAsdComplex[m] = new HypreParMatrix(MPI_COMM_WORLD, SpAsdComplex[m]->Size(), SpAsdComplexRowSizes[m], SpAsdComplex[m]);  // constructor with 4 arguments, v1
	    invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), MPI_COMM_WORLD);
#else
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
#endif

	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injComplexSD[m]), invAsdComplex[m], injComplexSD[m],
									false, false, false));
#else
	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injSD[m]), invAsd[m], injSD[m], false, false, false));
#endif

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

  virtual void Mult(const Vector & x, Vector & y) const
  {
    // x and y are vectors of true DOF's on the subdomain interfaces and exterior boundary. 
    // Degrees of freedom in x and y are ordered as follows: x = [x_0, x_1, ..., x_{N-1}];
    // N = numSubdomains, and on subdomain m, x_m = [u_m^s, f_m, \rho_m];
    // u_m^s is the vector of true DOF's of u on the entire surface of subdomain m, for a field u in a Nedelec space on subdomain m;
    // f_m = [f_{m,0}, ..., f_{m,p-1}] is an auxiliary vector of true DOF's in Nedelec spaces on all p interfaces on subdomain m;
    // \rho_m = [\rho_{m,0}, ..., \rho_{m,p-1}] is an auxiliary vector of true DOF's in H^1 (actually H^{1/2}) FE spaces on all
    // p interfaces on subdomain m.
    
    // The surface of subdomain m equals the union of subdomain interfaces and a subset of the exterior boundary.
    // There are redundant DOF's for f and \rho at subdomain corner edges (intersections of interfaces), i.e. discontinuity on corners.
    // The surface bilinear forms and their matrices are defined on subdomain interfaces, not the entire subdomain boundary.
    // The surface DOF's for a subdomain are indexed according to the entire subdomain mesh boundary, and we must use maps between
    // those surface DOF's and DOF's on the individual interfaces.
    
    globalOp->Mult(x, y);
  }  

#ifndef DDMCOMPLEX
  void FullSystemAnalyticTest()
  {
    Vector uSD, uIF, ubdry, ufullSD, AufullSD, diffIF;

    diffIF.SetSize(ifespace[0]->GetTrueVSize());

    VectorFunctionCoefficient E(3, test2_E_exact);
    VectorFunctionCoefficient y(3, test2_RHS_exact);
    VectorFunctionCoefficient f0(3, test2_f_exact_0);
    VectorFunctionCoefficient f1(3, test2_f_exact_1);
    FunctionCoefficient rho0(test2_rho_exact_0);
    FunctionCoefficient rho1(test2_rho_exact_1);

    int fullSize = 0;
    int ifSize = 0;

    std::vector<int> sdFullSize(numSubdomains);
    
    for (int m=0; m<numSubdomains; ++m)
      {
	int fullSize_m = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    fullSize_m += fespace[m]->GetTrueVSize();
	    ifSize += tdofsBdry[m].size();
	  }

	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    
	    fullSize_m += ifespace[interfaceIndex]->GetTrueVSize();
	    fullSize_m += iH1fespace[interfaceIndex]->GetTrueVSize();

	    ifSize += ifespace[interfaceIndex]->GetTrueVSize();
	    ifSize += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	fullSize += fullSize_m;
	sdFullSize[m] = fullSize_m;
      }

    Vector xfull(fullSize);
    xfull = 0.0;
    Vector Axfull(fullSize);
    Axfull = 0.0;
    Vector yfull(fullSize);
    yfull = 0.0;

    Vector xif(ifSize);
    Vector Cxif(ifSize);
    xif = 0.0;

    int osFull = 0;

    for (int m=0; m<numSubdomains; ++m)
      {
	ufullSD.SetSize(sdFullSize[m]);
	AufullSD.SetSize(sdFullSize[m]);

	ufullSD = 0.0;

	int osFullSD = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    uSD.SetSize(fespace[m]->GetTrueVSize());
	    ubdry.SetSize(tdofsBdry[m].size());

	    osFullSD += fespace[m]->GetTrueVSize();
	    
	    ParGridFunction x(fespace[m]);
	    x.ProjectCoefficient(E);
	    x.GetTrueDofs(uSD);
	    
	    tdofsBdryInjectionTranspose[m]->Mult(uSD, ubdry);

	    cout << m_rank << ": ubdry norm on SD " << m << " = " << ubdry.Norml2() << endl;

	    /*
	    { // Subtest, to check u restricted to the interface from each subdomain.
	      uIF.SetSize(ifespace[0]->GetTrueVSize());
	      InterfaceToSurfaceInjection[m][0]->MultTranspose(uSD, uIF);

	      if (m == 0)
		{
		  diffIF = uIF;

		  cout << m_rank << ": uIF[0] norm " << diffIF.Norml2() << endl;
		}
	      else
		{
		  ParGridFunction Eif(ifespace[0]);
		  Eif.ProjectCoefficient(E);
		  uIF = 0.0;
		  Eif.GetTrueDofs(uIF);
		  
		  diffIF -= uIF;
		  cout << m_rank << ": diffIF norm " << diffIF.Norml2() << endl;
		}
	    }
	    */
	    
	    for (int i=0; i<tdofsBdry[m].size(); ++i)
	      {
		xif[block_trueOffsets[m] + i] = ubdry[i];
	      }

	    for (int i=0; i<uSD.Size(); ++i)
	      {
		ufullSD[i] = uSD[i];
	      }

	    x.ProjectCoefficient(y);
	    x.GetTrueDofs(uSD);
	    
	    for (int i=0; i<uSD.Size(); ++i)
	      {
		yfull[osFull + i] = uSD[i];
	      }
	  }

	int osF = 0;
	
	int os = tdofsBdry[m].size();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    uIF.SetSize(ifespace[interfaceIndex]->GetTrueVSize());

	    ParGridFunction x(ifespace[interfaceIndex]);
	    if (m == 0)
	      x.ProjectCoefficient(f0);
	    else
	      x.ProjectCoefficient(f1);
	    
	    x.GetTrueDofs(uIF);

	    for (int j=0; j<uIF.Size(); ++j)
	      {
		xif[block_trueOffsets[m] + os + j] = uIF[j];
		ufullSD[osFullSD + j] = uIF[j];
	      }
	    
	    os += ifespace[interfaceIndex]->GetTrueVSize();

	    osF = osFullSD;
	    osFullSD += ifespace[interfaceIndex]->GetTrueVSize();

	    uIF.SetSize(iH1fespace[interfaceIndex]->GetTrueVSize());
	    
	    ParGridFunction r(iH1fespace[interfaceIndex]);
	    if (m == 0)
	      r.ProjectCoefficient(rho0);
	    else
	      r.ProjectCoefficient(rho1);
	    
	    r.GetTrueDofs(uIF);

	    for (int j=0; j<uIF.Size(); ++j)
	      {
		xif[block_trueOffsets[m] + os + j] = uIF[j];
		ufullSD[osFullSD + j] = uIF[j];
	      }
	    
	    os += iH1fespace[interfaceIndex]->GetTrueVSize();

	    osFullSD += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(sdFullSize[m] == osFullSD, "");

	Asd[m]->Mult(ufullSD, AufullSD);

	double normAuF = 0.0;
	for (int j=0; j<ifespace[0]->GetTrueVSize(); ++j)
	  normAuF += AufullSD[osF + j] * AufullSD[osF + j];

	normAuF = sqrt(normAuF);

	cout << m_rank << ": normAuF " << normAuF << " on sd " << m << endl;
	
	for (int j=0; j<sdFullSize[m]; ++j)
	  {
	    xfull[osFull + j] = ufullSD[j];
	    Axfull[osFull + j] = AufullSD[j];
	  }
	
	osFull += sdFullSize[m];
      }

    MFEM_VERIFY(osFull == fullSize, "");

    globalInterfaceOp->Mult(xif, Cxif);
    //Cxif = 0.0;  // TODO: remove
    
    double Cxif_norm = Cxif.Norml2();
    cout << m_rank << ": Cxif norm " << Cxif_norm << ", Axfull norm " << Axfull.Norml2() << endl;
    
    double Cxif_us = 0.0;
    
    osFull = 0;
    
    for (int m=0; m<numSubdomains; ++m)
      {
	int osFullSD = 0;

	if (pmeshSD[m] != NULL)
	  {
	    uSD.SetSize(fespace[m]->GetTrueVSize());
	    ubdry.SetSize(tdofsBdry[m].size());
    
	    for (int i=0; i<tdofsBdry[m].size(); ++i)
	      {
		ubdry[i] = Cxif[block_trueOffsets[m] + i];
	      }

	    double Cxif_us_m = ubdry.Norml2();
	    Cxif_us += Cxif_us_m * Cxif_us_m;
	    
	    tdofsBdryInjection[m]->Mult(ubdry, uSD);

	    for (int i=0; i<uSD.Size(); ++i)
	      {
		Axfull[osFull + i] += uSD[i];
	      }

	    osFullSD += fespace[m]->GetTrueVSize();
	  }

	int os = tdofsBdry[m].size();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    for (int j=0; j<ifespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		Axfull[osFull + osFullSD + j] += Cxif[block_trueOffsets[m] + os + j];
	      }

	    os += ifespace[interfaceIndex]->GetTrueVSize();
	    os += iH1fespace[interfaceIndex]->GetTrueVSize();

	    osFullSD += ifespace[interfaceIndex]->GetTrueVSize();
	    osFullSD += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(sdFullSize[m] == osFullSD, "");
	
	osFull += sdFullSize[m];
      }

    Cxif_us = sqrt(Cxif_us);
    
    cout << m_rank << ": Cxif u_s component norm " << Cxif_us << ", f and rho component norm " << Cxif_norm - Cxif_us << endl;

    // Now Axfull is (A + C) xfull, which should equal yfull.
    Vector dfull(fullSize);
    Vector dfull_u(fullSize);
    Vector dfull_us(fullSize);
    Vector dfull_f(fullSize);
    Vector dfull_rho(fullSize);
    Vector Axfull_u(fullSize);
    Vector Axfull_f(fullSize);
    Vector Axfull_rho(fullSize);

    dfull_u = 0.0;
    dfull_us = 0.0;
    dfull_f = 0.0;
    dfull_rho = 0.0;

    Axfull_u = 0.0;
    Axfull_f = 0.0;
    Axfull_rho = 0.0;
    
    for (int i=0; i<fullSize; ++i)
      dfull[i] = Axfull[i] - yfull[i];

    osFull = 0;
    for (int m=0; m<numSubdomains; ++m)
      {
	int osm = 0;
	
	if (pmeshSD[m] != NULL)
	  {
	    for (int i=0; i<fespace[m]->GetTrueVSize(); ++i)
	      {
		dfull_u[osFull + i] = Axfull[osFull + i] - yfull[osFull + i];
		Axfull_u[osFull + i] = Axfull[osFull + i];
	      }

	    osm += fespace[m]->GetTrueVSize();
	    
	    for (set<int>::const_iterator it = tdofsBdry[m].begin(); it != tdofsBdry[m].end(); ++it)
	      {
		dfull_us[osFull + (*it)] = dfull_u[osFull + (*it)];
	      }
	  }

	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];

	    for (int j=0; j<ifespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		dfull_f[osFull + osm + i] = Axfull[osFull + osm + i] - yfull[osFull + osm + i];
		Axfull_f[osFull + osm + i] = Axfull[osFull + osm + i];
	      }

	    osm += ifespace[interfaceIndex]->GetTrueVSize();

	    for (int j=0; j<iH1fespace[interfaceIndex]->GetTrueVSize(); ++j)
	      {
		dfull_rho[osFull + osm + i] = Axfull[osFull + osm + i] - yfull[osFull + osm + i];
		Axfull_rho[osFull + osm + i] = Axfull[osFull + osm + i];
	      }
	    
	    osm += iH1fespace[interfaceIndex]->GetTrueVSize();
	  }

	MFEM_VERIFY(osm == sdFullSize[m], "");
	osFull += sdFullSize[m];
      }
    
    cout << m_rank << ": dfull norm " << dfull.Norml2() << ", y norm " << yfull.Norml2() << ", (A+C)x norm " << Axfull.Norml2() << endl;
    cout << m_rank << ": dfull_u norm " << dfull_u.Norml2() << ", (A+C)x_u norm " << Axfull_u.Norml2() << endl;
    cout << m_rank << ": dfull_us norm " << dfull_us.Norml2() << endl;
    cout << m_rank << ": dfull_f norm " << dfull_f.Norml2() << ", (A+C)x_f norm " << Axfull_f.Norml2() << endl;
    cout << m_rank << ": dfull_rho norm " << dfull_rho.Norml2() << ", (A+C)x_rho norm " << Axfull_rho.Norml2() << endl;
  }
#endif
  
  void GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobalRe, Vector & sourceGlobalIm, Vector & sourceReduced) const
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
	
	if (pmeshSD[m] != NULL)
	  {
	    sourceSD.SetSize(fespace[m]->GetTrueVSize());

	    // Map from the global u to [u_m f_m \rho_m], with blocks corresponding to subdomains, and f_m = 0, \rho_m = 0.
#ifdef DDMCOMPLEX
	    // In the complex case, we map from the global u^{Re} and u^{Im} to [u_m^{Re} f_m^{Re} \rho_m^{Re} u_m^{Im} f_m^{Im} \rho_m^{Im}].

	    MFEM_VERIFY(AsdComplex[m]->Height() == block_ComplexOffsetsSD[m][2], "");
	    MFEM_VERIFY(AsdComplex[m]->Height() == 2*block_ComplexOffsetsSD[m][1], "");

	    cout << rank << ": Setting real subdomain DOFs, sd " << m << endl;
	    
	    ySD[m] = new Vector(AsdComplex[m]->Height());  // Size of [u_m f_m \rho_m], real and imaginary parts

	    wSD.SetSize(AsdComplex[m]->Height());
	    wSD = 0.0;
	    
	    SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalRe, sourceSD);

	    const bool explicitRHS = false;

	    if (explicitRHS)
	      {
		ParGridFunction x(fespace[m]);
		VectorFunctionCoefficient f2(3, test2_RHS_exact);

		//x.SetFromTrueDofs(sourceSD);
		x.ProjectCoefficient(f2);

		x.GetTrueDofs(sourceSD);
	      }

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
	    SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalIm, sourceSD);

	    cout << rank << ": Done setting imaginary subdomain DOFs, sd " << m << endl;

	    if (explicitRHS)
	      sourceSD = 0.0;

	    for (int i=0; i<sourceSD.Size(); ++i)
	      (*(ySD[m]))[block_ComplexOffsetsSD[m][1] + i] = sourceSD[i];  // Set the u_m block of ySD, imaginary part
	  }

	MPI_Barrier(MPI_COMM_WORLD);
	
	{
	    cout << rank << ": ySD norm: " << ySD[m]->Norml2() << ", wSD norm " << wSD.Norml2() << endl;
	    
	    cout << rank << ": Applying invAsdComplex[" << m << "]" << endl;
	    
	    invAsdComplex[m]->Mult(*(ySD[m]), wSD);
	    //AsdComplex[m]->Mult(*(ySD[m]), wSD);

	    cout << rank << ": Done applying invAsdComplex[" << m << "]" << endl;

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

	MPI_Barrier(MPI_COMM_WORLD);
      }
  }

#ifdef DDMCOMPLEX
  void PrintSubdomainError(const int sd, Vector & u)
  {
    ParGridFunction x(fespace[sd]);
    VectorFunctionCoefficient E(3, test2_E_exact);
    //x.ProjectCoefficient(E);

    Vector uSD(fespace[sd]->GetTrueVSize());

    Vector zeroVec(3);
    zeroVec = 0.0;
    VectorConstantCoefficient vzero(zeroVec);

    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      uSD[i] = u[i];

    x.SetFromTrueDofs(uSD);

    double errRe = x.ComputeL2Error(E);
    double normXRe = x.ComputeL2Error(vzero);

    DataCollection *dc = NULL;
    const bool visit = true;
    if (visit && sd == 1)
      {
	//x.ProjectCoefficient(E);

       bool binary = false;
       if (binary)
	 {
#ifdef MFEM_USE_SIDRE
	   dc = new SidreDataCollection("ddsol", pmeshSD[sd]);
#else
	   MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
	 }
       else
	 {
	   dc = new VisItDataCollection("usd1", pmeshSD[sd]);
	   dc->SetPrecision(8);
	   // To save the mesh using MFEM's parallel mesh format:
	   // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
	 }
       dc->RegisterField("solution", &x);
       dc->SetCycle(0);
       dc->SetTime(0.0);
       dc->Save();
     }

    // Overwrite real part with imaginary part, only for the subdomain true DOF's.
    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      uSD[i] = u[block_ComplexOffsetsSD[sd][1] + i];

    x.SetFromTrueDofs(uSD);
    double errIm = x.ComputeL2Error(vzero);

    ParGridFunction zerogf(fespace[sd]);
    zerogf = 0.0;
    double normE = zerogf.ComputeL2Error(E);
    //if (m_rank == 0)
      {
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Re = " << errRe << endl;
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Im = " << errIm << endl;
	cout << m_rank << ": sd " << sd << " || E_h ||_{L^2} Re = " << normXRe << endl;
	cout << m_rank << ": sd " << sd << " || E ||_{L^2} Re = " << normE << endl;
      }
  }
#endif

  void RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced, Vector & solDomain)
  {
    MFEM_VERIFY(solReduced.Size() == this->Height(), "");
    Vector w(this->Height());
    Vector v, u, uSD, wSD;

    MFEM_VERIFY(this->Height() == block_trueOffsets2[numSubdomains], "");

    // Assuming GetReducedSource has been called, ySD stores y_m (with full components [u_m f_m \rho_m]) for each subdomain m.

    // globalInterfaceOp represents the off-diagonal blocks C_{ij} R_j^T
    globalInterfaceOp->Mult(solReduced, w);

    // Now w = \sum_{j\neq i} C_{ij} R_j^T \overbar{u}_j

    for (int m=0; m<numSubdomains; ++m)
      {
	if (ySD[m] != NULL)
	  {
#ifdef DDMCOMPLEX
	    //MFEM_VERIFY(ySD[m]->Size() == block_trueOffsets2[m+1] - block_trueOffsets2[m], "");  wrong
	    MFEM_VERIFY(ySD[m]->Size() > block_trueOffsets2[m+1] - block_trueOffsets2[m], "");
	    MFEM_VERIFY(ySD[m]->Size() == AsdComplex[m]->Height(), "");

	    // Put the [u_m^s, f_m, \rho_m] entries of w (real and imaginary parts) into wSD.
	    wSD.SetSize(block_trueOffsets2[m+1] - block_trueOffsets2[m]);
	    uSD.SetSize(AsdComplex[m]->Height());

	    for (int i=0; i<block_trueOffsets2[m+1] - block_trueOffsets2[m]; ++i)
	      wSD[i] = w[block_trueOffsets2[m] + i];

	    injComplexSD[m]->Mult(wSD, uSD);

	    *(ySD[m]) -= uSD;
	    uSD = 0.0;
	    invAsdComplex[m]->Mult(*(ySD[m]), uSD);

	    PrintSubdomainError(m, uSD);
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
      }
  }
  
private:

  int m_rank;

  mutable bool testing;
  
  const double k2;

  bool realPart;
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;

  ParMesh *pmeshGlobal;
  ParMesh **pmeshSD;  // Subdomain meshes
  ParMesh **pmeshIF;  // Interface meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
  ParFiniteElementSpace **fespace, **ifespace, **iH1fespace;
  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad, **ifNDH1gradT, **ifH1mass, **ifND_FS;
  Operator **ifNDmassInv, **ifH1massInv;
  HypreParMatrix **sdND;
  HypreParMatrix **sdNDcopy;
  HypreParMatrix **A_SS;
  HypreParMatrix **sdNDPen;
  HypreParMatrix **sdNDPlusPen;
  SparseMatrix **sdNDPenSp;
  ParBilinearForm **bf_sdND;
  Operator **sdNDinv;
#ifdef DDMCOMPLEX
  Operator **AsdRe, **AsdIm, **AsdP, **invAsdComplex;
#ifdef SPARSE_ASDCOMPLEX
  SparseMatrix **SpAsdComplex;
  HypreParMatrix **HypreAsdComplex;
  HYPRE_Int **SpAsdComplexRowSizes;
#endif
  BlockOperator **AsdComplex;
  BlockDiagonalPreconditioner **precAsdComplex;
#else
  Operator **Asd;
#endif
  Operator **ASPsd;
  Operator **invAsd;
  Solver **precAsd;

  // TODO: it may be possible to eliminate ifND_FS. It is assembled as a convenience, to avoid summing entries input to ASPsd.
  
  Vector **ySD;
  
  BlockOperator **injSD;
  
  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

#ifdef DDMCOMPLEX
  BlockOperator *globalInterfaceOpRe, *globalInterfaceOpIm;
  std::vector<Array<int> > block_ComplexOffsetsSD;
  Array<int> block_trueOffsets2;  // Offsets used in globalOp
  BlockOperator **injComplexSD;
#endif

  BlockOperator *globalInterfaceOp;
  
  Operator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  BlockOperator *NDinv;
  
  vector<set<int> > tdofsBdry;
  SetInjectionOperator **tdofsBdryInjection;
  Operator **tdofsBdryInjectionTranspose;
  
  double alpha, beta, gamma;
  double alphaInverse, betaOverAlpha, gammaOverAlpha;
  double alphaRe, betaRe, gammaRe;
  double alphaIm, betaIm, gammaIm;

  bool preconditionerMode;
  
  std::vector<std::vector<InjectionOperator*> > InterfaceToSurfaceInjection;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionData;

  std::vector<Array<int> > rowTrueOffsetsSD, colTrueOffsetsSD;
#ifdef DDMCOMPLEX
  std::vector<Array<int> > rowTrueOffsetsComplexSD, colTrueOffsetsComplexSD;
  std::vector<Array<int> > rowTrueOffsetsComplexIF, colTrueOffsetsComplexIF;
#endif
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIF, colTrueOffsetsIF;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFL, colTrueOffsetsIFL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFR, colTrueOffsetsIFR;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBL, colTrueOffsetsIFBL;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFBR, colTrueOffsetsIFBR;
  std::vector<Array<int> > trueOffsetsSD;
  Array<int> rowSROffsetsSD, colSROffsetsSD;

#ifdef EQUATE_REDUNDANT_VARS
  std::vector<Array<int> > rowTrueOffsetsIFRR, colTrueOffsetsIFRR;
#endif
  
  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void SetParameters()
  {
    const double cTE = 0.5;  // From RawatLee2010
    const double cTM = 4.0;  // From RawatLee2010

    // Note that PengLee2012 recommends cTE = 1.5 * cTM. 

    // TODO: take these parameters from the mesh and finite element space.
    const double h = 1.4e-1;
    const double feOrder = 1.0;
    
    const double ktTE = cTE * M_PI * feOrder / h;
    const double ktTM = cTM * M_PI * feOrder / h;
    
    //const double kzTE = sqrt(8.0 * k2);
    //const double kzTM = sqrt(3.0 * k2);

    const double kzTE = sqrt((ktTE*ktTE) - k2);
    const double kzTM = sqrt((ktTM*ktTM) - k2);
    
    const double k = sqrt(k2);

    // PengLee2012
    // alpha = ik
    // beta = i / (k + i kzTE) = i * (k - i kzTE) / (k^2 + kzTE^2) = (kzTE + ik) / (k^2 + kzTE^2)
    // gamma = 1 / (k^2 + i k * kzTM) = (k^2 - i k * kzTM) / (k^4 + k^2 kzTM^2)
	
    // Real part
    alphaRe = 0.0;
    betaRe = kzTE / (k2 + (kzTE * kzTE));
    gammaRe = 1.0 / (k2 + (kzTM * kzTM));

    // Imaginary part
    alphaIm = k;
    betaIm = k / (k2 + (kzTE * kzTE));
    gammaIm = -kzTM / (k * (k2 + (kzTM * kzTM)));


    // RawatLee2010
    // alpha = -ik
    // beta = i / (k + kzTE) = i / (k - i sqrt((k_tau^{max,te})^2 - k^2)) = i / (k - i sqrt(ktTE^2 - k^2))
    //      = i (k + i sqrt(ktTE^2 - k^2)) / [k^2 + ktTE^2 - k^2] = i (k + i sqrt(ktTE^2 - k^2)) / ktTE^2
    //      = [-sqrt(ktTE^2 - k^2) + ik] / ktTE^2 = [-kzTE + ik] / ktTE^2
    // gamma = 1 / [k^2 + k kzTM] = 1 / [k^2 - ik sqrt((k_tau^{max,tm})^2 - k^2)] = 1 / [k^2 - ik sqrt(ktTM^2 - k^2)]
    //       = [k^2 + ik sqrt(ktTM^2 - k^2)] / (k^4 + k^2 ktTM^2 - k^4) = [k^2 + ik sqrt(ktTM^2 - k^2)] / (k^2 ktTM^2)
    //       = [1 + i (1/k) sqrt(ktTM^2 - k^2)] / ktTM^2 = [1 + i (kzTM / k)] / ktTM^2
    
    // Real part
    alphaRe = 0.0;
    betaRe = -kzTE / (ktTE * ktTE);
    gammaRe = 1.0 / (ktTM * ktTM);

    // Imaginary part
    alphaIm = -k;
    betaIm = k / (ktTE * ktTE);
    gammaIm = kzTM / (k * (ktTM * ktTM));

    /*
    {
      //////////// Testing 
      alphaRe = 1.0;
      betaRe = 1.0;
      gammaRe = 1.0;

      alphaIm = 0.0;
      betaIm = 0.0;
      gammaIm = 0.0;
    }
    */
  }

  void SetToRealParameters()
  {
    realPart = true;

    alpha = alphaRe;
    beta = betaRe;
    gamma = gammaRe;

    alphaInverse = alphaRe / ((alphaRe * alphaRe) + (alphaIm * alphaIm));

    betaOverAlpha = ((alphaRe * betaRe) + (alphaIm * betaIm)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
    gammaOverAlpha = ((alphaRe * gammaRe) + (alphaIm * gammaIm)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
  }
  
  void SetToImaginaryParameters()
  {
    realPart = false;

    alpha = alphaIm;
    beta = betaIm;
    gamma = gammaIm;

    alphaInverse = -alphaIm / ((alphaRe * alphaRe) + (alphaIm * alphaIm));

    betaOverAlpha = ((alphaRe * betaIm) - (alphaIm * betaRe)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
    gammaOverAlpha = ((alphaRe * gammaIm) - (alphaIm * gammaRe)) / ((alphaRe * alphaRe) + (alphaIm * alphaIm));
  }
  
  void SetToPreconditionerParameters()
  {
    realPart = true;

    /*
    alpha = 1.0;
    beta = 1.0;
    gamma = 1.0;
    */
    
    alpha = sqrt((alphaRe * alphaRe) + (alphaIm * alphaIm));
    beta = sqrt((betaRe * betaRe) + (betaIm * betaIm));
    gamma = sqrt((gammaRe * gammaRe) + (gammaIm * gammaIm));

    alphaInverse = 1.0 / alpha;

    betaOverAlpha = beta / alpha;
    gammaOverAlpha = gamma / alpha;
  }

  void CreateInterfaceMatrices(const int interfaceIndex)
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

#ifdef ZERO_RHO_BC
    {
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
    
    NDH1grad->Finalize();
    
    //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
    //NDH1grad->FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
    ifNDH1grad[interfaceIndex] = NDH1grad->ParallelAssemble();
    ifNDH1gradT[interfaceIndex] = ifNDH1grad[interfaceIndex]->Transpose();
    
    cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize()
	 << ", H1 true size " << iH1fespace[interfaceIndex]->GetTrueVSize()
	 << ", NDH1 height " << ifNDH1grad[interfaceIndex]->Height() << ", width " << ifNDH1grad[interfaceIndex]->Width() << endl;
  }
  
  // Create operator C_{sd0,sd1} in the block space corresponding to [u_m^s, f_i, \rho_i]. Note that the u_m^I blocks are omitted (just zeros).
  Operator* CreateCij(const int localInterfaceIndex, const int orientation)
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
    
    rowTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();

    rowTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();

    //rowTrueOffsetsIF[localInterfaceIndex][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIF[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
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

  // Create operator C_{sd0,sd1} R_{sd1}^T. The operator returned here is of size n_{sd0} by n_{sd1}, where n_{sd} is the sum of
  // tdofsBdry[sd].size() and ifespace[interfaceIndex]->GetTrueVSize() and iH1fespace[interfaceIndex]->GetTrueVSize() for all interfaces of subdomain sd.
  Operator* CreateInterfaceOperator(const int localInterfaceIndex, const int orientation)
  {
    const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
    const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

    const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];
    
    MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
    MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");

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
	  sd0os += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
	else
	  sd0osComp += ifespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd0][i]]->GetTrueVSize();
      }

    for (int i=0; i<subdomainLocalInterfaces[sd1].size(); ++i)
      {
	if (subdomainLocalInterfaces[sd1][i] == interfaceIndex)
	  {
	    MFEM_VERIFY(sd1if == -1, "");
	    sd1if = i;
	  }

	if (sd1if == -1)
	  sd1os += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
	else
	  sd1osComp += ifespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize() + iH1fespace[subdomainLocalInterfaces[sd1][i]]->GetTrueVSize();
      }
    
    MFEM_VERIFY(sd0if >= 0, "");
    MFEM_VERIFY(sd1if >= 0, "");

    sd0osComp -= ifespace[interfaceIndex]->GetTrueVSize();
    sd1osComp -= ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();

    Operator *Cij = CreateCij(localInterfaceIndex, orientation);

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
    rowTrueOffsetsIFR[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    rowTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFR[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    colTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFR[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *rightInjection = new BlockOperator(rowTrueOffsetsIFR[localInterfaceIndex][orientation], colTrueOffsetsIFR[localInterfaceIndex][orientation]);

    rightInjection->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd1][sd1if]),
						       tdofsBdryInjection[sd1], false, false));
    rightInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

    // Create left injection operator for sd0.

    rowTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
    colTrueOffsetsIFL[localInterfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

    rowTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    rowTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
    //rowTrueOffsetsIFL[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFL[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFL[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFL[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *leftInjection = new BlockOperator(rowTrueOffsetsIFL[localInterfaceIndex][orientation], colTrueOffsetsIFL[localInterfaceIndex][orientation]);

    leftInjection->SetBlock(0, 0, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][sd0if], false, false));
    leftInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

    TripleProductOperator *CijS = new TripleProductOperator(leftInjection, Cij, rightInjection, false, false, false);

    // CijS maps from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

    // Create block injection operator from (u^s, f_i) to (u^s, f_i, \rho_i) on sd0, where the range is over all sd0 interfaces.
    
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1
    colTrueOffsetsIFBL[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1

    rowTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = sd0os;
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize();
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation][4] = sd0osComp;
    
    rowTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFBL[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFBL[localInterfaceIndex][orientation][1] = tdofsBdry[sd0].size();
    colTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
    
    colTrueOffsetsIFBL[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *blockInjectionLeft = new BlockOperator(rowTrueOffsetsIFBL[localInterfaceIndex][orientation], colTrueOffsetsIFBL[localInterfaceIndex][orientation]);

    blockInjectionLeft->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd0].size()));
    blockInjectionLeft->SetBlock(2, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));

    // Create block injection operator from (u^s, f_i, \rho_i) to (u^s, f_i, \rho_i) on sd1, where the domain is over all sd1 interfaces
    // and the range is only this one interface.

    rowTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1
    colTrueOffsetsIFBR[localInterfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1

    rowTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
    rowTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
    colTrueOffsetsIFBR[localInterfaceIndex][orientation] = 0;
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][1] = tdofsBdry[sd1].size();
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = sd1os;
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    colTrueOffsetsIFBR[localInterfaceIndex][orientation][4] = sd1osComp;
    
    colTrueOffsetsIFBR[localInterfaceIndex][orientation].PartialSum();
    
    BlockOperator *blockInjectionRight = new BlockOperator(rowTrueOffsetsIFBR[localInterfaceIndex][orientation], colTrueOffsetsIFBR[localInterfaceIndex][orientation]);

    blockInjectionRight->SetBlock(0, 0, new IdentityOperator(tdofsBdry[sd1].size()));
    blockInjectionRight->SetBlock(1, 2, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));

#ifdef EQUATE_REDUNDANT_VARS
    if (orientation == 1 && realPart)
      {
	rowTrueOffsetsIFRR[localInterfaceIndex].SetSize(3 + 1);
	colTrueOffsetsIFRR[localInterfaceIndex].SetSize(3 + 1);
	
	rowTrueOffsetsIFRR[localInterfaceIndex] = 0;
	rowTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd0].size() + sd0os + ifespace[interfaceIndex]->GetTrueVSize();
	rowTrueOffsetsIFRR[localInterfaceIndex][2] = iH1fespace[interfaceIndex]->GetTrueVSize();
	rowTrueOffsetsIFRR[localInterfaceIndex][3] = sd0osComp - iH1fespace[interfaceIndex]->GetTrueVSize();
    
	rowTrueOffsetsIFRR[localInterfaceIndex].PartialSum();

	colTrueOffsetsIFRR[localInterfaceIndex] = 0;
	colTrueOffsetsIFRR[localInterfaceIndex][1] = tdofsBdry[sd1].size() + sd1os + ifespace[interfaceIndex]->GetTrueVSize();
	colTrueOffsetsIFRR[localInterfaceIndex][2] = iH1fespace[interfaceIndex]->GetTrueVSize();
	colTrueOffsetsIFRR[localInterfaceIndex][3] = sd1osComp;
    
	colTrueOffsetsIFRR[localInterfaceIndex].PartialSum();

	BlockOperator *Irhorho = new BlockOperator(rowTrueOffsetsIFRR[localInterfaceIndex], colTrueOffsetsIFRR[localInterfaceIndex]);
	Irhorho->SetBlock(1, 1, new IdentityOperator(iH1fespace[interfaceIndex]->GetTrueVSize()));
	
	return new SumOperator(new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false), Irhorho, false, false, 1.0, 1.0);
      }
    else
      return new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false);
#else
    return new TripleProductOperator(blockInjectionLeft, CijS, blockInjectionRight, false, false, false);
#endif
  }

  void CreateSubdomainMatrices(const int subdomain)
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
    MFEM_VERIFY(m_rank == 0, "");  // TODO: this code will not work in parallel, in general. 

    {
      Array<int> true_ess_dofs(fespace[subdomain]->GetTrueVSize());
      for (int i=0; i<fespace[subdomain]->GetTrueVSize(); ++i)
	true_ess_dofs[i] = 0;

      for (set<int>::const_iterator it = tdofsBdry[subdomain].begin(); it != tdofsBdry[subdomain].end(); ++it)
	true_ess_dofs[*it] = 1;

      MFEM_VERIFY(InterfaceToSurfaceInjectionData[subdomain].size() == subdomainLocalInterfaces[subdomain].size(), "");

      set<int> interfaceTDofs;  // True DOF's of fespace[subdomain] which lie on an interface.

      for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
	{
	  const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];
	  const int ifli = (*interfaceLocalIndex)[interfaceIndex];
	  MFEM_VERIFY(ifli >= 0, "");

	  for (int j=0; j<InterfaceToSurfaceInjectionData[subdomain][i].size(); ++j)
	    {
	      interfaceTDofs.insert(InterfaceToSurfaceInjectionData[subdomain][i][j]);
	      true_ess_dofs[InterfaceToSurfaceInjectionData[subdomain][i][j]] = 0;
	    }
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
  
  Solver* CreateSubdomainPreconditionerStrumpack(const int subdomain)
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

  void SetOffsetsSD(const int subdomain)
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
	
	MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
	MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
	
	trueOffsetsSD[subdomain][(2*i) + 2] += ifespace[interfaceIndex]->GetTrueVSize();
	trueOffsetsSD[subdomain][(2*i) + 3] += iH1fespace[interfaceIndex]->GetTrueVSize();
      }
    
    trueOffsetsSD[subdomain].PartialSum();
  }
  
  //#define SCHURCOMPSD
  
  // Create operator A_m for subdomain m, in the block space corresponding to [u_m, f_m^s, \rho_m^s].
  // We use mappings between interface and subdomain boundary DOF's, so there is no need for interior and surface blocks on each subdomain.
  Operator* CreateSubdomainOperator(const int subdomain)
  {
    BlockOperator *op = new BlockOperator(trueOffsetsSD[subdomain]);

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

		if (A_SS[subdomain] == NULL)
		  {
		    (*(ifNDmass[interfaceIndex])) *= -alpha;
		    (*(ifNDcurlcurl[interfaceIndex])) *= -beta;

		    HypreParMatrix *sumMassCC = ParAdd(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex]);
		    
		    (*(ifNDmass[interfaceIndex])) *= -1.0 / alpha;
		    (*(ifNDcurlcurl[interfaceIndex])) *= -1.0 / beta;

		    A_SS[subdomain] = AddSubdomainMatrixAndInterfaceMatrix(sdNDcopy[subdomain], sumMassCC, InterfaceToSurfaceInjectionData[subdomain][i],
									   ifespace[interfaceIndex], PENALTY_U_S);
		    delete sumMassCC;
		  }
		else
		  {
		    // TODO: add another interface operator to A_SS
		    MFEM_VERIFY(false, "TODO");
		  }
		
		op->SetBlock(0, 0, A_SS[subdomain]);
	      }
	    else
	      {
		// Sum operators abstractly, without adding matrices into a new HypreParMatrix.
		if (A_SS_op == NULL)
		  {
		    A_SS_op = new SumOperator(sdND[subdomain],
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
		else
		  {
		    // TODO: add another interface operator to A_SS_op
		    MFEM_VERIFY(false, "TODO");
		  }
		
		op->SetBlock(0, 0, A_SS_op);
	      }
	  }
	else  // if not real part
	  {
	    // Sum operators abstractly, without adding matrices into a new HypreParMatrix.
	    if (A_SS_op == NULL)
	      {
		A_SS_op = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][i],
						    new SumOperator(ifNDmass[interfaceIndex],
								    ifNDcurlcurl[interfaceIndex],
								    false, false, -alpha, -beta),
						    new TransposeOperator(InterfaceToSurfaceInjection[subdomain][i]),
						    false, false, false);
	      }
	    else
	      {
		// TODO: add another interface operator to A_SS_op
		MFEM_VERIFY(false, "TODO");
	      }
		
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
  
  // This is the same operator as CreateSubdomainOperator, except it is stored as a strumpack matrix rather than a block operator. 
  Operator* CreateSubdomainOperatorStrumpack(const int subdomain)
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
  
};
  
#endif  // DDOPER_HPP
