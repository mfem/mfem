#include "ddmesh.hpp"
#include "ddoper.hpp"


void NormalizeVector(MPI_Comm comm, Vector v)
{
  const double nrm = sqrt(InnerProduct(comm, v, v));
  v /= nrm;
}

InjectionOperator::InjectionOperator(MPI_Comm comm, ParFiniteElementSpace *subdomainSpace, FiniteElementSpace *interfaceSpace, int *a,
				     std::vector<int> const& gdofmap, std::vector<int> const& gflip) : id(a)
{
  m_comm = comm;
  
  MPI_Comm_size(comm, &m_nprocs);
  MPI_Comm_rank(comm, &m_rank);

  /*
  const HYPRE_Int sdtos = (subdomainSpace == NULL) ? 0 : subdomainSpace->GetMyTDofOffset();
  const int int_sdtos = sdtos; // TODO: what is HYPRE_Int as an MPI type?
  */
  
  const int sdtsize = (subdomainSpace == NULL) ? 0 : subdomainSpace->GetTrueVSize();
  const int iftsize = (interfaceSpace == NULL) ? 0 : interfaceSpace->GetTrueVSize();

  height = sdtsize;
  width = iftsize;
  //MFEM_VERIFY(height >= width, "InjectionOperator constructor");

  std::vector<int> allsdtos, alliftos, allsdtsize;
  allsdtos.assign(m_nprocs, 0);
  allsdtsize.assign(m_nprocs, 0);
  alliftos.assign(m_nprocs, 0);
  m_alliftsize.assign(m_nprocs, 0);
    
  MPI_Allgather(&sdtsize, 1, MPI_INT, allsdtsize.data(), 1, MPI_INT, comm);
  MPI_Allgather(&iftsize, 1, MPI_INT, m_alliftsize.data(), 1, MPI_INT, comm);

  for (int ifp=1; ifp<m_nprocs; ++ifp)
    {
      alliftos[ifp] = alliftos[ifp-1] + m_alliftsize[ifp-1];
      allsdtos[ifp] = allsdtos[ifp-1] + allsdtsize[ifp-1];
    }
  
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
#ifdef USE_SIGN_FLIP
  m_alltrueSDflip.assign(m_numTrueSD[m_rank], 0);
#endif
  
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
#ifdef USE_SIGN_FLIP
	      m_alltrueSDflip[os[ifp] + m_numLocalTrueSDmappedFromProc[ifp]] = gflip[alliftos[ifp] + i];
#endif
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

#ifdef AIRY_TEST
#include "gsl_sf_airy.h"

//#define K2_AIRY 1.0  // TODO: input k
//#define K2_AIRY 2.0  // TODO: input k
//#define K2_AIRY 686.3384931312 // 1.25 GHz
//#define K2_AIRY (2.0*686.3384931312) // 2.5 GHz
#define K2_AIRY 10981.4158900991  // 5 GHz
//#define K2_AIRY 43925.6635603965  // 10 GHz
//#define K2_AIRY 175702.65424  // 20 GHz
//#define K2_AIRY 1601.0

//#define AIRY_USE_Y

void test_Airy1_E_exact(const Vector &x, Vector &E)
{
  const double k = sqrt(K2_AIRY);  // TODO: input k
  const double beta = -pow(0.25 * k, 2.0/3.0);  // TODO: store this somehow?

  /*
  const double y = (4.0 * x(0)) - 1.0;
  E(0) = 0.0;
  E(1) = 0.0;
  E(2) = gsl_sf_airy_Ai(beta * y, GSL_PREC_DOUBLE);
  */

#ifdef AIRY_USE_Y
  const double y = (4.0 * x(1)) - 1.0;
#else
  const double y = (4.0 * x(2)) - 1.0;
#endif
  
  E(0) = gsl_sf_airy_Ai(beta * y, GSL_PREC_DOUBLE);
  E(1) = 0.0;
  E(2) = 0.0;

}

void test_Airy_epsilon(const Vector &x, Vector &e)
{
  /*
  e(0) = 1.0;
  e(1) = 1.0;
  e(2) = (4.0 * x(0)) - 1.0;
  */
  
#ifdef AIRY_USE_Y
  e(0) = (4.0 * x(1)) - 1.0;
#else
  e(0) = (4.0 * x(2)) - 1.0;
#endif
  e(1) = 1.0;
  e(2) = 1.0;

  
  e *= -K2_AIRY;
}

void Airy_epsilon(const Vector &x, Vector &e)
{
#ifdef AIRY_USE_Y
  e(0) = (4.0 * x(1)) - 1.0;
#else
  e(0) = (4.0 * x(2)) - 1.0;
#endif
  e(1) = 1.0;
  e(2) = 1.0;
}

void Airy_epsilon2(const Vector &x, Vector &e)
{
#ifdef AIRY_USE_Y
  e(0) = ((4.0 * x(1)) - 1.0) * ((4.0 * x(1)) - 1.0);
#else
  e(0) = ((4.0 * x(2)) - 1.0) * ((4.0 * x(2)) - 1.0);
#endif
  e(1) = 1.0;
  e(2) = 1.0;
}
#endif

#ifdef GPWD
GlobalPlaneParameters gpar;

void gpwd_ETE(const Vector &x, Vector &E)
{
  const double sinphi = sin(gpar.phi);
  const double cosphi = cos(gpar.phi);

  // Set (x,y) coordinates in the plane
  Vector xo(x);
  xo -= gpar.orig;
  const double xp = xo * gpar.bx;
  const double yp = xo * gpar.by;
  
  const double theta = gpar.k * ((cosphi * xp) + (sinphi * yp));
  
  E = gpar.bx;
  E *= sinphi;
  E.Add(-cosphi, gpar.by);

  if (gpar.Re)
    E *= cos(theta);  // real part
  else
    E *= sin(theta);  // imaginary part
}

void gpwd_bcr(const Vector &x, Vector &E)
{
  const double sinphi = sin(gpar.phi);
  const double cosphi = cos(gpar.phi);

  // Set (x,y) coordinates in the plane
  Vector xo(x);
  xo -= gpar.orig;
  const double xp = xo * gpar.bx;
  const double yp = xo * gpar.by;
  
  const double theta = gpar.k * ((cosphi * xp) + (sinphi * yp));
  
  E = gpar.bx;
  E *= cosphi;
  E.Add(sinphi, gpar.by);

  if (gpar.Re)
    E *= cos(theta);  // real part
  else
    E *= sin(theta);  // imaginary part
}

double gpwd_rhoc(const Vector &x)
{
  const double sinphi = sin(gpar.phi);
  const double cosphi = cos(gpar.phi);

  // Set (x,y) coordinates in the plane
  Vector xo(x);
  xo -= gpar.orig;
  const double xp = xo * gpar.bx;
  const double yp = xo * gpar.by;
  
  const double theta = gpar.k * ((cosphi * xp) + (sinphi * yp));
  
  if (gpar.Re)
    return cos(theta);  // real part
  else
    return sin(theta);  // imaginary part
}
#endif

void test2_E_exact(const Vector &x, Vector &E)
{
#ifdef AIRY_TEST
  test_Airy1_E_exact(x, E);
#else
  const double pi = M_PI;
  E(0) = sin(pi * x(1)) * sin(pi * x(2));
  E(1) = sin(pi * x(2)) * sin(pi * x(0));
  E(2) = sin(pi * x(0)) * sin(pi * x(1));
#endif
  
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

void test2_H_exact(const Vector &x, Vector &H)
{
#ifdef AIRY_TEST
  const double k = sqrt(K2_AIRY);  // TODO: input k
  const double beta = -pow(0.25 * k, 2.0/3.0);  // TODO: store this somehow?

#ifdef AIRY_USE_Y
  const double y = (4.0 * x(1)) - 1.0;
#else
  const double y = (4.0 * x(2)) - 1.0;
#endif

  H = 0.0;
 
#ifdef AIRY_USE_Y
  H(2) = -4.0 * beta * gsl_sf_airy_Ai_deriv(beta * y, GSL_PREC_DOUBLE);
#else
  H(1) = 4.0 * beta * gsl_sf_airy_Ai_deriv(beta * y, GSL_PREC_DOUBLE);
#endif
 
  H *= 1.0 / k;
  
#else
  H = 0.0;  // TODO
#endif
}

// Note that test2_f has zero tangential components on the exterior boundary.
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

  //int myos0 = 0;
  //int myos1 = 0;

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
      //const int osk = csr->i[k];
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

      /*
      bool nnzmatch = true;
      for (int i=0; i<globalNumRows; ++i)
	{
	  if (S->GetI()[i+1] != allnnz[i+1])
	    nnzmatch = false;
	}

      // Note that alldata may have some zeros, so S may be more compressed than alldata.
      */
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

  if (csr)
    delete csr;
  
  return S;
}

SparseMatrix* ReceiveSparseMatrix(MPI_Comm comm, const int source)
{
  int sizes[3];  // {matrix height, matrix width, number of nonzeros}

  MPI_Recv(sizes, 3, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);
  
  //MFEM_VERIFY(sizes[0] > 0 && sizes[1] > 0 && sizes[2] > 0, "");
  MFEM_VERIFY(sizes[0] > 0 && sizes[1] > 0, "");

  SparseMatrix *S = new SparseMatrix(sizes[0], sizes[1]);
  
  if (sizes[2] > 0)  // nnz
    {
      int *I = new int[sizes[0]+1];
      int *J = new int[sizes[2]];
      double *data = new double[sizes[2]];

      MPI_Recv(I, sizes[0]+1, MPI_INT, source, 1, comm, MPI_STATUS_IGNORE);
      MPI_Recv(J, sizes[2], MPI_INT, source, 2, comm, MPI_STATUS_IGNORE);
      MPI_Recv(data, sizes[2], MPI_DOUBLE, source, 3, comm, MPI_STATUS_IGNORE);

      MFEM_VERIFY(sizes[2] == I[sizes[0]], "");
  
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
    }
  
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

  if (sizes[2] > 0)
    {
      MPI_Send(S->GetI(), sizes[0]+1, MPI_INT, dest, 1, comm);
      MPI_Send(S->GetJ(), sizes[2], MPI_INT, dest, 2, comm);
      MPI_Send(S->GetData(), sizes[2], MPI_DOUBLE, dest, 3, comm);
    }
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
// Note that inj may be empty if ifespace == NULL, but ginj will always be the valid global injection map.
HypreParMatrix* AddSubdomainMatrixAndInterfaceMatrix(MPI_Comm ifcomm, HypreParMatrix *A,
#ifdef SERIAL_INTERFACES
						     SparseMatrix *I,
#else
						     HypreParMatrix *I,
#endif
						     std::vector<int> & inj, std::vector<int> & ginj,
#ifdef USE_SIGN_FLIP
						     std::vector<int> & gflip, const bool fliprows, const bool flipcols,
#endif
#ifdef SERIAL_INTERFACES						     
						     FiniteElementSpace *ifespace, FiniteElementSpace *ifespace2,
#else
						     ParFiniteElementSpace *ifespace, ParFiniteElementSpace *ifespace2,
#endif
						     const bool adding, const double cI, const double cA, const bool injRows)
{
  // inj maps from full local interface DOF's to local true DOF's in the subdomain. 

  // Gather the entire global interface matrix to one root process, namely rank 0 in the ifespace communicator.
#ifdef SERIAL_INTERFACES						     
  SparseMatrix *globalI = (I == NULL) ? NULL : new SparseMatrix(*I);
  if (globalI != NULL && cI != 0.0)
    {
      for (int i=0; i<globalI->Size(); ++i)
	{
	  globalI->Add(i, i, cI);
	}
    }
#else
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
#endif
  
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

  //cout << rank << ": ownerGlobalI " << ownerGlobalI << ", count " << count << ", nprocs " << nprocs << endl;
  
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

      //cout << rank << ": num rows " << csr_op->num_rows << ", RP0 " << A->RowPart()[0] << ", RP1 " << A->RowPart()[1] << endl;

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

#ifdef SERIAL_INTERFACES
  const int igsize = (I == NULL) ? 0 : I->Size();
#else
  const int igsize = (I == NULL) ? 0 : I->GetGlobalNumRows();
#endif
  
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
#ifdef SERIAL_INTERFACES
      const HYPRE_Int gtdof = i;
#else
      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
#endif
      
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
#ifdef SERIAL_INTERFACES
      const HYPRE_Int gtdof = i;
      const int ltdof = i;
#else
      const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(i);
      const int ltdof = ifespace->GetLocalTDofNumber(i);
#endif
      
      ifullsdt[i] = maxifgtsdt[gtdof];
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
#ifdef SERIAL_INTERFACES
      const int ltdof = i;
#else
      const int ltdof = ifespace2->GetLocalTDofNumber(i);
#endif
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

		  // TODO: find a better way to handle near-zero entries. Are they eliminated from the hypre matrix automatically?
		  if (fabs(d) < 1.0e-12 && colid == -1)
		    continue;

		  if (colid >= 0)
		    {
#ifdef USE_SIGN_FLIP
		      double s = (gflip[i] == 1 && fliprows) ? -1.0 : 1.0;
		      if (gflip[igcol] == 1 && flipcols)
			s *= -1.0;
		      
		      csr_op->data[colid] += (s * d);
#else
		      csr_op->data[colid] += d;
#endif
		    }
		  else
		    {
		      cout << "ERROR: allEntriesFound failure, d = " << d << endl;
		      allEntriesFound = false;
		    }
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
#ifdef SERIAL_INTERFACES
		  const HYPRE_Int gtdof = iftfull[i];
#else
		  const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(iftfull[i]);
#endif
		  SI[i+1] += globalI->GetI()[gtdof+1] - globalI->GetI()[gtdof];
		}
	    }
	  
	  // Partial sum
	  for (auto i=0; i<numLocRows; ++i)
	    SI[i+1] += SI[i];
	  
	  const HYPRE_Int nnz = SI[numLocRows];
	  HYPRE_Int *SJ = new HYPRE_Int[nnz+1];
	  double *Sdata = new double[nnz+1];

	  HYPRE_Int cnt = 0;

	  if (injRows) // Inject the rows from interface DOF's to subdomain DOF's.
	    {
#ifdef USE_SIGN_FLIP
	      MFEM_VERIFY(!flipcols, "");
#endif
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

#ifdef USE_SIGN_FLIP
			  const double s = (gflip[i] == 1 && fliprows) ? -1.0 : 1.0;
			  Sdata[SI[ltsd]+rowCount[ltsd]] = s * globalI->GetData()[k];
#else
			  Sdata[SI[ltsd]+rowCount[ltsd]] = globalI->GetData()[k];
#endif
			}
		    }
		}
	    }
	  else
	    {
	      MFEM_VERIFY(!mixed, "");
#ifdef USE_SIGN_FLIP
	      MFEM_VERIFY(!fliprows, "");
#endif

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
#ifdef SERIAL_INTERFACES
		  const HYPRE_Int gtdof = iftfull[i];
#else
		  const HYPRE_Int gtdof = ifespace->GetGlobalTDofNumber(iftfull[i]);
#endif
		  
		  for (int k=globalI->GetI()[gtdof]; k<globalI->GetI()[gtdof+1]; ++k, ++cnt, rowCount[i]++)
		    { // Copy values in row gtdof of globalI to local row i of S.
		      /*
		      if (maxifgtsdt[globalI->GetJ()[k]] < 0)
			cout << "BUG" << endl;
		      
		      SJ[SI[i]+rowCount[i]] = maxifgtsdt[globalI->GetJ()[k]];  // Column index in globalI is already global in ifespace or ifespace2. Map to global injected DOF.
		      */

		      SJ[SI[i]+rowCount[i]] = ginj[globalI->GetJ()[k]];  // Column index in globalI is already global in ifespace or ifespace2. Map to global injected DOF.

#ifdef USE_SIGN_FLIP
		      const double s = (gflip[globalI->GetJ()[k]] == 1 && flipcols) ? -1.0 : 1.0;
		      Sdata[SI[i]+rowCount[i]] = s * globalI->GetData()[k];
#else
		      Sdata[SI[i]+rowCount[i]] = globalI->GetData()[k];
#endif
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
#ifdef SERIAL_INTERFACES
	      const HYPRE_Int gtdof = mixed ? iftfull2[i] : iftfull[i];
#else
	      const HYPRE_Int gtdof = mixed ? ifespace2->GetGlobalTDofNumber(iftfull2[i]) : ifespace->GetGlobalTDofNumber(iftfull[i]);
#endif
	      SI[i+1] = SI[i] + globalImat->GetI()[gtdof+1] - globalImat->GetI()[gtdof];  // local offsets to rows
	    }

	  const HYPRE_Int nnz = SI[numLocRows];
	  HYPRE_Int *SJ = new HYPRE_Int[nnz];
	  double *Sdata = new double[nnz];

	  bool gciValid = true;
	  
	  HYPRE_Int cnt = 0;
	  for (int i=0; i<numLocRows; ++i)  // Loop over interface true DOF's, which correspond to local rows of S.
	    {
#ifdef SERIAL_INTERFACES
	      const HYPRE_Int gtdof = mixed ? iftfull2[i] : iftfull[i];
#else
	      const HYPRE_Int gtdof = mixed ? ifespace2->GetGlobalTDofNumber(iftfull2[i]) : ifespace->GetGlobalTDofNumber(iftfull[i]);
#endif
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
  
  if (csr_op)
    delete csr_op;
  
  return S;
}

// Row and column offsets are assumed to be the same, for each process.
// Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
// Different processes may have different numBlocks. To handle such generality, we use the block global indices blockGI. 
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
					       //#ifdef SERIAL_INTERFACES
					       Array2D<SparseMatrix*> const& blocksSp,
					       //#endif
					       Array2D<double> const& coefficient, std::vector<int> const& blockGI,
					       std::vector<std::vector<int> > const& blockProcOffsets,
					       std::vector<std::vector<int> > const& all_block_num_loc_rows)
{
  const int numBlocks = offsets.Size() - 1;

  /*
  MFEM_VERIFY(numBlocks == blockProcOffsets.size() && numBlocks == blockGI.size()
	      && numBlocks == all_block_num_loc_rows.size(), "");
  */
  
  const int num_loc_rows = offsets[numBlocks];

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> all_num_loc_rows(nprocs);
  std::vector<int> procOffsets(nprocs);
  /*
  std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
  std::vector<std::vector<int> > blockProcOffsets(numBlocks);
  */
  std::vector<std::vector<int> > procBlockOffsets(nprocs);

  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows.data(), 1, MPI_INT, comm);

  /*
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
  */
  
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
		  /*
		  const int nnz_k = csr_blocks(i, j)->i[k+1] - csr_blocks(i, j)->i[k];
		  const int osk = csr_blocks(i, j)->i[k];
		  */
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

// TODO: just have one version of CreateHypreParMatrixFromBlocks
// Row and column offsets are assumed to be the same, for each process.
// Array offsets stores process-local offsets with respect to the blocks. Process offsets are not included.
HypreParMatrix* blockgmg::CreateHypreParMatrixFromBlocks2(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
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

#ifdef MFEM_USE_STRUMPACK
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
#endif

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
bool FacesCoincideGeometrically(ParMesh *volumeMesh, const int face, ParMesh *surfaceMesh, const int elem, const bool surface)
{
  Array<int> faceVert;
  volumeMesh->GetFaceVertices(face, faceVert);

  Array<int> elemVert;
  if (surface)
    surfaceMesh->GetElementVertices(elem, elemVert);
  else
    surfaceMesh->GetFaceVertices(elem, elemVert);

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

// cp = a x b
void CrossProduct3D(Vector const& a, Vector const& b, Vector& cp)
{
  cp[0] = (a[1]*b[2]) - (a[2]*b[1]);
  cp[1] = (a[2]*b[0]) - (a[0]*b[2]);
  cp[2] = (a[0]*b[1]) - (a[1]*b[0]);
}

void GetFaceNormal(Mesh *mesh, const int face, Vector& normal, const bool elem=false)
{
  MFEM_VERIFY(mesh->SpaceDimension() == 3, "");

  Array<int> faceVert;
  if (elem)
    mesh->GetElementVertices(face, faceVert);
  else
    mesh->GetFaceVertices(face, faceVert);
  
  MFEM_VERIFY(faceVert.Size() > 2, "");

  Vector t1(3);
  Vector t2(3);

  for (int i=0; i<3; ++i)
    {
      t1[i] = mesh->GetVertex(faceVert[0])[i] - mesh->GetVertex(faceVert[1])[i];
      t2[i] = mesh->GetVertex(faceVert[2])[i] - mesh->GetVertex(faceVert[1])[i];
    }

  CrossProduct3D(t1, t2, normal);

  const double s = normal.Norml2();
  MFEM_VERIFY(s > 1.0e-8, "");
  
  normal /= s;
}

Mesh* Create2DMeshFrom3DSurfaceMesh(Mesh *mesh3D)
{
  MFEM_VERIFY(mesh3D->Dimension() == 2 && mesh3D->SpaceDimension() == 3, "");

  Mesh *mesh = new Mesh(2, mesh3D->GetNV(), mesh3D->GetNE());

  // Set the origin as the first vertex
  double *orig = mesh3D->GetVertex(0);

  // Plane basis vectors
  Vector b1(3);
  Vector b2(3);

  // Choose the first basis vector of the plane.
  {
    double *v1 = mesh3D->GetVertex(1);
    for (int i=0; i<3; ++i)
      b1[i] = v1[i] - orig[i];

    const double s = b1.Norml2();
    MFEM_VERIFY(s > 1.0e-8, "");

    b1 /= s;
  }

  // Choose the second basis vector of the plane.
  bool basisSet = false;
  for (int j=2; j<mesh3D->GetNV(); ++j)
    {
      double *v2 = mesh3D->GetVertex(j);
      for (int i=0; i<3; ++i)
	b2[i] = v2[i] - orig[i];

      const double dp = (b2 * b1);
      b2.Add(-dp, b1);
      
      const double s = b2.Norml2();

      if (s < 0.1)
	continue;
      
      b2 /= s;

      if ((b2 * b1) < 1.0e-8)
	{
	  basisSet = true;
	  break;
	}
    }
  
  MFEM_VERIFY(basisSet, "");

  double x[2];
  
  for (int i=0; i<mesh3D->GetNV(); ++i)
    {
      double *xi = mesh3D->GetVertex(i);

      x[0] = 0.0;
      x[1] = 0.0;
      for (int j=0; j<3; ++j)
	{
	  x[0] += b1[j] * (xi[j] - orig[j]);
	  x[1] += b2[j] * (xi[j] - orig[j]);
	}

      mesh->AddVertex(x);
    }

  const int elGeom = mesh3D->GetElementBaseGeometry(0);

  for (int i=0; i<mesh3D->GetNE(); ++i)
    {
      Element* el = mesh->NewElement(elGeom);
      el->SetAttribute(mesh3D->GetAttribute(i));

      Array<int> v;
      mesh3D->GetElementVertices(i, v);

      el->SetVertices(v);
      mesh->AddElement(el);
    }

  mesh->FinalizeTopology();

  return mesh;
}

int FindDofByPointValue(std::vector<double> const& facebv, const int dim, const int facendofs, const int facenip, const int face, const int dof,
			const int elem, ParFiniteElementSpace *fespace, Vector const& faceNormal, int& flip)
{
  // Note that the size of facebv is dim * (facendofs + 1) * facenip * numLocalFacesInInterface;

  //cout << "FindDofByPointValue ndofs " << facendofs << ", nip " << facenip << endl;
  
  const int osface = dim * (facendofs + 1) * facenip * face;

  const FiniteElement *fe = fespace->GetFE(elem);
  MFEM_VERIFY(fe->GetRangeType() != FiniteElement::SCALAR, "");

  ElementTransformation *Tr = fespace->GetElementTransformation(elem);

  Array<int> dofs;
  fespace->GetElementDofs(elem, dofs);
  const int ndofs = dofs.Size();

  int dofId = -1;
  //double dofsign = 1.0;
  {
    const int adof = (dof >= 0) ? dof : -1 - dof;
    for (int d=0; d<ndofs; ++d)
      {
	const int dof_d = (dofs[d] >= 0) ? dofs[d] : -1 - dofs[d];
	if (dof_d == adof)
	  {
	    MFEM_VERIFY(dofId == -1, "");
	    dofId = d;

	    /*
	    if ((dofs[d] < 0 && dof >= 0) || (dof < 0 && dofs[d] >= 0))
	      dofsign = -1.0;
	    */
	  }
      }
    
    MFEM_VERIFY(dofId >= 0, "");
  }
  
  IntegrationPoint ip;
  
  MFEM_VERIFY(dim == 3, "");
  
  Vector x(dim);  // Point in physical space.
  Vector v(dim);  // Basis function value at integration point.
  Vector fv(dim); // Basis function value at integration point, from facebv.
  Vector tmp(dim);
  
  DenseMatrix vshape(ndofs, dim);
  Vector data(ndofs);

  data = 0.0;
  data[dofId] = (dofs[dofId] >= 0) ? 1.0 : -1.0;

  Vector e(facendofs);

  int rdof = -1;
  
  for (int i=0; i<facenip; ++i)
    {
      for (int j=0; j<dim; ++j)
	x[j] = facebv[osface + (i * dim * (facendofs + 1)) + j];

      const bool found = (Tr->TransformBack(x, ip) == 0);
      MFEM_VERIFY(found, "");

      Tr->SetIntPoint(&ip);
      fe->CalcVShape(*Tr, vshape);

      vshape.MultTranspose(data, v);

      // Set v = n x (v x n).
      CrossProduct3D(v, faceNormal, tmp);
      CrossProduct3D(faceNormal, tmp, v);

      bool ambiguity_i = false;
      bool rdof_set_on_i = false;
      
      for (int signflip=0; signflip<2; ++signflip)
	{
	  double emin = 0.0;
	  int imin = 0;
      
	  for (int n=0; n<facendofs; ++n)
	    {
	      for (int j=0; j<dim; ++j)
		fv[j] = facebv[osface + (i * dim * (facendofs + 1)) + (dim * (1+n)) + j];

	      const double vnrm = std::max(v.Norml2(), fv.Norml2());

	      if (vnrm < 1.0e-4)
		{
		  e[n] = 1.0;
		  continue;
		}
	      
	      MFEM_VERIFY(vnrm > 1.0e-4, "");

	      if (signflip == 0)
		fv -= v;
	      else
		fv += v;
	  
	      e[n] = fv.Norml2() / vnrm;

	      if (n == 0 || e[n] < emin)
		{
		  emin = e[n];
		  imin = n;
		}
	    }

	  double emin2 = 0.0;
	  int imin2 = -1;

	  for (int n=0; n<facendofs; ++n)
	    {
	      if (n != imin && (imin2 == -1 || e[n] < emin2))
		{
		  emin2 = e[n];
		  imin2 = n;
		}
	    }

	  MFEM_VERIFY(imin != imin2 && imin2 >= 0, "");

	  if (emin2 > 1.0e-4 && emin / emin2 < 0.1)
	    {
	      if (rdof == -1)
		{
		  if (!ambiguity_i)
		    {
		      rdof = imin;
		      flip = signflip;
		      rdof_set_on_i = true;
		    }
		}
	      else
		{
		  //MFEM_VERIFY(rdof == imin && flip == signflip, "");
		  if (!(rdof == imin && flip == signflip))
		    {
		      ambiguity_i = true;

		      if (rdof_set_on_i)
			rdof = -1;
		    }
		}
	      
	      //return imin;
	    }
	}
    }
  
  return rdof;
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

// For full generality of parallel partitioning, we must allow for ifespace or fespace to be NULL (but not both), as this function
// needs to be called by all processes in ifsdcomm. There may be processes that touch an interface but not a subdomain, and vice versa.
//#define USE_IFMESH3D_VERT
void SetInterfaceToSurfaceDOFMap(MPI_Comm ifsdcomm,
#ifdef SERIAL_INTERFACES
				 FiniteElementSpace *ifespace,
				 const int interfaceFaceOffset,
				 std::vector<int> const& pmeshFacesInInterfaceOrdered,
#ifdef USE_IFMESH3D_VERT
				 Mesh *ifMesh3D,
#endif
#else
				 ParFiniteElementSpace *ifespace,
#endif
				 ParFiniteElementSpace *fespace, ParFiniteElementSpace *fespaceGlobal,
				 ParMesh *pmesh, const int sdAttribute,
				 std::set<int> const& pmeshFacesInInterface,
				 std::set<int> const& pmeshEdgesInInterface, std::set<int> const& pmeshVerticesInInterface, 
				 const FiniteElementCollection *fec, std::vector<int>& dofmap, //std::vector<int>& fdofmap,
				 std::vector<int>& gdofmap, std::set<int>& flippedSDDofs, std::vector<int>& gflip)
{
  const int ifSize = (ifespace == NULL) ? 0 : ifespace->GetVSize();  // Full DOF size
  const int iftSize = (ifespace == NULL) ? 0 : ifespace->GetTrueVSize();  // True DOF size, used for global gdofmap
  
#ifndef SERIAL_INTERFACES
  const HYPRE_Int iftos = (ifespace == NULL) ? 0 : ifespace->GetMyTDofOffset();
#endif
  const HYPRE_Int sdtos = (fespace == NULL) ? 0 : fespace->GetMyTDofOffset();
  
  int ifgSize = 0;

  // TODO: in general, this may need to be MPI_COMM_WORLD, since a process may see the subdomain but not the interface.
  //MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, ifespace->GetComm());
  //MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&iftSize, &ifgSize, 1, MPI_INT, MPI_SUM, ifsdcomm);

  //std::vector<int> fdofmap; // TODO: remove
  
  dofmap.assign(ifSize, -1);
  //fdofmap.assign(ifSize, -1);
  gdofmap.assign(ifgSize, -1);

#ifdef USE_SIGN_FLIP  
  gflip.assign(ifgSize, -1);
#endif
  
  std::vector<int> ifpedge, maxifpedge;
  ifpedge.assign(ifgSize, -1);
  maxifpedge.assign(ifgSize, -1);

  std::vector<int> ifpface, maxifpface;
  ifpface.assign(ifgSize, -1);
  maxifpface.assign(ifgSize, -1);
  
  const double vertexTol = 1.0e-12;

#ifdef SERIAL_INTERFACES
  Mesh *ifMesh = (ifespace == NULL) ? NULL : ifespace->GetMesh();  // Interface mesh
#else
  ParMesh *ifMesh = (ifespace == NULL) ? NULL : ifespace->GetParMesh();  // Interface mesh
#endif
  
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
	      std::set<int>::const_iterator itneg = pmeshFacesInInterface.find(-1 - elFaces[j]);
	      if (it != pmeshFacesInInterface.end() || itneg != pmeshFacesInInterface.end())
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

		  //bool onInterface = true;
		  bool touchingInterface = false;
		  
		  for (int l=0; l<2; ++l)
		    {
		      std::set<int>::const_iterator itv = pmeshVerticesInInterface.find(vert[l]);

		      /*
		      if (itv == pmeshVerticesInInterface.end())
			onInterface = false;
		      else
			touchingInterface = true;
		      */

		      if (itv != pmeshVerticesInInterface.end())
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

  // (1a) Loop over interface faces, only setting helper maps locally and globally (via some communication).

  std::map<HYPRE_Int, int> pgdofToFSet;

  int i = 0;
  
#ifdef SERIAL_INTERFACES
  std::vector<int> ifgi, iegi, ifeli;
  std::vector<double> ifemp;

  ifgi.assign(pmeshFacesInInterface.size(), -1);
  iegi.assign(pmeshEdgesInInterface.size(), -1);

  int numEdgesPerFace = 0;
  {
    Array<int> edge, ori;
    pmesh->GetFaceEdges(0, edge, ori);

    numEdgesPerFace = edge.Size();

    ifeli.assign(numEdgesPerFace * pmeshFacesInInterface.size(), -1);
    ifemp.assign(3 * numEdgesPerFace * pmeshFacesInInterface.size(), std::numeric_limits<double>::max());
  }

  std::map<int, int> pmeshEdgeToEdgeInInterface;

  for (std::set<int>::const_iterator it = pmeshEdgesInInterface.begin(); it != pmeshEdgesInInterface.end(); ++it, ++i)
    {
      const int pmeshEdge = *it;
      pmeshEdgeToEdgeInInterface[pmeshEdge] = i;
    }
  
  //bool nfpos = false;
  i = 0;
#endif

#ifdef SERIAL_INTERFACES
  /*
  // Copy all of the face indices in pmeshFacesInInterface as non-negative integers in another std::set for sorting in the right order.
  std::set<int> pmeshFacesInInterfaceSorted;

  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it)
    {
      const int pmeshFace = ((*it) >= 0) ? (*it) : -1 - (*it);
      pmeshFacesInInterfaceSorted.insert(pmeshFace);
    }

  for (std::set<int>::const_iterator it = pmeshFacesInInterfaceSorted.begin(); it != pmeshFacesInInterfaceSorted.end(); ++it, ++i)
  */
  for (std::vector<int>::const_iterator it = pmeshFacesInInterfaceOrdered.begin(); it != pmeshFacesInInterfaceOrdered.end(); ++it, ++i)
#else
  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it, ++i)  
#endif
    {
      const int drefit = *it;
      
      const int pmeshFace = ((*it) >= 0) ? (*it) : -1 - (*it);

      const int nf = fec->DofForGeometry(pmesh->GetFaceGeometryType(0));

#ifdef SERIAL_INTERFACES
      const bool faceInInterface = true;

      //if (nf > 0)
      //nfpos = true;
#else
      const bool faceInInterface = ((*it) >= 0);
#endif

      int osf = 0;
      
      /*
      double cm[3];  // center of mass for pmeshFace, for debugging purposes.

      {
	for (int l=0; l<3; ++l)
	  cm[l] = 0.0;
	
	Array<int> vert;
	pmesh->GetFaceVertices(pmeshFace, vert);
	for (int j=0; j<vert.Size(); ++j)
	  {
	    double *vc = pmesh->GetVertex(vert[j]);
	    for (int l=0; l<3; ++l)
	      cm[l] += vc[l];
	  }

	for (int l=0; l<3; ++l)
	  cm[l] /= (double) vert.Size();
      }
      */      
      
      { // Set pgdofToFSet, even if this edge is not in ifMesh
	const int nv = fec->DofForGeometry(Geometry::POINT);
	int numVerticesOnFace = 0;
	if (nv > 0)
	  {
	    Array<int> vert;
	    pmesh->GetFaceVertices(pmeshFace, vert);

	    numVerticesOnFace = vert.Size();
	  }

	const int ne = fec->DofForGeometry(Geometry::SEGMENT);
	int numEdgesOnFace = 0;
	if (ne > 0)
	  {
	    Array<int> edge, ori;
	    pmesh->GetFaceEdges(pmeshFace, edge, ori);

	    numEdgesOnFace = edge.Size();
	  }

	osf = (nv*numVerticesOnFace) + (ne*numEdgesOnFace);  // offset of face DOF's, which come after vertex and edge DOF's in the DOF arrays ifdofs and sddofs.
	
	Array<int> pdofs;
	fespaceGlobal->GetFaceDofs(pmeshFace, pdofs);

	if (nf > 0)
	  {
	    // Use the minimum global face DOF on the face, as a global identification for the face. The minimum resolves ambiguity resulting from orientations
	    // depending on the process.
	    /*
	    MFEM_VERIFY(nf == 2, "For nf > 2, generalize the computation of the minimum.");
	    const HYPRE_Int gtdof = std::min(fespaceGlobal->GetGlobalTDofNumber(pdofs[osf]), fespaceGlobal->GetGlobalTDofNumber(pdofs[osf+1]));
	    */

	    const int pdof_0 = (pdofs[osf] > 0) ? pdofs[osf] : -1 - pdofs[osf];
	    HYPRE_Int gtdof = fespaceGlobal->GetGlobalTDofNumber(pdof_0);
	    for (int j=1; j<nf; ++j)
	      {
		const int pdof_j = (pdofs[osf+j] > 0) ? pdofs[osf+j] : -1 - pdofs[osf+j];
		const HYPRE_Int gtdof_j = fespaceGlobal->GetGlobalTDofNumber(pdof_j);
		if (gtdof_j < gtdof)
		  gtdof = gtdof_j;
	      }
	    
	    pgdofToFSet[gtdof] = i;
	  }
      }

      if (faceInInterface && nf > 0)
	{
	  // Face pmeshFace of pmesh coincides with face i of ifMesh on this process

	  /*
	  int pmeshFaceOri = -1;
	  {
	    // Find the neighboring pmesh element.
	    std::map<int, int>::iterator ite = pmeshFaceToElem.find(pmeshFace);

	    if (ite == pmeshFaceToElem.end())  // This process does not have an element in this subdomain neighboring the face.
	      continue;
      
	    MFEM_VERIFY(ite->first == pmeshFace, "");

	    const int pmeshElem = ite->second;

	    MFEM_VERIFY(fespaceGlobal->GetOrder(pmeshElem) == 2, "hack only works for second order");

	    Array<int> pElFaces, pOri;
	    pmesh->GetElementFaces(pmeshElem, pElFaces, pOri);
	    for (int j=0; j<pElFaces.Size(); ++j)
	      {
		if (pElFaces[j] == pmeshFace)
		  pmeshFaceOri = pOri[j];
	      }
	  }

	  //cout << "Interface face " << i << " has pmeshFaceOri " << pmeshFaceOri << endl;

	  MFEM_VERIFY(pmeshFaceOri == 0 || pmeshFaceOri == 5, "pmeshFaceOri");
	  */

	  Array<int> pdofs, ifdofs;
	  fespaceGlobal->GetFaceDofs(pmeshFace, pdofs);

	  // Use the minimum global face DOF on the face, as a global identification for the face.
	  /*
	  MFEM_VERIFY(nf == 2, "For nf > 2, generalize the computation of the minimum.");
	  const HYPRE_Int gtdof = std::min(fespaceGlobal->GetGlobalTDofNumber(pdofs[osf]), fespaceGlobal->GetGlobalTDofNumber(pdofs[osf + 1]));
	  */

	  const int pdof_0 = (pdofs[osf] > 0) ? pdofs[osf] : -1 - pdofs[osf];
	  HYPRE_Int gtdof = fespaceGlobal->GetGlobalTDofNumber(pdof_0);
	  for (int j=1; j<nf; ++j)
	    {
	      const int pdof_j = (pdofs[osf+j] > 0) ? pdofs[osf+j] : -1 - pdofs[osf+j];
	      const HYPRE_Int gtdof_j = fespaceGlobal->GetGlobalTDofNumber(pdof_j);
	      if (gtdof_j < gtdof)
		gtdof = gtdof_j;
	    }
	  
#ifdef SERIAL_INTERFACES
	  ifgi[i] = gtdof;

	  {
	    Array<int> edge, ori;
	    pmesh->GetFaceEdges(pmeshFace, edge, ori);

	    MFEM_VERIFY(numEdgesPerFace == edge.Size(), "");
	    for (int j=0; j<numEdgesPerFace; ++j)
	      {
		std::map<int, int>::const_iterator it = pmeshEdgeToEdgeInInterface.find(edge[j]);
		
		MFEM_VERIFY(it != pmeshEdgeToEdgeInInterface.end(), "");
		MFEM_VERIFY(it->first == edge[j], "");
		
		ifeli[(numEdgesPerFace * i) + j] = it->second;

		Array<int> pmeshEdgeVert;
		pmesh->GetEdgeVertices(edge[j], pmeshEdgeVert);

		MFEM_VERIFY(pmeshEdgeVert.Size() == 2, "");
	
		for (int l=0; l<3; ++l)
		  ifemp[(3*((numEdgesPerFace * i) + j)) + l] = 0.5 * (pmesh->GetVertex(pmeshEdgeVert[0])[l] + pmesh->GetVertex(pmeshEdgeVert[1])[l]);
	      }
	  }
#else
	  ifespace->GetElementDofs(i, ifdofs);

	  MFEM_VERIFY(ifespace->GetOrder(i) == 2, "hack only works for second order");
	  MFEM_VERIFY(pdofs.Size() == ifdofs.Size() && pdofs.Size() == osf + nf, "hack only works for second order");

	  const HYPRE_Int ifgtdof = ifespace->GetGlobalTDofNumber(ifdofs[osf]);  // Just use the first face DOF on the face, as a global identification for the face.
	  MFEM_VERIFY(ifespace->GetGlobalTDofNumber(ifdofs[osf]) + 1 == ifespace->GetGlobalTDofNumber(ifdofs[osf+1]), "");
	  MFEM_VERIFY(ifdofs[osf] + 1 == ifdofs[osf+1], "");

	  ifpface[ifgtdof] = gtdof;
#endif

	  // TODO: is fswitch necessary?
	  //const int fswitch = 0; // (pmeshFaceOri > 0) ? (1 - (2*(d - osf))) : 0;  // TODO: this is valid only for second order!
	}
    }

#ifdef SERIAL_INTERFACES
  // TODO: the code will break for order 1.
  //MFEM_VERIFY(nfpos, "Otherwise ifeli, ifemp, etc. are not set, so even the edge DOF's are not mapped.");

  int myid, nprocs;
  MPI_Comm_size(ifsdcomm, &nprocs);
  MPI_Comm_rank(ifsdcomm, &myid);

  const int nie = pmeshEdgesInInterface.size();
  std::vector<int> allnie(nprocs);

  MPI_Gather(&nie, 1, MPI_INT, (int*) allnie.data(), 1, MPI_INT, 0, ifsdcomm);
    
  std::vector<int> dsple(nprocs);
  std::vector<int> allietoifedge;
  
  if (myid == 0)
    {
      int sumnie = 0;
      for (int i=0; i<nprocs; ++i)
	{
	  dsple[i] = sumnie;
	  sumnie += allnie[i];
	}

      allietoifedge.assign(sumnie, -1);
    }
  
  std::vector<int> allnif(nprocs);

  //if (nfpos)
    {
      // Set ifpface by using gathered ifgi values.
      
      const int nif = pmeshFacesInInterface.size();
      std::vector<int> allifos(nprocs);

      MPI_Gather(&nif, 1, MPI_INT, (int*) allnif.data(), 1, MPI_INT, 0, ifsdcomm);
      MPI_Gather(&interfaceFaceOffset, 1, MPI_INT, (int*) allifos.data(), 1, MPI_INT, 0, ifsdcomm);
    
      std::vector<int> allifgi;
      std::vector<int> dspl(nprocs);

      if (myid == 0)
	{
	  int sumnif = 0;
	  for (int i=0; i<nprocs; ++i)
	    {
	      dspl[i] = sumnif;
	      sumnif += allnif[i];

	      /*
	      if (dspl[i] != allifos[i])
		cout << "BUG" << endl;
	      
	      MFEM_VERIFY(dspl[i] == allifos[i], "");
	      */
	    }

	  MFEM_VERIFY(sumnif == ifMesh->GetNE(), "");

	  allifgi.assign(sumnif, -1);
	}

      MPI_Gatherv((int*) ifgi.data(), nif, MPI_INT, (int*) allifgi.data(), (int*) allnif.data(), (int*) dspl.data(), MPI_INT, 0, ifsdcomm);

      if (myid == 0)
	{
	  int osf = 0;
	  { // Set osf
	    const int nv = fec->DofForGeometry(Geometry::POINT);
	    int numVerticesOnFace = 0;
	    if (nv > 0)
	      {
		Array<int> vert;
		ifMesh->GetElementVertices(0, vert);

		numVerticesOnFace = vert.Size();
	      }

	    const int ne = fec->DofForGeometry(Geometry::SEGMENT);
	    int numEdgesOnFace = 0;
	    if (ne > 0)
	      {
		Array<int> edge, ori;
		ifMesh->GetElementEdges(0, edge, ori);

		numEdgesOnFace = edge.Size();
	      }

	    osf = (nv*numVerticesOnFace) + (ne*numEdgesOnFace);  // offset of face DOF's, which come after vertex and edge DOF's in the DOF arrays ifdofs and sddofs.
	  }

	  for (int p=0; p<nprocs; ++p)
	    {
	      for (int i=0; i<allnif[p]; ++i)  // Loop over pmesh faces in interface on proc p
		{
		  const int ifMeshElem = allifos[p] + i;  // The corresponding element index in ifMesh.

		  Array<int> ifdofs;
		  ifespace->GetElementDofs(ifMeshElem, ifdofs);

		  //MFEM_VERIFY(ifespace->GetOrder(ifMeshElem) == 2, "hack only works for second order");
		  MFEM_VERIFY(ifdofs[osf] + 1 == ifdofs[osf+1], "");

		  const HYPRE_Int ifgtdof = ifdofs[osf];  // Just use the first face DOF on the face, as a global identification for the face.
		  ifpface[ifgtdof] = allifgi[dspl[p] + i];
		}
	    }
	}

      // Set allietoifedge by using gathered ifeli and ifemp values.

      std::vector<int> allifeli;
      std::vector<double> allifemp;

      if (myid == 0)
	{
	  for (int i=0; i<nprocs; ++i)
	    {
	      dspl[i] *= numEdgesPerFace;
	      allnif[i] *= numEdgesPerFace;  // counts
	    }

	  allifeli.assign(numEdgesPerFace * allifgi.size(), -1);
	}
      
      MPI_Gatherv((int*) ifeli.data(), numEdgesPerFace * nif, MPI_INT, (int*) allifeli.data(), (int*) allnif.data(), (int*) dspl.data(), MPI_INT, 0, ifsdcomm);

      if (myid == 0)
	{
	  for (int i=0; i<nprocs; ++i)
	    {
	      dspl[i] *= 3;
	      allnif[i] *= 3;  // counts
	    }

	  allifemp.assign(3 * numEdgesPerFace * allifgi.size(), -1);
	}
      
      MPI_Gatherv((double*) ifemp.data(), 3 * numEdgesPerFace * nif, MPI_DOUBLE, (double*) allifemp.data(), (int*) allnif.data(),
		  (int*) dspl.data(), MPI_DOUBLE, 0, ifsdcomm);

      if (myid == 0)
	{
	  for (int p=0; p<nprocs; ++p)
	    {
	      allnif[p] /= (3 * numEdgesPerFace);  // restore the original value
	      dspl[p] /= (3 * numEdgesPerFace);  // restore the original value
	      
	      for (int i=0; i<allnif[p]; ++i)  // Loop over pmesh faces in interface on proc p
		{
		  const int ifMeshElem = allifos[p] + i;  // The corresponding element index in ifMesh.

		  Array<int> ifEdge, ifOri;
		  ifMesh->GetElementEdges(ifMeshElem, ifEdge, ifOri);
		  
		  MFEM_VERIFY(numEdgesPerFace == ifEdge.Size(), "");
		  
		  // Do a double-loop over numEdgesPerFace midpoints of edges in ifEdge and allifemp to find geometric matches.
		  for (int j=0; j<numEdgesPerFace; ++j)
		    {
		      int ifMeshEdge = -1;
		      
		      for (int k=0; k<numEdgesPerFace; ++k)
			{
			  Array<int> ifmeshEdgeVert;
			  ifMesh->GetEdgeVertices(ifEdge[k], ifmeshEdgeVert);

			  MFEM_VERIFY(ifmeshEdgeVert.Size() == 2, "");

			  bool pointsEqual = true;
			  for (int l=0; l<3; ++l)
			    {
#ifdef USE_IFMESH3D_VERT
			      const double ifMeshEdgeMidpoint_l = 0.5 * (ifMesh3D->GetVertex(ifmeshEdgeVert[0])[l] + ifMesh3D->GetVertex(ifmeshEdgeVert[1])[l]);
#else
			      const double ifMeshEdgeMidpoint_l = 0.5 * (ifMesh->GetVertex(ifmeshEdgeVert[0])[l] + ifMesh->GetVertex(ifmeshEdgeVert[1])[l]);
#endif

			      if (fabs(ifMeshEdgeMidpoint_l - allifemp[(3*numEdgesPerFace * (dspl[p] + i)) + (3*j) + l]) > vertexTol)
				pointsEqual = false;
			    }

			  if (pointsEqual)
			    {
			      MFEM_VERIFY(ifMeshEdge == -1, "");
			      ifMeshEdge = ifEdge[k];
			    }
			}
		      
		      MFEM_VERIFY(ifMeshEdge >= 0, "");

		      allietoifedge[dsple[p] + allifeli[(numEdgesPerFace * (dspl[p] + i)) + j]] = ifMeshEdge;
		    }
		}
	    }
	}
    }

  std::vector<double> ifbv;
  int ifnip = -1;
  int ifndofs = -1;
  
  {
    // Scatter basis function values at integration points on every ifMesh element.
    const int dim = pmesh->SpaceDimension();
    MFEM_VERIFY(dim == 3, "");
    const FiniteElement *fe0 = fespace->GetFE(0);

    MFEM_VERIFY(fe0->GetRangeType() != FiniteElement::SCALAR, "");

    std::vector<int> scnt(nprocs);
    std::vector<int> sdspl(nprocs);
    std::vector<double> vals;

    int ssizes[2];
    
    if (myid == 0)
      {
	// The following code is based on GridFunction::ComputeLpError and GridFunction::GetVectorValue
	
	const FiniteElement *ife0 = ifespace->GetFE(0);
	MFEM_VERIFY(fe0->GetOrder() == ife0->GetOrder(), "");

	int intorder = (2*ife0->GetOrder()) + 1;  // works for tets
	//int intorder = (3*ife0->GetOrder()) + 1;
	const IntegrationRule *ir = &(IntRules.Get(ife0->GetGeomType(), intorder));
	//const IntegrationRule *ir = &(RefinedIntRules.Get(ife0->GetGeomType(), intorder));

	const int nip = ir->Size();

	Array<int> dofs;
	ifespace->GetElementDofs(0, dofs);
	const int ndofs = dofs.Size();
	Vector data(ndofs);
#ifdef USE_IFMESH3D_VERT
	DenseMatrix vshape(ndofs, 2);
 	Vector v(2);
#else
	DenseMatrix vshape(ndofs, dim);
 	Vector v(dim);
#endif
	
	vals.resize(dim * (ndofs + 1) * nip * ifMesh->GetNE());
	
	ssizes[0] = ndofs;
	ssizes[1] = nip;
	
	sdspl[0] = 0;
	for (int i=0; i<nprocs; ++i)
	  {
	    scnt[i] = dim * (ndofs + 1) * nip * allnif[i];
	    if (i > 0)
	      sdspl[i] = sdspl[i-1] + scnt[i-1];
	  }
	
	int cnt = 0;
	for (int f=0; f<ifMesh->GetNE(); ++f)
	  {
	    const FiniteElement *ife = ifespace->GetFE(f);
	    MFEM_VERIFY(fe0->GetOrder() == ife->GetOrder(), "");
	    MFEM_VERIFY(ife->GetRangeType() != FiniteElement::SCALAR, "");

	    ElementTransformation *Tr = ifespace->GetElementTransformation(f);

	    ifespace->GetElementDofs(f, dofs);
	    MFEM_VERIFY(ndofs == dofs.Size() && ndofs == ife->GetDof(), "");
	    
	    for (int i=0; i<nip; ++i)
	      {
		IntegrationPoint const& ip = ir->IntPoint(i);
		Vector ipx(dim);
		Tr->Transform(ip, ipx);

		for (int j=0; j<dim; ++j, ++cnt)
		  {
		    vals[cnt] = ipx[j];
		  }

		Tr->SetIntPoint(&ip);
		ife->CalcVShape(*Tr, vshape);

		for (int d=0; d<ndofs; ++d)
		  {
		    data = 0.0;
		    data[d] = (dofs[d] >= 0) ? 1.0 : -1.0;

		    vshape.MultTranspose(data, v);
		    
		    for (int j=0; j<dim; ++j, ++cnt)
		      {
			vals[cnt] = v[j];
		      }
		  }
	      }
	  }

	MFEM_VERIFY(vals.size() == cnt, "");
      }

    MPI_Bcast(ssizes, 2, MPI_INT, 0, ifsdcomm);
    ifndofs = ssizes[0];
    ifnip = ssizes[1];

    const int nif = pmeshFacesInInterface.size();

    const int nrecv = dim * (ifndofs + 1) * ifnip * nif;
    
    ifbv.resize(nrecv);
    
    MPI_Scatterv((double*) vals.data(), (int*) scnt.data(), (int*) sdspl.data(), MPI_DOUBLE, (double*) ifbv.data(), nrecv, MPI_DOUBLE, 0, ifsdcomm);
  }
#endif

  MPI_Allreduce((int*) ifpface.data(), (int*) maxifpface.data(), ifgSize, MPI_INT, MPI_MAX, ifsdcomm);
  
  std::vector<int> pfaceToIFGDOF;
  pfaceToIFGDOF.assign(pmeshFacesInInterface.size(), -1);
  
  for (i=0; i<ifgSize; ++i)
    {
      const HYPRE_Int pgtdof = maxifpface[i]; // minimum of fespaceGlobal->GetGlobalTDofNumber() among face DOF's on its face.
      if (pgtdof >= 0)
	{
	  // Check whether pgtdof is on one of the pmesh faces in the interface on this process.
	  std::map<HYPRE_Int, int>::iterator it = pgdofToFSet.find(pgtdof);
	  if (it != pgdofToFSet.end())
	    {
	      MFEM_VERIFY(it->first == pgtdof, "");
	      const int pmfi = it->second;  // index in pmeshFacesInInterface
	      MFEM_VERIFY(pfaceToIFGDOF[pmfi] == -1 || pfaceToIFGDOF[pmfi] == i, "");
	      pfaceToIFGDOF[pmfi] = i;
	    }
	}
    }
	
  // (1b) Loop over interface faces.
  std::map<int, int> pmeshEdgeToAnyFaceInInterface;
  std::map<int, int> pmeshFaceToIFFace;

#ifdef SERIAL_INTERFACES
  i = 0; // interfaceFaceOffset;
#else
  i = 0;
#endif

#ifdef SERIAL_INTERFACES
  //for (std::set<int>::const_iterator it = pmeshFacesInInterfaceSorted.begin(); it != pmeshFacesInInterfaceSorted.end(); ++it, ++i)
  for (std::vector<int>::const_iterator it = pmeshFacesInInterfaceOrdered.begin(); it != pmeshFacesInInterfaceOrdered.end(); ++it, ++i)
#else
  for (std::set<int>::const_iterator it = pmeshFacesInInterface.begin(); it != pmeshFacesInInterface.end(); ++it, ++i)
#endif
    {
      //const int pmeshFace = *it;
      const int pmeshFace = ((*it) >= 0) ? (*it) : -1 - (*it);

#ifdef SERIAL_INTERFACES
      const bool faceInInterface = false;
#else
      const bool faceInInterface = ((*it) >= 0);
#endif
     
      if (faceInInterface)
	{
	  // Set maps pmeshFaceToIFFace and pmeshEdgeToAnyFaceInInterface, used only in (2a).
	  pmeshFaceToIFFace[pmeshFace] = i;
      
	  // Face pmeshFace of pmesh coincides with face i of ifMesh on this process (the same face may also exist on a different process in the same ifMesh,
	  // as there can be redundant overlapping faces in parallel, for communication).

	  { // For each pmesh edge of this face, map to pmeshFace.
	    Array<int> edges, ori;
	    pmesh->GetFaceEdges(pmeshFace, edges, ori);

	    for (int j=0; j<edges.Size(); ++j)
	      pmeshEdgeToAnyFaceInInterface[edges[j]] = pmeshFace;
	  }
	}
      
      // If !faceInInterface, then this pmesh face is not in ifMesh, so do not set any entries of dofmap. Instead, set face DOF entries of gdofmap.
      // It is not necessary to set edge or vertex DOF entries of gdofmap here, as that is done later in loop (2) and could be implemented in (3).
      
      // Find the neighboring pmesh element.
      std::map<int, int>::iterator ite = pmeshFaceToElem.find(pmeshFace);

      //MFEM_VERIFY(ite != pmeshFaceToElem.end(), "");

      if (ite == pmeshFaceToElem.end())  // This process does not have an element in this subdomain neighboring the face.
	continue;
      
      MFEM_VERIFY(ite->first == pmeshFace, "");

      const int pmeshElem = ite->second;

      int pmeshFaceOri = -1;
      {
	Array<int> pElFaces, pOri;
	pmesh->GetElementFaces(pmeshElem, pElFaces, pOri);
	for (int j=0; j<pElFaces.Size(); ++j)
	  {
	    if (pElFaces[j] == pmeshFace)
	      pmeshFaceOri = pOri[j];
	  }

	if (pElFaces.Size() == 4) // tetrahedral case
	  MFEM_VERIFY(pmeshFaceOri == 0 || pmeshFaceOri == 5, "pmeshFaceOri");
      }

      //cout << "Interface face " << i << " has pmeshFaceOri " << pmeshFaceOri << endl;
      
      // Find the neighboring sdMesh element, which coincides with pmeshElem in pmesh.
      std::map<int, int>::const_iterator itse = pmeshElemToSDmesh.find(pmeshElem);
      MFEM_VERIFY(itse != pmeshElemToSDmesh.end(), "");
      MFEM_VERIFY(itse->first == pmeshElem, "");

      const int sdMeshElem = itse->second;

      // Find the face of element sdMeshElem in sdMesh that coincides geometrically with the current interface face.
      Array<int> elFaces, ori;

      sdMesh->GetElementFaces(sdMeshElem, elFaces, ori);
      int sdMeshFace = -1;
      //int sdMeshFaceOri = -1;
      for (int j=0; j<elFaces.Size(); ++j)
	{
	  //if (FacesCoincideGeometrically(sdMesh, elFaces[j], ifMesh, i, true))
	  if (FacesCoincideGeometrically(sdMesh, elFaces[j], pmesh, pmeshFace, false))
	    {
	      sdMeshFace = elFaces[j];
	      //sdMeshFaceOri = ori[j];
	    }
	}

      MFEM_VERIFY(sdMeshFace >= 0, "");

      // Map vertex DOF's on ifMesh face i to vertex DOF's on sdMesh face sdMeshFace.
      // TODO: is this necessary, since FiniteElementSpace::GetEdgeDofs claims to return vertex DOF's as well?
      const int nv = fec->DofForGeometry(Geometry::POINT);
      int numVerticesOnFace = 0;
      if (nv > 0)
	{
	  Array<int> sdVert;
	  sdMesh->GetFaceVertices(sdMeshFace, sdVert);

	  numVerticesOnFace = sdVert.Size();
	}

      if (nv > 0 && faceInInterface)
	{
	  Array<int> sdVert;
	  sdMesh->GetFaceVertices(sdMeshFace, sdVert);

	  Array<int> ifVert;
	  ifMesh->GetElementVertices(i, ifVert);
	  MFEM_VERIFY(ifVert.Size() == sdVert.Size(), "");
	  
	  for (int j=0; j<ifVert.Size(); ++j)
	    {
#ifdef USE_IFMESH3D_VERT
	      double *ifv = ifMesh3D->GetVertex(ifVert[j]);
#else
	      double *ifv = ifMesh->GetVertex(ifVert[j]);
#endif
	      
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

			  /*
			  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
			  fdofmap[ifdofs[d]] = sddofs[d];
			  */
			  
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
      int numEdgesOnFace = 0;
      if (ne > 0)
	{
	  Array<int> sdEdge, sdOri;
	  sdMesh->GetFaceEdges(sdMeshFace, sdEdge, sdOri);

	  numEdgesOnFace = sdEdge.Size();
	}

      if (ne > 0 && faceInInterface)
	{
	  // TODO: could there be multiple DOF's on an edge with different orderings (depending on orientation) in ifespace and fespace?
	  // TODO: Check orientation for ND_HexahedronElement? Does ND_TetrahedronElement have orientation?

	  Array<int> sdEdge, sdOri;
	  sdMesh->GetFaceEdges(sdMeshFace, sdEdge, sdOri);
	  
	  Array<int> ifEdge, ifOri;
	  ifMesh->GetElementEdges(i, ifEdge, ifOri);
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
#ifdef USE_IFMESH3D_VERT
		      double *ifv = ifMesh3D->GetVertex(ifVert[v]);
#else
		      double *ifv = ifMesh->GetVertex(ifVert[v]);
#endif
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

		  /*
		  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
		  fdofmap[ifdofs[d]] = sddofs[d];
		  */
		  
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
	  //cout << "nv ne nf " << nv << " " << ne << " " << nf << endl;

	  Array<int> sddofs, ifdofs;
	  fespace->GetFaceDofs(sdMeshFace, sddofs);

	  const int osf = (nv*numVerticesOnFace) + (ne*numEdgesOnFace);  // offset of face DOF's, which come after vertex and edge DOF's in the DOF arrays ifdofs and sddofs.

	  MFEM_VERIFY(sddofs.Size() == osf + nf, "");

	  if (faceInInterface)
	    {
	      ifespace->GetElementDofs(i, ifdofs);
	      MFEM_VERIFY(ifdofs.Size() == sddofs.Size(), "");
	      MFEM_VERIFY(ifespace->GetOrder(i) == 2, "fswitch hack only works for second order");
	    }

	  for (int d=osf; d<sddofs.Size(); ++d)
	    {
	      //const int fswitch = (pmeshFaceOri > 0) ? (1 - (2*(d - osf))) : 0; // ((d - osf) + 1) % 2;  // TODO: this is valid only for second order!
#ifdef SERIAL_INTERFACES
	      const int fswitch2 = 0;
	      //const int fswitch2 = (pmeshFaceOri == 0) ? (1 - (2*(d - osf))) : 0; // ((d - osf) + 1) % 2;  // TODO: this is valid only for second order!
#else
	      const int fswitch2 = (pmeshFaceOri == 0) ? (1 - (2*(d - osf))) : 0; // ((d - osf) + 1) % 2;  // TODO: this is valid only for second order!
#endif
	      
	      //MFEM_VERIFY(fespace->GetOrder(sdMeshElem) == 2, "fswitch hack only works for second order");

#ifdef SERIAL_INTERFACES
	      const int sdtdof = -1;  // not used
#else
	      const int sddof_d = (sddofs[d + fswitch] >= 0) ? sddofs[d + fswitch] : -1 - sddofs[d + fswitch];
	      const int sdtdof = fespace->GetLocalTDofNumber(sddof_d);
#endif

	      const int sddof_d2 = (sddofs[d + fswitch2] >= 0) ? sddofs[d + fswitch2] : -1 - sddofs[d + fswitch2];
	      //const int sdtdof = fespace->GetLocalTDofNumber(sddofs[d]);
	      const int sdtdof2 = fespace->GetLocalTDofNumber(sddof_d2);

	      /*
	      MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
	      fdofmap[ifdofs[d]] = sddofs[d];
			  
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
		{
		  MFEM_VERIFY(dofmap[ifdofs[d]] == sdtdof || dofmap[ifdofs[d]] == -1, "");
		  dofmap[ifdofs[d]] = sdtdof;
		}
	      */

	      /*
	      MFEM_VERIFY(fdofmap[ifdof_d] == sddof_d || fdofmap[ifdof_d] == -1, "");
	      fdofmap[ifdof_d] = sddof_d;
	      */
#ifdef SERIAL_INTERFACES
	      if (sdtdof2 >= 0)  // if this is a true DOF of fespace.
#else
	      if (sdtdof >= 0)  // if this is a true DOF of fespace.
#endif
		{
		  if (faceInInterface)
		    {
		      // Set local dofmap
		      
		      const int ifdof_d = (ifdofs[d] >= 0) ? ifdofs[d] : -1 - ifdofs[d];

		      if (!(dofmap[ifdof_d] == sdtdof || dofmap[ifdof_d] == -1))
			cout << "BUG" << endl;

		      /*
		      if (ifdof_d == 1600 || ifdof_d == 1601)
			{
			  cout << "ifdof " << ifdof_d << " set on face " << i << ", sdMeshFaceOri " << sdMeshFaceOri
			       << ", ifdofs[d] " << ifdofs[d] << ", sddofs[d] " << sddofs[d] << ", pmeshFaceOri "
			       << pmeshFaceOri << endl;
			}
		      */
		      
		      MFEM_VERIFY(dofmap[ifdof_d] == sdtdof || dofmap[ifdof_d] == -1, "");
		      dofmap[ifdof_d] = sdtdof;
		    }
		  else
		    {
		      // Set global gdofmap
		      MFEM_VERIFY(pfaceToIFGDOF[i] >= 0, "");

		      /*
		      if (pfaceToIFGDOF[i] == 1600)
			{
			  cout << "ifdof " << pfaceToIFGDOF[i] + d - osf << " set on face " << i << ", sdMeshFaceOri " << sdMeshFaceOri
			       << ", sddofs[d] " << sddofs[d] << ", pmeshFaceOri " << pmeshFaceOri << endl;
			}
		      */
		      

		      // Note that pfaceToIFGDOF[i] is the global DOF of the first face DOF of the face in ifMesh corresponding to face i.

#ifdef SERIAL_INTERFACES
		      Vector normal(3);
		      GetFaceNormal(sdMesh, sdMeshFace, normal);

		      int flip = 0;
		      const int faceDofIndex = FindDofByPointValue(ifbv, sdMesh->SpaceDimension(), ifndofs, ifnip, i, sddofs[d], sdMeshElem, fespace, normal, flip);
		      //cout << "FindDofByPointValue result " << faceDofIndex << endl;
		      MFEM_VERIFY(faceDofIndex >= 0 && faceDofIndex >= osf, "");
		      
		      /*
		      const int ifMeshFaceDofIndex = [pfaceToIFGDOF[i] + d - osf];
		      gdofmap[pfaceToIFGDOF[i] + ifMeshFaceDofIndex] = sdtdof2 + sdtos;
		      */
		      
		      if (flip == 0)
			flippedSDDofs.insert(sdtdof2);
		      else
			flippedSDDofs.insert(-1 - sdtdof2);

		      if (!(gdofmap[pfaceToIFGDOF[i] + faceDofIndex - osf] == sdtdof2 + sdtos || gdofmap[pfaceToIFGDOF[i] + faceDofIndex - osf] == -1))
			cout << "BUG" << endl;

		      MFEM_VERIFY(gdofmap[pfaceToIFGDOF[i] + faceDofIndex - osf] == sdtdof2 + sdtos || gdofmap[pfaceToIFGDOF[i] + faceDofIndex - osf] == -1, "");

		      gdofmap[pfaceToIFGDOF[i] + faceDofIndex - osf] = sdtdof2 + sdtos;

#ifdef USE_SIGN_FLIP
		      MFEM_VERIFY(gflip[pfaceToIFGDOF[i] + faceDofIndex - osf] == flip || gflip[pfaceToIFGDOF[i] + faceDofIndex - osf] == -1, "");
		      gflip[pfaceToIFGDOF[i] + faceDofIndex - osf] = flip;
#endif
#else
		      if (!(gdofmap[pfaceToIFGDOF[i] + d - osf] == sdtdof2 + sdtos || gdofmap[pfaceToIFGDOF[i] + d - osf] == -1))
			cout << "BUG" << endl;
		      MFEM_VERIFY(gdofmap[pfaceToIFGDOF[i] + d - osf] == sdtdof2 + sdtos || gdofmap[pfaceToIFGDOF[i] + d - osf] == -1, "");

		      gdofmap[pfaceToIFGDOF[i] + d - osf] = sdtdof2 + sdtos;
#endif
		    }
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

#ifdef SERIAL_INTERFACES
	iegi[i] = gtdof;
#endif
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
#ifdef USE_IFMESH3D_VERT
		const double ifMeshEdgeMidpoint_k = 0.5 * (ifMesh3D->GetVertex(ifmeshEdgeVert[0])[k] + ifMesh3D->GetVertex(ifmeshEdgeVert[1])[k]);
#else
		const double ifMeshEdgeMidpoint_k = 0.5 * (ifMesh->GetVertex(ifmeshEdgeVert[0])[k] + ifMesh->GetVertex(ifmeshEdgeVert[1])[k]);
#endif
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
#ifdef USE_IFMESH3D_VERT
			    if (fabs(pmesh->GetVertex(pmeshEdgeVert[k])[l] - ifMesh3D->GetVertex(ifmeshEdgeVert[m])[l]) > vertexTol)
#else
			    if (fabs(pmesh->GetVertex(pmeshEdgeVert[k])[l] - ifMesh->GetVertex(ifmeshEdgeVert[m])[l]) > vertexTol)
#endif
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
#ifdef SERIAL_INTERFACES
	  const HYPRE_Int ifgtdof = ifdofs[0];  // Just use the first DOF on the edge, as a global identification for the edge.
#else
	  const HYPRE_Int ifgtdof = ifespace->GetGlobalTDofNumber(ifdofs[0]);  // Just use the first DOF on the edge, as a global identification for the edge.
#endif
	  
	  { // verification of a property assumed in the following code
	    bool continuousEdgeDOFs = true;
	    HYPRE_Int ifgtdofprev = ifgtdof;
	    for (int l=1; l<ifdofs.Size(); ++l)
	      {
#ifdef SERIAL_INTERFACES
		const HYPRE_Int ifgtdof_l = ifdofs[l];
#else
		const HYPRE_Int ifgtdof_l = ifespace->GetGlobalTDofNumber(ifdofs[l]);
#endif
		
		if (ifgtdof_l != ifgtdofprev + 1)
		  continuousEdgeDOFs = false;
		
		ifgtdofprev = ifgtdof_l;
		
		if (ifdofs[l] != ifdofs[l-1]+1)
		  continuousEdgeDOFs = false;
	      }
	    
	    MFEM_VERIFY(continuousEdgeDOFs, "continuousEdgeDOFs");
	  }
	  
#ifdef SERIAL_INTERFACES
	  MFEM_VERIFY(false, "This should not be reached in the case SERIAL_INTERFACES.");
#endif

	  ifpedge[ifgtdof] = gtdof;
	}
      }
    }

#ifdef SERIAL_INTERFACES
  // Set ifpedge by using gathered iegi values.
  {
    std::vector<int> alliegi;

    if (myid == 0)
      {
	int sumnie = 0;
	for (int i=0; i<nprocs; ++i)
	  {
	    sumnie += allnie[i];
	  }

	alliegi.assign(sumnie, -1);
      }

    MPI_Gatherv((int*) iegi.data(), nie, MPI_INT, (int*) alliegi.data(), (int*) allnie.data(), (int*) dsple.data(), MPI_INT, 0, ifsdcomm);

    if (myid == 0)
      {
	int ose = 0;
	{
	  const int nv = fec->DofForGeometry(Geometry::POINT);
	  ose = (2*nv);  // offset of edge DOF's, which come after vertex DOF's.
	}

	for (int p=0; p<nprocs; ++p)
	  {
	    for (int i=0; i<allnie[p]; ++i)  // Loop over pmesh edges in interface on proc p
	      {
		const int ifMeshEdge = allietoifedge[dsple[p] + i];  // The corresponding edge index in ifMesh.

		Array<int> ifdofs;
		ifespace->GetEdgeDofs(ifMeshEdge, ifdofs);

		const HYPRE_Int ifgtdof = ifdofs[ose];  // Just use the first edge DOF on the edge, as a global identification for the edge.
		ifpedge[ifgtdof] = alliegi[dsple[p] + i];
	      }
	  }
      }
  }
#endif
  
#ifdef SERIAL_INTERFACES
  // TODO: some MPI calls like this can be eliminated in the serial interface case.
#endif

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
	  MFEM_VERIFY(false, "BUG: i is not the index of the edge in ifMesh!");
	  ifMesh->GetEdgeVertices(i, ifVert);
	  sdMesh->GetEdgeVertices(sdMeshEdge, sdVert);

	  MFEM_VERIFY(ifVert.Size() == sdVert.Size(), "");
	  
	  for (int j=0; j<ifVert.Size(); ++j)
	    {
#ifdef USE_IFMESH3D_VERT
	      double *ifv = ifMesh3D->GetVertex(ifVert[j]);
#else
	      double *ifv = ifMesh->GetVertex(ifVert[j]);
#endif

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

			  /*
			  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
			  fdofmap[ifdofs[d]] = sddofs[d];
			  */
			  
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
	      
#ifdef SERIAL_INTERFACES
	      const int ltdof = ifdofs[0];
#else
	      const int ltdof = ifespace->GetLocalTDofNumber(ifdofs[0]);
	      
	      if (ltdof >= 0)
		{
		  const bool check = (pedgeToIFGDOF[i] == iftos + ltdof);

		  if (!check)
		    cout << "Failure on " << ltdof << endl;
	      
		  MFEM_VERIFY(pedgeToIFGDOF[i] == iftos + ltdof, "");
		}
#endif
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
		      MFEM_VERIFY(gdofmap[pedgeToIFGDOF[i] + d] == -1 || gdofmap[pedgeToIFGDOF[i] + d] == sdtdof + sdtos, "");
		      gdofmap[pedgeToIFGDOF[i] + d] = sdtdof + sdtos;
		    }
		}

	      /*
	      if (ifMeshEdge >= 0)
		{ // Set local fdofmap
		  MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
		  fdofmap[ifdofs[d]] = sddofs[d];
		}
	      */
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

	      //MFEM_VERIFY(fdofmap[ifdofs[d]] == sddofs[d] || fdofmap[ifdofs[d]] == -1, "");
	      //fdofmap[ifdofs[d]] = sddofs[d];
	      
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
#ifdef SERIAL_INTERFACES	    
	    //const int gtdof = i;
#else
	    const int gtdof = ifespace->GetGlobalTDofNumber(i);
	    
	    MFEM_VERIFY(gdofmap[gtdof] == -1 || gdofmap[gtdof] == dofmap[i] + sdtos, "");
	    gdofmap[gtdof] = dofmap[i] + sdtos;
#endif
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

#ifdef USE_SIGN_FLIP
    ifpedge = gflip;
    MPI_Allreduce((int*) ifpedge.data(), (int*) gflip.data(), ifgSize, MPI_INT, MPI_MAX, ifsdcomm);

    /*
    for (i=0; i<ifgSize; ++i)
      {
	if (gflip[i] < 0)
	  MFEM_VERIFY(gflip[i] >= 0, "");
      }
    */
#endif
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
  auto ssd_gf_host = ssd_gf.HostRead();

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

	  //MFEM_VERIFY(!((dofs[i] >= 0 && sddofs[i] < 0) || (dofs[i] < 0 && sddofs[i] >= 0)), "TODO: flip the sign as in SetSubdomainDofsFromDomainDofs!");
	  
	  if (ldof >= 0)
	    {
	      const double flip = ((dofs[i] >= 0 && sddofs[i] < 0) || (dofs[i] < 0 && sddofs[i] >= 0)) ? -1.0 : 1.0;
	      s[ldof] = ssd_gf_host[sddof_i] * flip;
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
	      /*
	      if ((dofs[i] >= 0 && sddofs[i] < 0) || (dofs[i] < 0 && sddofs[i] >= 0))
		{
		  cout << "Flip applied" << endl;
		  // Note that this occurs in parallel (1,2,1) for 4 SD's but the solution is the same as in serial. The flip seems correct.
		}
	      */

	      // TODO: It seems that this sign flipping may not be sufficient in general. So far, the debugging code below has not caught any
	      // incorrect signs, which may be the result of consistent vertex ordering in the subdomain and global domain meshes.
	      // In general, it may be necessary to multiply by the sign of the dot product of the tangent vectors for the edge in the
	      // subdomain mesh and the edge in the global domain mesh.
	      
	      const double flip = ((dofs[i] >= 0 && sddofs[i] < 0) || (dofs[i] < 0 && sddofs[i] >= 0)) ? -1.0 : 1.0;
	      ssd[lsddof] = s_gf[dof_i] * flip;
	    }
	}
    }

  { // debugging
    // Test orientations of the same DOF's in fespaceDomain and fespaceSD, by comparing projections of the same analytic function.
    VectorFunctionCoefficient E(3, test2_E_exact);

    ParGridFunction gfsd(fespaceSD);

    gfsd.ProjectCoefficient(E);
    s_gf.ProjectCoefficient(E);

    Vector vsd(fespaceSD->GetTrueVSize());
    Vector vg(fespaceDomain->GetTrueVSize());

    gfsd.GetTrueDofs(vsd);
    s_gf.GetTrueDofs(vg);

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

		if (fabs(vsd[lsddof] - (flip * s_gf[dof_i])) > 1.0e-8)
		  {
		    cout << "DISCREPANCY in sd/global map" << endl;
		  }
	      }
	  }
      }

    // Check whether all true SD DOF's get set.
    vsd = 0.0;
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
		vsd[lsddof] = 1.0;
	      }
	  }
      }

    for (int i=0; i<vsd.Size(); ++i)
      {
	if (fabs(vsd[i] - 1.0) > 1.0e-8)
	  cout << "DISCREPANCY BUG: not all SD DOF's set." << endl;
      }
  }
  
}

void GatherSetInt(MPI_Comm comm, std::vector<int> const& a, std::set<int>& allset)
{
  int rank, nprocs;
  
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &rank);

  const int s = a.size();
  
  std::vector<int> rcnt(nprocs);
  MPI_Allgather(&s, 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);
  
  std::vector<int> rdspl(nprocs);
  rdspl[0] = 0;
  
  for (int i=1; i<nprocs; ++i)
    {
      rdspl[i] = rdspl[i-1] + rcnt[i-1];
    }

  std::vector<int> recv;
  recv.assign(rdspl[nprocs-1] + rcnt[nprocs-1], 0);
  
  MPI_Allgatherv(a.data(), s, MPI_INT, recv.data(), rcnt.data(), rdspl.data(), MPI_INT, comm);

  for (auto i : recv)
    allset.insert(i);
}

void CompareOperators(Operator *A, Operator *B, MPI_Comm comm)
{
  MPI_Barrier(comm);

  const int n = A->Width();
  
  MFEM_VERIFY(A->Height() == n && B->Height() == n && B->Width() == n, "");

  Vector ej(n);
  Vector Aej(n);
  Vector Bej(n);
  
  for (int j=0; j<n; ++j)
    {
      ej = 0.0;
      ej[j] = 1.0;

      A->Mult(ej, Aej);
      B->Mult(ej, Bej);

      const double nrmA = Aej.Norml2();
      const double nrmB = Bej.Norml2();
      const double nrm = std::max(nrmA, nrmB);

      MFEM_VERIFY((nrm > 1.0e-8), "");

      Bej -= Aej;

      const double err = Bej.Norml2();

      if (err / nrm > 1.0e-8 || j % 100 == 0)
	cout << "WARNING: large operator diff " << err << " relative to " << nrm << ", " << j << " of " << n << endl;
    }
}

void BlockSubdomainPreconditioner::Mult(const Vector & x, Vector & y) const
{
  // TODO: In the case of ROBIN_TC, instead of block diagonal preconditioning, just solve the block lower triangular system?

  auto x_host = x.HostRead();

  // First, precondition the complex auxiliary block.
  
  for (int i=0; i<nAux; ++i) // Extract the complex auxiliary entries from x.
    {
      xaux[i] = x_host[nSD + i];
      xaux[nAux + i] = x_host[osIm + nSD + i];
    }
  
  invAuxComplex->Mult(xaux, yaux);

  for (int i=0; i<nAux; ++i)
    {
      y[nSD + i] = yaux[i];
      y[osIm + nSD + i] = yaux[nAux + i];
    }
  
#ifdef SD_ITERATIVE_COMPLEX
  for (int i=0; i<nSD; ++i)
    {
      xsd[i] = x_host[i];
      xsd[nSD + i] = x_host[osIm + i];
    }

  invSD->Mult(xsd, ysd);
  
  for (int i=0; i<nSD; ++i)
    {
      y[i] = ysd[i];
      y[osIm + i] = ysd[nSD + i];
    }
#else
  // Second, precondition the complex SD block by applying Gauss-Seidel to the 2x2 block system defined by real and imaginary parts.
  // For the system [ A_{Re} -A_{Im} ], Gauss-Seidel is y_Re = A_{Re}^{-1} (x_Re + A_{Im} y_Im)
  //                [ A_{Im}  A_{Re} ]                  y_Im = A_{Re}^{-1} (x_Im - A_{Im} y_Re)
  ysdRe = 0.0;
  ysdIm = 0.0;
  for (int iter=0; iter<2; ++iter)
    {
      sdImag->Mult(ysdIm, xsd);
      //xsd = 0.0;  // Jacobi
      for (int i=0; i<nSD; ++i)
	xsd[i] += x[i];  // += Re(x)
      
      invSD->Mult(xsd, ysdRe);

      sdImag->Mult(ysdRe, xsd);
      //xsd = 0.0;  // Jacobi
      for (int i=0; i<nSD; ++i)
	xsd[i] = x[osIm + i] - xsd[i];  // x_Im - A_{Im} y_Re
      
      invSD->Mult(xsd, ysdIm);
    }

  for (int i=0; i<nSD; ++i)
    {
      y[i] = ysdRe[i];
      y[osIm + i] = ysdIm[i];
    }
#endif
}

void E_exact_Re(const Vector &x, Vector &E)
{
  const double s = 1.0 / 8.0;
  /*
  E[0] = x[1] * x[2] * (s - x[1]) * (s - x[2]);
  E[1] = x[0] * x[1] * x[2] * (s - x[0]) * (s - x[2]);
  E[2] = x[0] * x[1] * (s - x[0]) * (s - x[1]);
  */
  
  E[0] = cos((x[0] + x[1] + x[2]) / (s * 1.73205));
  E[1] = 0.0;
  E[2] = 0.0;
}

void TestGMG(HypreParMatrix * A, std::vector<HypreParMatrix*> P, ParFiniteElementSpace *fespace)
{
  GMGSolver *gmg = new GMGSolver(A, P, GMGSolver::CoarseSolver::STRUMPACK);
  gmg->SetTheta(0.5);
  gmg->SetSmootherType(HypreSmoother::Jacobi);  // Some options: Jacobi, l1Jacobi, l1GS, GS

  const int n = A->Height();
  
  Vector x(n);
  Vector y(n);
  y = 0.0;
  
  for (int i=0; i<n; ++i)
    {
      x[i] = i;
    }

  {
    ParGridFunction sol(fespace);
    VectorFunctionCoefficient E_Re(3, E_exact_Re);
    sol.ProjectCoefficient(E_Re);
    A->Mult(sol, x);
  }

  GMRESSolver *gmres = new GMRESSolver(A->GetComm());

  gmres->SetOperator(*(A));

  gmres->SetRelTol(1e-12);
  gmres->SetMaxIter(100);
  gmres->SetPrintLevel(1);

  gmres->SetPreconditioner(*gmg);
  gmres->SetName("TestGMG");
  gmres->iterative_mode = false;

  cout << "TestGMG gmres" << endl;
  
  gmres->Mult(x, y);

  cout << "TestGMG gmres done" << endl;
  
  delete gmres;
  delete gmg;
}

void TestCGMG(HypreParMatrix * Re, HypreParMatrix * Im, std::vector<HypreParMatrix*> P)
{
  (*Im) *= 0.0;
  ComplexHypreParMatrix *AZ = new ComplexHypreParMatrix(Re, Im, false, false);
  //ComplexHypreParMatrix *AZ = new ComplexHypreParMatrix(Re, NULL, false, false);
  ComplexGMGSolver *cgmg = new ComplexGMGSolver(AZ, P, ComplexGMGSolver::CoarseSolver::STRUMPACK);
  cgmg->SetTheta(0.5);
  cgmg->SetSmootherType(HypreSmoother::Jacobi);  // Some options: Jacobi, l1Jacobi, l1GS, GS

  const int n = AZ->Height();
  
  Vector x(n);
  Vector y(n);

  for (int i=0; i<n; ++i)
    {
      x[i] = i;
    }

  GMRESSolver *gmres = new GMRESSolver(Re->GetComm());

  gmres->SetOperator(*(AZ));

  gmres->SetRelTol(1e-12);
  gmres->SetMaxIter(50);
  gmres->SetPrintLevel(1);

  gmres->SetPreconditioner(*cgmg);
  gmres->SetName("TestCGMG");
  gmres->iterative_mode = false;

  cout << "TestCGMG gmres" << endl;
  
  gmres->Mult(x, y);

  cout << "TestCGMG gmres done" << endl;
  
  delete gmres;
  delete cgmg;
  delete AZ;
}

#include "fem/intrules.hpp"

bool GetNormalVector(FiniteElementSpace *fespace, Vector& n)
{
  Vector ni(3);
  
  bool nset = false;
  
  for (int i=0; i < fespace->GetNE(); ++i)
    {
      const FiniteElement &fe = *fespace->GetFE(i);
      ElementTransformation *eltrans = fespace->GetElementTransformation(i);

      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), 2);  // order is arbitrary here
      const IntegrationPoint &ip = ir->IntPoint(0);

      eltrans->SetIntPoint(&ip);
      CalcOrtho(eltrans->Jacobian(), ni);

      ni /= ni.Norml2();  // Normalize
      MFEM_VERIFY(fabs(ni.Norml2() - 1.0) < 1.0e-4, "");

      if (!nset)
	{
	  n = ni;
	  nset = true;
	}
      else
	{
	  ni -= n;
	  MFEM_VERIFY(ni.Norml2() < 1.0e-4, "");
	}
    }

  return nset;
}

bool GetOutwardNormalVector(ParFiniteElementSpace *fespace, InjectionOperator *injop, Vector& n)
{
  // Determine all boundary elements of fespace lying on the interface associated with injop.
  
  const int isize = injop->Width();
  Vector ones(isize);
  Vector injOnesTrue(injop->Height());
  ParGridFunction injOnes(fespace);
  
  if (isize > 0)
    ones = 1.0;

  injop->Mult(ones, injOnesTrue);
  injOnes.SetFromTrueDofs(injOnesTrue);
  auto injOnes_host = injOnes.HostRead();

  ParMesh *pmesh = fespace->GetParMesh();
  
  const int nbe = fespace->GetNBE();

  Vector ni(3);

  bool nset = false;
  
  for (int i=0; i<nbe; ++i)
    {
      Array<int> dofs;
      fespace->GetBdrElementDofs(i, dofs);

      bool onInterface = true;
      
      for (int d=0; d<dofs.Size(); ++d)
	{
	  const int dof_d = (dofs[d] >= 0) ? dofs[d] : -1 - dofs[d];
	  
	  if (fabs(injOnes_host[dof_d]) < 0.1)
	    onInterface = false;
	}

      if (onInterface)
	{
	  const FiniteElement &be = *fespace->GetBE(i);
	  ElementTransformation *eltrans = fespace->GetBdrElementTransformation(i);
	  const IntegrationRule *ir = &IntRules.Get(be.GetGeomType(), 2);  // order is arbitrary here
	  const IntegrationPoint &ip = ir->IntPoint(0);

	  eltrans->SetIntPoint(&ip);
	  CalcOrtho(eltrans->Jacobian(), ni);

	  ni /= ni.Norml2();  // Normalize
	  MFEM_VERIFY(fabs(ni.Norml2() - 1.0) < 1.0e-4, "");

	  // Verify that the normal is outward from the neighboring element.
	  {
	    int el, info;
	    pmesh->GetBdrElementAdjacentElement(i, el, info);
	    Vector center(3);
	    pmesh->GetElementCenter(el, center);

	    Vector bvert(3);
	    Array<int> bv;
	    pmesh->GetBdrElementVertices(i, bv);
	    for (int j=0; j<3; ++j)
	      bvert[j] = pmesh->GetVertex(bv[0])[j];

	    bvert -= center;

	    const double dp = bvert * ni;
	    MFEM_VERIFY(dp > 0.0, "");
	  }
	  
	  if (!nset)
	    {
	      n = ni;
	      nset = true;
	    }
	  else
	    {
	      ni -= n;
	      MFEM_VERIFY(ni.Norml2() < 1.0e-4, "");
	    }
	}
    }

  return nset;
}

DDMInterfaceOperator::DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_,
					   ParFiniteElementSpace *fespaceGlobal, ParMesh **pmeshSD_,
#ifdef SERIAL_INTERFACES
					   Mesh **smeshIF_, std::vector<int> const& interfaceFaceOffset,
#else
					   ParMesh **pmeshIF_,
#endif
					   const int orderND_, const int spaceDim, std::vector<SubdomainInterface> *localInterfaces_,
					   std::vector<int> *interfaceLocalIndex_, const double k2_,
#ifdef GPWD
					   const int Nphi_,
#endif
#ifdef SD_ITERATIVE_GMG
					   std::vector<std::vector<HypreParMatrix*> > const& sdP,
					   std::vector<HypreParMatrix*> *sdcRe, std::vector<HypreParMatrix*> *sdcIm,
#endif
#ifdef SDFOSLS_PA
					   std::vector<Array2D<HypreParMatrix*> > *coarseFOSLS,
#endif
					   const double h_, const bool partialConstructor) :
  numSubdomains(numSubdomains_), numInterfaces(numInterfaces_), orderND(orderND_), pmeshSD(pmeshSD_),
#ifdef SERIAL_INTERFACES
  smeshIF(smeshIF_),
#else
  pmeshIF(pmeshIF_),
#endif
  fec(orderND_, spaceDim),
  fecbdry(orderND_, spaceDim-1), fecbdryH1(orderND_, spaceDim-1), localInterfaces(localInterfaces_), interfaceLocalIndex(interfaceLocalIndex_),
  subdomainLocalInterfaces(numSubdomains_), pmeshGlobal(pmesh_),
  k2(k2_), hmin(h_), realPart(true),
#ifdef GPWD
  Nphi(Nphi_),
#endif
  alpha(1.0), beta(1.0), gamma(1.0)  // TODO: set these to the right values
{
  int num_procs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  m_rank = rank;

  if (m_rank == 0)
    {
      std::cout << "DDM using k2 " << k2 << std::endl;

#ifdef GPWD
      cout << "DDM using GPWD with N " << Nphi << endl;
#endif
    }

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
#ifdef SD_ITERATIVE
  HypreDsdComplex = new HypreParMatrix*[numSubdomains];
#endif
  SpAsdComplexRowSizes = new HYPRE_Int*[numSubdomains];
#ifdef HYPRE_PARALLEL_ASDCOMPLEX    
  AsdRe_HypreBlocks.resize(numSubdomains);
  AsdIm_HypreBlocks.resize(numSubdomains);
  AsdRe_HypreBlockCoef.resize(numSubdomains);
  AsdIm_HypreBlockCoef.resize(numSubdomains);
  sd_com.resize(numSubdomains);
  sd_nonempty.resize(numSubdomains);
  //#ifdef SERIAL_INTERFACES
  AsdRe_SparseBlocks.resize(numSubdomains);
  AsdIm_SparseBlocks.resize(numSubdomains);
  //#endif
#ifdef SD_ITERATIVE
  DsdRe_HypreBlocks.resize(numSubdomains);
  DsdIm_HypreBlocks.resize(numSubdomains);
  DsdRe_HypreBlockCoef.resize(numSubdomains);
  DsdIm_HypreBlockCoef.resize(numSubdomains);  
  DsdRe_SparseBlocks.resize(numSubdomains);
  DsdIm_SparseBlocks.resize(numSubdomains);
  sdImag.resize(numSubdomains);
#endif
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

#ifdef SDFOSLS
  cfosls.resize(numSubdomains);
  for (int m=0; m<numSubdomains; ++m)
    cfosls[m] = NULL;
  injSD2 = new BlockOperator*[numSubdomains];
  injComplexSD2 = new BlockOperator*[numSubdomains];
#ifdef SERIAL_INTERFACES
  bf_ifNDtang = numInterfaces > 0 ? new BilinearForm*[numInterfaces] : NULL;
  ifNDtangSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;
#endif
#endif

  fespace = new ParFiniteElementSpace*[numSubdomains];
  ASPsd = new Operator*[numSubdomains];
  invAsd = new Operator*[numSubdomains];
  injSD = new BlockOperator*[numSubdomains];
  precAsd = new Solver*[numSubdomains];
  sdND = new HypreParMatrix*[numSubdomains];
  //sdNDcopy = new HypreParMatrix*[numSubdomains];
  A_SS = new HypreParMatrix*[numSubdomains];
  sdNDinv = new Operator*[numSubdomains];
  sdNDPen = new HypreParMatrix*[numSubdomains];
  sdNDPlusPen = new HypreParMatrix*[numSubdomains];
  sdNDPenSp = new SparseMatrix*[numSubdomains];
  bf_sdND = new ParBilinearForm*[numSubdomains];
  sd_ess_tdof_list.resize(numSubdomains);
#ifdef TEST_DECOUPLED_FOSLS
  sd_all_ess_tdof_list.resize(numSubdomains);
#endif
#ifdef SD_ITERATIVE_GMG_PA
  bfpa_sdND = new ParBilinearForm*[numSubdomains];
  oppa_sdND = new OperatorPtr[numSubdomains];
  AsdPARe = new BlockOperator*[numSubdomains];
  AsdPAIm = new BlockOperator*[numSubdomains];

  oppa_ifNDmass = new OperatorPtr[numInterfaces];
  oppa_ifNDcurlcurl = new OperatorPtr[numInterfaces];
  oppa_ifH1mass = new OperatorPtr[numInterfaces];
#endif
  ySD = new Vector*[numSubdomains];
  rhsSD = new Vector*[numSubdomains];
  srcSD = new Vector*[numSubdomains];

  if_ND_ess_tdof_list.resize(numInterfaces);
  if_H1_ess_tdof_list.resize(numInterfaces);
  
#ifdef SERIAL_INTERFACES
  ifespace = numInterfaces > 0 ? new FiniteElementSpace*[numInterfaces] : NULL;
  iH1fespace = numInterfaces > 0 ? new FiniteElementSpace*[numInterfaces] : NULL;

  ifNDmassSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;
  ifH1massSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;
  ifNDcurlcurlSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;
  ifNDH1gradSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;
  ifNDH1gradTSp = numInterfaces > 0 ? new SparseMatrix*[numInterfaces] : NULL;

  SIsender.resize(numInterfaces);
  SIreceiver.resize(numInterfaces);

  sd_root.assign(numSubdomains, -1);

  bf_ifNDmass = numInterfaces > 0 ? new BilinearForm*[numInterfaces] : NULL;
  bf_ifNDcurlcurl = numInterfaces > 0 ? new BilinearForm*[numInterfaces] : NULL;
  bf_ifH1mass = numInterfaces > 0 ? new BilinearForm*[numInterfaces] : NULL;
  bf_ifNDH1grad = numInterfaces > 0 ? new MixedBilinearForm*[numInterfaces] : NULL;
#else
  ifespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;
  iH1fespace = numInterfaces > 0 ? new ParFiniteElementSpace*[numInterfaces] : NULL;

  ifNDcurlcurl = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifND_FS = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  
  ifNDmass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifH1mass = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifNDH1grad = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
  ifNDH1gradT = numInterfaces > 0 ? new HypreParMatrix*[numInterfaces] : NULL;
#endif

  ifNDmassInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
  ifH1massInv = numInterfaces > 0 ? new Operator*[numInterfaces] : NULL;
  
  numLocalInterfaces = localInterfaces->size();
  globalInterfaceIndex.resize(numLocalInterfaces);
  globalInterfaceIndex.assign(numLocalInterfaces, -1);

  // QUESTION: is numLocalInterfaces == numInterfaces in general?
  //MFEM_VERIFY(numLocalInterfaces == numInterfaces, "");

  ifNDtrue.assign(numInterfaces, 0);
  ifH1true.assign(numInterfaces, 0);

  std::vector<int> interfaceSDs; // interfaceSDs[i] = (numSubdomains*sd0) + sd1
  interfaceSDs.assign(numInterfaces, 0);

#ifdef GPWD
  ifmrOffsetsComplex.resize(numInterfaces);
  block_MRoffsets.SetSize((2*numInterfaces) + 1);  // number of blocks + 1
  block_MRoffsets = 0;
#endif

#ifdef SERIAL_INTERFACES
#ifdef USE_2DINTERFACE
  smeshIF2D.resize(numInterfaces);
#endif
#endif
  
  for (int i=0; i<numInterfaces; ++i)
    {
#ifdef SERIAL_INTERFACES      
      if (smeshIF[i] == NULL)
#else
      if (pmeshIF[i] == NULL)
#endif
	{
	  ifespace[i] = NULL;
	  iH1fespace[i] = NULL;

#ifdef SERIAL_INTERFACES
#ifdef USE_2DINTERFACE
	  smeshIF2D[i] = NULL;
#endif

	  ifNDmassSp[i] = NULL;
	  ifNDH1gradSp[i] = NULL;
	  ifNDH1gradTSp[i] = NULL;
	  ifH1massSp[i] = NULL;
	  ifNDcurlcurlSp[i] = NULL;
#else
	  ifNDmass[i] = NULL;
	  ifNDH1grad[i] = NULL;
	  ifNDH1gradT[i] = NULL;
	  ifH1mass[i] = NULL;
	  ifNDcurlcurl[i] = NULL;
	  ifND_FS[i] = NULL;
#endif

#ifdef SDFOSLS
	  ifNDtangSp[i] = NULL;	  
#endif
	}
      else
	{
#ifdef SERIAL_INTERFACES
#ifdef USE_2DINTERFACE
	  smeshIF2D[i] = Create2DMeshFrom3DSurfaceMesh(smeshIF[i]);
	  
	  ifespace[i] = new FiniteElementSpace(smeshIF2D[i], &fecbdry);  // Nedelec space for f_{m,j} when interface i is the j-th interface of subdomain m. 
	  iH1fespace[i] = new FiniteElementSpace(smeshIF2D[i], &fecbdryH1);  // H^1 space \rho_{m,j} when interface i is the j-th interface of subdomain m.
#else
	  ifespace[i] = new FiniteElementSpace(smeshIF[i], &fecbdry);  // Nedelec space for f_{m,j} when interface i is the j-th interface of subdomain m. 
	  iH1fespace[i] = new FiniteElementSpace(smeshIF[i], &fecbdryH1);  // H^1 space \rho_{m,j} when interface i is the j-th interface of subdomain m.
#endif
	  
	  ifNDtrue[i] = ifespace[i]->GetVSize();
	  ifH1true[i] = iH1fespace[i]->GetVSize();
#else
	  ifespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdry);  // Nedelec space for f_{m,j} when interface i is the j-th interface of subdomain m. 
	  iH1fespace[i] = new ParFiniteElementSpace(pmeshIF[i], &fecbdryH1);  // H^1 space \rho_{m,j} when interface i is the j-th interface of subdomain m.
	  
	  ifNDtrue[i] = ifespace[i]->GetTrueVSize();
	  ifH1true[i] = iH1fespace[i]->GetTrueVSize();
#endif

	  /*
	  cout << rank << ": Interface " << i << " number of bdry attributes: " << pmeshIF[i]->bdr_attributes.Size()
	       << ", NE " << pmeshIF[i]->GetNE() << ", NBE " << pmeshIF[i]->GetNBE() << endl;
	  */
	  /*
	    for (int j=0; j<pmeshIF[i]->GetNBE(); ++j)
	    cout << rank << " Interface " << i << ", be " << j << ", bdrAttribute " << pmeshIF[i]->GetBdrAttribute(j) << endl;
	  */

	  CreateInterfaceMatrices(i, partialConstructor);
	}

#ifdef GPWD
      ifmrOffsetsComplex[i].SetSize(3);

      ifmrOffsetsComplex[i] = 0;

      ifmrOffsetsComplex[i][1] = (2*ifNDtrue[i]) + ifH1true[i];
      ifmrOffsetsComplex[i][2] = (2*ifNDtrue[i]) + ifH1true[i];

      ifmrOffsetsComplex[i].PartialSum();
      
      block_MRoffsets[(2*i)+1] = ifmrOffsetsComplex[i][2];
      block_MRoffsets[(2*i)+2] = ifmrOffsetsComplex[i][2];
#endif

      const int ifli = (*interfaceLocalIndex)[i];

      //MFEM_VERIFY((ifli >= 0) == (pmeshIF[i] != NULL), "");
	
      if (ifli >= 0)
	{
	  subdomainLocalInterfaces[(*localInterfaces)[ifli].FirstSubdomain()].push_back(i);
	  subdomainLocalInterfaces[(*localInterfaces)[ifli].SecondSubdomain()].push_back(i);

	  MFEM_VERIFY(globalInterfaceIndex[ifli] == i || globalInterfaceIndex[ifli] == -1, "");

	  globalInterfaceIndex[ifli] = i;

	  interfaceSDs[i] = (numSubdomains * (*localInterfaces)[ifli].FirstSubdomain()) + (*localInterfaces)[ifli].SecondSubdomain();
	}
    }

  {
    std::vector<int> localInterfaceSDs;
    localInterfaceSDs = interfaceSDs;
    MPI_Allreduce(localInterfaceSDs.data(), interfaceSDs.data(), numInterfaces, MPI_INT, MPI_MAX, MPI_COMM_WORLD);  // TODO: can this be done in place?
  }
    
  // For each subdomain parallel finite element space, determine all the true DOF's on the entire boundary. Also for each interface parallel finite element space, determine the number of true DOF's. Note that a true DOF on the boundary of a subdomain may coincide with an interface DOF that is not necessarily a true DOF on the corresponding interface mesh. The size of DDMInterfaceOperator will be the sum of the numbers of true DOF's on the subdomain mesh boundaries and interfaces.

  block_trueOffsets.SetSize(numSubdomains + 1); // number of blocks + 1
  block_trueOffsets = 0;

#ifdef SDFOSLS
  block_trueOffsets_FOSLS.SetSize(numSubdomains + 1); // number of blocks + 1
  block_trueOffsets_FOSLS = 0;
#endif
  
  int size = 0;

  InterfaceToSurfaceInjection.resize(numSubdomains);
  InterfaceToSurfaceInjectionData.resize(numSubdomains);
  //InterfaceToSurfaceInjectionFullData.resize(numSubdomains);
  InterfaceToSurfaceInjectionGlobalData.resize(numSubdomains);
  GlobalInterfaceToSurfaceInjectionGlobalData.resize(numSubdomains);
  allGlobalSubdomainInterfaces.resize(numSubdomains);
  gflip.resize(numSubdomains);

  /*
  std::vector<int> sdnp, gsdnp;
  sdnp.assign(numSubdomains, 0);
  gsdnp.assign(numSubdomains, 0);
  */

  StopWatch chronoSD;
  chronoSD.Clear();
  chronoSD.Start();

  for (int m=0; m<numSubdomains; ++m)  // TODO: make this scalable
    {
      MPI_Barrier(MPI_COMM_WORLD);
      GatherSetInt(MPI_COMM_WORLD, subdomainLocalInterfaces[m], allGlobalSubdomainInterfaces[m]);
    }
  
  for (int m=0; m<numSubdomains; ++m)
    {
      //InterfaceToSurfaceInjection[m].resize(subdomainLocalInterfaces[m].size());
      InterfaceToSurfaceInjection[m].resize(numInterfaces);
      InterfaceToSurfaceInjectionData[m].resize(subdomainLocalInterfaces[m].size());
      //InterfaceToSurfaceInjectionFullData[m].resize(subdomainLocalInterfaces[m].size());
      InterfaceToSurfaceInjectionGlobalData[m].resize(subdomainLocalInterfaces[m].size());
      GlobalInterfaceToSurfaceInjectionGlobalData[m].resize(numInterfaces);
      gflip[m].resize(numInterfaces);

      for (int i=0; i<numInterfaces; ++i)
	{
	  InterfaceToSurfaceInjection[m][i] = NULL;
	}
      
      //cout << rank << ": setup for subdomain " << m << endl;
	  
      if (pmeshSD[m] == NULL)
	{
	  fespace[m] = NULL;
	}
      else
	{
	  fespace[m] = new ParFiniteElementSpace(pmeshSD[m], &fec);  // Nedelec space for u_m

	  //sdnp[m] = 1;
	  
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

	  //if (m == 0)
	  //cout << rank << ": sd " << m << " ND space true size " << fespace[m]->GetTrueVSize() << ", full size " << fespace[m]->GetVSize() << endl;

	}

      {
	// Define a minimal communicator for each subdomain and its interfaces.
	int locSizeSD = (fespace[m] == NULL) ? 0 : fespace[m]->GetTrueVSize();
	for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	  {
	    const int interfaceIndex = subdomainLocalInterfaces[m][i];
	    locSizeSD += ifNDtrue[interfaceIndex];
	  }

#ifdef NO_COMM_SPLITTING
	sd_com[m] = MPI_COMM_WORLD;
#else
#ifdef SERIAL_INTERFACES
	sd_com[m] = (fespace[m] == NULL) ? MPI_COMM_NULL : fespace[m]->GetComm();
	sd_nonempty[m] = (fespace[m] != NULL);

	if (sd_nonempty[m])
	  {
	    sd_root[m] = m_rank;
	    MPI_Bcast(&(sd_root[m]), 1, MPI_INT, 0, sd_com[m]);
	  }
#else
	sd_nonempty[m] = (locSizeSD > 0);
	int color = (locSizeSD == 0);
	const int status = MPI_Comm_split(MPI_COMM_WORLD, color, m_rank, &(sd_com[m]));
	MFEM_VERIFY(status == MPI_SUCCESS, "Construction of comm failed");
#endif
#endif
      }

#ifdef SERIAL_INTERFACES
      if (sd_nonempty[m])
	{
#endif
      
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

	  if (m == -10)
	    {
	      cout << "SD 0 has interface " << interfaceIndex << " with ND, H1: " << ifNDtrue[interfaceIndex] << ", " << ifH1true[interfaceIndex] << endl;
	    }

#ifdef SDFOSLS
	  block_trueOffsets_FOSLS[m+1] += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
#endif

	  /*
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
	  */
	  
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
#ifdef SERIAL_INTERFACES
	}
#endif

      /*
      // TODO: make this scalable
      MPI_Barrier(MPI_COMM_WORLD);
      
      GatherSetInt(MPI_COMM_WORLD, subdomainLocalInterfaces[m], allGlobalSubdomainInterfaces[m]);
      */
      
      // Loop over all global interfaces touching this subdomain in a consistent order for all processes.
      int lastIF = -1;
      for (auto interfaceIndex : allGlobalSubdomainInterfaces[m])
	{
	  MFEM_VERIFY(interfaceIndex > lastIF, "");  // Verify ascending order of global indices.
	  lastIF = interfaceIndex;

	  const int ifli = (*interfaceLocalIndex)[interfaceIndex];
	  //MFEM_VERIFY(ifli >= 0, "");

	  int i = -1;
	  for (int j=0; j<subdomainLocalInterfaces[m].size(); ++j)
	    {
	      if (subdomainLocalInterfaces[m][j] == interfaceIndex)
		{
		  MFEM_VERIFY(i == -1, "");
		  i = j;
		}
	    }
	  
	  //MFEM_VERIFY(i >= 0, "");

	  if (sd_nonempty[m])
	    {
	      std::set<int> facesEmpty, edgesEmpty, verticesEmpty;
	      
	      const std::set<int> &ifFaces = (ifli >= 0) ? (*localInterfaces)[ifli].facesSet : facesEmpty;
	      const std::set<int> &ifEdges = (ifli >= 0) ? (*localInterfaces)[ifli].edges : edgesEmpty;
	      const std::set<int> &ifVertices = (ifli >= 0) ? (*localInterfaces)[ifli].vertices : verticesEmpty;

	      std::vector<int> dofmapEmpty;

	      std::vector<int> &dofmap = (i >= 0) ? InterfaceToSurfaceInjectionData[m][i] : dofmapEmpty;
	      //std::vector<int> &gdofmap = (i >= 0) ? InterfaceToSurfaceInjectionGlobalData[m][i] : gdofmapEmpty;
	      
	      /*
	      SetInterfaceToSurfaceDOFMap(sd_com[m], ifespace[interfaceIndex], fespace[m], fespaceGlobal, pmeshGlobal, m+1, (*localInterfaces)[ifli].faces, 
					  (*localInterfaces)[ifli].edges, (*localInterfaces)[ifli].vertices, &fecbdry,
					  InterfaceToSurfaceInjectionData[m][i], //InterfaceToSurfaceInjectionFullData[m][i],
					  InterfaceToSurfaceInjectionGlobalData[m][i]);
	      */
	      
#ifdef SERIAL_INTERFACES
	      std::vector<int> ifFacesOrderedEmpty;
	      const std::vector<int> &ifFacesOrdered = (ifli >= 0) ? (*localInterfaces)[ifli].faces : ifFacesOrderedEmpty;

	      MFEM_VERIFY(ifFacesOrdered.size() == ifFaces.size(), "");
#endif

	      const int ifos = (ifli >= 0) ? interfaceFaceOffset[ifli] : 0;

	      std::set<int> flippedSDDofs;

	      SetInterfaceToSurfaceDOFMap(sd_com[m], ifespace[interfaceIndex], 
#ifdef SERIAL_INTERFACES
					  ifos, ifFacesOrdered,
#ifdef USE_IFMESH3D_VERT
					  smeshIF[interfaceIndex],
#endif
#endif
					  fespace[m], fespaceGlobal, pmeshGlobal, m+1,
					  ifFaces, ifEdges, ifVertices, &fecbdry, dofmap,
					  GlobalInterfaceToSurfaceInjectionGlobalData[m][interfaceIndex], flippedSDDofs, gflip[m][interfaceIndex]);

	      if (i >= 0)
		InterfaceToSurfaceInjectionGlobalData[m][i] = GlobalInterfaceToSurfaceInjectionGlobalData[m][interfaceIndex];  // TODO: don't keep 2 copies.
	      
	      // InterfaceToSurfaceInjection[m][i] = new InjectionOperator(MPI_COMM_WORLD, fespace[m], ifespace[interfaceIndex],
	      /*
	      InterfaceToSurfaceInjection[m][i] = new InjectionOperator(sd_com[m], fespace[m], ifespace[interfaceIndex],
									&(InterfaceToSurfaceInjectionData[m][i][0]),
									InterfaceToSurfaceInjectionGlobalData[m][i]);
	      */
	      
	      InjectionOperator *injOp = new InjectionOperator(sd_com[m], fespace[m], ifespace[interfaceIndex],
							       (i >= 0) ? &(InterfaceToSurfaceInjectionData[m][i][0]) : NULL,
							       GlobalInterfaceToSurfaceInjectionGlobalData[m][interfaceIndex], gflip[m][interfaceIndex]);

#ifdef DOFMAP_DEBUG
	      //#ifndef SERIAL_INTERFACES
	      { // debugging
		// Test orientations of the same DOF's in fespace[m] and ifespace[interfaceIndex] by comparing projections of the same analytic function.
		VectorFunctionCoefficient E(3, test2_E_exact);

		const int ifsize = (ifespace[interfaceIndex] != NULL) ? ifespace[interfaceIndex]->GetTrueVSize() : 0;
		  
		Vector injIF(ifsize);
		Vector vif(ifsize);

		if (ifespace[interfaceIndex] != NULL)
		  {
#ifdef SERIAL_INTERFACES
		    GridFunction gfif(ifespace[interfaceIndex]);
		    Vector vifg(ifsize);
		    gfif.ProjectCoefficient(E);
		    gfif.GetTrueDofs(vifg);  // vifg will go out of scope with gfif
		    vif = vifg;
#else
		    ParGridFunction gfif(ifespace[interfaceIndex]);
		    gfif.ProjectCoefficient(E);
		    gfif.GetTrueDofs(vif);
#endif
		  }
		  
		Vector vsd((fespace[m] != NULL) ? fespace[m]->GetTrueVSize() : 0);
		if (fespace[m] != NULL)
		  {
		    ParGridFunction gfsd(fespace[m]);
		    gfsd.ProjectCoefficient(E);
		    gfsd.GetTrueDofs(vsd); // After this, vsd uses device
		  }

		double *vsd_host = (fespace[m] != NULL) ? vsd.HostReadWrite() : nullptr;

		const bool testHex = false;
		if (testHex && fespace[m] != NULL)
		  {
		    std::vector<int> dofori;
		    dofori.assign(fespace[m]->GetTrueVSize(), 0);
		    std::set<int> sdtdof;
		    injOp->GetAllTrueSD(sdtdof);
		    
		    Array<int> edofs;
		    
		    for (int el=0; el<pmeshSD[m]->GetNE(); ++el)
		      {
			fespace[m]->GetElementDofs(el, edofs);
			for (int j=0; j<edofs.Size(); ++j)
			  {
			    const int dofj = (edofs[j] >= 0) ? edofs[j] : -1 - edofs[j];
			    const int ltdof = fespace[m]->GetLocalTDofNumber(dofj);

			    std::set<int>::const_iterator it = sdtdof.find(ltdof);
			    if (it != sdtdof.end())
			      {
				const int ori = (edofs[j] >= 0) ? 1 : -1;
				if (!(ori == dofori[ltdof] || dofori[ltdof] == 0))
				  cout << "BUG" << endl;
			    
				MFEM_VERIFY(ori == dofori[ltdof] || dofori[ltdof] == 0, "");
				dofori[ltdof] = ori;
			      }
			  }
		      }
		    
		    for (int j=0; j<fespace[m]->GetTrueVSize(); ++j)
		      {
			if (dofori[j] != 0)
			  vsd_host[j] *= dofori[j];
		      }
		  }

		const bool testHex2 = false;
		if (testHex2 && fespace[m] != NULL)
		  {
		    for (int j=0; j<fespace[m]->GetTrueVSize(); ++j)
		      {
			std::set<int>::const_iterator it = flippedSDDofs.find(j);
			std::set<int>::const_iterator itm = flippedSDDofs.find(-1 - j);

			MFEM_VERIFY(it == flippedSDDofs.end() || itm == flippedSDDofs.end(), "");
			//MFEM_VERIFY(it != flippedSDDofs.end() || itm != flippedSDDofs.end(), "");
			
			if (itm != flippedSDDofs.end())
			  {
			    vsd_host[j] *= -1.0;
			  }
		      }
		  }
		
		injOp->MultTransposeRaw(vsd_host, injIF);

		auto vif_host = vif.HostRead();
		for (int j=0; j<ifsize; ++j)
		  {
		    if (fabs(injIF[j] - vif_host[j]) > 1.0e-8)
		      cout << m_rank << ": DISCREPANCY A sd " << m << ", entry " << j << ", inj " << injIF[j] << ", if " << vif[j] << endl;
		  }
	      }
	      //#endif
#endif
	      
	      /*
	      if (i >= 0)
		InterfaceToSurfaceInjection[m][i] = injOp;
	      */
	      InterfaceToSurfaceInjection[m][interfaceIndex] = injOp;
	    }
	  /*
	  else
	    {
	      if (i >= 0)
		InterfaceToSurfaceInjection[m][i] = NULL;
	    }
	  */
	}
      
      //MPI_Barrier(MPI_COMM_WORLD);
    }

  tdofsBdry.resize(numSubdomains);
  trueOffsetsSD.resize(numSubdomains);

#ifdef SD_ITERATIVE
  trueOffsetsAuxSD.resize(numSubdomains);
#endif
  
  tdofsBdryInjection = new SetInjectionOperator*[numSubdomains];
  tdofsBdryInjectionTranspose = new Operator*[numSubdomains];

  MPI_Barrier(MPI_COMM_WORLD);

  chronoSD.Stop();
  if (m_rank == 0)
    cout << m_rank << ": DDM constructor SD loop 1 timing " << chronoSD.RealTime() << endl;
  
  chronoSD.Clear();
  chronoSD.Start();

  for (int m=0; m<numSubdomains; ++m)
    {
      A_SS[m] = NULL;
      sdNDPen[m] = NULL;
      sdNDPlusPen[m] = NULL;
	
      if (pmeshSD[m] == NULL)
	{
	  sdND[m] = NULL;
	  //sdNDcopy[m] = NULL;
	    
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

#ifdef SDFOSLS
	  block_trueOffsets_FOSLS[m+1] += 2 * tdofsBdry[m].size();
#endif

	  tdofsBdryInjection[m] = new SetInjectionOperator(fespace[m]->GetTrueVSize(), &(tdofsBdry[m]));
	  tdofsBdryInjectionTranspose[m] = new TransposeOperator(tdofsBdryInjection[m]);
	  
	  //#ifndef SDFOSLS  // TODO: in SDFOSLS case, do not call CreateSubdomainMatrices, but do set attributes for pmeshSD as done in this function.
	  CreateSubdomainMatrices(m, partialConstructor);
	  //#endif
	  
	  /*
	    { // debugging
	    hypre_CSRMatrix* csr = GetHypreParMatrixData(*(sdND[m]));
	    cout << "csr nr " << csr->num_rows << endl;
	    }
	  */
	}
	
      SetOffsetsSD(m);
#ifdef SD_ITERATIVE
      SetOffsetsAuxSD(m);
#endif
      
      //GMRESSolver *gmres = new GMRESSolver(fespace[m]->GetComm());  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?
      GMRESSolver *gmres = new GMRESSolver(MPI_COMM_WORLD);  // TODO: this communicator is not necessarily the same as the pmeshIF communicators. Does GMRES actually use the communicator?

#ifdef DDMCOMPLEX
      // Real part
      SetToRealParameters();

#ifndef NO_SD_OPERATOR
      AsdRe[m] = CreateSubdomainOperator(m);
#endif

#ifdef SD_ITERATIVE_GMG_PA
      if (!partialConstructor)
	{
	  AsdPARe[m] = CreateSubdomainOperatorPA(m);
	}
#endif
      
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

#ifdef SD_ITERATIVE_GMG_PA
      if (partialConstructor)
#endif
      {
#ifndef SDFOSLS
	CreateSubdomainHypreBlocks(m, AsdRe_HypreBlocks[m],
				   //#ifdef SERIAL_INTERFACES
				   AsdRe_SparseBlocks[m],
				   //#endif				 
				   AsdRe_HypreBlockCoef[m]);
#endif
      }
      
#ifdef SD_ITERATIVE      
#ifdef SD_ITERATIVE_FULL
      CreateSubdomainDiagHypreBlocks(m, DsdRe_HypreBlocks[m], DsdRe_SparseBlocks[m], DsdRe_HypreBlockCoef[m]);
#else
      CreateSubdomainAuxiliaryHypreBlocks(m, DsdRe_HypreBlocks[m], DsdRe_SparseBlocks[m], DsdRe_HypreBlockCoef[m]);
#endif
#endif
#endif
	
      // Imaginary part
      SetToImaginaryParameters();
      
#ifndef NO_SD_OPERATOR
      AsdIm[m] = CreateSubdomainOperator(m);
#endif
#ifdef SD_ITERATIVE_GMG_PA
      if (!partialConstructor)
	{
	  AsdPAIm[m] = CreateSubdomainOperatorPA(m);
	}
#endif
      
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
#ifdef SD_ITERATIVE_GMG_PA
      if (partialConstructor)
#endif
	{
#ifndef SDFOSLS
	  CreateSubdomainHypreBlocks(m, AsdIm_HypreBlocks[m],
				     //#ifdef SERIAL_INTERFACES
				     AsdIm_SparseBlocks[m],
				     //#endif				 
				     AsdIm_HypreBlockCoef[m]);
#endif
	}
      
#ifdef SD_ITERATIVE      
#ifdef SD_ITERATIVE_FULL
      CreateSubdomainDiagHypreBlocks(m, DsdIm_HypreBlocks[m], DsdIm_SparseBlocks[m], DsdIm_HypreBlockCoef[m]);
#else
      CreateSubdomainAuxiliaryHypreBlocks(m, DsdIm_HypreBlocks[m], DsdIm_SparseBlocks[m], DsdIm_HypreBlockCoef[m]);      
#endif
#endif
#endif
   
      AsdP[m] = NULL;
      /*
      // Set real-valued subdomain operator for preconditioning purposes only.
      SetToPreconditionerParameters();

      preconditionerMode = true;  // TODO!
      AsdP[m] = CreateSubdomainOperator(m);
      preconditionerMode = false;

      SetParameters();
      SetToImaginaryParameters();
      */
      
      //gmres->SetOperator(*(AsdP[m]));  // TODO: maybe try an SPD version of Asd with a CG solver instead of GMRES.
#else
#ifndef NO_SD_OPERATOR
      Asd[m] = CreateSubdomainOperator(m);
#endif
      //gmres->SetOperator(*(Asd[m]));
#endif
	
      //ASPsd[m] = CreateSubdomainOperatorStrumpack(m);

      //precAsd[m] = CreateSubdomainPreconditionerStrumpack(m);

      gmres->SetRelTol(1e-12);
      gmres->SetMaxIter(100);  // 3333
      gmres->SetPrintLevel(0);

      //gmres->SetPreconditioner(*(precAsd[m]));
      gmres->SetName("invAsd" + std::to_string(m));
      gmres->iterative_mode = false;
	
      invAsd[m] = gmres;

      //MPI_Barrier(MPI_COMM_WORLD);
    }
    
  block_trueOffsets.PartialSum();
  MFEM_VERIFY(block_trueOffsets.Last() == size, "");

#ifdef SDFOSLS
  block_trueOffsets_FOSLS.PartialSum();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  
  chronoSD.Stop();
  if (m_rank == 0)
    cout << m_rank << ": DDM constructor SD loop 2 timing " << chronoSD.RealTime() << endl;

#ifndef SDFOSLS_PA
  if (partialConstructor)
    return;
#endif
  
  /*
  MPI_Allreduce(sdnp.data(), gsdnp.data(), numSubdomains, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (m_rank == 0)
    {
      for (int i=0; i<numSubdomains; ++i)
	{
	  cout << "gsdnp[" << i << "] " << gsdnp[i] << endl;
	}
    }
  */
  
#ifdef DDMCOMPLEX
#ifdef SDFOSLS
  height = 2*size;  // TODO
  width = 2*size;
#else
  height = 2*size;
  width = 2*size;
#endif

#ifdef SDFOSLS
  globalInterfaceOpRe = new BlockOperator(block_trueOffsets_FOSLS, block_trueOffsets);
  globalInterfaceOpIm = new BlockOperator(block_trueOffsets_FOSLS, block_trueOffsets);
#else
  globalInterfaceOpRe = new BlockOperator(block_trueOffsets);
  globalInterfaceOpIm = new BlockOperator(block_trueOffsets);
#endif
  
  block_trueOffsets2.SetSize(numSubdomains + 1);
  for (int i=0; i<numSubdomains + 1; ++i)
    block_trueOffsets2[i] = 2*block_trueOffsets[i];

  /*
#ifdef SDFOSLS
  block_trueOffsets22.SetSize(numSubdomains + 1);
  for (int i=0; i<numSubdomains + 1; ++i)
    block_trueOffsets22[i] = 4*block_trueOffsets[i];
#endif
  */
  
#ifdef SDFOSLS
  block_trueOffsets_FOSLS2.SetSize(numSubdomains + 1);
  for (int i=0; i<numSubdomains + 1; ++i)
    block_trueOffsets_FOSLS2[i] = 2*block_trueOffsets_FOSLS[i];
  
  globalInterfaceOp = new BlockOperator(block_trueOffsets_FOSLS2, block_trueOffsets2);

  rowTrueOffsetsComplexIF_FOSLS.resize(numInterfaces);
  colTrueOffsetsComplexIF_FOSLS.resize(numInterfaces);
#else
  globalInterfaceOp = new BlockOperator(block_trueOffsets2);
  if (m_rank == -10)
    {
      cout << "block_trueOffsets2 print, tdofsBdry[0].size() " << tdofsBdry[0].size() << endl;
      block_trueOffsets2.Print();
    }
#endif

  rowTrueOffsetsComplexIF.resize(numInterfaces);
  colTrueOffsetsComplexIF.resize(numInterfaces);
  /*
  rowTrueOffsetsComplexIF.resize(numLocalInterfaces);
  colTrueOffsetsComplexIF.resize(numLocalInterfaces);
  */
#else
  height = size;
  width = size;

  globalInterfaceOp = new BlockOperator(block_trueOffsets);
#endif
    
  rowTrueOffsetsIF.resize(numLocalInterfaces);
  colTrueOffsetsIF.resize(numLocalInterfaces);
  
  rowTrueOffsetsIFR.resize(numInterfaces);
  colTrueOffsetsIFR.resize(numInterfaces);

  rowTrueOffsetsIFL.resize(numInterfaces);
  colTrueOffsetsIFL.resize(numInterfaces);
    
  rowTrueOffsetsIFBR.resize(numInterfaces);
  colTrueOffsetsIFBR.resize(numInterfaces);

  rowTrueOffsetsIFBL.resize(numInterfaces);
  colTrueOffsetsIFBL.resize(numInterfaces);

#ifdef SDFOSLS
  rowTrueOffsetsIFL2.resize(numInterfaces);
  colTrueOffsetsIFL2.resize(numInterfaces);  
#endif

#ifdef GPWD
  block_MRoffsets.PartialSum();
  
  osM1.resize(numInterfaces);
  M1.resize(numInterfaces);
  eTE_Re.resize(numInterfaces);
  eTE_Im.resize(numInterfaces);
  eTE_M.resize(numInterfaces);
  bcr_Re.resize(numInterfaces);
  bcr_Im.resize(numInterfaces);
  bcr_M.resize(numInterfaces);
  rhoc_Re.resize(numInterfaces);
  rhoc_Im.resize(numInterfaces);
  rhoc_M.resize(numInterfaces);
  ifbx.resize(numInterfaces);
  ifby.resize(numInterfaces);
  iforig.resize(numInterfaces);
 
  cgp_u.resize(numInterfaces);
  cgp_f.resize(numInterfaces);
  cgp_rho.resize(numInterfaces);
  
  for (int i=0; i<numInterfaces; ++i)
    {
      M1[i] = NULL;
      cgp_u[i] = NULL;
      cgp_f[i] = NULL;
      cgp_rho[i] = NULL;
    }

  auxMR.SetSize(block_MRoffsets.Last());

  globalMR = new BlockOperator(block_MRoffsets, block_trueOffsets2);

  gpwd_size = 12*numInterfaces*Nphi;  // 2 subdomains per interface, 2 complex parts, 3 fields (u, f, rho).

  gpwd_v.SetSize(height);
  gpwd_w.SetSize(height);
  
  gpwd_u.SetSize(gpwd_size);  
  if (m_rank == 0)
    {
      ifroot.resize(numInterfaces);
      gpwd_z.SetSize(gpwd_size);  
      gpwd_cmat.SetSize(gpwd_size);
    }
#endif

  prec_z.SetSize(height);  
  prec_w.SetSize(height);  

#ifdef EQUATE_REDUNDANT_VARS
  rowTrueOffsetsIFRR.resize(numLocalInterfaces);
  colTrueOffsetsIFRR.resize(numLocalInterfaces);
#endif
  
#ifdef SDFOSLS
  outwardFromSD.resize(numInterfaces);
#endif
  
  chronoSD.Clear();
  chronoSD.Start();
  
  //for (auto interfaceIndex : allGlobalSubdomainInterfaces[m])
  //for (int ili=0; ili<numLocalInterfaces; ++ili)
  for (int i=0; i<numInterfaces; ++i)
    {
      const int ili = (*interfaceLocalIndex)[i];

      const int sd0 = interfaceSDs[i] / numSubdomains;
      const int sd1 = interfaceSDs[i] - (sd0 * numSubdomains);

      MFEM_VERIFY(0 <= sd0 && 0 <= sd1 && sd0 < numSubdomains && sd1 < numSubdomains, "");
      
      if (ili >= 0)
	{
	  MFEM_VERIFY(globalInterfaceIndex[ili] == i, "");
	  MFEM_VERIFY(sd0 == (*localInterfaces)[ili].FirstSubdomain() && sd1 == (*localInterfaces)[ili].SecondSubdomain(), "");
	}
      
      //const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
      //const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

      //cout << rank << ": Interface " << ili << " sd " << sd0 << ", " << sd1 << endl;
	
      MFEM_VERIFY(sd0 < sd1, "");
      
#ifdef SERIAL_INTERFACES
      int sdata[2] = {-1, -1};
      
      if (ili >= 0)
	{
	  if (!(*localInterfaces)[ili].IsEmpty())
	    {
	      MFEM_VERIFY((*localInterfaces)[ili].GetRank() == m_rank, "");
	      MFEM_VERIFY((*localInterfaces)[ili].GetOwningRank() == m_rank || (*localInterfaces)[ili].GetSharingRank() == m_rank, "");

	      int otherSubdomainRoot = -1;
		  
	      if ((sd_nonempty[sd0] || sd_nonempty[sd1]) && !(sd_nonempty[sd0] && sd_nonempty[sd1]))
		{
		  const int otherRank = ((*localInterfaces)[ili].GetOwningRank() == m_rank) ?
		    (*localInterfaces)[ili].GetSharingRank() : (*localInterfaces)[ili].GetOwningRank();

		  // Exchange subdomain root ranks.
		  
		  const int mysd = sd_nonempty[sd0] ? sd0 : sd1;
		  MFEM_VERIFY(sd_root[mysd] >= 0, "");

		  if (mysd == sd0)
		    MPI_Send(&(sd_root[mysd]), 1, MPI_INT, otherRank, i, MPI_COMM_WORLD);
		  else
		    MPI_Recv(&otherSubdomainRoot, 1, MPI_INT, otherRank, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  if (mysd == sd1)
		    MPI_Send(&(sd_root[mysd]), 1, MPI_INT, otherRank, i, MPI_COMM_WORLD);
		  else
		    MPI_Recv(&otherSubdomainRoot, 1, MPI_INT, otherRank, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  MFEM_VERIFY(otherSubdomainRoot >= 0, "");

		  int isOwner = ((*localInterfaces)[ili].GetOwningRank() == m_rank) ? 1 : 0;
		  sdata[0] = otherSubdomainRoot;
		  sdata[1] = isOwner;
		}
	    }
	}

      {
	// Send otherSubdomainRoot to this subdomain's root, for which ifNDtrue[i] > 0, via a gather for both subdomains.
	
	const int sds[2] = {sd0, sd1};
	for (int j=0; j<2; ++j)
	  {
	    const int sd = sds[j];
	    
	    if (sd_nonempty[sd])
	      {
		std::vector<int> allsdata;

		int sdnprocs = -1;
		MPI_Comm_size(sd_com[sd], &sdnprocs);
		
		if (sd_root[sd] == m_rank)
		  {
		    allsdata.resize(2*sdnprocs);
		  }
		
		MPI_Gather(sdata, 2, MPI_INT, allsdata.data(), 2, MPI_INT, 0, sd_com[sd]);

		if (sd_root[sd] == m_rank)
		  {
		    int otherSubdomainRoot = -1;
		    bool isOwner = false;
		    for (int p=0; p<sdnprocs; ++p)
		      {
			if (allsdata[2*p] >= 0)
			  {
			    MFEM_VERIFY(otherSubdomainRoot == -1 || (otherSubdomainRoot == allsdata[2*p] && isOwner == (allsdata[(2*p) + 1] == 1)), "");

			    otherSubdomainRoot = allsdata[2*p];
			    isOwner = (allsdata[(2*p) + 1] == 1);
			  }
		      }

		    MFEM_VERIFY(ifNDtrue[i] > 0, "");
		    MFEM_VERIFY(ili >= 0, "");
		    
		    if (isOwner)
		      {
			(*localInterfaces)[ili].OverwriteOwningRank(m_rank);
			(*localInterfaces)[ili].OverwriteSharingRank(otherSubdomainRoot);
		      }
		    else
		      {
			(*localInterfaces)[ili].OverwriteOwningRank(otherSubdomainRoot);
			(*localInterfaces)[ili].OverwriteSharingRank(m_rank);
		      }
		  }
	      }
	  }
      }
      
      if (ifNDtrue[i] > 0 && !(sd_nonempty[sd0] && sd_nonempty[sd1]))
	{
	  MFEM_VERIFY(sd_nonempty[sd0] || sd_nonempty[sd1], "");
	  MFEM_VERIFY(ili >= 0, "");
	  MFEM_VERIFY((*localInterfaces)[ili].GetRank() == m_rank, "");
	  MFEM_VERIFY((*localInterfaces)[ili].GetOwningRank() == m_rank || (*localInterfaces)[ili].GetSharingRank() == m_rank, "");
	  MFEM_VERIFY((*localInterfaces)[ili].GetOwningRank() != (*localInterfaces)[ili].GetSharingRank(), "");

	  const int otherRank = ((*localInterfaces)[ili].GetOwningRank() == m_rank) ? (*localInterfaces)[ili].GetSharingRank() : (*localInterfaces)[ili].GetOwningRank();
	  
	  SIsender[i] = new SerialInterfaceCommunicator((2*ifNDtrue[i]) + ifH1true[i], true, otherRank, i);
	  SIreceiver[i] = new SerialInterfaceCommunicator((2*ifNDtrue[i]) + ifH1true[i], false, otherRank, i);
	}
      else
	{
	  SIsender[i] = NULL;
	  SIreceiver[i] = NULL;
	}
#endif

#ifdef SDFOSLS
      // Determine the sign of the normal defined by the ifespace[i] element transformation Jacobians
      // relative to the outward normal of sd0.
      {
	bool if_root = false;
	if (ili >= 0)
	  {
	    if (!(*localInterfaces)[ili].IsEmpty() && m_rank == (*localInterfaces)[ili].GetOwningRank())
	      if_root = true;
	  }

	Vector ni(3);  // Set only for the process with if_root == true
	if (if_root)
	  {
	    MFEM_VERIFY(ifNDtrue[i] > 0, "");
	    const bool nset = GetNormalVector(ifespace[i], ni);
	    MFEM_VERIFY(nset, "");
	  }

	outwardFromSD[i].resize(2);

	const int sds[2] = {sd0, sd1};
	for (int j=0; j<2; ++j)
	  {
	    const int sd = sds[j];

	    Vector nsd(3);

	    const double tol = 1.0e-4;
	    
	    if (sd_nonempty[sd])
	      {
		const bool nset = GetOutwardNormalVector(fespace[sd], InterfaceToSurfaceInjection[sd][i], nsd);

		// Gather nsd to sd root

		std::vector<double> allnsd;
		std::vector<int> allnset;

		int sdnprocs = -1;
		MPI_Comm_size(sd_com[sd], &sdnprocs);
		
		if (sd_root[sd] == m_rank)
		  {
		    allnsd.resize(3*sdnprocs);
		    allnset.resize(sdnprocs);
		  }

		const int inset = nset;
		
		MPI_Gather(&inset, 1, MPI_INT, allnset.data(), 1, MPI_INT, 0, sd_com[sd]);
		MPI_Gather(nsd.GetData(), 3, MPI_DOUBLE, allnsd.data(), 3, MPI_DOUBLE, 0, sd_com[sd]);

		// Set nsd on sd root process and verify consistency among processes.
		if (sd_root[sd] == m_rank)
		  {
		    bool rootset = false;
		    for (int p=0; p<sdnprocs; ++p)
		      {
			if (allnset[p] == 1)
			  {
			    if (rootset)
			      { // Check for consistency
				for (int l=0; l<3; ++l)
				  {
				    MFEM_VERIFY(fabs(nsd[l] - allnsd[(3*p) + l]) < tol, "");
				  }
			      }
			    else
			      { // Set nsd
				for (int l=0; l<3; ++l)
				  nsd[l] = allnsd[(3*p) + l];

				rootset = true;
			      }
			  }
		      }

		    MFEM_VERIFY(rootset, "");

		    MFEM_VERIFY(ili >= 0, "");
		    MFEM_VERIFY(!(*localInterfaces)[ili].IsEmpty(), "");
		    MFEM_VERIFY((*localInterfaces)[ili].GetRank() == m_rank, "");
		    
		    if ((*localInterfaces)[ili].GetOwningRank() != m_rank)
		      {
			MFEM_VERIFY((*localInterfaces)[ili].GetSharingRank() == m_rank, "");
			MPI_Send(nsd.GetData(), 3, MPI_DOUBLE, (*localInterfaces)[ili].GetOwningRank(), 8000 + i, MPI_COMM_WORLD);
		      }
		  }
	      }  // if sd nonempty

	    if (if_root)
	      {
		if (sd_root[sd] != m_rank)
		  {
		    MFEM_VERIFY((*localInterfaces)[ili].GetSharingRank() != m_rank, "");

		    // Send nsd from sd_root[sd] to rank with if_root == true
		    double r[3];
		    MPI_Recv(r, 3, MPI_DOUBLE, (*localInterfaces)[ili].GetSharingRank(), 8000 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    for (int l=0; l<3; ++l)
		      {
			nsd[l] = r[l];
		      }
		  }
		
		// Compute the sign of the normal ni relative to nsd
		const double dp = ni * nsd;
		MFEM_VERIFY(fabs(fabs(dp) - 1.0) < 1.0e-4, "");

		outwardFromSD[i][j] = (dp > 0.0);
	      }
	  }  // j

	// Communicate outwardFromSD[i] from (*localInterfaces)[ili].GetOwningRank() to (*localInterfaces)[ili].GetSharingRank()
	{
	  if (if_root)
	    {
	      MFEM_VERIFY(outwardFromSD[i][0] + outwardFromSD[i][1] == 1, "");
	      if ((*localInterfaces)[ili].GetSharingRank() != m_rank)
		MPI_Send(outwardFromSD[i].data(), 2, MPI_INT, (*localInterfaces)[ili].GetSharingRank(), 9000 + i, MPI_COMM_WORLD);
	    }
	  
	  if (ili >= 0)
	    {
	      if (!(*localInterfaces)[ili].IsEmpty() && m_rank == (*localInterfaces)[ili].GetSharingRank() && m_rank != (*localInterfaces)[ili].GetOwningRank())
		{
		  int r[2];
		  MPI_Recv(r, 2, MPI_INT, (*localInterfaces)[ili].GetOwningRank(), 9000 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  for (int l=0; l<2; ++l)
		    outwardFromSD[i][l] = r[l];
		}
	    }
	}

	// Now outwardFromSD[i] is set in the same way on (*localInterfaces)[ili].GetOwningRank() and (*localInterfaces)[ili].GetSharingRank().
      }
#endif

      // Create operators for interface between subdomains sd0 and sd1, namely C_{sd0,sd1} R_{sd1}^T and the other.
      //if (ifNDtrue[ili] > 0)  // TODO: this is wrong, as it excludes the possibility that this process owns S but not F or rho.
      if (sd_nonempty[sd0] || sd_nonempty[sd1])
	{
#ifdef DDMCOMPLEX
	  // Real part
	  SetToRealParameters();
	  
	  globalInterfaceOpRe->SetBlock(sd0, sd1, CreateInterfaceOperator(sd0, sd1, ili, i, 0, partialConstructor));
#ifndef ELIMINATE_REDUNDANT_VARS
	  globalInterfaceOpRe->SetBlock(sd1, sd0, CreateInterfaceOperator(sd1, sd0, ili, i, 1, partialConstructor));
#endif

	  /*
	  if (sd0 == 0 && sd1 == 1)
	    {
	      const int nr = globalInterfaceOpRe->GetBlock(sd0, sd1).NumRows();
	      const int nc = globalInterfaceOpRe->GetBlock(sd0, sd1).NumCols();
	      Vector tmpx(nc);
	      Vector tmpy(nr);
	      tmpx = 1.0;
	      globalInterfaceOpRe->GetBlock(sd0, sd1).Mult(tmpx, tmpy);
	      //const double nrmx = InnerProduct(tmpx, tmpx);
	      //const double nrmy = InnerProduct(tmpy, tmpy);
	      cout << m_rank << ": GIF01 test " << tmpx.Norml2() << ", " << tmpy.Norml2() << endl;
	      //cout << m_rank << ": GIF01 test " << nrmx << ", " << nrmy << endl;
	    }
	  */

	  // Imaginary part
	  SetToImaginaryParameters();
	
	  globalInterfaceOpIm->SetBlock(sd0, sd1, CreateInterfaceOperator(sd0, sd1, ili, i, 0, partialConstructor));
#ifndef ELIMINATE_REDUNDANT_VARS
	  globalInterfaceOpIm->SetBlock(sd1, sd0, CreateInterfaceOperator(sd1, sd0, ili, i, 1, partialConstructor));
#endif
	}

#ifdef GPWD
      // Note that CreateInterfaceMassRestriction must be called after CreateInterfaceOperator, to ensure offset arrays are defined.

      {
	int isroot = 0;

	if (ifespace[i] != NULL)
	  {
	    int ifrank;
	    MPI_Comm_rank(ifespace[i]->GetComm(), &ifrank);
	    if (ifrank == 0)
	      isroot = 1;
	  }
	
	std::vector<int> allisroot(num_procs);
	MPI_Gather(&isroot, 1, MPI_INT, allisroot.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (m_rank == 0)
	  {
	    ifroot[i] = -1;
	    for (int j=0; j<num_procs; ++j)
	      {
		if (allisroot[j] == 1)
		  {
		    MFEM_VERIFY(ifroot[i] == -1, "");
		    ifroot[i] = j;
		  }
	      }
	    
	    MFEM_VERIFY(ifroot[i] >= 0, "");
	  }
      }
      
      Operator *ifmr0 = NULL;
      Operator *ifmr1 = NULL;
      
      if (sd_nonempty[sd0])
	{
	  ifmr0 = CreateInterfaceMassRestriction(sd0, ili, i, true);
	}
      
      if (sd_nonempty[sd1])
	{
	  ifmr1 = CreateInterfaceMassRestriction(sd1, ili, i, false);
	}
#endif
   
      // Set complex blocks
      rowTrueOffsetsComplexIF[i].SetSize(3);
      colTrueOffsetsComplexIF[i].SetSize(3);

      rowTrueOffsetsComplexIF[i] = 0;
      colTrueOffsetsComplexIF[i] = 0;

      rowTrueOffsetsComplexIF[i][1] = block_trueOffsets[sd0+1] - block_trueOffsets[sd0];
      rowTrueOffsetsComplexIF[i][2] = block_trueOffsets[sd0+1] - block_trueOffsets[sd0];
      
      colTrueOffsetsComplexIF[i][1] = block_trueOffsets[sd1+1] - block_trueOffsets[sd1];
      colTrueOffsetsComplexIF[i][2] = block_trueOffsets[sd1+1] - block_trueOffsets[sd1];

      rowTrueOffsetsComplexIF[i].PartialSum();
      colTrueOffsetsComplexIF[i].PartialSum();

#ifdef SDFOSLS
      rowTrueOffsetsComplexIF_FOSLS[i].SetSize(3);
      colTrueOffsetsComplexIF_FOSLS[i].SetSize(3);

      rowTrueOffsetsComplexIF_FOSLS[i] = 0;
      colTrueOffsetsComplexIF_FOSLS[i] = 0;
      
      rowTrueOffsetsComplexIF_FOSLS[i][1] = block_trueOffsets_FOSLS[sd0+1] - block_trueOffsets_FOSLS[sd0];
      rowTrueOffsetsComplexIF_FOSLS[i][2] = block_trueOffsets_FOSLS[sd0+1] - block_trueOffsets_FOSLS[sd0];

      colTrueOffsetsComplexIF_FOSLS[i][1] = block_trueOffsets_FOSLS[sd1+1] - block_trueOffsets_FOSLS[sd1];
      colTrueOffsetsComplexIF_FOSLS[i][2] = block_trueOffsets_FOSLS[sd1+1] - block_trueOffsets_FOSLS[sd1];

      rowTrueOffsetsComplexIF_FOSLS[i].PartialSum();
      colTrueOffsetsComplexIF_FOSLS[i].PartialSum();
#endif
      
      /*
	PrintNonzerosOfInterfaceOperator(globalInterfaceOpRe->GetBlock(sd0, sd1), fespace[sd0]->GetTrueVSize(), fespace[sd1]->GetTrueVSize(),
	tdofsBdry[sd0], tdofsBdry[sd1], num_procs, rank);
      */
      
      //if (ifNDtrue[ili] > 0)  // TODO: this is wrong, as it excludes the possibility that this process owns S but not F or rho.
      if (sd_nonempty[sd0] || sd_nonempty[sd1])
	{
#ifdef SDFOSLS
	  BlockOperator *complexBlock01 = new BlockOperator(rowTrueOffsetsComplexIF_FOSLS[i], colTrueOffsetsComplexIF[i]);
#else
	  BlockOperator *complexBlock01 = new BlockOperator(rowTrueOffsetsComplexIF[i], colTrueOffsetsComplexIF[i]);
#endif
	  complexBlock01->SetBlock(0, 0, &(globalInterfaceOpRe->GetBlock(sd0, sd1)));
	  complexBlock01->SetBlock(0, 1, &(globalInterfaceOpIm->GetBlock(sd0, sd1)), -1.0);
	  complexBlock01->SetBlock(1, 0, &(globalInterfaceOpIm->GetBlock(sd0, sd1)));
	  complexBlock01->SetBlock(1, 1, &(globalInterfaceOpRe->GetBlock(sd0, sd1)));
	  
#ifndef ELIMINATE_REDUNDANT_VARS
#ifdef SDFOSLS
	  BlockOperator *complexBlock10 = new BlockOperator(colTrueOffsetsComplexIF_FOSLS[i], rowTrueOffsetsComplexIF[i]);
#else
	  BlockOperator *complexBlock10 = new BlockOperator(colTrueOffsetsComplexIF[i], rowTrueOffsetsComplexIF[i]);
#endif
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
      
#ifdef GPWD
      if (ifNDtrue[i] > 0)
	{
	  int osRe = 0;
	  int osIm = ifmrOffsetsComplex[i][1];

	  // First, project u to eTE.
	  if (cgp_u[i] == NULL)
	    cgp_u[i] = new ComplexGlobalProjection(Nphi, osRe, osIm, ifespace[i]->GetComm(), &(eTE_Re[i]), &(eTE_Im[i]), &(eTE_M[i]));

	  // The range of cgp_u is of size 2*Nphi, for real and imaginary parts. The same is true for cgp_f and cgp_rho.
	  
	  // Second, project f to bcr.
	  
	  osRe += ifNDtrue[i];
	  osIm += ifNDtrue[i];

	  if (cgp_f[i] == NULL)
	    cgp_f[i] = new ComplexGlobalProjection(Nphi, osRe, osIm, ifespace[i]->GetComm(), &(bcr_Re[i]), &(bcr_Im[i]), &(bcr_M[i]));
	  
	  // Third, project rho to rhoc.
	  
	  osRe += ifNDtrue[i];
	  osIm += ifNDtrue[i];

	  if (cgp_rho[i] == NULL)
	    cgp_rho[i] = new ComplexGlobalProjection(Nphi, osRe, osIm, iH1fespace[i]->GetComm(), &(rhoc_Re[i]), &(rhoc_Im[i]), &(rhoc_M[i]));
	}

      if (sd_nonempty[sd0]) // && ifNDtrue[i] > 0)
	{
	  // Note that the range of ifmr0 has 3 blocks corresponding to spaces (ifespace[i], ifespace[i], iH1fespace[i]).
	  BlockOperator *complexIFMR = new BlockOperator(ifmrOffsetsComplex[i], rowTrueOffsetsComplexIF[i]);
	  complexIFMR->SetDiagonalBlock(0, ifmr0);
	  complexIFMR->SetDiagonalBlock(1, ifmr0);

	  globalMR->SetBlock(2*i, sd0, complexIFMR);
	}
      
      if (sd_nonempty[sd1]) // && ifNDtrue[i] > 0)
	{
	  BlockOperator *complexIFMR = new BlockOperator(ifmrOffsetsComplex[i], colTrueOffsetsComplexIF[i]);

	  complexIFMR->SetDiagonalBlock(0, ifmr1);
	  complexIFMR->SetDiagonalBlock(1, ifmr1);

	  globalMR->SetBlock((2*i)+1, sd1, complexIFMR);
	}
#endif // GPWD
#else // if DDMCOMPLEX is not defined
      if (ifNDtrue[ili] > 0)
	{
	  globalInterfaceOp->SetBlock(sd0, sd1, CreateInterfaceOperator(ili, 0));
	  globalInterfaceOp->SetBlock(sd1, sd0, CreateInterfaceOperator(ili, 1));
	}
#endif
    }

  MPI_Barrier(MPI_COMM_WORLD);

  chronoSD.Stop();
  if (m_rank == 0)
    cout << m_rank << ": DDM constructor IF loop timing " << chronoSD.RealTime() << endl;
  
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
#ifdef SDFOSLS
  rowTrueOffsetsComplexSD2.resize(numSubdomains);
  colTrueOffsetsComplexSD2.resize(numSubdomains);

  rowTrueOffsetsSD2.resize(numSubdomains);
  colTrueOffsetsSD2.resize(numSubdomains);
  
  globalSubdomainOp = new BlockOperator(block_trueOffsets2, block_trueOffsets_FOSLS2);
#else
  globalSubdomainOp = new BlockOperator(block_trueOffsets2);
#endif
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

  chronoSD.Clear();
  chronoSD.Start();

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

#ifdef SERIAL_INTERFACES
	if (!sd_nonempty[m])
	  ifsize = 0;
#endif
	
	rowTrueOffsetsSD[m][2] = ifsize;

	//cout << rank << ": SD " << m << " ifsize " << ifsize << endl;
	    
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

#ifdef SDFOSLS
	rowTrueOffsetsSD2[m].SetSize(3 + 1);  // Number of blocks + 1
	colTrueOffsetsSD2[m].SetSize(3 + 1);  // Number of blocks + 1

	rowTrueOffsetsSD2[m] = 0;
	rowTrueOffsetsSD2[m][1] = (fespace[m] != NULL) ? fespace[m]->GetTrueVSize() : 0;
	rowTrueOffsetsSD2[m][2] = rowTrueOffsetsSD2[m][1];

	rowTrueOffsetsSD2[m][3] = ifsize;

	colTrueOffsetsSD2[m] = rowTrueOffsetsSD2[m];
	colTrueOffsetsSD2[m][1] = tdofsBdry[m].size();
	colTrueOffsetsSD2[m][2] = tdofsBdry[m].size();

	rowTrueOffsetsSD2[m].PartialSum();
	colTrueOffsetsSD2[m].PartialSum();
	
	injSD2[m] = new BlockOperator(rowTrueOffsetsSD2[m], colTrueOffsetsSD2[m]);

	if (tdofsBdryInjection[m] != NULL)
	  {
	    injSD2[m]->SetBlock(0, 0, tdofsBdryInjection[m]);  // E block
	    injSD2[m]->SetBlock(1, 1, tdofsBdryInjection[m]);  // H block
	  }

	injSD2[m]->SetBlock(2, 2, new IdentityOperator(ifsize));
#endif

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

#ifdef SDFOSLS
	rowTrueOffsetsComplexSD2[m].SetSize(2 + 1);  // Number of blocks + 1
	colTrueOffsetsComplexSD2[m].SetSize(2 + 1);  // Number of blocks + 1

	rowTrueOffsetsComplexSD2[m] = 0;
	colTrueOffsetsComplexSD2[m] = 0;

	rowTrueOffsetsComplexSD2[m][1] = rowTrueOffsetsSD2[m][3];
	rowTrueOffsetsComplexSD2[m][2] = rowTrueOffsetsSD2[m][3];

	colTrueOffsetsComplexSD2[m][1] = colTrueOffsetsSD2[m][3];
	colTrueOffsetsComplexSD2[m][2] = colTrueOffsetsSD2[m][3];

	rowTrueOffsetsComplexSD2[m].PartialSum();
	colTrueOffsetsComplexSD2[m].PartialSum();
	
	injComplexSD2[m] = new BlockOperator(rowTrueOffsetsComplexSD2[m], colTrueOffsetsComplexSD2[m]);
	injComplexSD2[m]->SetBlock(0, 0, injSD2[m]);
	injComplexSD2[m]->SetBlock(1, 1, injSD2[m]);
#endif
	
	block_ComplexOffsetsSD[m].SetSize(3);  // number of blocks plus 1
	block_ComplexOffsetsSD[m] = 0;
	block_ComplexOffsetsSD[m][1] = rowTrueOffsetsSD[m][2];
	block_ComplexOffsetsSD[m][2] = rowTrueOffsetsSD[m][2];
	block_ComplexOffsetsSD[m].PartialSum();

#ifndef NO_SD_OPERATOR	
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
#endif
	  {
	    AsdComplex[m] = NULL;
	    precAsdComplex[m] = NULL;
	  }
	
	//MPI_Barrier(MPI_COMM_WORLD);
  
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

	MFEM_VERIFY((locSizeSD > 0) == sd_nonempty[m], "");
	
	if (locSizeSD > 0)
	  {
	    // Note: the following code cannot be removed and put in CreateHypreParMatrixFromBlocks, because the number of blocks varies on different
	    // processes due to the local ownership of subdomain interfaces.
	    
	    const int numBlocks = (2*subdomainLocalInterfaces[m].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
	    MFEM_VERIFY(numBlocks == trueOffsetsSD[m].Size()-1, "");
	    
	    std::vector<std::vector<int> > blockProcOffsets(numBlocks);
	    std::vector<int> blockGI(numBlocks);
	    blockGI[0] = -1;  // denotes subdomain
	    for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
	      {
		blockGI[(2*i) + 1] = 2*subdomainLocalInterfaces[m][i];
		blockGI[(2*i) + 2] = (2*subdomainLocalInterfaces[m][i]) + 1;
	      }

	    int sdnprocs, sdrank;
	    MPI_Comm_rank(sd_com[m], &sdrank);
	    MPI_Comm_size(sd_com[m], &sdnprocs);

	    std::vector<int> allnumrows(sdnprocs);
	    int blockNumRows = trueOffsetsSD[m][1];
	    MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, sd_com[m]);

	    std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
	    
	    blockProcOffsets[0].resize(sdnprocs);
	    blockProcOffsets[0][0] = 0;
	    for (int i=0; i<sdnprocs-1; ++i)
	      blockProcOffsets[0][i+1] = blockProcOffsets[0][i] + allnumrows[i];

	    all_block_num_loc_rows[0].resize(sdnprocs);
	    for (int i=0; i<sdnprocs; ++i)
	      all_block_num_loc_rows[0][i] = allnumrows[i];
	    
	    for (auto interfaceIndex : allGlobalSubdomainInterfaces[m])  // loop over ordered global interfaces for consistency in parallel
	      {
		int k = -1;
		for (int i=0; i<subdomainLocalInterfaces[m].size(); ++i)
		  {
		    if (subdomainLocalInterfaces[m][i] == interfaceIndex)
		      k = i;
		  }

		blockNumRows = ifNDtrue[interfaceIndex];
		MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, sd_com[m]);

		if (k >= 0)
		  {
		    const int j = (2*k) + 1;
		    blockProcOffsets[j].resize(sdnprocs);
		    blockProcOffsets[j][0] = 0;
		    for (int i=0; i<sdnprocs-1; ++i)
		      blockProcOffsets[j][i+1] = blockProcOffsets[j][i] + allnumrows[i];

		    all_block_num_loc_rows[j].resize(sdnprocs);
		    for (int i=0; i<sdnprocs; ++i)
		      all_block_num_loc_rows[j][i] = allnumrows[i];
		  }
		
		blockNumRows = ifH1true[interfaceIndex];
		MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, sd_com[m]);

		if (k >= 0)
		  {
		    const int j = (2*k) + 2;
		    blockProcOffsets[j].resize(sdnprocs);
		    blockProcOffsets[j][0] = 0;
		    for (int i=0; i<sdnprocs-1; ++i)
		      blockProcOffsets[j][i+1] = blockProcOffsets[j][i] + allnumrows[i];

		    all_block_num_loc_rows[j].resize(sdnprocs);
		    for (int i=0; i<sdnprocs; ++i)
		      all_block_num_loc_rows[j][i] = allnumrows[i];
		  }
	      }

	    {
	    /*
	      std::string filename = "ifNDmassSp" + std::to_string(m) + ".txt";
	      std::ofstream sfile(filename, std::ofstream::out);
	      ifNDmassSp[0]->Print(sfile);
	    */
	      /*
	      std::string filename = "ifNDcurlcurlSp" + std::to_string(m) + ".txt";
	      std::ofstream sfile(filename, std::ofstream::out);
	      ifNDcurlcurlSp[0]->Print(sfile);
	      */
	    }

	    {
	      /*
	      std::string filename = "ifNDmass" + std::to_string(m) + ".txt";
	      ifNDmass[0]->Print(filename.c_str());
	      */
	      /*
	      std::string filename = "ifNDcurlcurl" + std::to_string(m) + ".txt";
	      ifNDcurlcurl[0]->Print(filename.c_str());
	      */
	    }

	    // TODO: in case SD_ITERATIVE_GMG_PA, compute operator versions of HypreAsdComplexRe, HypreAsdComplexIm.
#ifndef SD_ITERATIVE_GMG_PA
#ifndef SDFOSLS
	    HypreParMatrix *HypreAsdComplexRe = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], AsdRe_HypreBlocks[m],
									       //#ifdef SERIAL_INTERFACES
									       AsdRe_SparseBlocks[m],
									       //#endif
									       AsdRe_HypreBlockCoef[m], blockGI, blockProcOffsets, all_block_num_loc_rows);
									       
	    HypreParMatrix *HypreAsdComplexIm = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], AsdIm_HypreBlocks[m],
									       //#ifdef SERIAL_INTERFACES
									       AsdIm_SparseBlocks[m],
									       //#endif
									       AsdIm_HypreBlockCoef[m], blockGI, blockProcOffsets, all_block_num_loc_rows);
#endif
#endif
    
#ifdef SD_ITERATIVE
#ifdef SD_ITERATIVE_FULL
	    HypreParMatrix *HypreDsdComplexRe = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], DsdRe_HypreBlocks[m], DsdRe_SparseBlocks[m],
									       DsdRe_HypreBlockCoef[m]); // , blockGI, blockProcOffsets, all_block_num_loc_rows);
									       
	    HypreParMatrix *HypreDsdComplexIm = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsSD[m], DsdIm_HypreBlocks[m], DsdIm_SparseBlocks[m],
									       DsdIm_HypreBlockCoef[m]); // , blockGI, blockProcOffsets, all_block_num_loc_rows);
#else
#ifndef IF_ITERATIVE
	    std::vector<std::vector<int> > blockProcOffsetsAux(numBlocks-1);
	    std::vector<std::vector<int> > all_block_num_loc_rows_Aux(numBlocks-1);

	    for (int i=0; i<numBlocks-1; ++i)
	      {
		blockProcOffsetsAux[i].resize(blockProcOffsets[i+1].size());
		all_block_num_loc_rows_Aux[i].resize(all_block_num_loc_rows[i+1].size());

		blockProcOffsetsAux[i] = blockProcOffsets[i+1];
		all_block_num_loc_rows_Aux[i] = all_block_num_loc_rows[i+1];
	      }

	    // TODO: this should just be a combination of serial sparse blocks, which could be factorized by UMFPACK on the subdomain root process.
#ifndef SDFOSLS_PA
	    HypreParMatrix *HypreDsdComplexRe = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsAuxSD[m], DsdRe_HypreBlocks[m], DsdRe_SparseBlocks[m],
									       DsdRe_HypreBlockCoef[m], blockGI, blockProcOffsetsAux, all_block_num_loc_rows_Aux);
									       
	    HypreParMatrix *HypreDsdComplexIm = CreateHypreParMatrixFromBlocks(sd_com[m], trueOffsetsAuxSD[m], DsdIm_HypreBlocks[m], DsdIm_SparseBlocks[m],
									       DsdIm_HypreBlockCoef[m], blockGI, blockProcOffsetsAux, all_block_num_loc_rows_Aux);
#endif
#endif
#endif
#endif

	    { // FreeSubdomainHypreBlocks(m);
	      /*
	      FreeHypreArray(AsdRe_HypreBlocks);
	      FreeHypreArray(AsdIm_HypreBlocks);

	      FreeSparseMatrixArray(AsdRe_SparseBlocks);
	      FreeSparseMatrixArray(AsdIm_SparseBlocks);
	      */
	    }
	    
#ifndef SD_ITERATIVE_GMG_PA
#ifndef SDFOSLS
	    if (m == -1)
	      {

		const int Nm = HypreAsdComplexIm->Height();
		MFEM_VERIFY(Nm > 0, "");

		Vector ej(Nm);
		Vector Aej(Nm);

		//double ejnorm = 0.0;
		double Aejnorm = 0.0;
		
		//for (int j=83024; j<(83024 + 2624); ++j)
		for (int j=0; j<Nm; ++j)
		  {
		    ej = 0.0;

		    if (sdrank == 0)
		      {
			MFEM_VERIFY(j < Nm, "");
			if (j >= Nm)
			  cout << "ERROR out of bounds" << endl;
		    
			ej[j] = 1.0;
		      }

		    //ejnorm = ej.Norml2();

		    //HypreAsdComplexRe->Mult(ej, Aej);

		    HypreAsdComplexIm->MultTranspose(ej, Aej);  // Aej is the j-th row of A
		    Aejnorm = Aej.Norml2();

		    cout << m_rank << ": sd " << m << " j " << j << " Aejnorm " << Aejnorm << endl;
		    
		    /*
		    if (sdrank == 0)
		      {
			ofstream tmpfile("rank36out3");
			Aej.Print(tmpfile);
			tmpfile.close();
		      }
		    */
		  }

		/*		
		Vector ej(std::max(ifH1true[1], 1));
		Vector Aej(std::max(ifNDtrue[1], 1));
		ej = 0.0;
		if (sdrank == 0)
		  ej[0] = 1.0;

		if (ifNDtrue[1] > 0)
		  ifNDH1grad[1]->Mult(ej, Aej);

		Aej *= gammaOverAlpha;
		
		if (sdrank == 0)
		  {
		    ofstream tmpfile("rank36out5");
		    Aej.Print(tmpfile);
		    tmpfile.close();
		  }
		*/
	      }

	    {
	      ComplexHypreParMatrix tmpComplex(HypreAsdComplexRe, HypreAsdComplexIm, false, false);
	      HypreAsdComplex[m] = tmpComplex.GetSystemMatrix();

	      delete HypreAsdComplexRe;
	      delete HypreAsdComplexIm;
	    }
#endif
#endif
	    
#ifdef SD_ITERATIVE
#ifndef IF_ITERATIVE
#ifndef SDFOSLS_PA
	    {
	      ComplexHypreParMatrix tmpDiagComplex(HypreDsdComplexRe, HypreDsdComplexIm, false, false);
	      HypreDsdComplex[m] = tmpDiagComplex.GetSystemMatrix();

	      delete HypreDsdComplexRe;
	      delete HypreDsdComplexIm;
	    }
#endif
#endif
#endif
	    
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
	    //if (m == 1) HypreAsdComplexRe->Print("HypreAsdComplexRe1_Par7");
	    //if (m == 1) HypreAsdComplexRe->Print("HypreAsdComplexRe1_Serial");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0_Serial");
	    //if (m == 1) HypreAsdComplexIm->Print("HypreAsdComplexIm1_Serial");
	    //if (m == 0) HypreAsdComplexIm->Print("HypreAsdComplexIm0_Par5");
	    //if (m == 1) HypreAsdComplexIm->Print("HypreAsdComplexIm1_Par6");
	    //if (m == 1) HypreAsdComplex[m]->Print("HypreAsdComplex_Par7");
	    //if (m == 1) HypreAsdComplex[m]->Print("HypreAsdComplex_Serial");
		
	    //invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), MPI_COMM_WORLD);
#ifdef EUCLID_INVERSE
	    GMRESSolver *gmres = new GMRESSolver(sd_com[m]);

	    gmres->SetOperator(*(HypreAsdComplex[m]));

	    gmres->SetRelTol(1e-12);
	    gmres->SetMaxIter(100);  // 3333
	    gmres->SetPrintLevel(1);

	    HypreEuclid *precEuclid = new HypreEuclid(*(HypreAsdComplex[m]));
	    gmres->SetPreconditioner(*precEuclid);
	    gmres->SetName("invAsdComplexEuclid" + std::to_string(m));
	    gmres->iterative_mode = false;
	    
	    invAsdComplex[m] = gmres;
#else
#ifdef SD_ITERATIVE
	    {
	      GMRESSolver *gmres = new GMRESSolver(sd_com[m]);

#ifdef SD_ITERATIVE_GMG_PA
#ifndef SDFOSLS
	      ComplexOperator *AsdPA = new ComplexOperator(AsdPARe[m], AsdPAIm[m], false, false);
	      gmres->SetOperator(*AsdPA);
#endif
#else
#ifndef SDFOSLS
	      gmres->SetOperator(*(HypreAsdComplex[m]));
#endif
#endif
	      
	      gmres->SetRelTol(1e-12);
	      gmres->SetMaxIter(50);
	      gmres->SetKDim(50);
	      gmres->SetPrintLevel(-1);

	      if (m == -10)
		gmres->SetPrintLevel(1);
	      
#ifdef SD_ITERATIVE_FULL
	      STRUMPACKSolver *sdprec = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreDsdComplex[m])), sd_com[m]);
#else
#ifdef SD_ITERATIVE_COMPLEX
#ifdef SD_ITERATIVE_GMG
	      {
#ifdef SD_ITERATIVE_GMG_PA
#ifndef SDFOSLS_PA
		ComplexGMGPASolver *cgmg = NULL;
#endif
		{
		  /*
		  BlockOperator* AsdRe_m = dynamic_cast<BlockOperator*>(AsdRe[m]);
		  BlockOperator* AsdIm_m = dynamic_cast<BlockOperator*>(AsdIm[m]);

		  MFEM_VERIFY(AsdRe_m && AsdIm_m, "");
		  */
		  /*
		  Vector diagRe, diagIm;
		  AsdRe_HypreBlocks[m](0,0)->GetDiag(diagRe);
		  AsdIm_HypreBlocks[m](0,0)->GetDiag(diagIm);
		  */
		  
		  Vector diagSDND;

		  if (fespace[m] != NULL)
		    {
		      ParGridFunction diag_pa(fespace[m]);
		      bfpa_sdND[m]->AssembleDiagonal(diag_pa);
		      diagSDND.SetSize(fespace[m]->GetTrueVSize());
		      diag_pa.GetTrueDofs(diagSDND);
		    }
		  
		  Array<int> ess_tdof_list;  // empty

		  //cgmg = new ComplexGMGPASolver(sd_com[m], &(AsdRe_m->GetBlock(0,0)), &(AsdIm_m->GetBlock(0,0)), diagRe, ess_tdof_list, sdP[m], (*sdcRe)[m], (*sdcIm)[m]);
		  //cgmg = new ComplexGMGPASolver(sd_com[m], AsdRe_HypreBlocks[m](0,0), AsdIm_HypreBlocks[m](0,0), diagRe, ess_tdof_list, sdP[m], (*sdcRe)[m], (*sdcIm)[m]);
		  //cgmg = new ComplexGMGPASolver(sd_com[m], AsdRe_HypreBlocks[m](0,0), AsdIm_HypreBlocks[m](0,0), diagSDND, ess_tdof_list, sdP[m], (*sdcRe)[m], (*sdcIm)[m]);
#ifdef TEST_SD_CMG
		  cgmg = new ComplexGMGPASolver(sd_com[m], oppa_sdND[m].Ptr(), NULL, diagSDND, sd_ess_tdof_list[m], sdP[m], (*sdcRe)[m], (*sdcIm)[m]); //, (m == 0));
		  if (m == -10)
		    sdP[0][0]->Print("P0.txt");
#else
#ifndef SDFOSLS_PA
		  cgmg = new ComplexGMGPASolver(sd_com[m], &(AsdPARe[m]->GetBlock(0,0)), &(AsdPAIm[m]->GetBlock(0,0)), diagSDND, sd_ess_tdof_list[m], sdP[m], (*sdcRe)[m], (*sdcIm)[m]); //, (m == 0));
#endif
#endif
		}
#else
#ifdef TEST_SD_CMG
		if (m == -10)
		  {
		    //HypreParMatrix *AC0 = RAP(sdND[m], sdP[0][0]);
		    //AC0->Print("AC0_.txt");

		    SparseMatrix spND, spP;

		    sdND[m]->GetDiag(spND);
		    sdP[0][0]->GetDiag(spP);

		    SparseMatrix *spR = Transpose(spP);
		    
		    /*
		    SparseMatrix *spRAP = RAP(spND, *spR);
		    ofstream rapout("spRAP0.txt");
		    
		    spRAP->Print(rapout);
		    rapout.close();
		    delete spRAP;
		    */
		    /*
		    ofstream srout("spR0.txt");
		    spR->Print(srout);
		    srout.close();

		    delete spR;
		    */
		    /*
		    ofstream ndout("spND0.txt");
		    ofstream spout("spP0.txt");

		    spND.Print(ndout);
		    spP.Print(spout);
		    
		    ndout.close();
		    spout.close();
		    */
		    
		    //delete AC0;
		  }
		
		ComplexHypreParMatrix *AZ = new ComplexHypreParMatrix(sdND[m], NULL, false, false);
#else
#ifndef SDFOSLS
		ComplexHypreParMatrix *AZ = new ComplexHypreParMatrix(AsdRe_HypreBlocks[m](0,0), AsdIm_HypreBlocks[m](0,0), false, false);
#endif
#endif
#ifndef SDFOSLS
#ifdef MFEM_USE_STRUMPACK
		ComplexGMGSolver *cgmg = new ComplexGMGSolver(AZ, sdP[m], ComplexGMGSolver::CoarseSolver::STRUMPACK); //, (m == 0));
#else
		ComplexGMGSolver *cgmg = new ComplexGMGSolver(AZ, sdP[m], ComplexGMGSolver::CoarseSolver::UMFPACK); //, (m == 0));
#endif
		cgmg->SetSmootherType(HypreSmoother::Jacobi);  // Some options: Jacobi, l1Jacobi, l1GS, GS
#endif
#endif

#ifdef SDFOSLS
		sdNDinv[m] = NULL;
#else
		cgmg->SetTheta(0.5);
		sdNDinv[m] = cgmg;
#endif
		
		/*
		{ // Use cgmg as a preconditioner for GMRES, rather than a solver.
		  GMRESSolver *gmresSD = new GMRESSolver(sd_com[m]);

#ifdef SD_ITERATIVE_GMG_PA
#ifdef TEST_SD_CMG
		  ComplexOperator *copSD = new ComplexOperator(oppa_sdND[m].Ptr(), NULL, false, false, ComplexOperator::HERMITIAN);
#else
		  ComplexOperator *copSD = new ComplexOperator(&(AsdPARe[m]->GetBlock(0,0)), &(AsdPAIm[m]->GetBlock(0,0)), false, false, ComplexOperator::HERMITIAN);
#endif
		  gmresSD->SetOperator(*copSD);
#else
		  gmresSD->SetOperator(*AZ);
#endif
		  
		  gmresSD->SetRelTol(1e-12);
		  gmresSD->SetMaxIter(100);
		  gmresSD->SetKDim(100);
		  gmresSD->SetPrintLevel(-1);
		  if (m == 0)
		    gmresSD->SetPrintLevel(1);
		  
		  gmresSD->SetPreconditioner(*cgmg);
		  gmresSD->SetName("sdNDcgmg" + std::to_string(m));
		  gmresSD->iterative_mode = false;

		  sdNDinv[m] = gmresSD;

		  if (m == -10)
		  { // Test gmresSD
#ifdef SD_ITERATIVE_GMG_PA
		    const int n_m = 2*oppa_sdND[m].Ptr()->Height();
#else
		    const int n_m = 2*sdND[m]->Height();
#endif
		    Vector tx(n_m);
		    Vector ty(n_m);
		    tx = 1.0;
		    ty = 0.0;
		    gmresSD->Mult(tx, ty);
		    //cgmg->Mult(tx, ty);
		    cout << "gmresSD test ty norm " << ty.Norml2() << endl;
		  }
		}
		*/
		
		if (m == 0)
		  {
#ifdef TEST_SD_CMG
		    TestGMG(sdND[m], sdP[m], fespace[m]);
#endif
		    //TestGMG(AsdRe_HypreBlocks[m](0,0), sdP[m]);
		    //TestCGMG(AsdRe_HypreBlocks[m](0,0), AsdIm_HypreBlocks[m](0,0), sdP[m]);
		  }
	      }
#else  // not GMG
	      {
		ComplexHypreParMatrix tmpSDComplex(AsdRe_HypreBlocks[m](0,0), AsdIm_HypreBlocks[m](0,0), false, false);
		HypreParMatrix *HypreSDComplex = tmpSDComplex.GetSystemMatrix();

#ifdef MFEM_USE_STRUMPACK
		sdNDinv[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreSDComplex)), sd_com[m]);
		delete HypreSDComplex;
#else
		sdNDinv[m] = CreateSerialUMFPackSolver(HypreSDComplex, sd_com[m]);
#endif
	      }
#endif
#else  // not SD_ITERATIVE_COMPLEX
	      sdNDinv[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(sdND[m])), sd_com[m]);
	      //sdNDinv[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(AsdRe_HypreBlocks[m](0,0))), sd_com[m]);  // sdND[m] plus the real interface terms.
	      //sdImag[m] = CreateSubdomainImaginaryPart(m);
	      sdImag[m] = AsdIm_HypreBlocks[m](0,0);
#endif
#ifdef IF_ITERATIVE
	      /*
	      HyprePCG *auxInv = new HyprePCG(*(HypreDsdComplex[m]));
	      auxInv->SetTol(1e-12);
	      auxInv->SetMaxIter(100);
	      auxInv->SetPrintLevel(-1);
	      //auxInv->SetPreconditioner(*ams);
	      */
#ifdef SD_ITERATIVE_GMG_PA
	      DDAuxSolver *auxInv = new DDAuxSolver(&(allGlobalSubdomainInterfaces[m]), oppa_ifNDmass, alphaIm, &ifH1true);
#else
	      DDAuxSolver *auxInv = new DDAuxSolver(&(allGlobalSubdomainInterfaces[m]), ifNDmassSp, alphaIm, &ifH1true);
#endif
#else
#ifdef SDFOSLS_PA
	      IdentityOperator *auxInv = new IdentityOperator(2*trueOffsetsAuxSD[m][trueOffsetsAuxSD[m].Size() - 1]);
#else
#ifdef MFEM_USE_STRUMPACK
	      STRUMPACKSolver *auxInv = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreDsdComplex[m])), sd_com[m]);
#else
	      Solver *auxInv = CreateSerialUMFPackSolver(HypreDsdComplex[m], sd_com[m]);
#endif
#endif
#endif
#ifndef SDFOSLS
#ifdef SD_ITERATIVE_GMG_PA
	      Solver *sdprec = new BlockSubdomainPreconditioner(AsdPA->Height(), sdNDinv[m], sdImag[m], auxInv);
#else
	      Solver *sdprec = new BlockSubdomainPreconditioner(HypreAsdComplex[m]->Height(), sdNDinv[m], sdImag[m], auxInv);		
#endif
#endif
#endif

#ifndef SDFOSLS
	      gmres->SetPreconditioner(*sdprec);
#endif
	      gmres->SetName("invAsdComplexIter" + std::to_string(m));
	      gmres->iterative_mode = false;
	      
#ifdef SDFOSLS
	      // TODO: in this case, do not construct everything that went into gmres
#ifdef ROBIN_TC
	      //if (m == 1)
	      if (true)
		{
		  ParGridFunction xsd(fespace[m]);
		  {
		    VectorFunctionCoefficient Eexact(3, test2_E_exact);
		    xsd.ProjectCoefficient(Eexact);
		  }

		  cfosls[m] = NULL; // TODO: remove this
		  invAsdComplex[m] = NULL; // TODO: remove this
		  
		  {
#ifdef TEST_DECOUPLED_FOSLS
		    {
		      Array<int> ess_bdr(pmeshSD[m]->bdr_attributes.Max());
		      ess_bdr = 1;
		      fespace[m]->GetEssentialTrueDofs(ess_bdr, sd_all_ess_tdof_list[m]);
		      cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], sd_all_ess_tdof_list[m], xsd, sdP[m], sqrt(k2));
		    }
#else
#ifdef TEST_DECOUPLED_FOSLS_PROJ_FULL
		    Array<int> ess_tdof_list_empty;
		    cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], ess_tdof_list_empty, xsd, sdP[m], sqrt(k2));
#else
#ifdef ZERO_ORDER_FOSLS
		    Array<int> ess_tdof_list_empty;
		    cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], ess_tdof_list_empty, xsd, sdP[m], sqrt(k2));
#else
#ifdef IFFOSLS
#ifdef IFFOSLS_ESS
		    cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], sd_ess_tdof_list[m], xsd, sdP[m], sqrt(k2));
#else
		    Array<int> ess_tdof_list_empty;
		    cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], ess_tdof_list_empty, xsd, sdP[m],
#ifdef SDFOSLS_PA
						(*coarseFOSLS)[m], partialConstructor,
#endif
						sqrt(k2));
#endif
#else
		    cfosls[m] = new FOSLSSolver(sd_com[m], fespace[m], sd_ess_tdof_list[m], xsd, sdP[m], sqrt(k2));
#endif
#endif
#endif
#endif

#ifdef IFFOSLS_H
		    invAsdComplex[m] = new LowerBlockTriangularSubdomainSolver(HypreAsdComplex[m], cfosls[m], auxInv, tdofsBdryInjectionTranspose[m],
									       &(allGlobalSubdomainInterfaces[m]), &(InterfaceToSurfaceInjection[m]),
									       &(ifNDtrue), &(ifH1true));

#else
		    invAsdComplex[m] = new LowerBlockTriangularSubdomainSolver(HypreAsdComplex[m], cfosls[m], auxInv);
#endif							       
		  }
		}
	      else
		invAsdComplex[m] = gmres;
#else
	      invAsdComplex[m] = NULL;  // TODO: in the SOTC case, the system is not lower block triangular, due to the (v,rho) block.
#endif
#else
	      invAsdComplex[m] = gmres;
#endif
	    }
#else
	    invAsdComplex[m] = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), sd_com[m]);
	    //invAsdComplex[m] = CreateStrumpackSolverApproxV2(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), sd_com[m]);
#endif
	    /*
	    {
	      GMRESSolver *gmres = new GMRESSolver(sd_com[m]);

	      gmres->SetOperator(*(HypreAsdComplex[m]));

	      gmres->SetRelTol(1e-12);
	      gmres->SetMaxIter(100);  // 3333
	      gmres->SetPrintLevel(1);

	      STRUMPACKSolver *sprec = CreateStrumpackSolverApproxV2(new STRUMPACKRowLocMatrix(*(HypreAsdComplex[m])), sd_com[m]);
	      gmres->SetPreconditioner(*sprec);
	      gmres->SetName("invAsdComplexHSSGMRES" + std::to_string(m));
	      gmres->iterative_mode = false;
	    
	      invAsdComplex[m] = gmres;
	    }
	    */

#ifndef TEST_SD_CMG
#ifndef SDFOSLS
	    // Delete subdomain blocks. Note that interface blocks must be deleted after the loop over subdomains, since they can be shared.
	    delete sdND[m];
#endif
#endif

#ifndef SD_ITERATIVE
	    delete HypreAsdComplex[m];
#endif
#endif
	    
	    //invAsdComplex[m] = HypreAsdComplex[m];
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

	//CompareOperators(HypreAsdComplex[m], AsdComplex[m], sd_com[m]);
	
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

#ifdef TEST_SD_SOLVER
	if (m == 2 && invAsdComplex[m] != NULL)
	  {
	    //CompareOperators(HypreAsdComplex[m], AsdComplex[m], sd_com[m]);

	    int sdrank = -1;
	    MPI_Comm_rank(sd_com[m], &sdrank);
  
	    const int Nm = AsdComplex[m]->Height();
	    MFEM_VERIFY(Nm > 0, "");
	    MFEM_VERIFY(Nm == AsdComplex[m]->Width(), "");

	    Vector ej(Nm);
	    Vector Aej(Nm);
	    Vector rej(Nm);

	    for (int j=0; j<300; ++j)
	      {
		ej = 0.0;

		if (sdrank == 0)
		  {
		    if (j >= Nm)
		      cout << "ERROR out of bounds" << endl;
		    
		    MFEM_VERIFY(j < Nm, "");

		    ej[j] = 1.0;
		  }
	    
		//AsdComplex[m]->Mult(ej, Aej);
		
		//AsdComplex[m]->Mult(ej, rej);
		//double opnrm = rej.Norml2();
		double opnrm = 0.0;
		
		HypreAsdComplex[m]->Mult(ej, Aej);
		const double Anrm = Aej.Norml2();

		/*
		if (sdrank == 0)
		  {
		    ofstream tmpfile("rank36out2");
		    rej.Print(tmpfile);
		    Aej.Print(tmpfile);
		    tmpfile.close();
		  }
		*/
		
		rej -= Aej;
		double diffnrm = rej.Norml2();

		invAsdComplex[m]->Mult(Aej, rej);

		double rnrm = rej.Norml2();
		double rnrm2 = rnrm*rnrm;
		double sumnrm2 = 0.0;

		const double locrnrm = rnrm;
		
		MPI_Allreduce(&rnrm2, &sumnrm2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);

		rej -= ej;
		
		rnrm = rej.Norml2();
		rnrm2 = rnrm*rnrm;
		double sumdnrm2 = 0.0;
		
		MPI_Allreduce(&rnrm2, &sumdnrm2, 1, MPI_DOUBLE, MPI_SUM, sd_com[m]);

		if (sumdnrm2 > 1.0e-18 || j % 100 == 0)
		  cout << m_rank << ": j " << j << " of " << Nm << ", norm of rej " << sqrt(sumnrm2) << ", norm of diff " << sqrt(sumdnrm2)
		       << ", local rej norm " << locrnrm << ", local Aej norm " << Anrm << ", local op*ej norm " << opnrm << ", local (op-H)*ej norm "
		       << diffnrm << ", sdrank " << sdrank << endl;
	      }
	    
	    /*
	    {
	      Vector uj(AsdIm[m]->Width());
	      Vector Auj(AsdIm[m]->Width());

	      uj = 0.0;
	      if (sdrank == 0)
		uj[120] = 1.0;

	      Vector uju(trueOffsetsSD[m][1]);
	      for (int l=trueOffsetsSD[m][0]; l<trueOffsetsSD[m][1]; ++l)
		uju[l] = uj[l];

	      Vector ujSIF(ifNDtrue[1]);
	      InterfaceToSurfaceInjection[m][1]->MultTranspose(uju, ujSIF);
	      const double ujSIFnrm = (ifNDtrue[1] > 0) ? ujSIF.Norml2() : 0.0;
	      
	      cout << m_rank << ": uju norm " << uju.Norml2() << ", restriction norm " << ujSIFnrm << endl;
	      
	      AsdIm[m]->Mult(uj, Auj);

	      if (sdrank == -1)
		{
		  ofstream tmpfile("rank36out6");
		  Auj.Print(tmpfile);
		  tmpfile.close();
		}
	      
	      if (sdrank == 0)
		{
		  cout << m_rank << ": trueOffsetsSD[" << m << "] size " << trueOffsetsSD[m].Size() << endl;

		  for (int j=0; j<subdomainLocalInterfaces[m].size(); ++j)
		    {
		      const int interfaceIndex = subdomainLocalInterfaces[m][j];
		      cout << m_rank << ": global interfaces for sd " << m << ": " << interfaceIndex << endl;
		    }
		    }
	  }
	    */
	  }
#endif
	
	if (invAsdComplex[m] != NULL)
	  {
#ifdef SDFOSLS
	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injComplexSD[m]), invAsdComplex[m], injComplexSD2[m],
									false, false, false));
#else
	    globalSubdomainOp->SetBlock(m, m, new TripleProductOperator(new TransposeOperator(injComplexSD[m]), invAsdComplex[m], injComplexSD[m],
									false, false, false));
#endif
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

  MPI_Barrier(MPI_COMM_WORLD);

  // Note that interface matrices cannot be deleted here, because they are used by Cij (see CreateCij()).
  
  chronoSD.Stop();
  if (m_rank == 0)
    cout << m_rank << ": DDM constructor SD loop 3 timing " << chronoSD.RealTime() << endl;

#ifdef SDFOSLS_PA
  if (partialConstructor)
    return;
#endif

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

  globalOffdiag = new ProductOperator(globalSubdomainOp, globalInterfaceOp, false, false);
  globalOp = new SumOperator(globalOffdiag, new IdentityOperator(height), false, false, 1.0, 1.0);

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

#ifdef GPWD
  if (false)  // Test ProjectionMult by checking whether P^2 = P.
    {
      std::default_random_engine generator;
      std::uniform_real_distribution<double> distribution(-1.0,1.0);

      for (int i=0; i<height; ++i)
	{
	  for (int j=0; j<height; ++j)
	    gpwd_w[j] = distribution(generator);

	  /*
	  for (int j=tdofsBdry[0].size(); j<height; ++j)
	    gpwd_w[j] = 0.0;
	  */
	  
	  const double nrm2w = InnerProduct(MPI_COMM_WORLD, gpwd_w, gpwd_w);

	  ProjectionMult(gpwd_w);
	  
	  Vector pw(gpwd_size);
	  
	  MPI_Bcast(gpwd_u.GetData(), 12*Nphi*numInterfaces, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  
	  if (m_rank == 0)
	    {
	      pw = gpwd_u;
	    }

	  // Lift
	  gpwd_w = 0.0;

	  std::vector<int> ossdif;
	  
	  for (int ri=0; ri<2; ++ri)  // loop over real and imaginary parts
	    {
	      ossdif.assign(numSubdomains, 0);
	      
	      for (int l=0; l<numInterfaces; ++l)
		{
		  int nghbSD[2];
		  nghbSD[0] = interfaceSDs[l] / numSubdomains;
		  nghbSD[1] = interfaceSDs[l] - (nghbSD[0] * numSubdomains);

		  for (int s=0; s<2; ++s)  // loop over sides of the interface
		    {
		      const int sd = nghbSD[s];
		      const int sdsize = (fespace[sd] == NULL) ? 0 : fespace[sd]->GetTrueVSize();

		      Vector sdtmp(sdsize);

		      const int osri = (ri == 0) ? 0 : (block_trueOffsets2[sd+1] - block_trueOffsets2[sd]) / 2;

		      // u^S component
		      if (InterfaceToSurfaceInjection[sd][l] != NULL)
			{
			  for (int b=0; b<Nphi; ++b)
			    {
			      if (ri == 0)
				InterfaceToSurfaceInjection[sd][l]->Mult(eTE_Re[l][b], sdtmp);
			      else
				InterfaceToSurfaceInjection[sd][l]->Mult(eTE_Im[l][b], sdtmp);
			  
			      if (tdofsBdryInjection[sd] != NULL)
				{
				  Vector uS(tdofsBdry[sd].size());
				  tdofsBdryInjection[sd]->MultTranspose(sdtmp, uS);

				  for (int q=0; q<uS.Size(); ++q)
				    {
				      //gpwd_w[block_trueOffsets2[sd] + osri + q] += gpwd_u[(12*Nphi*l) + (6*Nphi*s) + (3*Nphi*ri) + b] * uS[q];
				      gpwd_w[block_trueOffsets2[sd] + osri + q] += gpwd_u[(12*Nphi*l) + (2*Nphi*s) + (Nphi*ri) + b] * uS[q];
				      if (std::isnan(gpwd_w[block_trueOffsets2[sd] + osri + q]))
					cout << "nan" << endl;
				    }
				}
			    }
			}

		      // f component
		      for (int b=0; b<Nphi; ++b)
			{
			  //const double fcoef = gpwd_u[(12*Nphi*l) + (6*Nphi*s) + (3*Nphi*ri) + (Nphi) + b];
			  const double fcoef = gpwd_u[(12*Nphi*l) + (4*Nphi) + (2*Nphi*s) + (Nphi*ri) + b];
			  for (int q=0; q<ifNDtrue[l]; ++q)
			    {
			      if (ri == 0)
				gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q] += fcoef * bcr_Re[l][b][q];
			      else
				gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q] += fcoef * bcr_Im[l][b][q];

			      if (std::isnan(gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q]))
				cout << "nan" << endl;
			    }
			}
		      
		      ossdif[sd] += ifNDtrue[l];

		      // rho component
		      for (int b=0; b<Nphi; ++b)
			{
			  //const double rhocoef = gpwd_u[(12*Nphi*l) + (6*Nphi*s) + (3*Nphi*ri) + (2*Nphi) + b];
			  const double rhocoef = gpwd_u[(12*Nphi*l) + (8*Nphi) + (2*Nphi*s) + (Nphi*ri) + b];
			  for (int q=0; q<ifH1true[l]; ++q)
			    {
			      if (ri == 0)
				gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q] += rhocoef * rhoc_Re[l][b][q];
			      else
				gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q] += rhocoef * rhoc_Im[l][b][q];

			      if (std::isnan(gpwd_w[block_trueOffsets2[sd] + osri + tdofsBdry[sd].size() + ossdif[sd] + q]))
				cout << "nan" << endl;
			    }
			}
		      
		      ossdif[sd] += ifH1true[l];
		    }
		}
	    }
	  
	  const double nrm2lift = InnerProduct(MPI_COMM_WORLD, gpwd_w, gpwd_w);

	  ProjectionMult(gpwd_w);

	  if (m_rank == 0)
	    {
	      const double nrmP2u = gpwd_u.Norml2();
	      const double nrmPu = pw.Norml2();

	      pw -= gpwd_u;

	      const double nrmd = pw.Norml2();

	      cout << "ProjectionMult test " << i << " nrm diff " << nrmd << " relative to " << std::max(nrmPu, nrmP2u)
		   << ", rel " << nrmd / std::max(nrmPu, nrmP2u) << ", nrm2 lifted " << nrm2lift << endl;
	    }
	}
    }
  
  // Set global coarse matrix for GPWD.
  for (int j=0; j<gpwd_size; ++j)
    {
      if (m_rank == 0)
	{
	  gpwd_u = 0.0;
	  gpwd_u[j] = 1.0;
	}
      
      ProjectionMultTranspose(gpwd_v);
      globalOp->Mult(gpwd_v, gpwd_w);
      ProjectionMult(gpwd_w);

      if (m_rank == 0)
	{
	  for (int i=0; i<gpwd_size; ++i)
	    {
	      gpwd_cmat(i,j) = gpwd_u[i];
	    }
	}
    }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (m_rank == 0)
    {
      gpwd_cmat.Invert();
    }
#endif // GPWD  
}  // end of constructor for DDMInterfaceOperator

void DDMInterfaceOperator::GaussSeidelPreconditionerMult(const Vector & x, Vector & y) const
{
  Vector u, v, w;

  y = 0.0;
  
  for (int i=0; i<numSubdomains; ++i)
    {
      v.SetSize(block_trueOffsets2[i+1] - block_trueOffsets2[i]);
      w.SetSize(block_trueOffsets2[i+1] - block_trueOffsets2[i]);
      w = 0.0;
      
      for (int j=0; j<i; ++j)
	{
	  if (!globalInterfaceOp->IsZeroBlock(i,j))
	    {
	      u.SetSize(block_trueOffsets2[j+1] - block_trueOffsets2[j]);
	      for (int l=block_trueOffsets2[j]; l<block_trueOffsets2[j+1]; ++l)
		{
		  u[l-block_trueOffsets2[j]] = y[l];
		}
	      
	      globalInterfaceOp->GetBlock(i,j).Mult(u, v);
	      w.Add(globalInterfaceOp->GetBlockCoef(i,j), v);
	    }
	}

      // Now w equals the offdiagonal terms for block row i.
      MFEM_VERIFY((globalSubdomainOp->IsZeroBlock(i,i) == 0) == (block_trueOffsets2[i] == block_trueOffsets2[i+1]), "");
      
      if (!globalSubdomainOp->IsZeroBlock(i,i))
	{
	  globalSubdomainOp->GetBlock(i,i).Mult(w, v);

	  for (int l=block_trueOffsets2[i]; l<block_trueOffsets2[i+1]; ++l)
	    {
	      w[l-block_trueOffsets2[i]] = x[l];
	    }

	  w -= v;  // x_i - offdiagonal terms for block row i.
      
	  for (int l=block_trueOffsets2[i]; l<block_trueOffsets2[i+1]; ++l)
	    {
	      y[l] = w[l-block_trueOffsets2[i]];  // diagonal block is identity
	    }
	}
    }
}

#ifdef GPWD
void DDMInterfaceOperator::GPWD_PreconditionerMult(const Vector & x, Vector & y)
{
  ProjectionMult(x);

  if (m_rank == 0)
    {
      gpwd_cmat.Mult(gpwd_u, gpwd_z);
      gpwd_u = gpwd_z;
    }
  
  ProjectionMultTranspose(gpwd_v);
  globalOp->Mult(gpwd_v, gpwd_w);

  y = x;
  y -= gpwd_w;
}
#endif // GPWD  

void DDMInterfaceOperator::PolyPreconditionerMult(const Vector & x, Vector & y)
{
  y = x;  // identity term
  
  globalOffdiag->Mult(x, prec_z);
  y -= prec_z;  // y -= off diagonal terms times x

  globalOffdiag->Mult(prec_z, prec_w);
  y += prec_w;  // y += off diagonal terms squared times x
}

void DDMInterfaceOperator::GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobalRe, Vector & sourceGlobalIm,
					    Vector & sourceReduced, std::vector<int> const& sdOrder) const
{  
  //Vector sourceSD;
  Vector wSD, vSD;

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

  MFEM_VERIFY(numSubdomains == sdOrder.size(), "");

  MPI_Barrier(MPI_COMM_WORLD);
  
  //for (int m=0; m<numSubdomains; ++m)
  for (std::vector<int>::const_iterator it = sdOrder.begin(); it != sdOrder.end(); ++it)
    {
      const int m = *it;

      //cout << rank << ": GetReducedSource " << m << " nonempty " << sd_nonempty[m] << endl;

      ySD[m] = NULL;
      rhsSD[m] = NULL;
	
      //if (pmeshSD[m] != NULL)
      if (sd_nonempty[m])
	{
	  const int sdsize = (fespace[m] == NULL) ? 0 : fespace[m]->GetTrueVSize();
	  //sourceSD.SetSize(fespace[m]->GetTrueVSize());
	  //sourceSD.SetSize(sdsize);

	  // Map from the global u to [u_m f_m \rho_m], with blocks corresponding to subdomains, and f_m = 0, \rho_m = 0.
#ifdef DDMCOMPLEX
	  // In the complex case, we map from the global u^{Re} and u^{Im} to [u_m^{Re} f_m^{Re} \rho_m^{Re} u_m^{Im} f_m^{Im} \rho_m^{Im}].

	  MFEM_VERIFY(invAsdComplex[m]->Height() == block_ComplexOffsetsSD[m][2], "");
	  MFEM_VERIFY(invAsdComplex[m]->Height() == 2*block_ComplexOffsetsSD[m][1], "");
	  MFEM_VERIFY(invAsdComplex[m]->Height() > 0, "");

	  cout << rank << ": Setting real subdomain DOFs, sd " << m << ", size " << sdsize << endl;
	    
	  ySD[m] = new Vector(invAsdComplex[m]->Width());  // Size of [u_m f_m \rho_m], real and imaginary parts
	  rhsSD[m] = new Vector(invAsdComplex[m]->Height());  // Size of [u_m f_m \rho_m], real and imaginary parts

	  wSD.SetSize(invAsdComplex[m]->Height());	  
	  wSD = 0.0;

	  /*
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
	  */
	  //const bool explicitRHS = false;
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

if (sdsize > 0)
  {
    //MFEM_VERIFY(sourceSD.Size() == srcSD[m]->Size(), "");
#ifdef SDFOSLS
    const int nSD = cfosls[m]->rhs_E.Size();
    MFEM_VERIFY(4*nSD < ySD[m]->Size(), "");
    const int yImOS = ySD[m]->Size() / 2;
    MFEM_VERIFY(2*yImOS == ySD[m]->Size(), "");

    auto rhs_E_host = cfosls[m]->rhs_E.HostRead();
#endif

  for (int i=0; i<sdsize; ++i)
    {
      // Set the u_m block of ySD, real part
#ifdef SDFOSLS
      //(*(ySD[m]))[i] = cfosls[m]->rhs_E[i];
      (*(ySD[m]))[i] = rhs_E_host[i];
#ifdef ZERO_ORDER_FOSLS_COMPLEX
      //(*(ySD[m]))[yImOS + i] = -cfosls[m]->rhs_E[i];
      (*(ySD[m]))[yImOS + nSD + i] = -cfosls[m]->rhs_E[i];
#endif
#ifdef IFFOSLS
#ifdef IFFOSLS_ESS
      //(*(ySD[m]))[yImOS + i] = cfosls[m]->rhs_E[i];
#else
      //(*(ySD[m]))[yImOS + i] = -cfosls[m]->rhs_E[i];
#endif
#endif
#else
      (*(ySD[m]))[i] = (*(srcSD[m]))[i];
#endif
      //(*(ySD[m]))[i] = sourceSD[i];
    }
  }

  /*
  cout << rank << ": Setting imaginary subdomain DOFs, sd " << m << endl;
  if (sdsize > 0)
    {
      SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, sourceGlobalIm, sourceSD);
    }
    cout << rank << ": Done setting imaginary subdomain DOFs, sd " << m << endl;

    if (explicitRHS)
      sourceSD = 0.0;
  */

    /* // TODO: set imaginary part!
    for (int i=0; i<sourceSD.Size(); ++i)
      (*(ySD[m]))[block_ComplexOffsetsSD[m][1] + i] = sourceSD[i];  // Set the u_m block of ySD, imaginary part
    */
	} // if pmeshSD[m] != null

      //MPI_Barrier(MPI_COMM_WORLD);  // TODO: remove this barrier. The subdomain computations should be parallel.
      
      if (sd_nonempty[m])
      {
	cout << rank << ": ySD norm: " << ySD[m]->Norml2() << ", wSD norm " << wSD.Norml2() << " (sd " << m << ")" << endl;

	cout << rank << ": Applying invAsdComplex[" << m << "]" << endl;

	//if (m == 0) HypreAsdComplex[m]->Print("HypreAsdComplexCheck");

	if (invAsdComplex[m] != NULL)
	  {
	    invAsdComplex[m]->Mult(*(ySD[m]), wSD);
	    //AsdComplex[m]->Mult(*(ySD[m]), wSD);

	    /*
	    {
	      // Debugging: test residual of solution
	      Vector rSD(wSD.Size());
	      //AsdComplex[m]->Mult(wSD, rSD);
	      HypreAsdComplex[m]->Mult(wSD, rSD);
	      
	      rSD -= (*(ySD[m]));

	      const double rnrm = rSD.Norml2();
	      const double ynrm = ySD[m]->Norml2();

	      cout << rank << ": STRUMPACK residual for y on SD " << m << " absolute: " << rnrm << ", relative " << rnrm / ynrm << endl;
	    }
	    */
	  }
	
	cout << rank << ": Done applying invAsdComplex[" << m << "], norm of wSD: " << wSD.Norml2() << endl;

	// Extract only the [u_m^s, f_m, \rho_m] entries, real and imaginary parts.
	vSD.SetSize(block_trueOffsets2[m+1] - block_trueOffsets2[m]);
	injComplexSD[m]->MultTranspose(wSD, vSD);

	sourceReduced.AddOffset(1.0, vSD, block_trueOffsets2[m]);
	/*
	for (int i=0; i<vSD.Size(); ++i)
	  sourceReduced[block_trueOffsets2[m] + i] = vSD[i];
	*/
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

#ifdef DEBUG_RECONSTRUCTION
dsourcered.SetSize(sourceReduced.Size());
dsourcered = sourceReduced;
#endif
}

void DDMInterfaceOperator::CheckContinuityOfUS(const Vector & solReduced, const bool imag)
{
  for (int i=0; i<numInterfaces; ++i)
    {
      const int ili = (*interfaceLocalIndex)[i];

      int sd0 = -1;
      int sd1 = -1;

      if (ili >= 0)
	{
	  MFEM_VERIFY(globalInterfaceIndex[ili] == i, "");
	  sd0 = (*localInterfaces)[ili].FirstSubdomain();
	  sd1 = (*localInterfaces)[ili].SecondSubdomain();
	}
      else
	continue;
      
      //const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
      //const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

      cout << m_rank << ": CheckContinuityOfUS for interface " << i << " sd " << sd0 << ", " << sd1 << endl;
	
      MFEM_VERIFY(0 <= sd0 && sd0 < sd1, "");

      Vector uS0(tdofsBdry[sd0].size());
      Vector uS1(tdofsBdry[sd1].size());

      const int offsetImag0 = imag ? (block_trueOffsets2[sd0+1] - block_trueOffsets2[sd0]) / 2 : 0;
      const int offsetImag1 = imag ? (block_trueOffsets2[sd1+1] - block_trueOffsets2[sd1]) / 2 : 0;
      
      for (int j=0; j<tdofsBdry[sd0].size(); ++j)
	uS0[j] = solReduced[block_trueOffsets2[sd0] + offsetImag0 + j];
      
      for (int j=0; j<tdofsBdry[sd1].size(); ++j)
	uS1[j] = solReduced[block_trueOffsets2[sd1] + offsetImag1 + j];

      Vector u0(fespace[sd0]->GetTrueVSize());
      Vector u1(fespace[sd1]->GetTrueVSize());
      
      tdofsBdryInjection[sd0]->Mult(uS0, u0);
      tdofsBdryInjection[sd1]->Mult(uS1, u1);

      Vector uI0(ifNDtrue[i]);
      Vector uI1(ifNDtrue[i]);

      Vector f0(ifNDtrue[i]);
      Vector f1(ifNDtrue[i]);

      int osf0 = tdofsBdry[sd0].size();
      int osf1 = tdofsBdry[sd1].size();
      
      for (int j=0; j<subdomainLocalInterfaces[sd0].size(); ++j)
	{
	  if (subdomainLocalInterfaces[sd0][j] == i)
	    break;
	  else
	    osf0 += ifNDtrue[subdomainLocalInterfaces[sd0][j]] + ifH1true[subdomainLocalInterfaces[sd0][j]];
	}
      
      for (int j=0; j<subdomainLocalInterfaces[sd1].size(); ++j)
	{
	  if (subdomainLocalInterfaces[sd1][j] == i)
	    break;
	  else
	    osf1 += ifNDtrue[subdomainLocalInterfaces[sd1][j]] + ifH1true[subdomainLocalInterfaces[sd1][j]];
	}
      
      for (int j=0; j<ifNDtrue[i]; ++j)
	{
	  f0[j] = solReduced[block_trueOffsets2[sd0] + offsetImag0 + osf0 + j];
	  f1[j] = solReduced[block_trueOffsets2[sd1] + offsetImag1 + osf1 + j];
	}
      
      InterfaceToSurfaceInjection[sd0][i]->MultTranspose(u0, uI0);
      InterfaceToSurfaceInjection[sd1][i]->MultTranspose(u1, uI1);

      const double nrm0 = uI0.Norml2();
      const double nrm1 = uI1.Norml2();

      const double fnrm0 = f0.Norml2();
      const double fnrm1 = f1.Norml2();

      const double nrm = std::max(nrm0, nrm1);
      const double fnrm = std::max(fnrm0, fnrm1);

      uI0 -= uI1;
      const double err = uI0.Norml2();

      f0 += f1;
      
      cout << "Size of u discontinuity for ";

      if (imag)
	cout << "imaginary";
      else
	cout << "real";

      cout << " part: " << err << " relative to " << nrm << "; for f: " << f0.Norml2() << " relative to " << fnrm << endl;
    }
}

void DDMInterfaceOperator::RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced,
						 Vector const& femSol, Vector & solDomain)
{
  MFEM_VERIFY(solReduced.Size() == this->Height(), "");
  Vector w(globalInterfaceOp->Height());
  Vector v, u, uSD, wSD, uSD2;

  MFEM_VERIFY(this->Height() == block_trueOffsets2[numSubdomains], "");

#ifdef DEBUG_RECONSTRUCTION
  dsolred.SetSize(this->Height());
  dsolred = 0.0;
  
  dsolSD = new Vector*[numSubdomains];

  dcopysol.SetSize(this->Height());
  dcopysol = solReduced;

  double maxdsrcIm = 0.0;

  CheckContinuityOfUS(solReduced, false);  // real part
  CheckContinuityOfUS(solReduced, true);   // imaginary part
#endif
  
  // Assuming GetReducedSource has been called, ySD stores y_m (with full components [u_m f_m \rho_m]) for each subdomain m.

  // globalInterfaceOp represents the off-diagonal blocks C_{ij} R_j^T
  globalInterfaceOp->Mult(solReduced, w);
  auto w_host = w.HostRead();

  cout << m_rank << ": global w norm " << w.Norml2() << endl;
  
  // Now w = \sum_{j\neq i} C_{ij} R_j^T \overbar{u}_j

  Vector domainError(3), eSD(3), maxeSD(3);
  domainError = 0.0;

  std::vector<double> eSDall;
  eSDall.assign(3*numSubdomains, 0.0);
  
#ifdef TESTFEMSOL
  std::vector<Vector> uReSD(numSubdomains);
  Vector uReReduced(this->Height());
  Vector CuReReduced(this->Height());
  Vector CfReReduced(this->Height());
  
  uReReduced = 0.0;
  CfReReduced = 0.0;
#endif

#ifdef SDFOSLS
  MFEM_VERIFY(w.Size() == block_trueOffsets_FOSLS2[numSubdomains], "");
#endif

  for (int m=0; m<numSubdomains; ++m)
    {
      eSD = 0.0;

#ifdef DEBUG_RECONSTRUCTION      
      dsolSD[m] = NULL;
#endif
      
      if (ySD[m] != NULL)
	{
#ifdef DDMCOMPLEX
	  //MFEM_VERIFY(ySD[m]->Size() == block_trueOffsets2[m+1] - block_trueOffsets2[m], "");  wrong
	  //MFEM_VERIFY(ySD[m]->Size() > block_trueOffsets2[m+1] - block_trueOffsets2[m], "");
	  MFEM_VERIFY(ySD[m]->Size() == invAsdComplex[m]->Width(), "");
	  MFEM_VERIFY(invAsdComplex[m] != NULL, "");

#ifdef DEBUG_RECONSTRUCTION
	  dsolSD[m] = new Vector(invAsdComplex[m]->Height());
#endif
	  
	  // Put the [u_m^s, f_m, \rho_m] entries of w (real and imaginary parts) into wSD.
#ifdef SDFOSLS
	  wSD.SetSize(block_trueOffsets_FOSLS2[m+1] - block_trueOffsets_FOSLS2[m]);

	  /*
	  for (int i=0; i<block_trueOffsets_FOSLS2[m+1] - block_trueOffsets_FOSLS2[m]; ++i)
	    wSD[i] = w[block_trueOffsets_FOSLS2[m] + i];
	  */
	  wSD.SetOffset(1.0, w, block_trueOffsets_FOSLS2[m], 0, block_trueOffsets_FOSLS2[m+1] - block_trueOffsets_FOSLS2[m]);

	  uSD2.SetSize(invAsdComplex[m]->Width());
#else
	  wSD.SetSize(block_trueOffsets2[m+1] - block_trueOffsets2[m]);

	  for (int i=0; i<block_trueOffsets2[m+1] - block_trueOffsets2[m]; ++i)
	    wSD[i] = w_host[block_trueOffsets2[m] + i];
#endif

	  uSD.SetSize(invAsdComplex[m]->Height());

	  {
	    const int imsize = wSD.Size() / 2;
	    Vector wIm(imsize);

	    wIm.SetOffset(1.0, wSD, imsize, 0, imsize);
	    for (int i=0; i<imsize; ++i)
	      {
		//wIm[i] = wSD[imsize + i];
		
#ifdef DEBUG_RECONSTRUCTION		
		maxdsrcIm = std::max(maxdsrcIm, fabs(dsourcered[block_trueOffsets2[m] + imsize + i]));
		if (fabs(dsourcered[block_trueOffsets2[m] + imsize + i]) > 0.616)
		  cout << "dsourcered[" << block_trueOffsets2[m] + imsize + i << "] = "
		       << dsourcered[block_trueOffsets2[m] + imsize + i] << endl;
#endif
	      }

	    const double normIm = wIm.Norml2();
	    const double normIm2 = normIm*normIm;

	    const double normW = wSD.Norml2();
	    const double norm2 = normW*normW;

	    //for (int i=0; i<imsize; ++i)
	    //wIm[i] = solReduced[block_trueOffsets2[m] + i];

	    wIm.SetOffset(1.0, solReduced, block_trueOffsets2[m], 0, imsize);

	    const double normSolRe = wIm.Norml2();
	    const double normSolRe2 = normSolRe*normSolRe;

	    //for (int i=0; i<imsize; ++i)
	    //wIm[i] = solReduced[block_trueOffsets2[m] + imsize + i];

	    //wIm.SetOffset(1.0, solReduced, block_trueOffsets2[m] + imsize, 0, imsize);
	    // TODO: normSolIm is wrong, but it doesn't seem to be used.
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

#ifdef SDFOSLS
	  injComplexSD2[m]->Mult(wSD, uSD2);
#else
	  injComplexSD[m]->Mult(wSD, uSD);
#endif
	  
	  *(rhsSD[m]) = *(ySD[m]);
	  
#ifdef SDFOSLS
	  cout << m_rank << ": uSD2[" << m  << "] norm " << uSD2.Norml2() << endl;
	  *(ySD[m]) -= uSD2;
#else
	  *(ySD[m]) -= uSD;
#endif

	  /*
#ifdef TEST_DECOUPLED_FOSLS_PROJ  // TODO: remove this
	  {
	    MFEM_VERIFY(ySD[m]->Size() > cfosls[m]->rhs_E.Size(), "");

	    const int nSD = cfosls[m]->rhs_E.Size();
	    
	    (*(ySD[m])) = 0.0;
	    for (int i=0; i<nSD; ++i)
	      (*(ySD[m]))[i] = cfosls[m]->rhs_E[i];

	    Vector cfrhs(4*nSD);
	    Vector cfsol(4*nSD);

	    cfrhs = 0.0;
	    cfsol = 0.0;
	    for (int i=0; i<nSD; ++i)
	      cfrhs[i] = cfosls[m]->rhs_E[i];
	    
	    cfosls[m]->Mult(cfrhs, cfsol);

	    uSD = 0.0;
	    for (int i=0; i<nSD; ++i)
	      uSD[i] = cfsol[i];
	    
	    PrintSubdomainError(m, uSD, eSD, *(rhsSD[m]));
	  }
#else
	  */
	  uSD = 0.0;
	  if (invAsdComplex[m] != NULL)
	    invAsdComplex[m]->Mult(*(ySD[m]), uSD);

	  PrintSubdomainError(m, uSD, eSD, *(rhsSD[m]));
	  //#endif
#ifndef SERIAL_INTERFACES	  
	  PrintInterfaceError(m, uSD);
#endif
	  
	  if (fespace[m] != NULL)
	    {
	      if (fespace[m]->GetTrueVSize() > 0)
		{
		  uSD.SetSize(fespace[m]->GetTrueVSize());
		  SetDomainDofsFromSubdomainDofs(fespace[m], fespaceGlobal, uSD, solDomain);
		}
	    }
	  
#ifdef TESTFEMSOL
	  {
	    const int tds = tdofsBdry[m].size();
	    
	    Vector femSolSD(fespace[m]->GetTrueVSize());
	    SetSubdomainDofsFromDomainDofs(fespace[m], fespaceGlobal, femSol, femSolSD);
	    Vector eSDtmp(3);
	    PrintSubdomainInteriorError(m, femSolSD);

	    uReSD[m].SetSize(AsdComplex[m]->Height());
	    uReSD[m] = 0.0;
	    	    
	    for (int i=0; i<fespace[m]->GetTrueVSize(); ++i)
	      uReSD[m][i] = femSolSD[i];  // Set the real part of u

	    // Set real part of f on each interface of this subdomain
	    const int osIm = JustComputeF(m, femSolSD, *(rhsSD[m]), uReSD[m], CfReReduced);

	    MFEM_VERIFY(2*osIm == AsdComplex[m]->Height(), "");

	    Vector r_uReSD(block_trueOffsets2[m+1] - block_trueOffsets2[m]);
	    injComplexSD[m]->MultTranspose(uReSD[m], r_uReSD);

	    for (int i=0; i<r_uReSD.Size(); ++i)
	      uReReduced[block_trueOffsets2[m] + i] = r_uReSD[i];
	  }
#endif
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

      for (int i=0; i<3; ++i)
	eSDall[(3*m) + i] = eSD[i];
    } // loop (m) over subdomains

  for (int m=0; m<numSubdomains; ++m)
    {
      for (int i=0; i<3; ++i)
	eSD[i] = eSDall[(3*m) + i];
      
      maxeSD = 0.0;
      MPI_Allreduce(eSD.GetData(), maxeSD.GetData(), 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (m_rank == 0)
	{
	  for (int i=0; i<3; ++i)
	    domainError[i] += maxeSD[i] * maxeSD[i];
	}
    }
  
#ifdef TESTFEMSOL
#ifdef DEBUG_RECONSTRUCTION
  {
    hypre_CSRMatrix* mcsr = GetHypreParMatrixData(*(ifNDmass[0]));
    cout << "mcsr start " << mcsr->i[5] << " end " << mcsr->i[6] << endl;
  }
  
  CuReReduced = 0.0;
  globalInterfaceOp->Mult(uReReduced, CuReReduced);

  {
    cout << "Maximum imaginary entry of reduced source " << maxdsrcIm << endl;
    
    Vector DDuReReduced(this->Height());
    this->Mult(uReReduced, DDuReReduced);
    
    const double ddnrm = DDuReReduced.Norml2();
    const double srcnrm = dsourcered.Norml2();

    DDuReReduced -= dsourcered;

    int imax = 0;
    double dmax = fabs(DDuReReduced[0]);
    for (int i=1; i<DDuReReduced.Size(); ++i)
      {
	const double d = fabs(DDuReReduced[i]);
	if (d > dmax)
	  {
	    dmax = d;
	    imax = i;
	  }
      }
    
    cout << "Reduced residual norm " << DDuReReduced.Norml2() << " relative to " << std::max(ddnrm, srcnrm) << ", max entry "
	 << dmax << " at " << imax << endl;
  }
  
  Vector CuReReduced_Re(this->Height());
  CuReReduced_Re = CuReReduced;
  
  // Zero out the imaginary parts of CuReReduced_Re, which come from the real parts of uReReduced multiplied by imaginary operators.
  {
    for (int m=0; m<numSubdomains; ++m)
      {
	const int osIm = (block_trueOffsets2[m+1] - block_trueOffsets2[m]) / 2;
	for (int i=0; i<osIm; ++i)
	  CuReReduced_Re[block_trueOffsets2[m] + osIm + i] = 0.0;
      }
  }
  
  const double Cfnrm = CfReReduced.Norml2();
  const double Cunrm = CuReReduced_Re.Norml2();

  CfReReduced -= CuReReduced_Re;

  cout << m_rank << ": CfReReduced error " << CfReReduced.Norml2() << " relative to " << std::max(Cfnrm, Cunrm) << endl;
  
  for (int m=0; m<numSubdomains; ++m)
    {
      Vector CuReSD(AsdComplex[m]->Height());
      Vector r_CuReSD(block_trueOffsets2[m+1] - block_trueOffsets2[m]);

      for (int i=0; i<r_CuReSD.Size(); ++i)
	r_CuReSD[i] = CuReReduced[block_trueOffsets2[m] + i];
      
      injComplexSD[m]->Mult(r_CuReSD, CuReSD);
      
      Vector AuReSD(AsdComplex[m]->Height());
      AuReSD = 0.0;

      AsdComplex[m]->Mult(uReSD[m], AuReSD);

      AuReSD += CuReSD;
      
      Vector resSD(fespace[m]->GetTrueVSize());

      resSD = *(rhsSD[m]);

      for (int i=0; i<fespace[m]->GetTrueVSize(); ++i)
	resSD[i] -= AuReSD[i];

      const double diffRe = resSD.Norml2();

      const int osIm = AsdComplex[m]->Height() / 2;
      
      for (int i=0; i<fespace[m]->GetTrueVSize(); ++i)
	resSD[i] = AuReSD[osIm + i];

      const double diffIm = resSD.Norml2();

      cout << m_rank << ": FEMSOL res SD " << m << " Re " << diffRe << " relative to " << rhsSD[m]->Norml2()
	   << ", Im " << diffIm << endl;
    }
#endif
#endif
  
  if (m_rank == 0)
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

#ifdef DEBUG_RECONSTRUCTION
  Vector Asol(dsolred.Size());
  this->Mult(dsolred, Asol);
  //this->Mult(solReduced, Asol);

  const double nrmsrc = dsourcered.Norml2();
  const double nrmas = Asol.Norml2();
  cout << "Norm of source " << nrmsrc << ", norm of A*sol " << nrmas << endl;

  int os = 0;
  for (int sd=0; sd<numSubdomains; ++sd)
    {
      // Real part
      double nrm = 0.0;
      double nrmerr = 0.0;
      for (int i=0; i<tdofsBdry[sd].size(); ++i)
	{
	  nrm += dsourcered[i]*dsourcered[i];
	  nrmerr += (dsourcered[i] - Asol[i]) * (dsourcered[i] - Asol[i]);
	}

      cout << "REDUCED error on subdomain " << sd << ", real part " << sqrt(nrmerr) << " relative to " << sqrt(nrm) << endl;
      
      os += tdofsBdry[sd].size();

      for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
	{
	  const int interfaceIndex = subdomainLocalInterfaces[sd][i];
	  if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	    {
	      nrm = 0.0;
	      nrmerr = 0.0;
	      
	      for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
		{
		  nrm += dsourcered[os+j]*dsourcered[os+j];
		  nrmerr += (dsourcered[os+j] - Asol[os+j]) * (dsourcered[os+j] - Asol[os+j]);
		}

	      cout << "REDUCED error on subdomain " << sd << " interface " << interfaceIndex << ", real part " << sqrt(nrmerr)
		   << " relative to " << sqrt(nrm) << endl;

	      os += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	    }
	}

      // Imaginary part
      nrm = 0.0;
      nrmerr = 0.0;
      for (int i=0; i<tdofsBdry[sd].size(); ++i)
	{
	  nrm += dsourcered[i]*dsourcered[i];
	  nrmerr += (dsourcered[i] - Asol[i]) * (dsourcered[i] - Asol[i]);
	}

      cout << "REDUCED error on subdomain " << sd << ", imag part " << sqrt(nrmerr) << " relative to " << sqrt(nrm) << endl;
      
      os += tdofsBdry[sd].size();

      for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
	{
	  const int interfaceIndex = subdomainLocalInterfaces[sd][i];
	  if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	    {
	      nrm = 0.0;
	      nrmerr = 0.0;
	      
	      for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
		{
		  nrm += dsourcered[os+j]*dsourcered[os+j];
		  nrmerr += (dsourcered[os+j] - Asol[os+j]) * (dsourcered[os+j] - Asol[os+j]);
		}

	      cout << "REDUCED error on subdomain " << sd << " interface " << interfaceIndex << ", imag part " << sqrt(nrmerr)
		   << " relative to " << sqrt(nrm) << endl;

	      os += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	    }
	}
    }
  
#endif
}

void DDMInterfaceOperator::CreateInterfaceMatrices(const int interfaceIndex, const bool fullAssembly)
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
  
#ifdef SERIAL_INTERFACES
  bf_ifNDmass[interfaceIndex] = new BilinearForm(ifespace[interfaceIndex]);
  bf_ifNDcurlcurl[interfaceIndex] = new BilinearForm(ifespace[interfaceIndex]);
  bf_ifH1mass[interfaceIndex] = new BilinearForm(iH1fespace[interfaceIndex]);
  bf_ifNDH1grad[interfaceIndex] = new MixedBilinearForm(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);

#ifdef SDFOSLS
  bf_ifNDtang[interfaceIndex] = new BilinearForm(ifespace[interfaceIndex]);
#endif
  
#ifdef SD_ITERATIVE_GMG_PA
  if (!fullAssembly)
    {
      bf_ifNDmass[interfaceIndex]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      bf_ifNDcurlcurl[interfaceIndex]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      bf_ifH1mass[interfaceIndex]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      bf_ifNDH1grad[interfaceIndex]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
#else
  /*
  BilinearForm *NDmass = new BilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  BilinearForm *NDcurlcurl = new BilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  BilinearForm *H1mass = new BilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  MixedBilinearForm *NDH1grad = new MixedBilinearForm(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  */
  BilinearForm *ND_FS = new BilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.  
  BilinearForm *H1stiff = new BilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
#endif
#else
  ParBilinearForm *NDmass = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ParBilinearForm *NDcurlcurl = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ParBilinearForm *ND_FS = new ParBilinearForm(ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ParBilinearForm *H1mass = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ParBilinearForm *H1stiff = new ParBilinearForm(iH1fespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.
  ParMixedBilinearForm *NDH1grad = new ParMixedBilinearForm(iH1fespace[interfaceIndex], ifespace[interfaceIndex]);  // TODO: make this a class member and delete at the end.  
#endif

  // TODO: use variable wavenumber as coefficient?

  bf_ifNDmass[interfaceIndex]->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  bf_ifNDmass[interfaceIndex]->Assemble();

#ifdef SDFOSLS
  bf_ifNDtang[interfaceIndex]->AddDomainIntegrator(new VectorFEBoundaryTangentIntegrator(1.0));
  bf_ifNDtang[interfaceIndex]->Assemble();
#endif
  
#ifdef ZERO_IFND_BC  // TODO: fix this so that only the exterior boundary gets essential BC.
  {
#ifdef SERIAL_INTERFACES
#ifdef USE_2DINTERFACE
    ND_ess_bdr.SetSize(smeshIF2D[interfaceIndex]->bdr_attributes.Max());
#else
    ND_ess_bdr.SetSize(smeshIF[interfaceIndex]->bdr_attributes.Max());
#endif
#else
    ND_ess_bdr.SetSize(pmeshIF[interfaceIndex]->bdr_attributes.Max());
#endif
    
    ND_ess_bdr = 1;

    ifespace[interfaceIndex]->GetEssentialTrueDofs(ND_ess_bdr, ND_ess_tdof_list);
  }
#endif

  bf_ifNDcurlcurl[interfaceIndex]->AddDomainIntegrator(new CurlCurlIntegrator(one));
  bf_ifNDcurlcurl[interfaceIndex]->Assemble();

#ifndef SD_ITERATIVE_GMG_PA
  ND_FS->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  ND_FS->AddDomainIntegrator(new CurlCurlIntegrator(one));
  ND_FS->Assemble();
#endif

  //ifND_FS[interfaceIndex] = new HypreParMatrix();

#ifdef SERIAL_INTERFACES
  {
#ifdef SD_ITERATIVE_GMG_PA
    if (!fullAssembly)
      {
	GridFunction xif(ifespace[interfaceIndex]);

	{
	  Vector zeroVec(3);
	  zeroVec = 0.0;
	  VectorConstantCoefficient vzero(zeroVec);
	  xif.ProjectCoefficient(vzero);
	}
      
	LinearForm b(ifespace[interfaceIndex]);
	b.Assemble();
	Vector ifB, ifX;

	if_ND_ess_tdof_list[interfaceIndex] = ND_ess_tdof_list;  // deep copy

	/*
	if (m_rank == 0 && interfaceIndex == 0)
	  {
	    for (int j=0; j<if_ND_ess_tdof_list[interfaceIndex].Size(); ++j)
	      cout << m_rank << ": ifNDesstdof " << j << " = " << if_ND_ess_tdof_list[interfaceIndex][j] << endl;
	  }
	*/
	
	bf_ifNDmass[interfaceIndex]->FormLinearSystem(if_ND_ess_tdof_list[interfaceIndex], xif, b, oppa_ifNDmass[interfaceIndex], ifX, ifB);
	bf_ifNDcurlcurl[interfaceIndex]->FormLinearSystem(if_ND_ess_tdof_list[interfaceIndex], xif, b, oppa_ifNDcurlcurl[interfaceIndex], ifX, ifB);
      }
    else
#endif
      {
	ifNDmassSp[interfaceIndex] = new SparseMatrix();
	ifNDcurlcurlSp[interfaceIndex] = new SparseMatrix();

	bf_ifNDmass[interfaceIndex]->FormSystemMatrix(ND_ess_tdof_list, *(ifNDmassSp[interfaceIndex]));
	bf_ifNDcurlcurl[interfaceIndex]->FormSystemMatrix(ND_ess_tdof_list, *(ifNDcurlcurlSp[interfaceIndex]));
	//ND_FS->FormSystemMatrix(ND_ess_tdof_list, *(ifND_FS[interfaceIndex]));

#ifdef SDFOSLS
	ifNDtangSp[interfaceIndex] = new SparseMatrix();
#ifdef ZERO_IFND_BC
	bf_ifNDtang[interfaceIndex]->FormSystemMatrix(ND_ess_tdof_list, *(ifNDtangSp[interfaceIndex]));  // TODO: fix the BC.
#else
	Array<int> ess_tdof_list_empty;
	bf_ifNDtang[interfaceIndex]->FormSystemMatrix(ess_tdof_list_empty, *(ifNDtangSp[interfaceIndex]));
#endif	
#endif
	
	// Since BilinearForm uses DIAG_KEEP as the DiagonalPolicy, let's change it to DIAG_ONE manually here.
	for (int i = 0; i < ND_ess_tdof_list.Size(); i++)
	  {
	    const int sdof = ND_ess_tdof_list[i];
	    const int dof = (sdof >= 0) ? sdof : -1-sdof;
	    (*(ifNDmassSp[interfaceIndex]))(dof,dof) = 1.0;
	    (*(ifNDcurlcurlSp[interfaceIndex]))(dof,dof) = 1.0;
	  }
      }
    
#ifdef IF_ITERATIVE
    if (!fullAssembly)
      {
	CGSolver *cg = new CGSolver();
    
#ifdef SD_ITERATIVE_GMG_PA
	cg->SetOperator(*(oppa_ifNDmass[interfaceIndex].Ptr()));
#else
	cg->SetOperator(*(ifNDmassSp[interfaceIndex]));
#endif
	cg->SetRelTol(1e-12);
	cg->SetMaxIter(100);
	cg->SetPrintLevel(-1);
	cg->iterative_mode = false;
    
	ifNDmassInv[interfaceIndex] = cg;
      }
#else
#ifdef SDFOSLS_PA
    ifNDmassInv[interfaceIndex] = NULL;
#else
    UMFPackSolver *massInv = new UMFPackSolver();
    massInv->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    massInv->SetOperator(*(ifNDmassSp[interfaceIndex]));
    ifNDmassInv[interfaceIndex] = massInv;
#endif
#endif

    ifH1massSp[interfaceIndex] = new SparseMatrix();
  }
#else  // not SERIAL_INTERFACES
  ifNDmass[interfaceIndex] = new HypreParMatrix();
  ifNDcurlcurl[interfaceIndex] = new HypreParMatrix();

  NDmass->FormSystemMatrix(ND_ess_tdof_list, *(ifNDmass[interfaceIndex]));
  NDcurlcurl->FormSystemMatrix(ND_ess_tdof_list, *(ifNDcurlcurl[interfaceIndex]));
  //ND_FS->FormSystemMatrix(ND_ess_tdof_list, *(ifND_FS[interfaceIndex]));

  Operator *ifNDmassStrumpack = new STRUMPACKRowLocMatrix(*(ifNDmass[interfaceIndex]));
  ifNDmassInv[interfaceIndex] = CreateStrumpackSolver(ifNDmassStrumpack, ifespace[interfaceIndex]->GetComm());

  ifH1mass[interfaceIndex] = new HypreParMatrix();
#endif
  
  cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize() << ", mass height "
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
       << bf_ifNDmass[interfaceIndex]->Height() << ", width " << bf_ifNDmass[interfaceIndex]->Width() << ", ND V size "
#else
       << ifNDmassSp[interfaceIndex]->Height() << ", width " << ifNDmassSp[interfaceIndex]->Width() << ", ND V size "
#endif
#else
       << ifNDmass[interfaceIndex]->Height() << ", width " << ifNDmass[interfaceIndex]->Width() << ", ND V size "
#endif
       << ifespace[interfaceIndex]->GetVSize() << endl;
  
  // H^1 interface operators

  bf_ifH1mass[interfaceIndex]->AddDomainIntegrator(new MassIntegrator(one));
  bf_ifH1mass[interfaceIndex]->Assemble();
  
#ifdef ZERO_RHO_BC
  {
#ifdef SERIAL_INTERFACES
#ifdef USE_2DINTERFACE
    H1_ess_bdr.SetSize(smeshIF2D[interfaceIndex]->bdr_attributes.Max());
#else
    H1_ess_bdr.SetSize(smeshIF[interfaceIndex]->bdr_attributes.Max());
#endif
#else
    H1_ess_bdr.SetSize(pmeshIF[interfaceIndex]->bdr_attributes.Max());
#endif
    
    H1_ess_bdr = 1;

    iH1fespace[interfaceIndex]->GetEssentialTrueDofs(H1_ess_bdr, H1_ess_tdof_list);
  }    
#endif
  
  {
    /*
      Operator *ifH1massStrumpack = new STRUMPACKRowLocMatrix(*(ifH1mass[interfaceIndex]));
      ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massStrumpack, iH1fespace[interfaceIndex]->GetComm());
    */
#ifndef SD_ITERATIVE_GMG_PA
    H1stiff->AddDomainIntegrator(new MassIntegrator(one));
    H1stiff->AddDomainIntegrator(new DiffusionIntegrator(one));
    H1stiff->Assemble();
#endif
    
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
    {
      GridFunction xif(iH1fespace[interfaceIndex]);

      {
	ConstantCoefficient zero(0.0);
	xif.ProjectCoefficient(zero);
      }
      
      LinearForm b(iH1fespace[interfaceIndex]);
      b.Assemble();

      if_H1_ess_tdof_list[interfaceIndex] = H1_ess_tdof_list;  // deep copy
      
      Vector ifB, ifX;
      bf_ifH1mass[interfaceIndex]->FormLinearSystem(if_H1_ess_tdof_list[interfaceIndex], xif, b, oppa_ifH1mass[interfaceIndex], ifX, ifB);
    }
#else
    bf_ifH1mass[interfaceIndex]->FormSystemMatrix(H1_ess_tdof_list, *(ifH1massSp[interfaceIndex]));

    // Since BilinearForm uses DIAG_KEEP as the DiagonalPolicy, let's change it to DIAG_ONE manually here.
    for (int i = 0; i < H1_ess_tdof_list.Size(); i++)
      {
	const int sdof = H1_ess_tdof_list[i];
	const int dof = (sdof >= 0) ? sdof : -1-sdof;
	(*(ifH1massSp[interfaceIndex]))(dof,dof) = 1.0;
      }
#endif
    // Note that ifH1massInv is not used and does not need to be defined.
#else
    H1mass->FormSystemMatrix(H1_ess_tdof_list, *(ifH1mass[interfaceIndex]));
    
    HypreParMatrix *ifH1stiff = new HypreParMatrix();
    H1stiff->FormSystemMatrix(H1_ess_tdof_list, *ifH1stiff);
    Operator *ifH1massStrumpack = new STRUMPACKRowLocMatrix(*ifH1stiff);
    ifH1massInv[interfaceIndex] = CreateStrumpackSolver(ifH1massStrumpack, iH1fespace[interfaceIndex]->GetComm());
#endif
  }
    
  // Mixed interface operator
  bf_ifNDH1grad[interfaceIndex]->AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
  bf_ifNDH1grad[interfaceIndex]->Assemble();
#ifdef SD_ITERATIVE_GMG_PA
  if (fullAssembly)
#endif
  {
    bf_ifNDH1grad[interfaceIndex]->Finalize();
  }
  
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
  
  //ifNDH1grad[interfaceIndex] = new HypreParMatrix();
  //NDH1grad->FormSystemMatrix(ess_tdof_list, *(ifNDH1grad[interfaceIndex]));
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
  if (fullAssembly)
#endif
  {
    ifNDH1gradSp[interfaceIndex] = &(bf_ifNDH1grad[interfaceIndex]->SpMat());
    ifNDH1gradTSp[interfaceIndex] = Transpose(*(ifNDH1gradSp[interfaceIndex]));
  }
#else
  ifNDH1grad[interfaceIndex] = NDH1grad->ParallelAssemble();
  ifNDH1gradT[interfaceIndex] = ifNDH1grad[interfaceIndex]->Transpose();
#endif
  
  cout << myid << ": interface " << interfaceIndex << ", ND true size " << ifespace[interfaceIndex]->GetTrueVSize()
       << ", H1 true size " << iH1fespace[interfaceIndex]->GetTrueVSize()
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
       << ", NDH1 height " << bf_ifNDH1grad[interfaceIndex]->Height() << ", width " << bf_ifNDH1grad[interfaceIndex]->Width() << endl;
#else
       << ", NDH1 height " << ifNDH1gradSp[interfaceIndex]->Height() << ", width " << ifNDH1gradSp[interfaceIndex]->Width() << endl;
#endif
#else
       << ", NDH1 height " << ifNDH1grad[interfaceIndex]->Height() << ", width " << ifNDH1grad[interfaceIndex]->Width() << endl;
#endif
}

#ifdef SDFOSLS
#ifdef SERIAL_INTERFACES
Operator* DDMInterfaceOperator::CreateCij_FOSLS(const int localInterfaceIndex, const int orientation, const bool fullAssembly)
{
  const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
  const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

  const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];

  if (orientation == 0)
    {
      rowTrueOffsetsIF[localInterfaceIndex].resize(2);  // TODO: this does not seem to depend on orientation, so it could be just one array.
      colTrueOffsetsIF[localInterfaceIndex].resize(2);
    }
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(4);  // Number of blocks + 1
  colTrueOffsetsIF[localInterfaceIndex][orientation].SetSize(4);  // Number of blocks + 1
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;
  colTrueOffsetsIF[localInterfaceIndex][orientation][0] = 0;

  rowTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIF[localInterfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];

  rowTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIF[localInterfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];

  rowTrueOffsetsIF[localInterfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIF[localInterfaceIndex][orientation][3] = ifH1true[interfaceIndex];
    
  rowTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
  colTrueOffsetsIF[localInterfaceIndex][orientation].PartialSum();
    
  BlockOperator *op = new BlockOperator(rowTrueOffsetsIF[localInterfaceIndex][orientation], colTrueOffsetsIF[localInterfaceIndex][orientation]);

  const double tangSign = (outwardFromSD[interfaceIndex][1 - orientation] == 1) ? 1.0 : -1.0;
  //const double tangSign = (outwardFromSD[interfaceIndex][1 - orientation] == 1) ? -1.0 : 1.0;
  
  if (realPart)
    {
      // E row
#ifndef TEST_DECOUPLED_FOSLS
#ifndef TEST_DECOUPLED_FOSLS_PROJ
#ifdef SDFOSLS_PA
      if (fullAssembly)
	op->SetBlock(0, 0, ifNDmassSp[interfaceIndex], alphaIm);
      else
	op->SetBlock(0, 0, oppa_ifNDmass[interfaceIndex].Ptr(), alphaIm);
#else
      op->SetBlock(0, 0, ifNDmassSp[interfaceIndex], alphaIm);
#endif
#endif
#endif
      // H row
#ifndef ZERO_ORDER_FOSLS
#ifndef IFFOSLS
      op->SetBlock(1, 0, ifNDmassSp[interfaceIndex], alphaRe);
      op->SetBlock(1, 1, ifNDtangSp[interfaceIndex], -tangSign);  // (f, n x R) = -(n x f, R)
      //op->SetBlock(1, 1, ifNDtangSp[interfaceIndex], tangSign);  // (f, n x R) = -(n x f, R)
      // f row
      op->SetBlock(2, 0, ifNDmassSp[interfaceIndex], -1.0);
      op->SetBlock(2, 1, ifNDmassSp[interfaceIndex], alphaInverse);
#endif
#endif

#ifdef IFFOSLS_H
#ifdef SDFOSLS_PA
      if (fullAssembly)
	op->SetBlock(1, 1, ifNDmassSp[interfaceIndex], alphaIm);  // (f,R)
      else
	op->SetBlock(1, 1, oppa_ifNDmass[interfaceIndex].Ptr(), alphaIm);  // (f,R)
#else
      op->SetBlock(1, 1, ifNDmassSp[interfaceIndex], alphaIm);  // (f,R)
#endif
#endif
    }
  else
    {
      // E row
#ifdef ZERO_ORDER_FOSLS_COMPLEX
      //op->SetBlock(0, 0, ifNDmassSp[interfaceIndex], -alphaIm);
      op->SetBlock(1, 0, ifNDmassSp[interfaceIndex], -alphaIm);
#else
#ifdef IFFOSLS
      //op->SetBlock(0, 0, ifNDmassSp[interfaceIndex], -alphaIm);
#else
      op->SetBlock(0, 0, ifNDtangSp[interfaceIndex], -alphaRe * tangSign);
#endif
#endif

#ifdef IFFOSLS_H
      //op->SetBlock(1, 1, ifNDmassSp[interfaceIndex], -alphaIm);  // (f,R)
#endif
      
#ifndef IFFOSLS
#ifndef ZERO_ORDER_FOSLS
      op->SetBlock(0, 1, ifNDmassSp[interfaceIndex]);
#endif
      // H row
#ifndef ZERO_ORDER_FOSLS
      op->SetBlock(1, 0, ifNDtangSp[interfaceIndex], -alphaIm * tangSign);
#endif
      // f row
#ifndef ZERO_ORDER_FOSLS
      op->SetBlock(2, 1, ifNDmassSp[interfaceIndex], alphaInverse);
#endif
#endif
    }
  
  return op;
}
#endif
#endif

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
      rowTrueOffsetsIF[localInterfaceIndex].resize(2);  // TODO: this does not seem to depend on orientation, so it could be just one array.
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

  // TODO: essential BC should be removed by setting the diagonals to zero. 
  
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
#ifdef RL_VARFORM
    // RawatLee2010 assumes [[u]] = 0, so this term is dropped.
#else
    {
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
      op->SetBlock(0, 0, new SumOperator(oppa_ifNDmass[interfaceIndex].Ptr(), oppa_ifNDcurlcurl[interfaceIndex].Ptr(), false, false, alpha, beta));
#else
      op->SetBlock(0, 0, new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex], false, false, alpha, beta));
#endif
#else
      op->SetBlock(0, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, alpha, beta));
#endif
    }
#endif

#ifdef RL_VARFORM
  // RawatLee2010 assumes <<\mu_r^{-1} f>> = 0, so the A_m^{SF} block is nonzero but there is no C_{mn}^{SF} block.
  // -<\pi_{mn}(v_m), -\mu_r^{-1} f>_{S_{mn}}
#else
  // In PengLee2012 C_{mn}^{SF} corresponds to
  // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
  // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the C_{mn}^{SF} block is the part
  // -<\pi_{mn}(v_m), \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
  // This is an interface mass matrix.

#ifdef DDMCOMPLEX
  if (realPart)
#endif
    {
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
      op->SetBlock(0, 1, oppa_ifNDmass[interfaceIndex].Ptr(), -1.0);
#else
      op->SetBlock(0, 1, ifNDmassSp[interfaceIndex], -1.0);
#endif
#else
      op->SetBlock(0, 1, ifNDmass[interfaceIndex], -1.0);
#endif
    }
#endif
  
#ifdef RL_VARFORM
  // RawatLee2010 tests with F trial functions, rather than U plus F trial functions as in PengLee2012, and hence has no C_{mn}^{S\rho} block.
#else
  // In PengLee2012 C_{mn}^{S\rho} corresponds to
  // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
  // Since <<\rho>>_{mn} = \rho_m + \rho_n, the C_{mn}^{S\rho} block is the part
  // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_n >_{S_{mn}}
  // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
    
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
  op->SetBlock(0, 2, bf_ifNDH1grad[interfaceIndex], -gamma);
#else
  op->SetBlock(0, 2, ifNDH1gradSp[interfaceIndex], -gamma);
#endif
#else
  op->SetBlock(0, 2, ifNDH1grad[interfaceIndex], -gamma);
#endif
#endif
  
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
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
	  op->SetBlock(1, 0, new SumOperator(oppa_ifNDmass[interfaceIndex].Ptr(), oppa_ifNDcurlcurl[interfaceIndex].Ptr(), false, false, -1.0, -betaOverAlpha));
#else
	  op->SetBlock(1, 0, new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex], false, false, -1.0, -betaOverAlpha));
#endif
#else
	  op->SetBlock(1, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -1.0, -betaOverAlpha));
#endif
	}
      else
	{
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
	  op->SetBlock(1, 0, oppa_ifNDcurlcurl[interfaceIndex].Ptr(), -betaOverAlpha);
#else
	  op->SetBlock(1, 0, ifNDcurlcurlSp[interfaceIndex], -betaOverAlpha);
#endif
#else
	  op->SetBlock(1, 0, ifNDcurlcurl[interfaceIndex], -betaOverAlpha);
#endif
	}
    }
#else
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
  op->SetBlock(1, 0, new SumOperator(oppa_ifNDmass[interfaceIndex].Ptr(), oppa_ifNDcurlcurl[interfaceIndex].Ptr(), false, false, -1.0, -beta / alpha));
#else
  op->SetBlock(1, 0, new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex], false, false, -1.0, -beta / alpha));
#endif
#else
  op->SetBlock(1, 0, new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -1.0, -beta / alpha));
#endif
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
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
      op->SetBlock(1, 1, oppa_ifNDmass[interfaceIndex].Ptr(), alphaInverse);
#else
      op->SetBlock(1, 1, ifNDmassSp[interfaceIndex], alphaInverse);
#endif
#else
      op->SetBlock(1, 1, ifNDmass[interfaceIndex], alphaInverse);
#endif
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
#ifdef SERIAL_INTERFACES
#ifdef SD_ITERATIVE_GMG_PA
      op->SetBlock(1, 2, bf_ifNDH1grad[interfaceIndex], gammaOverAlpha);
#else
      op->SetBlock(1, 2, ifNDH1gradSp[interfaceIndex], gammaOverAlpha);
#endif
#else
      op->SetBlock(1, 2, ifNDH1grad[interfaceIndex], gammaOverAlpha);
#endif
    }
    
  // Row 2 is just zeros.
    
  return op;
}

Operator* NewIdentityOrEmptyOperator(const int n)
{
  if (n > 0)
    return new IdentityOperator(n);
  else
    return new EmptyOperator();
}

#ifdef GPWD
void DDMInterfaceOperator::ProjectionMult(const Vector & x)
{
  MFEM_VERIFY(x.Size() == height, "");

  globalMR->Mult(x, auxMR);  // restrict and multiply by interface mass matrices

  Vector p(2*Nphi);
  Vector s(12*Nphi);
  Vector v;

  for (int i=0; i<numInterfaces; ++i)
    {
      bool isroot = false;

      if (ifespace[i] != NULL)
	{
	  int ifrank;
	  MPI_Comm_rank(ifespace[i]->GetComm(), &ifrank);
	  if (ifrank == 0)
	    isroot = true;
	}

      if (ifNDtrue[i] > 0)
	{
	  v.SetSize(ifmrOffsetsComplex[i][2]);
	  
	  for (int j=0; j<2; ++j)  // Subdomains neighboring interface i are indexed by j.
	    {
	      for (int l=0; l<v.Size(); ++l)
		v[l] = auxMR[block_MRoffsets[(2*i)+j] + l];
		
	      cgp_u[i]->Mult(v, p);

	      for (int l=0; l<2*Nphi; ++l)
		s[(2*Nphi*j) + l] = p[l];

	      cgp_f[i]->Mult(v, p);

	      for (int l=0; l<2*Nphi; ++l)
		s[(4*Nphi) + (2*Nphi*j) + l] = p[l];

	      cgp_rho[i]->Mult(v, p);

	      for (int l=0; l<2*Nphi; ++l)
		s[(8*Nphi) + (2*Nphi*j) + l] = p[l];
	    }
	}

      // Send s from the root process for this interface to world rank 0.
      if (isroot && m_rank != 0)
	{
	  MPI_Send(s.GetData(), 12*Nphi, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
	}

      if (m_rank == 0 && ifroot[i] != 0)
	{
	  MPI_Recv(s.GetData(), 12*Nphi, MPI_DOUBLE, ifroot[i], i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

      if (m_rank == 0)
	{
	  for (int l=0; l<12*Nphi; ++l)
	    gpwd_u[(12*Nphi*i) + l] = s[l];
	}
    }
}

void DDMInterfaceOperator::ProjectionMultTranspose(Vector & x)  // x is output here
{
  MFEM_VERIFY(x.Size() == height, "");

  Vector p(2*Nphi);
  Vector s(12*Nphi);
  Vector v;

  for (int i=0; i<numInterfaces; ++i)
    {
      bool isroot = false;

      if (ifespace[i] != NULL)
	{
	  int ifrank;
	  MPI_Comm_rank(ifespace[i]->GetComm(), &ifrank);
	  if (ifrank == 0)
	    isroot = true;
	}

      if (m_rank == 0)
	{
	  for (int l=0; l<12*Nphi; ++l)
	    s[l] = gpwd_u[(12*Nphi*i) + l];
	}

      // Send s from world rank 0 to the root process for this interface.
      
      if (isroot && m_rank != 0)
	{
	  MPI_Recv(s.GetData(), 12*Nphi, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

      if (m_rank == 0 && ifroot[i] != 0)
	{
	  MPI_Send(s.GetData(), 12*Nphi, MPI_DOUBLE, ifroot[i], i, MPI_COMM_WORLD);
	}

      // Broadcast s from the root process for this interface to all processes in the interface communicator.
      if (ifespace[i] != NULL)
	{
	  MPI_Bcast(s.GetData(), 12*Nphi, MPI_DOUBLE, 0, ifespace[i]->GetComm());
	}
      
      if (ifNDtrue[i] > 0)
	{
	  v.SetSize(ifmrOffsetsComplex[i][2]);
	  
	  for (int j=0; j<2; ++j)  // Subdomains neighboring interface i are indexed by j.
	    {
	      v = 0.0;
	      
	      for (int l=0; l<2*Nphi; ++l)
		p[l] = s[(2*Nphi*j) + l];

	      cgp_u[i]->MultTranspose(p, v);

	      for (int l=0; l<2*Nphi; ++l)
		p[l] = s[(4*Nphi) + (2*Nphi*j) + l];

	      cgp_f[i]->MultTranspose(p, v);

	      for (int l=0; l<2*Nphi; ++l)
		p[l] = s[(8*Nphi) + (2*Nphi*j) + l];

	      cgp_rho[i]->MultTranspose(p, v);

	      for (int l=0; l<v.Size(); ++l)
		auxMR[block_MRoffsets[(2*i)+j] + l] = v[l];
	    }
	}
    }

  globalMR->MultTranspose(auxMR, x);
}

void ComplexGlobalProjection::Mult(const Vector & x, Vector & y) const
{
  MFEM_VERIFY(y.Size() == 2*N, "");
  
  for (int i=0; i<N; ++i)
    {
      for (int j=0; j<z.Size(); ++j)
	z[j] = x[osRe + j];  // z = Re(x)

      const double ipRR = InnerProduct(comm, z, (*b_Re)[i]);
      const double ipRI = InnerProduct(comm, z, (*b_Im)[i]);

      for (int j=0; j<z.Size(); ++j)
	z[j] = x[osIm + j];  // z = Im(x)

      const double ipIR = InnerProduct(comm, z, (*b_Re)[i]);
      const double ipII = InnerProduct(comm, z, (*b_Im)[i]);

      w[i] = ipRR - ipII;  // real part
      w[N + i] = ipRI + ipIR;  // imaginary part
    }

  Minv->Mult(w, y);
}

void ComplexGlobalProjection::MultTranspose(const Vector & x, Vector & y) const
{
  MFEM_VERIFY(x.Size() == 2*N, "");

  Minv->Mult(x, w);  // No need for transpose, assuming M is symmetric.

  for (int j=0; j<z.Size(); ++j)
    {
      y[osRe + j] = 0.0;
      y[osIm + j] = 0.0;

      for (int i=0; i<N; ++i)
	{
	  y[osRe + j] += (w[i] * (*b_Re)[i][j]) - (w[N+i] * (*b_Im)[i][j]);  // Real part
	  y[osIm + j] += (w[i] * (*b_Im)[i][j]) + (w[N+i] * (*b_Re)[i][j]);  // Imaginary part
	}
    }
}

void DDMInterfaceOperator::DefineInterfaceBasis(const int interfaceIndex)
{
  MPI_Comm comm = ifespace[interfaceIndex]->GetComm();

  int inprocs, irank;
  MPI_Comm_size(comm, &inprocs);
  MPI_Comm_rank(comm, &irank);

  ifbx[interfaceIndex].SetSize(3);
  ifby[interfaceIndex].SetSize(3);
  iforig[interfaceIndex].SetSize(3);
  
  if (irank == 0)
    {
      {
	Array<int> edges, eori;
	pmeshIF[interfaceIndex]->GetElementEdges(0, edges, eori);

	for (int k=0; k<2; ++k)  // loop over the first two edges
	  {
	    Array<int> vert;
	    pmeshIF[interfaceIndex]->GetEdgeVertices(edges[k], vert);

	    MFEM_VERIFY(vert.Size() == 2, "");
	    
	    double *vcrd0 = pmeshIF[interfaceIndex]->GetVertex(vert[0]);
	    double *vcrd1 = pmeshIF[interfaceIndex]->GetVertex(vert[1]);
	    for (int j=0; j<3; ++j)
	      {
		if (k == 0)
		  ifbx[interfaceIndex][j] = vcrd1[j] - vcrd0[j];
		else
		  ifby[interfaceIndex][j] = vcrd1[j] - vcrd0[j];
	      }
	  }
      }

      ifbx[interfaceIndex] /= ifbx[interfaceIndex].Norml2();

      if (-ifbx[interfaceIndex].Min() > ifbx[interfaceIndex].Max())
	ifbx[interfaceIndex] *= -1.0;
      
      const double dp = ifbx[interfaceIndex] * ifby[interfaceIndex];
      ifby[interfaceIndex].Add(-dp, ifbx[interfaceIndex]);

      MFEM_VERIFY(ifby[interfaceIndex].Norml2() > 1.0e-6, "");
      
      ifby[interfaceIndex] /= ifby[interfaceIndex].Norml2();

      if (-ifby[interfaceIndex].Min() > ifby[interfaceIndex].Max())
	ifby[interfaceIndex] *= -1.0;
    }

  MPI_Bcast(ifbx[interfaceIndex].GetData(), 3, MPI_DOUBLE, 0, comm);
  MPI_Bcast(ifby[interfaceIndex].GetData(), 3, MPI_DOUBLE, 0, comm);

  // Now the basis (ifbx, ifby) is defined.
  // Next, define the origin as the point closest to the origin, which is unique for Cartesian-aligned meshes and interfaces.
  double d2min = 0.0;
  int vmin = 0;
  MFEM_VERIFY(pmeshIF[interfaceIndex]->GetNV() > 0, "");
  for (int i=0; i<pmeshIF[interfaceIndex]->GetNV(); ++i)
    {
      double *v = pmeshIF[interfaceIndex]->GetVertex(i);
      const double d2 = (v[0]*v[0]) + (v[1]*v[1]) + (v[2]*v[2]);

      if (i == 0 || d2 < d2min)
	{
	  d2min = d2;
	  vmin = i;
	}
    }

  double alld2min = 0.0;
  MPI_Allreduce(&d2min, &alld2min, 1, MPI_DOUBLE, MPI_MIN, comm);

  const int ownmin = (d2min == alld2min);
  {
    int numown = 0;
    MPI_Allreduce(&ownmin, &numown, 1, MPI_INT, MPI_SUM, comm);
    MFEM_VERIFY(numown == 1, "");
  }

  int minowner = -1;

  if (ownmin)
    {
      minowner = irank;
      double *v = pmeshIF[interfaceIndex]->GetVertex(vmin);
      for (int j=0; j<3; ++j)
	iforig[interfaceIndex][j] = v[j];
    }

  {
    int thisMinOwner = minowner;
    MPI_Allreduce(&thisMinOwner, &minowner, 1, MPI_INT, MPI_MAX, comm);
  }
  
  MPI_Bcast(iforig[interfaceIndex].GetData(), 3, MPI_DOUBLE, minowner, comm);
}

Operator* DDMInterfaceOperator::CreateInterfaceMassRestriction(const int sd, const int localInterfaceIndex, const int interfaceIndex, const bool firstSD)
{
  if (ifNDtrue[interfaceIndex] > 0)
    MFEM_VERIFY(localInterfaceIndex >= 0, "");

  const int orientation = firstSD ? 1 : 0;
  
  BlockOperator *rightInjection = new BlockOperator(rowTrueOffsetsIFR[interfaceIndex][orientation],
						    colTrueOffsetsIFR[interfaceIndex][orientation]);

  if (tdofsBdryInjection[sd] == NULL)
    {
      if (InterfaceToSurfaceInjection[sd][interfaceIndex] != NULL)
	rightInjection->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd][interfaceIndex]),
							   new EmptyOperator(), false, false));
    }
  else
    {
      rightInjection->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd][interfaceIndex]),
							 tdofsBdryInjection[sd], false, false));
    }
  
  rightInjection->SetBlock(1, 1, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));

  BlockOperator *blockInjectionRight = new BlockOperator(rowTrueOffsetsIFBR[interfaceIndex][orientation], colTrueOffsetsIFBR[interfaceIndex][orientation]);

  blockInjectionRight->SetBlock(0, 0, NewIdentityOrEmptyOperator(tdofsBdry[sd].size()));
  blockInjectionRight->SetBlock(1, 2, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));
  
  BlockOperator *M = new BlockOperator(rowTrueOffsetsIFR[interfaceIndex][orientation]);

  if (M1[interfaceIndex] == NULL)
    {
      const int numBlocks = 2;

      osM1[interfaceIndex].SetSize(numBlocks + 1);
      osM1[interfaceIndex] = 0;
      osM1[interfaceIndex][1] = ifNDtrue[interfaceIndex];
      osM1[interfaceIndex][2] = ifH1true[interfaceIndex];
  
      osM1[interfaceIndex].PartialSum();

      M1[interfaceIndex] = new BlockOperator(osM1[interfaceIndex]);

      if (ifNDtrue[interfaceIndex] > 0)
	{
	  M1[interfaceIndex]->SetDiagonalBlock(0, ifNDmass[interfaceIndex]);
	  M1[interfaceIndex]->SetDiagonalBlock(1, ifH1mass[interfaceIndex]);
	}
      
      eTE_Re[interfaceIndex].resize(Nphi);
      eTE_Im[interfaceIndex].resize(Nphi);

      eTE_M[interfaceIndex].SetSize(2*Nphi);
 
      bcr_Re[interfaceIndex].resize(Nphi);
      bcr_Im[interfaceIndex].resize(Nphi);

      bcr_M[interfaceIndex].SetSize(2*Nphi);
 
      rhoc_Re[interfaceIndex].resize(Nphi);
      rhoc_Im[interfaceIndex].resize(Nphi);

      rhoc_M[interfaceIndex].SetSize(2*Nphi);

      if (ifNDtrue[interfaceIndex] > 0)
	{
	  DefineInterfaceBasis(interfaceIndex);
	}

      gpar.k = sqrt(k2);
      gpar.bx = ifbx[interfaceIndex];
      gpar.by = ifby[interfaceIndex];
      gpar.orig = iforig[interfaceIndex];
      
      for (int i=0; i<Nphi; ++i)
	{
	  eTE_Re[interfaceIndex][i].SetSize(ifNDtrue[interfaceIndex]);
	  eTE_Im[interfaceIndex][i].SetSize(ifNDtrue[interfaceIndex]);

	  bcr_Re[interfaceIndex][i].SetSize(ifNDtrue[interfaceIndex]);
	  bcr_Im[interfaceIndex][i].SetSize(ifNDtrue[interfaceIndex]);

	  rhoc_Re[interfaceIndex][i].SetSize(ifH1true[interfaceIndex]);
	  rhoc_Im[interfaceIndex][i].SetSize(ifH1true[interfaceIndex]);

	  if (ifNDtrue[interfaceIndex] > 0)
	    {
	      gpar.phi = (2*i) * M_PI / ((double) Nphi);
	      
	      ParGridFunction egf(ifespace[interfaceIndex]);
	      ParGridFunction rgf(iH1fespace[interfaceIndex]);

	      VectorFunctionCoefficient eTEexact(3, gpwd_ETE);
	      VectorFunctionCoefficient bcrexact(3, gpwd_bcr);
	      FunctionCoefficient rhocexact(gpwd_rhoc);
	      
	      // Real part
	      gpar.Re = true;

	      egf.ProjectCoefficient(eTEexact);
	      egf.GetTrueDofs(eTE_Re[interfaceIndex][i]);

	      NormalizeVector(ifespace[interfaceIndex]->GetComm(), eTE_Re[interfaceIndex][i]);
	      
	      egf.ProjectCoefficient(bcrexact);
	      egf.GetTrueDofs(bcr_Re[interfaceIndex][i]);

	      NormalizeVector(ifespace[interfaceIndex]->GetComm(), bcr_Re[interfaceIndex][i]);

	      rgf.ProjectCoefficient(rhocexact);
	      rgf.GetTrueDofs(rhoc_Re[interfaceIndex][i]);

	      NormalizeVector(iH1fespace[interfaceIndex]->GetComm(), rhoc_Re[interfaceIndex][i]);

	      // Imaginary part
	      gpar.Re = false;
	      
	      egf.ProjectCoefficient(eTEexact);
	      egf.GetTrueDofs(eTE_Im[interfaceIndex][i]);

	      NormalizeVector(ifespace[interfaceIndex]->GetComm(), eTE_Im[interfaceIndex][i]);

	      egf.ProjectCoefficient(bcrexact);
	      egf.GetTrueDofs(bcr_Im[interfaceIndex][i]);

	      NormalizeVector(ifespace[interfaceIndex]->GetComm(), bcr_Im[interfaceIndex][i]);

	      rgf.ProjectCoefficient(rhocexact);
	      rgf.GetTrueDofs(rhoc_Im[interfaceIndex][i]);

	      NormalizeVector(iH1fespace[interfaceIndex]->GetComm(), rhoc_Im[interfaceIndex][i]);
	    }
	}

      // Set the entries of eTE_M, bcr_M, rhoc_M.
      
      if (ifNDtrue[interfaceIndex] > 0)
	{
	  Vector u(ifNDtrue[interfaceIndex]);
	  Vector v(ifH1true[interfaceIndex]);

	  for (int i=0; i<Nphi; ++i)
	    {
	      for (int j=0; j<Nphi; ++j)
		{
		  // eTE_M
		  ifNDmass[interfaceIndex]->Mult(eTE_Re[interfaceIndex][i], u);

		  double ijRe = InnerProduct(ifespace[interfaceIndex]->GetComm(), u, eTE_Re[interfaceIndex][j]); // Re part
		  double ijIm = InnerProduct(ifespace[interfaceIndex]->GetComm(), u, eTE_Im[interfaceIndex][j]); // Im part

		  ifNDmass[interfaceIndex]->Mult(eTE_Im[interfaceIndex][i], u);

		  ijRe -= InnerProduct(ifespace[interfaceIndex]->GetComm(), u, eTE_Im[interfaceIndex][j]); // Re part
		  ijIm += InnerProduct(ifespace[interfaceIndex]->GetComm(), u, eTE_Im[interfaceIndex][j]); // Im part (symmetry of M implies this is unnecessary).
		  
		  eTE_M[interfaceIndex](i,j) = ijRe;
		  eTE_M[interfaceIndex](Nphi + i,j) = ijIm;
		  eTE_M[interfaceIndex](i,Nphi + j) = -ijIm;
		  eTE_M[interfaceIndex](Nphi + i,Nphi + j) = ijRe;

		  // bcr_M
		  ifNDmass[interfaceIndex]->Mult(bcr_Re[interfaceIndex][i], u);

		  ijRe = InnerProduct(ifespace[interfaceIndex]->GetComm(), u, bcr_Re[interfaceIndex][j]); // Re part
		  ijIm = InnerProduct(ifespace[interfaceIndex]->GetComm(), u, bcr_Im[interfaceIndex][j]); // Im part

		  ifNDmass[interfaceIndex]->Mult(bcr_Im[interfaceIndex][i], u);

		  ijRe -= InnerProduct(ifespace[interfaceIndex]->GetComm(), u, bcr_Im[interfaceIndex][j]); // Re part
		  ijIm += InnerProduct(ifespace[interfaceIndex]->GetComm(), u, bcr_Im[interfaceIndex][j]); // Im part (symmetry of M implies this is unnecessary).
		  
		  bcr_M[interfaceIndex](i,j) = ijRe;
		  bcr_M[interfaceIndex](Nphi + i,j) = ijIm;
		  bcr_M[interfaceIndex](i,Nphi + j) = -ijIm;
		  bcr_M[interfaceIndex](Nphi + i,Nphi + j) = ijRe;

		  // rhoc_M
		  ifH1mass[interfaceIndex]->Mult(rhoc_Re[interfaceIndex][i], v);

		  ijRe = InnerProduct(iH1fespace[interfaceIndex]->GetComm(), v, rhoc_Re[interfaceIndex][j]); // Re part
		  ijIm = InnerProduct(iH1fespace[interfaceIndex]->GetComm(), v, rhoc_Im[interfaceIndex][j]); // Im part

		  ifH1mass[interfaceIndex]->Mult(rhoc_Im[interfaceIndex][i], v);

		  ijRe -= InnerProduct(iH1fespace[interfaceIndex]->GetComm(), v, rhoc_Im[interfaceIndex][j]); // Re part
		  ijIm += InnerProduct(iH1fespace[interfaceIndex]->GetComm(), v, rhoc_Im[interfaceIndex][j]); // Im part (symmetry of M implies this is unnecessary).
		  
		  rhoc_M[interfaceIndex](i,j) = ijRe;
		  rhoc_M[interfaceIndex](Nphi + i,j) = ijIm;
		  rhoc_M[interfaceIndex](i,Nphi + j) = -ijIm;
		  rhoc_M[interfaceIndex](Nphi + i,Nphi + j) = ijRe;
		}
	    }

	  eTE_M[interfaceIndex].Invert();
	  bcr_M[interfaceIndex].Invert();
	  rhoc_M[interfaceIndex].Invert();
	}
    }

  if (ifNDtrue[interfaceIndex] > 0)
    {
      M->SetDiagonalBlock(0, ifNDmass[interfaceIndex]);
    }

  M->SetDiagonalBlock(1, M1[interfaceIndex]);
  
  // Create the product of interface mass matrix times the restriction operators. 
  TripleProductOperator *MR = new TripleProductOperator(M, rightInjection, blockInjectionRight, false, false, false);

  return MR;
}
#endif

Operator* DDMInterfaceOperator::CreateInterfaceOperator(const int sd0, const int sd1, const int localInterfaceIndex,
							const int interfaceIndex, const int orientation, const bool fullAssembly)
{
  /*
  const int sd0 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].FirstSubdomain() : (*localInterfaces)[localInterfaceIndex].SecondSubdomain();
  const int sd1 = (orientation == 0) ? (*localInterfaces)[localInterfaceIndex].SecondSubdomain() : (*localInterfaces)[localInterfaceIndex].FirstSubdomain();

  const int interfaceIndex = globalInterfaceIndex[localInterfaceIndex];
  */

  /*
  MFEM_VERIFY(ifespace[interfaceIndex] != NULL, "");
  MFEM_VERIFY(iH1fespace[interfaceIndex] != NULL, "");
  */
  
  // TODO: this function would be more concise with a local variable for ifNDtrue[interfaceIndex], etc., especially for the switch SERIAL_INTERFACES.
  
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

  /*
  MFEM_VERIFY(sd0if >= 0, "");
  MFEM_VERIFY(sd1if >= 0, "");
  */
  
  sd0osComp -= ifNDtrue[interfaceIndex];
  sd1osComp -= ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  sd0osComp -= ifespace[interfaceIndex]->GetTrueVSize();
  sd1osComp -= ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */

  if (ifNDtrue[interfaceIndex] > 0)
    MFEM_VERIFY(localInterfaceIndex >= 0, "");

#ifdef SDFOSLS
  Operator *Cij = (ifNDtrue[interfaceIndex] > 0) ? CreateCij_FOSLS(localInterfaceIndex, orientation, fullAssembly) : new EmptyOperator();
  //Operator *Cij = (ifNDtrue[interfaceIndex] > 0) ? CreateCij(localInterfaceIndex, orientation) : new EmptyOperator();  // TODO
#else
  Operator *Cij = (ifNDtrue[interfaceIndex] > 0) ? CreateCij(localInterfaceIndex, orientation) : new EmptyOperator();
#endif

  // Cij is in the local interface space only, mapping from (u^s, f_i, \rho_i) space to (u^s, f_i) space.

  // Compose Cij on the left and right with injection operators between the subdomain surfaces and the interface.

  // Create right injection operator for sd1.

  if (orientation == 0 && realPart)
    {
      rowTrueOffsetsIFR[interfaceIndex].resize(2);
      colTrueOffsetsIFR[interfaceIndex].resize(2);
      rowTrueOffsetsIFL[interfaceIndex].resize(2);
      colTrueOffsetsIFL[interfaceIndex].resize(2);

      rowTrueOffsetsIFBR[interfaceIndex].resize(2);
      colTrueOffsetsIFBR[interfaceIndex].resize(2);
      rowTrueOffsetsIFBL[interfaceIndex].resize(2);
      colTrueOffsetsIFBL[interfaceIndex].resize(2);

#ifdef SDFOSLS
      rowTrueOffsetsIFL2[interfaceIndex].resize(2);
      colTrueOffsetsIFL2[interfaceIndex].resize(2);
#endif
    }
    
  const int numBlocks = 2;  // 1 for the subdomain surface, 1 for the interface (f_{mn} and \rho_{mn}).
  rowTrueOffsetsIFR[interfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
  colTrueOffsetsIFR[interfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

  rowTrueOffsetsIFR[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFR[interfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  rowTrueOffsetsIFR[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
  rowTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */
      
  rowTrueOffsetsIFR[interfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFR[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFR[interfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  colTrueOffsetsIFR[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  /*
  colTrueOffsetsIFR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  */
  
  colTrueOffsetsIFR[interfaceIndex][orientation].PartialSum();

#ifdef SERIAL_INTERFACES
  if (!sd_nonempty[sd1])
    {
      colTrueOffsetsIFR[interfaceIndex][orientation] = 0;
    }
#endif

  BlockOperator *rightInjectionBO = new BlockOperator(rowTrueOffsetsIFR[interfaceIndex][orientation], colTrueOffsetsIFR[interfaceIndex][orientation]);

  if (tdofsBdryInjection[sd1] == NULL)
    {
      if (InterfaceToSurfaceInjection[sd1][interfaceIndex] != NULL)
	rightInjectionBO->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd1][interfaceIndex]),
							   new EmptyOperator(), false, false));
    }
  else
    {
      rightInjectionBO->SetBlock(0, 0, new ProductOperator(new TransposeOperator(InterfaceToSurfaceInjection[sd1][interfaceIndex]),
							 tdofsBdryInjection[sd1], false, false));
    }
  
#ifdef SERIAL_INTERFACES
  if (sd_nonempty[sd1])
#endif
    {
      rightInjectionBO->SetBlock(1, 1, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));
      //rightInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));
    }

  Operator *rightInjection = rightInjectionBO;
  
#ifdef SERIAL_INTERFACES
  MFEM_VERIFY(sd_nonempty[sd0] || sd_nonempty[sd1], "");
  // If both subdomains are nonempty, then no communication is done. 
  if (ifNDtrue[interfaceIndex] > 0 && !(sd_nonempty[sd0] && sd_nonempty[sd1]))
    {
      if (sd_nonempty[sd1])
	rightInjection = new ProductOperator(SIsender[interfaceIndex], rightInjectionBO, false, false);
      else
	rightInjection = new ProductOperator(SIreceiver[interfaceIndex], rightInjectionBO, false, false);
    }
#endif
  
  // Create left injection operator for sd0.

  rowTrueOffsetsIFL[interfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1
  colTrueOffsetsIFL[interfaceIndex][orientation].SetSize(numBlocks + 1);  // Number of blocks + 1

  rowTrueOffsetsIFL[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFL[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFL[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex]; // + iH1fespace[interfaceIndex]->GetTrueVSize();
  //rowTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
  //rowTrueOffsetsIFL[localInterfaceIndex][orientation][3] = iH1fespace[interfaceIndex]->GetTrueVSize();
    
  rowTrueOffsetsIFL[interfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFL[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFL[interfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIFL[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  /*
  colTrueOffsetsIFL[localInterfaceIndex][orientation][1] = ifespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIFL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize(); // + iH1fespace[interfaceIndex]->GetTrueVSize();
  */ 
  colTrueOffsetsIFL[interfaceIndex][orientation].PartialSum();

#ifdef SERIAL_INTERFACES
#ifdef SDFOSLS
  rowTrueOffsetsIFL2[interfaceIndex][orientation].SetSize(3 + 1);  // Number of blocks + 1
  colTrueOffsetsIFL2[interfaceIndex][orientation].SetSize(3 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFL2[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFL2[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFL2[interfaceIndex][orientation][2] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFL2[interfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];

  rowTrueOffsetsIFL2[interfaceIndex][orientation].PartialSum();
  colTrueOffsetsIFL2[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFL2[interfaceIndex][orientation][1] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIFL2[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIFL2[interfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];
  colTrueOffsetsIFL2[interfaceIndex][orientation].PartialSum();
#endif

  if (!sd_nonempty[sd0])
    {
      rowTrueOffsetsIFL[interfaceIndex][orientation] = 0;
#ifdef SDFOSLS
      rowTrueOffsetsIFL2[interfaceIndex][orientation] = 0;
#endif
    }
#endif

#if defined SERIAL_INTERFACES && defined SDFOSLS
  BlockOperator *leftInjection = new BlockOperator(rowTrueOffsetsIFL2[interfaceIndex][orientation], colTrueOffsetsIFL2[interfaceIndex][orientation]);
#else
  BlockOperator *leftInjection = new BlockOperator(rowTrueOffsetsIFL[interfaceIndex][orientation], colTrueOffsetsIFL[interfaceIndex][orientation]);
#endif
  
  if (tdofsBdryInjectionTranspose[sd0] == NULL)
    {
      if (InterfaceToSurfaceInjection[sd0][interfaceIndex] != NULL)
	{
	  leftInjection->SetBlock(0, 0, new ProductOperator(new EmptyOperator(), InterfaceToSurfaceInjection[sd0][interfaceIndex], false, false));
#if defined SERIAL_INTERFACES && defined SDFOSLS
	  leftInjection->SetBlock(1, 1, new ProductOperator(new EmptyOperator(), InterfaceToSurfaceInjection[sd0][interfaceIndex], false, false));
#endif
	}
    }
  else
    {
      leftInjection->SetBlock(0, 0, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][interfaceIndex], false, false));
#if defined SERIAL_INTERFACES && defined SDFOSLS
      leftInjection->SetBlock(1, 1, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][interfaceIndex], false, false));
#endif
    }


#ifdef SERIAL_INTERFACES
  if (sd_nonempty[sd0])
#endif
  {
#if defined SERIAL_INTERFACES && defined SDFOSLS
    leftInjection->SetBlock(2, 2, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex]));
#else
    leftInjection->SetBlock(1, 1, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex]));
    //leftInjection->SetBlock(1, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));
#endif
  }
  
  TripleProductOperator *CijS = new TripleProductOperator(leftInjection, Cij, rightInjection, false, false, false);

#if defined SERIAL_INTERFACES && defined SDFOSLS
  // CijS maps from (u^s, f_i, \rho_i) space to (E^s, H^s, f_i) space.
  // Create block injection operator from (E^s, H^s, f_i) to (E^s, H^s, f_i, \rho_i) on sd0, where the range is over all sd0 interfaces.

  rowTrueOffsetsIFBL[interfaceIndex][orientation].SetSize(5 + 1);  // Number of blocks + 1
  colTrueOffsetsIFBL[interfaceIndex][orientation].SetSize(3 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFBL[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFBL[interfaceIndex][orientation][2] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFBL[interfaceIndex][orientation][3] = sd0os;
  rowTrueOffsetsIFBL[interfaceIndex][orientation][4] = ifNDtrue[interfaceIndex];
  rowTrueOffsetsIFBL[interfaceIndex][orientation][5] = sd0osComp;
    
  colTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFBL[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  colTrueOffsetsIFBL[interfaceIndex][orientation][2] = tdofsBdry[sd0].size();
  colTrueOffsetsIFBL[interfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];
#else
  // CijS maps from (u^s, f_i, \rho_i) space to (u^s, f_i) space.
  // Create block injection operator from (u^s, f_i) to (u^s, f_i, \rho_i) on sd0, where the range is over all sd0 interfaces.
    
  rowTrueOffsetsIFBL[interfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1
  colTrueOffsetsIFBL[interfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFBL[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  rowTrueOffsetsIFBL[interfaceIndex][orientation][2] = sd0os;
  rowTrueOffsetsIFBL[interfaceIndex][orientation][3] = ifNDtrue[interfaceIndex];
  //rowTrueOffsetsIFBL[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize();
  rowTrueOffsetsIFBL[interfaceIndex][orientation][4] = sd0osComp;
    
  colTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFBL[interfaceIndex][orientation][1] = tdofsBdry[sd0].size();
  colTrueOffsetsIFBL[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex];
  //colTrueOffsetsIFBL[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize();
#endif
    
  rowTrueOffsetsIFBL[interfaceIndex][orientation].PartialSum();
  colTrueOffsetsIFBL[interfaceIndex][orientation].PartialSum();

#ifdef SERIAL_INTERFACES
  if (!sd_nonempty[sd0])
    {
      rowTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
      colTrueOffsetsIFBL[interfaceIndex][orientation] = 0;
    }
#endif
  
  BlockOperator *blockInjectionLeft = new BlockOperator(rowTrueOffsetsIFBL[interfaceIndex][orientation], colTrueOffsetsIFBL[interfaceIndex][orientation]);

  blockInjectionLeft->SetBlock(0, 0, NewIdentityOrEmptyOperator(tdofsBdry[sd0].size()));
#if defined SERIAL_INTERFACES && defined SDFOSLS
  blockInjectionLeft->SetBlock(1, 1, NewIdentityOrEmptyOperator(tdofsBdry[sd0].size()));
#endif
  
#ifdef SERIAL_INTERFACES
  if (!sd_nonempty[sd0])
    {
#if defined SERIAL_INTERFACES && defined SDFOSLS
      blockInjectionLeft->SetBlock(3, 2, new EmptyOperator());
#else
      blockInjectionLeft->SetBlock(2, 1, new EmptyOperator());
#endif
    }
  else
#endif
    {
#if defined SERIAL_INTERFACES && defined SDFOSLS
      blockInjectionLeft->SetBlock(3, 2, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex]));
#else
      blockInjectionLeft->SetBlock(2, 1, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex]));
      //blockInjectionLeft->SetBlock(2, 1, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()));
#endif
    }
  
  // Create block injection operator from (u^s, f_i, \rho_i) to (u^s, f_i, \rho_i) on sd1, where the domain is over all sd1 interfaces
  // and the range is only this one interface.

  rowTrueOffsetsIFBR[interfaceIndex][orientation].SetSize(2 + 1);  // Number of blocks + 1
  colTrueOffsetsIFBR[interfaceIndex][orientation].SetSize(4 + 1);  // Number of blocks + 1

  rowTrueOffsetsIFBR[interfaceIndex][orientation] = 0;
  rowTrueOffsetsIFBR[interfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  rowTrueOffsetsIFBR[interfaceIndex][orientation][2] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  //rowTrueOffsetsIFBR[localInterfaceIndex][orientation][2] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
    
  rowTrueOffsetsIFBR[interfaceIndex][orientation].PartialSum();
    
  colTrueOffsetsIFBR[interfaceIndex][orientation] = 0;
  colTrueOffsetsIFBR[interfaceIndex][orientation][1] = tdofsBdry[sd1].size();
  colTrueOffsetsIFBR[interfaceIndex][orientation][2] = sd1os;
  //colTrueOffsetsIFBR[localInterfaceIndex][orientation][3] = ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize();
  colTrueOffsetsIFBR[interfaceIndex][orientation][3] = ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
  colTrueOffsetsIFBR[interfaceIndex][orientation][4] = sd1osComp;
    
  colTrueOffsetsIFBR[interfaceIndex][orientation].PartialSum();

#ifdef SERIAL_INTERFACES
  if (!sd_nonempty[sd1])
    {
      rowTrueOffsetsIFBR[interfaceIndex][orientation] = 0;
      colTrueOffsetsIFBR[interfaceIndex][orientation] = 0;
    }
#endif
  
  BlockOperator *blockInjectionRightBO = new BlockOperator(rowTrueOffsetsIFBR[interfaceIndex][orientation], colTrueOffsetsIFBR[interfaceIndex][orientation]);

  blockInjectionRightBO->SetBlock(0, 0, NewIdentityOrEmptyOperator(tdofsBdry[sd1].size()));

#ifdef SERIAL_INTERFACES
  if (sd_nonempty[sd1])
#endif
    {
      blockInjectionRightBO->SetBlock(1, 2, NewIdentityOrEmptyOperator(ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex]));
      //blockInjectionRight->SetBlock(1, 2, new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize() + iH1fespace[interfaceIndex]->GetTrueVSize()));
    }

  Operator *blockInjectionRight = blockInjectionRightBO;
  
#ifdef SERIAL_INTERFACES
  if (!sd_nonempty[sd1])
    blockInjectionRight = new EmptyOperator();
#endif
  
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

void DDMInterfaceOperator::CreateSubdomainMatrices(const int subdomain, const bool fullAssembly)
{
  ConstantCoefficient one(1.0);
  ConstantCoefficient minusk2(-k2);

  //fespace[subdomain]->GetComm()

#ifdef SD_ITERATIVE_GMG_PA
  bfpa_sdND[subdomain] = new ParBilinearForm(fespace[subdomain]);
  bfpa_sdND[subdomain]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  bfpa_sdND[subdomain]->AddDomainIntegrator(new CurlCurlIntegrator(one));
  if (fullAssembly)
#endif
  {
    bf_sdND[subdomain] = new ParBilinearForm(fespace[subdomain]);  // TODO: make this a class member and delete at the end.
    bf_sdND[subdomain]->AddDomainIntegrator(new CurlCurlIntegrator(one));
  }
  
#ifdef AIRY_TEST
  VectorFunctionCoefficient epsilon(3, test_Airy_epsilon);

#ifdef SD_ITERATIVE_GMG_PA
  bfpa_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
  if (fullAssembly)
#endif
    {
      bf_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
    }
#else
#ifdef SD_ITERATIVE_GMG_PA
  bfpa_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(minusk2));
  if (fullAssembly)
#endif
    {
      bf_sdND[subdomain]->AddDomainIntegrator(new VectorFEMassIntegrator(minusk2));
    }
#endif

#ifdef SD_ITERATIVE_GMG_PA
  bfpa_sdND[subdomain]->Assemble();
  if (fullAssembly)
#endif
    {
      bf_sdND[subdomain]->Assemble();  
    }

  sdND[subdomain] = new HypreParMatrix();
  //sdNDcopy[subdomain] = new HypreParMatrix();

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
	Element *belem = pmeshSD[subdomain]->GetBdrElement(i);
	belem->SetAttribute(essBdryFace[i] + 1);  // 1 means face is interior to the domain; 2 means face is on the exterior boundary.

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
		    //#ifndef TEST_DECOUPLED_FOSLS_PROJ
#ifdef ZERO_IFND_BC
		    true_ess_dofs[ldof] = 1;
#endif
		    //#endif
		  }
	      }
	  }
      }

    pmeshSD[subdomain]->SetAttributes();

    fespace[subdomain]->MarkerToList(true_ess_dofs, ess_tdof_list);
  }

  //bf_sdND[subdomain]->FormSystemMatrix(ess_tdof_list, *(sdND[subdomain]));
  //bf_sdND[subdomain]->FormSystemMatrix(ess_tdof_list, *(sdNDcopy[subdomain]));  // TODO: is there a way to avoid making a copy of this matrix?

  if (fespace[subdomain] != NULL)
    {
      Vector zero(3);
      zero = 0.0;
      VectorConstantCoefficient vcc(zero);
      ParLinearForm *b = new ParLinearForm(fespace[subdomain]);
      const int sdim = pmeshSD[subdomain]->SpaceDimension();
      VectorFunctionCoefficient f(sdim, test2_RHS_exact);
      b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
      //b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vcc));
      b->Assemble();
      
      ParGridFunction xsd(fespace[subdomain]);

      {
	VectorFunctionCoefficient Eexact(3, test2_E_exact);
	//xsd.ProjectCoefficient(vcc);
	xsd.ProjectCoefficient(Eexact);
      }

      Vector sdB, sdX;
      srcSD[subdomain] = new Vector(fespace[subdomain]->GetTrueVSize());
    
      //a.FormLinearSystem(ess_tdof_list, xsd, *b, *(sdND[subdomain]), sdX, sdB);
      //bf_sdND[subdomain]->FormLinearSystem(ess_tdof_list, xsd, *b, *(sdND[subdomain]), sdX, sdB);

      /*
      Array<int> dbg_tdof_list;  // empty
      
      { // debugging!
	const int sdtsize = fespace[subdomain]->GetTrueVSize();
      
	Array<int> true_ess_dofs(sdtsize);
	for (int i=0; i<fespace[subdomain]->GetTrueVSize(); ++i)
	  true_ess_dofs[i] = 0;

	for (set<int>::const_iterator it = tdofsBdry[subdomain].begin(); it != tdofsBdry[subdomain].end(); ++it)
	  true_ess_dofs[*it] = 1;

	fespace[subdomain]->MarkerToList(true_ess_dofs, dbg_tdof_list);
      }
      */
      
      {
#ifdef SD_ITERATIVE_GMG_PA
	if (fullAssembly)
#endif
	  {
	    OperatorPtr Aptr;
	    /*
#ifdef SD_ITERATIVE_GMG_PA
	    if (fullAssembly)
	      {
		Array<int> empty_tdof_list;  // empty
		bf_sdND[subdomain]->FormLinearSystem(empty_tdof_list, xsd, *b, Aptr, sdX, *(srcSD[subdomain]));
	      }
	    else
#endif
	    */
	      {
		bf_sdND[subdomain]->FormLinearSystem(ess_tdof_list, xsd, *b, Aptr, sdX, *(srcSD[subdomain]));
	      }
	    
	      //bf_sdND[subdomain]->FormLinearSystem(dbg_tdof_list, xsd, *b, Aptr, sdX, *(srcSD[subdomain]));
	    sdND[subdomain] = Aptr.As<HypreParMatrix>();

	    cout << m_rank << ": sdND " << subdomain << " ess tdof size " << ess_tdof_list.Size() << endl;
	  }
#ifdef SD_ITERATIVE_GMG_PA
	else
	  {
	    cout << m_rank << ": sdND RHS size init " << srcSD[subdomain]->Size() << ", tvsize " << fespace[subdomain]->GetTrueVSize() << endl;
	    sd_ess_tdof_list[subdomain] = ess_tdof_list;  // deep copy
	    //sd_ess_tdof_list[subdomain] = dbg_tdof_list;  // deep copy
	    bfpa_sdND[subdomain]->FormLinearSystem(sd_ess_tdof_list[subdomain], xsd, *b, oppa_sdND[subdomain], sdX, *(srcSD[subdomain]));
	    
	    cout << m_rank << ": sdND RHS size computed " << srcSD[subdomain]->Size() << ", norm " << srcSD[subdomain]->Norml2() << ", sd " << subdomain << endl;
	  }
#endif

#ifndef SD_ITERATIVE_GMG_PA
	//sd_ess_tdof_list[subdomain] = ess_tdof_list;  // deep copy
	sd_ess_tdof_list[subdomain].SetSize(ess_tdof_list.Size());
	auto ess_tdof_list_host = ess_tdof_list.HostRead();
	for (int i=0; i<ess_tdof_list.Size(); ++i)
	  sd_ess_tdof_list[subdomain][i] = ess_tdof_list_host[i];
#endif

	  delete b;
	}
    }
  
  // Add sum over all interfaces of 
  // -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{mn}(u_m)>_{S_{mn}}

  //MFEM_VERIFY(false, "TODO: add boundary terms");
}

#ifdef MFEM_USE_STRUMPACK
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
#endif

#ifdef SD_ITERATIVE
void DDMInterfaceOperator::SetOffsetsAuxSD(const int subdomain)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size());  // 2 for each interface (f_{mn} and \rho_{mn}).
  trueOffsetsAuxSD[subdomain].SetSize(numBlocks + 1);  // Number of blocks + 1

  const bool sdNull = (fespace[subdomain] == NULL);
    
  trueOffsetsAuxSD[subdomain] = 0;

#ifdef SERIAL_INTERFACES
  if (sdNull)
    return;
#endif
  
  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      trueOffsetsAuxSD[subdomain][(2*i) + 1] += ifNDtrue[interfaceIndex];
      trueOffsetsAuxSD[subdomain][(2*i) + 2] += ifH1true[interfaceIndex];
    }
    
  trueOffsetsAuxSD[subdomain].PartialSum();
}
#endif

void DDMInterfaceOperator::SetOffsetsSD(const int subdomain)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
  trueOffsetsSD[subdomain].SetSize(numBlocks + 1);  // Number of blocks + 1
  //for (int i=0; i<numBlocks + 1; ++i)
  //trueOffsetsSD[subdomain][i] = 0;

  const bool sdNull = (fespace[subdomain] == NULL);
    
  trueOffsetsSD[subdomain] = 0;
  trueOffsetsSD[subdomain][1] = (!sdNull) ? fespace[subdomain]->GetTrueVSize() : 0;

#ifdef SERIAL_INTERFACES
  if (sdNull)
    return;
#endif
  
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

#ifndef SERIAL_INTERFACES  
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
#endif
}

#ifdef SD_ITERATIVE
Operator* DDMInterfaceOperator::CreateSubdomainImaginaryPart(const int subdomain)
{
  Operator *A = NULL;
  
  for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A.
      // Sum operators abstractly, without adding matrices into a new HypreParMatrix.
      if (A == NULL)
	{
	  if (ifNDtrue[interfaceIndex] > 0)
	    {
	      A = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef SERIAL_INTERFACES
					    new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
					    new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif
							    false, false, -alphaIm, -betaIm),
					    new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
							    false, false, false);
	    }
	}
      else
	{
	  if (ifNDtrue[interfaceIndex] > 0)
	    {
	      A = new SumOperator(A, new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef SERIAL_INTERFACES
							       new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
							       new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif											       
									       false, false, -alphaIm, -betaIm),
							       new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
									       false, false, false),
							       false, false, 1.0, 1.0);
	    }
	}
    }

  return A;
}
#endif

#ifdef SD_ITERATIVE_GMG_PA
BlockOperator* DDMInterfaceOperator::CreateSubdomainOperatorPA(const int subdomain)
{
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)
    return NULL;

  BlockOperator *op = new BlockOperator(trueOffsetsSD[subdomain]);
  
  Operator *opSS = realPart ? oppa_sdND[subdomain].Ptr() : NULL;

#ifdef RL_VARFORMOP
  MFEM_VERIFY(false, "version not implemented");
#endif

  for (auto interfaceIndex : allGlobalSubdomainInterfaces[subdomain])
    {
      int i = -1;
      for (int j=0; j<subdomainLocalInterfaces[subdomain].size(); ++j)
	{
	  if (subdomainLocalInterfaces[subdomain][j] == interfaceIndex)
	    {
	      MFEM_VERIFY(i == -1, "");
	      i = j;
	    }
	}
      
      Operator *ifsum = NULL;
      if (ifNDtrue[interfaceIndex] == 0)
	{
	  ifsum = &emptyOp;
	}
      else
	{
#ifdef SERIAL_INTERFACES
	  ifsum = new SumOperator(oppa_ifNDmass[interfaceIndex].Ptr(), oppa_ifNDcurlcurl[interfaceIndex].Ptr(), false, false, -alpha, -beta);
#else
	  ifsum = new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, -alpha, -beta);
#endif
	}
      
      Operator *ifop = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], ifsum,
						 new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
						 false, false, false);

      if (opSS == NULL)
	opSS = ifop;
      else
	{
	  opSS = new SumOperator(opSS, ifop, false, false, 1.0, 1.0);
	}

#ifdef SERIAL_INTERFACES
      op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], bf_ifNDH1grad[interfaceIndex], false, false), -gamma);
      op->SetBlock((2*i) + 1, (2*i) + 2, bf_ifNDH1grad[interfaceIndex], gammaOverAlpha);
      
      if (realPart)
	{
	  op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(oppa_ifNDmass[interfaceIndex].Ptr(), oppa_ifNDcurlcurl[interfaceIndex].Ptr(), false, false, 1.0, betaOverAlpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false));

#ifndef ROBIN_TC
	  op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(bf_ifNDH1grad[interfaceIndex]));
#endif

	  op->SetBlock((2*i) + 2, (2*i) + 2, oppa_ifH1mass[interfaceIndex].Ptr());
	}
      else
	{
	  op->SetBlock((2*i) + 1, 0, new ProductOperator(oppa_ifNDcurlcurl[interfaceIndex].Ptr(), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false),  betaOverAlpha);
	}

      op->SetBlock((2*i) + 1, (2*i) + 1, oppa_ifNDmass[interfaceIndex].Ptr(), alphaInverse);
#else
      MFEM_VERIFY(false, "not implemented");
#endif
    }

  op->SetBlock(0, 0, opSS);
  
  return op;
}
#endif

Operator* DDMInterfaceOperator::CreateSubdomainOperator(const int subdomain)
{
  BlockOperator *op = new BlockOperator(trueOffsetsSD[subdomain]);

  {
    const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).
    MFEM_VERIFY(numBlocks == trueOffsetsSD[subdomain].Size()-1, "");
  }
  
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)
    return NULL;  // Nothing to do, and the communicator is different for this process than that for processes with data.
  
  //const double sd1pen = (subdomain == 1) ? PENALTY_U_S : 0.0;
    
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

#ifdef RL_VARFORMOP
      // RawatLee2010 assumes [[u]] = 0, so the following term is dropped.
#else
      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A_m^{SS}.
#ifdef DDMCOMPLEX
      if (realPart)
	{
	  if (preconditionerMode)
	    {
	      // Create new HypreParMatrix for the sum
#ifdef SERIAL_INTERFACES
	      SparseMatrix *sumMassCC = NULL;
#else
	      HypreParMatrix *sumMassCC = NULL;
#endif
	      
	      if (ifNDtrue[interfaceIndex] > 0)
		{
#ifdef SERIAL_INTERFACES
		  sumMassCC = Add(-alpha, *(ifNDmassSp[interfaceIndex]), -beta, *(ifNDcurlcurlSp[interfaceIndex]));
#else
		  (*(ifNDmass[interfaceIndex])) *= -alpha;
		  (*(ifNDcurlcurl[interfaceIndex])) *= -beta;

		  sumMassCC = ParAdd(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex]);

		  (*(ifNDmass[interfaceIndex])) *= -1.0 / alpha;
		  (*(ifNDcurlcurl[interfaceIndex])) *= -1.0 / beta;
#endif
		}

	      if (A_SS[subdomain] == NULL)
		{
		  A_SS[subdomain] = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], sumMassCC, InterfaceToSurfaceInjectionData[subdomain][i],
									 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
#ifdef USE_SIGN_FLIP
									 gflip[subdomain][interfaceIndex], true, true,
#endif
									 ifespace[interfaceIndex], NULL, true, PENALTY_U_S);
		}
	      else
		{
		  // TODO: add another interface operator to A_SS
		  //MFEM_VERIFY(false, "TODO");

		  HypreParMatrix *previousSum = A_SS[subdomain];
		  
		  A_SS[subdomain] = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], previousSum, sumMassCC, InterfaceToSurfaceInjectionData[subdomain][i],
									 InterfaceToSurfaceInjectionGlobalData[subdomain][i],
#ifdef USE_SIGN_FLIP
									 gflip[subdomain][interfaceIndex], true, true,
#endif									 
									 ifespace[interfaceIndex], NULL, true, PENALTY_U_S);

		  delete previousSum;
		}

	      delete sumMassCC;

	      if (A_SS[subdomain] != NULL)
		op->SetBlock(0, 0, A_SS[subdomain]);
	    }
	  else  // not preconditionerMode
	    {
	      // Sum operators abstractly, without adding matrices into a new HypreParMatrix.
	      if (A_SS_op == NULL)
		{
		  if (ifNDtrue[interfaceIndex] == 0)
		    A_SS_op = sdND[subdomain];
		  else
		    {
		      Operator *ifop = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef EQUATE_REDUNDANT_VARS
								 new SumOperator(new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
										 new SumOperator(ifNDmass[interfaceIndex],
												 ifNDcurlcurl[interfaceIndex],
												 false, false, -alpha, -beta),
										 false, false, sd1pen, 1.0),
#else
#ifdef SERIAL_INTERFACES
								 new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
								 new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif
										 false, false, -alpha, -beta),
#endif
								 new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
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
						new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef EQUATE_REDUNDANT_VARS
									  new SumOperator(new IdentityOperator(ifespace[interfaceIndex]->GetTrueVSize()),
											  new SumOperator(ifNDmass[interfaceIndex],
													  ifNDcurlcurl[interfaceIndex],
													  false, false, -alpha, -beta),
											  false, false, sd1pen, 1.0),
#else
#ifdef SERIAL_INTERFACES
									  new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
									  new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif
											  false, false, -alpha, -beta),
#endif
									  new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
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
		  A_SS_op = new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef SERIAL_INTERFACES
						      new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
						      new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif
								      false, false, -alpha, -beta),
						      new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
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
		  A_SS_op = new SumOperator(A_SS_op, new TripleProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex],
#ifdef SERIAL_INTERFACES
									       new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex],
#else
									       new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex],
#endif											       
											       false, false, -alpha, -beta),
									       new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]),
									       false, false, false),
					    false, false, 1.0, 1.0);
		}
	    }
	  
	  if (A_SS_op != NULL)
	    op->SetBlock(0, 0, A_SS_op);
	}
#else
      // TODO: implement A_SS in the real case. 
#endif // DDMCOMPLEX
#endif // RL_VARFORM
      
#ifdef RL_VARFORMOP
      // RawatLee2010 assumes <<\mu_r^{-1} f>> = 0, so the A_m^{SF} block corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f>_{S_{mn}}
      // which is a mass matrix on the interface Nedelec space.
      if (ifNDmass[interfaceIndex] != NULL)
	op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], ifNDmass[interfaceIndex], false, false));
#else
      // In PengLee2012 A_m^{SF} corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0. The paper seems to confirm this in
      // a remark about equation (2.25) on page A1274. 

      // op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);
#endif
      
      // In PengLee2012 A_m^{S\rho} corresponds to
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

      if (ifNDtrue[interfaceIndex] == 0)
	continue;  // TODO: this seems to be a bug. In the empty case, an empty operator should be used, so that the injection operators can do their communication.

      // Note that this bug has gone unnoticed, since the hypre matrices are used rather than this block operator.
      
#ifdef SCHURCOMPSD
      // Modify the A_m^{SF} block by subtracting A_m^{S\rho} A_m^{\rho\rho}^{-1} A_m^{\rho F}; drop the A_m^{S\rho} block
      op->SetBlock(0, (2*i) + 1, new TripleProductOperator(new ProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], ifNDH1grad[interfaceIndex],
									       false, false),
							   ifH1massInv[interfaceIndex], ifNDH1gradT[interfaceIndex], false, false, false), gamma);
#else
#ifdef RL_VARFORMOP
      // RawatLee2010 tests with F trial functions, rather than U plus F trial functions as in PengLee2012, and hence has no A_m^{S\rho} block.
#else
#ifdef SERIAL_INTERFACES
      op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], ifNDH1gradSp[interfaceIndex], false, false), -gamma);
#else
      op->SetBlock(0, (2*i) + 2, new ProductOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex], ifNDH1grad[interfaceIndex], false, false), -gamma);
#endif
#endif
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
#ifdef SERIAL_INTERFACES
	  op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1gradSp[interfaceIndex], gammaOverAlpha);
#else
	  op->SetBlock((2*i) + 1, (2*i) + 2, ifNDH1grad[interfaceIndex], gammaOverAlpha);
#endif
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
#ifdef SERIAL_INTERFACES
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmassSp[interfaceIndex], ifNDcurlcurlSp[interfaceIndex], false, false, 1.0, betaOverAlpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false));
#else
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(new SumOperator(ifNDmass[interfaceIndex], ifNDcurlcurl[interfaceIndex], false, false, 1.0, betaOverAlpha), new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false));
#endif
	    }
#ifdef DDMCOMPLEX
	  else
	    {
#ifdef SERIAL_INTERFACES
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(ifNDcurlcurlSp[interfaceIndex], new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false),  betaOverAlpha);
#else
	      op->SetBlock((2*i) + 1, 0, new ProductOperator(ifNDcurlcurl[interfaceIndex], new TransposeOperator(InterfaceToSurfaceInjection[subdomain][interfaceIndex]), false, false),  betaOverAlpha);
#endif
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
#ifndef ROBIN_TC
#ifdef SERIAL_INTERFACES
	      op->SetBlock((2*i) + 2, (2*i) + 1, ifNDH1gradTSp[interfaceIndex]);
#else
	      op->SetBlock((2*i) + 2, (2*i) + 1, ifNDH1gradT[interfaceIndex]);
#endif
#endif
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
#ifdef SERIAL_INTERFACES
	  op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmassSp[interfaceIndex], alphaInverse);
#else
	  op->SetBlock((2*i) + 1, (2*i) + 1, ifNDmass[interfaceIndex], alphaInverse);
#endif
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
#ifdef SERIAL_INTERFACES
	      op->SetBlock((2*i) + 2, (2*i) + 2, ifH1massSp[interfaceIndex]);
#else
	      op->SetBlock((2*i) + 2, (2*i) + 2, ifH1mass[interfaceIndex]);
#endif
	    }
	}

      // TODO: should we equate redundant corner DOF's for f and \rho?
    }
    
  return op;
}

SparseMatrix * AddSp(double a, SparseMatrix * A, double b, SparseMatrix * B)
{
  if (a == 0.0 && b == 0.0)
    return NULL;

  if (b == 0.0)
    return A;

  return Add(a, *A, b, *B);
}

#ifdef SD_ITERATIVE
void DDMInterfaceOperator::CreateSubdomainAuxiliaryHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
							       Array2D<SparseMatrix*>& blockSp, Array2D<double>& blockCoefficient)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size());  // 2 for each interface (f_{mn} and \rho_{mn}).

  block.SetSize(numBlocks, numBlocks);
  blockCoefficient.SetSize(numBlocks, numBlocks);

  blockSp.SetSize(numBlocks, numBlocks);

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  block(i, j) = NULL;
	  blockSp(i, j) = NULL;
	  blockCoefficient(i, j) = 1.0;
	}
    }

  const int locSizeSD = trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1];
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)  // equivalent to checking sd_nonempty. TODO: replace with sd_nonempty.
    return;  // Nothing to do, and the communicator is different for this process than that for processes with data. 
  
  // Set blocks
  
  //for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
  for (auto interfaceIndex : allGlobalSubdomainInterfaces[subdomain])
    {
      //const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      int i = -1;
      for (int j=0; j<subdomainLocalInterfaces[subdomain].size(); ++j)
	{
	  if (subdomainLocalInterfaces[subdomain][j] == interfaceIndex)
	    {
	      MFEM_VERIFY(i == -1, "");
	      i = j;
	    }
	}
      
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
#ifdef SERIAL_INTERFACES
      if (ifNDH1gradSp[interfaceIndex] != NULL && i >= 0)
#else
      if (ifNDH1grad[interfaceIndex] != NULL && i >= 0)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i), (2*i) + 1) = ifNDH1gradSp[interfaceIndex];
#else
	  block((2*i), (2*i) + 1) = ifNDH1grad[interfaceIndex];
#endif
	  blockCoefficient((2*i), (2*i) + 1) = gammaOverAlpha;
	}
            
      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      //op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!

      if (ifNDtrue[interfaceIndex] == 0 || i < 0)
	continue;
      
#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifndef ROBIN_TC
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 1, (2*i)) = ifNDH1gradTSp[interfaceIndex];
#else
	  block((2*i) + 1, (2*i)) = ifNDH1gradT[interfaceIndex];
#endif
#endif
	}

      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.
      {
#ifdef SERIAL_INTERFACES
	blockSp((2*i), (2*i)) = ifNDmassSp[interfaceIndex];
#else
	block((2*i), (2*i)) = ifNDmass[interfaceIndex];
#endif
	blockCoefficient((2*i), (2*i)) = alphaInverse;
      }

      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 1, (2*i) + 1) = ifH1massSp[interfaceIndex];
#else
	  block((2*i) + 1, (2*i) + 1) = ifH1mass[interfaceIndex];
#endif
	}
    }
}

void DDMInterfaceOperator::CreateSubdomainDiagHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
							  Array2D<SparseMatrix*>& blockSp, Array2D<double>& blockCoefficient)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).

  block.SetSize(numBlocks, numBlocks);
  blockCoefficient.SetSize(numBlocks, numBlocks);

  blockSp.SetSize(numBlocks, numBlocks);

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  block(i, j) = NULL;
	  blockSp(i, j) = NULL;
	  blockCoefficient(i, j) = 1.0;
	}
    }

  const int locSizeSD = trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1];
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)  // equivalent to checking sd_nonempty. TODO: replace with sd_nonempty.
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
  std::vector<int> dofmapEmpty;
  
  //for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
  for (auto interfaceIndex : allGlobalSubdomainInterfaces[subdomain])
    {
      //const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      int i = -1;
      for (int j=0; j<subdomainLocalInterfaces[subdomain].size(); ++j)
	{
	  if (subdomainLocalInterfaces[subdomain][j] == interfaceIndex)
	    {
	      MFEM_VERIFY(i == -1, "");
	      i = j;
	    }
	}
      
      std::vector<int> &dofmap = (i >= 0) ? InterfaceToSurfaceInjectionData[subdomain][i] : dofmapEmpty;

#ifdef RL_VARFORM
      // RawatLee2010 assumes [[u]] = 0, so the following term is dropped.
#else
      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A_m^{SS}.
#ifdef DDMCOMPLEX
#ifdef SERIAL_INTERFACES      
      //SparseMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : AddSp(-alpha, (ifNDmassSp[interfaceIndex]), -beta, (ifNDcurlcurlSp[interfaceIndex]));
      SparseMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(-alpha, *(ifNDmassSp[interfaceIndex]), -beta, *(ifNDcurlcurlSp[interfaceIndex]));
#else
      HypreParMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(-alpha, *(ifNDmass[interfaceIndex]), -beta, *(ifNDcurlcurl[interfaceIndex]));
#endif

      const double sdNDcoef = realPart ? 1.0 : 0.0;
	
      if (A_SS_op == NULL)
	{
	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifSum, //InterfaceToSurfaceInjectionData[subdomain][i],
							 //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 dofmap,
							 GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
							 gflip[subdomain][interfaceIndex], true, true,
#endif
							 ifespace[interfaceIndex], NULL, true, 0.0, sdNDcoef);
	}
      else
	{
	  // TODO: add another interface operator to A_SS_op
	  //MFEM_VERIFY(false, "TODO");
	  
	  HypreParMatrix *previousSum = A_SS_op;

	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], previousSum, ifSum, //InterfaceToSurfaceInjectionData[subdomain][i],
							 //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 dofmap,
							 GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
							 gflip[subdomain][interfaceIndex], true, true,
#endif
							 ifespace[interfaceIndex], NULL, true, 0.0, 1.0);

	  delete previousSum;
	}

      if (A_SS_op != NULL)
	block(0, 0) = A_SS_op;

      if (ifSum != NULL)
	delete ifSum;
#else
      // TODO: implement A_SS in the real case. 
#endif // DDMCOMPLEX
#endif // RL_VARFORM
      
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
#ifdef SERIAL_INTERFACES
      if (ifNDH1gradSp[interfaceIndex] != NULL && i >= 0)
#else
      if (ifNDH1grad[interfaceIndex] != NULL && i >= 0)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 1, (2*i) + 2) = ifNDH1gradSp[interfaceIndex];
#else
	  block((2*i) + 1, (2*i) + 2) = ifNDH1grad[interfaceIndex];
#endif
	  blockCoefficient((2*i) + 1, (2*i) + 2) = gammaOverAlpha;
	}
            
      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      //op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!

      if (ifNDtrue[interfaceIndex] == 0 || i < 0)
	continue;
      
#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifndef ROBIN_TC
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 2, (2*i) + 1) = ifNDH1gradTSp[interfaceIndex];
#else
	  block((2*i) + 2, (2*i) + 1) = ifNDH1gradT[interfaceIndex];
#endif
#endif
	}

      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.
      {
#ifdef SERIAL_INTERFACES
	blockSp((2*i) + 1, (2*i) + 1) = ifNDmassSp[interfaceIndex];
#else
	block((2*i) + 1, (2*i) + 1) = ifNDmass[interfaceIndex];
#endif
	blockCoefficient((2*i) + 1, (2*i) + 1) = alphaInverse;
      }

      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 2, (2*i) + 2) = ifH1massSp[interfaceIndex];
#else
	  block((2*i) + 2, (2*i) + 2) = ifH1mass[interfaceIndex];
#endif
	}
    }
}
#endif // SD_ITERATIVE

void DDMInterfaceOperator::CreateSubdomainHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
						      //#ifdef SERIAL_INTERFACES
						      Array2D<SparseMatrix*>& blockSp,
						      //#endif				 
						      Array2D<double>& blockCoefficient)
{
  const int numBlocks = (2*subdomainLocalInterfaces[subdomain].size()) + 1;  // 1 for the subdomain, 2 for each interface (f_{mn} and \rho_{mn}).

  block.SetSize(numBlocks, numBlocks);
  blockCoefficient.SetSize(numBlocks, numBlocks);

  //#ifdef SERIAL_INTERFACES
  blockSp.SetSize(numBlocks, numBlocks);
  //#endif				 

  for (int i=0; i<numBlocks; ++i)
    {
      for (int j=0; j<numBlocks; ++j)
	{
	  block(i, j) = NULL;
	  //#ifdef SERIAL_INTERFACES
	  blockSp(i, j) = NULL;
	  //#endif
	  blockCoefficient(i, j) = 1.0;
	}
    }

  const int locSizeSD = trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1];
  if (trueOffsetsSD[subdomain][trueOffsetsSD[subdomain].Size()-1] < 1)  // equivalent to checking sd_nonempty. TODO: replace with sd_nonempty.
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
  std::vector<int> dofmapEmpty;
  
  //for (int i=0; i<subdomainLocalInterfaces[subdomain].size(); ++i)
  for (auto interfaceIndex : allGlobalSubdomainInterfaces[subdomain])
    {
      //const int interfaceIndex = subdomainLocalInterfaces[subdomain][i];

      int i = -1;
      for (int j=0; j<subdomainLocalInterfaces[subdomain].size(); ++j)
	{
	  if (subdomainLocalInterfaces[subdomain][j] == interfaceIndex)
	    {
	      MFEM_VERIFY(i == -1, "");
	      i = j;
	    }
	}
      
      std::vector<int> &dofmap = (i >= 0) ? InterfaceToSurfaceInjectionData[subdomain][i] : dofmapEmpty;

#ifdef RL_VARFORM
      // RawatLee2010 assumes [[u]] = 0, so the following term is dropped.
#else
      // Add -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m)>_{S_{mn}} - beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}} to A_m^{SS}.
#ifdef DDMCOMPLEX
#ifdef SERIAL_INTERFACES      
      //SparseMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : AddSp(-alpha, (ifNDmassSp[interfaceIndex]), -beta, (ifNDcurlcurlSp[interfaceIndex]));
      SparseMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(-alpha, *(ifNDmassSp[interfaceIndex]), -beta, *(ifNDcurlcurlSp[interfaceIndex]));
#else
      HypreParMatrix *ifSum = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(-alpha, *(ifNDmass[interfaceIndex]), -beta, *(ifNDcurlcurl[interfaceIndex]));
#endif
      
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
	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifSum, //InterfaceToSurfaceInjectionData[subdomain][i],
							 //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 dofmap,
							 GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
							 gflip[subdomain][interfaceIndex], true, true,
#endif
							 ifespace[interfaceIndex], NULL, true, 0.0, sdNDcoef);
	}
      else
	{
	  // TODO: add another interface operator to A_SS_op
	  //MFEM_VERIFY(false, "TODO");
	  
	  HypreParMatrix *previousSum = A_SS_op;

	  A_SS_op = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], previousSum, ifSum, //InterfaceToSurfaceInjectionData[subdomain][i],
							 //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
							 dofmap,
							 GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
							 gflip[subdomain][interfaceIndex], true, true,
#endif
							 ifespace[interfaceIndex], NULL, true, 0.0, 1.0);

	  delete previousSum;
	}

      if (A_SS_op != NULL)
	block(0, 0) = A_SS_op;

      if (ifSum != NULL)
	delete ifSum;
#else
      // TODO: implement A_SS in the real case. 
#endif // DDMCOMPLEX
#endif // RL_VARFORM

#ifdef RL_VARFORM
      // RawatLee2010 assumes <<\mu_r^{-1} f>> = 0, so the A_m^{SF} block corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f>_{S_{mn}}
      // which is a mass matrix on the interface Nedelec space.

      //if (ifNDmass[interfaceIndex] != NULL)
	{
	  HypreParMatrix* injectedM = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifNDmass[interfaceIndex],
									   dofmap,
									   GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
									   gflip[subdomain][interfaceIndex], true, false,
#endif
									   ifespace[interfaceIndex], NULL, false);
	  
	  if (injectedM != NULL && i >= 0)
	    block(0, (2*i) + 1) = injectedM;
	}
#else
      // In PengLee2012 A_m^{SF} corresponds to
      // -<\pi_{mn}(v_m), -\mu_r^{-1} f + <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0. The paper seems to confirm this in
      // a remark about equation (2.25) on page A1274. 
	
      // op->SetBlock(0, (2*i) + 1, new ProductOperator(InterfaceToSurfaceInjection[subdomain][i], ifNDmass[interfaceIndex], false, false), 1.0 / alpha);
#endif

#ifdef RL_VARFORM
      // RawatLee2010 tests with F trial functions, rather than U plus F trial functions as in PengLee2012, and hence has no A_m^{S\rho} block.
#else
      // In PengLee2012 A_m^{S\rho} corresponds to
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{S\rho} block is the part
      // -\gamma <\pi_{mn}(v_m), \nabla_\tau \rho_m >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.

#ifdef SERIAL_INTERFACES
      HypreParMatrix* injectedTr = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifNDH1gradSp[interfaceIndex],
#else
      HypreParMatrix* injectedTr = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifNDH1grad[interfaceIndex],
#endif
									//InterfaceToSurfaceInjectionData[subdomain][i],
									//InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									dofmap,
									GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
									gflip[subdomain][interfaceIndex], true, false,
#endif
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

      if (injectedTr != NULL && i >= 0)
	{
#ifdef MIXED_MATRIX_NO_TRANSPOSE
	  block(0, (2*i) + 2) = injectedTr;
#else
	  block(0, (2*i) + 2) = injectedTr->Transpose();  // not the most efficient implementation, to avoid premature optimization.
	  delete injectedTr;
#endif

	  blockCoefficient(0, (2*i) + 2) = -gamma;
	}
#endif // RL_VARFORM
	
      // In PengLee2012 A_m^{F\rho} corresponds to
      // gamma / alpha <w_m^s, \nabla_\tau <<\rho>>_{mn} >_{S_{mn}}
      // Since <<\rho>>_{mn} = \rho_m + \rho_n, the A_m^{F\rho} block is the part
      // gamma / alpha <w_m, \nabla_\tau \rho_n >_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
#ifdef SERIAL_INTERFACES
      if (ifNDH1gradSp[interfaceIndex] != NULL && i >= 0)
#else
      if (ifNDH1grad[interfaceIndex] != NULL && i >= 0)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 1, (2*i) + 2) = ifNDH1gradSp[interfaceIndex];
#else
	  block((2*i) + 1, (2*i) + 2) = ifNDH1grad[interfaceIndex];
#endif
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
#ifdef SERIAL_INTERFACES
	  //SparseMatrix *ifSumF = (ifNDtrue[interfaceIndex] == 0) ? NULL : AddSp(1.0, (ifNDmassSp[interfaceIndex]), betaOverAlpha, (ifNDcurlcurlSp[interfaceIndex]));
	  SparseMatrix *ifSumF = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(1.0, *(ifNDmassSp[interfaceIndex]), betaOverAlpha, *(ifNDcurlcurlSp[interfaceIndex]));
#else
	  HypreParMatrix *ifSumF = (ifNDtrue[interfaceIndex] == 0) ? NULL : Add(1.0, *(ifNDmass[interfaceIndex]), betaOverAlpha, *(ifNDcurlcurl[interfaceIndex]));
#endif
	  HypreParMatrix *matrixSum = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifSumF,
									   //InterfaceToSurfaceInjectionData[subdomain][i],
									   //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									   dofmap,
									   GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
									   gflip[subdomain][interfaceIndex], false, true,
#endif
									   ifespace[interfaceIndex], NULL, false,
									   0.0, 1.0, false);

#ifndef ZERO_ORDER_FOSLS
	  if (i >= 0)
	    block((2*i) + 1, 0) = matrixSum;
#endif
	  
	  if (ifSumF != NULL)
	    delete ifSumF;
	}
#ifdef DDMCOMPLEX
      else
	{
#ifdef SERIAL_INTERFACES
	  HypreParMatrix *matrixSum = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifNDcurlcurlSp[interfaceIndex],
#else
	  HypreParMatrix *matrixSum = AddSubdomainMatrixAndInterfaceMatrix(sd_com[subdomain], sdND[subdomain], ifNDcurlcurl[interfaceIndex],
#endif
									   //InterfaceToSurfaceInjectionData[subdomain][i],
									   //InterfaceToSurfaceInjectionGlobalData[subdomain][i],
									   dofmap,
									   GlobalInterfaceToSurfaceInjectionGlobalData[subdomain][interfaceIndex],
#ifdef USE_SIGN_FLIP
									   gflip[subdomain][interfaceIndex], false, true,
#endif
									   ifespace[interfaceIndex], NULL, false,
									   0.0, 1.0, false);
	  
	  if (i >= 0)
	    {
	      block((2*i) + 1, 0) = matrixSum;
	      blockCoefficient((2*i) + 1, 0) = betaOverAlpha;
	    }
	}
#endif
      
      // In PengLee2012 A_m^{\rho F} corresponds to
      // <\nabla_\tau \psi_m, \mu_{rm}^{-1} f_{mn}>_{S_{mn}}
      // The matrix is for a mixed bilinear form on the interface Nedelec space and H^1 space.
      //op->SetBlock((2*i) + 2, (2*i) + 1, new TransposeOperator(ifNDH1grad[interfaceIndex]));  // TODO: Without this block, the block diagonal preconditioner works very well!

      if (ifNDtrue[interfaceIndex] == 0 || i < 0)
	continue;
      
#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifndef ROBIN_TC
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 2, (2*i) + 1) = ifNDH1gradTSp[interfaceIndex];
#else
	  block((2*i) + 2, (2*i) + 1) = ifNDH1gradT[interfaceIndex];
#endif
#endif
	}

      // Diagonal blocks

      // In PengLee2012 A_m^{FF} corresponds to
      // 1/alpha <w_m^s, <<\mu_r^{-1} f>> >_{S_{mn}}
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{FF} block is the part
      // 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}
      // This is an interface mass matrix.
      {
#ifdef SERIAL_INTERFACES
	blockSp((2*i) + 1, (2*i) + 1) = ifNDmassSp[interfaceIndex];
#else
	block((2*i) + 1, (2*i) + 1) = ifNDmass[interfaceIndex];
#endif
#ifdef ZERO_ORDER_FOSLS
	blockCoefficient((2*i) + 1, (2*i) + 1) = 1.0;
#else
	blockCoefficient((2*i) + 1, (2*i) + 1) = alphaInverse;
#endif
      }

      // In PengLee2012 A_m^{\rho\rho} corresponds to
      // <\psi_m, \rho_m>_{S_{mn}}
      // This is an interface H^1 mass matrix.

#ifdef DDMCOMPLEX
      if (realPart)
#endif
	{
#ifdef SERIAL_INTERFACES
	  blockSp((2*i) + 2, (2*i) + 2) = ifH1massSp[interfaceIndex];
#else
	  block((2*i) + 2, (2*i) + 2) = ifH1mass[interfaceIndex];
#endif
	}
    }    
}

#ifdef MFEM_USE_STRUMPACK
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
      // Since <<\mu_r^{-1} f>> = \mu_{rm}^{-1} f_{mn} + \mu_{rn}^{-1} f_{nm}, the A_m^{SF} block is 0. 
	
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
#ifdef SERIAL_INTERFACES
      MFEM_VERIFY(false, "TODO");
#else
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
#endif
      // TODO: should we equate redundant corner DOF's for f and \rho?
    }

  return CreateStrumpackMatrixFromHypreBlocks(fespace[subdomain]->GetComm(), trueOffsetsSD[subdomain],
					      blocks, blockLeftInjection, blockRightInjection, blockCoefficient);
}
#endif

#ifndef SERIAL_INTERFACES
void DDMInterfaceOperator::TestProjectionError()
{
  // Reduced vectors
  Vector yr(height);
  Vector xr(width);

  xr = 0.0;
  yr = 0.0;

  /*
  Vector yA(height);
  yA = 0.0;
  */
  
  VectorFunctionCoefficient E(3, test2_E_exact);
  VectorFunctionCoefficient f(3, test2_f_exact_0);
  FunctionCoefficient rho(test2_rho_exact_0);

  // Test the full system, not reduced, but first set the reduced x to multiply by C_{ij} via globalInterfaceOp. 
  
  for (int m=0; m<numSubdomains; ++m)
    {
      if (pmeshSD[m] != NULL)
	{
	  ParGridFunction gfsd(fespace[m]);
	  gfsd.ProjectCoefficient(E);

	  Vector vsd(fespace[m]->GetTrueVSize());
	  gfsd.GetTrueDofs(vsd);

	  Vector vbdry(tdofsBdry[m].size());
	  
	  tdofsBdryInjection[m]->MultTranspose(vsd, vbdry);

	  for (int i=0; i<tdofsBdry[m].size(); ++i)
	    xr[block_trueOffsets2[m] + i] = vbdry[i];  // real part
	}
    }
  
  for (int i=0; i<numInterfaces; ++i)
    {
      const int ili = (*interfaceLocalIndex)[i];

      int sd0 = -1;
      int sd1 = -1;

      if (ili >= 0)
	{
	  MFEM_VERIFY(globalInterfaceIndex[ili] == i, "");
	  sd0 = (*localInterfaces)[ili].FirstSubdomain();
	  sd1 = (*localInterfaces)[ili].SecondSubdomain();
	}
      else
	continue;
      
      //const int sd0 = (*localInterfaces)[ili].FirstSubdomain();
      //const int sd1 = (*localInterfaces)[ili].SecondSubdomain();

      cout << m_rank << ": Interface " << ili << " sd " << sd0 << ", " << sd1 << endl;
	
      MFEM_VERIFY(0 <= sd0 && sd0 < sd1, "");
      
      if (ifNDtrue[i] > 0)
	{
	  // Set f_{sd0,sd1} = n_{sd0,sd1} \times \nabla\times u_{sd0}.

	  ParGridFunction gfi(ifespace[i]);
	  gfi.ProjectCoefficient(f);  // with normal (0, 1, 0)

	  Vector vi(ifNDtrue[i]);
	  gfi.GetTrueDofs(vi);

	  for (int isd=0; isd<2; ++isd)
	    {
	      const int sd = (isd == 0) ? sd0 : sd1;
	      int os = block_trueOffsets2[sd];  // real part
	      if (pmeshSD[sd] != NULL)
		os += tdofsBdry[sd].size();

	      const double flip = (isd == 0) ? 1.0 : -1.0;
	      
	      for (int j=0; j<ifNDtrue[i]; ++j)
		{
		  xr[os + j] = flip * vi[j];
		}
	    }
	}
      
      if (ifH1true[i] > 0)
	{
	  // Set f_{sd0,sd1} = n_{sd0,sd1} \times \nabla\times u_{sd0}.

	  ParGridFunction gfi(iH1fespace[i]);
	  gfi.ProjectCoefficient(rho);  // with normal (0, 1, 0)

	  Vector vi(ifH1true[i]);
	  gfi.GetTrueDofs(vi);

	  for (int isd=0; isd<2; ++isd)
	    {
	      const int sd = (isd == 0) ? sd0 : sd1;
	      int os = block_trueOffsets2[sd] + ifNDtrue[i];  // real part
	      if (pmeshSD[sd] != NULL)
		os += tdofsBdry[sd].size();

	      const double flip = (isd == 0) ? 1.0 : -1.0;
	      
	      for (int j=0; j<ifH1true[i]; ++j)
		{
		  xr[os + j] = flip * vi[j];
		}
	    }
	}
    }

  globalInterfaceOp->Mult(xr, yr);

  for (int m=0; m<numSubdomains; ++m)
    {
      if (sd_nonempty[m])
	{
	  Vector xm(xr.GetData() + block_trueOffsets2[m], injComplexSD[m]->Width());
	  //Vector ym(HypreAsdComplex[m]->Height());
	  Vector ym(yr.GetData() + block_trueOffsets2[m], injComplexSD[m]->Width());
	  
	  Vector u(HypreAsdComplex[m]->Width());
	  Vector v(HypreAsdComplex[m]->Height());
	  Vector vC(HypreAsdComplex[m]->Height());

	  u = 0.0;
	  v = 0.0;
	  
	  injComplexSD[m]->Mult(xm, u);
	  // Now u has [u_S, f, rho] set correctly, but not u_I. Next, overwrite [u_I, u_S] with the projection to u_m.
	  if (fespace[m] != NULL && fespace[m]->GetTrueVSize() > 0)
	    {
	      ParGridFunction gfsd(fespace[m]);
	      gfsd.ProjectCoefficient(E);

	      Vector vsd(fespace[m]->GetTrueVSize());
	      Vector Avsd(fespace[m]->GetTrueVSize());
	      gfsd.GetTrueDofs(vsd);

	      for (int j=0; j<fespace[m]->GetTrueVSize(); ++j)
		u[j] = vsd[j];  // real part

	      Avsd = 0.0;
	      sdND[m]->Mult(vsd, Avsd);
	      cout << "Norm of Avsd " << Avsd.Norml2() << endl;
	    }
	  
	  HypreAsdComplex[m]->Mult(u, v);
	  //injComplexSD[m]->MultTranspose(v, ym);

	  injComplexSD[m]->Mult(ym, vC);

	  v += vC;

	  // Now v is the full operator times the full solution, on subdomain m.
	  // Compare v against the full source.

	  const double ynorm = ySD[m]->Norml2();

	  v -= *(ySD[m]);
	  
	  const double enorm = v.Norml2();

	  cout << m_rank << ": subdomain " << m << " full projected error " << enorm << " relative to " << ynorm << endl;
	}
    }
}

int DDMInterfaceOperator::JustComputeF(const int sd, Vector const& u, Vector const& rhs, Vector& uFullSD, Vector& Cf)
{
  Vector Au(fespace[sd]->GetTrueVSize());
  Vector Rtu(fespace[sd]->GetTrueVSize());
  Vector Ru(tdofsBdry[sd].size());
  
  sdND[sd]->Mult(u, Au);

  for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
    {
      Au[i] = rhs[i] - Au[i];
    }
  
  VectorFunctionCoefficient f(3, test2_f_exact_0);

  int osf_i = tdofsBdry[sd].size();  // offset for f variable in ifespace on this interface, within the real part for this subdomain.
  int oss = (fespace[sd] == NULL) ? 0 : fespace[sd]->GetTrueVSize();

  for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[sd][i];
      // In the Robin TC case, beta = 0, and the A_m^{FS} block reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}} which is just ifNDmass[interfaceIndex] (coefficient 1).

      if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	{
	  // Now compute LHS terms (A_m^{FS} + C^{FS}) u_S + C^{FF} f
	  // C_{mn}^{FF} is the mass matrix ifNDmass corresponding to alpha^{-1} <w_m, \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
	  // In the Robin case, alpha^{-1} is purely imaginary. 

	  // C_{mn}^{FS} is an interface mass plus curl-curl stiffness matrix corresponding to
	  // -<w_m, \pi_{nm}(u_n)>_{S_{mn}} - beta/alpha <curl_\tau w_m, curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
	  // In the Robin case, beta = 0, and this reduces to the mass matrix -<w_m, \pi_{nm}(u_n)>_{S_{mn}}.

	  // Also in the Robin case, A_m^{FS} reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}}.
	  
	  // If u is real, then since (A_m^{FS} + C^{FS}) is also real, the real terms (A_m^{FS} + C^{FS}) u_S do not affect the imaginary terms
	  // A_m^{FF} f_{mn} + C_{mn}^{FF} f_{nm}, which must sum to zero.

	  // Note that A_m^{FF} is 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}. Thus A_m^{FF} f_{mn} + C_{mn}^{FF} f_{nm} = 0 just reduces to
	  // ifNDmass[interfaceIndex] (f_{mn} + f_{nm}) = 0. This does not uniquely determine f, but f is also in the u_S equation. That is,
	  // we must determine f_{nm} from the equation
	  // (Au)_S + C_{mn}^{SS} u_n^S + C_{mn}^{SF} f_{nm} = y_m^S,
	  // f_{nm} = (C_{mn}^{SF})^{-1} [y_m^S - (Au)_S - C_{mn}^{SS} u_n^S]

	  // C_{mn}^{SF} is an interface mass matrix corresponding to -<\pi_{mn}(v_m), \mu_{rn}^{-1} f_{nm}>_{S_{mn}}

	  // C_{mn}^{SS} corresponds to alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} + beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
	  // In the Robin case, C_{mn}^{SS} = alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} is a mass matrix scaled by alpha.

	  // Note that the imaginary term C_{mn}^{SS} u_n^S is balanced by the imaginary term -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} in (Au)_S.
	  // Considering only the real parts, we can drop C_{mn}^{SS} u_n^S and just compute (provided u is real)
	  // f_{nm} = (C_{mn}^{SF})^{-1} [y_m^S - (Au)_S]_{Re} = -ifNDmass^{-1} [y_m^S - (Au)_S]_{Re}
	  // f_{mn} = ifNDmass^{-1} [y_m^S - (Au)_S]_{Re}

	  Vector rS(ifNDtrue[interfaceIndex]);
	  Vector fS(ifNDtrue[interfaceIndex]);
	  Vector uS(ifNDtrue[interfaceIndex]);

	  InterfaceToSurfaceInjection[sd][interfaceIndex]->MultTranspose(Au, rS);

	  // C^{FS} is -M_f, so we compute C_{mn}^{FS} u_S = -ifNDmass u_S. Here, we use continuity of u_S. 
	  InterfaceToSurfaceInjection[sd][interfaceIndex]->MultTranspose(u, uS);
	  ifNDmass[interfaceIndex]->Mult(uS, fS);

	  for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
	    {
	      Cf[block_trueOffsets2[sd] + osf_i + j] -= fS[j];
	    }

	  // C^{SF} is -M_f, so C_{mn}^{SF} f_{nm} = [y_m^S - (Au)_S]_{Re} (real part only).
	  // In order to get the contribution only from this interface, first restrict to the interface (stored in rS now), then
	  // inject to the entire subdomain, then restrict to the surface.
	  Ru = 0.0;
	  Rtu = 0.0;
	  InterfaceToSurfaceInjection[sd][interfaceIndex]->Mult(rS, Rtu);
	  tdofsBdryInjection[sd]->MultTranspose(Rtu, Ru);
	  
	  for (int j=0; j<tdofsBdry[sd].size(); ++j)
	    {
	      Cf[block_trueOffsets2[sd] + j] += Ru[j];
	    }
	  
	  ifNDmassInv[interfaceIndex]->Mult(rS, fS);

	  {
	    Vector Mf(ifNDtrue[interfaceIndex]);
	    ifNDmass[interfaceIndex]->Mult(fS, Mf);

	    Mf -= rS;
	    cout << "Mf norm " << Mf.Norml2() << endl;
	  }

	  ParGridFunction gf(ifespace[interfaceIndex]);
	  gf.SetFromTrueDofs(fS);

	  const double err = gf.ComputeL2Error(f);
	  gf *= -1.0;
	  const double errm = gf.ComputeL2Error(f);
	  
	  gf = 0.0;
	  const double fL2 = gf.ComputeL2Error(f);

	  cout << m_rank << ": JustComputeF subdomain " << sd << " interface " << interfaceIndex << " f error " << err << " -f error " << errm
	       << " relative to " << fL2 << endl;

	  const double flip = (err < errm) ? 1.0 : -1.0;
	  
	  for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
	    uFullSD[oss + j] = fS[j]; // flip * fS[j];
	  
	  osf_i += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	  oss += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	}
    }

  return oss;
}
#endif

#ifdef DEBUG_RECONSTRUCTION
void DDMInterfaceOperator::ComputeF(const int sd, Vector const& u, Vector const& ufull, Vector const& rhs)
{
  Vector Au(fespace[sd]->GetTrueVSize());
  Vector Bd(fespace[sd]->GetTrueVSize());
  sdND[sd]->Mult(u, Au);

  *(dsolSD[sd]) = 0.0;

  Bd = Au;
  
  for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
    {
      Au[i] = rhs[i] - Au[i];
      (*(dsolSD[sd]))[i] = u[i];  // real part only
    }

  { // Set dsolred, u_{sd}^S real part
    const int os = block_trueOffsets2[sd];  // offset for real part of u_{sd}^S
    Vector u_S(tdofsBdryInjection[sd]->Width());

    // Also, compare against the reduced solution stored in dcopysol
    double nrm = 0.0;
    double err = 0.0;
    
    tdofsBdryInjection[sd]->MultTranspose(u, u_S);
    for (int j=0; j<tdofsBdryInjection[sd]->Width(); ++j)
      {
	dsolred[os + j] = u_S[j];

	nrm += dcopysol[os + j] * dcopysol[os + j];
	err += (dcopysol[os + j] - dsolred[os + j]) * (dcopysol[os + j] - dsolred[os + j]);
      }

    cout << "Subdomain " << sd << " error in real part of u_S after local full solution and restriction: " << sqrt(err) << " relative to " << sqrt(nrm) << endl;
  }
  
  VectorFunctionCoefficient f(3, test2_f_exact_0);
  
  int osf_i = tdofsBdry[sd].size();  // offset for f variable in ifespace on this interface, within the real part for this subdomain.
  int oss = (fespace[sd] == NULL) ? 0 : fespace[sd]->GetTrueVSize();

  for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[sd][i];
      // In the Robin TC case, beta = 0, and the A_m^{FS} block reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}} which is just ifNDmass[interfaceIndex] (coefficient 1).

      if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	{
	  // Now compute LHS terms (A_m^{FS} + C^{FS}) u_S + C^{FF} f
	  // C_{mn}^{FF} is the mass matrix ifNDmass corresponding to alpha^{-1} <w_m, \mu_{rn}^{-1} f_{nm}>_{S_{mn}}
	  // In the Robin case, alpha^{-1} is purely imaginary. 

	  // C_{mn}^{FS} is an interface mass plus curl-curl stiffness matrix corresponding to
	  // -<w_m, \pi_{nm}(u_n)>_{S_{mn}} - beta/alpha <curl_\tau w_m, curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
	  // In the Robin case, beta = 0, and this reduces to the mass matrix -<w_m, \pi_{nm}(u_n)>_{S_{mn}}.

	  // Also in the Robin case, A_m^{FS} reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}}.
	  
	  // If u is real, then since (A_m^{FS} + C^{FS}) is also real, the real terms (A_m^{FS} + C^{FS}) u_S do not affect the imaginary terms
	  // A_m^{FF} f_{mn} + C_{mn}^{FF} f_{nm}, which must sum to zero.

	  // Note that A_m^{FF} is 1/alpha <w_m^s, \mu_{rm}^{-1} f_{mn} >_{S_{mn}}. Thus A_m^{FF} f_{mn} + C_{mn}^{FF} f_{nm} = 0 just reduces to
	  // ifNDmass[interfaceIndex] (f_{mn} + f_{nm}) = 0. This does not uniquely determine f, but f is also in the u_S equation. That is,
	  // we must determine f_{nm} from the equation
	  // (Au)_S + C_{mn}^{SS} u_n^S + C_{mn}^{SF} f_{nm} = y_m^S,
	  // f_{nm} = (C_{mn}^{SF})^{-1} [y_m^S - (Au)_S - C_{mn}^{SS} u_n^S]

	  // C_{mn}^{SF} is an interface mass matrix corresponding to -<\pi_{mn}(v_m), \mu_{rn}^{-1} f_{nm}>_{S_{mn}}

	  // C_{mn}^{SS} corresponds to alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} + beta <curl_\tau \pi_{mn}(v_m), curl_\tau \pi_{nm}(u_n)>_{S_{mn}}
	  // In the Robin case, C_{mn}^{SS} = alpha <\pi_{mn}(v_m), \pi_{nm}(u_n)>_{S_{mn}} is a mass matrix scaled by alpha.

	  // Note that the imaginary term C_{mn}^{SS} u_n^S is balanced by the imaginary term -alpha <\pi_{mn}(v_m), \pi_{mn}(u_m>_{S_{mn}} in (Au)_S.
	  // Considering only the real parts, we can drop C_{mn}^{SS} u_n^S and just compute (provided u is real)
	  // f_{nm} = (C_{mn}^{SF})^{-1} [y_m^S - (Au)_S]_{Re} = -ifNDmass^{-1} [y_m^S - (Au)_S]_{Re}
	  // f_{mn} = ifNDmass^{-1} [y_m^S - (Au)_S]_{Re}

	  Vector rS(ifNDtrue[interfaceIndex]);
	  InterfaceToSurfaceInjection[sd][interfaceIndex]->MultTranspose(Au, rS);

	  Vector fS(ifNDtrue[interfaceIndex]);

	  ifNDmassInv[interfaceIndex]->Mult(rS, fS);

	  {
	    Vector tmpI(ifNDtrue[interfaceIndex]);
	    Vector tmpS(fespace[sd]->GetTrueVSize());
	    
	    ifNDmass[interfaceIndex]->Mult(fS, tmpI);
	    InterfaceToSurfaceInjection[sd][interfaceIndex]->Mult(tmpI, tmpS);

	    Bd += tmpS;
	  }

	  ParGridFunction gf(ifespace[interfaceIndex]);
	  gf.SetFromTrueDofs(fS);

	  const double err = gf.ComputeL2Error(f);
	  gf *= -1.0;
	  const double errm = gf.ComputeL2Error(f);
	  
	  gf = 0.0;
	  const double fL2 = gf.ComputeL2Error(f);

	  cout << m_rank << ": ComputeF subdomain " << sd << " interface " << interfaceIndex << " f error " << err << " -f error " << errm
	       << " relative to " << fL2 << endl;

	  // Also, check error between f taken from full solution and reduced solution.
	  double fnrm = 0.0;
	  double ferr = 0.0;
	  
	  // Set dsolred, f_{mn} real part
	  const int os = block_trueOffsets2[sd];  // offset for real part of u_{sd}^S
	  for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
	    {
	      //dsolred[os + osf_i + j] = fS[j];  // Set to reconstructed f
	      dsolred[os + osf_i + j] = ufull[oss + j];  // Set to original solution for f

	      fnrm += dcopysol[os + osf_i + j] * dcopysol[os + osf_i + j];
	      ferr += (ufull[oss + j] - dcopysol[os + osf_i + j]) * (ufull[oss + j] - dcopysol[os + osf_i + j]);
	      
	      (*(dsolSD[sd]))[oss + j] = fS[j];
	    }

	  cout << "Subdomain " << sd << ", interface " << interfaceIndex << " error in real part of f after local inversion " << sqrt(ferr)
	       << ", relative to " << sqrt(fnrm) << endl;
	  
	  osf_i += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	  oss += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	}
    }

  Vector Ad(AsdComplex[sd]->Height());
  AsdComplex[sd]->Mult(*(dsolSD[sd]), Ad);

  // At this point, the first real block of Ad of size fespace[sd]->GetTrueVSize() should equal rhs up to round-off, except it excludes the C blocks.
  // Instead, Bd should equal the first real block of rhs.
  
  Ad -= (*(rhsSD[sd]));

  double err = 0.0;
  double nrm = 0.0;
  
  oss = (fespace[sd] == NULL) ? 0 : fespace[sd]->GetTrueVSize();
  for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
    {
      nrm += (*(rhsSD[sd]))[i] * (*(rhsSD[sd]))[i];
      //err += Ad[i]*Ad[i];
      err += (Bd[i] - (*(rhsSD[sd]))[i])*(Bd[i] - (*(rhsSD[sd]))[i]);
    }

  cout << "SERIAL ComputeF subdomain " << sd << " reconstructed real u error " << sqrt(err) << " relative to " << sqrt(nrm) << endl;
  
  for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
    {
      const int interfaceIndex = subdomainLocalInterfaces[sd][i];
      // In the Robin TC case, beta = 0, and the A_m^{FS} block reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}} which is just ifNDmass[interfaceIndex] (coefficient 1).

      if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	{
	  err = 0.0;
	  nrm = 0.0;
	  
	  for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
	    {
	      nrm += (*(rhsSD[sd]))[oss + j] * (*(rhsSD[sd]))[oss + j];
	      err += Ad[oss + j] * Ad[oss + j];
	    }
	  
  	  oss += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];

	  // This is pointless. Ad neglects the C term multiplying u and f variables on neighboring subdomains.
	  cout << "SERIAL ComputeF subdomain " << sd << " interface " << interfaceIndex << " reconstructed real f error " << sqrt(err)
	       << " relative to " << sqrt(nrm) << endl;
	}
    }
}
#endif

void DDMInterfaceOperator::TestReconstructedFullDDSolution()
{
  // Reduced vectors
  Vector yr(height);
  Vector xr(width);

  xr = 0.0;
  yr = 0.0;
  
  globalInterfaceOp->Mult(xr, yr);
}

void DDMInterfaceOperator::CopySDMatrices(std::vector<HypreParMatrix*>& Re, std::vector<HypreParMatrix*>& Im)
{
  Re.resize(numSubdomains);
  Im.resize(numSubdomains);
  
  for (int m=0; m<numSubdomains; ++m)
    {
#ifdef TEST_SD_CMG
      Re[m] = sdND[m];
      Im[m] = NULL;
#else
      Re[m] = AsdRe_HypreBlocks[m](0,0);
      Im[m] = AsdIm_HypreBlocks[m](0,0);
#endif
    }
}

#ifdef SDFOSLS_PA
void DDMInterfaceOperator::CopyFOSLSMatrices(std::vector<Array2D<HypreParMatrix*> >& A)
{
  A.resize(numSubdomains);
  for (int m=0; m<numSubdomains; ++m)
    {
      if (cfosls[m] != NULL)
	cfosls[m]->GetMatrixPointers(A[m]);
    }
}
#endif
