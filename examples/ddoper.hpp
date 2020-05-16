#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"
#include "multigriddd.hpp"

using namespace mfem;
using namespace std;

#define AIRY_TEST

#define ZERO_RHO_BC
#define ZERO_IFND_BC  // TODO: fix this so that only the exterior boundary gets essential BC, or maybe even remove it.

//#define ELIMINATE_REDUNDANT_VARS
//#define EQUATE_REDUNDANT_VARS

#define MIXED_MATRIX_NO_TRANSPOSE


#define SPARSE_ASDCOMPLEX
#define HYPRE_PARALLEL_ASDCOMPLEX

//#define DEBUG_RECONSTRUCTION

#define ROBIN_TC
//#define PENGLEE12_SOTC
//#define RL_DDMXXI

//#define RL_VARFORM

//#define TESTFEMSOL

//#define TEST_SD_SOLVER

#define DOFMAP_DEBUG

#define PENALTY_U_S 0.0

//#define GPWD

//#define NO_COMM_SPLITTING

#define SERIAL_INTERFACES

//#define EUCLID_INVERSE

#define NO_SD_OPERATOR

#define USE_SIGN_FLIP

#define SD_ITERATIVE
#define SD_ITERATIVE_COMPLEX
//#define SD_ITERATIVE_GMG
//#define SD_ITERATIVE_GMG_PA
//#define SD_ITERATIVE_FULL

//#define IF_ITERATIVE

//#define USE_2DINTERFACE

//#define TEST_SD_CMG

//#define SDFOSLS
//#define SDFOSLS_PA

//#define IMPEDANCE_OTHER_SIGN

//#define IMPEDANCE_POSITIVE

//#define FOSLS_DIRECT_SOLVER

//#define ZERO_ORDER_FOSLS
//#define ZERO_ORDER_FOSLS_COMPLEX

//#define IFFOSLS
//#define IFFOSLS_H
//#define IFFOSLS_ESS

//#define FOSLS_INIT_GUESS

//#define TEST_DECOUPLED_FOSLS
//#define TEST_DECOUPLED_FOSLS_PROJ
//#define TEST_DECOUPLED_FOSLS_PROJ_FULL
//#define TEST_DECOUPLED_FOSLS_PROJ_FULL_EXT

// For FOSLS-DD, define SDFOSLS, IFFOSLS, IFFOSLS_H, SD_ITERATIVE_GMG, and do not define ZERO_IFND_BC. With PA, define SDFOSLS_PA and SD_ITERATIVE_GMG_PA, undefine FOSLS_DIRECT_SOLVER.

void test1_E_exact(const Vector &x, Vector &E);
void test1_RHS_exact(const Vector &x, Vector &f);
void test1_f_exact_0(const Vector &x, Vector &f);
void test1_f_exact_1(const Vector &x, Vector &f);
void test2_E_exact(const Vector &x, Vector &E);
void test2_H_exact(const Vector &x, Vector &H);
void test2_RHS_exact(const Vector &x, Vector &f);
void test2_f_exact_0(const Vector &x, Vector &f);
void test2_f_exact_1(const Vector &x, Vector &f);
double test2_rho_exact_0(const Vector &x);
double test2_rho_exact_1(const Vector &x);

#ifdef AIRY_TEST
void test_Airy_epsilon(const Vector &x, Vector &e);
#endif

SparseMatrix* GetSparseMatrixFromOperator(Operator *op);
void PrintNonzerosOfInterfaceOperator(Operator const& op, const int tsize0, const int tsize1, set<int> const& tdofsbdry0,
				      set<int> const& tdofsbdry1, const int nprocs, const int rank);

hypre_CSRMatrix* GetHypreParMatrixData(const HypreParMatrix & hypParMat);
SparseMatrix* GatherHypreParMatrix(HypreParMatrix *A);
SparseMatrix* ReceiveSparseMatrix(const int source);
void SendSparseMatrix(SparseMatrix *S, const int dest);


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
						     ParFiniteElementSpace *ifespace, ParFiniteElementSpace *ifespace2=NULL,
#endif
						     const bool adding=true, const double cI=0.0, const double cA=1.0, const bool injRows=true);

/*
HypreParMatrix* CreateHypreParMatrixFromBlocks(MPI_Comm comm, Array<int> const& offsets, Array2D<HypreParMatrix*> const& blocks,
					       Array2D<double> const& coefficient);
*/

Operator* CreateStrumpackMatrixFromHypreBlocks(MPI_Comm comm, const Array<int> & offsets, const Array2D<HypreParMatrix*> & blocks,
					       const Array2D<std::vector<int>*> & leftInjection,
					       const Array2D<std::vector<int>*> & rightInjection,
					       const Array2D<double> & coefficient);

void FindBoundaryTrueDOFs(ParFiniteElementSpace *pfespace, set<int>& tdofsBdry);
bool FacesCoincideGeometrically(ParMesh *volumeMesh, const int face, ParMesh *surfaceMesh, const int elem);
void PrintMeshBoundingBox(ParMesh *mesh);

/*
void SetInterfaceToSurfaceDOFMap(ParFiniteElementSpace *ifespace, ParFiniteElementSpace *fespace, ParFiniteElementSpace *fespaceGlobal,
				 ParMesh *pmesh, const int sdAttribute, const std::set<int>& pmeshFacesInInterface,
				 const std::set<int>& pmeshEdgesInInterface, const std::set<int>& pmeshVerticesInInterface, 
				 const FiniteElementCollection *fec, std::vector<int>& dofmap, //std::vector<int>& fdofmap,
				 std::vector<int>& gdofmap);
*/

void SetDomainDofsFromSubdomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & ssd, Vector & s);
void SetSubdomainDofsFromDomainDofs(ParFiniteElementSpace *fespaceSD, ParFiniteElementSpace *fespaceDomain, const Vector & s, Vector & ssd);

// TODO: can EmptyOperator be replaced by IdentityOperator(0)?
class EmptyOperator : public Operator
{
public:
  EmptyOperator() : Operator(0, 0) { }
  
  virtual void Mult(const Vector &x, Vector &y) const
  {
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
  }
  
  //virtual ~EmptyOperator();
};

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
    auto x_host = x.HostRead();
    auto y_host = y.HostReadWrite();

    //y = 0.0;
    for (int j=0; j<y.Size(); ++j)
      y_host[j] = 0.0;

    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y_host[*it] = x_host[i];
  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    auto x_host = x.HostRead();
    auto y_host = y.HostReadWrite();

    //y = 0.0;
    for (int j=0; j<y.Size(); ++j)
      y_host[j] = 0.0;

    int i = 0;
    for (std::set<int>::const_iterator it = id->begin(); it != id->end(); ++it, ++i)
      y_host[i] = x_host[*it];
  }
};

class InjectionOperator : public Operator
{
private:
  int *id;  // Size should be fullWidth, mapping to true subdomain DOF's.
  //mutable ParGridFunction gf;
  //int fullWidth;
  int m_nprocs, m_rank;
  
  std::vector<int> m_numTrueSD;
  std::vector<int> m_numLocalTrueSDmappedFromProc;
  std::vector<int> m_alliftsize;  // TODO: not needed as a class member.
  std::vector<int> m_alltrueSD;

#ifdef USE_SIGN_FLIP
  std::vector<int> m_alltrueSDflip;
#endif
  
  std::vector<int> m_iftToSDrank;  // map from interface true DOF to rank owning the corresponding subdomain true DOF.
  
  mutable std::vector<double> m_recv, m_send;
  std::vector<int> m_scnt, m_sdspl, m_rcnt, m_rdspl;

  MPI_Comm m_comm;
public:

  InjectionOperator(MPI_Comm comm, ParFiniteElementSpace *subdomainSpace, FiniteElementSpace *interfaceSpace, int *a,
		    std::vector<int> const& gdofmap, std::vector<int> const& gflip);
  
  ~InjectionOperator()
  {
  }

  /*
  void SetComm(MPI_Comm comm)
  {
    m_comm = comm;
  }
  */
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    //gf.SetFromTrueDofs(x);

    // Given an input vector x of local true interface DOF's, set the output vector y of local true SD DOF's, which may require values from other processes. 

    MFEM_VERIFY(m_alltrueSD.size() == m_recv.size(), "");
    MFEM_VERIFY(x.Size() == m_send.size(), "");

    std::vector<int> cnt;
    cnt.assign(m_nprocs, 0);

    auto x_host = x.HostRead();

    //for (int ifp=0; ifp<m_nprocs; ++ifp)
    for (int i=0; i<x.Size(); ++i) // loop over local true interface DOF's
      {
	m_send[m_sdspl[m_iftToSDrank[i]] + cnt[m_iftToSDrank[i]]] = x_host[i];
	cnt[m_iftToSDrank[i]]++;
      }

    { // debugging
      for (int i=0; i<m_nprocs; ++i)
	{
	  MFEM_VERIFY(cnt[i] == m_scnt[i], "");
	}
    }
    
    MPI_Alltoallv(m_send.data(), m_scnt.data(), m_sdspl.data(), MPI_DOUBLE, (double*) m_recv.data(), m_rcnt.data(), m_rdspl.data(), MPI_DOUBLE, m_comm);

    y = 0.0;
    /*
    for (int ifp=0; ifp<m_nprocs; ++ifp)
      {
	for (int i=0; i<m_rcnt[ifp]; ++i)
	  {
	    m_recv[m_rdspl[ifp] + i];
	  }
      }
    */

    for (int i=0; i<m_recv.size(); ++i)
      {
#ifdef USE_SIGN_FLIP
	y[m_alltrueSD[i]] = (m_alltrueSDflip[i] == 1) ? -m_recv[i] : m_recv[i];
#else
	y[m_alltrueSD[i]] = m_recv[i];
#endif
      }

  }

  virtual void MultTranspose(const Vector &x, Vector &y) const
  {
    // Given an input vector x of local true SD DOF's, set the output vector y of local true interface DOF's, which may require values from other processes.
    MFEM_VERIFY(y.Size() == m_send.size(), "");

    auto y_host = y.HostWrite();

    //y = 0.0;
    for (int j=0; j<y.Size(); ++j)
      y_host[j] = 0.0;

    auto x_host = x.HostRead();

    for (int i=0; i<m_recv.size(); ++i)
      {
#ifdef USE_SIGN_FLIP
	//m_recv[i] = (m_alltrueSDflip[i] == 1) ? -x(m_alltrueSD[i]) : x(m_alltrueSD[i]);
	m_recv[i] = (m_alltrueSDflip[i] == 1) ? -x_host[m_alltrueSD[i]] : x_host[m_alltrueSD[i]];
#else
	m_recv[i] = x_host[m_alltrueSD[i]];
#endif
      }

    // The roles of receive and send data are reversed.
    MPI_Alltoallv(m_recv.data(), m_rcnt.data(), m_rdspl.data(), MPI_DOUBLE, m_send.data(), m_scnt.data(), m_sdspl.data(), MPI_DOUBLE, m_comm);

    std::vector<int> cnt;
    cnt.assign(m_nprocs, 0);

    for (int i=0; i<y.Size(); ++i) // loop over local true interface DOF's
      {
	y_host[i] = m_send[m_sdspl[m_iftToSDrank[i]] + cnt[m_iftToSDrank[i]]];
	cnt[m_iftToSDrank[i]]++;
      }

    { // debugging
      for (int i=0; i<m_nprocs; ++i)
	{
	  MFEM_VERIFY(cnt[i] == m_scnt[i], "");
	}
    }
    
    /*
    //gf = 0.0;
    for (int i=0; i<fullWidth; ++i)
      {
      	if (id[i] >= 0)
	  gf[i] = x[id[i]];
      }
    
    gf.GetTrueDofs(y);
    */
  }

  void MultTransposeRaw(double *x, Vector &y) const
  {
    // Given an input vector x of local true SD DOF's, set the output vector y of local true interface DOF's, which may require values from other processes.
    MFEM_VERIFY(y.Size() == m_send.size(), "");

    y = 0.0;

    for (int i=0; i<m_recv.size(); ++i)
      {
#ifdef USE_SIGN_FLIP
	m_recv[i] = (m_alltrueSDflip[i] == 1) ? -x[m_alltrueSD[i]] : x[m_alltrueSD[i]];
#else
	m_recv[i] = x[m_alltrueSD[i]];
#endif
      }

    // The roles of receive and send data are reversed.
    MPI_Alltoallv(m_recv.data(), m_rcnt.data(), m_rdspl.data(), MPI_DOUBLE, m_send.data(), m_scnt.data(), m_sdspl.data(), MPI_DOUBLE, m_comm);

    std::vector<int> cnt;
    cnt.assign(m_nprocs, 0);

    for (int i=0; i<y.Size(); ++i) // loop over local true interface DOF's
      {
	y[i] = m_send[m_sdspl[m_iftToSDrank[i]] + cnt[m_iftToSDrank[i]]];
	cnt[m_iftToSDrank[i]]++;
      }

    { // debugging
      for (int i=0; i<m_nprocs; ++i)
	{
	  MFEM_VERIFY(cnt[i] == m_scnt[i], "");
	}
    }
  }

  void GetAllTrueSD(std::set<int> tdof)
  {
    for (auto i : m_alltrueSD)
      tdof.insert(i);
  }
};

#ifdef SERIAL_INTERFACES
class SerialInterfaceCommunicator : public Operator
{
private:
  const int n, otherRank, id;
  const bool isSender;
  
public:
  SerialInterfaceCommunicator(const int n_, const bool isSender_, const int otherRank_, const int id_)
    : Operator(n_), n(n_), isSender(isSender_), otherRank(otherRank_), id(id_)
  {
    MFEM_VERIFY(otherRank >= 0, "");
  }

  ~SerialInterfaceCommunicator()
  {
  }
  
  virtual void Mult(const Vector & x, Vector & y) const
  {
    MFEM_VERIFY(x.Size() == n && y.Size() == n, "");

    if (isSender)
      {
	auto x_host = x.HostRead();
	MPI_Send(x_host, n, MPI_DOUBLE, otherRank, id, MPI_COMM_WORLD);
	y = x;
      }
    else
      {
	auto y_host = y.HostWrite();
	MPI_Recv(y_host, n, MPI_DOUBLE, otherRank, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//MPI_Recv(y.GetData(), n, MPI_DOUBLE, otherRank, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  }
};

#endif
#ifdef GPWD
class GlobalPlaneParameters
{
public:
  GlobalPlaneParameters() : bx(3), by(3), orig(3)
  {
  }

  bool Re;
  Vector bx, by, orig;
  double phi, k;
};

// The complex system Au = y is rewritten in terms of real and imaginary parts as
// [ A^{Re} -A^{Im} ] [ u^{Re} ] = [ y^{Re} ]
// [ A^{Im}  A^{Re} ] [ u^{Im} ] = [ y^{Im} ]
class ComplexGlobalProjection : public Operator
{
public:
  ComplexGlobalProjection(const int N_, const int osRe_, const int osIm_, MPI_Comm comm_,
			  std::vector<Vector> *b_Re_, std::vector<Vector> *b_Im_, DenseMatrix *Minv_)
    : N(N_), osRe(osRe_), osIm(osIm_), comm(comm_), b_Re(b_Re_), b_Im(b_Im_), Minv(Minv_)
  {
    MFEM_VERIFY(b_Re->size() == N && b_Im->size() == N, "ComplexGlobalProjection");
    z.SetSize((*b_Re)[0].Size());
    w.SetSize(2*N);
  }

  virtual void Mult(const Vector & x, Vector & y) const;
  virtual void MultTranspose(const Vector &x, Vector &y) const;

private:
  const int N;
  const int osRe;
  const int osIm;

  MPI_Comm comm;  // communicator associated with Vector instances used for real and imaginary parts of the domain space. 

  std::vector<Vector> *b_Re, *b_Im;  // real and imaginary parts of basis
  DenseMatrix *Minv;
  
  mutable Vector z, w;
};
#endif

class BlockSubdomainPreconditioner : public Solver
{
public:
  BlockSubdomainPreconditioner(const int size, Operator *invSD_, Operator *sdImag_, Operator *invAuxComplex_)
    : Solver(size), invSD(invSD_), sdImag(sdImag_), invAuxComplex(invAuxComplex_)
  {
#ifdef SD_ITERATIVE_COMPLEX
    nSD = invSD->Height() / 2;
#else
    nSD = invSD->Height();
    MFEM_VERIFY(nSD == sdImag->Height(), "");
#endif
    
    nAux = invAuxComplex->Height() / 2;
    MFEM_VERIFY(2*(nSD + nAux) == size, "");
    //MFEM_VERIFY(, "");

    osIm = size / 2;

    xaux.SetSize(2*nAux);
    yaux.SetSize(2*nAux);

#ifdef SD_ITERATIVE_COMPLEX
    xsd.SetSize(2*nSD);
    ysd.SetSize(2*nSD);
#else
    xsd.SetSize(nSD);
 
    ysdRe.SetSize(nSD);
    ysdIm.SetSize(nSD);
#endif
  }
  
  ~BlockSubdomainPreconditioner()
  {
  }

  virtual void Mult(const Vector & x, Vector & y) const;

  virtual void SetOperator(const Operator &op)
  {
  }

private:
  Operator *invSD;
  Operator *sdImag;
  Operator *invAuxComplex;

  int nSD, nAux, osIm;

  mutable Vector xaux, yaux, xsd, ysd, ysdRe, ysdIm;
};

class DDAuxSolver : public Operator
{
public:
  DDAuxSolver(std::set<int> *allGlobalSubdomainInterfaces_,
#ifdef SD_ITERATIVE_GMG_PA
	      OperatorPtr *M_,
#else
	      SparseMatrix **M_,
#endif
	      const double alphaIm_, std::vector<int> *ifH1true_)
    : allGlobalSubdomainInterfaces(allGlobalSubdomainInterfaces_), M(M_), aIm(1.0 / alphaIm_), ifH1true(ifH1true_)
  {
    int os = 0;
    for (auto interfaceIndex : *allGlobalSubdomainInterfaces)
      {
	//HyprePCG *pcg = new HyprePCG(*(M[interfaceIndex]));
	CGSolver *pcg = new CGSolver();
#ifdef SD_ITERATIVE_GMG_PA
	if (M[interfaceIndex].Ptr() != NULL)
	  {
	    pcg->SetOperator(*(M[interfaceIndex].Ptr()));
#else
	if (M[interfaceIndex] != NULL)
	  {
	    pcg->SetOperator(*(M[interfaceIndex]));
#endif
	    pcg->SetRelTol(1e-12);
	    pcg->SetAbsTol(1e-12);
	    pcg->SetMaxIter(100);
	    pcg->SetPrintLevel(-1);
	  }
	
	Minv.push_back(pcg);
	fOS.push_back(os);

#ifdef SD_ITERATIVE_GMG_PA
	if (M[interfaceIndex].Ptr() != NULL)
	  os += 2 * ((*ifH1true)[interfaceIndex] + M[interfaceIndex].Ptr()->Height());
#else
	if (M[interfaceIndex] != NULL)
	  os += 2 * ((*ifH1true)[interfaceIndex] + M[interfaceIndex]->Height());
#endif
      }

    height = os;
    width = os;
  }

  virtual void Mult(const Vector & x, Vector & y) const
  {
    if (height == 0)
      return;

    MFEM_VERIFY(x.Size() == height && y.Size() == height, "");
    
    y = x;

    int id = 0;
    for (auto interfaceIndex : *allGlobalSubdomainInterfaces)
      {
	// For each interface, solve
	// [0   -1/a*M] [ y_Re ] = [ x_Re ]
	// [1/a*M   0 ] [ y_Im ]   [ x_Im ]
	// for y_f, where a = alphaIm. Set y_rho = 0.

#ifdef SD_ITERATIVE_GMG_PA
	if (M[interfaceIndex].Ptr() != NULL)
	  {
	    const int size = M[interfaceIndex].Ptr()->Height();
#else
	if (M[interfaceIndex] != NULL)
	  {
	    const int size = M[interfaceIndex]->Height();
#endif
	    
	    xt.SetSize(size);
	    yt.SetSize(size);

	    // Set xt = x_Re, f component.
	    for (int i=0; i<size; ++i)
	      {
		xt[i] = x[fOS[id] + i];
	      }

	    // Set yt = M^{-1} x_Re, f component.
	    Minv[id]->Mult(xt, yt);

	    // Set y_Im = -a M^{-1} x_Re, f component.
	    for (int i=0; i<size; ++i)
	      {
		y[fOS[id] + size + (*ifH1true)[interfaceIndex] + i] = -aIm * yt[i];
	      }

	    // Set xt = x_Im, f component.
	    for (int i=0; i<size; ++i)
	      {
		xt[i] = x[fOS[id] + size + (*ifH1true)[interfaceIndex] + i];
	      }

	    // Set yt = M^{-1} x_Im, f component.
	    Minv[id]->Mult(xt, yt);

	    // Set y_Re = a M^{-1} x_Im, f component.
	    for (int i=0; i<size; ++i)
	      {
		y[fOS[id] + i] = aIm * yt[i];
	      }
	  }
	
	id++;
      }
  }

private:
  std::set<int> *allGlobalSubdomainInterfaces;
#ifdef SD_ITERATIVE_GMG_PA
  OperatorPtr *M;
#else
  SparseMatrix **M;
#endif
  
  std::vector<Operator*> Minv;
  const double aIm;

  std::vector<int> *ifH1true;
  std::vector<int> fOS;

  mutable Vector xt, yt;
};

#define DDMCOMPLEX

class FOSLSSolver;
  
class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, const int numInterfaces_, ParMesh *pmesh_, ParFiniteElementSpace *fespaceGlobal, ParMesh **pmeshSD_,
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
		       const double h_, const bool partialConstructor=false);

  
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
    //globalOffdiag->Mult(x, y);

    //BlockOperator *globalSubdomainOp
    //globalSubdomainOp->Mult(x, y);
    //globalInterfaceOp->Mult(x, y);
  }  

  void CopySDMatrices(std::vector<HypreParMatrix*>& Re, std::vector<HypreParMatrix*>& Im);

#ifdef SDFOSLS_PA
  void CopyFOSLSMatrices(std::vector<Array2D<HypreParMatrix*> >& A);
#endif
  
  void TestIFMult(const Vector & x, Vector & y) const
  {
    globalInterfaceOp->Mult(x, y);
    cout << m_rank << ": TestIFMult norm " << y.Norml2() << endl;
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

  void GetReducedSource(ParFiniteElementSpace *fespaceGlobal, Vector & sourceGlobalRe, Vector & sourceGlobalIm, Vector & sourceReduced, std::vector<int> const& sdOrder) const;

#ifdef DEBUG_RECONSTRUCTION
  void ComputeF(const int sd, Vector const& u, Vector const& ufull, Vector const& rhs);
#endif
  
  
#ifdef DDMCOMPLEX
#ifndef SERIAL_INTERFACES
  int JustComputeF(const int sd, Vector const& u, Vector const& rhs, Vector& uFullSD, Vector& Cf);

  void PrintInterfaceError(const int sd, Vector const& u)
  {
    VectorFunctionCoefficient f(3, test2_f_exact_0);
    ConstantCoefficient zero(0.0);
    
    int osf_i = tdofsBdry[sd].size();  // offset for f variable in ifespace on this interface, within the real part for this subdomain.
    for (int i=0; i<subdomainLocalInterfaces[sd].size(); ++i)
      {
	const int interfaceIndex = subdomainLocalInterfaces[sd][i];
	// In the Robin TC case, beta = 0, and the A_m^{FS} block reduces to <w_m, \pi_{mn}(u_m)>_{S_{mn}} which is just ifNDmass[interfaceIndex] (coefficient 1).

	if (ifespace[interfaceIndex] != NULL && ifNDtrue[interfaceIndex] > 0)
	  {
	    Vector fS(ifNDtrue[interfaceIndex]);
	    
	    for (int j=0; j<ifNDtrue[interfaceIndex]; ++j)
	      {
		fS[j] = u[osf_i + j];
	      }

	    ParGridFunction gf(ifespace[interfaceIndex]);
	    gf.SetFromTrueDofs(fS);

	    const double err = gf.ComputeL2Error(f);
	    gf *= -1.0;
	    const double errm = gf.ComputeL2Error(f);
	  
	    gf = 0.0;
	    const double fL2 = gf.ComputeL2Error(f);

	    Vector rhoS(ifH1true[interfaceIndex]);
	    
	    for (int j=0; j<ifH1true[interfaceIndex]; ++j)
	      {
		rhoS[j] = u[osf_i + ifNDtrue[interfaceIndex] + j];
	      }

	    ParGridFunction gr(iH1fespace[interfaceIndex]);
	    gr.SetFromTrueDofs(rhoS);

	    const double errRho = gr.ComputeL2Error(zero);
	    
	    cout << m_rank << ": PrintInterfaceError subdomain " << sd << " interface " << interfaceIndex << " f error " << err << " -f error " << errm
		 << " relative to " << fL2 << ", rho L2 norm " << errRho << endl;
	    
	    osf_i += ifNDtrue[interfaceIndex] + ifH1true[interfaceIndex];
	  }
      }
  }
#endif
  
  void PrintSubdomainInteriorError(const int sd, Vector const& u)
  {
    if (fespace[sd] == NULL)
      return;
    
    ParGridFunction x(fespace[sd]);
    VectorFunctionCoefficient E(3, test2_E_exact);

    x.SetFromTrueDofs(u);

    Vector zeroVec(3);
    zeroVec = 0.0;
    VectorConstantCoefficient vzero(zeroVec);

    double err = x.ComputeL2Error(E);
    double normX = x.ComputeL2Error(vzero);

    cout << m_rank << ": Interior SD " << sd << " u error " << err << " relative to " << normX << endl;
  }
  
  void PrintSubdomainError(const int sd, Vector const& u, Vector & eSD, Vector const& rhs)
  {
    eSD = 0.0;

    if (fespace[sd] == NULL)
      return;
    
    Vector uSD(fespace[sd]->GetTrueVSize());
    auto u_host = u.HostRead();

    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      uSD[i] = u_host[i];
    
#ifdef DEBUG_RECONSTRUCTION
    ComputeF(sd, uSD, u, rhs);
#endif
    
    ParGridFunction x(fespace[sd]);
    VectorFunctionCoefficient E(3, test2_E_exact);
    //x.ProjectCoefficient(E);

    x.SetFromTrueDofs(uSD);

    Vector zeroVec(3);
    zeroVec = 0.0;
    VectorConstantCoefficient vzero(zeroVec);

    double errRe = x.ComputeL2Error(E);
    double normXRe = x.ComputeL2Error(vzero);

#ifdef IFFOSLS
    double normIFRe = 0.0;
    double normIFIm = 0.0;

    for (int i=fespace[sd]->GetTrueVSize(); i<block_ComplexOffsetsSD[sd][1]; ++i)
      {
	normIFRe = u_host[i] * u_host[i];
      }
#endif
    /*
#ifdef IFFOSLS_H
    Vector H(fespace[sd]->GetTrueVSize());
    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      H[i] = u[uSD.Size() + i];

    x.SetFromTrueDofs(H);

    VectorFunctionCoefficient Hexact(3, test2_H_exact);

    double errHRe = x.ComputeL2Error(Hexact);
    double normHRe = x.ComputeL2Error(vzero);
#endif
    */
    
    DataCollection *dc = NULL;
    const bool visit = false;
    if (visit && sd == 7)
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
	   dc = new VisItDataCollection("usd7", pmeshSD[sd]);
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
      uSD[i] = u_host[block_ComplexOffsetsSD[sd][1] + i];

    MFEM_VERIFY(u.Size() == block_ComplexOffsetsSD[sd][2], "");
    MFEM_VERIFY(u.Size() == 2*block_ComplexOffsetsSD[sd][1], "");

#ifdef IFFOSLS
    for (int i=fespace[sd]->GetTrueVSize() + block_ComplexOffsetsSD[sd][1]; i < u.Size(); ++i)
      {
	normIFIm = u_host[i] * u_host[i];
      }
#endif
    
    x.SetFromTrueDofs(uSD);
    const double errIm = x.ComputeL2Error(vzero);
    //const double errIm = x.ComputeL2Error(E);

    ParGridFunction zerogf(fespace[sd]);
    zerogf = 0.0;
    const double normE = zerogf.ComputeL2Error(E);

    const double relErrRe = errRe / normE;
    const double relErrTot = sqrt((errRe*errRe) + (errIm*errIm)) / normE;

    /*
#ifdef IFFOSLS_H
    for (int i=0; i<fespace[sd]->GetTrueVSize(); ++i)
      H[i] = u[block_ComplexOffsetsSD[sd][1] + uSD.Size() + i];

    x.SetFromTrueDofs(H);

    double normHIm = x.ComputeL2Error(vzero);
#endif
    */
    
    /*
    double relErrReMax = -1.0;
    double relErrTotMax = -1.0;
    
    MPI_Allreduce(&relErrRe, &relErrReMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&relErrTot, &relErrTotMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    */
    
    //if (m_rank == 0)
      {
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Re = " << errRe << endl;
	cout << m_rank << ": sd " << sd << " || E_h - E ||_{L^2} Im = " << errIm << endl;
	cout << m_rank << ": sd " << sd << " || E_h ||_{L^2} Re = " << normXRe << endl;
	cout << m_rank << ": sd " << sd << " || E ||_{L^2} Re = " << normE << endl;
	cout << m_rank << ": sd " << sd << " rel err Re " << relErrRe << endl; // ", max " << relErrReMax << endl;
	cout << m_rank << ": sd " << sd << " rel err tot " << relErrTot << endl; // ", max " << relErrTotMax << endl;

    /*
#ifdef IFFOSLS_H
	cout << m_rank << ": sd " << sd << " || H_h - H ||_{L^2} Re = " << errHRe << endl;
	cout << m_rank << ": sd " << sd << " || H_h ||_{L^2} Re = " << normHRe << endl;
	cout << m_rank << ": sd " << sd << " || H_h ||_{L^2} Im = " << normHIm << endl;
#endif
    */
	
#ifdef IFFOSLS
	cout << m_rank << ": sd " << sd << " || f,rho ||_{l2}^2 Re = " << normIFRe << endl;
	cout << m_rank << ": sd " << sd << " || f,rho ||_{l2}^2 Im = " << normIFIm << endl;
#endif
	
	eSD[0] = errRe;
	eSD[1] = errIm;
	eSD[2] = normE;
      }
  }
#endif

  void RecoverDomainSolution(ParFiniteElementSpace *fespaceGlobal, const Vector & solReduced, Vector const& femSol,
			     Vector & solDomain);

#ifndef SERIAL_INTERFACES
  void TestProjectionError();
#endif
  
  void TestReconstructedFullDDSolution();

#ifdef GPWD
  void GPWD_PreconditionerMult(const Vector & x, Vector & y);
#endif

  void PolyPreconditionerMult(const Vector & x, Vector & y);
  void GaussSeidelPreconditionerMult(const Vector & x, Vector & y) const;

private:

  int m_rank;

  mutable bool testing;
  
  const double k2;
  const double hmin;
  const int orderND;
  
  bool realPart;
  
  const int numSubdomains;
  int numInterfaces, numLocalInterfaces;

  ParMesh *pmeshGlobal;
  ParMesh **pmeshSD;  // Subdomain meshes
  ND_FECollection fec, fecbdry;
  H1_FECollection fecbdryH1;
  
#ifdef SERIAL_INTERFACES
  Mesh **smeshIF;  // Interface meshes in 3D
#ifdef USE_2DINTERFACE
  std::vector<Mesh*> smeshIF2D;  // Interface meshes in 2D
#endif
  
  FiniteElementSpace **ifespace, **iH1fespace;
  SparseMatrix **ifNDmassSp, **ifNDcurlcurlSp, **ifNDH1gradSp, **ifNDH1gradTSp, **ifH1massSp;

  std::vector<SerialInterfaceCommunicator*> SIsender, SIreceiver;

  std::vector<int> sd_root;
  
  //HypreParMatrix **ifNDmass, **ifNDH1grad, **ifNDH1gradT, **ifH1mass, **ifND_FS;
#else
  ParMesh **pmeshIF;  // Interface meshes
  ParFiniteElementSpace **ifespace, **iH1fespace;

  HypreParMatrix **ifNDmass, **ifNDcurlcurl, **ifNDH1grad, **ifNDH1gradT, **ifH1mass, **ifND_FS;
#endif

  EmptyOperator emptyOp;
  
  std::vector<Array2D<SparseMatrix*> > AsdRe_SparseBlocks;
  std::vector<Array2D<SparseMatrix*> > AsdIm_SparseBlocks;

  ParFiniteElementSpace **fespace;
  Operator **ifNDmassInv, **ifH1massInv;
  HypreParMatrix **sdND;
  //HypreParMatrix **sdNDcopy;
  HypreParMatrix **A_SS;
  HypreParMatrix **sdNDPen;
  HypreParMatrix **sdNDPlusPen;
  SparseMatrix **sdNDPenSp;
  ParBilinearForm **bf_sdND;
  Operator **sdNDinv;
#ifdef DDMCOMPLEX
  Operator **AsdRe, **AsdIm, **AsdP, **invAsdComplex;
#ifdef SD_ITERATIVE_GMG_PA
  BlockOperator **AsdPARe, **AsdPAIm;
  ParBilinearForm **bfpa_sdND;
  OperatorPtr *oppa_sdND, *oppa_ifNDmass, *oppa_ifNDcurlcurl, *oppa_ifH1mass;
#endif
  std::vector<Array<int> > sd_ess_tdof_list;
  std::vector<Array<int> > sd_all_ess_tdof_list;
  BilinearForm **bf_ifNDmass, **bf_ifNDcurlcurl, **bf_ifH1mass;
  MixedBilinearForm **bf_ifNDH1grad;
  std::vector<Array<int> > if_ND_ess_tdof_list, if_H1_ess_tdof_list;
#ifdef SPARSE_ASDCOMPLEX
  SparseMatrix **SpAsdComplex;
  HypreParMatrix **HypreAsdComplex;
  HYPRE_Int **SpAsdComplexRowSizes;
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
  std::vector<Array2D<HypreParMatrix*> > AsdRe_HypreBlocks;
  std::vector<Array2D<HypreParMatrix*> > AsdIm_HypreBlocks;
  std::vector<Array2D<double> > AsdRe_HypreBlockCoef;
  std::vector<Array2D<double> > AsdIm_HypreBlockCoef;
  std::vector<MPI_Comm> sd_com;
  std::vector<bool> sd_nonempty;
#ifdef SD_ITERATIVE
  HypreParMatrix **HypreDsdComplex;
  std::vector<Array2D<HypreParMatrix*> > DsdRe_HypreBlocks;
  std::vector<Array2D<HypreParMatrix*> > DsdIm_HypreBlocks;
  std::vector<Array2D<double> > DsdRe_HypreBlockCoef;
  std::vector<Array2D<double> > DsdIm_HypreBlockCoef;
  std::vector<Array2D<SparseMatrix*> > DsdRe_SparseBlocks;
  std::vector<Array2D<SparseMatrix*> > DsdIm_SparseBlocks;
  std::vector<Array<int> > trueOffsetsAuxSD;
  std::vector<Operator*> sdImag;
#endif
#endif
#endif
  BlockOperator **AsdComplex;
  BlockDiagonalPreconditioner **precAsdComplex;
#else
  Operator **Asd;
#endif
  Operator **ASPsd;
  Operator **invAsd;
  Solver **precAsd;

#ifdef SDFOSLS
  std::vector<FOSLSSolver*> cfosls;
  BlockOperator **injSD2;
  BlockOperator **injComplexSD2;
  std::vector<Array<int> > rowTrueOffsetsSD2, colTrueOffsetsSD2;
  std::vector<Array<int> > rowTrueOffsetsComplexSD2, colTrueOffsetsComplexSD2;
  std::vector<std::vector<Array<int> > > rowTrueOffsetsIFL2, colTrueOffsetsIFL2;
  BilinearForm **bf_ifNDtang;
  SparseMatrix **ifNDtangSp;
  std::vector<std::vector<int> > outwardFromSD;
  Array<int> block_trueOffsets_FOSLS;
  Array<int> block_trueOffsets_FOSLS2;
  std::vector<Array<int> > rowTrueOffsetsComplexIF_FOSLS, colTrueOffsetsComplexIF_FOSLS;
#endif

  // TODO: it may be possible to eliminate ifND_FS. It is assembled as a convenience, to avoid summing entries input to ASPsd.
  
  Vector **ySD;
  Vector **rhsSD;
  Vector **srcSD;

#ifdef DEBUG_RECONSTRUCTION
  mutable Vector dsolred, dsourcered, dcopysol;
  Vector **dsolSD;
#endif
  
  BlockOperator **injSD;
  
  std::vector<SubdomainInterface> *localInterfaces;
  std::vector<int> *interfaceLocalIndex;
  std::vector<int> globalInterfaceIndex;
  std::vector<std::vector<int> > subdomainLocalInterfaces;

  std::vector<int> ifNDtrue, ifH1true;
  
#ifdef DDMCOMPLEX
  BlockOperator *globalInterfaceOpRe, *globalInterfaceOpIm;
  std::vector<Array<int> > block_ComplexOffsetsSD;
  Array<int> block_trueOffsets2;  // Offsets used in globalOp
  //Array<int> block_trueOffsets22;  // Offsets used in globalSubdomainOp
  BlockOperator **injComplexSD;
#endif

  BlockOperator *globalInterfaceOp, *globalSubdomainOp;
  
  Operator *globalOp;  // Operator for all global subdomains (blocks corresponding to non-local subdomains will be NULL).
  Operator *globalOffdiag;
  
  Array<int> block_trueOffsets;  // Offsets used in globalOp

  BlockOperator *NDinv;
  
  vector<set<int> > tdofsBdry;
  SetInjectionOperator **tdofsBdryInjection;
  Operator **tdofsBdryInjectionTranspose;
  
#ifdef GPWD
  BlockOperator *globalMR;

  std::vector<Array<int> > osM1;
  std::vector<BlockOperator*> M1;

  std::vector<int> ifroot;
  
  const int Nphi;
  
  std::vector<std::vector<Vector> > eTE_Re, eTE_Im, bcr_Re, bcr_Im, rhoc_Re, rhoc_Im;
  std::vector<DenseMatrix> eTE_M, bcr_M, rhoc_M;

  std::vector<Vector> ifbx, ifby, iforig;
  
  std::vector<Array<int> > ifmrOffsetsComplex;

  Array<int> block_MRoffsets;  // Offsets used in globalMR

  int gpwd_size;
  
  Vector auxMR, gpwd_u, gpwd_v, gpwd_w, gpwd_z;

  DenseMatrix gpwd_cmat;
  
  std::vector<ComplexGlobalProjection*> cgp_u, cgp_f, cgp_rho;
  
  void DefineInterfaceBasis(const int interfaceIndex);
  Operator* CreateInterfaceMassRestriction(const int sd, const int localInterfaceIndex, const int interfaceIndex, const bool firstSD);

  void ProjectionMult(const Vector & x);
  void ProjectionMultTranspose(Vector & x);
#endif

  Vector prec_z, prec_w;
  
  double alpha, beta, gamma;
  double alphaInverse, betaOverAlpha, gammaOverAlpha;
  double alphaRe, betaRe, gammaRe;
  double alphaIm, betaIm, gammaIm;

  bool preconditionerMode;
  
  std::vector<std::vector<InjectionOperator*> > InterfaceToSurfaceInjection;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionData;
  //std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionFullData;
  std::vector<std::vector<std::vector<int> > > InterfaceToSurfaceInjectionGlobalData;

  std::vector<std::vector<std::vector<int> > > GlobalInterfaceToSurfaceInjectionGlobalData;
  std::vector<std::vector<std::vector<int> > > gflip;
  
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

  std::vector<std::set<int> > allGlobalSubdomainInterfaces;
  
#ifdef EQUATE_REDUNDANT_VARS
  std::vector<Array<int> > rowTrueOffsetsIFRR, colTrueOffsetsIFRR;
#endif
  
  // TODO: if the number of subdomains gets large, it may be better to define a local block operator only for local subdomains.

  void SetParameters()
  {
    const double k = sqrt(k2);

#ifdef PENGLEE12_SOTC
    const double cTM = 4.0;
    const double cTE = 1.5 * cTM;
#else
    const double cTE = 0.5;  // From RawatLee2010
    const double cTM = 4.0;  // From RawatLee2010
#endif
    
    // Note that PengLee2012 recommends cTE = 1.5 * cTM. 

    // TODO: take these parameters from the mesh and finite element space.
    //const double h = 0.0350769;
    //const double h = 0.0175385;
    //const double h = 1.4e-1;
    //const double h = 0.00877;

    const double h = hmin;

    const double feOrder = orderND;

    if (m_rank == 0)
      {
	cout << "Set parameters with k = " << k << ", h = " << h << ", kh " << k*h << ", cTE " << cTE << ", cTM " << cTM << ", FEM order " << feOrder << endl;
#ifdef ROBIN_TC
	cout << "Using Robin TC" << endl;
#else
#ifdef PENGLEE12_SOTC
	cout << "Using PL12 SOTC" << endl;
#else
	cout << "Using RL10 SOTC" << endl;
#endif
#endif
      }
    
#ifdef RL_DDMXXI
    const double ktTE = 0.5 * ((10.0*feOrder / h) + k);
    const double ktTM = 0.5 * ((20.0 * feOrder / (3.0 * h)) + k);
#else
    const double ktTE = cTE * M_PI * feOrder / h;
    const double ktTM = cTM * M_PI * feOrder / h;
#endif
    
    /*    
    const double ktTE = 3.0 * k; // From PengLee2012
    const double ktTM = 2.0 * k; // From PengLee2012
    */
    
    //const double kzTE = sqrt(8.0 * k2);
    //const double kzTM = sqrt(3.0 * k2);

    const double kzTE = sqrt((ktTE*ktTE) - k2);
    const double kzTM = sqrt((ktTM*ktTM) - k2);

#ifdef PENGLEE12_SOTC
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
#else
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
#endif
    
#ifdef ROBIN_TC
    // Robin TC

    // Real part
    alphaRe = 0.0;
    betaRe = 0.0;
    gammaRe = 0.0;

    // Imaginary part
    alphaIm = -k;
    betaIm = 0.0;
    gammaIm = 0.0;
#endif
    
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

  void CreateInterfaceMatrices(const int interfaceIndex, const bool fullAssembly);
  
  // Create operator C_{sd0,sd1} in the block space corresponding to [u_m^s, f_i, \rho_i]. Note that the u_m^I blocks are omitted (just zeros).
  Operator* CreateCij(const int localInterfaceIndex, const int orientation);

#ifdef SDFOSLS
  // Create operator C_{sd0,sd1} in the block space corresponding to [E_m^s, H_m^s, f_i, \rho_i] x [E_m^s, f_i, \rho_i].
  Operator* CreateCij_FOSLS(const int localInterfaceIndex, const int orientation, const bool fullAssembly);
#endif
  
  // Create operator C_{sd0,sd1} R_{sd1}^T. The operator returned here is of size n_{sd0} by n_{sd1}, where n_{sd} is the sum of
  // tdofsBdry[sd].size() and ifespace[interfaceIndex]->GetTrueVSize() and iH1fespace[interfaceIndex]->GetTrueVSize() for all interfaces of subdomain sd.
  Operator* CreateInterfaceOperator(const int sd0, const int sd1, const int localInterfaceIndex, const int interfaceIndex, const int orientation, const bool fullAssembly);

  void CreateSubdomainMatrices(const int subdomain, const bool fullAssembly);

#ifdef SD_ITERATIVE
  Operator* CreateSubdomainImaginaryPart(const int subdomain);
#endif

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
#else
  Solver* CreateSerialUMFPackSolver(HypreParMatrix *A, MPI_Comm comm)
  {
    int nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MFEM_VERIFY(nprocs == 1, "");

    SparseMatrix *Asp = new SparseMatrix();
    A->GetDiag(*Asp);  // Asp does not own the data

    UMFPackSolver *umf_solver = new UMFPackSolver();
    umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_solver->SetOperator(*Asp);
    return umf_solver;
  }
#endif

  /*
  STRUMPACKSolver* CreateStrumpackSolverApprox(Operator *Arow, MPI_Comm comm)
  {
    //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, comm);
    STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, comm);
    strumpack->SetPrintFactorStatistics(true);
    strumpack->SetPrintSolveStatistics(true);
    strumpack->SetHSS();
    strumpack->SetKrylovSolver(strumpack::KrylovSolver::PREC_GMRES);
    strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
    strumpack->SetOperator(*Arow);
    strumpack->SetFromCommandLine();
    return strumpack;
  }
  
  STRUMPACKSolver* CreateStrumpackSolverApproxV2(Operator *Arow, MPI_Comm comm)
  {
    //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, comm);
    STRUMPACKSolver * strumpack = new STRUMPACKSolver(0, NULL, comm);
    strumpack->SetPrintFactorStatistics(true);
    strumpack->SetPrintSolveStatistics(true);
    strumpack->SetHSS();
    strumpack->SetHssAbsTol(0.0);
    strumpack->SetHssRelTol(1e-4);
    strumpack->SetAbsTol(0.0);
    strumpack->SetRelTol(1e-6);
    strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
    strumpack->DisableMatching();
    strumpack->SetOperator(*Arow);
    strumpack->SetFromCommandLine();
    return strumpack;
  }
  */
  
#ifdef MFEM_USE_STRUMPACK
  Solver* CreateSubdomainPreconditionerStrumpack(const int subdomain);
#endif

  void SetOffsetsSD(const int subdomain);
#ifdef SD_ITERATIVE  
  void SetOffsetsAuxSD(const int subdomain);
#endif
  
  //#define SCHURCOMPSD
  
  // Create operator A_m for subdomain m, in the block space corresponding to [u_m, f_m^s, \rho_m^s].
  // We use mappings between interface and subdomain boundary DOF's, so there is no need for interior and surface blocks on each subdomain.
  Operator* CreateSubdomainOperator(const int subdomain);
#ifdef SD_ITERATIVE_GMG_PA
  BlockOperator* CreateSubdomainOperatorPA(const int subdomain);
#endif
  
#ifdef HYPRE_PARALLEL_ASDCOMPLEX
  void CreateSubdomainHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
				  //#ifdef SERIAL_INTERFACES
				  Array2D<SparseMatrix*>& blockSp,
				  //#endif
				  Array2D<double>& blockCoefficient);
#ifdef SD_ITERATIVE
  void CreateSubdomainAuxiliaryHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
					   Array2D<SparseMatrix*>& blockSp, Array2D<double>& blockCoefficient);
  
  void CreateSubdomainDiagHypreBlocks(const int subdomain, Array2D<HypreParMatrix*>& block,
				      Array2D<SparseMatrix*>& blockSp, Array2D<double>& blockCoefficient);
#endif
#endif
  
  // This is the same operator as CreateSubdomainOperator, except it is stored as a strumpack matrix rather than a block operator. 
#ifdef MFEM_USE_STRUMPACK
  Operator* CreateSubdomainOperatorStrumpack(const int subdomain);
#endif

  void CheckContinuityOfUS(const Vector & solReduced, const bool imag);
};

class DDMPreconditioner : public Solver
{
public:
  DDMPreconditioner(DDMInterfaceOperator *ddi_) : ddi(ddi_)
  {
  }
  
  virtual void Mult(const Vector &x, Vector &y) const
  {
#ifdef GPWD
    ddi->GPWD_PreconditionerMult(x, y);
#else
    ddi->GaussSeidelPreconditionerMult(x, y);
    //ddi->PolyPreconditionerMult(x, y);
#endif  // GPWD
  }

  virtual void SetOperator(const Operator &op)
  {
  }

private:
  DDMInterfaceOperator *ddi;
};

void Airy_epsilon(const Vector &x, Vector &e);
void Airy_epsilon2(const Vector &x, Vector &e);

class FOSLSSolver : public Solver
{
public:
  FOSLSSolver(MPI_Comm comm, ParFiniteElementSpace *fespace_, Array<int>& ess_tdof_list_E,
	      ParGridFunction& Eexact, std::vector<HypreParMatrix*> const& P,
#ifdef SDFOSLS_PA
	      Array2D<HypreParMatrix*> const& blockCoarseA, const bool fullAssembly,
#endif
	      const double omega_)
    : Solver(4 * fespace_->GetTrueVSize()), M_inv(comm), fespace(fespace_),
      n(fespace_->GetTrueVSize()), nfull(fespace_->GetVSize()), LSpcg(comm),
      omega(omega_)
  {
    z.SetSize(n);
    Minv_x.SetSize(n);
    rhs_E.SetSize(n);
    
    ParMesh *pmesh = fespace->GetParMesh();
    int dim = pmesh->Dimension();
    int sdim = pmesh->SpaceDimension();

    ConstantCoefficient pos(omega);
    ConstantCoefficient sigma(omega*omega);

#ifdef AIRY_TEST
    VectorFunctionCoefficient epsilon(dim, Airy_epsilon);
    VectorFunctionCoefficient epsilonT(dim, Airy_epsilon);  // transpose of epsilon
    VectorFunctionCoefficient epsilon2(dim, Airy_epsilon2);
    ScalarVectorProductCoefficient coeff(pos,epsilon);
    ScalarVectorProductCoefficient coeffT(pos,epsilonT);
    ScalarVectorProductCoefficient coeff2(sigma,epsilon2);
#else
    // TODO
#endif

    // Boundary attribute 1 means face is interior to the domain; 2 means face is on the exterior boundary.
    
    MFEM_VERIFY(pmesh->bdr_attributes.Max() == 1 || pmesh->bdr_attributes.Max() == 2, "");
    Array<int> bdr_marker(pmesh->bdr_attributes.Max());
    bdr_marker[0] = 1;  // add boundary integrators only in the interior of the domain
    if (pmesh->bdr_attributes.Max() == 2)
      {
#ifdef TEST_DECOUPLED_FOSLS_PROJ_FULL
	bdr_marker[1] = 1;
#else
#ifdef ZERO_ORDER_FOSLS
	bdr_marker[1] = 1;
#else
	bdr_marker[1] = 0;
#endif
#endif
      }

#ifdef IFFOSLS
    // Note that bdr_marker marks interfaces only. 
    Array<int> bdr_marker_everywhere(pmesh->bdr_attributes.Max());
    bdr_marker_everywhere = 1;
#endif
    
#if defined ZERO_ORDER_FOSLS || defined IFFOSLS
    Array<int> bdr_marker_ext(pmesh->bdr_attributes.Max());
    bdr_marker_ext[0] = 0;  // add boundary integrators only on the exterior boundary
    if (pmesh->bdr_attributes.Max() == 2)
      {
	bdr_marker_ext[1] = 1;
      }
#endif

    Array<int> ess_tdof_list_empty;  // empty

#ifndef SDFOSLS_PA
    bM = new ParBilinearForm(fespace);
    bM->AddDomainIntegrator(new VectorFEMassIntegrator());
    bM->Assemble();
    bM->Finalize();

    bM_eps = new ParBilinearForm(fespace);
    bM_eps->AddDomainIntegrator(new VectorFEMassIntegrator(epsilonT));
    bM_eps->Assemble();
    bM_eps->Finalize();

    bM_curl = new ParMixedBilinearForm(fespace,fespace);
    bM_curl->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator()); 
    bM_curl->Assemble();
    bM_curl->Finalize();
    
    bM->FormSystemMatrix(ess_tdof_list_empty, M);
    //bM_curl->FormColSystemMatrix(ess_tdof_list_empty, M_curl);
    //bM_curl->FormSystemMatrix(ess_tdof_list_empty, M_curl);  // TODO: if nonempty list is used, then this needs to be FormColSystemMatrix.
    bM_curl->FormRectangularSystemMatrix(ess_tdof_list_empty, ess_tdof_list_empty, M_curl);
    
    bM->FormSystemMatrix(ess_tdof_list_E, M_Ebc);
    bM_eps->FormSystemMatrix(ess_tdof_list_E, M_eps_Ebc);

    M_inv.SetAbsTol(1.0e-12);
    M_inv.SetRelTol(1.0e-12);
    M_inv.SetMaxIter(100);
    M_inv.SetOperator(M_Ebc);
    M_inv.SetPrintLevel(0);
#endif
    
    block_trueOffsets.SetSize(5);
    
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = n;
    block_trueOffsets[2] = n;
    block_trueOffsets[3] = n;
    block_trueOffsets[4] = n;
    block_trueOffsets.PartialSum();

    trueRhs = new BlockVector(block_trueOffsets);
    trueSol = new BlockVector(block_trueOffsets);

    //    _           _    _  _       _  _
    //   |             |  |    |     |    |
    //   |  A00   A01  |  | E  |     |F_E |
    //   |             |  |    |  =  |    |
    //   |  A10   A11  |  | H  |     |F_G |
    //   |_           _|  |_  _|     |_  _|
    //
    // A00 = (curl E, curl F) + \omega^2 (E,F)
    // A01 = - \omega *( (curl E, F) + (E,curl F)
    // A10 = - \omega *( (curl H, G) + (H,curl G)
    // A11 = (curl H, curl G) + \omega^2 (H,G)

    ParBilinearForm *a_EE = new ParBilinearForm(fespace);
    //a_EE = new ParBilinearForm(fespace);
    ParBilinearForm *a_HH = new ParBilinearForm(fespace);
    ParMixedBilinearForm *a_mix1 = new ParMixedBilinearForm(fespace,fespace);
    ParMixedBilinearForm *a_mix2 = new ParMixedBilinearForm(fespace,fespace);

#ifdef SDFOSLS_PA
    const bool pa = !fullAssembly;

    if (pa)
      {
	a_EE->SetAssemblyLevel(AssemblyLevel::PARTIAL);
	a_HH->SetAssemblyLevel(AssemblyLevel::PARTIAL);
	a_mix1->SetAssemblyLevel(AssemblyLevel::PARTIAL);
	a_mix2->SetAssemblyLevel(AssemblyLevel::PARTIAL);

	diag_PA_EE.SetSize(fespace->GetTrueVSize());
	diag_PA_HH.SetSize(fespace->GetTrueVSize());

	diag_pa.resize(4);
	diag_pa[0] = &diag_PA_EE;
	diag_pa[1] = &diag_PA_HH;
	diag_pa[2] = &diag_PA_EE;
	diag_pa[3] = &diag_PA_HH;
      }
#else
    const bool pa = false;
#endif

    a_EE->AddDomainIntegrator(new CurlCurlIntegrator());
    a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(coeff2));
#ifndef TEST_DECOUPLED_FOSLS
#ifdef IFFOSLS
#ifdef IFFOSLS_ESS
    a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator(pos), bdr_marker);
#else
    {
      //a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator(pos), bdr_marker_everywhere);//???
      VectorFEMassIntegrator *binteg = new VectorFEMassIntegrator(pos);
      binteg->isBdryInteg = true;
      binteg->el_marker = &bdr_marker_everywhere;
      a_EE->AddBoundaryIntegrator(binteg, bdr_marker_everywhere);

      // TODO: does the PA diagonal include the boundary integrator?
    }
#endif
#else
    a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator(pos), bdr_marker);
#endif
#endif
    a_EE->Assemble();
    if (!pa) a_EE->Finalize();
    //HypreParMatrix *A_EE = new HypreParMatrix;
    //a_EE->FormSystemMatrix(ess_tdof_list_E, *A_EE);
    //HypreParMatrix *A_EE = NULL;
    //OperatorPtr A_EE;
    {
      ParLinearForm b_E(fespace);
#ifndef AIRY_TEST
      MFEM_VERIFY(false, "TODO: add linear form integrator to RHS");
#endif
      VectorFunctionCoefficient E(3, test2_E_exact);
      ScalarVectorProductCoefficient kE(pos,E);
#ifdef ZERO_ORDER_FOSLS
#ifdef TEST_DECOUPLED_FOSLS_PROJ
      b_E.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(kE), bdr_marker);
#else
      b_E.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(kE), bdr_marker_ext);
#endif
#else
#ifdef IFFOSLS
#ifndef IFFOSLS_ESS
      b_E.AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(kE), bdr_marker_ext);
#endif
#endif
#endif
      
      b_E.Assemble();
      //OperatorPtr Aptr;
      Vector X, B;
      //a_EE->FormLinearSystem(ess_tdof_list_E, Eexact, b_E, A_EE, X, rhs_E);
      a_EE->FormLinearSystem(ess_tdof_list_E, Eexact, b_E, A_EE, X, B);
      MFEM_VERIFY(rhs_E.Size() == B.Size(), "");
      rhs_E = B; // necessary since B uses data from b_E and both go out of scope
      
      //a_EE->FormLinearSystem(ess_tdof_list_E, Eexact, b_E, Aptr, X, rhs_E);
      //A_EE = Aptr.As<HypreParMatrix>();
    }

    //#if defined ZERO_ORDER_FOSLS_COMPLEX || defined IFFOSLS
#ifdef ZERO_ORDER_FOSLS_COMPLEX
    ParBilinearForm *a_bM = new ParBilinearForm(fespace);
    a_bM->AddBoundaryIntegrator(new VectorFEMassIntegrator(pos), bdr_marker);
    a_bM->Assemble();
    a_bM->Finalize();
    HypreParMatrix *A_bM = new HypreParMatrix;
    a_bM->FormSystemMatrix(ess_tdof_list_E, *A_bM);
#endif

    /*
    { // Debugging
      int ninter = 0;
      int nexter = 0;
      int nundef = 0;
      for (int i = 0; i < fespace -> GetNBE(); i++)
      {
         const int bdr_attr = pmesh->GetBdrAttribute(i);
         if (bdr_marker[bdr_attr-1] == 0)
	   ninter++;
	 else if (bdr_marker[bdr_attr-1] == 1)
	   nexter++;
	 else
	   nundef++;
      }

      int nprocs, rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
      
      cout << rank << ": dbg bdr marker " << ninter << " " << nexter << " " << nundef << endl;
    }
    */
    
#ifdef IMPEDANCE_POSITIVE
    ConstantCoefficient negOne(-1.0);
    ParBilinearForm *a_EEm = new ParBilinearForm(fespace);
    a_EEm->AddDomainIntegrator(new CurlCurlIntegrator());
    a_EEm->AddDomainIntegrator(new VectorFEMassIntegrator(coeff2));
    a_EEm->AddBoundaryIntegrator(new VectorFEMassIntegrator(negOne));
    a_EEm->Assemble();
    a_EEm->Finalize();
    HypreParMatrix *A_EEm = new HypreParMatrix;
    a_EEm->FormSystemMatrix(ess_tdof_list, *A_EEm);

    ParBilinearForm *a_HHm = new ParBilinearForm(fespace);
    a_HHm->AddDomainIntegrator(new CurlCurlIntegrator());
    a_HHm->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    a_HHm->AddBoundaryIntegrator(new VectorFEMassIntegrator(negOne));
    a_HHm->Assemble();
    a_HHm->Finalize();
    HypreParMatrix *A_HHm = new HypreParMatrix;
    a_HHm->FormSystemMatrix(ess_tdof_list, *A_HHm);
#endif
    
    a_HH->AddDomainIntegrator(new CurlCurlIntegrator());
    a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
#ifndef ZERO_ORDER_FOSLS
#ifdef IFFOSLS_H
    {
      //a_HH->AddBoundaryIntegrator(new VectorFEMassIntegrator(pos), bdr_marker);
      VectorFEMassIntegrator *binteg = new VectorFEMassIntegrator(pos);
      binteg->isBdryInteg = true;
      binteg->el_marker = &bdr_marker;
      a_HH->AddBoundaryIntegrator(binteg, bdr_marker);
    }
#endif
#endif
    
    a_HH->Assemble();
    if (!pa) a_HH->Finalize();
    //HypreParMatrix *A_HH = new HypreParMatrix;
    //OperatorPtr A_HH;
    //a_HH->FormSystemMatrix(ess_tdof_list_empty, *A_HH);
    a_HH->FormSystemMatrix(ess_tdof_list_empty, A_HH);

#ifdef SDFOSLS_PA
    if (pa)
      {
	ParGridFunction diag_gf(fespace);
	diag_gf = 0.0;
	a_EE->AssembleDiagonal(diag_gf);
	diag_gf.GetTrueDofs(diag_PA_EE);

	diag_gf = 0.0;
	a_HH->AssembleDiagonal(diag_gf);
	diag_gf.GetTrueDofs(diag_PA_HH);
      }
#endif

#ifndef IFFOSLS
    ParBilinearForm *a_tang = new ParBilinearForm(fespace);
    //ParMixedBilinearForm *a_tang = new ParMixedBilinearForm(fespace, fespace);
    //bdr_marker[1] = 1;  // TODO: remove
    a_tang->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator(1.0), bdr_marker);
    a_tang->Assemble();
    a_tang->Finalize();

    OperatorHandle A_tang_ptr;
    a_tang->FormSystemMatrix(ess_tdof_list_empty, A_tang_ptr);
    HypreParMatrix *A_tang = A_tang_ptr.As<HypreParMatrix>();

    HypreParMatrix *A_tang_Ecol = A_tang->EliminateCols(ess_tdof_list_E);

    HypreParMatrix *A_tang_Erow = A_tang_Ecol->Transpose();  // other rotation
#endif
    
    // (k curl u, eps v) + (k u, curl v)
    a_mix1->AddDomainIntegrator(new MixedVectorCurlIntegrator(coeffT)); // Not supported for PA!
    a_mix1->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(pos));
    a_mix1->Assemble();
    if (!pa) a_mix1->Finalize();
    HypreParMatrix *A_mix1 = new HypreParMatrix;
    //if (!pa) a_mix1->FormColSystemMatrix(ess_tdof_list_empty, *A_mix1);
    if (!pa)
      {
	OperatorPtr A_mix1_ptr;
	a_mix1->FormRectangularSystemMatrix(ess_tdof_list_empty, ess_tdof_list_empty, A_mix1_ptr);
	A_mix1 = A_mix1_ptr.As<HypreParMatrix>();
      }
    
    // (k curl u, v) + (k eps u, curl v)
    a_mix2->AddDomainIntegrator(new MixedVectorCurlIntegrator(pos));
    a_mix2->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(coeff));
    a_mix2->Assemble();
    if (!pa) a_mix2->Finalize();
    HypreParMatrix *A_mix2 = new HypreParMatrix;
    //if (!pa) a_mix2->FormColSystemMatrix(ess_tdof_list_E, *A_mix2);
    if (!pa)
      {
	OperatorPtr A_mix2_ptr;
	a_mix2->FormRectangularSystemMatrix(ess_tdof_list_E, ess_tdof_list_empty, A_mix2_ptr);
	A_mix2 = A_mix2_ptr.As<HypreParMatrix>();
      }

#ifdef IFFOSLS
    HypreParMatrix *A_mix1_E = NULL;
    if (!pa) A_mix1_E = A_mix2->Transpose();
#endif

    /*
    { // TODO: remove
      int nprocs, rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

      //for (int i=0; i<pmesh->GetNEdges(); ++i)
	{
	  int i = 299;
	  Array<int> vert;
	  pmesh->GetEdgeVertices(i, vert);
	  MFEM_VERIFY(vert.Size() == 2, "");
	  
	  double *v1 = pmesh->GetVertex(vert[0]);
	  double *v2 = pmesh->GetVertex(vert[1]);
	  cout << rank << ": edge " << i << " (" << v1[0] << ", " << v1[1] << ", " << v1[2] << ") -> ("
	       << v2[0] << ", " << v2[1] << ", " << v2[2] << ")" << endl;

	  Array<int> dofs;
	  fespace->GetEdgeDofs(i, dofs);
	  MFEM_VERIFY(dofs.Size() == 2, "");
	  
	  int tdofs[2];
	  for (int j=0; j<2; ++j)
	    tdofs[j] = fespace->GetLocalTDofNumber(dofs[j]);
	  
	  cout << rank << ": edge " << i << " DOFs " << dofs[0] << ", " << dofs[1] << endl;
	  cout << rank << ": edge " << i << " tDOFs " << tdofs[0] << ", " << tdofs[1] << endl;
	}
      
      if (rank == -10)
	{
	  std::string filenameE = "essE0.txt";
	  std::ofstream sfile(filenameE, std::ofstream::out);
	  ess_tdof_list_E.Print(sfile);
	  
	  std::string filename = "AEH0.txt";
	  A_mix1_E->Print(filename.c_str());  // TODO: remove

	  std::string filename2 = "AHE0.txt";
	  A_mix2->Print(filename2.c_str());  // TODO: remove

	  std::string filename3 = "AEE0.txt";
	  A_EE->Print(filename3.c_str());  // TODO: remove
	}
    }
    */
    
    LS_Maxwellop = new BlockOperator(block_trueOffsets);
    const int numBlocks = 4;

#ifdef IMPEDANCE_OTHER_SIGN
    LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(1, 0, A_mix2, -1.0); // no bc
    LS_Maxwellop->SetBlock(3, 0, A_tang);
    LS_Maxwellop->SetBlock(0, 1, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HH);
#ifdef IMPEDANCE_POSITIVE
    LS_Maxwellop->SetBlock(2, 1, A_tang, -1.0);  // other rotation
#else
    LS_Maxwellop->SetBlock(2, 1, A_tang);  // other rotation
#endif    
    LS_Maxwellop->SetBlock(1, 2, A_tang, -1.0);
#ifdef IMPEDANCE_POSITIVE
    LS_Maxwellop->SetBlock(2, 2, A_EEm);
#else
    LS_Maxwellop->SetBlock(2, 2, A_EE);
#endif
    LS_Maxwellop->SetBlock(3, 2, A_mix2, -1.0);  // no bc
    LS_Maxwellop->SetBlock(0, 3, A_tang, -1.0);  // other rotation
    LS_Maxwellop->SetBlock(2, 3, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HH);
#else  // not IMPEDANCE_OTHER_SIGN, i.e. original version
#ifdef IMPEDANCE_POSITIVE
    LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(1, 0, A_mix2, -1.0); // no bc
    LS_Maxwellop->SetBlock(3, 0, A_tang);
    LS_Maxwellop->SetBlock(0, 1, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HHm);
    LS_Maxwellop->SetBlock(2, 1, A_tang);  // other rotation
    LS_Maxwellop->SetBlock(1, 2, A_tang, -1.0);
    LS_Maxwellop->SetBlock(2, 2, A_EEm);
    LS_Maxwellop->SetBlock(3, 2, A_mix2, -1.0);  // no bc
    LS_Maxwellop->SetBlock(0, 3, A_tang);  // other rotation
    LS_Maxwellop->SetBlock(2, 3, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HHm);
#else 
#ifdef ZERO_ORDER_FOSLS
    LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(1, 0, A_mix2, -1.0); // no bc
    //LS_Maxwellop->SetBlock(3, 0, A_tang_Ecol, -omega);
    LS_Maxwellop->SetBlock(0, 1, A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HH);
    //LS_Maxwellop->SetBlock(2, 1, A_tang_Erow, omega);  // other rotation
    //LS_Maxwellop->SetBlock(1, 2, A_tang_Ecol, omega);
    LS_Maxwellop->SetBlock(2, 2, A_EE);
    LS_Maxwellop->SetBlock(3, 2, A_mix2, -1.0);  // no bc
    //LS_Maxwellop->SetBlock(0, 3, A_tang_Erow, -omega);  // other rotation
    LS_Maxwellop->SetBlock(2, 3, A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HH);
#ifdef ZERO_ORDER_FOSLS_COMPLEX
    /*
    LS_Maxwellop->SetBlock(2, 0, A_bM, -1.0);
    LS_Maxwellop->SetBlock(0, 2, A_bM);
    */

    LS_Maxwellop->SetBlock(3, 0, A_bM, -1.0);  // Er
    LS_Maxwellop->SetBlock(1, 2, A_bM);  // Ei

#endif
#else
#ifdef IFFOSLS
    //Operator *a_mix2_tr = pa ? new TransposeOperator(a_mix2) : NULL;  // TODO: AddMultTransposePA not implemented for the BilinearFormIntegrators in a_mix2. Instead, use mix1.
    
    //LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(0, 0, A_EE.Ptr());
    /*
    LS_Maxwellop->SetBlock(1, 1, A_HH.Ptr());
    LS_Maxwellop->SetBlock(2, 2, A_EE.Ptr());
    LS_Maxwellop->SetBlock(3, 3, A_HH.Ptr());
    */
    ///* 1234
    LS_Maxwellop->SetBlock(1, 0, pa ? (Operator*) a_mix2 : (Operator*) A_mix2, -1.0); // no bc
    //LS_Maxwellop->SetBlock(3, 0, A_tang_Ecol, -omega);
    //LS_Maxwellop->SetBlock(0, 1, pa ? a_mix2_tr : (Operator*) A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(0, 1, pa ? a_mix1 : (Operator*) A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HH.Ptr());
    //LS_Maxwellop->SetBlock(2, 1, A_tang_Erow, omega);  // other rotation
    //LS_Maxwellop->SetBlock(1, 2, A_tang_Ecol, omega);
    //LS_Maxwellop->SetBlock(2, 2, A_EE);
    LS_Maxwellop->SetBlock(2, 2, A_EE.Ptr());
    LS_Maxwellop->SetBlock(3, 2, pa ? (Operator*) a_mix2 : (Operator*) A_mix2, -1.0);  // no bc
    //LS_Maxwellop->SetBlock(0, 3, A_tang_Erow, -omega);  // other rotation
    //LS_Maxwellop->SetBlock(2, 3, pa ? a_mix2_tr : (Operator*) A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(2, 3, pa ? a_mix1 : (Operator*) A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HH.Ptr());
    //*/
    /*
    // TODO: If BC are defined for E, then for A_bM put 0 on diagonal for eliminated entries. 
    LS_Maxwellop->SetBlock(2, 0, A_bM, -1.0);  // Er
    LS_Maxwellop->SetBlock(0, 2, A_bM);  // Ei
    
#ifdef IFFOSLS_H
    LS_Maxwellop->SetBlock(3, 1, A_bM, -1.0);  // Hr
    LS_Maxwellop->SetBlock(1, 3, A_bM);  // Hi
#endif
    */
#else  // original version (obsolete and broken)
    /*
    LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(1, 0, A_mix2, -1.0); // no bc
    LS_Maxwellop->SetBlock(3, 0, A_tang_Ecol, -omega);
    LS_Maxwellop->SetBlock(0, 1, A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HH);
    LS_Maxwellop->SetBlock(2, 1, A_tang_Erow, omega);  // other rotation
    LS_Maxwellop->SetBlock(1, 2, A_tang_Ecol, omega);
    LS_Maxwellop->SetBlock(2, 2, A_EE);
    LS_Maxwellop->SetBlock(3, 2, A_mix2, -1.0);  // no bc
    LS_Maxwellop->SetBlock(0, 3, A_tang_Erow, -omega);  // other rotation
    LS_Maxwellop->SetBlock(2, 3, A_mix1_E, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HH);
    */
#endif
#endif
#endif
#endif
   
    // Set up the preconditioner
#ifdef SDFOSLS_PA
    Array2D<Operator*> blockA(numBlocks, numBlocks);
#else
    Array2D<HypreParMatrix*> blockA(numBlocks, numBlocks);
#endif
    Array2D<double> blockAcoef(numBlocks, numBlocks);

    for (int i=0; i<numBlocks; ++i)
      {
	for (int j=0; j<numBlocks; ++j)
	  {
	    if (LS_Maxwellop->IsZeroBlock(i,j) == 0)
	      {
#ifdef SDFOSLS_PA
		blockA(i,j) = &(LS_Maxwellop->GetBlock(i,j));
#else
		blockA(i,j) = static_cast<HypreParMatrix *>(&LS_Maxwellop->GetBlock(i,j));
#endif
		blockAcoef(i,j) = LS_Maxwellop->GetBlockCoef(i,j);
	      }
	    else
	      {
		blockA(i,j) = NULL;
		blockAcoef(i,j) = 1.0;
	      }
	  }
      }

    LSpcg.SetAbsTol(1.0e-12);
    LSpcg.SetRelTol(1.0e-8);
    LSpcg.SetMaxIter(2000);
    LSpcg.SetOperator(*LS_Maxwellop);
    LSpcg.SetPrintLevel(0);
    
#ifdef FOSLS_DIRECT_SOLVER
    std::vector<std::vector<int> > blockProcOffsets(numBlocks);
    std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
    Array2D<SparseMatrix*> Asp;
    Asp.SetSize(numBlocks,numBlocks);

    {
      int nprocs, rank;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &nprocs);

      std::vector<int> allnumrows(nprocs);
      const int blockNumRows = n;
      MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, comm);

      for (int b=0; b<numBlocks; ++b)
	{
	  blockProcOffsets[b].resize(nprocs);
	  all_block_num_loc_rows[b].resize(nprocs);

	  for (int j=0; j<numBlocks; ++j)
	    Asp(b,j) = NULL;
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
   
    LSH = blockgmg::CreateHypreParMatrixFromBlocks2(comm, block_trueOffsets, blockA, Asp,
					 blockAcoef, blockProcOffsets, all_block_num_loc_rows);

    invLSH = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*LSH), comm);
#else
#ifdef SDFOSLS_PA
    if (!fullAssembly)
      {
	blockgmg::BlockMGPASolver *precMG = new blockgmg::BlockMGPASolver(comm, LS_Maxwellop->Height(), LS_Maxwellop->Width(), blockA, blockAcoef, blockCoarseA, P, diag_pa, ess_tdof_list_empty);
	precMG->SetTheta(0.5);
	LSpcg.SetPreconditioner(*precMG);
	LSpcg.iterative_mode = true;
#ifdef FOSLS_INIT_GUESS
	initialGuess.SetSize(LS_Maxwellop->Height());
	initialGuess = 0.0;
#endif
      }
#else
    blockgmg::BlockMGSolver *precMG = new blockgmg::BlockMGSolver(comm, LS_Maxwellop->Height(), LS_Maxwellop->Width(), blockA, blockAcoef, P);
    LSpcg.SetPreconditioner(*precMG);
#endif
#endif
  }
  
  void SetOperator(const Operator &op) { }

  void Mult(const Vector &x, Vector &y) const
  {
    MFEM_VERIFY(x.Size() == 4*n, "");

#ifdef FOSLS_DIRECT_SOLVER
    invLSH->Mult(x, y);
#else
#ifdef FOSLS_INIT_GUESS
    y = initialGuess;
#endif
    LSpcg.Mult(x, y);
#ifdef FOSLS_INIT_GUESS
    initialGuess = y;
#endif
#endif
  }
  
  void MultSolver(const Vector &x, Vector &y) const
  {
    // Solve (curl E, curl v) - k^2 (eps E, v) + ik <pi(u), pi(v)> = (x, v), with no BC,
    // where x is complex, using FOSLS. This is the Galerkin discretization of
    // curl curl u - k^2 eps u = x, with ik n x u x n - n x curl u = 0 on the boundary.

    MFEM_VERIFY(x.Size() == 2*n, "");

    (*trueRhs) = 0.0;
    
    for (int i=0; i<n; ++i)
      z[i] = x[i];  // Set z = x_Re
    
    M_inv.Mult(z, Minv_x);
    M_eps_Ebc.Mult(Minv_x, z);

    trueRhs->GetBlock(0) -= z;

    M_curl->Mult(Minv_x, z);
    z *= 1.0 / omega;  // TODO: essential DOF's?
    
    trueRhs->GetBlock(1) = z;

    for (int i=0; i<n; ++i)
      z[i] = x[n + i];  // Set z = x_Im
    
    M_inv.Mult(z, Minv_x);
    M_eps_Ebc.Mult(Minv_x, z);

    trueRhs->GetBlock(2) -= z;

    M_curl->Mult(Minv_x, z);
    z *= 1.0 / omega;  // TODO: essential DOF's?

    trueRhs->GetBlock(3) += z;
    
#ifdef FOSLS_DIRECT_SOLVER
    invLSH->Mult(*trueRhs, *trueSol);
#else
    LSpcg.Mult(*trueRhs, *trueSol);
#endif
    
    for (int i=0; i<n; ++i)
      y[i] = trueSol->GetBlock(0)[i];  // Set y_Re = E_Re

    for (int i=0; i<n; ++i)
      y[n + i] = trueSol->GetBlock(2)[i];  // Set y_Im = E_Im
  }

#ifdef MFEM_USE_STRUMPACK
  // TODO: just have one version of this 
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

  void GetMatrixPointers(Array2D<HypreParMatrix*>& A)
  {
    MFEM_VERIFY(LS_Maxwellop->NumRowBlocks() == 4 && LS_Maxwellop->NumColBlocks() == 4, "");
    const int n = 4;
    A.SetSize(n,n);

    for (int i=0; i<n; ++i)
      for (int j=0; j<n; ++j)
	{
	  if (LS_Maxwellop->IsZeroBlock(i,j))
	    A(i,j) = NULL;
	  else
	    A(i,j) = static_cast<HypreParMatrix *>(&LS_Maxwellop->GetBlock(i,j));
	}
  }
  
  Vector rhs_E;  // real part
  
  BlockOperator *LS_Maxwellop;
  
private:

  BlockVector *trueRhs, *trueSol;

  Array<int> block_trueOffsets;

  ParBilinearForm *bM, *bM_eps;
  ParMixedBilinearForm *bM_curl;
  
  HypreParMatrix M;
  OperatorPtr M_curl;
  HypreParMatrix M_Ebc, M_eps_Ebc;

  CGSolver M_inv;

  const int n;
  const int nfull;  // TODO: not used?
  
  mutable Vector z, Minv_x;
  
  CGSolver LSpcg;
  mutable Vector initialGuess;
  
  STRUMPACKSolver *invLSH;
  HypreParMatrix *LSH;

  ParFiniteElementSpace *fespace;

#ifdef SDFOSLS_PA
  std::vector<Vector*> diag_pa;
  Vector diag_PA_EE, diag_PA_HH;
#endif

  //ParBilinearForm *a_EE;
  
  OperatorPtr A_EE, A_HH;
  
  const double omega;
};


class LowerBlockTriangularSubdomainSolver : public Operator
{
public:
#ifdef IFFOSLS_H
  LowerBlockTriangularSubdomainSolver(HypreParMatrix *fullMat_, FOSLSSolver *sdInv_, Operator *auxInv_,
				      Operator *tdofsBdryInjectionTranspose_, std::set<int> *allGlobalSubdomainInterfaces_,
				      std::vector<InjectionOperator*> *InterfaceToSurfaceInjection_, std::vector<int> *ifNDtrue_,
				      std::vector<int> *ifH1true_)
    : fullMat(fullMat_), sdInv(sdInv_), auxInv(auxInv_), allGlobalSubdomainInterfaces(allGlobalSubdomainInterfaces_),
      tdofsBdryInjectionTranspose(tdofsBdryInjectionTranspose_), InterfaceToSurfaceInjection(InterfaceToSurfaceInjection_),
      ifNDtrue(ifNDtrue_), ifH1true(ifH1true_)
#else
  LowerBlockTriangularSubdomainSolver(HypreParMatrix *fullMat_, FOSLSSolver *sdInv_, Operator *auxInv_)
    : fullMat(fullMat_), sdInv(sdInv_), auxInv(auxInv_)
#endif
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef IFFOSLS_H
    MFEM_VERIFY(tdofsBdryInjectionTranspose != NULL, "");

    int os = 0;
    for (auto interfaceIndex : *allGlobalSubdomainInterfaces)
      {
	fOS.push_back(os);
	os += 2 * ((*ifNDtrue)[interfaceIndex] + (*ifH1true)[interfaceIndex]);
      }
#endif
    
    nSD = sdInv->Height() / 2;  // Discarding the H result
    nSDhalf = nSD / 2;   // Size of Er or Ei
    nAux = auxInv->Height();  // complex size
    nAuxHalf = nAux / 2;
    
    height = nSD + nAux;
    width = height + nSD;

#ifndef IFFOSLS
    MFEM_VERIFY(height == fullMat->Height(), "");
#endif
    
    tmp.SetSize(height);
    
    xSD.SetSize(2*nSD);
    ySD.SetSize(2*nSD);

    xAux.SetSize(nAux);
    yAux.SetSize(nAux);

#ifdef IFFOSLS_H
    MFEM_VERIFY(os == nAux, "");
#endif
  }

  virtual void Mult(const Vector & x, Vector & y) const
  {
    MFEM_VERIFY(x.Size() == width && y.Size() == height, "");
    MFEM_VERIFY(2 * (nSDhalf + nAuxHalf) == height, "");
    
    const int widthHalf = width / 2;

    /*
    for (int i=0; i<nSD; ++i)
      {
	xSD[i] = x[i];  // E and H real parts
	xSD[nSD + i] = x[widthHalf + i];  // E and H imaginary parts
      }
    */

    xSD.SetOffset(1.0, x, 0, 0, nSD);
    xSD.SetOffset(1.0, x, widthHalf, nSD, nSD);

    /*    
    auto xhost = x.HostRead();

    for (int i=0; i<nSD; ++i)
      {
	//xSD[i] = x[i];  // E and H real parts
	//xSD[nSD + i] = x[widthHalf + i];  // E and H imaginary parts
	xSD[i] = xhost[i];  // E and H real parts
	xSD[nSD + i] = xhost[widthHalf + i];  // E and H imaginary parts
      }
    */

    /*
    {
      Vector t(nSDhalf);
      Vector At(nSDhalf);
      //t = 0.0;
      //t[0] = 1.0;
      t = 1.0;
      At = 0.0;
      sdInv->LS_Maxwellop->GetBlock(1,1).Mult(t, At);
      //sdInv->LS_Maxwellop->Mult(t, At);
      cout << rank << ": LBTS A_EE * 1 norm " << At.Norml2() << endl;
      if (rank == 0)
	{
	  std::string filename = "Atpa7.txt";
	  std::ofstream sfile(filename, std::ofstream::out);
	  At.Print(sfile);
	  sfile.close();
	}
    }
    */
    
    //cout << rank << ": LBTS xSD norm " << xSD.Norml2() << endl;
    
    sdInv->Mult(xSD, ySD);  // Solve (0,0) block

    //cout << rank << ": LBTS ySD norm " << ySD.Norml2() << endl;

    y = 0.0;

    /*
    for (int i=0; i<nSDhalf; ++i)
      {
	y[i] = ySD[i];  // Er
	y[nSDhalf + nAuxHalf + i] = ySD[nSD + i];  // Ei
      }
    */
    y.SetOffset(1.0, ySD, 0, 0, nSDhalf);
    y.SetOffset(1.0, ySD, nSD, nSDhalf + nAuxHalf, nSDhalf);

#ifdef ZERO_ORDER_FOSLS
    for (int i=0; i<nAuxHalf; ++i)
      {
	y[nSDhalf + i] = x[nSD + i];  // real part
	y[nSD + nAuxHalf + i] = x[widthHalf + nSD + i];  // imaginary part
      }
#else
#ifdef IFFOSLS
#ifdef IFFOSLS_H
    /*
    for (int i=0; i<nAuxHalf; ++i)  // Apply identity for rho components, with f components overwritten below. 
      {
	y[nSDhalf + i] = x[nSD + i];  // real part
	y[nSD + nAuxHalf + i] = x[widthHalf + nSD + i];  // imaginary part
      }
    */

    y.SetOffset(1.0, x, nSD, nSDhalf, nAuxHalf);
    y.SetOffset(1.0, x, widthHalf + nSD, nSD + nAuxHalf, nAuxHalf);

    // f equation is f_{mn} = n x H_m x n (only in A, not C).

    //MFEM_VERIFY(fOS[fOS.size()-1] == nAux, "");

    {
      Vector ReH(nSDhalf);
      Vector ImH(nSDhalf);

      /*
      for (int i=0; i<nSDhalf; ++i)
	{
	  //ReH[i] = x[nSDhalf + i];
	  //ImH[i] = x[widthHalf + nSDhalf + i];
	  
	  ReH[i] = ySD[nSDhalf + i];
	  ImH[i] = ySD[nSD + nSDhalf + i];
	}
      */

      ReH.SetOffset(1.0, ySD, nSDhalf, 0, nSDhalf);
      ImH.SetOffset(1.0, ySD, nSD + nSDhalf, 0, nSDhalf);

      MFEM_VERIFY(tdofsBdryInjectionTranspose->Width() == nSDhalf, "");

      //cout << "ReH norm " << ReH.Norml2() << ", ImH " << ImH.Norml2() << endl;
      
      /*
      Vector ReHbdry(tdofsBdryInjectionTranspose->Height());
      Vector ImHbdry(tdofsBdryInjectionTranspose->Height());
	     
      tdofsBdryInjectionTranspose->Mult(ReH, ReHbdry);
      tdofsBdryInjectionTranspose->Mult(ImH, ImHbdry);
      */
      
      int os = 0;
      
      for (auto interfaceIndex : *allGlobalSubdomainInterfaces)
	{
	  //leftInjection->SetBlock(0, 0, new ProductOperator(tdofsBdryInjectionTranspose[sd0], InterfaceToSurfaceInjection[sd0][interfaceIndex], false, false));

	  MFEM_VERIFY((*ifNDtrue)[interfaceIndex] > 0, "");
	  MFEM_VERIFY((*ifNDtrue)[interfaceIndex] == (*InterfaceToSurfaceInjection)[interfaceIndex]->Width(), "");
	  MFEM_VERIFY(nSDhalf == (*InterfaceToSurfaceInjection)[interfaceIndex]->Height(), "");
  
	  Vector tmpx((*ifNDtrue)[interfaceIndex]);
	  //Vector tmpy((*ifNDtrue)[interfaceIndex]);

	  //Vector tmpu((*InterfaceToSurfaceInjection)[interfaceIndex]->Height());
	  
	  (*InterfaceToSurfaceInjection)[interfaceIndex]->MultTranspose(ReH, tmpx);
	  //tdofsBdryInjectionTranspose->Mult(tmpu);

	  /*
	  for (int i=0; i<(*ifNDtrue)[interfaceIndex]; ++i)
	    {
	      y[nSDhalf + os + i] = tmpx[i];  // real part
	    }
	  */
	  y.SetOffset(1.0, tmpx, 0, nSDhalf + os, (*ifNDtrue)[interfaceIndex]);

	  (*InterfaceToSurfaceInjection)[interfaceIndex]->MultTranspose(ImH, tmpx);

	  /*
	  for (int i=0; i<(*ifNDtrue)[interfaceIndex]; ++i)
	    {
	      y[nSD + nAuxHalf + os + i] = tmpx[i];  // imaginary part
	    }
	  */
	  y.SetOffset(1.0, tmpx, 0, nSD + nAuxHalf + os, (*ifNDtrue)[interfaceIndex]);

	  os += (*ifNDtrue)[interfaceIndex] + (*ifH1true)[interfaceIndex];
	  
	  //os += 2 * ((*ifNDtrue)[interfaceIndex] + (*ifH1true)[interfaceIndex]);
	}

      MFEM_VERIFY(os == nAuxHalf, "");
    }
#else
    for (int i=0; i<nAuxHalf; ++i)
      {
	y[nSDhalf + i] = x[nSD + i];  // real part
	y[nSD + nAuxHalf + i] = x[widthHalf + nSD + i];  // imaginary part
      }
#endif
#else
    tmp = 0.0;
    fullMat->Mult(y, tmp);  // Multiply solution for (0) block to get action of (1,0) block.

    for (int i=0; i<nAuxHalf; ++i)
      {
	xAux[i] = x[nSD + i] - tmp[nSDhalf + i];  // real part
	xAux[nAuxHalf + i] = x[widthHalf + nSD + i] - tmp[nSD + nAuxHalf + i];  // imaginary part
      }
    
    auxInv->Mult(xAux, yAux);

    for (int i=0; i<nAuxHalf; ++i)
      {
	y[nSDhalf + i] = yAux[i];
	y[nSD + nAuxHalf + i] = yAux[nAuxHalf + i];
      }
#endif
#endif
  }
  
private:
  //STRUMPACKSolver *auxInv;
  Operator *auxInv;
  FOSLSSolver *sdInv;
  int nSD, nSDhalf, nAux, nAuxHalf;
  int rank;
  
#ifdef IFFOSLS_H
  std::set<int> *allGlobalSubdomainInterfaces;
  Operator *tdofsBdryInjectionTranspose;  // TODO: remove?
  std::vector<InjectionOperator*> *InterfaceToSurfaceInjection;
  std::vector<int> *ifH1true;
  std::vector<int> *ifNDtrue;
  std::vector<int> fOS;  // TODO: remove?
#endif

  HypreParMatrix *fullMat;
  
  mutable Vector xSD, ySD, tmp, xAux, yAux;
};

  
#endif  // DDOPER_HPP
