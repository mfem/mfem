//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex3p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/mobius-strip.mesh -o 2 -f 0.1
//               mpirun -np 4 ex3p -m ../data/klein-bottle.mesh -o 2 -f 0.1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>

#include "ddmesh.hpp"
#include "ddoper.hpp"

//#include "testStrumpack.hpp"

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
double radiusFunction(const Vector &);
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;


void CreateInterfaceMeshes(const int numInterfaces, const int numSubdomains, const int myid, const int num_procs, SubdomainParMeshGenerator & sdMeshGen,
			   std::vector<int> & interfaceGlobalToLocalMap,
			   std::vector<int> const& interfaceGI, ParMesh **pmeshSD, std::vector<int> & interfaceFaceOffset,
			   vector<SubdomainInterface> & interfaces,
#ifdef SERIAL_INTERFACES
			   Mesh **smeshInterfaces
#else
			   ParMesh **pmeshInterfaces
#endif
			   );


#define NO_GLOBAL_FEM

//#define TEST_GMG

//#define SUBDOMAIN_MESH

#ifdef AIRY_TEST
//#define SIGMAVAL -686.3384931312 // 1.25 GHz
//#define SIGMAVAL (-2.0*686.3384931312) // 2.5 GHz
#define SIGMAVAL -10981.4158900991  // 5 GHz
//#define SIGMAVAL -43925.6635603965  // 10 GHz
//#define SIGMAVAL -175702.65424  // 20 GHz
//#define SIGMAVAL -1601.0
//#define SIGMAVAL -1009.0
//#define SIGMAVAL -211.0
//#define SIGMAVAL -2.0
//#define SIGMAVAL -1.0
#else
//#define SIGMAVAL -2.0
//#define SIGMAVAL -6007.0
//#define SIGMAVAL -191.0
#define SIGMAVAL -1009.0
//#define SIGMAVAL -511.0
//#define SIGMAVAL -6007.0
#endif

void test1_RHS_exact(const Vector &x, Vector &f)
{
  const double kappa = M_PI;
  const double sigma = SIGMAVAL;
  f(0) = (sigma + kappa * kappa) * sin(kappa * x(1));
  f(1) = (sigma + kappa * kappa) * sin(kappa * x(2));
  f(2) = (sigma + kappa * kappa) * sin(kappa * x(0));
}

void test2_RHS_exact(const Vector &x, Vector &f)
{
#ifdef AIRY_TEST
  f = 0.0;
#else
  const double pi = M_PI;
  const double sigma = SIGMAVAL;
  const double c = (2.0 * pi * pi) + sigma;

  f(0) = c * sin(pi * x(1)) * sin(pi * x(2));
  f(1) = c * sin(pi * x(2)) * sin(pi * x(0));
  f(2) = c * sin(pi * x(0)) * sin(pi * x(1));
#endif
  
  /*  
  f(0) = SIGMAVAL * x(1) * (1.0 - x(1)) * x(2) * (1.0 - x(2));
  f(1) = SIGMAVAL * x(0) * (1.0 - x(0)) * x(2) * (1.0 - x(2));
  f(2) = SIGMAVAL * x(1) * (1.0 - x(1)) * x(0) * (1.0 - x(0));

  f(0) += 2.0 * ((x(1) * (1.0 - x(1))) + (x(2) * (1.0 - x(2))));
  f(1) += 2.0 * ((x(0) * (1.0 - x(0))) + (x(2) * (1.0 - x(2))));
  f(2) += 2.0 * ((x(0) * (1.0 - x(0))) + (x(1) * (1.0 - x(1))));
  */
  /*
  f(0) = SIGMAVAL * x(0) * x(1) * (1.0 - x(1)) * x(2) * (1.0 - x(2));
  f(1) = SIGMAVAL * x(1) * x(0) * (1.0 - x(0)) * x(2) * (1.0 - x(2));
  f(2) = SIGMAVAL * x(2) * x(1) * (1.0 - x(1)) * x(0) * (1.0 - x(0));

  f(0) += ((x(1) * (1.0 - x(1))) + (x(2) * (1.0 - x(2))));
  f(1) += ((x(0) * (1.0 - x(0))) + (x(2) * (1.0 - x(2))));
  f(2) += ((x(0) * (1.0 - x(0))) + (x(1) * (1.0 - x(1))));
  */
}

void vecField1(const Vector &x, Vector &E)
{
  E(0) = x(0);
  E(1) = x(1);
  E(2) = x(2);
}

void TestProlongation(const int rank, HypreParMatrix *P, ParFiniteElementSpace & cspace, ParFiniteElementSpace & fspace,
		      ParGridFunction const& xc)
{
  //ParGridFunction xc(&cspace);
  ParGridFunction xf(&fspace);
  ParGridFunction Px(&fspace);

  VectorFunctionCoefficient E(3, vecField1);
  //xc.ProjectCoefficient(E);  // bug??? The cspace pmesh is the same as the fspace pmesh?

  xf.ProjectCoefficient(E);

  Vector txc(cspace.GetTrueVSize());
  Vector txf(fspace.GetTrueVSize());
  Vector tPx(fspace.GetTrueVSize());

  xc.GetTrueDofs(txc);

  xf.GetTrueDofs(txf);

  P->Mult(txc, tPx);

  //Px.SetFromTrueDofs(txf);

  const double tPx_norm = tPx.Norml2();
  tPx -= txf;
  
  cout << rank << ": norm of xf " << txf.Norml2() << ", norm of Px " << tPx_norm << ", norm of diff " << tPx.Norml2() << endl;
}

void CompareHypreParMatrices(HypreParMatrix *A, HypreParMatrix *B)
{
  const int m = A->Height();
  const int n = A->Width();
  MFEM_VERIFY(m == B->Height(), "");
  MFEM_VERIFY(n == B->Width(), "");

  Vector ej(n);
  Vector Aej(m);
  Vector Bej(m);

  double maxrelerr = 0.0;
  
  for (int i=0; i<n; ++i)
    {
      ej = 0.0;
      ej[i] = 1.0;

      A->Mult(ej, Aej);
      B->Mult(ej, Bej);

      const double Anrm = Aej.Norml2();
      const double mnrm = std::max(Anrm, Bej.Norml2());
      Aej -= Bej;
      const double relerr = Aej.Norml2() / mnrm;
      maxrelerr = std::max(maxrelerr, relerr);
    }

  cout << "CompareHypreParMatrices: " << maxrelerr << endl;
}

void TestHypreRectangularSerial()
{
  const int num_loc_cols = 100;
  const int num_loc_rows = 2 * num_loc_cols;
  const int nnz = num_loc_cols;
  
  HYPRE_Int rowStarts2[2];
  HYPRE_Int colStarts2[2];
  
  rowStarts2[0] = 0;
  rowStarts2[1] = num_loc_rows;

  colStarts2[0] = 0;
  colStarts2[1] = num_loc_cols;
  
  int *I_nnz = new int[num_loc_rows + 1];
  HYPRE_Int *J_col = new HYPRE_Int[nnz];

  I_nnz[0] = 0;

  // row 0: 0
  // row 1: 1
  // row 2: 0
  // ...

  // I_nnz row 0: 0
  // I_nnz row 1: 0
  // I_nnz row 2: 1
  // I_nnz row 3: 1
  // I_nnz row 4: 2
  // ...

  for (int i=0; i<num_loc_cols; ++i)
    {
      I_nnz[(2*i)+1] = i;
      I_nnz[(2*i)+2] = i+1;

      J_col[i] = i;
    }

  Vector diag(nnz);
  diag = 1.0;
  
  HypreParMatrix *A = new HypreParMatrix(MPI_COMM_WORLD, num_loc_rows, num_loc_rows, num_loc_cols, I_nnz, J_col, diag.GetData(), rowStarts2, colStarts2);

  Vector x(num_loc_cols);
  Vector y(num_loc_rows);

  x = 1.0;
  y = 0.0;

  cout << "Hypre serial test x norm " << x.Norml2() << endl;
  
  A->Mult(x, y);

  cout << "Hypre serial test y norm " << y.Norml2() << endl;

  delete [] I_nnz;
  delete [] J_col;
  delete A;
}

void TestHypreIdentity(MPI_Comm comm)
{
  int num_loc_rows = 100;
  HYPRE_Int size = 200;
  
  int nsdprocs, sdrank;
  MPI_Comm_size(comm, &nsdprocs);
  MPI_Comm_rank(comm, &sdrank);

  int *all_num_loc_rows = new int[nsdprocs];
		    
  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows, 1, MPI_INT, comm);

  int sumLocalSizes = 0;

  for (int i=0; i<nsdprocs; ++i)
    sumLocalSizes += all_num_loc_rows[i];

  MFEM_VERIFY(size == sumLocalSizes, "");
  
  HYPRE_Int *rowStarts = new HYPRE_Int[nsdprocs+1];
  HYPRE_Int *rowStarts2 = new HYPRE_Int[2];
  rowStarts[0] = 0;
  for (int i=0; i<nsdprocs; ++i)
    rowStarts[i+1] = rowStarts[i] + all_num_loc_rows[i];

  const int osj = rowStarts[sdrank];

  rowStarts2[0] = rowStarts[sdrank];
  rowStarts2[1] = rowStarts[sdrank+1];
  
  int *I_nnz = new int[num_loc_rows + 1];
  HYPRE_Int *J_col = new HYPRE_Int[num_loc_rows];

  for (int i=0; i<num_loc_rows + 1; ++i)
    I_nnz[i] = i;

  for (int i=0; i<num_loc_rows; ++i)
    J_col[i] = osj + i;

  Vector diag(num_loc_rows);
  diag = 1.0;
  
  HypreParMatrix *A = new HypreParMatrix(comm, num_loc_rows, size, size, I_nnz, J_col, diag.GetData(), rowStarts2, rowStarts2);

  /*
  {
    HypreParMatrix *B = new HypreParMatrix(comm, num_loc_rows, size, size, I_nnz, J_col, diag.GetData(), rowStarts2, rowStarts2);
    HypreParMatrix *C = ParAdd(A, B);

    A->Print("IA");
    B->Print("IB");
    C->Print("IC");
  }
  */
  
  Vector x(num_loc_rows);
  Vector y(num_loc_rows);

  x = 1.0;
  y = 0.0;
  
  A->Mult(x, y);

  cout << sdrank << ": Hypre test y norm " << y.Norml2() << endl;
  
  delete [] I_nnz;
  delete [] J_col;
  delete [] rowStarts;
  delete [] rowStarts2;
  delete [] all_num_loc_rows;
  delete A;
}

void VisitTestPlotParMesh(const std::string filename, ParMesh *pmesh, const int ifId, const int myid)
{
  if (pmesh == NULL)
    return;

  DataCollection *dc = NULL;
  bool binary = false;
  if (binary)
    {
#ifdef MFEM_USE_SIDRE
      dc = new SidreDataCollection(filename.c_str(), pmesh);
#else
      MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
    }
  else
    {
      dc = new VisItDataCollection(filename.c_str(), pmesh);
      dc->SetPrecision(8);
      // To save the mesh using MFEM's parallel mesh format:
      // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
    }

  // Define a grid function just to verify it is plotted correctly.
  H1_FECollection h1_coll(1, pmesh->Dimension());
  ParFiniteElementSpace fespace(pmesh, &h1_coll);

  if (ifId >= 0)
    cout << myid << ": interface " << ifId << " VISIT TEST: true V size " << fespace.GetTrueVSize() << ", V size " << fespace.GetVSize() << endl;
  
  ParGridFunction x(&fespace);
  FunctionCoefficient radius(radiusFunction);
  x.ProjectCoefficient(radius);
  
  dc->RegisterField("radius", &x);
  dc->SetCycle(0);
  dc->SetTime(0.0);
  dc->Save();

  delete dc;
}

void PrintDenseMatrixOfOperator(Operator const& op, const int nprocs, const int rank)
{
  const int n = op.Height();

  int ng = 0;
  MPI_Allreduce(&n, &ng, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  std::vector<int> alln(nprocs);
  MPI_Allgather(&n, 1, MPI_INT, alln.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  int myos = 0;

  int cnt = 0;
  for (int i=0; i<nprocs; ++i)
    {
      if (i < rank)
	myos += alln[i];

      cnt += alln[i];
    }

  MFEM_VERIFY(cnt == ng, "");
  
  Vector x(n);
  Vector y(n);

  /*
       Vector ej(Ndd);
     Vector Aej(Ndd);
     DenseMatrix ddd(Ndd);
     
     for (int j=0; j<Ndd; ++j)
       {
	 cout << "Computing column " << j << " of " << Ndd << " of ddi" << endl;
	 
	 ej = 0.0;
	 ej[j] = 1.0;
	 ddi.Mult(ej, Aej);

	 for (int i=0; i<Ndd; ++i)
	   {
	     ddd(i,j) = Aej[i];
	   }
       }
  */
}

void VerifyMeshesAreEqual(Mesh *a, Mesh *b)
{
  if (a == NULL)
    {
      MFEM_VERIFY(b == NULL, "");
      return;
    }

  MFEM_VERIFY(b != NULL, "");

  // Compare vertex coordinates
  MFEM_VERIFY(a->GetNV() == b->GetNV(), "");

  const int dim = a->SpaceDimension();
  MFEM_VERIFY(dim == b->SpaceDimension(), "");

  const double tol = 1.0e-12;
  
  bool eq = true;
  for (int i=0; i<a->GetNV(); ++i)
    {
      for (int j=0; j<dim; ++j)
	{
	  if (fabs(a->GetVertex(i)[j] - b->GetVertex(i)[j]) > tol)
	    eq = false;
	}
    }
  
  // Compare edge vertex indices

  MFEM_VERIFY(a->GetNEdges() == b->GetNEdges(), "");

  Array<int> av, bv, ao, bo;
  for (int i=0; i<a->GetNEdges(); ++i)
    {
      a->GetEdgeVertices(i, av);
      b->GetEdgeVertices(i, bv);

      MFEM_VERIFY(av.Size() == 2 && bv.Size() == 2, "");

      for (int j=0; j<2; ++j)
	{
	  if (av[j] != bv[j])
	    eq = false;
	}
    }
  
  // Compare face edge indices
  
  MFEM_VERIFY(a->GetNFaces() == b->GetNFaces(), "");

  for (int i=0; i<a->GetNFaces(); ++i)
    {
      a->GetFaceEdges(i, av, ao);
      b->GetFaceEdges(i, bv, bo);

      MFEM_VERIFY(av.Size() == bv.Size(), "");

      for (int j=0; j<av.Size(); ++j)
	{
	  if (av[j] != bv[j] || ao[j] != bo[j])
	    eq = false;
	}
    }

  // Compare element face indices
  
  MFEM_VERIFY(a->GetNE() == b->GetNE(), "");

  for (int i=0; i<a->GetNE(); ++i)
    {
      a->GetElementFaces(i, av, ao);
      b->GetElementFaces(i, bv, bo);

      MFEM_VERIFY(av.Size() == bv.Size(), "");

      for (int j=0; j<av.Size(); ++j)
	{
	  if (av[j] != bv[j] || ao[j] != bo[j])
	    eq = false;
	}
    }

  MFEM_VERIFY(eq, "");
}

#ifdef TEST_GMG
void TestGlobalGMG(ParMesh *pmesh, ParFiniteElementSpace *fespace, std::vector<HypreParMatrix*> const& P)
{
  ParGridFunction x(fespace);
  const int sdim = 3;
  VectorFunctionCoefficient E(sdim, test2_E_exact);
   
  x.ProjectCoefficient(E);

  Array<int> ess_tdof_list;
  Array<int> ess_bdr;
  if (pmesh->bdr_attributes.Size())
    {
      //Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  // 8. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (f,phi_i) where f is given by the function f_exact and phi_i are the
  //    basis functions in the finite element fespace.
  //VectorFunctionCoefficient f(sdim, f_exact);
  VectorFunctionCoefficient f(sdim, test2_RHS_exact);
  ParLinearForm *b = new ParLinearForm(fespace);
  b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
  b->Assemble();

  // 9. Define the solution vector x as a parallel finite element grid function
  //    corresponding to fespace. Initialize x by projecting the exact
  //    solution. Note that only values from the boundary edges will be used
  //    when eliminating the non-homogeneous boundary condition to modify the
  //    r.h.s. vector b.
  //VectorFunctionCoefficient E(sdim, E_exact);
 
  // 10. Set up the parallel bilinear form corresponding to the EM diffusion
  //     operator curl muinv curl + sigma I, by adding the curl-curl and the
  //     mass domain integrators.
  Coefficient *muinv = new ConstantCoefficient(1.0);
  Coefficient *sigma = new ConstantCoefficient(SIGMAVAL);
  //Coefficient *sigmaAbs = new ConstantCoefficient(fabs(SIGMAVAL));
  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));

#ifdef AIRY_TEST
  VectorFunctionCoefficient epsilon(3, test_Airy_epsilon);
  a->AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
#else
  a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
#endif
   
  //cout << myid << ": NBE " << pmesh->GetNBE() << endl;

  // 11. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  a->Assemble();
  a->Finalize();

  /*
    Vector exactSol(fespace->GetTrueVSize());
    x.GetTrueDofs(exactSol);
  */
   
  HypreParMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  GMGSolver *gmg = new GMGSolver(&A, P, GMGSolver::CoarseSolver::STRUMPACK);
  gmg->SetTheta(0.5);
  gmg->SetSmootherType(HypreSmoother::Jacobi);  // Some options: Jacobi, l1Jacobi, l1GS, GS

  GMRESSolver *gmres = new GMRESSolver(A.GetComm());

  gmres->SetOperator(A);

  gmres->SetRelTol(1e-12);
  gmres->SetMaxIter(50);
  gmres->SetPrintLevel(1);

  gmres->SetPreconditioner(*gmg);
  gmres->SetName("TestGlobalGMG");
  gmres->iterative_mode = false;

  cout << "TestGlobalGMG gmres" << endl;
  
  gmres->Mult(B, X);

  cout << "TestGlobalGMG gmres done" << endl;
  
  delete gmres;
  delete gmg;
}
#endif

#ifdef SD_ITERATIVE_GMG
void MapElementDofsByValue(const int dim, ParFiniteElementSpace *fespaceA, const int elemA, ParFiniteElementSpace *fespaceB, const int elemB,
			   std::vector<int>& dofmap, std::vector<int>& flip)
{
  const FiniteElement *feA = fespaceA->GetFE(elemA);
  MFEM_VERIFY(feA->GetRangeType() != FiniteElement::SCALAR, "");

  ElementTransformation *TrA = fespaceA->GetElementTransformation(elemA);

  Array<int> dofsA, dofsB;
  fespaceA->GetElementDofs(elemA, dofsA);
  const int ndofs = dofsA.Size();

  const FiniteElement *feB = fespaceB->GetFE(elemB);
  MFEM_VERIFY(feB->GetRangeType() != FiniteElement::SCALAR, "");

  ElementTransformation *TrB = fespaceB->GetElementTransformation(elemB);
  fespaceB->GetElementDofs(elemB, dofsB);

  MFEM_VERIFY(ndofs == dofsB.Size(), "");
  
  dofmap.resize(ndofs);
  flip.resize(ndofs);
  
  IntegrationPoint ip;

  MFEM_VERIFY(dim == 3, "");

  Vector x(dim);  // Point in physical space.
  Vector va(dim);  // Basis function value at integration point.
  Vector vb(dim);  // Basis function value at integration point.

  int intorder = (2*feA->GetOrder()) + 1;
  const IntegrationRule *ir = &(IntRules.Get(feA->GetGeomType(), intorder));

  MFEM_VERIFY(feA->GetOrder() == feB->GetOrder() && feA->GetGeomType() == feB->GetGeomType(), "");

  const int nip = ir->Size();

  DenseMatrix vshape(ndofs, dim);
  Vector dataA(ndofs);
  Vector dataB(ndofs);

  Vector e(ndofs);

  for (int d=0; d<ndofs; ++d)
    {
      dataA = 0.0;
      dataA[d] = (dofsA[d] >= 0) ? 1.0 : -1.0;

      dofmap[d] = -1;
      
      for (int i=0; i<nip; ++i)
	{
	  ip = ir->IntPoint(i);
	  TrA->Transform(ip, x);

	  TrA->SetIntPoint(&ip);
	  feA->CalcVShape(*TrA, vshape);

	  vshape.MultTranspose(dataA, va);

	  if (va.Norml2() < 1.0e-4)
	    continue;
	  
	  const bool foundB = (TrB->TransformBack(x, ip) == 0);
	  MFEM_VERIFY(foundB, "");
	  
	  TrB->SetIntPoint(&ip);
	  feB->CalcVShape(*TrB, vshape);

	  for (int signflip=0; signflip<1; ++signflip)
	    {
	      double emin = 0.0;
	      int imin = 0;
      
	      for (int n=0; n<ndofs; ++n)
		{
		  dataB = 0.0;
		  dataB[n] = (dofsB[n] >= 0) ? 1.0 : -1.0;
		  
		  vshape.MultTranspose(dataB, vb);
		  
		  const double vnrm = std::max(va.Norml2(), vb.Norml2());

		  MFEM_VERIFY(vnrm > 1.0e-4, "");

		  if (signflip == 0)
		    vb -= va;
		  else
		    vb += va;
	  
		  e[n] = vb.Norml2() / vnrm;

		  if (n == 0 || e[n] < emin)
		    {
		      emin = e[n];
		      imin = n;
		    }
		}

	      double emin2 = 0.0;
	      int imin2 = -1;

	      for (int n=0; n<ndofs; ++n)
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
		  flip[d] = signflip;
		  MFEM_VERIFY(dofmap[d] == -1 || dofmap[d] == imin, "");
		  
		  dofmap[d] = imin;
		}
	    }
	}

      MFEM_VERIFY(dofmap[d] >= 0, "");
    }
}

// This function assumes (i) pmeshA and pmeshB have the same elements on each process, but they can have different ordering for all mesh entities;
// (ii) the element attribute in pmeshA and pmeshB is 1-based and distinct with respect to elements, and elements with the same attribute in the
// two meshes coincide geometrically;
// (iii) each DOF in fespaceA maps to exactly one DOF in fespaceB, with possibly a sign flip but no scaling (valid for hexahedral meshes);
// (iv) A and B have the same MPI_Comm.
// The matrix returned is the representation of the DOF map from fespaceA to fespaceB.
HypreParMatrix * CreateFESpaceMapForReorderedMeshes(ParMesh *pmeshA, ParFiniteElementSpace *fespaceA, ParMesh *pmeshB, FiniteElementCollection * fec)
{
  std::map<int, int> attributeToIndex;

  const int dim = pmeshA->SpaceDimension();
  
  for (int i=0; i<pmeshA->GetNE(); ++i)
    {
      const int attr = pmeshA->GetAttribute(i);

      std::map<int, int>::const_iterator it = attributeToIndex.find(attr);
      MFEM_VERIFY(it == attributeToIndex.end(), "");

      attributeToIndex[attr] = i;
    }

  ParFiniteElementSpace *fespaceB = new ParFiniteElementSpace(pmeshB, fec);

  const int ntdof = fespaceB->GetTrueVSize();
  const int num_loc_rows = ntdof;
  
  MPI_Comm comm = pmeshA->GetComm();  // Note that A and B should have the same comm.

  int nprocs, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  std::vector<int> all_num_loc_rows(nprocs);
  
  MPI_Allgather(&num_loc_rows, 1, MPI_INT, all_num_loc_rows.data(), 1, MPI_INT, comm);
  int glob_nrows = 0;
  int first_loc_row = 0;  
  for (int i=0; i<nprocs; ++i)
    {
      glob_nrows += all_num_loc_rows[i];
      if (i < rank)
	first_loc_row += all_num_loc_rows[i];
    }

  std::vector<HYPRE_Int> rowStarts2(2);
  rowStarts2[0] = first_loc_row;
  rowStarts2[1] = first_loc_row + all_num_loc_rows[rank];

  const int num_loc_cols = fespaceA->GetTrueVSize();
  std::vector<int> all_num_loc_cols(nprocs);
  
  MPI_Allgather(&num_loc_cols, 1, MPI_INT, all_num_loc_cols.data(), 1, MPI_INT, comm);
  int glob_ncols = 0;
  int first_loc_col = 0;  
  for (int i=0; i<nprocs; ++i)
    {
      glob_ncols += all_num_loc_cols[i];
      if (i < rank)
	first_loc_col += all_num_loc_cols[i];
    }

  std::vector<HYPRE_Int> colStarts2(2);
  colStarts2[0] = first_loc_col;
  colStarts2[1] = first_loc_col + all_num_loc_cols[rank];
  
  std::vector<int> opI(num_loc_rows+1);
  //std::vector<int> cnt(num_loc_rows);

  opI[0] = 0;
  for (int i=1; i<num_loc_rows+1; ++i)
    {
      opI[i] = 1;  // 1 entry per row
      opI[i] += opI[i-1];  // partial sum for offsets.
    }

  const int nnz = opI[num_loc_rows];

  std::vector<HYPRE_Int> opJ;
  opJ.assign(nnz, -1);
  std::vector<double> data;
  data.assign(nnz, 0.0);

  std::vector<int> elemMap, flip;
  
  for (int elemB=0; elemB<pmeshB->GetNE(); ++elemB)
    {
      const int attr = pmeshB->GetAttribute(elemB);

      std::map<int, int>::const_iterator it = attributeToIndex.find(attr);
      MFEM_VERIFY(it != attributeToIndex.end(), "");
      MFEM_VERIFY(it->first == attr, "");

      const int elemA = it->second;

      MapElementDofsByValue(dim, fespaceA, elemA, fespaceB, elemB, elemMap, flip);

      Array<int> dofsA, dofsB;
      fespaceA->GetElementDofs(elemA, dofsA);
      fespaceB->GetElementDofs(elemB, dofsB);

      MFEM_VERIFY(dofsA.Size() == elemMap.size() && dofsB.Size() == elemMap.size(), "");

      for (int i=0; i<elemMap.size(); ++i)
	{
	  const int dofB = (dofsB[elemMap[i]] >= 0) ? dofsB[elemMap[i]] : -1 - dofsB[elemMap[i]];
	  
	  const int ltdofB = fespaceB->GetLocalTDofNumber(dofB);

	  if (ltdofB >= 0)
	    {
	      const int dofA = (dofsA[i] >= 0) ? dofsA[i] : -1 - dofsA[i];
	      const HYPRE_Int gtdofA = fespaceA->GetGlobalTDofNumber(dofA);

	      if (!(opJ[opI[ltdofB]] == gtdofA || opJ[opI[ltdofB]] == -1))
		cout << "BUG" << endl;
	  
	      MFEM_VERIFY(opJ[opI[ltdofB]] == gtdofA || opJ[opI[ltdofB]] == -1, "");

	      opJ[opI[ltdofB]] = gtdofA;
	      data[opI[ltdofB]] = (flip[i] == 0) ? 1.0 : -1.0;
	    }
	}
    }
  
  for (int i=0; i<nnz; ++i)
    {
      if (opJ[i] < 0)
	MFEM_VERIFY(opJ[i] >= 0, "");	
    }

  delete fespaceB;
  
  HypreParMatrix *fesmap = new HypreParMatrix(comm, ntdof, glob_nrows, glob_ncols, (int*) opI.data(), (HYPRE_Int*) opJ.data(), (double*) data.data(),
					      (HYPRE_Int*) rowStarts2.data(), (HYPRE_Int*) colStarts2.data());

  return fesmap;
}
#endif

int main(int argc, char *argv[])
{
   StopWatch chronoMain;
   chronoMain.Clear();
   chronoMain.Start();
  
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   // 2. Parse command-line options.
   //const char *mesh_file = "../data/beam-tet.mesh";
#ifdef AIRY_TEST
   //const char *mesh_file = "inline-tetHalf.mesh";
   //const char *mesh_file = "inline-tetHalf2.mesh";
   //const char *mesh_file = "../data/inline-tet.mesh";
   const char *mesh_file = "inline-hexHalf.mesh";
   //const char *mesh_file = "inline-hexHalf2.mesh";
#else
   const char *mesh_file = "../data/inline-tet.mesh";
#endif

   int order = 2;
   bool static_cond = false;
   bool visualization = 1;
   bool visit = true;
#ifdef MFEM_USE_STRUMPACK
   bool use_strumpack = true;
#endif
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_STRUMPACK
   args.AddOption(&use_strumpack, "-strumpack", "--strumpack-solver",
                  "-no-strumpack", "--no-strumpack-solver",
                  "Use STRUMPACK's double complex linear solver.");
#endif

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   kappa = freq * M_PI;

   /*
   // Redirect mfem output
   ofstream outfile("mfem.txt");
   mfem::out.SetStream(outfile);
   */
   
   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

#ifdef SUBDOMAIN_MESH
   for (int i=0; i<mesh->GetNE(); ++i)  // Loop over all elements, to set the attribute as the subdomain index.
     {
       mesh->SetAttribute(i, i+1);  // Set each element to be a subdomain.
     }
   
   const int numSubdomains = mesh->GetNE();
#endif
   
   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
	(int)floor(log(10000./mesh->GetNE())/log(2.)/dim);  // h = 0.0701539, 1/16
      //(int)floor(log(100000./mesh->GetNE())/log(2.)/dim);  // h = 0.0350769, 1/32
      //(int)floor(log(1000000./mesh->GetNE())/log(2.)/dim);  // h = 0.0175385, 1/64
	//(int)floor(log(10000000./mesh->GetNE())/log(2.)/dim);  // h = 0.00876923, 1/128
	//(int)floor(log(100000000./mesh->GetNE())/log(2.)/dim);  // exceeds memory with slab subdomains, first-order

      // Note: with nx=6 in inline-tetHalf2.mesh, 1/64 becomes 1/96; 1/128 becomes 1/192.
      
      //(int)floor(log(100000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

#ifndef SUBDOMAIN_MESH
   // 4.5. Partition the mesh in serial, to define subdomains.
   // Note that the mesh attribute is overwritten here for convenience, which is bad if the attribute is needed.
   int nxyzSubdomains[3] = {4, 4, 4};
   const int numSubdomains = nxyzSubdomains[0] * nxyzSubdomains[1] * nxyzSubdomains[2];
   {
     int *subdomain = mesh->CartesianPartitioning(nxyzSubdomains);
     for (int i=0; i<mesh->GetNE(); ++i)  // Loop over all elements, to set the attribute as the subdomain index.
       {
	 mesh->SetAttribute(i, subdomain[i]+1);
       }
     delete subdomain;
   }
#endif
   
   if (numSubdomains > 10000)
     {
       MFEM_VERIFY(numSubdomains < 10000, "SubdomainInterface::SetGlobalIndex will overflow");
       return 3;
     }

   if (myid == 0)
     cout << "Serial mesh number of elements: " << mesh->GetNE() << endl;

   std::vector<int> sdOrder(numSubdomains);
   
#ifdef SUBDOMAIN_MESH
   for (int i=0; i<numSubdomains; ++i)
     sdOrder[i] = i;
#else
   if (myid == 0)
     {
       cout << "Subdomain partition " << nxyzSubdomains[0] << ", " << nxyzSubdomains[1] << ", " << nxyzSubdomains[2] << endl;
     }

   {
     const int nh = numSubdomains / 2;

     /*
     for (int i=0; i<numSubdomains; ++i)
       {
	 sdOrder[i] = i;
       }

     for (int i=0; i<nh; ++i)
       {
	 sdOrder[i] = 2*i;
	 sdOrder[i + nh] = (2*i) + 1;
       }
     */
     /*
     // Checkerboard ordering, which is bad
     int cnt = 0;
     for (int i=0; i<nxyzSubdomains[0]; ++i)
       {
	 for (int j=0; j<nxyzSubdomains[1]; ++j)
	   {
	     for (int l=0; l<nxyzSubdomains[2]; ++l)
	       {
		 const int sdid = (i * nxyzSubdomains[1] * nxyzSubdomains[2]) + (j * nxyzSubdomains[2]) + l;
		 const int sds = i + j + l;
		 
		 if (sds % 2 == 0)
		   sdOrder[cnt / 2] = sdid;
		 else
		   sdOrder[(cnt / 2) + nh] = sdid;

		 cnt++;
	       }
	   }
       }
     */
     // 2x2x2 block ordering
     int ngt = 1;
     //int nhalf[3];
     int ngrp[3];
     
     for (int i=0; i<3; ++i)
       {
	 if (nxyzSubdomains[i] > 1)
	   {
	     ngt *= 2;
	     ngrp[i] = 2;
	     //nhalf[i] = nxyzSubdomains[i] / 2;
	   }
	 else
	   {
	     //nhalf[i] = 1;
	     ngrp[i] = 1;
	   }
       }

     std::vector<int> gcnt(ngt); // group size
     std::vector<int> gos(ngt);  // group offset
     gcnt.assign(ngt, 0);
     gos.assign(ngt, 0);
     
     for (int i=0; i<nxyzSubdomains[2]; ++i)
       {
	 for (int j=0; j<nxyzSubdomains[1]; ++j)
	   {
	     for (int l=0; l<nxyzSubdomains[0]; ++l)
	       {
		 const int g = ((i % 2) * ngrp[1] * ngrp[0]) + ((j % 2) * ngrp[0]) + (l % 2);
		 gcnt[g]++;  // count the number of subdomains in each group
	       }
	   }
       }

     // Set group offsets from counts
     for (int i=1; i<ngt; ++i)
       {
	 gos[i] = gos[i-1] + gcnt[i-1];
       }
     
     gcnt.assign(ngt, 0);
     
     for (int i=0; i<nxyzSubdomains[2]; ++i)
       {
	 for (int j=0; j<nxyzSubdomains[1]; ++j)
	   {
	     for (int l=0; l<nxyzSubdomains[0]; ++l)
	       {
		 const int g = ((i % 2) * ngrp[1] * ngrp[0]) + ((j % 2) * ngrp[0]) + (l % 2);

		 const int sdid = (i * nxyzSubdomains[1] * nxyzSubdomains[0]) + (j * nxyzSubdomains[0]) + l;
		 
		 sdOrder[gos[g] + gcnt[g]] = sdid;
		 gcnt[g]++;
	       }
	   }
       }
   }
#endif
   
   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = NULL;

#ifdef SUBDOMAIN_MESH
   const bool geometricPartition = false;
#else
   const bool geometricPartition = true;
#endif

   if (geometricPartition)
     {
       //int nxyzGlobal[3] = {1, 1, 1};
       //int nxyzGlobal[3] = {2, 2, 4};
       //int nxyzGlobal[3] = {1, 1, 4};
       //int nxyzGlobal[3] = {1, 2, 1};
       //int nxyzGlobal[3] = {2, 2, 2};
       //int nxyzGlobal[3] = {2, 2, 4};
       //int nxyzGlobal[3] = {3, 3, 4};
       int nxyzGlobal[3] = {4, 4, 4};
       //int nxyzGlobal[3] = {6, 6, 6};
       //int nxyzGlobal[3] = {6, 6, 12};  // 432
       //int nxyzGlobal[3] = {6, 12, 12};  // 864
       //int nxyzGlobal[3] = {2, 2, 8};
       //int nxyzGlobal[3] = {6, 6, 8};  // 288
       //int nxyzGlobal[3] = {8, 6, 6};  // 288
       //int nxyzGlobal[3] = {6, 12, 8};  // 576
       //int nxyzGlobal[3] = {12, 6, 8};  // 576
       //int nxyzGlobal[3] = {12, 12, 4};  // 576
       //int nxyzGlobal[3] = {6, 6, 16};  // 576
       //int nxyzGlobal[3] = {12, 12, 8};  // 1152
       //int nxyzGlobal[3] = {6, 12, 16};  // 1152
       //int nxyzGlobal[3] = {6, 6, 32};  // 1152
       //int nxyzGlobal[3] = {4, 4, 8};
       //int nxyzGlobal[3] = {4, 8, 8};  // 256
       //int nxyzGlobal[3] = {12, 12, 12};  // 1728
       //int nxyzGlobal[3] = {24, 12, 12};  // 3456
       //int nxyzGlobal[3] = {24, 24, 12};  // 6912
       //int nxyzGlobal[3] = {16, 16, 8};
       //int nxyzGlobal[3] = {8, 8, 8};
       //int nxyzGlobal[3] = {12, 12, 16};  // 2304
       //int nxyzGlobal[3] = {24, 24, 4};  // 2304
       //int nxyzGlobal[3] = {24, 24, 8};  // 4608
       //int nxyzGlobal[3] = {12, 6, 6};  // 432
       
       int *partition = mesh->CartesianPartitioning(nxyzGlobal);
       //int *partition = mesh->CartesianPartitioningXY(nxyzGlobal, 2, 2);
       //int *partition = mesh->CartesianPartitioningXY(nxyzGlobal, 6, 6);
       //int *partition = mesh->CartesianPartitioningXY(nxyzGlobal, 3, 3);
       
       pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partition);

       delete partition;
       
       // pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

       if (myid == 0)
	 {
	   cout << "Parallel partition " << nxyzGlobal[0] << ", " << nxyzGlobal[1] << ", " << nxyzGlobal[2] << endl;
	 }

       /*
       std::vector<int> partition;
       partition.assign(mesh->GetNE(), -1);

       Vector ec;
       
       for (int i=0; i<mesh->GetNE(); ++i)
	 {
	   mesh->GetElementCenter(i, ec);
	   partition[i] = 0;
	 }
       
       pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partition.data());
       */
     }
   else
     {
       pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
     }
   
   delete mesh;
   //pmesh->ReorientTetMesh();

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   
#ifdef SD_ITERATIVE_GMG
   ParMesh** pmeshSDcoarse = NULL;
   {
     SubdomainParMeshGenerator sdCoarseMeshGen(numSubdomains, pmesh);
     pmeshSDcoarse = sdCoarseMeshGen.CreateParallelSubdomainMeshes();

     // Note that the element attribute in pmeshSDcoarse[sd] is the index of the corresponding coarse pmesh element plus one.

     if (pmeshSDcoarse == NULL)
       return 2;
   }

   std::vector<std::vector<HypreParMatrix*> > sdP(numSubdomains);
   std::vector<ParFiniteElementSpace*> sdfespace(numSubdomains);
#endif

#ifdef TEST_GMG
   std::vector<HypreParMatrix*> gmgP;
   ParFiniteElementSpace *gmgfespace = new ParFiniteElementSpace(pmesh, fec);
#endif

#ifdef SD_ITERATIVE_GMG
   std::vector<HypreParMatrix*> sdcRe;
   std::vector<HypreParMatrix*> sdcIm;
#endif

#ifdef SDFOSLS_PA
   std::vector<Array2D<HypreParMatrix*> > coarseFOSLS;
#endif

   {
      int par_ref_levels = 1;
      
      if (myid == 0)
	cout << "Parallel refinement levels: " << par_ref_levels << endl;
      
#ifdef TEST_GMG
      gmgP.resize(par_ref_levels);
#endif

#ifdef SD_ITERATIVE_GMG
#ifdef SD_ITERATIVE_GMG_PA      
      { // Construct a DD operator just to get coarse subdomain and interface matrices with correct boundary conditions.
	SubdomainInterfaceGenerator sdInterfaceGen_C(numSubdomains, pmesh);
	vector<SubdomainInterface> interfaces_C;  // Local interfaces
	sdInterfaceGen_C.CreateInterfaces(interfaces_C);
	
	std::vector<int> interfaceGlobalToLocalMap_C, interfaceGI_C;
	const int numInterfaces_C = sdInterfaceGen_C.GlobalToLocalInterfaceMap(interfaces_C, interfaceGlobalToLocalMap_C, interfaceGI_C);
   
	cout << myid << ": created " << numSubdomains << " COARSE subdomains with " << numInterfaces_C <<  " interfaces" << endl;

	SubdomainParMeshGenerator sdMeshGen_C(numSubdomains, pmesh);
	ParMesh **pmeshSD_C = sdMeshGen_C.CreateParallelSubdomainMeshes();
	// TODO: isn't pmeshSD_C identical to pmeshSDcoarse? If so, just use pmeshSDcoarse and do not create pmeshSD_C.
	
	ParFiniteElementSpace fespace_C(pmesh, fec);

#ifdef SERIAL_INTERFACES
	Mesh **smeshInterfaces_C = (numInterfaces_C > 0 ) ? new Mesh*[numInterfaces_C] : NULL;
#else
	ParMesh **pmeshInterfaces_C = (numInterfaces_C > 0 ) ? new ParMesh*[numInterfaces_C] : NULL;
#endif

	std::vector<int> interfaceFaceOffset_C;
   
	CreateInterfaceMeshes(numInterfaces_C, numSubdomains, myid, num_procs, sdMeshGen_C, 
			      interfaceGlobalToLocalMap_C, interfaceGI_C, pmeshSD_C, interfaceFaceOffset_C, interfaces_C,
#ifdef SERIAL_INTERFACES
			      smeshInterfaces_C
#else
			      pmeshInterfaces_C
#endif
			      );

	std::vector<Array2D<HypreParMatrix*> > dummyCoarseFOSLS(numSubdomains);
	
	DDMInterfaceOperator ddiC(numSubdomains, numInterfaces_C, pmesh, &fespace_C, pmeshSD_C,
#ifdef SERIAL_INTERFACES
				  smeshInterfaces_C, interfaceFaceOffset_C, 
#else
				  pmeshInterfaces_C,
#endif
				  order, pmesh->Dimension(),
				  &interfaces_C, &interfaceGlobalToLocalMap_C, -SIGMAVAL,
#ifdef GPWD
				  1,
#endif
#ifdef SD_ITERATIVE_GMG
				  sdP, NULL, NULL,  // not used here
#endif
#ifdef SDFOSLS_PA
				  &dummyCoarseFOSLS,  // not used here
#endif
				  1.0, true);  // hmin value not relevant here

#ifdef SDFOSLS_PA
	ddiC.CopyFOSLSMatrices(coarseFOSLS);
#else
	ddiC.CopySDMatrices(sdcRe, sdcIm);
#endif
	
	for (int i=0; i<numInterfaces_C; ++i)
	  {
#ifdef SERIAL_INTERFACES
	    delete smeshInterfaces_C[i];
#else
	    delete pmeshInterfaces_C[i];
#endif
	  }

#ifdef SERIAL_INTERFACES
	delete smeshInterfaces_C;
#else
	delete pmeshInterfaces_C;
#endif
	
	for (int sd=0; sd<numSubdomains; ++sd)
	  {
	    delete pmeshSD_C[sd];
	  }
	
	delete pmeshSD_C;
      }
#endif
      
      for (int sd=0; sd<numSubdomains; ++sd)
	{
	  if (pmeshSDcoarse[sd] != NULL)
	    {
	      //pmeshSDcoarse[sd]->ReorientTetMesh();
	      sdP[sd].resize(par_ref_levels);
	      sdfespace[sd] = new ParFiniteElementSpace(pmeshSDcoarse[sd], fec);
	    }
	}
#endif
      
      for (int l = 0; l < par_ref_levels; l++)
      {
#ifdef SD_ITERATIVE_GMG
	const int numCoarseElem = pmesh->GetNE();
#endif
	
	pmesh->UniformRefinement();
	
#ifdef SD_ITERATIVE_GMG
	CoarseFineTransformations const& cftr = pmesh->GetRefinementTransforms();
	MFEM_VERIFY(cftr.embeddings.Size() == pmesh->GetNE(), "");

	std::vector<int> numFinePerCoarse;
	numFinePerCoarse.assign(numCoarseElem, 0);
	
	for (int i=0; i<pmesh->GetNE(); ++i)
	  {
	    numFinePerCoarse[cftr.embeddings[i].parent]++;
	  }

	const int nfpc = numFinePerCoarse[0];
	
	{
	  bool mixedMesh = false;
	  for (int i=1; i<numCoarseElem; ++i)
	    {
	      if (numFinePerCoarse[i] != nfpc)
		mixedMesh = true;
	    }

	  MFEM_VERIFY(!mixedMesh && nfpc > 1, "");
	}
	
	std::vector<int> coarseToFine;
	coarseToFine.assign(nfpc*numCoarseElem, -1);

	numFinePerCoarse.assign(numCoarseElem, 0);

	for (int i=0; i<pmesh->GetNE(); ++i)
	  {
	    const int coarse = cftr.embeddings[i].parent;
	    coarseToFine[(nfpc*coarse) + numFinePerCoarse[coarse]] = i;
	    numFinePerCoarse[coarse]++;
	  }

	for (int sd=0; sd<numSubdomains; ++sd)
	  {
	    if (pmeshSDcoarse[sd] != NULL)
	      {
		ParFiniteElementSpace cfespace(*(sdfespace[sd]));
		/*
		// For TestProlongation
		ParGridFunction xc(&cfespace);
		{
		  VectorFunctionCoefficient ETP(3, vecField1);
		  xc.ProjectCoefficient(ETP);
		}
		*/
		
		pmeshSDcoarse[sd]->UniformRefinement();
		//pmeshSDcoarse[sd]->ReorientTetMesh();

		sdfespace[sd]->Update();
		OperatorHandle Tr(Operator::Hypre_ParCSR);
		sdfespace[sd]->GetTrueTransferOperator(cfespace, Tr);
		Tr.SetOperatorOwner(false);
		Tr.Get(sdP[sd][l]);

		/*
		if (sd == 0)
		  TestProlongation(myid, sdP[sd][l], cfespace, *(sdfespace[sd]), xc);
		*/
		if (sd == 0 && myid == -10)
		  sdP[0][0]->Print("rawP0.txt");
		
		// Update the element attribute in pmeshSDcoarse[sd] to be the index of the corresponding pmesh element plus one.
		Vector pc(3);
		Vector sc(3);
		for (int i=0; i<pmeshSDcoarse[sd]->GetNE(); ++i)
		  {
		    pmeshSDcoarse[sd]->GetElementCenter(i, sc);
		    const int pmeshCoarseElem = pmeshSDcoarse[sd]->GetAttribute(i) - 1;
		    int pmeshElem = -1;
		    for (int j=0; j<nfpc; ++j)
		      {
			const int pf = coarseToFine[(nfpc*pmeshCoarseElem) + j];
			pmesh->GetElementCenter(pf, pc);
			pc -= sc;
			if (pc.Norml2() < 1.0e-4 * pmesh->GetElementSize(pf))
			  {
			    MFEM_VERIFY(pmeshElem == -1, "");
			    pmeshElem = pf;
			  }
		      }

		    MFEM_VERIFY(pmeshElem >= 0, "");
		    pmeshSDcoarse[sd]->SetAttribute(i, pmeshElem+1);
		  }

		/*
#ifdef SD_ITERATIVE_GMG_PA
	 {
	   MFEM_VERIFY(pmeshSD[sd] != NULL, "");

	   // Note that the element attribute in pmeshSD[sd] and pmeshSDcoarse[sd] is the index of the corresponding pmesh element plus one.
	   
	   HypreParMatrix *fesMapSD = CreateFESpaceMapForReorderedMeshes(pmeshSDcoarse[sd], sdfespace[sd], pmeshSD[sd], fec);

	   // On the finest level, replace sdP[sd][par_ref_levels-1] with fesMapSD times itself.
	   HypreParMatrix * MP = ParMult(fesMapSD, sdP[sd][sdP[sd].size() - 1]);
	   sdP[sd][sdP[sd].size() - 1] = MP;

	   delete fesMapSD;
	 }
#endif
		*/
		
	      }
	  }
#endif

#ifdef TEST_GMG
	const ParFiniteElementSpace cgmgfespace(*gmgfespace);
#endif
	
#ifdef TEST_GMG
	gmgfespace->Update();
	OperatorHandle Tr(Operator::Hypre_ParCSR);
	gmgfespace->GetTrueTransferOperator(cgmgfespace, Tr);
	Tr.SetOperatorOwner(false);
	Tr.Get(gmgP[l]);
#endif
      }
   }
   pmesh->ReorientTetMesh();

#ifdef TEST_GMG
   TestGlobalGMG(pmesh, gmgfespace, gmgP);

   return 0;
#endif
 
   /*
#ifdef SD_ITERATIVE_GMG
   for (int sd=0; sd<numSubdomains; ++sd)
     {
       if (pmeshSDcoarse[sd] != NULL)
	 {
	   pmeshSDcoarse[sd]->ReorientTetMesh();
	 }
     }
#endif
   */
   
   double hmin = 0.0;   
   {
     double minsize = pmesh->GetElementSize(0);
     double maxsize = minsize;
     for (int i=1; i<pmesh->GetNE(); ++i)
       {
	 const double size_i = pmesh->GetElementSize(i);
	 minsize = std::min(minsize, size_i);
	 maxsize = std::max(maxsize, size_i);
       }

     cout << myid << ": Element size range: (" << minsize << ", " << maxsize << ")" << endl;

     MPI_Allreduce(&minsize, &hmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   }

   // 5.1. Determine subdomain interfaces, and for each interface create a set of local vertex indices in pmesh.
   SubdomainInterfaceGenerator sdInterfaceGen(numSubdomains, pmesh);
   vector<SubdomainInterface> interfaces;  // Local interfaces
   sdInterfaceGen.CreateInterfaces(interfaces);
   
   /*
   { // Debugging
     for (std::set<int>::const_iterator it = interfaces[0].faces.begin(); it != interfaces[0].faces.end(); ++it)
       {
	 cout << myid << ": iface " << *it << endl;
       }
   }
   */
   
   std::vector<int> interfaceGlobalToLocalMap, interfaceGI;
   const int numInterfaces = sdInterfaceGen.GlobalToLocalInterfaceMap(interfaces, interfaceGlobalToLocalMap, interfaceGI);
   
   cout << myid << ": created " << numSubdomains << " subdomains with " << numInterfaces <<  " interfaces" << endl;
   
   // 5.2. Create parallel subdomain meshes.
   SubdomainParMeshGenerator sdMeshGen(numSubdomains, pmesh);
   ParMesh **pmeshSD = sdMeshGen.CreateParallelSubdomainMeshes();

   if (pmeshSD == NULL)
     return 2;

#ifdef SD_ITERATIVE_GMG
   //std::vector<HypreParMatrix*> fesMapSD(numSubdomains);

   for (int sd=0; sd<numSubdomains; ++sd)
     {
       //#ifndef SD_ITERATIVE_GMG_PA
       //VerifyMeshesAreEqual(pmeshSDcoarse[sd], pmeshSD[sd]);
       if (pmeshSDcoarse[sd] != NULL)
	 {
	   MFEM_VERIFY(pmeshSD[sd] != NULL, "");

	   // Note that the element attribute in pmeshSD[sd] and pmeshSDcoarse[sd] is the index of the corresponding pmesh element plus one.

	   HypreParMatrix *fesMapSD = CreateFESpaceMapForReorderedMeshes(pmeshSDcoarse[sd], sdfespace[sd], pmeshSD[sd], fec);

	   if (sd == 0 && myid == -10)
	     fesMapSD->Print("Fmap.txt");
	   
	   // On the finest level, replace sdP[sd][par_ref_levels-1] with fesMapSD times itself.
	   HypreParMatrix * MP = ParMult(fesMapSD, sdP[sd][sdP[sd].size() - 1]);
	   if (sd == 0 && myid == -10)
	     CompareHypreParMatrices(MP, sdP[sd][sdP[sd].size() - 1]);
	   
	   sdP[sd][sdP[sd].size() - 1] = MP;
	   
	   delete fesMapSD;
	 }
       else
	 MFEM_VERIFY(pmeshSD[sd] == NULL, "");
       //#endif
       
       // Now we can delete everything related to pmeshSDcoarse except sdP.
       if (pmeshSDcoarse[sd] != NULL)
	 {
	   delete sdfespace[sd];
	   delete pmeshSDcoarse[sd];
	 }
     }
#endif

   // 5.3. Create interface meshes.

#ifdef SERIAL_INTERFACES
   Mesh **smeshInterfaces = (numInterfaces > 0 ) ? new Mesh*[numInterfaces] : NULL;
#else
   ParMesh **pmeshInterfaces = (numInterfaces > 0 ) ? new ParMesh*[numInterfaces] : NULL;
#endif

   std::vector<int> interfaceFaceOffset;
   
   CreateInterfaceMeshes(numInterfaces, numSubdomains, myid, num_procs, sdMeshGen, 
			 interfaceGlobalToLocalMap, interfaceGI, pmeshSD, interfaceFaceOffset, interfaces,
#ifdef SERIAL_INTERFACES
			 smeshInterfaces
#else
			 pmeshInterfaces
#endif
			 );

   /*
#ifdef SERIAL_INTERFACES
   Mesh **smeshInterfaces = (numInterfaces > 0 ) ? new Mesh*[numInterfaces] : NULL;
#else
   ParMesh **pmeshInterfaces = (numInterfaces > 0 ) ? new ParMesh*[numInterfaces] : NULL;
#endif

#ifdef SERIAL_INTERFACES
   {
     // For each interface, the root process in pmeshSD[sd]->GetComm() for each of the neighboring subdomains sd
     // must create an empty interface if it does not exist already. 

     std::map<int, int> sdToId;
     std::vector<std::vector<int> > sdif;
     
     for (int i=0; i<numInterfaces; ++i)
       {
	 const int iloc = interfaceGlobalToLocalMap[i];  // Local interface index
       
	 if (iloc >= 0)
	   {
	     MFEM_VERIFY(interfaceGI[i] == interfaces[iloc].GetGlobalIndex(), "");
	     int sds[2] = {interfaces[iloc].FirstSubdomain(), interfaces[iloc].SecondSubdomain()};

	     for (int j=0; j<2; ++j)
	       {
		 if (pmeshSD[sds[j]] != NULL)
		   {
		     std::map<int, int>::iterator it = sdToId.find(sds[j]);
		     if (it == sdToId.end())
		       {
			 std::vector<int> ifi;
			 ifi.push_back(i);

			 sdToId[sds[j]] = sdif.size();
			 sdif.push_back(ifi);
		       }
		     else
		       {
			 MFEM_VERIFY(it->first == sds[j], "");

			 const int sdid = it->second;

			 sdif[sdid].push_back(i);
		       }
		   }
	       }
	   }
       }
     
     for (int sd=0; sd<numSubdomains; ++sd)
       {
	 if (pmeshSD[sd] != NULL)
	   {
	     int nif = 0;
	     std::vector<int> sbuf, recv;

	     int sdrank = -1;
	     int sdnprocs = -1;
	     MPI_Comm_rank(pmeshSD[sd]->GetComm(), &sdrank);
	     MPI_Comm_size(pmeshSD[sd]->GetComm(), &sdnprocs);
	     
	     std::vector<int> allnif(sdnprocs);
	     std::vector<int> rdspl(sdnprocs);

	     std::map<int, int>::iterator it = sdToId.find(sd);
	     if (it != sdToId.end())
	       {
		 MFEM_VERIFY(it->first == sd, "");
		 const int sdid = it->second;

		 nif = 2 * sdif[sdid].size();
		 for (auto ifid : sdif[sdid])
		   {
		     sbuf.push_back(ifid);
		     
		     MFEM_VERIFY(interfaceGlobalToLocalMap[ifid] >= 0, "");
		     sbuf.push_back(interfaces[interfaceGlobalToLocalMap[ifid]].GetOwningRank());
		   }
	       }

	     MPI_Gather(&nif, 1, MPI_INT, allnif.data(), 1, MPI_INT, 0, pmeshSD[sd]->GetComm());

	     int sumcnt = 0;
	     if (sdrank == 0)
	       {
		 for (int p=0; p<sdnprocs; ++p)
		   {
		     rdspl[p] = sumcnt;
		     sumcnt += allnif[p];
		   }
	     
		 recv.resize(sumcnt);
	       }
	     
	     MPI_Gatherv(sbuf.data(), nif, MPI_INT, recv.data(), allnif.data(), rdspl.data(), MPI_INT, 0, pmeshSD[sd]->GetComm());

	     if (sdrank == 0)
	       {
		 for (int j=0; j<sumcnt/2; ++j)
		   {
		     const int ifid = recv[2*j];
		     const int owningRank = recv[(2*j) + 1];
		     
		     if (interfaceGlobalToLocalMap[ifid] < 0)
		       {
			 const int sd0 = interfaceGI[ifid] / numSubdomains;
			 const int sd1 = interfaceGI[ifid] - (sd0 * numSubdomains);
			 
			 interfaceGlobalToLocalMap[ifid] = interfaces.size();
			 interfaces.push_back(SubdomainInterface(sd0, sd1, myid, true));  // Create empty interface

			 interfaces[interfaceGlobalToLocalMap[ifid]].SetOwningRank(owningRank);

			 const int ifgi = interfaces[interfaceGlobalToLocalMap[ifid]].SetGlobalIndex(numSubdomains);
			 MFEM_VERIFY(ifgi == interfaceGI[ifid], "");
		       }
		   }
	       }
	   }
       }
   }

   interfaceFaceOffset.assign(interfaces.size(), 0);
#endif

   for (int i=0; i<numInterfaces; ++i)
     {
       const int iloc = interfaceGlobalToLocalMap[i];  // Local interface index

       if (iloc >= 0)
	 {
	   MFEM_VERIFY(interfaceGI[i] == interfaces[iloc].GetGlobalIndex(), "");
#ifdef SERIAL_INTERFACES
	   int mustBuild = 1;
	   int sds[2] = {interfaces[iloc].FirstSubdomain(), interfaces[iloc].SecondSubdomain()};

	   for (int j=0; j<2; ++j)
	     {
	       if (pmeshSD[sds[j]] != NULL)
		 {
		   // Limit the serial mesh construction to one process having non-null pmeshSD, for each neighboring subdomain.
		   int sdrank = -1;
		   MPI_Comm_rank(pmeshSD[sds[j]]->GetComm(), &sdrank);

		   if (sdrank == 0)
		     mustBuild = 2;
		 }
	     }
	   
	   smeshInterfaces[i] = sdMeshGen.CreateSerialInterfaceMesh(interfaceFaceOffset[iloc], interfaces[iloc], mustBuild);
	   //interfaceFaceOffset[iloc] = 0;
#else
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(interfaces[iloc]);
#endif
	 }
       else
	 {
	   // This is not elegant. The problem is that SubdomainParMeshGenerator uses MPI_COMM_WORLD, so every
	   // process must call its functions, even if the process does not touch the interface. A solution would
	   // be to use the appropriate communicator or point-to-point communication, which would make the interface
	   // mesh generation more parallel.
	   const int sd0 = interfaceGI[i] / numSubdomains;  // globalIndex = (numSubdomains * sd0) + sd1;
	   const int sd1 = interfaceGI[i] - (numSubdomains * sd0);  // globalIndex = (numSubdomains * sd0) + sd1;
	   SubdomainInterface emptyInterface(sd0, sd1, myid, true);
	   emptyInterface.SetGlobalIndex(numSubdomains);
#ifdef SERIAL_INTERFACES
	   int elemOffset = 0;
	   Mesh* ifmesh = sdMeshGen.CreateSerialInterfaceMesh(elemOffset, emptyInterface, 1);
	   MFEM_VERIFY(ifmesh == NULL, "");
	   smeshInterfaces[i] = NULL;
#else
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(emptyInterface);
#endif
	 }
     }

   { // At this point, owningRank is set locally in each SubdomainInterface. Now determine the non-owning process.
     
     // First, count the number of local interfaces owned by each rank.

     std::vector<int> numOwnedByRank, numSharedByRank;
     numOwnedByRank.assign(num_procs, 0);
     numSharedByRank.assign(num_procs, 0);

     for (auto si : interfaces)
       {
	 if (!si.IsEmpty())
	   numOwnedByRank[si.GetOwningRank()]++;
       }

     MPI_Alltoall(numOwnedByRank.data(), 1, MPI_INT, numSharedByRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
     
#ifdef SERIAL_INTERFACES
     // Second, send interface indices to their owners.
     for (int r=0; r<num_procs; ++r)
       {
	 if (numOwnedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifid(numOwnedByRank[r]);
	     int cnt = 0;
	     
	     // This looks like bad complexity, but the small number of local interfaces and neighboring processes should keep this from being slow.
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 if (interfaceGlobalToLocalMap[i] >= 0)
		   {
		     if ((!interfaces[interfaceGlobalToLocalMap[i]].IsEmpty()) && interfaces[interfaceGlobalToLocalMap[i]].GetOwningRank() == r)
		       {
			 ifid[cnt] = i;
			 cnt++;
		       }
		   }
	       }

	     MFEM_VERIFY(cnt == numOwnedByRank[r], "");

	     MPI_Send(ifid.data(), numOwnedByRank[r], MPI_INT, r, myid, MPI_COMM_WORLD);
	   }

	 if (numSharedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifid(numSharedByRank[r]);

	     MPI_Recv(ifid.data(), numSharedByRank[r], MPI_INT, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	     for (int i=0; i<numSharedByRank[r]; ++i)
	       {
		 MFEM_VERIFY(interfaceGlobalToLocalMap[ifid[i]] >= 0, "");
		 interfaces[interfaceGlobalToLocalMap[ifid[i]]].SetSharingRank(r);
	       }
	   }
       }

     // Third, verify that owning and sharing ranks are set.
     for (auto si : interfaces)
       {
	 //MFEM_VERIFY(si.GetOwningRank() == myid || si.GetSharingRank() == myid, "");
	 MFEM_VERIFY(si.GetOwningRank() >= 0, ""); // && si.GetSharingRank() >= 0, "");
       }

     // Fourth, overwrite interfaceFaceOffset on sharing ranks with the values received from owning ranks.
     for (int r=0; r<num_procs; ++r)
       {
	 if (numSharedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifos(numSharedByRank[r]);
	     int cnt = 0;
	     
	     // This looks like bad complexity, but the small number of local interfaces and neighboring processes should keep this from being slow.
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 const int ifloc = interfaceGlobalToLocalMap[i];
		 if (ifloc >= 0)
		   {
		     if (interfaces[ifloc].GetSharingRank() == r)
		       {
			 ifos[cnt] = interfaceFaceOffset[ifloc];
			 cnt++;
		       }
		   }
	       }

	     MFEM_VERIFY(cnt == numSharedByRank[r], "");

	     MPI_Send(ifos.data(), numSharedByRank[r], MPI_INT, r, num_procs + myid, MPI_COMM_WORLD);
	   }

	 if (numOwnedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifos(numOwnedByRank[r]);

	     MPI_Recv(ifos.data(), numOwnedByRank[r], MPI_INT, r, num_procs + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	     int cnt = 0;
	     
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 const int ifloc = interfaceGlobalToLocalMap[i];
		 if (ifloc >= 0)
		   {
		     if (interfaces[ifloc].GetOwningRank() == r)
		       {
			 interfaceFaceOffset[ifloc] = ifos[cnt];
			 cnt++;
		       }
		   }
	       }
	     
	     MFEM_VERIFY(cnt == numOwnedByRank[r], "");
	   }
       }

     // Note that the interface owning and sharing ranks were used only to set interfaceFaceOffset and are no longer needed.
#endif
   }
   */
   
   MPI_Barrier(MPI_COMM_WORLD);
   
   /*
   {
     // Print the first vertex on each process, in order of rank.
     // This shows that as rank increases, the x-coordinate increases fastest, z slowest.
     // Consequently, partitioning the subdomains in the z direction will ensure that 
     // different subdomains are on different nodes, so that matrix memory is distributed.

     if (myid > 0)
       {
	 MPI_Status stat;
	 int num;
	 MPI_Recv(&num, 1, MPI_INT, myid-1, myid-1, MPI_COMM_WORLD, &stat);
       }
     
     cout << myid << ": first vertex " << pmesh->GetVertex(0)[0] << " " << pmesh->GetVertex(0)[1] << " "  << pmesh->GetVertex(0)[2] << endl;

     if (myid < num_procs-1)
       {
	 MPI_Send(&myid, 1, MPI_INT, myid+1, myid, MPI_COMM_WORLD);
       }
     
     return;
   }
   */
   /*
   { // debugging
     for (int i=0; i<numSubdomains; ++i)
       {
	 cout << "SD " << i << " y coordinate for vertex 0 " << pmeshSD[i]->GetVertex(0)[1] << endl;
       }

     for (int i=0; i<numInterfaces; ++i)
       {
	 cout << "IF " << i << " y coordinate for vertex 0 " << pmeshInterfaces[i]->GetVertex(0)[1] << endl;
       }
   }
   */
   
   // Note that subdomains do not overlap element-wise, and the parallel mesh of an individual subdomain has no element overlap on different processes.
   // However, the parallel mesh of an individual interface may have element (face) overlap on different processes, for the purpose of communication.
   // It is even possible (if an interface lies on a process boundary) for an entire interface to be duplicated on two processes, with zero true DOF's
   // on one process. 
   
   const bool testSubdomains = false;
   if (testSubdomains)
     {
       for (int i=0; i<numSubdomains; ++i)
	 {
	   ostringstream filename;
	   filename << "sd" << setfill('0') << setw(3) << i;
	   VisitTestPlotParMesh(filename.str(), pmeshSD[i], -1, myid);
	   //PrintMeshBoundingBox2(pmeshSD[i]);
	 }

#ifndef SERIAL_INTERFACES
       for (int i=0; i<numInterfaces; ++i)
	 {
	   ostringstream filename;
	   filename << "sdif" << setfill('0') << setw(3) << i;
	   VisitTestPlotParMesh(filename.str(), pmeshInterfaces[i], i, myid);
	 }
#endif
       
       const bool printInterfaceVertices = false;
       if (printInterfaceVertices)
	 {
	   for (int i=0; i<interfaces.size(); ++i)
	     {
	       cout << myid << ": Interface " << interfaces[i].GetGlobalIndex() << " has " << interfaces[i].NumVertices() << endl;
	       interfaces[i].PrintVertices(pmesh);
	     }
	 }
     }

   //TestHypreIdentity(MPI_COMM_WORLD);
   //TestHypreRectangularSerial();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   //FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   long globalNE = pmesh->GetGlobalNE();
   if (myid == 0)
   {
     cout << "Number of mesh elements: " << globalNE << endl;
     cout << "Number of finite element unknowns: " << size << endl;
     cout << "Root local number of finite element unknowns: " << fespace->TrueVSize() << endl;

     /*
     cout << "Maximum value for int: " << std::numeric_limits<int>::max() << endl;
     cout << "Maximum value for HYPRE_Int: " << std::numeric_limits<HYPRE_Int>::max() << endl;
     cout << "Maximum value for double: " << std::numeric_limits<double>::max() << endl;
     */
     
     {
       const HYPRE_Int maxint = 0.9*std::numeric_limits<int>::max();
       if (size > maxint)
	 cout << "WARNING: int max exceeded! Change type int to HYPRE_Int?" << endl;
       MFEM_VERIFY(size < maxint, "");
     }
   }
 
   /*
   if (myid == 0)
   { // Print the interface mesh
     ostringstream mesh_name;
     mesh_name << "ifmesh." << setfill('0') << setw(6) << myid;

     ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     pmeshInterfaces[0]->Print(mesh_ofs);
   }
   */

   StopWatch chronoDDC;
   chronoDDC.Clear();
   chronoDDC.Start();
   
   // 6.1. Create interface operator.

   DDMInterfaceOperator ddi(numSubdomains, numInterfaces, pmesh, fespace, pmeshSD,
#ifdef SERIAL_INTERFACES
			    smeshInterfaces, interfaceFaceOffset, 
#else
			    pmeshInterfaces,
#endif
			    order, pmesh->Dimension(),
			    &interfaces, &interfaceGlobalToLocalMap, -SIGMAVAL,
#ifdef GPWD
			    1,
#endif
#ifdef SD_ITERATIVE_GMG
			    sdP, &sdcRe, &sdcIm,
#endif
#ifdef SDFOSLS_PA
			    &coarseFOSLS,
#endif
			    hmin);

   cout << myid << ": DDI size " << ddi.Height() << " by " << ddi.Width() << endl;

   chronoDDC.Stop();
   if (myid == 0)
     cout << myid << ": DDMInterfaceOperator constructor timing " << chronoDDC.RealTime() << endl;

   /*
   {
     Vector t(ddi.Width());
     Vector s(ddi.Width());
     t = 1.0;

     //for (int i=ddi.Width()/2; i<ddi.Width(); ++i)
     //  t[i] = 2.0;

     for (int i=0; i<ddi.Width(); ++i)
       t[i] = i;

     if (myid == 1)
       {
	 //t = 2.0;
	 for (int i=0; i<ddi.Width(); ++i)
	   t[i] += ddi.Width();
       }
     
     s = 0.0;
     ddi.TestIFMult(t, s);
   }
   */
   
   //return 0;
   
   //PrintDenseMatrixOfOperator(ddi, num_procs, myid);
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, test2_E_exact);
   
#ifdef NO_GLOBAL_FEM
   Vector B(fespace->GetTrueVSize());
   Vector X(fespace->GetTrueVSize());
#else
   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
     //Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   //VectorFunctionCoefficient f(sdim, f_exact);
   VectorFunctionCoefficient f(sdim, test2_RHS_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   //VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(SIGMAVAL);
   //Coefficient *sigmaAbs = new ConstantCoefficient(fabs(SIGMAVAL));
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));

#ifdef AIRY_TEST
   VectorFunctionCoefficient epsilon(3, test_Airy_epsilon);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
#else
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
#endif
   
   //cout << myid << ": NBE " << pmesh->GetNBE() << endl;

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();
   a->Finalize();

   /*
   Vector exactSol(fespace->GetTrueVSize());
   x.GetTrueDofs(exactSol);
   */
   
   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   //a->FormSystemMatrix(ess_tdof_list, A);

   /*
   {
     Vector Ax(fespace->GetTrueVSize());
     A.Mult(exactSol, Ax);

     Ax -= B;
     cout << "Global projected error " << Ax.Norml2() << " relative to " << B.Norml2() << endl;
   }
   */
#endif // NO_GLOBAL_FEM
   
   //HypreParMatrix *Apa = a->ParallelAssemble();
   DenseMatrixInverse dddinv;
   
   if (false)
   { // Output ddi as a DenseMatrix
     const int Ndd = ddi.Height();
     Vector ej(Ndd);
     Vector Aej(Ndd);
     //DenseMatrix ddd(Ndd);

     ofstream sp("ddisparse.txt");

     const bool writeFile = true;
     
     for (int j=0; j<Ndd; ++j)
       {
	 cout << "Computing column " << j << " of " << Ndd << " of ddi" << endl;
	 
	 ej = 0.0;
	 ej[j] = 1.0;
	 ddi.Mult(ej, Aej);

	 for (int i=0; i<Ndd; ++i)
	   {
	     //ddd(i,j) = Aej[i];
	     
	     if (writeFile)
	       {
		 if (fabs(Aej[i]) > 1.0e-15)
		   sp << i+1 << " " << j+1 << " " << Aej[i] << endl;
	       }
	   }
       }

     sp.close();

     /*
     cout << "Factoring dense matrix" << endl;
     dddinv.Factor(ddd);
     cout << "Done factoring dense matrix" << endl;

     if (writeFile)
       {
	 ofstream ost("ddimat5.mat", std::ofstream::out);
	 ddd.PrintMatlab(ost);
       }
     */
   }
   
#ifndef NO_GLOBAL_FEM
   if (false)
   { // Test projection as solution
     ParBilinearForm *mbf = new ParBilinearForm(fespace);
     mbf->AddDomainIntegrator(new VectorFEMassIntegrator(*muinv));
     mbf->Assemble();
     mbf->Finalize();

     HypreParMatrix Mtest;

     mbf->FormSystemMatrix(ess_tdof_list, Mtest);
     //HypreParMatrix *Mpa = mbf->ParallelAssemble();
     
     ParGridFunction tgf(fespace);

     Vector uproj(fespace->GetTrueVSize());
     Vector Auproj(fespace->GetTrueVSize());
     Vector yproj(fespace->GetTrueVSize());
     Vector Myproj(fespace->GetTrueVSize());
     Vector MinvAuproj(fespace->GetTrueVSize());

     VectorFunctionCoefficient utest(3, test2_E_exact);
     VectorFunctionCoefficient ytest(3, test2_RHS_exact);

     tgf.ProjectCoefficient(utest);
     tgf.GetTrueDofs(uproj);

     tgf.ProjectCoefficient(ytest);
     tgf.GetTrueDofs(yproj);

     cout << myid << ": Norm of yproj " << yproj.Norml2() << endl;

     Mtest.Mult(yproj, Myproj);
     //Mpa->Mult(yproj, Myproj);

     cout << myid << ": Norm of Myproj " << Myproj.Norml2() << endl;

     A.Mult(uproj, Auproj);
     //Apa->Mult(uproj, Auproj);

     {
       HypreSolver *amg = new HypreBoomerAMG(Mtest);
       HyprePCG *pcg = new HyprePCG(Mtest);
       pcg->SetTol(1e-12);
       pcg->SetMaxIter(200);
       pcg->SetPrintLevel(2);
       pcg->SetPreconditioner(*amg);
       pcg->Mult(Auproj, MinvAuproj);

       //MinvAuproj -= yproj;
       
       tgf.SetFromTrueDofs(MinvAuproj);

       Vector zeroVec(3);
       zeroVec = 0.0;
       VectorConstantCoefficient vzero(zeroVec);

       //double L2e = tgf.ComputeL2Error(vzero);
       double L2e = tgf.ComputeL2Error(ytest);

       cout << myid << ": L2 error of MinvAuproj - yproj: " << L2e << endl;
     }
     
     cout << myid << ": Norm of Auproj " << Auproj.Norml2() << endl;

     Myproj -= Auproj;
     cout << myid << ": Norm of diff " << Myproj.Norml2() << endl;

     delete mbf;
   }

   //return;
   
   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }
   
   {
     cout << myid << ": A size " << A.Height() << " x " << A.Width() << endl;
     cout << myid << ": X size " << X.Size() << ", B size " << B.Size() << endl;
   }
#endif // NO_GLOBAL_FEM

   cout << myid << ": fespace size " << fespace->GetVSize() << ", true size " << fespace->GetTrueVSize() << endl;

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   //TestStrumpackConstructor();

   const bool solveDD = true;
   if (solveDD)
     {
       //cout << myid << ": B size " << B.Size() << ", norm " << B.Norml2() << endl;
       //cout << myid << ": fespace true V size " << fespace->GetTrueVSize() << endl;

       Vector Bdd(ddi.Width());
       Vector xdd(ddi.Width());

       Vector B_Im(B.Size());
       B_Im = 0.0;
       //B_Im = B;
       //B = 0.0;
       
       ddi.GetReducedSource(fespace, B, B_Im, Bdd, sdOrder);

       //ddi.TestProjectionError();
       
       //ddi.FullSystemAnalyticTest();

       //Bdd = 1.0;

       /*
       if (myid == 0)
	 {
	   ofstream ddrhsfile("bddtmp");
	   Bdd.Print(ddrhsfile);
	 }
       if (myid == 1)
	 {
	   ofstream ddrhsfile("bddtmp1");
	   Bdd.Print(ddrhsfile);
	 }
       */

       cout << myid << ": Bdd norm " << Bdd.Norml2() << endl;

       /*
       {
	 xdd = 0.0;
	 ddi.TestIFMult(Bdd, xdd);
       }
       */
       
       cout << "Solving DD system with gmres" << endl;

       GMRESSolver *gmres = new GMRESSolver(fespace->GetComm());
       //MINRESSolver *gmres = new MINRESSolver(fespace->GetComm());
       //BiCGSTABSolver *gmres = new BiCGSTABSolver(fespace->GetComm());
       //OrthominSolver *gmres = new OrthominSolver(fespace->GetComm());
       
       gmres->SetOperator(ddi);
       gmres->SetRelTol(1e-8);
       gmres->SetMaxIter(100);
       gmres->SetKDim(100);
       gmres->SetPrintLevel(1);

       /*
       DDMPreconditioner ddprec(&ddi);
       gmres->SetPreconditioner(ddprec);
       */
       
       //gmres->SetName("ddi");

       StopWatch chronoSolver;
       chronoSolver.Clear();
       chronoSolver.Start();
       
       xdd = 0.0;
       gmres->Mult(Bdd, xdd);
       //ddi.Mult(Bdd, xdd);

       //dddinv.Mult(Bdd, xdd);
       /*
       {
	 cout << "Reading FEM solution from file" << endl;
	 ifstream ddsolfile("xfemsp");
	 //ifstream ddsolfile("xfemsp2sd");
	 for (int i=0; i<X.Size(); ++i)
	   ddsolfile >> X[i];
       }
       */
       /*       
       {
	 cout << "Reading solution from file" << endl;
	 //ifstream ddsolfile("xddd4sd");
	 ifstream ddsolfile("xgm4sd0rho");
	 for (int i=0; i<xdd.Size(); ++i)
	   ddsolfile >> xdd[i];
	 
	 //xdd.Load(ddsolfile);
       }
       */
       
       cout << myid << ": xdd norm " << xdd.Norml2() << endl;

       /*
       if (myid == 0)
       {
	 ofstream ddsolfile("xgm4sd");
	 xdd.Print(ddsolfile);
       }
       */

       X = 0.0;

       Vector Xfem(X.Size());
       Xfem = X;
       
       ddi.RecoverDomainSolution(fespace, xdd, Xfem, X);

       //ddi.TestReconstructedFullDDSolution();
       
       chronoSolver.Stop();
       if (myid == 0)
	 cout << myid << ": Solver and recovery only timing " << chronoSolver.RealTime() << endl;

       delete gmres;
     }

   chrono.Stop();
   if (myid == 0)
     cout << myid << ": Total DDM timing (setup, solver, recovery) " << chrono.RealTime() << endl;

   /*
   { // Sleep in order to check memory usage using top.
     cout << "Sleeping" << endl;
     int q = -1;
     if (myid == 0)
       {
         MPI_Recv(&q, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }
     else
       {
         MPI_Recv(&q, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }

     return;
   }
   */
   
#ifdef NO_GLOBAL_FEM
   x.SetFromTrueDofs(X);
#else
   //#ifdef MFEM_USE_STRUMPACK
   if (use_strumpack)
     {
       const bool fullDirect = true;

       if (fullDirect)
	 {
	   /*
	   cout << "FULL DIRECT SOLVER" << endl;

	   Operator * Arow = new STRUMPACKRowLocMatrix(A);

	   STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
	   strumpack->SetPrintFactorStatistics(true);
	   strumpack->SetPrintSolveStatistics(false);
	   strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
	   strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
	   // strumpack->SetMC64Job(strumpack::MC64Job::NONE);
	   // strumpack->SetSymmetricPattern(true);
	   strumpack->SetOperator(*Arow);
	   strumpack->SetFromCommandLine();
	   //Solver * precond = strumpack;

	   strumpack->Mult(B, X);

	   if (myid == -10)
	     {
	       ofstream solfile("xairy27b");
	       X.Print(solfile);
	     }
	   
	   delete strumpack;
	   delete Arow;
	   */
	 }
       else
	 {
	   ParFiniteElementSpace *prec_fespace =
	     (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
	   HypreSolver *ams = new HypreAMS(A, prec_fespace);

#ifdef HYPRE_DYLAN
	   {
	     Vector Xtmp(X);
	     ams->Mult(B, Xtmp);  // Just a hack to get ams to run its setup function. There should be a better way.
	   }

	   GMRESSolver *gmres = new GMRESSolver(fespace->GetComm());
	   //FGMRESSolver *gmres = new FGMRESSolver(fespace->GetComm());
	   //BiCGSTABSolver *gmres = new BiCGSTABSolver(fespace->GetComm());
	   //MINRESSolver *gmres = new MINRESSolver(fespace->GetComm());
	   
	   gmres->SetOperator(A);
	   gmres->SetRelTol(1e-16);
	   gmres->SetMaxIter(1000);
	   gmres->SetPrintLevel(1);

	   gmres->SetPreconditioner(*ams);
	   gmres->Mult(B, X);
#else
	   HypreGMRES *gmres = new HypreGMRES(A);
	   gmres->SetTol(1e-12);
	   gmres->SetMaxIter(100);
	   gmres->SetPrintLevel(10);
	   gmres->SetPreconditioner(*ams);
	   gmres->Mult(B, X);

	   delete gmres;
	   //delete iams;
	   //delete ams;
#endif
	 }
     }
   else
     //#endif
     {
       // 12. Define and apply a parallel PCG solver for AX=B with the AMS
       //     preconditioner from hypre.
       ParFiniteElementSpace *prec_fespace =
	 (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
       HypreSolver *ams = new HypreAMS(A, prec_fespace);
       HyprePCG *pcg = new HyprePCG(A);
       pcg->SetTol(1e-12);
       pcg->SetMaxIter(100);
       pcg->SetPrintLevel(2);
       pcg->SetPreconditioner(*ams);
       pcg->Mult(B, X);

       delete pcg;
       delete ams;
     }
   
   {
     // Check residual
     Vector res(X.Size());
     Vector ssol(X.Size());
     ssol = X;

     const double Bnrm = B.Norml2();
     const double Bnrm2 = Bnrm*Bnrm;
	     
     A.Mult(ssol, res);
     res -= B;

     const double Rnrm = res.Norml2();
     const double Rnrm2 = Rnrm*Rnrm;

     double sumBnrm2 = 0.0;
     double sumRnrm2 = 0.0;
     MPI_Allreduce(&Bnrm2, &sumBnrm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&Rnrm2, &sumRnrm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

     if (myid == 0)
       {
	 cout << myid << ": Real FEM system residual norm " << sqrt(sumRnrm2) << ", B norm " << sqrt(sumBnrm2) << endl;
       }
   }
   
   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);
#endif // NO_GLOBAL_FEM

   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      Vector zeroVec(3);
      zeroVec = 0.0;
      VectorConstantCoefficient vzero(zeroVec);
      ParGridFunction zerogf(fespace);
      zerogf = 0.0;
      double normE = zerogf.ComputeL2Error(E);
      double normX = x.ComputeL2Error(vzero);
      if (myid == 0)
      {
         cout << "|| E_h - E ||_{L^2} = " << err << endl;
	 cout << "|| E_h ||_{L^2} = " << normX << endl;
	 cout << "|| E ||_{L^2} = " << normE << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   
   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }
  
   //cout << "Final element 0 size " << pmesh->GetElementSize(0) << ", number of elements " << pmesh->GetGlobalNE() << endl;
   
   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
     {
       bool binary = false;
       if (binary)
	 {
#ifdef MFEM_USE_SIDRE
	   dc = new SidreDataCollection("ddsol", pmesh);
#else
	   MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
	 }
       else
	 {
	   dc = new VisItDataCollection("ddsol", pmesh);
	   dc->SetPrecision(8);
	   // To save the mesh using MFEM's parallel mesh format:
	   // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
	 }
       dc->RegisterField("solution", &x);
       dc->SetCycle(0);
       dc->SetTime(0.0);
       dc->Save();
     }
   
   // 17. Free the used memory.
#ifndef NO_GLOBAL_FEM
   delete a;
   delete sigma;
   delete muinv;
   delete b;
#endif
   delete fespace;
   delete fec;
   delete pmesh;

   chronoMain.Stop();
   if (myid == 0)
     cout << myid << ": Total main timing " << chronoMain.RealTime() << endl;
   
   MPI_Finalize();

   //outfile.close();
   
   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (SIGMAVAL + kappa * kappa) * sin(kappa * x(1));
      f(1) = (SIGMAVAL + kappa * kappa) * sin(kappa * x(2));
      f(2) = (SIGMAVAL + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

double radiusFunction(const Vector &x)
{
  double f = 0.0;
  for (int i=0; i<dim; ++i)
    f += x[i]*x[i];
  
  return sqrt(f);
}

void CreateInterfaceMeshes(const int numInterfaces, const int numSubdomains, const int myid, const int num_procs, SubdomainParMeshGenerator & sdMeshGen,
			   std::vector<int> & interfaceGlobalToLocalMap,
			   std::vector<int> const& interfaceGI, ParMesh **pmeshSD, std::vector<int> & interfaceFaceOffset,
			   vector<SubdomainInterface> & interfaces,
#ifdef SERIAL_INTERFACES
			   Mesh **smeshInterfaces
#else
			   ParMesh **pmeshInterfaces
#endif
			   )
{
  /*
#ifdef SERIAL_INTERFACES
  Mesh **smeshInterfaces = (numInterfaces > 0 ) ? new Mesh*[numInterfaces] : NULL;
#else
  ParMesh **pmeshInterfaces = (numInterfaces > 0 ) ? new ParMesh*[numInterfaces] : NULL;
#endif
  */
  
#ifdef SERIAL_INTERFACES
   {
     // For each interface, the root process in pmeshSD[sd]->GetComm() for each of the neighboring subdomains sd
     // must create an empty interface if it does not exist already. 

     std::map<int, int> sdToId;
     std::vector<std::vector<int> > sdif;
     
     for (int i=0; i<numInterfaces; ++i)
       {
	 const int iloc = interfaceGlobalToLocalMap[i];  // Local interface index
       
	 if (iloc >= 0)
	   {
	     MFEM_VERIFY(interfaceGI[i] == interfaces[iloc].GetGlobalIndex(), "");
	     int sds[2] = {interfaces[iloc].FirstSubdomain(), interfaces[iloc].SecondSubdomain()};

	     for (int j=0; j<2; ++j)
	       {
		 if (pmeshSD[sds[j]] != NULL)
		   {
		     std::map<int, int>::iterator it = sdToId.find(sds[j]);
		     if (it == sdToId.end())
		       {
			 std::vector<int> ifi;
			 ifi.push_back(i);

			 sdToId[sds[j]] = sdif.size();
			 sdif.push_back(ifi);
		       }
		     else
		       {
			 MFEM_VERIFY(it->first == sds[j], "");

			 const int sdid = it->second;

			 sdif[sdid].push_back(i);
		       }
		   }
	       }
	   }
       }
     
     for (int sd=0; sd<numSubdomains; ++sd)
       {
	 if (pmeshSD[sd] != NULL)
	   {
	     int nif = 0;
	     std::vector<int> sbuf, recv;

	     int sdrank = -1;
	     int sdnprocs = -1;
	     MPI_Comm_rank(pmeshSD[sd]->GetComm(), &sdrank);
	     MPI_Comm_size(pmeshSD[sd]->GetComm(), &sdnprocs);
	     
	     std::vector<int> allnif(sdnprocs);
	     std::vector<int> rdspl(sdnprocs);

	     std::map<int, int>::iterator it = sdToId.find(sd);
	     if (it != sdToId.end())
	       {
		 MFEM_VERIFY(it->first == sd, "");
		 const int sdid = it->second;

		 nif = 2 * sdif[sdid].size();
		 for (auto ifid : sdif[sdid])
		   {
		     sbuf.push_back(ifid);
		     
		     MFEM_VERIFY(interfaceGlobalToLocalMap[ifid] >= 0, "");
		     sbuf.push_back(interfaces[interfaceGlobalToLocalMap[ifid]].GetOwningRank());
		   }
	       }

	     MPI_Gather(&nif, 1, MPI_INT, allnif.data(), 1, MPI_INT, 0, pmeshSD[sd]->GetComm());

	     int sumcnt = 0;
	     if (sdrank == 0)
	       {
		 for (int p=0; p<sdnprocs; ++p)
		   {
		     rdspl[p] = sumcnt;
		     sumcnt += allnif[p];
		   }
	     
		 recv.resize(sumcnt);
	       }
	     
	     MPI_Gatherv(sbuf.data(), nif, MPI_INT, recv.data(), allnif.data(), rdspl.data(), MPI_INT, 0, pmeshSD[sd]->GetComm());

	     if (sdrank == 0)
	       {
		 for (int j=0; j<sumcnt/2; ++j)
		   {
		     const int ifid = recv[2*j];
		     const int owningRank = recv[(2*j) + 1];
		     
		     if (interfaceGlobalToLocalMap[ifid] < 0)
		       {
			 const int sd0 = interfaceGI[ifid] / numSubdomains;
			 const int sd1 = interfaceGI[ifid] - (sd0 * numSubdomains);
			 
			 interfaceGlobalToLocalMap[ifid] = interfaces.size();
			 interfaces.push_back(SubdomainInterface(sd0, sd1, myid, true));  // Create empty interface

			 interfaces[interfaceGlobalToLocalMap[ifid]].SetOwningRank(owningRank);

			 const int ifgi = interfaces[interfaceGlobalToLocalMap[ifid]].SetGlobalIndex(numSubdomains);
			 MFEM_VERIFY(ifgi == interfaceGI[ifid], "");
		       }
		   }
	       }
	   }
       }
   }

   //std::vector<int> interfaceFaceOffset;
   interfaceFaceOffset.assign(interfaces.size(), 0);
#endif

   for (int i=0; i<numInterfaces; ++i)
     {
       const int iloc = interfaceGlobalToLocalMap[i];  // Local interface index

       if (iloc >= 0)
	 {
	   MFEM_VERIFY(interfaceGI[i] == interfaces[iloc].GetGlobalIndex(), "");
#ifdef SERIAL_INTERFACES
	   int mustBuild = 1;
	   int sds[2] = {interfaces[iloc].FirstSubdomain(), interfaces[iloc].SecondSubdomain()};

	   for (int j=0; j<2; ++j)
	     {
	       if (pmeshSD[sds[j]] != NULL)
		 {
		   // Limit the serial mesh construction to one process having non-null pmeshSD, for each neighboring subdomain.
		   int sdrank = -1;
		   MPI_Comm_rank(pmeshSD[sds[j]]->GetComm(), &sdrank);

		   if (sdrank == 0)
		     mustBuild = 2;
		 }
	     }
	   
	   smeshInterfaces[i] = sdMeshGen.CreateSerialInterfaceMesh(interfaceFaceOffset[iloc], interfaces[iloc], mustBuild);
	   //interfaceFaceOffset[iloc] = 0;
#else
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(interfaces[iloc]);
#endif
	 }
       else
	 {
	   // This is not elegant. The problem is that SubdomainParMeshGenerator uses MPI_COMM_WORLD, so every
	   // process must call its functions, even if the process does not touch the interface. A solution would
	   // be to use the appropriate communicator or point-to-point communication, which would make the interface
	   // mesh generation more parallel.
	   const int sd0 = interfaceGI[i] / numSubdomains;  // globalIndex = (numSubdomains * sd0) + sd1;
	   const int sd1 = interfaceGI[i] - (numSubdomains * sd0);  // globalIndex = (numSubdomains * sd0) + sd1;
	   SubdomainInterface emptyInterface(sd0, sd1, myid, true);
	   emptyInterface.SetGlobalIndex(numSubdomains);
#ifdef SERIAL_INTERFACES
	   int elemOffset = 0;
	   Mesh* ifmesh = sdMeshGen.CreateSerialInterfaceMesh(elemOffset, emptyInterface, 1);
	   MFEM_VERIFY(ifmesh == NULL, "");
	   smeshInterfaces[i] = NULL;
#else
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(emptyInterface);
#endif
	 }
     }

   { // At this point, owningRank is set locally in each SubdomainInterface. Now determine the non-owning process.
     
     // First, count the number of local interfaces owned by each rank.

     std::vector<int> numOwnedByRank, numSharedByRank;
     numOwnedByRank.assign(num_procs, 0);
     numSharedByRank.assign(num_procs, 0);

     for (auto si : interfaces)
       {
	 if (!si.IsEmpty())
	   numOwnedByRank[si.GetOwningRank()]++;
       }

     MPI_Alltoall(numOwnedByRank.data(), 1, MPI_INT, numSharedByRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
     
#ifdef SERIAL_INTERFACES
     // Second, send interface indices to their owners.
     for (int r=0; r<num_procs; ++r)
       {
	 if (numOwnedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifid(numOwnedByRank[r]);
	     int cnt = 0;
	     
	     // This looks like bad complexity, but the small number of local interfaces and neighboring processes should keep this from being slow.
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 if (interfaceGlobalToLocalMap[i] >= 0)
		   {
		     if ((!interfaces[interfaceGlobalToLocalMap[i]].IsEmpty()) && interfaces[interfaceGlobalToLocalMap[i]].GetOwningRank() == r)
		       {
			 ifid[cnt] = i;
			 cnt++;
		       }
		   }
	       }

	     MFEM_VERIFY(cnt == numOwnedByRank[r], "");

	     MPI_Send(ifid.data(), numOwnedByRank[r], MPI_INT, r, myid, MPI_COMM_WORLD);
	   }

	 if (numSharedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifid(numSharedByRank[r]);

	     MPI_Recv(ifid.data(), numSharedByRank[r], MPI_INT, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	     for (int i=0; i<numSharedByRank[r]; ++i)
	       {
		 MFEM_VERIFY(interfaceGlobalToLocalMap[ifid[i]] >= 0, "");
		 interfaces[interfaceGlobalToLocalMap[ifid[i]]].SetSharingRank(r);
	       }
	   }
       }

     // Third, verify that owning and sharing ranks are set.
     for (auto si : interfaces)
       {
	 //MFEM_VERIFY(si.GetOwningRank() == myid || si.GetSharingRank() == myid, "");
	 MFEM_VERIFY(si.GetOwningRank() >= 0, ""); // && si.GetSharingRank() >= 0, "");
       }

     // Fourth, overwrite interfaceFaceOffset on sharing ranks with the values received from owning ranks.
     for (int r=0; r<num_procs; ++r)
       {
	 if (numSharedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifos(numSharedByRank[r]);
	     int cnt = 0;
	     
	     // This looks like bad complexity, but the small number of local interfaces and neighboring processes should keep this from being slow.
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 const int ifloc = interfaceGlobalToLocalMap[i];
		 if (ifloc >= 0)
		   {
		     if (interfaces[ifloc].GetSharingRank() == r)
		       {
			 ifos[cnt] = interfaceFaceOffset[ifloc];
			 cnt++;
		       }
		   }
	       }

	     MFEM_VERIFY(cnt == numSharedByRank[r], "");

	     MPI_Send(ifos.data(), numSharedByRank[r], MPI_INT, r, num_procs + myid, MPI_COMM_WORLD);
	   }

	 if (numOwnedByRank[r] > 0 && r != myid)
	   {
	     std::vector<int> ifos(numOwnedByRank[r]);

	     MPI_Recv(ifos.data(), numOwnedByRank[r], MPI_INT, r, num_procs + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	     int cnt = 0;
	     
	     for (int i=0; i<numInterfaces; ++i)
	       {
		 const int ifloc = interfaceGlobalToLocalMap[i];
		 if (ifloc >= 0)
		   {
		     if (interfaces[ifloc].GetOwningRank() == r)
		       {
			 interfaceFaceOffset[ifloc] = ifos[cnt];
			 cnt++;
		       }
		   }
	       }
	     
	     MFEM_VERIFY(cnt == numOwnedByRank[r], "");
	   }
       }

     // Note that the interface owning and sharing ranks were used only to set interfaceFaceOffset and are no longer needed.
#endif
   }
}
