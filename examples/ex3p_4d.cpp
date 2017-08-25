//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh
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
#include "./spe10_coeff.cpp"

using namespace std;
using namespace mfem;


int* LoadIterations(int NRows, int NCol)
{
  ifstream in("iter_curl.txt");

  //initialize
  int *iters = new int[NCol*NRows];
  for(int col = 0; col < NCol; col++)
  {
    for(int row = 0; row < NRows; row++)
    {
      iters[row*NCol+col] = -1;
    }
  }

  if (!in)
  {
    cout << "Cannot open file.\n";
    return iters;
  }

  for(int row = 0; row < NRows; row++)
	  for(int col = 0; col < NCol; col++)
		{
		  if(in.eof())
		  {
			  in.close();
			  return iters;
		  }
		  in >> iters[row*NCol+col];
		}


  in.close();

  return iters;
}

void putIterationsInArray(int iter, int row, int col, int NCol, int* iters)
{
	iters[row*NCol+col] = iter;
}

void WriteIterations(int *iters, int NRows, int NCol)
{
	ofstream out;
	out.open("iter_curl.txt",fstream::out);

	if (!out)
	{
		cout << "Cannot open file.\n";
		delete[] iters;

		return;
	}

	for(int row = 0; row < NRows; row++)
	{
		for(int col = 0; col < NCol; col++)
		{
		  out << iters[row*NCol+col] << "\t";
		}
		out << endl;
	}
	out.close();

	delete[] iters;
}


// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa = 1.0;
int dim;

double osziCoeff(const Vector &x)
{
	return 1.0001 + sin(100*x(0))*sin(200*x(1))*sin(300*x(2))*sin(400*x(3));
}

class Curl4dPrec : public Solver
{

private:
	HypreParMatrix *A;
	ParFiniteElementSpace *fespace;
	Coefficient *alpha_, *beta_;

	HypreParMatrix *idMat;
	HypreParMatrix *H1VecLaplaceMat;
	HypreBoomerAMG *amgVecH1;


	HypreParMatrix *gradMat;
	HypreParMatrix *H1LaplaceMat;
	HypreBoomerAMG *amgH1;

	HypreSmoother * smoother;
	CGSolver *pcgGrad;
	CGSolver *pcgH1Vec;

	Vector *f;
	Vector *fGrad, *uGrad;
	Vector *fH1Vec, *uH1Vec;

	bool exactSolves;

public:
	~Curl4dPrec()
	{
		delete pcgH1Vec;
		delete pcgGrad;

		delete 	f, fGrad, uGrad, fH1Vec, uH1Vec;

		delete smoother;

		delete amgVecH1, H1VecLaplaceMat;
		delete idMat;
		delete amgH1, H1LaplaceMat;
		delete gradMat;
	}

	Curl4dPrec(HypreParMatrix *AUser, ParFiniteElementSpace *fespaceUser, Coefficient *alpha, Coefficient *beta,
			const Array<int> &essBnd, int orderKernel=1, bool exactSolvesUser=false)
	{
		A = AUser;
		fespace = fespaceUser;
		alpha_ = alpha;
		beta_ = beta;

		ParMesh *pmesh = fespace->GetParMesh();
		int dim = pmesh->Dimension();

		exactSolves = exactSolvesUser;

		int orderIm=1;  //vecH1  --> H(curl)
		int orderKer=orderKernel; //grad V --> H(curl)

		smoother = new HypreSmoother(*A, 16, 3);

//		//for the pure dirichlet case
//		Array<int> essBnd(pmesh->bdr_attributes.Max()); essBnd = 1;

		Array<int> HCurl_essDof(fespace->GetVSize()); HCurl_essDof = 0;
		fespace->GetEssentialVDofs(essBnd, HCurl_essDof);

		//setup the H1 FESpace
		FiniteElementCollection* fecH1;
		   if(orderKer==1) fecH1 = new LinearFECollection;
		   else fecH1 = new QuadraticFECollection;

		ParFiniteElementSpace *H1FESpace = new ParFiniteElementSpace(pmesh, fecH1);
		Array<int> H1_essDof(H1FESpace->GetVSize()); H1_essDof = 0;
		H1FESpace->GetEssentialVDofs(essBnd, H1_essDof);

		//setup the discrete gradient
		ParDiscreteLinearOperator *disGrad = new ParDiscreteLinearOperator(H1FESpace, fespace);
		disGrad->AddDomainInterpolator(new GradientInterpolator);
		disGrad->Assemble();
		disGrad->Finalize();
		SparseMatrix* smat = &(disGrad->SpMat());
		smat->EliminateCols(H1_essDof);
		for(int dof=0; dof<HCurl_essDof.Size(); dof++) if(HCurl_essDof[dof]<0) smat->EliminateRow(dof);
		gradMat = disGrad->ParallelAssemble();
		delete disGrad;

		//setup the H1 preconditioner
		ParBilinearForm* H1Varf = new ParBilinearForm(H1FESpace);
		H1Varf->AddDomainIntegrator(new DiffusionIntegrator(*beta_));
//		H1Varf->AddDomainIntegrator(new MassIntegrator);
		H1Varf->Assemble();
		H1Varf->Finalize();

		SparseMatrix &matH1(H1Varf->SpMat());
		for(int dof=0; dof<H1_essDof.Size(); dof++) if(H1_essDof[dof]<0) matH1.EliminateRowCol(dof);
		H1LaplaceMat = H1Varf->ParallelAssemble();
		delete H1Varf;
		amgH1 = new HypreBoomerAMG(*H1LaplaceMat);


		//setup the H1 injection
		FiniteElementCollection* fecH1Vec;
		   if(orderIm==1) fecH1Vec = new LinearFECollection;
		   else fecH1Vec = new QuadraticFECollection;
		ParFiniteElementSpace *H1VecFESpace = new ParFiniteElementSpace(pmesh, fecH1Vec, dim, Ordering::byVDIM);
		Array<int> H1Vec_essDof(H1VecFESpace->GetVSize()); H1Vec_essDof = 0;
		H1VecFESpace->GetEssentialVDofs(essBnd, H1Vec_essDof);

		//setup the discrete gradient
		ParDiscreteLinearOperator *disInterpol = new ParDiscreteLinearOperator(H1VecFESpace, fespace);
		disInterpol->AddDomainInterpolator(new IdentityInterpolator);
		disInterpol->Assemble();
		disInterpol->Finalize();
		SparseMatrix* smatID = &(disInterpol->SpMat());
		smatID->EliminateCols(H1Vec_essDof);
		for(int dof=0; dof<HCurl_essDof.Size(); dof++) if(HCurl_essDof[dof]<0) smatID->EliminateRow(dof);
		idMat = disInterpol->ParallelAssemble();
		delete disInterpol;

		//setup the H1-vec preconditioner
		ParBilinearForm* H1VecVarf = new ParBilinearForm(H1VecFESpace);
		H1VecVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_));
		H1VecVarf->AddDomainIntegrator(new VectorMassIntegrator(-1, beta_));
		H1VecVarf->Assemble();
		H1VecVarf->Finalize();

		SparseMatrix &matH1Vec(H1VecVarf->SpMat());
		for(int dof=0; dof<H1Vec_essDof.Size(); dof++) if(H1Vec_essDof[dof]<0) matH1Vec.EliminateRowCol(dof);
		H1VecLaplaceMat = H1VecVarf->ParallelAssemble();
		delete H1VecVarf;
		amgVecH1 = new HypreBoomerAMG(*H1VecLaplaceMat);
		amgVecH1->SetSystemsOptions(dim);


		f = new Vector(fespace->GetTrueVSize());

		fGrad = new Vector(H1FESpace->GetTrueVSize());
		uGrad = new Vector(H1FESpace->GetTrueVSize());

		fH1Vec = new Vector(H1VecFESpace->GetTrueVSize());
		uH1Vec = new Vector(H1VecFESpace->GetTrueVSize());


		amgH1->Mult(*fGrad, *uGrad);
		amgVecH1->Mult(*fH1Vec, *uH1Vec);

		pcgGrad = new CGSolver(MPI_COMM_WORLD);
		pcgGrad->SetOperator(*H1LaplaceMat);
		pcgGrad->SetPreconditioner(*amgH1);
		pcgGrad->SetRelTol(1e-16);
		pcgGrad->SetMaxIter(100000000);
		pcgGrad->SetPrintLevel(-2);

		pcgH1Vec = new CGSolver(MPI_COMM_WORLD);
		pcgH1Vec->SetOperator(*H1VecLaplaceMat);
		pcgH1Vec->SetPreconditioner(*amgVecH1);
		pcgH1Vec->SetRelTol(1e-16);
		pcgH1Vec->SetMaxIter(100000000);
		pcgH1Vec->SetPrintLevel(-2);

		delete H1FESpace; delete fecH1;
		delete H1VecFESpace; delete fecH1Vec;

	}

	void setExactSolve(bool exSol)
	{
		exactSolves = exSol;
	}

	virtual void Mult(const Vector &x, Vector &y) const
	{
		smoother->Mult(x,y);

		idMat->MultTranspose(x,*fH1Vec);
		*uH1Vec = 0.0;
		if(exactSolves) pcgH1Vec->Mult(*fH1Vec, *uH1Vec);
		else amgVecH1->Mult(*fH1Vec, *uH1Vec);
		idMat->Mult(1.0, *uH1Vec, 1.0, y);

		gradMat->MultTranspose(x,*fGrad);
		*uGrad = 0.0;
		if(exactSolves) pcgGrad->Mult(*fGrad, *uGrad);
		else amgH1->Mult(*fGrad, *uGrad);
		gradMat->Mult(1.0, *uGrad, 1.0, y);

	}

	virtual void SetOperator(const Operator &op) {};

};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   bool verbose = (myid==0);
   
   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int order = 1;
   bool set_bc = true;
   bool static_cond = false;
   bool visualization = 1;
   int sequ_ref_levels = 0;
   int par_ref_levels = 0;
   double tol = 1e-6;
   double coeffWeight = 1.0;
   bool exactH1Solver = false;
   bool spe10Coeff = false;
   bool standardCG = true;

   int NExpo = 8;
   int weightStart = -NExpo;
   int weightEnd = NExpo;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sequ_ref_levels, "-sr", "--seqrefinement",
                     "Number of sequential refinement steps.");
   args.AddOption(&par_ref_levels, "-pr", "--parrefinement",
                     "Number of parallel refinement steps.");
   args.AddOption(&order, "-o", "--order",
                     "Polynomial order of the finite element space.");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&tol, "-tol", "--tol",
                     "A parameter.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&coeffWeight, "-c", "--coeffMass",
                  "the weight for the mass term.");
   args.AddOption(&exactH1Solver, "-exH1Sol", "--exactH1Solver", "-H1prec", "--H1preconditioner",
                  "Use exact H1 solvers for the preconditioner.");
   args.AddOption(&spe10Coeff, "-spe10", "--useSPE10Coeff", "-constCoeff", "--constCoeff",
                  "Switch between the coefficients for the mass bilinear form.");
   args.AddOption(&standardCG, "-sCG", "--stdCG", "-rCG", "--resCG",
                  "Switch between standard PCG or recompute residuals in every step and use the residuals itself for the stopping criteria.");
   args.AddOption(&weightStart, "-ws", "--weightStart",
                  "the exponent for the starting weight (for the mass term).");
   args.AddOption(&weightEnd, "-we", "--weightEnd",
                  "the exponent for the weight at the end (for the mass term).");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if(verbose) args.PrintOptions(cout);
   
   kappa = freq * M_PI;

   Mesh *mesh;
   ifstream imesh(mesh_file);
   if(!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }

   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   
   if(dim !=4 || sdim != 4)
   {
	   MPI_Finalize();
	   return 0;
   }

   for(int i=0; i<sequ_ref_levels; i++) mesh->UniformRefinement();
   if(verbose) mesh->PrintCharacteristics();

   if(verbose) cout << "now we partition the mesh..." << endl << endl;

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for(int i=0; i<par_ref_levels; i++) pmesh->UniformRefinement();

      pmesh->ReorientTetMesh();
   
   pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec;
   if(dim==4)
   {
	   if(order==1) fec = new ND1_4DFECollection;
	   else fec = new ND2_4DFECollection;
   }
   else fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = set_bc ? 1 : 0;
   if (pmesh->bdr_attributes.Size())
   {
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.


   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);

   for(int expo=weightStart; expo<=weightEnd; expo++)
   {
	   double weight = pow(10.0,expo);
	   kappa = weight;

	   VectorFunctionCoefficient f(sdim, f_exact);
	   ParLinearForm *b = new ParLinearForm(fespace);
	   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
	   b->Assemble();

	   x.ProjectCoefficient(E);

	   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
	   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
	   //     mass domain integrators.
//	   std::string permFile = "spe_perm.dat";
//	   InversePermeabilityFunction::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);

	   Coefficient *alpha = new ConstantCoefficient(1.0);
	   Coefficient *beta;
//	   if(spe10Coeff) beta = new FunctionCoefficient(InversePermeabilityFunction::Norm2Permeability);
//	   else
		   beta = new ConstantCoefficient(weight);

	   ParBilinearForm *a = new ParBilinearForm(fespace);
	   a->AddDomainIntegrator(new CurlCurlIntegrator(*alpha));
	   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));

	   // 11. Assemble the parallel bilinear form and the corresponding linear
	   //     system, applying any necessary transformations such as: parallel
	   //     assembly, eliminating boundary conditions, applying conforming
	   //     constraints for non-conforming AMR, static condensation, etc.
	   if (static_cond) { a->EnableStaticCondensation(); }
	   a->Assemble();

	   HypreParMatrix A;
	   Vector B, X;
	   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

	   if (myid == 0)
	   {
		  cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
	   }

	   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
	   //     preconditioner from hypre.
	   ParFiniteElementSpace *prec_fespace =
		  (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
	   Solver *prec;
	   if(dim<=3) prec = new HypreAMS(A, prec_fespace);
	   else if(dim==4) prec = new Curl4dPrec(&A, fespace, alpha, beta, ess_bdr, order, false);
	   IterativeSolver *pcg = new CGSolver(MPI_COMM_WORLD);
	   pcg->SetOperator(A);
	   pcg->SetRelTol(tol);
	   pcg->SetMaxIter(5000);
	   pcg->SetPrintLevel(1);
	   pcg->SetPreconditioner(*prec);
	   pcg->Mult(B, X);

	   int iter = pcg->GetNumIterations();
	   if(myid==0)
	   {
		   cout << "Weigth: " << weight << " " << iter << endl;

		   int *iters = LoadIterations(10, 2*NExpo+1);
		   putIterationsInArray(iter, sequ_ref_levels+par_ref_levels, expo+NExpo, 2*NExpo+1, iters);
		   WriteIterations(iters, 10, 2*NExpo+1);
	   }

	   // 13. Recover the parallel grid function corresponding to X. This is the
	   //     local finite element solution on each processor.
	   a->RecoverFEMSolution(X, *b, x);

	   // 14. Compute and print the L^2 norm of the error.
	   {
		  double err = x.ComputeL2Error(E);
		  if (myid == 0)
		  {
			 cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
		  }
	   }

	   // 15. Save the refined mesh and the solution in parallel. This output can
	   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
	//   {
	//      ostringstream mesh_name, sol_name;
	//      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
	//      sol_name << "sol." << setfill('0') << setw(6) << myid;
	//
	//      ofstream mesh_ofs(mesh_name.str().c_str());
	//      mesh_ofs.precision(8);
	//      pmesh->Print(mesh_ofs);
	//
	//      ofstream sol_ofs(sol_name.str().c_str());
	//      sol_ofs.precision(8);
	//      x.Save(sol_ofs);
	//   }

	//   // 16. Send the solution by socket to a GLVis server.
	//   if (visualization)
	//   {
	//      char vishost[] = "localhost";
	//      int  visport   = 19916;
	//      socketstream sol_sock(vishost, visport);
	//      sol_sock << "parallel " << num_procs << " " << myid << "\n";
	//      sol_sock.precision(8);
	//      sol_sock << "solution\n" << *pmesh << x << flush;
	//   }

	   delete pcg;
	   delete prec;
	   delete a;
	   delete alpha;
	   delete beta;
	   delete b;

   }

   // 17. Free the used memory.

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if(dim==4)
   {
      E(0) =  sin(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
      E(1) = -cos(M_PI*x(0))*sin(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
      E(2) =  cos(M_PI*x(0))*cos(M_PI*x(1))*sin(M_PI*x(2))*cos(M_PI*x(3));
      E(3) = -cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*sin(M_PI*x(3));
   }
   else if (dim == 3)
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
   //f_exact = E +  DivSkew P( curl E ), where P is the 4d permutation operator
   if(dim==4)
   {
	   f(0) =  (kappa+4.0*M_PI*M_PI)*sin(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
	   f(1) = -(kappa+4.0*M_PI*M_PI)*cos(M_PI*x(0))*sin(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
	   f(2) =  (kappa+4.0*M_PI*M_PI)*cos(M_PI*x(0))*cos(M_PI*x(1))*sin(M_PI*x(2))*cos(M_PI*x(3));
	   f(3) = -(kappa+4.0*M_PI*M_PI)*cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*sin(M_PI*x(3));
   }
   else if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
