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
  ifstream in("iter_DivSkew.txt");

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
	out.open("iter_DivSkew.txt",fstream::out);

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
void E_exact_vec(const Vector &x, Vector &E);
void E_exact(const Vector &, DenseMatrix &);
void f_exact(const Vector &, DenseMatrix &);


class DivSkew4dPrec : public Solver
{

private:
   HypreParMatrix *A;
   ParFiniteElementSpace *fespace;
   Coefficient *alpha_, *beta_;

   //kernel operators
   HypreParMatrix *P_d_HCurl_HDivSkew;


   HypreParMatrix *P_H1_HCurl;
   HypreParMatrix *H1_KernelMat;
   HypreBoomerAMG *amgH1_Kernel;

   //"image" operators
   HypreParMatrix *P_H1_HDivSkew;
   HypreParMatrix *H1_ImageMat;
   HypreBoomerAMG *amgH1_Image;


   HypreParMatrix *HCurlMat;
   HypreSmoother * smootherDivSkew;
   HypreSmoother * smootherCurl;

   CGSolver *pcgKernel;
   CGSolver *pcgImage;

   Vector *f;
   Vector *fKernel, *uKernel;
   Vector *fImage, *uImage;
   Vector *fCurl, *uCurl;

   bool exactSolves;

   FiniteElementCollection* fecHCurlKernel;
   ParFiniteElementSpace *HCurlKernelFESpace;


public:
   ~DivSkew4dPrec()
   {
	   delete pcgImage, pcgKernel;

	   delete f, fKernel, uKernel, fImage, uImage, fCurl, uCurl;

	   delete smootherCurl, HCurlMat;

	   delete P_d_HCurl_HDivSkew, P_H1_HDivSkew, P_H1_HCurl;

	   delete amgH1_Image, H1_ImageMat;
	   delete amgH1_Kernel, H1_KernelMat;

	   delete smootherDivSkew;

	   delete HCurlKernelFESpace, fecHCurlKernel;
   }
   DivSkew4dPrec(HypreParMatrix *AUser, ParFiniteElementSpace *fespaceUser, Coefficient *alpha, Coefficient *beta,
                 const Array<int> &essBnd, int orderKernel=1, bool exactSolvesUser=false)
   {
      A = AUser;
      fespace = fespaceUser;
	  alpha_ = alpha;
	  beta_ = beta;

      ParMesh *pmesh = fespace->GetParMesh();
      int dim = pmesh->Dimension();

      exactSolves = exactSolvesUser;




      int orderIm=1;  //H1  --> H(divSkew)
      int orderKer=orderKernel; //curl V --> H(divSkew)



      smootherDivSkew = new HypreSmoother(*A, 16, 3);

      Array<int> HDivSkew_essDof(fespace->GetVSize()); HDivSkew_essDof = 0;
      fespace->GetEssentialVDofs(essBnd, HDivSkew_essDof);




      //setup the H1 FESpace for the kernel
      FiniteElementCollection* fecH1Kernel;
      if (orderKer==1) { fecH1Kernel = new LinearFECollection; }
      else { fecH1Kernel = new QuadraticFECollection; }

      ParFiniteElementSpace *H1KernelFESpace = new ParFiniteElementSpace(pmesh,
                                                                         fecH1Kernel, dim, Ordering::byVDIM);
      Array<int> H1Kernel_essDof(H1KernelFESpace->GetVSize()); H1Kernel_essDof = 0;
      H1KernelFESpace->GetEssentialVDofs(essBnd, H1Kernel_essDof);


      //setup the H(curl) FESpace for the kernel
      if (orderKer==1) { fecHCurlKernel = new ND1_4DFECollection; }
      else { fecHCurlKernel = new ND2_4DFECollection; }

      HCurlKernelFESpace = new ParFiniteElementSpace(pmesh,
                                                                            fecHCurlKernel);
      Array<int> HCurlKernel_essDof(HCurlKernelFESpace->GetVSize());
      HCurlKernel_essDof = 0;
      HCurlKernelFESpace->GetEssentialVDofs(essBnd, HCurlKernel_essDof);


      //setup the FESpace for the H1 injection
      FiniteElementCollection* fecH1Vec;
      if (orderIm==1) { fecH1Vec = new LinearFECollection; }
      else { fecH1Vec = new QuadraticFECollection; }
      ParFiniteElementSpace *H1_ImageFESpace = new ParFiniteElementSpace(pmesh,
                                                                         fecH1Vec, 6, Ordering::byVDIM);
      Array<int> H1Image_essDof(H1_ImageFESpace->GetVSize()); H1Image_essDof = 0;
      H1_ImageFESpace->GetEssentialVDofs(essBnd, H1Image_essDof);



      //setup the H1 preconditioner for the kernel
      ParBilinearForm* H1Varf = new ParBilinearForm(H1KernelFESpace);
      H1Varf->AddDomainIntegrator(new VectorDiffusionIntegrator(*beta_));
//      H1Varf->AddDomainIntegrator(new VectorMassIntegrator);
      H1Varf->Assemble();
      H1Varf->Finalize();
      SparseMatrix &matH1(H1Varf->SpMat());
      for (int dof=0; dof<H1Kernel_essDof.Size(); dof++) if (H1Kernel_essDof[dof]<0) { matH1.EliminateRowCol(dof); }
      H1_KernelMat = H1Varf->ParallelAssemble();
      delete H1Varf;
      amgH1_Kernel = new HypreBoomerAMG(*H1_KernelMat);
      amgH1_Kernel->SetSystemsOptions(dim);

      //setup the H1 preconditioner for the image
      ParBilinearForm* H1VecVarf = new ParBilinearForm(H1_ImageFESpace);
      H1VecVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_, 6));
      H1VecVarf->AddDomainIntegrator(new VectorMassIntegrator(6, beta));
      H1VecVarf->Assemble();
      H1VecVarf->Finalize();
      SparseMatrix &matH1Vec(H1VecVarf->SpMat());
      for (int dof=0; dof<H1Image_essDof.Size(); dof++) if (H1Image_essDof[dof]<0) { matH1Vec.EliminateRowCol(dof); }
      H1_ImageMat = H1VecVarf->ParallelAssemble();
      delete H1VecVarf;
      amgH1_Image = new HypreBoomerAMG(*H1_ImageMat);
      amgH1_Image->SetSystemsOptions(6);


      //setup the injection of H1 into H(curl)
      ParDiscreteLinearOperator *disInterpol = new ParDiscreteLinearOperator(
         H1KernelFESpace, HCurlKernelFESpace);
      disInterpol->AddDomainInterpolator(new IdentityInterpolator);
      disInterpol->Assemble();
      disInterpol->Finalize();
      SparseMatrix* smatID = &(disInterpol->SpMat());
      smatID->EliminateCols(H1Kernel_essDof);
      for (int dof=0; dof<HCurlKernel_essDof.Size();
           dof++) if (HCurlKernel_essDof[dof]<0) { smatID->EliminateRow(dof); }
      P_H1_HCurl = disInterpol->ParallelAssemble();
      delete disInterpol;

      //setup the injection of H1 into H(DivSkew)
      ParDiscreteLinearOperator *disInterpolIm = new ParDiscreteLinearOperator(
         H1_ImageFESpace, fespace);
      disInterpolIm->AddDomainInterpolator(new IdentityInterpolator);
      disInterpolIm->Assemble();
      disInterpolIm->Finalize();
      SparseMatrix* smatIDIm = &(disInterpolIm->SpMat());
      smatIDIm->EliminateCols(H1Image_essDof);
      for (int dof=0; dof<HDivSkew_essDof.Size(); dof++) if (HDivSkew_essDof[dof]<0) { smatIDIm->EliminateRow(dof); }
      P_H1_HDivSkew = disInterpolIm->ParallelAssemble();
      delete disInterpolIm;


      //setup the injection of the curl(H(curl)) into H(DivSkew)
      ParDiscreteLinearOperator *disCurl = new ParDiscreteLinearOperator(
         HCurlKernelFESpace, fespace);
      disCurl->AddDomainInterpolator(new CurlInterpolator);
      disCurl->Assemble();
      disCurl->Finalize();
      SparseMatrix* smatCurl = &(disCurl->SpMat());
      smatCurl->EliminateCols(HCurlKernel_essDof);
      for (int dof=0; dof<HDivSkew_essDof.Size(); dof++) if (HDivSkew_essDof[dof]<0) { smatCurl->EliminateRow(dof); }
      P_d_HCurl_HDivSkew = disCurl->ParallelAssemble();
      delete disCurl;



      //setup the smoother for H(curl)
//      Coefficient *massC = new ConstantCoefficient(1.0);
//      Coefficient *CurlCurlC = new ConstantCoefficient(1.0);
      ParBilinearForm *a_HCurl = new ParBilinearForm(HCurlKernelFESpace);
      a_HCurl->AddDomainIntegrator(new CurlCurlIntegrator(*beta_));
//      a_HCurl->AddDomainIntegrator(new CurlCurlIntegrator(*CurlCurlC));
//      a_HCurl->AddDomainIntegrator(new VectorFEMassIntegrator(*massC));
      a_HCurl->Assemble();
      a_HCurl->Finalize();
      SparseMatrix &matHCurl(a_HCurl->SpMat());
      for (int dof=0; dof<HCurlKernel_essDof.Size();
           dof++) if (HCurlKernel_essDof[dof]<0) { matHCurl.EliminateRowCol(dof); }
      HCurlMat = a_HCurl->ParallelAssemble();
      delete a_HCurl;
      smootherCurl = new HypreSmoother(*HCurlMat, 16, 3);



      f = new Vector(fespace->GetTrueVSize());

      fKernel = new Vector(H1KernelFESpace->GetTrueVSize());
      uKernel = new Vector(H1KernelFESpace->GetTrueVSize());

      fImage = new Vector(H1_ImageFESpace->GetTrueVSize());
      uImage = new Vector(H1_ImageFESpace->GetTrueVSize());

      fCurl = new Vector(HCurlKernelFESpace->GetTrueVSize());
      uCurl = new Vector(HCurlKernelFESpace->GetTrueVSize());


      amgH1_Kernel->Mult(*fKernel, *uKernel);
      amgH1_Image->Mult(*fImage, *uImage);

      pcgKernel = new CGSolver(MPI_COMM_WORLD);
      pcgKernel->SetOperator(*H1_KernelMat);
      pcgKernel->SetPreconditioner(*amgH1_Kernel);
      pcgKernel->SetRelTol(1e-16);
      pcgKernel->SetMaxIter(100000000);
      pcgKernel->SetPrintLevel(-2);

      pcgImage = new CGSolver(MPI_COMM_WORLD);
      pcgImage->SetOperator(*H1_ImageMat);
      pcgImage->SetPreconditioner(*amgH1_Image);
      pcgImage->SetRelTol(1e-16);
      pcgImage->SetMaxIter(100000000);
      pcgImage->SetPrintLevel(-2);


      delete H1KernelFESpace, fecH1Kernel;
      delete H1_ImageFESpace, fecH1Vec;
   }

   void setExactSolve(bool exSol)
   {
      exactSolves = exSol;
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      smootherDivSkew->Mult(x,y);

      P_H1_HDivSkew->MultTranspose(x,*fImage);
      *uImage = 0.0;
      if (exactSolves) { pcgImage->Mult(*fImage, *uImage); }
      else { amgH1_Image->Mult(*fImage, *uImage); }
      P_H1_HDivSkew->Mult(1.0, *uImage, 1.0, y);


      *uCurl = 0.0;
      P_d_HCurl_HDivSkew->MultTranspose(x,*fCurl);

      smootherCurl->Mult(*fCurl, *uCurl);

      P_H1_HCurl->MultTranspose(*fCurl,*fKernel);
      *uKernel = 0.0;
      if (exactSolves) { pcgKernel->Mult(*fKernel, *uKernel); }
      else { amgH1_Kernel->Mult(*fKernel, *uKernel); }
      P_H1_HCurl->Mult(1.0, *uKernel, 1.0, *uCurl);

      P_d_HCurl_HDivSkew->Mult(1.0, *uCurl, 1.0, y);
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
   if (verbose) { args.PrintOptions(cout); }

   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }

   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   if (dim !=4 || sdim != 4)
   {
      MPI_Finalize();
      return 0;
   }

   for (int i=0; i<sequ_ref_levels; i++) { mesh->UniformRefinement(); }
   if (verbose) { mesh->PrintCharacteristics(); }

   if (verbose) { cout << "now we partition the mesh..." << endl << endl; }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int i=0; i<par_ref_levels; i++) { pmesh->UniformRefinement(); }

   pmesh->PrintInfo(std::cout);
   if (verbose) { cout << endl; }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec;
   if (order==1) { fec = new DivSkew1_4DFECollection; }
   //    else fec = new F2K1_4DFECollection;

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

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

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;

   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   MatrixFunctionCoefficient f(sdim, f_exact);
   MatrixFunctionCoefficient solMat(sdim, E_exact);
   VectorFunctionCoefficient solVec(6, E_exact_vec);




   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new MatFEDomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);

   for(int expo=weightStart; expo<=weightEnd; expo++)
   {
	   double weight = pow(10.0,expo);

	   x.ProjectCoefficient(solVec);

	   //   cout << x << endl;
	   //   x = 0.0;

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
	   a->AddDomainIntegrator(new DivSkewDivSkewIntegrator(*alpha));
	   a->AddDomainIntegrator(new VectorFE_DivSkewMassIntegrator(*beta));

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


	   //Define the preconditioner

	   if (myid == 0) { cout << "Set up the preconditioner" << endl; }
	   Solver *prec;
	   if (dim==4) { prec = new DivSkew4dPrec(&A, fespace, alpha, beta, ess_bdr, order, exactH1Solver); }

	   IterativeSolver *pcg = new CGSolver(MPI_COMM_WORLD);
	   pcg->SetOperator(A);
	   pcg->SetRelTol(tol);
	   pcg->SetMaxIter(500);
	   pcg->SetPrintLevel(1);
	   pcg->SetPreconditioner(*prec);
	   pcg->Mult(B, X);

	   delete prec;

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
		  double error = 0.0;
		  for (int i = 0; i < fespace->GetNE(); i++)
		  {
			 const FiniteElement* fe = fespace->GetFE(i);
			 int fdof = fe->GetDof();
			 ElementTransformation* transf = fespace->GetElementTransformation(i);
			 DenseMatrix shape(fdof,dim*dim);

			 int intorder = 2*fe->GetOrder() + 1; // <----------
			 const IntegrationRule *ir;
			 ir = &(IntRules.Get(fe->GetGeomType(), intorder));

			 Vector elSol(dim*dim);
			 DenseMatrix elSolMat(dim,dim);
			 DenseMatrix exactSol(dim,dim);
			 Vector exactSolVec(dim*dim);



			 Array<int> vdofs;
			 fespace->GetElementVDofs(i, vdofs);
			 for (int j = 0; j < ir->GetNPoints(); j++)
			 {
				const IntegrationPoint &ip = ir->IntPoint(j);
				transf->SetIntPoint(&ip);

				fe->CalcVShape(*transf, shape);

				elSol = 0.0;
				for (int k = 0; k < fdof; k++)
				{
				   if (vdofs[k] >= 0)
				   {
					  for (int l=0; l<dim*dim; l++) { elSol(l) += shape(k,l)*x(vdofs[k]); }
				   }
				   else
				   {
					  for (int l=0; l<dim*dim; l++) { elSol(l) -= shape(k,l)*x(-1-vdofs[k]); }
				   }
				}
				for (int k=0; k<dim; k++)
				   for (int l=0; l<dim; l++)
				   {
					  elSolMat(k,l) = elSol(dim*k+l);
				   }


				solMat.Eval(exactSol,*transf, ip);
				for (int k=0; k<dim; k++)
				   for (int l=0; l<dim; l++)
				   {
					  exactSolVec(dim*k+l) = exactSol(k,l);
				   }
				elSol.Add(-1.0, exactSolVec);

				error += ip.weight * fabs(transf->Weight()) * (elSol * elSol);
			 }
		  }
		  double globalError = 0.0;
		  MPI_Allreduce(&error, &globalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		  if (myid==0) { std::cout << "L2 error: " << sqrt(globalError) << std::endl; }


	   }

	   delete pcg;
	   delete a;
	   delete alpha;
	   delete beta;
   }

   // 17. Free the used memory.

   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

void E_exact_vec(const Vector &x, Vector &E)
{
   int dim = x.Size();

   if (dim==4)
   {
      E.SetSize(6);

      double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
             s3 = sin(M_PI*x(3));
      double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
             c3 = cos(M_PI*x(3));

      E(0) =  c0*c1*s2*s3;
      E(1) = -c0*s1*c2*s3;
      E(2) =  c0*s1*s2*c3;
      E(3) =  s0*c1*c2*s3;
      E(4) = -s0*c1*s2*c3;
      E(5) =  s0*s1*c2*c3;
   }
}

void E_exact(const Vector &x, DenseMatrix &E)
{
   int dim = x.Size();

   E.SetSize(dim*dim);

   if (dim==4)
   {
      Vector vecE; E_exact_vec(x, vecE);

      E = 0.0;

      E(0,1) = vecE(0);
      E(0,2) = vecE(1);
      E(0,3) = vecE(2);
      E(1,2) = vecE(3);
      E(1,3) = vecE(4);
      E(2,3) = vecE(5);

      E(1,0) =  -E(0,1);
      E(2,0) =  -E(0,2);
      E(3,0) =  -E(0,3);
      E(2,1) =  -E(1,2);
      E(3,1) =  -E(1,3);
      E(3,2) =  -E(2,3);
   }
}



//f_exact = E + 0.5 * P( curl DivSkew E ), where P is the 4d permutation operator
void f_exact(const Vector &x, DenseMatrix &f)
{
   int dim = x.Size();

   f.SetSize(dim,dim);

   if (dim==4)
   {
      f = 0.0;

      double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
             s3 = sin(M_PI*x(3));
      double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
             c3 = cos(M_PI*x(3));

      f(0,1) =  (1.0 + 1.0  * M_PI*M_PI)*c0*c1*s2*s3;
      f(0,2) = -(1.0 + 0.0  * M_PI*M_PI)*c0*s1*c2*s3;
      f(0,3) =  (1.0 + 1.0  * M_PI*M_PI)*c0*s1*s2*c3;
      f(1,2) =  (1.0 - 1.0  * M_PI*M_PI)*s0*c1*c2*s3;
      f(1,3) = -(1.0 + 0.0  * M_PI*M_PI)*s0*c1*s2*c3;
      f(2,3) =  (1.0 + 1.0  * M_PI*M_PI)*s0*s1*c2*c3;

      f(1,0) =  -f(0,1);
      f(2,0) =  -f(0,2);
      f(3,0) =  -f(0,3);
      f(2,1) =  -f(1,2);
      f(3,1) =  -f(1,3);
      f(3,2) =  -f(2,3);
   }
}
