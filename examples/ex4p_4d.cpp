//                       MFEM Example 4 - Parallel Version
//
// Compile with: make ex4p
//
// Sample runs:  mpirun -np 4 ex4p -m ../data/square-disc.mesh
//               mpirun -np 4 ex4p -m ../data/star.mesh
//               mpirun -np 4 ex4p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex4p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex4p -m ../data/escher.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/fichera.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex4p -m ../data/fichera-q3.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/square-disc-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/beam-hex-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/periodic-square.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/periodic-cube.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/star-surf.mesh -o 3 -hb
//
// Description:  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Bilinear form
//               hybridization and static condensation are also illustrated.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "./spe10_coeff.cpp"

using namespace std;
using namespace mfem;


int* LoadIterations(int NRows, int NCol)
{
  ifstream in("iter_div.txt");

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
	out.open("iter_div.txt",fstream::out);

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


// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;



class div4dPrec : public Solver
{

private:
	HypreParMatrix *A;
	ParFiniteElementSpace *fespace;

	Coefficient *alpha_, *beta_;

	//kernel operators
	HypreParMatrix *P_d_HSkewDiv_Hdiv;

	HypreParMatrix *P_H1_HDivSkew;
	HypreParMatrix *H1_KernelMat;
	HypreBoomerAMG *amgH1_Kernel;

	//"image" operators
	HypreParMatrix *P_H1_Hdiv;
	HypreParMatrix *H1_ImageMat;
	HypreBoomerAMG *amgH1_Image;

	HypreParMatrix *HDivSkewMat;
	HypreSmoother * smootherdiv;
	HypreSmoother * smootherDivSkew;

	CGSolver *pcgKernel;
	CGSolver *pcgImage;

	Vector *f;
	Vector *fKernel, *uKernel;
	Vector *fImage, *uImage;
	Vector *fDivSkew, *uDivSkew;

	FiniteElementCollection* fecHDivSkewKernel;
	ParFiniteElementSpace *HDivSkewKernelFESpace;

	bool exactSolves;

public:
	~div4dPrec()
	{
		delete pcgImage;
		delete pcgKernel;

		delete uDivSkew, fDivSkew, uImage, fImage, uKernel, fKernel, f;

		delete smootherDivSkew;
		delete HDivSkewMat;

		delete P_d_HSkewDiv_Hdiv;
		delete P_H1_Hdiv;
		delete P_H1_HDivSkew;

		delete amgH1_Image, H1_ImageMat;
		delete amgH1_Kernel, H1_KernelMat;

		delete smootherdiv;

		delete HDivSkewKernelFESpace;
		delete fecHDivSkewKernel;
	}
	div4dPrec(HypreParMatrix *AUser, ParFiniteElementSpace *fespaceUser, Coefficient *alpha, Coefficient *beta, const Array<int> &essBnd, int orderKernel=1, bool exactSolvesUser=false)
	{
		A = AUser;
		fespace = fespaceUser;
		alpha_ = alpha;
		beta_ = beta;


		ParMesh *pmesh = fespace->GetParMesh();
		int dim = pmesh->Dimension();

		exactSolves = exactSolvesUser;




		int orderIm=1;  //H1  --> H(div)
		int orderKer=orderKernel; //DivSkew V --> H(div)



		smootherdiv = new HypreSmoother(*A, 16, 3);

		Array<int> Hdiv_essDof(fespace->GetVSize()); Hdiv_essDof = 0;
		fespace->GetEssentialVDofs(essBnd, Hdiv_essDof);




		//setup the H1 FESpace for the kernel
		FiniteElementCollection* fecH1Kernel;
		   if(orderKer==1) fecH1Kernel = new LinearFECollection;
		   else fecH1Kernel = new QuadraticFECollection;
		ParFiniteElementSpace *H1KernelFESpace = new ParFiniteElementSpace(pmesh, fecH1Kernel, 6, Ordering::byVDIM);
		Array<int> H1Kernel_essDof(H1KernelFESpace->GetVSize()); H1Kernel_essDof = 0;
		H1KernelFESpace->GetEssentialVDofs(essBnd, H1Kernel_essDof);


		//setup the H(DivSkew) FESpace for the kernel
		   if(orderKer==1) fecHDivSkewKernel = new DivSkew1_4DFECollection;
//		   else fecHDivSkewKernel = new DivSkewFull1_4DFECollection;
		HDivSkewKernelFESpace = new ParFiniteElementSpace(pmesh, fecHDivSkewKernel);
		Array<int> HDivSkewKernel_essDof(HDivSkewKernelFESpace->GetVSize()); HDivSkewKernel_essDof = 0;
		HDivSkewKernelFESpace->GetEssentialVDofs(essBnd, HDivSkewKernel_essDof);


		//setup the FESpace for the H1 injection
		FiniteElementCollection* fecH1Vec;
		   if(orderIm==1) fecH1Vec = new LinearFECollection;
		   else fecH1Vec = new QuadraticFECollection;
		ParFiniteElementSpace *H1_ImageFESpace = new ParFiniteElementSpace(pmesh, fecH1Vec, dim, Ordering::byVDIM);
		Array<int> H1Image_essDof(H1_ImageFESpace->GetVSize()); H1Image_essDof = 0;
		H1_ImageFESpace->GetEssentialVDofs(essBnd, H1Image_essDof);



		//setup the H1 preconditioner for the kernel
		ParBilinearForm* H1Varf = new ParBilinearForm(H1KernelFESpace);
//		H1Varf->AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_, 6));
//		H1Varf->AddDomainIntegrator(new VectorMassIntegrator(6, beta_));

		H1Varf->AddDomainIntegrator(new VectorDiffusionIntegrator(*beta_, 6));
		H1Varf->Assemble();
		H1Varf->Finalize();
		SparseMatrix &matH1(H1Varf->SpMat());
		for(int dof=0; dof<H1Kernel_essDof.Size(); dof++) if(H1Kernel_essDof[dof]<0) matH1.EliminateRowCol(dof);
		H1_KernelMat = H1Varf->ParallelAssemble();
		delete H1Varf;
		amgH1_Kernel = new HypreBoomerAMG(*H1_KernelMat);
		amgH1_Kernel->SetSystemsOptions(6);

		//setup the H1 preconditioner for the image
		ParBilinearForm* H1VecVarf = new ParBilinearForm(H1_ImageFESpace);
		H1VecVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_));
		H1VecVarf->AddDomainIntegrator(new VectorMassIntegrator(-1, beta_));
		H1VecVarf->Assemble();
		H1VecVarf->Finalize();
		SparseMatrix &matH1Vec(H1VecVarf->SpMat());
		for(int dof=0; dof<H1Image_essDof.Size(); dof++) if(H1Image_essDof[dof]<0) matH1Vec.EliminateRowCol(dof);
		H1_ImageMat = H1VecVarf->ParallelAssemble();
		delete H1VecVarf;
		amgH1_Image = new HypreBoomerAMG(*H1_ImageMat);
		amgH1_Image->SetSystemsOptions(dim);


		//setup the injection of H1 into H(DivSkew)
		ParDiscreteLinearOperator *disInterpolIm = new ParDiscreteLinearOperator(H1KernelFESpace, HDivSkewKernelFESpace);
		disInterpolIm->AddDomainInterpolator(new IdentityInterpolator);
		disInterpolIm->Assemble();
		disInterpolIm->Finalize();
		SparseMatrix* smatIDIm = &(disInterpolIm->SpMat());
		smatIDIm->EliminateCols(H1Kernel_essDof);
		for(int dof=0; dof<HDivSkewKernel_essDof.Size(); dof++) if(HDivSkewKernel_essDof[dof]<0) smatIDIm->EliminateRow(dof);
		P_H1_HDivSkew = disInterpolIm->ParallelAssemble();
		delete disInterpolIm;

		//setup the injection of H1 into H(div)
		ParDiscreteLinearOperator *disInterpol = new ParDiscreteLinearOperator(H1_ImageFESpace, fespace);
		disInterpol->AddDomainInterpolator(new IdentityInterpolator);
		disInterpol->Assemble();
		disInterpol->Finalize();
		SparseMatrix* smatID = &(disInterpol->SpMat());
		smatID->EliminateCols(H1Image_essDof);
		for(int dof=0; dof<Hdiv_essDof.Size(); dof++) if(Hdiv_essDof[dof]<0) smatID->EliminateRow(dof);
		P_H1_Hdiv = disInterpol->ParallelAssemble();
		delete disInterpol;




		//setup the injection of the DivSkew(H(DivSkew)) into H(div)
		ParDiscreteLinearOperator *disDivSkew = new ParDiscreteLinearOperator(HDivSkewKernelFESpace, fespace);
		disDivSkew->AddDomainInterpolator(new DivSkewInterpolator);
		disDivSkew->Assemble();
		disDivSkew->Finalize();
		SparseMatrix* smatDivSkew= &(disDivSkew->SpMat());
		smatDivSkew->EliminateCols(HDivSkewKernel_essDof);
		for(int dof=0; dof<Hdiv_essDof.Size(); dof++) if(Hdiv_essDof[dof]<0) smatDivSkew->EliminateRow(dof);
		P_d_HSkewDiv_Hdiv = disDivSkew->ParallelAssemble();
		delete disDivSkew;


		//setup the smoother for H(DivSkew)
		ParBilinearForm *a_HDivSkew = new ParBilinearForm(HDivSkewKernelFESpace);
//		a_HDivSkew->AddDomainIntegrator(new DivSkewDivSkewIntegrator(*alpha_));
//		a_HDivSkew->AddDomainIntegrator(new VectorFE_DivSkewMassIntegrator(*beta_));

		a_HDivSkew->AddDomainIntegrator(new DivSkewDivSkewIntegrator(*beta_));

		a_HDivSkew->Assemble();
		a_HDivSkew->Finalize();
		SparseMatrix &matHDivSkew(a_HDivSkew->SpMat());
		for(int dof=0; dof<HDivSkewKernel_essDof.Size(); dof++) if(HDivSkewKernel_essDof[dof]<0) matHDivSkew.EliminateRowCol(dof);
		HDivSkewMat = a_HDivSkew->ParallelAssemble();
		delete a_HDivSkew;
		smootherDivSkew = new HypreSmoother(*HDivSkewMat, 16, 3);



		f = new Vector(fespace->GetTrueVSize());

		fKernel = new Vector(H1KernelFESpace->GetTrueVSize());
		uKernel = new Vector(H1KernelFESpace->GetTrueVSize());

		fImage = new Vector(H1_ImageFESpace->GetTrueVSize());
		uImage = new Vector(H1_ImageFESpace->GetTrueVSize());

		fDivSkew = new Vector(HDivSkewKernelFESpace->GetTrueVSize());
		uDivSkew = new Vector(HDivSkewKernelFESpace->GetTrueVSize());

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

		delete H1_ImageFESpace;
		delete H1KernelFESpace;
		delete fecH1Kernel;
		delete fecH1Vec;
	}

	void setExactSolve(bool exSol)
	{
		exactSolves = exSol;
	}

	virtual void Mult(const Vector &x, Vector &y) const
	{
		smootherdiv->Mult(x,y);

		P_H1_Hdiv->MultTranspose(x,*fImage);
		*uImage = 0.0;
		if(exactSolves) pcgImage->Mult(*fImage, *uImage);
		else amgH1_Image->Mult(*fImage, *uImage);
		P_H1_Hdiv->Mult(1.0, *uImage, 1.0, y);


		*uDivSkew = 0.0;
		P_d_HSkewDiv_Hdiv->MultTranspose(x,*fDivSkew);

			smootherDivSkew->Mult(*fDivSkew, *uDivSkew);

			P_H1_HDivSkew->MultTranspose(*fDivSkew,*fKernel);
				*uKernel = 0.0;
				if(exactSolves) pcgKernel->Mult(*fKernel, *uKernel);
				else amgH1_Kernel->Mult(*fKernel, *uKernel);
			P_H1_HDivSkew->Mult(1.0, *uKernel, 1.0, *uDivSkew);

		P_d_HSkewDiv_Hdiv->Mult(1.0, *uDivSkew, 1.0, y);
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
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool set_bc = true;
   bool static_cond = false;
   bool hybridization = false;
   bool visualization = 1;
   int sequ_ref_levels = 0;
   int par_ref_levels = 0;
   double tol = 1e-6;
   double coeffWeight = 1.0;
   bool spe10Coeff = false;
   bool exactH1Solver = false;
   bool standardCG = true;

   int NExpo = 8;
   int weightStart = -NExpo;
   int weightEnd = NExpo;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sequ_ref_levels, "-sr", "--seqrefinement",
                     "Number of sequential refinement steps.");
   args.AddOption(&par_ref_levels, "-pr", "--parrefinement",
                     "Number of parallel refinement steps.");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume, as well as periodic meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      for (int l = 0; l < sequ_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them (this is needed in the ADS solver below).
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec;
   if(dim==4) fec = new RT0_4DFECollection;
   else fec = new RT_FECollection(order-1, dim);
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
   VectorFunctionCoefficient f(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient F(sdim, F_exact);

   for(int expo=weightStart; expo<=weightEnd; expo++)
   {
	   double weight = pow(10.0,expo);
	   kappa = weight;

	   x.ProjectCoefficient(F);

	   // 10. Set up the parallel bilinear form corresponding to the H(div)
	   //     diffusion operator grad alpha div + beta I, by adding the div-div and
	   //     the mass domain integrators.

//	   std::string permFile = "spe_perm.dat";
//	   InversePermeabilityFunction::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);

	   Coefficient *alpha = new ConstantCoefficient(1.0);
	   Coefficient *beta;
//	   if(spe10Coeff) beta = new FunctionCoefficient(InversePermeabilityFunction::Norm2Permeability);
//	   else
		   beta = new ConstantCoefficient(weight);

	   ParBilinearForm *a = new ParBilinearForm(fespace);
	   a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
	   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));

	   // 11. Assemble the parallel bilinear form and the corresponding linear
	   //     system, applying any necessary transformations such as: parallel
	   //     assembly, eliminating boundary conditions, applying conforming
	   //     constraints for non-conforming AMR, static condensation,
	   //     hybridization, etc.
	   FiniteElementCollection *hfec = NULL;
	   ParFiniteElementSpace *hfes = NULL;
	   if (static_cond)
	   {
		  a->EnableStaticCondensation();
	   }
	   else if (hybridization)
	   {
		  hfec = new DG_Interface_FECollection(order-1, dim);
		  hfes = new ParFiniteElementSpace(pmesh, hfec);
		  a->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
								 ess_tdof_list);
	   }
	   a->Assemble();

	   HypreParMatrix A;
	   Vector B, X;
	   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

	   HYPRE_Int glob_size = A.GetGlobalNumRows();
	   if (myid == 0)
	   {
		  cout << "Size of linear system: " << glob_size << endl;
	   }

	   // 12. Define and apply a parallel PCG solver for A X = B with the 2D AMS or
	   //     the 3D ADS preconditioners from hypre. If using hybridization, the
	   //     system is preconditioned with hypre's BoomerAMG.
	   Solver *prec = NULL;
	   if (hybridization) { prec = new HypreBoomerAMG(A); }
	   else
	   {
		  ParFiniteElementSpace *prec_fespace =
			 (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
		  if (dim == 2)   { prec = new HypreAMS(A, prec_fespace); }
		  else if(dim==3) { prec = new HypreADS(A, prec_fespace); }
		  else if(dim==4) prec = new div4dPrec(&A, fespace, alpha, beta, ess_bdr, order, exactH1Solver);
		  else prec = NULL;
	   }

		  int iter = -1;
		   if(standardCG)
		   {
			   IterativeSolver *pcg = new CGSolver(MPI_COMM_WORLD);
			   pcg->SetOperator(A);
			   pcg->SetRelTol(tol);
			   pcg->SetMaxIter(500);
			   pcg->SetPrintLevel(1);
			   pcg->SetPreconditioner(*prec);
			   pcg->Mult(B, X);

			   iter = pcg->GetNumIterations();

			   delete pcg;
		   }
		   else
		   {
			   HyprePCG *pcg = new HyprePCG(A);
			   pcg->SetTol(tol);
			   pcg->SetMaxIter(5000);
			   pcg->SetResidualConvergenceOptions(1,tol);
			   pcg->SetPrintLevel(2);
//			   pcg->SetPreconditioner(*prec);
			   pcg->Mult(B, X);

			   pcg->GetNumIterations(iter);

			   delete pcg;
		   }

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
		  double err = x.ComputeL2Error(F);
		  if (myid == 0)
		  {
			 cout << "\n|| F_h - F ||_{L^2} = " << err << '\n' << endl;
		  }
	   }

	   // 15. Save the refined mesh and the solution in parallel. This output can
	   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
//	   {
//		  ostringstream mesh_name, sol_name;
//		  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
//		  sol_name << "sol." << setfill('0') << setw(6) << myid;
//
//		  ofstream mesh_ofs(mesh_name.str().c_str());
//		  mesh_ofs.precision(8);
//		  pmesh->Print(mesh_ofs);
//
//		  ofstream sol_ofs(sol_name.str().c_str());
//		  sol_ofs.precision(8);
//		  x.Save(sol_ofs);
//	   }

	   // 16. Send the solution by socket to a GLVis server.
//	   if (visualization)
//	   {
//		  char vishost[] = "localhost";
//		  int  visport   = 19916;
//		  socketstream sol_sock(vishost, visport);
//		  sol_sock << "parallel " << num_procs << " " << myid << "\n";
//		  sol_sock.precision(8);
//		  sol_sock << "solution\n" << *pmesh << x << flush;
//	   }

	   if(prec!=NULL) delete prec;
	   delete hfes;
	   delete hfec;
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


// The exact solution (for non-surface meshes)
void F_exact(const Vector &p, Vector &F)
{
   int dim = p.Size();

   if(dim==4)
   {
	   double s0 = sin(M_PI*p(0)), s1 = sin(M_PI*p(1)), s2 = sin(M_PI*p(2)), s3 = sin(M_PI*p(3));
	   double c0 = cos(M_PI*p(0)), c1 = cos(M_PI*p(1)), c2 = cos(M_PI*p(2)), c3 = cos(M_PI*p(3));

	   F(0) = c0 * s1 * s2 * s3;
	   F(1) = s0 * c1 * s2 * s3;
	   F(2) = s0 * s1 * c2 * s3;
	   F(3) = s0 * s1 * s2 * c3;
   }
   else
   {
	   double x = p(0);
	   double y = p(1);
	   // double z = (dim == 3) ? p(2) : 0.0;

	   F(0) = cos(kappa*x)*sin(kappa*y);
	   F(1) = cos(kappa*y)*sin(kappa*x);
	   if (dim == 3)
	   {
		  F(2) = 0.0;
	   }
   }
}

// The right hand side
void f_exact(const Vector &p, Vector &f)
{
   int dim = p.Size();
   if(dim==4)
   {
	   double s0 = sin(M_PI*p(0)), s1 = sin(M_PI*p(1)), s2 = sin(M_PI*p(2)), s3 = sin(M_PI*p(3));
	   double c0 = cos(M_PI*p(0)), c1 = cos(M_PI*p(1)), c2 = cos(M_PI*p(2)), c3 = cos(M_PI*p(3));

	   f(0) = c0 * s1 * s2 * s3;
	   f(1) = s0 * c1 * s2 * s3;
	   f(2) = s0 * s1 * c2 * s3;
	   f(3) = s0 * s1 * s2 * c3;

	   f *= (kappa+4.0 * M_PI*M_PI);
   }
   else
   {
	   double x = p(0);
	   double y = p(1);
	   // double z = (dim == 3) ? p(2) : 0.0;

	   double temp = 1 + 2*kappa*kappa;

	   f(0) = temp*cos(kappa*x)*sin(kappa*y);
	   f(1) = temp*cos(kappa*y)*sin(kappa*x);
	   if (dim == 3)
	   {
		  f(2) = 0;
	   }
   }
}
