//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "./spe10_coeff.cpp"


int* LoadIterations(int NRows, int NCol)
{
  ifstream in("iter_grad.txt");

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
	out.open("iter_grad.txt",fstream::out);

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


using namespace std;
using namespace mfem;

double kappa = 1.0;

double u_exact(const Vector &x)
{
	int dim = x.Size();

	if(dim==4)
	{
		return cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
	}
	else return 0.0;
}

double f_exact(const Vector &x)
{
	int dim = x.Size();

	if(dim==4)
	{
		return (kappa + 4.0 * M_PI*M_PI) * cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2))*cos(M_PI*x(3));
	}
	else return 0.0;
}


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
   bool static_cond = false;
   bool visualization = 1;
   int sequ_ref_levels = 0;
   int par_ref_levels = 0;
   double tol = 1e-6;
   bool set_bc = true;
   bool standardCG = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sequ_ref_levels, "-sr", "--seqrefinement",
                     "Number of sequential refinement steps.");
   args.AddOption(&par_ref_levels, "-pr", "--parrefinement",
                     "Number of parallel refinement steps.");
   args.AddOption(&order, "-o", "--order",
                     "Polynomial order of the finite element space.");
   args.AddOption(&tol, "-tol", "--tol",
                     "A parameter.");
   args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                  "Impose or not essential boundary conditions.");
   args.AddOption(&standardCG, "-sCG", "--stdCG", "-rCG", "--resCG",
                  "Switch between standard PCG or recompute residuals in every step and use the residuals itself for the stopping criteria.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if(verbose) args.PrintOptions(cout);

   Mesh *mesh;
   ifstream imesh(mesh_file);
   if(!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }

   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

//   if(dim !=4 || sdim != 4)
//   {
//	   MPI_Finalize();
//	   return 0;
//   }

   for(int i=0; i<sequ_ref_levels; i++) mesh->UniformRefinement();
   if(verbose) mesh->PrintCharacteristics();

   if(verbose) cout << "now we partition the mesh..." << endl << endl;

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for(int i=0; i<par_ref_levels; i++) pmesh->UniformRefinement();

   pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
	  if(dim==4)
	  {
		if(order==1) fec = new LinearFECollection;
		else fec = new QuadraticFECollection;
	  }
	  else fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
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
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = set_bc ? 1 : 0;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   FunctionCoefficient uExact(u_exact);
   ParGridFunction x(fespace);

   int NExpo =8;
   for(int expo=-NExpo; expo<=NExpo; expo++)
   {
	   double weight = pow(10.0,expo);
	   kappa = weight;

	   x.ProjectCoefficient(uExact);

	   ParLinearForm *b = new ParLinearForm(fespace);
	   FunctionCoefficient ffunc(f_exact);
	   b->AddDomainIntegrator(new DomainLFIntegrator(ffunc));
	   b->Assemble();

	   x = 0.0;

	   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
	   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
	   //     domain integrator.

//	   std::string permFile = "spe_perm.dat";
//	   InversePermeabilityFunction::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);
//	   FunctionCoefficient *cspe10 = new FunctionCoefficient(InversePermeabilityFunction::Norm2Permeability);
	   Coefficient *beta = new ConstantCoefficient(weight);

	   ParBilinearForm *a = new ParBilinearForm(fespace);
	   a->AddDomainIntegrator(new DiffusionIntegrator);
	   a->AddDomainIntegrator(new MassIntegrator(*beta));

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

	   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
	   //     preconditioner from hypre.
	   HypreSolver *amg = new HypreBoomerAMG(A);

	  int iter = -1;
	   if(standardCG)
	   {
		   IterativeSolver *pcg = new CGSolver(MPI_COMM_WORLD);
		   pcg->SetOperator(A);
		   pcg->SetRelTol(tol);
		   pcg->SetMaxIter(5000);
		   pcg->SetPrintLevel(1);
		   pcg->SetPreconditioner(*amg);
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
		   pcg->SetPreconditioner(*amg);
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

	   {
		  double err = x.ComputeL2Error(uExact);
		  if (myid == 0)
		  {
			 cout << "\n|| u - u_h ||_{L^2} = " << err << '\n' << endl;
		  }
	   }

	   // 14. Save the refined mesh and the solution in parallel. This output can
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

	   // 15. Send the solution by socket to a GLVis server.
	//   if (visualization)
	//   {
	//      char vishost[] = "localhost";
	//      int  visport   = 19916;
	//      socketstream sol_sock(vishost, visport);
	//      sol_sock << "parallel " << num_procs << " " << myid << "\n";
	//      sol_sock.precision(8);
	//      sol_sock << "solution\n" << *pmesh << x << flush;
	//   }

	   delete amg;
	   delete a;
	   delete beta;
	   delete b;
   }

   // 16. Free the used memory.

   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
