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

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

#define SIGMAVAL -250.0
//#define FORM_DEFINITE
#define SOLVE_A2
#define ITER_A2


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
#ifdef MFEM_USE_STRUMPACK
   bool use_strumpack = false;
#endif

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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(100000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

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
   }
   
   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   long globalNE = pmesh->GetGlobalNE();
   if (myid == 0)
   {
     cout << "Number of mesh elements: " << globalNE << endl;
     cout << "Number of finite element unknowns: " << size << endl;
     cout << "Root local number of finite element unknowns: " << fespace->TrueVSize() << endl;
   }
   
   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
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
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(SIGMAVAL);
   Coefficient *sigmaAbs = new ConstantCoefficient(fabs(SIGMAVAL));
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

#ifdef FORM_DEFINITE
   ParBilinearForm *adef = new ParBilinearForm(fespace);
   adef->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   adef->AddDomainIntegrator(new VectorFEMassIntegrator(*sigmaAbs));

   if (static_cond) { adef->EnableStaticCondensation(); }
   adef->Assemble();

   HypreParMatrix Adef;
   Vector Bdef, Xdef;
   adef->FormLinearSystem(ess_tdof_list, x, *b, Adef, Xdef, Bdef);
#endif

#ifdef ITER_A2
   Vector Bdef, Xdef;
   
   ParBilinearForm *Mform =  new ParBilinearForm(fespace);
   Mform->AddDomainIntegrator(new VectorFEMassIntegrator(*sigmaAbs));
   Mform->Assemble();
   
   Mform->Finalize();

   HypreParMatrix Mmat, Mcopy;
   Mform->FormLinearSystem(ess_tdof_list, x, *b, Mmat, Xdef, Bdef);
   Mform->FormLinearSystem(ess_tdof_list, x, *b, Mcopy, Xdef, Bdef);  // There must be a better way to implement M^2.

   /*
   HypreParMatrix *Mmat = Mform->ParallelAssemble();
   HypreParMatrix *Mcopy = Mform->ParallelAssemble();  // There must be a better way to implement M^2.
   */
   
   ParBilinearForm *Sform =  new ParBilinearForm(fespace);
   Sform->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   Sform->Assemble();

   HypreParMatrix Smat, Scopy;
   Sform->FormLinearSystem(ess_tdof_list, x, *b, Smat, Xdef, Bdef);
   Sform->FormLinearSystem(ess_tdof_list, x, *b, Scopy, Xdef, Bdef);
#endif
   
   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A, Acopy;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   {
     Vector Bdum, Xdum;
     a->FormLinearSystem(ess_tdof_list, x, *b, Acopy, Xdum, Bdum);
   }
   
   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   //A.Print("maxwell1000_2");
   
#ifdef MFEM_USE_STRUMPACK
   if (use_strumpack)
     {
       const bool fullDirect = false;

       if (fullDirect)
	 {
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
       
	   delete strumpack;
	   delete Arow;
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

	   HypreIAMS *iams = new HypreIAMS(A, (HypreAMS*) ams, argc, argv);

	   GMRESSolver *gmres = new GMRESSolver();
	   gmres->SetOperator(A);
	   gmres->SetRelTol(1e-12);
	   gmres->SetMaxIter(100);
	   gmres->SetPrintLevel(1);

#ifdef SOLVE_A2
	   {
	     StopWatch chronoA2;
	     chronoA2.Clear();
	     chronoA2.Start();

	     HypreParMatrix * A2 = ParMult(&A, &Acopy);

	     chronoA2.Stop();
	     cout << "A2 setup time " << chronoA2.RealTime() << endl;
	     
	     Vector AB(B);
	     A.Mult(B, AB);
	     gmres->SetOperator(*A2);

	     HypreSolver *ams2 = new HypreAMS(A, prec_fespace);
	     {
	       Vector Xtmp(X);
	       ams2->Mult(B, Xtmp);  // Just a hack to get ams to run its setup function. There should be a better way.
	     }

#ifdef ITER_A2
	     // Iteratively solve 0.5 (A^2 + S^2 + M^2) u^{k+1} = 0.5 (SM + MS) u^k + Ab 

	     StopWatch chronoIterA2;
	     chronoIterA2.Clear();
	     chronoIterA2.Start();

	     HypreParMatrix * M2 = ParMult(&Mmat, &Mcopy);
	     HypreParMatrix * S2 = ParMult(&Smat, &Scopy);

	     HypreParMatrix * MS = ParMult(&Mmat, &Scopy);
	     HypreParMatrix * SM = ParMult(&Smat, &Mcopy);

             HypreParMatrix * Bmat = ParAdd(SM, MS);
	     (*Bmat) *= 0.5;
	     
	     // TODO: there must be a better way to form a sum of three matrices. Of course, we could define an operator that does 3 mat-vecs. 
             //HypreParMatrix * S2M2 = ParAdd(S2, M2);
	     //HypreParMatrix * iterMat = ParAdd(A2, S2M2);
	     HypreParMatrix * iterMat = ParAdd(A2, Bmat);
	     
	     chronoIterA2.Stop();
	     cout << "Iter A2 setup time " << chronoIterA2.RealTime() << endl;

	     /*
	     HypreSolver *ams3 = new HypreAMS(*iterMat, prec_fespace);
	     {
	       Vector Xtmp(X);
	       ams3->Mult(B, Xtmp);  // Just a hack to get ams to run its setup function. There should be a better way.
	     }
	     */
	     	     
	     /*
	     // GMRES
	     gmres->SetOperator(*iterMat);
	     gmres->SetPreconditioner(*ams2);
	     */
	     

	     //HypreBoomerAMG *amg = new HypreBoomerAMG(*iterMat);
	     HypreBoomerAMG *amg = new HypreBoomerAMG(*A2);

	     // PCG
	     HyprePCG *pcg = new HyprePCG(*iterMat);
	     //HyprePCG *pcg = new HyprePCG(*A2);
	     pcg->SetTol(1e-12);
	     pcg->SetMaxIter(10);
	     pcg->SetPrintLevel(2);
	     pcg->SetPreconditioner(*amg);

	     /*
	     // Strumpack linear solver
	     Operator * Arow = new STRUMPACKRowLocMatrix(*iterMat);

	     STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
	     strumpack->SetPrintFactorStatistics(true);
	     strumpack->SetPrintSolveStatistics(false);
	     strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
	     strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
	     strumpack->SetOperator(*Arow);
	     strumpack->SetFromCommandLine();
	     */
	     
	     Vector iterRHS(AB);
	     Vector iterU(AB);
	     Vector iterU0(AB);

	     iterU = 0.0;
	     iterU0 = 0.0;
	     
	     bool iterate = true;
	     int numIter = 0;
	     while (iterate)
	       {
		 iterRHS = iterU;
		 iterRHS.Add(-1.0, iterU0);

		 cout << "Iteration " << numIter + 1 << ": diff norm " << iterRHS.Norml2() << endl;
		 
		 iterU0 = iterU;

		 Bmat->Mult(iterU0, iterRHS);
		 //iterRHS.Add(2.0, AB);
		 iterRHS.Add(1.0, AB);
				 
		 //gmres->Mult(iterRHS, iterU);
		 pcg->Mult(iterRHS, iterU);
		 //strumpack->Mult(iterRHS, iterU);

		 numIter++;

		 if (numIter > 100)
		   iterate = false;
	       }

	     //delete strumpack;
	     //delete Arow;

	     delete pcg;

	     X = iterU;
#else
	     //HypreIAMS *iams2 = new HypreIAMS(*A2, (HypreAMS*) ams2, argc, argv);
	     //gmres->SetPreconditioner(*iams2);
	     cout << myid << ": Solving" << endl;
	     gmres->SetPreconditioner(*ams2);
	     gmres->Mult(AB, X);
	     cout << myid << ": Solved" << endl;
	     return 3;
#endif
	   }
#else
	   gmres->SetPreconditioner(*iams);
	   gmres->Mult(B, X);
#endif
#else
	   HypreGMRES *gmres = new HypreGMRES(A);
	   gmres->SetTol(1e-12);
	   gmres->SetMaxIter(100);
	   gmres->SetPrintLevel(10);

#ifdef FORM_DEFINITE
	   HypreSolver *amsdef = new HypreAMS(Adef, prec_fespace);
	   gmres->SetPreconditioner(*amsdef);
#else
	   gmres->SetPreconditioner(*ams);
#endif
	   gmres->Mult(B, X);
#endif

	   delete gmres;
	   //delete iams;
	   //delete ams;
	 }
     }
   else
#endif
     {
       // 12. Define and apply a parallel PCG solver for AX=B with the AMS
       //     preconditioner from hypre.
       ParFiniteElementSpace *prec_fespace =
	 (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
       HypreSolver *ams = new HypreAMS(A, prec_fespace);
       HyprePCG *pcg = new HyprePCG(A);
       pcg->SetTol(1e-12);
       pcg->SetMaxIter(500);
       pcg->SetPrintLevel(2);
       pcg->SetPreconditioner(*ams);
       pcg->Mult(B, X);

       delete pcg;
       delete ams;
     }
   
   chrono.Stop();
   cout << myid << ": Solver time " << chrono.RealTime() << endl;

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

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
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
	 cout << "\n|| E_h ||_{L^2} = " << normX << '\n' << endl;
	 cout << "\n|| E ||_{L^2} = " << normE << '\n' << endl;
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

   // 17. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

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
