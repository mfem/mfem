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
//#define SOLVE_A2
//#define ITER_A2

//#define USE_CSL

//#define USE_HELMHOLTZ

//#define TEST_MULTIPLE_SP

#ifdef USE_HELMHOLTZ
void GetHelmholtzMatrix(ParMesh *pmesh, const int dir, HypreParMatrix *A)
{
  const int order = 1;
  FiniteElementCollection *fec;
  fec = new H1_FECollection(order, dim);
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

  Array<int> ess_tdof_list;

  const bool homogeneousBCeverywhere = false;
  if (homogeneousBCeverywhere)
    {
      if (pmesh->bdr_attributes.Size())
	{
	  Array<int> ess_bdr(pmesh->bdr_attributes.Max());
	  ess_bdr = 1;
	  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	}
    }
  else
    { // Set boundary conditions, depending on dir.
      MFEM_VERIFY(dim == 3, "");
      for (int i=0; i<pmesh->GetNBE(); ++i)
	{
	  Element *elem = pmesh->GetBdrElement(i);
	  MFEM_VERIFY(elem->GetNVertices() >= 3, "");
	  const int *vertices = elem->GetVertices();
	  double *v[3];
	  for (int j=0; j<3; ++j)
	    v[j] = pmesh->GetVertex(vertices[j]);

	  double u[3];
	  double w[3];
	  for (int j=0; j<3; ++j)
	    {
	      u[j] = v[1][j] - v[0][j];  // An edge tangent
	      w[j] = v[2][j] - v[1][j];  // Another edge tangent, not parallel to u.
	    }
	  
	  double n[3];  // normal vector, taken as the cross product u x v
	  n[0] = (u[1]*w[2]) - (u[2]*w[1]);
	  n[1] = (u[2]*w[0]) - (u[0]*w[2]);
	  n[2] = (u[0]*w[1]) - (u[1]*w[0]);

	  double t = sqrt((n[0]*n[0]) + (n[1]*n[1]) + (n[2]*n[2]));

	  int d = -1;
	  for (int j=0; j<3; ++j)
	    {
	      n[j] /= t;  // normalize
	      if (fabs(fabs(n[j]) - 1.0) < 1.0e-8)
		d = j;
	    }

	  MFEM_VERIFY(d >= 0, "");

	  if (d != dir)  // face has essential BC at all DOF's.
	    {
	      elem->SetAttribute(1);
	    }
	  else
	    {
	      elem->SetAttribute(0);
	    }
	}

      Array<int> ess_bdr(2);
      ess_bdr = 0;
      ess_bdr[1] = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  ParBilinearForm *a = new ParBilinearForm(fespace);
  ConstantCoefficient one(1.0);
  ConstantCoefficient neg(SIGMAVAL);

  a->AddDomainIntegrator(new DiffusionIntegrator(one));
  a->AddDomainIntegrator(new MassIntegrator(neg));

  ParLinearForm *b = new ParLinearForm(fespace);
  ConstantCoefficient zero(0.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(zero));
  b->Assemble();

  bool static_cond = false;

  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  ParGridFunction x(fespace);
  x = 0.0;

  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, *A, X, B);
}
#endif


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int order = 2;
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
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      //(int)floor(log(100000./mesh->GetNE())/log(2.)/dim);
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

   //cout << myid << ": NBE " << pmesh->GetNBE() << endl;
   
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

#ifdef USE_CSL
   Vector Bdef, Xdef;
   
   ParBilinearForm *Mform =  new ParBilinearForm(fespace);
   Mform->AddDomainIntegrator(new VectorFEMassIntegrator(*sigmaAbs));
   Mform->Assemble();
   
   // Mform->Finalize();

   HypreParMatrix Mmat, Smat, Mcopy;
   Mform->FormLinearSystem(ess_tdof_list, x, *b, Mmat, Xdef, Bdef);
   Mform->FormLinearSystem(ess_tdof_list, x, *b, Mcopy, Xdef, Bdef);  // There must be a better way than creating two identical matrices.

   ParBilinearForm *Sform =  new ParBilinearForm(fespace);
   Sform->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   Sform->Assemble();

   Sform->FormLinearSystem(ess_tdof_list, x, *b, Smat, Xdef, Bdef);

   ParBilinearForm *agrad = new ParBilinearForm(fespace);
   //agrad->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   agrad->AddDomainIntegrator(new VectorFEMassIntegrator(*muinv));

   if (static_cond) { agrad->EnableStaticCondensation(); }
   agrad->Assemble();
   HypreParMatrix Agrad;
   agrad->FormLinearSystem(ess_tdof_list, x, *b, Agrad, Xdef, Bdef);
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

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

#ifdef SOLVE_A2
   HypreParMatrix Acopy;
   {
     Vector Bdum, Xdum;
     a->FormLinearSystem(ess_tdof_list, x, *b, Acopy, Xdum, Bdum);
   }
#endif
   
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
       const bool fullDirect = true;

#ifdef USE_CSL
       const double beta1 = 1.0;
       const double beta2 = 0.5;

       Mmat *= -beta1;
	   
       // HypreParMatrix *cslRe = Add(1.0, Smat, -beta1, Mmat);
       HypreParMatrix * cslRe = ParAdd(&Smat, &Mmat);
	   
       Mcopy *= beta2;
	   
       //ComplexHypreParMatrix chpm(cslRe, &Mcopy, false, false);
       ComplexHypreParMatrix chpm(&A, &Mcopy, false, false);  // For the case beta1 = 1.

       HypreParMatrix *cSysMat = chpm.GetSystemMatrix();

       Array<int> block_offsets(3); // number of variables + 1
       block_offsets[0] = 0;
       block_offsets[1] = fespace->GetVSize();
       block_offsets[2] = fespace->GetVSize();
       block_offsets.PartialSum();

       Array<int> block_trueOffsets(3); // number of variables + 1
       block_trueOffsets[0] = 0;
       block_trueOffsets[1] = fespace->TrueVSize();
       block_trueOffsets[2] = fespace->TrueVSize();
       block_trueOffsets.PartialSum();

       //cout << myid << ": V size " << fespace->GetVSize() << ", true " << fespace->TrueVSize() << ", global true " << size << ", B size "
       //<< B.Size() << ", X size " << X.Size() << endl;
	   
       // Note that B is of true size.
       BlockVector trueY(block_trueOffsets), trueX(block_trueOffsets), trueRhs(block_trueOffsets);
	   
       trueRhs.GetBlock(0) = B;
       trueRhs.GetBlock(1) = 0.0;
	   
       Operator * Arow = new STRUMPACKRowLocMatrix(*cSysMat);

       STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
       strumpack->SetPrintFactorStatistics(true);
       strumpack->SetPrintSolveStatistics(false);
       strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
       strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
       // strumpack->SetMC64Job(strumpack::MC64Job::NONE);
       // strumpack->SetSymmetricPattern(true);
       strumpack->SetOperator(*Arow);
       strumpack->SetFromCommandLine();
#endif
       
       if (fullDirect)
	 {
#ifdef USE_CSL
	   //Solver * precond = strumpack;

	   // strumpack->Mult(B, X);

	   BlockOperator blockDiagA(block_trueOffsets);
	   for (int i=0; i<2; ++i)
	     blockDiagA.SetDiagonalBlock(i, &A);

	   ParFiniteElementSpace *prec_fespace =
	     (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
	   HypreSolver *amsgrad = new HypreAMS(Agrad, prec_fespace);

#ifdef HYPRE_DYLAN
	   {
	     Vector Xtmp(X);
	     amsgrad->Mult(B, Xtmp);  // Just a hack to get ams to run its setup function. There should be a better way.
	   }

	   HypreAMSG *amsg = new HypreAMSG((HypreAMS*) amsgrad, argc, argv);
	   BlockOperator blockDiagP(block_trueOffsets);
	   for (int i=0; i<2; ++i)
	     blockDiagP.SetDiagonalBlock(i, amsg);

	   TripleProductOperator strumpackProj(&blockDiagP, strumpack, &blockDiagP, false, false, false);
	   ProductOperator prod(&blockDiagA, &strumpackProj, false, false);
#else
	   ProductOperator prod(&blockDiagA, strumpack, false, false);
#endif
	   
	   GMRESSolver *gmres = new GMRESSolver(fespace->GetComm());
	   //BiCGSTABSolver *gmres = new BiCGSTABSolver(fespace->GetComm());
	   
	   gmres->SetOperator(prod);
	   gmres->SetRelTol(1e-12);
	   gmres->SetMaxIter(1000);
	   gmres->SetPrintLevel(1);

	   gmres->Mult(trueRhs, trueY);
	   strumpack->Mult(trueY, trueX);

	   X = trueX.GetBlock(0);
	   double xim2 = trueX.GetBlock(1).Norml2();
	   xim2 *= xim2;
	   double sumxim2 = 0.0;
	   
	   MPI_Allreduce(&xim2, &sumxim2, 1, MPI_DOUBLE, MPI_SUM, fespace->GetComm());

	   if (myid == 0)
	     cout << myid << ": norm of Xim " << trueX.GetBlock(1).Norml2() << ", global " << sqrt(sumxim2) << endl;
	   
	   delete gmres;
	   delete strumpack;
	   delete Arow;
#else
	   cout << "Solving with STRUMPACK" << endl;

#ifdef TEST_MULTIPLE_SP
	   const int Ns = 2;
	   std::vector<Operator*> Arows(Ns);
	   std::vector<STRUMPACKSolver*> strumpacks(Ns);

	   //Operator * Arow = new STRUMPACKRowLocMatrix(A);

	   for (int m=0; m<Ns; ++m)
	     {
	       Arows[m] = new STRUMPACKRowLocMatrix(A);

	       //STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
	       strumpacks[m] = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
	       strumpacks[m]->SetPrintFactorStatistics(true);
	       strumpacks[m]->SetPrintSolveStatistics(false);
	       strumpacks[m]->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
	       strumpacks[m]->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
	       // strumpack->SetMC64Job(strumpack::MC64Job::NONE);
	       // strumpack->SetSymmetricPattern(true);
	       strumpacks[m]->SetOperator(*Arows[m]);
	       strumpacks[m]->SetFromCommandLine();
	       //Solver * precond = strumpack;

	       strumpacks[m]->Mult(B, X);
       
	       //delete strumpack;
	       //delete Arow;
	     }
#else	   
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

	   cout << "Solving with strumpack one time" << endl;
	   
	   strumpack->Mult(B, X);
       
	   delete strumpack;
	   delete Arow;
#endif
#endif
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

	   HypreParMatrix H[3];
#ifdef USE_HELMHOLTZ
	   for (int i=0; i<3; ++i)
	     GetHelmholtzMatrix(pmesh, i, &(H[i]));
#endif

#ifdef USE_CSL
	   HypreIAMS *iams = new HypreIAMS(A, H, strumpack, &trueX, &trueY, (HypreAMS*) ams, argc, argv);
#else
	   HypreIAMS *iams = new HypreIAMS(A, H, strumpack, NULL, NULL, (HypreAMS*) ams, argc, argv);
#endif

	   GMRESSolver *gmres = new GMRESSolver(fespace->GetComm());
	   //FGMRESSolver *gmres = new FGMRESSolver(fespace->GetComm());
	   //BiCGSTABSolver *gmres = new BiCGSTABSolver(fespace->GetComm());
	   //MINRESSolver *gmres = new MINRESSolver(fespace->GetComm());
	   
	   gmres->SetOperator(A);
	   gmres->SetRelTol(1e-16);
	   gmres->SetMaxIter(1000);
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
