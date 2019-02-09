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

#include "ddmesh.hpp"
#include "ddoper.hpp"

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
double radiusFunction(const Vector &);
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

#define SIGMAVAL -250.0


void VisitTestPlotParMesh(const std::string filename, ParMesh *pmesh)
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

  ParGridFunction x(&fespace);
  FunctionCoefficient radius(radiusFunction);
  x.ProjectCoefficient(radius);
  
  dc->RegisterField("radius", &x);
  dc->SetCycle(0);
  dc->SetTime(0.0);
  dc->Save();

  delete dc;
}

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
   bool visit = true;   
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

   // 4.5. Partition the mesh in serial, to define subdomains.
   // Note that the mesh attribute is overwritten here for convenience, which is bad if the attribute is needed.
   int nxyzSubdomains[3] = {2, 1, 1};
   const int numSubdomains = nxyzSubdomains[0] * nxyzSubdomains[1] * nxyzSubdomains[2];
   {
     int *subdomain = mesh->CartesianPartitioning(nxyzSubdomains);
     for (int i=0; i<mesh->GetNE(); ++i)  // Loop over all elements, to set the attribute as the subdomain index.
       {
	 mesh->SetAttribute(i, subdomain[i]+1);
       }
     delete subdomain;
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

   // 5.1. Determine subdomain interfaces, and for each interface create a set of local vertex indices in pmesh.
   SubdomainInterfaceGenerator sdInterfaceGen(numSubdomains, pmesh);
   vector<SubdomainInterface> interfaces;  // Local interfaces
   sdInterfaceGen.CreateInterfaces(interfaces);
   std::vector<int> interfaceGlobalToLocalMap, interfaceGI;
   const int numInterfaces = sdInterfaceGen.GlobalToLocalInterfaceMap(interfaces, interfaceGlobalToLocalMap, interfaceGI);
   
   cout << myid << ": created " << numSubdomains << " subdomains with " << numInterfaces <<  " interfaces" << endl;
   
   // 5.2. Create parallel subdomain meshes.
   SubdomainParMeshGenerator sdMeshGen(numSubdomains, pmesh);
   ParMesh **pmeshSD = sdMeshGen.CreateParallelSubdomainMeshes();

   if (pmeshSD == NULL)
     return 2;

   // 5.3. Create parallel interface meshes.
   ParMesh **pmeshInterfaces = (numInterfaces > 0 ) ? new ParMesh*[numInterfaces] : NULL;

   for (int i=0; i<numInterfaces; ++i)
     {
       const int iloc = interfaceGlobalToLocalMap[i];  // Local interface index
       if (iloc >= 0)
	 {
	   MFEM_VERIFY(interfaceGI[i] == interfaces[iloc].GetGlobalIndex(), "");
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(interfaces[iloc]);
	 }
       else
	 {
	   // This is not elegant. 
	   const int sd0 = interfaceGI[i] / numSubdomains;  // globalIndex = (numSubdomains * sd0) + sd1;
	   const int sd1 = interfaceGI[i] - (numSubdomains * sd0);  // globalIndex = (numSubdomains * sd0) + sd1;
	   SubdomainInterface emptyInterface(sd0, sd1);
	   emptyInterface.SetGlobalIndex(numSubdomains);
	   pmeshInterfaces[i] = sdMeshGen.CreateParallelInterfaceMesh(emptyInterface);
	 }
     }
   
   const bool testSubdomains = true;
   if (testSubdomains)
     {
       for (int i=0; i<numSubdomains; ++i)
	 {
	   ostringstream filename;
	   filename << "sd" << setfill('0') << setw(3) << i;
	   VisitTestPlotParMesh(filename.str(), pmeshSD[i]);
	 }
       
       for (int i=0; i<numInterfaces; ++i)
	 {
	   ostringstream filename;
	   filename << "sdif" << setfill('0') << setw(3) << i;
	   VisitTestPlotParMesh(filename.str(), pmeshInterfaces[i]);
	 }
       
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

   // 6.1. Create interface operator.
   DDMInterfaceOperator ddi(numSubdomains, numInterfaces, pmeshSD, pmeshInterfaces, order, pmesh->Dimension(),
			    &interfaces, &interfaceGlobalToLocalMap);  // PengLee2012 uses order 2 
     
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

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

#ifdef MFEM_USE_STRUMPACK
   if (use_strumpack)
     {
       const bool fullDirect = true;

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
#endif
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

double radiusFunction(const Vector &x)
{
  double f = 0.0;
  for (int i=0; i<dim; ++i)
    f += x[i]*x[i];
  
  return sqrt(f);
}
