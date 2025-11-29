//                       MFEM Example 1 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../../data/amr-quad.mesh --petscopts rc_ex1p
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda --petscopts rc_ex1p_device
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
//               The example also shows how PETSc Krylov solvers can be used by
//               wrapping a HypreParMatrix (or not) and a Solver, together with
//               customization using an options file (see rc_ex1p) We also
//               provide an example on how to visualize the iterative solution
//               inside a PETSc solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

class UserMonitor : public PetscSolverMonitor
{
private:
   ParBilinearForm *a;
   ParLinearForm *b;

public:
   UserMonitor(ParBilinearForm *a_, ParLinearForm *b_)
      : PetscSolverMonitor(true,false), a(a_), b(b_) {}

   void MonitorSolution(PetscInt it, PetscReal norm, const Vector &X)
   {
      // we plot the first 5 iterates
      if (!it || it > 5) { return; }
      ParFiniteElementSpace *fespace = a->ParFESpace();
      ParMesh *mesh = fespace->GetParMesh();
      ParGridFunction x(fespace);
      a->RecoverFEMSolution(X, *b, x);

      char vishost[] = "localhost";
      int  visport   = 19916;
      int  num_procs, myid;

      MPI_Comm_size(mesh->GetComm(),&num_procs);
      MPI_Comm_rank(mesh->GetComm(),&myid);
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x
               << "window_title 'Iteration no " << it << "'" << flush;
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool visualization = false;
   const char *device_config = "cpu";
   bool use_petsc = true;
   const char *petscrc_file = "";
   bool petscmonitor = false;
   bool forcewrap = false;
   bool useh2 = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&petscmonitor, "-petscmonitor", "--petscmonitor",
                  "-no-petscmonitor", "--no-petscmonitor",
                  "Enable or disable GLVis visualization of residual.");
   args.AddOption(&forcewrap, "-forcewrap", "--forcewrap",
                  "-noforce-wrap", "--noforce-wrap",
                  "Force matrix-free.");
   args.AddOption(&useh2, "-useh2", "--useh2", "-no-h2",
                  "--no-h2",
                  "Use or not the H2 matrix solver.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3b. We initialize PETSc
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
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

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     If using MFEM with HYPRE
   //       * With full assembly, use the BoomerAMG preconditioner from hypre.
   //       * With partial assembly, use Jacobi smoothing, for now.
   //     If using MFEM with PETSc
   //       * With full assembly, use command line options or H2 matrix solver
   //       * With partial assembly, wrap Jacobi smoothing, for now.
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(*fespace))
      {
         prec = new OperatorJacobiSmoother(*a, ess_tdof_list);
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }

   if (!use_petsc)
   {
      CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
      if (prec) { pcg->SetPreconditioner(*prec); }
      pcg->SetOperator(*A);
      pcg->SetRelTol(1e-12);
      pcg->SetMaxIter(200);
      pcg->SetPrintLevel(1);
      pcg->Mult(B, X);
      delete pcg;
   }
   else
   {
      PetscPCGSolver *pcg;
      // If petscrc_file has been given, we convert the HypreParMatrix to a
      // PetscParMatrix; the user can then experiment with PETSc command line
      // options unless forcewrap is true.
      bool wrap = forcewrap ? true : (pa ? true : !strlen(petscrc_file));
      if (wrap)
      {
         pcg = new PetscPCGSolver(MPI_COMM_WORLD);
         pcg->SetOperator(*A);
         if (useh2)
         {
            delete prec;
            prec = new PetscH2Solver(*A.Ptr(),fespace);
         }
         else if (!pa) // We need to pass the preconditioner constructed from the HypreParMatrix
         {
            delete prec;
            HypreParMatrix *hA = A.As<HypreParMatrix>();
            prec = new HypreBoomerAMG(*hA);
         }
         if (prec) { pcg->SetPreconditioner(*prec); }
      }
      else // Not wrapping, pass the HypreParMatrix so that users can experiment with command line
      {
         HypreParMatrix *hA = A.As<HypreParMatrix>();
         pcg = new PetscPCGSolver(*hA, false);
         if (useh2)
         {
            delete prec;
            prec = new PetscH2Solver(*hA,fespace);
         }
      }
      pcg->iterative_mode = true; // iterative_mode is true by default with CGSolver
      pcg->SetRelTol(1e-12);
      pcg->SetAbsTol(1e-12);
      pcg->SetMaxIter(200);
      pcg->SetPrintLevel(1);

      UserMonitor mymon(a,b);
      if (visualization && petscmonitor)
      {
         pcg->SetMonitor(&mymon);
         pcg->iterative_mode = true;
         X.Randomize();
      }
      pcg->Mult(B, X);
      delete pcg;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

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
   if (delete_fec)
   {
      delete fec;
   }
   delete a;
   delete b;
   delete fespace;
   delete pmesh;
   delete prec;

   // We finalize PETSc
   MFEMFinalizePetsc();

   return 0;
}
