//                       MFEM Example 11 - Parallel Version
//
// Compile with: make ex11p
//
// Sample runs:  mpirun -np 4 ex11p -m ../data/square-disc.mesh
//               mpirun -np 4 ex11p -m ../data/star.mesh
//               mpirun -np 4 ex11p -m ../data/escher.mesh
//               mpirun -np 4 ex11p -m ../data/fichera.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex11p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex11p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/disc-nurbs.mesh -o -1 -n 20
//               mpirun -np 4 ex11p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex11p -m ../data/star-surf.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex11p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex11p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex11p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex11p -m ../data/mobius-strip.mesh -n 8
//               mpirun -np 4 ex11p -m ../data/klein-bottle.mesh -n 10
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               eigenvalue problem -Delta u = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of the LOBPCG eigenvalue solver
//               together with the BoomerAMG preconditioner in HYPRE, as well as
//               optionally the SuperLU or STRUMPACK parallel direct solvers.
//               Reusing a single GLVis visualization window for multiple
//               eigenfunctions is also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

double InitialDisplacement(const Vector &x);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   int ref_levels = 1;
   int order = 1;
   bool slu_solver  = false;
   double dt = 0.01;
   double tmax = 5.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
   args.AddOption(&dt, "-dt", "--delta-time",
                  "Size of the time step.");
   args.AddOption(&tmax, "-tm", "--tmax",
                  "Length of time to run the simulation.");
   args.Parse();
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel bilinear forms k(.,.) and m(.,.) on the finite
   //    element space. The first corresponds to the Laplacian operator -Delta,
   //    while the second is a simple mass matrix
   ConstantCoefficient one(1.0);
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm *k = new ParBilinearForm(fespace);
   k->AddDomainIntegrator(new DiffusionIntegrator(one));
   k->Assemble();
   k->Finalize();
   HypreParMatrix *K = k->ParallelAssemble();
   delete k;

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   //m->EliminateEssentialBC(ess_bdr);
   m->Finalize();
   HypreParMatrix *M = m->ParallelAssemble();
   delete m;

   // 7. Setup the grid functions and vectors with their initial conditions
   ParGridFunction u_tp1(fespace), u_t(fespace), u_tm1(fespace), b(fespace);
   HypreParVector U_TP1(*M), U_T(*M), U_TM1(*M), B(*M);
   FunctionCoefficient u_0(InitialDisplacement);
   u_tp1.ProjectCoefficient(u_0);
   u_tp1.GetTrueDofs(U_TP1);
   U_T = U_TP1;
   U_TM1 = U_TP1;

   // 8. Define and configure the solver.  We will use HYPRE
   //    with either BoomerAMG as the preconditioner, or use 
   //    SuperLU as the "preconditioner"  which will handle
   //    the full solve in one go.
   Solver *solver = NULL;
   Operator *Mrow = NULL;
   HypreBoomerAMG *amg = NULL;
   if (!slu_solver)
   {
      HypreBoomerAMG * amg = new HypreBoomerAMG(*M);
      amg->SetPrintLevel(0);
      HyprePCG *pcg = new HyprePCG(*M);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(200);
      pcg->SetPrintLevel(1);
      pcg->SetPreconditioner(*amg);
      solver = pcg;
   }
   else
   {
#ifdef MFEM_USE_SUPERLU
      Mrow = new SuperLURowLocMatrix(*M);
      SuperLUSolver * superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(true);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      superlu->SetOperator(*Mrow);
      solver = superlu;
#endif
   }



   // 9. 
   int num_steps = int(tmax / dt);
   for (int step = 1; step <= num_steps; ++step)
   {
      //Move the values from the current time step and laggard down the line
      U_TM1 = U_T;
      U_T = U_TP1;

      //Compute the RHS B = dt^2 K U_T + 2 M U_T - M U_TM1
      K->Mult(U_T, B, dt*dt, 0.0);
      M->Mult(U_T, B, 2.0, 1.0);
      M->Mult(U_TM1, B, -1.0, 1.0);

      //Apply the dirichlet boundary conditions to B
      //m->EliminateEssentialBC(ess_bdr, U_TP1, B);
      //Now solve M U_TP1 = B
      solver->Mult(B, U_TP1);

      //Dump the resulting displacements for this time step to output
   }



   // 12. Free the used memory.
   delete M;
   delete K;

   delete solver;
   if (slu_solver)
   {
#if defined(MFEM_USE_SUPERLU)
      delete Mrow;
#endif
   }
   else
   {
      delete amg;
   }



   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}


//This will be a "cone" initial displacement with 1.0 at the center
//tending to 0.0 at distance 1.0 away from the center.
double InitialDisplacement(const Vector &x)
{
   double r = 0.0;
   for (int d = 0; d < x.Size(); ++d)
   {
      r += x[d]*x[d]; 
   }
   r = sqrt(r);
   return (r < 1.0) ? 1.0 - r : 0.0;
}