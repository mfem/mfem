//                       MFEM Example 25 - Parallel Version
//
// Compile with: make ex25p
//
// Sample runs:  mpirun -np 4 ex25p -m ../data/star.mesh
//               mpirun -np 4 ex25p -m ../data/fichera.mesh
//               mpirun -np 4 ex25p -m ../data/beam-hex.mesh
//
// Device sample runs:
//               mpirun -np 4 ex25p -d cuda
//               mpirun -np 4 ex25p -d occa-cuda
//               mpirun -np 4 ex25p -d raja-omp
//               mpirun -np 4 ex25p -d ceed-cpu
//               mpirun -np 4 ex25p -d ceed-cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions
//               as in example 1. It highlights on the creation of a hierarchy
//               of discretization spaces with partial assembly and the
//               construction of an efficient p-multigrid preconditioner for the
//               iterative solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 4;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

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
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
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
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
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

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   if (pmesh->bdr_attributes.Size())
   {
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
   a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     * With p-multigrid preconditioner
   Solver *prec = NULL;
   ParSpaceHierarchy *spaceHierarchy = NULL;
   Array<H1_FECollection*> *collections = NULL;
   ParMultigridBilinearForm* mgOperator = NULL;
   spaceHierarchy = new ParSpaceHierarchy;
   collections = new Array<H1_FECollection*>;
   Array<int> orders;
   orders.Append(order);
   int coarseOrder = order / 2;
   while (coarseOrder > 0)
   {
      orders.Append(coarseOrder);
      coarseOrder /= 2;
   }
   orders.Sort();
   for (int level = 0; level < orders.Size() - 1; ++level)
   {
      collections->Append(new H1_FECollection(orders[level], dim));
      ParFiniteElementSpace* fesp =
         new ParFiniteElementSpace(pmesh, collections->Last());
      spaceHierarchy->AddLevel(mesh, fesp, false, true);
   }
   spaceHierarchy->AddLevel(mesh, fespace, false, false);
   mgOperator = new ParMultigridBilinearForm(*spaceHierarchy, *a, ess_bdr);
   prec = new MultigridSolver(mgOperator, MultigridSolver::CycleType::VCYCLE,
                              1, 1);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.Mult(B, X);
   delete prec;

   delete mgOperator;
   for (int level = 0; level < collections->Size(); ++level)
   {
      delete (*collections)[level];
   }
   delete collections;
   delete spaceHierarchy;

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
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
