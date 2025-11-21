//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../../data/star.mesh
//               mpirun -np 4 ex1p -m ../../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../../data/escher.mesh
//               mpirun -np 4 ex1p -m ../../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../../data/octahedron.mesh -o 1
//               mpirun -np 4 ex1p -m ../../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -fa -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
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

#ifdef MFEM_USE_CUDSS
#include <cuda_runtime.h>
#include "cudss.h"
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool cudss_solver = false;
   bool visualization = true;
   bool algebraic_ceed = false;

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
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CUDSS
   args.AddOption(&cudss_solver, "-cudss", "--cudss-solver", "-no-cudss",
                  "--no-cudss-solver", "Use the cuDSS Solver.");
#endif
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   int device_id = 0;
#ifdef MFEM_USE_CUDA
   // Assign devices round-robin over MPI ranks if GPU support is enabled.
   int ngpu = Device::GetDeviceCount();
   device_id = myid % ngpu;
#endif
   Device device(device_config, device_id);

#ifdef MFEM_USE_CUDA
#if defined(HYPRE_WITH_GPU_AWARE_MPI) || defined(HYPRE_USING_GPU_AWARE_MPI)
   device.SetGPUAwareMPI(true);
#endif
#endif

   if (myid == 0)
   {
      device.Print();
   }
   // Initialize Hypre
   Hypre::Init();

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
          (int)floor(log(10000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
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
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
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
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the external boundary attributes from the mesh as
   //    essential (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      // Apply boundary conditions on all external boundaries:
      pmesh.MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // pmesh.MarkNamedBoundaries(set_name, ess_bdr)

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a.EnableStaticCondensation();
   }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13a. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
         }
         else
         {
            prec = new OperatorJacobiSmoother(a, ess_tdof_list);
         }
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec)
   {
      cg.SetPreconditioner(*prec);
   }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

#ifdef MFEM_USE_CUDSS
   // 13b. Solve A Y = B using cuDSS
   if (!pa && cudss_solver)
   {
      ParGridFunction y(&fespace);
      y = 0.0;
      Vector Y;
      a.FormLinearSystem(ess_tdof_list, y, b, A, Y, B);

      CuDSSSolver cu(MPI_COMM_WORLD);
      cu.SetMatrixSymType(CuDSSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
      cu.SetMatrixViewType(CuDSSSolver::MatViewType::UPPER);
      cu.SetOperator(*A);
      cu.Mult(B, Y);

      a.RecoverFEMSolution(Y, b, y);
      y.SaveAsOne("sol_ex1p_cudss.gf", 8);

      // Check the difference between X and Y
      Y -= X;
      real_t diff = Y * Y;
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
      diff = std::sqrt(diff);

      if (myid == 0)
      {
         cout << "||X - Y|| = " << diff << endl;
      }
   }
#endif

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "sol_ex1p.mesh";
      sol_name << "sol_ex1p.gf";

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.PrintAsOne(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.SaveAsOne(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << pmesh << x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
