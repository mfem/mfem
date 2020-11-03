//                       MFEM Example 1 - Parallel Version
//                             SuperLU Modification
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
//               mpirun -np 4 ex1p -m ../../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../../data/mobius-strip.mesh
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

#ifndef MFEM_USE_SUPERLU
#error This example requires that MFEM is built with MFEM_USE_SUPERLU=YES
#endif

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
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = true;
   int slu_colperm = 4;
   int slu_rowperm = 1;
   int slu_iterref = 2;

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
   args.AddOption(&slu_colperm, "-cp", "--colperm",
                  "SuperLU Column Permutation Method:  0-NATURAL, 1-MMD-ATA "
                  "2-MMD_AT_PLUS_A, 3-COLAMD, 4-METIS_AT_PLUS_A, 5-PARMETIS "
                  "6-ZOLTAN");
   args.AddOption(&slu_rowperm, "-rp", "--rowperm",
                  "SuperLU Row Permutation Method:  0-NOROWPERM, 1-LargeDiag");
   args.AddOption(&slu_iterref, "-rp", "--rowperm",
                  "SuperLU Iterative Refinement:  0-NOREFINE, 1-Single, "
                  "2-Double, 3-Extra");

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
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
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
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13. Solve the linear system A X = B utilizing SuperLU.
   SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
   Operator *SLU_A = new SuperLURowLocMatrix(*A.As<HypreParMatrix>());
   superlu->SetPrintStatistics(true);
   superlu->SetSymmetricPattern(false);

   if (slu_colperm == 0)
   {
      superlu->SetColumnPermutation(superlu::NATURAL);
   }
   else if (slu_colperm == 1)
   {
      superlu->SetColumnPermutation(superlu::MMD_ATA);
   }
   else if (slu_colperm == 2)
   {
      superlu->SetColumnPermutation(superlu::MMD_AT_PLUS_A);
   }
   else if (slu_colperm == 3)
   {
      superlu->SetColumnPermutation(superlu::COLAMD);
   }
   else if (slu_colperm == 4)
   {
      superlu->SetColumnPermutation(superlu::METIS_AT_PLUS_A);
   }
   else if (slu_colperm == 5)
   {
      superlu->SetColumnPermutation(superlu::PARMETIS);
   }
   else if (slu_colperm == 6)
   {
      superlu->SetColumnPermutation(superlu::ZOLTAN);
   }

   if (slu_rowperm == 0)
   {
      superlu->SetRowPermutation(superlu::NOROWPERM);
   }
   else if (slu_rowperm == 1)
   {
#ifdef MFEM_USE_SUPERLU5
      superlu->SetRowPermutation(superlu::LargeDiag);
#else
      superlu->SetRowPermutation(superlu::LargeDiag_MC64);
#endif
   }

   if (slu_iterref == 0)
   {
      superlu->SetIterativeRefine(superlu::NOREFINE);
   }
   else if (slu_iterref == 1)
   {
      superlu->SetIterativeRefine(superlu::SLU_SINGLE);
   }
   else if (slu_iterref == 2)
   {
      superlu->SetIterativeRefine(superlu::SLU_DOUBLE);
   }
   else if (slu_iterref == 3)
   {
      superlu->SetIterativeRefine(superlu::SLU_EXTRA);
   }

   superlu->SetOperator(*SLU_A);
   superlu->SetPrintStatistics(true);
   superlu->Mult(B, X);

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

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
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }
   MPI_Finalize();

   return 0;
}
