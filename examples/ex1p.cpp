//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
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

using namespace std;
using namespace mfem;

double u_exact(const Vector &x)
{
   return sin(sqrt(2)*x(0))*exp(x(1))*exp(x(2));
}

void u_grad_exact(const Vector &x, Vector &u)
{
   u(0) = sqrt(2)*cos(sqrt(2)*x(0))*exp(x(1))*exp(x(2));
   u(1) = sin(sqrt(2)*x(0))*exp(x(1))*exp(x(2));
   u(2) = sin(sqrt(2)*x(0))*exp(x(1))*exp(x(2));
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-oneHex.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = false;
   bool use_serendip = false;
   int total_refinements = -1;
   int cg_num_its = -85716;
   double solve_time = -0.85716;

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
   args.AddOption(&use_serendip, "-ser", "--use-serendipity",
                  "-no-ser", "--not-serendipity",
                  "Use serendipity element collection.");
   args.AddOption(&total_refinements, "-r", "--refine",
                  "Number of uniform refinements to do.");
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

   //   Start the timer.
   if (myid == 0)
   {
      tic_toc.Clear();
      tic_toc.Start();
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

   int ref_levels =
      (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
   if (total_refinements > -1)
   {
      ref_levels = total_refinements;
   }

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
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
      if (use_serendip)
      {
         fec = new H1Ser_FECollection(order,dim);
      }
      else
      {
         fec = new H1_FECollection(order, dim);
      }
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

   int total_dofs = size;
   if (myid == 0)
   {
      cout << "total_dofs = " << total_dofs << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   // if (pmesh->bdr_attributes.Size())
   // {
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);

   // For Delta u = 0:
   ConstantCoefficient zero(0.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(zero));

   // For Delta u = 1:
   // ConstantCoefficient one(1.0);
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));

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

   // Define exact solution, for boundary
   FunctionCoefficient *uExCoeff = new FunctionCoefficient(u_exact);
   x.ProjectBdrCoefficient(*uExCoeff, ess_bdr);

   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator);
   // a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;

   // cout <<  endl << "******* randomizing RHS for condition testing ****" << endl;
   // B.Randomize(1);

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   double assemble_time = -0.85716;

   if (myid == 0)
   {
      tic_toc.Stop();
      assemble_time = tic_toc.RealTime();
      cout << " Assembly took " << assemble_time << "s." << endl;
      tic_toc.Clear();
      tic_toc.Start();
   }

   // cout << "Size of linear system: " << A->Height() << endl;

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use no preconditioner, for now.
   HypreBoomerAMG *amg =  new HypreBoomerAMG;

   // HYPRE_BoomerAMGSetRelaxType(*amg, 18); // use l1-scaled Jacobi relaxation method
   // HYPRE_BoomerAMGSetRelaxType(*amg, 16); // Chebyshev
   HYPRE_BoomerAMGSetRelaxType(*amg,
                               8); // $\ell_1$-scaled hybrid symmetric Gauss-Seidel

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(*amg);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   if (myid == 0)
   {
      cg_num_its = cg.GetNumIterations();
   }
   delete amg;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".

   ostringstream mesh_name, sol_name;
   mesh_name << "mesh." << setfill('0') << setw(6) << myid;
   sol_name << "sol." << setfill('0') << setw(6) << myid;

   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->Print(mesh_ofs);

   ofstream sol_ofs(sol_name.str().c_str());
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   if (myid == 0)
   {
      tic_toc.Stop();
      solve_time = tic_toc.RealTime();
      cout << " ex1p: Done timing with solve_time = " << solve_time << endl;
      tic_toc.Clear();
      tic_toc.Start();
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

   // Compute and print the L^2 and H^1 norms of the error.
   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient *(uExCoeff_grad) = new VectorFunctionCoefficient(dim,
                                                                              u_grad_exact);

   double l2_err = x.ComputeL2Error(*uExCoeff);
   double h1_err = x.ComputeH1Error(uExCoeff, uExCoeff_grad, &one, 1.0, 1);

   if (myid == 0)
   {
      tic_toc.Stop();
      cout << endl << endl;
      cout << "Computing the error took " << tic_toc.RealTime() << "s." << endl;
      tic_toc.Clear();

      cout << "mesh = " << mesh_file << endl;
      cout << "exact solution u(x,y,z) = sin(sqrt(2)x) e^y e^z" << endl;
      cout << "Solver = BoomerAMG with l_1-scaled hybrid symmetric Gauss-Seidel relaxation"
           << endl;
      cout << "Used static condensation (N/A for serendipity): " << static_cond <<
           endl;

      cout << endl << "***********" << endl;
      cout << "Number of uniform refinements = " << ref_levels << endl;
      cout << "Used serendipity = " << use_serendip << endl;
      cout << "Order of element = " << order << endl;
      cout << "L2 err = " << l2_err << endl;
      cout << "H1 err = " << h1_err << endl;
      cout << "Assembly time (s) = " << assemble_time << endl;
      cout << "Solve time (s) = " << solve_time << endl;
      cout << "# of global dofs = " << total_dofs << endl;
      cout << "# of solver iterations = " << cg_num_its << endl;
      cout << endl << "Writing the above information to data_out.txt.  Done." << endl;

      // Write to file
      std::ofstream fileForWriting;
      fileForWriting.open("data_out.txt", std::ios_base::app);
      fileForWriting << ref_levels  << '\t';
      fileForWriting << use_serendip  << '\t';
      fileForWriting << order << '\t';
      fileForWriting << l2_err << '\t';
      fileForWriting << h1_err << '\t';
      fileForWriting << assemble_time << '\t';
      fileForWriting << solve_time << '\t';
      fileForWriting << total_dofs << '\t';
      fileForWriting << cg_num_its << '\t';
      fileForWriting << std::endl;
      fileForWriting.close();

      delete uExCoeff_grad;
   }

   // 17. Free the used memory.
   delete uExCoeff;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}


