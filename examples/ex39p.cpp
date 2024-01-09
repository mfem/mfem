//                       MFEM Example 39 - Parallel Version
//
// Compile with: make ex39p
//
// Sample runs:  mpirun -np 4 ex39p
//               mpirun -np 4 ex39p -ess "Southern Boundary"
//               mpirun -np 4 ex39p -src Base
//
// Device sample runs:
//               mpirun -np 4 ex39p -fa -d cuda
//
// Description:  This example code demonstrates the use of named attribute
//               sets in MFEM to specify material regions, boundary regions,
//               or source regions by name rather than attribute numbers. It
//               also demonstrates how new named attribute sets may be created
//               from arbitrary groupings of attribute number and used as a
//               convenient shorthand to refer to those groupings in other
//               portions of the application or through the command line.
//
//               The particular problem being solved here is nearly the same
//               as that in example 1 i.e. a simple finite element
//               discretization of the Laplace problem -Delta u = 1 with
//               homogeneous Dirichlet boundary conditions and, in this case,
//               an inhomogeneous diffusion coefficient. The diffusion
//               coefficient is given a small default value throughout the
//               domain which is increased by two separate amounts in two named
//               regions.
//
//               The example highlights the use of named attribute sets for
//               both subdomains and boundaries in different contexts as well
//               as basic methods to create named sets from existing attributes.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/compass.msh";
   int order = 1;
   string source_name = "Rose Even";
   string ess_name = "Boundary";
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&source_name,"-src","--source-attr-name",
                  "Name of attribute set containing source.");
   args.AddOption(&ess_name,"-ess","--ess-attr-name",
                  "Name of attribute set containing essential BC.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
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

   // Display attribute set names contained in the initial mesh
   if (myid == 0)
   {
      std::set<string> names = pmesh.GetAttributeSetNames();
      cout << "Element Attribute Set Names: ";
      for (auto const &set_name : names)
      {
         cout << " \"" << set_name << "\"";
      }
      cout << endl;

      std::set<string> bdr_names = pmesh.GetBdrAttributeSetNames();
      cout << "Boundary Attribute Set Names: ";
      for (auto const &bdr_set_name : bdr_names)
      {
         cout << " \"" << bdr_set_name << "\"";
      }
      cout << endl;
   }

   // Define new regions based on existing attribute sets
   {
      Array<int> & Na = pmesh.GetAttributeSet("N Even");
      Array<int> & Nb = pmesh.GetAttributeSet("N Odd");
      Array<int> & Sa = pmesh.GetAttributeSet("S Even");
      Array<int> & Sb = pmesh.GetAttributeSet("S Odd");
      Array<int> & Ea = pmesh.GetAttributeSet("E Even");
      Array<int> & Eb = pmesh.GetAttributeSet("E Odd");
      Array<int> & Wa = pmesh.GetAttributeSet("W Even");
      Array<int> & Wb = pmesh.GetAttributeSet("W Odd");

      // Create a new set spanning the North point
      pmesh.SetAttributeSet("North", Na);
      pmesh.AddToAttributeSet("North", Nb);

      // Create a new set spanning the South point
      pmesh.SetAttributeSet("South", Sa);
      pmesh.AddToAttributeSet("South", Sb);

      // Create a new set spanning the East point
      pmesh.SetAttributeSet("East", Ea);
      pmesh.AddToAttributeSet("East", Eb);

      // Create a new set spanning the West point
      pmesh.SetAttributeSet("West", Wa);
      pmesh.AddToAttributeSet("West", Wb);

      // Create a new set consisting of the "a" sides of the compass rose
      pmesh.SetAttributeSet("Rose Even", Na);
      pmesh.AddToAttributeSet("Rose Even", Sa);
      pmesh.AddToAttributeSet("Rose Even", Ea);
      pmesh.AddToAttributeSet("Rose Even", Wa);

      // Create a new set consisting of the "b" sides of the compass rose
      pmesh.SetAttributeSet("Rose Odd", Nb);
      pmesh.AddToAttributeSet("Rose Odd", Sb);
      pmesh.AddToAttributeSet("Rose Odd", Eb);
      pmesh.AddToAttributeSet("Rose Odd", Wb);


      // Create a new set consisting of the full compass rose
      Array<int> & Ra = pmesh.GetAttributeSet("Rose Even");
      Array<int> & Rb = pmesh.GetAttributeSet("Rose Odd");
      pmesh.SetAttributeSet("Rose", Ra);
      pmesh.AddToAttributeSet("Rose", Rb);
   }
   // Define new boundary regions based on existing boundary attribute sets
   {
      Array<int> & NNE = pmesh.GetBdrAttributeSet("NNE");
      Array<int> & NNW = pmesh.GetBdrAttributeSet("NNW");
      Array<int> & ENE = pmesh.GetBdrAttributeSet("ENE");
      Array<int> & ESE = pmesh.GetBdrAttributeSet("ESE");
      Array<int> & SSE = pmesh.GetBdrAttributeSet("SSE");
      Array<int> & SSW = pmesh.GetBdrAttributeSet("SSW");
      Array<int> & WNW = pmesh.GetBdrAttributeSet("WNW");
      Array<int> & WSW = pmesh.GetBdrAttributeSet("WSW");

      pmesh.SetBdrAttributeSet("Northern Boundary", NNE);
      pmesh.AddToBdrAttributeSet("Northern Boundary", NNW);

      pmesh.SetBdrAttributeSet("Southern Boundary", SSE);
      pmesh.AddToBdrAttributeSet("Southern Boundary", SSW);

      pmesh.SetBdrAttributeSet("Eastern Boundary", ENE);
      pmesh.AddToBdrAttributeSet("Eastern Boundary", ESE);

      pmesh.SetBdrAttributeSet("Western Boundary", WNW);
      pmesh.AddToBdrAttributeSet("Western Boundary", WSW);

      pmesh.SetBdrAttributeSet("Boundary",
                               pmesh.GetBdrAttributeSet("Northern Boundary"));
      pmesh.AddToBdrAttributeSet("Boundary",
                                 pmesh.GetBdrAttributeSet("Southern Boundary"));
      pmesh.AddToBdrAttributeSet("Boundary",
                                 pmesh.GetBdrAttributeSet("Eastern Boundary"));
      pmesh.AddToBdrAttributeSet("Boundary",
                                 pmesh.GetBdrAttributeSet("Western Boundary"));
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
   //    by marking all the boundary regions corresponding to the boundary
   //    attributes contained in the set named "ess_name" as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr_marker;
      pmesh.BdrAttrToMarker(pmesh.GetBdrAttributeSet(ess_name),
                            ess_bdr_marker);
      fespace.GetEssentialTrueDofs(ess_bdr_marker, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1_s,phi_i) where phi_i are the basis functions in fespace and 1_s
   //    is an indicator function equal to 1 on the region defined by the
   //    named set "source_name" and zero elsewhere.

   Array<int> source_marker;
   pmesh.AttrToMarker(pmesh.GetAttributeSet(source_name), source_marker);

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one), source_marker);
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
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }

   ConstantCoefficient defaultCoef(1.0e-6);
   ConstantCoefficient baseCoef(1.0);
   ConstantCoefficient roseCoef(2.0);

   Array<int> base_marker;
   Array<int> rose_marker;
   pmesh.AttrToMarker(pmesh.GetAttributeSet("Base"), base_marker);
   pmesh.AttrToMarker(pmesh.GetAttributeSet("Rose Even"), rose_marker);

   a.AddDomainIntegrator(new DiffusionIntegrator(defaultCoef));
   a.AddDomainIntegrator(new DiffusionIntegrator(baseCoef), base_marker);
   a.AddDomainIntegrator(new DiffusionIntegrator(roseCoef), rose_marker);

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13. Solve the linear system A X = B.
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
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

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
      sol_sock << "solution\n" << pmesh << x << "keys Rjmm" << flush;
   }

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
