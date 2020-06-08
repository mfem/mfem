//              MFEM Example 1 - Parallel High-Performance Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../../data/fichera.mesh -perf -mf  -pc lor
//               mpirun -np 4 ex1p -m ../../data/fichera.mesh -perf -asm -pc ho
//               mpirun -np 4 ex1p -m ../../data/fichera.mesh -perf -asm -pc ho -sc
//               mpirun -np 4 ex1p -m ../../data/fichera.mesh -std  -asm -pc ho
//               mpirun -np 4 ex1p -m ../../data/fichera.mesh -std  -asm -pc ho -sc
//               mpirun -np 4 ex1p -m ../../data/amr-hex.mesh -perf -asm -pc ho -sc
//               mpirun -np 4 ex1p -m ../../data/amr-hex.mesh -std  -asm -pc ho -sc
//               mpirun -np 4 ex1p -m ../../data/ball-nurbs.mesh -perf -asm -pc ho  -sc
//               mpirun -np 4 ex1p -m ../../data/ball-nurbs.mesh -std  -asm -pc ho  -sc
//               mpirun -np 4 ex1p -m ../../data/pipe-nurbs.mesh -perf -mf  -pc lor
//               mpirun -np 4 ex1p -m ../../data/pipe-nurbs.mesh -std  -asm -pc ho  -sc
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

#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define template parameters for optimized build.
const Geometry::Type geom     = Geometry::CUBE; // mesh elements  (default: hex)
const int            mesh_p   = 3;              // mesh curvature (default: 3)
const int            sol_p    = 3;              // solution order (default: 3)
const int            rdim     = Geometry::Constants<geom>::Dimension;
const int            ir_order = 2*sol_p+rdim-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_p>          sol_fe_t;
typedef H1_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t> HPCBilinearForm;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/fichera.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 1;
   int order = sol_p;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool static_cond = false;
   const char *pc = "lor";
   bool perf = true;
   bool matrix_free = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial;"
                  " -1 = auto: <= 10,000 elements.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                  "Enable high-performance, tensor-based, assembly/evaluation.");
   args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                  "Use matrix-free evaluation or efficient matrix assembly in "
                  "the high-performance version.");
   args.AddOption(&pc, "-pc", "--preconditioner",
                  "Preconditioner: lor - low-order-refined (matrix-free) AMG, "
                  "ho - high-order (assembled) AMG, none.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   if (static_cond && perf && matrix_free)
   {
      if (myid == 0)
      {
         cout << "\nStatic condensation can not be used with matrix-free"
              " evaluation!\n" << endl;
      }
      MPI_Finalize();
      return 2;
   }
   MFEM_VERIFY(perf || !matrix_free,
               "--standard-version is not compatible with --matrix-free");
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   enum PCType { NONE, LOR, HO };
   PCType pc_choice;
   if (!strcmp(pc, "ho")) { pc_choice = HO; }
   else if (!strcmp(pc, "lor")) { pc_choice = LOR; }
   else if (!strcmp(pc, "none")) { pc_choice = NONE; }
   else
   {
      mfem_error("Invalid Preconditioner specified");
      return 3;
   }

   if (myid == 0)
   {
      cout << "\nMFEM SIMD width: " << MFEM_SIMD_BYTES/sizeof(double)
           << " doubles\n" << endl;
   }

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   if (myid == 0)
   {
      cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Check if the optimized version matches the given mesh
   if (perf)
   {
      if (myid == 0)
      {
         cout << "High-performance version using integration rule with "
              << int_rule_t::qpts << " points ..." << endl;
      }
      if (!mesh_t::MatchesGeometry(*mesh))
      {
         if (myid == 0)
         {
            cout << "The given mesh does not match the optimized 'geom' parameter.\n"
                 << "Recompile with suitable 'geom' value." << endl;
         }
         delete mesh;
         MPI_Finalize();
         return 4;
      }
      else if (!mesh_t::MatchesNodes(*mesh))
      {
         if (myid == 0)
         {
            cout << "Switching the mesh curvature to match the "
                 << "optimized value (order " << mesh_p << ") ..." << endl;
         }
         mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
      }
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements, or as specified on the command line with the
   //    option '--refine-serial'.
   {
      int ref_levels = (ser_ref_levels != -1) ? ser_ref_levels :
                       (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (!perf && mesh->NURBSext)
   {
      const int new_mesh_p = std::min(sol_p, mesh_p);
      if (myid == 0)
      {
         cout << "NURBS mesh: switching the mesh curvature to be "
              << "min(sol_p, mesh_p) = " << new_mesh_p << " ..." << endl;
      }
      mesh->SetCurvature(new_mesh_p, false, -1, Ordering::byNODES);
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   if (pmesh->MeshGenerator() & 1) // simplex mesh
   {
      MFEM_VERIFY(pc_choice != LOR, "triangle and tet meshes do not support"
                  " the LOR preconditioner yet");
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim, basis);
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
      fec = new H1_FECollection(order = 1, dim, basis);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   ParMesh *pmesh_lor = NULL;
   FiniteElementCollection *fec_lor = NULL;
   ParFiniteElementSpace *fespace_lor = NULL;
   if (pc_choice == LOR)
   {
      int basis_lor = basis;
      if (basis == BasisType::Positive) { basis_lor=BasisType::ClosedUniform; }
      pmesh_lor = new ParMesh(pmesh, order, basis_lor);
      fec_lor = new H1_FECollection(1, dim);
      fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);
   }

   // 8. Check if the optimized version matches the given space
   if (perf && !sol_fes_t::Matches(*fespace))
   {
      if (myid == 0)
      {
         cout << "The given order does not match the optimized parameter.\n"
              << "Recompile with suitable 'sol_p' value." << endl;
      }
      delete fespace;
      delete fec;
      delete mesh;
      MPI_Finalize();
      return 5;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
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

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system, which in this case is
   //     (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     that will hold the matrix corresponding to the Laplacian operator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ParBilinearForm *a_pc = NULL;
   if (pc_choice == LOR) { a_pc = new ParBilinearForm(fespace_lor); }
   if (pc_choice == HO)  { a_pc = new ParBilinearForm(fespace); }

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a->EnableStaticCondensation();
      MFEM_VERIFY(pc_choice != LOR,
                  "cannot use LOR preconditioner with static condensation");
   }

   if (myid == 0)
   {
      cout << "Assembling the matrix ..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   // Pre-allocate sparsity assuming dense element matrices
   a->UsePrecomputedSparsity();

   HPCBilinearForm *a_hpc = NULL;
   Operator *a_oper = NULL;

   if (!perf)
   {
      // Standard assembly using a diffusion domain integrator
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
      a->Assemble();
   }
   else
   {
      // High-performance assembly/evaluation using the templated operator type
      a_hpc = new HPCBilinearForm(integ_t(coeff_t(1.0)), *fespace);
      if (matrix_free)
      {
         a_hpc->Assemble(); // partial assembly
      }
      else
      {
         a_hpc->AssembleBilinearForm(*a); // full matrix assembly
      }
   }
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 14. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.

   // Setup the operator matrix (if applicable)
   HypreParMatrix A;
   Vector B, X;
   if (perf && matrix_free)
   {
      a_hpc->FormLinearSystem(ess_tdof_list, x, *b, a_oper, X, B);
      HYPRE_Int glob_size = fespace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Size of linear system: " << glob_size << endl;
      }
   }
   else
   {
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      HYPRE_Int glob_size = A.GetGlobalNumRows();
      if (myid == 0)
      {
         cout << "Size of linear system: " << glob_size << endl;
      }
      a_oper = &A;
   }

   // Setup the matrix used for preconditioning
   if (myid == 0)
   {
      cout << "Assembling the preconditioning matrix ..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();

   HypreParMatrix A_pc;
   if (pc_choice == LOR)
   {
      // TODO: assemble the LOR matrix using the performance code
      a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
      a_pc->UsePrecomputedSparsity();
      a_pc->Assemble();
      a_pc->FormSystemMatrix(ess_tdof_list, A_pc);
   }
   else if (pc_choice == HO)
   {
      if (!matrix_free)
      {
         A_pc.MakeRef(A); // matrix already assembled, reuse it
      }
      else
      {
         a_pc->UsePrecomputedSparsity();
         a_hpc->AssembleBilinearForm(*a_pc);
         a_pc->FormSystemMatrix(ess_tdof_list, A_pc);
      }
   }
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // Solve with CG or PCG, depending if the matrix A_pc is available
   CGSolver *pcg;
   pcg = new CGSolver(MPI_COMM_WORLD);
   pcg->SetRelTol(1e-6);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(1);

   HypreSolver *amg = NULL;

   pcg->SetOperator(*a_oper);
   if (pc_choice != NONE)
   {
      amg = new HypreBoomerAMG(A_pc);
      pcg->SetPreconditioner(*amg);
   }

   tic_toc.Clear();
   tic_toc.Start();

   pcg->Mult(B, X);

   tic_toc.Stop();
   delete amg;

   if (myid == 0)
   {
      // Note: In the pcg algorithm, the number of operator Mult() calls is
      //       N_iter and the number of preconditioner Mult() calls is N_iter+1.
      cout << "Time per CG step: "
           << tic_toc.RealTime() / pcg->GetNumIterations() << "s." << endl;
   }

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   if (perf && matrix_free)
   {
      a_hpc->RecoverFEMSolution(X, *b, x);
   }
   else
   {
      a->RecoverFEMSolution(X, *b, x);
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
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

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete a_hpc;
   if (a_oper != &A) { delete a_oper; }
   delete a_pc;
   delete b;
   delete fespace;
   delete fespace_lor;
   delete fec_lor;
   delete pmesh_lor;
   if (order > 0) { delete fec; }
   delete pmesh;
   delete pcg;

   MPI_Finalize();

   return 0;
}
