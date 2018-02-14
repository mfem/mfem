//              MFEM Example 3 - Parallel High-Performance Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../../data/fichera.mesh -perf -mf  -pc lor
//               mpirun -np 4 ex3p -m ../../data/fichera.mesh -perf -asm -pc ho
//               mpirun -np 4 ex3p -m ../../data/fichera.mesh -perf -asm -pc ho -sc
//               mpirun -np 4 ex3p -m ../../data/fichera.mesh -std  -asm -pc ho
//               mpirun -np 4 ex3p -m ../../data/fichera.mesh -std  -asm -pc ho -sc
//               mpirun -np 4 ex3p -m ../../data/amr-hex.mesh -perf -asm -pc ho -sc
//               mpirun -np 4 ex3p -m ../../data/amr-hex.mesh -std  -asm -pc ho -sc
//               mpirun -np 4 ex3p -m ../../data/ball-nurbs.mesh -perf -asm -pc ho  -sc
//               mpirun -np 4 ex3p -m ../../data/ball-nurbs.mesh -std  -asm -pc ho  -sc
//               mpirun -np 4 ex3p -m ../../data/pipe-nurbs.mesh -perf -mf  -pc lor
//               mpirun -np 4 ex3p -m ../../data/pipe-nurbs.mesh -std  -asm -pc ho  -sc
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define TWO_DIMENSIONS 0

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

// Define template parameters for optimized build.
#if TWO_DIMENSIONS
const Geometry::Type geom     = Geometry::SQUARE;
#else
const Geometry::Type geom     = Geometry::CUBE; // mesh elements  (default: hex)
#endif
const int            mesh_p   = 3;              // mesh curvature (default: 3)
const int            sol_p    = 3;              // solution order (default: 3)
const int            rdim     = Geometry::Constants<geom>::Dimension;
const int            ir_order = 2*sol_p+rdim-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef ND_FiniteElement<geom,sol_p>          sol_fe_t;
typedef ND_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
typedef TIntegrator<coeff_t,THcurlMassKernel> mass_integ_t;
typedef TIntegrator<coeff_t,THcurlcurlKernel> curl_integ_t;

typedef NDShapeEvaluator<sol_fe_t,int_rule_t>  sol_Shape_Eval;
typedef NDFieldEvaluator<sol_fes_t,ScalarLayout,int_rule_t> sol_Field_Eval;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,
                      mass_integ_t,sol_Shape_Eval,sol_Field_Eval> HPCMassBilinearForm;
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,
                      curl_integ_t,sol_Shape_Eval,sol_Field_Eval> HPCCurlBilinearForm;

class SumOperator : public Operator
{
public:
   SumOperator(const Operator& a_mass_oper,
               const Operator& a_curl_oper);

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   const Operator& a_mass_oper_;
   const Operator& a_curl_oper_;

   mutable Vector temp_;
};

SumOperator::SumOperator(const Operator& a_mass_oper,
                         const Operator& a_curl_oper)
   :
   Operator(a_mass_oper.Height()),
   a_mass_oper_(a_mass_oper),
   a_curl_oper_(a_curl_oper),
   temp_(a_mass_oper.Height())
{
}

void SumOperator::Mult(const Vector &x, Vector &y) const
{
   a_mass_oper_.Mult(x, y);
   a_curl_oper_.Mult(x, temp_);
   y += temp_;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
#if TWO_DIMENSIONS
   const char *mesh_file = "../../data/beam-quad.mesh";
#else
   const char *mesh_file = "../../data/beam-hex.mesh";
#endif
   int ser_ref_levels = -1;
   int par_ref_levels = 1;
   int order = sol_p;
   bool static_cond = false;
   const char *pc = "none";
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
   kappa = freq * M_PI;

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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
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
   FiniteElementCollection *fec = new ND_FECollection(order, dim);;
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
      pmesh_lor = new ParMesh(*pmesh);
      fec_lor = new ND_FECollection(1, dim);
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
   VectorFunctionCoefficient f(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     that will hold the matrix corresponding to the Laplacian operator.
   ParBilinearForm *a_form = new ParBilinearForm(fespace);
   ParBilinearForm *a_pc = NULL;
   if (pc_choice == LOR) { a_pc = new ParBilinearForm(fespace_lor); }
   if (pc_choice == HO)  { a_pc = new ParBilinearForm(fespace); }

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a_form->EnableStaticCondensation();
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
   a_form->UsePrecomputedSparsity();

   HPCMassBilinearForm *a_mass_hpc = NULL;
   HPCCurlBilinearForm *a_curl_hpc = NULL;
   Operator *a_oper = NULL;
   Operator *a_unconstrained_oper = NULL;
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   if (!perf)
   {
      // Standard assembly using a diffusion domain integrator
      a_form->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
      a_form->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
      a_form->Assemble();
   }
   else
   {
      // High-performance assembly/evaluation using the templated operator type
      a_mass_hpc = new HPCMassBilinearForm(mass_integ_t(coeff_t(1.0)), *fespace);
      a_curl_hpc = new HPCCurlBilinearForm(curl_integ_t(coeff_t(1.0)), *fespace);
      if (matrix_free)
      {
         a_mass_hpc->Assemble(); // partial assembly
         a_curl_hpc->Assemble();
      }
      else
      {
         a_mass_hpc->AssembleBilinearForm(*a_form); // full matrix assembly
         a_curl_hpc->AssembleBilinearForm(*a_form);
      }
   }
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 14. Define and apply a parallel PCG solver for AX=B

   // Setup the operator matrix (if applicable)
   HypreParMatrix A;
   Vector B, X;
   if (perf && matrix_free)
   {
      a_unconstrained_oper = new SumOperator(*a_mass_hpc, *a_curl_hpc);
      a_unconstrained_oper->FormLinearSystem(ess_tdof_list, x, *b, a_oper, X, B);
      HYPRE_Int glob_size = fespace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Size of linear system: " << glob_size << endl;
      }
   }
   else
   {
      a_form->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
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
      a_pc->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
      a_pc->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
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
         a_mass_hpc->AssembleBilinearForm(*a_pc);
         a_curl_hpc->AssembleBilinearForm(*a_pc);
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
      //amg = new HypreBoomerAMG(A_pc);
      //pcg->SetPreconditioner(*amg);
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
      a_oper->RecoverFEMSolution(X, *b, x);
   }
   else
   {
      a_form->RecoverFEMSolution(X, *b, x);
   }

   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
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
   delete a_form;
   delete a_mass_hpc;
   delete a_curl_hpc;
   delete muinv;
   delete sigma;
   if (a_oper != &A) { delete a_oper; }
   delete a_unconstrained_oper;
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
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
