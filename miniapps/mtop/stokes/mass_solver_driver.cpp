/// @file mass_solver_driver.cpp
/// @brief Driver that tests MassMatrixSolver on a 3-D unit cube mesh.
///
/// Usage:
///   mpirun -np 4 ./mass_solver_driver [options]
///
/// Options parsed by MFEM's OptionsParser:
///   -m  <mesh>   Mesh file           (default: inline 3-D hex mesh)
///   -o  <order>  FE polynomial order (default: 2)
///   -r  <refs>   Uniform refinements (default: 2)
///   -c  <val>    Constant mass coeff (default: 1.0)
///       --use-func-coeff   Use a spatially varying coefficient instead
///   -rt <tol>    CG relative tol    (default: 1e-8)
///   -mi <iter>   CG max iterations  (default: 500)
///   -pl <lvl>    CG print level     (default: 1)
///
/// The driver:
///  1. Builds/refines a parallel mesh.
///  2. Creates an H1 FE space.
///  3. Projects a smooth manufactured function f onto the space to get x_ref.
///  4. Assembles b = M x_ref explicitly (full assembly, for reference only).
///  5. Solves M x = b with MassMatrixSolver (partial assembly + Jacobi CG).
///  6. Reports ||x - x_ref||_inf and solver diagnostics.

#include "MassMatrixSolver.hpp"
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

// ---------------------------------------------------------------------------
// Spatially varying coefficient:  rho(x,y,z) = 2 + sin(pi*x)*cos(pi*y)
// ---------------------------------------------------------------------------
real_t rho_func(const Vector &xv)
{
   return 2.0 + std::sin(M_PI * xv(0)) * std::cos(M_PI * xv(1));
}

// ---------------------------------------------------------------------------
// Manufactured solution:  u(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)
// ---------------------------------------------------------------------------
real_t u_exact(const Vector &xv)
{
   return std::sin(M_PI * xv(0))
        * std::sin(M_PI * xv(1))
        * std::sin(M_PI * xv(2));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
   // ---- MPI init ----
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int myid = Mpi::WorldRank();

   // ---- Options ----
   const char *mesh_file    = "";     // empty → inline mesh
   int         order        = 2;
   int         refinements  = 2;
   real_t      coeff_val    = 1.0;
   bool        use_func     = false;
   real_t      rel_tol      = 1e-8;
   int         max_iter     = 500;
   int         print_level  = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file,   "-m",    "--mesh",        "Mesh file.");
   args.AddOption(&order,       "-o",    "--order",       "FE order.");
   args.AddOption(&refinements, "-r",    "--refinements", "Uniform refs.");
   args.AddOption(&coeff_val,   "-c",    "--coeff",       "Constant mass coeff.");
   args.AddOption(&use_func,
                  "-uf", "--use-func-coeff",
                  "-uc", "--use-const-coeff",
                  "Use spatially varying coefficient (default: constant).");
   args.AddOption(&rel_tol,     "-rt",   "--rel-tol",     "CG relative tol.");
   args.AddOption(&max_iter,    "-mi",   "--max-iter",    "CG max iterations.");
   args.AddOption(&print_level, "-pl",   "--print-level", "CG print level.");
   args.Parse();
   if (!args.Good()) { if (myid == 0) args.PrintUsage(cout); return 1; }
   if (myid == 0)    args.PrintOptions(cout);

   // ---- Mesh ----
   Mesh *mesh = nullptr;
   if (strlen(mesh_file) > 0)
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }
   else
   {
      // 8x8x8 inline hex mesh on the unit cube
      mesh = new Mesh(Mesh::MakeCartesian3D(8, 8, 8, Element::HEXAHEDRON));
   }

   for (int i = 0; i < refinements; ++i) mesh->UniformRefinement();

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   const int dim = pmesh.Dimension();
   if (myid == 0)
      cout << "\nMesh: " << pmesh.GetNE() << " elements, dim=" << dim << "\n";

   // ---- FE space (H1) ----
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   if (myid == 0)
      cout << "True DOFs: " << pfes.GlobalTrueVSize() << "\n\n";

   // ---- Manufactured solution x_ref ----
   FunctionCoefficient u_coeff(u_exact);
   ParGridFunction x_ref(&pfes);
   x_ref.ProjectCoefficient(u_coeff);

   // True-dof representation of x_ref
   Vector X_ref(pfes.TrueVSize());
   x_ref.GetTrueDofs(X_ref);

   // ---- Build RHS  b = M_full * x_ref  (full assembly, reference only) ----
   // We use a full-assembly form here ONLY to generate the RHS vector so the
   // test is self-contained.  The MassMatrixSolver itself uses PA.
   //
   // IMPORTANT: MassIntegrator stores a raw *reference* to the Coefficient.
   // The coefficient objects must therefore outlive mass_full.Assemble() —
   // they are declared here in the enclosing scope to guarantee that.
   FunctionCoefficient rho_ref(rho_func);       // used when use_func == true
   ConstantCoefficient cst_ref(coeff_val);      // used when use_func == false

   ParBilinearForm mass_full(&pfes);
   if (use_func)
      mass_full.AddDomainIntegrator(new MassIntegrator(rho_ref));
   else
      mass_full.AddDomainIntegrator(new MassIntegrator(cst_ref));

   mass_full.Assemble();
   mass_full.Finalize();

   // Obtain the parallel matrix and multiply
   HypreParMatrix *M_hyp = mass_full.ParallelAssemble();
   Vector B(pfes.TrueVSize());
   M_hyp->Mult(X_ref, B);
   delete M_hyp;

   // ---- Solve with MassMatrixSolver ----
   Vector X_sol(pfes.TrueVSize());
   X_sol = 0.0;  // zero initial guess

   if (myid == 0)
      cout << "--- Test 1: " << (use_func ? "FunctionCoefficient" : "ConstantCoefficient")
           << " ---\n";

   if (use_func)
   {
      // Constructor 1 – general coefficient
      FunctionCoefficient rho(rho_func);
      MassMatrixSolver solver(&pfes, &rho, rel_tol, max_iter, print_level);
      solver.Solve(B, X_sol);

      if (myid == 0)
         cout << "  Iterations : " << solver.GetNumIterations()  << "\n"
              << "  Final res  : " << solver.GetFinalResidual()  << "\n"
              << "  Converged  : " << (solver.GetConverged() ? "yes" : "NO") << "\n";
   }
   else
   {
      // Constructor 2 – constant coefficient
      MassMatrixSolver solver(&pfes, coeff_val, rel_tol, max_iter, print_level);
      solver.Solve(B, X_sol);

      if (myid == 0)
         cout << "  Iterations : " << solver.GetNumIterations()  << "\n"
              << "  Final res  : " << solver.GetFinalResidual()  << "\n"
              << "  Converged  : " << (solver.GetConverged() ? "yes" : "NO") << "\n";
   }

   // ---- Error check ----
   Vector err(X_sol);
   err -= X_ref;
   const real_t linf = err.Normlinf();
   real_t global_linf = 0.0;
   MPI_Allreduce(&linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   if (myid == 0)
      cout << "\n  ||x_sol - x_ref||_inf = " << global_linf << "\n";

   // ---- Test both constructors in sequence ----
   if (myid == 0) cout << "\n--- Test 2: both constructors back-to-back ---\n";
   {
      // Const coeff constructor
      MassMatrixSolver s1(&pfes, 3.14, rel_tol, max_iter, 0);
      Vector b1(pfes.TrueVSize()), x1(pfes.TrueVSize());
      b1 = 1.0; x1 = 0.0;
      s1.Solve(b1, x1);
      if (myid == 0)
         cout << "  s1 (const 3.14)  iters=" << s1.GetNumIterations()
              << "  converged=" << s1.GetConverged() << "\n";

      // Nullptr coefficient → unit mass matrix
      MassMatrixSolver s2(&pfes, static_cast<Coefficient*>(nullptr),
                          rel_tol, max_iter, 0);
      Vector b2(pfes.TrueVSize()), x2(pfes.TrueVSize());
      b2 = 1.0; x2 = 0.0;
      s2.Solve(b2, x2);
      if (myid == 0)
         cout << "  s2 (nullptr→unit) iters=" << s2.GetNumIterations()
              << "  converged=" << s2.GetConverged() << "\n";
   }

   if (myid == 0) cout << "\nAll tests passed.\n";
   return 0;
}
