//               IGA with LOR Preconditioning
//
// Compile with: make nurbs_lor
//
// Sample runs:  nurbs_lor -ref 2 -incdeg 3 -pc 1
//
// Description:  This example code solves a diffusion problem with
//               different preconditioners and records some stats.
//               Preconditioner (-pc) choices:
//                 - 0: No PC
//                 - 1: LOR AMG (R = Identity)
//                 - 2: LOR AMG (choose interpolation with -interp)
//

#include "nurbs_lor.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/cube-nurbs.mesh";
   bool pa = false;
   bool patchAssembly = false;
   bool reducedIntegration = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int interp_rule_ = 1;
   int preconditioner = 0;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&reducedIntegration, "-rint", "--reduced-integration", "-fint",
                  "--full-integration", "Enable reduced integration rules.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&interp_rule_, "-int", "--interpolation-rule",
                  "Interpolation rule: 0 - Greville, 1 - Botella, 2 - Demko, 3 - Uniform");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - diagonal, 2 - LOR AMG");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   // MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");
   MFEM_VERIFY(!(pa && !patchAssembly), "PA must be used with -patcha");

   NURBSInterpolationRule interp_rule = static_cast<NURBSInterpolationRule>
                                        (interp_rule_);

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int NP = mesh.NURBSext->GetNP();

   // Verify mesh is valid for this problem
   MFEM_VERIFY(mesh.IsNURBS(), "Example is for NURBS meshes");
   MFEM_VERIFY(mesh.GetNodes(), "NURBS mesh must have nodes");

   // 3. Optionally, increase the NURBS degree.
   if (nurbs_degree_increase>0)
   {
      mesh.DegreeElevate(nurbs_degree_increase);
   }

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.NURBSUniformRefinement();
   }

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec);

   const int Ndof = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << NP << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   // mesh.MarkExternalBoundaries(ess_bdr);
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Define the solution vector x as a finite element grid function
   GridFunction x(&fespace);
   x = 0.0;

   // 8. Setup linear form b(.)
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Setup bilinear form a(.,.)
   DiffusionIntegrator *di = new DiffusionIntegrator(one);

   if (patchAssembly)
   {
      if (reducedIntegration)
      {
         di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE_REDUCED);
      }
      else
      {
         di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      }
      SetPatchIntegrationRules(mesh, SplineIntegrationRule::FULL_GAUSSIAN, di);
   }

   // 10. Assembly
   StopWatch sw;
   sw.Start();

   // Define and assemble bilinear form
   cout << "Assembling a ... " << flush;
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(di);
   a.Assemble();
   cout << "done." << endl;

   // Form linear system
   cout << "Forming linear system ... " << flush;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace.GetTrueVSize() << ")" << endl;

   sw.Stop();
   const real_t timeAssemble_A = sw.RealTime();
   sw.Clear();
   sw.Start();

   // 11. Get the preconditioner
   // We define solver here because SetOperator needs to be used before
   // SetPreconditioner *if* we are using hypre
   CGSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(*A);

   // No preconditioner
   if (preconditioner == 0)
   {
      cout << "No preconditioner set ... " << endl;
   }
   // LOR AMG
   else if ((preconditioner == 1) || (preconditioner == 2))
   {
      cout << "Setting up preconditioner (LOR AMG) ... " << endl;

      // Create the LOR mesh
      const int vdim = fespace.GetVDim();
      Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(interp_rule, vdim, NULL);

      // Write low order mesh to file
      if (visualization)
      {
         ofstream ofs("lo_mesh.mesh");
         ofs.precision(8);
         lo_mesh.Print(ofs);
      }

      FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
      cout << "lo_fec order: " << lo_fec->GetOrder() << endl;
      FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec);
      const int lo_Ndof = lo_fespace.GetTrueVSize();
      MFEM_VERIFY(Ndof == lo_Ndof, "Low-order problem requires same Ndof");

      Array<int> lo_ess_tdof_list, lo_ess_bdr(lo_mesh.bdr_attributes.Max());
      lo_ess_bdr = 1;
      // lo_mesh.MarkExternalBoundaries(lo_ess_bdr);
      lo_fespace.GetEssentialTrueDofs(lo_ess_bdr, lo_ess_tdof_list);

      GridFunction lo_x(&lo_fespace);
      lo_x = 0.0;

      LinearForm lo_b(&lo_fespace);
      lo_b.AddDomainIntegrator(new DomainLFIntegrator(one));
      lo_b.Assemble();

      DiffusionIntegrator *lo_di = new DiffusionIntegrator(one);

      // Set up problem
      BilinearForm lo_a(&lo_fespace);
      lo_a.AddDomainIntegrator(lo_di);
      lo_a.Assemble();

      // Define linear system
      OperatorPtr lo_A;
      Vector lo_B, lo_X;
      lo_a.FormLinearSystem(lo_ess_tdof_list, lo_x, lo_b, lo_A, lo_X, lo_B);
      // Array<int> nobcs;
      // lo_a.FormLinearSystem(nobcs, lo_x, lo_b, lo_A, lo_X, lo_B);

      // Set up HypreBoomerAMG on the low-order problem
      HYPRE_BigInt row_starts[2] = {0, Ndof};
      SparseMatrix *lo_Amat = new SparseMatrix(lo_a.SpMat());
      HypreParMatrix *lo_A_hypre = new HypreParMatrix(
         MPI_COMM_WORLD,
         HYPRE_BigInt(Ndof),
         row_starts,
         lo_Amat
      );
      HypreBoomerAMG *lo_P = new HypreBoomerAMG(*lo_A_hypre);

      if (preconditioner == 1)
      {
         cout << "LOR AMG: R = Identity ... " << endl;
         solver.SetPreconditioner(*lo_P);
      }
      else
      {
         cout << "LOR AMG: R = X^-1 ... " << endl;

         NURBSInterpolator* interpolator = new NURBSInterpolator(&mesh, &lo_mesh);
         interpolator->SetBandwidths(1e-5, true); // uses initial guess of order*2+1
         NURBSLORPreconditioner *P = new NURBSLORPreconditioner(interpolator, ess_tdof_list, lo_P);
         solver.SetPreconditioner(*P);
      }
   }
   else if (preconditioner == 3)
   {
      cout << "Setting up preconditioner (HO AMG) ... " << endl;

      HYPRE_BigInt row_starts[2] = {0, Ndof};
      SparseMatrix *Amat = new SparseMatrix(a.SpMat());
      HypreParMatrix *A_hypre = new HypreParMatrix(
         MPI_COMM_WORLD,
         HYPRE_BigInt(Ndof),
         row_starts,
         Amat
      );
      HypreBoomerAMG *P = new HypreBoomerAMG(*A_hypre);

      solver.SetPreconditioner(*P);
   }
   else
   {
      MFEM_ABORT("Invalid preconditioner setting.")
   }

   sw.Stop();
   const real_t timeAssemble_PC = sw.RealTime();
   sw.Clear();
   sw.Start();
   const real_t timeAssemble = timeAssemble_A + timeAssemble_PC;

   // 12. Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;
   solver.SetMaxIter(1e5);
   solver.SetPrintLevel(1);
   // These tolerances end up getting squared
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0);

   solver.Mult(B, X);

   cout << "Done solving system." << endl;
   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Collect results and write to file
   const long Niter = solver.GetNumIterations();
   const long dof_per_sec_solve = (long)Ndof * Niter / timeSolve;
   const long dof_per_sec_total = (long)Ndof * Niter / timeTotal;
   cout << "Time to assemble A: " << timeAssemble_A << " seconds" << endl;
   cout << "Time to assemble PC: " << timeAssemble_PC << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;
   cout << "Total time: " << timeTotal << " seconds" << endl;
   cout << "Dof/sec (solve): " << dof_per_sec_solve << endl;
   cout << "Dof/sec (total): " << dof_per_sec_total << endl;

   ofstream results_ofs("nurbs_lor_solve_results.csv", ios_base::app);
   // If file does not exist, write the header
   if (results_ofs.tellp() == 0)
   {
      results_ofs << "patcha, pa, rint, pc, proj, "   // settings
                  << "mesh, refs, deg_inc, ndof, "    // mesh
                  << "niter, absnorm, relnorm, "      // solver
                  << "linf, l2, "                     // solution
                  << "t_assemble, t_A, t_PC, "        // timing
                  << "t_solve, t_total, "
                  << "dof/s_solve, dof/s_total"       // benchmarking
                  << endl;
   }

   results_ofs << patchAssembly << ", "               // settings
               << pa << ", "
               << reducedIntegration << ", "
               << preconditioner << ", "
               << interp_rule_ << ", "
               << mesh_file << ", "                   // mesh
               << ref_levels << ", "
               << nurbs_degree_increase << ", "
               << Ndof << ", "
               << Niter << ", "                       // solver
               << solver.GetFinalNorm() << ", "
               << solver.GetFinalRelNorm() << ", "
               << x.Normlinf() << ", "                // solution
               << x.Norml2() << ", "
               << timeAssemble << ", "                // timing
               << timeAssemble_A << ", "
               << timeAssemble_PC << ", "
               << timeSolve << ", "
               << timeTotal << ", "
               << dof_per_sec_solve << ", "           // benchmarking
               << dof_per_sec_total << endl;

   results_ofs.close();

   // 14. Save the mesh and the solution
   if (visualization)
   {
      cout << "Saving mesh and solution to file..." << endl;
      ofstream mesh_ofs("mesh.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(16);
      x.Save(sol_ofs);
   }

   // 15. Free the used memory.
   // delete R;

   return 0;
}
