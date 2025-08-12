//               MFEM Example 2 - Linear elasticity + patch partial assembly
//
// Compile with: make nurbs_patch_ex2
//
// Sample runs:  nurbs_patch_ex2 -incdeg 2 -rf 4 -patcha -pa
//               nurbs_patch_ex2 -incdeg 2 -rf 4 -patcha -pa -int 1
//               nurbs_patch_ex2 -incdeg 2 -rf 4 -patcha -pa -int 1 -pc 1
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               This example is a specialization of ex2 which demonstrates
//               patch-wise partial assembly on NURBS meshes.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   bool pa = false;
   bool patchAssembly = false;
   int refinement_factor = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int spline_integration_type = 0;
   int preconditioner = 0;
   int visport = 19916;
   bool csv_info = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&refinement_factor, "-rf", "--refinement-factor",
                  "Refinement factor for the NURBS mesh.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&spline_integration_type, "-int", "--integration-type",
                  "Integration rule type: 0 - full order Gauss Legendre, "
                  "1 - reduced order Gaussian Legendre");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - Jacobi");
   args.AddOption(&csv_info, "-csv", "--csv-info", "-no-csv",
                  "--no-csv-info",
                  "Enable or disable dump of info into csv.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(pa && !patchAssembly), "Patch assembly must be used with -pa");
   MFEM_VERIFY(spline_integration_type >= 0 && spline_integration_type < 2,
               "Spline integration type must be 0 or 1 for this example");

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const bool isNURBS = mesh.NURBSext;

   // Verify mesh is valid for this problem
   MFEM_VERIFY(isNURBS, "Example is for NURBS meshes");
   MFEM_VERIFY(mesh.GetNodes(), "NURBS mesh must have nodes");
   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      cout << "\nInput mesh should have at least two boundary"
           << "attributes! (See schematic in ex2.cpp)\n"
           << endl;
   }

   // 3. Optionally, increase the NURBS degree.
   if (nurbs_degree_increase>0)
   {
      mesh.DegreeElevate(nurbs_degree_increase);
   }

   // 4. Refine the mesh to increase the resolution.
   if (refinement_factor > 1)
   {
      mesh.NURBSUniformRefinement(refinement_factor);
   }

   // 5. Define a finite element space on the mesh.
   // Node ordering is important - right now, only works with byVDIM
   FiniteElementCollection * fec = mesh.GetNodes()->OwnFEC();
   cout << "fec order = " << fec->GetOrder() << endl;

   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec,
                                                   dim, Ordering::byVDIM);
   cout << "Finite Element Collection: " << fec->Name() << endl;
   const int ndof = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << ndof << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.)
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm b(&fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "Assembling RHS ... " << flush;
   b.Assemble();
   cout << "done." << endl;

   // 8. Define the solution vector x as a finite element grid function
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.)

   // Lame parameters
   Vector lambda(mesh.attributes.Max());
   Vector mu(mesh.attributes.Max());
   lambda = 1.0; lambda(0) = lambda(1)*50;
   mu = 1.0; mu(0) = mu(1)*50;

   PWConstCoefficient lambda_func(lambda);
   PWConstCoefficient mu_func(mu);

   // Bilinear integrator
   ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
   NURBSMeshRules* meshRules = nullptr;
   if (patchAssembly)
   {
      ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      // Integration rule for the 1d bases defined on each knotvector
      SplineIntegrationRule splineRule(spline_integration_type);
      // Set the patch integration rules
      meshRules = new NURBSMeshRules(mesh, splineRule);
      ei->SetNURBSPatchIntRule(meshRules);
   }

   // 10. Assembly
   StopWatch sw;
   sw.Start();

   // Define and assemble bilinear form
   cout << "Assembling a ... " << flush;
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(ei);
   a.Assemble();
   cout << "done." << endl;

   // Form linear system
   cout << "Forming linear system ... " << flush;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace.GetTrueVSize() << ")" << endl;

   // 11. Get the preconditioner
   CGSolver solver;
   solver.SetOperator(*A);

   // No preconditioner
   if (preconditioner == 0)
   {
      cout << "No preconditioner set ... " << endl;
   }
   // Jacobi
   else if (preconditioner == 1)
   {
      cout << "Setting up preconditioner (Jacobi) ... " << endl;
      OperatorJacobiSmoother P(a, ess_tdof_list);
      solver.SetPreconditioner(P);
   }
   else
   {
      MFEM_ABORT("Invalid preconditioner setting.")
   }

   sw.Stop();
   const real_t timeAssemble = sw.RealTime();
   sw.Clear();
   sw.Start();

   // 12. Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;
   solver.SetMaxIter(1e4);
   solver.SetPrintLevel(1);
   solver.SetRelTol(sqrt(1e-6));
   solver.SetAbsTol(sqrt(1e-14));

   solver.Mult(B, X);

   cout << "Done solving system." << endl;

   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Collect results and write to file
   const int niter = solver.GetNumIterations();
   const int dof_per_sec_solve = ndof * niter / timeSolve;
   const int dof_per_sec_total = ndof * niter / timeTotal;
   cout << "Time to assemble: " << timeAssemble << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;
   cout << "Total time: " << timeTotal << " seconds" << endl;
   cout << "Dof/sec (solve): " << dof_per_sec_solve << endl;
   cout << "Dof/sec (total): " << dof_per_sec_total << endl;

   if (csv_info)
   {
      ofstream results_ofs("ex2_results.csv", ios_base::app);
      // If file does not exist, write the header
      if (results_ofs.tellp() == 0)
      {
         results_ofs << "patcha, pa, pc, sint, "         // settings
                     << "mesh, rf, deg_inc, ndof, "      // mesh
                     << "niter, absnorm, relnorm, "      // solver
                     << "linf, l2, "                     // solution
                     << "t_assemble, t_solve, t_total, " // timing
                     << "dof/s_solve, dof/s_total"       // benchmarking
                     << endl;
      }

      results_ofs << patchAssembly << ", "               // settings
                  << pa << ", "
                  << preconditioner << ", "
                  << spline_integration_type << ", "
                  << mesh_file << ", "                   // mesh
                  << refinement_factor << ", "
                  << nurbs_degree_increase << ", "
                  << ndof << ", "
                  << niter << ", "                       // solver
                  << solver.GetFinalNorm() << ", "
                  << solver.GetFinalRelNorm() << ", "
                  << x.Normlinf() << ", "                // solution
                  << x.Norml2() << ", "
                  << timeAssemble << ", "                // timing
                  << timeSolve << ", "
                  << timeTotal << ", "
                  << dof_per_sec_solve << ", "           // benchmarking
                  << dof_per_sec_total << endl;

      results_ofs.close();
   }

   // 14. Save the displaced mesh and the inverted solution
   {
      cout << "Saving mesh and solution to file..." << endl;
      GridFunction *nodes = mesh.GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(16);
      x.Save(sol_ofs);
   }

   // 15. Send the data by socket to a GLVis server.
   if (visualization)
   {
      // send to socket
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x;
      sol_sock << "window_geometry " << 0 << " " << 0 << " "
               << 800 << " " << 800 << "\n"
               << "keys agc\n" << std::flush;
   }

   // 16. Free the used memory.
   delete meshRules;

   return 0;
}
