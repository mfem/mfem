//               MFEM Example 2 - NURBS with patch-wise assembly
//
// Compile with: make nurbs_patch_ex2
//
// Sample runs:  nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha
//               nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha -pa
//               nurbs_patch_ex2 -incdeg 3 -ref 2 -iro 8 -patcha -fint
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
//               patch-wise matrix assembly and partial assembly on NURBS
//               meshes.

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
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int ir_order = -1;
   int preconditioner = 0;
   int visport = 19916;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&ir_order, "-iro", "--integration-order",
                  "Order of integration rule.");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - diagonal, 2 - LOR.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   bool isNURBS = mesh.NURBSext;

   // Verify mesh is valid for this problem
   MFEM_VERIFY(isNURBS, "Example is for NURBS meshes");
   MFEM_VERIFY(mesh.GetNodes(), "NURBS mesh must have nodes");
   if (mesh.bdr_attributes.Max() < 2)
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
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define a finite element space on the mesh.
   // Node ordering is important - right now, only works with byVDIM
   const Ordering::Type fes_ordering = Ordering::byVDIM;

   FiniteElementCollection * fec = nullptr;
   fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec, dim,
                                                        fes_ordering);
   // FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
   cout << "Finite Element Collection: " << fec->Name() << endl;
   const real_t Ndof = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof <<
        endl;
   cout << "Number of elements: " << fespace->GetNE() << std::endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << std::endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

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

   LinearForm b(fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "Assembling RHS ... " << flush;
   b.Assemble();
   cout << "done." << endl;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.

   // Lame parameters
   Vector lambda(mesh.attributes.Max());
   lambda = 10.0;
   // lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 10.0;
   // mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   // Bilinear integrator
   ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
   if (patchAssembly)
   {
      ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
   }

   // Patch rule
   {
      if (ir_order == -1) { ir_order = 2*fec->GetOrder(); }
      cout << "Integration rule order: " << ir_order << endl;

      NURBSMeshRules * patchRule  = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
      // Loop over patches and set a different rule for each patch.
      for (int p=0; p < mesh.NURBSext->GetNP(); ++p)
      {
         Array<const KnotVector*> kv(dim);
         mesh.NURBSext->GetPatchKnotVectors(p, kv);

         std::vector<const IntegrationRule*> ir1D(dim);
         const IntegrationRule *ir = &IntRules.Get(Geometry::SEGMENT, ir_order);

         // Construct 1D integration rules by applying the rule ir to each knot span.
         for (int i=0; i<dim; ++i)
         {
            ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
         }

         patchRule->SetPatchRules1D(p, ir1D);
      }  // loop (p) over patches

      patchRule->Finalize(mesh);
      ei->SetNURBSPatchIntRule(patchRule);
   }

   StopWatch sw;
   sw.Start();

   // 10. Assemble and solve the linear system

   // Define and assemble bilinear form
   cout << "Assembling a ... " << flush;
   BilinearForm a(fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(ei);
   a.Assemble();
   cout << "done." << endl;

   // Define linear system
   cout << "Forming linear system ... " << flush;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace->GetTrueVSize() << ")" << endl;

   // Get the preconditioner
   CGSolver solver;
   if (preconditioner == 1)
   {
      cout << "Getting diagonal for Jacobi PC ... " << endl;
      OperatorJacobiSmoother M(a, ess_tdof_list);
      solver.SetPreconditioner(M);
   }
   else if (preconditioner == 2)
   {
      cout << "Getting LOR PC ... " << endl;
      MFEM_ABORT("Not implemented yet.");
   }

   sw.Stop();
   const real_t timeAssemble = sw.RealTime();
   sw.Clear();
   sw.Start();

   // Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;
   solver.SetMaxIter(1e5);
   solver.SetPrintLevel(1);
   solver.SetRelTol(sqrt(1e-8));
   solver.SetAbsTol(sqrt(1e-14));
   solver.SetOperator(*A);

   solver.Mult(B, X);

   // Apply operator once
   // A->AddMult(X, X);

   cout << "Done solving system." << endl;

   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Collect results and write to file
   const int Niter = solver.GetNumIterations();
   // const int Niter = 1;
   const int dof_per_sec_solve = Ndof * Niter / timeSolve;
   const int dof_per_sec_total = Ndof * Niter / timeTotal;
   cout << "Time to assemble: " << timeAssemble << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;
   cout << "Total time: " << timeTotal << " seconds" << endl;
   cout << "Dof/sec (solve): " << dof_per_sec_solve << endl;
   cout << "Dof/sec (total): " << dof_per_sec_total << endl;

   ofstream results_ofs("ex2_results.csv", ios_base::app);
   bool file_exists = results_ofs.tellp() != 0;
   // header
   if (!file_exists)
   {
      results_ofs << "int.patch, int.pa, "
                  << "problem.mesh, problem.refs, problem.degree_inc, problem.ndof, "
                  << "solver.iter, solver.absnorm, solver.relnorm, "
                  << "solution.linf, solution.l2, "
                  << "time.assemble, time.solve, time.total, "
                  << "dof_per_sec_solve, "
                  << "dof_per_sec_total" << endl;
   }

   results_ofs << patchAssembly << ", "
               << pa << ", "
               << mesh_file << ", "
               << ref_levels << ", "
               << nurbs_degree_increase << ", "
               << Ndof << ", "
               << Niter << ", "
               << solver.GetFinalNorm() << ", "
               << solver.GetFinalRelNorm() << ", "
               << x.Normlinf() << ", "
               << x.Norml2() << ", "
               << timeAssemble << ", "
               << timeSolve << ", "
               << timeTotal << ", "
               << dof_per_sec_solve << ", "
               << dof_per_sec_total << endl;

   results_ofs.close();

   // 12. Save the displaced mesh and the inverted solution
   {
      GridFunction *nodes = mesh.GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 13. Send the above data by socket to a GLVis server.
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

   // 14. Free the used memory.
   delete fespace;

   return 0;
}
