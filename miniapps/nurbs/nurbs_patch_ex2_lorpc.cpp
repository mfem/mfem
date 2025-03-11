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

#include "fem/fe_coll.hpp"
#include "fem/gridfunc.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



struct Problem
{
   FiniteElementCollection * fec;
   FiniteElementSpace * fespace;
   Array<int> ess_tdof_list;
   BilinearForm * a;
   LinearForm * b;
   GridFunction * x;
   int Ndof;
   OperatorPtr A;
   Vector B, X;
   real_t timeAssemble;

   Problem(Mesh &mesh,
         bool pa = false,
         bool patchAssembly = false,
         int ref_levels = 0,
         int nurbs_degree_increase = 0, // Elevate the NURBS mesh degree by this
         int ir_order = -1)
         // bool reorder_space = false)
   {
      // 1) Mesh
      int dim = mesh.Dimension();
      bool isNURBS = mesh.NURBSext;

      // 1a) Verify mesh is valid for this problem
      MFEM_VERIFY(!(isNURBS && !mesh.GetNodes()), "NURBS mesh must have nodes");
      if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
      {
         cout << "\nInput mesh should have at least two materials and "
            << "two boundary attributes! (See schematic in ex2.cpp)\n"
            << endl;
      }
      // 1b) Optionally, increase the NURBS degree.
      if (isNURBS && nurbs_degree_increase>0)
      {
         mesh.DegreeElevate(nurbs_degree_increase);
      }

      // 1c) Refine the mesh to increase the resolution.
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }

      // 2) Define a finite element space on the mesh.
      // Node ordering is important - right now, only works with byVDIM
      // const Ordering::Type fes_ordering =
      //    reorder_space ? Ordering::byNODES : Ordering::byVDIM;

      fec = isNURBS ? mesh.GetNodes()->OwnFEC() : new H1_FECollection(1, dim);
      fespace = new FiniteElementSpace(&mesh, fec, dim, Ordering::byVDIM);// fes_ordering);
      Ndof = fespace->GetTrueVSize();

      // 2a) Print some info
      cout << "Finite Element Collection: " << fec->Name() << endl;
      cout << "Number of finite element unknowns: " << Ndof << endl;
      cout << "Number of elements: " << fespace->GetNE() << std::endl;
      if (isNURBS)
      {
         cout << "Number of patches: " << mesh.NURBSext->GetNP() << std::endl;
      }

      // 3) Determine the list of true (i.e. conforming) essential boundary dofs.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 4) Set up the linear form b(.)

      // 4a) Define pull force only on the last dimension
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

      // 4b) Define the BoundaryIntegrator
      b = new LinearForm(fespace);
      b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));

      // 5) Define the solution vector x as a finite element grid function
      x = new GridFunction(fespace);
      // Initialize x with 0, which satisfies the essential BC.
      *x = 0.0;

      // 6) Set up the bilinear form

      // 6a) Define Lame parameters
      Vector lambda(mesh.attributes.Max());
      lambda = 1.0;
      // lambda(0) = lambda(1)*50;
      PWConstCoefficient lambda_func(lambda);
      Vector mu(mesh.attributes.Max());
      mu = 1.0;
      // mu(0) = mu(1)*50;
      PWConstCoefficient mu_func(mu);

      // 6b) Define the Elasticity Integrator
      ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
      if (patchAssembly)
      {
         ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      }

      // 6c) Set up patch rule
      NURBSMeshRules *patchRule = nullptr;
      if (isNURBS)
      {
         if (ir_order == -1) { ir_order = 2*fec->GetOrder(); }
         cout << "Integration rule order: " << ir_order << endl;

         patchRule = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
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

      // 6d) Define the bilinear form. Set assembly level and add the integrator
      a = new BilinearForm(fespace);
      if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      a->AddDomainIntegrator(ei);

      // 7) Assemble linear + bilinear forms

      // Time the assembly
      StopWatch sw;
      sw.Start();

      cout << "Assembling RHS ... " << flush;
      b->Assemble();
      cout << "done." << endl;
      cout << "Assembling a ... " << flush;
      a->Assemble();
      cout << "done." << endl;

      // 8) Form the linear system
      cout << "Forming linear system ... " << flush;
      a->FormLinearSystem(ess_tdof_list, *x, *b, A, X, B);

      sw.Stop();
      timeAssemble = sw.RealTime();
      sw.Clear();

   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   bool pa = false;
   bool patchAssembly = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int ir_order = -1;
   // bool reorder_space = false;
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
   // args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
   //                "Default ordering is byVDIM");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");

   // 2. Read the mesh from the given mesh file.
   cout << "Setting up problem..." << endl;
   // Mesh mesh(mesh_file, 1, 1);
   Mesh mesh("nurbs-meshes/high_order_p4_i8.mesh", 1, 1);
   bool isNURBS = mesh.NURBSext;
   Problem hop(mesh, pa, patchAssembly, ref_levels,
               nurbs_degree_increase, ir_order);

   cout << "Setting up low-order problem..." << endl;
   Mesh lop_mesh("nurbs-meshes/low_order_p4_i8.mesh", 1, 1);
   Problem lop(lop_mesh, pa, patchAssembly, 0,
               0, ir_order);

   StopWatch sw;
   sw.Start();

   // Diagonal preconditioner
   cout << "Getting diagonal ... " << endl;
   OperatorJacobiSmoother M(*hop.a, hop.ess_tdof_list);

   // Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;
   // GSSmoother M(A);
   // GSSmoother M((SparseMatrix&)(*A));
   // PCG(*A, M, B, X, 1, 200, 1e-20, 0.0);

   // PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
   // CG(*A, B, X, 1, 10, 1e-8, 1e-10);

   // GSSmoother M(A);
   // PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
   // GMRESSolver solver;
   // solver.SetMaxIter(1000);
   CGSolver solver;
   solver.SetMaxIter(1e5);

   solver.SetPrintLevel(1);
   solver.SetRelTol(sqrt(1e-8));
   solver.SetAbsTol(sqrt(1e-14));
   solver.SetOperator(*hop.A);
   // solver.SetPreconditioner(M);

   solver.Mult(hop.B, hop.X);

   cout << "Done solving system." << endl;

   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = hop.timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   hop.a->RecoverFEMSolution(hop.X, *hop.b, *hop.x);

   // Collect results and write to file
   const int Niter = solver.GetNumIterations();
   const int dof_per_sec_solve = hop.Ndof * Niter / timeSolve;
   const int dof_per_sec_total = hop.Ndof * Niter / timeTotal;
   cout << "Time to assemble: " << hop.timeAssemble << " seconds" << endl;
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
                  << "solver.iter, solver.absnorm, solver.relnorm, solver.converged, "
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
               << hop.Ndof << ", "
               << Niter << ", "
               << solver.GetFinalNorm() << ", "
               << solver.GetFinalRelNorm() << ", "
               << solver.GetConverged() << ", "
               << hop.x->Normlinf() << ", "
               << hop.x->Norml2() << ", "
               << hop.timeAssemble << ", "
               << timeSolve << ", "
               << timeTotal << ", "
               << dof_per_sec_solve << ", "
               << dof_per_sec_total << endl;

   results_ofs.close();

   if (!isNURBS)
   {
      mesh.SetNodalFESpace(hop.fespace);
   }

   // Save the displaced mesh and the inverted solution
   cout << "Saving solution" << endl;
   GridFunction x(*hop.x);
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

   // Send the above data by socket to a GLVis server.
   cout << "Sending data to GLVis" << endl;
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

   cout << "done " << endl;

   return 0;
}
