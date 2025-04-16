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

enum class SplineIntegrationRule { FULL_GAUSSIAN, REDUCED_GAUSSIAN, };
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi);

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   // const char *mesh_file = "../../../miniapps/nurbs/meshes/beam-hex-nurbs-onepatch.mesh";
   bool pa = false;
   bool patchAssembly = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   bool reduced_integration = 1;
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
   args.AddOption(&reduced_integration, "-ri", "--reduced-integration",
                  "-fi", "--full-integration", "Use reduced integration.");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - diagonal, 2 - LOR AMG");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");
   if (preconditioner == 2)
   {
      MFEM_VERIFY(nurbs_degree_increase > 0,
                  "LOR preconditioner requires degree increase");
   }

   // Integration rule for the 1d bases defined on each knotvector
   // Reduced Gaussian rule as in Zou 2022 - equation 14
   auto splineRule = reduced_integration
                     ? SplineIntegrationRule::REDUCED_GAUSSIAN
                     : SplineIntegrationRule::FULL_GAUSSIAN;

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const bool isNURBS = mesh.NURBSext;

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
      // mesh.UniformRefinement();
      mesh.NURBSUniformRefinement();
   }

   // 5. Define a finite element space on the mesh.
   // Node ordering is important - right now, only works with byVDIM
   FiniteElementCollection * fec = mesh.GetNodes()->OwnFEC();
   cout << "fec order = " << fec->GetOrder() << endl;

   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, mesh.NURBSext, fec,
                                                        dim,
                                                        Ordering::byVDIM);
   cout << "Finite Element Collection: " << fec->Name() << endl;
   const int Ndof = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof << endl;
   cout << "Number of elements: " << fespace->GetNE() << endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;

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
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.)

   // Lame parameters
   Vector lambda(mesh.attributes.Max());
   Vector mu(mesh.attributes.Max());
   lambda = 1.0; lambda(0) = lambda(1)*50;
   mu = 1.0; mu(0) = mu(1)*50;
   // Constant coefficients
   // lambda = 10.0; mu = 10.0;

   PWConstCoefficient lambda_func(lambda);
   PWConstCoefficient mu_func(mu);

   // Bilinear integrator
   ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
   if (patchAssembly)
   {
      ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      // Set the patch integration rules
      SetPatchIntegrationRules(mesh, splineRule, ei);
   }

   // 10. Assembly
   StopWatch sw;
   sw.Start();

   // Define and assemble bilinear form
   cout << "Assembling a ... " << flush;
   BilinearForm a(fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(ei);
   a.Assemble();
   cout << "done." << endl;

   // Form linear system
   cout << "Forming linear system ... " << flush;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace->GetTrueVSize() << ")" << endl;

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
   // Jacobi
   else if (preconditioner == 1)
   {
      cout << "Setting up preconditioner (Jacobi) ... " << endl;
      OperatorJacobiSmoother *P = new OperatorJacobiSmoother(a, ess_tdof_list);
      solver.SetPreconditioner(*P);
   }
   // LOR AMG
   else if (preconditioner == 2)
   {
      cout << "Setting up preconditioner (LOR AMG) ... " << endl;
      // Read in mesh again, but don't increase order; refine so that Ndof is equivalent
      Mesh lo_mesh(mesh_file, 1, 1);
      lo_mesh.NURBSUniformRefinement(pow(2,ref_levels) + nurbs_degree_increase);

      FiniteElementCollection * lo_fec = lo_mesh.GetNodes()->OwnFEC();
      FiniteElementSpace *lo_fespace = new FiniteElementSpace(&lo_mesh, lo_fec, dim,
                                                              Ordering::byVDIM);
      const int lo_Ndof = lo_fespace->GetTrueVSize();
      MFEM_VERIFY(Ndof == lo_Ndof, "Low-order problem requires same Ndof");

      // We can reuse variables that are defined by mesh attribute: ess_bdr, f, lambda, mu
      Array<int> lo_ess_tdof_list;
      lo_fespace->GetEssentialTrueDofs(ess_bdr, lo_ess_tdof_list);

      LinearForm lo_b(lo_fespace);
      lo_b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
      lo_b.Assemble();

      GridFunction lo_x(lo_fespace);
      lo_x = 0.0;

      ElasticityIntegrator *lo_ei = new ElasticityIntegrator(lambda_func, mu_func);
      // Set up problem
      BilinearForm lo_a(lo_fespace);
      lo_a.AddDomainIntegrator(lo_ei);
      lo_a.Assemble();

      // Define linear system
      OperatorPtr lo_A;
      Vector lo_B, lo_X;
      lo_a.FormLinearSystem(lo_ess_tdof_list, lo_x, lo_b, lo_A, lo_X, lo_B);

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
      // Make sure we set byVDIM ordering (second argument == false)
      lo_P->SetSystemsOptions(dim, false);

      // Use low-order AMG as preconditioner for high-order problem
      solver.SetPreconditioner(*lo_P);
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
   solver.SetMaxIter(1e5);
   solver.SetPrintLevel(1);
   // These tolerances end up getting squared
   solver.SetRelTol(sqrt(1e-6));
   solver.SetAbsTol(sqrt(1e-14));

   solver.Mult(B, X);

   // Apply operator once
   // A->AddMult(X, X);

   cout << "Done solving system." << endl;

   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Collect results and write to file
   const int Niter = solver.GetNumIterations();
   const int dof_per_sec_solve = Ndof * Niter / timeSolve;
   const int dof_per_sec_total = Ndof * Niter / timeTotal;
   cout << "Time to assemble: " << timeAssemble << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;
   cout << "Total time: " << timeTotal << " seconds" << endl;
   cout << "Dof/sec (solve): " << dof_per_sec_solve << endl;
   cout << "Dof/sec (total): " << dof_per_sec_total << endl;

   ofstream results_ofs("ex2_results.csv", ios_base::app);
   // If file does not exist, write the header
   if (results_ofs.tellp() == 0)
   {
      results_ofs << "patcha, pa, pc, ri, "           // settings
                  << "mesh, refs, deg_inc, ndof, "    // mesh
                  << "niter, absnorm, relnorm, "      // solver
                  << "linf, l2, "                     // solution
                  << "t_assemble, t_solve, t_total, " // timing
                  << "dof/s_solve, dof/s_total"       // benchmarking
                  << endl;
   }

   results_ofs << patchAssembly << ", "               // settings
               << pa << ", "
               << preconditioner << ", "
               << reduced_integration << ", "
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
               << timeSolve << ", "
               << timeTotal << ", "
               << dof_per_sec_solve << ", "           // benchmarking
               << dof_per_sec_total << endl;

   results_ofs.close();

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

   // Compare with another saved solution
   // {
   //    ifstream sol_ifs("sol_ri.gf");
   //    GridFunction x2(&mesh, sol_ifs);
   //    GridFunctionCoefficient x2_gfc(&x2);
   //    cout << "L2 error w.r.t sol_ri.gf = " << x.ComputeL2Error(x2_gfc) << endl;
   // }

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

   return 0;
}

// For each patch, sets
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi)
{
   const int dim = mesh.Dimension();
   NURBSMeshRules * patchRules  = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
   // Loop over patches and set a different rule for each patch.
   for (int p=0; p < mesh.NURBSext->GetNP(); ++p)
   {
      Array<const KnotVector*> kv(dim);
      mesh.NURBSext->GetPatchKnotVectors(p, kv);

      std::vector<const IntegrationRule*> ir1D(dim);
      // Construct 1D integration rules by applying the rule ir to each knot span.
      for (int i=0; i<dim; ++i)
      {
         if ( splineRule == SplineIntegrationRule::FULL_GAUSSIAN )
         {
            const int order = kv[i]->GetOrder();
            const IntegrationRule ir = IntRules.Get(Geometry::SEGMENT, 2*order);
            ir1D[i] = IntegrationRule::ApplyToKnotIntervals(ir,*kv[i]);
         }
         else if ( splineRule == SplineIntegrationRule::REDUCED_GAUSSIAN )
         {
            ir1D[i] = IntegrationRule::GetIsogeometricReducedGaussianRule(*kv[i]);
         }
         else
         {
            MFEM_ABORT("Unknown PatchIntegrationRule1D")
         }
      }

      patchRules->SetPatchRules1D(p, ir1D);
   }  // loop (p) over patches

   patchRules->Finalize(mesh);
   bfi->SetNURBSPatchIntRule(patchRules);
}