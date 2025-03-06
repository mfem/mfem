#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   int test_case = 1;
   int ref_levels = 0;
   int order = 1;
   int visport = 19916;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&test_case, "-c", "--case", "Test case");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);

   bool iga = false;
   bool pa = false;
   bool patcha = false;
   bool reducedint = false;
   int ir_order = -1;
   if (test_case == 1)
   {
      cout << "Test case 1: NURBS + PA" << endl;
      pa = true;
   }
   else if (test_case == 2)
   {
      cout << "Test case 2: NURBS + IGA + Patch + PA" << endl;
      iga = true;
      pa = true;
      patcha = true;
      // order -= 1; // nurbs degree increase (mesh is order 1)
   }
   else if (test_case == 3)
   {
      cout << "Test case 3: NURBS + IGA + Patch + PA + ReducedInt" << endl;
      iga = true;
      pa = true;
      patcha = true;
      reducedint = true;
      ir_order = 8;
   }
   else if (test_case == 4)
   {
      cout << "Test case 4: NURBS + IGA" << endl;
      iga = true;
   }
   else
   {
      cout << "Unknown test case: " << test_case << endl;
      return 1;
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   bool isNURBS = mesh.NURBSext;

   MFEM_VERIFY(isNURBS, "NURBS mesh required.");


   // 3. Increase the NURBS degree.
   if (iga && order > 0)
   {
      cout << "Increasing NURBS degree by " << order << endl;
      mesh.DegreeElevate(order);
   }

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection * fec = nullptr;
   fec = iga ? mesh.GetNodes()->OwnFEC() : new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
   cout << "Finite Element Collection: " << fec->Name() << endl;
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl;
   cout << "Number of elements: " << fespace->GetNE() << std::endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << std::endl;


   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      mesh.MarkExternalBoundaries(ess_bdr);
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   LinearForm b(fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(fespace);
   x = 0.0;

   BilinearForm a(fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (!pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   DiffusionIntegrator *di = new DiffusionIntegrator(one);
   if (patcha)
   {
      if (reducedint)
      {
         di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE_REDUCED);
      }
      else
      {
         di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      }
   }


   // Patch rule
   NURBSMeshRules *patchRule = nullptr;
   if (patcha)
   {
      if (ir_order==-1)
      {
         ir_order = 2*fec->GetOrder();
      }
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
      di->SetNURBSPatchIntRule(patchRule);
   }

   StopWatch sw;
   sw.Start();

   // 10. Assemble and solve the linear system

   // Define and assemble bilinear form
   a.AddDomainIntegrator(di);
   cout << "Assembling a ... " << flush;
   a.Assemble();
   cout << "done." << endl;

   // Define linear system
   cout << "Forming linear system ... " << flush;

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace->GetTrueVSize() << ")" << endl;

   sw.Stop();
   const real_t timeAssemble = sw.RealTime();
   sw.Clear();
   sw.Start();

   // Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;

   CGSolver solver;
   // OperatorJacobiSmoother M(a, ess_tdof_list);
   // solver.SetPreconditioner(M);
   solver.SetMaxIter(4e2);
   solver.SetPrintLevel(1);
   solver.SetRelTol(sqrt(1e-12));
   solver.SetAbsTol(0.0);
   solver.SetOperator(*A);
   solver.Mult(B, X);

   cout << "Done solving system." << endl;
   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   cout << "Time to assemble: " << timeAssemble << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Append timings and problem info to file
   ofstream results_ofs("ex1_comparison.csv", ios_base::app);
   bool file_exists = results_ofs.tellp() != 0;
   // header
   if (!file_exists)
   {
      results_ofs << "int.test_case, "
                  << "problem.mesh, problem.refs, problem.ndof, "
                  << "solver.iter, solver.absnorm, solver.relnorm, solver.converged, "
                  << "solution.linf, solution.l2, "
                  << "time.assemble, time.solve, time.total, "
                  << "dof_per_sec_solve" << endl;
   }

   results_ofs << test_case << ", "
               << mesh_file << ", "
               << ref_levels << ", "
               << fespace->GetTrueVSize() << ", "
               << solver.GetNumIterations() << ", "
               << solver.GetFinalNorm() << ", "
               << solver.GetFinalRelNorm() << ", "
               << solver.GetConverged() << ", "
               << x.Normlinf() << ", "
               << x.Norml2() << ", "
               << timeAssemble << ", "
               << timeSolve << ", "
               << (timeAssemble + timeSolve) << ", "
               << fespace->GetTrueVSize() * solver.GetNumIterations() / timeSolve << endl;

   results_ofs.close();


   // Save solution
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}
