//               MFEM Example 1 - NURBS with patch-wise assembly
//
// Compile with: make nurbs_patch_ex1
//
// Sample runs:  nurbs_patch_ex1 -incdeg 3 -ref 2 -iro 8 -patcha
//               nurbs_patch_ex1 -incdeg 3 -ref 2 -iro 8 -patcha -pa
//               nurbs_patch_ex1 -incdeg 3 -ref 2 -iro 8 -patcha -fint
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               This example is a specialization of ex1 which demonstrates
//               patch-wise matrix assembly and partial assembly on NURBS
//               meshes. There is the option to compare run times of patch
//               and element assembly, as well as relative error computation.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void AssembleAndSolve(LinearForm & b, BilinearFormIntegrator * bfi,
                      Array<int> const& ess_tdof_list, const bool pa,
                      const bool algebraic_ceed, GridFunction & x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   int order = -1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   bool patchAssembly = false;
   bool reducedIntegration = true;
   bool compareToElementWise = true;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int ref_levels = 0;
   int ir_order = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&reducedIntegration, "-rint", "--reduced-integration", "-fint",
                  "--full-integration", "Enable reduced integration rules.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&ir_order, "-iro", "--integration-order",
                  "Order of integration rule.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&compareToElementWise, "-cew", "--compare-element",
                  "-no-compare", "-no-compare-element",
                  "Compute element-wise solution for comparison");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   MFEM_VERIFY(!(pa && !patchAssembly), "Patch assembly must be used with -pa");

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. For this NURBS patch example,
   //    only 3D hexahedral meshes are currently supported. The NURBS degree is
   //    optionally increased.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (nurbs_degree_increase > 0) { mesh.DegreeElevate(nurbs_degree_increase); }

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define an isoparametric/isogeometric finite element space on the mesh.
   FiniteElementCollection *fec = nullptr;
   bool delete_fec;
   if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      MFEM_ABORT("Mesh must have nodes");
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   DiffusionIntegrator *di = new DiffusionIntegrator(one);

   if (patchAssembly && reducedIntegration && !pa)
   {
#ifdef MFEM_USE_SINGLE
      cout << "Reduced integration is not supported in single precision.\n";
      return MFEM_SKIP_RETURN_VALUE;
#endif

      di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE_REDUCED);
   }
   else if (patchAssembly)
   {
      di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
   }

   NURBSMeshRules *patchRule = nullptr;
   if (order < 0)
   {
      if (ir_order == -1) { ir_order = 2*fec->GetOrder(); }
      cout << "Using ir_order " << ir_order << endl;

      patchRule = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
      // Loop over patches and set a different rule for each patch.
      for (int p=0; p<mesh.NURBSext->GetNP(); ++p)
      {
         Array<const KnotVector*> kv(dim);
         mesh.NURBSext->GetPatchKnotVectors(p, kv);

         std::vector<const IntegrationRule*> ir1D(dim);
         const IntegrationRule *ir = &IntRules.Get(Geometry::SEGMENT, ir_order);

         // Construct 1D integration rules by applying the rule ir to each
         // knot span.
         for (int i=0; i<dim; ++i)
         {
            ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
         }

         patchRule->SetPatchRules1D(p, ir1D);
      }  // loop (p) over patches

      patchRule->Finalize(mesh);
      di->SetNURBSPatchIntRule(patchRule);
   }

   // 10. Assemble and solve the linear system
   cout << "Assembling system patch-wise and solving" << endl;
   AssembleAndSolve(b, di, ess_tdof_list, pa, algebraic_ceed, x);

   delete patchRule;

   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   // 13. Optionally assemble element-wise and solve the linear system, to
   //     compare timings and compute relative error.
   if (compareToElementWise)
   {
      Vector x_pw, x_ew;
      x.GetTrueDofs(x_pw);

      cout << "Assembling system element-wise and solving" << endl;
      DiffusionIntegrator *d = new DiffusionIntegrator(one);
      // Element-wise partial assembly is not supported on NURBS meshes, so we
      // pass pa = false here.
      AssembleAndSolve(b, d, ess_tdof_list, false, algebraic_ceed, x);

      x.GetTrueDofs(x_ew);

      const real_t solNorm = x_ew.Norml2();
      x_ew -= x_pw;

      cout << "Element-wise solution norm " << solNorm << endl;
      cout << "Relative error of patch-wise solution "
           << x_ew.Norml2() / solNorm << endl;
   }

   // 14. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

// This function deletes bfi when the BilinearForm goes out of scope.
void AssembleAndSolve(LinearForm & b, BilinearFormIntegrator * bfi,
                      Array<int> const& ess_tdof_list, const bool pa,
                      const bool algebraic_ceed, GridFunction & x)
{
   FiniteElementSpace *fespace = b.FESpace();
   BilinearForm a(fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   a.AddDomainIntegrator(bfi);  // Takes ownership of bfi

   StopWatch sw;
   sw.Start();

   // Assemble the bilinear form and the corresponding linear system, applying
   // any necessary transformations such as: eliminating boundary conditions,
   // applying conforming constraints for non-conforming AMR, etc.
   a.Assemble();

   sw.Stop();

   const real_t timeAssemble = sw.RealTime();

   sw.Clear();
   sw.Start();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   sw.Stop();

   const real_t timeFormLinearSystem = sw.RealTime();

   cout << "Timing for Assemble: " << timeAssemble << " seconds" << endl;
   cout << "Timing for FormLinearSystem: " << timeFormLinearSystem << " seconds"
        << endl;
   cout << "Timing for entire setup: " << timeAssemble + timeFormLinearSystem
        << " seconds" << endl;

   sw.Clear();
   sw.Start();

   // Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-20, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(*fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-20, 0.0);
      }
   }

   sw.Stop();
   cout << "Timing for solve " << sw.RealTime() << endl;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
}
