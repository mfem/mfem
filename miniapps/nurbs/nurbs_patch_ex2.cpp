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

void AssembleAndSolve(LinearForm & b, BilinearFormIntegrator * bfi,
                      Array<int> const& ess_tdof_list, const bool pa,
                      const bool algebraic_ceed, GridFunction & x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   bool pa = false;
   bool algebraic_ceed = false;
   bool patchAssembly = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int ir_order = -1;
   int visport = 19916;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   // args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
   //                "--no-static-condensation", "Enable static condensation.");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&ir_order, "-iro", "--integration-order",
                  "Order of integration rule.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   MFEM_VERIFY(!(pa && !patchAssembly), "Patch assembly must be used with -pa");

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   MFEM_VERIFY(mesh->NURBSext, "Mesh must be a NURBS mesh");
   MFEM_VERIFY(mesh->GetNodes(), "Mesh must have nodes");

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Optionally, increase the NURBS degree.
   if (nurbs_degree_increase > 0) { mesh->DegreeElevate(nurbs_degree_increase); }

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define an isogeometric finite element space on the mesh.
   FiniteElementCollection *fec = mesh->GetNodes()->OwnFEC();
   cout << "Using isoparametric FEs: " << fec->Name() << endl;

   FiniteElementSpace *fespace = mesh->GetNodes()->FESpace();
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.

   // Lame parameters
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   // Bilinear integrator
   ElasticityIntegrator *ei = new ElasticityIntegrator(lambda_func, mu_func);
   if (patchAssembly)
   {
      ei->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
   }


   NURBSMeshRules *patchRule = nullptr;
   if (ir_order == -1) { ir_order = 2*fec->GetOrder(); }
   cout << "Using ir_order " << ir_order << endl;

   patchRule = new NURBSMeshRules(mesh->NURBSext->GetNP(), dim);
   // Loop over patches and set a different rule for each patch.
   for (int p=0; p<mesh->NURBSext->GetNP(); ++p)
   {
      Array<const KnotVector*> kv(dim);
      mesh->NURBSext->GetPatchKnotVectors(p, kv);

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

   patchRule->Finalize(*mesh);
   ei->SetNURBSPatchIntRule(patchRule);


   // 10. Assemble and solve the linear system
   cout << "Assembling system and solving" << endl;
   AssembleAndSolve(*b, ei, ess_tdof_list, pa, algebraic_ceed, x);
   delete patchRule;

   // 14. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete b;
   delete mesh;

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

   // if (static_cond) { a->EnableStaticCondensation(); }
   // Assemble the bilinear form
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // Solve the linear system A X = B.
   if (!pa)
   {
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-8, 0.0);
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

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
}