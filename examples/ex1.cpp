//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void ComputeStandardSolution(GridFunction & x, bool pa);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   bool patchAssembly = false;
   int ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   //mesh.DegreeElevate(1);
   //mesh.Save("beam4.mesh");

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   /*
   {
      int ref_levels =
         (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   */

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   //mesh.Save("beam4ref.mesh");
   //return 0;

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
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
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   DiffusionIntegrator *di = new DiffusionIntegrator(one, nullptr, patchAssembly);
   NURBSPatchRule *patchRule = nullptr;
   if (order < 0)
   {
      const int ir_order = 2*fec->GetOrder();
      cout << "Using ir_order " << ir_order << endl;

      bool uniformRule = false;
      if (uniformRule)
      {
         IntegrationRule *ir = const_cast<IntegrationRule*> (&IntRules.Get(
                                                                Geometry::SEGMENT, ir_order));
         patchRule = new NURBSPatchRule(ir, ir, dim == 3 ? ir : nullptr);
      }
      else
      {
         patchRule = new NURBSPatchRule(mesh.NURBSext->GetNP(), dim);
         // Loop over patches and set a different rule for each patch.
         for (int p=0; p<mesh.NURBSext->GetNP(); ++p)
         {
            Array<const KnotVector*> kv(dim);
            mesh.NURBSext->GetPatchKnotVectors(p, kv);

            std::vector<const IntegrationRule*> ir1D(dim);
            const IntegrationRule *ir = &IntRules.Get(Geometry::SEGMENT, ir_order);

            // NOTE: I tried using a fine rule for the entire patch, but it results
            // in garbage. The only thing that works so far is an ordinary rule applied
            // to each knot span.
            //const IntegrationRule *irFine = &IntRules.Get(Geometry::SEGMENT, 10);

            for (int i=0; i<dim; ++i)
            {
               ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);

               /*
               for (int j=0; j<kv[i]->Size(); ++j)
               cout << "kv[" << i << "][" << j << "] " << (*kv[i])[j] << endl;
                    */
            }

            /*
            // TODO: scale ir knots by kv physical length
            IntegrationRule *irFineScaled = new IntegrationRule(irFine->Size());
            const double scale = 4.0;

            for (int i=0; i<irFine->Size(); ++i)
              {
            (*irFineScaled)[i].Set(scale * (*irFine)[i].x, 0.0, 0.0, (*irFine)[i].weight);
            //(*irFineScaled)[i].Set1w(scale * (*irFine)[i].x, (*irFine)[i].weight);
            cout << i << ": irFine " << (*irFine)[i].x << " -> " << (*irFineScaled)[i].x << endl;
                 }

                    for (int i=0; i<dim; ++i)
                    {
                       ir1D[i] = ir;
                    }
               ir1D[0] = irFineScaled;
               */

            /*
                 if (dim == 3)
                 {
              patchRule->SetPatchRule(p, new IntegrationRule(*ir1D[0], *ir1D[1], *ir1D[2]));
                 }
                 else
                 {
                    patchRule->SetPatchRule(p, new IntegrationRule(*ir1D[0], *ir1D[1]));
                 }
            */

            patchRule->SetPatchRules1D(p, ir1D);
         }  // loop (p) over patches

         patchRule->SetPointToElement(mesh);
      }

      di->SetNURBSPatchIntRule(patchRule);
   }

   a.AddDomainIntegrator(di);

   StopWatch ta, ts, tfls;
   ta.Start();

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();
   delete patchRule;

   tfls.Start();
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   tfls.Stop();
   cout << "Timing for FormLinearSystem " << tfls.RealTime() << endl;

   ta.Stop();
   cout << "Timing for assembly " << ta.RealTime() << endl;

   cout << "Size of linear system: " << A->Height() << endl;

   /* Debugging
   SparseMatrix *Asp = (SparseMatrix*) A.Ptr();
   ofstream outs("Apatch.mat");
   Asp->PrintMatlab(outs);
   */

   ts.Start();

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
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
      if (UsesTensorBasis(fespace))
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
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   ts.Stop();
   cout << "Timing for patch solve " << ts.RealTime() << endl;

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 12b. Solve with a standard element-wise IntegrationRule and compare solutions.
   GridFunction xe(&fespace);
   ComputeStandardSolution(xe, pa);

   const double solNorm = xe.Norml2();
   xe -= x;
   cout << "Solution norm " << solNorm << endl;
   cout << "Relative error of solution " << xe.Norml2() / solNorm << endl;

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

void ComputeStandardSolution(GridFunction & x, bool pa)
{
   FiniteElementSpace & fespace = *x.FESpace();
   Mesh & mesh = *fespace.GetMesh();

   const bool static_cond = false;
   const bool algebraic_ceed = false;

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
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   DiffusionIntegrator *di = new DiffusionIntegrator(one);
   const int ir_order = 2*fespace.FEColl()->GetOrder();

   IntegrationRule *ir = const_cast<IntegrationRule*> (&IntRules.Get(
                                                          Geometry::SEGMENT, ir_order));
   di->SetIntRule(new IntegrationRule(*ir, *ir, *ir));
   a.AddDomainIntegrator(di);

   StopWatch ta, ts;
   ta.Start();

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   ta.Stop();
   cout << "Timing for element assembly " << ta.RealTime() << endl;

   cout << "Size of linear system: " << A->Height() << endl;

   /* Debugging
   SparseMatrix *Asp = (SparseMatrix*) A.Ptr();
   ofstream outs("Ast.mat");
   Asp->PrintMatlab(outs);
   */

   ts.Start();
   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
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
      if (UsesTensorBasis(fespace))
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
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   ts.Stop();
   cout << "Timing for element solve " << ts.RealTime() << endl;

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
}
