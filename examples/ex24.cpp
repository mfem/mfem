//                               MFEM Example 24
//
// Compile with: make ex24
//
// Sample runs:  ex24 -m ../data/star.mesh
//               ex24 -m ../data/square-disc.mesh -o 2
//               ex24 -m ../data/beam-tet.mesh
//               ex24 -m ../data/beam-hex.mesh -o 2 -pa
//               ex24 -m ../data/escher.mesh
//               ex24 -m ../data/escher.mesh -o 2
//               ex24 -m ../data/fichera.mesh
//               ex24 -m ../data/fichera-q2.vtk
//               ex24 -m ../data/fichera-q3.mesh
//               ex24 -m ../data/square-disc-nurbs.mesh
//               ex24 -m ../data/beam-hex-nurbs.mesh
//               ex24 -m ../data/amr-quad.mesh -o 2
//               ex24 -m ../data/amr-hex.mesh
//
// Device sample runs:
//               ex24 -m ../data/star.mesh -pa -d cuda
//               ex24 -m ../data/star.mesh -pa -d raja-cuda
//               ex24 -m ../data/star.mesh -pa -d raja-omp
//               ex24 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code illustrates usage of mixed finite element
//               spaces. Using two different approaches, we project a gradient
//               of a function in H^1 to H(curl). Other spaces and example
//               computations are to be added in the future.
//
//               We recommend viewing examples 1 and 3 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double p_exact(const Vector &x);
void gradp_exact(const Vector &, Vector &);

int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   mesh->ReorientTetMesh();

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementCollection *H1fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   FiniteElementSpace *H1fespace = new FiniteElementSpace(mesh, H1fec);

   int size = fespace->GetTrueVSize();
   int H1size = H1fespace->GetTrueVSize();
   cout << "Number of Nedelec finite element unknowns: " << size << endl;
   cout << "Number of H1 finite element unknowns: " << H1size << endl;

   // 6. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   FunctionCoefficient p_coef(p_exact);
   GridFunction p(H1fespace);
   p.ProjectCoefficient(p_coef);
   p.SetTrueVector();
   p.SetFromTrueVector();

   VectorFunctionCoefficient gradp_coef(sdim, gradp_exact);

   // 7. Set up the bilinear forms.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   MixedBilinearForm *a_NDH1 = new MixedBilinearForm(H1fespace, fespace);
   if (pa)
   {
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a_NDH1->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   // First approach: L2 projection
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
   a_NDH1->AddDomainIntegrator(new MixedVectorGradientIntegrator(*muinv));

   // 8. Assemble the parallel bilinear form and the corresponding linear
   //    system, applying any necessary transformations such as: parallel
   //    assembly, eliminating boundary conditions, applying conforming
   //    constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }

   a->Assemble();
   if (!pa) { a->Finalize(); }

   a_NDH1->Assemble();
   if (!pa) { a_NDH1->Finalize(); }

   if (pa)
   {
      a_NDH1->Mult(p, x);
   }
   else
   {
      SparseMatrix& NDH1 = a_NDH1->SpMat();
      NDH1.Mult(p, x);
   }

   // 9. Define and apply a PCG solver for Ax = b with Jacobi preconditioner.
   {
      GridFunction rhs(fespace);
      rhs = x;
      x = 0.0;

      CGSolver cg;
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      if (pa)
      {
         Array<int> ess_tdof_list; // empty
         OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

         cg.SetOperator(*a);
         cg.SetPreconditioner(Jacobi);
         cg.Mult(rhs, x);
      }
      else
      {
         SparseMatrix& Amat = a->SpMat();
         DSmoother Jacobi(Amat);

         cg.SetOperator(Amat);
         cg.SetPreconditioner(Jacobi);
         cg.Mult(rhs, x);
      }
   }

   // 10. Second approach: compute the same solution by applying
   //     GradientInterpolator in H(curl).
   DiscreteLinearOperator grad(H1fespace, fespace);
   grad.AddDomainInterpolator(new GradientInterpolator());
   grad.Assemble();

   GridFunction gradp(fespace);
   grad.Mult(p, gradp);

   // 11. Compute the projection of the exact grad p.
   GridFunction exact_gradp(fespace);
   exact_gradp.ProjectCoefficient(gradp_coef);
   exact_gradp.SetTrueVector();
   exact_gradp.SetFromTrueVector();

   // 12. Compute and print the L^2 norm of the error.
   {
      double errSol = x.ComputeL2Error(gradp_coef);
      double errInterp = gradp.ComputeL2Error(gradp_coef);
      double errProj = exact_gradp.ComputeL2Error(gradp_coef);

      cout << "\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): "
           "|| E_h - grad p ||_{L^2} = " << errSol << '\n' << endl;
      cout << " Gradient interpolant E_h = grad p_h in H(curl): || E_h - grad p"
           "||_{L^2} = " << errInterp << '\n' << endl;
      cout << " Projection E_h of exact grad p in H(curl): || E_h - grad p "
           "||_{L^2} = " << errProj << '\n' << endl;
   }

   // 13. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
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
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete a_NDH1;
   delete sigma;
   delete muinv;
   delete fespace;
   delete H1fespace;
   delete fec;
   delete H1fec;
   delete mesh;

   return 0;
}

double p_exact(const Vector &x)
{
   if (dim == 3)
   {
      return sin(x(0)) * sin(x(1)) * sin(x(2));
   }
   else if (dim == 2)
   {
      return sin(x(0)) * sin(x(1));
   }

   return 0.0;
}

void gradp_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = cos(x(0)) * sin(x(1)) * sin(x(2));
      f(1) = sin(x(0)) * cos(x(1)) * sin(x(2));
      f(2) = sin(x(0)) * sin(x(1)) * cos(x(2));
   }
   else
   {
      f(0) = cos(x(0)) * sin(x(1));
      f(1) = sin(x(0)) * cos(x(1));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
