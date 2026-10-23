//                                MFEM Example 24
//
// Compile with: make ex24
//
// Sample runs:  ex24 -m ../data/star.mesh
//               ex24 -m ../data/square-disc.mesh -o 2
//               ex24 -m ../data/beam-tet.mesh
//               ex24 -m ../data/beam-hex.mesh -o 2 -pa
//               ex24 -m ../data/beam-hex.mesh -o 2 -pa -p 1
//               ex24 -m ../data/beam-hex.mesh -o 2 -pa -p 2
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
//               spaces, with three variants:
//
//               1) (grad p, u) for p in H^1 tested against u in H(curl)
//               2) (curl v, u) for v in H(curl) tested against u in H(div), 3D
//               3) (div v, q) for v in H(div) tested against q in L_2
//
//               Using different approaches, we project the gradient, curl, or
//               divergence to the appropriate space.
//
//               We recommend viewing examples 1, 3, and 5 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t p_exact(const Vector &x);
void gradp_exact(const Vector &, Vector &);
real_t div_gradp_exact(const Vector &x);
void v_exact(const Vector &x, Vector &v);
void curlv_exact(const Vector &x, Vector &cv);

int dim;
real_t freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   int prob = 0;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&prob, "-p", "--problem-type",
                  "Choose between 0: grad, 1: curl, 2: div");
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
   kappa = freq * M_PI;

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

   // 5. Define a finite element space on the mesh. Here we use Nedelec or
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *trial_fec = NULL;
   FiniteElementCollection *test_fec = NULL;

   if (prob == 0)
   {
      trial_fec = new H1_FECollection(order, dim);
      test_fec = new ND_FECollection(order, dim);
   }
   else if (prob == 1)
   {
      trial_fec = new ND_FECollection(order, dim);
      test_fec = new RT_FECollection(order-1, dim);
   }
   else
   {
      trial_fec = new RT_FECollection(order-1, dim);
      test_fec = new L2_FECollection(order-1, dim);
   }

   FiniteElementSpace trial_fes(mesh, trial_fec);
   FiniteElementSpace test_fes(mesh, test_fec);

   int trial_size = trial_fes.GetTrueVSize();
   int test_size = test_fes.GetTrueVSize();

   if (prob == 0)
   {
      cout << "Number of Nedelec finite element unknowns: " << test_size << endl;
      cout << "Number of H1 finite element unknowns: " << trial_size << endl;
   }
   else if (prob == 1)
   {
      cout << "Number of Nedelec finite element unknowns: " << trial_size << endl;
      cout << "Number of Raviart-Thomas finite element unknowns: " << test_size <<
           endl;
   }
   else
   {
      cout << "Number of Raviart-Thomas finite element unknowns: "
           << trial_size << endl;
      cout << "Number of L2 finite element unknowns: " << test_size << endl;
   }

   // 6. Define the solution vector as a finite element grid function
   //    corresponding to the trial fespace.
   GridFunction gftest(&test_fes);
   GridFunction gftrial(&trial_fes);
   GridFunction x(&test_fes);
   FunctionCoefficient p_coef(p_exact);
   VectorFunctionCoefficient gradp_coef(sdim, gradp_exact);
   VectorFunctionCoefficient v_coef(sdim, v_exact);
   VectorFunctionCoefficient curlv_coef(sdim, curlv_exact);
   FunctionCoefficient divgradp_coef(div_gradp_exact);

   if (prob == 0)
   {
      gftrial.ProjectCoefficient(p_coef);
   }
   else if (prob == 1)
   {
      gftrial.ProjectCoefficient(v_coef);
   }
   else
   {
      gftrial.ProjectCoefficient(gradp_coef);
   }

   gftrial.SetTrueVector();
   gftrial.SetFromTrueVector();

   // 7. Set up the bilinear forms for L2 projection.
   ConstantCoefficient one(1.0);
   BilinearForm a(&test_fes);
   MixedBilinearForm a_mixed(&trial_fes, &test_fes);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a_mixed.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   if (prob == 0)
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_mixed.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
   }
   else if (prob == 1)
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_mixed.AddDomainIntegrator(new MixedVectorCurlIntegrator(one));
   }
   else
   {
      a.AddDomainIntegrator(new MassIntegrator(one));
      a_mixed.AddDomainIntegrator(new VectorFEDivergenceIntegrator(one));
   }

   // 8. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }

   a.Assemble();
   if (!pa) { a.Finalize(); }

   a_mixed.Assemble();
   if (!pa) { a_mixed.Finalize(); }

   if (pa)
   {
      a_mixed.Mult(gftrial, x);
   }
   else
   {
      SparseMatrix& mixed = a_mixed.SpMat();
      mixed.Mult(gftrial, x);
   }

   // 9. Define and apply a PCG solver for Ax = b with Jacobi preconditioner.
   {
      GridFunction rhs(&test_fes);
      rhs = x;
      x = 0.0;

      CGSolver cg;
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      if (pa)
      {
         Array<int> ess_tdof_list; // empty
         OperatorJacobiSmoother Jacobi(a, ess_tdof_list);

         cg.SetOperator(a);
         cg.SetPreconditioner(Jacobi);
         cg.Mult(rhs, x);
      }
      else
      {
         SparseMatrix& Amat = a.SpMat();
         DSmoother Jacobi(Amat);

         cg.SetOperator(Amat);
         cg.SetPreconditioner(Jacobi);
         cg.Mult(rhs, x);
      }
   }

   // 10. Compute the same field by applying a DiscreteInterpolator.
   GridFunction discreteInterpolant(&test_fes);
   DiscreteLinearOperator dlo(&trial_fes, &test_fes);
   if (prob == 0)
   {
      dlo.AddDomainInterpolator(new GradientInterpolator());
   }
   else if (prob == 1)
   {
      dlo.AddDomainInterpolator(new CurlInterpolator());
   }
   else
   {
      dlo.AddDomainInterpolator(new DivergenceInterpolator());
   }

   dlo.Assemble();
   dlo.Mult(gftrial, discreteInterpolant);

   // 11. Compute the projection of the exact field.
   GridFunction exact_proj(&test_fes);
   if (prob == 0)
   {
      exact_proj.ProjectCoefficient(gradp_coef);
   }
   else if (prob == 1)
   {
      exact_proj.ProjectCoefficient(curlv_coef);
   }
   else
   {
      exact_proj.ProjectCoefficient(divgradp_coef);
   }

   exact_proj.SetTrueVector();
   exact_proj.SetFromTrueVector();

   // 12. Compute and print the L_2 norm of the error.
   if (prob == 0)
   {
      real_t errSol = x.ComputeL2Error(gradp_coef);
      real_t errInterp = discreteInterpolant.ComputeL2Error(gradp_coef);
      real_t errProj = exact_proj.ComputeL2Error(gradp_coef);

      cout << "\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): "
           "|| E_h - grad p ||_{L_2} = " << errSol << '\n' << endl;
      cout << " Gradient interpolant E_h = grad p_h in H(curl): || E_h - grad p"
           " ||_{L_2} = " << errInterp << '\n' << endl;
      cout << " Projection E_h of exact grad p in H(curl): || E_h - grad p "
           "||_{L_2} = " << errProj << '\n' << endl;
   }
   else if (prob == 1)
   {
      real_t errSol = x.ComputeL2Error(curlv_coef);
      real_t errInterp = discreteInterpolant.ComputeL2Error(curlv_coef);
      real_t errProj = exact_proj.ComputeL2Error(curlv_coef);

      cout << "\n Solution of (E_h,w) = (curl v_h,w) for E_h and w in H(div): "
           "|| E_h - curl v ||_{L_2} = " << errSol << '\n' << endl;
      cout << " Curl interpolant E_h = curl v_h in H(div): || E_h - curl v "
           "||_{L_2} = " << errInterp << '\n' << endl;
      cout << " Projection E_h of exact curl v in H(div): || E_h - curl v "
           "||_{L_2} = " << errProj << '\n' << endl;
   }
   else
   {
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t errSol = x.ComputeL2Error(divgradp_coef, irs);
      real_t errInterp = discreteInterpolant.ComputeL2Error(divgradp_coef, irs);
      real_t errProj = exact_proj.ComputeL2Error(divgradp_coef, irs);

      cout << "\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: "
           "|| f_h - div v ||_{L_2} = " << errSol << '\n' << endl;
      cout << " Divergence interpolant f_h = div v_h in L_2: || f_h - div v "
           "||_{L_2} = " << errInterp << '\n' << endl;
      cout << " Projection f_h of exact div v in L_2: || f_h - div v "
           "||_{L_2} = " << errProj << '\n' << endl;
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
   delete trial_fec;
   delete test_fec;
   delete mesh;

   return 0;
}

real_t p_exact(const Vector &x)
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

real_t div_gradp_exact(const Vector &x)
{
   if (dim == 3)
   {
      return -3.0 * sin(x(0)) * sin(x(1)) * sin(x(2));
   }
   else if (dim == 2)
   {
      return -2.0 * sin(x(0)) * sin(x(1));
   }

   return 0.0;
}

void v_exact(const Vector &x, Vector &v)
{
   if (dim == 3)
   {
      v(0) = sin(kappa * x(1));
      v(1) = sin(kappa * x(2));
      v(2) = sin(kappa * x(0));
   }
   else
   {
      v(0) = sin(kappa * x(1));
      v(1) = sin(kappa * x(0));
      if (x.Size() == 3) { v(2) = 0.0; }
   }
}

void curlv_exact(const Vector &x, Vector &cv)
{
   if (dim == 3)
   {
      cv(0) = -kappa * cos(kappa * x(2));
      cv(1) = -kappa * cos(kappa * x(0));
      cv(2) = -kappa * cos(kappa * x(1));
   }
   else
   {
      cv = 0.0;
   }
}
