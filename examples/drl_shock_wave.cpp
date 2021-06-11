// Compile with: make drl_shock_wave
//
// drl_shock_wave -o 2 -m ../data/inline-quad.mesh
// for multi agent local, set the mesh to use 20x20 grid because that is what 
// was used for training.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "drl_shock_wave.hpp"
#include "multi_agent_local_refiner.hpp"

#define MFEM_USE_RLLIB
#ifdef MFEM_USE_RLLIB
#include <Python.h>
#include "numpy/arrayobject.h"
#endif

using namespace std;
using namespace mfem;

double bdry_function(const Vector& x)
{
   return 0.0; // default
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

#ifdef MFEM_USE_RLLIB
   Py_Initialize();
   import_array(); // numpy init
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 4. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }
   else {
      // mesh.UniformRefinement();
      //mesh.UniformRefinement();
      mesh.EnsureNCMesh();
   }

   // 5. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // Create 0-order L2 gridfunction to hold errors
   L2_FECollection fec0(0, dim);
   FiniteElementSpace fes0(&mesh, &fec0);
   GridFunction err(&fes0);

   // 6. As in Example 1, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.SetDiagonalPolicy(Operator::DIAG_ONE);
   }
   LinearForm b(&fespace);

   FunctionCoefficient bdry(bdry_function);
   FunctionCoefficient rhs(layer2_laplace);
   FunctionCoefficient exact(layer2_exsol);
   ConstantCoefficient one(1.0);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   int int_order = 8;
   int geom_type = mesh.GetElementBaseGeometry(0);
   DomainLFIntegrator* dlfi = new DomainLFIntegrator(rhs);
   dlfi->SetIntRule(&IntRules.Get(geom_type, int_order));
   b.AddDomainIntegrator(dlfi);

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   GridFunction x(&fespace);
   x = 0.0;

   // 8. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   socketstream err_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
      err_sock.open(vishost, visport);
   }

   // 10. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     that uses the ComputeElementFlux method of the DiffusionIntegrator to
   //     recover a smoothed flux (gradient) that is subtracted from the element
   //     flux to get an error indicator. We need to supply the space for the
   //     smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   FiniteElementSpace flux_fespace(&mesh, &fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   //KellyErrorEstimator estimator2(*integ, x, flux_fespace);
   //estimator.SetAnisotropic();

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.

#if 0
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.10);
#else
   MAL_DRLRefiner refiner(x);
   //DRLRefiner refiner(x);
#endif
  
#if 1
   x.ProjectCoefficient(exact);
   refiner.Apply(mesh);
   fespace.Update();
   fes0.Update();
   x.Update();
   x.ProjectCoefficient(exact);
   refiner.Apply(mesh);
   fespace.Update();
   fes0.Update();
   x.Update();
   {
      ofstream sol_ofs("sol.gf");
      x.Save(sol_ofs);
   }
   {
      ofstream mesh_ofs("amr.mesh");
      mesh_ofs.precision(14);
      mesh.Print(mesh_ofs);
   }
   MFEM_ABORT(" ");
#endif

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 2000;
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 13. Assemble the right-hand side.
      b.Assemble();

      // 14. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(exact, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Assemble the stiffness matrix.
      a.Assemble();
      
      // 16. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      if (!pa)
      {
#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif
      }
      else // Diagonal preconditioning in partial assembly mode.
      {
         OperatorJacobiSmoother M(a, ess_tdof_list);
         PCG(*A, M, B, X, 3, 2000, 1e-12, 0.0);
      }

      // 18. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // Compute error against exact solution

      x.ComputeElementL2Errors(exact, err);

      // 19. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }
      if (visualization && err_sock.good())
      {
         err_sock.precision(8);
         err_sock << "solution\n" << mesh << err << flush;
      }

      if (cdofs > max_dofs || it == 3)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.

      std::cout << " apply refiner\n";
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 21. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      fes0.Update();
      x.Update();
      err.Update();

      // 22. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }


   {
      ofstream sol_ofs("sol.gf");
      x.Save(sol_ofs);
   }

   {
      ofstream mesh_ofs("amr.mesh");
      mesh_ofs.precision(14);
      mesh.Print(mesh_ofs);
   }

   return 0;
}
