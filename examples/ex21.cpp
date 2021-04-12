//                                MFEM Example 21
//
// Compile with: make ex21
//
// Sample runs:  ex21
//               ex21 -o 3
//               ex21 -m ../data/beam-quad.mesh
//               ex21 -m ../data/beam-quad.mesh -o 3
//               ex21 -m ../data/beam-quad.mesh -o 3 -f 1
//               ex21 -m ../data/beam-tet.mesh
//               ex21 -m ../data/beam-tet.mesh -o 2
//               ex21 -m ../data/beam-hex.mesh
//               ex21 -m ../data/beam-hex.mesh -o 2
//
// Description:  This is a version of Example 2 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the linear
//               elasticity describing a multi-material cantilever beam.
//               The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear and curved meshes. Interpolation of functions from
//               coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Examples 2 and 6 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   int flux_averaging = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&flux_averaging, "-f", "--flux-averaging",
                  "Flux averaging: 0 - global, 1 - by mesh attribute.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, and hexahedral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   MFEM_VERIFY(mesh.SpaceDimension() == dim, "invalid mesh");

   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Since a NURBS mesh can currently only be refined uniformly, we need to
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

   // 4. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   // 5. As in Example 2, we set up the linear form b(.) which corresponds to
   //    the right-hand side of the FEM linear system. In this case, b_i equals
   //    the boundary integral of f*phi_i where f represents a "pull down"
   //    force on the Neumann part of the boundary and phi_i are the basis
   //    functions in the finite element fespace. The force is defined by the
   //    VectorArrayCoefficient object f, which is a vector of Coefficient
   //    objects. The fact that f is non-zero on boundary attribute 2 is
   //    indicated by the use of piece-wise constants coefficient for its last
   //    component. We don't assemble the discrete problem yet, this will be
   //    done in the main loop.
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

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorBoundaryLFIntegrator(f));

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   BilinearForm a(&fespace);
   BilinearFormIntegrator *integ =
      new ElasticityIntegrator(lambda_func,mu_func);
   a.AddDomainIntegrator(integ);
   if (static_cond) { a.EnableStaticCondensation(); }

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_vec_coeff(zero_vec);
   GridFunction x(&fespace);
   x = 0.0;

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.  The conversion to true dofs will be done in the
   //    main loop.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   // 10. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     that uses the ComputeElementFlux method of the ElasticityIntegrator to
   //     recover a smoothed flux (stress) that is subtracted from the element
   //     flux to get an error indicator. We need to supply the space for the
   //     smoothed flux: an (H1)^tdim (i.e., vector-valued) space is used here.
   //     Here, tdim represents the number of components for a symmetric (dim x
   //     dim) tensor.
   const int tdim = dim*(dim+1)/2;
   FiniteElementSpace flux_fespace(&mesh, &fec, tdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   estimator.SetFluxAveraging(flux_averaging);

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 50000;
   const int max_amr_itr = 20;
   for (int it = 0; it <= max_amr_itr; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 13. Assemble the stiffness matrix and the right-hand side.
      a.Assemble();
      b.Assemble();

      // 14. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(zero_vec_coeff, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      SparseMatrix A;
      Vector B, X;
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

#ifndef MFEM_USE_SUITESPARSE
      // 16. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with PCG.
      GSSmoother M(A);
      PCG(A, M, B, X, 3, 2000, 1e-12, 0.0);
#else
      // 16. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
      //     the linear system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(B, X);
#endif

      // 17. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // 18. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         GridFunction nodes(&fespace), *nodes_p = &nodes;
         mesh.GetNodes(nodes);
         nodes += x;
         int own_nodes = 0;
         mesh.SwapNodes(nodes_p, own_nodes);
         x.Neg(); // visualize the backward displacement
         sol_sock << "solution\n" << mesh << x << flush;
         x.Neg();
         mesh.SwapNodes(nodes_p, own_nodes);
         if (it == 0)
         {
            sol_sock << "keys '" << ((dim == 2) ? "Rjl" : "") << "m'" << endl;
         }
         sol_sock << "window_title 'AMR iteration: " << it << "'\n"
                  << "pause" << endl;
         cout << "Visualization paused. "
              "Press <space> in the GLVis window to continue." << endl;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 19. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 20. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   {
      ofstream mesh_ref_out("ex21_reference.mesh");
      mesh_ref_out.precision(16);
      mesh.Print(mesh_ref_out);

      ofstream mesh_out("ex21_deformed.mesh");
      mesh_out.precision(16);
      GridFunction nodes(&fespace), *nodes_p = &nodes;
      mesh.GetNodes(nodes);
      nodes += x;
      int own_nodes = 0;
      mesh.SwapNodes(nodes_p, own_nodes);
      mesh.Print(mesh_out);
      mesh.SwapNodes(nodes_p, own_nodes);

      ofstream x_out("ex21_displacement.sol");
      x_out.precision(16);
      x.Save(x_out);
   }

   return 0;
}
