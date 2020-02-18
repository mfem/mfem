//                                MFEM Example 21
//
// Compile with: make ex21p
//
// Sample runs:  mpirun -np 4 ex21p
//               mpirun -np 4 ex21p -o 3
//               mpirun -np 4 ex21p -m ../data/beam-quad.mesh
//               mpirun -np 4 ex21p -m ../data/beam-quad.mesh -o 3
//               mpirun -np 4 ex21p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex21p -m ../data/beam-tet.mesh -o 2
//               mpirun -np 4 ex21p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex21p -m ../data/beam-hex.mesh -o 2
//
// Description:  This is a version of Example 2p with a simple adaptive mesh
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
//               We recommend viewing Examples 2p and 6p before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-tri.mesh";
   int serial_ref_levels = 0;
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of uniform serial refinements (before parallel"
                  " partitioning)");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

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
      MPI_Finalize();
      return 3;
   }

   // 3. Refine the mesh before parallel partitioning. Since a NURBS mesh can
   //    currently only be refined uniformly, we need to convert it to a
   //    piecewise-polynomial curved mesh. First we refine the NURBS mesh a bit
   //    more and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext && serial_ref_levels == 0)
   {
      serial_ref_levels = 2;
   }
   for (int i = 0; i < serial_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim);

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
      Vector pull_force(pmesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorBoundaryLFIntegrator(f));

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(pmesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm a(&fespace);
   BilinearFormIntegrator *integ =
      new ElasticityIntegrator(lambda_func,mu_func);
   a.AddDomainIntegrator(integ);
   if (static_cond) { a.EnableStaticCondensation(); }

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_vec_coeff(zero_vec);
   ParGridFunction x(&fespace);
   x = 0.0;

   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.  The conversion to true dofs will be done in the
   //    main loop.
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

   // 9. GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;

   // 10. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     that uses the ComputeElementFlux method of the ElasticityIntegrator to
   //     recover a smoothed flux (stress) that is subtracted from the element
   //     flux to get an error indicator. We need to supply the space for the
   //     smoothed flux: an (H1)^tdim (i.e., vector-valued) space is used here.
   //     Here, tdim represents the number of components for a symmetric (dim x
   //     dim) tensor.
   const int tdim = dim*(dim+1)/2;
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fespace(&pmesh, &flux_fec, tdim);
   ParFiniteElementSpace smooth_flux_fespace(&pmesh, &fec, tdim);
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace,
                                       smooth_flux_fespace);

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
      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

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

      HypreParMatrix A;
      Vector B, X;
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 16. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);
      // amg.SetSystemsOptions(dim); // optional
      CGSolver pcg(A.GetComm());
      pcg.SetPreconditioner(amg);
      pcg.SetOperator(A);
      pcg.SetRelTol(1e-6);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(3); // print the first and the last iterations only
      pcg.Mult(B, X);

      // 17. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // 18. Send solution by socket to the GLVis server.
      if (visualization && it == 0)
      {
         sol_sock.open(vishost, visport);
         sol_sock.precision(8);
      }
      if (visualization && sol_sock.good())
      {
         GridFunction nodes(&fespace), *nodes_p = &nodes;
         pmesh.GetNodes(nodes);
         nodes += x;
         int own_nodes = 0;
         pmesh.SwapNodes(nodes_p, own_nodes);
         x.Neg(); // visualize the backward displacement
         sol_sock << "parallel " << num_procs << ' ' << myid << '\n';
         sol_sock << "solution\n" << pmesh << x << flush;
         x.Neg();
         pmesh.SwapNodes(nodes_p, own_nodes);
         if (it == 0)
         {
            sol_sock << "keys '" << ((dim == 2) ? "Rjl" : "") << "m'" << endl;
         }
         sol_sock << "window_title 'AMR iteration: " << it << "'\n"
                  << "pause" << endl;
         if (myid == 0)
         {
            cout << "Visualization paused. "
                 "Press <space> in the GLVis window to continue." << endl;
         }
      }

      if (global_dofs > max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 19. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
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

      // 21. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         x.Update();
      }

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   {
      ostringstream mref_name, mesh_name, sol_name;
      mref_name << "ex21p_reference_mesh." << setfill('0') << setw(6) << myid;
      mesh_name << "ex21p_deformed_mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex21p_displacement." << setfill('0') << setw(6) << myid;

      ofstream mesh_ref_out(mref_name.str().c_str());
      mesh_ref_out.precision(16);
      pmesh.Print(mesh_ref_out);

      ofstream mesh_out(mesh_name.str().c_str());
      mesh_out.precision(16);
      GridFunction nodes(&fespace), *nodes_p = &nodes;
      pmesh.GetNodes(nodes);
      nodes += x;
      int own_nodes = 0;
      pmesh.SwapNodes(nodes_p, own_nodes);
      pmesh.Print(mesh_out);
      pmesh.SwapNodes(nodes_p, own_nodes);

      ofstream x_out(sol_name.str().c_str());
      x_out.precision(16);
      x.Save(x_out);
   }

   MPI_Finalize();
   return 0;
}
