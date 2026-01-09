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
   const char *mesh_file = "../../data/beam-tri.mesh";
   int order = 1;
   real_t td = 0.5;
   bool trace_h1 = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default H1).");
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
   L2_FECollection R_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection W_coll(order, dim, BasisType::GaussLobatto);

   const int dim_lame = 1 + dim * (dim+1) / 2;
   FiniteElementSpace R_space(&mesh, &R_coll, dim_lame);
   FiniteElementSpace W_space(&mesh, &W_coll, dim);

   DarcyForm darcy(&R_space, &W_space);
   const Array<int> &block_offsets = darcy.GetOffsets();

   // 5. In this example, the Dirichlet boundary conditions are defined by
   //    marking boundary attributes 1 and 2 in the marker Array 'dir_bdr'.
   //    These b.c. are imposed weakly, by adding the appropriate boundary
   //    integrators over the marked 'dir_bdr' to the bilinear and linear forms.
   //    With this DG formulation, there are no essential boundary conditions.
   Array<int> ess_stress_tdofs_list; // no essential b.c. (empty list)
   Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet

   Array<int> neu_bdr(dir_bdr.Size());
   for (int i = 0; i < neu_bdr.Size(); i++)
   {
      neu_bdr[i] = dir_bdr[i] ? 0 : 1;
   }

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
   VectorArrayCoefficient force_c(dim);
   for (int i = 0; i < dim-1; i++)
   {
      force_c.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = +1.0e-2;
      force_c.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm *g = darcy.GetPotentialRHS();
   g->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(force_c));

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_c(mu);

   // diffusion coefficient lambda+2*mu
   SumCoefficient sumlame_c(lambda_c, mu_c, 1., 2.);

   // 1/lambda coefficient
   Vector ilambda(lambda.Size());
   for (int i = 0; i < lambda.Size(); i++)
   {
      ilambda(i) = 1./lambda(i);
   }
   PWConstCoefficient ilambda_c(ilambda);

   // 1/2*mu coefficient
   Vector i2mu(mu.Size());
   for (int i = 0; i < mu.Size(); i++)
   {
      i2mu(i) = 1./(2.*mu(i));
   }
   PWConstCoefficient i2mu_c(i2mu);

   // inverse Lame coefficients for the decomposed stress tensor
   VectorArrayCoefficient ilame_c(dim_lame);
   ilame_c.Set(0, &ilambda_c, false);
   for (int i = 1; i < dim_lame; i++)
   {
      ilame_c.Set(i, &i2mu_c, false);
   }

   BilinearForm *Ms = darcy.GetFluxMassForm();
   MixedBilinearForm *Bs = darcy.GetFluxDivForm();
   BilinearForm *Mu = darcy.GetPotentialMassForm();

   Ms->AddDomainIntegrator(new VectorMassIntegrator(ilame_c));
   Bs->AddDomainIntegrator(new StressDivergenceIntegrator(-1.));
   Bs->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                    new DGNormalStressIntegrator(+1.)));
   Bs->AddBdrFaceIntegrator(new TransposeIntegrator(
                               new DGNormalStressIntegrator(+2.)), neu_bdr);
   Mu->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                    dim, new HDGDiffusionIntegrator(sumlame_c, td)));

   //set hybridization / assembly level

   FiniteElementCollection *trace_coll;
   FiniteElementSpace *trace_space;

   if (trace_h1)
   {
      trace_coll = new H1_Trace_FECollection(max(order, 1), dim);
   }
   else
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
   }
   trace_space = new FiniteElementSpace(&mesh, trace_coll, dim);
   darcy.EnableHybridization(trace_space,
                             new NormalStressJumpIntegrator(-1.),
                             ess_stress_tdofs_list);

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_vec_coeff(zero_vec);

   BlockVector x(block_offsets);
   x = 0.0;

   GridFunction u, uhat;
   u.MakeRef(&W_space, x.GetBlock(1), 0);

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
   BilinearFormIntegrator *estimator_integ = (*Mu->GetFBFI())[0];
   HDGErrorEstimator estimator(*estimator_integ, uhat, u);

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
      const int q_dofs = R_space.GetTrueVSize();
      const int u_dofs = W_space.GetTrueVSize();
      const int uhat_dofs = trace_space->GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of flux unknowns: " << q_dofs << endl;
      cout << "Number of potential unknowns: " << u_dofs << endl;
      cout << "Number of trace unknowns: " << uhat_dofs << endl;

      // 13. Assemble the stiffness matrix and the right-hand side.
      darcy.Assemble();

      // 14. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.

      // 15. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      SparseMatrix A;
      Vector B, X;
      const int copy_interior = 1;
      darcy.FormLinearSystem(ess_stress_tdofs_list, x, A, X, B, copy_interior);

#ifdef MFEM_USE_SUITESPARSE
      // 16. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with GMRES.
      GSSmoother M(A);
      GMRES(A, M, B, X, 3, 2000, 500, 1e-12, 0.0);
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
      darcy.RecoverFEMSolution(X, x);
      uhat.MakeTRef(trace_space, X, 0);
      uhat.SetFromTrueVector();

      // 18. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         GridFunction nodes(&W_space), *nodes_p = &nodes;
         mesh.GetNodes(nodes);
         nodes += u;
         int own_nodes = 0;
         mesh.SwapNodes(nodes_p, own_nodes);
         u.Neg(); // visualize the backward displacement
         sol_sock << "solution\n" << mesh << u << flush;
         u.Neg();
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

      if (u_dofs > max_dofs)
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
      R_space.Update();
      W_space.Update();
      trace_space->Update();

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      darcy.Update();
      x.Update(darcy.GetOffsets());

      x = 0.;
      u.MakeRef(&W_space, x.GetBlock(1), 0);

      darcy.EnableHybridization(trace_space,
                                new NormalStressJumpIntegrator(-1.),
                                ess_stress_tdofs_list);
   }

   {
      ofstream mesh_ref_out("ex21_reference.mesh");
      mesh_ref_out.precision(16);
      mesh.Print(mesh_ref_out);

      ofstream mesh_out("ex21_deformed.mesh");
      mesh_out.precision(16);
      GridFunction nodes(&W_space), *nodes_p = &nodes;
      mesh.GetNodes(nodes);
      nodes += u;
      int own_nodes = 0;
      mesh.SwapNodes(nodes_p, own_nodes);
      mesh.Print(mesh_out);
      mesh.SwapNodes(nodes_p, own_nodes);

      ofstream u_out("ex21_displacement.sol");
      u_out.precision(16);
      u.Save(u_out);
   }

   // clean-up
   delete trace_space;
   delete trace_coll;

   return 0;
}
