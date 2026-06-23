//                          MFEM Example 6 - HDG Version
//
// Compile with: make ex6
//
// Sample runs:  ex6 -m ../../data/square-disc.mesh -o 1 -dg
//               ex6 -m ../../data/square-disc.mesh -o 2 -brt
//               ex6 -m ../../data/square-disc-nurbs.mesh -o 2 -dg -trh1
//               ex6 -m ../../data/star.mesh -o 3
//               ex6 -m ../../data/escher.mesh -o 2 -dg
//               ex6 -m ../../data/fichera.mesh -o 2 -dg -trh1
//               ex6 -m ../../data/disc-nurbs.mesh -o 2 -brt
//               ex6 -m ../../data/ball-nurbs.mesh -dg
//               ex6 -m ../../data/pipe-nurbs.mesh -brt
//               ex6 -m ../../data/amr-quad.mesh -brt -trh1
//
// Device sample runs:
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Poisson
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to HDG error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear and curved meshes. Interpolation of functions from
//               coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing examples 1 and 5 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool dg = false;
   bool brt = false;
   real_t td = 0.5;
   bool trace_h1 = false;
   const char *device_config = "cpu";
   int max_dofs = 50000;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
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

   // 5. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   FiniteElementCollection *R_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      R_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      R_coll = new BrokenRT_FECollection(order, dim);
   }
   else
   {
      R_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim);

   FiniteElementSpace *R_space = new FiniteElementSpace(&mesh, R_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *W_space = new FiniteElementSpace(&mesh, W_coll);

   // 6. As in Example 1, we set up bilinear and linear forms corresponding to
   //    the Poisson problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.

   DarcyForm darcy(R_space, W_space);

   ConstantCoefficient one(1.0), negone(-1.0);

   LinearForm *f = darcy.GetPotentialRHS();
   f->AddDomainIntegrator(new DomainLFIntegrator(negone));

   BilinearForm *Mq = darcy.GetFluxMassForm();
   MixedBilinearForm *Bq = darcy.GetFluxDivForm();
   BilinearForm *Mu = (dg)?(darcy.GetPotentialMassForm()):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator());
      Bq->AddDomainIntegrator(new VectorDivergenceIntegrator());
      Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      Mu->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator());
      Bq->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      if (brt)
      {
         Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                          new DGNormalTraceIntegrator(-1.)));
      }
   }

   // Set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   if (trace_h1)
   {
      trace_coll = new H1_Trace_FECollection(max(order, 1), dim);
   }
   else
   {
      trace_coll = new DG_Interface_FECollection(order, dim);
   }
   trace_space = new FiniteElementSpace(&mesh, trace_coll);
   darcy.EnableHybridization(trace_space,
                             new NormalTraceJumpIntegrator(),
                             ess_flux_tdofs_list);

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(darcy.GetOffsets(), mt);
   x = 0.0;

   GridFunction u_h, uhat_h;
   u_h.MakeRef(W_space, x.GetBlock(1), 0);

   // 8. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 9. Set up an error estimator. Here we use the HDG estimator which
   //    evaluates the difference between the face values of the potential and
   //    the trace variable in an energy norm with respect to a given operator,
   //    which is represented by the provided integrator implementing
   //    ComputeHDGFaceEnergy() method.
   HDGDiffusionIntegrator estimator_integ(one, td);
   HDGErrorEstimator estimator(estimator_integ, uhat_h, u_h);

   // 10. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 11. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
   {
      const int q_dofs = R_space->GetTrueVSize();
      const int u_dofs = W_space->GetTrueVSize();
      const int uhat_dofs = trace_space->GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of flux unknowns: " << q_dofs << endl;
      cout << "Number of potential unknowns: " << u_dofs << endl;
      cout << "Number of trace unknowns: " << uhat_dofs << endl;

      // 12. Assemble the stiffness matrix.
      darcy.Assemble();

      // 13. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      darcy.FormLinearSystem(ess_flux_tdofs_list, x, A, X, B);

      // 14. Solve the linear system A X = B.
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with GMRES.
      GSSmoother M((SparseMatrix&)(*A));
      GMRES(*A, M, B, X, 3, 200, 50, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver solver;
      solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      solver.SetOperator(*A);
      solver.Mult(B, X);
#endif

      // 15. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      darcy.RecoverFEMSolution(X, x);
      uhat_h.MakeTRef(trace_space, X, 0);
      uhat_h.SetFromTrueVector();

      // 16. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << u_h << flush;
      }

      if (u_dofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 17. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 18. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      R_space->Update();
      W_space->Update();
      trace_space->Update();

      // 19. Inform also the bilinear and linear forms that the space has
      //     changed.

      darcy.Update();
      x.Update(darcy.GetOffsets(), mt);

      x = 0.;
      u_h.MakeRef(W_space, x.GetBlock(1), 0);

      darcy.EnableHybridization(trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);
   }

   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;
   return 0;
}
