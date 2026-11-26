//                    MFEM Example 6 (Maxwell) - HDG Version
//
// Compile with: make ex6-max
//
// Sample runs:  ex6-max -m ../../data/square-disc.mesh -o 1
//               ex6-max -m ../../data/square-disc.mesh -o 2 -trh1
//               ex6-max -m ../../data/square-disc-nurbs.mesh -o 2
//               ex6-max -m ../../data/star.mesh -o 3
//               ex6-max -m ../../data/amr-quad.mesh -o 2
//               ex6-max -m ../../data/disc-nurbs.mesh -o 2 -trh1
//               ex6-max -m ../../data/fichera.mesh -o 1
//               ex6-max -m ../../data/escher.mesh -o 1
//
// Description:  This is a version of Example 5 (Maxwell) with a simple
//               adaptive mesh refinement loop. The problem being solved is the
//               steady magnetic diffusion corresponding to the mixed system
//
//                                 sigma*E - curl B = 0
//                                 curl E  + mu*B   = g
//
//               i.e., curl(1/sigma curl B) + mu*B = g, with the electric
//               boundary condition E x n = <given electric field> x n. We use
//               a given exact solution (E,B) and compute the
//               corresponding r.h.s. g. It is the solution of Example 5
//               (Maxwell) with the roles of the electric and magnetic fields
//               interchanged, so the forcing is applied in the magnetic
//               equation instead. The electric field E and the magnetic field B
//               are discretized by piecewise discontinuous polynomials and the
//               system is hybridized with the tangential trace of B as the
//               trace unknown. The problem is solved on a sequence of meshes
//               which are locally refined in a conforming (triangles,
//               tetrahedrons) or non-conforming (quadrilaterals, hexahedra)
//               manner according to the HDG error estimator.
//
//               As in Example 5 (Maxwell), the magnetic mass term mu*B makes
//               the problem well-posed, otherwise the magnetic field would be
//               determined only up to the kernel of the curl operator.
//
//               We recommend viewing Example 6 and Example 5 (Maxwell) to
//               understand the adaptive refinement loop and the construction
//               with the Hybridizable Discontinuous Galerkin method,
//               respectively.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void EFun_ex(const Vector & x, Vector & E);
void BFun_ex(const Vector & x, Vector & B);
void gFun(const Vector & x, Vector & g);
real_t freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
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
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();
   kappa = freq * M_PI;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int dim_curl = dim == 3 ? 3 : 1;

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

   // 5. Define the finite element spaces on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   //    We chose a closed basis for the electric field as it is customary for
   //    HDG to match trace DOFs, but an open basis can be used instead.
   auto E_coll = make_unique<L2_FECollection>(order, dim,
                                              BasisType::GaussLobatto);
   auto B_coll = make_unique<L2_FECollection>(order, dim);

   auto E_space = make_unique<FiniteElementSpace>(&mesh, E_coll.get(), dim);
   auto B_space = make_unique<FiniteElementSpace>(&mesh, B_coll.get(), dim_curl);

   // 6. As in Example 5 (Maxwell), we set up bilinear and linear forms
   //    corresponding to the magnetic diffusion problem. We don't assemble the
   //    discrete problem yet, this will be done in the main loop.
   MixedSystemForm mixed_system(E_space.get(), B_space.get());

   ConstantCoefficient sigmacoeff(1.0); // Diffusivity
   ConstantCoefficient invsigmacoeff(1.0); // Inverse diffusivity
   ConstantCoefficient mucoeff(1.0); // Permeability

   // Note the negative sign of g in the potential RHS, which is given by the
   // sign convention of the mixed system (see also Example 6)
   VectorFunctionCoefficient gcoeff(dim_curl, gFun); // Magnetic rhs
   ScalarVectorProductCoefficient neg_gcoeff(-1., gcoeff);
   VectorFunctionCoefficient Ecoeff(dim, EFun_ex); // Electric field
   VectorFunctionCoefficient Bcoeff(dim_curl, BFun_ex); // Magnetic field

   // Magnetic RHS
   LinearForm *gform = mixed_system.GetPotentialRHS();
   gform->AddDomainIntegrator(new VectorDomainLFIntegrator(neg_gcoeff));

   BilinearForm *Me = mixed_system.GetFluxMassForm();
   MixedBilinearForm *Ce = mixed_system.GetFluxDivForm();
   BilinearForm *Mb = mixed_system.GetPotentialMassForm();

   Me->AddDomainIntegrator(new VectorMassIntegrator(sigmacoeff));
   Ce->AddDomainIntegrator(new MixedCurlIntegrator(
                              MixedCurlIntegrator::Mode::OutOfPlane));
   Ce->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                    new DGTangentialTraceIntegrator(+1.)));
   Mb->AddInteriorFaceIntegrator(new HDGCurlCurlIntegrator(invsigmacoeff, td));

   if (dim_curl > 1)
   {
      Mb->AddDomainIntegrator(new VectorMassIntegrator(mucoeff));
   }
   else
   {
      Mb->AddDomainIntegrator(new MassIntegrator(mucoeff));
   }

   // The boundary faces are treated as the interior ones and the electric BC
   // is imposed weakly through the trace RHS
   Ce->AddBdrFaceIntegrator(new TransposeIntegrator(
                               new DGTangentialTraceIntegrator(+2.)));
   Mb->AddBdrFaceIntegrator(new HDGCurlCurlIntegrator(invsigmacoeff, td));

   // Set hybridization

   Array<int> ess_flux_tdofs_list;

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<FiniteElementSpace> trace_space;

   if (trace_h1)
   {
      trace_coll = make_unique<H1_Trace_FECollection>(max(order, 1), dim);
   }
   else
   {
      trace_coll = make_unique<DG_Interface_FECollection>(order, dim);
   }
   trace_space = make_unique<FiniteElementSpace>(&mesh, trace_coll.get(),
                                                 dim - 1);
   mixed_system.EnableHybridization(trace_space.get(),
                                    new TangentTraceJumpIntegrator(-1.),
                                    ess_flux_tdofs_list);

   // The natural electric BC is imposed through the trace RHS. The linear
   // form is assembled in the main loop.
   LinearForm hform(trace_space.get());
   hform.AddBoundaryIntegrator(new VectorBoundaryTangentTraceLFIntegrator(
                                  Ecoeff));

   // 7. The solution vector x and the associated finite element grid functions
   //    will be maintained over the AMR iterations. We initialize it to zero.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(mixed_system.GetOffsets(), mt);
   x = 0.0;

   GridFunction E_h, B_h, Bhat_h;
   E_h.MakeRef(E_space.get(), x.GetBlock(0), 0);
   B_h.MakeRef(B_space.get(), x.GetBlock(1), 0);

   // 8. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 9. Set up an error estimator. Here we use the HDG estimator which
   //    evaluates the difference between the tangential face values of the
   //    magnetic field and the trace variable in an energy norm with respect
   //    to a given operator, which is represented by the provided integrator
   //    implementing ComputeHDGFaceEnergy() method.
   HDGCurlCurlIntegrator estimator_integ(invsigmacoeff, td);
   HDGErrorEstimator estimator(estimator_integ, Bhat_h, B_h);

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
      const int E_dofs = E_space->GetTrueVSize();
      const int B_dofs = B_space->GetTrueVSize();
      const int Bhat_dofs = trace_space->GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of electric field unknowns: " << E_dofs << endl;
      cout << "Number of magnetic field unknowns: " << B_dofs << endl;
      cout << "Number of trace unknowns: " << Bhat_dofs << endl;

      // 12. Set the natural electric BC through the trace RHS. The true vector
      //     refers directly to the linear form when no prolongation is needed.
      hform.Assemble();
      Vector B;
      const SparseMatrix *P = trace_space->GetConformingProlongation();
      if (P)
      {
         B.SetSize(Bhat_dofs);
         P->MultTranspose(hform, B);
      }
      else
      {
         B.MakeRef(hform, 0, hform.Size());
      }

      // 13. Assemble the system.
      mixed_system.Assemble();

      // 14. Create the linear system: constrain hanging nodes and possibly
      //     apply other transformations. The system will be solved for true
      //     (unconstrained) DOFs only.
      OperatorPtr A;
      Vector X;

      mixed_system.FormLinearSystem(ess_flux_tdofs_list, x, A, X, B);

      // 15. Solve the linear system A X = B.
#ifdef MFEM_USE_SUITESPARSE
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
      // system.
      UMFPackSolver solver;
      solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      solver.SetOperator(*A);
      solver.Mult(B, X);
#else
      // Use a simple symmetric Gauss-Seidel preconditioner with GMRES.
      GSSmoother M((SparseMatrix&)(*A));
      GMRES(*A, M, B, X, 1, 1000, 50, 1e-12, 0.0);
#endif

      // 16. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      mixed_system.RecoverFEMSolution(X, x);
      Bhat_h.MakeTRef(trace_space.get(), X, 0);
      Bhat_h.SetFromTrueVector();

      // Compute the L2 error norms.
      {
         const int order_quad = max(2, 2*order+1);
         const IntegrationRule *irs[Geometry::NumGeom];
         for (int i = 0; i < Geometry::NumGeom; ++i)
         {
            irs[i] = &(IntRules.Get(i, order_quad));
         }

         const real_t err_E = E_h.ComputeL2Error(Ecoeff, irs);
         const real_t err_B = B_h.ComputeL2Error(Bcoeff, irs);
         const real_t norm_E = ComputeLpNorm(2., Ecoeff, mesh, irs);
         const real_t norm_B = ComputeLpNorm(2., Bcoeff, mesh, irs);
         cout << "|| E_h - E_ex || / || E_ex || = " << err_E / norm_E << endl;
         cout << "|| B_h - B_ex || / || B_ex || = " << err_B / norm_B << endl;
      }
      // 17. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << B_h
                  << "window_title 'Magnetic field'" << flush;
      }

      if (B_dofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 18. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 19. Update the spaces to reflect the new state of the mesh.
      E_space->Update();
      B_space->Update();
      trace_space->Update();

      // 20. Inform also the bilinear and linear forms that the space has
      //     changed.
      mixed_system.Update();
      hform.Update();
      x.Update(mixed_system.GetOffsets(), mt);

      x = 0.;
      E_h.MakeRef(E_space.get(), x.GetBlock(0), 0);
      B_h.MakeRef(B_space.get(), x.GetBlock(1), 0);

      mixed_system.EnableHybridization(trace_space.get(),
                                       new TangentTraceJumpIntegrator(-1.),
                                       ess_flux_tdofs_list);
   }

   return 0;
}

// The magnetic field takes the role of the electric field in Example 5
// (Maxwell), and the electric field is then given by E = 1/sigma curl B
void BFun_ex(const Vector & x, Vector &B)
{
   const int dim = x.Size();
   if (dim == 3)
   {
      B(0) = sin(kappa * x(1));
      B(1) = sin(kappa * x(2));
      B(2) = sin(kappa * x(0));
   }
   else
   {
      B(0) = sin(kappa * x(0)) + sin(kappa * x(1));
   }
}

void EFun_ex(const Vector & x, Vector &E)
{
   const int dim = x.Size();
   if (dim == 3)
   {
      E(0) = -kappa * cos(kappa * x(2));
      E(1) = -kappa * cos(kappa * x(0));
      E(2) = -kappa * cos(kappa * x(1));
   }
   else
   {
      // curl of the out-of-plane field B e_z
      E(0) = +kappa * cos(kappa * x(1));
      E(1) = -kappa * cos(kappa * x(0));
   }
}

// g = curl E + mu B = (1 + kappa^2) B
void gFun(const Vector & x, Vector &g)
{
   BFun_ex(x, g);
   g *= 1. + kappa * kappa;
}
