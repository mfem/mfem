//                                MFEM DRL4AMR
//
// Compile with: make
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"

#include <list>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define dbg(...) \
   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

#include "drl4amr.hpp"

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 8;
constexpr double sharpness = 100.0;

double x0(const Vector &x)
{
   double result = 0.0;
   const double t = x[0] + tan(theta)*x[1];
   for (double o: offsets) { result += tanh(sharpness*(o - t)); }
   return result/static_cast<double>(discs);
}

Drl4Amr::Drl4Amr(int o):
   order(o),
   device(device_config), // Enable hardware devices
   mesh(nx, ny, type, generate_edges, sx, sy, sfc), // Create the mesh
   dim(mesh.Dimension()),
   sdim(mesh.SpaceDimension()),
   fec(order, dim, BasisType::Positive),
   fespace(&mesh, &fec),
   a(&fespace),
   b(&fespace),
   one(1.0),
   zero(0.0),
   integ(new DiffusionIntegrator(one)),
   x(&fespace),
   ess_bdr(mesh.bdr_attributes.Max()),
   iteration(0),
   flux_fespace(&mesh, &fec, sdim),
   estimator(*integ, x, flux_fespace),
   refiner(estimator)
{
   dbg("Drl4Amr o:%d",o);
   device.Print();

   mesh.EnsureNodes();
   mesh.PrintCharacteristics();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);

   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // Connect to GLVis.
   if (visualization) { sol_sock.open(vishost, visport); }

   // Initialize theta, offsets and x from x0_coeff
   srand48(time(NULL));
   theta = M_PI*drand48()/2.0;
   discs = static_cast<int>(1 + nb_discs_max*drand48());
   offsets.SetSize(discs);
   for (int i=0; i < discs; i++) { offsets[i] = drand48(); }
   offsets.Sort();
   dbg("theta = %f, discontinuities:%d", theta, discs);
   for (double offset: offsets) { dbg("%f ", offset); }
   FunctionCoefficient x0_coeff(x0);
   x.ProjectCoefficient(x0_coeff);

   if (visualization && sol_sock.good())
   {
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "pause" << endl;
   }

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0, "BC attributes required!");
   ess_bdr = 1;

   // Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   // that uses the ComputeElementFlux method of the DiffusionIntegrator to
   // recover a smoothed flux (gradient) that is subtracted from the element
   // flux to get an error indicator. We need to supply the space for the
   // smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   estimator.SetAnisotropic();

   // A refiner selects and refines elements based on a refinement strategy.
   // The strategy here is to refine elements with errors larger than a
   // fraction of the maximum element error. Other strategies are possible.
   // The refiner will call the given error estimator.
   refiner.SetTotalErrorFraction(0.7);
}

int Drl4Amr::Compute()
{
   dbg("Compute: %d", iteration);
   const int cdofs = fespace.GetTrueVSize();
   cout << "\nAMR iteration " << iteration << endl;
   cout << "Number of unknowns: " << cdofs << endl;

   // Assemble the right-hand side.
   b.Assemble();

   // Set Dirichlet boundary values in the GridFunction x.
   // Determine the list of Dirichlet true DOFs in the linear system.
   Array<int> ess_tdof_list;
   x.ProjectBdrCoefficient(zero, ess_bdr);
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Assemble the stiffness matrix.
   a.Assemble();

   // Create the linear system: eliminate boundary conditions, constrain
   // hanging nodes and possibly apply other transformations. The system
   // will be solved for true (unconstrained) DOFs only.
   OperatorPtr A;
   Vector B, X;

   const int copy_interior = 1;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

   // Solve the linear system A X = B.
   CG(*A, B, X, 3, 2000, 1e-12, 0.0);

   // After solving the linear system, reconstruct the solution as a
   // finite element GridFunction. Constrained nodes are interpolated
   // from true DOFs (it may therefore happen that x.Size() >= X.Size()).
   a.RecoverFEMSolution(X, b, x);

   // Send solution by socket to the GLVis server.
   if (visualization && sol_sock.good())
   {
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "pause" << endl;
      cout << "Visualization paused. "
           "Press <space> in the GLVis window to continue." << endl;
   }

   if (cdofs > max_dofs)
   {
      cout << "Reached the maximum number of dofs. Stop." << endl;
      return 1;
   }
   iteration += 1;
   return 0;
}

int Drl4Amr::Refine()
{
   dbg("Refine");
   // Call the refiner to modify the mesh. The refiner calls the error
   // estimator to obtain element errors, then it selects elements to be
   // refined and finally it modifies the mesh. The Stop() method can be
   // used to determine if a stopping criterion was met.
   refiner.Apply(mesh);
   if (refiner.Stop())
   {
      cout << "Stopping criterion satisfied. Stop." << endl;
      return 1;
   }
   return 0;
}

int Drl4Amr::Update()
{
   dbg("Update");
   // Update the space to reflect the new state of the mesh. Also,
   // interpolate the solution x so that it lies in the new space but
   // represents the same function. This saves solver iterations later
   // since we'll have a good initial guess of x in the next step.
   // Internally, FiniteElementSpace::Update() calculates an
   // interpolation matrix which is then used by GridFunction::Update().
   fespace.Update();
   x.Update();

   // Inform also the bilinear and linear forms that the space has changed.
   a.Update();
   b.Update();
   return 0;
}

extern "C" {
   Drl4Amr* Ctrl(int order) { return new Drl4Amr(order); }
   int Compute(Drl4Amr *ctrl) { return ctrl->Compute(); }
   int Refine(Drl4Amr *ctrl) { return ctrl->Refine(); }
   int Update(Drl4Amr *ctrl) { return ctrl->Update(); }
}
