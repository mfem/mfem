//                           hp-Refinement Test
//
// Compile with: make hptest
//
// Sample runs:
//
// Description:
//

#include "mfem.hpp"
#include <fstream>

#include "exact.hpp"
#include "util.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "layer-quad.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int int_order = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
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

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Load and adjust the Mesh
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   //int sdim = mesh.SpaceDimension();

   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh(true);

   // We don't support mixed meshes at the moment
   MFEM_VERIFY(mesh.GetNumGeometries(dim) == 1, "Mixed meshes not supported.");
   Geometry::Type geom = mesh.GetElementGeometry(0);

   // Prepare exact solution Coefficients
   FunctionCoefficient exsol(layer2_exsol);
      /*prob_type ? ((dim == 3) ? layer3_exsol  : layer2_exsol)
                : ((dim == 3) ? fichera_exsol : lshape_exsol) );*/

   VectorFunctionCoefficient exgrad(dim, layer2_exgrad);
      /*prob_type ? ((dim == 3) ? layer3_exgrad  : layer2_exgrad)
                : ((dim == 3) ? fichera_exgrad : lshape_exgrad) );*/

   FunctionCoefficient rhs(layer2_laplace);
      /*prob_type ? ((dim == 3) ? layer3_laplace  : layer2_laplace)
                : ((dim == 3) ? fichera_laplace : lshape_laplace) );*/

   // Define a finite element space on the mesh. Initially the polynomial
   // order is constant everywhere.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   GridFunction x(&fespace);
   x = 0.0;

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // The main AMR loop. In each iteration we solve the problem on the
   // current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 50000;
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // Assemble the linear form. The right hand side is manufactured
      // so that the solution is the analytic solution.
      LinearForm lf(&fespace);
      DomainLFIntegrator *dlfi = new DomainLFIntegrator(rhs);
      dlfi->SetIntRule(&IntRules.Get(geom, int_order));
      lf.AddDomainIntegrator(dlfi);
      lf.Assemble();

      // Assemble bilinear form.
      BilinearForm bf(&fespace);
      if (pa) { bf.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      bf.AddDomainIntegrator(new DiffusionIntegrator());
      bf.Assemble();

      // Set Dirichlet boundary values in the GridFunction x.
      // Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(exsol, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 16. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      if (!pa)
      {
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
      }
      else // No preconditioning for now in partial assembly mode.
      {
         CG(*A, B, X, 3, 2000, 1e-12, 0.0);
      }

      // 18. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      bf.RecoverFEMSolution(X, lf, x);

      // 19. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }


      // TODO refine
      mesh.RandomRefinement(0.5);


      // 21. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 22. Inform also the bilinear and linear forms that the space has
      //     changed.
      bf.Update();
      lf.Update();
   }

   return 0;
}
