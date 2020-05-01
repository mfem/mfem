//                           hp-Refinement Demo
//
// Compile with: make hptest
//
// Sample runs: TODO
//
// Description: This is a demo of the hp-refinement capability of MFEM.
//              One of the benchmark problems with a known exact solution is
//              solved on a sequence of meshes where both the size (h) and the
//              polynomial order (p) of elements is adapted.
//

#include "mfem.hpp"
#include <fstream>

#include "exact.hpp"
#include "util.hpp"
#include "error.hpp"

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
   double ref_threshold = 0.7;
   bool aniso = true;
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

      // Calculate the H^1_0 errors of elements as well as the total error.
      Array<double> elem_error;
      Array<int> ref_type;
      double error;
      {
         error = CalculateH10Error2(&x, &exgrad, &elem_error, &ref_type, int_order);
         error = std::sqrt(error);
      }

      // Refine elements
      Array<Refinement> refinements;
      {
         double err_max = elem_error.Max();
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (elem_error[i] > ref_threshold * err_max)
            {
               int type = aniso ? ref_type[i] : 7;
               refinements.Append(Refinement(i, type));
            }
         }
      }
      mesh.GeneralRefinement(refinements);

      // Update the space, interpolate the solution.
      fespace.Update();
      x.Update();

      // Inform also the bilinear and linear forms that the space has changed.
      bf.Update();
      lf.Update();
   }

   return 0;
}
