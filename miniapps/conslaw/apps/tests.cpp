#include "mfem.hpp"
#include "advection.hpp"
#include "mass.hpp"
#include <fstream>
#include <iostream>
#include <random>

using namespace std;
using namespace mfem;
using namespace dg;

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   for (int d = 0; d < dim; ++d)
   {
      v(d) = 1.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file =
      "../../data/periodic-hexagon.mesh";
   int ref_levels = -1;
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.

   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   PartialAssembly dgpa(&fes);
   Mass mass(&dgpa);
   MassInverse massinv(&mass);

   // Test mass matrix routines
   GridFunction u(&fes), u2(&fes), Mu(&fes), Mu2(&fes);
   // Generate random grid function
   mt19937 re(20);
   uniform_real_distribution<double> unif(0.0,1.0);
   for (int i = 0; i < u.Size(); ++i) { u[i] = unif(re); }

   mass.Mult(u, Mu);
   massinv.Mult(Mu, u2);
   u2 -= u;
   cout << "Difference u and M^{-1}M u = " << u2.Normlinf() << '\n';

   DGMassPA mass_pa(&dgpa);
   mass_pa.Mult(u, Mu2);
   Mu2 -= Mu;
   cout << "Difference M u and M_{PA} u = " << Mu2.Normlinf() << '\n';

   // Test advection integrators
   Advection adv_pa(&dgpa, dim);
   GridFunction Au(&fes), Au2(&fes);

   // Compare with standard MFEM integrator
   tic();
   BilinearForm k(&fes);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.Assemble();
   k.Finalize();
   toc();
   std::cout << "Standard assembly:    " << tic_toc.RealTime() << " s\n";
   tic();
   k.Mult(u, Au);
   toc();
   std::cout << "Standard application: " << tic_toc.RealTime() << " s\n";
   // Compute using partial assembly
   tic();
   adv_pa.Mult(u, Au2);
   toc();
   std::cout << "PA application:       " << tic_toc.RealTime() << " s\n";
   Au2 -= Au;
   cout << "Difference A u and A_{PA} u = " << Au2.Normlinf() << '\n';

   return 0;
}