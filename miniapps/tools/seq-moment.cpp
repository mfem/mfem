#include "mfem.hpp"
#include "../common/mesh_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

static int prob_ = -1;

double densityFunc(const Vector &x)
{
  switch(prob_)
    {
    case 0:
      // Linear in the radius
      return sqrt(x * x);
    case 1:
      // Hydrostatic equilibrium (1D)
      return pow(cosh(sqrt(0.5 * (x * x))), -2.0);
    case 2:
      // Hydrostatic equilibrium (2D)
      return pow(1.0 + 0.125 * (x * x), -2.0);
    case 3:
      // Hydrostatic equilibrium (3D), approx.
      return pow(1.0 + (x * x) / 9.0, -1.5);
    case 4:
      // Off-center Gaussian
      {
	double s_data[3];
	Vector s(s_data, x.Size());
	s = 0.0; s[0] = -1.0; s += x;
	return exp(-(s * s));
      }
    }  
  // Default to homogeneous
  return 1.0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/ball-nurbs.mesh";
   int ref_levels = 2;
   int ir_order = 2;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&ir_order, "-o", "--order",
                  "Integration rule order.");
   args.AddOption(&prob_, "-d", "--density",
                  "Density profile:\n"
		  "  0 - Linear increase with radius,\n"
		  "  1 - Hydrostatic equilibrium in 1D,\n"
		  "  2 - Hydrostatic equilibrium in 2D,\n"
		  "  3 - Hydrostatic equilibrium in 3D (approximately),\n"
		  "  4 - Gaussian centered at x = 1,\n"
		  "  Default - homogeneous.");
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
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement where 'ref_levels' is user defined.
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   FunctionCoefficient density(densityFunc);

   L2_FECollection fec(0, dim);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace vfes(&mesh, &fec, sdim);
   GridFunction elemMass(&fes);
   GridFunction elemCent(&vfes);

   double vol = ComputeVolume(mesh, ir_order);
   double area = ComputeSurfaceArea(mesh, ir_order);

   Vector mom1(sdim);
   Vector cent(sdim); cent = 0.0;
   DenseMatrix mom2(sdim);

   double mass  = ComputeZerothMoment(mesh, density, ir_order);
   double mass1 = ComputeFirstMoment(mesh, density, ir_order, mom1);
   double mass2 = ComputeSecondMoment(mesh, density, cent,
                                      ir_order, mom2);

   cout << "Volume:       " << vol << endl;
   cout << "Surface Area: " << area << endl;
   cout << "Mass:         " << mass << endl;
   cout << "Mass:         " << mass1 << endl;
   cout << "Mass:         " << mass2 << endl;

   cout << "First Moment:   "; mom1.Print(cout);
   mom1 /= mass1;
   cout << "Center of Mass: "; mom1.Print(cout);
   cout << "Second Moment (moment of inertia):\n";
   mom2.Print(cout);

   ComputeElementZerothMoments(mesh, density, ir_order, elemMass);
   ComputeElementCentersOfMass(mesh, density, ir_order, elemCent);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream m_sock(vishost, visport);
      m_sock.precision(8);
      m_sock << "solution\n" << mesh << elemMass
             << "window_title 'Element Masses'" << flush;

      socketstream c_sock(vishost, visport);
      c_sock.precision(8);
      c_sock << "solution\n" << mesh << elemCent
             << "window_title 'Element Centers'"
             << "keys vvv" << flush;
   }

   return 0;
}
