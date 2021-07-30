//                            MFEM Example 30
//
// Compile with: make ex30
//
// Sample runs:  ex30 -m ../data/square-disc.mesh -o 1
//               ex30 -m ../data/square-disc.mesh -o 2
//               ex30 -m ../data/square-disc-nurbs.mesh -o 2  ???
//               ex30 -m ../data/star.mesh -o 3
//               ex30 -m ../data/escher.mesh -o 2  ???
//               ex30 -m ../data/fichera.mesh -o 2 !!!
//               ex30 -m ../data/disc-nurbs.mesh -o 2
//               ex30 -m ../data/ball-nurbs.mesh
//               ex30 -m ../data/pipe-nurbs.mesh
//               ex30 -m ../data/star-surf.mesh -o 2 ???
//               ex30 -m ../data/square-disc-surf.mesh -o 2  ???
//               ex30 -m ../data/amr-quad.mesh
//
// Description:  This is an example of adaptive mesh refinement preprocessing
//               which lowers the data oscillation [1] to a user-defined
//               relative threshold. There is no PDE being solved.
//
//               MFEM's capability to work with both conforming and 
//               nonconforming meshes is demonstrated in example 6. In some
//               problems, the material data or loading data is not sufficiently
//               resolved on the initial mesh. This missing fine scale data 
//               reduces the accuracy of the solution as well as the accuracy
//               of some local error estimators. By preprocessing the mesh
//               before the solving the PDE, many issues can be avoided.
//
//               [1] Morin, P., Nochetto, R. H., & Siebert, K. G. (2000). 
//                   Data oscillation and convergence of adaptive FEM. SIAM
//                   Journal on Numerical Analysis, 38(2), 466-488.


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double function0(const Vector &p)
{
   double x = p(0), y = p(1);
   return 1.0 + x + y;
}

double function1(const Vector &p)
{
   if (p.Normlp(2.0) > 0.4 && p.Normlp(2.0) < 0.6) { return 1; }
   if (p.Normlp(2.0) < 0.4 || p.Normlp(2.0) > 0.6) { return 2; }
   return 0;
}

double function2(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   double xc = 0.75, yc = 0.5;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   double num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   double denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = max(denom,1e-8);
   return num / denom;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int nc_limit = 1;
   int max_elems = 1e5;
   bool visualization = true;
   double osc_threshold = 1e-3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nc_limit, "-l", "--nc-limit",
                  "Maximum level of hanging nodes.");
   args.AddOption(&max_elems, "-me", "--max-elems",
                  "Stop after reaching this many elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&osc_threshold, "-e", "--error",
                  "relative data oscillation threshold");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNCMesh();

   // // 2. Since a NURBS mesh can currently only be refined uniformly, we need to
   // //    convert it to a piecewise-polynomial curved mesh. First we refine the
   // //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   // if (mesh.NURBSext)
   // {
   //    for (int i = 0; i < 2; i++)
   //    {
   //       mesh.UniformRefinement();
   //    }
   //    mesh.SetCurvature(2);
   // }

   // 2. Define functions and refiner.
   FunctionCoefficient coeff0(function0);
   FunctionCoefficient coeff1(function1);
   FunctionCoefficient coeff2(function2);
   CoefficientRefiner coeffrefiner(order);

   // 2. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + 5;
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   // 3. Preprocess mesh to control osc
   coeffrefiner.SetCoefficient(coeff0);
   coeffrefiner.PreprocessMesh(mesh);

   mfem::out  << "\n";
   mfem::out << "Function 0 (affine) \n";
   mfem::out << "Number of Elements " << mesh.GetNE() << "\n";
   mfem::out << "Osc error " << coeffrefiner.GetOsc() << "\n";
   mfem::out  << "\n";

   // coeffrefiner.SetIntRule(irs);
   coeffrefiner.SetMaxElements( (long) max_elems);
   coeffrefiner.SetThreshold(osc_threshold);
   coeffrefiner.SetNCLimit(nc_limit);

   coeffrefiner.SetCoefficient(coeff1);
   coeffrefiner.PreprocessMesh(mesh);

   mfem::out << "Function 1 (discontinuous) \n";
   mfem::out << "Number of Elements " << mesh.GetNE() << "\n";
   mfem::out << "Osc error " << coeffrefiner.GetOsc() << "\n";
   mfem::out  << "\n";

   coeffrefiner.SetCoefficient(coeff2);
   coeffrefiner.PreprocessMesh(mesh);

   mfem::out << "Function 2 (singular) \n";
   mfem::out << "Number of Elements " << mesh.GetNE() << "\n";
   mfem::out << "Osc error " << coeffrefiner.GetOsc() << "\n";

   sol_sock.precision(8);
   sol_sock << "mesh\n" << mesh << flush;


   return 0;
}
