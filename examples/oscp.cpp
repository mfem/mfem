//                                MFEM Example 6
//
// Compile with: make ex6
//
// Sample runs:  ex6 -m ../data/square-disc.mesh -o 1
//               ex6 -m ../data/square-disc.mesh -o 2
//               ex6 -m ../data/square-disc-nurbs.mesh -o 2
//               ex6 -m ../data/star.mesh -o 3
//               ex6 -m ../data/escher.mesh -o 2
//               ex6 -m ../data/fichera.mesh -o 2
//               ex6 -m ../data/disc-nurbs.mesh -o 2
//               ex6 -m ../data/ball-nurbs.mesh
//               ex6 -m ../data/pipe-nurbs.mesh
//               ex6 -m ../data/star-surf.mesh -o 2
//               ex6 -m ../data/square-disc-surf.mesh -o 2
//               ex6 -m ../data/amr-quad.mesh
//               ex6 -m ../data/inline-segment.mesh -o 1 -md 100
//
// Device sample runs:
//               ex6 -pa -d cuda
//               ex6 -pa -d occa-cuda
//               ex6 -pa -d raja-omp
//               ex6 -pa -d ceed-cpu
//             * ex6 -pa -d ceed-cuda
//               ex6 -pa -d ceed-cuda:/gpu/cuda/shared
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
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double wavefront_exsol(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   // double xc = -0.05, yc = -0.05;
   double xc = 0.0, yc = 0.0;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   return atan(alpha * (r - r0));
}

void wavefront_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   double xc = -0.05, yc = -0.05;
   double r0 = 0.7;
   grad(0) = 0.0;
   grad(1) = 0.0;
}

double wavefront_laplace(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   // double xc = -0.05, yc = -0.05;
   double xc = 0.0, yc = 0.0;
   double r0 = 0.7;
   double r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   double num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   double denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = max(denom,1e-8);
   // return num / denom;
   if (p.Normlp(2.0) > 0.4 && p.Normlp(2.0) < 0.6) { return 1; }
   if (p.Normlp(2.0) < 0.4 || p.Normlp(2.0) > 0.6) { return 2; }
   return 0;
}

double wavefront_laplace_alt(const Vector &p)
{
   double x = p(0), y = p(1);
   double alpha = 1000.0;
   double xc = -0.5, yc = -0.5;
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
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   const char *device_config = "cpu";
   int max_dofs = 50000;
   bool visualization = true;
   int nc_limit = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
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

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   mesh.EnsureNCMesh();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Coefficient * exsol = nullptr;
   Coefficient * rhs = nullptr;
   exsol = new FunctionCoefficient(wavefront_exsol);
   rhs = new FunctionCoefficient(wavefront_laplace);


   // 8. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Connect to GLVis.
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

   // 11.5. Preprocess mesh to control osc
   double osc_tol = 1e-3;
   CoefficientRefiner coeffrefiner(0);
   coeffrefiner.SetCoefficient(*rhs);
   coeffrefiner.SetIntRule(irs);
   coeffrefiner.SetThreshold(osc_tol);
   coeffrefiner.SetNCLimit(0);
   coeffrefiner.PreprocessMesh(pmesh);

   // Coefficient * rhs2 = nullptr;
   // rhs2 = new FunctionCoefficient(wavefront_laplace_alt);
   // coeffrefiner.SetCoefficient(*rhs2);
   // coeffrefiner.PreprocessMesh(pmesh);


   cout << "Number of Elements " << pmesh.GetGlobalNE() << endl;
   cout << "Osc error " << coeffrefiner.GetOsc() << endl;


   sol_sock.precision(8);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock << "mesh\n" << pmesh << flush;


   MPI_Finalize();
   return 0;
}
