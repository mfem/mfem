//                       MFEM Example 30 - Parallel Version
//
// Compile with: make ex30p
//
// Sample runs:  mpirun -np 4 ex30p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex30p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex30p -m ../data/square-disc.mesh -o 2 -me 1e+4
//               mpirun -np 4 ex30p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex30p -m ../data/star.mesh -o 2 -eo 4
//               mpirun -np 4 ex30p -m ../data/fichera.mesh -o 2 -me 1e+5 -e 5e-2
//               mpirun -np 4 ex30p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex30p -m ../data/ball-nurbs.mesh -o 2 -eo 3 -e 5e-2 -me 1e+5
//               mpirun -np 4 ex30p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex30p -m ../data/square-disc-surf.mesh -o 2
//               mpirun -np 4 ex30p -m ../data/amr-quad.mesh -l 2
//
// Description:  This is an example of adaptive mesh refinement preprocessing
//               which lowers the data oscillation [1] to a user-defined
//               relative threshold. There is no PDE being solved.
//
//               MFEM's capability to work with both conforming and
//               nonconforming meshes is demonstrated in example 6. In some
//               problems, the material data or loading data is not sufficiently
//               resolved on the initial mesh. This missing fine scale data
//               reduces the accuracy of the solution as well as the accuracy of
//               some local error estimators. By preprocessing the mesh before
//               solving the PDE, many issues can be avoided.
//
//               [1] Morin, P., Nochetto, R. H., & Siebert, K. G. (2000). Data
//                   oscillation and convergence of adaptive FEM. SIAM Journal
//                   on Numerical Analysis, 38(2), 466-488.
//
//               [2] Mitchell, W. F. (2013). A collection of 2D elliptic
//                   problems for testing adaptive grid refinement algorithms.
//                   Applied mathematics and computation, 220, 350-364.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Piecewise-affine function which is sometimes mesh-conforming
real_t affine_function(const Vector &p)
{
   real_t x = p(0), y = p(1);
   if (x < 0.0)
   {
      return 1.0 + x + y;
   }
   else
   {
      return 1.0;
   }
}

// Piecewise-constant function which is never mesh-conforming
real_t jump_function(const Vector &p)
{
   if (p.Normlp(2.0) > 0.4 && p.Normlp(2.0) < 0.6)
   {
      return 1.0;
   }
   else
   {
      return 5.0;
   }
}

// Singular function derived from the Laplacian of the "steep wavefront" problem
// in [2].
real_t singular_function(const Vector &p)
{
   real_t x = p(0), y = p(1);
   real_t alpha = 1000.0;
   real_t xc = 0.75, yc = 0.5;
   real_t r0 = 0.7;
   real_t r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   real_t num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   real_t denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = std::max(denom, (real_t) 1.0e-8);
   return num / denom;
}

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int nc_limit = 1;
   int max_elems = 1e5;
   real_t double_max_elems = real_t(max_elems);
   bool visualization = true;
   bool nc_simplices = true;
   real_t osc_threshold = 1e-3;
   int enriched_order = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nc_limit, "-l", "--nc-limit",
                  "Maximum level of hanging nodes.");
   args.AddOption(&double_max_elems, "-me", "--max-elems",
                  "Stop after reaching this many elements.");
   args.AddOption(&osc_threshold, "-e", "--error",
                  "relative data oscillation threshold.");
   args.AddOption(&enriched_order, "-eo", "--enriched_order",
                  "Enriched quadrature order.");
   args.AddOption(&nc_simplices, "-ns", "--nonconforming-simplices",
                  "-cs", "--conforming-simplices",
                  "For simplicial meshes, enable/disable nonconforming"
                  " refinement");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   max_elems = int(double_max_elems);
   Mesh mesh(mesh_file, 1, 1);

   // 2. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }

   // 3. Make sure the mesh is in the non-conforming mode to enable local
   //    refinement of quadrilaterals/hexahedra. Simplices can be refined either
   //    in conforming or in non-conforming mode. The conforming mode however
   //    does not support dynamic partitioning.
   mesh.EnsureNCMesh(nc_simplices);

   // 4. Define a parallel mesh by partitioning the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define functions and refiner.
   FunctionCoefficient affine_coeff(affine_function);
   FunctionCoefficient jump_coeff(jump_function);
   FunctionCoefficient singular_coeff(singular_function);
   CoefficientRefiner  coeffrefiner(affine_coeff,order);

   // 6. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 7. Define custom integration rule (optional).
   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + enriched_order;
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   // 8. Apply custom refiner settings.
   coeffrefiner.SetIntRule(irs);
   coeffrefiner.SetMaxElements(max_elems);
   coeffrefiner.SetThreshold(osc_threshold);
   coeffrefiner.SetNCLimit(nc_limit);
   coeffrefiner.PrintWarnings();

   // 9. Preprocess mesh to control osc (piecewise-affine function). This is
   //    mostly just a verification check. The oscillation should be zero if the
   //    function is mesh-conforming and order > 0.
   coeffrefiner.PreprocessMesh(pmesh);

   int globalNE = pmesh.GetGlobalNE();
   real_t osc = coeffrefiner.GetOsc();
   if (myid == 0)
   {
      mfem::out << "\n";
      mfem::out << "Function 0 (affine) \n";
      mfem::out << "Number of Elements " << globalNE << "\n";
      mfem::out << "Osc error " << osc << "\n";
   }

   // 10. Preprocess mesh to control osc (jump function).
   coeffrefiner.ResetCoefficient(jump_coeff);
   coeffrefiner.PreprocessMesh(pmesh);

   globalNE = pmesh.GetGlobalNE();
   osc = coeffrefiner.GetOsc();
   if (myid == 0)
   {
      mfem::out << "\n";
      mfem::out << "Function 1 (discontinuous) \n";
      mfem::out << "Number of Elements " << globalNE << "\n";
      mfem::out << "Osc error " << osc << "\n";
   }

   // 11. Preprocess mesh to control osc (singular function).
   coeffrefiner.ResetCoefficient(singular_coeff);
   coeffrefiner.PreprocessMesh(pmesh);

   globalNE = pmesh.GetGlobalNE();
   osc = coeffrefiner.GetOsc();
   if (myid == 0)
   {
      mfem::out << "\n";
      mfem::out << "Function 2 (singular) \n";
      mfem::out << "Number of Elements " << globalNE << "\n";
      mfem::out << "Osc error " << osc << "\n";
   }

   if (visualization)
   {
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "mesh\n" << pmesh << flush;
   }

   return 0;
}
