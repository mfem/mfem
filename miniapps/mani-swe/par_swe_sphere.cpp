//                                MFEM MINIAPP MANI-SWE
//
// Compile with: make swe
//
// Sample runs:  swe
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include "manihyp.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 1. Parse command-line options.
   int order = 1;
   bool visualization = true;
   int level_serial = 2;
   int level_parallel = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&level_serial, "-rs", "--refine",
                  "Refinement level");
   args.AddOption(&level_parallel, "-rp", "--refine-parallel",
                  "Parallel refinement level");
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

   std::unique_ptr<ParMesh> mesh(static_cast<ParMesh*>(sphericalMesh(1.0, Element::Type::TRIANGLE, order,
                                            level_serial, level_parallel)));
   L2_FECollection fec(order, 2);
   ParFiniteElementSpace fes_scalar(mesh.get(), &fec);
   ParFiniteElementSpace fes_sdim(mesh.get(), &fec, 2);
   ParFiniteElementSpace fes_vdim(mesh.get(), &fec, 2+1);

   ParGridFunction gf(&fes_scalar);
   gf = 1.0;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << gf << flush;
   }
   return 0;
}
