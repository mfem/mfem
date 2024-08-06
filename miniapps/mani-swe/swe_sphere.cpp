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
   // 1. Parse command-line options.
   int order = 1;
   bool visualization = true;
   int level = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&level, "-r", "--refine",
                  "Refinement level");
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

   std::unique_ptr<Mesh> mesh(sphericalMesh(1.0, Element::Type::TRIANGLE, order,
                                            level));
   L2_FECollection fec(order, 2);
   FiniteElementSpace fes_scalar(mesh.get(), &fec);
   FiniteElementSpace fes_sdim(mesh.get(), &fec, 2);
   FiniteElementSpace fes_vdim(mesh.get(), &fec, 2+1);

   GridFunction gf(&fes_scalar);
   gf = 1.0;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << gf << flush;
   }
   return 0;
}
