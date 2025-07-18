//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "remap.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int dim = 2;
   int order = 3;
   int ref_levels = 0;
   int nrTest = 1000;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.AddOption(&nrTest, "-n", "--test", "Number of tests to run");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh = dim == 2
               ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   // Mesh ser_mesh("../../data/mobius-strip.mesh", 1, 1);
   for (int i=0; i<ref_levels; i++)
   {
      mesh.UniformRefinement();
   }


   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   L2_FECollection fec(order, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fespace(&mesh, &fec);
   out << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   FunctionCoefficient cf([](const Vector &x)
   { return std::sin(x[0]*M_PI)*std::sin(x[1]*M_PI); });

   QuadratureSpace qspace(&mesh, order*2);
   QuadratureFunction qf(qspace);
   cf.Project(qf);

   QuadratureFunctionCoefficient qf_cf(qf);
   LinearForm lf(&fespace);
   lf.AddDomainIntegrator(new QuadratureLFIntegrator(qf_cf));
   tic();
   for (int i=0; i<nrTest; i++)
   {
      lf.Assemble();
   }
   real_t lf_time = toc();

   QuadratureLinearForm qlf(qspace, fespace);
   Vector qlf_vec(fespace.GetTrueVSize());
   tic();
   for (int i=0; i<nrTest; i++)
   {
      qlf.Mult(qf, qlf_vec);
   }
   real_t qlf_time = toc();

   out << "QuadratureLFIntegrator time: " << lf_time*1e6/nrTest << " micro sec" <<
        endl;
   out << "QuadratureLinearForm time: " << qlf_time*1e6/nrTest << " micro sec" <<
        endl;
   out << "Speedup: " << lf_time/qlf_time << endl;
   real_t err = qlf_vec.DistanceTo(lf);
   out << "Error: " << err << std::endl;
   return 0;
}
