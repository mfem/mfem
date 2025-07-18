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
   int qorder = 6;
   int ref_levels = 0;
   int nrTest = 1000;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&qorder, "-qo", "--quad-order", "Quadrature order");
   args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.AddOption(&nrTest, "-n", "--test", "Number of tests to run");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   // Mesh mesh = dim == 2
   //             ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
   //             : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   Mesh mesh("../../data/mobius-strip.mesh", 1, 1);
   for (int i=0; i<ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   auto f = [](const Vector &x)->real_t
   {
      return exp(x[0]*M_PI)*sin(x[1]*M_PI)+1;
   };
   auto f_ad = [](Vector &, ad::ADVectorType &x,
                  ad::ADVectorType &f) -> void
   {
      f[0] = exp(x[0]*M_PI)*sin(x[1]*M_PI)+1;
   };

   auto df = [](const Vector &x, Vector &y) -> void
   {
      y.SetSize(2);
      y[0] = M_PI * exp(x[0]*M_PI)*sin(x[1]*M_PI);
      y[1] = M_PI * exp(x[0]*M_PI)*cos(x[1]*M_PI);
   };

   QuadratureSpace qspace(&mesh, qorder);
   H1_FECollection fec(order, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fespace(&mesh, &fec);
   std::vector<FiniteElementSpace*> fes({&fespace});

   Array<int> offsets({0, qspace.GetSize(), fespace.GetTrueVSize()});
   offsets.PartialSum();
   BlockVector x(offsets);
   QuadratureFunction qf(&qspace, x.GetBlock(0).GetData());
   GridFunction gf(&fespace);
   gf.MakeTRef(&fespace, x.GetBlock(1).GetData());

   FunctionCoefficient coord0([](const Vector &x) { return x[0]; });
   FunctionCoefficient coord1([](const Vector &x) { return x[1]; });
   coord0.Project(qf);
   gf.ProjectCoefficient(coord1);
   gf.SetTrueVector();

   Array<int> space_idx({-1, 0});
   ComposedFunctional func(f, df, qspace, fes, space_idx);
   ComposedADFunctional<2> adfunc(f_ad, qspace, fes, space_idx);

   Vector y(1), y_ad(1);
   BlockVector g(offsets), g_ad(offsets);

   func.Mult(x, y);
   func.GetGradient().Mult(x, g);
   adfunc.Mult(x, y_ad);
   adfunc.GetGradient().Mult(x, g_ad);

   QuadratureFunction gf_qf(qspace);
   gf_qf.ProjectGridFunction(gf);
   std::vector<QuadratureFunction> qf_out({QuadratureFunction(qspace), QuadratureFunction(qspace)});
   FunctionCoefficient result_cf(f);
   VectorArrayCoefficient qf_gf_cf(2);
   qf_gf_cf.Set(0, new QuadratureFunctionCoefficient(qf));
   qf_gf_cf.Set(1, new GridFunctionCoefficient(&gf));
   QuadratureFunction qf_all(qspace, 2);
   qf_gf_cf.Project(qf_all);
   Vector all_point(2);
   Vector grad_point(2);
   real_t result = 0.0;
   for (int i=0; i<qspace.GetSize(); i++)
   {
      all_point.MakeRef(qf_all, i*2);
      result += f(all_point)*qspace.GetWeights()[i];
      df(all_point, grad_point);
      all_point = grad_point;
      qf_out[0][i] = grad_point[0];
      qf_out[1][i] = grad_point[1];
   }

   out << "Functional value: " << y[0] << endl;
   out << "Expected value: " << result << endl;
   out << "Difference: " << std::abs(result - y[0]) << endl;
   out << "Difference(AD): " << std::abs(result - y_ad[0]) << endl;

   qf_out[0] *= qspace.GetWeights();
   out << "Gradient diff [0]: " << g.GetBlock(0).DistanceTo(
          qf_out[0]) << std::endl;
   out << "Gradient diff (AD) [0]: " << g_ad.GetBlock(0).DistanceTo(
          qf_out[0]) << std::endl;
   LinearForm lf(&fespace);
   QuadratureFunctionCoefficient qf_out1_cf(qf_out[1]);
   lf.AddDomainIntegrator(new QuadratureLFIntegrator(qf_out1_cf));
   lf.Assemble();
   out << "Gradient diff [1]: " << g.GetBlock(1).DistanceTo(lf) << std::endl;
   out << "Gradient diff (AD) [1]: " << g_ad.GetBlock(1).DistanceTo(
          lf) << std::endl;
}
