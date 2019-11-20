#include "mfem.hpp"
#include "catch.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

namespace pa_kernels
{

double test_nl_convection_nd(int dim)
{
   Mesh *mesh;
   if (dim == 2)
   {
      mesh = new Mesh(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   }
   if (dim == 3)
   {
      mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0);
   }

   int order = 2;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, dim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(3);

   NonlinearForm nlf_fa(&fes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_fa.Mult(x, y_fa);

   NonlinearForm nlf_pa(&fes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_pa.Setup();
   nlf_pa.Mult(x, y_pa);

   y_fa -= y_pa;
   double difference = y_fa.Norml2();

   delete mesh;

   return difference;
}

TEST_CASE("Nonlinear Convection", "[PartialAssembly], [NonlinearPA]")
{
   SECTION("2D")
   {
      REQUIRE(test_nl_convection_nd(2) == Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_nl_convection_nd(3) == Approx(0.0));
   }
}

}
