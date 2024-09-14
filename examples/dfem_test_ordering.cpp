#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"
#include "fem/coefficient.hpp"
#include "fem/pgridfunc.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_ordering(std::string mesh_file,
                  int refinements,
                  int polynomial_order)
{
   constexpr int dim = 2;
   constexpr int vdim = dim;

   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   const IntegrationRule &ir =
      IntRules.Get(mesh_fes.GetFE(0)->GetGeomType(),
                   2 * mesh_fes.FEColl()->GetOrder() - 1);

   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      out << "(" << ir.IntPoint(q).x << ", " << ir.IntPoint(q).y << ")\n";
   }

   ParGridFunction u(&mesh_fes);
   auto f = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = x*x*y + 1.0;
      u(1) = y*y*x*x + 2.0;
   };

   VectorFunctionCoefficient uc(dim, f);
   u.ProjectCoefficient(uc);

   auto kernel = [](const tensor<double, dim> &xi,
                    const tensor<double, vdim, dim> &J,
                    const tensor<double, dim> &u,
                    const tensor<double, vdim, dim> &dudxi)
   {
      out << "xi: " << xi << "\n";
      out << "J: " << J << "\n";
      out << "u: " << u << "\n";
      out << "dudxi: " << dudxi << "\n\n";
      return serac::tuple{J};
   };

   serac::tuple argument_operators{Value{"coordinates"}, Gradient{"coordinates"}, Value{"potential"}, Gradient{"potential"}};
   serac::tuple output_operator{Gradient{"potential"}};

   ElementOperator op{kernel, argument_operators, output_operator};

   std::array solutions{FieldDescriptor{&mesh_fes, "potential"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop{solutions, parameters, serac::tuple{op}, mesh, ir};

   Vector y(u);

   dop.SetParameters({mesh_nodes});
   dop.Mult(u, y);

   print_vector(y);

   return 0;
}

DFEM_TEST_MAIN(test_ordering);
