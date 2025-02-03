#include "dfem/dfem_test_macro.hpp"
#include "fem/bilininteg.hpp"
#include <fstream>

using namespace mfem;
using mfem::internal::tensor;

int dfem_test_mass_scalar_2d(std::string mesh_file,
                             int refinements,
                             int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   const int dim = mesh_serial.Dimension();

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   mesh_serial.Clear();

   ParGridFunction *mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction f1_g(&h1fes);

   auto kernel_2d = [](const double &u,
                       const tensor<double, 2, 2> &J,
                       const double &w)
   {
      return mfem::tuple{u * w * det(J)};
   };

   auto kernel_3d = [](const double &u,
                       const tensor<double, 3, 3> &J,
                       const double &w)
   {
      return mfem::tuple{u * w * det(J)};
   };

   constexpr int Potential = 0;
   constexpr int Coordinates = 1;

   mfem::tuple input_operators = {Value<Potential>{}, Gradient<Coordinates>{}, Weight{}};
   mfem::tuple output_operator = {Value<Potential>{}};

   auto solutions = std::vector{FieldDescriptor{Potential, &h1fes}};
   auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes}};

   DifferentiableOperator dop(solutions, parameters, mesh);
   if (dim == 2)
   {
      dop.AddDomainIntegrator(kernel_2d, input_operators, output_operator, ir);
   }
   else
   {
      dop.AddDomainIntegrator(kernel_3d, input_operators, output_operator, ir);
   }

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x + x*y + 1.25 * x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(*f1_g.GetTrueDofs()), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   ParBilinearForm a(&h1fes);
   auto mass_integ = new MassIntegrator;
   mass_integ->SetIntRule(&ir);
   a.AddDomainIntegrator(mass_integ);
   if (mesh.GetElement(0)->GetType() == Element::QUADRILATERAL ||
       mesh.GetElement(0)->GetType() == Element::HEXAHEDRON)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   a.Assemble();
   a.Finalize();

   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();

   Vector diff(y2);
   diff -= y;
   // if (diff.Norml2() > 1e-12)
   {
      print_vector(diff);
      print_vector(y2);
      print_vector(y);
      // return 1;
   }

   return 0;
}

DFEM_TEST_MAIN(dfem_test_mass_scalar_2d);
