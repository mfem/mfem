#include "dfem/dfem_test_macro.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "fem/bilininteg.hpp"
#include "fem/pbilinearform.hpp"
#include "fem/plinearform.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_objective(
   std::string mesh_file, int refinements, int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   2 * h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetDim() - 1);

   ParGridFunction f1_g(&h1fes);

   auto kernel = [] MFEM_HOST_DEVICE (
                    const real_t &u,
                    const tensor<real_t, 2> &dudxi,
                    const tensor<real_t, 2, 2> &J,
                    const real_t &w)
   {
      return mfem::tuple{(u * u * norm(dudxi * inv(J))) * w * det(J)};
   };

   constexpr int Potential = 0;
   constexpr int Coordinates = 1;

   auto input_operators = mfem::tuple{Value<Potential>{}, Gradient<Potential>{}, Gradient<Coordinates>{}, Weight{}};
   auto output_operator = mfem::tuple{One<Potential>{}};

   auto solutions = std::vector{FieldDescriptor{Potential, &h1fes}};
   auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes}};

   DifferentiableOperator dop(solutions, parameters, mesh);
   auto derivatives = std::integer_sequence<size_t, Potential> {};
   dop.AddDomainIntegrator(kernel, input_operators, output_operator, ir,
                           derivatives);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return x + y;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(*f1_g.GetTrueDofs());

   Vector y(1);

   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   out << "Objective value ∫ u^2 dx (u = x+y) on reference element:\n";
   print_vector(y);

   auto dfdp = dop.GetDerivative(Potential, {&f1_g}, {mesh_nodes});
   Vector dfdpv(1);
   x = 1.0;
   dfdp->Mult(x, dfdpv);

   out << "Derivative of the objective wrt u:\n";
   print_vector(dfdpv);

   Vector dfdp_vec;
   dfdp->AssembleVector(dfdp_vec);
   out << "dfdp:\n";
   print_vector(dfdp_vec);

   {
      x = *f1_g.GetTrueDofs();
      FDJacobian fd_jac(dop, x);
      x = 1.0;
      fd_jac.Mult(x, y);
      out << "fdjvp\n";
      print_vector(y);
      out << "FD Jacobian:\n";
      fd_jac.PrintMatlab(out);
   }

   return 0;
}

DFEM_TEST_MAIN(test_objective);
