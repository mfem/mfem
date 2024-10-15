#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_interpolate_linear_scalar(std::string mesh_file,
                                   int refinements,
                                   int polynomial_order)
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
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction f1_g(&h1fes);

   auto kernel = [](const double &u, const tensor<double, 2, 2> &J,
                    const double &w)
   {
      return mfem::tuple{u};
   };

   mfem::tuple argument_operators = {Value{"potential"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple output_operator = {None{"potential"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = mfem::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x + y;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(*f1_g.GetTrueDofs()), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   Vector f_test(h1fes.GetElementRestriction(
                    ElementDofOrdering::LEXICOGRAPHIC)->Height());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         f_test((e * ir.GetNPoints()) + qp) = f1_c.Eval(*T, ip);
      }
   }

   Vector diff(f_test);
   diff -= y;
   if (diff.Norml2() > 1e-10)
   {
      print_vector(diff);
      print_vector(f_test);
      print_vector(y);
      return 1;
   }

   return 0;
}

DFEM_TEST_MAIN(test_interpolate_linear_scalar);
