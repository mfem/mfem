#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_interpolate_gradient_linear_scalar_3d(std::string mesh_file,
                                               int refinements,
                                               int polynomial_order)
{
   constexpr int dim = 3;
   Mesh mesh_serial = Mesh(mesh_file);
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "wrong mesh dimension");

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

   // const IntegrationRule &ir =
   //    IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);
   IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = gll_rules.Get(h1fes.GetFE(0)->GetGeomType(),
                                             2 * polynomial_order - 1);

   ParGridFunction f1_g(&h1fes);

   ParametricSpace pspace(dim, dim, ir.GetNPoints(),
                          dim * ir.GetNPoints() * mesh.GetNE());
   ParametricFunction qdata(pspace);

   auto kernel = [](const tensor<double, dim> &dudxi,
                    const tensor<double, dim, dim> &J)
   {
      return mfem::tuple{dudxi * inv(J)};
   };

   mfem::tuple argument_operators = {Gradient{"potential"}, Gradient{"coordinates"}};
   mfem::tuple output_operator = {None{"qdata"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = mfem::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array
   {
      FieldDescriptor{&mesh_fes, "coordinates"},
      FieldDescriptor{&pspace, "qdata"}
   };

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      const double z = coords(2);
      return 2.345 + x * y * z + y * z;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(*f1_g.GetTrueDofs()), y(h1fes.TrueVSize() * dim);
   dop.SetParameters({mesh_nodes, &qdata});
   dop.Mult(x, y);

   Vector f_test(h1fes.GetElementRestriction(
                    ElementDofOrdering::LEXICOGRAPHIC)->Height() * dim);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);

      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector g(dim);
         f1_g.GetGradient(*T, g);
         // printf("(%f, %f, %f): (%f, %f, %f)\n", ip.x, ip.y, ip.z,  g(0), g(1), g(2));
         for (int d = 0; d < dim; d++)
         {
            int qpo = qp * dim;
            int eo = e * (ir.GetNPoints() * dim);
            f_test(d + qpo + eo) = g(d);
         }
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

DFEM_TEST_MAIN(test_interpolate_gradient_linear_scalar_3d);
