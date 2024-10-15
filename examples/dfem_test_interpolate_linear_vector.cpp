#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_interpolate_linear_vector(std::string mesh_file, int refinements,
                                   int polynomial_order)
{
   constexpr int vdim = 2;
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
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   QuadratureSpace qspace(mesh, ir);
   QuadratureFunction qf(&qspace, vdim);

   ParGridFunction f1_g(&h1fes);

   auto kernel = [](const tensor<double, 2> &u)
   {
      return mfem::tuple{u};
   };

   mfem::tuple argument_operators = {Value{"potential"}};
   mfem::tuple output_operator = {None{"potential"}};

   ElementOperator eop{kernel, argument_operators, output_operator};
   auto ops = mfem::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = 2.345 + x + y;
      u(1) = 12.345 + x + y;
   };

   VectorFunctionCoefficient f1_c(vdim, f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g), y(f1_g.Size());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   Vector f_test(qf.Size());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation *T = mesh.GetElementTransformation(e);
      for (int qp = 0; qp < ir.GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir.IntPoint(qp);
         T->SetIntPoint(&ip);

         Vector f(vdim);
         f1_g.GetVectorValue(*T, ip, f);
         for (int d = 0; d < vdim; d++)
         {
            int qpo = qp * vdim;
            int eo = e * (ir.GetNPoints() * vdim);
            f_test(d + qpo + eo) = f(d);
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

DFEM_TEST_MAIN(test_interpolate_linear_vector);
