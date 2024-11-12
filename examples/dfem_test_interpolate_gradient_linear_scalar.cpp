#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_interpolate_gradient_linear_scalar(std::string mesh_file,
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

   mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   ParGridFunction *mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction f1_g(&h1fes);

   // ParametricSpace qdata_space(dim, 1, ir.GetNPoints(),
   //                             ir.GetNPoints() * mesh.GetNE());
   ParametricSpace qdata_space(1, dim, ir.GetNPoints(),
                               dim * ir.GetNPoints() * mesh.GetNE());
   ParametricFunction qdata(qdata_space);

   auto kernel_2d = [] MFEM_HOST_DEVICE (
                       const tensor<real_t, 2> &dudxi,
                       const tensor<real_t, 2, 2> &J)
   {
      out << "J: " << J << std::endl;
      out << "dudxi: " << dudxi << std::endl;
      out << inv(J) << std::endl;
      return mfem::tuple{dudxi * inv(J)};
   };

   auto kernel_3d = [] MFEM_HOST_DEVICE (
                       const tensor<real_t, 3> &dudxi,
                       const tensor<real_t, 3, 3> &J)
   {
      out << "J: " << J << std::endl;
      out << "dudxi: " << dudxi << std::endl;
      out << inv(J) << std::endl;
      return mfem::tuple{dudxi * inv(J)};
   };

   constexpr int Potential = 0;
   constexpr int Coordinates = 1;
   constexpr int Qdata = 2;

   auto input_operators = mfem::tuple{Gradient<Potential>{}, Gradient<Coordinates>{}};
   auto output_operator = mfem::tuple{None<Qdata>{}};

   auto solutions = std::vector{FieldDescriptor{Potential, &h1fes}};
   auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes},
                                 FieldDescriptor{Qdata, &qdata_space}};

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
      if (coords.Size() > 2)
      {
         const double z = coords(2);
         return 2.345 + x * y * z + y * z;
      }
      else
      {
         return 2.345 + x * y + y;
      }
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(*f1_g.GetTrueDofs());
   dop.SetParameters({mesh_nodes, &qdata});
   dop.Mult(x, qdata);

   Vector f_test(qdata.Size());
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
   diff -= qdata;
   if (diff.Norml2() > 1e-12)
   {
      print_vector(diff);
      print_vector(f_test);
      print_vector(qdata);
      return 1;
   }

   return 0;
}

DFEM_TEST_MAIN(test_interpolate_gradient_linear_scalar);
