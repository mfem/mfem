#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_neo_hookean_elasticity_2d(
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
   MFEM_ASSERT(dim == 2, "This test is for 2D meshes only");
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction*>(mesh.GetNodes());
   ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, dim);

   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

   out << "#qp: " << ir.GetNPoints() << "\n";

   ParGridFunction u_g(&h1fes);

   auto kernel = [] MFEM_HOST_DEVICE(const tensor<double, 2, 2>& J,
                                     const double& w,
                                     const tensor<double, 2, 2>& dudxi)
   {
      // Neo-Hookean parameters
      const double lambda = 1.0;
      const double mu = 0.5;

      static constexpr auto I = mfem::internal::IsotropicIdentity<2>();
      auto F = I + (dudxi * inv(J));
      auto E = 0.5 * (transpose(F) * F - I);
      auto invF = inv(F);

      // 2D plane strain formulation
      auto P = mu * (F - transpose(invF)) + lambda * log(det(F)) * transpose(invF);

      return mfem::tuple{P * det(J) * w};
   };

   mfem::tuple argument_operators = {Gradient{"coordinates"}, Weight{},
                                     Gradient{"displacement"}
                                    };
   mfem::tuple output_operator = {Gradient{"displacement"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = mfem::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "displacement"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto displacement = [](const Vector& coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = 0.1 * x * y;
      u(1) = 0.1 * y * x;
   };

   VectorFunctionCoefficient disp_coeff(2, displacement);
   u_g.ProjectCoefficient(disp_coeff);

   Vector x(u_g), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);
   y.HostRead();

   // Test linearization
   auto dFdu = dop.GetDerivativeWrt<0>({&u_g}, {mesh_nodes});
   dFdu->Mult(x, y);

   // Finite difference Jacobian test
   {
      double eps = 1.0e-6;
      Vector v(x), xpv(x), xmv(x), fxpv(x.Size()), fxmv(x.Size());
      v *= eps;
      xpv += v;
      xmv -= v;
      dop.Mult(xpv, fxpv);
      dop.Mult(xmv, fxmv);
      fxpv -= fxmv;
      fxpv /= (2.0*eps);

      fxpv -= y;
      if (fxpv.Norml2() > eps)
      {
         out << "||dFdu_FD u^* - ex||_l2 = " << fxpv.Norml2() << "\n";
         return 1;
      }
   }

   return 0;
}

DFEM_TEST_MAIN(test_neo_hookean_elasticity_2d);
