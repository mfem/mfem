#include "dfem/dfem_refactor.hpp"
#include "linalg/hypre.hpp"

using namespace mfem;
using mfem::internal::tensor;
using mfem::internal::dual;

int test_diffusion_integrator(std::string mesh_file,
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
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

   ParGridFunction f1_g(&h1fes);
   ParGridFunction rho_g(&h1fes);

   auto rho_f = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return x + y;
   };

   FunctionCoefficient rho_c(rho_f);
   rho_g.ProjectCoefficient(rho_c);

   auto kernel = [](const tensor<dual<double, double>, 2> &grad_u,
                    const dual<double, double> &rho,
                    const tensor<double, 2, 2> &J,
                    const double &w)
   {
      auto invJ = inv(J);
      return std::tuple{rho*rho * grad_u * invJ * transpose(invJ) * det(J) * w};
   };

   std::tuple argument_operators = {Gradient{"potential"}, Value{"density"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator = {Gradient{"potential"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = std::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array
   {
      FieldDescriptor{&h1fes, "density"},
      FieldDescriptor{&mesh_fes, "coordinates"}
   };

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + 0.25 * x*x*y + y*y*x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g), y(h1fes.TrueVSize());
   dop.SetParameters({&rho_g, mesh_nodes});
   dop.Mult(x, y);

   ParBilinearForm a(&h1fes);
   TransformedCoefficient rho_c2(&rho_c, [](double c) {return c*c;});
   a.AddDomainIntegrator(new DiffusionIntegrator(rho_c2));
   a.Assemble();
   a.Finalize();

   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2 -= y;
   if (y2.Norml2() > 1e-10)
   {
      out << "||F(u) - ex||_l2 = " << y2.Norml2() << "\n";
      return 1;
   }

   // Test linearization here as well
   auto dFdu = dop.GetDerivativeWrt<0>({&f1_g}, {&rho_g, mesh_nodes});

   // HypreParMatrix A;
   // dFdu->Assemble(A);

   if (dFdu->Height() != h1fes.GetTrueVSize())
   {
      out << "dFdu unexpected height of " << dFdu->Height() << "\n";
      return 1;
   }

   dFdu->Mult(x, y);
   a.Mult(x, y2);
   y2 -= y;
   if (y2.Norml2() > 1e-10)
   {
      out << "||dFdu u^* - ex||_l2 = " << y2.Norml2() << "\n";
      return 1;
   }

   // fd jacobian test
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

   f1_g.ProjectCoefficient(f1_c);
   rho_g.ProjectCoefficient(rho_c);
   auto dFdrho = dop.GetDerivativeWrt<1>({&f1_g}, {&rho_g, mesh_nodes});
   if (dFdrho->Height() != h1fes.GetTrueVSize())
   {
      out << "dFdrho unexpected height of " << dFdrho->Height() << "\n";
      return 1;
   }

   dFdrho->Mult(rho_g, y);

   // fd test
   {
      double eps = 1.0e-6;
      Vector v(rho_g), rhopv(rho_g), rhomv(rho_g), frhopv(x.Size()), frhomv(x.Size());
      v *= eps;
      rhopv += v;
      rhomv -= v;
      dop.SetParameters({&rhopv, mesh_nodes});
      dop.Mult(x, frhopv);
      dop.SetParameters({&rhomv, mesh_nodes});
      dop.Mult(x, frhomv);
      frhopv -= frhomv;
      frhopv /= (2.0*eps);

      frhopv -= y;
      if (frhopv.Norml2() > eps)
      {
         out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
         return 1;
      }
   }

   return 0;
}

int test_qoi(std::string mesh_file,
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
   ParFiniteElementSpace h1fes(&mesh, &h1fec, dim);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

   ParGridFunction rho_g(&h1fes);

   auto rho_f = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = x + y;
      u(1) = x + y;
   };

   VectorFunctionCoefficient rho_c(dim, rho_f);
   rho_g.ProjectCoefficient(rho_c);

   auto kernel = [](const tensor<dual<double, double>, 2> &rho,
                    const tensor<dual<double, double>, 2, 2> &drhodxi,
                    const tensor<double, 2, 2> &J,
                    const double &w)
   {
      const double eps = 1.2345;
      const auto drhodx = drhodxi * inv(J);
      return std::tuple{(0.5 * eps * dot(rho, rho) + ddot(drhodx, drhodx)) * det(J) * w};
   };

   std::tuple argument_operators = {Value{"density"}, Gradient{"density"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator = {One{"density"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = std::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "density"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   Vector x(rho_g), y(1);
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   // print_vector(y);

   auto dFdrho = dop.GetDerivativeWrt<0>({&rho_g}, {mesh_nodes});
   // Vector dFdrho_vec;
   // dFdrho->Assemble(dFdrho_vec);

   // print_vector(dFdrho_vec);

   // fd jacobian test
   {
      double eps = 1.0e-8;
      Vector v(x), fxpv(1), fxmv(1), dfdx(x.Size());
      for (int i = 0; i < x.Size(); i++)
      {
         v(i) += eps;
         dop.Mult(v, fxpv);
         v(i) -= 2.0 * eps;
         dop.Mult(v, fxmv);
         fxpv -= fxmv;
         fxpv /= (2.0*eps);
         dfdx(i) = fxpv(0);
      }

      // print_vector(dfdx);
      dfdx -= dFdrho_vec;
      if (dfdx.Norml2() > 1e-6)
      {
         out << "||dFdu_FD u^* - ex||_l2 = " << dfdx.Norml2() << "\n";
         return 1;
      }
   }

   return 0;
}

int main(int argc, char *argv[])
{
   Mpi::Init();

   std::cout << std::setprecision(9);

   const char *mesh_file = "../data/star.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.ParseCheck();

   out << std::setprecision(12);

   int ret;

   ret = test_diffusion_integrator(mesh_file,
                                   refinements,
                                   polynomial_order);
   out << "test_diffusion_integrator";
   ret ? out << " FAILURE\n" : out << " OK\n";

   ret = test_qoi(mesh_file, refinements, polynomial_order);
   out << "test_qoi";
   ret ? out << " FAILURE\n" : out << " OK\n";

   return 0;
}
