#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_diffusion(
   std::string mesh_file, int refinements, int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   MFEM_ASSERT(mesh_serial.Dimension() == 2, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction*>(mesh.GetNodes());
   ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

   out << "#qp: " << ir.GetNPoints() << "\n";

   ParGridFunction f1_g(&h1fes);
   ParGridFunction rho_g(&h1fes);

   constexpr int Potential = 0;
   constexpr int Coordinates = 1;

   auto kernel = [] MFEM_HOST_DEVICE(
                    const tensor<double, 2, 2>& J,
                    const double& w, const tensor<double, 2>& dudxi)
   {
      auto invJ = inv(J);
      return mfem::tuple{dudxi * invJ * transpose(invJ) * det(J) * w};
   };

   mfem::tuple argument_operators =
   {
      Gradient<Coordinates>{}, Weight{}, Gradient<Potential>{}
   };
   mfem::tuple output_operator = {Gradient<Potential>{}};

   auto solutions = std::vector{FieldDescriptor{Potential, &h1fes}};
   auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes}};

   DifferentiableOperator dop(solutions, parameters, mesh);

   auto f1 = [](const Vector& coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + 0.25 * x * x * y + y * y * x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);
   y.HostRead();

   ParBilinearForm a(&h1fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   a.Finalize();

   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();
   Vector diff(y2);
   diff -= y;
   if (diff.Norml2() > 1e-10)
   {
      print_vector(diff);
      print_vector(y2);
      print_vector(y);
      return 1;
   }

   // // Test linearization here as well
   // auto dFdu = dop.GetDerivativeWrt<0>({&f1_g}, {mesh_nodes});

   // if (dFdu->Height() != h1fes.GetTrueVSize())
   // {
   //    out << "dFdu unexpected height of " << dFdu->Height() << "\n";
   //    return 1;
   // }

   // dFdu->Mult(x, y);
   // y.HostRead();
   // a.Mult(x, y2);
   // y2.HostRead();

   // diff = y2;
   // diff -= y;
   // if (diff.Norml2() > 1e-10)
   // {
   //    print_vector(diff);
   //    print_vector(y2);
   //    print_vector(y);
   //    return 1;
   // }

   // // fd jacobian test
   // {
   //    double eps = 1.0e-6;
   //    Vector v(x), xpv(x), xmv(x), fxpv(x.Size()), fxmv(x.Size());
   //    v *= eps;
   //    xpv += v;
   //    xmv -= v;
   //    dop.Mult(xpv, fxpv);
   //    dop.Mult(xmv, fxmv);
   //    fxpv -= fxmv;
   //    fxpv /= (2.0*eps);

   //    fxpv -= y;
   //    if (fxpv.Norml2() > eps)
   //    {
   //       out << "||dFdu_FD u^* - ex||_l2 = " << fxpv.Norml2() << "\n";
   //       return 1;
   //    }
   // }

   // f1_g.ProjectCoefficient(f1_c);
   // rho_g.ProjectCoefficient(rho_c);
   // auto dFdrho = dop.GetDerivativeWrt<1>({&f1_g}, {&rho_g, mesh_nodes});
   // if (dFdrho->Height() != h1fes.GetTrueVSize())
   // {
   //    out << "dFdrho unexpected height of " << dFdrho->Height() << "\n";
   //    return 1;
   // }

   // dFdrho->Mult(rho_g, y);

   // // fd test
   // {
   //    double eps = 1.0e-6;
   //    Vector v(rho_g), rhopv(rho_g), rhomv(rho_g), frhopv(x.Size()),
   //    frhomv(x.Size()); v *= eps; rhopv += v; rhomv -= v;
   //    dop.SetParameters({&rhopv, mesh_nodes});
   //    dop.Mult(x, frhopv);
   //    dop.SetParameters({&rhomv, mesh_nodes});
   //    dop.Mult(x, frhomv);
   //    frhopv -= frhomv;
   //    frhopv /= (2.0*eps);

   //    frhopv -= y;
   //    if (frhopv.Norml2() > eps)
   //    {
   //       out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
   //       return 1;
   //    }
   // }

   return 0;
}

DFEM_TEST_MAIN(test_diffusion);
