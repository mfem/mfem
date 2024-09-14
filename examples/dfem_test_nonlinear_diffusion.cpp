#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_nonlinear_diffusion(
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

   bool inactive_derivative = false;

   auto kernel = [] MFEM_HOST_DEVICE(
                    const tensor<double, 2, 2>& J,
                    const double& w,
                    const tensor<double, 2>& dudxi,
                    const double& u)
   {
      auto invJ = inv(J);
      return serac::tuple{(u * u) * dudxi * invJ * transpose(invJ) * det(J) * w};
   };

   serac::tuple argument_operators =
   {
      Gradient{"coordinates"},
      Weight{},
      Gradient{"potential"},
      Value{"potential"}
   };

   serac::tuple output_operator =
   {
      Gradient{"potential"}
   };

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = serac::tuple{eop};

   auto solutions = std::array
   {
      FieldDescriptor{&h1fes, "potential"}
   };
   auto parameters = std::array
   {
      FieldDescriptor{&mesh_fes, "coordinates"}
   };

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

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
   GridFunctionCoefficient f1gc(&f1_g);
   TransformedCoefficient tf_c(&f1gc, [](double f) { return f * f; });
   a.AddDomainIntegrator(new DiffusionIntegrator(tf_c));
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   a.Finalize();

   Vector y2(h1fes.TrueVSize()), diff(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();
   diff = y2;
   diff -= y;
   if (diff.Norml2() > 1e-10)
   {
      out << "||F(u) - ex||_l2 = " << diff.Norml2() << "\n";
      print_vector(diff);
      print_vector(y);
      print_vector(y2);
      return 1;
   }

   // Test linearization here as well
   auto dFdu = dop.GetDerivativeWrt<0>({&f1_g}, {mesh_nodes});
   dFdu->Mult(x, y);

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

   // ParBilinearForm da(&h1fes);
   // TransformedCoefficient dtf_c(&f1gc, [](double f) { return 2.0 * f; });
   // da.AddDomainIntegrator(new DiffusionIntegrator(dtf_c));
   // da.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // da.Assemble();
   // da.Finalize();

   // if (dFdu->Height() != h1fes.GetTrueVSize())
   // {
   //    out << "dFdu unexpected height of " << dFdu->Height() << "\n";
   //    return 1;
   // }

   // dFdu->Mult(x, y);
   // print_vector(y);
   // da.Mult(x, y2);
   // print_vector(y2);
   // y2 -= y;
   // out << "||dFdu x - A x||_l2 = " << y2.Norml2() << "\n";
   // if (y2.Norml2() > 1e-10)
   // {
   //    out << "||dFdu u^* - ex||_l2 = " << y2.Norml2() << "\n";
   // }

   return 0;
}

DFEM_TEST_MAIN(test_nonlinear_diffusion);
