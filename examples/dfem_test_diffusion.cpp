#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <int dim = 2>
int test_diffusion(
   std::string mesh_file, int refinements, int polynomial_order)
{
   constexpr int num_samples = 1;
   Mesh mesh_serial = Mesh(mesh_file);
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction*>(mesh.GetNodes());
   ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   L2_FECollection l2fec(0, dim);
   ParFiniteElementSpace l2fes(&mesh, &l2fec);

   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(
                      0)->GetDim() - 1);

   printf("#ndof per el = %d\n", h1fes.GetFE(0)->GetDof());
   printf("#nqp = %d\n", ir.GetNPoints());
   printf("#q1d = %d\n", (int)floor(pow(ir.GetNPoints(), 1.0/dim) + 0.5));

   std::shared_ptr<ParametricSpace> qdata_space;
   if (mesh.GetElement(0)->GetType() == Element::QUADRILATERAL ||
       mesh.GetElement(0)->GetType() == Element::HEXAHEDRON)
   {
      qdata_space =
         std::make_shared<ParametricSpace>(
            dim, dim * dim, ir.GetNPoints(), dim * dim * ir.GetNPoints() * mesh.GetNE());
   }
   else
   {
      qdata_space =
         std::make_shared<ParametricSpace>(
            1, dim * dim, ir.GetNPoints(), dim * dim * ir.GetNPoints() * mesh.GetNE());
   }

   ParametricFunction qdata(*qdata_space);

   ParGridFunction f1_g(&h1fes);
   ParGridFunction rho_g(&l2fes);

   auto f1 = [](const Vector& coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      if (dim == 3)
      {
         const double z = coords(2);
         return 2.345 + x + x*y + 1.25 * z*x;
      }
      else
      {
         return x + x*y + 2.345;
      }
   };
   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   rho_g = 2.0;

   Vector x(f1_g);

   Vector y1(h1fes.GetTrueVSize());

   auto diffusion_mf_kernel =
      [] MFEM_HOST_DEVICE (
         const tensor<real_t, dim>& dudxi,
         const real_t& rho,
         const tensor<real_t, dim, dim>& J,
         const real_t& w)
   {
      auto invJ = inv(J);
      return mfem::tuple{(pow(rho, 3.0)*(dudxi * invJ)) * transpose(invJ) * det(J) * w};
   };

   constexpr int Potential = 3;
   constexpr int Diffusivity = 44;
   constexpr int Coordinates = 55;

   auto input_operators = mfem::tuple
   {
      Gradient<Potential>{},
      Value<Diffusivity>{},
      Gradient<Coordinates>{},
      Weight{}
   };
   auto output_operator = mfem::tuple{Gradient<Potential>{}};

   auto solutions = std::vector
   {
      FieldDescriptor{Potential, &h1fes}
   };
   auto parameters = std::vector
   {
      FieldDescriptor{Diffusivity, &l2fes},
      FieldDescriptor{Coordinates, &mesh_fes}
   };

   DifferentiableOperator dop(solutions, parameters, mesh);
   auto derivatives = std::integer_sequence<size_t, Potential, Diffusivity> {};
   Array<int> domain_attributes(mesh.attributes.Size());
   domain_attributes = 1;
   dop.AddDomainIntegrator(
      diffusion_mf_kernel, input_operators, output_operator, ir, domain_attributes,
      derivatives);

   dop.SetParameters({&rho_g, mesh_nodes});
   StopWatch sw;
   sw.Start();
   for (int i = 0; i < num_samples; i++)
   {
      dop.Mult(x, y1);
   }
   sw.Stop();
   printf("dfem mf:       %fs\n", sw.RealTime() / num_samples);
   y1.HostRead();

   auto dfdp = dop.GetDerivative(Diffusivity, {&f1_g}, {&rho_g, mesh_nodes});

   dfdp->Mult(rho_g, y1);

   // printf("y1: ");
   // print_vector(y1);

   {
      // Create a direction vector for rho
      Vector dir(rho_g);

      // Small parameter for finite difference
      double eps = 1.0e-6;

      // Compute f(rho + eps*dir)
      Vector rho_plus(rho_g);
      rho_plus.Add(eps, dir);
      dop.SetParameters({&rho_plus, mesh_nodes});
      Vector f_plus(x.Size());
      dop.Mult(x, f_plus);

      // Compute f(rho - eps*dir)
      Vector rho_minus(rho_g);
      rho_minus.Add(-eps, dir);
      dop.SetParameters({&rho_minus, mesh_nodes});
      Vector f_minus(x.Size());
      dop.Mult(x, f_minus);

      // Finite difference approximation of the derivative action
      Vector fd_result(x.Size());
      subtract(f_plus, f_minus, fd_result);
      fd_result *= 1.0/(2.0*eps);

      // printf("fd: ");
      // print_vector(fd_result);

      fd_result -= y1;
      double absolute_error = fd_result.Norml2();
      double relative_error = absolute_error / y1.Norml2();
      out << "Absolute error ||dFdrho_FD * rho - dfem||_l2 = " << absolute_error <<
          "\n";
      out << "Relative error ||dFdrho_FD * rho - dfem||_l2 / ||dfem||_l2 = " <<
          relative_error << "\n";      // if (frhopv.Norml2() > eps)
      // {
      //    out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
      //    return 1;
      // }
   }

   return 0;
}

DFEM_TEST_MAIN(test_diffusion<2>);
