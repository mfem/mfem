#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"
#include "examples/dfem/dfem_parametricspace.hpp"
#include "fem/bilininteg.hpp"
#include "general/tic_toc.hpp"

using namespace mfem;
using mfem::internal::tensor;

int test_diffusion_3d(
   std::string mesh_file, int refinements, int polynomial_order)
{
   constexpr int num_samples = 10;
   constexpr int dim = 3;
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


   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(
                      0)->GetDim() - 1);

   printf("#ndof per el = %d\n", h1fes.GetFE(0)->GetDof());
   printf("#nqp = %d\n", ir.GetNPoints());
   printf("#q1d = %d\n", (int)floor(pow(ir.GetNPoints(), 1.0/dim) + 0.5));

   ParametricSpace qdata_space(dim, dim * dim, ir.GetNPoints(),
                               dim * dim * ir.GetNPoints() * mesh.GetNE());
   ParametricFunction qdata(qdata_space);

   ParGridFunction f1_g(&h1fes);
   ParGridFunction rho_g(&h1fes);

   auto f1 = [](const Vector& coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      const double z = coords(2);
      return 2.345 + x + x*y + 1.25 * z*x;
   };
   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g);
   {
      auto diffusion_setup_kernel =
         [] MFEM_HOST_DEVICE (
            const tensor<double, dim, dim>& J,
            const double& w)
      {
         auto invJ = inv(J);
         return mfem::tuple{invJ * transpose(invJ) * det(J) * w};
      };

      mfem::tuple argument_operators = {Gradient{"coordinates"}, Weight{}};
      mfem::tuple output_operator = {None{"qdata"}};

      ElementOperator eop = {diffusion_setup_kernel, argument_operators, output_operator};
      auto ops = mfem::tuple{eop};

      auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
      auto parameters = std::array
      {
         FieldDescriptor{&mesh_fes, "coordinates"},
         FieldDescriptor{&qdata_space, "qdata"}};

      DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

      dop.SetParameters({mesh_nodes, &qdata});
      StopWatch sw;
      sw.Start();
      for (int i = 0; i < num_samples; i++)
      {
         dop.Mult(x, qdata);
      }
      sw.Stop();
      printf("dfem setup: %fs\n", sw.RealTime() / num_samples);
      qdata.HostRead();
   }

   // printf("qdata: ");
   // print_vector(qdata);

   Vector y(h1fes.GetTrueVSize());
   {
      auto diffusion_apply_kernel =
         [] MFEM_HOST_DEVICE (
            const tensor<double, dim>& dudxi,
            const tensor<double, dim, dim>& qdata)
      {
         return mfem::tuple{qdata * dudxi};
      };

      mfem::tuple argument_operators = {Gradient{"potential"}, None{"qdata"}};
      mfem::tuple output_operator = {Gradient{"potential"}};

      ElementOperator eop = {diffusion_apply_kernel, argument_operators, output_operator};
      auto ops = mfem::tuple{eop};

      auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
      auto parameters = std::array{FieldDescriptor{&qdata_space, "qdata"}};

      DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

      dop.SetParameters({&qdata});
      StopWatch sw;
      sw.Start();
      for (int i = 0; i < num_samples; i++)
      {
         dop.Mult(x, y);
      }
      sw.Stop();
      printf("dfem apply: %fs\n", sw.RealTime() / num_samples);
      y.HostRead();
   }

   // printf("y: ");
   // print_vector(y);

   Vector y2(h1fes.TrueVSize());
   {
      ParBilinearForm a(&h1fes);
      auto diff_integ = new DiffusionIntegrator;
      diff_integ->SetIntRule(&ir);
      a.AddDomainIntegrator(diff_integ);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

      StopWatch sw;
      sw.Start();
      a.Assemble();
      sw.Stop();
      printf("mfem setup: %fs\n", sw.RealTime());

      a.Finalize();
      sw.Clear();
      sw.Start();
      for (int i = 0; i < num_samples; i++)
      {
         a.Mult(x, y2);
      }
      sw.Stop();
      printf("mfem apply: %fs\n", sw.RealTime() / num_samples);
      y2.HostRead();
   }
   // printf("y2: ");
   // print_vector(y2);

   Vector diff(y2);
   diff -= y;
   if (diff.Norml2() > 1e-15)
   {
      // printf("y ");
      // print_vector(y);
      // printf("y2: ");
      // print_vector(y2);
      // printf("diff: ");
      // print_vector(diff);
      return 1;
   }

   // Test linearization here as well
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

DFEM_TEST_MAIN(test_diffusion_3d);
