#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"
#include "fem/bilininteg.hpp"
#include "fem/normal_deriv_restriction.hpp"
#include <fstream>

using namespace mfem;
using mfem::internal::tensor;

int dfem_test_mass_scalar_2d(std::string mesh_file,
                             int refinements,
                             int polynomial_order)
{
   constexpr int dim = 2;
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

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   // IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
   // const IntegrationRule &ir = gll_rules.Get(h1fes.GetFE(0)->GetGeomType(),
   //                                           2 * polynomial_order - 1);

   printf("#nqp = %d\n", ir.GetNPoints());
   printf("#q1d = %d\n", (int)floor(pow(ir.GetNPoints(), 1.0/dim) + 0.5));

   ParGridFunction f1_g(&h1fes);

   auto kernel = [](const double& u,
                    const tensor<double, dim> x,
                    const tensor<double, dim, dim> J,
                    const double& w)
   {
      out << x << ": " << u << "\n";
      return serac::tuple{u * w * det(J)};
   };

   serac::tuple argument_operators = {Value{"potential"}, Value{"coordinates"}, Gradient{"coordinates"}, Weight{}};
   serac::tuple output_operator = {Value{"potential"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = serac::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x + x*y + 1.25 * x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector f1_g_e(f1_g.Size());
   auto R = h1fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   // R->Mult(f1_g, f1_g_e);
   auto r_out = std::ofstream("r_mat.mtx");
   R->PrintMatlab(r_out);
   r_out.close();
   print_vector(f1_g);
   // print_vector(f1_g_e);

   Vector x(*f1_g.GetTrueDofs()), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   ParBilinearForm a(&h1fes);
   auto mass_integ = new MassIntegrator;
   mass_integ->SetIntRule(&ir);
   a.AddDomainIntegrator(mass_integ);
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

   return 0;
}

DFEM_TEST_MAIN(dfem_test_mass_scalar_2d);
