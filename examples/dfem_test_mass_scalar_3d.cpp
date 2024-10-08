#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"
#include "fem/bilininteg.hpp"
#include "fem/fe/fe_base.hpp"

using namespace mfem;
using mfem::internal::tensor;

int dfem_test_mass_scalar_3d(std::string mesh_file,
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

   mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   ParGridFunction *mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   polynomial_order * h1fec.GetOrder() + 2);

   // IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
   // const IntegrationRule &ir = gll_rules.Get(h1fes.GetFE(0)->GetGeomType(),
   //                                           2 * polynomial_order - 1);

   auto dtq = h1fes.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
   printf("\n B: ");
   dtq.B.Print(out, dtq.B.Size());
   printf("\n G: ");
   dtq.G.Print(out, dtq.G.Size());
   printf("\n w: ");
   ir.GetWeights().Print(out, ir.GetWeights().Size());

   printf("#ndof per el = %d\n", h1fes.GetFE(0)->GetDof());
   printf("#nqp = %d\n", ir.GetNPoints());
   printf("#q1d = %d\n", (int)floor(pow(ir.GetNPoints(), 1.0/dim) + 0.5));

   printf("nodes: ");
   print_vector(*mesh_nodes);

   ParGridFunction f1_g(&h1fes);

   auto kernel = [](const double& u,
                    const tensor<double, dim> x,
                    const tensor<double, dim, dim> J,
                    const double& w)
   {
      return serac::tuple{u * det(J) * w};
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
      const double z = coords(2);
      return 2.345 + x + x*y + 1.25 * z*x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);
   printf("\nf1_g: ");
   print_vector(f1_g);

   auto R = h1fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   Vector f1_g_e(R->Height());
   R->Mult(f1_g, f1_g_e);
   printf("\nf1_g_e: ");
   print_vector(f1_g_e);
   auto r_out = std::ofstream("r_mat.mtx");
   R->PrintMatlab(r_out);
   r_out.close();

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
   if (diff.Norml2() > 1e-15)
   {
      printf("y ");
      print_vector(y);
      printf("y2: ");
      print_vector(y2);
      printf("diff: ");
      print_vector(diff);
      return 1;
   }

   return 0;
}

DFEM_TEST_MAIN(dfem_test_mass_scalar_3d);
