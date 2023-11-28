#include "mfem.hpp"
#include "dfem/dfem_refactor.hpp"

using namespace mfem;

auto main(int argc, char *argv[]) -> int
{
   Mpi::Init();

   std::cout << std::setprecision(9);

   const char *mesh_file = "../data/star.mesh";
   int polynomial_order = 1;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.ParseCheck();

   Mesh mesh_serial(mesh_file, 1, 1);
   mesh_serial.SetCurvature(1);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   const int dim = mesh_serial.Dimension();

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   constexpr int vdim = 1;

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   std::cout << "nqpts = " << ir.GetNPoints() << std::endl;
   std::cout << "ndofs = " << h1fes.GlobalTrueVSize() << std::endl;

   ParGridFunction u(&h1fes);

   auto exact_solution = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + x + y;
   };

   FunctionCoefficient exact_solution_coeff(exact_solution);

   u.ProjectCoefficient(exact_solution_coeff);

   auto domain_qf = [](const double &u,
                       const tensor<double, 2, 2> &J,
                       const double &w)
   {
      out << u << "\n" << J << "\n" << w << "\n\n";
      return std::tuple{u * det(J) * w};
   };

   std::tuple input_descriptors = {Value{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_descriptors = {Value{"potential"}};
   ElementOperator eop{domain_qf, input_descriptors, output_descriptors};

   auto ops = std::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};
   DifferentiableOperator dop{solutions, parameters, ops, mesh, ir};

   Vector x(h1fes.GetTrueVSize()), y(h1fes.GetTrueVSize());

   u.GetTrueDofs(x);

   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);

   // Derivative wrt "potential", indicated by the index 0 of the set {solutions} \cup {parameters}
   auto dFd0 = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});
   dFd0->Mult(x, y);

   Vector dFd0_vec;
   dFd0->Assemble(dFd0_vec);

   // Derivative wrt "coordinates", indicated by the index 1 of the set {solutions} \cup {parameters}
   auto dFd1 = dop.GetDerivativeWrt<1>({&u}, {mesh_nodes});
   dFd1->Mult(x, y);

   return 0;
}
