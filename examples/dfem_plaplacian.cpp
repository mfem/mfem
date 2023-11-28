#include "dfem.hpp"

int main(int argc, char *argv[])
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
      // PRESENT
      return pow(x,2) + 0.5*x*pow(y,2);
   };

   FunctionCoefficient exact_solution_coeff(exact_solution);

   auto plaplacian = [](double u,
                        tensor<double, 2> dudxi,
                        tensor<double, 2, 2> J,
                        double w)
   {
      using mfem::internal::tensor;
      auto dudx = dudxi * inv(J);
      auto JxW = transpose(inv(J)) * det(J) * w;
      // PRESENT: Implement (1+u^2) * ∇u
      return (1.0 + u*u) * dudx * JxW;
   };

   // PRESENT: Implement descriptors
   std::tuple input_descriptors = {Value{"potential"}, Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   // PRESENT: Implement descriptors
   std::tuple output_descriptors = {Gradient{"potential"}};

   ElementOperator qf {plaplacian, input_descriptors, output_descriptors};

   ElementOperator forcing_qf
   {
      [](tensor<double, 2> coords, tensor<double, 2, 2> J, double w)
      {
         int p = 2;
         double x = coords(0);
         double y = coords(1);
         // *INDENT-OFF*
         double mathematica_please_help_me = 2.*pow(x,2)*pow(y,2)*(pow(x,2) + 0.5*x*pow(y,2)) + 2*pow(2*x + 0.5*pow(y,2),2)*(pow(x,2) + 0.5*x*pow(y,2)) + 2*(1 + pow(pow(x,2) + 0.5*x*pow(y,2),2)) + 1.*x*(1 + pow(pow(x,2) + 0.5*x*pow(y,2),2));
         return mathematica_please_help_me * det(J) * w;
        // *INDENT-ON*
      },
      // inputs
      std::tuple{
         Value{"coordinates"},
         Gradient{"coordinates"},
         Weight{"integration_weight"}},
      // outputs
      std::tuple{
         Value{"potential"}}
   };

   std::tuple list_of_qfs{qf_1, qf_2, qf_n};

   std::vector<Field> solutions{{&u, "potential"}};
   std::vector<Field> parameters{{mesh.GetNodes(), "coordinates"}};
   DifferentiableForm dop(solutions, parameters, mesh);
   dop.SetEssentialTrueDofs(ess_tdof_list);

   auto R = dop.GetResidual(list_of_qfs, ir);
   auto Jacobian_aka_dRdu = dop.GetDerivative<0>(list_of_qfs, ir);

   // R(u) = (\grad u, \grad v) + (f, v)
   // dop.AddElementOperator<AD::Enzyme>(qf, ir);
   // dop.AddElementOperator<AD::None>(forcing_qf, ir);

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(5000);
   gmres.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(gmres);
   newton.SetOperator(dop);
   newton.SetRelTol(1e-12);
   newton.SetMaxIter(100);
   newton.SetPrintLevel(1);

   u = 1e-6;
   u.ProjectBdrCoefficient(exact_solution_coeff, ess_bdr);
   Vector x;
   u.GetTrueDofs(x);

   Vector zero;
   newton.Mult(zero, x);

   u.Distribute(x);

   std::cout << "|u-u_ex|_L2 = " << u.ComputeL2Error(exact_solution_coeff) << "\n";

   return 0;
}
