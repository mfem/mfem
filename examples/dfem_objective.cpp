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

   constexpr int vdim = 2;

   // test_partial_assembly_setup_qf(mesh, 1, polynomial_order);
   // exit(0);

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
   ParGridFunction g(&h1fes);
   ParGridFunction rho(&h1fes);

   auto exact_solution = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      u(0) = x*x + y;
      u(1) = x + 0.5*y*y;
   };

   VectorFunctionCoefficient exact_solution_coeff(dim, exact_solution);

   auto objective = [](tensor<double, 2> u, double rho,
                       tensor<double, 2, 2> J,
                       double w)
   {
      return sqnorm(u) * det(J) * w;
   };

   std::tuple inputs{Value{"displacement"}, Value{"density"}, Gradient{"coordinates"}, Weight{"integration_weight"}};
   std::tuple outputs{ One{"integral"} };
   ElementOperator objective_eop { objective, inputs, outputs };

   std::vector<Field> solution_fields{{&u, "displacement"}};
   std::vector<Field> parameter_fields{{mesh.GetNodes(), "coordinates"}, {&rho, "density"}};
   std::vector<Field> dependent_variables{{&u, "displacement"}};
   DifferentiableForm dop(solution_fields, parameter_fields, dependent_variables,
                          mesh);

   dop.AddElementOperator(objective_eop, ir);

   u.ProjectCoefficient(exact_solution_coeff);
   Vector zero;

   Vector y(1);
   Vector utdof;
   u.GetTrueDofs(utdof);
   dop.Mult(utdof, y);

   // finite difference test
   Vector dgdu(u.Size());
   Vector fx(y);
   out << "g: ";
   print_vector(fx);
   out << "\n";

   for (int i = 0; i < u.Size(); i++)
   {
      double h = 1e-6;
      u(i) += h;
      dop.Mult(u, y);
      u(i) -= h;
      y -= fx;
      y /= h;
      dgdu(i) = y(0);
   }

   out << "dgdu: ";
   print_vector(dgdu);

   // Vector dgdu = dop.GetGradientWrt({&u, "displacement"});

   return 0;
}
