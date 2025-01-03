#include "dfem/dfem_refactor.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <int dim = 2>
class AdvDiffQFunction
{
public:
   AdvDiffQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (const real_t &u,
                    const tensor<real_t, dim>& dudxi,
                    const tensor<real_t, dim, dim>& J,
                    const real_t& w) const
   {
      auto invJ = inv(J);

      // Advection
      auto b = tensor<real_t, dim> {1.0, 1.0};
      auto advection = -b * u;

      // Diffusion
      auto K = 1.0 / (1.0 + u*u);
      auto diffusion = K * (dudxi * invJ);

      return mfem::tuple{(advection + diffusion) * transpose(invJ) * det(J) * w};
   }
};

real_t mms_solution(const Vector& coords)
{
   const double x = coords(0);
   const double y = coords(1);
   return pow(cos(y),2) + pow(sin(x),2);
};

real_t mms_forcing(const Vector &coords)
{
   const double x = coords(0);
   const double y = coords(1);
   return 2*cos(x)*sin(x) + (8*pow(cos(x),2)*pow(sin(x),2)*(pow(cos(y),
                                                                2) + pow(sin(x),2)))/pow(1 + pow(pow(cos(y),2) + pow(sin(x),2),2),
                                                                      2) - (2*pow(cos(x),2))/(1 + pow(pow(cos(y),2) + pow(sin(x),2),
                                                                            2)) + (2*pow(cos(y),2))/(1 + pow(pow(cos(y),2) + pow(sin(x),2),
                                                                                  2)) + (2*pow(sin(x),2))/(1 + pow(pow(cos(y),2) + pow(sin(x),2),
                                                                                        2)) - 2*cos(y)*sin(y) + (8*pow(cos(y),2)*(pow(cos(y),2) + pow(sin(x),
                                                                                              2))*pow(sin(y),2))/pow(1 + pow(pow(cos(y),2) + pow(sin(x),2),2),
                                                                                                    2) - (2*pow(sin(y),2))/(1 + pow(pow(cos(y),2) + pow(sin(x),2),2));
}

template <int dim = 2>
class AdvDiffOp : public TimeDependentOperator
{
   static constexpr int Concentration = 0;
   static constexpr int Coordinates = 1;

   class AdvDiffGradientOp : public Operator
   {
   public:
      AdvDiffGradientOp(const AdvDiffOp &a, const Vector &x, real_t h) :
         Operator(a.Height()),
         a(a),
         concentration_l(a.fes.GetTrueVSize()),
         h(h)
      {
         ParGridFunction g(&a.fes, concentration_l);
         a.fes.GetProlongationMatrix()->Mult(x, g);
         dRdu = a.adv_diff->GetDerivative(Concentration, {&g}, {a.mesh_nodes});
      }

      void Mult(const Vector &k, Vector &y) const override
      {
         // column elimination for essential dofs
         k_elim = k;
         k_elim.SetSubVector(a.ess_tdof_list, 0.0);

         dRdu->Mult(k_elim, y);
         y *= h;
         a.M->AddMult(k_elim, y);

         for (int i = 0; i < a.ess_tdof_list.Size(); i++)
         {
            y[a.ess_tdof_list[i]] = k[a.ess_tdof_list[i]];
         }
      }

      const AdvDiffOp &a;
      mutable Vector concentration_l;
      mutable Vector k_elim;
      real_t h;
      std::unique_ptr<Operator> dRdu;
   };

   class AdvDiffResidualOp : public Operator
   {
   public:
      AdvDiffResidualOp(const AdvDiffOp &a, real_t dt, const Vector &x) :
         Operator(a.Height()),
         dt(dt),
         a(a),
         x(x),
         u(x.Size()),
         z(x.Size()) {}

      void Mult(const Vector &k, Vector &R) const override
      {
         u = k;
         u *= dt;
         u += x;

         a.M->Mult(k, R);
         a.adv_diff->AddMult(u, R);
         R -= a.mms_forcing_rhs;

         R.SetSubVector(a.ess_tdof_list, 0.0);
      }

      Operator& GetGradient(const Vector &k) const override
      {
         u = k;
         u *= dt;
         u += x;
         jacobian.reset(new AdvDiffGradientOp(a, u, dt));

         return *jacobian;

         // fd_jacobian.reset(new FDJacobian(*this, k));
         // fd_jacobian->PrintMatlab(std::cout);
         // return *fd_jacobian;
      }

      const AdvDiffOp &a;
      double dt;
      Vector x;
      mutable Vector u, z;

      mutable std::shared_ptr<FDJacobian> fd_jacobian;

      // AD Jacobian operator dRdu
      mutable std::shared_ptr<AdvDiffGradientOp> jacobian;
   };

public:
   AdvDiffOp(ParFiniteElementSpace &fes, const IntegrationRule &ir,
             const Array<int> ess_tdof_list) :
      TimeDependentOperator(fes.GetTrueVSize()),
      ess_tdof_list(ess_tdof_list),
      fes(fes),
      Mform(&fes)
   {
      auto mesh = fes.GetParMesh();
      mesh_nodes = static_cast<ParGridFunction*>
                   (mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      auto input_operators = mfem::tuple{Value<Concentration>{}, Gradient<Concentration>{}, Gradient<Coordinates>{}, Weight{}};
      auto output_operator = mfem::tuple{Gradient<Concentration>{}};

      auto solutions = std::vector{FieldDescriptor{Concentration, &fes}};
      auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes}};

      adv_diff = std::make_unique<DifferentiableOperator>(solutions, parameters,
                                                          *mesh);
      auto derivatives = std::integer_sequence<size_t, Concentration> {};
      AdvDiffQFunction<2> advdiff_qf{};
      adv_diff->AddDomainIntegrator(advdiff_qf, input_operators, output_operator, ir,
                                    derivatives);

      adv_diff->SetParameters({mesh_nodes});

      FunctionCoefficient mms_forcing_c(mms_forcing);
      ParLinearForm Lform(&fes);
      Lform.AddDomainIntegrator(new DomainLFIntegrator(mms_forcing_c, &ir));
      Lform.Assemble();
      Lform.ParallelAssemble(mms_forcing_rhs);

      Mform.AddDomainIntegrator(new MassIntegrator);
      Mform.Assemble();
      Mform.FormSystemMatrix(Array<int> {}, M);
   }

   void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      auto residual = AdvDiffResidualOp(*this, dt, x);

      GMRESSolver krylov(MPI_COMM_WORLD);
      krylov.SetRelTol(1e-6);
      krylov.SetMaxIter(1000);
      // krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetOperator(residual);
      newton.SetSolver(krylov);
      newton.SetRelTol(1e-12);
      newton.SetMaxIter(10);
      // newton.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

      Vector zero;
      k = x;
      k.SetSubVector(ess_tdof_list, 0.0);
      newton.Mult(zero, k);
   }

private:
   std::unique_ptr<DifferentiableOperator> adv_diff;
   Array<int> ess_tdof_list;
   ParFiniteElementSpace &fes;
   ParGridFunction *mesh_nodes = nullptr;
   Vector mms_forcing_rhs;
   ParBilinearForm Mform;
   OperatorHandle M;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "../data/inline-quad.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;
   real_t dt = 1.0;
   real_t t_final = 1.0;
   int vis_steps = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   out << std::setprecision(8);

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

   ParGridFunction concentration_exact(&h1fes);

   FunctionCoefficient mms_solution_c(mms_solution);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   AdvDiffOp advdiff(h1fes, ir, ess_tdof_list);

   ODESolver *ode_solver = new SDIRK23Solver;
   ode_solver->Init(advdiff);

   Vector zero, x(h1fes.GetTrueVSize());
   concentration_exact = 0.0;
   concentration_exact.ProjectBdrCoefficient(mms_solution_c, ess_bdr);
   concentration_exact.GetTrueDofs(x);

   // print_vector(x);

   real_t t = 0.0;

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = std::min(dt, t_final - t);
      ode_solver->Step(x, t, dt_real);
      // print_vector(x);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            std::cout << "time step: " << ti << ", time: " << t << std::endl;
         }
      }
   }

   concentration_exact.ProjectCoefficient(mms_solution_c);
   ParGridFunction sol(&h1fes), exact_sol(&h1fes), err(&h1fes);
   exact_sol.ProjectCoefficient(mms_solution_c);
   sol.SetFromTrueDofs(x);
   real_t l2err = sol.ComputeL2Error(mms_solution_c);
   out << "l2err = " << l2err << "\n";

   for (int i = 0; i < err.Size(); i++)
   {
      err[i] = abs(sol[i] - concentration_exact[i]);
   }

   ParaViewDataCollection dc("dfem_nonlinear_advdiff", &mesh);
   dc.RegisterField("concentration", &sol);
   dc.RegisterField("exact", &exact_sol);
   dc.RegisterField("err", &err);
   dc.Save();

   return 0;
}
