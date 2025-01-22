#include "dfem/dfem_refactor.hpp"
#include "linalg/ode.hpp"

using namespace mfem;
using mfem::internal::tensor;

class TimeStepEstimateQFunction
{
public:
   TimeStepEstimateQFunction() = default;

   using vecd = tensor<real_t, 2>;
   using matd = tensor<real_t, 2, 2>;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &h0,
      const real_t &order_v,
      const real_t &w)
   {
      real_t dt_est = 0.0;
      return mfem::tuple{dt_est};
   }
};

void velocity(const Vector &c, Vector &u)
{
   const double x = c(0);
   const double y = c(1);

   u(0) = 0.5 - y;
   u(1) = x - 0.5;
}

template <int problem = 0>
real_t three_bodies_ic(const Vector &X)
{
   const real_t x = X(0);
   const real_t y = X(1);
   const real_t r0 = 0.15;
   real_t x0 = 0.0;
   real_t y0 = 0.0;

   auto region = [&r0](const real_t x, const real_t y, const real_t x0,
                       const real_t y0)
   {
      return sqrt(pow(x-x0, 2.0) + pow(y-y0, 2.0));
   };

   x0 = 0.25;
   y0 = 0.5;
   const real_t hump = 0.25 + 0.25 * cos(M_PI * region(x, y, x0, y0) / r0);
   if (region(x, y, x0, y0) <= r0)
   {
      return hump;
   }

   if constexpr (problem == 1)
   {
      return 0.0;
   }

   x0 = 0.5;
   y0 = 0.25;
   const real_t cone = (1.0 - region(x, y, x0, y0) / r0);
   if (region(x, y, x0, y0) <= r0)
   {
      return cone;
   }

   x0 = 0.5;
   y0 = 0.75;
   if ((region(x, y, x0, y0) <= r0) && (fabs(x - 0.5) >= 0.025 || y >= 0.85))
   {
      return 1.0;
   }

   return 0.0;
}

MFEM_HOST_DEVICE
template <int dim = 2>
tensor<real_t, dim> get_velocity(const tensor<real_t, dim>& x)
{
   return {0.5 - x(1), x(0) - 0.5};
}

MFEM_HOST_DEVICE
template <int dim = 2>
real_t compute_tau(
   const tensor<real_t, dim>& b,
   const tensor<real_t, dim, dim>& J,
   const real_t& dt,
   const int& p)
{
   const real_t h_min = calcsv(J, dim-1) / static_cast<real_t>(p);
   real_t velocity_norm = sqrt(dot(b, b));
   if (velocity_norm < 1e-12) { velocity_norm = 1e-12; }
   auto tau_ugn_1 = h_min;
   auto tau_ugn_2 = dt / 2.0;
   auto tau = 1.0 / sqrt(1.0/pow(tau_ugn_1, 2) + 1.0/pow(tau_ugn_2, 2));
   return tau;
}

template <int dim = 2>
class SUPGMassQFunction
{
public:
   SUPGMassQFunction(const real_t &dt, const int &p) :
      dt(dt),
      p(p)
   {}

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &k,
      const tensor<real_t, dim>& x,
      const tensor<real_t, dim, dim>& J,
      const real_t& w) const
   {
      auto b = get_velocity(x);
      auto tau = compute_tau(b, J, dt, p);

      return mfem::tuple{tau * k * b * transpose(inv(J)) * det(J) * w};
   }

   const real_t &dt;  // time step
   const int p;  // polynomial order
};

template <int dim = 2>
class AdvQFunction
{
public:
   AdvQFunction(const real_t &dt, const int &p, bool use_stabilization) :
      dt(dt),
      p(p),
      use_stabilization(use_stabilization)
   {};

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &u,
      const tensor<real_t, dim>& dudxi,
      const tensor<real_t, dim>& x,
      const tensor<real_t, dim, dim>& J,
      const real_t& w) const
   {
      auto invJ = inv(J);
      auto b = get_velocity(x);

      // Advection
      auto advection = -b * u;

      if (use_stabilization)
      {
         auto residual = dot(b, (transpose(inv(J)) * dudxi));
         auto tau = compute_tau(b, J, dt, p);
         auto stab = tau * residual * b;
         return mfem::tuple{(advection + stab) * transpose(invJ) * det(J) * w};
      }
      else
      {
         return mfem::tuple{advection * transpose(invJ) * det(J) * w};
      }
   }

   const bool use_stabilization;
   const real_t &dt;  // time step
   const int p;  // polynomial order
};

template <int dim = 2>
class AdvOp : public TimeDependentOperator
{
   static constexpr int Concentration = 0;
   static constexpr int Coordinates = 1;

   class AdvGradientOp : public Operator
   {
   public:
      AdvGradientOp(const AdvOp &a, const Vector &x, real_t h) :
         Operator(a.Height()),
         a(a),
         concentration_l(a.fes.GetVSize()),
         h(h)
      {
         ParGridFunction g(&a.fes, concentration_l);
         a.fes.GetProlongationMatrix()->Mult(x, g);
         dRdu = a.adv->GetDerivative(Concentration, {&g}, {a.mesh_nodes});
      }

      void Mult(const Vector &k, Vector &y) const override
      {
         // column elimination for essential dofs
         k_elim = k;
         k_elim.SetSubVector(a.ess_tdof_list, 0.0);

         dRdu->Mult(k_elim, y);
         y *= h;
         a.M->AddMult(k_elim, y);
         if (a.use_stabilization)
         {
            a.Msupg_dk->AddMult(k_elim, y);
         }

         for (int i = 0; i < a.ess_tdof_list.Size(); i++)
         {
            y[a.ess_tdof_list[i]] = k[a.ess_tdof_list[i]];
         }
      }

      const AdvOp &a;
      mutable Vector concentration_l;
      mutable Vector k_elim;
      real_t h;
      std::shared_ptr<DerivativeOperator> dRdu;
   };

   class AdvResidualOp : public Operator
   {
   public:
      AdvResidualOp(const AdvOp &a, real_t dt, const Vector &x) :
         Operator(a.Height()),
         dt(dt),
         a(a),
         x(x),
         u(x.Size()),
         z(x.Size())
      {}

      void Mult(const Vector &k, Vector &R) const override
      {
         u = k;
         u *= dt;
         u += x;

         a.M->Mult(k, R);
         if (a.use_stabilization)
         {
            a.Msupg_dk->AddMult(k, R);
         }
         a.adv->AddMult(u, R);

         R.SetSubVector(a.ess_tdof_list, 0.0);
      }

      Operator& GetGradient(const Vector &k) const override
      {
         u = k;
         u *= dt;
         u += x;

         jacobian.reset(new AdvGradientOp(a, u, dt));
         return *jacobian;

         // fd_jacobian.reset(new FDJacobian(*this, k));
         // return *fd_jacobian;
      }

      const AdvOp &a;
      double dt;
      Vector x;
      mutable Vector u, z;

      mutable std::shared_ptr<FDJacobian> fd_jacobian;

      // AD Jacobian operator dRdu
      mutable std::shared_ptr<AdvGradientOp> jacobian;
   };

public:
   AdvOp(ParFiniteElementSpace &fes, const IntegrationRule &ir,
         const Array<int> ess_tdof_list, const double &dt, const int &polynomial_order,
         bool disable_tensor_product_structure = false, bool use_stabilization = false) :
      TimeDependentOperator(fes.GetTrueVSize()),
      ess_tdof_list(ess_tdof_list),
      fes(fes),
      Mform(&fes),
      use_stabilization(use_stabilization)
   {
      auto mesh = fes.GetParMesh();
      mesh_nodes = static_cast<ParGridFunction*>
                   (mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      auto solutions = std::vector{FieldDescriptor{Concentration, &fes}};
      auto parameters = std::vector{FieldDescriptor{Coordinates, &mesh_fes}};

      adv = std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
      adv->DisableTensorProductStructure(disable_tensor_product_structure);
      auto derivatives = std::integer_sequence<size_t, Concentration> {};
      AdvQFunction<2> adv_qf(dt, polynomial_order, use_stabilization);
      {
         auto input_operators = mfem::tuple
         {
            Value<Concentration>{},
            Gradient<Concentration>{},
            Value<Coordinates>{},
            Gradient<Coordinates>{},
            Weight{}
         };
         auto output_operator = mfem::tuple{Gradient<Concentration>{}};

         adv->AddDomainIntegrator(adv_qf, input_operators, output_operator, ir,
                                  derivatives);
      }
      adv->SetParameters({mesh_nodes});

      supg_mass = std::make_shared<DifferentiableOperator>(solutions, parameters,
                                                           *mesh);
      supg_mass->DisableTensorProductStructure(disable_tensor_product_structure);
      SUPGMassQFunction<2> supg_mass_qf(dt, polynomial_order);
      {
         auto input_operators = mfem::tuple
         {
            Value<Concentration>{},
            Value<Coordinates>{},
            Gradient<Coordinates>{},
            Weight{}
         };
         auto output_operator = mfem::tuple{Gradient<Concentration>{}};

         supg_mass->AddDomainIntegrator(supg_mass_qf, input_operators, output_operator,
                                        ir, derivatives);
      }
      supg_mass->SetParameters({mesh_nodes});

      // Compute SUPG mass matrix by linearizing around a dummy variable
      {
         ParGridFunction g(&fes);
         g = 1.0;
         Msupg_dk = supg_mass->GetDerivative(Concentration, {&g}, {mesh_nodes});
      }

      Mform.AddDomainIntegrator(new MassIntegrator);
      Mform.Assemble();
      Mform.FormSystemMatrix(Array<int> {}, M);
   }

   void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      auto residual = AdvResidualOp(*this, dt, x);

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
   std::shared_ptr<DifferentiableOperator> adv;
   std::shared_ptr<DifferentiableOperator> supg_mass;
   std::shared_ptr<DerivativeOperator> Msupg_dk;
   Array<int> ess_tdof_list;
   ParFiniteElementSpace &fes;
   ParGridFunction *mesh_nodes = nullptr;
   ParBilinearForm Mform;
   OperatorHandle M;
   const bool use_stabilization;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "../data/ref-square.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;
   real_t dt = 1.0;
   real_t t_final = 2.0 * M_PI;
   int vis_steps = 5;
   bool disable_tensor_product_structure = false;
   bool use_stabilization = false;
   int problem = 0;

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
   args.AddOption(&disable_tensor_product_structure, "-disable-tp", "--disable-tp",
                  "-enable-tp", "--enable-tp", "");
   args.AddOption(&use_stabilization, "-enable-stab", "--enable-stab",
                  "-disable-stab", "--disable-stab", "");
   args.AddOption(&problem, "-prob", "--problem", "problem number");
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

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   const int global_tdof = h1fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "#dofs " << global_tdof << "\n";
   }

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(
                      0)->GetDim() - 1);

   ParGridFunction concentration(&h1fes);

   FunctionCoefficient *concentration_ic_coef = nullptr;
   if (problem == 0)
   {
      concentration_ic_coef = new FunctionCoefficient(three_bodies_ic<0>);
   }
   else
   {
      concentration_ic_coef = new FunctionCoefficient(three_bodies_ic<1>);
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   AdvOp advdiff(h1fes, ir, ess_tdof_list, dt, polynomial_order,
                 disable_tensor_product_structure, use_stabilization);

   ODESolver *ode_solver = new SDIRK33Solver;
   ode_solver->Init(advdiff);

   Vector zero, x(h1fes.GetTrueVSize());
   concentration.ProjectCoefficient(*concentration_ic_coef);
   concentration.GetTrueDofs(x);

   // print_vector(x);

   ParGridFunction sol(&h1fes), err(&h1fes);
   sol.SetFromTrueDofs(x);

   err.ProjectCoefficient(*concentration_ic_coef);

   real_t t = 0.0;

   ParaViewDataCollection dc("dfem_advection_supg", &mesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(polynomial_order);
   dc.RegisterField("concentration", &sol);
   dc.RegisterField("err", &err);
   dc.SetCycle(0);
   dc.SetTime(t);
   dc.Save();

   bool done = false;
   for (int ti = 0; !done;)
   {
      real_t dt_real = std::min(dt, t_final - t);
      ode_solver->Step(x, t, dt_real);
      // print_vector(x);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         sol.SetFromTrueDofs(x);

         err.ProjectCoefficient(*concentration_ic_coef);
         for (int i = 0; i < err.Size(); i++)
         {
            err[i] = abs(sol[i] - err[i]);
         }

         dc.SetCycle(ti);
         dc.SetTime(t);
         dc.Save();

         if (Mpi::Root())
         {
            std::cout << "time step: " << ti << ", time: " << t << std::endl;
         }
      }
   }

   const real_t l2err = sol.ComputeL2Error(*concentration_ic_coef);
   const real_t maxerr = sol.ComputeMaxError(*concentration_ic_coef);
   if (Mpi::Root())
   {
      std::cout << "|u - u_ic|_L2 = " << l2err << "\n";
      std::cout << "|u - u_ic|_max = " << maxerr << "\n";
   }

   return 0;
}
