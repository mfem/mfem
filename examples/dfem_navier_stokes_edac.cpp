#include "dfem/dfem_refactor.hpp"

using namespace mfem;
using mfem::internal::tensor;

void vel_ldc_ic(const Vector &coords, Vector &u)
{
   real_t x = coords(0);
   real_t y = coords(1);

   if (y >= 1.0)
   {
      u(0) = 1.0;
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

void vel_ldc_dt(const Vector &coords, Vector &u)
{
   real_t x = coords(0);
   real_t y = coords(1);

   u(0) = 0.0;
   u(1) = 0.0;
}

void vel_shear_ic(const Vector &x, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   real_t rho = 80.0;
   real_t delta = 0.05;

   if (yi <= 0.5)
   {
      u(0) = tanh(rho * (yi - 0.25));
   }
   else
   {
      u(0) = tanh(rho * (0.75 - yi));
   }

   u(1) = delta * sin(2.0 * M_PI * (xi + 0.25));
}

void vel_2d_cyl_ic(const Vector &coords, Vector &u)
{
   real_t x = coords(0);
   real_t y = coords(1);

   real_t H = 0.41;
   // real_t A = 1.5;
   // real_t U = A * sin(M_PI * t / 8.0);
   real_t U = 0.3;
   u(0) = 4.0 * U * y * (H - y) / (H * H);
   u(1) = 0.0;
}


void vel_2d_cyl(const Vector &coords, const real_t &t, Vector &u)
{
   real_t x = coords(0);
   real_t y = coords(1);

   real_t H = 0.41;
   // real_t A = 1.5;
   // real_t U = A * sin(M_PI * t / 8.0);
   real_t U = 0.3;
   if (x == 0.0)
   {
      u(0) = 4.0 * U * y * (H - y) / (H * H);
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

void vel_2d_cyl_dt(const Vector &coords, const real_t &t, Vector &u)
{
   real_t x = coords(0);
   real_t y = coords(1);

   real_t H = 0.41;
   // real_t A = 1.5;
   // real_t dUdt = 1.0 / 8.0 * A * M_PI * cos(M_PI * t / 8.0);
   real_t dUdt = 0.0;
   if (x == 0.0)
   {
      u(0) = 4.0 * dUdt * y * (H - y) / (H * H);
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}


// -\nabla \cdot (\nabla u + p * I) -> (\nabla u + p * I, \nabla v)
template <int dim = 2>
class MomentumQFunction
{
public:
   MomentumQFunction(const real_t &kinematic_viscosity) :
      kinematic_viscosity(kinematic_viscosity) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &u,
                   const tensor<real_t, dim, dim> &dudxi,
                   const real_t &p,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto viscous_stress = -p * I + kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{(outer(u, u) - viscous_stress) * JxW};
   }

   // TODO: this might not be ok on GPU
   const real_t kinematic_viscosity;
};

template <int dim = 2>
class ViscousStressQFunction
{
public:
   ViscousStressQFunction(const real_t &kinematic_viscosity) :
      kinematic_viscosity(kinematic_viscosity) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dudxi,
                   const real_t &p,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      // auto viscous_stress = -p * I + 2.0 * kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{-(kinematic_viscosity * dudxi) * JxW};
   }

   // TODO: this might not be ok on GPU
   const real_t kinematic_viscosity;
};

template <int dim = 2>
class ConvectionQFunction
{
public:
   ConvectionQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &u,
                   const tensor<real_t, dim, dim> &dudxi,
                   const tensor<real_t, dim> &dpdxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto dpdx = dpdxi * invJ;
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{(-dot(dudx, u) - dpdx) * JxW};
   }
};

template <int dim = 2>
class ContinuityQFunction
{
public:
   ContinuityQFunction(real_t kinematic_viscosity,
                       real_t mach_number) :
      kinematic_viscosity(kinematic_viscosity),
      mach_number(mach_number) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &u,
                   const real_t &p,
                   const tensor<real_t, dim> &dpdxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      auto invJ = inv(J);
      auto dpdx = dpdxi * invJ;
      auto JxW = det(J) * w * transpose(invJ);
      auto r = mfem::tuple{((1.0 / (mach_number*mach_number)) * u - kinematic_viscosity * dpdx) * JxW};
      return r;
   }

   real_t kinematic_viscosity;
   real_t mach_number;
};

class NavierStokesOperator : public TimeDependentOperator
{
   static constexpr int Velocity = 0;
   static constexpr int Pressure = 1;
   static constexpr int Coordinates = 2;

public:
   NavierStokesOperator(ParFiniteElementSpace &velocity_fes,
                        ParFiniteElementSpace &pressure_fes,
                        Array<int> &offsets,
                        Array<int> &vel_ess_bdr_attr,
                        Array<int> &vel_ess_tdofs,
                        const real_t &kinematic_viscosity,
                        const IntegrationRule &velocity_ir,
                        const IntegrationRule &pressure_ir) :
      TimeDependentOperator(offsets.Last()),
      block_offsets(offsets),
      vel_ess_bdr_attr(vel_ess_bdr_attr),
      vel_ess_tdofs(vel_ess_tdofs),
      velocity_fes(velocity_fes),
      pressure_fes(pressure_fes),
      Mv_form(&velocity_fes),
      Mp_form(&pressure_fes),
      mv_diag_inv(velocity_fes.GetTrueVSize()),
      mp_diag_inv(pressure_fes.GetTrueVSize()),
      rhsu(velocity_fes.GetTrueVSize()),
      zu(velocity_fes.GetTrueVSize()),
      rhsp(pressure_fes.GetTrueVSize()),
      xu_gf(&velocity_fes),
      ku_gf(&velocity_fes),
      xp_gf(&pressure_fes)
   {
      auto mesh = velocity_fes.GetParMesh();
      auto mesh_nodes = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      auto parameters = std::vector
      {
         FieldDescriptor{Coordinates, &mesh_fes}
      };

      auto solutions = std::vector
      {
         FieldDescriptor{Velocity, &velocity_fes},
         FieldDescriptor{Pressure, &pressure_fes}
      };

      momentum =
         std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

      continuity =
         std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

      {
         mfem::tuple input_operators{Value<Velocity>{}, Gradient<Velocity>{}, Value<Pressure>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple output_operators{Gradient<Velocity>{}};

         MomentumQFunction<2> momentum_qf(kinematic_viscosity);
         momentum->AddDomainIntegrator(
            momentum_qf, input_operators, output_operators, velocity_ir);

         momentum->SetParameters({mesh_nodes});
      }

      {
         mfem::tuple input_operators{Value<Velocity>{}, Value<Pressure>{}, Gradient<Pressure>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple output_operators{Gradient<Pressure>{}};

         ContinuityQFunction<2> continuity_qf(kinematic_viscosity, 0.01);
         continuity->AddDomainIntegrator(
            continuity_qf, input_operators, output_operators, velocity_ir);

         continuity->SetParameters({mesh_nodes});
      }

      BilinearFormIntegrator *integ = new VectorMassIntegrator;
      integ->SetIntegrationRule(velocity_ir);
      Mv_form.AddDomainIntegrator(integ);
      Mv_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      Mv_form.Assemble();
      Mv_form.FormSystemMatrix(vel_ess_tdofs, Mv);

      MvInvPC = new OperatorJacobiSmoother(Mv_form, vel_ess_tdofs);

      Array<int> outlet(velocity_fes.GetParMesh()->bdr_attributes.Max());
      outlet = 0;
      // outlet[1] = 1;
      pressure_fes.GetEssentialTrueDofs(outlet, pres_ess_tdofs);

      integ = new MassIntegrator;
      integ->SetIntegrationRule(pressure_ir);
      Mp_form.AddDomainIntegrator(integ);
      Mp_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      Mp_form.Assemble();

      Mp_form.FormSystemMatrix(pres_ess_tdofs, Mp);

      MpInvPC = new OperatorJacobiSmoother(Mp_form, pres_ess_tdofs);
   }

   void Mult(const Vector &x, Vector &k) const override
   {
      Vector xu(x.GetData() + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector xp(x.GetData() + block_offsets[1],
                block_offsets[2] - block_offsets[1]);
      Vector ku(k.ReadWrite() + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector kp(k.ReadWrite() + block_offsets[1],
                block_offsets[2] - block_offsets[1]);

      // velocity_fes.GetProlongationMatrix()->Mult(xu, xu_gf);
      // xu_gf.ProjectBdrCoefficient(*u_coef, vel_ess_bdr_attr);
      // velocity_fes.GetProlongationMatrix()->MultTranspose(xu_gf, xu);

      // Momentum solve
      {
         momentum->Mult(x, rhsu);

         ku_gf = 0.0;
         ku_gf.ProjectBdrCoefficient(*dudt_coef, vel_ess_bdr_attr);
         velocity_fes.GetProlongationMatrix()->MultTranspose(ku_gf, ku);

         Mv.As<ConstrainedOperator>()->EliminateRHS(ku, rhsu);

         CGSolver cg(MPI_COMM_WORLD);
         cg.SetOperator(*Mv);
         cg.SetPreconditioner(*MvInvPC);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(IterativeSolver::PrintLevel().None());

         cg.Mult(rhsu, ku);
      }

      // Continuity solve
      {
         continuity->Mult(x, rhsp);

         kp = 0.0;
         Mp.As<ConstrainedOperator>()->EliminateRHS(kp, rhsp);

         CGSolver cg(MPI_COMM_WORLD);
         cg.SetOperator(*Mp);
         cg.SetPreconditioner(*MpInvPC);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(IterativeSolver::PrintLevel().None());

         cg.Mult(rhsp, kp);
      }
   }

   void SetDVelocityDtDirichlet(VectorCoefficient *u, VectorCoefficient *dudt)
   {
      u_coef = u;
      dudt_coef = dudt;
   }

   void SetTime(real_t t) override
   {
      u_coef->SetTime(t);
      dudt_coef->SetTime(t);
   }

   std::shared_ptr<DifferentiableOperator> momentum;
   std::shared_ptr<DifferentiableOperator> continuity;

   Vector mv_diag_inv, mp_diag_inv;
   mutable Vector rhsu, rhsp, zu;
   mutable ParGridFunction xu_gf, ku_gf, xp_gf;

   ParBilinearForm Mv_form;
   ParBilinearForm Mp_form;
   OperatorHandle Mv;
   OperatorHandle Mp;

   mutable OperatorJacobiSmoother *MvInvPC, *MpInvPC;

   const Array<int> block_offsets;
   const Array<int> vel_ess_tdofs;
   Array<int> pres_ess_tdofs;
   const Array<int> vel_ess_bdr_attr;

   ParFiniteElementSpace &velocity_fes;
   ParFiniteElementSpace &pressure_fes;

   VectorCoefficient *u_coef = nullptr;
   VectorCoefficient *dudt_coef = nullptr;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "../data/fsi.msh";
   int polynomial_order = 2;
   int ir_order = 2;
   int refinements = 0;
   real_t kinematic_viscosity = 1.0e-3;
   real_t t_final = 1.0;
   real_t dt = 1e-4;
   int vis_steps = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&kinematic_viscosity, "-kv", "--kv", "");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
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
   mesh_serial.EnsureNodes();
   // GridFunction *nodes = mesh_serial.GetNodes();
   // *nodes -= -1.0;
   // *nodes /= 2.0;

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   // mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   H1_FECollection velocity_fec(polynomial_order, dim);
   ParFiniteElementSpace velocity_fes(&mesh, &velocity_fec, dim);

   H1_FECollection pressure_fec(polynomial_order - 1, dim);
   ParFiniteElementSpace pressure_fes(&mesh, &pressure_fec);

   out << velocity_fes.GetTrueVSize() << "\n";
   out << pressure_fes.GetTrueVSize() << "\n";

   const auto &velocity_ir = IntRules.Get(velocity_fes.GetFE(0)->GetGeomType(),
                                          2 * polynomial_order);

   const auto &pressure_ir = IntRules.Get(pressure_fes.GetFE(0)->GetGeomType(),
                                          2 * polynomial_order);

   Array<int> vel_ess_bdr_attr(mesh.bdr_attributes.Max());
   vel_ess_bdr_attr = 1;
   // vel_ess_bdr_attr[1] = 0; // outlet
   // vel_ess_bdr_attr[4] = 0; // beam walls

   Array<int> vel_ess_tdofs;
   velocity_fes.GetEssentialTrueDofs(vel_ess_bdr_attr, vel_ess_tdofs);
   // velocity_fes.GetEssentialTrueDofs(Array<int> {}, vel_ess_tdofs);

   ParGridFunction u(&velocity_fes);
   ParGridFunction p(&pressure_fes);

   // auto u_ic_coef = VectorFunctionCoefficient(dim, vel_2d_cyl_ic);
   // auto u_coef = VectorFunctionCoefficient(dim, vel_2d_cyl);
   // auto dudt_coef = VectorFunctionCoefficient(dim, vel_2d_cyl_dt);

   auto u_ic_coef = VectorFunctionCoefficient(dim, vel_ldc_ic);
   auto u_coef = VectorFunctionCoefficient(dim, vel_ldc_ic);
   auto dudt_coef = VectorFunctionCoefficient(dim, vel_ldc_dt);

   u.ProjectCoefficient(u_ic_coef);
   u.ProjectBdrCoefficient(u_coef, vel_ess_bdr_attr);
   p = 0.0;

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = velocity_fes.GetTrueVSize();
   block_offsets[2] = pressure_fes.GetTrueVSize();
   block_offsets.PartialSum();

   NavierStokesOperator navierstokes(velocity_fes, pressure_fes, block_offsets,
                                     vel_ess_bdr_attr, vel_ess_tdofs, kinematic_viscosity, velocity_ir, pressure_ir);

   navierstokes.SetDVelocityDtDirichlet(&u_coef, &dudt_coef);

   BlockVector x(block_offsets);
   u.ParallelProject(x.GetBlock(0));

   RK3SSPSolver ode_solver;

   real_t t = 0.0;
   ode_solver.Init(navierstokes);

   ParGridFunction w(&pressure_fes);
   CurlGridFunctionCoefficient curlu(&u);
   w.ProjectCoefficient(curlu);

   ParaViewDataCollection dc("dfem_navier_stokes_edac", &mesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(polynomial_order);
   dc.RegisterField("velocity", &u);
   dc.RegisterField("pressure", &p);
   dc.RegisterField("curl_u", &w);
   dc.SetTime(t);
   dc.SetCycle(0);
   dc.Save();

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = std::min(dt, t_final - t);
      ode_solver.Step(x, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            std::cout << "time step: " << ti << ", time: " << t << std::endl;
         }

         u.SetFromTrueDofs(x.GetBlock(0));
         p.SetFromTrueDofs(x.GetBlock(1));
         w.ProjectCoefficient(curlu);

         dc.SetTime(t);
         dc.SetCycle(ti);
         dc.Save();
      }
   }

   ParMixedBilinearForm div_form(&velocity_fes, &pressure_fes);
   div_form.AddDomainIntegrator(new VectorDivergenceIntegrator);
   div_form.Assemble();

   div_form.Mult(u, p);

   out << p.Norml2() << "\n";


   return 0;
}
