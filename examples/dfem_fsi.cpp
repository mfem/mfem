#include "dfem/dfem_refactor.hpp"
#include "general/tic_toc.hpp"
#include "linalg/hypre.hpp"
#include "linalg/solvers.hpp"

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

enum ProblemType
{
   CFD_EX_TEST = 0,
   DFG_CFD1_TEST = 1,
   DFG_CFD2_TEST = 2,
};

float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f)
{
   if (x < lowerlimit) { return lowerlimit; }
   if (x > upperlimit) { return upperlimit; }
   return x;
}

struct CFD_TEST_CTX
{
   const int L = 1;
} cfd_test_ctx;

template <int dim = 2>
struct VelocityMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{v * det(J) * w};
   }
};

template <int dim = 2>
struct PressureMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const real_t &p,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{p * det(J) * w};
   }
};

template <int dim = 2>
struct NavierStokesMomentumConvectiveQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim> &v,
                   const tensor<real_t, dim, dim> &dvdxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{dot(dvdxi * inv(J), v) * det(J) * w};
   }
};

template <int dim = 2>
struct NavierStokesMomentumViscousQFunction
{
   NavierStokesMomentumViscousQFunction(real_t &kinematic_viscosity) :
      kinematic_viscosity(kinematic_viscosity) {};

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dvdxi,
                   const real_t &p,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dvdxi * invJ;
      auto viscous_stress = -p * I + 2.0 * kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{(viscous_stress) * JxW};
   }
   const real_t kinematic_viscosity;
};

template <int dim = 2>
struct NavierStokesContinuityQFunction
{
   NavierStokesContinuityQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, dim, dim> &dvdxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{tr(dvdxi * inv(J)) * det(J) * w};
   }
};

class ALEFSIOperator : public TimeDependentOperator
{
   static constexpr int Position = 0;
   static constexpr int Velocity = 1;
   static constexpr int Pressure = 2;

   class ALEFSIResidual : public Operator
   {
   public:
      class ALEFSIResJac : public Operator
      {
      public:
         ALEFSIResJac(const ALEFSIResidual &res, const Vector &u) :
            Operator(u.Size()),
            res(res),
            u(u)
         {
            fd_jacobian = std::make_shared<FDJacobian>(res, u);
         }

         void Mult(const Vector &x, Vector &y) const override
         {
            fd_jacobian->Mult(x, y);
         }

         const ALEFSIResidual &res;
         const Vector u;
         mutable std::shared_ptr<FDJacobian> fd_jacobian;
      };

      class ALEFSIJacPrec : public Solver
      {
      public:
         ALEFSIJacPrec() : Solver() {}

         void SetOperator(const Operator &jac) override
         {
            this->height = jac.Height();
            this->width = jac.Width();

            auto alefsi_jac = dynamic_cast<const ALEFSIResJac*>(&jac);
            MFEM_VERIFY(alefsi_jac != nullptr, "invalid operator");

            const ALEFSIResidual &res = alefsi_jac->res;
            ALEFSIOperator &op = res.op;

            Vector uv, up;
            auto uptr = const_cast<Vector*>(&alefsi_jac->u);
            uv.MakeRef(*uptr, 0, op.H1vtsize);
            up.MakeRef(*uptr, op.H1vtsize, op.H1tsize);

            auto x_gf = static_cast<ParGridFunction*>(op.H1vfes.GetParMesh()->GetNodes());
            op.v_gf.SetFromTrueDofs(uv);
            op.p_gf.SetFromTrueDofs(up);

            HypreParMatrix Mv, Mp, Aconv, Avisc, B;

            auto dMDv = op.velocity_mass->GetDerivative(Velocity, {&op.v_gf}, {x_gf});
            auto dMDp = op.pressure_mass->GetDerivative(Pressure, {&op.p_gf}, {x_gf});
            auto dFcvDv = op.fluid_momentum_convective->GetDerivative(Velocity, {&op.v_gf}, {x_gf});
            auto dFvvDv = op.fluid_momentum_viscous->GetDerivative(Velocity, {&op.v_gf}, {&op.p_gf, x_gf});
            auto dCDv = op.fluid_continuity->GetDerivative(Velocity, {&op.v_gf}, {&op.p_gf, x_gf});

            dMDv->Assemble(Mv);
            dMDp->Assemble(Mp);
            dFcvDv->Assemble(Aconv);
            dFvvDv->Assemble(Avisc);
            dCDv->Assemble(B);

            std::shared_ptr<HypreParMatrix> A0, A;
            A0.reset(Add(1.0, Mv, op.theta * res.gamma, Aconv));
            A.reset(Add(1.0, *A0, op.theta * res.gamma, Avisc));
            auto Bt = B.Transpose();

            Array2D<const HypreParMatrix*> blocks(2, 2);
            blocks(0, 0) = A.get();
            blocks(0, 1) = Bt;
            blocks(1, 0) = &B;
            blocks(1, 1) = &Mp;

            Array2D<real_t> blockCoeff(2, 2);
            blockCoeff(0, 0) = 1.0;
            blockCoeff(0, 1) = -1.0 * res.gamma;
            blockCoeff(1, 0) = -1.0;
            blockCoeff(1, 1) = 0.0;

            K.reset(HypreParMatrixFromBlocks(blocks, &blockCoeff));

            // std::ofstream kmout("K.m");
            // kmout.precision(16);
            // K->PrintMatlab(kmout);
            // kmout.close();

            Array<int> combined_ess_tdof(op.vel_ess_tdof.Size() + op.pres_ess_tdof.Size());
            for (int i = 0; i < op.vel_ess_tdof.Size(); i++)
            {
               combined_ess_tdof[i] = op.vel_ess_tdof[i];
            }
            for (int i = 0; i < op.pres_ess_tdof.Size(); i++)
            {
               combined_ess_tdof[i + op.vel_ess_tdof.Size()] =
                  op.pres_ess_tdof[i] + op.H1vtsize;
            }

            auto Ke = K->EliminateRowsCols(combined_ess_tdof);
            delete Ke;

            // std::ofstream kemout("Kelim.m");
            // kemout.precision(16);
            // K->PrintMatlab(kemout);
            // kemout.close();

            // auto fd_jacobian = std::make_shared<FDJacobian>(res, alefsi_jac->u);
            // std::ofstream fdout("Kfd.m");
            // fdout.precision(16);
            // fd_jacobian->PrintMatlab(fdout);
            // fdout.close();

            // exit(0);

            slu = std::make_shared<SuperLUSolver>(MPI_COMM_WORLD);
            slu->SetPrintStatistics(false);
            A_SLU = std::make_shared<SuperLURowLocMatrix>(*K);
            slu->SetOperator(*A_SLU);
         }

         void Mult(const Vector &x, Vector &y) const override
         {
            // GMRESSolver krylov(MPI_COMM_WORLD);
            // krylov.SetRelTol(1e-8);
            // krylov.SetMaxIter(1000);
            // krylov.SetOperator(*K);
            // krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
            // krylov.Mult(x, y);

            slu->Mult(x, y);

            // y = x;
         }

         std::shared_ptr<HypreParMatrix> K;
         std::shared_ptr<SuperLURowLocMatrix> A_SLU;
         std::shared_ptr<SuperLUSolver> slu;
      };

      ALEFSIResidual(ALEFSIOperator &op, const real_t &gamma, const Vector &S,
                     const Vector &prevS) :
         Operator(op.offsets.Last()),
         op(op),
         gamma(gamma),
         prevu(prevS),
         z(S.Size()),
         v_gf(&op.H1vfes),
         p_gf(&op.H1fes),
         H1vtsize(op.H1vfes.GetTrueVSize()),
         H1tsize(op.H1fes.GetTrueVSize()) {}

      void Mult(const Vector &u, Vector &R) const override
      {
         auto uptr = const_cast<Vector*>(&u);

         Vector uv, up;
         uv.MakeRef(*uptr, 0, H1vtsize);
         up.MakeRef(*uptr, H1vtsize, H1tsize);

         Vector prevuv, prevup;
         prevuv.MakeRef(prevu, 0, H1vtsize);
         prevup.MakeRef(prevu, H1vtsize, H1tsize);

         Vector Rv, Rp;
         Rv.MakeRef(R, 0, H1vtsize);
         Rp.MakeRef(R, H1vtsize, H1tsize);

         Rv = 0.0;
         Rp = 0.0;

         auto x_gf = static_cast<ParGridFunction*>(op.H1vfes.GetParMesh()->GetNodes());

         op.velocity_mass->SetParameters({x_gf});
         op.velocity_mass->Mult(uv, Rv);

         // Current F_v(U)
         op.fluid_momentum_convective->SetParameters({x_gf});
         op.fluid_momentum_convective->AddMult(uv, Rv, op.theta*gamma);
         p_gf.SetFromTrueDofs(up);
         p_gf *= 1.0 / op.theta;
         op.fluid_momentum_viscous->SetParameters({&p_gf, x_gf});
         op.fluid_momentum_viscous->AddMult(uv, Rv, op.theta*gamma);

         // Previous F_v(U)
         op.fluid_momentum_convective->SetParameters({x_gf});
         op.fluid_momentum_convective->AddMult(prevuv, Rv, (1.0-op.theta)*gamma);
         p_gf = 0.0;
         op.fluid_momentum_viscous->SetParameters({&p_gf, x_gf});
         op.fluid_momentum_viscous->AddMult(prevuv, Rv, (1.0-op.theta)*gamma);

         // Previous time stepping terms
         op.velocity_mass->SetParameters({x_gf});
         op.velocity_mass->AddMult(prevuv, Rv, -1.0);

         p_gf.SetFromTrueDofs(up);
         op.fluid_continuity->SetParameters({&p_gf, x_gf});
         op.fluid_continuity->AddMult(uv, Rp, -1.0);

         Rv.SetSubVector(op.vel_ess_tdof, 0.0);
         Rp.SetSubVector(op.pres_ess_tdof, 0.0);
      }

      Operator& GetGradient(const Vector &u) const override
      {
         jacobian.reset(new ALEFSIResJac(*this, u));
         return *jacobian;
      }

      ALEFSIOperator &op;
      const real_t gamma;

      mutable ParGridFunction v_gf, p_gf;

      const int H1vtsize;
      const int H1tsize;

      mutable Vector z, prevu;

      mutable std::shared_ptr<ALEFSIResJac> jacobian;
   };

public:
   ALEFSIOperator(
      real_t theta,
      real_t &kinematic_viscosity,
      ParFiniteElementSpace &H1vfes,
      ParFiniteElementSpace &H1fes,
      Array<int> &offsets,
      Array<int> &vel_ess_bdr,
      Array<int> &pres_ess_bdr,
      VectorCoefficient &vel_bdr_coeff,
      Coefficient &pres_bdr_coeff,
      const IntegrationRule &ir,
      bool enable_tps) :
      TimeDependentOperator(offsets.Last()),
      theta(theta),
      kinematic_viscosity(kinematic_viscosity),
      offsets(offsets),
      H1vfes(H1vfes),
      H1fes(H1fes),
      H1vtsize(H1vfes.GetTrueVSize()),
      H1tsize(H1fes.GetTrueVSize()),
      vel_ess_bdr(vel_ess_bdr),
      pres_ess_bdr(pres_ess_bdr),
      vel_bdr_coeff(&vel_bdr_coeff),
      pres_bdr_coeff(&pres_bdr_coeff),
      ir(ir),
      prevS(offsets.Last()),
      v_gf(&H1vfes),
      p_gf(&H1fes)
   {
      auto mesh = H1vfes.GetParMesh();
      x_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *x_gf->ParFESpace();

      H1vfes.GetEssentialTrueDofs(vel_ess_bdr, vel_ess_tdof);
      H1fes.GetEssentialTrueDofs(pres_ess_bdr, pres_ess_tdof);

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         velocity_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            velocity_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Velocity>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Velocity>{}};

         auto mass_qf = VelocityMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         velocity_mass->AddDomainIntegrator(mass_qf, inputs, outputs, ir, derivatives);
      }

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_momentum_convective =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_momentum_convective->DisableTensorProductStructure();
         }

         mfem::tuple inputs
         {
            Value<Velocity>{},
            Gradient<Velocity>{},
            Gradient<Position>{},
            Weight{}
         };

         mfem::tuple outputs{Value<Velocity>{}};

         auto momentum_qf = NavierStokesMomentumConvectiveQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         fluid_momentum_convective->AddDomainIntegrator(
            momentum_qf, inputs, outputs, ir, derivatives);
      }

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_momentum_viscous =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_momentum_viscous->DisableTensorProductStructure();
         }

         mfem::tuple inputs
         {
            Gradient<Velocity>{},
            Value<Pressure>{},
            Gradient<Position>{},
            Weight{}
         };

         mfem::tuple outputs{Gradient<Velocity>{}};

         auto momentum_qf =
            NavierStokesMomentumViscousQFunction<DIMENSION>(kinematic_viscosity);
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         fluid_momentum_viscous->AddDomainIntegrator(
            momentum_qf, inputs, outputs, ir, derivatives);
      }

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &H1vfes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
            FieldDescriptor{Position, &mesh_fes}
         };

         fluid_continuity =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            fluid_continuity->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Gradient<Velocity>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Pressure>{}};
         auto continuity_qf = NavierStokesContinuityQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         fluid_continuity->AddDomainIntegrator(
            continuity_qf, inputs, outputs, ir, derivatives);
      }

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Pressure, &H1fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         pressure_mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         if (!enable_tps)
         {
            pressure_mass->DisableTensorProductStructure();
         }

         mfem::tuple inputs{Value<Pressure>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Pressure>{}};
         auto pressure_mass_qf = PressureMassQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Pressure> {};
         pressure_mass->AddDomainIntegrator(
            pressure_mass_qf, inputs, outputs, ir, derivatives);
      }
   }

   void SetTime(const real_t t) override
   {
      vel_bdr_coeff->SetTime(t);
      pres_bdr_coeff->SetTime(t);
   }

   void Step(Vector &S, real_t &t, const real_t &dt)
   {
      this->SetTime(t);
      prevS = S;

      this->SetTime(t + dt);
      Vector Sv, Sp;
      Sv.MakeRef(S, 0, H1vtsize);
      Sp.MakeRef(S, H1vtsize, H1tsize);

      v_gf.SetFromTrueDofs(Sv);
      v_gf.ProjectBdrCoefficient(*vel_bdr_coeff, vel_ess_bdr);
      v_gf.GetTrueDofs(Sv);

      p_gf.SetFromTrueDofs(Sp);
      p_gf.ProjectBdrCoefficient(*pres_bdr_coeff, pres_ess_bdr);
      p_gf.GetTrueDofs(Sp);

      ALEFSIResidual residual(*this, dt, S, prevS);

      ALEFSIOperator::ALEFSIResidual::ALEFSIJacPrec prec;

      FGMRESSolver krylov(MPI_COMM_WORLD);
      krylov.SetRelTol(1e-4);
      krylov.SetMaxIter(1000);
      krylov.SetPreconditioner(prec);
      krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetOperator(residual);
      newton.SetSolver(prec);
      newton.SetRelTol(1e-8);
      newton.SetMaxIter(10);
      newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());

      Vector zero;
      newton.Mult(zero, S);

      t += dt;
   }

   real_t theta;
   real_t kinematic_viscosity;
   Array<int> offsets;
   std::shared_ptr<DifferentiableOperator> fluid_momentum_convective;
   std::shared_ptr<DifferentiableOperator> fluid_momentum_viscous;
   std::shared_ptr<DifferentiableOperator> fluid_continuity;
   std::shared_ptr<DifferentiableOperator> velocity_mass;
   std::shared_ptr<DifferentiableOperator> pressure_mass;

   ParGridFunction *x_gf, v_gf, p_gf;

   Vector prevS;

   const Array<int> vel_ess_bdr, pres_ess_bdr;
   Array<int> vel_ess_tdof, pres_ess_tdof;

   VectorCoefficient *vel_bdr_coeff;
   Coefficient *pres_bdr_coeff;

   ParFiniteElementSpace &H1vfes;
   ParFiniteElementSpace &H1fes;
   const int H1vtsize, H1tsize;
   IntegrationRule ir;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();
   Hypre::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "";
   int polynomial_order_velocity = 2;
   int refinements = 0;
   int problem_type = 0;
   real_t t_final = 0.0;
   real_t dt = 1e-3;
   real_t kinematic_viscosity = 1.0;
   int vis_steps = 1;
   real_t theta = 0.5;
   bool enable_tps = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem_type, "-prob", "--problem", "Problem #");
   args.AddOption(&polynomial_order_velocity, "-ov", "--order-velocity", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&dt, "-dt", "--dt", "");
   args.AddOption(&kinematic_viscosity, "-kv", "--kinematic-viscosity", "");
   args.AddOption(&theta, "-theta", "--theta", "");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&enable_tps, "-tps", "--enable-tps", "-no-tps",
                  "--no-tps", "Enable tensor product structure for quad/hex.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   int polynomial_order_pressure = polynomial_order_velocity - 1;

   Mesh mesh_serial;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      mesh_serial.EnsureNodes();
      auto nodes = mesh_serial.GetNodes();
      *nodes -= 0.5;
      *nodes *= cfd_test_ctx.L;
   }
   else if (problem_type == ProblemType::DFG_CFD1_TEST ||
            problem_type == ProblemType::DFG_CFD2_TEST)
   {
      mesh_serial = Mesh::LoadFromFile(mesh_file);
      // Array<int> domains(1);
      // domains[0] = 1;
      // mesh_serial = SubMesh::CreateFromDomain(m, domains);
      mesh_serial.EnsureNodes();
   }
   else
   {
      MFEM_ABORT("invalid problem type");
   }
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.EnsureNodes();
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   H1_FECollection velocity_fec(polynomial_order_velocity, dim);
   H1_FECollection pressure_fec(polynomial_order_pressure);

   ParFiniteElementSpace H1vfes(&mesh, &velocity_fec, dim);
   ParFiniteElementSpace H1fes(&mesh, &pressure_fec);

   HYPRE_BigInt global_size_velocity = H1vfes.GlobalTrueVSize();
   HYPRE_BigInt global_size_pressure = H1fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "Number of velocity unknowns: " << global_size_velocity << "\n";
      out << "Number of pressure unknowns: " << global_size_pressure << "\n";
   }

   const IntegrationRule &integration_rule =
      IntRules.Get(H1vfes.GetFE(0)->GetGeomType(),
                   2 * H1vfes.GetFE(0)->GetOrder() + 1);

   Array<int> vel_ess_attr(mesh.bdr_attributes.Max());
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      vel_ess_attr = 1;
   }
   else if (problem_type == ProblemType::DFG_CFD1_TEST ||
            problem_type == ProblemType::DFG_CFD2_TEST)
   {
      // everywhere
      vel_ess_attr = 1;
      // beam
      vel_ess_attr[4] = 0;
      // outlet
      vel_ess_attr[1] = 0;
   }

   Array<int> pres_ess_attr(mesh.bdr_attributes.Max());
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      pres_ess_attr = 1;
   }
   else if (problem_type == ProblemType::DFG_CFD1_TEST ||
            problem_type == ProblemType::DFG_CFD2_TEST)
   {
      // everywhere
      pres_ess_attr = 0;
      // outlet
      // pres_ess_attr[1] = 1;
   }

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = H1vfes.GetTrueVSize();
   block_offsets[2] = H1fes.GetTrueVSize();
   block_offsets.PartialSum();

   BlockVector S(block_offsets, Device::GetDeviceMemoryType());

   ParGridFunction v_gf(&H1vfes), p_gf(&H1fes);

   std::function<void(const Vector &, real_t, Vector &)> velocity_exact;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      velocity_exact = [nu = kinematic_viscosity](
                          const Vector &coords, real_t t, Vector &u)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t f = exp(-4.0 * nu * M_PI * M_PI * t);
         u(0) = -sin(2.0 * M_PI * y) * f;
         u(1) = sin(2.0 * M_PI * x) * f;
      };
   }
   else if (problem_type == ProblemType::DFG_CFD1_TEST ||
            problem_type == ProblemType::DFG_CFD2_TEST)
   {
      velocity_exact = [problem_type](const Vector &coords, real_t t, Vector &u)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t H = 0.41;
         real_t U = 0.3;
         if (problem_type == DFG_CFD2_TEST)
         {
            U = 1.5;
         }
         auto smoothstep = [t](const real_t edge0, const real_t edge1, real_t x)
         {
            x = clamp((x - edge0) / (edge1 - edge0));
            return x * x * (3.0 - 2.0 * x);
         };
         if (x == 0.0)
         {
            u(0) = 4.0 * U * y * (H - y) / powf(H, 2.0) * smoothstep(0.0, 0.1, t);
         }
         else
         {
            u(0) = 0.0;
         }
         u(1) = 0.0;
      };
   }

   std::function<real_t(const Vector &, real_t)> pressure_exact;
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      pressure_exact = [nu = kinematic_viscosity](const Vector &coords, real_t t)
      {
         const real_t x = coords(0);
         const real_t y = coords(1);
         const real_t f = exp(-8.0 * nu * M_PI * M_PI * t);
         return -cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y) * f;
      };
   }
   else if (problem_type == ProblemType::DFG_CFD1_TEST ||
            problem_type == ProblemType::DFG_CFD2_TEST)
   {
      pressure_exact = [](const Vector &, real_t)
      {
         return 0.0;
      };
   }

   VectorFunctionCoefficient vel_exact(dim, velocity_exact);
   FunctionCoefficient pres_exact(pressure_exact);

   v_gf.ProjectCoefficient(vel_exact);
   v_gf.GetTrueDofs(S.GetBlock(0));

   p_gf.ProjectCoefficient(pres_exact);
   p_gf.GetTrueDofs(S.GetBlock(1));

   ALEFSIOperator alefsi(
      theta,
      kinematic_viscosity,
      H1vfes,
      H1fes,
      block_offsets,
      vel_ess_attr,
      pres_ess_attr,
      vel_exact,
      pres_exact,
      integration_rule,
      enable_tps);

   real_t t = 0.0;
   out << "time step: " << dt << "\n";
   real_t t_old;
   bool last_step = false;

   ParGridFunction verr_gf(&H1vfes), vex_gf(&H1vfes);
   verr_gf = 0.0;

   ParGridFunction perr_gf(&H1fes), pex_gf(&H1fes);
   perr_gf = 0.0;

   ParaViewDataCollection dc("dfem_fsi", &mesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(polynomial_order_velocity);
   dc.RegisterField("velocity", &v_gf);
   dc.RegisterField("pressure", &p_gf);
   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      dc.RegisterField("velocity_exact", &vex_gf);
      dc.RegisterField("velocity_error", &verr_gf);
      dc.RegisterField("pressure_exact", &pex_gf);
      dc.RegisterField("pressure_error", &perr_gf);
   }
   dc.SetCycle(0);
   dc.SetTime(0);
   dc.Save();

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - 1e-8*dt)
      {
         dt = t_final - t;
         last_step = true;
      }

      alefsi.Step(S, t, dt);

      if (last_step || (ti % vis_steps == 0))
      {
         v_gf.SetFromTrueDofs(S.GetBlock(0));
         p_gf.SetFromTrueDofs(S.GetBlock(1));

         if (Mpi::Root())
         {
            out << "step " << std::setw(5) << ti
                << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt;
            out << std::endl;
         }

         dc.SetCycle(ti);
         dc.SetTime(t);
         dc.Save();
      }
   }

   if (problem_type == ProblemType::CFD_EX_TEST)
   {
      vel_exact.SetTime(t);
      vex_gf.ProjectCoefficient(vel_exact);
      real_t vel_l2err = v_gf.ComputeL2Error(vel_exact);

      pres_exact.SetTime(t);
      pex_gf.ProjectCoefficient(pres_exact);
      real_t pres_l2err = p_gf.ComputeL2Error(pres_exact);

      if (Mpi::Root())
      {
         out << "|u - u_exact|_L2 = " << vel_l2err
             << "\n|p - p_exact|_L2 = " << pres_l2err;
      }

      for (int i = 0; i < verr_gf.Size(); i++)
      {
         verr_gf(i) = abs(vex_gf(i) - v_gf(i));
      }

      for (int i = 0; i < perr_gf.Size(); i++)
      {
         perr_gf(i) = abs(pex_gf(i) - p_gf(i));
      }

      dc.Save();
   }

   if (problem_type == ProblemType::DFG_CFD1_TEST ||
       problem_type == ProblemType::DFG_CFD2_TEST)
   {
      DenseMatrix points(dim, 1);
      Vector pointA(2), pointB(2);

      pointA(0) = 0.25;
      pointA(1) = 0.2;

      pointB(0) = 0.15;
      pointB(1) = 0.2;

      Array<int> elem_ids;
      Array<IntegrationPoint> ips;
      points.SetCol(0, pointA);
      mesh.FindPoints(points, elem_ids, ips);
      real_t pA = p_gf.GetValue(elem_ids[0], ips[0]);

      points.SetCol(0, pointB);
      mesh.FindPoints(points, elem_ids, ips);
      real_t pB = p_gf.GetValue(elem_ids[0], ips[0]);

      out << "p(B) - p(A) = " << pB - pA << "\n";
   }

   Hypre::Finalize();
   Mpi::Finalize();

   return 0;
}
