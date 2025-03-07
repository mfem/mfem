#include "dfem/dfem_refactor.hpp"

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

template <typename T, int dim>
MFEM_HOST_DEVICE inline
tensor<T, 3, 3> tensor_to_3D(const tensor<T, dim, dim>& A)
{
   tensor<T, 3, 3> A3D{};
   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         A3D[i][j] = A[i][j];
      }
   }
   return A3D;
}

template <typename Material, int dim = DIMENSION>
struct InternalStateQFunction
{
   InternalStateQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, dim, dim> &dudxi,
      const tensor<real_t, dim, dim> &J,
      const tensor<real_t, 10> &internal_state,
      const double &w) const
   {
      auto invJ = inv(J);
      auto dudX = dudxi * invJ;
      auto dudX3D = tensor_to_3D(dudX);
      auto P3D = material.UpdateInternalState(dudX3D, internal_state);
      auto Q = mfem::internal::make_tensor<dim, dim>([&P3D](int i, int j) { return P3D[i][j]; });
      return mfem::tuple{Q};
   }

   Material material;
};

template <typename Material, int dim = DIMENSION>
struct MomentumRefStateQFunction
{
   MomentumRefStateQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, dim, dim> &dudxi,
      const tensor<real_t, dim, dim> &J,
      const tensor<real_t, 10> &internal_state,
      const double &w) const
   {
      auto invJ = inv(J);
      auto dudX = dudxi * invJ;
      auto dudX3D = tensor_to_3D(dudX);
      auto P3D = material.UpdateStress(dudX3D, internal_state);
      auto P = mfem::internal::make_tensor<dim, dim>([&P3D](int i, int j) { return P3D[i][j]; });
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{P * JxW};
   }

   Material material;
};


struct StVenantKirchhoff
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, 3, 3> & dudX) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<3>();
      const real_t lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);
      auto E = 0.5 * (dudX + transpose(dudX) + dot(transpose(dudX), dudX));
      auto S = lambda * tr(E) * I + 2.0 * mu * E;
      auto F = mfem::internal::IsotropicIdentity<3>() + dudX;
      return dot(F, S);
   }
   
   real_t mu;
   real_t nu;
};

struct J2SmallStrain {
  static constexpr int dim = 3;         ///< spatial dimension
  static constexpr int n_internal_states = 10;
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  real_t E;                 ///< Young's modulus
  real_t nu;                ///< Poisson's ratio
  real_t sigma_y;                ///< Yield strength
  double Hk;                ///< Kinematic hardening modulus
  double density;           ///< Mass density

  /// @brief variables required to characterize the hysteresis response
  struct InternalState {
    tensor<double, dim, dim> plastic_strain;  ///< plastic strain
    double accumulated_plastic_strain;        ///< uniaxial equivalent plastic strain
  };

  MFEM_HOST_DEVICE inline
  InternalState unpack_internal_state(const tensor<real_t, n_internal_states> & packed_state) const
  {
      // we could use type punning here to avoid copies
      auto plastic_strain = mfem::internal::make_tensor<dim, dim>(
         [&packed_state](int i, int j) { return packed_state[dim*i + j]; });
      real_t accumulated_plastic_strain = packed_state[n_internal_states - 1];
      return {plastic_strain, accumulated_plastic_strain};
  }

  MFEM_HOST_DEVICE inline
  tensor<real_t, n_internal_states> pack_internal_state(const tensor<real_t, dim, dim> & plastic_strain, real_t accumulated_plastic_strain) const
  {
      tensor<real_t, n_internal_states> packed_state{};
      for (int i = 0, ij = 0; i < dim; i++) {
         for (int j = 0; j < dim; j++, ij++) {
            packed_state[ij] = plastic_strain[i][j];
         }
      }
      packed_state[n_internal_states - 1] = accumulated_plastic_strain;
      return packed_state;
  }

  MFEM_HOST_DEVICE inline
  tuple<tensor<real_t, dim, dim>, tensor<real_t, n_internal_states>>
  Update(const tensor<real_t, dim, dim> & dudX, const tensor<real_t, n_internal_states> & internal_state) const
  {
      auto I = mfem::internal::Identity<dim>();
      const real_t K = E / (3.0 * (1.0 - 2.0 * nu));
      const real_t G = 0.5 * E / (1.0 + nu);

      auto [plastic_strain, accumulated_plastic_strain] = unpack_internal_state(internal_state);

      // (i) elastic predictor
      auto el_strain = sym(dudX) - plastic_strain;
      auto p = K * tr(el_strain);
      auto s = 2.0 * G * dev(el_strain);
      auto sigma_b = 2.0 / 3.0 * Hk * plastic_strain;
      auto eta = s - sigma_b;
      auto q = sqrt(1.5) * norm(eta);
      real_t delta_eqps = 0.0;

      // (ii) admissibility
      if (q > tol * sigma_y) {
         // (iii) return mapping
         real_t delta_eqps = (q - sigma_y)/(3*G + Hk);
         auto Np = 1.5 * eta / q;
         s = s - 2.0 * G * delta_eqps * Np;
         plastic_strain += delta_eqps * Np;
         accumulated_plastic_strain += delta_eqps;
         // out << accumulated_plastic_strain << std::endl;
      }
      auto stress = s + p * I;
      auto internal_state_new = pack_internal_state(plastic_strain, accumulated_plastic_strain);
      return {stress, internal_state_new};
  }

  MFEM_HOST_DEVICE inline
  auto UpdateStress(const tensor<real_t, dim, dim> & du_dX, const tensor<real_t, n_internal_states> & internal_state) const
  {
   return get<0>(Update(du_dX, internal_state));
  }

  MFEM_HOST_DEVICE inline
  auto UpdateInternalState(const tensor<real_t, dim, dim> & du_dX, const tensor<real_t, n_internal_states> & internal_state) const
  {
      return get<1>(Update(du_dX, internal_state));
  }
};

class ElasticityOperator : public Operator
{
   static constexpr int Displacement = 0;
   static constexpr int Coordinates = 1;
   static constexpr int InternalState = 2;

public:
   class ElasticityJacobianPreconditioner : public Solver
   {
   public:
      ElasticityJacobianPreconditioner() : Solver() {}

      void SetOperator(const Operator &op) override
      {
         this->height = op.Height();
         this->width = op.Width();

         auto elasticity_jacobian = dynamic_cast<const ElasticityJacobianOperator*>(&op);
         MFEM_VERIFY(elasticity_jacobian != nullptr, "invalid operator");

         A = std::make_shared<HypreParMatrix>();
         elasticity_jacobian->momentum_du->Assemble(*A);
         auto Ae = A->EliminateRowsCols(
                      elasticity_jacobian->elasticity->displacement_ess_tdof);
         delete Ae;

         amg = std::make_shared<HypreBoomerAMG>();
         amg->SetOperator(*A);
         amg->SetPrintLevel(0);
         amg->SetSystemsOptions(
            elasticity_jacobian->elasticity->mesh_nodes->ParFESpace()->GetMesh()->Dimension(),
            true);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         amg->Mult(x, y);
      }

      std::shared_ptr<HypreParMatrix> A;
      std::shared_ptr<HypreBoomerAMG> amg;
   };

   class ElasticityJacobianOperator : public Operator
   {
   public:
      ElasticityJacobianOperator(const ElasticityOperator *elasticity,
                                 const Vector &x) :
         Operator(elasticity->Height()),
         elasticity(elasticity),
         z(elasticity->Height())
      {
         ParGridFunction u(&elasticity->displacement_fes);
         u.SetFromTrueDofs(x);

         auto mesh_nodes = static_cast<ParGridFunction*>
                           (elasticity->displacement_fes.GetParMesh()->GetNodes());
         momentum_du = elasticity->momentum->GetDerivative(Displacement, {&u}, {mesh_nodes, &elasticity->internal_state});
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(elasticity->displacement_ess_tdof, 0.0);

         momentum_du->Mult(z, y);

         for (int i = 0; i < elasticity->displacement_ess_tdof.Size(); i++)
         {
            y[elasticity->displacement_ess_tdof[i]] =
               x[elasticity->displacement_ess_tdof[i]];
         }
      }

      const ElasticityOperator *elasticity;
      std::shared_ptr<DerivativeOperator> momentum_du;
      mutable Vector z;
   };

   template <typename Material>
   ElasticityOperator(ParFiniteElementSpace &displacement_fes,
                      Array<int> &vel_ess_tdofs,
                      const IntegrationRule &displacement_ir,
                      ParametricFunction &internal_state,
                      Material material) :
      Operator(displacement_fes.GetTrueVSize()),
      density(1.0e3),
      displacement_ess_tdof(vel_ess_tdofs),
      displacement_fes(displacement_fes),
      displacement_ir(displacement_ir),
      internal_state(internal_state),
      body_force(displacement_fes.GetTrueVSize())
   {
      auto mesh = displacement_fes.GetParMesh();
      mesh_nodes = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Displacement, &displacement_fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Coordinates, &mesh_fes},
            FieldDescriptor{InternalState, &internal_state.space}
         };

         momentum =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
         // momentum->DisableTensorProductStructure();

         mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Coordinates>{}, None<InternalState>{}, Weight{}};
         mfem::tuple outputs{Gradient<Displacement>{}};

         auto momentum_qf = MomentumRefStateQFunction<Material, DIMENSION> {.material = material};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         Array<int> solid_domain_attr(mesh->attributes.Max());
         solid_domain_attr[0] = 1;
         momentum->AddDomainIntegrator(
            momentum_qf, inputs, outputs, displacement_ir, solid_domain_attr, derivatives);
      }

      {
         Vector g(DIMENSION);
         g = 0.0;

         ParLinearForm body_force_lf(&displacement_fes);
         body_force_coef = new VectorConstantCoefficient(g);
         auto integ = new VectorDomainLFIntegrator(*body_force_coef);
         integ->SetIntRule(&displacement_ir);
         body_force_lf.AddDomainIntegrator(integ);
         body_force_lf.Assemble();
         body_force_lf.ParallelAssemble(body_force);
      }
   }

   void Mult(const Vector &displacement, Vector &r) const override
   {
      momentum->SetParameters({mesh_nodes, &internal_state});
      momentum->Mult(displacement, r);
      r -= body_force;
      r.SetSubVector(displacement_ess_tdof, 0.0);
   }

   void Reaction(const Vector &displacement, Vector &r) const
   {
      momentum->SetParameters({mesh_nodes, &internal_state});
      momentum->Mult(displacement, r);
      r -= body_force;
      r.Neg();
   }

   Operator &GetGradient(const Vector &x) const override
   {
      jacobian_operator = std::make_shared<ElasticityJacobianOperator>(this, x);
      return *jacobian_operator;

      // fd_jacobian = std::make_shared<FDJacobian>(*this, x);
      // return *fd_jacobian;
   }

   real_t density;
   std::shared_ptr<DifferentiableOperator> momentum;
   mutable std::shared_ptr<HypreParMatrix> A;
   VectorConstantCoefficient *body_force_coef = nullptr;
   Vector body_force;

   ParGridFunction *mesh_nodes;

   const Array<int> displacement_ess_tdof;

   ParFiniteElementSpace &displacement_fes;
   IntegrationRule displacement_ir;

   ParametricFunction& internal_state;

   mutable std::shared_ptr<ElasticityJacobianOperator> jacobian_operator;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;
};

#if 0
class InternalStateUpdater : public Operator
{
   public:
   
   InternalStateUpdater::InternalStateUpdater(ParFiniteElementSpace &displacement_fes,
                                              const IntegrationRule &displacement_ir,
                                              ParametricFunction &internal_state,
                                              Material material) :
      Operator(displacement_fes.GetTrueVSize()),
      displacement_ir(displacement_ir),
      internal_state(internal_state),
   {
      auto solutions = std::vector
      {
         FieldDescriptor{InternalState, &internal_state.space}
      };

      auto parameters = std::vector
      {
         FieldDescriptor{Coordinates, &mesh_fes},
         FieldDescriptor{Displacement, &displacement_fes}
      };

      auto mesh = displacement_fes.GetParMesh();

      dop = std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

      mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Coordinates>{}, None<InternalState>{}, Weight{}};
      mfem::tuple outputs{None<InternalState>{}};


   }


   IntegrationRule displacement_ir;

   ParametricFunction& internal_state;
};
#endif

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "/Users/andrej1/dump/fsi.msh";
   // const char* mesh_file = "../data/ref-square.mesh";
   int polynomial_order = 2;
   int ir_order = 2;
   int refinements = 0;
   int nonlinear_solver_type = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&nonlinear_solver_type, "-nls", "--nonlinear-solver", "");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   out << std::setprecision(8);

   Mesh mesh_serial = Mesh::MakeCartesian2D(20, 2, Element::QUADRILATERAL,
      false, 1.0, 0.1);
   mesh_serial.EnsureNodes();
   auto mesh_beam = ParMesh(MPI_COMM_WORLD, mesh_serial);

   out << "#el: " << mesh_beam.GetNE() << "\n";

   H1_FECollection displacement_fec(polynomial_order, dim);
   ParFiniteElementSpace displacement_fes(&mesh_beam, &displacement_fec, dim);

   HYPRE_BigInt global_size = displacement_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "Number of unknowns: " << global_size << "\n";
   }

   const IntegrationRule &displacement_ir =
      IntRules.Get(displacement_fes.GetFE(0)->GetGeomType(),
                   2 * ir_order + displacement_fes.GetFE(0)->GetOrder());
   
   constexpr int n_internal_state_variables = 10;
   ParametricSpace internal_state_space(dim, n_internal_state_variables, displacement_ir.GetNPoints(),
      n_internal_state_variables*displacement_ir.GetNPoints()*mesh_beam.GetNE());

   ParametricFunction internal_state(internal_state_space);
   internal_state = 0.0;

   Array<int> bdr_attr_is_ess(mesh_beam.bdr_attributes.Max());
   Array<int> displacement_ess_tdof;
   Array<int> bc_tdof;

   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[0] = 1;
   displacement_fes.GetEssentialTrueDofs(bdr_attr_is_ess, bc_tdof, 1);
   for (auto td : bc_tdof) { displacement_ess_tdof.Append(td); };

   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[3] = 1;
   displacement_fes.GetEssentialTrueDofs(bdr_attr_is_ess, bc_tdof, 0);
   for (auto td : bc_tdof) { displacement_ess_tdof.Append(td); };

   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[1] = 1;
   displacement_fes.GetEssentialTrueDofs(bdr_attr_is_ess, bc_tdof, 0);
   for (auto td : bc_tdof) { displacement_ess_tdof.Append(td); };

   // Applied displacement boundary condition
   constexpr real_t applied_displacement = 0.2;
   ParGridFunction u(&displacement_fes);
   u = 0.0;
   u.SetSubVector(bc_tdof, applied_displacement);

   using Material = J2SmallStrain; // StVenantKirchhoff
   Material material{.E = 1000.0, .nu = 0.25, .sigma_y = 0.53333, .Hk = 40.0};
   // Material material{.mu = 0.5e6, .nu = 0.4};

   ElasticityOperator elasticity(displacement_fes, displacement_ess_tdof,
                                 displacement_ir, internal_state, material);

   ElasticityOperator::ElasticityJacobianPreconditioner prec;

   CGSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-5);
   // solver.SetKDim(500);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(2);
   solver.SetPreconditioner(prec);

   // NewtonSolver newton(MPI_COMM_WORLD);
   // newton.SetOperator(elasticity);
   // newton.SetSolver(solver);
   // newton.SetRelTol(1e-12);
   // newton.SetMaxIter(50);
   // newton.SetPrintLevel(1);

   std::shared_ptr<NewtonSolver> nonlinear_solver;
   if (nonlinear_solver_type == 0)
   {
      nonlinear_solver = std::make_shared<NewtonSolver>(MPI_COMM_WORLD);
   }
   // else if (nonlinear_solver_type == 1)
   // {
   //    nonlinear_solver = std::make_shared<KINSolver>(MPI_COMM_WORLD, KIN_LINESEARCH);
   // }
   else
   {
      MFEM_ABORT("invalid nonlinear solver type");
   }
   nonlinear_solver->SetOperator(elasticity);
   nonlinear_solver->SetRelTol(1e-10);
   nonlinear_solver->SetMaxIter(50);
   nonlinear_solver->SetSolver(solver);
   nonlinear_solver->SetPrintLevel(1);

   Vector zero, x(displacement_fes.GetTrueVSize());
   u.GetTrueDofs(x);

   nonlinear_solver->Mult(zero, x);

   u.SetFromTrueDofs(x);

   // Compute reactions
   Vector r(displacement_fes.GetTrueVSize());
   elasticity.Reaction(x, r);
   ParGridFunction reaction(&displacement_fes);
   reaction.SetFromTrueDofs(r);

   ParaViewDataCollection dc("dfem_plasticity", &mesh_beam);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(1);
   dc.RegisterField("displacement", &u);
   dc.RegisterField("reaction", &reaction);
   dc.Save();

   return 0;
}
