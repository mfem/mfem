#include "dfem/dfem_refactor.hpp"

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

template <int dim = DIMENSION>
struct MomentumRefStateQFunction
{
   MomentumRefStateQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, dim, dim> &dudxi,
      const tensor<real_t, dim, dim> &J,
      const double &w) const
   {
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      constexpr real_t nu = 0.4;
      constexpr real_t mu = 0.5 * 10e6;
      constexpr real_t lambda = -((mu * nu) / (-1.0 + 2.0 * nu));

      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto F = I + dudx;
      auto detF = det(F);
      // St. Venant-Kirchhoff model
      auto E = 0.5 * (transpose(F) * F - I);
      auto PK2 = lambda * tr(E) * I + 2.0 * mu * E;
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{F * PK2 * JxW};
      // auto invJ = inv(J);
      // auto eps = sym(dudxi * invJ);
      // return mfem::tuple{transpose(lambda * tr(eps) * I + 2.0 * mu * eps) * det(J) * w * transpose(invJ)};
   }
};

class ElasticityOperator : public Operator
{
   static constexpr int Displacement = 0;
   static constexpr int Coordinates = 1;

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
         momentum_du = elasticity->momentum->GetDerivative(Displacement, {&u}, {mesh_nodes});
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

   ElasticityOperator(ParFiniteElementSpace &displacement_fes,
                      Array<int> &vel_ess_tdofs,
                      const IntegrationRule &displacement_ir) :
      Operator(displacement_fes.GetTrueVSize()),
      density(1.0e3),
      displacement_ess_tdof(vel_ess_tdofs),
      displacement_fes(displacement_fes),
      displacement_ir(displacement_ir),
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
            FieldDescriptor{Coordinates, &mesh_fes}
         };

         momentum =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
         // momentum->DisableTensorProductStructure();

         mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple outputs{Gradient<Displacement>{}};

         auto momentum_qf = MomentumRefStateQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Displacement> {};
         momentum->AddDomainIntegrator(
            momentum_qf, inputs, outputs, displacement_ir, derivatives);
      }

      {
         Vector g(DIMENSION);
         g = 0.0;
         g(1) = 2.0 * density;

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
      momentum->SetParameters({mesh_nodes});
      momentum->Mult(displacement, r);
      r -= body_force;
      r.SetSubVector(displacement_ess_tdof, 0.0);
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

   mutable std::shared_ptr<ElasticityJacobianOperator> jacobian_operator;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;
};

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

   Mesh mesh_serial = Mesh(mesh_file);
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.EnsureNodes();
   mesh_serial.Clear();

   Array<int> beam_attributes(1);
   beam_attributes[0] = 2;
   auto mesh_beam = ParSubMesh::CreateFromDomain(mesh, beam_attributes);

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

   Array<int> bdr_attr_is_ess(mesh_beam.bdr_attributes.Max());
   out << bdr_attr_is_ess.Size() << "\n";
   bdr_attr_is_ess = 0;
   bdr_attr_is_ess[6] = 1;
   Array<int> displacement_ess_tdof;
   displacement_fes.GetEssentialTrueDofs(bdr_attr_is_ess, displacement_ess_tdof);

   ParGridFunction u(&displacement_fes);
   // u.Randomize(1234);
   // u.SetSubVector(displacement_ess_tdofs, 0.0);
   u = 0.0;

   ElasticityOperator elasticity(displacement_fes, displacement_ess_tdof,
                                 displacement_ir);

   ElasticityOperator::ElasticityJacobianPreconditioner prec;

   CGSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-4);
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
   nonlinear_solver->SetRelTol(1e-6);
   nonlinear_solver->SetMaxIter(50);
   nonlinear_solver->SetSolver(solver);
   nonlinear_solver->SetPrintLevel(1);

   Vector zero, x(displacement_fes.GetTrueVSize());
   u.GetTrueDofs(x);

   nonlinear_solver->Mult(zero, x);

   u.SetFromTrueDofs(x);

   // Compute Newton residual
   Vector r(displacement_fes.GetTrueVSize());
   elasticity.Mult(x, r);
   r.SetSubVector(displacement_ess_tdof, 0.0);
   double rnorm = r.Norml2();
   out << "||F(x) - b||_2 = " << rnorm << "\n";

   // Compute CG residual
   Vector z(displacement_fes.GetTrueVSize());
   z = x;
   z.SetSubVector(displacement_ess_tdof, 0.0);
   elasticity.GetGradient(x).Mult(z, r);
   r.Neg();
   r += elasticity.body_force;
   for (int i = 0; i < displacement_ess_tdof.Size(); i++)
   {
      r[displacement_ess_tdof[i]] = x[displacement_ess_tdof[i]];
   }
   double cg_rnorm = r.Norml2();
   out << "||b - Ax||_2 = " << cg_rnorm << "\n";
   out << "||b||_2 = " << elasticity.body_force.Norml2() << "\n";
   out << "||b - Ax||_2 / ||b||_2 = " << cg_rnorm / elasticity.body_force.Norml2()
       << "\n";

   ParaViewDataCollection dc("dfem_elasticity", &mesh_beam);
   dc.SetHighOrderOutput(true);
   dc.RegisterField("deformation", &u);
   dc.Save();

   return 0;
}
