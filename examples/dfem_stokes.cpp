#include "dfem/dfem_refactor.hpp"
#include "fem/pgridfunc.hpp"
#include "linalg/hypre.hpp"
#include "linalg/solvers.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <int dim = 2>
struct StokesMomentumQFunction
{
   StokesMomentumQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      // velocity gradient in reference space
      const tensor<real_t, dim, dim> &dudxi, // internal::dual<real_t, real_t>
      const real_t &p, // internal::dual<real_t, real_t>
      const tensor<real_t, dim, dim> &J,
      const double &w) const
   {
      constexpr real_t kinematic_viscosity = 1.0;
      auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto viscous_stress = -p * I + 2.0 * kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{-viscous_stress * JxW};
   }
};

template <int dim = 2, int sdim = 2>
struct StokesMassConservationQFunction
{
   StokesMassConservationQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      // velocity gradient in reference space
      const tensor<double, dim, dim> &dudxi,
      const tensor<double, sdim, dim> &J,
      const double &w) const
   {
      return mfem::tuple{tr(dudxi * inv(J)) * det(J) * w};
   }
};

class StokesOperator : public Operator
{
   static constexpr int Velocity = 0;
   static constexpr int Pressure = 1;
   static constexpr int Coordinates = 2;

   class StokesJacobianOperator : public Operator
   {
   public:
      StokesJacobianOperator(const StokesOperator *ns, const Vector &x) :
         Operator(ns->Height()),
         ns(ns),
         block_op(ns->block_offsets)
      {
         xtmp = x;
         BlockVector xb(xtmp.ReadWrite(), ns->block_offsets);

         ParGridFunction u(&ns->velocity_fes);
         ParGridFunction p(&ns->pressure_fes);

         u.SetFromTrueDofs(xb.GetBlock(0));
         p.SetFromTrueDofs(xb.GetBlock(1));

         auto mesh_nodes = static_cast<ParGridFunction*>
                           (ns->velocity_fes.GetParMesh()->GetNodes());
         momentum_du = ns->momentum->GetDerivative(Velocity, {&u}, {&p, mesh_nodes});

         // Get a HypreParMatrix
         //
         // HypreParMatrix A;
         // static_cast<DerivativeOperator *>(momentum_du.get())->Assemble(A);
         //
         // or directly
         //
         // HypreParMatrix A;
         // ns->momentum->GetDerivative(Velocity, {&u,}, {&p, mesh_nodes})->Assemble(A);

         dRdp = ns->mass_conservation;
         dRdpT = std::make_shared<TransposeOperator>(*dRdp);

         block_op.SetBlock(0, 0, momentum_du.get());
         block_op.SetBlock(0, 1, dRdpT.get());
         block_op.SetBlock(1, 0, dRdp.get());
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         BlockVector xb(const_cast<double*>(x.Read()), ns->block_offsets);
         // column elimination for essential dofs
         xtmp = x;

         BlockVector xtmpb(xtmp.ReadWrite(), ns->block_offsets);
         xtmpb.GetBlock(0).SetSubVector(ns->vel_ess_tdofs, 0.0);

         block_op.Mult(xtmpb, y);

         BlockVector yb(y.ReadWrite(), ns->block_offsets);
         for (int i = 0; i < ns->vel_ess_tdofs.Size(); i++)
         {
            yb.GetBlock(0)[ns->vel_ess_tdofs[i]] = xb.GetBlock(0)[ns->vel_ess_tdofs[i]];
         }
      }

      const StokesOperator *ns;
      std::shared_ptr<Operator> momentum_du;
      std::shared_ptr<Operator> convective_du;

      std::shared_ptr<Operator> dRdu;
      std::shared_ptr<Operator> dRdp;
      std::shared_ptr<TransposeOperator> dRdpT;
      BlockOperator block_op;

      mutable Vector xtmp;
   };

public:
   StokesOperator(ParFiniteElementSpace &velocity_fes,
                  ParFiniteElementSpace &pressure_fes,
                  Array<int> &offsets,
                  Array<int> &vel_ess_tdofs,
                  const IntegrationRule &velocity_ir,
                  const IntegrationRule &pressure_ir) :
      Operator(offsets.Last()),
      block_offsets(offsets),
      vel_ess_tdofs(vel_ess_tdofs),
      velocity_fes(velocity_fes),
      pressure_fes(pressure_fes),
      mass_conservation_form(&velocity_fes, &pressure_fes)
   {
      auto mesh = velocity_fes.GetParMesh();
      mesh_nodes = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Velocity, &velocity_fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Pressure, &pressure_fes},
            FieldDescriptor{Coordinates, &mesh_fes}
         };

         momentum =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         mfem::tuple inputs{Gradient<Velocity>{}, Value<Pressure>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple outputs{Gradient<Velocity>{}};

         auto stokes_momemtum_qf = StokesMomentumQFunction{};
         auto derivatives = std::integer_sequence<size_t, Velocity> {};
         momentum->AddDomainIntegrator(stokes_momemtum_qf, inputs, outputs, velocity_ir,
                                       derivatives);
      }

      // Standard MFEM integrator
      auto vdfi = new VectorDivergenceIntegrator;
      vdfi->SetIntegrationRule(pressure_ir);
      mass_conservation_form.AddDomainIntegrator(vdfi);
      mass_conservation_form.Assemble();
      mass_conservation_form.Finalize();
      mass_conservation.reset(mass_conservation_form.ParallelAssemble());

      // dFEM
      // {
      //    auto solutions = std::vector
      //    {
      //       FieldDescriptor{Pressure, &pressure_fes},
      //    };

      //    auto parameters = std::vector
      //    {
      //       FieldDescriptor{Velocity, &velocity_fes},
      //       FieldDescriptor{Coordinates, &mesh_fes}
      //    };

      //    mass_conservation = std::make_shared<DifferentiableOperator>(solutions,
      //                                                                 parameters,
      //                                                                 mesh);

      //    mfem::tuple inputs{Gradient<Velocity>{}, Gradient<Coordinates>{}, Weight{}};
      //    mfem::tuple outputs{Value<Pressure>{}};

      //    auto stokes_mass_conservation_qf = StokesMassConservationQFunction{};
      //    mass_conservation->AddDomainIntegrator(stokes_mass_conservation_qf, inputs,
      //                                           outputs,
      //                                           pressure_ir);
      // }
   }

   void Mult(const Vector &x, Vector &r) const override
   {
      Vector xu(const_cast<double *>(x.Read()) + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector xp(const_cast<double *>(x.Read()) + block_offsets[1],
                block_offsets[2] - block_offsets[1]);
      Vector ru(r.ReadWrite() + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector rp(r.ReadWrite() + block_offsets[1],
                block_offsets[2] - block_offsets[1]);

      ParGridFunction p(&pressure_fes);
      p.SetFromTrueDofs(xp);
      momentum->SetParameters({&p, mesh_nodes});

      momentum->Mult(xu, ru);
      mass_conservation->Mult(xu, rp);

      ru.SetSubVector(vel_ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      jacobian_operator = std::make_shared<StokesJacobianOperator>(this, x);
      return *jacobian_operator;

      // fd_jacobian = std::make_shared<FDJacobian>(*this, x);
      // return *fd_jacobian;
   }

   std::shared_ptr<DifferentiableOperator> momentum;
   std::shared_ptr<DifferentiableOperator> continuity;

   ParGridFunction *mesh_nodes;

   ParMixedBilinearForm mass_conservation_form;
   std::shared_ptr<Operator> mass_conservation;
   const Array<int> block_offsets;
   const Array<int> vel_ess_tdofs;

   ParFiniteElementSpace &velocity_fes;
   ParFiniteElementSpace &pressure_fes;

   mutable std::shared_ptr<StokesJacobianOperator> jacobian_operator;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "../data/inline-quad.mesh";
   int polynomial_order = 2;
   int ir_order = 2;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   out << "#el: " << mesh.GetNE() << "\n";

   H1_FECollection velocity_fec(polynomial_order, dim);
   ParFiniteElementSpace velocity_fes(&mesh, &velocity_fec, dim);

   H1_FECollection pressure_fec(polynomial_order - 1, dim);
   ParFiniteElementSpace pressure_fes(&mesh, &pressure_fec);

   out << velocity_fes.GetTrueVSize() << "\n";
   out << pressure_fes.GetTrueVSize() << "\n";

   const IntegrationRule &velocity_ir =
      IntRules.Get(velocity_fes.GetFE(0)->GetGeomType(),
                   ir_order * velocity_fec.GetOrder());

   const IntegrationRule &pressure_ir =
      IntRules.Get(pressure_fes.GetFE(0)->GetGeomType(),
                   ir_order * pressure_fec.GetOrder());

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   Array<int> vel_ess_tdofs;
   velocity_fes.GetEssentialTrueDofs(bdr_attr_is_ess, vel_ess_tdofs);

   ParGridFunction u(&velocity_fes);
   ParGridFunction p(&pressure_fes);

   auto u_f = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double y = coords(1);
      if (y >= 1.0)
      {
         u(0) = 1.0;
      }
      else
      {
         u(0) = 0.0;
      }
      u(1) = 0.0;
   };
   auto u_coef = VectorFunctionCoefficient(dim, u_f);

   u.ProjectCoefficient(u_coef);
   p = 0.0;

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = velocity_fes.GetTrueVSize();
   block_offsets[2] = pressure_fes.GetTrueVSize();
   block_offsets.PartialSum();

   StokesOperator stokes(velocity_fes, pressure_fes, block_offsets,
                         vel_ess_tdofs, velocity_ir, pressure_ir);

   BlockVector x(block_offsets), y(block_offsets);
   u.ParallelProject(x.GetBlock(0));
   x.GetBlock(1) = 0.0;

   GMRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-8);
   // solver.SetKDim(100);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(2);
   // solver.SetPreconditioner(prec);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(stokes);
   newton.SetSolver(solver);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(50);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   ParaViewDataCollection dc("dfem_stokes", &mesh);
   dc.SetHighOrderOutput(true);
   dc.RegisterField("velocity", &u);
   dc.RegisterField("pressure", &p);
   dc.Save();

   return 0;
}
