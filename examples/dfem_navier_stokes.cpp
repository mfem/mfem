#include "dfem/dfem_refactor.hpp"
#include "fem/bilininteg.hpp"
#include "fem/intrules.hpp"
#include "fem/pbilinearform.hpp"
#include "linalg/hypre.hpp"
#include "linalg/operator.hpp"
#include "linalg/tensor.hpp"

using namespace mfem;
using mfem::internal::tensor;

// -\nabla \cdot (\nabla u + p * I) -> (\nabla u + p * I, \nabla v)
template <int dim = 2>
class MomentumQFunction
{
public:
   MomentumQFunction(const double &kinematic_viscosity,
                     const bool &formulation) :
      kinematic_viscosity(kinematic_viscosity),
      formulation(formulation) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<double, dim> &u,
                   const tensor<double, dim, dim> &dudxi,
                   const double &p,
                   const tensor<double, dim, dim> &J,
                   const double &w) const
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto viscous_stress = -p * I + 2.0 * kinematic_viscosity * sym(dudx);
      auto JxW = det(J) * w * transpose(invJ);
      if (formulation == 0)
      {
         return mfem::tuple{(-outer(u, u) + viscous_stress) * JxW};
      }
      else
      {
         return mfem::tuple{viscous_stress * JxW};
      }
   }

   // TODO: this might not be ok on GPU
   const double kinematic_viscosity;
   const bool formulation;
};

template <int dim = 2>
class ContinuityQFunction
{
public:
   ContinuityQFunction(const int &formulation) :
      formulation(formulation) {}

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<double, dim> &u,
                   const tensor<double, dim, dim> &dudxi,
                   const tensor<double, dim, dim> &J,
                   const double &w) const
   {
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto JxW = det(J) * w;
      auto convective = dot(dudx, u);
      if (formulation == 1)
      {
         return mfem::tuple{convective * JxW};
      }
      else if (formulation == 2)
      {
         return mfem::tuple{(convective + 0.5 * tr(dudx) * u) * JxW};
      }
      else if (formulation == 3)
      {
         // ONLY VALID FOR dim == 2
         real_t curl_u = dudx(1, 0) - dudx(0, 1);
         // cross product u x curl(u)
         tensor<real_t, dim> u_cross_curl_u;
         u_cross_curl_u(0) = u(1) * curl_u;
         u_cross_curl_u(1) = -u(0) * curl_u;
         return mfem::tuple{-u_cross_curl_u * JxW};
      }
      else if (formulation == 4)
      {
         return mfem::tuple{(2.0 * sym(dudx) * u + 0.5 * tr(dudx) * u) * JxW};
      }
   }

   const int formulation;
};

class NavierStokesOperator : public Operator
{
   static constexpr int Velocity = 0;
   static constexpr int Pressure = 1;
   static constexpr int Coordinates = 2;

   class NavierStokesJacobianOperator : public Operator
   {
   public:
      NavierStokesJacobianOperator(
         const NavierStokesOperator *ns,
         const Vector &x) :
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

         momentum_du = ns->momentum->GetDerivative(Velocity, {&u, &p}, {mesh_nodes});
         dRdp = ns->mass_conservation;
         dRdpT = std::make_shared<TransposeOperator>(*dRdp);

         if (ns->formulation == 0)
         {
            block_op.SetBlock(0, 0, momentum_du.get());
         }
         else
         {
            convective_du = ns->continuity->GetDerivative(Velocity, {&u, &p}, {mesh_nodes});
            dRdu = std::make_shared<SumOperator>(momentum_du.get(), 1.0,
                                                 convective_du.get(), 1.0, false, false);
            block_op.SetBlock(0, 0, dRdu.get());
         }

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

      const NavierStokesOperator *ns;
      std::shared_ptr<Operator> momentum_du;
      std::shared_ptr<Operator> convective_du;

      std::shared_ptr<Operator> dRdu;
      std::shared_ptr<Operator> dRdp;
      std::shared_ptr<TransposeOperator> dRdpT;
      BlockOperator block_op;

      mutable Vector xtmp;
   };

public:
   NavierStokesOperator(ParFiniteElementSpace &velocity_fes,
                        ParFiniteElementSpace &pressure_fes,
                        Array<int> &offsets,
                        Array<int> &vel_ess_tdofs,
                        const double &kinematic_viscosity,
                        const IntegrationRule &velocity_ir,
                        const IntegrationRule &pressure_ir,
                        const int &formulation) :
      Operator(offsets.Last()),
      block_offsets(offsets),
      vel_ess_tdofs(vel_ess_tdofs),
      velocity_fes(velocity_fes),
      pressure_fes(pressure_fes),
      mass_conservation_form(&velocity_fes, &pressure_fes),
      formulation(formulation)
   {
      auto mesh = velocity_fes.GetParMesh();
      auto mesh_nodes = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      auto solutions = std::vector
      {
         FieldDescriptor{Velocity, &velocity_fes},
         FieldDescriptor{Pressure, &pressure_fes}
      };

      auto parameters = std::vector
      {
         FieldDescriptor{Coordinates, &mesh_fes}
      };

      {
         mfem::tuple input_operators{Value<Velocity>{}, Gradient<Velocity>{}, Value<Pressure>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple output_operators{Gradient<Velocity>{}};

         momentum =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         auto derivatives = std::integer_sequence<size_t, Velocity> {};

         MomentumQFunction<2> momentum_qf(kinematic_viscosity, formulation);
         momentum->AddDomainIntegrator(
            momentum_qf, input_operators, output_operators, velocity_ir,
            derivatives);

         momentum->SetParameters({mesh_nodes});
      }

      auto vdfi = new VectorDivergenceIntegrator;
      vdfi->SetIntegrationRule(pressure_ir);
      mass_conservation_form.AddDomainIntegrator(vdfi);
      mass_conservation_form.Assemble();
      mass_conservation_form.Finalize();
      mass_conservation.reset(mass_conservation_form.ParallelAssemble());

      {
         mfem::tuple input_operators{Value<Velocity>{}, Gradient<Velocity>{}, Gradient<Coordinates>{}, Weight{}};
         mfem::tuple output_operators{Value<Velocity>{}};

         continuity =
            std::make_shared<DifferentiableOperator>(
               std::vector{FieldDescriptor{Velocity, &velocity_fes}}, parameters, *mesh);

         auto derivatives = std::integer_sequence<size_t, Velocity> {};

         ContinuityQFunction<2> convective_qf(formulation);
         continuity->AddDomainIntegrator(
            convective_qf, input_operators, output_operators, velocity_ir,
            derivatives);

         continuity->SetParameters({mesh_nodes});
      }
   }

   void Mult(const Vector &x, Vector &r) const override
   {
      Vector xu(const_cast<double *>(x.Read()) + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector ru(r.ReadWrite() + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector rp(r.ReadWrite() + block_offsets[1],
                block_offsets[2] - block_offsets[1]);

      momentum->Mult(x, ru);

      if (formulation != 0)
      {
         continuity->AddMult(xu, ru);
      }

      mass_conservation->Mult(xu, rp);

      ru.SetSubVector(vel_ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      // jacobian_operator = std::make_shared<NavierStokesJacobianOperator>(this, x);
      // return *jacobian_operator;

      fd_jacobian = std::make_shared<FDJacobian>(*this, x);
      return *fd_jacobian;
   }

   std::shared_ptr<DifferentiableOperator> momentum;
   std::shared_ptr<DifferentiableOperator> continuity;

   ParMixedBilinearForm mass_conservation_form;
   std::shared_ptr<Operator> mass_conservation;
   const Array<int> block_offsets;
   const Array<int> vel_ess_tdofs;

   ParFiniteElementSpace &velocity_fes;
   ParFiniteElementSpace &pressure_fes;

   mutable std::shared_ptr<NavierStokesJacobianOperator> jacobian_operator;
   mutable std::shared_ptr<FDJacobian> fd_jacobian;

   const bool formulation;
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
   double kinematic_viscosity = 1.0;
   int formulation = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&kinematic_viscosity, "-kv", "--kv", "");
   args.AddOption(&formulation, "-f", "--f",
                  "Formulation:"
                  "0 - conservative form"
                  "1 - convective form"
                  "2 - convective skew-symmetric form");
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

   NavierStokesOperator navierstokes(velocity_fes, pressure_fes, block_offsets,
                                     vel_ess_tdofs, kinematic_viscosity, velocity_ir, pressure_ir,
                                     formulation);

   BlockVector x(block_offsets), y(block_offsets);
   u.ParallelProject(x.GetBlock(0));

   GMRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-4);
   solver.SetKDim(100);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(2);
   // solver.SetPreconditioner(prec);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(navierstokes);
   newton.SetSolver(solver);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(50);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   ParaViewDataCollection dc("dfem_navier_stokes", &mesh);
   dc.SetHighOrderOutput(true);
   dc.RegisterField("velocity", &u);
   dc.RegisterField("pressure", &p);
   dc.Save();

   return 0;
}
