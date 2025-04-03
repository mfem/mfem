// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <mfem.hpp>

// TODO: Do we want this to be included from mfem.hpp automatically now?
#include <fem/dfem/doperator.hpp>
#include <linalg/tensor.hpp>

#include <fstream>

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

template <typename T, int dim>
MFEM_HOST_DEVICE inline
tensor<T, 3, 3> tensor_to_3D(const tensor<T, dim, dim>& A)
{
   tensor<T, 3, 3> A3D{};
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
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
      //auto internal_state_new = get<1>(material(dudX3D, internal_state));
      auto [stress, internal_state_new] = material(dudX3D, internal_state);
      // real_t vm = sqrt(1.5)*norm(dev(stress));
      // out << vm << " " << internal_state_new[9] << std::endl;
      return mfem::tuple{internal_state_new};
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
      auto [P3D, Qnew] = material(dudX3D, internal_state);
      auto P = mfem::internal::make_tensor<dim, dim>([&P3D](int i, int j) { return P3D[i][j]; });
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{P * JxW};
   }

   Material material;
};


struct J2SmallStrain
{
   static constexpr int dim = 3;         ///< spatial dimension
   static constexpr int n_internal_states = 10;
   static constexpr double tol =
      1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

   real_t E;                 ///< Young's modulus
   real_t nu;                ///< Poisson's ratio
   real_t sigma_y;           ///< Yield strength
   real_t Hi;                ///< Isotropic hardening modulus
   real_t density;           ///< Mass density

   /// @brief variables required to characterize the hysteresis response
   struct InternalState
   {
      tensor<double, dim, dim> plastic_strain;  ///< plastic strain
      double accumulated_plastic_strain;        ///< uniaxial equivalent plastic strain
   };

   MFEM_HOST_DEVICE inline
   InternalState unpack_internal_state(const tensor<real_t, n_internal_states> &
                                       packed_state) const
   {
      // we could use type punning here to avoid copies
      auto plastic_strain = mfem::internal::make_tensor<dim, dim>(
      [&packed_state](int i, int j) { return packed_state[dim*i + j]; });
      real_t accumulated_plastic_strain = packed_state[n_internal_states - 1];
      return {plastic_strain, accumulated_plastic_strain};
   }

   MFEM_HOST_DEVICE inline
   tensor<real_t, n_internal_states> pack_internal_state(const
                                                         tensor<real_t, dim, dim> & plastic_strain,
                                                         real_t accumulated_plastic_strain) const
   {
      tensor<real_t, n_internal_states> packed_state{};
      for (int i = 0, ij = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++, ij++)
         {
            packed_state[ij] = plastic_strain[i][j];
         }
      }
      packed_state[n_internal_states - 1] = accumulated_plastic_strain;
      return packed_state;
   }

   MFEM_HOST_DEVICE inline
   tuple<tensor<real_t, dim, dim>, tensor<real_t, n_internal_states>>
                                                                   operator()(const tensor<real_t, dim, dim> & dudX,
                                                                              const tensor<real_t, n_internal_states> & internal_state) const
   {
      auto I = mfem::internal::Identity<dim>();
      const real_t K = E / (3.0 * (1.0 - 2.0 * nu));
      const real_t G = 0.5 * E / (1.0 + nu);

      auto [plastic_strain, accumulated_plastic_strain] = unpack_internal_state(
                                                             internal_state);

      // (i) elastic predictor
      auto el_strain = sym(dudX) - plastic_strain;
      auto p = K * tr(el_strain);
      auto s = 2.0 * G * dev(el_strain);
      auto q = sqrt(1.5) * norm(s);
      [[maybe_unused]] real_t delta_eqps = 0.0;

      [[maybe_unused]] auto flow_strength = [this](real_t eqps) { return this->sigma_y + this->Hi*eqps; };

      // (ii) admissibility
      if (q - (sigma_y + Hi*accumulated_plastic_strain) > tol*sigma_y)
      {
         // (iii) return mapping
         real_t delta_eqps = (q - sigma_y - Hi*accumulated_plastic_strain)/(3*G + Hi);
         auto Np = 1.5 * s / q;
         s -= 2.0 * G * delta_eqps * Np;
         plastic_strain += delta_eqps * Np;
         accumulated_plastic_strain += delta_eqps;
      }
      auto stress = s + p * I;
      auto internal_state_new = pack_internal_state(plastic_strain,
                                                    accumulated_plastic_strain);
      return {stress, internal_state_new};
   }
};

class ElasticityOperator : public Operator
{
   static constexpr int Displacement = 0;
   static constexpr int Coordinates = 1;
   static constexpr int InternalState = 2;

public:
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
      body_force(displacement_fes.GetTrueVSize()),
      displacement_ess_tdof(vel_ess_tdofs),
      displacement_fes(displacement_fes),
      displacement_ir(displacement_ir),
      internal_state(internal_state)
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
         momentum->DisableTensorProductStructure();

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


class InternalStateUpdater : public Operator
{
public:

   static constexpr int Displacement = 0;
   static constexpr int Coordinates = 1;
   static constexpr int InternalState = 2;

   template <typename Material>
   InternalStateUpdater(ParFiniteElementSpace &displacement_fes,
                        const IntegrationRule &displacement_ir,
                        ParametricFunction &internal_state,
                        Material material) :
      Operator(displacement_fes.GetTrueVSize()),
      displacement_fes(displacement_fes),
      displacement_ir(displacement_ir),
      internal_state(internal_state)
   {
      auto mesh = displacement_fes.GetParMesh();
      mesh_nodes = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

      auto solutions = std::vector
      {
         FieldDescriptor{Displacement, &displacement_fes}
      };

      auto parameters = std::vector
      {
         FieldDescriptor{Coordinates, &mesh_fes},
         FieldDescriptor{InternalState, &internal_state.space}
      };

      op = std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);
      op->DisableTensorProductStructure();

      mfem::tuple inputs{Gradient<Displacement>{}, Gradient<Coordinates>{}, None<InternalState>{}, Weight{}};
      mfem::tuple outputs{None<InternalState>{}};

      auto qfunction = InternalStateQFunction<Material, DIMENSION> {.material = material};
      // just a placeholder for now. We want vjps wrt both displacement and old internal state eventually
      auto derivatives = std::integer_sequence<size_t, Displacement> {};
      Array<int> solid_domain_attr(mesh->attributes.Max());
      solid_domain_attr[0] = 1;
      op->AddDomainIntegrator(
         qfunction, inputs, outputs, displacement_ir, solid_domain_attr, derivatives);
   }

   void Mult(const Vector &displacement, Vector& internal_state_new) const override
   {
      op->SetParameters({mesh_nodes, &internal_state});
      op->Mult(displacement, internal_state_new);
   }

   void VjpDisplacement(ParGridFunction &u, Vector& internal_state_old,
                        Vector& internal_state_new_bar, Vector& displacement_bar) const
   {
      // u, internal_state_old, internal_state_new_bar should be const
      out << "Sizes " << "u " << u.Size() << ", qold " << internal_state_old.Size() <<
          ", qbar " << internal_state_new_bar.Size() << ", ubar " <<
          displacement_bar.Size() << std::endl;
      auto grad_op = op->GetDerivative(Displacement, {&u}, {mesh_nodes, &internal_state_old});
      out << "grad_op " << grad_op->Height() << " " << grad_op->Width() << std::endl;
      out << "grad_op^T " << grad_op->Width() << " " << grad_op->Height() <<
          std::endl;
      grad_op->MultTranspose(internal_state_new_bar, displacement_bar);
   }

   ParGridFunction *mesh_nodes;
   ParFiniteElementSpace &displacement_fes;
   std::shared_ptr<DifferentiableOperator> op;
   IntegrationRule displacement_ir;
   ParametricFunction& internal_state;
};


int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;
   int nonlinear_solver_type = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--refinements", "");
   args.AddOption(&ir_order, "-iro", "--integration-rule-order", "");
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

   Mesh mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
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
   ParametricSpace internal_state_space(dim, n_internal_state_variables,
                                        displacement_ir.GetNPoints(),
                                        n_internal_state_variables*displacement_ir.GetNPoints()*mesh_beam.GetNE());

   ParametricFunction internal_state(internal_state_space);
   internal_state = 0.0;
   ParametricFunction internal_state_old(internal_state_space);
   internal_state_old = 0.0;

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

   ParGridFunction u(&displacement_fes);
   u = 0.0;

   using Material = J2SmallStrain; // StVenantKirchhoff
   Material material{.E = 1000.0, .nu = 0.25, .sigma_y = 0.53333, .Hi = 40.0, .density = 1.0};
   // Material material{.mu = 0.5e6, .nu = 0.4};

   ElasticityOperator elasticity(displacement_fes, displacement_ess_tdof,
                                 displacement_ir, internal_state, material);

   CGSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-10);
   solver.SetMaxIter(1000);
   solver.SetPrintLevel(2);

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
   nonlinear_solver->SetRelTol(1e-9);
   nonlinear_solver->SetMaxIter(25);
   nonlinear_solver->SetSolver(solver);
   nonlinear_solver->SetPrintLevel(1);

   // variables for output
   QuadratureSpace output_internal_state_space(mesh_beam, displacement_ir);
   QuadratureFunction output_internal_state(&output_internal_state_space,
                                            internal_state.GetData(), material.n_internal_states);
   Vector r(displacement_fes.GetTrueVSize());
   ParGridFunction reaction(&displacement_fes);
   Vector end_forces_x(bc_tdof.Size());

   ParaViewDataCollection dc("dfem_plasticity", &mesh_beam);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(1);
   dc.RegisterField("displacement", &u);
   dc.RegisterField("reaction", &reaction);
   dc.RegisterQField("internal_state", &output_internal_state);
   dc.SetCycle(0);
   dc.Save();

   InternalStateUpdater internal_state_update(displacement_fes, displacement_ir,
                                              internal_state, material);
   //Vector q(internal_state_space.GetTotalSize());

   auto applied_displacement = [](double t) { return 1.2e-2*t; };

   real_t time = 0.0;
   std::ofstream history_file("history_output.csv");
   history_file << applied_displacement(time) << " " << 0.0 << std::endl;

   Vector zero, x(displacement_fes.GetTrueVSize());

   constexpr int max_cycles = 3;
   const real_t dt = 1.0/(max_cycles - 1);
   for (int cycle = 1; cycle < max_cycles; cycle++)
   {
      time += dt;
      out << "-------------------------------------------" << std::endl;
      out << "TIME STEP " << cycle << std::endl;
      out << "t = " << time << std::endl;

      real_t ubc = applied_displacement(time);
      u.SetSubVector(bc_tdof, ubc);

      u.GetTrueDofs(x);
      nonlinear_solver->Mult(zero, x);
      u.SetFromTrueDofs(x);

      // update internal variables
      internal_state_old.Set(1.0, internal_state);
      internal_state_update.Mult(u, internal_state);

      // Compute reactions
      elasticity.Reaction(x, r);
      reaction.SetFromTrueDofs(r);
      reaction.GetSubVector(bc_tdof, end_forces_x);
      real_t force = -end_forces_x.Sum();
      out << "u = " << applied_displacement(time) << ", Force = " << force <<
          std::endl;
      history_file << applied_displacement(time) << " " << force << std::endl;

      output_internal_state = internal_state;

      dc.SetCycle(cycle);
      dc.SetTime(time);
      dc.Save();
   }

   // try to use the derivative to see if it works
   ParametricFunction internal_state_bar(internal_state_space);
   internal_state_bar = 1.0;
   //ParGridFunction u_bar(displacement_fes);
   Vector u_bar(displacement_fes.GetTrueVSize());
   internal_state_update.VjpDisplacement(u, internal_state_old, internal_state_bar,
                                         u_bar);

   pretty_print(u_bar);

   history_file.close();
   return 0;
}
