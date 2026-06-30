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

#include "../unit_tests.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif


constexpr int dim = 3;
constexpr int Displacement = 0;
constexpr int Coords = 1;
constexpr int Energy = 2;

template <typename dscalar_t>
struct NeoHookeanEnergy
{
   real_t D1 = 100.0;
   real_t C1 = 50.0;

   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &energy) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;
      const auto C = transpose(F) * F;
      const auto JF = det(F);
      const auto I1_bar = pow(JF, -2.0_r / 3.0_r) * tr(C);
      const auto psi = D1 * (JF - 1.0_r) * (JF - 1.0_r)
                       + C1 * (I1_bar - real_t(dim));
      energy = psi * det(J) * w;
   }
};

template <typename dscalar_t>
struct NeoHookeanStress
{
   real_t D1 = 100.0;
   real_t C1 = 50.0;

   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   tensor<dscalar_t, dim, dim> &dvdxi) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;
      const auto JF = det(F);
      const auto FinvT = transpose(inv(F));
      const auto I1 = tr(transpose(F) * F);
      const auto P = 2.0_r * D1 * JF * (JF - 1.0_r) * FinvT
                     + 2.0_r * C1 * pow(JF, -2.0_r / 3.0_r)
                     * (F - (I1 / 3.0_r) * FinvT);
      dvdxi = P * transpose(invJ) * det(J) * w;
   }
};

class HyperelasticityProblem
{
public:
   HyperelasticityProblem(ParFiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          bool use_energy) :
      fes(fes),
      use_energy(use_energy),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      q(qspace_vec)
   {
      auto &mesh_nodes = *static_cast<ParGridFunction *>
                         (fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Displacement, &fes},
         {Coords, mesh_nodes_fes}
      };
      std::vector<FieldDescriptor> outputs;
      if (use_energy)
      {
         outputs = std::vector<FieldDescriptor>
         {
            {Energy, &qspace_vec}
         };
      }
      else
      {
         outputs = std::vector<FieldDescriptor>
         {
            {Displacement, &fes}
         };
      }

      dop = std::make_shared<DifferentiableOperator>(inputs, outputs,
                                                     *fes.GetParMesh());

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      auto derivatives = std::integer_sequence<size_t, Displacement> {};
      if (use_energy)
      {
         NeoHookeanEnergy<dscalar_t> energy;
         dop->AddDomainIntegrator<LocalQFBackend, true>(
            energy,
            Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
            Outputs<Identity<Energy>> {},
            ir, all_domain_attr, derivatives);
      }
      else
      {
         NeoHookeanStress<dscalar_t> stress;
         dop->AddDomainIntegrator<LocalQFBackend>(
            stress,
            Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
            Outputs<Gradient<Displacement>> {},
            ir, all_domain_attr, derivatives);
      }
   }

   void SetEssentialAttributes(const Array<int> &ess_bdr)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   void SetPrescribedDisplacementAttributes(const Array<int> &disp_bdr)
   {
      fes.GetEssentialTrueDofs(disp_bdr, prescribed_tdofs);
   }

   const Array<int>& GetPrescribedDisplacementTDofs() const
   {
      return prescribed_tdofs;
   }

   const Array<int>& GetEssentialTDofs() const
   {
      return ess_tdofs;
   }

   void Residual(const Vector &x, Vector &r) const
   {
      MultiVector X{x, mesh_nodes_tdofs};
      MultiVector R{r};
      if (use_energy)
      {
         dop->GetDerivative(Displacement)->Mult(X, R);
      }
      else
      {
         dop->Mult(X, R);
      }
      r.SetSubVector(ess_tdofs, 0.0);
   }

   void GradientAction(const Vector &x, const Vector &dx, Vector &y) const
   {
      MultiVector X{x, mesh_nodes_tdofs};
      std::shared_ptr<DerivativeOperator> derivative;
      if (use_energy)
      {
         derivative = dop->GetSecondDerivative(Displacement, X);
      }
      else
      {
         derivative = dop->GetDerivative(Displacement, X);
      }

      Vector local_dx(dx);
      local_dx.SetSubVector(ess_tdofs, 0.0);

      MultiVector Y{y};
      derivative->Mult(local_dx, Y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

private:
   ParFiniteElementSpace &fes;
   bool use_energy = false;
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   QuadratureFunction q;
   Vector mesh_nodes_tdofs;
   Array<int> ess_tdofs;
   Array<int> prescribed_tdofs;
   std::shared_ptr<DifferentiableOperator> dop;
};

struct HyperelasticityTestContext
{
   HyperelasticityTestContext(bool use_energy)
   {
      Mesh mesh = Mesh::MakeCartesian3D(8, 2, 2, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
      mesh.EnsureNodes();
      pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
      mesh.Clear();
      pmesh->EnsureNodes();

      fec = std::make_unique<H1_FECollection>(1, dim);
      fes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get(), dim,
                                                    Ordering::byNODES);
      const IntegrationRule &ir = IntRules.Get(pmesh->GetTypicalElementGeometry(), 3);
      problem = std::make_unique<HyperelasticityProblem>(*fes, ir, use_energy);

      Array<int> ess_attr(pmesh->bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[4] = 1;
      ess_attr[2] = 1;
      problem->SetEssentialAttributes(ess_attr);

      Array<int> disp_attr(pmesh->bdr_attributes.Max());
      disp_attr = 0;
      disp_attr[2] = 1;
      problem->SetPrescribedDisplacementAttributes(disp_attr);

      state.SetSize(fes->GetTrueVSize());
      state.Randomize(11);
      state -= 0.5;
      state *= 1.0e-3;
      state.SetSubVector(problem->GetEssentialTDofs(), 0.0);
      state.SetSubVector(problem->GetPrescribedDisplacementTDofs(), 1.0e-2);

      direction.SetSize(fes->GetTrueVSize());
      direction.Randomize(17);
      direction -= 0.5;
   }

   std::unique_ptr<ParMesh> pmesh;
   std::unique_ptr<H1_FECollection> fec;
   std::unique_ptr<ParFiniteElementSpace> fes;
   std::unique_ptr<HyperelasticityProblem> problem;
   Vector state;
   Vector direction;
};


TEST_CASE("dfem neo-hookean energy and stress agree",
          "[Parallel][dFEM][Hyperelasticity]")
{
   HyperelasticityTestContext energy(true);
   HyperelasticityTestContext stress(false);

   REQUIRE(energy.state.Size() == stress.state.Size());

   // Check residuals from energy and stress formulations should match.
   Vector energy_residual(energy.state.Size());
   Vector stress_residual(stress.state.Size());
   energy.problem->Residual(energy.state, energy_residual);
   stress.problem->Residual(stress.state, stress_residual);

   Vector residual_diff(energy_residual);
   residual_diff -= stress_residual;
   REQUIRE(residual_diff.Norml2() < 1e-12);

   // Check the energy Hessian action should match the stress Jacobian action.
   Vector energy_action(energy.state.Size());
   Vector stress_action(stress.state.Size());
   energy.problem->GradientAction(energy.state, energy.direction, energy_action);
   stress.problem->GradientAction(stress.state, stress.direction, stress_action);

   REQUIRE(energy_action.Norml2() > 0.0);
   REQUIRE(stress_action.Norml2() > 0.0);

   Vector action_diff(energy_action);
   action_diff -= stress_action;
   REQUIRE(action_diff.Norml2() < 1e-10);

   // Print  the residuals, grad actions, for debugging purposes.
   mfem::out << "Energy residual norm: " << energy_residual.Norml2() << std::endl;
   mfem::out << "Stress residual norm: " << stress_residual.Norml2() << std::endl;
   mfem::out << "Residual difference norm: " << residual_diff.Norml2() << std::endl;
   mfem::out << "Energy action norm: " << energy_action.Norml2() << std::endl;
   mfem::out << "Stress action norm: " << stress_action.Norml2() << std::endl;
   mfem::out << "Action difference norm: " << action_diff.Norml2() << std::endl;
}

#endif