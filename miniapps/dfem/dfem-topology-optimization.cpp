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
//
// --------------------------------------------------------
// Hyperelastic Topology Optimization with dFEM (ex37-style)
// --------------------------------------------------------
//
// Compile with: make dfem-hyperelasticity_topopt
//   (assumes ../../examples/ex37.hpp is reachable, i.e. a normal MFEM
//    source tree with miniapps/dfem/ and examples/ as siblings)
//
// Description: This is MFEM Example 37 (density-filtered topology
//   optimization via entropic mirror descent [2]) with the state problem
//   and sensitivity replaced to handle hyperelastic materials via dFEM:
//
//     * State: linear elasticity (one linear solve)
//       -->     nonlinear hyperelasticity (Newton, using dFEM first/second
//                derivatives of a strain-energy functional -- see
//                dfem-hyperelasticity_energy.cpp).
//
//     * Sensitivity: closed-form pointwise coefficient
//       (StrainEnergyDensityCoefficient), valid because linear-elasticity
//       compliance is self-adjoint (dual = -primal, see ex37's own
//       preamble below)
//       --> two options, following Klarbring & Strömberg [4]:
//             c1, potential energy (adjoint free) -- obtained directly via dFEM's
//               GetDerivative(Density).
//             c2, compliance: no longer self-adjoint once the material is
//               nonlinear -- needs one explicit adjoint solve, reusing
//               the converged Newton tangent (which *is* symmetric).
//
//   Filtering, the sigmoid/latent-variable bound constraint, and the
//   mirror-descent update are otherwise unmodified from ex37.
//
// [1] Andreassen et al. (2011), 88 lines of code, Struct Multidisc Optim 43.
// [2] Keith, B. and Surowiec, T. (2023), Proximal Galerkin, arXiv:2307.12444.
// [3] Lazarov, B. S., & Sigmund, O. (2011), Helmholtz filters, IJNME 86(6).
// [4] Klarbring, A., Strömberg, N. (2013), Struct Multidisc Optim 47:37-48.
//
// Sample runs for comparable material optimizations:
//
// To increase the number of iterations, modify the -mi option (default 35)
//
// Linear elasticity
// mpirun -np 4 /dfem-topology-optimization -mat linear-elastic -obj c1 -p1 0.1 -p2 0.1 -r 4 -mi 50 -vf 0.5 -epsilon 0.01 -alpha 1.0 -growth 1.5 -ntol 1e-4 -itol 1e-2 -of /home/shared/Output/dfem-TopOp/TestMaterials/LinearElastic --paraview
//
// NeoHookean
// mpirun -np 4 ./dfem-topology-optimization -mat neo-hookean -obj c1 -p1 1 -p2 1 -r 4 -vf 0.5 -epsilon 0.01 -alpha 1.0 -growth 1.5 -ntol 1e-4 -itol 1e-2 -of /home/shared/Output/dfem-TopOp/TestMaterials/NeoHookean/ --paraview
//
// Mooney-Rivlin
// mpirun -np 4 ./dfem-topology-optimization -mat mooney-rivlin -obj c1 -p1 2 -p2 2 -r 4 -vf 0.5 -epsilon 0.01 -alpha 1.0 -growth 1.5 -ntol 1e-4 -itol 1e-2 -of /home/shared/Output/dfem-TopOp/TestMaterials/MooneyRivlin/ --parav
//

#include "mfem.hpp"
#include "fem/dfem/doperator.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"
#include <cstdio>
#include <functional>

namespace mfem
{

real_t inv_sigmoid(real_t x)
{
   real_t tol = 1e-12;
   x = std::min(std::max(tol,x), real_t(1.0)-tol);
   return std::log(x/(1.0-x));
}

real_t sigmoid(real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      return std::exp(x)/(1.0+std::exp(x));
   }
}

MFEM_HOST_DEVICE inline real_t SimpPower(real_t rho, real_t exponent)
{
   if (exponent == 1.0) { return rho; }
   if (exponent == 2.0) { return rho * rho; }
   if (exponent == 3.0) { return rho * rho * rho; }
   if (exponent == 4.0) { const real_t rho2 = rho * rho; return rho2 * rho2; }
   return pow(rho, exponent);
}

void ClampDensity(Vector &rho, const char *name)
{
   for (int i = 0; i < rho.Size(); i++)
   {
      MFEM_VERIFY(IsFinite(rho(i)), name << " contains non-finite value at index "
                  << i << ": " << rho(i));
      rho(i) = std::min(std::max(rho(i), real_t(0.0)), real_t(1.0));
   }
}

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t)> fun;

public:
   MappedGridFunctionCoefficient()
      : GridFunctionCoefficient(), fun([](real_t x) { return x; }) { }

   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<real_t(const real_t)> fun_,
                                 int comp=1)
      : GridFunctionCoefficient(gf, comp), fun(fun_) { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
};

class DiffMappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   const GridFunction *OtherGridF;
   GridFunctionCoefficient OtherGridF_cf;
   std::function<real_t(const real_t)> fun;

public:
   DiffMappedGridFunctionCoefficient()
      : GridFunctionCoefficient(), OtherGridF(nullptr), OtherGridF_cf(),
        fun([](real_t x) { return x; }) { }

   DiffMappedGridFunctionCoefficient(const GridFunction *gf,
                                     const GridFunction *other_gf,
                                     std::function<real_t(const real_t)> fun_,
                                     int comp=1)
      : GridFunctionCoefficient(gf, comp), OtherGridF(other_gf),
        OtherGridF_cf(OtherGridF), fun(fun_) { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const real_t value1 = fun(GridFunctionCoefficient::Eval(T, ip));
      const real_t value2 = fun(OtherGridF_cf.Eval(T, ip));
      return value1 - value2;
   }
};

class VolumeForceCoefficient : public VectorCoefficient
{
public:
   VolumeForceCoefficient(real_t r_, Vector &center_, Vector &force_)
      : VectorCoefficient(center_.Size()), r(r_), center(center_), force(force_) { }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x;
      x.SetSize(T.GetDimension());
      T.Transform(ip, x);
      x -= center;

      V.SetSize(T.GetDimension());
      if (x.Norml2() <= r)
      {
         V = force;
      }
      else
      {
         V = 0.0;
      }
   }

private:
   real_t r;
   Vector center;
   Vector force;
};

real_t proj(GridFunction &psi, GridFunction &alpha_grad, real_t target_volume,
            real_t tol = 1e-12, int max_its = 100)
{
#ifdef MFEM_USE_MPI
   FiniteElementSpace *fes = psi.FESpace();
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
#endif
   ConstantCoefficient zero_cf(0.0);
   real_t a = -alpha_grad.ComputeMaxError(zero_cf);
   real_t b = -a;
   real_t y = 0.0;

   MappedGridFunctionCoefficient sigmoid_psi(
   &psi, [&y](const real_t x) { return sigmoid(x + y); });
   std::unique_ptr<LinearForm> int_sigmoid_psi;
#ifdef MFEM_USE_MPI
   ParGridFunction *par_psi = dynamic_cast<ParGridFunction *>(&psi);
   if (par_psi)
   {
      int_sigmoid_psi.reset(new ParLinearForm(par_psi->ParFESpace()));
   }
   else
   {
      int_sigmoid_psi.reset(new LinearForm(psi.FESpace()));
   }
#else
   int_sigmoid_psi.reset(new LinearForm(psi.FESpace()));
#endif
   int_sigmoid_psi->AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));

   y = a;
   int_sigmoid_psi->Assemble();
   real_t f_a = int_sigmoid_psi->Sum();

   y = b;
   int_sigmoid_psi->Assemble();
   real_t f_b = int_sigmoid_psi->Sum();
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &f_a, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &f_b, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
   }
#endif
   f_a -= target_volume;
   f_b -= target_volume;
   real_t c = 0.0;
   real_t f_c = 0.0;
   int side = 0;

   bool done = false;
   for (int k=0; k < max_its; k++)
   {
      c = (f_a * b - f_b * a) / (f_a - f_b);

      if (abs(b - a) < tol * abs(b + a)) { done = true; break; }

      y = c;
      int_sigmoid_psi->Assemble();
      f_c = int_sigmoid_psi->Sum();
#ifdef MFEM_USE_MPI
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &f_c, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, MPI_COMM_WORLD);
      }
#endif
      f_c -= target_volume;

      if (f_c * f_b > 0)
      {
         b = c;
         f_b = f_c;
         if (side == -1) { f_a /= 2.0; }
         side = -1;
      }
      else if (f_c * f_a > 0)
      {
         a = c;
         f_a = f_c;
         if (side == 1) { f_b /= 2.0; }
         side = 1;
      }
      else
      {
         done = true; break;
      }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. "
                   "Result may not be accurate.");
   }
   y = 0.0;
   psi += c;
   int_sigmoid_psi->Assemble();
   real_t material_volume = int_sigmoid_psi->Sum();
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &material_volume, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   }
#endif
   return material_volume;
}

} // namespace mfem

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

/*
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  State problem (nonlinear, replacing ex37's linear elasticity):
 *
 *      find u such that  dV(rho_tilde,u)/du = F + lambda*e,   e^T u = delta
 *
 *  where V(rho_tilde,u) = int_Omega  r(rho_tilde) psi(F(u))  dx  is the SIMP-
 *  weighted hyperelastic strain energy, r(rho_tilde) = rho_min + rho_tilde^n (1-rho_min),
 *  and delta is a prescribed displacement enforced through the reaction
 *  force / Lagrange multiplier lambda (as in [4], not present in ex37).
 *
 *  Objective (choose one; K&S [4] eq. 27):
 *
 *      c1(rho) = -(V(rho_tilde,u(rho)) - F.u(rho) - lambda(rho)*(delta - e.u(rho)))
 *      c2(rho) = 1/2 F.u(rho) - 1/2 lambda(rho)*delta
 *
 *  Unlike ex37's compliance objective, neither c1 nor c2 satisfies
 *  "dual = -primal" in general once V is nonlinear:
 *
 *      c1 sensitivity: self-adjoint by stationarity of the Lagrangian
 *                       (not by symmetry) -- ds1/drho_tilde = dV/drho_tilde, no
 *                       adjoint solve (K&S eq. 30-32).
 *      c2 sensitivity: needs one adjoint solve, eq. (33)-(35), reusing
 *                       the converged Newton tangent (which is symmetric).
 *
 *  Discretization choices (unchanged from ex37):
 *
 *     u  in V   subset (H1)^d      (order p)
 *     psi in L2 (order p-1), rho = sigmoid(psi)
 *     rho_tilde in H1                   (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  1. Initialize psi = inv_sigmoid(vol_fraction).
 *
 *  While not converged:
 *
 *     2. Filter solve (DiffusionSolver, unmodified from ex37):
 *
 *           (epsilon^2 grad(rho_tilde), grad(v)) + (rho_tilde,v) = (rho,v)   for all v.
 *
 *     3. Nonlinear state solve (Newton, replaces ex37's single linear
 *        solve): dV/du = F + lambda*e,  e^T u = delta. Extract lambda as
 *        the reaction on the prescribed-displacement dofs.
 *
 *     4. Sensitivity wrt rho_tilde (K&S eq. 30-35; replaces ex37's closed-form
 *        StrainEnergyDensityCoefficient):
 *          c1: s = dV/drho_tilde                                 (no solve)
 *          c2: solve J^T w = 1/2[F; -delta];  s = -w^T dh/drho_tilde  (one solve)
 *
 *     5. Adjoint filter solve, same discrete operator as Step 2 (self-
 *        adjoint), applied to the pointwise sensitivity field:
 *
 *           (epsilon^2 grad(w~), grad(v)) + (w~,v) = (s,v)   for all v.
 *
 *     6. Project the gradient onto the discrete latent space:
 *
 *                          (G,v) = (w~,v)   for all v in L2.
 *
 *     7. Bregman proximal gradient update (unmodified from ex37):
 *
 *                            psi <- psi - alpha*G + c,
 *
 *     where c enforces  int_Omega sigmoid(psi - alpha*G + c) dx = theta*vol(Omega).
 *
 *  end
 */

#ifndef MFEM_USE_ENZYME
// Enzyme is not available, trigger a compile-time error
#error "Enzyme is not available"
#endif

constexpr int dim = 2;

// The hyperelastic energies below are written in dim dimensions throughout:
// the isochoric split is J^(-2/dim) and each invariant is offset by its own
// value at F = I, so psi(I) = 0 and the reference state is stress free for
// any dim. For dim == 3 these are the standard 3D expressions.
//
// I1 = tr(C) and I2 = (tr(C)^2 - tr(C^2))/2 are the first two invariants of
// the dim x dim tensor C, so at F = I they equal dim and dim(dim-1)/2.
constexpr real_t I1_ref = real_t(dim);
constexpr real_t I2_ref = real_t(dim * (dim - 1) / 2);

constexpr int Displacement = 0;
constexpr int Coords       = 1;
constexpr int Density      = 2;   // filtered (physical) density, rho_tilde
constexpr int Energy       = 3;

enum class MaterialType { NeoHookean, LinearElastic, MooneyRivlin, Holzapfel };
enum class ObjectiveType { PotentialEnergy, Compliance };   // c1, c2
enum class PreconditionerType { None, Diagonal, AMG };

MaterialType ParseMaterial(const char *material)
{
   const std::string name(material);
   if (name == "neo-hookean" || name == "neohookean") { return MaterialType::NeoHookean; }
   if (name == "linear-elastic" || name == "linear")  { return MaterialType::LinearElastic; }
   if (name == "mooney-rivlin" || name == "mooney" || name == "rivlin")
   {
      return MaterialType::MooneyRivlin;
   }
   if (name == "holzapfel" || name == "fiber" || name == "fiber-reinforced")
   {
      return MaterialType::Holzapfel;
   }
   MFEM_ABORT("Unknown material '" << name
              << "'. Available materials: neo-hookean, linear-elastic, "
              << "mooney-rivlin, holzapfel.");
   return MaterialType::NeoHookean;
}

ObjectiveType ParseObjective(const char *objective)
{
   const std::string name(objective);
   if (name == "c1" || name == "potential-energy") { return ObjectiveType::PotentialEnergy; }
   if (name == "c2" || name == "compliance")       { return ObjectiveType::Compliance; }
   MFEM_ABORT("Unknown objective '" << name << "'. Available: c1, c2.");
   return ObjectiveType::PotentialEnergy;
}

// SIMP-interpolated strain-energy q-function: rho is the filtered
// (physical) density sampled at the quadrature point via Value<Density>.
template <typename Material, typename dscalar_t>
struct SimpHyperelasticEnergyQFunction
{
   real_t penal   = 3.0;
   real_t rho_min = 1e-3;

   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &rho,
                   const real_t &w,
                   dscalar_t &energy) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;
      // No clamping of rho. A hard clip is C^0 at rho = 1
      // (g'(1^-) = penal*(1-rho_min), g'(1^+) = 0), which is exactly where
      // solid regions of a converged design sit. Both AD paths take the same
      // branch, so symmetry thru Schwarz would still be satisfied, but mixed
      // block is one-sided and wrong along that bound. rho in [0,1] holds
      // automatically given box constraints plus a convex-average filter;
      // the assert is here to catch the case where that stops being true.
      MFEM_ASSERT(rho >= -1e-12 && rho <= 1.0 + 1e-12,
                  "filtered density outside [0,1]: check the box "
                  "constraints and that the filter weights are a "
                  "convex average.");
      MFEM_ASSERT(rho_min > 0.0, "rho_min must be strictly positive.");
      MFEM_ASSERT(penal >= 1.0,
                  "penal < 1 makes rho^penal non-C^1 at rho = 0, which "
                  "breaks the mixed Hessian block used by the adjoint.");
      const real_t rho_simp = rho_min + (1.0 - rho_min) * SimpPower(rho, penal);
      energy = rho_simp * material().psi(F, dudx) * det(J) * w;
   }

private:
   MFEM_HOST_DEVICE inline
   const Material& material() const { return static_cast<const Material&>(*this); }
};

// mu = 2*C1 = 100, kappa = 2*D1 = 200. Reference calibration for the others.
template <typename dscalar_t>
struct NeoHookeanEnergy :
   SimpHyperelasticEnergyQFunction<NeoHookeanEnergy<dscalar_t>, dscalar_t>
{
   real_t D1 = 100.0;
   real_t C1 = 50.0;
   bool trace_params = false;

   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> &F,
                 const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {

      if (trace_params)
      {
         printf("NeoHookeanEnergy::psi retained parameters: C1=%g D1=%g\n",
                static_cast<double>(C1), static_cast<double>(D1));
      }

      const auto C = transpose(F) * F;
      const auto J = det(F);
      const auto I1_bar = pow(J, -2.0 / real_t(dim)) * tr(C);
      return D1 * (J - 1.0) * (J - 1.0) + C1 * (I1_bar - I1_ref);
   }
};

template <typename dscalar_t>
struct LinearElasticEnergy :
   SimpHyperelasticEnergyQFunction<LinearElasticEnergy<dscalar_t>, dscalar_t>
{
   real_t lambda = (dim == 2) ? 100.0 : 400.0 / 3.0;
   real_t mu = 100.0;

   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> & /* F */,
                 const tensor<dscalar_t, dim, dim> &dudx) const
   {
      const auto strain = sym(dudx);
      const auto tr_strain = tr(strain);
      return 0.5 * lambda * tr_strain * tr_strain + mu * ddot(strain, strain);
   }
};

template <typename dscalar_t>
struct MooneyRivlinEnergy :
   SimpHyperelasticEnergyQFunction<MooneyRivlinEnergy<dscalar_t>, dscalar_t>
{
   real_t c1 = (dim == 2) ? 50.0 : 25.0;
   real_t c2 = (dim == 2) ?  0.0 : 25.0;
   real_t kappa = 200.0;

   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> &F,
                 const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {
      const auto C = transpose(F) * F;
      const auto J = det(F);
      const auto Jm2d = pow(J, -2.0 / real_t(dim));
      const auto I1 = tr(C);
      const auto I2 = 0.5 * (I1 * I1 - ddot(C, C));
      const auto I1_bar = Jm2d * I1;
      const auto I2_bar = Jm2d * Jm2d * I2;
      const auto log_J = log(J);
      return c1 * (I1_bar - I1_ref) + c2 * (I2_bar - I2_ref)
             + 0.5 * kappa * log_J * log_J;
   }
};

template <typename dscalar_t>
struct HolzapfelEnergy :
   SimpHyperelasticEnergyQFunction<HolzapfelEnergy<dscalar_t>, dscalar_t>
{
   real_t c = 100.0;
   real_t kappa = 200.0;
   real_t k1 = 10.0;
   real_t k2 = 20.0;

   tensor<real_t, dim> a0 = make_tensor<dim>(
   [](int) { return 1.0 / sqrt(real_t(dim)); });


   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> &F,
                 const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {
      const auto C = transpose(F) * F;
      const auto J = det(F);
      const auto Jm2d = pow(J, -2.0 / real_t(dim));
      const auto I1_bar = Jm2d * tr(C);
      const auto I4 = dot(a0, C * a0);
      const auto fiber_strain = I4 - 1.0;
      const auto log_J = log(J);
      const auto psi_vol = 0.5 * kappa * log_J * log_J;
      const auto psi_iso = 0.5 * c * (I1_bar - I1_ref);
      const auto psi_aniso = (k1 / (2.0 * k2)) * (exp(k2 * fiber_strain *
                                                      fiber_strain) - 1.0);
      //const auto psi_aniso = (fiber_strain > 0.0)
      //                       ? (k1 / (2.0 * k2)) *
      //                         (exp(k2 * fiber_strain * fiber_strain) - 1.0)
      //                       : dscalar_t(0.0);
      return psi_vol + psi_iso + psi_aniso;
   }
};

// Nonlinear mechanics operator with a Density input. Owns the dFEM
// DifferentiableOperator, the Newton residual/tangent, and both the c1
// (adjoint-free) and c2 (adjoint) density sensitivities.
class HyperelasticTopOptOperator : public Operator
{
public:

   //=============================
   ///<--- Hessian operator
   //=============================
   class HessianOperator : public Operator
   {
   public:
      HessianOperator(const HyperelasticTopOptOperator &oper, const Vector &state) :
         Operator(oper.Height()), ess_tdofs(oper.ess_tdofs), state(state),
         z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs, oper.rho_tdofs};
         hessian = oper.internal_energy_dop->GetSecondDerivative(Displacement, X, true);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(ess_tdofs, 0.0);
         MultiVector Y{y};
         hessian->Mult(z, Y);
         auto d_y = y.ReadWrite();
         const auto d_x = x.Read();
         const auto d_dofs = ess_tdofs.Read();
         mfem::forall(ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_y[d_dofs[i]] = d_x[d_dofs[i]];
         });
      }

      void AssembleDiagonal(Vector &diag) const override
      {
         hessian->AssembleDiagonal(diag);
         auto d_diag = diag.ReadWrite();
         const auto d_dofs = ess_tdofs.Read();
         mfem::forall(ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_diag[d_dofs[i]] = 1.0;
         });
      }

      // Matrix counterpart of AssembleDiagonal, for preconditioners that need
      // a real matrix. @a A can be uninitialized; it is allocated by dFEM and
      // ownership is passed to the caller. Eliminating the essential rows and
      // columns puts 1.0 on their diagonal, matching Mult above.
      void AssembleHessian(HypreParMatrix *&A) const
      {
         hessian->Assemble(A);
         auto Ae = A->EliminateRowsCols(ess_tdofs);
         delete Ae;
      }

   private:
      const Array<int>& ess_tdofs;
      Vector state;
      mutable Vector z;
      std::shared_ptr<DerivativeOperator> hessian;
   };

public:
   HyperelasticTopOptOperator(ParFiniteElementSpace &fes,
                              ParFiniteElementSpace &rho_fes,
                              const IntegrationRule &ir,
                              MaterialType material,
                              real_t penal,
                              real_t rho_min,
                              real_t param1 = -1.0,
                              real_t param2 = -1.0,
                              bool trace_qf_params = false) :
      Operator(fes.GetTrueVSize()),
      fes(fes), rho_fes(rho_fes), ir(ir),
      qspace(*fes.GetParMesh(), ir), qspace_vec(qspace, 1), q(qspace_vec)
   {
      auto &mesh_nodes = *static_cast<ParGridFunction *>
                         (fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      rho_tdofs.SetSize(rho_fes.GetTrueVSize());
      rho_tdofs = 1.0;

      const std::vector<FieldDescriptor> inputs =
      {
         {Displacement, &fes}, {Coords, mesh_nodes_fes}, {Density, &rho_fes}
      };
      const std::vector<FieldDescriptor> outputs = { {Energy, &qspace_vec} };

      internal_energy_dop = std::make_shared<DifferentiableOperator>(
                               inputs, outputs, *fes.GetParMesh());

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      auto derivatives = std::integer_sequence<size_t, Displacement, Density> {};
      auto second_derivatives = SecondDerivatives<Pairs::All> {};
      auto register_material = [&](auto energy)
      {
         energy.penal = penal;
         energy.rho_min = rho_min;
         internal_energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            Inputs<Gradient<Displacement>, Gradient<Coords>, Value<Density>, Weight> {},
            Outputs<FunctionalValue<Energy>> {},
            ir, all_domain_attr, derivatives, second_derivatives);
      };

      // param1 and param2 carry the same physical meaning for every model:
      //
      //     param1 -> shear (deviatoric) modulus knob
      //     param2 -> volumetric (bulk) modulus knob
      //
      // so that -p1/-p2 are comparable across -mat choices. A negative value
      // leaves that model's own calibrated default in place. The models are
      // calibrated to a common reference of mu = 100, kappa = 200:
      //
      //     linear-elastic  param1 = mu,  param2 = lambda
      //     neo-hookean     param1 = C1,  param2 = D1      (mu = 2 C1, kappa = 2 D1)
      //     mooney-rivlin   param1 = c1,  param2 = kappa   (mu = 2 (c1 + c2))
      //
      // Mooney-Rivlin's c2 keeps its dim-dependent default: in 2D the second
      // invariant is degenerate and c2 has no effect at all.
      const bool set1 = param1 >= 0.0;
      const bool set2 = param2 >= 0.0;

      switch (material)
      {
         case MaterialType::NeoHookean:
         {
            NeoHookeanEnergy<dscalar_t> energy;
            if (set1) { energy.C1 = param1; }
            if (set2) { energy.D1 = param2; }
            energy.trace_params = trace_qf_params;
            register_material(energy);
            break;
         }
         case MaterialType::LinearElastic:
         {
            LinearElasticEnergy<dscalar_t> energy;
            if (set1) { energy.mu = param1; }
            if (set2) { energy.lambda = param2; }
            register_material(energy);
            break;
         }
         case MaterialType::MooneyRivlin:
         {
            MooneyRivlinEnergy<dscalar_t> energy;
            if (set1) { energy.c1 = param1; }
            if (set2) { energy.kappa = param2; }
            register_material(energy);
            break;
         }
         case MaterialType::Holzapfel:
         {
            HolzapfelEnergy<dscalar_t> energy;
            if (set1) { energy.c = param1; }
            if (set2) { energy.kappa = param2; }
            register_material(energy);
            break;
         }
      }

      gradient = internal_energy_dop->GetDerivative(Displacement);
      density_sensitivity = internal_energy_dop->GetDerivative(Density);
   }

   void SetEssentialAttributes(const Array<int> &ess_bdr)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   void SetPrescribedDisplacementAttributes(const Array<int> &disp_bdr)
   {
      fes.GetEssentialTrueDofs(disp_bdr, prescribed_tdofs);
   }

   const Array<int>& GetPrescribedDisplacementTDofs() const { return prescribed_tdofs; }
   const Array<int>& GetEssentialTDofs() const { return ess_tdofs; }

   void SetDensity(const Vector &rho_phys_tdofs) { rho_tdofs = rho_phys_tdofs; }
   void SetExternalLoad(const Vector &load_tdofs) { external_load_tdofs = load_tdofs; }

   void UnconstrainedResidual(const Vector &x, Vector &y) const
   {
      MultiVector X{x, mesh_nodes_tdofs, rho_tdofs};
      MultiVector Y{y};
      gradient->Mult(X, Y);
      y -= external_load_tdofs;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      UnconstrainedResidual(x, y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator& GetGradient(const Vector &x) const override
   {
      hessian = std::make_shared<HessianOperator>(*this, x);
      return *hessian;
   }

   // c1 sensitivity, K&S eq. (31)-(32): s1 = dV/drho_tilde, no adjoint solve.
   void DensitySensitivity(const Vector &u, Vector &s) const
   {
      MultiVector X{u, mesh_nodes_tdofs, rho_tdofs};
      MultiVector S{s};
      density_sensitivity->Mult(X, S);
   }

   // Temporary hardcoded c2 sensitivity:
   // approximate -w^T dR/drho with a centered finite-difference residual
   // contraction. This is only a stopgap until a mixed dFEM derivative
   // is implemented directly.
   //
   // Debug-only and deliberately serial-order: the perturbed dof is swept in
   // global order, one dof at a time, so every rank makes the same number of
   // (collective) residual evaluations and the w^T dR contraction is reduced
   // across ranks. Cost is 2 * (global rho t-dofs) residual evaluations, so
   // this is only usable on small meshes.
   void MixedDensitySensitivityFD(const Vector &u, const Vector &w,
                                  const Vector &rho_phys_tdofs, Vector &s)
   {
      const real_t h_base = 1e-6;
      const MPI_Comm comm = fes.GetComm();
      const int nranks = Mpi::WorldSize();
      const int myrank = Mpi::WorldRank();

      Vector rho_saved = rho_tdofs;
      Vector rho_pert = rho_phys_tdofs;
      Vector R_plus;
      Vector R_minus;

      s.SetSize(rho_phys_tdofs.Size());
      s = 0.0;

      // Local t-dof counts of every rank, so the sweep is identical everywhere.
      Array<int> local_sizes(nranks);
      const int my_size = rho_pert.Size();
      MPI_Allgather(&my_size, 1, MPI_INT, local_sizes.GetData(), 1, MPI_INT, comm);

      for (int r = 0; r < nranks; r++)
      {
         for (int j = 0; j < local_sizes[r]; j++)
         {
            const bool mine = (r == myrank);
            const real_t rho_j = mine ? rho_pert(j) : 0.0;
            const real_t h = h_base * std::max(real_t(1.0), std::abs(rho_j));

            if (mine) { rho_pert(j) = rho_j + h; }
            SetDensity(rho_pert);
            UnconstrainedResidual(u, R_plus);

            if (mine) { rho_pert(j) = rho_j - h; }
            SetDensity(rho_pert);
            UnconstrainedResidual(u, R_minus);

            real_t contraction = 0.0;
            for (int i = 0; i < w.Size(); i++)
            {
               contraction += w(i) * (R_plus(i) - R_minus(i));
            }
            real_t global_contraction = 0.0;
            MPI_Allreduce(&contraction, &global_contraction, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_SUM, comm);

            if (mine)
            {
               s(j) = -0.5 * global_contraction / h;
               rho_pert(j) = rho_j;
            }
         }
      }

      rho_tdofs = rho_saved;
   }

   // Compute the c2 sensitivity
   //   1. solve J^T w = 1/2[F; -delta],
   //   2. s = -(dR/d rho_tilde)^T w,  with R = grad_u V.
   void MixedDensitySensitivity(const Vector &u, const Vector &w, Vector &s) const
   {
      MultiVector X{u, mesh_nodes_tdofs, rho_tdofs};
      MultiVector S{s};
      internal_energy_dop->GetSecondDerivative(Density, Displacement, X)->Mult(w, S);
      s.Neg();
   }

private:
   ParFiniteElementSpace &fes;
   ParFiniteElementSpace &rho_fes;
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   const IntegrationRule &ir;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   QuadratureFunction q;
   Vector mesh_nodes_tdofs;
   Vector rho_tdofs;
   Vector external_load_tdofs;
   Array<int> ess_tdofs;
   Array<int> prescribed_tdofs;

   std::shared_ptr<DifferentiableOperator> internal_energy_dop;
   std::shared_ptr<DerivativeOperator> gradient;
   std::shared_ptr<DerivativeOperator> density_sensitivity;
   mutable std::shared_ptr<HessianOperator> hessian;
};

// BoomerAMG preconditioner for the matrix-free Hessian. HypreBoomerAMG needs a
// real HypreParMatrix, so on every SetOperator we let dFEM assemble the second
// derivative and setup AMG.
class HessianAMG : public Solver
{
public:
   HessianAMG(ParFiniteElementSpace *fes_) : fes(fes_)
   {
      amg.SetPrintLevel(0);
   }

   void SetOperator(const Operator &op) override
   {
      const auto *H =
         dynamic_cast<const HyperelasticTopOptOperator::HessianOperator *>(&op);
      MFEM_VERIFY(H, "HessianAMG requires a HessianOperator");
      height = width = op.Height();

      delete A;
      A = nullptr;
      H->AssembleHessian(A);
      amg.SetOperator(*A);
      // Tell BoomerAMG this is a dim-component displacement system rather than
      // a scalar one. order_bynodes MUST be true: the state space is built as
      // ParFiniteElementSpace(&pmesh, &fec, dim), which is Ordering::byNODES,
      // while hypre's default assumes byVDIM. Getting this flag wrong hands
      // hypre a bogus dof -> function map and converges *worse* than passing no
      // systems options at all. Must follow SetOperator: the dof map is sized
      // from height, which SetOperator establishes.
      amg.SetSystemsOptions(fes->GetVDim(), /*order_bynodes=*/true);
   }

   void Mult(const Vector &x, Vector &y) const override { amg.Mult(x, y); }

   ~HessianAMG() { delete A; }

private:
   ParFiniteElementSpace *fes;
   HypreParMatrix *A = nullptr;
   HypreBoomerAMG amg;
};

// Build the preconditioner selected by -pc. The state solve and the c2 adjoint
// solve both use it, on the same operator (the converged Newton tangent), but
// each needs its OWN instance: IterativeSolver::SetOperator forwards to the
// preconditioner, so a shared instance would be torn down and re-setup by
// whichever of the two solves ran last.
std::unique_ptr<Solver> MakePreconditioner(PreconditionerType type,
                                          ParFiniteElementSpace *fes)
{
   switch (type)
   {
      case PreconditionerType::None:
         return nullptr;
      case PreconditionerType::Diagonal:
         return std::make_unique<OperatorJacobiSmoother>();
      case PreconditionerType::AMG:
         return std::make_unique<HessianAMG>(fes);
      default:
         MFEM_ABORT("Unknown preconditioner type: " << static_cast<int>(type));
   }
   return nullptr;
}

class HelmholtzFilter
{
   // For this case we assume pure Neumann boundary conditions (bilinear form is coercive due to the mass term).
public:
   HelmholtzFilter(ParFiniteElementSpace &fes_, real_t epsilon,
                   real_t mass = 1.0) :
      fes(fes_), solution_gf(&fes_)
   {

      ConstantCoefficient mass_coeff(mass);
      ConstantCoefficient eps2(epsilon * epsilon);

      a = std::make_unique<ParBilinearForm>(&fes);
      a->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      a->AddDomainIntegrator(new MassIntegrator(mass_coeff));
      a->Assemble();
      Array<int> empty;
      a->FormSystemMatrix(empty, A);

      prec = new HypreBoomerAMG(*A.As<HypreParMatrix>());

      prec->SetPrintLevel(0);
      solver = new CGSolver(fes.GetComm());
      solver->SetRelTol(1e-12);
      solver->SetMaxIter(10000);
      solver->SetPrintLevel(0);
      solver->SetOperator(*A);
      solver->SetPreconditioner(*prec);

      solution_gf = 0.0;

      tdof_size = fes.GetTrueVSize();
   }

   ~HelmholtzFilter()
   {
      delete solver;
      delete prec;
   }

   void Solve(Coefficient &rhs)
   {
      ParLinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs));
      b.Assemble();
      B.SetSize(tdof_size);
      b.ParallelAssemble(B);

      Solve(B);
   }

   void Solve(Vector &rhs)
   {
      X.SetSize(tdof_size);
      X = 0.0;
      solver->Mult(rhs, X);
      solution_gf.SetFromTrueDofs(X);
   }

   ParGridFunction &GetSolution() { return solution_gf; }

private:

   ParFiniteElementSpace &fes;
   std::unique_ptr<ParBilinearForm> a;
   ParGridFunction solution_gf;
   OperatorHandle A;
   Vector B, X;
   HypreBoomerAMG* prec;
   CGSolver* solver;
   int tdof_size;
};


int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

#ifndef MFEM_USE_ENZYME
   if (Mpi::Root())
   {
      mfem::out << "This miniapp requires MFEM_USE_ENZYME=YES because it uses "
                << "dFEM functional second derivatives.\n";
   }
   return 0;
#endif

   //======================================================================
   ///<--- Parse command-line options (names follow ex37p where the concepts
   ///     overlap).
   //======================================================================
   int order = 2;
   int ref_levels = 5;
   real_t alpha = 1.0;
   real_t growth = 2.0;
   real_t alpha_max = 300;
   real_t epsilon = 0.01;
   real_t vol_fraction = 0.5;
   int max_it = 1000;
   real_t itol = 1e-2;
   real_t ntol = 1e-4;
   real_t rho_min = 1e-3;
   real_t penal = 3.0;

   // Material parameters. Negative values keep each model's calibrated defaults.
   real_t param1 = -1.0;
   real_t param2 = -1.0;

   const char *material_name = "linear-elastic";
   const char *objective_name = "c1";
   int prec_type = static_cast<int>(PreconditionerType::Diagonal);
   bool paraview_output = false;
   bool trace_qf_params = false;
   bool fd_sensitivity_check = false;
   const char *outfolder = "output";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for mirror descent.");
   args.AddOption(&alpha_max, "-amax", "--alpha-max",
                  "Upper bound on the mirror descent step length.");
   args.AddOption(&growth, "-growth", "--alpha-growth-rate",
                  "Growth rate of step length for mirror descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for the Helmholtz filter.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of design iterations.");
   args.AddOption(&ntol, "-ntol", "--rel-tol", "Normalized exit tolerance.");
   args.AddOption(&itol, "-itol", "--abs-tol", "Increment exit tolerance.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&rho_min, "-rmin", "--rho-min", "Minimum SIMP density (void).");
   args.AddOption(&penal, "-p", "--penalization", "SIMP penalization exponent.");
   args.AddOption(&param1, "-p1", "--param1",
                  "Material parameter 1: shear/deviatoric knob. Defaults: "
                  "linear mu, Neo-Hookean C1, Mooney-Rivlin c1, Holzapfel c.");
   args.AddOption(&param2, "-p2", "--param2",
                  "Material parameter 2: volumetric/bulk knob. Defaults: "
                  "linear lambda, Neo-Hookean D1, Mooney-Rivlin kappa, "
                  "Holzapfel kappa.");
   args.AddOption(&material_name, "-mat", "--material",
                  "Material: neo-hookean, linear-elastic, mooney-rivlin, holzapfel.");
   args.AddOption(&objective_name, "-obj", "--objective",
                  "Objective: c1 (potential energy, adjoint-free) or c2 (compliance).");
   args.AddOption(&prec_type, "-pc", "--preconditioner",
                  "Preconditioner: 0 = none, 1 = diagonal/Jacobi, 2 = AMG.");
   args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Enable or disable ParaView output.");
   args.AddOption(&trace_qf_params, "-trace-qf-params", "--trace-qf-params",
                  "-no-trace-qf-params", "--no-trace-qf-params",
                  "Print Neo-Hookean q-function parameters from Func calls.");
   args.AddOption(&fd_sensitivity_check, "-fdcheck", "--fd-sensitivity-check",
                  "-no-fdcheck", "--no-fd-sensitivity-check",
                  "Finite-difference check of the c2 mixed density sensitivity. "
                  "Very expensive: 2 residual evaluations per density t-dof per "
                  "design step. Small meshes only.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for ParaView files.");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(std::cout); }

   const MaterialType material = ParseMaterial(material_name);
   const ObjectiveType objective = ParseObjective(objective_name);

   Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL,
                                     true, 3.0, 1.0);
   MFEM_VERIFY(mesh.Dimension() == dim, "Unexpected mesh dimension.");

   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      Element *be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      const real_t *coords1 = mesh.GetVertex(vertices[0]);
      const real_t *coords2 = mesh.GetVertex(vertices[1]);
      const real_t center_x = 0.5 * (coords1[0] + coords2[0]);

      be->SetAttribute(std::abs(center_x) < 1e-10 ? 1 : 2);
   }
   mesh.SetAttributes();

   mesh.EnsureNodes();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();


   //======================================================================
   ///<--- Define the necessary finite element spaces.
   //======================================================================
   H1_FECollection state_fec(order, dim);               // space for u
   H1_FECollection filter_fec(order, dim);              // space for rho_tilde
   L2_FECollection control_fec(std::max(order - 1, 0), dim,
                               BasisType::GaussLobatto); // space for psi
   ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   const HYPRE_BigInt state_size = state_fes.GlobalTrueVSize();
   const HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
   const HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of state unknowns: "   << state_size << "\n"
                << "Number of filter unknowns: "  << filter_size << "\n"
                << "Number of control unknowns: " << control_size << std::endl;
   }

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order + 1);


   //======================================================================
   ///<--- Set up the nonlinear elasticity state problem.
   //======================================================================
   HyperelasticTopOptOperator elasticity_op(state_fes, filter_fes, ir, material,
                                            penal, rho_min, param1, param2,
                                            trace_qf_params);

   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_attr(pmesh.bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[0] = 1;
      elasticity_op.SetEssentialAttributes(ess_attr);
   }

   Vector center(dim);
   center(0) = 2.9;
   center(1) = 0.5;
   Vector force(dim);
   force = 0.0;
   force(1) = -1.0;
   real_t forceadius = 0.05;
   VolumeForceCoefficient vforce_cf(forceadius, center, force);
   ParLinearForm load_form(&state_fes);
   load_form.AddDomainIntegrator(new VectorDomainLFIntegrator(vforce_cf));
   load_form.Assemble();
   Vector load_tdofs;
   load_tdofs.SetSize(state_fes.GetTrueVSize());
   load_form.ParallelAssemble(load_tdofs);
   elasticity_op.SetExternalLoad(load_tdofs);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(10000);
   cg.SetPrintLevel(2);

   std::unique_ptr<Solver> pc =
      MakePreconditioner(static_cast<PreconditionerType>(prec_type), &state_fes);
   if (pc) { cg.SetPreconditioner(*pc); }

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(cg);
   newton.SetOperator(elasticity_op);
#ifdef MFEM_USE_SINGLE
   newton.SetRelTol(1e-4);
#else
   newton.SetRelTol(1e-6);
#endif
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);


   //======================================================================
   ///<--- Set up the Helmholtz filter.
   //======================================================================
   HelmholtzFilter filter(filter_fes, epsilon);

   ConstantCoefficient one(1.0);
   ParBilinearForm mass_control(&control_fes);
   mass_control.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(
                                                             one)));
   mass_control.Assemble();
   HypreParMatrix M_control;
   Array<int> empty;
   mass_control.FormSystemMatrix(empty, M_control);

   //======================================================================
   ///<--- Set the initial guess for rho.
   //======================================================================
   ParGridFunction u(&state_fes);
   ParGridFunction psi(&control_fes), psi_old(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = vol_fraction;
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   ParGridFunction rho_gf(&control_fes);
   DiffMappedGridFunctionCoefficient succ_diffho(&psi, &psi_old, sigmoid);

   // Measure of non-discreteness, Mnd = int 4 rho (1-rho) dx / |Omega|. It is 0
   // for a fully 0/1 design and 1 for a uniformly grey rho = 1/2, so it says
   // whether a plateau in the objective is a real converged structure or the
   // optimizer stalling in grey material.
   MappedGridFunctionCoefficient nondiscrete_cf(&psi, [](const real_t x)
   {
      const real_t s = sigmoid(x);
      return 4.0 * s * (1.0 - s);
   });

   ParGridFunction grad(&control_fes);
   ParGridFunction w_filter(&filter_fes);

   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes); onegf = 1.0;
   ParGridFunction zerogf(&control_fes); zerogf = 0.0;
   ParLinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   const real_t domain_volume = vol_form(onegf);
   const real_t target_volume = domain_volume * vol_fraction;


   //======================================================================
   ///<--- ParaView output
   //======================================================================
   ParaViewDataCollection* paraview_dc = nullptr;
   if (paraview_output)
   {
      paraview_dc = new ParaViewDataCollection("dfem-topop-hyperelasticity", &pmesh);
      rho_gf.ProjectCoefficient(rho);
      paraview_dc->SetPrefixPath(outfolder);
      if (order > 1)
      {
         paraview_dc->SetLevelsOfDetail(order);
         paraview_dc->SetHighOrderOutput(true);
      }
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->RegisterField("displacement", &u);
      paraview_dc->RegisterField("density", &rho_gf);
      paraview_dc->RegisterField("filtered_density", &rho_filter);
      paraview_dc->Save();
   }


   // Vectors for the state solve, sensitivity, and adjoint solve.
   Vector U, R_full;

   // Solver for the adjoint objective (c2) sensitivity, K&S eq. (33)-(35).
   std::unique_ptr<CGSolver> cg_obj;
   std::unique_ptr<Solver> pc_obj;
   if (objective == ObjectiveType::Compliance)
   {
      cg_obj = std::make_unique<CGSolver>(MPI_COMM_WORLD);
      cg_obj->SetRelTol(1e-8);
      cg_obj->SetMaxIter(10000);
      cg_obj->SetPrintLevel(2);
      // Same -pc choice as the state solve, but a separate instance: both are
      // applied to the Newton tangent, and SetOperator forwards to whichever
      // preconditioner is attached, so sharing one would make each solve
      // invalidate the other's setup.
      pc_obj = MakePreconditioner(static_cast<PreconditionerType>(prec_type), &state_fes);
      if (pc_obj) { cg_obj->SetPreconditioner(*pc_obj); }
   }

   //======================================================================
   ///<--- Main optimization loop.
   //======================================================================
   const real_t alpha0 = alpha;
   for (int k = 1; k <= max_it; k++)
   {
      if (k > 1)
      {
         alpha = std::min(alpha0 * std::pow((real_t) k, growth), alpha_max);
      }
      if (Mpi::Root()) { mfem::out << "\nStep = " << k << std::endl; }

      //======================================================================
      // Step 2 - Filter solve: (epsilon^2 grad(rho_tilde), grad(v)) + (rho_tilde,v) = (rho,v)
      //======================================================================
      filter.Solve(rho);
      rho_filter = filter.GetSolution();

      Vector rho_filter_tdofs;
      rho_filter.GetTrueDofs(rho_filter_tdofs);
      ClampDensity(rho_filter_tdofs, "rho_filter");
      elasticity_op.SetDensity(rho_filter_tdofs);

      //======================================================================
      // Step 3 - Nonlinear state solve: dV/du = F.
      //======================================================================
      u.GetTrueDofs(U);
      newton.iterative_mode = true;
      Vector zerohs;
      newton.Mult(zerohs, U);
      u.SetFromTrueDofs(U);

      // Full residual for diagnostics and the c2 adjoint path.
      R_full.SetSize(U.Size());
      elasticity_op.UnconstrainedResidual(U, R_full);

      // Compliance, c2 = 1/2 F.u (K&S eq. (30)). The -1/2 lambda delta term is
      // absent because no prescribed displacement is applied here. For c1 the
      // objective is the potential energy, which is not assembled in this
      // miniapp, so the compliance is reported as a diagnostic instead.
      const real_t compliance =
         0.5 * InnerProduct(MPI_COMM_WORLD, load_tdofs, U);


      //======================================================================
      ///<--- Step 4 - Sensitivity wrt rho_tilde, K&S eq. (30)-(35).
      //======================================================================
      Vector s_phys(filter_fes.GetTrueVSize());
      if (objective == ObjectiveType::PotentialEnergy)
      {
         // s1 = dV/d rho_tilde
         elasticity_op.DensitySensitivity(U, s_phys);   // c1: adjoint-free
         s_phys.Neg();
      }
      else
      {
         // For a general objective c=f(u,rho), a DifferentiableOperator can
         // provide the local partials, and the adjoint closes the total sensitivity.

         // c2 compliance sensitivity:
         // 1. State residual: R(u,rho_tilde) = dV/du - external loads.
         // 2. Solve adjoint: (dR/du)^T w = dc2/du. For hyperelastic energy,
         //    dR/du = d^2V/du^2 is the symmetric displacement Hessian.
         // 3. Density sensitivity needs the residual-density coupling:
         //      dc2/d rho_tilde = direct term - w^T dR/d rho_tilde.
         // 4. Since dR/d rho_tilde = d^2V/(du d rho_tilde), this requires
         //    a mixed dFEM derivative. This path is disabled until that exists.

         // c2 = 1/2 F^T u - 1/2 lambda delta, K&S eq. (30)

         // Solve the adjoint problem: (dR/du)^T w = dc2/du
         // dc2/du = 1/2 F - 1/2 lambda delta     
         Operator &Jt = elasticity_op.GetGradient(U);
         Vector rhs(U.Size()), w_adj(U.Size());
         rhs = 0.5;
         rhs *= load_tdofs;
         rhs.SetSubVector(elasticity_op.GetEssentialTDofs(), 0.0);
         w_adj = 0.0;
         cg_obj->SetOperator(Jt);
         if (Mpi::Root()) { mfem::out << "adjoint solve (c2):\n"; }
         cg_obj->Mult(rhs, w_adj);

         elasticity_op.MixedDensitySensitivity(U, w_adj, s_phys);

         // Optional debug check, linear elasticity only: compliance is
         // self-adjoint there, so R = K u - F gives K w = 1/2 F and hence
         // w = 1/2 u exactly. This pins down the adjoint right-hand side and
         // the adjoint solve -- the two links the finite-difference check
         // below cannot see, since it holds w fixed on both sides. It does
         // not hold for the hyperelastic materials: losing self-adjointness
         // is precisely why c2 needs an adjoint solve at all.
         if (fd_sensitivity_check && material == MaterialType::LinearElastic)
         {
            Vector w_exact(U);
            w_exact *= 0.5;
            w_exact.SetSubVector(elasticity_op.GetEssentialTDofs(), 0.0);
            const real_t w_ref = GlobalLpNorm(2.0, w_exact.Norml2(), MPI_COMM_WORLD);
            w_exact -= w_adj;
            const real_t w_err = GlobalLpNorm(2.0, w_exact.Norml2(), MPI_COMM_WORLD);
            if (Mpi::Root())
            {
               mfem::out << "self-adjointness check: ||w - u/2||_2 / ||u/2||_2 = "
                         << (w_ref > 0.0 ? w_err / w_ref : w_err) << "\n";
            }
         }

         // Optional debug check: finite-difference the mixed density
         // sensitivity. Costs 2 * (global rho t-dofs) residual evaluations per
         // design step, so it is off by default.
         if (fd_sensitivity_check)
         {
            Vector s_phys_fd(filter_fes.GetTrueVSize());
            elasticity_op.MixedDensitySensitivityFD(U, w_adj, rho_filter_tdofs, s_phys_fd);
            Vector s_phys_diff(s_phys);
            s_phys_diff -= s_phys_fd;
            const real_t diff_norm = GlobalLpNorm(2.0, s_phys_diff.Norml2(), MPI_COMM_WORLD);
            const real_t fd_norm = GlobalLpNorm(2.0, s_phys_fd.Norml2(), MPI_COMM_WORLD);
            if (Mpi::Root())
            {
               mfem::out << "mixed density sensitivity check: ||mixed - fd||_2 = "
                         << diff_norm << ", ||fd||_2 = " << fd_norm << "\n";
            }
         }
      }

      //======================================================================
      ///<--- Step 5 - Adjoint filter solve, same discrete operator as Step 2.
      // (Our sensitivity comes out of dFEM already assembled as a dual
      // vector, so it is handed directly to the Helmholtz filter.)
      //======================================================================
      filter.Solve(s_phys);
      w_filter = filter.GetSolution();

      //======================================================================
      ///<---  Step 6 - Project the gradient onto the discrete latent space:
      //       G = M_control^{-1} w~.
      //======================================================================
      GridFunctionCoefficient w_cf(&w_filter);
      ParLinearForm whs(&control_fes);
      whs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      whs.Assemble();
      M_control.Mult(whs, grad);

      //======================================================================
      ///<--- Step 7 - Bregman proximal gradient update: ψ ← proj(ψ - αG)
      //======================================================================
      psi.Add(-alpha, grad);
      ParGridFunction alpha_grad(grad);
      alpha_grad *= alpha;
      const real_t material_volume = proj(psi, alpha_grad, target_volume);

      real_t norm_increment = zerogf.ComputeL1Error(succ_diffho);
      // 4 rho (1-rho) >= 0, so the L1 error against zero is the integral.
      const real_t nondiscreteness =
         zerogf.ComputeL1Error(nondiscrete_cf) / domain_volume;
      real_t normeduced_gradient = norm_increment / alpha;
      psi_old = psi;

      if (Mpi::Root())
      {
         mfem::out << (objective == ObjectiveType::Compliance
                       ? "objective (c2)   = "
                       : "compliance (c2)  = ") << compliance << "\n"
                   << "norm of the reduced gradient = " << normeduced_gradient << "\n"
                   << "norm of the increment        = " << norm_increment << "\n"
                   << "volume fraction               = " << material_volume / domain_volume
                   << "\nnon-discreteness (Mnd)       = " << nondiscreteness
                   << std::endl;
      }

      if (paraview_output)
      {
         rho_gf.ProjectCoefficient(rho);
         paraview_dc->SetCycle(k);
         paraview_dc->SetTime((real_t)k);
         paraview_dc->Save();
      }

      if (normeduced_gradient < ntol && norm_increment < itol) { break; }
   }

   delete paraview_dc;

   return 0;
}
