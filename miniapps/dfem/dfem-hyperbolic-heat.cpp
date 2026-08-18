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
//                --------------------------------------------
//                Nonlinear Hyperbolic Heat Equation with dFEM
//                --------------------------------------------
//
// Compile with: make dfem-hyperbolic-heat
//
// Sample runs:  mpirun -np 4 dfem-hyperbolic-heat
//               mpirun -np 4 dfem-hyperbolic-heat -b 0
//               mpirun -np 4 dfem-hyperbolic-heat -tau 0.02
//               mpirun -np 4 dfem-hyperbolic-heat -q 5 -rc 4
//               mpirun -np 4 dfem-hyperbolic-heat -r 1 -tf 0.6
//               mpirun -np 4 dfem-hyperbolic-heat -dt 0.2 -tf 2
//               mpirun -np 4 dfem-hyperbolic-heat -al legacy
//
// Description:  This miniapp solves the hyperbolic (Maxwell-Cattaneo) heat
//               equation with a nonlinear conductivity,
//
//                 tau d^2T/dt^2 + rho c dT/dt = div( k(T) grad T ) + Q,
//
//               on the unit cube (set the compile-time constant dim to 2 for
//               the unit square), starting from a Gaussian hot spot at rest.
//               The relaxation term tau d^2T/dt^2 turns the parabolic heat
//               equation into a damped wave equation to comply with the 
//               finite speed sqrt(k/tau) of heat propagation. 
//               Small tau recovers diffusive behavior (Fourier's law), 
//               large tau a wave-like behavior.
//
//               The conductivity k(T) = k0 (1 + beta T^2) is the only
//               nonlinearity. It is written as a dFEM q-function, so both the
//               diffusion residual
//
//                 F(T) = int k(T) grad T . grad w dx
//
//               and its tangent J_F(T) come from the same pointwise
//               description; nothing here differentiates k(T) by hand. 
//
//               We recommend viewing examples 16 and 23, and dfem miniapps
//               before viewing this example.

#include "mfem.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "../../fem/dfem/backends/local_qf/prelude.hpp"

#include <memory>

using namespace mfem;

// This example code demonstrates the use of new features in MFEM that are in
// development but exposed through the mfem::future namespace. All features
// under this namespace might change their interface or behavior in upcoming
// releases until they have stabilized.
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

// Space dimension. The q-function and the diffusion operator are templated on
// it, so it is fixed at compile time; set it to 2 to run the same problem on
// the unit square.
constexpr int dim = 3;

// Field IDs used by the dFEM integrator.
static constexpr int Temperature = 0;
static constexpr int Coords = 1;

enum class PreconditionerType { None, Diagonal };

/// Map an AssemblyLevel name onto the MFEM assembly level used for the mass operators.
/// Only partial assembly keeps the whole solve matrix free; the others are here
/// so the cost of that choice can be measured against the alternatives.
static AssemblyLevel ParseAssemblyLevel(const char *name)
{
   const std::string s(name);
   if (s == "partial") { return AssemblyLevel::PARTIAL; }
   if (s == "legacy")  { return AssemblyLevel::LEGACY; }
   if (s == "full")    { return AssemblyLevel::FULL; }
   MFEM_ABORT("Unknown assembly level '" << name << "'. Available levels: "
              "partial, legacy, full.");
   return AssemblyLevel::LEGACY;
}

// ----------------------------------------------------------------------------
// Pointwise description of the nonlinear diffusion term.
// ----------------------------------------------------------------------------

/// Quadrature point kernel for F(T) = int k(T) grad T . grad w dx with the
/// temperature dependent conductivity k(T) = k0 (1 + beta T^2). 
template <int DIM>
struct NonlinearDiffusionQFunction
{
   real_t k0 = 1.0;
   real_t beta = 1.0;

   MFEM_HOST_DEVICE inline
   auto operator()(const dscalar_t &T,
                   const tensor<dscalar_t, DIM> &dTdxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<dscalar_t, DIM> &dwdx) const
   {
      const auto invJ = inv(J);
      const auto dTdx = dTdxi * invJ;
      const dscalar_t k = k0 * (1.0_r + beta * T * T);
      dwdx = k * dTdx * transpose(invJ) * det(J) * w;
   }
};

/// dFEM wrapper for the nonlinear diffusion residual F(T) and its tangent
/// J_F(T). 
///
/// Neither F nor J_F applies any essential-dof treatment. That belongs to the
/// operator that assembles the full residual, since the boundary condition
/// constrains the acceleration, not the diffusion term by itself.
template <int DIM>
class NonlinearDiffusionOperator
{
public:
   NonlinearDiffusionOperator(ParFiniteElementSpace &fes, const IntegrationRule &ir,
                      real_t k0, real_t beta)
   {
      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Temperature, &fes}, {Coords, mesh_nodes_fes}
      };
      // The output operator is the gradient of the test function basis, which
      // completes the weak form B^T D(B T, B x, w).
      const std::vector<FieldDescriptor> outputs = {{Temperature, &fes}};

      dop = std::make_shared<DifferentiableOperator>(
               inputs, outputs, *fes.GetParMesh());

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      NonlinearDiffusionQFunction<DIM> qf;
      qf.k0 = k0;
      qf.beta = beta;

      // Requesting the derivative with respect to Temperature is what makes
      // J_F available later through GetDerivative.
      auto derivatives = std::integer_sequence<size_t, Temperature> {};
      dop->AddDomainIntegrator<LocalQFBackend>(
         qf,
         Inputs < Value<Temperature>, Gradient<Temperature>,
         Gradient<Coords>, Weight > {},
         Outputs<Gradient<Temperature>> {},
         ir, all_domain_attr, derivatives);
   }

   /// F(T), the nonlinear diffusion residual.
   void Mult(const Vector &T, Vector &F) const
   {
      MultiVector X{T, mesh_nodes_tdofs};
      MultiVector Y{F};
      dop->Mult(X, Y);
   }

   /// J_F(T), the tangent of F linearized at @a T. The returned operator
   /// captures @a T, so it has to be rebuilt whenever the linearization point
   /// moves.
   std::shared_ptr<DerivativeOperator> GetGradient(const Vector &T) const
   {
      MultiVector X{T, mesh_nodes_tdofs};
      return dop->GetDerivative(Temperature, X);
   }

private:
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   Vector mesh_nodes_tdofs;
   std::shared_ptr<DifferentiableOperator> dop;
};

// ----------------------------------------------------------------------------
// Newton residual for one implicit stage.
// ----------------------------------------------------------------------------
///
/// Residual of the semi-discrete system, written in the acceleration unknown
/// a = d^2T/dt^2 that MFEM's second order ODE solvers solve for. With the stage
/// quantities requested by the integrator,
///
///    T_stage = T + fac0 a,      v_stage = dT/dt + fac1 a,
///
/// the weak form M_tau a + M_rhoc v + F(T) - Q = 0 becomes
///
///    R(a) = M_tau a + M_rhoc v_stage + F(T_stage) - Q.
///
class ResidualOperator : public Operator
{
public:
   /// Newton tangent
   ///
   ///    J_R(a) = M_tau + fac1 M_rhoc + fac0 J_F(T + fac0 a).
   ///
   /// None of the three terms is a matrix: the mass operators are partially
   /// assembled and J_F is a dFEM derivative action, so this class only ever
   /// composes actions and diagonals.
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(const ResidualOperator &oper, const Vector &T_stage) :
         Operator(oper.Height()), oper(oper), z(oper.Height()), w(oper.Height())
      {
         diffusion_tangent = oper.diffusion.GetGradient(T_stage);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         // Essential directions are removed before the apply and restored as
         // identity rows afterwards.
         z = x;
         z.SetSubVector(oper.ess_tdofs, 0.0);

         oper.M_tau.Mult(z, y);
         oper.M_rhoc.AddMult(z, y, oper.fac1);

         MultiVector W{w};
         diffusion_tangent->Mult(z, W);
         y.Add(oper.fac0, w);

         auto d_y = y.ReadWrite();
         const auto d_x = x.Read();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_y[d_dofs[i]] = d_x[d_dofs[i]];
         });
      }

      void AssembleDiagonal(Vector &diag) const override
      {
         oper.M_tau.AssembleDiagonal(diag);

         oper.M_rhoc.AssembleDiagonal(w);
         diag.Add(oper.fac1, w);

         diffusion_tangent->AssembleDiagonal(w);
         diag.Add(oper.fac0, w);

         auto d_diag = diag.ReadWrite();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_diag[d_dofs[i]] = 1.0;
         });
      }

   private:
      const ResidualOperator &oper;
      mutable Vector z, w;
      std::shared_ptr<DerivativeOperator> diffusion_tangent;
   };

   /// @a M_tau and @a M_rhoc are the mass operators as returned by
   /// ParBilinearForm::FormSystemMatrix, i.e. constrained true-dof operators.
   ResidualOperator(const Operator &M_tau, const Operator &M_rhoc,
                    const NonlinearDiffusionOperator<dim> &diffusion,
                    const Array<int> &ess_tdofs) :
      Operator(M_tau.Height()),
      M_tau(M_tau), M_rhoc(M_rhoc), diffusion(diffusion), ess_tdofs(ess_tdofs),
      T_stage(height), v_stage(height), z(height) { }

   /// Describe the stage the time integrator is asking about. @a T, @a dTdt and
   /// @a Q are referenced, not copied, so they have to outlive the Newton solve.
   void SetParameters(real_t fac0_, real_t fac1_, const Vector *T_,
                      const Vector *dTdt_, const Vector *Q_)
   {
      fac0 = fac0_;
      fac1 = fac1_;
      T = T_;
      dTdt = dTdt_;
      Q = Q_;
   }

   void Mult(const Vector &a, Vector &R) const override
   {
      MFEM_ASSERT(T && dTdt && Q, "call SetParameters() first");

      add(*T, fac0, a, T_stage);
      add(*dTdt, fac1, a, v_stage);

      M_tau.Mult(a, R);
      M_rhoc.AddMult(v_stage, R);

      diffusion.Mult(T_stage, z);
      R += z;

      R -= *Q;

      R.SetSubVector(ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &a) const override
   {
      // The tangent has to be linearized at the stage temperature, not at the
      // temperature of the previous step.
      add(*T, fac0, a, T_stage);
      jacobian = std::make_shared<JacobianOperator>(*this, T_stage);
      return *jacobian;
   }

private:
   const Operator &M_tau;
   const Operator &M_rhoc;
   const NonlinearDiffusionOperator<dim> &diffusion;
   const Array<int> &ess_tdofs;

   real_t fac0 = 0.0, fac1 = 0.0;
   const Vector *T = nullptr;
   const Vector *dTdt = nullptr;
   const Vector *Q = nullptr;

   mutable Vector T_stage, v_stage, z;
   mutable std::shared_ptr<JacobianOperator> jacobian;
};

// ----------------------------------------------------------------------------
// The second order time dependent operator.
// ----------------------------------------------------------------------------

/// Semi-discrete hyperbolic heat equation
///
///     M_tau a + M_rhoc v + F(T) - Q = 0,
///
class HyperbolicHeatOperator : public SecondOrderTimeDependentOperator
{
public:
   HyperbolicHeatOperator(ParFiniteElementSpace &fes,
                          const IntegrationRule &ir,
                          const Array<int> &ess_bdr,
                          real_t tau, real_t rho_c, real_t k0, real_t beta,
                          real_t source, PreconditionerType pc_type,
                          AssemblyLevel assembly);

   /// Compute a = M_tau^{-1} ( Q - M_rhoc v - F(T) ).
   using SecondOrderTimeDependentOperator::Mult;
   void Mult(const Vector &T, const Vector &dTdt,
             Vector &d2Tdt2) const override;

   /// Solve R(a) = M_tau a + M_rhoc (dTdt + fac1 a) + F(T + fac0 a) - Q = 0 for
   /// the unknown acceleration a = d2Tdt2.
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   void ImplicitSolve(const real_t fac0, const real_t fac1,
                      const Vector &T, const Vector &dTdt,
                      Vector &d2Tdt2) override;

   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// Newton iterations taken by the last ImplicitSolve.
   int GetNewtonIterations() const { return newton_solver.GetNumIterations(); }

   /// Compute the physical L2 norm directly from a true-dof vector.
   real_t ComputeL2Norm(const Vector &x) const;

   ~HyperbolicHeatOperator() override;

private:
   /// Set up one mass operator weighted by the constant coefficient @a w at the
   /// requested @a assembly level, and hand back its constrained true-dof form
   /// in @a M.
   void SetupMass(ParBilinearForm &m, ConstantCoefficient &w,
                  const IntegrationRule &ir, AssemblyLevel assembly,
                  OperatorHandle &M);

   ParFiniteElementSpace &fes;
   Array<int> ess_tdof_list; // empty for pure Neumann boundary conditions

   // Mass operators, weighted by tau and by rho*c. The forms have to outlive

   // the handles: under partial assembly the operator in the handle refers to
   // the quadrature point data held by the form.
   ConstantCoefficient tau_coeff, rhoc_coeff;
   real_t tau;
   ParBilinearForm m_tau, m_rhoc;
   OperatorHandle M_tau, M_rhoc;

   std::unique_ptr<NonlinearDiffusionOperator<dim>> diffusion;
   Vector Q;

   CGSolver M_solver;
   OperatorJacobiSmoother M_prec;

   std::unique_ptr<ResidualOperator> residual_op;
   NewtonSolver newton_solver;
   std::unique_ptr<IterativeSolver> krylov;
   std::unique_ptr<Solver> krylov_prec;

   mutable Vector cached_accel;
   mutable Vector rhs, z;
};

static std::unique_ptr<Solver> MakePreconditioner(PreconditionerType type)
{
   switch (type)
   {
      case PreconditionerType::None:     return nullptr;
      case PreconditionerType::Diagonal:
         return std::make_unique<OperatorJacobiSmoother>();
      default:
         MFEM_ABORT("Unknown preconditioner type: " << static_cast<int>(type));
   }
   return nullptr;
}

void HyperbolicHeatOperator::SetupMass(ParBilinearForm &m,
                                       ConstantCoefficient &w,
                                       const IntegrationRule &ir,
                                       AssemblyLevel assembly,
                                       OperatorHandle &M)
{
   m.SetAssemblyLevel(assembly);
   m.AddDomainIntegrator(new MassIntegrator(w, &ir));
   m.Assemble();
   m.FormSystemMatrix(ess_tdof_list, M);
}

HyperbolicHeatOperator::HyperbolicHeatOperator(ParFiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const Array<int> &ess_bdr,
                                               real_t tau, real_t rho_c,
                                               real_t k0, real_t beta,
                                               real_t source,
                                               PreconditionerType pc_type,
                                               AssemblyLevel assembly)
   : SecondOrderTimeDependentOperator(fes.GetTrueVSize(), (real_t) 0.0),
     fes(fes),
     tau_coeff(tau), rhoc_coeff(rho_c),
     tau(tau),
     m_tau(&fes), m_rhoc(&fes),
     M_solver(fes.GetComm()),
     newton_solver(fes.GetComm()),
     cached_accel(height), rhs(height), z(height)
{
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   /// Setup the mass operators for the acceleration and velocity terms. 
   SetupMass(m_tau, tau_coeff, ir, assembly, M_tau);
   SetupMass(m_rhoc, rhoc_coeff, ir, assembly, M_rhoc);

   /// Create the nonlinear diffusion operator as a dFEM operator 
   diffusion = std::make_unique<NonlinearDiffusionOperator<dim>>(fes, ir, k0, beta);

   /// The source term is a constant coefficient, so we can assemble it once and for all.
   ConstantCoefficient source_coeff(source);
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(source_coeff, &ir));
   b.Assemble();
   Q.SetSize(height);
   b.ParallelAssemble(Q);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-10);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*M_tau);

   /// Set up the nonlinear residual operator used by the Newton solver. 
   residual_op = std::make_unique<ResidualOperator>(*M_tau, *M_rhoc, *diffusion,
                                                    ess_tdof_list);

   /// Set up the Newton solver and Krylov solver for linearized systems. 
   if (beta == 0.0)
   {
      krylov = std::make_unique<CGSolver>(fes.GetComm());
   }
   else
   {
      krylov = std::make_unique<GMRESSolver>(fes.GetComm());
   }
   krylov->SetRelTol(1e-10);
   krylov->SetAbsTol(0.0);
   krylov->SetMaxIter(500);
   krylov->SetPrintLevel(0);

   krylov_prec = MakePreconditioner(pc_type);
   if (krylov_prec) { krylov->SetPreconditioner(*krylov_prec); }

   newton_solver.SetOperator(*residual_op);
   newton_solver.SetSolver(*krylov);
#ifdef MFEM_USE_SINGLE
   newton_solver.SetRelTol(1e-5);
   newton_solver.SetAbsTol(1e-8);
#else
   newton_solver.SetRelTol(1e-8);
   newton_solver.SetAbsTol(1e-12);
#endif
   newton_solver.SetMaxIter(25);
   newton_solver.SetPrintLevel(0);

   cached_accel = 0.0;
}

real_t HyperbolicHeatOperator::ComputeL2Norm(const Vector &x) const
{
   M_tau->Mult(x, z);
   return sqrt(InnerProduct(fes.GetComm(), x, z) / tau);
}

void HyperbolicHeatOperator::Mult(const Vector &T, const Vector &dTdt,
                                  Vector &d2Tdt2) const
{
   diffusion->Mult(T, rhs);
   M_rhoc->Mult(dTdt, z);
   rhs += z;
   rhs.Neg();
   rhs += Q;

   rhs.SetSubVector(ess_tdof_list, 0.0);
   M_solver.Mult(rhs, d2Tdt2);
   d2Tdt2.SetSubVector(ess_tdof_list, 0.0);

   cached_accel = d2Tdt2;
}

void HyperbolicHeatOperator::ImplicitSolve(const real_t fac0, const real_t fac1,
                                           const Vector &T, const Vector &dTdt,
                                           Vector &d2Tdt2)
{
   residual_op->SetParameters(fac0, fac1, &T, &dTdt, &Q);

   d2Tdt2 = cached_accel;
   d2Tdt2.SetSubVector(ess_tdof_list, 0.0);

   Vector zero;
   newton_solver.Mult(zero, d2Tdt2);
   MFEM_VERIFY(newton_solver.GetConverged(), "Nonlinear solve failed.");

   cached_accel = d2Tdt2;
}

HyperbolicHeatOperator::~HyperbolicHeatOperator() = default;

static real_t InitialTemperature(const Vector &x)
{
   real_t r2 = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      const real_t s = x(d) - 0.5;
      r2 += s * s;
   }
   return exp(-30.0 * r2);
}

static real_t InitialRate(const Vector &) { return 0.0; }




int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options
   int order = 2;
   int refinements = 0;
   const char *device_config = "cpu";
   int ode_solver_type = 11;
   real_t t_final = 0.4;
   real_t dt = 0.02;
   real_t tau = 1.0;
   real_t rho_c = 1.0;
   real_t k0 = 1.0;
   real_t beta = 1.0;
   real_t source = 0.0;
   int prec_type = static_cast<int>(PreconditionerType::Diagonal);
   const char *assembly_name = "partial";
   bool dirichlet = true;
   bool visualization = false;
   bool paraview = false;
   int vis_steps = 5;
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   // FEM
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of uniform refinements.");
   args.AddOption(&prec_type, "-pc", "--preconditioner",
                  "Preconditioner for the Newton tangent: 0 = none, "
                  "1 = diagonal/Jacobi.");
   args.AddOption(&assembly_name, "-al", "--assembly-level",
                  "Assembly level of the two mass operators: partial, legacy, or full.");
   // Time integration
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  SecondOrderODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   // Parameters/Physics
   args.AddOption(&tau, "-tau", "--tau",
                  "Thermal relaxation coefficient of d^2T/dt^2. The wave speed "
                  "is sqrt(k/tau); tau -> 0 recovers the parabolic limit.");
   args.AddOption(&rho_c, "-rc", "--rho-c",
                  "Volumetric heat capacity rho*c, the damping of the wave.");
   args.AddOption(&k0, "-k", "--conductivity",
                  "Conductivity k0 in k(T) = k0 (1 + beta T^2).");
   args.AddOption(&beta, "-b", "--beta",
                  "Nonlinearity beta in k(T) = k0 (1 + beta T^2). "
                  "beta = 0 gives a linear, symmetric problem.");
   args.AddOption(&source, "-q", "--source", "Constant volumetric source Q.");
   args.AddOption(&dirichlet, "-dir", "--dirichlet", "-neu", "--neumann",
                  "BC switch: T = 0 or zero flux on the whole boundary.");
   // Visualization
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.ParseCheck();

   MFEM_VERIFY(tau > 0.0, "-tau must be positive: the hyperbolic term is what "
               "makes this a second order problem in time.");
   MFEM_VERIFY(rho_c >= 0.0, "-rc must be non-negative.");
   MFEM_VERIFY(k0 > 0.0, "-k must be positive.");
   MFEM_VERIFY(beta >= 0.0, "-b must be non-negative to keep k(T) positive.");

   const AssemblyLevel assembly = ParseAssemblyLevel(assembly_name);

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   if (Mpi::Root())
   {
      mfem::out << "k(T) = " << k0 << " (1 + " << beta << " T^2)"
                << ", wave speed sqrt(k0/tau) = " << sqrt(k0 / tau)
                << ", relaxation time tau/(rho c) = "
                << (rho_c > 0.0 ? tau / rho_c
                    : std::numeric_limits<real_t>::infinity())
                << ", tangent: "
                << (beta == 0.0 ? "symmetric (CG)" : "non-symmetric (GMRES)")
                << ", mass assembly: " << assembly_name << std::endl;
   }

   // 4. Define the second order ODE solver used for time integration
   std::unique_ptr<SecondOrderODESolver> ode_solver(
      SecondOrderODESolver::Select(ode_solver_type));

   // 5. Create the unit square/cube mesh and refine it
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(8, 8, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(8, 8, 8, Element::HEXAHEDRON);
   for (int l = 0; l < refinements; l++) { mesh.UniformRefinement(); }
   mesh.SetCurvature(order);

   // 6. Define a parallel mesh. dFEM reads the mesh coordinates as a field, so
   //    the nodes have to exist.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();

   // 7. Define a scalar H1 space for the temperature. GlobalTrueVSize is
   //    collective, so every rank has to reach it.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   const HYPRE_BigInt global_dofs = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of temperature unknowns: " << global_dofs
                << std::endl;
   }

   // 8. Select quadrature for the nonlinear weak term
   //    k(T) grad(T).grad(w). For k(T) = k0 (1 + beta T^2), its
   //    polynomial degree on affine elements is up to 4*p - 2.
   const int integration_order =
       (beta == 0.0) ? 2 * order : 4 * order - 2;
   const IntegrationRule &ir =
       IntRules.Get(pmesh.GetTypicalElementGeometry(), integration_order);

   // 9. Set the initial conditions. We assume a gaussian hot spot at rest.
   ParGridFunction T_gf(&fes), dTdt_gf(&fes);

   FunctionCoefficient T_0(InitialTemperature);
   T_gf.ProjectCoefficient(T_0);
   Vector T;
   T_gf.GetTrueDofs(T);

   FunctionCoefficient dTdt_0(InitialRate);
   dTdt_gf.ProjectCoefficient(dTdt_0);
   Vector dTdt;
   dTdt_gf.GetTrueDofs(dTdt);

   // 10. Initialize the hyperbolic heat operator
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = dirichlet ? 1 : 0;
   }

   HyperbolicHeatOperator oper(fes, ir, ess_bdr, tau, rho_c, k0, beta, source,
                               static_cast<PreconditionerType>(prec_type), assembly);

   // The projected initial data does not necessarily satisfy T = 0 exactly on
   // the boundary, and the operator keeps the acceleration at zero there, so the
   // constraint is imposed on the initial state instead.
   T.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);
   dTdt.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);
   T_gf.SetFromTrueDofs(T);
   dTdt_gf.SetFromTrueDofs(dTdt);

   // 11. Set up visualization
   socketstream sout;
   if (visualization)
   {
      sout.open("localhost", visport);
      if (!sout)
      {
         if (Mpi::Root())
         {
            mfem::out << "Unable to connect to GLVis server at localhost:"
                      << visport << "\nGLVis visualization disabled.\n";
         }
         visualization = false;
      }
      else
      {
         sout.precision(8);
         sout << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
              << "\n";
         sout << "solution\n" << pmesh << T_gf;
         sout << "window_title 'Temperature'\n" << std::flush;
      }
   }

   ParaViewDataCollection pd("dfem-hyperbolic-heat-output", &pmesh);
   if (paraview)
   {
      pd.RegisterField("temperature", &T_gf);
      pd.RegisterField("rate", &dTdt_gf);
      pd.SetDataFormat(VTKFormat::BINARY);
      if (order > 1)
      {
         pd.SetHighOrderOutput(true);
         pd.SetLevelsOfDetail(order);
      }
      pd.SetCycle(0);
      pd.SetTime(0.0);
      pd.Save();
   }

   {
      const real_t T_norm = oper.ComputeL2Norm(T);
      const real_t dTdt_norm = oper.ComputeL2Norm(dTdt);
      if (Mpi::Root())
      {
         mfem::out << "step        t      ||T||_L2   ||dT/dt||_L2   newton\n";
         mfem::out << "   0  " << std::setw(8) << 0.0
                   << "  " << std::setw(11) << T_norm
                   << "  " << std::setw(13) << dTdt_norm
                   << "  " << std::setw(7) << "-" << std::endl;
      }
   }

   // 12. Perform time integration
   ode_solver->Init(oper);
   real_t t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt / 2) { last_step = true; }

      ode_solver->Step(T, dTdt, t, dt);

      const real_t T_norm = oper.ComputeL2Norm(T);
      const real_t dTdt_norm = oper.ComputeL2Norm(dTdt);
      if (Mpi::Root())
      {
         mfem::out << std::setw(4) << ti << "  " << std::setw(8) << t
                   << "  " << std::setw(11) << T_norm
                   << "  " << std::setw(13) << dTdt_norm
                   << "  " << std::setw(7) << oper.GetNewtonIterations()
                   << std::endl;
      }

      if ((last_step || (ti % vis_steps) == 0) && (visualization || paraview))
      {
         T_gf.SetFromTrueDofs(T);
         dTdt_gf.SetFromTrueDofs(dTdt);

         if (visualization)
         {
            sout << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
                 << "\n";
            sout << "solution\n" << pmesh << T_gf << std::flush;
         }

         if (paraview)
         {
            pd.SetCycle(ti);
            pd.SetTime(t);
            pd.Save();
         }
      }
   }

   return 0;
}

