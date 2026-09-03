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
//               ----------------------------------------------
//               Nonlinear Poisson (Diffusion) Problem with dFEM
//               ----------------------------------------------
//
// Compile with: make dfem-nonlinear-poisson
//
// Sample runs:  mpirun -np 4 dfem-nonlinear-poisson -k grad -f energy -no-vis
//               mpirun -np 4 dfem-nonlinear-poisson -k grad -f residual -no-vis
//               mpirun -np 4 dfem-nonlinear-poisson -k sol -f residual -no-vis
//               mpirun -np 4 dfem-nonlinear-poisson -o 2 -r 4 -pc 2 -no-vis
//
// Description:  This miniapp solves the nonlinear diffusion problem
//
//                   -div( kappa(u, grad u) grad u ) = f   in Omega
//                                                 u = 0   on dOmega
//
//               on the unit square (set the compile-time constant dim to 3 for
//               the unit cube). It shows the two ways a nonlinear
//               problem can be given to dFEM, and the rule that decides which
//               one is available: the coefficient (-k) determines whether an
//               energy exists, and the formulation (-f) picks how it is
//               written.
//
//               -k grad : kappa = 1 + |grad u|^2 depends only on the gradient,
//                         so the operator is the first variation of
//                         Pi(u) = int psi(|grad u|^2) dx with 2 psi' = kappa.
//                         The tangent is symmetric positive definite, and both
//                         formulations apply.
//
//               -k sol  : kappa = 1 + u^2 depends on the solution, so the
//                         operator is NOT the gradient of any functional: the
//                         tangent gains int kappa'(u) du (grad u . grad v) dx,
//                         which is not symmetric. Only -f residual applies, and
//                         the linear solves need GMRES rather than CG.
//
//               -f energy   : the q-function returns the energy density psi as
//                             a FunctionalValue. The residual is GetDerivative,
//                             the Newton tangent is GetSecondDerivative.
//
//               -f residual : the q-function returns the pointwise residual, as
//                             in the minimal surface miniapp, and the tangent
//                             is GetDerivative.
//
//               With -k grad both formulations describe the same problem and
//               produce the same answer. Convergence of both against a
//               manufactured solution is verified in
//               tests/unit/dfem/test_nonlinear_poisson.cpp.

#include "mfem.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "../../fem/dfem/backends/local_qf/prelude.hpp"

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

// Space dimension. The q-functions and the operator are templated on it, so it
// is fixed at compile time; set it to 3 to run the same problem on the unit
// cube.
constexpr int dim = 2;

// Field IDs shared by every integrator in this miniapp.
static constexpr int Solution = 0;
static constexpr int Coords = 1;
static constexpr int Energy = 2;

// The coefficient determines whether an energy formulation exists at all.
enum class KappaType
{
   GradientDependent, // kappa = 1 + |grad u|^2, symmetric, energy available
   SolutionDependent  // kappa = 1 + u^2, non-symmetric, residual only
};

enum class FormType
{
   Energy,  // q-function returns psi, tangent = GetSecondDerivative
   Residual // q-function returns the residual, tangent = GetDerivative
};

enum class PreconditionerType { None, Diagonal, AMG };

// ----------------------------------------------------------------------------
// Pointwise description of the problem: the coefficients and one q-function per
// formulation.
// ----------------------------------------------------------------------------
template <int DIM>
struct NonlinearPoisson
{
   // kappa = 1 + |grad u|^2, as a function of s = |grad u|^2.
   template <typename T>
   MFEM_HOST_DEVICE static inline T Kappa_grad(const T &s)
   {
      return 1.0_r + s;
   }

   // kappa = 1 + u^2.
   template <typename T>
   MFEM_HOST_DEVICE static inline T Kappa_sol(const T &u)
   {
      return 1.0_r + u * u;
   }

   // Energy density with 2 Psi_grad'(s) = Kappa_grad(s), so that the first
   // variation of int psi dx is int kappa grad u . grad v dx.
   template <typename T>
   MFEM_HOST_DEVICE static inline T Psi_grad(const T &s)
   {
      return 0.5_r * s + 0.25_r * s * s;
   }

   // The q-function supplies only the energy density; dFEM forms both the first
   // variation (the residual) and the second variation (the Newton tangent)
   // from it. Only Kappa_grad has a potential, so only it can be written here.
   struct EnergyBased
   {
      MFEM_HOST_DEVICE inline
      auto operator()(const tensor<dscalar_t, DIM> &dudxi,
                      const tensor<real_t, DIM, DIM> &J,
                      const real_t &w,
                      dscalar_t &energy) const
      {
         const auto dudx = dudxi * inv(J);
         // det(J) * w is the quadrature form of dx in Pi(u) = int psi dx.
         energy = Psi_grad(sqnorm(dudx)) * det(J) * w;
      }
   };

   // KAPPA selects the coefficient family. Both variants take the same inputs,
   // so they register identically; the solution value is interpolated either
   // way even though only Kappa_sol consumes it.
   template <KappaType KAPPA>
   struct ResidualBased
   {
      MFEM_HOST_DEVICE inline
      auto operator()(const dscalar_t &u,
                      const tensor<dscalar_t, DIM> &dudxi,
                      const tensor<real_t, DIM, DIM> &J,
                      const real_t &w,
                      tensor<dscalar_t, DIM> &dvdx) const
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         dscalar_t kappa;
         if constexpr (KAPPA == KappaType::GradientDependent)
         {
            kappa = Kappa_grad(sqnorm(dudx));
         }
         else
         {
            kappa = Kappa_sol(u);
         }
         dvdx = kappa * dudx * transpose(invJ) * det(J) * w;
      }
   };
};

// ----------------------------------------------------------------------------
// Depending on @a form, the DifferentiableOperator below holds either the
// energy Pi(u), whose first derivative is the residual, or the residual R(u)
// directly. Both expose a tangent through the same DerivativeOperator
// interface, so the Newton plumbing is shared.
// ----------------------------------------------------------------------------
template <int DIM>
class NonlinearPoissonOperator : public Operator
{
public:
   // Matrix-free Newton tangent. Essential directions are removed before the
   // apply and restored as identity rows afterwards.
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(const NonlinearPoissonOperator &oper,
                       const Vector &state) :
         Operator(oper.Height()), oper(oper), z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs};
         // In the energy form the residual is already the first derivative of
         // the energy, so its tangent is the second derivative. In the residual
         // form the residual is differentiated directly.
         tangent = (oper.form == FormType::Energy)
                   ? oper.dop->GetSecondDerivative(Solution, X)
                   : oper.dop->GetDerivative(Solution, X);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(oper.ess_tdofs, 0.0);

         MultiVector Y{y};
         tangent->Mult(z, Y);

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
         tangent->AssembleDiagonal(diag);
         auto d_diag = diag.ReadWrite();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_diag[d_dofs[i]] = 1.0;
         });
      }

      // Matrix counterpart of AssembleDiagonal, for preconditioners that need a
      // real matrix. @a A can be uninitialized; it is allocated by dFEM and
      // ownership is passed to the caller. Eliminating the essential rows and
      // columns puts 1.0 on their diagonal, matching Mult above.
      void AssembleJacobian(HypreParMatrix *&A) const
      {
         tangent->Assemble(A);
         auto Ae = A->EliminateRowsCols(oper.ess_tdofs);
         delete Ae;
      }

   private:
      const NonlinearPoissonOperator &oper;
      mutable Vector z;
      std::shared_ptr<DerivativeOperator> tangent;
   };

   NonlinearPoissonOperator(ParFiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            FormType form, KappaType kappa) :
      Operator(fes.GetTrueVSize()),
      fes(fes),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      form(form)
   {
      MFEM_VERIFY(form == FormType::Residual ||
                  kappa == KappaType::GradientDependent,
                  "the energy formulation requires -k grad: a solution "
                  "dependent coefficient is not the gradient of a functional");

      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Solution, &fes}, {Coords, mesh_nodes_fes}
      };

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      auto derivatives = std::integer_sequence<size_t, Solution> {};

      // We need the action, AssembleDiagonal and AssembleMatrix
      // (for jacobi, AMG preconditioning). AllAssembly is a convenience for both.
      constexpr auto kernels =
         DerivativeKernels::Action |
         DerivativeKernels::AllAssembly;

      if (form == FormType::Energy)
      {
         // A single scalar per quadrature point, summed by dFEM into the value
         // of the functional.
         const std::vector<FieldDescriptor> outputs = {{Energy, &qspace_vec}};
         dop = std::make_shared<DifferentiableOperator>(
                  inputs, outputs, *fes.GetParMesh());

         // Requesting the second derivative here is what makes the Newton
         // tangent available later through GetSecondDerivative.
         auto second_derivatives = SecondDerivatives<Pairs::All> {};
         typename NonlinearPoisson<DIM>::EnergyBased qf;
         dop->AddDomainIntegrator<LocalQFBackend, kernels>(
            qf,
            Inputs<Gradient<Solution>, Gradient<Coords>, Weight> {},
            Outputs<FunctionalValue<Energy>> {},
            ir, all_domain_attr, derivatives, second_derivatives);

         // The first variation of a functional is stateless: the current
         // solution is passed to gradient->Mult(X, Y) on every residual
         // evaluation, so this wrapper is built once and reused.
         gradient = dop->GetDerivative(Solution);
      }
      else
      {
         // The output operator is the gradient of the test function basis,
         // completing the diffusion-like weak form B^T D(B u, B x, w).
         const std::vector<FieldDescriptor> outputs = {{Solution, &fes}};
         dop = std::make_shared<DifferentiableOperator>(
                  inputs, outputs, *fes.GetParMesh());

         const auto in = Inputs < Value<Solution>, Gradient<Solution>,
                    Gradient<Coords>, Weight > {};
         const auto out = Outputs<Gradient<Solution>> {};

         if (kappa == KappaType::GradientDependent)
         {
            typename NonlinearPoisson<DIM>::template
            ResidualBased<KappaType::GradientDependent> qf;
            dop->AddDomainIntegrator<LocalQFBackend, kernels>(
               qf, in, out, ir, all_domain_attr, derivatives);
         }
         else
         {
            typename NonlinearPoisson<DIM>::template
            ResidualBased<KappaType::SolutionDependent> qf;
            dop->AddDomainIntegrator<LocalQFBackend, kernels>(
               qf, in, out, ir, all_domain_attr, derivatives);
         }
      }
   }

   void SetEssentialAttributes(const Array<int> &ess_bdr)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   // The load stays outside the dFEM functional: the full potential is
   // Pi(u) - (b, u), whose first variation is the residual below.
   void SetLoad(const Vector &b) { load = b; }

   void Mult(const Vector &x, Vector &y) const override
   {
      MultiVector X{x, mesh_nodes_tdofs};
      MultiVector Y{y};
      // For a functional, GetDerivative returns the gradient action directly,
      // including the pointwise reverse seed for the summed energy.
      if (form == FormType::Energy) { gradient->Mult(X, Y); }
      else { dop->Mult(X, Y); }

      y -= load;
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator& GetGradient(const Vector &x) const override
   {
      jacobian = std::make_shared<JacobianOperator>(*this, x);
      return *jacobian;
   }

private:
   ParFiniteElementSpace &fes;
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   FormType form;
   Vector mesh_nodes_tdofs, load;
   Array<int> ess_tdofs;

   std::shared_ptr<DifferentiableOperator> dop;
   std::shared_ptr<DerivativeOperator> gradient;
   mutable std::shared_ptr<JacobianOperator> jacobian;
};

// BoomerAMG for the matrix-free tangent. HypreBoomerAMG needs a real
// HypreParMatrix, so on every SetOperator we let dFEM assemble the tangent and
// set up AMG. This works for both formulations, since GetDerivative and
// GetSecondDerivative both return a DerivativeOperator that can assemble itself.
template <int DIM>
class JacobianAMG : public Solver
{
   using JacobianOperator =
      typename NonlinearPoissonOperator<DIM>::JacobianOperator;

public:
   JacobianAMG() { amg.SetPrintLevel(0); }

   void SetOperator(const Operator &op) override
   {
      const auto *Jop = dynamic_cast<const JacobianOperator *>(&op);
      MFEM_VERIFY(Jop, "JacobianAMG requires a JacobianOperator");
      height = width = op.Height();

      delete A;
      A = nullptr;
      Jop->AssembleJacobian(A);
      amg.SetOperator(*A);
   }

   void Mult(const Vector &x, Vector &y) const override { amg.Mult(x, y); }

   ~JacobianAMG() { delete A; }

private:
   HypreParMatrix *A = nullptr;
   HypreBoomerAMG amg;
};

template <int DIM>
std::unique_ptr<Solver> MakePreconditioner(PreconditionerType type)
{
   switch (type)
   {
      case PreconditionerType::None:     return nullptr;
      case PreconditionerType::Diagonal: return
            std::make_unique<OperatorJacobiSmoother>();
      case PreconditionerType::AMG:      return std::make_unique<JacobianAMG<DIM>>();
      default:
         MFEM_ABORT("Unknown preconditioner type: " << static_cast<int>(type));
   }
   return nullptr;
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options
   int order = 1;
   int refinements = 3;
   const char *device_config = "cpu";
   const char *outfolder = "./Output";
   const char *kappa_name = "grad";
   const char *form_name = "energy";
   int prec_type = static_cast<int>(PreconditionerType::AMG);
   real_t krylov_tol = 1e-8;
   real_t source = 1.0;
   bool visualization = true;
   bool paraview = false;
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of uniform refinements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&kappa_name, "-k", "--kappa",
                  "Diffusion coefficient: grad = 1 + |grad u|^2, "
                  "sol = 1 + u^2.");
   args.AddOption(&form_name, "-f", "--form",
                  "dFEM formulation: energy (requires -k grad), residual.");
   args.AddOption(&prec_type, "-pc", "--preconditioner",
                  "Preconditioner: 0 = none, 1 = diagonal/Jacobi, 2 = AMG.");
   args.AddOption(&krylov_tol, "-tol", "--krylov-tol",
                  "Relative tolerance for the linear solver.");
   args.AddOption(&source, "-s", "--source", "Constant source term.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for ParaView DataCollection files.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.ParseCheck();

   const std::string kappa_str(kappa_name), form_str(form_name);
   MFEM_VERIFY(kappa_str == "grad" || kappa_str == "sol",
               "-k must be grad or sol");
   MFEM_VERIFY(form_str == "energy" || form_str == "residual",
               "-f must be energy or residual");

   const KappaType kappa = (kappa_str == "grad")
                           ? KappaType::GradientDependent
                           : KappaType::SolutionDependent;
   const FormType form = (form_str == "energy") ? FormType::Energy
                         : FormType::Residual;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   if (Mpi::Root())
   {
      mfem::out << "coefficient: "
                << (kappa == KappaType::GradientDependent ? "1 + |grad u|^2"
                    : "1 + u^2")
                << ", formulation: " << form_str
                << ", tangent: "
                << (kappa == KappaType::GradientDependent
                    ? "symmetric (CG)" : "non-symmetric (GMRES)")
                << std::endl;
   }

   // 4. Create the unit square/cube mesh and refine it
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   for (int l = 0; l < refinements; l++) { mesh.UniformRefinement(); }
   mesh.SetCurvature(order);

   // 5. Define a parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();

   // 6. Define a scalar H1 space on the parallel mesh. GlobalTrueVSize is
   //    collective, so every rank has to reach it.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   const HYPRE_BigInt global_dofs = fes.GlobalTrueVSize();
   if (Mpi::Root()) { mfem::out << "#dofs: " << global_dofs << std::endl; }

   // 7. Set up the integration rule. It has to resolve the nonlinear
   //    coefficient, not just the bilinear part, hence the extra order
   //    compared to a linear problem.
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order + 2);

   // 8. Create the nonlinear operator for the chosen coefficient and
   //    formulation
   NonlinearPoissonOperator<dim> poisson_op(fes, ir, form, kappa);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   poisson_op.SetEssentialAttributes(ess_bdr);

   // 9. Assemble the constant source term. The boundary data is homogeneous.
   ConstantCoefficient source_coeff(source);
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(source_coeff, &ir));
   b.Assemble();

   Vector B(fes.GetTrueVSize());
   b.ParallelAssemble(B);
   poisson_op.SetLoad(B);

   ParGridFunction u_gf(&fes);
   u_gf = 0.0;

   Vector U(fes.GetTrueVSize());
   u_gf.GetTrueDofs(U);

   // 10. Set up the linear solver used within Newton's method. The
   //     solution-dependent coefficient gives a non-symmetric tangent, so CG
   //     does not apply there.
   std::unique_ptr<IterativeSolver> krylov;
   if (kappa == KappaType::GradientDependent)
   {
      krylov = std::make_unique<CGSolver>(MPI_COMM_WORLD);
   }
   else
   {
      krylov = std::make_unique<GMRESSolver>(MPI_COMM_WORLD);
   }
   krylov->SetAbsTol(0.0);
   krylov->SetRelTol(krylov_tol);
   krylov->SetMaxIter(2000);
   krylov->SetPrintLevel(0);

   std::unique_ptr<Solver> pc =
      MakePreconditioner<dim>(static_cast<PreconditionerType>(prec_type));
   if (pc) { krylov->SetPreconditioner(*pc); }

   // 11. Set up the nonlinear solver (Newton)
   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(poisson_op);
   newton.SetSolver(*krylov);
   newton.SetAbsTol(0.0);
#ifdef MFEM_USE_SINGLE
   newton.SetRelTol(1e-6);
#else
   newton.SetRelTol(1e-10);
#endif
   newton.SetMaxIter(30);
   newton.SetPrintLevel(1);

   // 12. Solve the nonlinear system using Newton's method
   Vector zero;
   newton.Mult(zero, U);
   MFEM_VERIFY(newton.GetConverged(), "Newton did not converge");

   u_gf.Distribute(U);

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank()
               << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u_gf << std::flush;
   }

   // 14. Save the solution in parallel using ParaView format
   if (paraview)
   {
      ParaViewDataCollection pd("dfem-nonlinear-poisson-output", &pmesh);
      pd.SetPrefixPath(outfolder);
      pd.RegisterField("solution", &u_gf);
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

   return 0;
}
