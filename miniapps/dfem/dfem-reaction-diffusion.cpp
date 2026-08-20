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
//              ------------------------------------------------
//              Nonlinear Reaction-Diffusion with dFEM (two
//              outputs from a single quadrature point function)
//              ------------------------------------------------
//
// Compile with: make dfem-reaction-diffusion
//
// Sample runs:  mpirun -np 4 dfem-reaction-diffusion -b 1 -no-vis
//               mpirun -np 4 dfem-reaction-diffusion -b 1 -o 2 -r 4 -pc 2 -no-vis
//               mpirun -np 4 dfem-reaction-diffusion -b 10 -a 0 -no-vis
//               mpirun -np 4 dfem-reaction-diffusion -b 0 -check -no-vis
//
// Description:  This miniapp solves the nonlinear reaction-diffusion problem
//
//                   -div( kappa grad u ) + alpha u + beta u^3 = f   in Omega
//                                                          u = 0   on dOmega
//
//               on the unit square (set the compile-time constant dim to 3 for
//               the unit cube). 
//
//               This miniapp demonstrates the use of dFEM in multiple-output mode. 
//               The q-function writes two outputs onto the same test field, e.g.:
//
//                   Outputs<Gradient<Solution>, Value<Solution>>
//
//               dFEM integrates each output against its own test basis
//               operation and accumulates all of them into a single output
//               vector,
//
//                   R_i = int (kappa grad u) . grad phi_i dx
//                       + int (alpha u + beta u^3 - f) phi_i dx
//
//               so the diffusion and the reaction term share one kernel and one
//               restriction of u. 
//

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

// Space dimension. The q-function and the operator are templated on it, so it
// is fixed at compile time; set it to 3 to run the same problem on the unit
// cube.
constexpr int dim = 2;

// Field IDs used by the dFEM integrator.
static constexpr int Solution = 0;
static constexpr int Coords = 1;

enum class PreconditionerType { None, Diagonal, AMG };

// ----------------------------------------------------------------------------
// Pointwise description of the problem.
// ----------------------------------------------------------------------------

/// Quadrature point kernel for
///
///   R(u; v) = int kappa grad u . grad v dx
///           + int (alpha u + beta u^3 - f) v dx
///
/// The two integrands leave through two different outputs: @a dvdx is paired
/// with Gradient<Solution> and @a v with Value<Solution>.
/// However, both are attached to the same fieldID, so dFEM sums their contributions
/// into one residual vector.
template <int DIM>
struct ReactionDiffusion
{
   real_t kappa = 1.0;
   real_t alpha = 1.0;
   real_t beta = 1.0;
   real_t source = 1.0;

   MFEM_HOST_DEVICE inline
   auto operator()(const dscalar_t &u,
                   const tensor<dscalar_t, DIM> &dudxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<dscalar_t, DIM> &dvdx,
                   dscalar_t &v) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const real_t dxw = det(J) * w;

      dvdx = kappa * dudx * transpose(invJ) * dxw;
      v = (alpha * u + beta * u * u * u - source) * dxw;
   }
};

// ----------------------------------------------------------------------------
// The residual operator. A single DifferentiableOperator holds both terms, so
// the Newton tangent below is the derivative of the whole q-function.
// ----------------------------------------------------------------------------
template <int DIM>
class ReactionDiffusionOperator : public Operator
{
public:
   /// Matrix-free Newton tangent K + (alpha + 3 beta u^2) M. Essential
   /// directions are removed before the apply and restored as identity rows
   /// afterwards.
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(const ReactionDiffusionOperator &oper,
                       const Vector &state) :
         Operator(oper.Height()), oper(oper), z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs};
         // Differentiating the q-function linearizes both outputs at once,
         // so no term has to be added by hand here.
         tangent = oper.dop->GetDerivative(Solution, X);
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
      const ReactionDiffusionOperator &oper;
      mutable Vector z;
      std::shared_ptr<DerivativeOperator> tangent;
   };

   ReactionDiffusionOperator(ParFiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             real_t kappa, real_t alpha, real_t beta,
                             real_t source) :
      Operator(fes.GetTrueVSize()),
      fes(fes)
   {
      MFEM_VERIFY(kappa > 0.0, "the diffusion coefficient has to be positive");
      MFEM_VERIFY(alpha >= 0.0 && beta >= 0.0,
                  "negative reaction coefficients make the tangent indefinite; "
                  "this miniapp assumes -a and -b are non-negative so that the "
                  "problem stays convex and CG applies");

      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Solution, &fes}, {Coords, mesh_nodes_fes}
      };
      // One output field, even though the q-function writes two outputs onto
      // it: the number of FieldOperators does not set the number of blocks.
      const std::vector<FieldDescriptor> outputs = {{Solution, &fes}};

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      dop = std::make_shared<DifferentiableOperator>(
               inputs, outputs, *fes.GetParMesh());

      ReactionDiffusion<DIM> qf{kappa, alpha, beta, source};
      dop->AddDomainIntegrator<LocalQFBackend>(
         qf,
         Inputs < Value<Solution>, Gradient<Solution>,
         Gradient<Coords>, Weight > {},
         Outputs<Gradient<Solution>, Value<Solution>> {},
         ir, all_domain_attr, Derivatives<Solution> {});
   }

   void SetEssentialAttributes(const Array<int> &ess_bdr)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MultiVector X{x, mesh_nodes_tdofs};
      MultiVector Y{y};
      // Both terms and the load are inside the q-function, so there is nothing
      // to add to the residual here.
      dop->Mult(X, Y);
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
   Vector mesh_nodes_tdofs;
   Array<int> ess_tdofs;

   std::shared_ptr<DifferentiableOperator> dop;
   mutable std::shared_ptr<JacobianOperator> jacobian;
};

// BoomerAMG for the matrix-free tangent. HypreBoomerAMG needs a real
// HypreParMatrix, so on every SetOperator we let dFEM assemble the tangent and
// set up AMG. The assembled matrix carries both outputs, so the reaction term
// is present in the preconditioner as well.
template <int DIM>
class JacobianAMG : public Solver
{
   using JacobianOperator =
      typename ReactionDiffusionOperator<DIM>::JacobianOperator;

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
// Verification of the two-output path against a classical assembly.
// ----------------------------------------------------------------------------

/// Largest absolute entry of @a v across all ranks.
static real_t GlobalNormlinf(MPI_Comm comm, const Vector &v)
{
   real_t local = v.Normlinf(), global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
}

/// With beta = 0 and f = 0 the operator is exactly
/// DiffusionIntegrator(kappa) + MassIntegrator(alpha), which a single
/// ParBilinearForm can build. Comparing against it checks that the Gradient and
/// the Value output really do accumulate into the same residual vector: drop
/// either FieldOperator from the Outputs tuple and these comparisons fail.
///
/// The operator used here is built without essential attributes, so no
/// elimination happens on either side and the raw action and matrix are
/// compared.
template <int DIM>
static void CheckLinearLimit(ParFiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             real_t kappa, real_t alpha)
{
   MPI_Comm comm = fes.GetComm();
   const int tvsize = fes.GetTrueVSize();

   ReactionDiffusionOperator<DIM> lin_op(fes, ir, kappa, alpha, 0.0, 0.0);

   ConstantCoefficient kappa_coeff(kappa), alpha_coeff(alpha);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(kappa_coeff, &ir));
   a.AddDomainIntegrator(new MassIntegrator(alpha_coeff, &ir));
   a.Assemble();
   a.Finalize();

   Vector U(tvsize);
   U.Randomize(0x9e3779b9);
   ParGridFunction u_gf(&fes), y_gf(&fes);
   u_gf.SetFromTrueDofs(U);

   // 1. Action of the two-output q-function against the two integrators.
   Vector Y(tvsize), Yref(tvsize);
   lin_op.Mult(U, Y);
   a.Mult(u_gf, y_gf);
   fes.GetProlongationMatrix()->MultTranspose(y_gf, Yref);

   Vector diff(Y);
   diff -= Yref;
   const real_t action_err = GlobalNormlinf(comm, diff);

   // 2. Assembled tangent and its diagonal. Both assembly paths walk the
   //    quadrature point cache written by DerivativeSetup, which is laid out
   //    over every output FieldOperator, so both have to pick up the reaction
   //    term as well as the diffusion term. The matrix comparison is done
   //    through matrix-vector products, so that it does not depend on the
   //    sparsity pattern or the ordering of the two assemblies.
   const auto &J = lin_op.GetGradient(U);
   const auto *Jop =
      dynamic_cast<const typename ReactionDiffusionOperator<DIM>::JacobianOperator *>
      (&J);
   MFEM_VERIFY(Jop, "expected a JacobianOperator");

   HypreParMatrix *A = nullptr;
   Jop->AssembleJacobian(A);
   HypreParMatrix *Aref = a.ParallelAssemble();

   real_t matrix_err = 0.0;
   Vector v(tvsize), w1(tvsize), w2(tvsize);
   for (int k = 0; k < 3; k++)
   {
      v.Randomize(0x01000193 + k);
      A->Mult(v, w1);
      Aref->Mult(v, w2);
      w1 -= w2;
      matrix_err = std::max(matrix_err, GlobalNormlinf(comm, w1));
   }

   Vector d1(tvsize), d2(tvsize);
   Jop->AssembleDiagonal(d1);
   Aref->GetDiag(d2);
   d1 -= d2;
   const real_t diag_err = GlobalNormlinf(comm, d1);

   delete A;
   delete Aref;

   if (Mpi::Root())
   {
      mfem::out << "linear limit (beta = 0, f = 0) vs "
                << "DiffusionIntegrator + MassIntegrator:\n"
                << "  action   max |dFEM - reference| = " << action_err << '\n'
                << "  tangent  max |dFEM - reference| = " << matrix_err << '\n'
                << "  diagonal max |dFEM - reference| = " << diag_err
                << std::endl;
   }

   // Loose enough for the accumulated round-off of two different assembly
   // paths, tight enough that a missing output term cannot slip through.
   const real_t tol = 1e-10 * std::max(1.0_r, kappa + alpha);
   MFEM_VERIFY(action_err < tol, "two-output action does not match the "
               "DiffusionIntegrator + MassIntegrator reference");
   MFEM_VERIFY(matrix_err < tol, "assembled two-output tangent does not match "
               "the DiffusionIntegrator + MassIntegrator reference");
   MFEM_VERIFY(diag_err < tol, "diagonal of the two-output tangent does not "
               "match the DiffusionIntegrator + MassIntegrator reference");
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
   real_t kappa = 1.0;
   real_t alpha = 1.0;
   real_t beta = 1.0;
   real_t source = 1.0;
   int prec_type = static_cast<int>(PreconditionerType::AMG);
   real_t krylov_tol = 1e-8;
   bool check = false;
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
   args.AddOption(&kappa, "-k", "--kappa", "Diffusion coefficient, positive.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Linear reaction coefficient, non-negative.");
   args.AddOption(&beta, "-b", "--beta",
                  "Cubic reaction coefficient, non-negative. Zero gives the "
                  "linear screened Poisson problem.");
   args.AddOption(&source, "-s", "--source", "Constant source term.");
   args.AddOption(&prec_type, "-pc", "--preconditioner",
                  "Preconditioner: 0 = none, 1 = diagonal/Jacobi, 2 = AMG.");
   args.AddOption(&krylov_tol, "-tol", "--krylov-tol",
                  "Relative tolerance for the linear solver.");
   args.AddOption(&check, "-check", "--check-linear", "-no-check",
                  "--no-check-linear",
                  "Compare the beta = 0 limit against a ParBilinearForm "
                  "carrying DiffusionIntegrator and MassIntegrator.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   if (Mpi::Root())
   {
      mfem::out << "operator: -div(" << kappa << " grad u) + " << alpha
                << " u + " << beta << " u^3 = " << source
                << ", outputs: Gradient<Solution>, Value<Solution>"
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

   // 7. Set up the integration rule. It has to resolve the cubic reaction
   //    term, not just the bilinear part.
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order + 2);

   // 8. Optionally verify the two-output path against a classical assembly.
   //    This uses its own operator, so it is independent of -b and -s.
   if (check) { CheckLinearLimit<dim>(fes, ir, kappa, alpha); }

   // 9. Create the nonlinear operator
   ReactionDiffusionOperator<dim> rd_op(fes, ir, kappa, alpha, beta, source);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   rd_op.SetEssentialAttributes(ess_bdr);

   ParGridFunction u_gf(&fes);
   u_gf = 0.0;

   Vector U(fes.GetTrueVSize());
   u_gf.GetTrueDofs(U);

   // 10. Set up the linear solver used within Newton's method. For
   //     non-negative alpha and beta the tangent is symmetric positive
   //     definite, so CG works.
   CGSolver krylov(MPI_COMM_WORLD);
   krylov.SetAbsTol(0.0);
   krylov.SetRelTol(krylov_tol);
   krylov.SetMaxIter(2000);
   krylov.SetPrintLevel(0);

   std::unique_ptr<Solver> pc =
      MakePreconditioner<dim>(static_cast<PreconditionerType>(prec_type));
   if (pc) { krylov.SetPreconditioner(*pc); }

   // 11. Set up the nonlinear solver (Newton)
   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(rd_op);
   newton.SetSolver(krylov);
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
      ParaViewDataCollection pd("dfem-reaction-diffusion-output", &pmesh);
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
