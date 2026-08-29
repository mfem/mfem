// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
// Shared command-line, eigenfrequency, and reporting utilities for the
// frequency-domain cantilever example and manufactured-solution regression.

#ifndef MFEM_MTOP_FREQUENCY_DOMAIN_DRIVER_UTILS_HPP
#define MFEM_MTOP_FREQUENCY_DOMAIN_DRIVER_UTILS_HPP

#include "frequency_domain_elasticity_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

namespace mfem
{
namespace frequency_domain
{

/// Shared outer-solver, block-preconditioner, and H-inverse controls.
struct SolverOptions
{
   const char *linear_solver = "automatic";
   const char *preconditioner = "presb";
   const char *h_inverse = "lor-amg";
   const char *lor_ordering = "nodes";
   int h_amg_cycles = 1;
   real_t relative_tolerance = 1.0e-10;
   real_t absolute_tolerance = 0.0;
   int max_iterations = 500;
   int kdim = 50;
   int print_level = 0;
   real_t preconditioner_relative_tolerance = 1.0e-2;
   real_t preconditioner_absolute_tolerance = 0.0;
   int preconditioner_max_iterations = 50;
   int preconditioner_print_level = -1;

   void AddOptions(OptionsParser &args)
   {
      args.AddOption(&linear_solver, "-ls", "--linear-solver",
                     "Linear solver: automatic, gmres, fgmres, minres, mumps.");
      args.AddOption(&preconditioner, "-pc", "--preconditioner",
                     "Block preconditioner: presb or block-diagonal.");
      args.AddOption(&h_inverse, "-hi", "--h-inverse",
                     "H inverse: lor-amg, lor-cg-amg, or mumps.");
      args.AddOption(&lor_ordering, "-lo", "--lor-ordering",
                     "LOR vector ordering: nodes or vdim.");
      args.AddOption(&h_amg_cycles, "-hac", "--h-amg-cycles",
                     "Fixed AMG cycles per H-inverse application.");
      args.AddOption(&relative_tolerance, "-rtol", "--relative-tolerance",
                     "Outer relative tolerance.");
      args.AddOption(&absolute_tolerance, "-atol", "--absolute-tolerance",
                     "Outer absolute tolerance.");
      args.AddOption(&max_iterations, "-mi", "--max-iterations",
                     "Outer maximum iteration count.");
      args.AddOption(&kdim, "-kdim", "--kdim",
                     "GMRES or FGMRES restart dimension.");
      args.AddOption(&print_level, "-pl", "--print-level",
                     "Outer solver print level.");
      args.AddOption(&preconditioner_relative_tolerance, "-prtol",
                     "--preconditioner-relative-tolerance",
                     "Nested H-solve relative tolerance.");
      args.AddOption(&preconditioner_absolute_tolerance, "-patol",
                     "--preconditioner-absolute-tolerance",
                     "Nested H-solve absolute tolerance.");
      args.AddOption(&preconditioner_max_iterations, "-pmi",
                     "--preconditioner-max-iterations",
                     "Nested H-solve maximum iteration count.");
      args.AddOption(&preconditioner_print_level, "-ppl",
                     "--preconditioner-print-level",
                     "Nested H-solve and AMG print level.");
   }

   void Validate() const
   {
      const std::string ls(linear_solver);
      const std::string pc(preconditioner);
      const std::string hi(h_inverse);
      const std::string lo(lor_ordering);
      MFEM_VERIFY(ls == "automatic" || ls == "gmres" || ls == "fgmres" ||
                  ls == "minres" || ls == "mumps",
                  "Unknown linear solver.");
      MFEM_VERIFY(pc == "presb" || pc == "block-diagonal",
                  "Unknown block preconditioner.");
      MFEM_VERIFY(hi == "lor-amg" || hi == "lor-cg-amg" || hi == "mumps",
                  "Unknown H inverse.");
      MFEM_VERIFY(lo == "nodes" || lo == "vdim", "Unknown LOR ordering.");
      MFEM_VERIFY(h_amg_cycles > 0,
                  "H-inverse AMG cycle count must be positive.");
      MFEM_VERIFY(relative_tolerance >= 0.0 && absolute_tolerance >= 0.0 &&
                  max_iterations > 0 && kdim > 0,
                  "Invalid outer solver controls.");
      MFEM_VERIFY(preconditioner_relative_tolerance >= 0.0 &&
                  preconditioner_absolute_tolerance >= 0.0 &&
                  preconditioner_max_iterations > 0,
                  "Invalid H-inverse controls.");
      MFEM_VERIFY(ls != "minres" ||
                  (pc == "block-diagonal" && hi != "lor-cg-amg"),
                  "MINRES requires block-diagonal with a fixed H inverse.");
      MFEM_VERIFY(ls != "gmres" || hi != "lor-cg-amg",
                  "Variable nested H solves require FGMRES.");
   }

   void Apply(FrequencyDomainLinearElasticitySolver &solver) const
   {
      using FD = FrequencyDomainLinearElasticitySolver;
      const std::string ls(linear_solver);
      const std::string pc(preconditioner);
      const std::string hi(h_inverse);
      solver.SetLinearSolverType(
         ls == "gmres" ? FD::LinearSolverType::GMRES :
         ls == "fgmres" ? FD::LinearSolverType::FGMRES :
         ls == "minres" ? FD::LinearSolverType::MINRES :
         ls == "mumps" ? FD::LinearSolverType::MUMPS :
         FD::LinearSolverType::Automatic);
      solver.SetPreconditionerType(
         pc == "block-diagonal" ? FD::PreconditionerType::BlockDiagonal :
         FD::PreconditionerType::PRESB);
      solver.SetHInverseType(
         hi == "lor-cg-amg" ? FD::HInverseType::LORMonolithicCGAMG :
         hi == "mumps" ? FD::HInverseType::MUMPS :
         FD::HInverseType::LORMonolithicAMG);
      solver.SetLOROrdering(std::string(lor_ordering) == "vdim" ?
                            Ordering::byVDIM : Ordering::byNODES);
      solver.SetHInverseAMGCycles(h_amg_cycles);
      solver.SetRelTol(relative_tolerance);
      solver.SetAbsTol(absolute_tolerance);
      solver.SetMaxIter(max_iterations);
      solver.SetKDim(kdim);
      solver.SetPrintLevel(print_level);
      solver.SetPreconditionerRelTol(preconditioner_relative_tolerance);
      solver.SetPreconditionerAbsTol(preconditioner_absolute_tolerance);
      solver.SetPreconditionerMaxIter(preconditioner_max_iterations);
      solver.SetPreconditionerPrintLevel(preconditioner_print_level);
   }
};

/// Select and configure Rayleigh or independent isotropic damping.
struct DampingOptions
{
   const char *model = "rayleigh";
   real_t alpha = 0.02;
   real_t beta = 0.0;
   real_t mass = 0.02;
   real_t lambda = 0.0;
   real_t mu = 0.0;

   void AddOptions(OptionsParser &args)
   {
      args.AddOption(&model, "-dm", "--damping-model",
                     "Damping model: rayleigh or independent.");
      args.AddOption(&alpha, "-da", "--damping-alpha",
                     "Rayleigh mass-proportional coefficient.");
      args.AddOption(&beta, "-db", "--damping-beta",
                     "Rayleigh stiffness-proportional coefficient.");
      args.AddOption(&mass, "-dcm", "--mass-damping",
                     "Independent mass damping coefficient.");
      args.AddOption(&lambda, "-dcl", "--damping-lambda",
                     "Independent lambda-like damping coefficient.");
      args.AddOption(&mu, "-dcu", "--damping-mu",
                     "Independent mu-like damping coefficient.");
   }

   void Validate() const
   {
      const std::string name(model);
      MFEM_VERIFY(name == "rayleigh" || name == "independent",
                  "Damping model must be rayleigh or independent.");
      MFEM_VERIFY(alpha >= 0.0 && beta >= 0.0 && mass >= 0.0 &&
                  lambda >= 0.0 && mu >= 0.0,
                  "Damping coefficients must be nonnegative.");
   }

   void Apply(FrequencyDomainLinearElasticitySolver &solver) const
   {
      if (std::string(model) == "rayleigh")
      {
         solver.SetRayleighDamping(alpha, beta);
      }
      else
      {
         solver.SetDampingCoefficients(
            std::make_shared<ConstantCoefficient>(mass),
            std::make_shared<ConstantCoefficient>(lambda),
            std::make_shared<ConstantCoefficient>(mu));
      }
   }
};

/// Controls for the one-mode LOBPCG diagnostic solve.
struct EigenOptions
{
   real_t tolerance = 1.0e-8;
   int max_iterations = 200;
   int seed = 75;
   int print_level = 0;

   void AddOptions(OptionsParser &args)
   {
      args.AddOption(&tolerance, "-etol", "--eigen-tolerance",
                     "LOBPCG eigenvalue tolerance.");
      args.AddOption(&max_iterations, "-emi", "--eigen-max-iterations",
                     "LOBPCG maximum iteration count.");
      args.AddOption(&seed, "-eseed", "--eigen-seed",
                     "LOBPCG random seed.");
      args.AddOption(&print_level, "-epl", "--eigen-print-level",
                     "LOBPCG print level.");
   }

   void Validate() const
   {
      MFEM_VERIFY(tolerance > 0.0 && max_iterations > 0 && seed >= 0,
                  "Invalid LOBPCG controls.");
   }
};

/// First-mode spectral and damping measurements.
struct EigenDiagnostics
{
   real_t lambda1 = 0.0;
   real_t omega1 = 0.0;
   real_t modal_damping = 0.0;
   real_t damping_ratio = 0.0;
   double solve_time = 0.0;
};

/// Solve K phi=lambda M phi and evaluate the damping of its lowest mode.
inline EigenDiagnostics ComputeEigenDiagnostics(
   ParFiniteElementSpace &space, Coefficient &lambda, Coefficient &mu,
   Coefficient &density, const Array<int> &essential_boundary,
   const DampingOptions &damping, const EigenOptions &options)
{
   ParBilinearForm stiffness(&space);
   stiffness.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
   stiffness.Assemble();
   stiffness.EliminateEssentialBCDiag(essential_boundary, 1.0);
   stiffness.Finalize();
   std::unique_ptr<HypreParMatrix> K(stiffness.ParallelAssemble());

   ParBilinearForm mass(&space);
   mass.AddDomainIntegrator(new VectorMassIntegrator(density));
   mass.Assemble();
   mass.EliminateEssentialBCDiag(
      essential_boundary, std::numeric_limits<real_t>::min());
   mass.Finalize();
   std::unique_ptr<HypreParMatrix> M(mass.ParallelAssemble());

   HypreBoomerAMG amg(*K);
   amg.SetSystemsOptions(space.GetVDim(),
                         space.GetOrdering() == Ordering::byNODES);
   amg.SetPrintLevel(0);
   HypreLOBPCG lobpcg(space.GetComm());
   lobpcg.SetNumModes(1);
   lobpcg.SetRandomSeed(options.seed);
   lobpcg.SetPreconditioner(amg);
   lobpcg.SetMaxIter(options.max_iterations);
   lobpcg.SetTol(options.tolerance);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(options.print_level);
   lobpcg.SetMassMatrix(*M);
   lobpcg.SetOperator(*K);
   StopWatch timer;
   timer.Start();
   lobpcg.Solve();
   timer.Stop();

   Array<real_t> eigenvalues;
   lobpcg.GetEigenvalues(eigenvalues);
   MFEM_VERIFY(eigenvalues.Size() > 0 && eigenvalues[0] > 0.0,
               "LOBPCG did not return a positive lowest eigenvalue.");
   EigenDiagnostics result;
   result.lambda1 = eigenvalues[0];
   result.omega1 = std::sqrt(result.lambda1);
   result.solve_time = timer.RealTime();

   if (std::string(damping.model) == "rayleigh")
   {
      result.modal_damping = damping.alpha + damping.beta*result.lambda1;
   }
   else
   {
      ConstantCoefficient damping_mass(damping.mass);
      ConstantCoefficient damping_lambda(damping.lambda);
      ConstantCoefficient damping_mu(damping.mu);
      ParBilinearForm damping_form(&space);
      damping_form.AddDomainIntegrator(
         new ElasticityIntegrator(damping_lambda, damping_mu));
      damping_form.AddDomainIntegrator(
         new VectorMassIntegrator(damping_mass));
      damping_form.Assemble();
      damping_form.EliminateEssentialBCDiag(essential_boundary, 0.0);
      damping_form.Finalize();
      std::unique_ptr<HypreParMatrix> C(damping_form.ParallelAssemble());
      const HypreParVector &mode = lobpcg.GetEigenvector(0);
      Vector Mmode(mode.Size()), Cmode(mode.Size());
      M->Mult(mode, Mmode);
      C->Mult(mode, Cmode);
      const real_t local_products[2] = {mode*Mmode, mode*Cmode};
      real_t global_products[2] = {0.0, 0.0};
      MPI_Allreduce(local_products, global_products, 2,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, space.GetComm());
      const real_t denominator = global_products[0];
      MFEM_VERIFY(denominator > 0.0, "The first mode has zero mass norm.");
      result.modal_damping = global_products[1]/denominator;
   }
   result.damping_ratio = result.modal_damping/(2.0*result.omega1);
   return result;
}

/// Use an absolute frequency, or a positive multiple of omega1 when supplied.
inline real_t ResolveFrequency(const real_t frequency,
                               const real_t frequency_factor,
                               const EigenDiagnostics &eigen)
{
   return frequency_factor > 0.0 ? frequency_factor*eigen.omega1 : frequency;
}

/// Return omega*c1/|lambda1-omega^2|, or infinity at numerical resonance.
inline real_t LossFactor(const real_t omega,
                         const EigenDiagnostics &eigen)
{
   const real_t denominator = std::abs(eigen.lambda1 - omega*omega);
   const real_t scale = std::max(eigen.lambda1, omega*omega);
   if (denominator <= 100.0*std::numeric_limits<real_t>::epsilon()*scale)
   {
      return std::numeric_limits<real_t>::infinity();
   }
   return omega*eigen.modal_damping/denominator;
}

/// Return the first-mode value lambda1-omega^2+omega*c1 for H=W+T.
inline real_t HIndicator(const real_t omega,
                         const EigenDiagnostics &eigen)
{
   return eigen.lambda1 - omega*omega + omega*eigen.modal_damping;
}

/// Reduce several wall times with one collective operation.
inline void GlobalMaximum(const double *values, double *maximum,
                          const int count, const MPI_Comm communicator)
{
   MFEM_VERIFY(count >= 0, "The timing array length must be nonnegative.");
   MPI_Allreduce(values, maximum, count, MPI_DOUBLE, MPI_MAX, communicator);
}

/// Return one maximum wall time across an MPI communicator.
inline double GlobalMaximum(const double value, const MPI_Comm communicator)
{
   double maximum = 0.0;
   GlobalMaximum(&value, &maximum, 1, communicator);
   return maximum;
}

inline const char *LinearSolverName(
   const FrequencyDomainLinearElasticitySolver::LinearSolverType type)
{
   using Type = FrequencyDomainLinearElasticitySolver::LinearSolverType;
   return type == Type::GMRES ? "gmres" :
          type == Type::FGMRES ? "fgmres" :
          type == Type::MINRES ? "minres" :
          type == Type::MUMPS ? "mumps" : "automatic";
}

inline const char *PreconditionerName(
   const FrequencyDomainLinearElasticitySolver::PreconditionerType type)
{
   using Type = FrequencyDomainLinearElasticitySolver::PreconditionerType;
   return type == Type::BlockDiagonal ? "block-diagonal" : "presb";
}

inline const char *HInverseName(
   const FrequencyDomainLinearElasticitySolver::HInverseType type)
{
   using Type = FrequencyDomainLinearElasticitySolver::HInverseType;
   return type == Type::LORMonolithicCGAMG ? "lor-cg-amg" :
          type == Type::MUMPS ? "mumps" : "lor-amg";
}

inline const char *LOROrderingName(const Ordering::Type ordering)
{
   return ordering == Ordering::byVDIM ? "vdim" : "nodes";
}

} // namespace frequency_domain
} // namespace mfem

#endif
