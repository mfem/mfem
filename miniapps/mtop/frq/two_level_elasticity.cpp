// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
//
/** @file
    @brief Compare a modal two-level method with LOR-AMG for a static beam.

    This MPI example solves the linear-elasticity problem

        K u = f

    on a two- or three-dimensional Cartesian cantilever. The x-min end is
    clamped and a constant traction is applied at x-max. Both the high-order
    stiffness K and vector mass M use partial assembly.

    The coarse basis consists of the requested lowest modes of

        K_f phi_i = lambda_i M_f phi_i,

    where the subscript f denotes restriction to unconstrained true dofs.
    FreeDofOperator lets HypreLOBPCG apply the PA operators without including
    essential unknowns in its distributed vectors. A LOR-AMG cycle accelerates
    the eigensolve. The modes are expanded to full true vectors, explicitly
    zero on the clamp, mass-normalized, and stored in TwoLevelPreconditioner.

    Smoothing can be applied before the coarse correction, after it, or on
    both sides. The symmetric two-sided cycle is solved with PCG; the
    nonsymmetric one-sided cycles are solved with GMRES. The smoother is either
    Jacobi based on an element L1 or scaled-L2 row-norm diagonal, or a symmetric
    LOR-AMG V-cycle. With smoothing disabled, the example instead solves the
    compatible deflated system

        (A - A Q A) x_hat = (I - A Q) f,
        u = Q f + (I - Q A) x_hat,

    where Q = Z (Z^T A Z)^dagger Z^T. Every run also solves the original PA
    system with LOR-AMG preconditioning and reports accuracy and timings. Use
    --gmres to select GMRES for all static solves instead of CG wherever the
    latter is permitted.

    ParaView output contains both displacements, their difference, and up to
    the first ten modes. For example:

        mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -sn l1
        mpirun -np 4 ./two_level_elasticity -st lor-amg
        mpirun -np 4 ./two_level_elasticity -st lor-amg -gmres
        mpirun -np 4 ./two_level_elasticity -st l1 -sp pre
        mpirun -np 4 ./two_level_elasticity -dim 2 -nm 10 -no-sm

    Use --help for the complete option list. The target requires MPI, HYPRE,
    double precision, and LAPACK.
*/

#include "frequency_domain_preconditioners.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/** Restrict a full PA true-dof operator to unconstrained true dofs.

    Given the injection E from free to full local true vectors, Mult applies
    E^T A E. The wrapper holds a non-owning reference to @a op, copies the
    free-dof map, and owns mutable full-size work vectors. Concurrent calls on
    one instance are therefore not supported.
*/
class FreeDofOperator : public Operator
{
public:
   FreeDofOperator(const Operator &op, const Array<int> &free_tdofs)
      : Operator(free_tdofs.Size()), op_(op), free_tdofs_(free_tdofs),
        full_input_(op.Width()), full_output_(op.Height())
   {
      MFEM_VERIFY(op.Height() == op.Width(),
                  "The restricted operator must be square.");
   }

   MemoryClass GetMemoryClass() const override
   {
      return op_.GetMemoryClass();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(),
                  "Restricted operator input size mismatch.");
      full_input_ = 0.0;
      full_input_.SetSubVector(free_tdofs_, x);
      op_.Mult(full_input_, full_output_);
      full_output_.GetSubVector(free_tdofs_, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Height(),
                  "Restricted transpose input size mismatch.");
      full_input_ = 0.0;
      full_input_.SetSubVector(free_tdofs_, x);
      op_.MultTranspose(full_input_, full_output_);
      full_output_.GetSubVector(free_tdofs_, y);
   }

private:
   const Operator &op_;
   Array<int> free_tdofs_;
   mutable Vector full_input_;
   mutable Vector full_output_;
};

/** Apply a full true-dof solver to a vector containing only free true dofs.

    This is the solver analogue of FreeDofOperator: expand a reduced residual,
    apply the non-owning full-space solver, and restrict its correction. The
    wrapped solver must outlive this adapter and remains configured for the
    full true-dof matrix.
*/
class FreeDofSolver : public Solver
{
public:
   FreeDofSolver(Solver &solver, const Array<int> &free_tdofs)
      : Solver(free_tdofs.Size()), solver_(solver), free_tdofs_(free_tdofs),
        full_input_(solver.Width()), full_output_(solver.Height())
   {
      MFEM_VERIFY(solver.Height() == solver.Width(),
                  "The restricted solver must be square.");
   }

   MemoryClass GetMemoryClass() const override
   {
      return solver_.GetMemoryClass();
   }

   void SetOperator(const Operator &op) override
   {
      MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                  "Restricted solver operator size mismatch.");
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(),
                  "Restricted solver input size mismatch.");
      full_input_ = 0.0;
      full_input_.SetSubVector(free_tdofs_, x);
      solver_.Mult(full_input_, full_output_);
      full_output_.GetSubVector(free_tdofs_, y);
   }

private:
   Solver &solver_;
   Array<int> free_tdofs_;
   mutable Vector full_input_;
   mutable Vector full_output_;
};

/** Expose a deliberately symmetric operator interface.

    HypreBoomerAMG does not implement MultTranspose(). For the LOR-AMG
    smoother this adapter defines the transpose action to be the forward
    action. This is valid only because ConfigureAMG selects the same positive
    L1-Jacobi relaxation in both directions of a Galerkin V-cycle.
*/
class SymmetricOperatorAdapter : public Operator
{
public:
   explicit SymmetricOperatorAdapter(const Operator &op)
      : Operator(op.Height(), op.Width()), op_(op)
   {
      MFEM_VERIFY(op.Height() == op.Width(),
                  "The symmetric adapter requires a square operator.");
   }

   MemoryClass GetMemoryClass() const override
   {
      return op_.GetMemoryClass();
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      op_.Mult(x, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      op_.Mult(x, y);
   }

private:
   const Operator &op_;
};

/// Expose the symmetric deflated operator A-AQA to the complementary CG solve.
class DeflatedSystemOperator : public Operator
{
public:
   explicit DeflatedSystemOperator(const TwoLevelPreconditioner &two_level)
      : Operator(two_level.Height()), two_level_(two_level) { }

   void Mult(const Vector &x, Vector &y) const override
   {
      two_level_.MultDeflatedOperator(x, y);
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      // The example uses this adapter only for symmetric elasticity.
      two_level_.MultDeflatedOperator(x, y);
   }

private:
   const TwoLevelPreconditioner &two_level_;
};

struct SolveResult
{
   Vector solution;
   int iterations = 0;
   bool converged = false;
   real_t relative_residual = 0.0;
   double solve_time = 0.0;
};

/// Euclidean norm of a distributed true-dof vector.
real_t GlobalNorm(MPI_Comm comm, const Vector &vector)
{
   return std::sqrt(std::max(real_t(0.0),
                             InnerProduct(comm, vector, vector)));
}

/// Maximum wall-clock measurement over all ranks.
double GlobalMaximum(MPI_Comm comm, const double local_value)
{
   double global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, comm);
   return global_value;
}

/// Configure one LOR-AMG V-cycle for byNODES elasticity vectors.
void ConfigureAMG(HypreBoomerAMG &amg, const int dimension,
                  const int print_level)
{
   amg.SetSystemsOptions(dimension, true);
   amg.SetRelaxType(18);
   amg.SetTol(0.0);
   amg.SetMaxIter(1);
   amg.SetPrintLevel(print_level);
}

/// Assemble a conservative element-row-norm diagonal without assembling K.
void AssembleRowNormDiagonal(ParBilinearForm &stiffness,
                             const Array<int> &essential_tdofs,
                             const bool use_l2, Vector &true_diagonal)
{
   ParFiniteElementSpace &space = *stiffness.ParFESpace();
   Vector local_diagonal(space.GetVSize());
   local_diagonal = 0.0;

   Array<int> vdofs;
   DenseMatrix element_matrix;
   for (int element = 0; element < space.GetNE(); ++element)
   {
      stiffness.ComputeElementMatrix(element, element_matrix);
      space.GetElementVDofs(element, vdofs);
      MFEM_VERIFY(element_matrix.Height() == vdofs.Size(),
                  "Element matrix and vector-dof sizes do not match.");

      const real_t l2_scale = std::sqrt(real_t(element_matrix.Width()));
      for (int i = 0; i < element_matrix.Height(); ++i)
      {
         real_t row_value = 0.0;
         for (int j = 0; j < element_matrix.Width(); ++j)
         {
            const real_t entry = element_matrix(i, j);
            row_value += use_l2 ? entry*entry : std::abs(entry);
         }
         if (use_l2) { row_value = l2_scale*std::sqrt(row_value); }

         const int signed_dof = vdofs[i];
         const int dof = signed_dof >= 0 ? signed_dof : -1-signed_dof;
         local_diagonal(dof) += row_value;
      }
   }

   const Operator *prolongation = space.GetProlongationMatrix();
   MFEM_VERIFY(prolongation, "Parallel true-dof prolongation is unavailable.");
   true_diagonal.SetSize(space.GetTrueVSize());
   true_diagonal.UseDevice(true);
   prolongation->MultTranspose(local_diagonal, true_diagonal);
   true_diagonal.SetSubVector(essential_tdofs, 1.0);
   for (int i = 0; i < true_diagonal.Size(); ++i)
   {
      MFEM_VERIFY(true_diagonal(i) > 0.0,
                  "The smoother diagonal must be positive.");
   }
}

/// Run zero-initial-guess CG and collect globally comparable timing data.
SolveResult RunCG(MPI_Comm comm, const Operator &op, const Vector &rhs,
                  Solver *preconditioner, const real_t relative_tolerance,
                  const real_t absolute_tolerance, const int max_iterations,
                  const int print_level)
{
   CGSolver solver(comm);
   solver.SetRelTol(relative_tolerance);
   solver.SetAbsTol(absolute_tolerance);
   solver.SetMaxIter(max_iterations);
   solver.SetPrintLevel(print_level);
   solver.SetOperator(op);
   // Set the auxiliary preconditioner after the high-order operator so the
   // iterative solver does not replace its LOR or coarse operator.
   if (preconditioner) { solver.SetPreconditioner(*preconditioner); }

   SolveResult result;
   result.solution.SetSize(op.Width());
   result.solution.UseDevice(true);
   result.solution = 0.0;
   StopWatch timer;
   timer.Start();
   solver.Mult(rhs, result.solution);
   timer.Stop();
   result.solve_time = GlobalMaximum(comm, timer.RealTime());
   result.iterations = solver.GetNumIterations();
   result.converged = solver.GetConverged();
   return result;
}

/// Run zero-initial-guess GMRES and collect globally comparable timing data.
SolveResult RunGMRES(MPI_Comm comm, const Operator &op, const Vector &rhs,
                     Solver *preconditioner,
                     const real_t relative_tolerance,
                     const real_t absolute_tolerance,
                     const int max_iterations, const int print_level)
{
   GMRESSolver solver(comm);
   solver.SetRelTol(relative_tolerance);
   solver.SetAbsTol(absolute_tolerance);
   solver.SetMaxIter(max_iterations);
   solver.SetPrintLevel(print_level);
   solver.SetOperator(op);
   if (preconditioner) { solver.SetPreconditioner(*preconditioner); }

   SolveResult result;
   result.solution.SetSize(op.Width());
   result.solution.UseDevice(true);
   result.solution = 0.0;
   StopWatch timer;
   timer.Start();
   solver.Mult(rhs, result.solution);
   timer.Stop();
   result.solve_time = GlobalMaximum(comm, timer.RealTime());
   result.iterations = solver.GetNumIterations();
   result.converged = solver.GetConverged();
   return result;
}

/// Compute ||A x-b||_2/||b||_2 using the original distributed operator.
real_t ComputeRelativeResidual(MPI_Comm comm, const Operator &op,
                               const Vector &rhs, const Vector &solution)
{
   Vector residual(op.Height());
   op.Mult(solution, residual);
   residual -= rhs;
   const real_t rhs_norm = GlobalNorm(comm, rhs);
   return GlobalNorm(comm, residual)/(rhs_norm > 0.0 ? rhs_norm : 1.0);
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

#if !defined(MFEM_USE_LAPACK)
   if (Mpi::Root())
   {
      mfem::err << "two_level_elasticity requires MFEM_USE_LAPACK.\n";
   }
   return EXIT_FAILURE;
#else
   // 1. Parse geometry, material, eigensolver, linear-solver, and output
   // controls. Defaults define a moderately resolved 2D cantilever.
   const char *device_configuration = "cpu";
   const char *smoother_norm_name = "l1";
   const char *smoother_type_name = "l1";
   const char *smoother_placement_name = "both";
   const char *output_prefix = "ParaView";
   const char *csv_path = "";
   bool use_smoother = true;
   bool use_gmres = false;
   bool visualization = true;
   int dimension = 2;
   int nx = 24;
   int ny = 6;
   int nz = 6;
   int order = 2;
   int serial_refinements = 0;
   int parallel_refinements = 0;
   int load_component = -1;
   int num_modes = 10;
   int eigen_max_iterations = 200;
   int eigen_seed = 75;
   int eigen_print_level = 0;
   int max_iterations = 500;
   int print_level = 0;
   real_t length = 4.0;
   real_t height = 1.0;
   real_t width = 1.0;
   real_t lame_lambda = 2.3;
   real_t lame_mu = 1.7;
   real_t density = 0.9;
   real_t load_amplitude = -1.0;
   real_t eigen_tolerance = 1.0e-8;
   real_t relative_tolerance = 1.0e-10;
   real_t absolute_tolerance = 1.0e-14;

   OptionsParser args(argc, argv);
   args.AddOption(&dimension, "-dim", "--dimension",
                  "Spatial dimension: 2 or 3.");
   args.AddOption(&device_configuration, "-d", "--device",
                  "MFEM device configuration.");
   args.AddOption(&nx, "-nx", "--x-elements",
                  "Number of elements along the beam.");
   args.AddOption(&ny, "-ny", "--y-elements",
                  "Number of elements through the beam height.");
   args.AddOption(&nz, "-nz", "--z-elements",
                  "Number of elements through the 3D beam width.");
   args.AddOption(&order, "-o", "--order", "H1 polynomial degree.");
   args.AddOption(&serial_refinements, "-rs", "--serial-refinements",
                  "Number of serial refinements.");
   args.AddOption(&parallel_refinements, "-rp", "--parallel-refinements",
                  "Number of parallel refinements.");
   args.AddOption(&length, "-lx", "--length", "Beam length.");
   args.AddOption(&height, "-ly", "--height", "Beam height.");
   args.AddOption(&width, "-lz", "--width", "3D beam width.");
   args.AddOption(&lame_lambda, "-la", "--lambda",
                  "First Lame coefficient.");
   args.AddOption(&lame_mu, "-mu", "--mu", "Shear modulus.");
   args.AddOption(&density, "-rho", "--density", "Mass density.");
   args.AddOption(&load_component, "-c", "--component",
                  "Zero-based traction component; -1 selects the last.");
   args.AddOption(&load_amplitude, "-a", "--amplitude",
                  "Constant free-end traction component.");
   args.AddOption(&num_modes, "-nm", "--num-modes",
                  "Number of lowest eigenmodes in the coarse space.");
   args.AddOption(&use_smoother, "-sm", "--smoother", "-no-sm",
                  "--no-smoother", "Legacy smoother enable/disable alias.");
   args.AddOption(&smoother_norm_name, "-sn", "--smoother-norm",
                  "Legacy diagonal smoother selection: l1 or l2.");
   args.AddOption(&smoother_type_name, "-st", "--smoother-type",
                  "Two-level smoother: none, l1, l2, or lor-amg.");
   args.AddOption(&smoother_placement_name, "-sp", "--smoother-placement",
                  "Smoother placement: pre, post, or both.");
   args.AddOption(&use_gmres, "-gmres", "--gmres", "-cg", "--cg",
                  "Use GMRES instead of CG for the static solves.");
   args.AddOption(&eigen_tolerance, "-etol", "--eigen-tolerance",
                  "LOBPCG relative tolerance.");
   args.AddOption(&eigen_max_iterations, "-emi", "--eigen-max-iterations",
                  "LOBPCG maximum iteration count.");
   args.AddOption(&eigen_seed, "-eseed", "--eigen-seed",
                  "LOBPCG random seed.");
   args.AddOption(&eigen_print_level, "-epl", "--eigen-print-level",
                  "LOBPCG print level.");
   args.AddOption(&relative_tolerance, "-rtol", "--relative-tolerance",
                  "Krylov-solver relative tolerance.");
   args.AddOption(&absolute_tolerance, "-atol", "--absolute-tolerance",
                  "Krylov-solver absolute tolerance.");
   args.AddOption(&max_iterations, "-mi", "--max-iterations",
                  "Krylov-solver maximum iteration count.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Krylov-solver and AMG print level.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView output.");
   args.AddOption(&output_prefix, "-out", "--output-prefix",
                  "ParaView output directory.");
   args.AddOption(&csv_path, "-csv", "--csv",
                  "Optional performance CSV output file.");
   args.ParseCheck();

   const std::string smoother_norm(smoother_norm_name);
   std::string smoother_type(smoother_type_name);
   const std::string smoother_placement(smoother_placement_name);
   // Preserve the original flags. The new --smoother-type option should not
   // be combined with these compatibility aliases.
   if (!use_smoother) { smoother_type = "none"; }
   else if (smoother_type == "l1") { smoother_type = smoother_norm; }
   MFEM_VERIFY(dimension == 2 || dimension == 3,
               "The spatial dimension must be 2 or 3.");
   MFEM_VERIFY(nx > 0 && ny > 0 && (dimension == 2 || nz > 0),
               "Mesh element counts must be positive.");
   MFEM_VERIFY(order > 0 && serial_refinements >= 0 &&
               parallel_refinements >= 0,
               "Order and refinement counts are invalid.");
   MFEM_VERIFY(length > 0.0 && height > 0.0 &&
               (dimension == 2 || width > 0.0),
               "Beam dimensions must be positive.");
   MFEM_VERIFY(lame_lambda > 0.0 && lame_mu > 0.0 && density > 0.0,
               "Material parameters must be positive.");
   MFEM_VERIFY(num_modes > 0, "The number of modes must be positive.");
   MFEM_VERIFY(smoother_norm == "l1" || smoother_norm == "l2",
               "The legacy smoother norm must be l1 or l2.");
   MFEM_VERIFY(smoother_type == "none" || smoother_type == "l1" ||
               smoother_type == "l2" || smoother_type == "lor-amg",
               "The smoother type must be none, l1, l2, or lor-amg.");
   MFEM_VERIFY(smoother_placement == "pre" ||
               smoother_placement == "post" ||
               smoother_placement == "both",
               "The smoother placement must be pre, post, or both.");
   MFEM_VERIFY(eigen_tolerance > 0.0 && eigen_max_iterations > 0 &&
               eigen_seed >= 0, "Invalid LOBPCG controls.");
   MFEM_VERIFY(relative_tolerance > 0.0 && absolute_tolerance >= 0.0 &&
               max_iterations > 0, "Invalid Krylov-solver controls.");
   MFEM_VERIFY(load_component >= -1,
               "The traction component must be -1 or nonnegative.");
   if (load_component < 0) { load_component = dimension - 1; }
   MFEM_VERIFY(load_component < dimension,
               "The traction component is outside the spatial dimension.");

   Device device(device_configuration);
   if (Mpi::Root()) { device.Print(); }

   // 2. Build the Cartesian cantilever and its vector H1 true-dof space.
   // MakeCartesian2D uses attributes 4/2 for x-min/x-max; the corresponding
   // 3D attributes are 5/3.
   Mesh serial_mesh;
   if (dimension == 2)
   {
      serial_mesh = Mesh::MakeCartesian2D(
                       nx, ny, Element::QUADRILATERAL, true, length, height);
   }
   else
   {
      serial_mesh = Mesh::MakeCartesian3D(
                       nx, ny, nz, Element::HEXAHEDRON,
                       length, height, width);
   }
   for (int level = 0; level < serial_refinements; ++level)
   {
      serial_mesh.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int level = 0; level < parallel_refinements; ++level)
   {
      mesh.UniformRefinement();
   }
   MPI_Comm comm = mesh.GetComm();

   H1_FECollection collection(order, dimension, BasisType::GaussLobatto);
   ParFiniteElementSpace space(&mesh, &collection, dimension,
                               Ordering::byNODES);
   const int support_attribute = dimension == 2 ? 4 : 5;
   const int free_surface_attribute = dimension == 2 ? 2 : 3;
   Array<int> essential_boundary(mesh.bdr_attributes.Max());
   essential_boundary = 0;
   essential_boundary[support_attribute-1] = 1;
   Array<int> essential_tdofs;
   space.GetEssentialTrueDofs(essential_boundary, essential_tdofs);

   // LOBPCG must operate in a space where the mass inner product is positive
   // definite. Build an explicit local map containing only free true dofs.
   Array<int> essential_marker(space.GetTrueVSize());
   essential_marker = 0;
   for (const int tdof : essential_tdofs) { essential_marker[tdof] = 1; }
   Array<int> free_tdofs;
   free_tdofs.Reserve(space.GetTrueVSize()-essential_tdofs.Size());
   for (int tdof = 0; tdof < space.GetTrueVSize(); ++tdof)
   {
      if (!essential_marker[tdof]) { free_tdofs.Append(tdof); }
   }

   HYPRE_BigInt local_free_dofs = free_tdofs.Size();
   HYPRE_BigInt global_free_dofs = 0;
   MPI_Allreduce(&local_free_dofs, &global_free_dofs, 1,
                 HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   if (Mpi::Root())
   {
      std::cout << "Total global true DOFs: " << space.GlobalTrueVSize()
                << '\n'
                << "Free global true DOFs: " << global_free_dofs << '\n';
   }
   MFEM_VERIFY(num_modes <= global_free_dofs,
               "The number of modes exceeds the number of free true dofs.");

   // 3. Assemble the high-order PA elasticity and vector-mass operators and
   // the static free-end traction. FormLinearSystem imposes the clamp on K.
   ConstantCoefficient lambda_coefficient(lame_lambda);
   ConstantCoefficient mu_coefficient(lame_mu);
   ConstantCoefficient density_coefficient(density);
   ParBilinearForm stiffness(&space);
   stiffness.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   stiffness.AddDomainIntegrator(
      new ElasticityIntegrator(lambda_coefficient, mu_coefficient));
   ParBilinearForm mass(&space);
   mass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass.AddDomainIntegrator(new VectorMassIntegrator(density_coefficient));

   Vector traction_value(dimension);
   traction_value = 0.0;
   traction_value(load_component) = load_amplitude;
   VectorConstantCoefficient traction(traction_value);
   Array<int> traction_marker(mesh.bdr_attributes.Max());
   traction_marker = 0;
   traction_marker[free_surface_attribute-1] = 1;
   ParLinearForm load(&space);
   load.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                              traction_marker);

   StopWatch common_assembly_timer;
   common_assembly_timer.Start();
   stiffness.Assemble();
   mass.Assemble();
   load.Assemble();
   ParGridFunction zero_displacement(&space);
   zero_displacement = 0.0;
   OperatorHandle stiffness_operator;
   Vector initial_solution, system_rhs;
   stiffness.FormLinearSystem(essential_tdofs, zero_displacement, load,
                              stiffness_operator, initial_solution,
                              system_rhs);
   OperatorHandle mass_operator;
   mass.FormSystemMatrix(essential_tdofs, mass_operator);
   common_assembly_timer.Stop();
   const double common_assembly_time =
      GlobalMaximum(comm, common_assembly_timer.RealTime());
   Operator &K = *stiffness_operator.Ptr();
   Operator &M = *mass_operator.Ptr();

   // 4. Assemble one low-order-refined stiffness matrix. Its AMG hierarchy is
   // reused first as the modal eigensolver preconditioner and later as the
   // reference preconditioner for the static PA system.
   StopWatch lor_assembly_timer;
   lor_assembly_timer.Start();
   ParLORDiscretization lor(stiffness, essential_tdofs);
   HypreParMatrix &lor_matrix = lor.GetAssembledMatrix();
   MFEM_VERIFY(lor_matrix.Height() == K.Height() &&
               lor_matrix.Width() == K.Width(),
               "LOR and high-order true-dof sizes do not match.");
   lor_assembly_timer.Stop();
   const double lor_assembly_time =
      GlobalMaximum(comm, lor_assembly_timer.RealTime());

   StopWatch eigen_amg_timer;
   eigen_amg_timer.Start();
   HypreBoomerAMG eigen_amg(lor_matrix);
   ConfigureAMG(eigen_amg, dimension, 0);
   Vector amg_rhs(lor_matrix.Height());
   Vector amg_solution(lor_matrix.Width());
   amg_rhs = 0.0;
   amg_solution = 0.0;
   eigen_amg.Setup(amg_rhs, amg_solution);
   eigen_amg_timer.Stop();
   const double eigen_amg_time =
      GlobalMaximum(comm, eigen_amg_timer.RealTime());

   // 5. Solve the generalized eigenproblem entirely on free true dofs. The
   // wrappers expand/restrict each PA or AMG application without assembling
   // either high-order matrix.
   FreeDofOperator free_stiffness(K, free_tdofs);
   FreeDofOperator free_mass(M, free_tdofs);
   FreeDofSolver free_eigen_amg(eigen_amg, free_tdofs);
   HypreLOBPCG lobpcg(comm);
   lobpcg.SetNumModes(num_modes);
   lobpcg.SetRandomSeed(eigen_seed);
   lobpcg.SetPreconditioner(free_eigen_amg);
   lobpcg.SetMaxIter(eigen_max_iterations);
   lobpcg.SetTol(eigen_tolerance);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(eigen_print_level);
   lobpcg.SetMassMatrix(free_mass);
   lobpcg.SetOperator(free_stiffness);
   StopWatch eigen_timer;
   eigen_timer.Start();
   lobpcg.Solve();
   eigen_timer.Stop();
   const double eigen_time = GlobalMaximum(comm, eigen_timer.RealTime());

   Array<real_t> eigenvalues;
   lobpcg.GetEigenvalues(eigenvalues);
   StopWatch mode_processing_timer;
   mode_processing_timer.Start();
   std::vector<Vector> modes;
   std::vector<Vector> mass_modes;
   std::vector<real_t> eigen_residuals(num_modes);
   modes.reserve(num_modes);
   mass_modes.reserve(num_modes);
   Vector mass_image(K.Height()), stiffness_image(K.Height());
   for (int mode_index = 0; mode_index < num_modes; ++mode_index)
   {
      MFEM_VERIFY(std::isfinite(eigenvalues[mode_index]) &&
                  eigenvalues[mode_index] > 0.0,
                  "LOBPCG returned invalid eigenvalue "
                  << eigenvalues[mode_index] << " for mode " << mode_index
                  << ". Try -epl 1 to inspect its convergence history.");
      Vector mode(K.Height());
      mode = 0.0;
      // Expansion through free_tdofs makes the clamped values exactly zero.
      mode.SetSubVector(free_tdofs, lobpcg.GetEigenvector(mode_index));
      M.Mult(mode, mass_image);
      const real_t mass_norm = std::sqrt(std::max(
         real_t(0.0), InnerProduct(comm, mode, mass_image)));
      MFEM_VERIFY(mass_norm > 0.0, "An eigenmode has zero mass norm.");
      mode /= mass_norm;

      K.Mult(mode, stiffness_image);
      M.Mult(mode, mass_image);
      const real_t stiffness_norm = GlobalNorm(comm, stiffness_image);
      stiffness_image.Add(-eigenvalues[mode_index], mass_image);
      const real_t denominator =
         stiffness_norm +
         std::abs(eigenvalues[mode_index])*GlobalNorm(comm, mass_image);
      eigen_residuals[mode_index] =
         GlobalNorm(comm, stiffness_image)/(denominator > 0.0 ?
                                             denominator : 1.0);
      modes.push_back(mode);
      mass_modes.push_back(mass_image);
   }

   real_t mass_orthogonality_error = 0.0;
   for (int j = 0; j < num_modes; ++j)
   {
      for (int i = 0; i < num_modes; ++i)
      {
         const real_t value = InnerProduct(comm, modes[i], mass_modes[j]);
         const real_t expected = i == j ? 1.0 : 0.0;
         mass_orthogonality_error = std::max(
                                      mass_orthogonality_error,
                                      std::abs(value-expected));
      }
   }
   mode_processing_timer.Stop();
   const double mode_processing_time =
      GlobalMaximum(comm, mode_processing_timer.RealTime());

   // 6. Construct the selected smoother. A diagonal smoother is assembled
   // from element row bounds. The LOR-AMG choice gets a separate hierarchy so
   // its setup cost is attributed to the two-level method, not the comparison.
   std::unique_ptr<OperatorJacobiSmoother> diagonal_smoother;
   std::unique_ptr<HypreBoomerAMG> lor_amg_smoother;
   std::unique_ptr<SymmetricOperatorAdapter> symmetric_lor_amg_smoother;
   const Operator *smoother = nullptr;
   double smoother_setup_time = 0.0;
   if (smoother_type != "none")
   {
      StopWatch smoother_timer;
      smoother_timer.Start();
      if (smoother_type == "lor-amg")
      {
         lor_amg_smoother.reset(new HypreBoomerAMG(lor_matrix));
         ConfigureAMG(*lor_amg_smoother, dimension, print_level);
         amg_solution = 0.0;
         lor_amg_smoother->Setup(amg_rhs, amg_solution);
         symmetric_lor_amg_smoother.reset(
            new SymmetricOperatorAdapter(*lor_amg_smoother));
         smoother = symmetric_lor_amg_smoother.get();
      }
      else
      {
         Vector smoother_diagonal;
         AssembleRowNormDiagonal(stiffness, essential_tdofs,
                                 smoother_type == "l2", smoother_diagonal);
         diagonal_smoother.reset(
            new OperatorJacobiSmoother(smoother_diagonal, essential_tdofs));
         smoother = diagonal_smoother.get();
      }
      smoother_timer.Stop();
      smoother_setup_time =
         GlobalMaximum(comm, smoother_timer.RealTime());
   }

   TwoLevelPreconditioner two_level(comm, K, num_modes);
   if (smoother)
   {
      if (smoother_placement == "both")
      {
         two_level.SetSmoother(*smoother);
      }
      else if (smoother_placement == "pre")
      {
         two_level.SetPreSmoother(*smoother);
      }
      else
      {
         two_level.SetPostSmoother(*smoother);
      }
   }
   for (const Vector &mode : modes) { two_level.AddCoarseVector(mode); }
   StopWatch coarse_timer;
   coarse_timer.Start();
   two_level.Assemble();
   coarse_timer.Stop();
   const double coarse_setup_time =
      GlobalMaximum(comm, coarse_timer.RealTime());

   // 7. Two-sided smoothing gives a symmetric PCG preconditioner. A one-sided
   // cycle is nonsymmetric and therefore always uses GMRES. --gmres also
   // selects GMRES for symmetric and deflated solves. Without smoothing, solve
   // the compatible semidefinite deflated problem and reconstruct u.
   SolveResult two_level_result;
   const bool one_sided_smoothing =
      smoother_type != "none" && smoother_placement != "both";
   const bool two_level_uses_gmres = use_gmres || one_sided_smoothing;
   if (smoother_type != "none")
   {
      if (!two_level_uses_gmres)
      {
         two_level_result = RunCG(comm, K, system_rhs, &two_level,
                                  relative_tolerance, absolute_tolerance,
                                  max_iterations, print_level);
      }
      else
      {
         two_level_result = RunGMRES(comm, K, system_rhs, &two_level,
                                     relative_tolerance, absolute_tolerance,
                                     max_iterations, print_level);
      }
   }
   else
   {
      DeflatedSystemOperator deflated_operator(two_level);
      Vector deflated_rhs;
      two_level.FormDeflatedRHS(system_rhs, deflated_rhs);
      SolveResult complementary;
      if (two_level_uses_gmres)
      {
         complementary = RunGMRES(
            comm, deflated_operator, deflated_rhs, nullptr,
            relative_tolerance, absolute_tolerance,
            max_iterations, print_level);
      }
      else
      {
         complementary = RunCG(
            comm, deflated_operator, deflated_rhs, nullptr,
            relative_tolerance, absolute_tolerance,
            max_iterations, print_level);
      }
      two_level_result = complementary;
      two_level.RecoverDeflatedSolution(system_rhs, complementary.solution,
                                        two_level_result.solution);
   }
   two_level_result.relative_residual = ComputeRelativeResidual(
                                          comm, K, system_rhs,
                                          two_level_result.solution);

   // 8. Solve the same PA static system with LOR-AMG preconditioning.
   StopWatch comparison_amg_timer;
   comparison_amg_timer.Start();
   HypreBoomerAMG comparison_amg(lor_matrix);
   ConfigureAMG(comparison_amg, dimension, print_level);
   amg_solution = 0.0;
   comparison_amg.Setup(amg_rhs, amg_solution);
   comparison_amg_timer.Stop();
   const double comparison_amg_time =
      GlobalMaximum(comm, comparison_amg_timer.RealTime());
   SolveResult lor_result;
   if (use_gmres)
   {
      lor_result = RunGMRES(
         comm, K, system_rhs, &comparison_amg, relative_tolerance,
         absolute_tolerance, max_iterations, print_level);
   }
   else
   {
      lor_result = RunCG(
         comm, K, system_rhs, &comparison_amg, relative_tolerance,
         absolute_tolerance, max_iterations, print_level);
   }
   lor_result.relative_residual = ComputeRelativeResidual(
                                    comm, K, system_rhs,
                                    lor_result.solution);

   Vector solution_difference(two_level_result.solution);
   solution_difference -= lor_result.solution;
   const real_t relative_solution_difference =
      GlobalNorm(comm, solution_difference)/
      std::max(GlobalNorm(comm, lor_result.solution), real_t(1.0e-30));

   const double two_level_setup_time = lor_assembly_time + eigen_amg_time +
      eigen_time + mode_processing_time + smoother_setup_time +
      coarse_setup_time;
   const double lor_setup_time = lor_assembly_time + comparison_amg_time;
   if (Mpi::Root())
   {
      std::cout << std::setprecision(12)
                << "Static cantilever dimension: " << dimension << '\n'
                << "Total global true DOFs: " << space.GlobalTrueVSize()
                << '\n'
                << "Free global true DOFs: " << global_free_dofs << '\n'
                << "Coarse eigenmodes: " << num_modes << '\n'
                << "Mass-orthogonality max error: "
                << mass_orthogonality_error << '\n'
                << "Smoother: " << smoother_type
                << (smoother_type == "none" ? " (deflated solve)" : "")
                << '\n'
                << "Smoother placement: "
                << (smoother_type == "none" ? "n/a" : smoother_placement)
                << '\n'
                << "Requested static solver: "
                << (use_gmres ? "GMRES" : "CG") << '\n'
                << "Common PA assembly time: " << common_assembly_time << '\n'
                << "LOR matrix assembly time: " << lor_assembly_time << '\n'
                << "Eigen AMG setup time: " << eigen_amg_time << '\n'
                << "Eigenmode solve time: " << eigen_time << '\n'
                << "Eigenmode processing time: "
                << mode_processing_time << '\n'
                << "Smoother setup time: " << smoother_setup_time << '\n'
                << "Coarse SVD setup time: " << coarse_setup_time << '\n';
      for (int i = 0; i < num_modes; ++i)
      {
         std::cout << "Mode " << i+1 << ": lambda=" << eigenvalues[i]
                   << ", omega=" << std::sqrt(eigenvalues[i])
                   << ", relative residual=" << eigen_residuals[i] << '\n';
      }
      std::cout << "\nMethod                         setup(s)  iterations"
                   "  solve(s)  true-rel-residual  converged\n"
                << (smoother_type == "none" ?
                    (two_level_uses_gmres ?
                     "deflated GMRES             " :
                     "deflated CG                ") :
                    (two_level_uses_gmres ?
                     "two-level GMRES            " :
                     "two-level PCG              "))
                << two_level_setup_time << "  "
                << two_level_result.iterations << "  "
                << two_level_result.solve_time << "  "
                << two_level_result.relative_residual << "  "
                << (two_level_result.converged ? "yes" : "no") << '\n'
                << (use_gmres ?
                    "LOR-AMG GMRES               " :
                    "LOR-AMG PCG                 ") << lor_setup_time << "  "
                << lor_result.iterations << "  " << lor_result.solve_time
                << "  " << lor_result.relative_residual << "  "
                << (lor_result.converged ? "yes" : "no") << '\n'
                << "Relative solution difference: "
                << relative_solution_difference << '\n';
   }

   if (std::string(csv_path).size() > 0)
   {
      // CSV contains one row per method and is written only by rank zero.
      std::unique_ptr<std::ofstream> csv;
      int csv_open = 1;
      if (Mpi::Root())
      {
         csv.reset(new std::ofstream(csv_path));
         csv_open = *csv ? 1 : 0;
      }
      MPI_Bcast(&csv_open, 1, MPI_INT, 0, comm);
      MFEM_VERIFY(csv_open, "Unable to open CSV output file: " << csv_path);
      if (Mpi::Root())
      {
         *csv << "method,dimension,true_dofs,modes,smoother,placement,"
                 "setup_time,"
                 "iterations,solve_time,true_relative_residual,converged\n";
         *csv << (smoother_type == "none" ?
                  (two_level_uses_gmres ?
                   "deflated_gmres" : "deflated_cg") :
                  (two_level_uses_gmres ?
                   "two_level_gmres" : "two_level_pcg")) << ','
              << dimension << ',' << space.GlobalTrueVSize() << ','
              << num_modes << ','
              << smoother_type << ','
              << (smoother_type == "none" ? "n/a" : smoother_placement) << ','
              << two_level_setup_time << ','
              << two_level_result.iterations << ','
              << two_level_result.solve_time << ','
              << two_level_result.relative_residual << ','
              << (two_level_result.converged ? 1 : 0) << '\n';
         *csv << (use_gmres ? "lor_amg_gmres," : "lor_amg_pcg,")
              << dimension << ','
              << space.GlobalTrueVSize() << ',' << num_modes << ",n/a,n/a,"
              << lor_setup_time << ',' << lor_result.iterations << ','
              << lor_result.solve_time << ','
              << lor_result.relative_residual << ','
              << (lor_result.converged ? 1 : 0) << '\n';
      }
   }

   if (visualization)
   {
      // ParaView fields use the high-order vector space. Modes are already
      // mass-normalized and satisfy the homogeneous clamp exactly.
      ParGridFunction two_level_displacement(&space);
      ParGridFunction lor_displacement(&space);
      ParGridFunction displacement_difference(&space);
      two_level_displacement.SetFromTrueDofs(two_level_result.solution);
      lor_displacement.SetFromTrueDofs(lor_result.solution);
      displacement_difference = two_level_displacement;
      displacement_difference -= lor_displacement;

      const int output_modes = std::min(10, num_modes);
      std::vector<std::unique_ptr<ParGridFunction>> mode_fields;
      mode_fields.reserve(output_modes);
      for (int i = 0; i < output_modes; ++i)
      {
         mode_fields.emplace_back(new ParGridFunction(&space));
         mode_fields.back()->SetFromTrueDofs(modes[i]);
      }

      const std::string collection_name =
         "two_level_elasticity_" + std::to_string(dimension) + "d";
      ParaViewDataCollection paraview(collection_name, &mesh);
      paraview.SetPrefixPath(output_prefix);
      paraview.SetLevelsOfDetail(order);
      paraview.SetDataFormat(VTKFormat::BINARY);
      paraview.SetHighOrderOutput(true);
      paraview.SetCycle(0);
      paraview.SetTime(0.0);
      paraview.RegisterField("displacement_two_level",
                             &two_level_displacement);
      paraview.RegisterField("displacement_lor_amg", &lor_displacement);
      paraview.RegisterField("displacement_difference",
                             &displacement_difference);
      for (int i = 0; i < output_modes; ++i)
      {
         const std::string name = "mode_" +
                                  (i+1 < 10 ? std::string("0") : "") +
                                  std::to_string(i+1);
         paraview.RegisterField(name, mode_fields[i].get());
      }
      paraview.Save();
   }

   return two_level_result.converged && lor_result.converged ?
          EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
