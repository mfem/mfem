// test_mean_free_par_diffusion.cpp
//
// Standalone parallel MFEM driver for testing a mean-free pure-Neumann
// diffusion solver.  The test problem is
//
//     -div(kappa grad u) = f        in [0,1]^d,
//        kappa grad u . n = 0       on boundary,
//        integral(u) = 0,
//
// with manufactured solution
//
//     u_exact(x) = prod_i cos(pi x_i),
//     f(x)       = kappa * d * pi^2 * u_exact(x).
//
// This exact solution has zero Neumann flux and zero mean on the unit box.
// The driver also solves a deliberately incompatible RHS, b + alpha*m,
// where m_i = int phi_i dx.  The solver should project alpha*m away, so the
// second solution should match the first solution up to solver tolerance.
//
// ParaView output can be enabled with -pv.  The fields written are:
//
//     solution, exact, error

#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace mfem;
using namespace std;

namespace
{
/// Diffusion coefficient used by the manufactured RHS callback.
real_t g_kappa = 1.0;

/// Spatial dimension used by the manufactured RHS callback.
int g_dim = 2;

/**
 * @brief Return the mathematical constant pi.
 *
 * The implementation uses the identity pi = 4 atan(1) so that the driver does
 * not depend on non-standard constants such as M_PI.
 *
 * @return Approximation of pi in MFEM's real_t precision.
 */
real_t Pi()
{
   return 4.0*atan(1.0);
}

/**
 * @brief Evaluate the manufactured exact solution at a physical point.
 *
 * The exact solution is
 *
 *     u(x) = prod_i cos(pi x_i),
 *
 * on the unit box.  It has zero normal derivative on each coordinate-aligned
 * boundary face and zero integral over the domain.
 *
 * @param x Physical coordinates of the evaluation point.
 * @return Value of the exact scalar solution at @a x.
 */
real_t ExactSolution(const Vector &x)
{
   const real_t pi = Pi();
   real_t u = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      u *= cos(pi*x(d));
   }
   return u;
}

/**
 * @brief Evaluate the forcing term for the manufactured Neumann problem.
 *
 * For constant kappa and the exact solution used in this file,
 *
 *     -div(kappa grad u) = kappa * dim * pi^2 * u.
 *
 * The global values @c g_kappa and @c g_dim are set from the command-line
 * options before this callback is used by MFEM's FunctionCoefficient.
 *
 * @param x Physical coordinates of the evaluation point.
 * @return Value of the scalar RHS at @a x.
 */
real_t RightHandSide(const Vector &x)
{
   const real_t pi = Pi();
   return g_kappa * g_dim * pi*pi * ExactSolution(x);
}

/**
 * @brief Compute the parallel Euclidean norm of a true-dof vector.
 *
 * This helper uses MFEM's parallel inner product, which performs the MPI
 * reduction over all ranks in @a comm.
 *
 * @param comm MPI communicator associated with the vector distribution.
 * @param v True-dof vector whose l2 norm is requested.
 * @return sqrt((v, v)) over the distributed true-dof vector.
 */
real_t ParNorml2(MPI_Comm comm, const Vector &v)
{
   return sqrt(InnerProduct(comm, v, v));
}
}

/**
 * @brief Mean-free parallel diffusion solver for pure-Neumann problems.
 *
 * This class owns a true-dof diffusion operator assembled from a parallel H1
 * finite element space and a scalar diffusion coefficient.  The matrix is
 * singular for a pure-Neumann problem because constants are in the nullspace.
 * The solver removes that ambiguity by:
 *
 *   1. projecting every RHS to the compatible subspace, z^T b = 0;
 *   2. solving the projected system with CG preconditioned by BoomerAMG;
 *   3. subtracting the integral mean from the computed solution.
 *
 * Here @c z is the true-dof coefficient vector representing the constant
 * function 1, and @c m is the true-dof mass/load vector with entries
 * m_i = int phi_i dx.  The projection is
 *
 *     b <- b - (z^T b / |Omega|) m,
 *
 * and the final gauge choice is
 *
 *     x <- x - (int x dx / |Omega|) z.
 *
 * All public vector inputs and outputs are true-dof vectors.
 */
class MeanFreeParDiffusionSolver : public Solver
{
public:
   /**
    * @brief Assemble and configure a mean-free diffusion solver.
    *
    * The constructor assembles the parallel true-dof matrix corresponding to
    * the bilinear form (kappa grad u, grad v), builds the constant null vector
    * and mass/load vector used for projections, and configures a CG solver with
    * HypreBoomerAMG as a preconditioner.
    *
    * @param fes Parallel finite element space defining the H1 trial and test
    *            space.  The object must outlive this solver because it is
    *            stored by reference.
    * @param kappa Scalar diffusion coefficient used in the domain diffusion
    *              integrator.  It is only needed during assembly.
    * @param rel_tol Relative convergence tolerance for CG.
    * @param max_iter Maximum number of CG iterations.
    * @param print_level MFEM CG print level.
    */
   MeanFreeParDiffusionSolver(ParFiniteElementSpace &fes,
                              Coefficient &kappa,
                              real_t rel_tol = 1e-12,
                              int max_iter = 500,
                              int print_level = 0)
      : Solver(fes.GetTrueVSize(), false),
        fes_(fes),
        comm_(fes.GetComm()),
        cg_(comm_),
        b_(fes.GetTrueVSize()),
        z_(fes.GetTrueVSize()),
        m_(fes.GetTrueVSize())
   {
      ParBilinearForm a(&fes_);
      a.AddDomainIntegrator(new DiffusionIntegrator(kappa));
      a.Assemble();
      a.Finalize();

      A_.reset(a.ParallelAssemble());

      BuildConstantModeAndMassVector();

      amg_.reset(new HypreBoomerAMG);
      amg_->SetPrintLevel(print_level);

      cg_.SetRelTol(rel_tol);
      cg_.SetAbsTol(0.0);
      cg_.SetMaxIter(max_iter);
      cg_.SetPrintLevel(print_level);
      cg_.SetPreconditioner(*amg_);
      cg_.SetOperator(*A_);
   }

   /**
    * @brief Reject external operator replacement.
    *
    * The solver owns a matrix assembled from the finite element space and
    * coefficient provided to the constructor.  Replacing the operator through
    * the generic Solver interface would leave the stored nullspace projection
    * data inconsistent, so this method aborts.
    *
    * @param op Ignored operator argument from the base-class interface.
    */
   void SetOperator(const Operator &op) override
   {
      (void) op;
      MFEM_ABORT("MeanFreeParDiffusionSolver owns its operator; rebuild it "
                 "after changing the finite element space or coefficient.");
   }

   /**
    * @brief Solve the mean-free Neumann diffusion problem.
    *
    * The input RHS is first projected to satisfy the compatibility condition
    * z^T rhs = 0.  The projected singular system is then solved from a zero
    * initial guess, and the result is shifted to have zero integral mean.
    *
    * @param rhs True-dof RHS vector.  It may be compatible or incompatible.
    * @param x On output, true-dof solution vector with zero integral mean.
    */
   void Mult(const Vector &rhs, Vector &x) const override
   {
      MFEM_VERIFY(rhs.Size() == A_->Height(), "RHS must be a true-dof vector.");

      x.SetSize(A_->Width());

      ProjectRHS(rhs, b_);

      x = 0.0;
      cg_.Mult(b_, x);

      SetZeroMean(x);
   }

   /**
    * @brief Project a RHS vector onto the compatible range of the operator.
    *
    * For a pure-Neumann diffusion matrix, the compatibility condition is that
    * the total load against the constant function vanish: z^T b = 0.  This
    * routine computes
    *
    *     projected_rhs = rhs - (z^T rhs / |Omega|) m.
    *
    * @param rhs Input true-dof RHS vector.
    * @param projected_rhs Output true-dof RHS vector after compatibility
    *                      projection.
    */
   void ProjectRHS(const Vector &rhs, Vector &projected_rhs) const
   {
      projected_rhs = rhs;
      MakeCompatible(projected_rhs);
   }

   /**
    * @brief Access the assembled true-dof diffusion matrix.
    *
    * @return Reference to the owned Hypre parallel matrix.
    */
   const HypreParMatrix &GetMatrix() const { return *A_; }

   /**
    * @brief Access the true-dof vector representing the constant function 1.
    *
    * @return Reference to the constant null-mode vector @c z.
    */
   const Vector &GetConstantMode() const { return z_; }

   /**
    * @brief Access the true-dof mass/load vector for integration.
    *
    * The vector represents the linear functional v -> int_Omega v dx, so the
    * integral of a true-dof vector @c x is @c InnerProduct(comm, m, x).
    *
    * @return Reference to the mass/load vector @c m.
    */
   const Vector &GetMassVector() const { return m_; }

   /**
    * @brief Return the global measure of the computational domain.
    *
    * @return |Omega| computed as z^T m.
    */
   real_t GetVolume() const { return volume_; }

   /**
    * @brief Compute the integral mean of a true-dof vector.
    *
    * The mean is (int_Omega x dx) / |Omega|, evaluated using the mass/load
    * vector built in the constructor.
    *
    * @param x True-dof vector representing a scalar finite element function.
    * @return Integral mean of @a x over the global domain.
    */
   real_t Mean(const Vector &x) const
   {
      return InnerProduct(comm_, m_, x)/volume_;
   }

   /**
    * @brief Compute the total load of a true-dof RHS vector.
    *
    * For a RHS vector @a rhs, this returns z^T rhs, where @c z is the true-dof
    * vector representing the constant function 1.  A compatible pure-Neumann
    * RHS has total load equal to zero, up to roundoff and quadrature error.
    *
    * @param rhs True-dof RHS vector.
    * @return Global total load z^T rhs.
    */
   real_t TotalLoad(const Vector &rhs) const
   {
      return InnerProduct(comm_, z_, rhs);
   }

private:
   /// Parallel finite element space used to assemble the owned operator.
   ParFiniteElementSpace &fes_;

   /// MPI communicator associated with @c fes_ and all true-dof vectors.
   MPI_Comm comm_;

   /// Owned true-dof diffusion matrix.
   unique_ptr<HypreParMatrix> A_;

   /// Owned BoomerAMG preconditioner for the CG iteration.
   unique_ptr<HypreBoomerAMG> amg_;

   /// Mutable CG solver because Solver::Mult is const in MFEM.
   //mutable CGSolver cg_;
   mutable GMRESSolver cg_;

   /// Work vector storing the projected RHS used by Mult().
   mutable Vector b_;

   /// True-dof coefficient vector for the constant function 1.
   Vector z_;

   /// True-dof load vector m_i = int_Omega phi_i dx.
   Vector m_;

   /// Global domain measure, computed as z^T m.
   real_t volume_ = 0.0;

   /**
    * @brief Build the constant mode, mass/load vector, and domain volume.
    *
    * The constant mode @c z_ is obtained by projecting the coefficient 1 into
    * the finite element space and then converting to true DOFs.  The vector
    * @c m_ is assembled as the linear form int_Omega v dx.  Their inner
    * product gives the domain volume.
    */
   void BuildConstantModeAndMassVector()
   {
      ConstantCoefficient one(1.0);

      ParGridFunction one_gf(&fes_);
      one_gf.ProjectCoefficient(one);
      one_gf.ParallelProject(z_);

      ParLinearForm lf_one(&fes_);
      lf_one.AddDomainIntegrator(new DomainLFIntegrator(one));
      lf_one.Assemble();
      lf_one.ParallelAssemble(m_);

      volume_ = InnerProduct(comm_, z_, m_);
      MFEM_VERIFY(volume_ > 0.0, "Non-positive domain volume.");
   }

   /**
    * @brief Modify a RHS vector in place so it satisfies z^T rhs = 0.
    *
    * The correction subtracts the constant load component using the integration
    * vector @c m_:
    *
    *     rhs <- rhs - (z^T rhs / |Omega|) m.
    *
    * @param rhs True-dof RHS vector to modify in place.
    */
   void MakeCompatible(Vector &rhs) const
   {
      const real_t total_load = InnerProduct(comm_, z_, rhs);
      rhs.Add(-total_load/volume_, m_);
   }

   /**
    * @brief Shift a true-dof solution vector to have zero integral mean.
    *
    * Since constants are in the nullspace of the pure-Neumann operator, the
    * computed solution is unique only up to an additive constant.  This method
    * chooses the representative satisfying int_Omega x dx = 0.
    *
    * @param x True-dof solution vector to modify in place.
    */
   void SetZeroMean(Vector &x) const
   {
      const real_t mean = Mean(x);
      x.Add(-mean, z_);
   }
};

/**
 * @brief Run the parallel manufactured-solution test driver.
 *
 * The driver constructs a Cartesian unit-box mesh, refines it, builds a
 * parallel H1 space, assembles a manufactured RHS, solves the pure-Neumann
 * diffusion problem with MeanFreeParDiffusionSolver, and checks:
 *
 *   - L2 error against the exact solution;
 *   - projected residual norm;
 *   - zero mean of the numerical solution;
 *   - invariance under adding a constant incompatible load.
 *
 * Optional MFEM-format output is controlled by @c -s, and optional ParaView
 * output is controlled by @c -pv.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array.
 * @return 0 if all checks pass, 1 for option parsing failure, 2 for invalid
 *         option values, or 3 if at least one numerical check fails.
 */
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();
   MPI_Comm comm = MPI_COMM_WORLD;

   int dim = 2;
   int nx = 8;
   int order = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   real_t kappa_value = 1.0;
   real_t cg_rel_tol = 1e-12;
   int cg_max_iter = 500;
   int cg_print = 0;
   real_t incompatible_shift = 1.0;
   real_t l2_tol = 1e-3;
   real_t residual_tol = 1e-8;
   real_t mean_tol = 1e-12;
   bool save = false;
   bool paraview = true;
   const char *paraview_prefix = "ParaView";
   const char *paraview_name = "mean_free_neumann";
   bool paraview_binary = true;
   bool paraview_high_order = true;
   int paraview_lod = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension of the unit box test mesh: 2 or 3.");
   args.AddOption(&nx, "-n", "--num-elements",
                  "Number of elements per direction before refinement.");
   args.AddOption(&order, "-o", "--order",
                  "H1 finite element order.");
   args.AddOption(&ser_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of uniform serial refinements before partitioning.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of uniform parallel refinements after partitioning.");
   args.AddOption(&kappa_value, "-k", "--kappa",
                  "Constant diffusion coefficient.");
   args.AddOption(&cg_rel_tol, "-rtol", "--relative-tolerance",
                  "CG relative tolerance.");
   args.AddOption(&cg_max_iter, "-mi", "--max-iterations",
                  "Maximum CG iterations.");
   args.AddOption(&cg_print, "-pl", "--print-level",
                  "CG print level.");
   args.AddOption(&incompatible_shift, "-shift", "--incompatible-shift",
                  "Amount of constant load added for the incompatibility test.");
   args.AddOption(&l2_tol, "-l2tol", "--l2-tolerance",
                  "Pass/fail tolerance for the L2 error of the compatible solve.");
   args.AddOption(&residual_tol, "-restol", "--residual-tolerance",
                  "Pass/fail tolerance for the relative projected residual.");
   args.AddOption(&mean_tol, "-meantol", "--mean-tolerance",
                  "Pass/fail tolerance for absolute solution mean.");
   args.AddOption(&save, "-s", "--save", "-no-s", "--no-save",
                  "Save parallel mesh and solution files in MFEM format.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Save ParaView/PVTU output.");
   args.AddOption(&paraview_prefix, "-pv-prefix", "--paraview-prefix",
                  "Directory prefix for ParaView output.");
   args.AddOption(&paraview_name, "-pv-name", "--paraview-name",
                  "ParaView data collection name.");
   args.AddOption(&paraview_binary, "-pvbin", "--paraview-binary",
                  "-pvtxt", "--paraview-ascii",
                  "Use binary or ASCII ParaView output.");
   args.AddOption(&paraview_high_order, "-pvho", "--paraview-high-order",
                  "-no-pvho", "--no-paraview-high-order",
                  "Use high-order ParaView output.");
   args.AddOption(&paraview_lod, "-pv-lod", "--paraview-levels-of-detail",
                  "ParaView output refinement level.  Use -1 to match the FE order.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (dim != 2 && dim != 3)
   {
      if (myid == 0) { cerr << "ERROR: -dim must be 2 or 3.\n"; }
      return 2;
   }
   if (nx < 1 || order < 1)
   {
      if (myid == 0) { cerr << "ERROR: -n and -o must be positive.\n"; }
      return 2;
   }
   if (paraview_lod == 0 || paraview_lod < -1)
   {
      if (myid == 0)
      {
         cerr << "ERROR: -pv-lod must be positive, or -1 to match the FE order.\n";
      }
      return 2;
   }

   if (myid == 0) { args.PrintOptions(cout); }

   g_dim = dim;
   g_kappa = kappa_value;

   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL,
                                   true, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON,
                                   1.0, 1.0, 1.0);
   }

   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);


   {
      long int gn=fes.GlobalTrueVSize();

      if (myid == 0)
      {
         cout << "Global true dofs: " << gn << '\n';
      }
   }

   ConstantCoefficient kappa(kappa_value);
   FunctionCoefficient fcoef(RightHandSide);
   FunctionCoefficient ucoef(ExactSolution);

   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(fcoef));
   b.Assemble();

   Vector B; B.SetSize(fes.GetTrueVSize());
   b.ParallelAssemble(B);

   MeanFreeParDiffusionSolver solver(fes, kappa, cg_rel_tol,
                                     cg_max_iter, cg_print);

   Vector B_projected;
   solver.ProjectRHS(B, B_projected);

   const real_t raw_total_load = solver.TotalLoad(B);
   const real_t projected_total_load = solver.TotalLoad(B_projected);

   Vector X;
   solver.Mult(B, X);

   Vector AX(B.Size());
   solver.GetMatrix().Mult(X, AX);
   AX -= B_projected;

   const real_t projected_rhs_norm = ParNorml2(comm, B_projected);
   const real_t residual_norm = ParNorml2(comm, AX);
   const real_t relative_residual = residual_norm / max(projected_rhs_norm, real_t(1.0));
   const real_t solution_mean = solver.Mean(X);

   ParGridFunction uh(&fes);
   uh.SetFromTrueDofs(X);

   ParGridFunction uex_gf(&fes);
   uex_gf.ProjectCoefficient(ucoef);

   ParGridFunction error_gf(&fes);
   error_gf = uh;
   error_gf -= uex_gf;

   const real_t l2_error = uh.ComputeL2Error(ucoef);

   // Incompatibility test: add alpha*m to the RHS.  The solver should remove
   // this constant load and recover the same mean-free solution.
   Vector B_bad(B.Size());
   B_bad = B;
   B_bad.Add(incompatible_shift, solver.GetMassVector());

   Vector B_bad_projected;
   solver.ProjectRHS(B_bad, B_bad_projected);

   Vector X_bad;
   solver.Mult(B_bad, X_bad);

   Vector dX(X.Size());
   dX = X_bad;
   dX -= X;
   const real_t solution_difference = ParNorml2(comm, dX);

   Vector dB(B.Size());
   dB = B_bad_projected;
   dB -= B_projected;
   const real_t projected_rhs_difference = ParNorml2(comm, dB);

   const bool pass_l2 = (l2_error <= l2_tol);
   const bool pass_residual = (relative_residual <= residual_tol);
   const bool pass_mean = (fabs(solution_mean) <= mean_tol);
   const bool pass_projection =
      (projected_rhs_difference <= 10.0*residual_tol*max(projected_rhs_norm, real_t(1.0)) &&
       solution_difference <= 100.0*residual_tol*max(ParNorml2(comm, X), real_t(1.0)));

   real_t vol=solver.GetVolume();
   real_t tl=solver.TotalLoad(B_bad);

   if (myid == 0)
   {
      cout << setprecision(16);
      cout << "\nMean-free Neumann diffusion test\n"
           << "  volume                         = " << vol << '\n'
           << "  raw total load z^T B            = " << raw_total_load << '\n'
           << "  projected total load z^T Pc(B)  = " << projected_total_load << '\n'
           << "  ||Pc(B)||_2                     = " << projected_rhs_norm << '\n'
           << "  ||A X - Pc(B)||_2               = " << residual_norm << '\n'
           << "  relative projected residual     = " << relative_residual << '\n'
           << "  mean(X)                         = " << solution_mean << '\n'
           << "  L2 error                        = " << l2_error << '\n'
           << "\nIncompatible-RHS projection test\n"
           << "  incompatible shift alpha         = " << incompatible_shift << '\n'
           << "  total load z^T (B + alpha*m)    = " << tl << '\n'
           << "  ||Pc(B + alpha*m) - Pc(B)||_2   = " << projected_rhs_difference << '\n'
           << "  ||X_bad - X||_2                 = " << solution_difference << '\n'
           << "\nChecks\n"
           << "  L2 error                         " << (pass_l2 ? "PASS" : "FAIL") << '\n'
           << "  projected residual               " << (pass_residual ? "PASS" : "FAIL") << '\n'
           << "  zero mean                        " << (pass_mean ? "PASS" : "FAIL") << '\n'
           << "  incompatible RHS projection      " << (pass_projection ? "PASS" : "FAIL") << '\n';
   }

   if (save)
   {
      ostringstream mesh_name, sol_name, exact_name, error_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      exact_name << "exact." << setfill('0') << setw(6) << myid;
      error_name << "error." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(16);
      uh.Save(sol_ofs);

      ofstream exact_ofs(exact_name.str().c_str());
      exact_ofs.precision(16);
      uex_gf.Save(exact_ofs);

      ofstream error_ofs(error_name.str().c_str());
      error_ofs.precision(16);
      error_gf.Save(error_ofs);
   }

   if (paraview)
   {
      const int levels_of_detail = (paraview_lod < 0) ? order : paraview_lod;

      ParaViewDataCollection pvdc(paraview_name, &pmesh);
      pvdc.SetPrefixPath(paraview_prefix);
      pvdc.RegisterField("solution", &uh);
      pvdc.RegisterField("exact", &uex_gf);
      pvdc.RegisterField("error", &error_gf);
      pvdc.SetLevelsOfDetail(levels_of_detail);
      pvdc.SetDataFormat(paraview_binary ? VTKFormat::BINARY : VTKFormat::ASCII);
      pvdc.SetHighOrderOutput(paraview_high_order);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();

      if (myid == 0)
      {
         cout << "\nSaved ParaView output under: " << paraview_prefix
              << "/" << paraview_name << "\n";
      }
   }

   return (pass_l2 && pass_residual && pass_mean && pass_projection) ? 0 : 3;
}
