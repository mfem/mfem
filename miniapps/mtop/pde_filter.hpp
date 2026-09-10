#pragma once

#include "mfem.hpp"
#include <memory>

namespace toopt
{

using namespace mfem;

/// @brief Options controlling solver behaviour of the PDE filter.
struct PDEFilterOptions
{
    double filter_radius = 0.05;  ///< r in -div(r^2 A grad u) + u = rho

    double solver_rtol    = 1e-10;
    double solver_atol    = 1e-12;
    int    solver_maxiter = 500;

    /// 0=silent, 1=final residual, 2=every iteration
    int print_level = 0;
};

/// @brief Parallel PDE (Helmholtz) filter for topology optimization.
///
/// Inherits mfem::Operator and maps true-dof vectors:
///
///   Mult          (x_ctrl, y_filt) : y = F(x) — forward filter
///   MultTranspose (x_filt, y_ctrl) : y = F^T(x) — adjoint filter
///
/// An optional prescribed-output projection makes Mult affine.  In that mode,
/// MultTranspose applies the transpose of Mult's Jacobian; see
/// SetPrescribedOutputDofs().
///
/// Operator dimensions:
///   height = fes_filter.GetTrueVSize()   (output of Mult)
///   width  = fes_control.GetTrueVSize()  (input  of Mult)
///
/// With a positive radius, the filter solves:
///   (r^2 K + M) rho_tilde = M_fc rho
/// A zero radius intentionally removes Helmholtz smoothing and leaves the
/// exact L2-to-H1 mass projection M rho_tilde = M_fc rho.
///
/// where K is the H1 diffusion matrix, M the H1 mass matrix, and M_fc
/// the mixed mass matrix (control trial, filter test).
///
/// Typical usage:
/// @code
///   PDEFilter filter(fes_filter, fes_control, opts);
///   filter.SetDiffusionCoeff(my_coeff);  // optional
///   filter.Assemble();
///
///   // True-dof interface (Operator)
///   Vector rho_tdof(filter.Width()), rho_tilde_tdof(filter.Height());
///   filter.Mult(rho_tdof, rho_tilde_tdof);
///
///   Vector sens_tdof(filter.Width());
///   filter.MultTranspose(rho_tilde_tdof, sens_tdof);
///
///   // GridFunction convenience wrappers
///   ParGridFunction rho(&fes_control), rho_tilde(&fes_filter), sens(&fes_control);
///   filter.Mult(rho, rho_tilde);
///   filter.MultTranspose(rho_tilde, sens);
/// @endcode
class PDEFilter : public Operator
{
public:
    /// @brief Construct from externally owned FE spaces.
    ///
    /// @param fes_filter   Solution space — MUST be H1.
    /// @param fes_control  Input/control space — H1 or L2.
    /// @param opts         Solver and filter-radius options.
    PDEFilter(ParFiniteElementSpace& fes_filter,
              ParFiniteElementSpace& fes_control,
              const PDEFilterOptions& opts = PDEFilterOptions());

    ~PDEFilter() = default;

    PDEFilter(const PDEFilter&)            = delete;
    PDEFilter& operator=(const PDEFilter&) = delete;

    // -----------------------------------------------------------------------
    // Configuration — must be called before Assemble()
    // -----------------------------------------------------------------------

    /// @brief Isotropic spatially-varying diffusion: r^2 * a(x) * I.
    void SetDiffusionCoeff(Coefficient& coeff);

    /// @brief Axis-aligned anisotropic diffusion: r^2 * diag(v(x)).
    void SetDiffusionCoeff(VectorCoefficient& coeff);

    /// @brief Full-tensor diffusion: r^2 * M(x).
    void SetDiffusionCoeff(MatrixCoefficient& coeff);

    /// @brief Prescribe selected filtered true DOFs to a constant value.
    ///
    /// This optional operation is intended for fixed physical/passive regions.
    /// With C denoting the diagonal projector that is one on unconstrained
    /// filtered DOFs and zero on prescribed ones, the forward map becomes
    ///
    ///   y = C F x + (I-C) value.
    ///
    /// It is therefore affine rather than linear.  When prescribed outputs are
    /// enabled, MultTranspose() applies the transpose of the forward map's
    /// Jacobian, F^T C, not the transpose of the affine map itself.  This is the
    /// derivative required by topology-optimization adjoints.
    ///
    /// The list contains local true-DOF indices in the filter space.  It may be
    /// configured before or after Assemble(), since it does not change the
    /// Helmholtz matrix.
    void SetPrescribedOutputDofs(const Array<int>& tdofs, double value);

    /// Remove an existing prescribed-output projection.
    void ClearPrescribedOutputDofs();

    /// @brief Assemble the system matrix and set up the AMG+PCG solver.
    ///
    /// Must be called exactly once, after SetDiffusionCoeff (if any).
    void Assemble();

    // -----------------------------------------------------------------------
    // mfem::Operator interface — true-dof Vectors
    // -----------------------------------------------------------------------

    /// @brief Forward filter: y_filt = F(x_ctrl), optionally followed by the
    /// prescribed-output affine projection configured above.
    ///
    /// @param x  Control true-dof vector, size Width()  = ctrl TrueVSize.
    /// @param y  Filter  true-dof vector, size Height() = filt TrueVSize.
    void Mult(const Vector& x, Vector& y) const override;

    /// @brief Adjoint filter: y_ctrl = F^T(x_filt), or F^T C x_filt when
    /// prescribed outputs are enabled.
    ///
    /// Because (r^2 K + M) is SPD the adjoint reuses the same solver.
    ///
    /// @param x  Filter  true-dof vector, size Height() = filt TrueVSize.
    /// @param y  Control true-dof vector, size Width()  = ctrl TrueVSize.
    void MultTranspose(const Vector& x, Vector& y) const override;

    // -----------------------------------------------------------------------
    // GridFunction convenience wrappers
    // -----------------------------------------------------------------------

    /// @brief Forward filter on ParGridFunctions.
    void Mult(const ParGridFunction& rho, ParGridFunction& rho_tilde) const;

    /// @brief Adjoint filter on ParGridFunctions.
    void MultTranspose(const ParGridFunction& lambda,
                       ParGridFunction& sens_out) const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// H1 solution/filter space (not owned).
    ParFiniteElementSpace* GetFilterFESpace()  const { return fes_filter_; }

    /// Control/input space (not owned).
    ParFiniteElementSpace* GetControlFESpace() const { return fes_control_; }

    /// Assembled system matrix.  Null before Assemble().
    HypreParMatrix* GetSystemMatrix() const { return filter_mat_.get(); }

    MPI_Comm GetComm() const { return fes_filter_->GetComm(); }

    bool HasPrescribedOutputDofs() const
    { return prescribed_output_enabled_; }

    const Array<int>& GetPrescribedOutputDofs() const
    { return prescribed_output_tdofs_; }

    double GetPrescribedOutputValue() const
    { return prescribed_output_value_; }

private:
    void AssembleBilinearForm_();
    void AssembleMixedMass_();
    void SetupSolver_();
    void CheckConvergence_(const Vector& b,
                           const Vector& x,
                           const char*   context) const;

    ParFiniteElementSpace* fes_filter_;
    ParFiniteElementSpace* fes_control_;

    PDEFilterOptions opts_;

    std::unique_ptr<ParBilinearForm>      filter_bf_;
    std::unique_ptr<ParMixedBilinearForm> mixed_mass_;
    std::unique_ptr<HypreParMatrix>       mixed_mass_mat_;
    std::unique_ptr<HypreParMatrix>       filter_mat_;
    std::unique_ptr<HypreBoomerAMG>       amg_prec_;
    std::unique_ptr<HyprePCG>             solver_;

    struct DiffCoeff {
        Coefficient*       scalar = nullptr;
        VectorCoefficient* vector = nullptr;
        MatrixCoefficient* matrix = nullptr;
    } diff_;

    Array<int> prescribed_output_tdofs_;
    double prescribed_output_value_ = 0.0;
    // Separate from the local list size: in parallel a rank may own no
    // prescribed DOFs even though the globally configured projection is on.
    bool prescribed_output_enabled_ = false;

    // Owned wrapper coefficients created in AssembleBilinearForm_().
    // DiffusionIntegrator does not take ownership of its coefficient, so we
    // keep these alive for the lifetime of filter_bf_.
    // The scalar/default branches produce a Coefficient; the vector/matrix
    // branches produce a MatrixCoefficient (separate hierarchy in MFEM).
    std::unique_ptr<Coefficient>       diff_scalar_owned_;  ///< scalar or default
    std::unique_ptr<MatrixCoefficient> diff_matrix_owned_;  ///< vector or matrix

    bool assembled_ = false;
};

} // namespace toopt
