#pragma once

#include "mfem.hpp"
#include <memory>
#include <vector>

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
///   Mult          (x_ctrl,   y_filt)  :  y = F(x)   — forward filter
///   MultTranspose (x_filt,   y_ctrl)  :  y = F^T(x) — adjoint filter
///
/// Operator dimensions:
///   height = fes_filter.GetTrueVSize()   (output of Mult)
///   width  = fes_control.GetTrueVSize()  (input  of Mult)
///
/// The filter solves:
///   (r^2 K + M) rho_tilde = M_fc rho
///
/// where K is the H1 diffusion matrix, M the H1 mass matrix, and M_fc
/// the mixed mass matrix (control trial, filter test).
///
/// Optional essential Dirichlet conditions use the same configuration pattern
/// as the diffusion solvers in the stokes directory:
///
///   AddBoundaryCondition(attribute, coefficient_or_value);
///   Assemble();
///
/// With nonzero Dirichlet data, Mult() is an affine filter map. In that case
/// MultTranspose() applies the transpose of its linearization: essential output
/// rows, which do not depend on the control vector, contribute zero sensitivity.
///
/// Typical usage:
/// @code
///   PDEFilter filter(fes_filter, fes_control, opts);
///   filter.SetDiffusionCoeff(my_coeff);  // optional
///   filter.AddBoundaryCondition(1, 0.0); // optional
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

    /// @brief Add or replace coefficient-valued Dirichlet data.
    ///
    /// The boundary attribute is a one-based MFEM boundary attribute id.
    /// The coefficient is borrowed and must remain alive through Assemble().
    void AddBoundaryCondition(int boundary_attribute, Coefficient& coefficient);

    /// @brief Add or replace an internally owned constant Dirichlet value.
    void AddBoundaryCondition(int boundary_attribute, real_t value);

    /// @brief Remove all recorded Dirichlet boundary conditions.
    void ClearBoundaryConditions();

    /// @brief Assemble the system matrix and set up the AMG+PCG solver.
    ///
    /// Must be called exactly once, after SetDiffusionCoeff (if any).
    void Assemble();

    // -----------------------------------------------------------------------
    // mfem::Operator interface — true-dof Vectors
    // -----------------------------------------------------------------------

    /// @brief Forward filter: y_filt = F(x_ctrl).
    ///
    /// @param x  Control true-dof vector, size Width()  = ctrl TrueVSize.
    /// @param y  Filter  true-dof vector, size Height() = filt TrueVSize.
    void Mult(const Vector& x, Vector& y) const override;

    /// @brief Adjoint filter: y_ctrl = F^T(x_filt).
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

    /// Uneliminated filter matrix. Null before Assemble().
    HypreParMatrix* GetFullMatrix() const { return filter_full_mat_.get(); }

    /// Return the number of recorded boundary-attribute conditions.
    int GetNumBoundaryConditions() const
    { return static_cast<int>(boundary_conditions_.size()); }

    /// Return true if the assembled filter has essential boundary conditions.
    bool HasEssentialBoundaryConditions() const
    { return assembled_ && !boundary_conditions_.empty(); }

    /// Return true until Assemble() has completed.
    bool NeedsAssembly() const { return !assembled_; }

    /// Return the assembled essential true-DOF list.
    const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list_; }

    /// Return the assembled boundary-attribute marker.
    const Array<int>& GetBoundaryAttributeMarker() const
    { return bdr_attr_marker_; }

    /// Return prescribed boundary values in filter true-DOF layout.
    const Vector& GetEssentialTrueDofValues() const { return x_bc_; }

    MPI_Comm GetComm() const { return fes_filter_->GetComm(); }

private:
    struct BoundaryConditionEntry
    {
        int boundary_attribute = 0;
        Coefficient* coefficient = nullptr;
        std::unique_ptr<Coefficient> owned_coefficient;
    };

    void AssembleBilinearForm_();
    void AssembleMixedMass_();
    void BuildBoundaryValuesAndMarkers_();
    void SetupSolver_();
    void FormSystemRHS_(const Vector& rhs, Vector& system_rhs) const;
    void ZeroEssentialValues_(Vector& x) const;
    void CopyEssentialValues_(Vector& x) const;
    int MaxBoundaryAttribute_() const;
    void ValidateBoundaryAttribute_(int boundary_attribute) const;
    void RemoveBoundaryCondition_(int boundary_attribute);
    void CheckConvergence_(const Vector& b,
                           const Vector& x,
                           const char*   context) const;

    ParFiniteElementSpace* fes_filter_;
    ParFiniteElementSpace* fes_control_;

    PDEFilterOptions opts_;

    std::unique_ptr<ParBilinearForm>      filter_bf_;
    std::unique_ptr<ParMixedBilinearForm> mixed_mass_;
    std::unique_ptr<HypreParMatrix>       mixed_mass_mat_;
    std::unique_ptr<HypreParMatrix>       filter_full_mat_;
    std::unique_ptr<HypreParMatrix>       filter_mat_;
    std::unique_ptr<HypreParMatrix>       filter_eliminated_mat_;
    std::unique_ptr<HypreBoomerAMG>       amg_prec_;
    std::unique_ptr<HyprePCG>             solver_;

    std::vector<BoundaryConditionEntry> boundary_conditions_;
    Array<int> bdr_attr_marker_;
    Array<int> ess_tdof_list_;
    Vector x_bc_;

    struct DiffCoeff {
        Coefficient*       scalar = nullptr;
        VectorCoefficient* vector = nullptr;
        MatrixCoefficient* matrix = nullptr;
    } diff_;

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
