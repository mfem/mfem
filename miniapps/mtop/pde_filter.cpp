#include "pde_filter.hpp"
#include <algorithm>   // std::max
#include <cstdio>      // std::snprintf
#include <cmath>       // std::sqrt

namespace toopt
{

// =============================================================================
//  File-local coefficient helpers (inside toopt namespace so the compiler
//  can resolve the mfem:: base class inheritance via `using namespace mfem`)
// =============================================================================

class ScaledMatrixCoefficient : public mfem::MatrixCoefficient
{
public:
    ScaledMatrixCoefficient(double scale, mfem::MatrixCoefficient& mc)
        : mfem::MatrixCoefficient(mc.GetHeight(), mc.GetWidth()),
          scale_(scale), mc_(mc) {}
    void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
    { mc_.Eval(M, T, ip); M *= scale_; }
private:
    double scale_; mfem::MatrixCoefficient& mc_;
};

class DiagVecAsMatrixCoefficient : public mfem::MatrixCoefficient
{
public:
    DiagVecAsMatrixCoefficient(double scale, mfem::VectorCoefficient& vc)
        : mfem::MatrixCoefficient(vc.GetVDim(), vc.GetVDim()),
          scale_(scale), vc_(vc) {}
    void Eval(mfem::DenseMatrix& M, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
    {
        mfem::Vector v(height); vc_.Eval(v, T, ip);
        M.SetSize(height, width); M = 0.0;
        for (int i = 0; i < height; ++i) M(i,i) = scale_ * v(i);
    }
private:
    double scale_; mfem::VectorCoefficient& vc_;
};

// =============================================================================
//  Constructor
// =============================================================================
PDEFilter::PDEFilter(ParFiniteElementSpace& fes_filter,
                     ParFiniteElementSpace& fes_control,
                     const PDEFilterOptions& opts)
    : Operator(fes_filter.GetTrueVSize(),   // height = filter output size
               fes_control.GetTrueVSize()), // width  = control input size
      fes_filter_(&fes_filter),
      fes_control_(&fes_control),
      opts_(opts)
{
    MFEM_VERIFY(opts_.filter_radius > 0.0,
                "PDEFilter: filter_radius must be positive");
    const FiniteElementCollection* fec = fes_filter_->FEColl();
    MFEM_VERIFY(fec && std::string(fec->Name()).rfind("H1", 0) == 0,
                "PDEFilter: fes_filter must be an H1 space");
    MFEM_VERIFY(fes_filter_->GetParMesh() == fes_control_->GetParMesh(),
                "PDEFilter: spaces must share the same ParMesh");
}

// =============================================================================
//  SetDiffusionCoeff overloads
// =============================================================================
void PDEFilter::SetDiffusionCoeff(Coefficient& c)
{
    MFEM_VERIFY(!assembled_, "PDEFilter: cannot change coefficient after Assemble()");
    diff_ = { &c, nullptr, nullptr };
}
void PDEFilter::SetDiffusionCoeff(VectorCoefficient& c)
{
    MFEM_VERIFY(!assembled_, "PDEFilter: cannot change coefficient after Assemble()");
    diff_ = { nullptr, &c, nullptr };
}
void PDEFilter::SetDiffusionCoeff(MatrixCoefficient& c)
{
    MFEM_VERIFY(!assembled_, "PDEFilter: cannot change coefficient after Assemble()");
    diff_ = { nullptr, nullptr, &c };
}

// =============================================================================
//  Assemble
// =============================================================================
void PDEFilter::Assemble()
{
    MFEM_VERIFY(!assembled_, "PDEFilter: Assemble() called more than once");
    AssembleBilinearForm_();
    AssembleMixedMass_();
    filter_bf_->Finalize();
    filter_mat_.reset(filter_bf_->ParallelAssemble());
    SetupSolver_();
    assembled_ = true;
}

// =============================================================================
//  AssembleBilinearForm_
// =============================================================================
void PDEFilter::AssembleBilinearForm_()
{
    const double r2 = opts_.filter_radius * opts_.filter_radius;
    filter_bf_ = std::make_unique<ParBilinearForm>(fes_filter_);

    // The DiffusionIntegrator stores a raw pointer to its coefficient and does
    // NOT take ownership.  We therefore store the dynamically-created wrapper
    // coefficient in diff_coeff_owned_ so it lives as long as filter_bf_.
    if (diff_.matrix)
    {
        diff_matrix_owned_.reset(new ScaledMatrixCoefficient(r2, *diff_.matrix));
        filter_bf_->AddDomainIntegrator(
            new DiffusionIntegrator(*diff_matrix_owned_));
    }
    else if (diff_.vector)
    {
        diff_matrix_owned_.reset(new DiagVecAsMatrixCoefficient(r2, *diff_.vector));
        filter_bf_->AddDomainIntegrator(
            new DiffusionIntegrator(*diff_matrix_owned_));
    }
    else if (diff_.scalar)
    {
        diff_scalar_owned_.reset(new mfem::ProductCoefficient(r2, *diff_.scalar));
        filter_bf_->AddDomainIntegrator(
            new DiffusionIntegrator(*diff_scalar_owned_));
    }
    else
    {
        diff_scalar_owned_.reset(new mfem::ConstantCoefficient(r2));
        filter_bf_->AddDomainIntegrator(
            new DiffusionIntegrator(*diff_scalar_owned_));
    }

    filter_bf_->AddDomainIntegrator(new MassIntegrator());
    filter_bf_->Assemble(0);
}

// =============================================================================
//  AssembleMixedMass_
// =============================================================================
void PDEFilter::AssembleMixedMass_()
{
    mixed_mass_ = std::make_unique<ParMixedBilinearForm>(fes_control_,
                                                          fes_filter_);
    mixed_mass_->AddDomainIntegrator(new MixedScalarMassIntegrator());
    mixed_mass_->Assemble();
    mixed_mass_->Finalize();
    mixed_mass_mat_.reset(mixed_mass_->ParallelAssemble());
}

// =============================================================================
//  SetupSolver_
// =============================================================================
void PDEFilter::SetupSolver_()
{
    amg_prec_ = std::make_unique<HypreBoomerAMG>(*filter_mat_);
    amg_prec_->SetPrintLevel(0);

    solver_ = std::make_unique<HyprePCG>(*filter_mat_);
    solver_->SetTol(opts_.solver_rtol);
    solver_->SetAbsTol(opts_.solver_atol);
    solver_->SetMaxIter(opts_.solver_maxiter);
    solver_->SetPrintLevel(opts_.print_level);
    solver_->SetPreconditioner(*amg_prec_);
}

// =============================================================================
//  CheckConvergence_
// =============================================================================
void PDEFilter::CheckConvergence_(const Vector& b,
                                   const Vector& x,
                                   const char*   context) const
{
    MPI_Comm comm = fes_filter_->GetComm();

    Vector res(b.Size());
    filter_mat_->Mult(x, res);
    res -= b;

    const double res_norm = std::sqrt(InnerProduct(comm, res, res));
    const double rhs_norm = std::sqrt(InnerProduct(comm, b,   b  ));
    const double tol = 100.0 * std::max(opts_.solver_atol,
                                         opts_.solver_rtol * rhs_norm);
    if (res_norm > tol)
    {
        int rank = 0; MPI_Comm_rank(comm, &rank);
        if (rank == 0)
        {
            char msg[512];
            std::snprintf(msg, sizeof(msg),
                          "%s: PCG solver did not converge.\n"
                          "  ||r|| = %.6e   tol = %.6e   ||b|| = %.6e",
                          context, res_norm, tol, rhs_norm);
            MFEM_WARNING(msg);
        }
    }
}

// =============================================================================
//  Mult  —  forward filter on true-dof Vectors
//
//  y = F(x):  y_filt = (r^2 K + M)^{-1} M_fc x_ctrl
// =============================================================================
void PDEFilter::Mult(const Vector& x, Vector& y) const
{
    MFEM_VERIFY(assembled_, "PDEFilter: call Assemble() before Mult()");
    MFEM_VERIFY(x.Size() == Width(),
                "PDEFilter::Mult: x.Size() != Width() (control TrueVSize)");

    y.SetSize(Height());

    // rhs = M_fc * x_ctrl
    Vector rhs(Height());
    mixed_mass_mat_->Mult(x, rhs);

    y = 0.0;
    solver_->Mult(rhs, y);
    CheckConvergence_(rhs, y, "PDEFilter::Mult");
}

// =============================================================================
//  MultTranspose  —  adjoint filter on true-dof Vectors
//
//  y = F^T(x):  y_ctrl = M_fc^T (r^2 K + M)^{-1} x_filt
// =============================================================================
void PDEFilter::MultTranspose(const Vector& x, Vector& y) const
{
    MFEM_VERIFY(assembled_, "PDEFilter: call Assemble() before MultTranspose()");
    MFEM_VERIFY(x.Size() == Height(),
                "PDEFilter::MultTranspose: x.Size() != Height() (filter TrueVSize)");

    // psi = (r^2 K + M)^{-1} x_filt
    Vector psi(Height());
    psi = 0.0;
    solver_->Mult(x, psi);
    CheckConvergence_(x, psi, "PDEFilter::MultTranspose");

    // y = M_fc^T * psi
    y.SetSize(Width());
    mixed_mass_mat_->MultTranspose(psi, y);
}

// =============================================================================
//  GridFunction convenience wrappers
// =============================================================================
void PDEFilter::Mult(const ParGridFunction& rho,
                     ParGridFunction& rho_tilde) const
{
    MFEM_VERIFY(rho.ParFESpace() == fes_control_,
                "PDEFilter::Mult: rho must live on the control FE space");
    MFEM_VERIFY(rho_tilde.ParFESpace() == fes_filter_,
                "PDEFilter::Mult: rho_tilde must live on the filter FE space");

    Vector x, y;
    rho.GetTrueDofs(x);
    Mult(x, y);
    rho_tilde.SetFromTrueDofs(y);
}

void PDEFilter::MultTranspose(const ParGridFunction& lambda,
                               ParGridFunction& sens_out) const
{
    MFEM_VERIFY(lambda.ParFESpace() == fes_filter_,
                "PDEFilter::MultTranspose: lambda must live on the filter FE space");
    MFEM_VERIFY(sens_out.ParFESpace() == fes_control_,
                "PDEFilter::MultTranspose: sens_out must live on the control FE space");

    Vector x, y;
    lambda.GetTrueDofs(x);
    MultTranspose(x, y);
    sens_out.SetFromTrueDofs(y);
}

} // namespace toopt
