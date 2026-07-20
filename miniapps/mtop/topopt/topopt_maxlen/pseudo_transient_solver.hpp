#pragma once

#include "mfem.hpp"
#include <cmath>
#include <memory>

namespace mfem
{

// Time dependent solver for the DG upwind advection system:
// du/dt = M^{-1} ( -K u + b )
class LinearEvolutionOperator : public TimeDependentOperator
{
private:
    const Operator *K, *Minv;
    const Vector *b;

    mutable Vector z;

public:
    LinearEvolutionOperator(const Operator &K_, const Operator &Minv_)
        : TimeDependentOperator(K_.Height()),
          K(&K_), Minv(&Minv_), b(nullptr), z(K_.Height())
    { }

    void SetSource(const Vector &b_) { b = &b_; }

    // y = M^{-1} ( -K x + b )
    void Mult(const Vector &x, Vector &y) const override;
};

// Pseudo-Transient Solver for solving the pure advection system with
// a pseudo time derivative. The solver solves for the following system:
//       M du/dt = - K   u + b,       u^{n+1} = u^n + du/dt * dt
// The solver is equipped with an adjoint solve:
//       M dλ/dt = - K^T λ + ∂j/∂u,   λ^{n+1} = λ^n + dλ/dt * dt
class PseudoTransientSolver : public Operator
{
private:
    const Operator *K, *Minv;
    MPI_Comm comm;

    const Vector *rhs = nullptr;
    const Vector *adj_rhs = nullptr;

    std::unique_ptr<ODESolver> ode;

    mutable int iter_count = 0;
    real_t dt = 0.1;
    real_t final_t = 3;
    real_t tol = 1e-6;

    // forward and backward march using the same scheme
    // stopping criterion is when relative rate of change below tolerance
    void MarchToSteadyState(LinearEvolutionOperator &evol, Vector &u) const;

public:
    PseudoTransientSolver(const Operator &K_, const Operator &Minv_, MPI_Comm comm_)
        : Operator(K_.Height()), K(&K_), Minv(&Minv_), comm(comm_)
    { }

    // construct the solver without Minv, set later
    PseudoTransientSolver(const Operator &K_, MPI_Comm comm_)
        : Operator(K_.Height()), K(&K_), Minv(nullptr), comm(comm_)
    { }

    // set the true vdof for the right hand side for forward/adjoint solves
    void SetRhs(const Vector &rhs_) { rhs = &rhs_; }
    void SetAdjointRhs(const Vector &adj_rhs_) { adj_rhs = &adj_rhs_; }

    // set the mass matrix inverse operator
    void SetMinv(const Operator &Minv_) { Minv = &Minv_; }

    void SetODESolver(std::unique_ptr<ODESolver> ode_) { ode = std::move(ode_); }

    // parameter values for the solver
    void SetTimeStep(const real_t dt_) { dt = dt_; }
    void SetTerminalTime(const real_t final_t_) { final_t = final_t_; }
    void SetTol(const real_t tol_) { tol = tol_; }

    int GetIterCount() const { return iter_count; }

    // Forward solve of the advection equation using the pseudo time scheme
    // computes:    u^{n+1} = u^n + dt*M^{-1} * (-K u + b)  until steady state
    void Mult(const Vector &x, Vector &y) const override;

    // Backward solve of the advection equation using the pseudo time scheme
    // computes:    λ^{n+1} = λ^n + dt*M^{-1} * (-K^T λ + b)  until steady state
    void MultTranspose(const Vector &x, Vector &y) const override;
};


// Calculates the thickness of the design over a ray vector field by solving an
// advection problem:
//                        v . grad(rho_a) = rho_p
// where rho_a on the outflow boundary gives the solution of the thickness measure.
// rho_p is the current filtered density. The steady states solves are performed
// using Mult() and MultTranspose(). Since the linear solve is ill-conditioned, we
// define FSolve() and ASolve(), which use a pseudo-transient approach.
class MaterialThicknessSolver : public Operator
{
private:
    ParMesh* design_mesh;       // mesh for design variable, can be a submesh
    ParMesh* eval_mesh;         // mesh for thickness evaluation
    bool same_mesh;

    ParFiniteElementSpace *design_fes;  // design fes for rho_filter
    ParFiniteElementSpace *sol_fes;     // rho_a fes
    ParFiniteElementSpace *src_fes;     // rho_p fes
    bool pa;        // boolean for partial assembly

    ParBilinearForm *K;                 // Bilinear form for the advection term
    OperatorHandle Kopt, Nopt;          // Operator for Stiffness and src Mass

    std::unique_ptr<TransposeOperator> Kt;      // transpose view of Kopt for the adjoint solve
    std::unique_ptr<HypreParMatrix> KoptT;      // explicit transpose matrix for the adjoint preconditioner

    std::unique_ptr<BlockILU> fwd_prec, adj_prec;           // preconditioners for the linear solves
    std::unique_ptr<GMRESSolver> fwd_gmres, adj_gmres;      // gmres solvers for linear solves

    DGMassInverse *Minv = nullptr;      // M^-1 operator for the pseudotime derivative

    PseudoTransientSolver *pt_solver;       // pseudotransient solver for this advection solve

    ParGridFunction rho_a;               // rho_a solution gridfunction
    mutable ParGridFunction rho_p;      // rho_p for appropriate source for this solve

    VectorCoefficient &v_cf;        // vector representing ray field

    Vector adj_rhs;
    Vector fwd_rhs;
    Vector lambda_tv;
    Vector sens_tv;

    // builds the true dof vector for the right hand side
    void BuildRhs(const Vector &design_tv, Vector &rhs_tv) const;

public:
    MaterialThicknessSolver(ParFiniteElementSpace &design_fes_,
                            ParFiniteElementSpace &eval_fes_,
                            VectorCoefficient &v_cf_,
                            bool pa_ = false);

    ~MaterialThicknessSolver();

    // returns the owned pseudo transient solver
    PseudoTransientSolver &GetSolver() { return *pt_solver; }

    const ParGridFunction &GetRhoA() const { return rho_a; }
    const ParGridFunction &GetRhoP() const { return rho_p; }

    Vector &GetAdjoint() { return lambda_tv; }
    Vector &GetSensitivity() { return sens_tv; }

    void SetRhs(const Vector &rhs_) { fwd_rhs = rhs_; }
    void SetAdjointRhs(const Vector &adj_rhs_) { adj_rhs = adj_rhs_; }

    void SetMinv(DGMassInverse &Minv_) { Minv = &Minv_; pt_solver->SetMinv(Minv_); }

    // assemble the stiffness bilinear form and the mixed mass operator
    void Assemble();

    // assembles and caches the GMRES solvers (and preconditioners) used by
    // Mult() and MultTranspose()
    void AssembleLinearSolver();

    // Mult() and MultTranspose() perform linear solve using GMRES
    void Mult(const Vector &x, Vector &y) const override;
    void MultTranspose(const Vector &x, Vector &y) const override;

    // FSolve() and Asolve() uses the pseudoTransientSovler
    void FSolve();
    void ASolve();
};

} // namespace mfem
