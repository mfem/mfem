#ifndef MFEM_LINEAR_EVOLVER
#define MFEM_LINEAR_EVOLVER

#include "mfem.hpp"

namespace mfem
{

    /// Class that can handle implicit or explicit time marching of linear or
    /// nonlinear ODEs
    /// TODO: think about how to handle partial assebmly of residual jacobian and
    ///       stiffness matrices
    class EulerEvolver : public TimeDependentOperator
    {
    public:
        /// Serves as an base class for linear/nonlinear explicit/implicit time
        /// marching problems
        /// \param[in] mass - bilinear form for mass matrix (not owned)
        /// \param[in] res - nonlinear residual operator (not owned)
        /// \param[in] start_time - time to start integration from
        ///                         (important for time-variant sources)
        /// \param[in] type - solver type; explicit or implicit
        /// \note supports partial assembly of mass and stiffness matrices for
        ///       explicit time marching
        EulerEvolver(BilinearForm *mass,
                     NonlinearForm *res,
                     double start_time,
                     mfem::TimeDependentOperator::Type type = EXPLICIT);

        /// Perform the action of the operator: y = k = f(x, t), where k solves
        /// the algebraic equation F(x, k, t) = G(x, t) and t is the current time.
        /// Compute k = M^-1(R(x,t) + Kx + l)
        void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

        /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
        /// the current time t.
        /// Currently implemented for the implicit midpoit method
        void ImplicitSolve(const double dt, const mfem::Vector &x,
                           mfem::Vector &k) override;

        /// Variant of `mfem::ImplicitSolve` for entropy constrained systems
        /// \param[in] dt_stage - the full step size
        /// \param[in] dt - a partial step, `dt` < `dt_stage`.
        /// \param[in] x - baseline state
        /// \param[out] k - the desired slope
        /// \note This may need to be generalized further
        void ImplicitSolve(const double dt_stage, const double dt,
                           const mfem::Vector &x, mfem::Vector &k);

        /// Return a reference to the Jacobian of the combined operator
        /// \param[in] x - the current state
        mfem::Operator &GetGradient(const mfem::Vector &x) const override;

        void solveForState(){};

        virtual ~EulerEvolver();

    protected:
        /// pointer to mass bilinear form (not owned)
        BilinearForm *mass;
        /// pointer to nonlinear form (not owned)
        NonlinearForm *res;

/// solver for inverting mass matrix for explicit solves
/// \note supports partially assembled mass bilinear form
#ifdef MFEM_USE_SUITESPARSE
        // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
        UMFPackSolver mass_solver;
#else
        //mfem::CGSolver mass_solver;
        mfem::GMRESSolver mass_solver;
#endif
        /// preconditioner for inverting mass matrix
        //DSmoother mass_prec;
        Solver *mass_prec;

        /// Newton solver for implicit problems (not owned)
        mfem::NewtonSolver newton_solver;
        /// pointer-to-implementation idiom
        /// Hides implementation details of this operator, and because it's private,
        /// it doesn't pollute the mach namespace
        class SystemOperator;
        /// Operator that combines the linear/nonlinear spatial discretization with
        /// the load vector into one operator used for implicit solves
        std::unique_ptr<SystemOperator> combined_oper;
        /// TimeDependentOperator
        std::unique_ptr<mfem::TimeDependentOperator> evolver;
        /// work vectors
        mutable mfem::Vector x_work, r_work1, r_work2;

        /// sets the state and dt for the combined operator
        /// \param[in] dt - time increment
        /// \param[in] x - the current state
        /// \param[in] dt_stage - time step for full stage/step
        void setOperParameters(double dt, const mfem::Vector *x,
                               double dt_stage = -1.0);
    };

    class EulerEvolver::SystemOperator : public Operator
    {
    public:
        /// Nonlinear operator of the form that combines the mass, res, stiff,
        /// and load elements for implicit/explicit ODE integration
        /// \param[in] nonlinear_mass - nonlinear mass matrix operator (not owned)
        /// \param[in] mass - bilinear form for mass matrix (not owned)
        /// \param[in] res - nonlinear residual operator (not owned)
        /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
        /// \note The mfem::NewtonSolver class requires the operator's width and
        /// height to be the same; here we use `GetTrueVSize()` to find the process
        /// local height=width
        SystemOperator(BilinearForm *_mass, NonlinearForm *_res)
            : Operator(_mass->FESpace()->GetTrueVSize()),
              mass(_mass),
              res(_res), Jacobian(NULL),
              dt(0.0), x(NULL), x_work(width), r_work(height)
        {
        }

        /// Compute r = N(x + dt_stage*k,t + dt) - N(x,t) + M@k + R(x + dt*k,t) + K@(x+dt*k) + l
        /// (with `@` denoting matrix-vector multiplication)
        /// \param[in] k - dx/dt
        /// \param[out] r - the residual
        /// \note the signs on each operator must be accounted for elsewhere
        void Mult(const mfem::Vector &k, mfem::Vector &r) const override
        {
            r = 0.0;
            // x_work = x + dt*k = x + dt*dx/dt = x + dx
            add(1.0, *x, dt, k, x_work);
            res->Mult(x_work, r_work);
            r += r_work;
            mass->AddMult(k, r);
        }

        /// Compute J = grad(N(x + dt_stage*k)) + M + dt * grad(R(x + dt*k, t)) + dt * K
        /// \param[in] k - dx/dt
        mfem::Operator &GetGradient(const mfem::Vector &k) const override
        {

            SparseMatrix *jac = nullptr;

            // x_work = x + dt*k = x + dt*dx/dt = x + dx
            add(1.0, *x, dt, k, x_work);
            jac = Add(1.0, mass->SpMat(),
                      dt, *dynamic_cast<const SparseMatrix *>(&res->GetGradient(x_work)));
            return *jac;
        }

        /// Set current dt and x values - needed to compute action and Jacobian.
        /// \param[in] _dt - the step used to define where RHS is evaluated
        /// \praam[in] _x - current state
        /// \param[in] _dt_stage - the step for this entire stage/step
        /// \note `_dt` is the step usually assumed in mfem.  `_dt_stage` is needed
        /// by the nonlinear mass form and can be ignored if not needed.
        void setParameters(double _dt, const mfem::Vector *_x, double _dt_stage = -1.0)
        {
            dt = _dt;
            x = _x;
            dt_stage = _dt_stage;
        };

        ~SystemOperator() { delete Jacobian; };

    private:
        NonlinearForm *nonlinear_mass; // not used
        //DSmoother mass_prec;
        Solver *mass_prec;
        
        BilinearForm *mass;
        NonlinearForm *res;
        BilinearForm *stiff; // not used

        mutable SparseMatrix *Jacobian;

        double dt;
        double dt_stage;

        const mfem::Vector *x;

        mutable mfem::Vector x_work;
        mutable mfem::Vector r_work;
    };

    EulerEvolver::EulerEvolver(BilinearForm *_mass, NonlinearForm *_res,
                               double start_time, TimeDependentOperator::Type type)
        : TimeDependentOperator(_mass->FESpace()->GetTrueVSize(),
                                start_time, type),
          mass(_mass), res(_res), x_work(width), r_work1(mass->Height()), r_work2(height)
    {
        combined_oper.reset(new SystemOperator(_mass, _res));
        if (_mass != nullptr)
        {
#ifdef MFEM_USE_SUITESPARSE
            mass_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
            mass_solver.SetOperator(mass->SpMat());
#else
            //mass_prec = DSmoother(mass->SpMat());
            mass_prec = new DSmoother(1);
            mass_solver.SetPreconditioner(*mass_prec);
            mass_solver.SetOperator(mass->SpMat());
            mass_solver.iterative_mode = false;
            mass_solver.SetRelTol(1e-9);
            mass_solver.SetAbsTol(0.0);
            mass_solver.SetMaxIter(500);
            mass_solver.SetPrintLevel(-1);
#endif
        }
        newton_solver.iterative_mode = false;
        newton_solver.SetSolver(mass_solver);
        newton_solver.SetOperator(*combined_oper);
        newton_solver.SetPrintLevel(1); // print Newton iterations
        newton_solver.SetRelTol(1e-9);
        newton_solver.SetAbsTol(1e-11);
        newton_solver.SetMaxIter(1000);
    }

    EulerEvolver::~EulerEvolver() = default;

    void EulerEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
    {
        res->Mult(x, r_work1);
        mass_solver.Mult(r_work1, y);
        y *= -1.0;
    }

    void EulerEvolver::ImplicitSolve(const double dt, const Vector &x,
                                     Vector &k)
    {
        //cout << "ImplicitSolve is called " << endl;
        setOperParameters(dt, &x);
        Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
        k = 0.0;     // In case iterative mode is set to true
        newton_solver.Mult(zero, k);
        MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge!");
    }

    void EulerEvolver::ImplicitSolve(const double dt_stage, const double dt,
                                     const Vector &x, Vector &k)
    {
        setOperParameters(dt, &x, dt_stage);
        Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
        k = 0.0;     // In case iterative mode is set to true
        newton_solver.Mult(zero, k);
        MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge!");
    }

    mfem::Operator &EulerEvolver::GetGradient(const mfem::Vector &x) const
    {
        return combined_oper->GetGradient(x);
    }

    void EulerEvolver::setOperParameters(double dt, const mfem::Vector *x,
                                         double dt_stage)
    {
        combined_oper->setParameters(dt, x, dt_stage);
    }

} // namespace mfem
#endif
