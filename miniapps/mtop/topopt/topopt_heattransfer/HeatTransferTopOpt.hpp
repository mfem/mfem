#ifndef HEATTRANSFER_OPT_HPP
#define HEATTRANSFER_OPT_HPP

#include "mfem.hpp"
#include "../topopt_transient/ObjectiveFunctional.hpp"     // TimeIntegratedObjective (J, dJ/du)
#include "../../pde_filter.hpp"
#include "../topopt_transient/ElastodynamicsSolver.hpp"
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

namespace mfem
{

// =============================================================================
// ABSTRACT BASE CLASS: TerminalObjective
// =============================================================================
// Interface for terminal objective functionals J = ∫_Ω j(u,T) dx
//
// Subclasses must implement:
//   - ComputeObjective: actually computes cost 
//   - ComputeObjectiveGradient: compute ∂J/∂u at one timestep (for adjoint)
//
class TerminalObjective
{
   protected:
   ParFiniteElementSpace *fespace;
   real_t cost;
   MPI_Comm comm;
   int myid;
   
   public:
   TerminalObjective(ParFiniteElementSpace *fes, MPI_Comm comm_)
      : fespace(fes), cost(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   virtual ~TerminalObjective() = default;

   void Reset() { cost = 0.0; }

   real_t GetObjective() const { return cost; }

   inline void ComputeObjective(const ParGridFunction &u)
   {
      (void)u;
      cost = 0.0;
   }


   /// Compute objective gradient ∂J/∂u (for adjoint)
   virtual void ComputeObjectiveGradient(const ParGridFunction &u, ParLinearForm &grad_form) = 0;
};

// =============================================================================
// Terminal L2 OBJECTIVE: minimize ∫ |u(t)|² dx in subdomain
// =============================================================================
class TerminalL2Objective : public TerminalObjective
{
   private:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;

   public:
   /// Borrow an externally-owned indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(&indicator) {}

   /// Take ownership of an indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator)) {}

   /// Backward-compatible constructor for legacy call sites.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : TerminalObjective(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr) {}

   void ComputeObjective(const ParGridFunction &u)
   {
      ConstantCoefficient zero(0.0);
      cost = u.ComputeL2Error(zero)*u.ComputeL2Error(zero);
   }

   virtual ~TerminalL2Objective() = default;

   /// Compute objective gradient ∂J/∂u (for adjoint) NEED TO FIX
   void ComputeObjectiveGradient(const ParGridFunction &u, ParLinearForm &grad_form) override
   {
      VectorGridFunctionCoefficient u_coef(&u);
      class ObjectiveGradientCoef : public VectorCoefficient
      {
      private:
         VectorGridFunctionCoefficient *u_cf;

      public:
         ObjectiveGradientCoef(int vdim, VectorGridFunctionCoefficient *uc)
            : VectorCoefficient(vdim), u_cf(uc) {}

         void Eval(Vector &V, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            u_cf->Eval(V, T, ip);
            // const real_t chi_val = chi->Eval(T, ip);
            V *= 2.0;
         }
      };

      ObjectiveGradientCoef grad_coef(u.VectorDim(), &u_coef);

      class HighOrderVectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, q_vec;
         VectorCoefficient &q;

      public:
         HighOrderVectorDomainLFIntegrator(VectorCoefficient &q_)
            : q(q_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = q.GetVDim();
            const int dof = el.GetDof();

            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);

               el.CalcPhysShape(T, shape);
               q.Eval(q_vec, T, ip);

               const real_t trans_weight = T.Weight();
               for (int k = 0; k < vdim; k++)
               {
                  const real_t coeff = ip.weight * trans_weight * q_vec(k);
                  for (int s = 0; s < dof; s++)
                  {
                     elvect(dof*k + s) += coeff * shape(s);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderVectorDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};


class Implicit_Solver : public Solver
{
private:
   HypreParMatrix &M, &S;
   HypreParMatrix *A;
   CGSolver linear_solver;
   real_t dt;
   SparseMatrix M_diag;
   MPI_Comm comm;
public:
   Implicit_Solver(HypreParMatrix &M_, HypreParMatrix &S_,
                   const ParFiniteElementSpace &fes, real_t &dt_, MPI_Comm comm_)
      : M(M_),
        S(S_),
        A(nullptr),
        comm(comm_),
        linear_solver(comm_),
        dt(dt_)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);

      M.GetDiag(M_diag);
      // Form initial operator A = M + dt*S so the linear solver has an operator
      A = Add(dt, S, 1.0, M);
      linear_solver.SetOperator(*A);
   }

   void SetTimeStep(real_t dt_)
   {
      real_t ddt = dt-dt_;

      // syncronize ddt across all processes
      // MPI_Comm comm = M.GetComm();
      int myrank;
      MPI_Comm_rank(comm, &myrank);
      MPI_Bcast(&ddt, 1, MPI_DOUBLE, 0, comm);

      real_t epsilon;
      epsilon = std::numeric_limits<real_t>::epsilon();
      // allow for some tolerance in the time stepping process
      epsilon*=10;

      if (fabs(ddt) > epsilon)
      {
         if (0==myrank)
         {
            std::cout << "Updating Implicit_Solver time step from " << dt
                 << " to " << dt_ << std::endl;
         }
         delete A;
         dt = dt_;
         // Form operator A = M + dt*S
         A = Add(dt, S, 1.0, M);
         linear_solver.SetOperator(*A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }

   void SetPreconditioner(Solver &precond)
   {
      linear_solver.SetPreconditioner(precond);
   }

   ~Implicit_Solver() override
   {
      delete A;
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of the advection-diffusion equation is (M + dt S) du/dt = Su - K u + b
    , where M and K are the mass and advection matrices, and b describes the
    flow on the boundary. In the case of IMEX evolution, the diffusion term is
    treated implicitly, and the advection term is treated explicitly.  */
class IMEXAdvectionDiffusionSolver : public TimeDependentOperator
{
   private:
   ParFiniteElementSpace *fespace;
   ParBilinearForm *M, *K, *S, *A;
   std::unique_ptr<HypreParMatrix> M_mat, S_mat, K_mat;
   ParLinearForm *b;
   std::unique_ptr<HypreParVector> b_vec;
   Solver *M_prec;
   CGSolver *M_solver;
   Implicit_Solver *implicit_solver;
   LORSolver<HypreBoomerAMG>* lor_solver;
   mutable ParGridFunction q_gf;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr_attr;
   Array<int> inflow_bdr;
   int true_size;
   MPI_Comm comm;
   //  std::unique_ptr<ODESolver> ode_solver;

   VectorFunctionCoefficient velocity_coeff;
   ConstantCoefficient diffusion_coeff;
   ForwardTrajectoryStorage *trajectory;
   TerminalObjective *objective;
   FunctionCoefficient q0;
   FunctionCoefficient inflow;
   // mutable int current_adjoint_step;
   real_t dt;
   ConstantCoefficient dt_diff_coeff;
   mutable Vector z;
   mutable Vector w;
   // real_t curr_time;
   public:
   IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes, VectorFunctionCoefficient &velocity_coeff, ConstantCoefficient &dt_diff_coeff, ConstantCoefficient &diffusion_coeff, FunctionCoefficient &inflow, Array<int> &ess_tdof_list_, Array<int> &ess_bdr_attr_, FunctionCoefficient &q0, real_t dt, MPI_Comm comm, ForwardTrajectoryStorage *traj = nullptr, TerminalObjective *obj = nullptr);
   void Mult1(const Vector &x, Vector &y) const;
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k);
   void Mult(const Vector &x, Vector &y) const override
   {
      if (TimeDependentOperator::EvalMode::ADDITIVE_TERM_1 == GetEvalMode())
      {
        Mult1(x,y);
      }
      else
      {
        mfem_error("TimeDependentOperator::Mult() is not overridden!");
      }
   }
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override
   {
      if (TimeDependentOperator::EvalMode::ADDITIVE_TERM_2 == GetEvalMode())
      {
         ImplicitSolve2(dt,x,k);
      }
      else
      {
         mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
      }
   }
   virtual void JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs) const;
   const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }
   //  void InitTimeStepping(); 
   //  void Step();

   void UpdateDt(real_t dt_real){dt = dt_real;}

   ParGridFunction& Getq() { return q_gf; }

    

   void SetTrajectory(ForwardTrajectoryStorage *traj) { trajectory = traj; }

   void SetObjective(TerminalObjective *obj) { objective = obj; }

   void ComputeObjectiveGradient(int step, Vector &grad_vec) const
   {
      grad_vec = 0.0;
      if (!objective || !trajectory) return;
      // Get the state variable;
      // if (!q_gf) return;
      // Set grid function from stored state
      // Compute ∂J_Ω/∂u = 2 χ_Ω̃ u (from ObjectiveFunctional)
      ParLinearForm grad_form(fespace);
      objective->ComputeObjectiveGradient(q_gf, grad_form);
      grad_form.ParallelAssemble(grad_vec);
    }

    // void SetAdjointTimestep(int step) { current_adjoint_step = step; }

    // /// Store current state in trajectory (call during forward march)
    // void StoreTrajectoryStep(int step, const Vector &state) const
    // {
    //     if (!trajectory) return;

    //     BlockVector bstate(const_cast<Vector&>(state), block_true_offsets);
    //     Vector u_true(bstate.GetBlock(0).GetData(), true_size);
    //     Vector v_true(bstate.GetBlock(1).GetData(), true_size);

    //     trajectory->Store(step, u_true, v_true, Kmat);
    // }

    // /// Initialize terminal condition for adjoint solve
    // /// From paper eq. 996-997: (A_N^+)^T η^N = q^N
    // /// For J_T = 0 (no terminal objective), η^N = 0
    // void InitializeTerminalAdjoint(Vector &eta_final) const
    // {
    //     eta_final = 0.0;

    //     // If terminal objective exists: η^N = ∂J_T/∂z^N
    //     // For our case J_T = 0, so terminal adjoint is zero
    //     // This would be modified if we have a terminal cost
    // }

    // /// Compute objective gradient at current timestep
    // /// Returns q^n = [∂J_Ω/∂u^n, ∂J_Ω/∂v^n]

    // Update Destructor
   virtual ~IMEXAdvectionDiffusionSolver()
   {
      delete implicit_solver;
      delete lor_solver;
      delete M_prec;
   }
};



IMEXAdvectionDiffusionSolver::IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes_, VectorFunctionCoefficient &velocity_coeff_, ConstantCoefficient &dt_diff_coeff_, ConstantCoefficient &diffusion_coeff_, FunctionCoefficient &inflow_, Array<int> &ess_tdof_list_, Array<int> &ess_bdr_attr_, FunctionCoefficient &q0_, real_t dt_, MPI_Comm comm_, ForwardTrajectoryStorage *traj_, TerminalObjective *obj_)
   : TimeDependentOperator(fes_.GetTrueVSize()), 
   fespace(&fes_), 
   velocity_coeff(velocity_coeff_), 
   diffusion_coeff(diffusion_coeff_), 
   dt_diff_coeff(dt_diff_coeff_),
   inflow(inflow_), 
   trajectory(traj_), 
   q0(q0_),
   objective(obj_),
   ess_bdr_attr(ess_bdr_attr_),
   ess_tdof_list(ess_tdof_list_),
   comm(comm_),
   z(height),
   w(height),
   dt(dt_)
{
   int order = fespace->GetOrder(0);
   real_t kappa = (order + 1)*(order + 1);
   const real_t sigma = -1.0;
   
   int myid = Mpi::WorldRank();
   //  fespace.GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);

   //  std::unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);
   //  *ode_solver = *ode_solver_up;

    // Form the Mass Integrator 
   M = new ParBilinearForm(fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   // Form the DG Conevection Matrix
   constexpr real_t alpha = -1.0;
   K = new ParBilinearForm(fespace);
   K->AddDomainIntegrator(new ConvectionIntegrator(velocity_coeff, alpha));
   K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_coeff, alpha));                                                       
   K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_coeff, alpha));
   // Form DG Stiffness Matrix
   S = new ParBilinearForm(fespace);
   S->AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
   S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diffusion_coeff, sigma, kappa));
   S->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diffusion_coeff, sigma, kappa));;
   // For the preconditioner - create billinear form corresponding to
   // operator (M + dt S)
   A = new ParBilinearForm(fespace);
   A->AddDomainIntegrator(new MassIntegrator);
   A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_coeff));
   A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma, kappa));
   A->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_diff_coeff, sigma, kappa));
   M->Assemble();
   K->Assemble();
   S->Assemble();
   A->Assemble();
   M->Finalize();
   K->Finalize();
   S->Finalize();
   A->Finalize();
   //  Array<int> inflow_bdr = ess_bdr_attr;
   //  inflow_bdr = 0;
   //  inflow_bdr[0] = 1;

   // Set up boundary markers for loading
   ParMesh *pmesh = fespace->GetParMesh();
   int max_bdr_attr = pmesh->bdr_attributes.Max();
   inflow_bdr.SetSize(max_bdr_attr);
   inflow_bdr = 0;
   inflow_bdr[0] = 1;
   
   b = new ParLinearForm(fespace);
   b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity_coeff, alpha), inflow_bdr);
   b_vec.reset(b->ParallelAssemble());


   //  A->Reset(A->ParallelAssemble(), true);
   M_mat.reset(M->ParallelAssemble());
   S_mat.reset(S->ParallelAssemble());
   K_mat.reset(K->ParallelAssemble());
   HypreSmoother *hypre_prec = new HypreSmoother(*M_mat, HypreSmoother::Jacobi);
   M_prec = hypre_prec;
   implicit_solver = new Implicit_Solver(*M_mat, *S_mat, *fespace, dt, comm);
   lor_solver = new LORSolver<HypreBoomerAMG>(*A, ess_tdof_list);
   lor_solver->GetSolver().SetSystemsOptions(fespace->GetVDim(), true);
   implicit_solver -> SetPreconditioner(*lor_solver);
   t = 0.0;
   // allocate temporary vectors used by Mult/ImplicitSolve
   //  z.SetSize(fespace.GetTrueVSize());
   //  w.SetSize(fespace.GetTrueVSize());
   q_gf.SetSpace(fespace);
   q_gf.ProjectCoefficient(q0);
   M_solver = new CGSolver(comm);
   M_solver->SetOperator(*M_mat);
   M_solver->SetPreconditioner(*M_prec);
   M_solver->iterative_mode = false;
   M_solver->SetRelTol(1e-9);
   M_solver->SetAbsTol(0.0);
   M_solver->SetMaxIter(100);
   M_solver->SetPrintLevel(0);
}

void IMEXAdvectionDiffusionSolver::Mult1(const Vector &x, Vector &y) const
{
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K_mat->Mult(x, z);
   z += *b_vec;
   M_solver->Mult(z, y);
}

void IMEXAdvectionDiffusionSolver::ImplicitSolve2(const real_t dt, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   S_mat->Mult(x, z);
   z *= -1.0;
   implicit_solver->SetTimeStep(dt);
   implicit_solver->Mult(z, k);
}

void IMEXAdvectionDiffusionSolver::JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs) const
{
   // Plain transpose of the forward RHS Jacobian:
   // G(u) = M^{-1} (K u + b)
   // lam_rhs = 0.0;

   // Adjoint RHS evaluation for discrete adjoint 
   // Jac(G) = M^{-1} K 
   // Jac(G)^T = K^{T} M^{-T} 
   K_mat->MultTranspose(lam, z);
   M_solver->Mult(z, lam_rhs);
}

// void IMEXAdvectionDiffusionSolver::InitTimeStepping()
// {
//     ode_solver->Init(*this);
// }

// void IMEXAdvectionDiffusionSolver::Step()
// {
//     HypreParVector *q_vec = q_gf.GetTrueDofs();
//     ode_solver->Step(*q_vec, t, dt);
//     q_gf = *q_vec;
// }
}
#endif 