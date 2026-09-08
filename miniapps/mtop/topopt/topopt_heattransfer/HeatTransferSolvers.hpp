#ifndef HT_SOLVERS_HPP
#define HT_SOLVERS_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "ObjFunc.hpp"     // TimeIntegratedObjective (J, dJ/du)
#include "HeatTransferLinForms.hpp"
#include "../../pde_filter.hpp"
#include "diffusion_mass_solver.hpp"
#include "diffusion_mass_solver.cpp"

namespace mfem
{

static int GlobalMax(MPI_Comm comm, int value)
{
   int global = 0;
   MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, comm);
   return global;
}

// =============================================================================
// FORWARD TRAJECTORY STORAGE
// =============================================================================
// Storage for forward state needed by adjoint solver
struct ForwardTrajectoryStorage
{
   Array<Vector*> q_traj;       // Displacement at each timestep

   int num_steps;
   bool storage_enabled;

   ForwardTrajectoryStorage(int n) : num_steps(n), storage_enabled(false)
   {
      q_traj.SetSize(n);

      for (int i = 0; i < n; i++)
      {
         q_traj[i] = nullptr;
      }
   }

   void EnableStorage() { storage_enabled = true; }

   void Store(int step, const Vector &q)
   {
      if (!storage_enabled) return;

      if (step >= num_steps) return;

      if (q_traj[step]) delete q_traj[step];

      q_traj[step] = new Vector(q);
   }

   Vector Get(int step){return *q_traj[step]; }

   real_t Size(){return q_traj.Size();}

   ~ForwardTrajectoryStorage()
   {
      for (int i = 0; i < num_steps; i++)
      {
         delete q_traj[i];
      }
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
      linear_solver.SetPrintLevel(-1);

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
            // std::cout << "Updating Implicit_Solver time step from " << dt 
            //      << " to " << dt_ << std::endl;
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
      // int myrank;
      // MPI_Comm_rank(comm, &myrank);
      // std::cout << "My rank " << myrank << std::endl;
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


class TopOptTimeDependentOperator : public TimeDependentOperator
{
   public:
   TopOptTimeDependentOperator(int n);
   virtual void AdjointMult(const Vector &lam, Vector &lam_rhs) const = 0;
   virtual void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) = 0;  
   virtual void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) = 0;
   virtual void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) = 0;
   virtual void ExplicitMultCoupledStateGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdu, ParFiniteElementSpace &vfes) = 0;
   virtual void ImplicitSolveCoupledStateGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdu, ParFiniteElementSpace &vfes) = 0;  
};

TopOptTimeDependentOperator::TopOptTimeDependentOperator(int n) : TimeDependentOperator(n)
{}




/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of the advection-diffusion equation is (M + dt S) du/dt = Su - K u + b
    , where M and K are the mass and advection matrices, and b describes the
    flow on the boundary. In the case of IMEX evolution, the diffusion term is
    treated implicitly, and the advection term is treated explicitly.  */
class IMEXAdvectionDiffusionSolver : public TopOptTimeDependentOperator
{
    protected:
    // Finite Element Spaces, Operators, and Solvers
    ParFiniteElementSpace *fespace;
    ParFiniteElementSpace *filter_fes;
    ParBilinearForm *M, *K, *S, *A; 
    std::unique_ptr<HypreParMatrix> M_mat, S_mat, K_mat;
    mutable ParLinearForm *b;
    mutable std::unique_ptr<HypreParVector> b_vec;
    Solver *M_prec;
    CGSolver *M_solver;
    Implicit_Solver *implicit_solver;
    LORSolver<HypreBoomerAMG>* lor_solver;
    real_t kappa;

    // Solution Storage
    GridFunctionCoefficient q0; 
    mutable ParGridFunction q_gf;
    ForwardTrajectoryStorage *trajectory;

    // Boundary Stuff
    Array<int> ess_bdr_attr;
    Array<int> ess_tdof_list;
    mutable Array<int> inflow_bdr_attr;

    // Design Optimization
    mutable ParGridFunction rho_tilde;
    mutable Vector design_gradient;
    HeatTransferObjectiveFunction *objective;
    SIMPCoefficient SIMP_cf;

    // Time Integration Related
    real_t dt;
    real_t t_final;
    int current_step;

    // PDE Coefficients
    real_t raw_diff_term;
    mutable VectorGridFunctionCoefficient v_base;
    mutable FunctionCoefficient raw_inflow;
    real_t dt_diff_term;
    
    // misc
    int true_size;
    MPI_Comm comm;

    // Helpers
    mutable Vector z;
    mutable Vector w;
    int problem_type;
    






    public:
    IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes, 
        FunctionCoefficient &raw_inflow, 
        VectorGridFunctionCoefficient &v_base, 
        real_t &dt_diff_term, 
        real_t &raw_diff_term,  
        GridFunctionCoefficient &q0, 
        ParGridFunction &rho_tilde, 
        real_t dt, 
        real_t t_final, 
        SIMPCoefficient SIMP_cf, 
        MPI_Comm comm, 
        HeatTransferObjectiveFunction *obj);

    IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes, 
        FunctionCoefficient &raw_inflow, 
        VectorGridFunctionCoefficient &v_base, 
        real_t &dt_diff_term, 
        real_t &raw_diff_term,  
        GridFunctionCoefficient &q0, 
        ParGridFunction &rho_tilde, 
        real_t dt, 
        real_t t_final, 
        SIMPCoefficient SIMP_cf, 
        MPI_Comm comm, 
        Array<int> &ess_bdr_attr,
        Array<int> &inflow_bdr_attr,
        HeatTransferObjectiveFunction *obj);
    
    void InitializeInjectionProblem();
    void InitializeFlowProblem();
    void Mult1(const Vector &x, Vector &y) const;
    void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k);
    void JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs) const;
    void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
    void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override; 
    void ExplicitMultCoupledStateGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdu, ParFiniteElementSpace &vfes) override;
    void ImplicitSolveCoupledStateGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdu, ParFiniteElementSpace &vfes) override; 
    void AdjointImplicitSolve2(const real_t dt, const Vector &lam, Vector &k);
    void Mult(const Vector &x, Vector &y) const override
    {
        Mult1(x,y);
    }
    void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override 
    {
        ImplicitSolve2(dt_pass,x,k);
    }
    void AdjointMult(const Vector &lam, Vector &lam_rhs) const override
    {
        JacobianMult1Transpose(lam, lam_rhs);
    }
    void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override
    {
        AdjointImplicitSolve2(dt_pass,lam,k);
    }


    const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }

    void UpdateDt(real_t dt_real)
    {
        MPI_Bcast(&dt_real, 1, MPI_DOUBLE, 0, comm);
        dt = dt_real;
    }

    void SetInflowBdrAttr(Array<int> &inflow_bdr) {inflow_bdr_attr = inflow_bdr;}

    ParGridFunction& Getq() { return q_gf; }


    void Updateq(ParGridFunction &new_q_gf) {q_gf = new_q_gf;}

        

    void SetTrajectory(ForwardTrajectoryStorage *traj) { trajectory = traj; }

    // void SetObjective(HeatTransferObjectiveFunction obj) { objective = obj; }

    void StoreTraj(int step, Vector &q_vec){trajectory->Store(step, q_vec);}

    void GetTraj(int step, Vector &q_vec){q_vec = trajectory->Get(step);}

    void SetStep(int new_step){current_step = new_step;}

    int GetStep(){return current_step;}

    Vector GetDesignGrad(){return design_gradient;}


    void ComputeObjectiveGradient(Vector &grad_vec) const
    {
        grad_vec = 0.0;
        if (!objective || !trajectory) return;
        // Get the state variable;
        // if (!q_gf) return;
        // Set grid function from stored state
        // Compute ∂J_Ω/∂u = 2 χ_Ω̃ u (from ObjectiveFunctional)
        ParLinearForm grad_form(fespace);
        // objective->ComputeObjectiveGradient(q_gf, grad_form);
        // grad_form.ParallelAssemble(grad_vec);
        }

        // Update Destructor
    virtual ~IMEXAdvectionDiffusionSolver()
    {
        delete implicit_solver;
        delete lor_solver;
        delete M_prec;
        delete M_solver;
        delete trajectory;
        delete M;
        delete K;
        delete S;
        delete A;
        delete b;
    }
};




IMEXAdvectionDiffusionSolver::IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes_, 
        FunctionCoefficient &raw_inflow_, 
        VectorGridFunctionCoefficient &v_base_, 
        real_t &dt_diff_term_, 
        real_t &raw_diff_term_,  
        GridFunctionCoefficient &q0_, 
        ParGridFunction &rho_tilde_, 
        real_t dt_, 
        real_t t_final_, 
        SIMPCoefficient SIMP_cf_, 
        MPI_Comm comm_, 
        HeatTransferObjectiveFunction *obj = nullptr)
   : TopOptTimeDependentOperator(fes_.GetTrueVSize()), 
   fespace(&fes_), 
   dt_diff_term(dt_diff_term_),
   q0(q0_),
   objective(obj),
   comm(comm_),
   v_base(v_base_),
   z(fes_.GetTrueVSize()),
   w(fes_.GetTrueVSize()),
   rho_tilde(rho_tilde_),
   t_final(t_final_),
   raw_diff_term(raw_diff_term_),
   raw_inflow(raw_inflow_),
   dt(dt_),
   SIMP_cf(SIMP_cf_)
{
   int order = fespace->GetOrder(0);
   kappa = (order + 1)*(order + 1);
   int myid = Mpi::WorldRank();
   ParMesh *pmesh = fespace->GetParMesh();

   rho_tilde.ExchangeFaceNbrData();
   
   t = 0.0;

   q_gf.SetSpace(fespace);
   q_gf.ProjectCoefficient(q0);
   q_gf.ExchangeFaceNbrData();

   filter_fes = rho_tilde.ParFESpace();
   design_gradient.SetSize(filter_fes->GetTrueVSize());
   design_gradient = 0.0;

   int n_steps = (int)ceil(t_final / dt);
   trajectory = new ForwardTrajectoryStorage(n_steps);
   trajectory->EnableStorage();
   Vector q_vec = q_gf;
   trajectory->Store(0, q_vec);
   problem_type = 0;
}

IMEXAdvectionDiffusionSolver::IMEXAdvectionDiffusionSolver(ParFiniteElementSpace &fes_, 
        FunctionCoefficient &raw_inflow_, 
        VectorGridFunctionCoefficient &v_base_, 
        real_t &dt_diff_term_, 
        real_t &raw_diff_term_,  
        GridFunctionCoefficient &q0_, 
        ParGridFunction &rho_tilde_, 
        real_t dt_, 
        real_t t_final_, 
        SIMPCoefficient SIMP_cf_, 
        MPI_Comm comm_, 
        Array<int> &ess_bdr_attr_,
        Array<int> &inflow_bdr_attr_,
        HeatTransferObjectiveFunction *obj = nullptr)
        : TopOptTimeDependentOperator(fes_.GetTrueVSize()), 
   fespace(&fes_), 
   dt_diff_term(dt_diff_term_),
   q0(q0_),
   objective(obj),
   comm(comm_),
   v_base(v_base_),
   z(fes_.GetTrueVSize()),
   w(fes_.GetTrueVSize()),
   rho_tilde(rho_tilde_),
   t_final(t_final_),
   raw_diff_term(raw_diff_term_),
   raw_inflow(raw_inflow_),
   dt(dt_),
   SIMP_cf(SIMP_cf_),
   ess_bdr_attr(ess_bdr_attr_),
   inflow_bdr_attr(inflow_bdr_attr_)
{
   int order = fespace->GetOrder(0);
   kappa = (order + 1)*(order + 1);
   int myid = Mpi::WorldRank();
   ParMesh *pmesh = fespace->GetParMesh();

   rho_tilde.ExchangeFaceNbrData();
   
   t = 0.0;

   q_gf.SetSpace(fespace);
   q_gf.ProjectCoefficient(q0);
   q_gf.ExchangeFaceNbrData();

   filter_fes = rho_tilde.ParFESpace();
   design_gradient.SetSize(filter_fes->GetTrueVSize());
   design_gradient = 0.0;

   int n_steps = (int)ceil(t_final / dt);
   trajectory = new ForwardTrajectoryStorage(n_steps);
   trajectory->EnableStorage();
   Vector q_vec = q_gf;
   trajectory->Store(0, q_vec);
   problem_type = 0;

   fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
   fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);   

}

void IMEXAdvectionDiffusionSolver::InitializeInjectionProblem()
{   
    problem_type = 1;
    const real_t sigma = -1.0;
    M = new ParBilinearForm(fespace);
    M->AddDomainIntegrator(new MassIntegrator());
    GridFunctionCoefficient rho_til_cf(&rho_tilde);

    // Form the DG Conevection Matrix
    constexpr real_t alpha = -1.0;
    K = new ParBilinearForm(fespace);
    K->AddDomainIntegrator(new ConvectionIntegrator(v_base, alpha));
    K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(v_base, alpha));                                                       
    
    // Form DG Stiffness Matrix
    S = new ParBilinearForm(fespace);
    ConstantCoefficient raw_diff_cf(raw_diff_term);
    S->AddDomainIntegrator(new DiffusionIntegrator(raw_diff_cf));
    S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(raw_diff_cf, sigma, kappa));

    // For the preconditioner - create billinear form corresponding to
    // operator (M + dt S)
    A = new ParBilinearForm(fespace);
    ConstantCoefficient dt_diff_cf(dt_diff_term);
    A->AddDomainIntegrator(new MassIntegrator);
    A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_cf));
    A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_cf, sigma, kappa));

    M->Assemble();
    K->Assemble();
    S->Assemble();
    A->Assemble();
    M->Finalize();
    K->Finalize();
    S->Finalize();
    A->Finalize();

    raw_inflow.SetTime(0.0);
    b = new ParLinearForm(fespace);
    ProductCoefficient inflow(rho_til_cf, raw_inflow);
    b->AddDomainIntegrator(new DomainLFIntegrator(inflow));
    b->Assemble();
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
    lor_solver->GetSolver().SetPrintLevel(-1);
    implicit_solver -> SetPreconditioner(*lor_solver);

    M_solver = new CGSolver(comm);
    M_solver->SetOperator(*M_mat);
    M_solver->SetPreconditioner(*M_prec);
    M_solver->iterative_mode = false;
    M_solver->SetRelTol(1e-13);
    M_solver->SetAbsTol(0.0);
    M_solver->SetMaxIter(100);
    M_solver->SetPrintLevel(0);
}

void IMEXAdvectionDiffusionSolver::InitializeFlowProblem()
{ 
   // Boundary Conditions   
    if (ess_bdr_attr.Size() == 0)
    {
      ess_bdr_attr.SetSize(fespace->GetParMesh()->bdr_attributes.Max());   
      ess_bdr_attr = 0;   
      fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
      fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);  
    } 
    if (inflow_bdr_attr.Size() == 0)
    {
      inflow_bdr_attr.SetSize(fespace->GetParMesh()->bdr_attributes.Max()); 
      inflow_bdr_attr = 0;
      inflow_bdr_attr[1] = 1;    
    }
     
    problem_type = 2;
    const real_t sigma = -1.0;
    M = new ParBilinearForm(fespace);
    M->AddDomainIntegrator(new MassIntegrator());
    GridFunctionCoefficient rho_til_cf(&rho_tilde);

    // Form the DG Conevection Matrix
    constexpr real_t alpha = -1.0;
    ScalarVectorProductCoefficient velocity_cf(SIMP_cf, v_base);   
    K = new ParBilinearForm(fespace);
    K->AddDomainIntegrator(new ConvectionIntegrator(velocity_cf, alpha));
    K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha));                                                       
    K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha), inflow_bdr_attr);
    
    // Form DG Stiffness Matrix
    ProductCoefficient diff_cf(raw_diff_term, SIMP_cf);
    S = new ParBilinearForm(fespace);
    S->AddDomainIntegrator(new DiffusionIntegrator(diff_cf));
    S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_cf, sigma, kappa));

    // For the preconditioner - create billinear form corresponding to
    // operator (M + dt S)
    ProductCoefficient dt_diff_cf(dt_diff_term, SIMP_cf); 
    A = new ParBilinearForm(fespace);
    A->AddDomainIntegrator(new MassIntegrator);
    A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_cf));
    A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_cf, sigma, kappa));

    M->Assemble();
    K->Assemble();
    S->Assemble();
    A->Assemble();
    M->Finalize();
    K->Finalize();
    S->Finalize();
    A->Finalize();


    b = new ParLinearForm(fespace);
    b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(raw_inflow, velocity_cf, alpha), inflow_bdr_attr);
    b->Assemble();
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
    lor_solver->GetSolver().SetPrintLevel(-1);
    implicit_solver -> SetPreconditioner(*lor_solver);

    M_solver = new CGSolver(comm);
    M_solver->SetOperator(*M_mat);
    M_solver->SetPreconditioner(*M_prec);
    M_solver->iterative_mode = false;
    M_solver->SetRelTol(1e-13);
    M_solver->SetAbsTol(0.0);
    M_solver->SetMaxIter(100);
    M_solver->SetPrintLevel(0);
}

void IMEXAdvectionDiffusionSolver::Mult1(const Vector &x, Vector &y) const
{
   int myrank;
   MPI_Comm_rank(comm, &myrank);
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K_mat->Mult(x, z);
   z += *b_vec;
   M_solver->Mult(z, y);

   // raw_inflow.SetTime(t);
   // GridFunctionCoefficient rho_til_cf(&rho_tilde);
   // ProductCoefficient inflow(rho_til_cf, raw_inflow);
   // //b->Update();
   // b = new ParLinearForm(fespace);
   // b->AddDomainIntegrator(new DomainLFIntegrator(inflow));
   // b->Assemble();
   // b_vec.reset(b->ParallelAssemble());
}

void IMEXAdvectionDiffusionSolver::ImplicitSolve2(const real_t dt_pass, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");

   int myrank;
   MPI_Comm_rank(comm, &myrank);
   z = 0.0;
   S_mat->Mult(x, z);
   z *= -1.0;
   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(z, k);
}

void IMEXAdvectionDiffusionSolver::AdjointImplicitSolve2(const real_t dt_pass, const Vector &lam, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");

   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(lam, z);
   z *= -1.0;
   S_mat->Mult(z, k);
}

void IMEXAdvectionDiffusionSolver::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
{
   MFEM_VERIFY(implicit_solver != NULL, "Implicit time integration is not supported with partial assembly");
   implicit_solver->SetTimeStep(dt);
   if (problem_type == 1)
   {
      //dfdrho_tilde = 0.0; 
      // No dependence on rho, do nothing.
   }
   else if (problem_type == 2)
   { 
      //lam A^{-1} dS/drho A^{-1} S q
      Vector k_d(dual_vector.Size()); 
      Vector y(dual_vector.Size());
      Vector u(x.Size());
      implicit_solver->Mult(dual_vector, w); // w = A^{-1} lam, A is self adjoint
      //Vector q_vec = trajectory->Get(current_step-1);
      M_mat->Mult(x, u);
      implicit_solver->Mult(u, y); // y = A^{-1}S q
      ParLinearForm stiff_lf1(filter_fes); 
      ParGridFunction w_gf(fespace);
      ParGridFunction y_gf(fespace);
      w_gf.SetFromTrueDofs(w);
      y_gf.SetFromTrueDofs(y);
      rho_tilde.ExchangeFaceNbrData();
      w_gf.ExchangeFaceNbrData();
      y_gf.ExchangeFaceNbrData();
      stiff_lf1.AddDomainIntegrator(new DGStiffnessDesignLFIntegrator(rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
      stiff_lf1.AddInteriorFaceIntegrator(new DGStiffnessDesignLFIntegrator(rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
      stiff_lf1.Assemble();
      std::unique_ptr<HypreParVector> stiff_vec1(stiff_lf1.ParallelAssemble());   
      dfdrho_tilde.Add(a, *stiff_vec1);
      // design_gradient.Add(dt, *stiff_vec1); 
   }
   else{MFEM_ABORT("Unknown Problem Type (Design Gradient): " << problem_type);}
}

void IMEXAdvectionDiffusionSolver::ImplicitSolveCoupledStateGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdu, ParFiniteElementSpace &vfes)
{
   MFEM_VERIFY(implicit_solver != NULL, "Implicit time integration is not supported with partial assembly");
   implicit_solver->SetTimeStep(dt);
   if (problem_type == 1)
   {
      //dfdrho_tilde = 0.0; 
      // No dependence on rho, do nothing.
   }
   else if (problem_type == 2)
   { 
      // no dependence on u, do nothing
   }
   else{MFEM_ABORT("Unknown Problem Type (Design Gradient): " << problem_type);}
}


void IMEXAdvectionDiffusionSolver::JacobianMult1Transpose(const Vector &lam, Vector &lam_rhs) const
{
   // Plain transpose of the forward RHS Jacobian:
   // G(u) = M^{-1} (K u + b)
   // lam_rhs = 0.0;
   // Adjoint RHS evaluation for discrete adjoint 
   // Jac(G) = M^{-1} K 
   // Jac(G)^T = K^{T} M^{-T} 
   z = 0.0;
   M_solver->Mult(lam, z);
   K_mat->MultTranspose(z, lam_rhs);
}

void IMEXAdvectionDiffusionSolver::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
{
   // Update the design gradient
   M_solver->Mult(dual_vector, w);
   // Vector q_vec = trajectory->Get(current_step-1);
   // std::cout<<"current step = "<<current_step << std::endl;
   // Vector wf(filter_fes->GetTrueVSize()), qf(filter_fes->GetTrueVSize());
   // Mixed_Mass_mat->Mult(w, wf);
   // Mixed_Mass_mat->Mult(q_vec, qf);

   ParGridFunction lam_gf(fespace);
   lam_gf.SetFromTrueDofs(w);
   ParGridFunction qq_gf(fespace);
   qq_gf.SetFromTrueDofs(x);
   rho_tilde.ExchangeFaceNbrData();
   lam_gf.ExchangeFaceNbrData();
   qq_gf.ExchangeFaceNbrData();

   if (problem_type == 1)
   {
      raw_inflow.SetTime(t);
      ParLinearForm dom_flow_lf(filter_fes);
      dom_flow_lf.AddDomainIntegrator(new DomainDesignLFIntegrator(lam_gf, raw_inflow));
      dom_flow_lf.Assemble();
      std::unique_ptr<HypreParVector> dom_flow_vec(dom_flow_lf.ParallelAssemble());
      //design_gradient.Add(-dt, *dom_flow_vec);
      dgdrho_tilde.Add(-dt_pass, *dom_flow_vec);
   }
   else if (problem_type == 2)
   {
      ParLinearForm adv_lf(filter_fes);
      adv_lf.AddDomainIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base, SIMP_cf));
      adv_lf.AddBdrFaceIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base, SIMP_cf), inflow_bdr_attr);
      adv_lf.AddInteriorFaceIntegrator(new DGAdvectionDesignLFIntegrator(rho_tilde, qq_gf, lam_gf, v_base, SIMP_cf));
      adv_lf.Assemble();
      std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
      dgdrho_tilde.Add(-dt_pass, *adv_vec);
      //design_gradient.Add(-dt, *adv_vec);
      ParLinearForm bdr_flow_lf(filter_fes);
      bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowDesignLFIntegrator(rho_tilde, lam_gf, raw_inflow, v_base, SIMP_cf),inflow_bdr_attr);
      bdr_flow_lf.Assemble();
      std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
      dgdrho_tilde.Add(dt_pass, *bdr_flow_vec);
      //design_gradient.Add(dt, *bdr_flow_vec);
   }
   else{MFEM_ABORT("Unknown Problem Type (Design Gradient): " << problem_type);}
}

void IMEXAdvectionDiffusionSolver::ExplicitMultCoupledStateGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdu, ParFiniteElementSpace &vfes)
{
   // Update the design gradient
   M_solver->Mult(dual_vector, w);
   // Vector q_vec = trajectory->Get(current_step-1);
   // std::cout<<"current step = "<<current_step << std::endl;
   // Vector wf(filter_fes->GetTrueVSize()), qf(filter_fes->GetTrueVSize());
   // Mixed_Mass_mat->Mult(w, wf);
   // Mixed_Mass_mat->Mult(q_vec, qf);

   ParGridFunction lam_gf(fespace);
   lam_gf.SetFromTrueDofs(w);
   ParGridFunction qq_gf(fespace);
   qq_gf.SetFromTrueDofs(x);
   rho_tilde.ExchangeFaceNbrData();
   lam_gf.ExchangeFaceNbrData();
   qq_gf.ExchangeFaceNbrData();

   if (problem_type == 1)
   {

   }
   else if (problem_type == 2)
   {
      ParLinearForm adv_lf(&vfes);
      if(Mpi::Root()){std::cout<<"adv lf norm pre = " << adv_lf.Norml2() << std::endl;}
      if(Mpi::Root()){std::cout<<"pre lin form q = " << qq_gf.Norml2() << std::endl;}
      if(Mpi::Root()){std::cout<<"pre lin form l = " << lam_gf.Norml2() << std::endl;}
      adv_lf.AddDomainIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, qq_gf, lam_gf, SIMP_cf, v_base));
      if(Mpi::Root()){std::cout<<"adv lf norm pre = " << adv_lf.Norml2() << std::endl;}
      adv_lf.AddBdrFaceIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, qq_gf, lam_gf, SIMP_cf, v_base), inflow_bdr_attr);
      if(Mpi::Root()){std::cout<<"adv lf norm pre = " << adv_lf.Norml2() << std::endl;}
      adv_lf.AddInteriorFaceIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, qq_gf, lam_gf, SIMP_cf, v_base));
      adv_lf.Assemble();
      if(Mpi::Root()){std::cout<<"adv lf norm post = " << adv_lf.Norml2() << std::endl;}
      std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
      if(Mpi::Root()){std::cout<<"adv portion = " << adv_vec->Norml2() << std::endl;}
      // for(int idx = 0; idx < adv_vec->Size(); idx++)
      // {
      //    if(Mpi::Root()){std::cout<<"idx = " << idx << ", adv_vec val = " << (*adv_vec)(idx) << std::endl;}
      // }
      dgdu.Add(-dt_pass, *adv_vec);
      //design_gradient.Add(-dt, *adv_vec);
      ParLinearForm bdr_flow_lf(&vfes);
      bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowVelocityGradientLFIntegrator(rho_tilde, lam_gf, raw_inflow, v_base, SIMP_cf), inflow_bdr_attr);
      bdr_flow_lf.Assemble();
      std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
      if(Mpi::Root()){std::cout<<"inflow portion = " << bdr_flow_vec->Norml2() << std::endl;}
      dgdu.Add(dt_pass, *bdr_flow_vec);
      //design_gradient.Add(dt, *bdr_flow_vec);
   }
   else{MFEM_ABORT("Unknown Problem Type (Design Gradient): " << problem_type);} 
}

class TopOptSteadyStateOperator : public Operator
{
   public:
   TopOptSteadyStateOperator(int n);
   TopOptSteadyStateOperator(int h, int w);
   virtual void AdjointSolve(Vector &lam_rhs_update) = 0;
   virtual void AddDesignGradient(Vector &dgdrho_tilde, real_t dt) = 0; 
};

TopOptSteadyStateOperator::TopOptSteadyStateOperator(int n) : Operator(n)
{}

TopOptSteadyStateOperator::TopOptSteadyStateOperator(int h, int w) : Operator(h, w)
{}

class TopOptSteadyStateStokesSolver : public TopOptSteadyStateOperator
{
   protected:
   ParFiniteElementSpace *V_fes;
   ParFiniteElementSpace *P_fes;
   ParFiniteElementSpace *filter_fes;
   BrinkmanCoefficient brinkman_cf;
   bool pa;
   BrinkmanStokesSolver *brinkman_stokes_solver;
   ConstantCoefficient viscosity_cf;

   ParGridFunction brinkman_gf;
   mutable ParGridFunction u_gf, p_gf;
   mutable ParGridFunction v_gf, ap_gf;
   mutable ParGridFunction q_gf, l_gf;

   BlockVector x, rhs, trueX, trueRhs;

   Array<int> inlet_bdr;
   Array<int> noslip_bdr;
   Array<int> all_ess_bdr;
   int max_attr;

   mutable ParGridFunction rho_tilde;

   // misc
   MPI_Comm comm;

   // PDE Coefficients
   mutable VectorFunctionCoefficient inlet_cf;

   public:
   TopOptSteadyStateStokesSolver(ParFiniteElementSpace &V_fes,
   ParFiniteElementSpace &P_fes,
   ParFiniteElementSpace &filter_fes,
   Array<int> &inlet_bdr,
   Array<int> &noslip_bdr,
   Array<int> &all_ess_bdr,
   ParGridFunction &rho_tilde,
   BrinkmanCoefficient &brinkman_cf,
   ConstantCoefficient &viscosity_cf,
   VectorFunctionCoefficient &inlet_cf,
   bool pa,
   MPI_Comm comm);

   void SetQ(ParGridFunction &q) {q_gf = q;}
   void SetLam(ParGridFunction &l) {l_gf = l;}

   virtual ~TopOptSteadyStateStokesSolver()
   {
      delete brinkman_stokes_solver;
    }

   ParGridFunction GetU(){return u_gf;}

   void SolveStokes();

   void Mult(const Vector &x, Vector &y) const override
   {
      brinkman_stokes_solver->Mult(x,y);
   }

   void AdjointSolve(Vector &lam_rhs_update) override
   {
      // //trueX = 0.0;
      // // BlockVector adjRhs = trueRhs;
      // // adjRhs.GetBlock(0).Add(1.0, lam_rhs_update);
      // VectorConstantCoefficient adj_rhs(lam_rhs_update);
      // for (int attr = 1; attr <= max_attr; attr++)
      // {
      //    brinkman_stokes_solver->Acceleration().Add(attr, adj_rhs);
      // }
      // brinkman_stokes_solver->Solve(x);
      // v_gf.SetFromTrueDofs(x.GetBlock(0));
      // ap_gf.SetFromTrueDofs(x.GetBlock(0));
      std::cout << "to be implemented " << std::endl;
   }

   void AddDesignGradient(Vector &dgdrho_tilde, real_t dt)
   {
      ParLinearForm mass_b_lf(filter_fes);
      mass_b_lf.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(rho_tilde, u_gf, v_gf, brinkman_cf));
      mass_b_lf.Assemble();
      std::unique_ptr<HypreParVector> mass_b_vec(mass_b_lf.ParallelAssemble());
      dgdrho_tilde.Add(dt, *mass_b_vec);
   }
};

TopOptSteadyStateStokesSolver::TopOptSteadyStateStokesSolver(ParFiniteElementSpace &V_fes_,
   ParFiniteElementSpace &P_fes_,
   ParFiniteElementSpace &filter_fes_,
   Array<int> &inlet_bdr_,
   Array<int> &noslip_bdr_,
   Array<int> &all_ess_bdr_,
   ParGridFunction &rho_tilde_,
   BrinkmanCoefficient &brinkman_cf_,
   ConstantCoefficient &viscosity_cf_,
   VectorFunctionCoefficient &inlet_cf_,
   bool pa_,
   MPI_Comm comm_)
   : TopOptSteadyStateOperator(V_fes_.GlobalTrueVSize() + P_fes_.GlobalTrueVSize()),
   V_fes(&V_fes_),
   P_fes(&P_fes_),
   filter_fes(&filter_fes_),
   inlet_bdr(inlet_bdr_),
   noslip_bdr(noslip_bdr_),
   all_ess_bdr(all_ess_bdr_),
   rho_tilde(rho_tilde_),
   brinkman_cf(brinkman_cf_),
   viscosity_cf(viscosity_cf_),
   inlet_cf(inlet_cf_),
   u_gf(&V_fes_),
   p_gf(&P_fes_),
   v_gf(&V_fes_),
   ap_gf(&P_fes_),
   brinkman_gf(&filter_fes_),
   pa(pa_),
   comm(comm_)
   {
      max_attr = GlobalMax(V_fes->GetParMesh()->GetComm(),
                                  V_fes->GetParMesh()->attributes.Size()
                                  ? V_fes->GetParMesh()->attributes.Max() : 0);
      int dim = V_fes -> GetVDim();
      int order_v = V_fes -> GetElementOrder(0);
      int order_p = P_fes -> GetElementOrder(0);

      rho_tilde.ExchangeFaceNbrData();
      brinkman_gf.ProjectCoefficient(brinkman_cf);

      brinkman_stokes_solver = new BrinkmanStokesSolver(*V_fes, *P_fes);
      brinkman_stokes_solver->SetSolverType(StokesSolver::KrylovSolver::GMRES);
      brinkman_stokes_solver->SetVelocityPreconditionerType(StokesSolver::VelocityPreconditioner::AMG);
      brinkman_stokes_solver->SetPressurePreconditionerType(StokesSolver::PressurePreconditioner::CAHOUET_CHABARD);
      brinkman_stokes_solver->SetCCDiffusionSolverType(StokesSolver::CCDiffusionSolver::GMRES);
      brinkman_stokes_solver->SetLSCVelocityOperatorType(StokesSolver::LSCVelocityOperator::ASSEMBLED);
      brinkman_stokes_solver->SetLSCDiagonalOperatorType(StokesSolver::LSCDiagonalOperator::MATCH_VELOCITY);
      brinkman_stokes_solver->SetLSCQPreconditionerType(StokesSolver::LSCQPreconditioner::OPERATOR_JACOBI);
      brinkman_stokes_solver->SetRelTol(1e-10);
      brinkman_stokes_solver->SetAbsTol(0.0);
      brinkman_stokes_solver->SetMaxIter(500);
      brinkman_stokes_solver->SetVelocityAMGElasticityNearNullspace(false);
      brinkman_stokes_solver->SetVelocityPreconditionerCGRelTol(1.0e-8); 
      brinkman_stokes_solver->SetVelocityPreconditionerCGAbsTol(0.0);
      brinkman_stokes_solver->SetVelocityPreconditionerCGMaxIter(100);
      brinkman_stokes_solver->SetPressurePreconditionerCGRelTol(1e-8);
      brinkman_stokes_solver->SetPressurePreconditionerCGAbsTol(0.0);
      brinkman_stokes_solver->SetPressurePreconditionerCGMaxIter(100);
      brinkman_stokes_solver->SetKDim(50);
      brinkman_stokes_solver->SetPrintLevel(-1);

      brinkman_stokes_solver->SetViscosity(viscosity_cf);
      brinkman_stokes_solver->SetBrinkmanPenalization(brinkman_gf);

      for (int attr = 1; attr <= all_ess_bdr.Size(); attr++)
      {
         if (all_ess_bdr[attr-1] == 1){brinkman_stokes_solver->VelocityBoundary().Add(attr, inlet_cf);}
      }

      // Vector zero(dim);
      // zero = 0.0;

      // VectorConstantCoefficient accel(zero);
      // for (int attr = 1; attr <= max_attr; attr++)
      // {
      //    std::cout << "....or accel" << std::endl;
      //    brinkman_stokes_solver->Acceleration().Add(attr, accel);
      // }
      std::cout << "done" << std::endl;
   }

   void TopOptSteadyStateStokesSolver::SolveStokes()
   {
      x.Update(brinkman_stokes_solver->GetBlockOffsets());
      brinkman_stokes_solver->Solve(x);
      u_gf.SetFromTrueDofs(x.GetBlock(0));
      p_gf.SetFromTrueDofs(x.GetBlock(1));
      std::cout <<  std::setprecision(14) << "velocity norm = " << u_gf.Norml2() << ", pressure norm = " << p_gf.Norml2() << std::endl;
   }




/**   Abstract Class for MixedMultiPhysics Operators. Assumed to be a time dependent operator, but it is possible to implement with static problems.
 *    Class is designed to incorporate IMEX Time-stepping, although other time integration schemes are possible. 
 *    
 *    Designed to be used for Topology Optimization, which is the purpose of the ExplicitMultDesignGradient and ImplicitSolveDesignGradient methods.
 *
 *    Note that Mult and ImplicitSolve are inherited from TimeDependentOperator, and must be implemented in derived classes of MixedMultiPhysicsOperator.
*/
class MixedMultiPhysicsOperator : public TimeDependentOperator
{
   protected:
   int num_constraints; // Essentially, the number of PDEs in the mixed-multiphysics system.
   MPI_Comm comm;
   int current_step;
   real_t dt;
   real_t t_final;
   ForwardTrajectoryStorage trajectory; //Storage for the State Trajectory
   int n_steps;

   public:
   MixedMultiPhysicsOperator(int n, int num_constraints, MPI_Comm &comm, real_t dt, real_t t_final);

   // This is where you initialize the operators, bilinear forms, solvers, etc.
   virtual void InitializeOperators(ParGridFunction &new_rho_til) = 0;

   // Perform computation of the explicit portion of the adjoint equation. 
   virtual void AdjointMult(const Vector &lam, Vector &lam_rhs) const = 0;

   // Perform computation of the implicit portion of the adjoint equation. 
   virtual void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) = 0;
   
   // Perform computation of the design gradient of explicit portion. 
   virtual void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) = 0;

   // Perform computation of the design gradient of implicit portion. 
   virtual void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) = 0;

   // Return the state.
   virtual void GetState(Vector &state_vec) = 0;

   // return the current step index.
   int GetStep(){return current_step;}

   // Set the current time-step.
   void SetStep(int new_step){current_step = new_step;}

   // Store the current state
   void StoreTraj(int step, Vector &state_vec){trajectory.Store(step, state_vec);}

   // Get the Trajectory at a given time-step
   void GetTraj(int step, Vector &state_vec) {state_vec = trajectory.Get(step);}

   // Update Dt in case of variable time-stepping
   void UpdateDt(real_t dt_real)   
   {
      MPI_Bcast(&dt_real, 1, MPI_DOUBLE, 0, comm);
      dt = dt_real;
   }
};

MixedMultiPhysicsOperator::MixedMultiPhysicsOperator(int n, 
   int num_constraints_, 
   MPI_Comm &comm_, 
   real_t dt_,
   real_t t_final_) 
   : TimeDependentOperator(n), 
   dt(dt_),
   num_constraints(num_constraints_),
   comm(comm_),
   current_step(0),
   trajectory((int)ceil(t_final_ / dt_)),
   t_final(t_final_)
{
   n_steps = (int)ceil(t_final_ / dt_);
}

/**  "Mixed Multi-Physics" Operator where the only operation is Advection-Diffusion.
 *    Spatial Discretization is Interior Penalty DG. Designed to be used in conjunction with 
 *    the IMEX-RK schemes implemented in TopOptIMEXIntegrators.hpp. The advection term is treated explicitly, and diffusion implicit.
*/
class AdvectionDiffusionMixedMultiPhysicsOperator : public MixedMultiPhysicsOperator
{
   protected:
   // Finite Element Spaces, Operators, and Solvers
   ParFiniteElementSpace *fespace;
   ParFiniteElementSpace *filter_fes;
   ParBilinearForm *M, *K, *S, *A; 
   std::unique_ptr<HypreParMatrix> M_mat, S_mat, K_mat;
   mutable ParLinearForm *b;
   mutable std::unique_ptr<HypreParVector> b_vec;
   Solver *M_prec;
   CGSolver *M_solver;
   Implicit_Solver *implicit_solver;
   LORSolver<HypreBoomerAMG>* lor_solver;
   real_t kappa;

   // Solution Storage
   GridFunctionCoefficient q0; 
   mutable ParGridFunction q_gf;

   // Boundary Stuff
   Array<int> ess_bdr_attr;
   Array<int> ess_tdof_list;
   mutable Array<int> inflow_bdr_attr;

   // Design Optimization
   mutable ParGridFunction* rho_tilde;
   SIMPCoefficient SIMP_cf;

   // PDE Coefficients
   real_t raw_diff_term;
   mutable VectorFunctionCoefficient v_base;
   mutable FunctionCoefficient raw_inflow;
   real_t dt_diff_term;
   
   // misc
   int true_size;

   // Helpers
   mutable Vector z;
   mutable Vector w;

   public:
   AdvectionDiffusionMixedMultiPhysicsOperator(ParFiniteElementSpace &fes,  
      VectorFunctionCoefficient &v_base, 
      real_t &dt_diff_term, 
      real_t &raw_diff_term,  
      GridFunctionCoefficient &q0, 
      ParGridFunction * rho_tilde, 
      real_t dt, 
      real_t t_final, 
      SIMPCoefficient SIMP_cf, 
      FunctionCoefficient raw_inflow,
      Array<int> inflow_bdr_attr,
      MPI_Comm &comm);

   void UpdateGridFuncWithStateVec(Vector &state_vec){q_gf.SetFromTrueDofs(state_vec);}

   /**
    * Set the essential boundary conditions. Maybe be non-homogenous.
    */
   void SetEssentialBoundaryConditions(Array<int> ess_bdr_attr_);
   /**
    * Set the inflow boundary.
    */
   void SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, FunctionCoefficient &inflow_cf);


   // Return the state vector  corresponding to the true dofs
   void GetState(Vector &state_vec) override;
   void InitializeOperators(ParGridFunction &new_rho_til) override; 
   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override;
   void AdjointMult(const Vector &lam, Vector &lam_rhs) const override;
   void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override;
   void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
   void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override;
   ParGridFunction& GetQ(){return q_gf;}
   

   virtual ~AdvectionDiffusionMixedMultiPhysicsOperator()
   {
      delete implicit_solver;
      delete lor_solver;
      delete M_prec;
      delete M_solver;
      //delete trajectory;
      delete M;
      delete K;
      delete S;
      delete A;
      delete b;
      //delete b_vec;
   }
};

AdvectionDiffusionMixedMultiPhysicsOperator::AdvectionDiffusionMixedMultiPhysicsOperator(ParFiniteElementSpace &fes_,  
   VectorFunctionCoefficient &v_base_, 
   real_t &dt_diff_term_, 
   real_t &raw_diff_term_,  
   GridFunctionCoefficient &q0_, 
   ParGridFunction * rho_tilde_, 
   real_t dt_, 
   real_t t_final_, 
   SIMPCoefficient SIMP_cf_, 
   FunctionCoefficient raw_inflow_,
   Array<int> inflow_bdr_attr_,
   MPI_Comm &comm_) :
   MixedMultiPhysicsOperator(fes_.GetTrueVSize(), 1, comm_, dt_, t_final_),
   fespace(&fes_),
   v_base(v_base_),
   dt_diff_term(dt_diff_term_),
   raw_diff_term(raw_diff_term_),
   q0(q0_),
   rho_tilde(rho_tilde_),
   SIMP_cf(SIMP_cf_),
   raw_inflow(raw_inflow_),
   inflow_bdr_attr(inflow_bdr_attr_),
   z(fes_.GetTrueVSize()), 
   w(fes_.GetTrueVSize())
{
   int order = fespace->GetOrder(0);
   kappa = (order + 1)*(order + 1);
   rho_tilde->ExchangeFaceNbrData();
   t = 0.0;

   q_gf.SetSpace(fespace);
   q_gf.ProjectCoefficient(q0);
   q_gf.ExchangeFaceNbrData();

   filter_fes = rho_tilde->ParFESpace();

   int n_steps = (int)ceil(t_final / dt);
   //trajectory = new ForwardTrajectoryStorage(n_steps);
   trajectory.EnableStorage();
   Vector q_vec = q_gf;
   trajectory.Store(0, q_vec);  
}

void AdvectionDiffusionMixedMultiPhysicsOperator::SetEssentialBoundaryConditions(Array<int> ess_bdr_attr_)
{
   ess_bdr_attr = ess_bdr_attr_;
   fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
   fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list); 
}

void AdvectionDiffusionMixedMultiPhysicsOperator::SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, 
   FunctionCoefficient &inflow_cf)
{
   inflow_bdr_attr = inflow_bdr_attr_; 
   raw_inflow = inflow_cf;
}

void AdvectionDiffusionMixedMultiPhysicsOperator::GetState(Vector &state_vec)
{
   state_vec = *(q_gf.GetTrueDofs());
}

void AdvectionDiffusionMixedMultiPhysicsOperator::InitializeOperators(ParGridFunction &new_rho_til)
{
   *rho_tilde = new_rho_til;
   // Boundary Conditions   
   if (ess_bdr_attr.Size() == 0)
   {
     ess_bdr_attr.SetSize(fespace->GetParMesh()->bdr_attributes.Max());   
     ess_bdr_attr = 0;   
     fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
     fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);  
   } 
   if (inflow_bdr_attr.Size() == 0)
   {
     inflow_bdr_attr.SetSize(fespace->GetParMesh()->bdr_attributes.Max()); 
     inflow_bdr_attr = 0;
     inflow_bdr_attr[1] = 1;    
   }
   const real_t sigma = -1.0;
   M = new ParBilinearForm(fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   // Form the DG Conevection Matrix
   constexpr real_t alpha = -1.0;
   ScalarVectorProductCoefficient velocity_cf(SIMP_cf, v_base);   
   K = new ParBilinearForm(fespace);
   K->AddDomainIntegrator(new ConvectionIntegrator(velocity_cf, alpha));
   K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha));                                   
   K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha), inflow_bdr_attr);
   
   // Form DG Stiffness Matrix
   ProductCoefficient diff_cf(raw_diff_term, SIMP_cf);
   S = new ParBilinearForm(fespace);
   S->AddDomainIntegrator(new DiffusionIntegrator(diff_cf));
   S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_cf, sigma, kappa));

   // For the preconditioner - create billinear form corresponding to
   // operator (M + dt S)
   ProductCoefficient dt_diff_cf(dt_diff_term, SIMP_cf); 
   A = new ParBilinearForm(fespace);
   A->AddDomainIntegrator(new MassIntegrator);
   A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_cf));
   A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_cf, sigma, kappa));
   
   M->Assemble();
   K->Assemble();
   S->Assemble();
   A->Assemble();
   M->Finalize();
   K->Finalize();
   S->Finalize();
   A->Finalize();
   
   b = new ParLinearForm(fespace);
   b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(raw_inflow, velocity_cf, alpha), inflow_bdr_attr);
   b->Assemble();
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
   lor_solver->GetSolver().SetPrintLevel(-1);
   implicit_solver -> SetPreconditioner(*lor_solver);
   
   M_solver = new CGSolver(comm);
   M_solver->SetOperator(*M_mat);
   M_solver->SetPreconditioner(*M_prec);
   M_solver->iterative_mode = false;
   M_solver->SetRelTol(1e-13);
   M_solver->SetAbsTol(0.0);
   M_solver->SetMaxIter(100);
   M_solver->SetPrintLevel(0);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::Mult(const Vector &x, Vector &y) const
{
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K_mat->Mult(x, z);
   z += *b_vec;
   M_solver->Mult(z, y);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::AdjointMult(const Vector &lam, Vector &lam_rhs) const
{
   // Plain transpose of the forward RHS Jacobian:
   // G(u) = M^{-1} (K u + b)
   // lam_rhs = 0.0;
   // Adjoint RHS evaluation for discrete adjoint 
   // Jac(G) = M^{-1} K 
   // Jac(G)^T = K^{T} M^{-T} 
   z = 0.0;
   M_solver->Mult(lam, z);
   K_mat->MultTranspose(z, lam_rhs);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   z = 0.0;
   S_mat->Mult(x, z);
   z *= -1.0;
   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(z, k);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(implicit_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   implicit_solver->SetTimeStep(dt_pass);
   implicit_solver->Mult(lam, z);
   z *= -1.0;
   S_mat->Mult(z, k);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
{
   // Update the design gradient
   // Compute w = M^{-1} lambda
   M_solver->Mult(dual_vector, w);
   ParGridFunction lam_gf(fespace);
   lam_gf.SetFromTrueDofs(w);
   // Update q_gf to be x. 
   q_gf.SetFromTrueDofs(x);
   rho_tilde->ExchangeFaceNbrData();
   lam_gf.ExchangeFaceNbrData();
   q_gf.ExchangeFaceNbrData();
   // Gradient from the Convection term
   ParLinearForm adv_lf(filter_fes);
   adv_lf.AddDomainIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf));
   adv_lf.AddBdrFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf), inflow_bdr_attr);
   adv_lf.AddInteriorFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf));
   adv_lf.Assemble();
   std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
   dgdrho_tilde.Add(-dt_pass, *adv_vec);
   // Gradient from the rhs
   ParLinearForm bdr_flow_lf(filter_fes);
   bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowDesignLFIntegrator(*rho_tilde, lam_gf, raw_inflow, v_base, SIMP_cf),inflow_bdr_attr);
   bdr_flow_lf.Assemble();
   std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
   dgdrho_tilde.Add(dt_pass, *bdr_flow_vec);
}

void AdvectionDiffusionMixedMultiPhysicsOperator::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
{
   MFEM_VERIFY(implicit_solver != NULL, "Implicit time integration is not supported with partial assembly");
   implicit_solver->SetTimeStep(dt);
   //lam A^{-1} dS/drho A^{-1} S q
   Vector k_d(dual_vector.Size()); 
   Vector y(dual_vector.Size());
   Vector u(x.Size());
   implicit_solver->Mult(dual_vector, w); // w = A^{-1} lam, A is self adjoint
   M_mat->Mult(x, u);
   implicit_solver->Mult(u, y); // y = A^{-1}S q
   ParLinearForm stiff_lf1(filter_fes); 
   ParGridFunction w_gf(fespace);
   ParGridFunction y_gf(fespace);
   w_gf.SetFromTrueDofs(w);
   y_gf.SetFromTrueDofs(y);
   rho_tilde->ExchangeFaceNbrData();
   w_gf.ExchangeFaceNbrData();
   y_gf.ExchangeFaceNbrData();
   stiff_lf1.AddDomainIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
   stiff_lf1.AddInteriorFaceIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
   stiff_lf1.Assemble();
   std::unique_ptr<HypreParVector> stiff_vec1(stiff_lf1.ParallelAssemble());   
   dfdrho_tilde.Add(a, *stiff_vec1);
}
}


#endif 