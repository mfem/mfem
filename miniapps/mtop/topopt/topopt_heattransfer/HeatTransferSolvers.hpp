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

namespace mfem
{

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
    mutable ParBilinearForm *Kd;
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
    mutable VectorFunctionCoefficient v_base;
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
        VectorFunctionCoefficient &v_base, 
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
        VectorFunctionCoefficient &v_base, 
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
        VectorFunctionCoefficient &v_base_, 
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
        VectorFunctionCoefficient &v_base_, 
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
      //raw_inflow.SetTime(t);
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

// // =============================================================================
// // IMEX ODESolvers for Design Opt
// // =============================================================================
// // Note, Time dependent operator f must have adjointmult

// class TopOptIMEXSolver : public ODESolver
// {
// protected:
//    IMEXAdvectionDiffusionSolver *f;
// public:
//    virtual void Init(IMEXAdvectionDiffusionSolver &f_) = 0;
//    virtual void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) = 0;
//    virtual void Step(Vector &x, real_t &t, real_t &dt) = 0;
//    // virtual ~TopOptIMEXSolver();
// };

// void TopOptIMEXSolver::Init(IMEXAdvectionDiffusionSolver &f_)
// {
//    this->f = &f_;
//    mem_type = GetMemoryType(f_.GetMemoryClass());
// }


// class TopOptIMEXExpImplEuler : public TopOptIMEXSolver
// {
// private:
//    Vector k1; Vector k2;
// public:
//    void Init(IMEXAdvectionDiffusionSolver &f_) override;

//    void Step(Vector &x, real_t &t, real_t &dt) override;

//    void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) override;
// };

// void TopOptIMEXExpImplEuler::Init(IMEXAdvectionDiffusionSolver &f_)
// {
//    TopOptIMEXSolver::Init(f_);
//    int n = f->Width();
//    k1.SetSize(n, mem_type);
//    k2.SetSize(n, mem_type);
// }

// void TopOptIMEXExpImplEuler::Step(Vector &x, real_t &t, real_t &dt)
// {
//    f->SetTime(t);
//    f->Mult(x, k1);

//    f->SetTime(t+dt);
//    f->ImplicitSolve(dt, x, k2);

//    f->SetTime(t);
//    x.Add(dt, k1);
//    x.Add(dt, k2);
//    t += dt;
// }

// void TopOptIMEXExpImplEuler::AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x)
// {
//    f->SetTime(t);
//    f->AdjointMult(lam, k1, x);

//    f->SetTime(t+dt);
//    f->AdjointImplicitSolve(dt, lam, x, k2);

//    f->SetTime(t);
//    lam.Add(dt, k1);
//    lam.Add(dt, k2);
//    t += dt;
// }

// /// Second order, two-stage implicit-explicit (IMEX) Runge-Kutta (RK) method
// /** L-stable IMEX RK2 method adopted from "On the Stability of IMEX Upwind gSBP
//     Schemes for 1D Linear Advection‑Difusion Equations" by Sigrun Ortleb. Same
//     as (2,2,2) from "Implicit-explicit Runge-Kutta methods for time-dependent
//     partial differential equations" by Ascher, Ruuth and Spiteri, Applied
//     Numerical Mathematics (1997). */
// class TopOptIMEXRK2 : public TopOptIMEXSolver
// {
// private:
//    Vector k1_exp; Vector k2_exp; Vector k_imp;
//    //helper vector
//    Vector y;
// public:
//    void Init(IMEXAdvectionDiffusionSolver &f_) override;

//    void Step(Vector &x, real_t &t, real_t &dt) override;

//    void AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x) override;
// };

// void TopOptIMEXRK2::Init(IMEXAdvectionDiffusionSolver &f_)
// {
//    TopOptIMEXSolver::Init(f_);
//    int n = f->Width();
//    k1_exp.SetSize(n, mem_type);
//    k2_exp.SetSize(n, mem_type);
//    k_imp.SetSize(n, mem_type);
//    y.SetSize(n, mem_type);
// }

// void TopOptIMEXRK2::Step(Vector &x, real_t &t, real_t &dt)
// {
//    double gamma = 1 - sqrt(2)/2;
//    double delta = 1 - 1/(2*gamma);

//    f->SetTime(t);

//    //K1 exp is just f_1(t, x)
//    f->Mult(x, k1_exp);

//    //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
//    f->SetTime(t + gamma*dt);
//    add(x, dt*gamma, k1_exp, y);
//    f->Mult(y, k2_exp);

//    //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
//    f->ImplicitSolve(dt*gamma, x, k_imp);
//    //reuse k_imp to avoid extra vector

//    //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
//    f -> SetTime(t + dt);
//    //add(x, dt*(1-gamma), k2_imp, z);
//    //optimization to avoid extra vector
//    x.Add(dt*(1-gamma), k_imp);
//    //f->ImplicitSolve(dt*gamma, z, k3_imp);
//    //reuse k_imp to avoid extra vector
//    f->ImplicitSolve(dt*gamma, x, k_imp);

//    //add it all up
//    f->SetTime(t);
//    x.Add(dt*delta, k1_exp);
//    x.Add(dt*(1-delta), k2_exp);
//    //x.Add(dt*(1-gamma), k2_imp); it is already added to x above
//    x.Add(dt*gamma, k_imp);
//    t += dt;
// }

// void TopOptIMEXRK2::AdjointStep(Vector &lam, real_t &t, real_t &dt, Vector &x)
// {
//    double gamma = 1 - sqrt(2)/2;
//    double delta = 1 - 1/(2*gamma);
//    int n = lam.Size();

//    f->SetTime(t);

//    Vector x1(n), x2(n), x3(n), ys(n), yi(n), x4(n);
//    f->Mult(x, x1);

//    //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
//    f->SetTime(t + gamma*dt);
//    add(x, dt*gamma, k1_exp, ys);
//    f->UpdateDt(gamma*dt);
//    f->Mult(ys, x2);
//    f->UpdateDt(dt);

//    //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
//    f->ImplicitSolve(dt*gamma, x, x3);
//    //reuse k_imp to avoid extra vector

//    //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
//    f -> SetTime(t + dt);
//    //add(x, dt*(1-gamma), k2_imp, z);
//    //optimization to avoid extra vector
//    add(x, dt*(1-gamma), x3, yi);
//    //f->ImplicitSolve(dt*gamma, z, k3_imp);
//    //reuse k_imp to avoid extra vector
//    f->ImplicitSolve(dt*gamma, yi, x4);

//    /////////////////////////////

//    //K1 exp is just f_1(t, x)
//    f->UpdateDt(delta*dt);
//    f->AdjointMult(lam, k1_exp, x);

//    //K2 exp is f_1(t + gamma dt, x + dt gamma K1)
//    f->SetTime(t + gamma*dt);
//    add(lam, dt*gamma, k1_exp, y);
//    f->UpdateDt((1-delta)*dt);
//    f->AdjointMult(y, k2_exp, x);
//    // f->UpdateDt(dt);

//    //K2_imp = f_2(t + gamma dt, x + dt gamma K2_imp)
//    f->UpdateDt((1-gamma)*dt);
//    f->AdjointImplicitSolve(dt*gamma, lam, x, k_imp);
//    //reuse k_imp to avoid extra vector

//    //K3_imp = f_2(t+dt,x + dt(1-gamma)K2_imp + dt gamma K3_imp)
//    f -> SetTime(t + dt);
//    //add(x, dt*(1-gamma), k2_imp, z);
//    //optimization to avoid extra vector
//    lam.Add(dt*(1-gamma), k_imp);
//    //f->ImplicitSolve(dt*gamma, z, k3_imp);
//    //reuse k_imp to avoid extra vector
//    f->UpdateDt(gamma*dt);
//    f->AdjointImplicitSolve(dt*(gamma), lam, yi, k_imp);
//    f->UpdateDt(dt);

//    f->SetTime(t);

//    //add it all up
//    lam.Add(dt*delta, k1_exp);
//    lam.Add(dt*(1.0-delta), k2_exp);
//    //x.Add(dt*(1-gamma), k2_imp); it is already added to x above
//    lam.Add(dt*gamma, k_imp);
//    t += dt;
// }



// std::unique_ptr<TopOptIMEXSolver> SelectDesignOptIMEX(const int ode_solver_type)
// {
//    using ode_ptr = std::unique_ptr<TopOptIMEXSolver>;
//    switch (ode_solver_type)
//    {
//       // L-stable IMEX methods for design opt
//       case 1: return ode_ptr(new TopOptIMEXExpImplEuler);
//       case 2: return ode_ptr(new TopOptIMEXRK2);

//       default: MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
//    }
// }
}
#endif 