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
      
      // Allow dynamic resizing if the forward loop takes extra steps
      while (step >= q_traj.Size()) 
      {  
         q_traj.Append(nullptr);
         num_steps++;
      }
      
      if (q_traj[step]) delete q_traj[step];
      q_traj[step] = new Vector(q);
   }

   Vector& Get(int step) { return *q_traj[step]; }
   const Vector& Get(int step) const { return *q_traj[step]; }

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
      MPI_Bcast(&dt_, 1, MPI_DOUBLE, 0, comm);
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
   virtual void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const = 0;
   virtual void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) = 0;  
   virtual void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) = 0;
   virtual void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) = 0;
   virtual void ExplicitMultCoupledStateGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdu, ParFiniteElementSpace &vfes) = 0;
   virtual void ImplicitSolveCoupledStateGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdu, ParFiniteElementSpace &vfes) = 0;  
};

TopOptTimeDependentOperator::TopOptTimeDependentOperator(int n) : TimeDependentOperator(n)
{}

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
   Array<int> offsets;

   public:
   MixedMultiPhysicsOperator(int n, int num_constraints, MPI_Comm &comm, real_t dt, real_t t_final);

   // This is where you initialize the operators, bilinear forms, solvers, etc.
   virtual BlockVector InitializeOperators(ParGridFunction &new_rho_til) = 0;

   // Perform computation of the explicit portion of the adjoint equation. 
   virtual void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const = 0;

   // Perform computation of the implicit portion of the adjoint equation. 
   virtual void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) = 0;
   
   // Perform computation of the design gradient of explicit portion. 
   virtual void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) = 0;

   // Perform computation of the design gradient of implicit portion. 
   virtual void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) = 0;

   // Add design gradient which is not time-integrated (i.e. from a static portion of the problem).
   virtual void AddStaticDesignGradient(const Vector &lam, Vector &dgdrho_tilde) { }; // Default behavior is to do nothing.

   virtual const Array<int> GetSystemOffsets(){return offsets;}

   // return the current step index.
   int GetStep(){return current_step;}

   // Set the current time-step.
   virtual void SetStep(int new_step){current_step = new_step;}

   // Store the current state
   virtual void StoreTraj(int step, BlockVector &new_state_vec){trajectory.Store(step, new_state_vec);}

   // Get the Trajectory at a given time-step
   virtual void GetTraj(int step, BlockVector &state_vec) {
      state_vec.Update(trajectory.Get(step).GetData(), offsets);
   }

   // Update Dt in case of variable time-stepping
   virtual void UpdateDt(real_t &dt_real)   
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
   offsets.SetSize(num_constraints+1);
}

/**  "Mixed Multi-Physics" Operator where the only operation is Advection-Diffusion.
 *    Spatial Discretization is Interior Penalty DG. Designed to be used in conjunction with 
 *    the IMEX-RK schemes implemented in TopOptIMEXIntegrators.hpp. The advection term is treated explicitly, and diffusion implicit.
*/
// class AdvectionDiffusionMixedMultiPhysicsOperator : public MixedMultiPhysicsOperator
// {
//    protected:
//    // Finite Element Spaces, Operators, and Solvers
//    ParFiniteElementSpace *fespace;
//    ParFiniteElementSpace *filter_fes;
//    ParBilinearForm *M, *K, *S, *A; 
//    std::unique_ptr<HypreParMatrix> M_mat, S_mat, K_mat;
//    mutable ParLinearForm *b;
//    mutable std::unique_ptr<HypreParVector> b_vec;
//    Solver *M_prec;
//    CGSolver *M_solver;
//    Implicit_Solver *implicit_solver;
//    LORSolver<HypreBoomerAMG>* lor_solver;
//    real_t kappa;

//    // Solution Storage
//    GridFunctionCoefficient q0; 
//    mutable ParGridFunction q_gf;

//    // Boundary Stuff
//    Array<int> ess_bdr_attr;
//    Array<int> ess_tdof_list;
//    mutable Array<int> inflow_bdr_attr;

//    // Design Optimization
//    mutable ParGridFunction* rho_tilde;
//    SIMPCoefficient SIMP_cf;

//    // PDE Coefficients
//    real_t raw_diff_term;
//    mutable VectorGridFunctionCoefficient v_base;
//    mutable FunctionCoefficient raw_inflow;
//    real_t dt_diff_term;
   
//    // misc
//    int true_size;

//    // Helpers
//    mutable Vector z;
//    mutable Vector w;

//    public:
//    AdvectionDiffusionMixedMultiPhysicsOperator(ParFiniteElementSpace &fes,  
//       VectorGridFunctionCoefficient &v_base, 
//       real_t &dt_diff_term, 
//       real_t &raw_diff_term,  
//       GridFunctionCoefficient &q0, 
//       ParGridFunction * rho_tilde, 
//       real_t dt, 
//       real_t t_final, 
//       SIMPCoefficient SIMP_cf, 
//       FunctionCoefficient raw_inflow,
//       Array<int> inflow_bdr_attr,
//       MPI_Comm &comm);

//    /**
//     * Set the essential boundary conditions. Maybe be non-homogenous.
//     */
//    void SetEssentialBoundaryConditions(Array<int> ess_bdr_attr_);
//    /**
//     * Set the inflow boundary.
//     */
//    void SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, FunctionCoefficient &inflow_cf);

//    void Minv(const Vector &x, Vector &y) const {y = 0.0; M_solver->Mult(x,y);}

//    // Return the state vector  corresponding to the true dofs
//    BlockVector InitializeOperators(ParGridFunction &new_rho_til) override; 
//    void Mult(const Vector &x, Vector &y) const override;
//    void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override;
//    void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const override;
//    void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override;
//    void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
//    void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override;

//    virtual ~AdvectionDiffusionMixedMultiPhysicsOperator()
//    {
//       delete implicit_solver;
//       delete lor_solver;
//       delete M_prec;
//       delete M_solver;
//       delete M;
//       delete K;
//       delete S;
//       delete A;
//       delete b;
//       //delete b_vec;
//    }
// };

// AdvectionDiffusionMixedMultiPhysicsOperator::AdvectionDiffusionMixedMultiPhysicsOperator(ParFiniteElementSpace &fes_,  
//    VectorGridFunctionCoefficient &v_base_, 
//    real_t &dt_diff_term_, 
//    real_t &raw_diff_term_,  
//    GridFunctionCoefficient &q0_, 
//    ParGridFunction * rho_tilde_, 
//    real_t dt_, 
//    real_t t_final_, 
//    SIMPCoefficient SIMP_cf_, 
//    FunctionCoefficient raw_inflow_,
//    Array<int> inflow_bdr_attr_,
//    MPI_Comm &comm_) :
//    MixedMultiPhysicsOperator(fes_.GetTrueVSize(), 1, comm_, dt_, t_final_),
//    fespace(&fes_),
//    v_base(v_base_),
//    dt_diff_term(dt_diff_term_),
//    raw_diff_term(raw_diff_term_),
//    q0(q0_),
//    rho_tilde(rho_tilde_),
//    SIMP_cf(SIMP_cf_),
//    raw_inflow(raw_inflow_),
//    inflow_bdr_attr(inflow_bdr_attr_),
//    z(fes_.GetTrueVSize()), 
//    w(fes_.GetTrueVSize()),
//    M(nullptr),
//    K(nullptr),
//    S(nullptr),
//    A(nullptr),
//    b(nullptr),
//    M_prec(nullptr),
//    M_solver(nullptr),
//    implicit_solver(nullptr),
//    lor_solver(nullptr)
// {
//    int order = fespace->GetOrder(0);
//    kappa = (order + 1)*(order + 1);
//    rho_tilde->ExchangeFaceNbrData();
//    t = 0.0;

//    filter_fes = rho_tilde->ParFESpace();

//    offsets[0] = 0;
//    offsets[1] = fes_.GetTrueVSize();
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::SetEssentialBoundaryConditions(Array<int> ess_bdr_attr_)
// {
//    ess_bdr_attr = ess_bdr_attr_;
//    fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
//    fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list); 
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, 
//    FunctionCoefficient &inflow_cf)
// {
//    inflow_bdr_attr = inflow_bdr_attr_; 
//    raw_inflow = inflow_cf;
// }

// BlockVector AdvectionDiffusionMixedMultiPhysicsOperator::InitializeOperators(ParGridFunction &new_rho_til)
// {
//    ParGridFunction q_gf(fespace);
//    q_gf.ProjectCoefficient(q0);
//    q_gf.ExchangeFaceNbrData();
//    trajectory.EnableStorage();
//    BlockVector q_vec;
//    q_vec.Update(offsets);
//    q_gf.GetTrueDofs(q_vec.GetBlock(0));
//    trajectory.Store(0, q_vec);  


//    *rho_tilde = new_rho_til;


//    // Boundary Conditions   
//    if (ess_bdr_attr.Size() == 0)
//    {
//        int local_max_bdr = fespace->GetParMesh()->bdr_attributes.Size() ? fespace->GetParMesh()->bdr_attributes.Max() : 0;
//        int global_max_bdr = 0;
//        MPI_Allreduce(&local_max_bdr, &global_max_bdr, 1, MPI_INT, MPI_MAX, comm);

//        ess_bdr_attr.SetSize(global_max_bdr);   
//        ess_bdr_attr = 0;   
//        fespace->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
//        fespace->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list);  
//    }
//    if (inflow_bdr_attr.Size() == 0)
//    {
//        int local_max_bdr = fespace->GetParMesh()->bdr_attributes.Size() ? fespace->GetParMesh()->bdr_attributes.Max() : 0;
//        int global_max_bdr = 0;
//        MPI_Allreduce(&local_max_bdr, &global_max_bdr, 1, MPI_INT, MPI_MAX, comm);

//        inflow_bdr_attr.SetSize(global_max_bdr);   
//        inflow_bdr_attr = 0;   
//        fespace->GetParMesh()->MarkExternalBoundaries(inflow_bdr_attr);  
//    }

//    // if(Mpi::Root())
//    // {
//    //    for (int i = 0; i < inflow_bdr_attr.Size(); i++)
//    //    {
//    //       std::cout << "inflow bdr attr in ad = " << inflow_bdr_attr[i] << std::endl;
//    //    }
//    // }
//    const real_t sigma = -1.0;
//    M = new ParBilinearForm(fespace);      
//    M->AddDomainIntegrator(new MassIntegrator());
//    // Form the DG Conevection Matrix
//    constexpr real_t alpha = -1.0;
//    ScalarVectorProductCoefficient velocity_cf(SIMP_cf, v_base);   
//    K = new ParBilinearForm(fespace);
//    K->AddDomainIntegrator(new ConvectionIntegrator(velocity_cf, alpha));
//    K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha));                                   
//    K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_cf, alpha), inflow_bdr_attr);
   
//    // Form DG Stiffness Matrix
//    ProductCoefficient diff_cf(raw_diff_term, SIMP_cf);
//    S = new ParBilinearForm(fespace);
//    S->AddDomainIntegrator(new DiffusionIntegrator(diff_cf));
//    S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_cf, sigma, kappa));

//    // For the preconditioner  - create billinear form corresponding to
//    // operator (M + dt S)
//    ProductCoefficient dt_diff_cf(dt_diff_term, SIMP_cf); 
//    A = new ParBilinearForm(fespace);
//    A->AddDomainIntegrator(new MassIntegrator);
//    A->AddDomainIntegrator(new DiffusionIntegrator(dt_diff_cf));
//    A->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_diff_cf, sigma, kappa));
   
//    M->Assemble();
//    K->Assemble();
//    S->Assemble();
//    A->Assemble();
//    M->Finalize();
//    K->Finalize();
//    S->Finalize();
//    A->Finalize();
   
//    b = new ParLinearForm(fespace);
//    b->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(raw_inflow, velocity_cf, alpha), inflow_bdr_attr);
//    b->Assemble();
//    b_vec.reset(b->ParallelAssemble());
   
//    //  A->Reset(A->ParallelAssemble(), true);
//    M_mat.reset(M->ParallelAssemble());
//    S_mat.reset(S->ParallelAssemble());
//    K_mat.reset(K->ParallelAssemble());
//    HypreSmoother *hypre_prec = new HypreSmoother(*M_mat, HypreSmoother::Jacobi);
//    M_prec = hypre_prec;
//    implicit_solver = new Implicit_Solver(*M_mat, *S_mat, *fespace, dt, comm);
//    lor_solver = new LORSolver<HypreBoomerAMG>(*A, ess_tdof_list); 
//    lor_solver->GetSolver().SetSystemsOptions(fespace->GetVDim(), true);
//    lor_solver->GetSolver().SetPrintLevel(-1);
//    implicit_solver -> SetPreconditioner(*lor_solver);
   
//    M_solver = new CGSolver(comm);
//    M_solver->SetOperator(*M_mat);
//    M_solver->SetPreconditioner(*M_prec);
//    M_solver->iterative_mode = false;
//    M_solver->SetRelTol(1e-13);
//    M_solver->SetAbsTol(0.0);
//    M_solver->SetMaxIter(100);
//    M_solver->SetPrintLevel(0);

//    return q_vec;
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::Mult(const Vector &x, Vector &y) const
// {
//    // Perform the explicit step
//    // y = M^{-1} (K x + b)
//    z = 0.0;
//    K_mat->Mult(x, z);
//    z += *b_vec;
//    M_solver->Mult(z, y);
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const
// {
//    // Plain transpose of the forward RHS Jacobian:
//    // G(u) = M^{-1} (K u + b)
//    // lam_rhs = 0.0;
//    // Adjoint RHS evaluation for discrete adjoint 
//    // Jac(G) = M^{-1} K 
//    // Jac(G)^T = K^{T} M^{-T} 
//    z.SetSize(lam.Size());
//    z = 0.0;

//    lam_rhs.SetSize(lam.Size());
//    lam_rhs = 0.0;

//    M_solver->Mult(lam, z);
//    //std::cout << std::setprecision(14) <<"z norm l2 = " << z.Norml2() << std::endl;
//    MFEM_VERIFY(z.Size() == lam.Size(), "Invalid mass-solve output size.");
//    K_mat->MultTranspose(z, lam_rhs);
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k)
// {
//    // Perform the implicit step
//    // solve for k, k = -(M+dt S)^{-1} S x
//    MFEM_VERIFY(implicit_solver != NULL,
//                "Implicit time integration is not supported with partial assembly");
//    z = 0.0;
//    S_mat->Mult(x, z);
//    z *= -1.0;
//    implicit_solver->SetTimeStep(dt_pass);
//    implicit_solver->Mult(z, k);
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k)
// {
//    // Perform the implicit step
//    // solve for k, k = -(M+dt S)^{-1} S x
//    MFEM_VERIFY(implicit_solver != NULL,
//                "Implicit time integration is not supported with partial assembly");
//    implicit_solver->SetTimeStep(dt_pass);
//    z= 0.0;
//    implicit_solver->Mult(lam, z);
//    z *= -1.0;
//    S_mat->Mult(z, k);
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
// {
//    // Update the design gradient
//    // Compute w = M^{-1} lambda
//    w = 0.0;
//    M_solver->Mult(dual_vector, w);
//    ParGridFunction lam_gf(fespace);
//    lam_gf.SetFromTrueDofs(w);
//    // Update q_gf to be x. 

//    ParGridFunction q_gf(fespace);
//    q_gf.SetFromTrueDofs(x);
//    rho_tilde->ExchangeFaceNbrData();
//    lam_gf.ExchangeFaceNbrData();
//    q_gf.ExchangeFaceNbrData();
//    // Gradient from the Convection term
//    ParLinearForm adv_lf(filter_fes);
//    adv_lf.AddDomainIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf));
//    adv_lf.AddBdrFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf), inflow_bdr_attr);
//    adv_lf.AddInteriorFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, q_gf, lam_gf, v_base, SIMP_cf));
//    adv_lf.Assemble();
//    std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
//    dgdrho_tilde.Add(-dt_pass, *adv_vec);
//    // Gradient from the rhs
//    ParLinearForm bdr_flow_lf(filter_fes);
//    bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowDesignLFIntegrator(*rho_tilde, lam_gf, raw_inflow, v_base, SIMP_cf),inflow_bdr_attr);
//    bdr_flow_lf.Assemble();
//    std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
//    dgdrho_tilde.Add(-dt_pass, *bdr_flow_vec);
// }

// void AdvectionDiffusionMixedMultiPhysicsOperator::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
// {
//    MFEM_VERIFY(implicit_solver != NULL, "Implicit time integration is not supported with partial assembly");
//    implicit_solver->SetTimeStep(dt);
//    //lam A^{-1} dS/drho A^{-1} S q
//    Vector k_d(dual_vector.Size());  
//    Vector y(dual_vector.Size());
//    Vector u(x.Size());
//    k_d = 0.0;
//    y = 0.0;
//    u = 0.0;
//    w = 0.0;
//    implicit_solver->Mult(dual_vector, w); // w = A^{-1} lam, A is self adjoint
//    M_mat->Mult(x, u);
//    implicit_solver->Mult(u, y); // y = A^{-1}S q
//    ParLinearForm stiff_lf1(filter_fes); 
//    ParGridFunction w_gf(fespace);
//    ParGridFunction y_gf(fespace); 
//    w_gf.SetFromTrueDofs(w);
//    y_gf.SetFromTrueDofs(y);
//    rho_tilde->ExchangeFaceNbrData();
//    w_gf.ExchangeFaceNbrData();
//    y_gf.ExchangeFaceNbrData();
//    stiff_lf1.AddDomainIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
//    stiff_lf1.AddInteriorFaceIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf, w_gf, raw_diff_term, kappa, SIMP_cf));
//    stiff_lf1.Assemble();
//    std::unique_ptr<HypreParVector> stiff_vec1(stiff_lf1.ParallelAssemble());   
//    dfdrho_tilde.Add(a, *stiff_vec1);
// }




// /**   MixedMultiPhysicsOperator with coupled Stokes and Advection-Diffusion.
//  *    Spatial Discretization is Interior Penalty DG for Advection Diffusion. Taylor-Hood for Stokes. Designed to be used in conjunction with 
//  *    the IMEX-RK schemes implemented in TopOptIMEXIntegrators.hpp. The advection term is treated explicitly, and diffusion implicit.
// */
// class CoupledSteadyStokesAdvectionDiffusionOperator : public MixedMultiPhysicsOperator
// {
//    protected:
//    AdvectionDiffusionMixedMultiPhysicsOperator *adv_diff_oper;
//    ParFiniteElementSpace *V_fes;
//    ParFiniteElementSpace *P_fes;
//    ParFiniteElementSpace *filter_fes;
//    BrinkmanCoefficient brinkman_cf;
//    bool pa;
//    BrinkmanStokesSolver *brinkman_stokes_solver;
//    ConstantCoefficient viscosity_cf;

//    ParGridFunction brinkman_gf;
//    mutable ParGridFunction u_gf, p_gf;
//    mutable ParGridFunction v_gf, ap_gf;
//    mutable ParGridFunction q_gf, l_gf;

//    BlockVector x; // holds the stokes solution

//    mutable Array<int> inlet_bdr;
//    Array<int> noslip_bdr;
//    Array<int> all_ess_bdr;
//    int max_attr;

//    mutable ParGridFunction *rho_tilde;

//    // PDE Coefficients
//    mutable VectorFunctionCoefficient inlet_cf;

//    // Advection Diffusion stuff
//    ParFiniteElementSpace *adv_fes;
//    real_t dt_diff_term; 
//    real_t raw_diff_term;
//    GridFunctionCoefficient q0;
//    mutable SIMPCoefficient SIMP_cf;
//    mutable FunctionCoefficient raw_inflow;

//    public:
//    CoupledSteadyStokesAdvectionDiffusionOperator(ParFiniteElementSpace &V_fes,  
//       ParFiniteElementSpace &P_fes,
//       Array<int> &inlet_bdr,
//       Array<int> &noslip_bdr,
//       Array<int> &all_ess_bdr,
//       ParGridFunction * rho_tilde,
//       BrinkmanCoefficient &brinkman_cf,
//       ConstantCoefficient &viscosity_cf,
//       VectorFunctionCoefficient &inlet_cf,
//       bool pa,
//       ParFiniteElementSpace &adv_fes,
//       real_t &dt_diff_term, 
//       real_t &raw_diff_term,  
//       GridFunctionCoefficient &q0, 
//       real_t dt, 
//       real_t t_final, 
//       SIMPCoefficient SIMP_cf, 
//       FunctionCoefficient raw_inflow,
//       MPI_Comm &comm);

   
//    void SetStep(int new_step) override {current_step = new_step; adv_diff_oper->SetStep(new_step);}

//    // Update Dt in case of variable time-stepping
//    void UpdateDt(real_t &dt_real)  override
//    {
//       MPI_Bcast(&dt_real, 1, MPI_DOUBLE, 0, comm);
//       dt = dt_real;
//       adv_diff_oper->UpdateDt(dt_real);
//    }

//    BlockVector InitializeOperators(ParGridFunction &new_rho_til) override; 
//    void Mult(const Vector &x, Vector &y) const override;
//    void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override;
//    void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const override;
//    void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override;
//    void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
//    void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override;
//    void AddStaticDesignGradient(const Vector &lam, Vector &dgdrho_tilde) override;

//    virtual ~CoupledSteadyStokesAdvectionDiffusionOperator()
//    {
//       // if (adv_diff_oper) {
//       delete adv_diff_oper;
//       // }
//       // if (brinkman_stokes_solver) {
//       delete brinkman_stokes_solver;
//       // }
//    }
// };

// CoupledSteadyStokesAdvectionDiffusionOperator::CoupledSteadyStokesAdvectionDiffusionOperator(ParFiniteElementSpace &V_fes_,  
//       ParFiniteElementSpace &P_fes_,
//       Array<int> &inlet_bdr_,
//       Array<int> &noslip_bdr_,
//       Array<int> &all_ess_bdr_,
//       ParGridFunction * rho_tilde_,
//       BrinkmanCoefficient &brinkman_cf_,
//       ConstantCoefficient &viscosity_cf_,
//       VectorFunctionCoefficient &inlet_cf_,
//       bool pa_,
//       ParFiniteElementSpace &adv_fes_,
//       real_t &dt_diff_term_, 
//       real_t &raw_diff_term_,  
//       GridFunctionCoefficient &q0_, 
//       real_t dt_, 
//       real_t t_final_, 
//       SIMPCoefficient SIMP_cf_, 
//       FunctionCoefficient raw_inflow_,
//       MPI_Comm &comm_) :
//    MixedMultiPhysicsOperator(adv_fes_.GetTrueVSize() + V_fes_.GetTrueVSize() + P_fes_.GetTrueVSize(), 3, comm_, dt_, t_final_),
//    V_fes(&V_fes_),
//    P_fes(&P_fes_),
//    inlet_bdr(inlet_bdr_),
//    noslip_bdr(noslip_bdr_),
//    all_ess_bdr(all_ess_bdr_),
//    rho_tilde(rho_tilde_),
//    brinkman_cf(brinkman_cf_),
//    viscosity_cf(viscosity_cf_),
//    inlet_cf(inlet_cf_),
//    pa(pa_),
//    adv_fes(&adv_fes_),
//    dt_diff_term(dt_diff_term_),
//    raw_diff_term(raw_diff_term_),
//    q0(q0_),
//    SIMP_cf(SIMP_cf_), 
//    raw_inflow(raw_inflow_),
//    u_gf(&V_fes_),
//    v_gf(&V_fes_),
//    p_gf(&P_fes_),
//    brinkman_stokes_solver(nullptr),
//    adv_diff_oper(nullptr)
// {
//    rho_tilde->ExchangeFaceNbrData();
//    t = 0.0;

//    q_gf.SetSpace(adv_fes);
//    q_gf.ProjectCoefficient(q0);
//    q_gf.ExchangeFaceNbrData();

//    filter_fes = rho_tilde->ParFESpace();
//    brinkman_gf.SetSpace(filter_fes);

//    offsets[0] = 0;
//    offsets[1] = adv_fes->GetTrueVSize();
//    offsets[2] = V_fes->GetTrueVSize();
//    offsets[3] = P_fes->GetTrueVSize();
//    offsets.PartialSum();
// }


// BlockVector CoupledSteadyStokesAdvectionDiffusionOperator::InitializeOperators(ParGridFunction &new_rho_til)
// {
//    *rho_tilde = new_rho_til;

//    max_attr = GlobalMax(comm, V_fes->GetParMesh()->attributes.Size() ? V_fes->GetParMesh()->attributes.Max() : 0);
//    int dim = V_fes -> GetVDim();
//    int order_v = V_fes -> GetElementOrder(0);
//    int order_p = P_fes -> GetElementOrder(0);

//    //brinkman_cf.UpdateRho(*rho_tilde);
//    rho_tilde->ExchangeFaceNbrData();
//    brinkman_gf.ProjectCoefficient(brinkman_cf);

//    if (brinkman_stokes_solver) {
//        delete brinkman_stokes_solver;
//    }
//    brinkman_stokes_solver = new BrinkmanStokesSolver(*V_fes, *P_fes);
//    brinkman_stokes_solver->SetSolverType(StokesSolver::KrylovSolver::MINRES);
//    brinkman_stokes_solver->SetVelocityPreconditionerType(StokesSolver::VelocityPreconditioner::AMG);
//    brinkman_stokes_solver->SetPressurePreconditionerType(StokesSolver::PressurePreconditioner::CAHOUET_CHABARD);
//    brinkman_stokes_solver->SetCCDiffusionSolverType(StokesSolver::CCDiffusionSolver::GMRES);
//    brinkman_stokes_solver->SetLSCVelocityOperatorType(StokesSolver::LSCVelocityOperator::ASSEMBLED);
//    brinkman_stokes_solver->SetLSCDiagonalOperatorType(StokesSolver::LSCDiagonalOperator::MATCH_VELOCITY);
//    brinkman_stokes_solver->SetLSCQPreconditionerType(StokesSolver::LSCQPreconditioner::OPERATOR_JACOBI);
//    brinkman_stokes_solver->SetRelTol(1e-10);
//    brinkman_stokes_solver->SetAbsTol(0.0);
//    brinkman_stokes_solver->SetMaxIter(500);
//    brinkman_stokes_solver->SetVelocityAMGElasticityNearNullspace(false);
//    brinkman_stokes_solver->SetVelocityPreconditionerCGRelTol(1.0e-8); 
//    brinkman_stokes_solver->SetVelocityPreconditionerCGAbsTol(1e-12);
//    brinkman_stokes_solver->SetVelocityPreconditionerCGMaxIter(100);
//    brinkman_stokes_solver->SetPressurePreconditionerCGRelTol(1e-8);
//    brinkman_stokes_solver->SetPressurePreconditionerCGAbsTol(1e-12);
//    brinkman_stokes_solver->SetPressurePreconditionerCGMaxIter(100);
//    brinkman_stokes_solver->SetKDim(50);
//    brinkman_stokes_solver->SetPrintLevel(-1);

//    brinkman_stokes_solver->SetViscosity(viscosity_cf);
//    brinkman_stokes_solver->SetBrinkmanPenalization(brinkman_gf);


//    int max_bdr_attr = GlobalMax(comm, V_fes->GetParMesh()->bdr_attributes.Size() ? V_fes->GetParMesh()->bdr_attributes.Max() : 0);


//    for (int attr = 1; attr <= max_bdr_attr; attr++)
//    {
//       if (all_ess_bdr[attr-1] == 1){brinkman_stokes_solver->VelocityBoundary().Add(attr, inlet_cf);}
//    }


//    x.Update(brinkman_stokes_solver->GetBlockOffsets());
//    x = 0.0;
//    brinkman_stokes_solver->Solve(x);
//    u_gf.SetFromTrueDofs(x.GetBlock(0));
//    p_gf.SetFromTrueDofs(x.GetBlock(1));
 

//    VectorGridFunctionCoefficient u_cf(&u_gf);

//    if (adv_diff_oper) {
//       delete adv_diff_oper;
//    }

//    adv_diff_oper = new AdvectionDiffusionMixedMultiPhysicsOperator(
//       *adv_fes,
//       u_cf,
//       dt_diff_term, 
//       raw_diff_term,
//       q0,
//       rho_tilde,
//       dt,
//       t_final,
//       SIMP_cf,
//       raw_inflow,
//       inlet_bdr,
//       comm);


//    q_gf.SetSpace(adv_fes);
//    q_gf.ProjectCoefficient(q0);
//    q_gf.ExchangeFaceNbrData();
//    const Array<int> ad_offsets = adv_diff_oper->GetSystemOffsets();
//    BlockVector q_vec(ad_offsets);
//    q_vec = adv_diff_oper->InitializeOperators(*rho_tilde);

//    BlockVector initial_state(offsets); 

//    initial_state.GetBlock(2) = x.GetBlock(1);
//    initial_state.GetBlock(1) = x.GetBlock(0);
//    q_gf.GetTrueDofs(initial_state.GetBlock(0));

//    trajectory.EnableStorage();
//    trajectory.Store(0, initial_state); 
//    // std::cout << std::setprecision(13)<<  "=============INITIAL===================" << std::endl;
//    // // std::cout << std::setprecision(13)<< "state heat norm = " << initial_state.GetBlock(0).Norml2() << std::endl;
//    // // std::cout << std::setprecision(13)<< "state vel norm = " << initial_state.GetBlock(1).Norml2() << std::endl;
//    // std::cout << std::setprecision(13) << "brinkman norm = " << brinkman_gf.Norml2() << std::endl;
//    // std::cout << std::setprecision(13) << "state vel norm = " << u_gf.Norml2() << std::endl;
//    // // std::cout << std::setprecision(13)<< "state pressure norm = " << initial_state.GetBlock(2).Norml2() << std::endl;
//    // std::cout << std::setprecision(13)<< "=============INITIAL===================" << std::endl;

//    int num_procs = Mpi::WorldSize();   
//    int myid = Mpi::WorldRank();   

//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       socketstream u_sock(vishost, visport);
//       u_sock << "parallel " << num_procs << " " << myid << "\n";
//       u_sock.precision(8);
//       u_sock << "solution\n" << *(V_fes->GetParMesh()) << u_gf << "window_title 'Velocity'"
//              << std::endl;
//       // Make sure all ranks have sent their 'u' solution before initiating
//       // another set of GLVis connections (one from each rank):
//       MPI_Barrier(comm);
//       socketstream p_sock(vishost, visport);
//       p_sock << "parallel " << num_procs << " " << myid << "\n";
//       p_sock.precision(8);
//       p_sock << "solution\n" << *(P_fes->GetParMesh()) << p_gf << "window_title 'Pressure'"
//              << std::endl;

//       // socketstream b_sock(vishost, visport);
//       // b_sock << "parallel " << num_procs << " " << myid << "\n";
//       // b_sock.precision(8);
//       // b_sock << "solution\n" << *(filter_fes->GetParMesh()) << brinkman_gf << "window_title 'brinkman field'"
//       //        << std::endl;

//       // socketstream rho_sock(vishost, visport);
//       // rho_sock << "parallel " << num_procs << " " << myid << "\n";
//       // rho_sock.precision(8);
//       // rho_sock << "solution\n" << *(filter_fes->GetParMesh()) << *rho_tilde << "window_title 'rho field'"
//       //        << std::endl;
//    }
//    return initial_state;
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::Mult(const Vector &x, Vector &y) const
// {
//    BlockVector bx(const_cast<Vector&>(x), offsets);  
//    BlockVector by(y, offsets);
//    y = 0.0;
//    adv_diff_oper->Mult(bx.GetBlock(0),by.GetBlock(0));
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const
// {

//    BlockVector bl(const_cast<Vector&>(lam), offsets);
//    BlockVector br(lam_rhs, offsets);
//    lam_rhs = 0.0;
//    br.GetBlock(0) = 0.0;
//    adv_diff_oper->AdjointMult(bl.GetBlock(0), br.GetBlock(0), x);

//    VectorGridFunctionCoefficient u_cf(&u_gf);
 
//    Vector w(bl.GetBlock(0).Size());
//    w = 0.0;
//    adv_diff_oper->Minv(bl.GetBlock(0), w);
//    //std::cout << std::setprecision(14) << "w norm l2 = " << w.Norml2() << std::endl;
//    ParGridFunction lam_gf(adv_fes);
//    lam_gf.SetFromTrueDofs(w);

//    BlockVector state_v(x, offsets);
//    ParGridFunction qq_gf(adv_fes);
//    qq_gf.SetFromTrueDofs(state_v.GetBlock(0)); 
//    rho_tilde->ExchangeFaceNbrData();
//    lam_gf.ExchangeFaceNbrData();
//    qq_gf.ExchangeFaceNbrData();


//    ParLinearForm adv_lf(V_fes);
//    adv_lf.AddDomainIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, SIMP_cf, u_cf));
//    adv_lf.AddBdrFaceIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, SIMP_cf, u_cf), inlet_bdr);
//    adv_lf.AddInteriorFaceIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, SIMP_cf, u_cf));
//    adv_lf.Assemble();
//    std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
 
   

//    ParLinearForm bdr_flow_lf(V_fes);
//    bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowVelocityGradientLFIntegrator(*rho_tilde, lam_gf, raw_inflow, u_cf, SIMP_cf), inlet_bdr);
//    bdr_flow_lf.Assemble(); 
//    std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
//    //std::cout<<"norm bdr grad = " << bdr_flow_vec->Norml2() << std::endl;

//    adv_vec->Add(1.0, *bdr_flow_vec);

//    *adv_vec *= -1.0;

//    // int max_bdr_attr = GlobalMax(comm, V_fes->GetParMesh()->bdr_attributes.Size() ? V_fes->GetParMesh()->bdr_attributes.Max() : 0);
//    // brinkman_stokes_solver->VelocityBoundary().Clear();
//    // Vector zero_v(V_fes->GetVDim());
//    // zero_v = 0.0;
//    // auto zero_cf = std::make_shared<VectorConstantCoefficient>(zero_v);
//    // for (int attr = 1; attr <= max_bdr_attr; attr++)
//    // {
//    //    if (all_ess_bdr[attr-1] == 1) { 
//    //       brinkman_stokes_solver->VelocityBoundary().Add(attr, zero_cf);
//    //    }
//    // }

//    //*adv_vec *= -1.0;

//    // 1. Create a BlockVector for the right-hand side
//    BlockVector rhs_stokes(brinkman_stokes_solver->GetBlockOffsets());
//    rhs_stokes = 0.0;
   
//    // 2. Assign the assembled adjoint forcing to the velocity block (Block 0)
//    //    (The pressure forcing in Block 1  remains 0.0 for incompressibility)
//    rhs_stokes.GetBlock(0) = *adv_vec;
//    MFEM_VERIFY(adv_vec->Size() == V_fes->GetTrueVSize(),
//                "Adjoint velocity RHS does not match the local velocity size.");

//    HYPRE_BigInt local_size = adv_vec->Size();
//    HYPRE_BigInt global_size = 0;
//    MPI_Allreduce(&local_size, &global_size, 1, HYPRE_MPI_BIG_INT, MPI_SUM,
//                  comm);
//    MFEM_VERIFY(global_size == V_fes->GlobalTrueVSize(),
//                "Adjoint velocity RHS does not match the global velocity size.");
 
//    BlockVector vv(brinkman_stokes_solver->GetBlockOffsets());

//    vv = 0.0; 


//    brinkman_stokes_solver->MultTranspose(rhs_stokes, vv);

//    br.GetBlock(1) = vv.GetBlock(0);
//    br.GetBlock(2) = vv.GetBlock(1);

//    // brinkman_stokes_solver->VelocityBoundary().Clear();
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k)
// {
//    BlockVector bx(const_cast<Vector&>(x), offsets);
//    BlockVector bk(k, offsets);
//    k = 0.0;
//    adv_diff_oper->ImplicitSolve(dt_pass, bx.GetBlock(0), bk.GetBlock(0));
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k)
// {
//    BlockVector bl(const_cast<Vector&>(lam), offsets);
//    BlockVector bk(k, offsets);
//    k = 0.0;
//    adv_diff_oper->AdjointImplicitSolve(dt_pass, bl.GetBlock(0), bk.GetBlock(0));
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
// {
//    BlockVector bl(dual_vector, offsets);
//    BlockVector bx(x, offsets);
//    adv_diff_oper->ExplicitMultDesignGradient(dt_pass, bl.GetBlock(0), bx.GetBlock(0), dgdrho_tilde);

//    // ParLinearForm mass_b_lf(filter_fes);
//    // v_gf.SetFromTrueDofs(bl.GetBlock(1));
//    // mass_b_lf.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(*rho_tilde, u_gf, v_gf, brinkman_cf));
//    // mass_b_lf.Assemble();
//    // std::unique_ptr<HypreParVector> mass_b_vec(mass_b_lf.ParallelAssemble());
//    // dgdrho_tilde.Add(dt_pass, *mass_b_vec);  

// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
// {
//    BlockVector bl(dual_vector, offsets);
//    BlockVector bx(x, offsets);
//    adv_diff_oper->ImplicitSolveDesignGradient(dt_pass, a, bl.GetBlock(0), bx.GetBlock(0), dfdrho_tilde);
// }

// void CoupledSteadyStokesAdvectionDiffusionOperator::AddStaticDesignGradient(const Vector &lam, Vector &dgdrho_tilde)
// {
//    // Extract the accumulated velocity adjoint from Block 1
//    BlockVector bl(const_cast<Vector&>(lam), offsets);
//    v_gf.SetFromTrueDofs(bl.GetBlock(1)); 
   
//    ParLinearForm mass_b_lf(filter_fes);
//    mass_b_lf.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(*rho_tilde, u_gf, v_gf, brinkman_cf));
//    mass_b_lf.Assemble();
//    std::unique_ptr<HypreParVector> mass_b_vec(mass_b_lf.ParallelAssemble());

//    dgdrho_tilde.Add(-1.0, *mass_b_vec);  
// }



/**  "Mixed Multi-Physics" Operator where the only operation is Advection-Diffusion. 
 *    Solves for 2 variables  - heat field over a solid plate, and over a fluid design region
 *    Spatial Discretization is Interior Penalty DG. Designed to be used in conjunction with 
 *    the IMEX-RK schemes implemented in TopOptIMEXIntegrators.hpp. The advection term is treated explicitly, and diffusion implicit.
*/
class Pseudo3DAdvectionDiffusionOperator : public MixedMultiPhysicsOperator
{
   protected:
   // Finite Element Spaces
   ParFiniteElementSpace *fluid_heat_fes;
   ParFiniteElementSpace *solid_heat_fes;
   ParFiniteElementSpace *filter_fes;

   // Fluid Region Operators
   ParBilinearForm *Mf, *Kf, *Sf, *Af; 
   std::unique_ptr<HypreParMatrix> Mf_mat, Sf_mat, Kf_mat, Mf_no_interp_mat;

   //Solid Region Operators 
   ParBilinearForm *Ms, *S, *As;
   std::unique_ptr<HypreParMatrix> Ms_mat, S_mat, Ms_no_interp_mat;

   ParBilinearForm *Ms_no_interp;
   ParBilinearForm *Mf_no_interp;

   // Forcing RHS  
   mutable ParLinearForm *bin;
   mutable std::unique_ptr<HypreParVector> bin_vec;

   // Heat Productions Forcing
   mutable ParLinearForm *b_prod;
   mutable std::unique_ptr<HypreParVector> b_prod_vec;


   // Solvers and Preconditioners
   Solver *Mf_prec, *Ms_prec;
   CGSolver *Mf_solver, *Ms_solver;
   Implicit_Solver *implicit_solver_solid, *implicit_solver_fluid;
   LORSolver<HypreBoomerAMG>* lor_solver_solid;
   LORSolver<HypreBoomerAMG>* lor_solver_fluid;
   real_t kappa_f, kappa_s;

   // Base Plate Solution Storage
   GridFunctionCoefficient q0_s, q0_f; 
   mutable ParGridFunction qs_gf, qf_gf;

   // Inflow Boundary. The base plate has perfect insulation. Design Region has inflow/outflow boundary
   mutable Array<int> inflow_bdr_attr;

   // Design Optimization, and interpolants
   mutable ParGridFunction* rho_tilde;
   RAMPCoefficient *RAMP_k, *RAMP_h;
   SIMPCoefficient *SIMP_cf;

   // PDE Coefficients
   real_t kf, ks, dtkf, dtks; // diffusion terms
   real_t cf, rf; // heat capacity and density of fluid, respectively
   real_t Q_prod; // heat production rate
   real_t plate_thickness;
   real_t finn_height; 
   real_t hs, hf; // heat transfer coefficients

   mutable VectorGridFunctionCoefficient v_base;
   mutable ConstantCoefficient raw_inflow;

   ProductCoefficient *SIMP_product_cf;
   ScalarVectorProductCoefficient *velocity_cf;
   ConstantCoefficient *ks_cf;
   ProductCoefficient *dt_diff_cf;
   ConstantCoefficient *dt_diff_cf_s;
   ConstantCoefficient *prod_cf;
   // misc
   int true_size;


   public:
   Pseudo3DAdvectionDiffusionOperator(
      ParFiniteElementSpace &fluid_heat_fes, 
      ParFiniteElementSpace &solid_heat_fes,
      GridFunctionCoefficient &q0f,
      GridFunctionCoefficient &q0s,
      Array<int> &inflow_bdr_attr,
      ParGridFunction * rho_tilde, 
      VectorGridFunctionCoefficient &v_base, 
      real_t &kf, real_t &ks, real_t &dtkf, real_t &dtks, 
      real_t &cf, real_t &rf,
      real_t &Q_prod,
      real_t &plate_thickness,
      real_t &finn_height,
      real_t &hs, real_t &hf,
      real_t dt, 
      real_t t_final, 
      ConstantCoefficient raw_inflow,
      MPI_Comm &comm);

   /**
    * Set the inflow boundary.
    */
   void SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, ConstantCoefficient &inflow_cf);

   void Minv(const Vector &x, Vector &y) const {y = 0.0; Mf_solver->Mult(x,y);}

   // Return the state vector  corresponding to the true dofs
   BlockVector InitializeOperators(ParGridFunction &new_rho_til) override; 
   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override;
   void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const override;
   void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override;
   void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
   void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override;

   virtual ~Pseudo3DAdvectionDiffusionOperator()
   {
      delete implicit_solver_fluid;
      delete implicit_solver_solid;
      delete lor_solver_solid;
      delete lor_solver_fluid;
      delete Ms_prec; delete Mf_prec;
      delete Ms_solver; delete Mf_solver;
      delete Mf;
      delete Kf;
      delete Sf;
      delete Af;
      delete Ms;
      delete S;
      delete bin;
      delete b_prod;
      delete RAMP_k;
      delete RAMP_h;
      delete As;
      delete SIMP_product_cf;
      delete velocity_cf;
      delete ks_cf;
      delete dt_diff_cf;
      delete dt_diff_cf_s;
      delete prod_cf;
   }
};

Pseudo3DAdvectionDiffusionOperator::Pseudo3DAdvectionDiffusionOperator(
      ParFiniteElementSpace &fluid_heat_fes_, 
      ParFiniteElementSpace &solid_heat_fes_,
      GridFunctionCoefficient &q0f_,
      GridFunctionCoefficient &q0s_,
      Array<int> &inflow_bdr_attr_,
      ParGridFunction * rho_tilde_, 
      VectorGridFunctionCoefficient &v_base_, 
      real_t &kf_, real_t &ks_, real_t &dtkf_, real_t &dtks_, 
      real_t &cf_, real_t &rf_,
      real_t &Q_prod_,
      real_t &plate_thickness_,
      real_t &finn_height_,
      real_t &hs_, real_t &hf_,
      real_t dt_, 
      real_t t_final_, 
      ConstantCoefficient raw_inflow_,
      MPI_Comm &comm_):
   MixedMultiPhysicsOperator(fluid_heat_fes_.GetTrueVSize() + solid_heat_fes_.GetTrueVSize(), 2, comm_, dt_, t_final_),
   fluid_heat_fes(&fluid_heat_fes_),
   solid_heat_fes(&solid_heat_fes_),
   q0_f(q0f_),
   q0_s(q0s_),
   inflow_bdr_attr(inflow_bdr_attr_),
   rho_tilde(rho_tilde_),
   v_base(v_base_),
   kf(kf_), ks(ks_), dtkf(dtkf_), dtks(dtks_),
   cf(cf_), rf(rf_),
   Q_prod(Q_prod_),
   plate_thickness(plate_thickness_),
   finn_height(finn_height_),
   hf(hf_), hs(hs_),
   raw_inflow(raw_inflow_),
   Mf_no_interp(nullptr),
   Ms_no_interp(nullptr),
   Mf(nullptr),
   Kf(nullptr),
   Sf(nullptr),
   Af(nullptr),
   Ms(nullptr),
   As(nullptr),
   S(nullptr),
   bin(nullptr),
   b_prod(nullptr),
   Ms_prec(nullptr),
   Mf_prec(nullptr),
   Ms_solver(nullptr),
   Mf_solver(nullptr),
   implicit_solver_solid(nullptr),
   implicit_solver_fluid(nullptr),
   RAMP_h(nullptr),
   RAMP_k(nullptr),
   SIMP_cf(nullptr),
   lor_solver_solid(nullptr),
   lor_solver_fluid(nullptr)
{
   int order_solid = solid_heat_fes->GetOrder(0);
   int order_fluid = fluid_heat_fes->GetOrder(0);
   kappa_f = (order_fluid + 1)*(order_fluid + 1);
   kappa_s = (order_solid + 1)*(order_solid + 1);

   rho_tilde->ExchangeFaceNbrData();
   t = 0.0;

   filter_fes = rho_tilde->ParFESpace();

   offsets[0] = 0;
   offsets[1] = solid_heat_fes_.GetTrueVSize();
   offsets[2] = fluid_heat_fes_.GetTrueVSize();
   offsets.PartialSum();
}

void Pseudo3DAdvectionDiffusionOperator::SetInflowBoundaryConditions(Array<int> inflow_bdr_attr_, 
   ConstantCoefficient &inflow_cf)
{
   inflow_bdr_attr = inflow_bdr_attr_; 
   raw_inflow = inflow_cf;
}

BlockVector Pseudo3DAdvectionDiffusionOperator::InitializeOperators(ParGridFunction &new_rho_til)
{
   qs_gf.SetSpace(solid_heat_fes);
   qf_gf.SetSpace(fluid_heat_fes);

   qs_gf.ProjectCoefficient(q0_s);
   qf_gf.ProjectCoefficient(q0_f);

   qs_gf.ExchangeFaceNbrData();
   qf_gf.ExchangeFaceNbrData();

   trajectory.EnableStorage();
   BlockVector q_vec;
   q_vec.Update(offsets);
   qs_gf.GetTrueDofs(q_vec.GetBlock(0));
   qf_gf.GetTrueDofs(q_vec.GetBlock(1));
   trajectory.Store(0, q_vec);  

   *rho_tilde = new_rho_til;

   const real_t sigma = -1.0;

   //Coefficients
   RAMP_h = new RAMPCoefficient(rho_tilde, hf, hs, 0.5);
   RAMP_k = new RAMPCoefficient(rho_tilde, kf, ks, 1.0);

   // Mass Matrices
   Ms = new ParBilinearForm(solid_heat_fes);
   Mf = new ParBilinearForm(fluid_heat_fes);     
   Ms->AddDomainIntegrator(new MassIntegrator(*RAMP_h));
   Mf->AddDomainIntegrator(new MassIntegrator(*RAMP_h));
   Ms->Assemble();
   Mf->Assemble();
   Ms->Finalize();
   Mf->Finalize();

   Ms_no_interp = new ParBilinearForm(solid_heat_fes);
   Mf_no_interp = new ParBilinearForm(fluid_heat_fes);     
   Ms_no_interp->AddDomainIntegrator(new MassIntegrator());
   Mf_no_interp->AddDomainIntegrator(new MassIntegrator());
   Ms_no_interp->Assemble();
   Mf_no_interp->Assemble();
   Ms_no_interp->Finalize();
   Mf_no_interp->Finalize();
   
   // Form the DG Conevection Matrix
   constexpr real_t alpha = -1.0;
   real_t convection_coeff = cf*rf;
   SIMP_cf = new SIMPCoefficient(rho_tilde, 1e-6, 1.0, 3.0);
   SIMP_product_cf = new ProductCoefficient(convection_coeff, *SIMP_cf);
   velocity_cf = new ScalarVectorProductCoefficient(*SIMP_product_cf, v_base);   
   Kf = new ParBilinearForm(fluid_heat_fes);
   Kf->AddDomainIntegrator(new ConvectionIntegrator(*velocity_cf, alpha));
   Kf->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity_cf, alpha));                                   
   Kf->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity_cf, alpha), inflow_bdr_attr);
   Kf->Assemble();
   Kf->Finalize();
   
   // Form DG Stiffness Matrices
   Sf = new ParBilinearForm(fluid_heat_fes);
   Sf->AddDomainIntegrator(new DiffusionIntegrator(*RAMP_k));
   Sf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*RAMP_k, sigma, kappa_f));
   Sf->Assemble();
   Sf->Finalize();
   ks_cf = new ConstantCoefficient(ks);
   S = new ParBilinearForm(solid_heat_fes);
   S->AddDomainIntegrator(new DiffusionIntegrator(*ks_cf));
   S->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*ks_cf, sigma, kappa_s));
   S->Assemble();
   S->Finalize();

   // For the preconditioner  - create billinear form corresponding to
   // operator (M + dt S)
   dt_diff_cf = new ProductCoefficient(dt, *RAMP_k); 
   Af = new ParBilinearForm(fluid_heat_fes);
   Af->AddDomainIntegrator(new MassIntegrator);
   Af->AddDomainIntegrator(new DiffusionIntegrator(*dt_diff_cf));
   Af->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dt_diff_cf, sigma, kappa_f));
   Af->Assemble();
   Af->Finalize();
   dt_diff_cf_s = new ConstantCoefficient(dt*ks); 
   As = new ParBilinearForm(solid_heat_fes);
   As->AddDomainIntegrator(new MassIntegrator);
   As->AddDomainIntegrator(new DiffusionIntegrator(*dt_diff_cf_s));
   As->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dt_diff_cf_s, sigma, kappa_s));
   As->Assemble();
   As->Finalize();


   // Inflow Vector
   bin = new ParLinearForm(fluid_heat_fes);
   bin->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(raw_inflow, *velocity_cf, alpha), inflow_bdr_attr);
   bin->Assemble();
   bin_vec.reset(bin->ParallelAssemble());

   // Heat Production RHS
   prod_cf = new ConstantCoefficient(Q_prod / plate_thickness);
   b_prod = new ParLinearForm(solid_heat_fes);
   b_prod->AddDomainIntegrator(new DomainLFIntegrator(*prod_cf));
   b_prod->Assemble();
   b_prod_vec.reset(b_prod->ParallelAssemble());

   //Assemble Martrices
   Mf_mat.reset(Mf->ParallelAssemble());
   Mf_no_interp_mat.reset(Mf_no_interp->ParallelAssemble());
   Sf_mat.reset(Sf->ParallelAssemble());
   Kf_mat.reset(Kf->ParallelAssemble());
   Ms_mat.reset(Ms->ParallelAssemble());
   S_mat.reset(S->ParallelAssemble());
   Ms_no_interp_mat.reset(Ms_no_interp->ParallelAssemble());

   // Mass Matrix Preconditioners and Solvers
   HypreSmoother *hypre_prec1 = new HypreSmoother(*Ms_no_interp_mat, HypreSmoother::Jacobi);
   Ms_prec = hypre_prec1;
   HypreSmoother *hypre_prec2 = new HypreSmoother(*Mf_no_interp_mat, HypreSmoother::Jacobi);
   Mf_prec = hypre_prec2;
   Ms_solver = new CGSolver(comm);
   Ms_solver->SetOperator(*Ms_no_interp_mat);
   Ms_solver->SetPreconditioner(*Ms_prec);
   Ms_solver->iterative_mode = false;
   Ms_solver->SetRelTol(1e-13);
   Ms_solver->SetAbsTol(0.0);
   Ms_solver->SetMaxIter(100);
   Ms_solver->SetPrintLevel(0);
   Mf_solver = new CGSolver(comm);
   Mf_solver->SetOperator(*Mf_no_interp_mat);
   Mf_solver->SetPreconditioner(*Mf_prec);
   Mf_solver->iterative_mode = false;
   Mf_solver->SetRelTol(1e-13);
   Mf_solver->SetAbsTol(0.0);
   Mf_solver->SetMaxIter(100);
   Mf_solver->SetPrintLevel(0);

   int local_max_bdr = solid_heat_fes->GetParMesh()->bdr_attributes.Size() ? solid_heat_fes->GetParMesh()->bdr_attributes.Max() : 0;
   int global_max_bdr = 0;
   MPI_Allreduce(&local_max_bdr, &global_max_bdr, 1, MPI_INT, MPI_MAX, comm);

   Array<int> ess_bdr_attr;
   Array<int> ess_tdof_list_s;
   Array<int> ess_tdof_list_f;

   ess_bdr_attr.SetSize(global_max_bdr);   
   ess_bdr_attr = 0;   
   solid_heat_fes->GetParMesh()->MarkExternalBoundaries(ess_bdr_attr);  
   solid_heat_fes->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list_s);  
   fluid_heat_fes->GetEssentialTrueDofs(ess_bdr_attr, ess_tdof_list_f);  

   // Implicit Solvers
   implicit_solver_solid = new Implicit_Solver(*Ms_no_interp_mat, *S_mat, *solid_heat_fes, dt, comm);
   lor_solver_solid = new LORSolver<HypreBoomerAMG>(*As, ess_tdof_list_s); 
   lor_solver_solid->GetSolver().SetSystemsOptions(solid_heat_fes->GetVDim(), true);
   lor_solver_solid->GetSolver().SetPrintLevel(0);
   implicit_solver_solid -> SetPreconditioner(*lor_solver_solid);
   implicit_solver_fluid = new Implicit_Solver(*Mf_no_interp_mat, *Sf_mat, *fluid_heat_fes, dt, comm);
   lor_solver_fluid = new LORSolver<HypreBoomerAMG>(*Af, ess_tdof_list_f); 
   lor_solver_fluid->GetSolver().SetSystemsOptions(fluid_heat_fes->GetVDim(), true);
   lor_solver_fluid->GetSolver().SetPrintLevel(0);
   implicit_solver_fluid -> SetPreconditioner(*lor_solver_fluid);

   return q_vec;
}

void Pseudo3DAdvectionDiffusionOperator::Mult(const Vector &x, Vector &y) const
{
   BlockVector bx(const_cast<Vector&>(x), offsets);  
   BlockVector by(y, offsets);

   Vector z(bx.GetBlock(1).Size());
   Vector w(bx.GetBlock(1).Size());
   Vector v(bx.GetBlock(1).Size());

   y = 0.0;
   
   //Fluid Portion
   Kf_mat->Mult(bx.GetBlock(1), z);
   z += *bin_vec;
   Mf_mat->Mult(bx.GetBlock(1), w);
   w *= 1.0 / finn_height;
   z += w;
   qs_gf.SetFromTrueDofs(bx.GetBlock(0));
   GridFunctionCoefficient qs_cf(&qs_gf);
   ParGridFunction qs_gf_f(fluid_heat_fes);
   qs_gf_f.ProjectCoefficient(qs_cf);
   Mf_mat->Mult(qs_gf_f, v);
   v *= 1.0/finn_height;
   z -= v;
   Mf_solver->Mult(z, by.GetBlock(1));

   z.SetSize(bx.GetBlock(0).Size());
   w.SetSize(bx.GetBlock(0).Size());
   v.SetSize(bx.GetBlock(0).Size());

   // Solid Portion
   Ms_mat->Mult(bx.GetBlock(0), z);
   z *= -1.0 / plate_thickness;
   z += *b_prod_vec;
   qf_gf.SetFromTrueDofs(bx.GetBlock(1));
   GridFunctionCoefficient qf_cf(&qf_gf);
   ParGridFunction qf_gf_s(solid_heat_fes);
   qf_gf_s.ProjectCoefficient(qf_cf);
   Ms_mat->Mult(qf_gf_s, v);
   v *= 1.0 / plate_thickness;
   z += v;
   Ms_solver->Mult(z, by.GetBlock(0));
}

void Pseudo3DAdvectionDiffusionOperator::AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const
{
   // Plain transpose of the forward RHS Jacobian:
   // G(u) = M^{-1} (K u + b)
   // lam_rhs = 0.0;
   // Adjoint RHS evaluation for discrete adjoint 
   // Jac(G) = M^{-1} K 
   // Jac(G)^T = K^{T} M^{-T} 

   BlockVector bl(const_cast<Vector&>(lam), offsets);
   BlockVector br(lam_rhs, offsets);
   lam_rhs = 0.0;

   Vector z(bl.GetBlock(1).Size());
   Vector w(bl.GetBlock(1).Size());
   Vector v(bl.GetBlock(1).Size());
   Vector zz(bl.GetBlock(1).Size());
   Vector ww(bl.GetBlock(1).Size());

   //Fluid Portion
   Mf_solver->Mult(bl.GetBlock(1), zz);
   Kf_mat->MultTranspose(zz, br.GetBlock(1));
   Mf_mat->Mult(zz, v);
   v *= 1.0 / finn_height;
   br.GetBlock(1) += v;
   qs_gf.SetFromTrueDofs(bl.GetBlock(0));
   GridFunctionCoefficient qs_cf(&qs_gf);
   ParGridFunction qs_gf_f(fluid_heat_fes);
   qs_gf_f.ProjectCoefficient(qs_cf);
   Mf_solver->Mult(qs_gf_f,w);
   Mf_mat->Mult(w, ww);
   ww *= 1.0 / finn_height;
   br.GetBlock(1) += ww;

   z.SetSize(bl.GetBlock(0).Size());
   w.SetSize(bl.GetBlock(0).Size());
   v.SetSize(bl.GetBlock(0).Size());
   zz.SetSize(bl.GetBlock(0).Size());
   ww.SetSize(bl.GetBlock(0).Size());

   // Solid Portion
   Ms_solver->Mult(bl.GetBlock(0), zz);
   Ms_mat->Mult(zz, br.GetBlock(0));
   qf_gf.SetFromTrueDofs(bl.GetBlock(1));
   GridFunctionCoefficient qf_cf(&qf_gf);
   ParGridFunction qf_gf_s(solid_heat_fes);
   qf_gf_s.ProjectCoefficient(qf_cf);
   Ms_solver->Mult(qf_gf_s,w);
   Ms_mat->Mult(w, ww);
   br.GetBlock(0) += ww;
   br.GetBlock(0) *= 1.0 / plate_thickness;


}

void Pseudo3DAdvectionDiffusionOperator::ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k)
{
   BlockVector bx(const_cast<Vector&>(x), offsets);
   BlockVector bk(k, offsets);
   k = 0.0;

   Vector z(bx.GetBlock(1).Size());
   z = 0.0;

   //Fluid Portion
   Sf_mat->Mult(bx.GetBlock(1), z);
   z *= -1.0;
   implicit_solver_fluid->SetTimeStep(dt_pass);
   implicit_solver_fluid->Mult(z, bk.GetBlock(1));

   //Solid Portion
   z.SetSize(bx.GetBlock(0).Size());
   S_mat->Mult(bx.GetBlock(0), z);
   z *= -1.0;
   implicit_solver_solid->SetTimeStep(dt_pass);
   implicit_solver_solid->Mult(z, bk.GetBlock(0));
}

void Pseudo3DAdvectionDiffusionOperator::AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k)
{
   BlockVector bl(const_cast<Vector&>(lam), offsets);
   BlockVector bk(k, offsets);
   k = 0.0;

   Vector z(bl.GetBlock(1).Size());
   z = 0.0;

   //Fluid Portion
   implicit_solver_fluid->SetTimeStep(dt_pass);
   implicit_solver_fluid->Mult(bl.GetBlock(1), z);
   z *= -1.0;
   Sf_mat->Mult(z, bk.GetBlock(1));

   z.SetSize(bl.GetBlock(0).Size());

   // Solid Portion
   implicit_solver_solid->SetTimeStep(dt_pass);
   implicit_solver_solid->Mult(bl.GetBlock(0), z);
   //z *= -1.0;
   S_mat->Mult(z, bk.GetBlock(0));
}

void Pseudo3DAdvectionDiffusionOperator::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
{
   BlockVector bl(dual_vector, offsets);
   BlockVector bx(x, offsets);

   Vector ws(bl.GetBlock(0).Size());
   Vector wf(bl.GetBlock(1).Size());
   Mf_solver->Mult(bl.GetBlock(1), wf);
   Ms_solver->Mult(bl.GetBlock(0), ws);

   ParGridFunction lam_s_gf(solid_heat_fes);
   ParGridFunction lam_f_gf(fluid_heat_fes);

   lam_s_gf.SetFromTrueDofs(ws);
   lam_f_gf.SetFromTrueDofs(wf);
   // Update q_gf to be x. 

   qs_gf.SetFromTrueDofs(bx.GetBlock(0));
   qf_gf.SetFromTrueDofs(bx.GetBlock(1));
   rho_tilde->ExchangeFaceNbrData();
   lam_s_gf.ExchangeFaceNbrData();
   lam_f_gf.ExchangeFaceNbrData();
   qs_gf.ExchangeFaceNbrData();
   qf_gf.ExchangeFaceNbrData();

   // Gradient from the Convection term
   ParLinearForm adv_lf(filter_fes);
   adv_lf.AddDomainIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, qf_gf, lam_f_gf, v_base, *SIMP_cf));
   adv_lf.AddBdrFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, qf_gf, lam_f_gf, v_base, *SIMP_cf), inflow_bdr_attr);
   adv_lf.AddInteriorFaceIntegrator(new DGAdvectionDesignLFIntegrator(*rho_tilde, qf_gf, lam_f_gf, v_base, *SIMP_cf));
   adv_lf.Assemble();
   std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
   dgdrho_tilde.Add(-dt_pass*cf*rf, *adv_vec);

   // Gradient from the rhs
   ParLinearForm bdr_flow_lf(filter_fes);
   bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowDesignLFIntegrator(*rho_tilde, lam_f_gf, raw_inflow, v_base, *SIMP_cf),inflow_bdr_attr);
   bdr_flow_lf.Assemble();
   std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
   dgdrho_tilde.Add(-dt_pass*cf*rf, *bdr_flow_vec);

   // Gradient with respect to mass terms
   ParLinearForm mass_s_lf(filter_fes);
   mass_s_lf.AddDomainIntegrator(new MassRAMPDesignGradientLFIntegrator(*rho_tilde, qs_gf, lam_s_gf, *RAMP_h));
   mass_s_lf.Assemble();
   std::unique_ptr<HypreParVector> mass_s_vec(mass_s_lf.ParallelAssemble());
   dgdrho_tilde.Add(dt_pass*((1.0 / finn_height) + (1.0 / plate_thickness)), *mass_s_vec);

   ParLinearForm mass_f_lf(filter_fes);
   mass_f_lf.AddDomainIntegrator(new MassRAMPDesignGradientLFIntegrator(*rho_tilde, qf_gf, lam_f_gf, *RAMP_h));
   mass_f_lf.Assemble();
   std::unique_ptr<HypreParVector> mass_f_vec(mass_f_lf.ParallelAssemble());
   dgdrho_tilde.Add(-dt_pass*((1.0 / finn_height) + (1.0 / plate_thickness)), *mass_f_vec);
}

void Pseudo3DAdvectionDiffusionOperator::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
{
   implicit_solver_fluid->SetTimeStep(dt);

   BlockVector bl(dual_vector, offsets);
   BlockVector bx(x, offsets);

   //lam A^{-1} dS/drho A^{-1} S q
   Vector wf(bl.GetBlock(1).Size());
   Vector k_df(bl.GetBlock(1).Size());  
   Vector yf(bl.GetBlock(1).Size());
   Vector uf(bx.GetBlock(1).Size());

   k_df = 0.0;
   yf = 0.0;
   uf = 0.0;
   wf = 0.0;


   implicit_solver_fluid->Mult(bl.GetBlock(1), wf); // w = A^{-1} lam, A is self adjoint
   Mf_no_interp_mat->Mult(bx.GetBlock(1), uf);
   implicit_solver_fluid->Mult(uf, yf); // y = A^{-1}S q


   ParLinearForm stiff_lff(filter_fes); 
   ParGridFunction w_gf_f(fluid_heat_fes);
   ParGridFunction y_gf_f(fluid_heat_fes); 
   w_gf_f.SetFromTrueDofs(wf);
   y_gf_f.SetFromTrueDofs(yf);
   rho_tilde->ExchangeFaceNbrData();
   w_gf_f.ExchangeFaceNbrData();
   y_gf_f.ExchangeFaceNbrData();
   real_t one = 1.0;
   stiff_lff.AddDomainIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf_f, w_gf_f, one, kappa_f, *RAMP_k));
   stiff_lff.AddInteriorFaceIntegrator(new DGStiffnessDesignLFIntegrator(*rho_tilde, y_gf_f, w_gf_f, one, kappa_f, *RAMP_k));
   stiff_lff.Assemble();
   std::unique_ptr<HypreParVector> stiff_vecf(stiff_lff.ParallelAssemble());   
   dfdrho_tilde.Add(a, *stiff_vecf);
}

/**   MixedMultiPhysicsOperator with coupled Stokes and Advection-Diffusion for Pseudo 3D Model.
 *    Spatial Discretization is Interior Penalty DG for Advection Diffusion. Taylor-Hood for Stokes. Designed to be used in conjunction with 
 *    the IMEX-RK schemes implemented in TopOptIMEXIntegrators.hpp. The advection term and mass terms are treated explicitly, and diffusion implicit.
*/
class Pseudo3DStokesOperator : public MixedMultiPhysicsOperator
{
   protected:
   // Advection Diffusion Operator, defined above
   Pseudo3DAdvectionDiffusionOperator *adv_diff_oper;

   // Finite Element Spaces
   ParFiniteElementSpace *V_fes;
   ParFiniteElementSpace *P_fes;
   ParFiniteElementSpace *filter_fes;
   ParFiniteElementSpace *fluid_heat_fes;
   ParFiniteElementSpace *solid_heat_fes;

   // Coefficients
   BrinkmanCoefficient brinkman_cf;
   mutable ConstantCoefficient raw_inflow;
   ConstantCoefficient viscosity_cf;
   GridFunctionCoefficient q0_f;
   GridFunctionCoefficient q0_s;
   mutable VectorFunctionCoefficient inlet_cf;
   SIMPCoefficient *SIMP_cf;

   // Stokes Solver
   bool pa;
   BrinkmanStokesSolver *brinkman_stokes_solver;

   // Gridfunctions, including rho
   ParGridFunction brinkman_gf;
   mutable ParGridFunction u_gf, p_gf;
   mutable ParGridFunction v_gf;
   mutable ParGridFunction q_s_gf, l_s_gf;
   mutable ParGridFunction q_f_gf, l_f_gf;
   mutable ParGridFunction *rho_tilde;

   BlockVector x; // holds the stokes solution

   //Boundary stuff
   mutable Array<int> inlet_bdr;
   Array<int> noslip_bdr;
   Array<int> all_ess_bdr;
   Array<int> helper_offsets; 
   int max_attr;

   //constants
   real_t kf, ks, dtkf, dtks;
   real_t cf, rf;
   real_t Q_prod;
   real_t plate_thickness;
   real_t finn_height;
   real_t hs, hf;

   public:
   Pseudo3DStokesOperator(
      ParFiniteElementSpace &V_fes,  
      ParFiniteElementSpace &P_fes,
      ParFiniteElementSpace &fluid_heat_fes,  
      ParFiniteElementSpace &solid_heat_fes,
      BrinkmanCoefficient &brinkman_cf,
      ConstantCoefficient &viscosity_cf,
      VectorFunctionCoefficient &inlet_cf,
      ConstantCoefficient &raw_inflow,
      GridFunctionCoefficient &q0_f,
      GridFunctionCoefficient &q0_s, 
      ParGridFunction * rho_tilde,
      Array<int> &inlet_bdr,
      Array<int> &noslip_bdr,
      Array<int> &all_ess_bdr,
      real_t &kf, real_t &ks, real_t &dtkf, real_t &dtks,
      real_t &cf, real_t &rf,
      real_t &Q_prod,
      real_t &plate_thickness,
      real_t &finn_height,
      real_t &hs, real_t &hf,
      real_t dt, 
      real_t t_final, 
      MPI_Comm &comm);

   
   void SetStep(int new_step) override {current_step = new_step; adv_diff_oper->SetStep(new_step);}

   // Update Dt in case of variable time-stepping
   void UpdateDt(real_t &dt_real)  override
   {
      MPI_Bcast(&dt_real, 1, MPI_DOUBLE, 0, comm);
      dt = dt_real;
      adv_diff_oper->UpdateDt(dt_real);
   }

   BlockVector InitializeOperators(ParGridFunction &new_rho_til) override; 
   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k) override;
   void AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const override;
   void AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k) override;
   void ExplicitMultDesignGradient(const real_t dt, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde) override;
   void ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a, Vector &dual_vector, Vector &x, Vector &dfdrho_tilde) override;
   void AddStaticDesignGradient(const Vector &lam, Vector &dgdrho_tilde) override;

   virtual ~Pseudo3DStokesOperator()
   {
      delete adv_diff_oper;
      delete brinkman_stokes_solver;
      delete SIMP_cf;
   }
};

Pseudo3DStokesOperator::Pseudo3DStokesOperator(
      ParFiniteElementSpace &V_fes_,  
      ParFiniteElementSpace &P_fes_,
      ParFiniteElementSpace &fluid_heat_fes_,  
      ParFiniteElementSpace &solid_heat_fes_,
      BrinkmanCoefficient &brinkman_cf_,
      ConstantCoefficient &viscosity_cf_,
      VectorFunctionCoefficient &inlet_cf_,
      ConstantCoefficient &raw_inflow_,
      GridFunctionCoefficient &q0_f_,
      GridFunctionCoefficient &q0_s_, 
      ParGridFunction * rho_tilde_,
      Array<int> &inlet_bdr_,
      Array<int> &noslip_bdr_,
      Array<int> &all_ess_bdr_,
      real_t &kf_, real_t &ks_, real_t &dtkf_, real_t &dtks_,
      real_t &cf_, real_t &rf_,
      real_t &Q_prod_,
      real_t &plate_thickness_,
      real_t &finn_height_,
      real_t &hs_, real_t &hf_,
      real_t dt_, 
      real_t t_final_, 
      MPI_Comm &comm_) :
   MixedMultiPhysicsOperator(solid_heat_fes_.GetTrueVSize() + fluid_heat_fes_.GetTrueVSize() + V_fes_.GetTrueVSize() + P_fes_.GetTrueVSize(), 
   4, comm_, dt_, t_final_),
   V_fes(&V_fes_),
   P_fes(&P_fes_),
   fluid_heat_fes(&fluid_heat_fes_),
   solid_heat_fes(&solid_heat_fes_),
   brinkman_cf(brinkman_cf_),
   viscosity_cf(viscosity_cf_),
   inlet_cf(inlet_cf_),
   raw_inflow(raw_inflow_),
   q0_f(q0_f_),
   q0_s(q0_s_),
   rho_tilde(rho_tilde_),
   inlet_bdr(inlet_bdr_),
   noslip_bdr(noslip_bdr_),
   all_ess_bdr(all_ess_bdr_),
   pa(false), 
   kf(kf_), ks(ks_), dtkf(dtkf_), dtks(dtks_),
   cf(cf_), rf(rf_),
   Q_prod(Q_prod_),
   plate_thickness(plate_thickness_),
   finn_height(finn_height_),
   hf(hf_), hs(hs_),
   u_gf(&V_fes_),
   v_gf(&V_fes_),
   p_gf(&P_fes_),
   brinkman_stokes_solver(nullptr),
   adv_diff_oper(nullptr),
   SIMP_cf(nullptr)
{
   rho_tilde->ExchangeFaceNbrData();
   t = 0.0;

   q_f_gf.SetSpace(fluid_heat_fes);
   q_f_gf.ProjectCoefficient(q0_f);
   q_f_gf.ExchangeFaceNbrData();

   q_s_gf.SetSpace(solid_heat_fes);
   q_s_gf.ProjectCoefficient(q0_s);
   q_s_gf.ExchangeFaceNbrData();

   filter_fes = rho_tilde->ParFESpace();
   brinkman_gf.SetSpace(filter_fes);

   offsets[0] = 0;
   offsets[1] = solid_heat_fes->GetTrueVSize();
   offsets[2] = fluid_heat_fes->GetTrueVSize();
   offsets[3] = V_fes->GetTrueVSize();
   offsets[4] = P_fes->GetTrueVSize();
   offsets.PartialSum();

   helper_offsets.SetSize(4);
   helper_offsets[0] = 0;
   helper_offsets[1] = solid_heat_fes->GetTrueVSize() + fluid_heat_fes->GetTrueVSize();
   helper_offsets[2] = V_fes->GetTrueVSize();
   helper_offsets[3] = P_fes->GetTrueVSize();
   helper_offsets.PartialSum();

   SIMP_cf = new SIMPCoefficient(rho_tilde, 1e-6, 1.0, 3.0);
}


BlockVector Pseudo3DStokesOperator::InitializeOperators(ParGridFunction &new_rho_til)
{
   *rho_tilde = new_rho_til;

   max_attr = GlobalMax(comm, V_fes->GetParMesh()->attributes.Size() ? V_fes->GetParMesh()->attributes.Max() : 0);
   int dim = V_fes -> GetVDim();
   int order_v = V_fes -> GetElementOrder(0);
   int order_p = P_fes -> GetElementOrder(0);

   //brinkman_cf.UpdateRho(*rho_tilde);
   rho_tilde->ExchangeFaceNbrData();
   brinkman_gf.ProjectCoefficient(brinkman_cf);

   if (brinkman_stokes_solver) {
       delete brinkman_stokes_solver;
   }
   brinkman_stokes_solver = new BrinkmanStokesSolver(*V_fes, *P_fes);
   brinkman_stokes_solver->SetSolverType(StokesSolver::KrylovSolver::MINRES);
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
   brinkman_stokes_solver->SetVelocityPreconditionerCGAbsTol(1e-12);
   brinkman_stokes_solver->SetVelocityPreconditionerCGMaxIter(100);
   brinkman_stokes_solver->SetPressurePreconditionerCGRelTol(1e-8);
   brinkman_stokes_solver->SetPressurePreconditionerCGAbsTol(1e-12);
   brinkman_stokes_solver->SetPressurePreconditionerCGMaxIter(100);
   brinkman_stokes_solver->SetKDim(50);
   brinkman_stokes_solver->SetPrintLevel(0);

   brinkman_stokes_solver->SetViscosity(viscosity_cf);
   brinkman_stokes_solver->SetBrinkmanPenalization(brinkman_gf);


   int max_bdr_attr = GlobalMax(comm, V_fes->GetParMesh()->bdr_attributes.Size() ? V_fes->GetParMesh()->bdr_attributes.Max() : 0);


   for (int attr = 1; attr <= max_bdr_attr; attr++)
   {
      if (all_ess_bdr[attr-1] == 1){brinkman_stokes_solver->VelocityBoundary().Add(attr, inlet_cf);}
   }


   x.Update(brinkman_stokes_solver->GetBlockOffsets());
   x = 0.0;
   brinkman_stokes_solver->Solve(x);
   u_gf.SetFromTrueDofs(x.GetBlock(0));
   p_gf.SetFromTrueDofs(x.GetBlock(1));
 

   VectorGridFunctionCoefficient u_cf(&u_gf);

   if (adv_diff_oper) {
      delete adv_diff_oper;
   }

   adv_diff_oper = new Pseudo3DAdvectionDiffusionOperator(
      *fluid_heat_fes,
      *solid_heat_fes,
      q0_f,
      q0_s,
      inlet_bdr,
      rho_tilde,
      u_cf,
      kf, ks, dtkf, dtks,
      cf, rf,
      Q_prod,
      plate_thickness,
      finn_height,
      hs,
      hf,
      dt,
      t_final,
      raw_inflow,
      comm);

   q_s_gf.ExchangeFaceNbrData();
   q_f_gf.ExchangeFaceNbrData();
   const Array<int> ad_offsets = adv_diff_oper->GetSystemOffsets();
   BlockVector q_vec(ad_offsets);
   q_vec = adv_diff_oper->InitializeOperators(*rho_tilde);

   BlockVector initial_state(offsets); 

   initial_state.GetBlock(3) = x.GetBlock(1);
   initial_state.GetBlock(2) = x.GetBlock(0);
   q_s_gf.GetTrueDofs(initial_state.GetBlock(0));
   q_f_gf.GetTrueDofs(initial_state.GetBlock(1));

   trajectory.EnableStorage();
   trajectory.Store(0, initial_state); 
   std::cout << std::setprecision(13)<<  "=============INITIAL===================" << std::endl;
   // std::cout << std::setprecision(13)<< "state heat norm = " << initial_state.GetBlock(0).Norml2() << std::endl;
   // std::cout << std::setprecision(13)<< "state vel norm = " << initial_state.GetBlock(1).Norml2() << std::endl;
   std::cout << std::setprecision(13) << "brinkman norm = " << brinkman_gf.Norml2() << std::endl;
   std::cout << std::setprecision(13) << "state vel norm = " << u_gf.Norml2() << std::endl;
   // std::cout << std::setprecision(13)<< "state pressure norm = " << initial_state.GetBlock(2).Norml2() << std::endl;
   std::cout << std::setprecision(13)<< "=============INITIAL===================" << std::endl;

   int num_procs = Mpi::WorldSize();   
   int myid = Mpi::WorldRank();   

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *(V_fes->GetParMesh()) << u_gf << "window_title 'Velocity'"
             << std::endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(comm);
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *(P_fes->GetParMesh()) << p_gf << "window_title 'Pressure'"
             << std::endl;

      // socketstream s_sock(vishost, visport);
      // s_sock << "parallel " << num_procs << " " << myid << "\n";
      // s_sock.precision(8);
      // s_sock << "solution\n" << *(solid_heat_fes->GetParMesh()) << q_s_gf << "window_title 'solid heat'"
      //        << std::endl;

      // socketstream f_sock(vishost, visport);
      // f_sock << "parallel " << num_procs << " " << myid << "\n";
      // f_sock.precision(8);
      // f_sock << "solution\n" << *(fluid_heat_fes->GetParMesh()) << q_f_gf << "window_title 'fluid heat'"
      //        << std::endl;
   }
   return initial_state;
}

void Pseudo3DStokesOperator::Mult(const Vector &x, Vector &y) const
{
   BlockVector bx(const_cast<Vector&>(x), helper_offsets);  
   BlockVector by(y, helper_offsets);
   y = 0.0;
   adv_diff_oper->Mult(bx.GetBlock(0),by.GetBlock(0));
}

void Pseudo3DStokesOperator::AdjointMult(const Vector &lam, Vector &lam_rhs, Vector &x) const
{

   BlockVector bl(const_cast<Vector&>(lam), helper_offsets);
   BlockVector br(lam_rhs, helper_offsets);
   lam_rhs = 0.0;
   br.GetBlock(0) = 0.0;
   adv_diff_oper->AdjointMult(bl.GetBlock(0), br.GetBlock(0), x);

   VectorGridFunctionCoefficient u_cf(&u_gf);
 
   bl.Update(offsets);
   Vector w(bl.GetBlock(1).Size());
   w = 0.0;
   adv_diff_oper->Minv(bl.GetBlock(1), w);
   //std::cout << std::setprecision(14) << "w norm l2 = " << w.Norml2() << std::endl;
   ParGridFunction lam_gf(fluid_heat_fes);
   lam_gf.SetFromTrueDofs(w);

   BlockVector state_v(x, offsets);
   ParGridFunction qq_gf(fluid_heat_fes);
   qq_gf.SetFromTrueDofs(state_v.GetBlock(1)); 
   rho_tilde->ExchangeFaceNbrData();
   lam_gf.ExchangeFaceNbrData();
   qq_gf.ExchangeFaceNbrData();

   ParLinearForm adv_lf(V_fes);
   adv_lf.AddDomainIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, *SIMP_cf, u_cf));
   adv_lf.AddBdrFaceIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, *SIMP_cf, u_cf), inlet_bdr);
   adv_lf.AddInteriorFaceIntegrator(new StokesVelocityGradientLFIntegrator(*rho_tilde, qq_gf, lam_gf, *SIMP_cf, u_cf));
   adv_lf.Assemble();
   std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
 
   

   ParLinearForm bdr_flow_lf(V_fes);
   bdr_flow_lf.AddBdrFaceIntegrator(new BdrFlowVelocityGradientLFIntegrator(*rho_tilde, lam_gf, raw_inflow, u_cf, *SIMP_cf), inlet_bdr);
   bdr_flow_lf.Assemble(); 
   std::unique_ptr<HypreParVector> bdr_flow_vec(bdr_flow_lf.ParallelAssemble());
   //std::cout<<"norm bdr grad = " << bdr_flow_vec->Norml2() << std::endl;

   adv_vec->Add(1.0, *bdr_flow_vec);

   *adv_vec *= -rf*cf;

   // int max_bdr_attr = GlobalMax(comm, V_fes->GetParMesh()->bdr_attributes.Size() ? V_fes->GetParMesh()->bdr_attributes.Max() : 0);
   // brinkman_stokes_solver->VelocityBoundary().Clear();
   // Vector zero_v(V_fes->GetVDim());
   // zero_v = 0.0;
   // auto zero_cf = std::make_shared<VectorConstantCoefficient>(zero_v);
   // for (int attr = 1; attr <= max_bdr_attr; attr++)
   // {
   //    if (all_ess_bdr[attr-1] == 1) { 
   //       brinkman_stokes_solver->VelocityBoundary().Add(attr, zero_cf);
   //    }
   // }

   //*adv_vec *= -1.0;

   // 1. Create a BlockVector for the right-hand side
   BlockVector rhs_stokes(brinkman_stokes_solver->GetBlockOffsets());
   rhs_stokes = 0.0;
   
   // 2. Assign the assembled adjoint forcing to the velocity block (Block 0)
   //    (The pressure forcing in Block 1  remains 0.0 for incompressibility)
   rhs_stokes.GetBlock(0) = *adv_vec;
   MFEM_VERIFY(adv_vec->Size() == V_fes->GetTrueVSize(),
               "Adjoint velocity RHS does not match the local velocity size.");

   HYPRE_BigInt local_size = adv_vec->Size();
   HYPRE_BigInt global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, HYPRE_MPI_BIG_INT, MPI_SUM,
                 comm);
   MFEM_VERIFY(global_size == V_fes->GlobalTrueVSize(),
               "Adjoint velocity RHS does not match the global velocity size.");
 
   BlockVector vv(brinkman_stokes_solver->GetBlockOffsets());

   vv = 0.0; 


   brinkman_stokes_solver->MultTranspose(rhs_stokes, vv);

   br.GetBlock(1) = vv.GetBlock(0);
   br.GetBlock(2) = vv.GetBlock(1);

   // brinkman_stokes_solver->VelocityBoundary().Clear();
}

void Pseudo3DStokesOperator::ImplicitSolve(const real_t dt_pass, const Vector &x, Vector &k)
{
   BlockVector bx(const_cast<Vector&>(x), helper_offsets);
   BlockVector bk(k, helper_offsets);
   k = 0.0;
   adv_diff_oper->ImplicitSolve(dt_pass, bx.GetBlock(0), bk.GetBlock(0));
}

void Pseudo3DStokesOperator::AdjointImplicitSolve(const real_t dt_pass, const Vector &lam, Vector &k)
{
   BlockVector bl(const_cast<Vector&>(lam), helper_offsets);
   BlockVector bk(k, helper_offsets);
   k = 0.0;
   adv_diff_oper->AdjointImplicitSolve(dt_pass, bl.GetBlock(0), bk.GetBlock(0));
}

void Pseudo3DStokesOperator::ExplicitMultDesignGradient(const real_t dt_pass, Vector &dual_vector, Vector &x, Vector &dgdrho_tilde)
{
   BlockVector bl(dual_vector, helper_offsets);
   BlockVector bx(x, helper_offsets);
   adv_diff_oper->ExplicitMultDesignGradient(dt_pass, bl.GetBlock(0), bx.GetBlock(0), dgdrho_tilde);

   // ParLinearForm mass_b_lf(filter_fes);
   // v_gf.SetFromTrueDofs(bl.GetBlock(1));
   // mass_b_lf.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(*rho_tilde, u_gf, v_gf, brinkman_cf));
   // mass_b_lf.Assemble();
   // std::unique_ptr<HypreParVector> mass_b_vec(mass_b_lf.ParallelAssemble());
   // dgdrho_tilde.Add(dt_pass, *mass_b_vec);  

}

void Pseudo3DStokesOperator::ImplicitSolveDesignGradient(const real_t dt_pass, const real_t a,Vector &dual_vector, Vector &x, Vector &dfdrho_tilde)
{
   BlockVector bl(dual_vector, helper_offsets);
   BlockVector bx(x, helper_offsets);
   adv_diff_oper->ImplicitSolveDesignGradient(dt_pass, a, bl.GetBlock(0), bx.GetBlock(0), dfdrho_tilde);
}

void Pseudo3DStokesOperator::AddStaticDesignGradient(const Vector &lam, Vector &dgdrho_tilde)
{
   // Extract the accumulated velocity adjoint from Block 1
   BlockVector bl(const_cast<Vector&>(lam), helper_offsets);
   v_gf.SetFromTrueDofs(bl.GetBlock(1)); 
   
   ParLinearForm mass_b_lf(filter_fes);
   mass_b_lf.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(*rho_tilde, u_gf, v_gf, brinkman_cf));
   mass_b_lf.Assemble();
   std::unique_ptr<HypreParVector> mass_b_vec(mass_b_lf.ParallelAssemble());

   dgdrho_tilde.Add(-1.0, *mass_b_vec);  
}

}


#endif 