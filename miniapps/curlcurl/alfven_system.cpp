// 
// Put it in mfem/example and compile  
//
// we solve system: start with this: (M + nu dt K2) B^{n+1} = M * B0 + K1 * V0
//
// Sample run: mpirun -np 4 alfven -dt 0.01 -o 2
//
// TODO :
//  + Change AA class to cal FPMult() rather than Mult, then just
//  define special FPMult() function in AlfvenOperator class.
//  + This doesnt work because it is only Operator in AA, need more mods...


#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <set>
#include <deque>

#include "mfem.hpp"

using namespace mfem;

double c0 = 10.0;
void computeB0(const Vector& x, Vector& B0);
void computeV0(const Vector& x, Vector& V0);

// next three functions are used in earlier versions of the code
void computeCrossB0(const Vector& x, DenseMatrix& CrossB0);
void computeCrossB0transpose(const Vector& x, DenseMatrix& CrossB0);
void computeOmega(const Vector& x, DenseMatrix& omega);

/// Nonlinear Picard fixed-point iteration
class PicardIteration : public IterativeSolver
{
public:
  PicardIteration() : IterativeSolver() { }

#ifdef MFEM_USE_MPI
  PicardIteration(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif

  void Mult(const Vector &x, Vector &y) const { }

  virtual void SetOperator(const Operator &op)
  {
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square Operator is required.");
  }

  void Solve(Vector &x) const
  {
    Vector temp = x;
    double resid = std::sqrt(InnerProduct(GetComm(), temp, temp));
    if (print_level == 1) {
       mfem::out << "  Picard iteration : " << std::setw(3) << 0
               << "  ||u|| = " << resid << std::endl;
    }
    double final_norm = std::max(rel_tol*resid, abs_tol);

    int iter = 0;
    while (resid > final_norm) {
      iter ++;
      oper->Mult(x, x);
      temp -= x;                 // Measure resid
      resid = std::sqrt(InnerProduct(GetComm(), temp, temp));
      temp = x;
      if (print_level == 1) {
         mfem::out << "  Picard iteration : " << std::setw(3) << iter
                 << "  ||G(u) - u|| = " << resid << std::endl;
      }

      if (iter >= max_iter) {
        mfem_warning("Nonlinear iteration did not converge!");
        break;
      }
    }
  }
};

/// Nonlinear Anderson Acceleration
class AndersonAcceleration : public IterativeSolver
{
protected:
   int maxVecs; // see SetKDim()
   int AAstart;
   bool restart;
   bool isFixedPointOp;
   double omega;

   /// Apply fixed-point Mult and compute residual.
   //  For !isFixedPointOp:
   //    y <-- G(x) = x + A(x) - b and r <-- G(x) - x = A(x) - b
   //  For isFixedPointOp:
   //    y <-- G(x) = A(x) - b and r <-- G(x) - x = A(x) - b - x
   void FixedPointMult(const Vector &x, Vector &y, Vector &r) const
   {
      // Assume Operator represents a fixed-point operator *with
      // right-hand side*, so we iterate x_{k+1} = M^{-1}G(x_k)
      oper->Mult(x, y);
      r = y;
      r -= x;
   }

   // Helper function for Anderson Acceleration
   void QRdelete(std::deque<Vector *> &Q, DenseMatrix &R) const
   {
      Vector temp(Q[0]->Size());
      for (int i=0; i<(maxVecs-1); i++) {
         double d = sqrt( R(i,i+1)*R(i,i+1) + R(i+1,i+1)*R(i+1,i+1) );
         double c = R(i,i+1) / d;
         double s = R(i+1,i+1) / d;
         R(i,i+1) = d;
         R(i+1,i+1) = 0;

         if (i < (maxVecs-2)) {
            for (int j=(i+2); j<maxVecs; j++) {
               d = c*R(i,j) + s*R(i+1,j);
               R(i+1,j) = -s*R(i,j) + c*R(i+1,j);
               R(i,j) = d;
            }
         }
         // temp = c*Q[i] + s*Q[i+1];
         add(c, *(Q[i]), s, *(Q[i+1]), temp);
         // Q[i+1] = -s*Q[i] + c*Q[i+1];
         *(Q[i+1]) *= c;
         Q[i+1] -> Add(-s, *(Q[i]));
         *(Q[i]) = temp;
      }
      
      // Shift Q <- Q[:,0:(m-2)], i.e., delete last column of Q
      delete Q.back();
      Q.pop_back();

      // Shift columns of R to the left by one, R = R[0:(m−2), 1:(m-1)]
      for (int j=1; j<maxVecs; j++) {
         for (int i=0; i<maxVecs; i++) {
            R(i,j-1) = R(i,j);
         }
      }
   }

public:
   AndersonAcceleration() : IterativeSolver(), maxVecs(25), AAstart(0),
    omega(1), restart(false), isFixedPointOp(false) { }

#ifdef MFEM_USE_MPI
   AndersonAcceleration(MPI_Comm _comm) : IterativeSolver(_comm),
      maxVecs(25), AAstart(0), omega(1), restart(false),
      isFixedPointOp(false) { }
#endif

   /// Maximum number of vectors to store in Krylov-like space
   void SetKDim(int dim) { maxVecs = dim; }

   /// Number of fixed-point iterations to do before starting AA
   void SetAAStart(int start_) { AAstart = start_; }

   /// Boolean to restart, that is, erase entire space after maxVecs
   //  are stored (AAstart=true) or use a sliding space (AAstart=false)
   //  where one vector is deleted to make room for a new one.
   void SetRestart(bool restart_) { restart = restart_; }

   /// Set relaxation weight
   void SetWeight(double omega_) { omega = omega_; }

   void Mult(const Vector &x, Vector &y) const { }

   virtual void SetOperator(const Operator &op)
   {
      oper = &op;
      height = op.Height();
      width = op.Width();
      MFEM_ASSERT(height == width, "square Operator is required.");
   }

   void Solve(Vector &x) const
   {
      MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
     
      int n = width;
      int numVecs = 0;
      double resid, norm_df;
      double min_diag = 1e-13;
      final_norm = -1;

      // Check vector is initialized, set to zero for
      // iterative_mode = false
      if (x.Size() != n) {
         std::cout << "Warning!!! - AA input vector wrong size!\n";
      }

      // Storage containers for acceleration
      std::deque<Vector *> G;
      std::deque<Vector *> Q;
      DenseMatrix R(maxVecs);
      R = 0.0;
      Vector g_old(n);
      Vector g_current(n);
      Vector f_old(n);
      Vector f_current(n);
      Vector gamma(maxVecs);
      Vector rhs(maxVecs);
      Vector correction;
      if (omega > 0 && std::abs(omega -  1) > 1e-14) {
         correction.SetSize(n);
      }
      Vector *dg;
      Vector *df;

      resid = std::sqrt(InnerProduct(GetComm(), x, x));
      if (print_level == 1) {
         mfem::out << "  AA iteration : " << std::setw(3) << 0
                 << "  ||u|| = " << resid << std::endl;
      }
      if (final_norm < 0) {
         final_norm = std::max(rel_tol*resid, abs_tol);
      }

      // Loop over AA iterations
      int k;
      for (k=0; k<max_iter; k++) {

         // Compute g_current = G(x), f_current = G(x) - x
         this->FixedPointMult(x, g_current, f_current); 

         // Check norm of current approximation to fixed point G(u) = u
         resid = Norm(f_current);
         MFEM_ASSERT(IsFinite(resid), "||G(u) - u|| = " << resid);
         if (print_level == 1)
         {
            mfem::out << "  AA iteration : " << std::setw(3) << k+1
                    << "  ||G(u) - u|| = " << resid << std::endl;
         }

         // Check for convergence
         if (resid <= final_norm)
         {
            final_norm = resid;
            final_iter = k;
            converged = 1;
            goto finish;
         }

         // Start Anderson Acceleration after AAstart FP iterations
         if (k > AAstart) {
            // df = f_current - f_old; 
            df = new Vector(n);
            add(1.0, f_current, -1.0, f_old, *df);
            // dg = g_current - g_old;
            dg = new Vector(n);
            add(1.0, g_current, -1.0, g_old, *dg);
         
            if (numVecs < maxVecs) {
               G.push_back(dg);
            }
            else {
               delete G[0];
               G.pop_front();
               G.push_back(dg);
            }
            numVecs++;
            dg = NULL;
         }
         
         f_old = f_current;
         g_old = g_current;
         
         // First iteration or initial fixed-point iterations
         if (numVecs == 0) {
            x = g_current;
            continue;
         }

         // All later iterations: orthogonalize and find best approximation
         if (numVecs == 1) {
            norm_df = Norm(*df);
            MFEM_ASSERT(IsFinite(norm_df), "norm_df = " << norm_df);
            (*df) /= norm_df;
            Q.push_back(df);
            R(0,0) = norm_df;
            df = NULL;
         }
         else {
            // Remove first column in basis F and R, reorthogonalize
            if (numVecs > maxVecs) {
               this->QRdelete(Q, R);
               numVecs--;
            }
            // Compute last column of R
            for (int i=0; i<(numVecs-1); i++) {
               R(i,numVecs-1) = Dot(*(Q[i]), *df);
               // df -= R(i,numVecs-1) * Q[i]
               df -> Add(-R(i,numVecs-1), *(Q[i]));
            }
            norm_df = Norm(*df);
            MFEM_ASSERT(IsFinite(norm_df), "norm_df = " << norm_df);
            (*df) /= norm_df;
            Q.push_back(df);
            R(numVecs-1, numVecs-1) = norm_df;
            df = NULL;
         }

         // Back solve for new weights, R\gamma = Q^T * f_current
         rhs = 0.0;
         gamma = 0.0;
         for (int i=0; i<numVecs; i++) {
            rhs(i) = Dot(*(Q[i]), f_current);   // Form right hand side
         }
         for (int i=(numVecs-1); i>=0; i--) {
            double temp = rhs(i);
            for (int j=(i+1); j<numVecs; j++) {
               temp -= R(i,j)*gamma(j);
            }
            if (std::abs(R(i,i)) < min_diag) {
               gamma(i) = 0.0;
               std::cout << "Diagonal of R -- " << R(i,i) << " ~ 0.\n";
            }
            else {
               gamma(i) = temp / R(i,i);            
            }
         }

         /// DEBUG --> test backsolve
         Vector test(numVecs);
         for (int i=0; i<numVecs; i++) {
            test(i) = 0;
            for (int j=i; j<numVecs; j++) {
               test(i) += R(i,j) * gamma(j);
            }
            if (std::abs(rhs(i) - test(i)) > 1e-10) {
               std::cout << "Bad solve! Err = " << rhs(i) - test(i) << "\n";
            }
         }

         // Compute updated solution x = g_current − G*\gamma 
         x = g_current;
         for (int i=0; i<numVecs; i++) {
            // x -= gamma(i)*G[i]
            x.Add(-gamma(i), *(G[i]));
         }

         // Apply damped iteration for \omega \in (0,1),
         //   x -= (1−omega) * (f_current − Q*R*gamma);
         if (omega > 0 && std::abs(omega -  1) > 1e-14) {
            // Redefine rhs = R*gamma
            for(int i=0; i<numVecs; i++) {
               rhs(i) = 0;
               for (int j=i; j<numVecs; j++) {
                  rhs(i) += R(i,j)*gamma(j);
               }
            }
            correction = f_current;
            for (int i=0; i<numVecs; i++) {
               // correction -= rhs(i)*Q[i]
               correction.Add(-rhs(i), *(Q[i]));
            }
            // x -= (1 - omega) * correction;
            x.Add( -(1 - omega), correction);
         }

         // Restart AA minimization by eliminating all vectors but the most recent
         if (restart && (numVecs == maxVecs)) {
            for (int i=0; i<(maxVecs-1); i++) {
               delete G[0];
               G.pop_front();          
               delete Q[0];
               Q.pop_front();
            }
            R = 0.0;
            R(0,0) = norm_df;
            numVecs = 1;
            if (print_level == 1)
            {
               mfem::out << "Restarting..." << '\n';
            }
         }
      }

      // Compute final residual, save counts for solve
      this->FixedPointMult(x, g_current, f_current); 
      resid = Norm(f_current);
      MFEM_ASSERT(IsFinite(resid), "||G(u) - u|| = " << resid);
      final_norm = resid;
      final_iter = max_iter;
      if (resid <= final_norm) converged = 1;
      else converged = 0;

   finish:
      if (print_level == 3)
      {
         mfem::out << "  Iteration : " << std::setw(3) << k
                 << "  ||G(u) - u|| = " << resid << std::endl;
      }
      else if (print_level == 2)
      {
         mfem::out << "Anderson Acceleration: Number of iterations: " << final_iter << '\n';
      }
      if (print_level >= 0 && !converged)
      {
         mfem::out << "Anderson Acceleration: No convergence!\n";
      }

      // Cleanup pointers
      for (int i=0; i<numVecs; i++) {
         delete G[0];
         G.pop_front();          
         delete Q[0];
         Q.pop_front();
      }
      delete dg;
      delete df;
   }
};

/*******************************************************************
* Alfven time-dependent non-linear operator. Explicit interface 
* solves the following system of ODEs:
*
*   dV/dt = inv(Mv) * [-B x (\nabla x B)]
*   dB/dt = inv(Mb) * [\nabla x (V x B) - 1/S \nabla x \nabla x B]
*
* where V in H_0(div) and B in H(curl). 
****************************************************************** */
class AlfvenOperator : public TimeDependentOperator {
public:
  AlfvenOperator(std::shared_ptr<ParFiniteElementSpace>& vspace,
                 std::shared_ptr<ParFiniteElementSpace>& bspace,
                 Array<int>& ess_tdof_list,
                 double gmres_tol,
                 int nonlinear_solve,
                 int myid);
  ~AlfvenOperator() {
    // if (A11_) delete A11_;
  }

  // required interface
  virtual void Mult(const Vector& u, Vector& du_dt) const;
  virtual void ImplicitSolve(const double dt, const Vector& u, Vector& du_dt);

  // other member functions
  void Init(double eta);
  void UpdateMatrices(const Vector& u) const;
  void UpdateLinearSolver(double dt, const Vector& u) const;
  void FormBlockSystem(int mode, double dt=0) const;
  void MassMult(const Vector& u, Vector& v);

  // getters / setters
  std::shared_ptr<BlockOperator> get_system() { return system_; }
  Array<int> get_block_offsets() { return block_offsets_; }
  Array<int> get_block_true_offsets() { return block_true_offsets_; }
  HypreParMatrix* get_system_block(int i, int j) { return static_cast<HypreParMatrix*>(&system_->GetBlock(i, j)); }

  std::pair<int, int> get_explicit_iterations() { return std::make_pair(itrs0_, itrs1_); }
  int get_implicit_iterations() { return itrs2_; }

private:
  std::shared_ptr<ParFiniteElementSpace> vspace_, bspace_;
  int ntv_, ntb_, myid_, nonlinear_solve_;
  bool fixed_point_mult_;

  mutable std::shared_ptr<BlockOperator> system_;
  Array<int> block_offsets_, block_true_offsets_; // number of variables + 1
  Array<int>& ess_tdof_list_;

  double eta_;
  mutable double current_dt_;

  std::shared_ptr<HypreParMatrix> M00_, M11_, S11_;
  mutable std::shared_ptr<HypreParMatrix> P11add_;
  mutable std::shared_ptr<HypreParMatrix> A11_, B01_;
  mutable TransposeOperator *B10_;

  mutable std::shared_ptr<GMRESSolver> gmres_; // solver for M u + dt K u, where u = (V, B)
  double gmres_tol_;
  Vector z_;

  mutable std::shared_ptr<BlockDiagonalPreconditioner> pc_;
  mutable std::shared_ptr<HypreSolver> inv00_, inv11_;

  // statistics
  mutable int itrs0_, itrs1_, itrs2_;
};


/*******************************************************************
* Alfven operator constructor
****************************************************************** */
AlfvenOperator::AlfvenOperator(std::shared_ptr<ParFiniteElementSpace>& vspace,
                               std::shared_ptr<ParFiniteElementSpace>& bspace,
                               Array<int>& ess_tdof_list,
                               double gmres_tol,
                               int nonlinear_solve,
                               int myid)
  : TimeDependentOperator(vspace->TrueVSize() + bspace->TrueVSize(), 0.0), 
    vspace_(vspace),
    bspace_(bspace),
    ess_tdof_list_(ess_tdof_list),
    gmres_tol_(gmres_tol),
    nonlinear_solve_(nonlinear_solve),
    myid_(myid),
    fixed_point_mult_(false)
{
  // A11_ = nullptr; // FIXME (shares ptr?)
}


/*******************************************************************
* Alfven operator API for explicit time integration
****************************************************************** */
void AlfvenOperator::Mult(const Vector& u, Vector& du_dt) const
{
  // Modified mult to specifically use in Anderson Acceleration
  // Fixed-point solve
  if (fixed_point_mult_) {
    UpdateMatrices(u);                  // Update linearized nonlinear operator L[x]
    FormBlockSystem(2, current_dt_);    // Form matrix (M - dt*L[x])
    UpdateLinearSolver(current_dt_, u); // Construct preconditioner
    gmres_->Mult(z_, du_dt);            // Solve linearized system
    if (myid_ == 0) {
      std::cout << "\tgmres iterations: " << gmres_->GetNumIterations() << std::endl;
    }
  }
  // Standard Mult as used by, e.g., explicit time integration
  else {
    // Update nonlinear operators
    UpdateMatrices(u);
    FormBlockSystem(1);

    Vector f(u);
    system_->Mult(u, f); 

    CGSolver M00_solver(MPI_COMM_WORLD), M11_solver(MPI_COMM_WORLD);
    HypreSmoother M00_prec, M11_prec;

    M00_solver.iterative_mode = false;
    M00_solver.SetRelTol(1e-6);
    M00_solver.SetAbsTol(0.0);
    M00_solver.SetMaxIter(30);
    M00_solver.SetPrintLevel(0);

    M00_prec.SetType(HypreSmoother::Jacobi);
    M00_solver.SetPreconditioner(M00_prec);
    M00_solver.SetOperator(*M00_);

    M11_solver.iterative_mode = false;
    M11_solver.SetRelTol(1e-6);
    M11_solver.SetAbsTol(0.0);
    M11_solver.SetMaxIter(30);
    M11_solver.SetPrintLevel(0);

    M11_prec.SetType(HypreSmoother::Jacobi);
    M11_solver.SetPreconditioner(M11_prec);
    M11_solver.SetOperator(*M11_);

    Vector v(f.GetData(), ntv_);
    Vector dv_dt(du_dt.GetData(), ntv_);
    M00_solver.Mult(v, dv_dt);
    itrs0_ = M00_solver.GetNumIterations();

    Vector b(f.GetData() + ntv_, ntb_);
    Vector db_dt(du_dt.GetData() + ntv_, ntb_);
    M11_solver.Mult(b, db_dt);
    itrs1_ = M11_solver.GetNumIterations();

    // negate the time rate, see the definition of the block system for 
    // explicit time integrator
    du_dt.Neg();

    if (myid_ == 0) {
      std::cout << "  mass matrix solvers: " << itrs0_ << " " << itrs1_ << std::endl;
    }
  }
}

/*******************************************************************
* Alfven operator API for implicit time integration
****************************************************************** */
void AlfvenOperator::ImplicitSolve(const double dt, const Vector& u, Vector& du_dt)
{
  // Here we use Picard iterations to solve a nonlinear equation
  // for the Runge-Kutta stage vector k,
  //
  //    M*k = N(u+dt*k)                     (1)
  //
  // for nonlinear operator N. We assume N can be written as
  //
  //    N(u+dt*k) := L[u+dt*k](u+dt*k) + f(t)
  //
  // where L is a matrix-valued operator evaluated at u+dt*k and f(t)
  // a (potentially zero) time-dependent forcing vector. (1) can be
  // rewritten as a fixed-point equation
  //
  //    x = (M - dt*L[x])^{-1} (Mu + f)     (2)
  //
  // where x := u + dt*k, which can be solved using a Picard iteration,
  // where a function G(x) = x is solved via iterations x_{k+1} = G(x_k).
  //
  //    Note, system_ = M + dtK

  double tol = 1e-6;
  int maxiter = 100;
  current_dt_ = dt;

  // Right-hand side for nonlinear iteration
  z_.SetSize(u.Size());    // Vector for right-hand side
  MassMult(u, z_);   // NOTE : Add forcing function here if one exists
  du_dt = u;        // Set u as initial guess for x (2)

  fixed_point_mult_ = true;
  if (nonlinear_solve_ == 0) {
    PicardIteration PI(MPI_COMM_WORLD);
    PI.SetOperator(*this);
    PI.SetRelTol(tol);
    PI.SetMaxIter(maxiter);
    PI.SetPrintLevel(1);
    PI.Solve(du_dt);
  }
  else {
    AndersonAcceleration AA(MPI_COMM_WORLD);
    AA.SetOperator(*this);
    AA.SetKDim(10);
    AA.SetRestart(true); // WORKS
    AA.SetAAStart(0);    // WORKS
    AA.SetWeight(1.0);   // Not robust but seems to work
    AA.SetRelTol(tol);
    AA.SetMaxIter(maxiter);
    AA.SetPrintLevel(1);
    AA.Solve(du_dt);
  }
  fixed_point_mult_ = false;

  // Above we solved for x = u + dt*k, where k is the desired update
  // Map du_dt -> k.
  du_dt -= u;
  du_dt /= dt;
}


/*******************************************************************
* Define 2x2 block system
****************************************************************** */
void AlfvenOperator::Init(double eta)
{
  block_offsets_.Append(0);
  block_offsets_.Append(vspace_->GetVSize());
  block_offsets_.Append(bspace_->GetVSize());
  block_offsets_.PartialSum();

  ntv_ = vspace_->TrueVSize();
  ntb_ = bspace_->TrueVSize();

  block_true_offsets_.Append(0);
  block_true_offsets_.Append(ntv_);
  block_true_offsets_.Append(ntb_);
  block_true_offsets_.PartialSum();

  system_ = std::make_shared<BlockOperator>(block_true_offsets_);

  eta_ = eta;

  // ---------- Precompute operators fixed for all time ---------- //
  // block 00 is mass matrix for velocity uses the unit coefficient
  ConstantCoefficient one(1.0);
  ParBilinearForm *form00 = new ParBilinearForm(vspace_.get());
  form00->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  form00->Assemble();
  form00->Finalize();

  // -- block 11 is mass matrix plus (1/S) curl-curl matrix for B
  ParBilinearForm *form11m = new ParBilinearForm(bspace_.get());
  form11m->AddDomainIntegrator(new VectorFEMassIntegrator(one));
  form11m->Assemble();
  form11m->Finalize();

  ParBilinearForm *form11s = new ParBilinearForm(bspace_.get());
  form11s->AddDomainIntegrator(new CurlCurlIntegrator(one));
  form11s->Assemble();
  form11s->Finalize();

  // TODO : why eliminate only for form00, not form11m/form11s?
  form00->EliminateEssentialBC(ess_tdof_list_);

  M00_ = std::shared_ptr<HypreParMatrix>(form00->ParallelAssemble());
  M11_ = std::shared_ptr<HypreParMatrix>(form11m->ParallelAssemble());
  S11_ = std::shared_ptr<HypreParMatrix>(form11s->ParallelAssemble());

  delete form00;
  delete form11m;
  delete form11s;
}

/*******************************************************************
* Apply mass matrices on both spaces, v = Mu, where u,v are block vectors
****************************************************************** */
void AlfvenOperator::MassMult(const Vector& u, Vector& v)
{
  Vector u0(u.GetData(), ntv_);
  Vector u1(u.GetData() + ntv_, ntb_);
  Vector v0(v.GetData(), ntv_);
  Vector v1(v.GetData() + ntv_, ntb_);

  M00_->Mult(u0, v0);  
  M11_->Mult(u1, v1);  
}

/*******************************************************************
* Update matrices using the previous time-step solution
****************************************************************** */
void AlfvenOperator::UpdateMatrices(const Vector& u) const
{
  // block 10 is <T0^t curl B, V'> = <B x curl B, V'>
  // MatrixFunctionCoefficient tensor01(3, computeCrossB0transpose);
  // auto form01 = std::make_shared<ParMixedBilinearForm>(bspace_.get(), vspace_.get());
  // form01->AddDomainIntegrator(new MixedVectorCurlIntegrator(tensor01));

  ParGridFunction bfun(bspace_.get());
  Vector u_b(u.GetData() + ntv_, ntb_); // shallow subvector
  bfun.SetFromTrueDofs(u_b);
  VectorGridFunctionCoefficient vector10(&bfun);
  auto form01 = std::make_shared<ParMixedBilinearForm>(bspace_.get(), vspace_.get());
  form01->AddDomainIntegrator(new MixedCrossCurlIntegrator(vector10));
  form01->Assemble();
  form01->Finalize();

  form01->EliminateTestDofs(ess_tdof_list_);
  B01_ = std::shared_ptr<HypreParMatrix>(form01->ParallelAssemble());

  // -- block 10 is negative transport of block 01
  //    we use transpose, but should revisit due to different boundary conditions
  B10_ = new TransposeOperator(B01_.get());
}


/*******************************************************************
* Update solver and preconditioner using the previous solution
****************************************************************** */
void AlfvenOperator::UpdateLinearSolver(double dt, const Vector& u) const
{
  // update matrices
  ParGridFunction bfun(bspace_.get());
  Vector u_b(u.GetData() + ntv_, ntb_);
  bfun.SetFromTrueDofs(u_b);
  VectorGridFunctionCoefficient vector11(&bfun);
  CrossCrossCoefficient tensorOmega(1.0, vector11);

  // NOTE: integrator takes ownership of the input pointer
  ParBilinearForm *form11p = new ParBilinearForm(bspace_.get());
  form11p->AddDomainIntegrator(new CurlCurlIntegrator(tensorOmega));
  form11p->Assemble();
  form11p->Finalize();

  double alpha = dt * dt;
  HypreParMatrix *P11;
  P11add_ = std::shared_ptr<HypreParMatrix>(form11p->ParallelAssemble());
  P11 = Add(1.0, *A11_, alpha, *P11add_);

  // form block-diagonal preconditioner
  inv00_ = std::make_shared<HypreDiagScale>(*M00_);
  inv11_ = std::make_shared<HypreAMS>(*P11, bspace_.get()); // no Schur complement: use A11
   
  inv00_->iterative_mode = false;
  inv11_->iterative_mode = false;
 
  pc_ = std::make_shared<BlockDiagonalPreconditioner>(block_true_offsets_);
  pc_->SetDiagonalBlock(0, inv00_.get());
  pc_->SetDiagonalBlock(1, inv11_.get());

  // linearized solver for implicit time integrator
  gmres_ = std::make_shared<GMRESSolver>(MPI_COMM_WORLD);
  gmres_->SetKDim(100);
  gmres_->SetOperator(*system_);
  gmres_->SetPreconditioner(*pc_);
  gmres_->SetRelTol(gmres_tol_);
  gmres_->SetMaxIter(5000);
  gmres_->SetPrintLevel(0);

  // cleaning
  delete form11p;
}


/*******************************************************************
* Form block system from elemental matrices
* mode:
*   0 : Qi's hardcoded implicit solver, requires dt
*   1 : explicit mult, no dt required
*   2 : implicit solve, dt required
* Only provide dt for type 0 or type 2.
****************************************************************** */
void AlfvenOperator::FormBlockSystem(int mode, double dt) const
{
  // a stand-alone one-time step solver, line in Qi's solver
  if (mode == 0) {
    A11_ = std::shared_ptr<HypreParMatrix>(Add(1.0, *M11_, eta_ * dt, *S11_));

    system_->SetBlock(0, 0, M00_.get());
    system_->SetBlock(0, 1, B01_.get(), dt);
    system_->SetBlock(1, 0, B10_, -dt);
    system_->SetBlock(1, 1, A11_.get());

  // an explicit time integrator du/dt + A(u) = 0
  } else if (mode == 1) {
    system_->SetBlock(0, 0, M00_.get(), 0.0); // FIXME
    system_->SetBlock(0, 1, B01_.get());
    system_->SetBlock(1, 0, B10_, -1.0);
    system_->SetBlock(1, 1, S11_.get(), eta_);

  // an implicit time integrator uses linearized operator T = M + dt K
  } else if (mode == 2) {
    A11_ = std::shared_ptr<HypreParMatrix>(Add(1.0, *M11_, eta_ * dt, *S11_));

    system_->SetBlock(0, 0, M00_.get()); // FIXME
    system_->SetBlock(0, 1, B01_.get(), dt);
    system_->SetBlock(1, 0, B10_, -dt);
    system_->SetBlock(1, 1, A11_.get());

  } else {
    exit(0);
  }
}


/*******************************************************************
* Alfven linearized system
****************************************************************** */
int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  // int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // default values for skipped input
  int nz = 3; // 3 is the minimal value for a periodic mesh
  int dim(3), order(1), refinement(1), ode_solver_type(11), nonlinear_solve(1);
  double dt(1.0/64), tfin(1.0), io_freq(0.002), eta(1e-5), gmres_tol(1e-6);
  bool visualization(false);

  // Parse command line
  OptionsParser args(argc, argv);
  args.AddOption(&nz, "-nz", "--num-elem", "number of elements.");
  args.AddOption(&order, "-o", "--order", "Finite element order RT(o-1) + ND(o).");
  args.AddOption(&refinement, "-r", "--refinement", "Refinement leveles for parallel mesh.");
  args.AddOption(&c0, "-c0", "--c0", "set c0 in the background B.");
  args.AddOption(&dt, "-dt", "--dt", "set dt.");
  args.AddOption(&tfin, "-tfin", "--tfin", "set final time.");
  args.AddOption(&io_freq, "-io", "--io-frequency", "time frequency of i/o snapshots.");
  args.AddOption(&eta, "-eta", "--eta", "set eta (i.e., 1/S).");

  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                 "            11 - Forward Euler, 12 - RK2,\n\t"
                 "            13 - RK3 SSP, 14 - RK4."
                 "            22 - Implicit Midpoint Method,\n\t"
                 "            23 - SDIRK23 (A-stable), 24 - SDIRK34");

  args.AddOption(&nonlinear_solve, "-nl-solve", "--nonlinear-solve", "nonlinear solver, Picard=0, AA=1 (default).");
  args.AddOption(&gmres_tol, "-gmres_tol", "--gmres_tol", "set tolerance for linear solver (GMRES).");

  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) args.PrintUsage(std::cout);
    return 1;
  }
  if (myid == 0) args.PrintOptions(std::cout);

  // Generate a base mesh
  Mesh orig_mesh = Mesh::MakeCartesian3D(nz, nz, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);

  // Make z direction periodic
  std::vector<Vector> translations = { Vector({0.0, 0.0, 1.0}) };

  Mesh mesh = Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));
  mesh.RemoveInternalBoundaries();

  // Refine serial mesh once
  mesh.UniformRefinement();

  // Save the final serial mesh
  std::ofstream mesh_ofs("periodic-cube-z.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
 
  // Define a parallel mesh by a partitioning of the serial mesh. Refine
  // this mesh further in parallel to increase the resolution. Once the
  // parallel mesh is defined, the serial mesh can be deleted.
  auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);
  auto ne0 = pmesh->GetGlobalNE();

  for (int l = 0; l < refinement; l++) {
    pmesh->UniformRefinement();
  }
  auto ne1 = pmesh->GetGlobalNE();
  if (myid == 0) {
    std::cout << "Number of cells in initial and final meshes: " << ne0 << " " << ne1 << std::endl;
  }

  FiniteElementCollection *rt_coll = new RT_FECollection(order - 1, dim);
  FiniteElementCollection *nd_coll = new ND_FECollection(order, dim);

  auto vspace = std::make_shared<ParFiniteElementSpace>(pmesh.get(), rt_coll);
  auto bspace = std::make_shared<ParFiniteElementSpace>(pmesh.get(), nd_coll);

  HYPRE_BigInt sizeb = bspace->GlobalTrueVSize();
  HYPRE_BigInt sizev = vspace->GlobalTrueVSize();
  if (myid == 0) {
    std::cout << "Number of finite element unknowns (B&V): " << sizeb << " " << sizev << std::endl;
  }

  // set up zero velocity on the whole boundary
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size() > 0) {
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    vspace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    if (myid == 0) {
      std::cout << "Number of velocity dofs slated for essential BCs: " << ess_tdof_list.Size() << std::endl;
    }
  }

  // parse initial conditions
  // -- velocity field
  ParGridFunction V0(vspace.get());
  VectorFunctionCoefficient Vcoeff(dim, computeV0);
  V0.ProjectCoefficient(Vcoeff);

  // -- magnetic field
  ParGridFunction B0(bspace.get());
  VectorFunctionCoefficient Bcoeff(dim, computeB0);
  B0.ProjectCoefficient(Bcoeff);

  // explicit time-stepping
  ODESolver *ode_solver;
  switch (ode_solver_type) {
    // Implicit L-stable methods
    case 1:  ode_solver = new BackwardEulerSolver; break;
    case 2:  ode_solver = new SDIRK23Solver(2); break;
    case 3:  ode_solver = new SDIRK33Solver; break;

    // Explicit methods
    case 11: ode_solver = new ForwardEulerSolver; break;
    case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
    case 13: ode_solver = new RK3SSPSolver; break;
    case 14: ode_solver = new RK4Solver; break;
    case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;

    // Implicit A-stable methods (not L-stable)
    case 22: ode_solver = new ImplicitMidpointSolver; break;
    case 23: ode_solver = new SDIRK23Solver; break;
    case 24: ode_solver = new SDIRK34Solver; break;

    default:
      if (myid == 0) std::cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      exit(1);
  }
  bool implicit = (ode_solver_type < 4 || ode_solver_type > 21) ? true : false;

  // initialized block system
  AlfvenOperator op(vspace, bspace, ess_tdof_list,
    gmres_tol, nonlinear_solve, myid);
  op.Init(eta);
  op.SetTime(0.0);
  ode_solver->Init(op);

  // initialzie solution vectors
  auto block_offsets = op.get_block_offsets();
  auto block_true_offsets = op.get_block_true_offsets();

  BlockVector sol(block_offsets), rhs(block_offsets);
  BlockVector trueSol(block_true_offsets), trueRhs(block_true_offsets);
   
  V0.GetTrueDofs(trueSol.GetBlock(0));
  B0.GetTrueDofs(trueSol.GetBlock(1));

  double vnorm = GlobalLpNorm(2.0, trueSol.GetBlock(0).Norml2(), MPI_COMM_WORLD);
  double bnorm = GlobalLpNorm(2.0, trueSol.GetBlock(1).Norml2(), MPI_COMM_WORLD);
  if (myid == 0) {
    std::cout << "  ||B0|| = " << bnorm << ", ||V0|| = " << vnorm << std::endl;
  }

  // initialize io
  int nloop(0);
  double t(0.0), dt_limited, t_io(dt);

  ParaViewDataCollection io("alfven_system", pmesh.get());
  io.SetPrefixPath("ParaView");
  io.SetLevelsOfDetail(order);
  io.SetDataFormat(VTKFormat::BINARY);
  io.SetHighOrderOutput(true);

  while (t < tfin) {
    dt_limited = std::min(dt, tfin - t);
    if (myid == 0) {
      std::cout << "\nCycle " << nloop << ":  t=" << t << " dt=" << dt_limited << std::endl;
    }

    ode_solver->Step(trueSol, t, dt_limited);

    double vnorm = GlobalLpNorm(2.0, trueSol.GetBlock(0).Norml2(), MPI_COMM_WORLD);
    double bnorm = GlobalLpNorm(2.0, trueSol.GetBlock(1).Norml2(), MPI_COMM_WORLD);
    if (myid == 0) {
      std::cout << "  ||B|| = " << bnorm << ", ||V|| = " << vnorm << std::endl;
    }

    if (visualization && std::fabs(t - t_io) < dt) {
      if (myid == 0) std::cout << "  paraview io..." << std::endl;
      ParGridFunction Vfinal(vspace.get());
      ParGridFunction Bfinal(bspace.get());

      Vfinal.SetFromTrueDofs(trueSol.GetBlock(0));
      Bfinal.SetFromTrueDofs(trueSol.GetBlock(1));

      io.SetCycle(nloop);
      io.SetTime(t);
      io.RegisterField("B", &Bfinal);
      io.RegisterField("V", &Vfinal);
      io.Save();

      t_io += io_freq;
    }

    nloop++;
  }
  if (myid == 0) std::cout << "SIMULATION SUCCESSFUL\n";

  delete rt_coll;
  delete nd_coll;
  delete ode_solver;

  return 0;
}


/*******************************************************************
* Compute B0
****************************************************************** */
void computeB0(const Vector& x, Vector& B0)
{
  double x0(0.5), y0(0.5);
  double Az = 0.5 * std::pow(x(0) - x0, 2.0) + 1.0/32.0 * pow(sin(2*M_PI * (x(1) - y0)), 2.0);
  B0(0) = M_PI/8.0 * std::sin(2 * M_PI * (x(1) - y0)) * cos(2*M_PI * (x(1) - y0));
  B0(1) = -x(0) + x0;
  B0(2) = c0 * fabs(Az);
}


/*******************************************************************
* Compute V0
****************************************************************** */
void computeV0(const Vector& x, Vector& V0)
{
  V0(0) = 0.0;
  V0(1) = 0.0;
  V0(2) = 0.0;
}


/*******************************************************************
* OBSOLETE. Compute tensor T in T w = (w x B0) or its transpose
****************************************************************** */
void computeCrossB0(const Vector& x, DenseMatrix& crossB0)
{
   crossB0.SetSize(3);
   Vector B0(3);

   computeB0(x, B0);
   
   crossB0(0, 1) = B0(2);
   crossB0(0, 2) =-B0(1);
   crossB0(1, 0) =-B0(2);
   crossB0(1, 2) = B0(0);
   crossB0(2, 0) = B0(1);
   crossB0(2, 1) =-B0(0);
}

void computeCrossB0transpose(const Vector& x, DenseMatrix& crossB0)
{
   crossB0.SetSize(3);
   Vector B0(3);

   computeB0(x, B0);
   
   crossB0(0, 1) =-B0(2);
   crossB0(0, 2) = B0(1);
   crossB0(1, 0) = B0(2);
   crossB0(1, 2) =-B0(0);
   crossB0(2, 0) =-B0(1);
   crossB0(2, 1) = B0(0);
}


/*******************************************************************
* OBSOLETE. Compute Omega = |B0|^2 I - B0 B0^T
****************************************************************** */
void computeOmega(const Vector& x, DenseMatrix& omega)
{
  omega.SetSize(3);
  Vector B0(3);

  computeB0(x, B0);

  double norm2 = B0.Norml2();
  norm2 *= norm2;
  
  double tol = norm2 * 0.00; // % of B0-norml
   
  omega(0, 0) = norm2 - B0(0) * B0(0) + tol; 
  omega(1, 1) = norm2 - B0(1) * B0(1) + tol;
  omega(2, 2) = norm2 - B0(2) * B0(2) + tol;
  omega(0, 1) =-B0(0) * B0(1);
  omega(0, 2) =-B0(0) * B0(2);
  omega(1, 2) =-B0(1) * B0(2);

  // symmetrize
  omega(1, 0) = omega(0, 1);
  omega(2, 0) = omega(0, 2);
  omega(2, 1) = omega(1, 2);
}
