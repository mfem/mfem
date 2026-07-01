#include "mfem.hpp"
#include <cmath>
#include <memory>

using namespace std;
using namespace mfem;


// SIMP coefficient: r(rho~) = E_min + rho~^exponent (E_max - E_min).
class SIMPCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter;
   real_t E_min, E_max, exponent;

public:
   SIMPCoefficient(GridFunction *rho_filter_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_filter(rho_filter_), E_min(E_min_), E_max(E_max_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      // r(rho~) = E_min + rho~^exponent (E_max - E_min)
      return E_min + std::pow(val, exponent) * (E_max - E_min);
   }
};

class SIMPGradCoefficient : public SIMPCoefficient
{
public:
   using SIMPCoefficient::SIMPCoefficient;   // inherits members + constructor

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      // r'(rho~) = exponent * rho~^(exponent-1) (E_max - E_min)
      return exponent * std::pow(val, exponent - 1.0) * (E_max - E_min);
   }
};

// Strain-energy-density (compliance sensitivity) coefficient.
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient *lambda = nullptr;
   Coefficient *mu = nullptr;
   GridFunction *u = nullptr;             // displacement state
   DenseMatrix grad;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_, 
                                  GridFunction *u_)
      : lambda(lambda_), mu(mu_), u(u_)
   {
      MFEM_ASSERT(u, "displacement not set");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t L = lambda->Eval(T, ip);
      real_t M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      real_t div_u = grad.Trace();

      // psi0(u) = lambda (div u)^2 + 2 mu |eps(u)|^2  (strain energy density)
      real_t density = L * div_u * div_u;
      int dim = T.GetSpaceDim();
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            density += M * grad(i, j) * (grad(i, j) + grad(j, i));
         }
      }
      return density;
   }
};

// Time stepping for du/dt = M^-1 (K x + b)
class DG_AdvectionSolver : public TimeDependentOperator
{
private:
   MPI_Comm comm;

   DGMassInverse *m_inv;          // M^{-1}, element-local DG mass inverse

   VectorCoefficient &v_cf;
   HypreParMatrix *Kmat;          // advection operator K

   const Vector *b;
   mutable Vector z;

public:
   DG_AdvectionSolver(HypreParMatrix &Kmat_, ParFiniteElementSpace &fes_,
                      VectorCoefficient &v_cf_)
      : TimeDependentOperator(Kmat_.Height()),
        comm(Kmat_.GetComm()), m_inv(nullptr),
        v_cf(v_cf_), Kmat(&Kmat_),
        b(nullptr), z(Kmat_.Height())
   {
      m_inv = new DGMassInverse(fes_);
   }

   ~DG_AdvectionSolver()
   {
      delete m_inv;
   }

   // set the (constant) source vector b in  du/dt = M^-1 (K x + b)
   void SetSource(const Vector &b_) { b = &b_; }
   HypreParMatrix &GetKMat() { return *Kmat; }

   // forward rate:  y = M^{-1} (K x + b)
   void Mult(const Vector &x, Vector &y) const override
   {
      Kmat->Mult(x, z);
      if (b) { z += *b; }
      m_inv->Mult(z, y);
   }
};

// Pseudo-transient solver for the advection equation  K rho_a = -b,  b = N rho_p
// forward : Mult   -> direct solve  K rho_a = -b
//           FSolve -> pseudo-transient march  M(drho_a/dt)=K rho_a+b
// adjoint : ASolve -> linear solve  K^T lambda = rhs  (steady state)
class PseudoTransientSolver : public Operator
{
private:
   MPI_Comm comm;
   ParFiniteElementSpace *dgfes;   // DG space of rho_a

   VectorCoefficient &v_cf;        // advection direction
   GridFunctionCoefficient rf_gfcf;  // rho_filter coefficient

   ParGridFunction rho_a_gf;       // forward DG field 
   ParGridFunction *rho_filter;    // source rho_p 

   ParBilinearForm *K;
   HypreParMatrix  *Kmat;
   DG_AdvectionSolver *adv;
   mutable Vector b;               // source  b = N rho_p  
   Vector adj_rhs;                 // adjoint RHS 
   Vector lambda;

   real_t dt = 1e-3;               // pseudo-transient time step

   // forward transport solve  K rho_a = -b   (direct, design-independent K)
   std::unique_ptr<BlockILU>       fwd_prec;
   std::unique_ptr<GMRESSolver>    fwd_gmres;

   // adjoint transport solve  K^T lambda = rhs
   std::unique_ptr<HypreParMatrix> Kt;
   std::unique_ptr<BlockILU>       adj_prec;
   std::unique_ptr<GMRESSolver>    adj_gmres;

   void Assemble();
   void AssembleSource(const ParGridFunction &rf) const;  // b = N rf
   void rho_a_init(Vector &r_a);
   void SetupForwardSolver();
   void SetupAdjointSolver();

public:
   PseudoTransientSolver(ParGridFunction &rho_a_, ParGridFunction &rho_filter_,
                         VectorCoefficient &v_cf_)
   : v_cf(v_cf_), rho_a_gf(rho_a_), rho_filter(&rho_filter_), rf_gfcf(&rho_filter_),
     K(nullptr), Kmat(nullptr), adv(nullptr)
   {
      dgfes = rho_a_gf.ParFESpace();
      comm  = dgfes->GetComm();

      K = new ParBilinearForm(dgfes);

      Assemble();                                    // -> Kmat
      adv = new DG_AdvectionSolver(*Kmat, *dgfes, v_cf);

      SetupForwardSolver();                          // K   solver, reused every solve
      SetupAdjointSolver();                          // K^T solver, reused every solve
   }

   ~PseudoTransientSolver()
   { delete adv; delete Kmat; delete K; }

   ParGridFunction &GetRhoA() { return rho_a_gf; }
   Vector &GetAdjoint() { return lambda; }
   void GetFilterGrad(Vector &dGdrf) const;        // dG/drho_filter = -N^T lambda 

   // set the adjoint RHS from an assembled dG/drho_a
   void SetAdjointRHS(const Vector &rhs) { adj_rhs = rhs; }
   void SetTimeStep(const real_t dt_) { dt = dt_; }

   // forward direct steady solve  K r_a = -b(r_f)
   void Mult(const Vector &r_f, Vector &r_a) const override;
   void FSolve();    // time march forward solve
   void ASolve();    // solve for lambda -> K^T lambda = (rhs)
};

// K (upwind discretization of -v.grad)
// M drho_a/dt = K rho_a + b.
void PseudoTransientSolver::Assemble()
{
   constexpr real_t alpha = -1.0;
   K->AddDomainIntegrator(new ConvectionIntegrator(v_cf, alpha));
   K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(v_cf, alpha));
   K->AddBdrFaceIntegrator (new NonconservativeDGTraceIntegrator(v_cf, alpha));
   K->Assemble();
   K->Finalize();
   Kmat = K->ParallelAssemble();
}

// source  b = N rf = INT rf w_dg
void PseudoTransientSolver::AssembleSource(const ParGridFunction &rf) const
{
   ParLinearForm src(dgfes);
   GridFunctionCoefficient rf_cf(&rf);
   src.AddDomainIntegrator(new DomainLFIntegrator(rf_cf));
   src.Assemble();
   std::unique_ptr<HypreParVector> bv(src.ParallelAssemble());
   b = *bv;
   adv->SetSource(b);
}

void PseudoTransientSolver::rho_a_init(Vector &r_a)
{
   ParGridFunction rf_dg_gf(dgfes);
   rf_dg_gf.ProjectDiscCoefficient(rf_gfcf);
   rf_dg_gf.GetTrueDofs(r_a);
}

// gmres solver for K^-1  (forward steady transport)
void PseudoTransientSolver::SetupForwardSolver()
{
   const int bs = dgfes->GetFE(0)->GetDof();   // DG dofs/element
   fwd_prec  = std::make_unique<BlockILU>(*Kmat, bs);
   fwd_gmres = std::make_unique<GMRESSolver>(comm);
   fwd_gmres->SetKDim(100);
   fwd_gmres->SetRelTol(1e-12);
   fwd_gmres->SetAbsTol(0.0);
   fwd_gmres->SetMaxIter(5000);
   fwd_gmres->SetPrintLevel(0);
   fwd_gmres->SetOperator(*Kmat);
   fwd_gmres->SetPreconditioner(*fwd_prec);
}

// gmres solver for K^-T
void PseudoTransientSolver::SetupAdjointSolver()
{
   Kt.reset(adv->GetKMat().Transpose());
   const int bs = dgfes->GetFE(0)->GetDof();   // DG dofs/element
   adj_prec  = std::make_unique<BlockILU>(*Kt, bs);
   adj_gmres = std::make_unique<GMRESSolver>(comm);
   adj_gmres->SetKDim(100);
   adj_gmres->SetRelTol(1e-12);
   adj_gmres->SetAbsTol(0.0);
   adj_gmres->SetMaxIter(5000);
   adj_gmres->SetPrintLevel(0);
   adj_gmres->SetOperator(*Kt);
   adj_gmres->SetPreconditioner(*adj_prec);
}

// forward direct steady solve:  rho_filter -> rho_a
// assemble the DG source b = N r_f, then solve  K r_a = -b.
void PseudoTransientSolver::Mult(const Vector &r_f, Vector &r_a) const
{
   ParGridFunction rf_gf(rho_filter->ParFESpace());
   rf_gf.SetFromTrueDofs(r_f);
   AssembleSource(rf_gf);         // b = N r_f  (H1 -> DG)

   Vector rhs(b); rhs.Neg();      // rhs = -b
   r_a.SetSize(adv->Height());
   r_a = 0.0;
   fwd_gmres->Mult(rhs, r_a);     // K r_a = -b   =>   r_a = -K^{-1} b
}

// forward pseudo-transient solve:  M (drho_a/dt) = K rho_a + b
void PseudoTransientSolver::FSolve()
{
   AssembleSource(*rho_filter);   // b = N rho_filter
   const int n = adv->Height();

   Vector r_a(n), rate(n);
   rho_a_init(r_a);     // initial guess for rho_a
   
   RK3SSPSolver ode; ode.Init(*adv);

   real_t t = 0.0;     // explicit step  (set < CFL limit)
   real_t eps = 1e-10;            // steady-state stop tolerance
   real_t res = 2.0 * eps;

   const int max_steps = 100000;  // safety cap
   int step = 0;
   while (res > eps && step < max_steps)
   {
      adv->Mult(r_a, rate);                       // rate = M^{-1}(K rho + b)
      res = std::sqrt(InnerProduct(comm, rate, rate));
      ode.Step(r_a, t, dt);                       // 3-stage SSP-RK3 step
      step++;
   }

   rho_a_gf.SetFromTrueDofs(r_a);
}

// steady adjoint transport:  K^T lambda = adj_rhs
void PseudoTransientSolver::ASolve()
{
   MFEM_VERIFY(adj_rhs.Size() > 0, "call SetAdjointRHS() before ASolve()");
   lambda.SetSize(dgfes->GetTrueVSize());
   lambda = 0.0;
   adj_gmres->Mult(adj_rhs, lambda);
}

// dG/drho_filter (H1) = -N^T lambda
void PseudoTransientSolver::GetFilterGrad(Vector &dGdrf) const
{
   ParGridFunction lam_gf(dgfes);
   lam_gf.SetFromTrueDofs(lambda);
   GridFunctionCoefficient lam_cf(&lam_gf);

   ParLinearForm lf(rho_filter->ParFESpace());
   lf.AddDomainIntegrator(new DomainLFIntegrator(lam_cf));
   lf.Assemble();
   std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());
   dGdrf = *v;
   dGdrf.Neg();
}