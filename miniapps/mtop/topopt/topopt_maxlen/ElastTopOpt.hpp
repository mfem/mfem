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
// M and K are assembled externally and passed in by reference (not owned here).
class DG_AdvectionSolver : public TimeDependentOperator
{
private:
   MPI_Comm comm;

   DGMassInverse *m_inv;          // M^{-1}, element-local DG mass inverse

   VectorCoefficient &v_cf;
   HypreParMatrix *Kmat;          // not owned

   const Vector *b;               // optional constant source, not owned
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

// pseudo transient solver that solve the steady state by adding a time derivative
// forward solve: time stepping solve for  M (drho_a/dt) = K rho_a + b,  b = N rho_p
// adjoint solve: linear solve  K^T lambda = (rhs)  in steady state
class PseudoTransientSolver : public Operator
{
private:
   MPI_Comm comm;
   ParFiniteElementSpace *dgfes;   // DG space of rho_a (borrowed from rho_a_gf)

   VectorCoefficient &v_cf;        // advection direction

   ParGridFunction rho_a_gf;       // forward DG field (unknown / output)
   ParGridFunction *rho_filter;    // source rho_p (H1, live; borrowed from driver)

   // forward operators (owned)
   ParBilinearForm *K;
   HypreParMatrix  *Kmat;
   DG_AdvectionSolver *adv;
   mutable Vector b;               // source  b = N rho_p  (T-dof)
   Vector adj_rhs;                 // adjoint RHS (T-dof)
   Vector lambda;

   // adjoint transport solve  K^T lambda = rhs (built once, K is design-independent)
   std::unique_ptr<HypreParMatrix> Kt;
   std::unique_ptr<BlockILU>       adj_prec;
   std::unique_ptr<GMRESSolver>    adj_gmres;

   void Assemble();
   void AssembleSource(const ParGridFunction &rf) const;  // b = N rf, adv->SetSource(b)
   void SetupAdjointSolver();

public:
   PseudoTransientSolver(ParGridFunction &rho_a_, ParGridFunction &rho_filter_,
                         VectorCoefficient &v_cf_)
   : v_cf(v_cf_), rho_a_gf(rho_a_), rho_filter(&rho_filter_),
     K(nullptr), Kmat(nullptr), adv(nullptr)
   {
      dgfes = rho_a_gf.ParFESpace();
      comm  = dgfes->GetComm();

      K = new ParBilinearForm(dgfes);

      Assemble();                                    // -> Kmat
      adv = new DG_AdvectionSolver(*Kmat, *dgfes, v_cf);

      SetupAdjointSolver();                          // K^T solver, reused every solve
   }

   ~PseudoTransientSolver()
   {
      delete adv;
      delete Kmat;
      delete K;
   }

   ParGridFunction &GetRhoA() { return rho_a_gf; }  // forward field (full DG)
   Vector &GetAdjoint() { return lambda; }
   void GetFilterGrad(Vector &dGdrf) const;    // dG/drho_filter (H1) = -N^T lambda 

   // set the adjoint RHS from an assembled dG/drho_a (T-dof)
   void SetAdjointRHS(const Vector &rhs) { adj_rhs = rhs; }

   void Mult(const Vector &x, Vector &y) const override;

   // solve for rho_a -> M (drho_a/dt) = K rho_a + N rho_p
   void FSolve();

   // solve for lambda -> K^T lambda = (rhs)
   void ASolve();
};

// K (upwind discretization of -v.grad); K carries the minus sign so that
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

// forward map: filtered density (H1 T-dof) -> steady advected density (DG T-dof)
void PseudoTransientSolver::Mult(const Vector &r_f, Vector &r_a) const
{
   ParGridFunction rf_gf(rho_filter->ParFESpace());
   rf_gf.SetFromTrueDofs(r_f);
   AssembleSource(rf_gf);

   const int n = adv->Height();
   r_a.SetSize(n);
   r_a = 0.0;
   Vector rate(n);

   RK3SSPSolver ode;
   ode.Init(*adv);

   real_t t = 0.0, dt = 1e-3;     // explicit step  (set < CFL limit)
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
}

// forward solve on the live rho_filter -> rho_a_gf
void PseudoTransientSolver::FSolve()
{
   Vector r_f;
   rho_filter->GetTrueDofs(r_f);

   Vector r_a;
   Mult(r_f, r_a);

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

// dG/drho_filter (H1) = -N^T lambda,  (N^T lambda)_j = INT phi_j^H1 lambda_h
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