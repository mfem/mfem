#include "mfem.hpp"
#include <cmath>
#include "QuantityOfInterest.hpp"

using namespace std;
using namespace mfem;


// SIMP coefficient: r(rho~) = min_val + rho~^exponent (max_val - min_val).
class SIMPCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter;
   real_t min_val, max_val, exponent;

public:
   SIMPCoefficient(GridFunction *rho_filter_, real_t min_val_ = 1e-6,
                   real_t max_val_ = 1.0, real_t exponent_ = 3.0)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      // r(rho~) = min_val + rho~^exponent (max_val - min_val)
      return min_val + std::pow(val, exponent) * (max_val - min_val);
   }
};

// Strain-energy-density (compliance sensitivity) coefficient.
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient *lambda = nullptr;          // unit-modulus lambda
   Coefficient *mu = nullptr;              // unit-modulus mu
   GridFunction *u = nullptr;              // displacement state
   GridFunction *rho_filter = nullptr;     // filtered density rho~
   DenseMatrix grad;                       // scratch, used in Eval
   real_t exponent, rho_min;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction *u_, GridFunction *rho_filter_,
                                  real_t rho_min_ = 1e-6, real_t exponent_ = 3.0)
      : lambda(lambda_), mu(mu_), u(u_), rho_filter(rho_filter_),
        exponent(exponent_), rho_min(rho_min_)
   {
      MFEM_ASSERT(rho_min_ >= 0.0 && rho_min_ < 1.0, "rho_min must be in [0,1)");
      MFEM_ASSERT(u && rho_filter, "displacement / density field not set");
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
      
      real_t val = rho_filter->GetValue(T, ip);
      // dc/drho~ = -p rho~^{p-1} (1 - rho_min) psi0(u)
      return -exponent * std::pow(val, exponent - 1.0) * (1.0 - rho_min) * density;
   }
};

// PDE density filter: -r^2 Lap(rho~) + rho~ = g, Neumann BCs.
class FilterSolver : Operator
{
private:
   ParMesh *pmesh = nullptr;
   int order = 1, dim = 0;
   Coefficient *diffcf = nullptr;          // r^2
   Coefficient *masscf = nullptr;          // 1
   Coefficient *rhscf  = nullptr;          // forward source (rho)
   Coefficient *adjcf  = nullptr;          // adjoint source (strain energy)

   FiniteElementCollection *fec = nullptr;
   ParFiniteElementSpace *fes = nullptr;
   Array<int> ess_bdr, ess_tdof_list;
   ParGridFunction *u = nullptr;           // last solution (rho~ or w~)

   ParBilinearForm *a = nullptr;
   OperatorPtr A;
   HypreBoomerAMG *amg = nullptr;
   CGSolver *cg = nullptr;

   // Solve  (r^2 grad u, grad v) + (u, v) = (src, v)  for all v in H1
   void Solve(Coefficient *rhs_coeff)
   {
      ParLinearForm b(fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeff));
      b.Assemble();

      std::unique_ptr<HypreParVector> B(b.ParallelAssemble());
      Vector X(fes->GetTrueVSize());  
      X = 0.0;

      cg->Mult(*B, X);                       // Neumann: no BC elimination
      u->SetFromTrueDofs(X);
   }

public:
   FilterSolver() { }
   ~FilterSolver()
   { delete u; delete fes; delete fec; delete a; delete amg; delete cg; }

   void SetMesh(ParMesh *m) { pmesh = m; }
   void SetOrder(int o) { order = o; }
   void SetDiffusionCoefficient(Coefficient *c) { diffcf = c; }
   void SetMassCoefficient(Coefficient *c) { masscf = c; }
   void SetRHSCoefficient(Coefficient *c) { rhscf = c; }
   void SetAdjointRHSCoefficient(Coefficient *c) { adjcf = c; }
   void SetEssentialBoundary(const Array<int> &bdr) { ess_bdr = bdr; }

   void SetupFEM()
   {
      dim = pmesh->Dimension();
      fec = new H1_FECollection(order, dim);
      fes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(fes);  
      *u = 0.0;

      if (!ess_bdr.Size() && pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());  
         ess_bdr = 0; 
      }
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      a = new ParBilinearForm(fes);
      a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
      a->Assemble();
      a->FormSystemMatrix(ess_tdof_list, A);

      amg = new HypreBoomerAMG(*A.As<HypreParMatrix>());
      amg->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
      cg->SetRelTol(1e-10);
      cg->SetMaxIter(2000);
      cg->SetPrintLevel(0);
      cg->SetPreconditioner(*amg);
      cg->SetOperator(*A);

      height = width = fes->GetTrueVSize();  // make this a well-formed Operator
   }

   void FSolve() { Solve(rhscf); }          // forward filter -> rho~
   void ASolve() { Solve(adjcf); }          // adjoint filter -> w~

   // Forward action of the filter operator:  y = A x,  A = r^2 K + M.
   // (FSolve/ASolve apply A^{-1}; this applies A itself.)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      y.SetSize(fes->GetTrueVSize());
      A.Ptr()->Mult(x, y);
   }

   ParGridFunction *GetFEMSolution() { return u; }
};

// SIMP-scaled linear elasticity: -div(r(rho~) C eps(u)) = f.
class LinearElasticitySolver : Operator
{
private:
   ParMesh *pmesh = nullptr;
   int order = 1, dim = 0;
   Coefficient *lambda_cf = nullptr;               // SIMP-scaled lambda
   Coefficient *mu_cf = nullptr;                   // SIMP-scaled mu
   VectorCoefficient *rhs_bd_cf = nullptr;         // surface traction t
   VectorFunctionCoefficient *rhs_cf = nullptr;    // body force f

   FiniteElementCollection *fec = nullptr;
   ParFiniteElementSpace *fes = nullptr;   // vector H1
   Array<int> ess_bdr, load_bdr, ess_tdof_list;
   ParGridFunction *u = nullptr;           // displacement
   ParLinearForm *b = nullptr;             // load (assembled once)
   std::unique_ptr<HypreParVector> load_true;

   real_t compliance = 0.0;

   // Solve a(u,v) = INT r(rho~)[ lambda (div u)(div v) + 2 mu eps(u):eps(v) ] = (t, v)
   void Solve()
   {
      // 
      ParBilinearForm a(fes);
      a.AddDomainIntegrator(new ElasticityIntegrator(*lambda_cf, *mu_cf));
      a.Assemble();

      OperatorPtr A;  
      Vector B, X;
      *u = 0.0;                                  // homogeneous clamp
      a.FormLinearSystem(ess_tdof_list, *u, *b, A, X, B);

      HypreBoomerAMG amg(*A.As<HypreParMatrix>());
      amg.SetPrintLevel(0);
      CGSolver cg(pmesh->GetComm());
      cg.SetRelTol(1e-10);  
      cg.SetMaxIter(10000);  
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(amg);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a.RecoverFEMSolution(X, *b, *u);

      Compliance comp(pmesh->GetComm(), *load_true, X);
      compliance = comp.Eval();  // global f . u
   }

public:
   LinearElasticitySolver() { }
   ~LinearElasticitySolver() { delete u; delete fes; delete fec; delete b; }

   void SetMesh(ParMesh *m) { pmesh = m; }
   void SetOrder(int o) { order = o; }
   void SetLameCoefficients(Coefficient *l, Coefficient *m) { lambda_cf = l; mu_cf = m; }
   void SetRHSbdrCoefficient(VectorCoefficient *c) { rhs_bd_cf = c; }               // traction t
   void SetRHSdomainCoefficient(VectorFunctionCoefficient *vc) { rhs_cf = vc; }     // bodyforce f
   void SetEssentialBoundary(const Array<int> &bdr) { ess_bdr = bdr; }
   void SetLoadBoundary(const Array<int> &bdr) { load_bdr = bdr; }

   void SetupFEM()
   {
      dim = pmesh->Dimension();
      fec = new H1_FECollection(order, dim);
      fes = new ParFiniteElementSpace(pmesh, fec, dim);
      u   = new ParGridFunction(fes);  *u = 0.0;

      if (!ess_bdr.Size() && pmesh->bdr_attributes.Size())
      { ess_bdr.SetSize(pmesh->bdr_attributes.Max());  ess_bdr = 0; }
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      b = new ParLinearForm(fes);
      if (rhs_cf)
         b->AddDomainIntegrator(new VectorDomainLFIntegrator(*rhs_cf));
      if (rhs_bd_cf && load_bdr)
         b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*rhs_bd_cf), load_bdr);
      b->Assemble();
      load_true.reset(b->ParallelAssemble());

      height = width = fes->GetTrueVSize();  // make this a well-formed Operator
   }

   void FSolve() { Solve(); }               // primal state  K u = f
   void ASolve() { Solve(); }               // adjoint state (self-adjoint: == FSolve)

   // Forward action of the SIMP-scaled stiffness:  y = K(rho~) x.
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      ParBilinearForm a(fes);
      a.AddDomainIntegrator(new ElasticityIntegrator(*lambda_cf, *mu_cf));
      a.Assemble();
      a.Finalize();
      std::unique_ptr<HypreParMatrix> K(a.ParallelAssemble());
      y.SetSize(fes->GetTrueVSize());
      K->Mult(x, y);
   }

   ParGridFunction *GetFEMSolution() { return u; }
   real_t GetCompliance() const { return compliance; }
};
