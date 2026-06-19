#include "mfem.hpp"
#include <cmath>
#include "QuantityOfInterest.hpp"

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

// PDE density filter: -r^2 Lap(rho~) + rho~ = g
class FilterSolver : Operator
{
private:
   ParMesh *pmesh = nullptr;
   int order = 1, dim = 0;
   Coefficient *diffcf = nullptr;          // r^2
   Coefficient *masscf = nullptr;          // 1
   Coefficient *rhscf  = nullptr;          // forward rhs
   Coefficient *adjcf  = nullptr;          // adjoint rhs

   FiniteElementCollection *fec = nullptr;
   ParFiniteElementSpace *fes = nullptr;
   Array<int> ess_bdr, ess_tdof_list;
   ParGridFunction *u = nullptr;

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

      cg->Mult(*B, X);
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

      height = width = fes->GetTrueVSize();
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