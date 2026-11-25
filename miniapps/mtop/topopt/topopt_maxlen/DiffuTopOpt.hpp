#pragma once

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include "../QuantityOfInterest.hpp"

using namespace std;
using namespace mfem;


// SIMP coefficient for thermal conductivity: k(rho~) = k_min + rho~^exponent (k_max - k_min).
class SIMPCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter;
   real_t k_min, k_max, exponent;

public:
   SIMPCoefficient(GridFunction *rho_filter_, real_t k_min_ = 1e-6,
                   real_t k_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_filter(rho_filter_), k_min(k_min_), k_max(k_max_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = std::max(real_t(0), std::min(real_t(1), rho_filter->GetValue(T, ip)));
      return k_min + std::pow(val, exponent) * (k_max - k_min);
   }
};

class SIMPGradCoefficient : public SIMPCoefficient
{
public:
   using SIMPCoefficient::SIMPCoefficient;   // inherits members + constructor

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = std::max(real_t(0), std::min(real_t(1), rho_filter->GetValue(T, ip)));
      return exponent * std::pow(val, exponent - 1.0) * (k_max - k_min);
   }
};

// |grad T|^2 coefficient.  Multiply by k(rho~) in the caller to get dissipation.
class GradTNorm2Coefficient : public Coefficient
{
protected:
   GridFunction *T = nullptr;
   Vector gradT;

public:
   GradTNorm2Coefficient(GridFunction *T_) : T(T_)
   {
      MFEM_ASSERT(T, "temperature not set");
   }

   real_t Eval(ElementTransformation &Trans, const IntegrationPoint &ip) override
   {
      T->GetGradient(Trans, gradT);
      return gradT * gradT;
   }
};

// grad T . grad lambda coefficient.  Multiply by k'(rho~) in the caller to get
// the adjoint sensitivity integrand  k'(rho~) grad T . grad lambda.
class GradTGradLambdaCoefficient : public Coefficient
{
protected:
   GridFunction *T, *lambda;
   Vector gradT, gradL;

public:
   GradTGradLambdaCoefficient(GridFunction *T_, GridFunction *lambda_)
      : T(T_), lambda(lambda_) { }

   real_t Eval(ElementTransformation &Trans, const IntegrationPoint &ip) override
   {
      T->GetGradient(Trans, gradT);
      lambda->GetGradient(Trans, gradL);
      return gradT * gradL;
   }
};

// Heat diffusion solver: -div(k(rho~) grad T) = q,  T = ess_value on the
// essential boundary.  K(rho~) is symmetric, so the adjoint uses the same
// matrix, AMG and PCG with homogeneous essential BC.
//
// Usage per design iteration:   Assemble();  FSolve();  [ASolve(rhs);]
// Solve() = Assemble() + FSolve()  (kept for existing drivers).
class DiffusionSolver
{
protected:
   ParFiniteElementSpace *fes;
   FiniteElementCollection *fec = nullptr;
   bool own_fes = true;
   Coefficient *kappa;        // SIMP conductivity k(rho~)   (not owned)
   Coefficient *q = nullptr;  // heat source                 (not owned)
   Array<int> ess_bdr;        // essential boundary marker
   Array<int> ess_tdof_list;
   real_t ess_value = 0.0;
   ParGridFunction T;         // forward solution
   ParGridFunction lambda;    // adjoint solution

   // Assembled system for the current design (rebuilt by Assemble()).
   std::unique_ptr<ParBilinearForm> a;
   std::unique_ptr<ParLinearForm> b;
   OperatorPtr A;
   Vector B, X;
   std::unique_ptr<HypreBoomerAMG> amg;
   std::unique_ptr<HyprePCG> pcg;

   void ClearSystem()
   {
      pcg.reset();  amg.reset();   // reference A, release first
      A.Clear();
      b.reset();  a.reset();       // a owns the HypreParMatrix behind A
   }

public:
   DiffusionSolver(ParMesh *pmesh, int order, Coefficient &kappa_)
      : kappa(&kappa_)
   {
      fec = new H1_FECollection(order, pmesh->Dimension());
      fes = new ParFiniteElementSpace(pmesh, fec);
      T.SetSpace(fes);       T = 0.0;
      lambda.SetSpace(fes);  lambda = 0.0;
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   // Use an externally owned (scalar H1) state space.
   DiffusionSolver(ParFiniteElementSpace &state_fes, Coefficient &kappa_)
      : fes(&state_fes), own_fes(false), kappa(&kappa_)
   {
      T.SetSpace(fes);       T = 0.0;
      lambda.SetSpace(fes);  lambda = 0.0;
      ess_bdr.SetSize(fes->GetParMesh()->bdr_attributes.Max());
      ess_bdr = 0;
   }

   ~DiffusionSolver()
   {
      ClearSystem();
      if (own_fes) { delete fes; delete fec; }
   }

   void SetEssentialBC(int attr, real_t value)
   {
      ess_bdr[attr-1] = 1;
      ess_value = value;
   }

   void SetHeatSource(Coefficient &q_) { q = &q_; }

   // Assemble K(rho~) and q for the current design and set up AMG/PCG.
   void Assemble()
   {
      MFEM_VERIFY(q, "heat source not set");
      ClearSystem();
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      a.reset(new ParBilinearForm(fes));
      a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
      a->Assemble();

      b.reset(new ParLinearForm(fes));
      b->AddDomainIntegrator(new DomainLFIntegrator(*q));
      b->Assemble();

      T = ess_value;
      a->FormLinearSystem(ess_tdof_list, T, *b, A, X, B);

      HypreParMatrix &Ah = *A.As<HypreParMatrix>();
      amg.reset(new HypreBoomerAMG(Ah));
      amg->SetPrintLevel(0);

      pcg.reset(new HyprePCG(Ah));
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(2000);
      pcg->SetPrintLevel(0);
      pcg->SetPreconditioner(*amg);
   }

   // Forward solve:  K T = q,  T = ess_value on the essential boundary.
   void FSolve()
   {
      MFEM_VERIFY(pcg, "call Assemble() before FSolve()");
      pcg->Mult(B, X);
      a->RecoverFEMSolution(X, *b, T);
   }

   // Adjoint solve:  K lambda = rhs,  lambda = 0 on the essential boundary.
   // rhs is a true-dof vector on GetFESpace(); its essential entries are
   // ignored (set to zero).
   void ASolve(const Vector &rhs)
   {
      MFEM_VERIFY(pcg, "call Assemble() before ASolve()");
      MFEM_VERIFY(rhs.Size() == fes->GetTrueVSize(), "adjoint rhs size mismatch");
      Vector R(rhs), L(rhs.Size());
      R.SetSubVector(ess_tdof_list, 0.0);
      L = 0.0;
      pcg->Mult(R, L);
      lambda.SetFromTrueDofs(L);
   }

   // Assemble and solve  -div(k(rho~) grad T) = q  for the current design.
   void Solve() { Assemble(); FSolve(); }

   ParGridFunction& GetTemperature() { return T; }
   ParGridFunction& GetAdjoint() { return lambda; }
   ParFiniteElementSpace* GetFESpace() { return fes; }
   const Array<int>& GetEssentialTrueDofs() const { return ess_tdof_list; }
};

// Heat dissipation objective: J = ∫ k(rho~) |grad T|^2 dx
class HeatObjective : public QuantityOfInterest
{
protected:
   ParFiniteElementSpace *fes;
   MPI_Comm comm;
   Coefficient &integrand;   // caller owns; should evaluate k(rho~)|grad T|^2

public:
   HeatObjective(MPI_Comm comm_, ParFiniteElementSpace *fes_, Coefficient &integrand_)
      : fes(fes_), comm(comm_), integrand(integrand_) { }

   real_t Eval() override
   {
      ParLinearForm lf(fes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(integrand));
      lf.Assemble();
      std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());

      real_t loc, val;
      loc = v->Sum();
      MPI_Allreduce(&loc, &val, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return val;
   }
};

// Max-temperature slack residual
//     G = 1/(2s) ||T - α||²_{L2(Ω)},    s = T_max² |Ω|,
// with α a slack field in the temperature space, bounded 0 ≤ α ≤ T_max.
// Uses the assembled true-dof mass matrix M, so  G = 1/(2s) rᵀ M r  with
// r = T − α, and the gradients below are its exact derivatives.
class MaxTempResidual : public QuantityOfInterest
{
protected:
    MPI_Comm comm;
    ParFiniteElementSpace *fes;     // temperature space (T and α)
    ParGridFunction *T, *alpha;     // borrowed, live
    std::unique_ptr<HypreParMatrix> M;
    real_t scale;                   // s = T_max² |Ω|
    Vector r, Mr;                   // r = T − α (true dofs),  M r

    void Update()
    {
        Vector a_tv;
        T->GetTrueDofs(r);
        alpha->GetTrueDofs(a_tv);
        r -= a_tv;
        Mr.SetSize(r.Size());
        M->Mult(r, Mr);
    }

public:
    MaxTempResidual(ParGridFunction &T_, ParGridFunction &alpha_, real_t T_max)
    : comm(T_.ParFESpace()->GetComm()), fes(T_.ParFESpace()),
      T(&T_), alpha(&alpha_)
    {
        MFEM_VERIFY(alpha_.ParFESpace() == fes, "T and alpha must share a space");

        ParBilinearForm m(fes);
        m.AddDomainIntegrator(new MassIntegrator);
        m.Assemble();
        m.Finalize();
        M.reset(m.ParallelAssemble());

        ConstantCoefficient one(1.0);
        ParLinearForm vol(fes);
        vol.AddDomainIntegrator(new DomainLFIntegrator(one));
        vol.Assemble();
        std::unique_ptr<HypreParVector> v(vol.ParallelAssemble());
        real_t loc = v->Sum(), domain_volume;
        MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

        scale = T_max * T_max * domain_volume;
    }

    // G = 1/(2s) (T − α)ᵀ M (T − α)
    real_t Eval() override
    {
        Update();
        return 0.5 * InnerProduct(comm, r, Mr) / scale;
    }

    // dG/dT = M (T − α)/s,   dG/dα = −M (T − α)/s   (true-dof vectors)
    void GetGrad(Vector &dGdT, Vector &dGdalpha) override
    {
        Update();
        dGdT = Mr;
        dGdT /= scale;
        dGdalpha = dGdT;
        dGdalpha.Neg();
    }
};

// p-norm of the temperature (volume-averaged):
//
//    P(T) = ( 1/|Ω| ∫ |T|^p dx )^{1/p}            P ≤ max|T|,   P → max|T| as p → ∞
//    dP/dT_i = P^{1−p} / |Ω| ∫ |T|^{p−1} sgn(T) φ_i dx
//
// T ≥ 0 by the maximum principle (q ≥ 0, T = 0 on the sink); |T| only guards
// against small high-order undershoots.  |T|^p is a high-degree polynomial, so
// both the value and the gradient use a rule of order ceil(p) * fe_order, unless
// ir_order > 0 is given.
class TemperaturePNorm : public QuantityOfInterest
{
protected:
   ParGridFunction &T;
   ParFiniteElementSpace *fes;
   MPI_Comm comm;
   real_t p;
   int ir_order;
   real_t domain_volume;

   // |T|^q sgn(T)^s  at a quadrature point
   class PowCoefficient : public Coefficient
   {
      GridFunction &T;
      real_t q;
      bool signed_;
   public:
      PowCoefficient(GridFunction &T_, real_t q_, bool signed__)
         : T(T_), q(q_), signed_(signed__) { }
      real_t Eval(ElementTransformation &Tr, const IntegrationPoint &ip) override
      {
         const real_t t = T.GetValue(Tr, ip);
         const real_t v = std::pow(std::abs(t), q);
         return (signed_ && t < 0.0) ? -v : v;
      }
   };

   const IntegrationRule &GetRule(int e) const
   {
      return IntRules.Get(fes->GetFE(e)->GetGeomType(), ir_order);
   }

   // global  ∫ |T|^p dx
   real_t IntegratePow()
   {
      PowCoefficient Tp(T, p, false);
      real_t loc = 0.0;
      for (int e = 0; e < fes->GetNE(); e++)
      {
         ElementTransformation *Tr = fes->GetElementTransformation(e);
         const IntegrationRule &ir = GetRule(e);
         for (int k = 0; k < ir.GetNPoints(); k++)
         {
            const IntegrationPoint &ip = ir.IntPoint(k);
            Tr->SetIntPoint(&ip);
            loc += ip.weight * Tr->Weight() * Tp.Eval(*Tr, ip);
         }
      }
      real_t glob;
      MPI_Allreduce(&loc, &glob, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return glob;
   }

public:
   TemperaturePNorm(ParGridFunction &T_, real_t p_, int ir_order_ = 0)
      : T(T_), fes(T_.ParFESpace()), comm(T_.ParFESpace()->GetComm()), p(p_)
   {
      MFEM_VERIFY(p >= 1.0, "p-norm exponent must be >= 1");
      ir_order = (ir_order_ > 0) ? ir_order_
                 : (int)std::ceil(p) * fes->GetMaxElementOrder();

      ConstantCoefficient one(1.0);
      ParLinearForm vol(fes);
      vol.AddDomainIntegrator(new DomainLFIntegrator(one));
      vol.Assemble();
      ParGridFunction ones(fes);  ones = 1.0;
      domain_volume = vol(ones);   // ParLinearForm::operator() is global
   }

   // P(T) for the current T
   real_t Eval() override
   {
      return std::pow(IntegratePow() / domain_volume, 1.0 / p);
   }

   // dP/dT as a true-dof vector on T's space
   void GetGrad(Vector &dPdT) override
   {
      const real_t P = Eval();
      PowCoefficient Tp1(T, p - 1.0, true);

      ParLinearForm lf(fes);
      // a single rule for all elements (one element type in this mesh)
      lf.AddDomainIntegrator(new DomainLFIntegrator(Tp1, &GetRule(0)));
      lf.Assemble();
      std::unique_ptr<HypreParVector> v(lf.ParallelAssemble());

      dPdT.SetSize(fes->GetTrueVSize());
      dPdT = *v;
      dPdT *= (P > 0.0) ? std::pow(P, 1.0 - p) / domain_volume : 0.0;
   }

   real_t GetDomainVolume() const { return domain_volume; }
   int GetIntRuleOrder() const { return ir_order; }
};
