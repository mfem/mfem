#pragma once

#include "mfem.hpp"
#include <cmath>
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

// Heat diffusion solver: -div(k(rho~) grad T) = q  with homogeneous essential BC.
// The conductivity k(rho~) changes every design iteration, so the system is
// re-assembled from scratch in each Solve() (cf. toopt::PDEFilter).
class DiffusionSolver
{
protected:
   ParFiniteElementSpace *fes;
   FiniteElementCollection *fec;
   Coefficient *kappa;        // SIMP conductivity k(rho~)   (not owned)
   Coefficient *q = nullptr;  // heat source                 (not owned)
   Array<int> ess_bdr;        // essential boundary marker
   real_t ess_value = 0.0;
   ParGridFunction T;

public:
   DiffusionSolver(ParMesh *pmesh, int order, Coefficient &kappa_)
      : kappa(&kappa_)
   {
      fec = new H1_FECollection(order, pmesh->Dimension());
      fes = new ParFiniteElementSpace(pmesh, fec);
      T.SetSpace(fes);
      T = 0.0;
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   ~DiffusionSolver()
   {
      delete fes;
      delete fec;
   }

   void SetEssentialBC(int attr, real_t value)
   {
      ess_bdr[attr-1] = 1;
      ess_value = value;
   }

   void SetHeatSource(Coefficient &q_) { q = &q_; }

   // Assemble and solve  -div(k(rho~) grad T) = q  for the current design.
   void Solve()
   {
      Array<int> ess_tdof_list;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      ParBilinearForm a(fes);
      a.AddDomainIntegrator(new DiffusionIntegrator(*kappa));
      a.Assemble();

      ParLinearForm b(fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(*q));
      b.Assemble();

      T = ess_value;            // impose the Dirichlet value

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, T, b, A, X, B);

      HypreParMatrix &Ah = *A.As<HypreParMatrix>();
      HypreBoomerAMG amg(Ah);
      amg.SetPrintLevel(0);

      HyprePCG pcg(Ah);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(2000);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);

      a.RecoverFEMSolution(X, b, T);
   }

   ParGridFunction& GetTemperature() { return T; }
   ParFiniteElementSpace* GetFESpace() { return fes; }
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