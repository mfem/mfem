#pragma once
#include "mfem.hpp"

/// @brief Inverse sigmoid function
double inv_sigmoid(const double x)
{
   const double tol = 1e-12;
   const double tmp = std::min(std::max(tol,x),1.0-tol);
   return std::log(tmp/(1.0-tmp));
}

/// @brief Sigmoid function
double sigmoid(const double x)
{
   return x>=0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x));
}

/// @brief Derivative of sigmoid function
double der_sigmoid(const double x)
{
   const double tmp = sigmoid(x);
   return tmp*(1.0 - tmp);
}

/// @brief SIMP function, ρ₀ + (ρ̄ - ρ₀)*x^k
double simp(const double x, const double rho_0=1e-06, const double k=3.0,
            const double rho_max=1.0)
{
   return rho_0 + std::pow(x, k) * (rho_max - rho_0);
}

/// @brief Derivative of SIMP function, k*(ρ̄ - ρ₀)*x^(k-1)
double der_simp(const double x, const double rho_0=1e-06,
                const double k=3.0, const double rho_max=1.0)
{
   return k * std::pow(x, k - 1) * (rho_max - rho_0);
}

namespace mfem
{
class DiscL2ProjectionIntegrator : public MixedScalarIntegrator
{
private:
   std::unique_ptr<BilinearFormIntegrator> invmass, mass;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix invM, M;
#endif
public:
   DiscL2ProjectionIntegrator();
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};
// Elasticity solver
class EllipticSolver
{
protected:
   BilinearForm &a; // LHS
   LinearForm &b; // RHS
   std::unique_ptr<GridFunction> u;
   Array2D<int> ess_bdr; // Component-wise essential boundary marker
   bool parallel; // Flag for ParFiniteElementSpace
   bool symmetric;
public:
   /// @brief Linear solver for elliptic problem with given Essential BC
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Essential boundary marker for boundary attributes
   EllipticSolver(BilinearForm &a, LinearForm &b, Array<int> &ess_bdr_list);
   /// @brief Linear solver for elliptic problem with given component-wise essential BC
   /// ess_bdr[0,:] - All components, ess_bdr[i,:] - ith-direction
   /// @param a Bilinear Form
   /// @param b Linear Form
   /// @param ess_bdr_list Component-wise essential boundary marker for boundary attributes
   EllipticSolver(BilinearForm &a, LinearForm &b, Array2D<int> &ess_bdr);

   /// @brief Solve linear system and return FEM solution in x.
   /// @param x FEM solution
   /// @param A_assembled If true, skip assembly of LHS (bilinearform)
   /// @param b_Assembled If true, skip assembly of RHS (linearform)
   /// @return convergence flag
   bool Solve(GridFunction &x, bool A_assembled=false, bool b_Assembled=false);
   bool SolveTranspose(GridFunction &x, LinearForm *f, bool A_assembled=false,
                       bool b_Assembled=false);

   bool isParallel() { return parallel; }
   bool isSymmetric() { return symmetric; }
protected:
   /// @brief Get true dofs related to the boundaries in @ess_bdr
   /// @return True dof list
   Array<int> GetEssentialTrueDofs();
private:
};

class DensityFilter
{
public:
protected:
private:

public:
   DensityFilter() = default;
   virtual void Apply(const GridFunction &rho, GridFunction &frho) const = 0;
   virtual void Apply(Coefficient &rho, GridFunction &frho) const = 0;
protected:
private:
};

class HelmholtzFilter : public DensityFilter
{
public:
protected:
   FiniteElementSpace &fes;
   std::unique_ptr<BilinearForm> filter;
   ConstantCoefficient eps2;
private:

public:
   HelmholtzFilter(FiniteElementSpace &fes, const double eps);
   void Apply(const GridFunction &rho, GridFunction &frho) const override
   {
      GridFunctionCoefficient rho_cf(&rho);
      Apply(rho_cf, frho);
   }
   void Apply(Coefficient &rho, GridFunction &frho) const override;
protected:
private:
};

class DesignDensity
{
   // variables
public:
protected:
   std::unique_ptr<GridFunction> x_gf;
   std::unique_ptr<GridFunction> frho;
   std::unique_ptr<Coefficient> rho_cf;
   DensityFilter &filter;
   double target_volume_fraction;
   double target_volume;
   double current_volume;
   double vol_tol;
private:

   // functions
public:
   DesignDensity(FiniteElementSpace &fes, DensityFilter &filter,
                 FiniteElementSpace &filter_fes, double vol_frac,
                 double volume_tolerance=1e-09);
   FiniteElementSpace *FESpace() {return x_gf->FESpace(); }
   FiniteElementSpace *FESpace_filter() {return frho->FESpace(); }
   double GetVolume() { return current_volume; }
   GridFunction &GetGridFunction() { return *x_gf; }
   Coefficient &GetDensityCoefficient() { return *rho_cf; }
   GridFunction &GetFilteredDensity() { return *frho; }
   void UpdateFilteredDensity()
   {
      filter.Apply(*rho_cf, *frho);
   }
   DensityFilter &GetFilter() { return filter; }

   virtual void Project() = 0;
   virtual double StationarityError(GridFunction &grad) = 0;
protected:
   virtual void ComputeVolume() = 0;
private:
};

class DensityProjector
{
public:
   virtual Coefficient &GetPhysicalDensity(GridFunction &frho) = 0;
   virtual Coefficient &GetDerivative(GridFunction &frho) = 0;
};
class SIMPProjector : public DensityProjector
{
private:
   const double k, rho0;
   std::unique_ptr<MappedGridFunctionCoefficient> phys_density, dphys_dfrho;
public:
   SIMPProjector(const double k, const double rho0):k(k), rho0(rho0)
   {
      phys_density.reset(new MappedGridFunctionCoefficient(
      nullptr, [rho0, k](double x) {return simp(x, rho0, k);}));
      dphys_dfrho.reset(new MappedGridFunctionCoefficient(
      nullptr, [rho0, k](double x) {return der_simp(x, rho0, k);}));
   }
   Coefficient &GetPhysicalDensity(GridFunction &frho) override
   {
      phys_density->SetGridFunction(&frho);
      return *phys_density;
   }
   Coefficient &GetDerivative(GridFunction &frho) override
   {
      dphys_dfrho->SetGridFunction(&frho);
      return *dphys_dfrho;
   }
};

class ThresholdProjector : public DensityProjector
{
private:
   const double beta, eta;
   std::unique_ptr<MappedGridFunctionCoefficient> phys_density, dphys_dfrho;
   ThresholdProjector(const double beta, const double eta):beta(beta), eta(eta)
   {
      const double c1 = std::tanh(beta*eta);
      const double c2 = std::tanh(beta*(1-eta));
      const double inv_denominator = 1.0 / (c1 + c2);
      phys_density.reset(new MappedGridFunctionCoefficient(
                            nullptr, [c1, c2, beta, eta](double x)
      {
         return (c1 + std::tanh(beta*(x - eta))) / (c1 + c2);
      }));
      dphys_dfrho.reset(new MappedGridFunctionCoefficient(
                           nullptr, [c1, c2, beta, eta](double x)
      {
         return beta*std::pow(1.0/std::cosh(beta*(x - eta)), 2.0) / (c1 + c2);
      }));
   }
   Coefficient &GetPhysicalDensity(GridFunction &frho) override
   {
      phys_density->SetGridFunction(&frho);
      return *phys_density;
   }
   Coefficient &GetDerivative(GridFunction &frho) override
   {
      dphys_dfrho->SetGridFunction(&frho);
      return *dphys_dfrho;
   }
};

class LatentDesignDensity : public DesignDensity
{
   // variables
public:
protected:
private:
   GridFunction zero_gf;

   // functions
public:
   LatentDesignDensity(FiniteElementSpace &fes, DensityFilter &filter,
                       FiniteElementSpace &fes_filter,
                       double vol_frac):
      DesignDensity(fes, filter, fes_filter, vol_frac), zero_gf(&fes)
   {
      *x_gf = inv_sigmoid(vol_frac);
      rho_cf.reset(new MappedGridFunctionCoefficient(x_gf.get(), sigmoid));
      zero_gf = 0.0;
   }
   void Project() override;
   double StationarityError(GridFunction &grad) override
   {
      return StationarityError(grad, false);
   };
   double StationarityError(GridFunction &grad, bool useL2norm);
   double StationarityErrorL2(GridFunction &grad);
   double ComputeBregmanDivergence(GridFunction *p, GridFunction *q,
                                   double log_tol=1e-09);
protected:
   void ComputeVolume() override
   {
      current_volume = zero_gf.ComputeL1Error(*rho_cf.get());
   }
private:
};

class PrimalDesignDensity : public DesignDensity
{
   // variables
public:
protected:
private:
   ConstantCoefficient zero_cf;

   // functions
public:
   PrimalDesignDensity(FiniteElementSpace &fes, DensityFilter& filter,
                       FiniteElementSpace &fes_filter,
                       double vol_frac):
      DesignDensity(fes, filter, fes_filter, vol_frac), zero_cf(0.0)
   {
      rho_cf.reset(new GridFunctionCoefficient(x_gf.get()));
   }
   void Project() override;
   double StationarityError(GridFunction &grad) override;
protected:
   void ComputeVolume() override
   {
      current_volume = x_gf->ComputeL1Error(zero_cf);
   }
private:
};

class ParametrizedLinearEquation
{
public:
protected:
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm> b;
   GridFunction &frho;
   DensityProjector &projector;
   bool AisStationary;
   bool BisStationary;
   Array2D<int> &ess_bdr;
private:


public:
   ParametrizedLinearEquation(FiniteElementSpace &fes,
                              GridFunction &filtered_density,
                              DensityProjector &projector, Array2D<int> &ess_bdr);
   void SetBilinearFormStationary(bool isStationary=true);
   void SetLinearFormStationary(bool isStationary=true);
   void Solve(GridFunction &x);

   /// @brief Solve a(λ, v)=F(v) assuming a is symmetric.
   /// @param
   /// @param u
   /// @return
   void DualSolve(GridFunction &x, LinearForm &new_b);
   virtual std::unique_ptr<Coefficient> GetdEdfrho(GridFunction &u,
                                                   GridFunction &lambda, GridFunction &frho) = 0;
   FiniteElementSpace *FESpace() { return a->FESpace(); }
   LinearForm &GetLinearForm() {return *b;}
protected:
private:
};

class TopOptProblem
{
public:
protected:
   LinearForm &obj;
   ParametrizedLinearEquation &state_equation;
   DesignDensity &density;
   std::shared_ptr<GridFunction> gradF;
   std::shared_ptr<GridFunction> gradF_filter;
   std::shared_ptr<GridFunction> state, dual_solution;
   std::unique_ptr<MixedBilinearForm> filter_to_density;
   std::unique_ptr<LinearForm> gradF_filter_form;
   std::unique_ptr<Coefficient> dEdfrho;
   const bool skip_dual;
private:

public:

   /// @brief Create Topology optimization problem
   /// @param objective Objective linear functional, F(u)
   /// @param state_equation State equation, a(u,v) = b(v)
   /// @param density Density object, ρ
   /// @param skip_dual If true, kip dual solve, a(v,λ)=F(v) and assume λ=u
   /// @note It assume that the state equation is symmetric and objective
   TopOptProblem(LinearForm &objective,
                 ParametrizedLinearEquation &state_equation,
                 DesignDensity &density, bool skip_dual);

   double Eval();
   void UpdateGradient();
   GridFunction &GetGradient() { return *gradF.get(); }
   GridFunction &GetGridFunction() { return density.GetGridFunction(); }
   Coefficient &GetDensity() { return density.GetDensityCoefficient(); }
   GridFunction &GetState() {return *state;}
protected:
private:
};


/// @brief Strain energy density coefficient
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &lambda;
   Coefficient &mu;
   GridFunction &u1; // displacement
   GridFunction &u2; // displacement
   Coefficient &dphys_dfrho;
   DenseMatrix grad1, grad2; // auxiliary matrix, used in Eval

public:
   StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                  GridFunction &u, DensityProjector &projector, GridFunction &frho)
      : lambda(lambda), mu(mu),  u1(u),  u2(u),
        dphys_dfrho(projector.GetDerivative(frho))
   { }
   StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                  GridFunction &u1, GridFunction &u2, DensityProjector &projector,
                                  GridFunction &frho)
      : lambda(lambda), mu(mu),  u1(u1),  u2(u2),
        dphys_dfrho(projector.GetDerivative(frho))
   { }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class ParametrizedElasticityEquation : public ParametrizedLinearEquation
{
public:
protected:
   Coefficient &lambda;
   Coefficient &mu;
   GridFunction &filtered_density;
   ProductCoefficient phys_lambda;
   ProductCoefficient phys_mu;
   VectorCoefficient &f;
private:

public:
   ParametrizedElasticityEquation(FiniteElementSpace &fes,
                                  GridFunction &filtered_density,
                                  DensityProjector &projector,
                                  Coefficient &lambda, Coefficient &mu,
                                  VectorCoefficient &f, Array2D<int> &ess_bdr);
   virtual std::unique_ptr<Coefficient> GetdEdfrho(GridFunction &u,
                                                   GridFunction &dual_solution, GridFunction &frho) override
   { return std::unique_ptr<Coefficient>(new StrainEnergyDensityCoefficient(lambda, mu, u, dual_solution, projector, frho)); }
protected:
private:
};


/**
 * @brief Perform a gradient step with given initial step size. Shrink step size until armijo condition is satisfied.
 *        Assume problem is evaluated at the current point and gradient is updated.
 *
 * @param problem Underlying topology optimization problem
 * @param val Current value
 * @param c1 Control parameter << 1
 * @param step_size initial step size
 * @param shrink_factor step size shrink factor, for each iteration, step_size *= shrink_factor
 */
double Step_Armijo(TopOptProblem &problem, const double val, const double c1,
                   double step_size, const double shrink_factor=0.5);

/// @brief Volumetric force for linear elasticity
class VolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
public:
   VolumeForceCoefficient(double r_,Vector &  center_, Vector & force_);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   void Set(double r_,Vector & center_, Vector & force_);
   void UpdateSize();
};

/// @brief Volumetric force for linear elasticity
class LineVolumeForceCoefficient : public VectorCoefficient
{
private:
   double r2;
   Vector &center;
   Vector &force;
   int direction_dim;
public:
   LineVolumeForceCoefficient(double r_, Vector &center_, Vector & force_,
                              int direction_dim);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   void Set(double r_,Vector & center_, Vector & force_);
   void UpdateSize();
};

}