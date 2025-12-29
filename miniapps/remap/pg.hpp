#pragma once
#include "mfem.hpp"
#include "ad_native.hpp"

namespace mfem
{
// PGStepSizeRule defines the step size rule for the Proximal Galerkin (PG) method.
// See RuleType for the available rules
struct PGStepSizeRule
{
   enum RuleType
   {
      CONSTANT, // alpha0
      POLY, // alpha0 * (iter+1)^ratio
      EXP, // alpha0 * ratio^iter
      DOUBLE_EXP, // alpha0 * ratio^(ratio2^iter)
      // ... add more rules as needed
      INVALID // used to check for valid rule types
   };
   RuleType rule_type;

   real_t max_alpha;
   real_t alpha0; // initial step size
   real_t ratio; // poly degree (POLY), exponential base (EXP, DOUBLE_EXP)
   real_t ratio2; // nested exponential base (DOUBLE_EXP)

   PGStepSizeRule(int rule_type,
                  real_t alpha0 = 1.0, real_t max_alpha = 1e06,
                  real_t ratio = -1.0, real_t ratio2 = -1.0);

   /// Get the step size for the given iteration
   real_t Get(int iter) const;
};

// Base struct for dual entropy functions
struct ADEntropy : public ADFunction
{
   ADEntropy(int n_input)
      : ADFunction(n_input) { }
};

template <typename T>
std::vector<T*> uniquevec2ptrvec(std::vector<std::unique_ptr<T>> &vec)
{
   std::vector<T*> ptrs(vec.size());
   for (int i=0; i<vec.size(); i++)
   {
      ptrs[i] = vec[i].get();
   }
   return ptrs;
}

// Construct augmented energy for proximal Galerkin
// psi =
// L(u, psi) = f(u) + (1/alpha)(u*(psi-psi_k) - E^*(psi))
// so that
// dL/du = df/du + (1/alpha)(psi-psi_k)
// dL/dpsi = (1/alpha)(u - dE^*(psi))
// When primal is not full vector, set primal_begin
// The parameter should be [org_param, entropy_param, alpha, psi_k]
struct ADPGFunctional : public ADFunction
{
   ADFunction &f;
   std::vector<ADEntropy*> dual_entropy;
   std::vector<int> primal_idx;
   std::vector<int> dual_idx;
   std::vector<int> entropy_size;
   std::vector<GridFunction*> latent_k_gf;
   mutable std::vector<Vector> latent_k;
   mutable Vector jac;
   mutable DenseMatrix hess;
   real_t alpha = 1.0;
   std::unique_ptr<VectorCoefficient> owned_cf;

   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy, int idx=0)
      : ADFunction(f.n_input + dual_entropy.n_input)
      , f(f), dual_entropy{&dual_entropy}
      , primal_idx(1)
      , dual_idx(1)
      , entropy_size(1)
      , latent_k_gf(1)
      , latent_k(1)
   {
      this->primal_idx[0] = idx;
      entropy_size[0] = dual_entropy.n_input;
      MFEM_VERIFY(f.n_input >= this->primal_idx[0] + entropy_size[0],
                  "ADPGFunctional: f.n_input must not exceed "
                  "primal_begin + dual_entropy.n_input:"
                  << f.n_input << " >= " << n_input);
      latent_k[0].SetSize(entropy_size[0]);
      dual_idx[0] = f.n_input;
   }
   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy,
                  GridFunction &latent_k, int idx=0)
      : ADPGFunctional(f, dual_entropy, idx)
   {
      SetPrevLatent(latent_k);
   }
   // Multiple entropies
   ADPGFunctional(ADFunction &f, std::vector<ADEntropy*> dual_entropy_,
                  std::vector<int> &primal_begin)
      : ADFunction(f.n_input)
      , f(f), dual_entropy(std::move(dual_entropy_))
      , primal_idx(primal_begin)
      , dual_idx(dual_entropy.size())
      , entropy_size(dual_entropy.size())
      , latent_k_gf(dual_entropy.size())
      , latent_k(dual_entropy.size())
   {
      int dual_entropy_size = 0;
      int max_primal_index = 0;
      for (int i=0; i<dual_entropy.size(); i++)
      {
         dual_entropy_size += dual_entropy[i]->n_input;
         max_primal_index = std::max(max_primal_index,
                                     primal_begin[i] + dual_entropy[i]->n_input);
      }
      n_input += dual_entropy_size;
      MFEM_VERIFY(f.n_input >= max_primal_index,
                  "ADPGFunctional: f.n_input must be larger than "
                  "primal_begin[i] + dual_entropy.n_input[i] for all i");
      int offset = 0;
      for (int i=0; i<dual_entropy.size(); i++)
      {
         entropy_size[i] = dual_entropy[i]->n_input;
         latent_k[i].SetSize(entropy_size[i]);
         dual_idx[i] = f.n_input + offset;
         offset += entropy_size[i];
      }
   }

   ADPGFunctional(ADFunction &f, std::vector<ADEntropy*> dual_entropy,
                  std::vector<GridFunction*> latent_k_gf, std::vector<int> &primal_begin)
      : ADPGFunctional(f, std::move(dual_entropy), primal_begin)
   {
      MFEM_VERIFY(latent_k_gf.size() == this->dual_entropy.size(),
                  "ADPGFunctional: latent_k must have the same size as dual_entropy: "
                  << latent_k_gf.size() << " != " << dual_entropy.size());
      MFEM_VERIFY(latent_k_gf.size() == primal_begin.size(),
                  "ADPGFunctional: latent_k must have the same size as primal_begin"
                  << latent_k_gf.size() << " != " << primal_begin.size());
      for (int i=0; i<latent_k_gf.size(); i++)
      {
         SetPrevLatent(*latent_k_gf[i], i);
      }
   }
   // Multiple entropies
   ADPGFunctional(ADFunction &f, std::vector<std::unique_ptr<ADEntropy>> &dual_entropy,
                  std::vector<int> &primal_begin)
      : ADPGFunctional(f, uniquevec2ptrvec(dual_entropy), primal_begin)
   {}
   ADPGFunctional(ADFunction &f, std::vector<std::unique_ptr<ADEntropy>> &dual_entropy,
                  std::vector<std::unique_ptr<GridFunction>> &latent_k_gf, std::vector<int> primal_begin)
      : ADPGFunctional(f, uniquevec2ptrvec(dual_entropy),
                       uniquevec2ptrvec(latent_k_gf), primal_begin)
   {}

   ADFunction &GetObjective() const
   { return f; }

   ADEntropy &GetEntropy() const
   {
      MFEM_VERIFY(dual_entropy.size() == 1,
                  "ADPGFunctional: GetEntropy() can only be called when there is a single entropy");
      return *dual_entropy[0];
   }
   const std::vector<ADEntropy*> &GetEntropies() const
   { return dual_entropy; }

   // Set the penalty parameter alpha
   void SetAlpha(real_t alpha) { this->alpha = alpha; }

   real_t GetAlpha() const { return alpha; }

   void SetPrevLatent(GridFunction &psi_k, int i=0)
   {
      MFEM_VERIFY(psi_k.FESpace()->GetVDim() == entropy_size[i],
                  "ADPGFunctional: psi_k must have the same dimension as "
                  "dual_entropy.n_input");
      MFEM_VERIFY(i < latent_k.size(),
                  "ADPGFunctional: i must be less than latent_k.size()");
      latent_k_gf[i] = &psi_k;
   }
   GridFunction &GetPrevLatent(int i=0) const
   {
      MFEM_ASSERT(i < latent_k_gf.size(),
                  "ADPGFunctional: i must be less than latent_k_gf.size()");
      return *latent_k_gf[i];
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      for (int i=0; i<latent_k.size(); i++)
      {
         MFEM_ASSERT(latent_k_gf[i] != nullptr,
                     "ADPGFunctional: latent_k_cf is not set. Use SetPrevLatent() to set it.");
         latent_k_gf[i]->GetVectorValue(Tr, ip, latent_k[i]);
         dual_entropy[i]->ProcessParameters(Tr, ip);
      }
      f.ProcessParameters(Tr, ip);
   }

   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &J) const override
   {
      ProcessParameters(Tr, ip);
      Vector x_primal(x.GetData(), f.n_input);
      Vector x_dual;
      f.Gradient(x_primal, J);
      J.SetSize(n_input);

      for (int i=0; i<dual_entropy.size(); i++)
      {
         x_dual.SetDataAndSize(x.GetData() + dual_idx[i],
                               entropy_size[i]);
         dual_entropy[i]->Gradient(x_dual, jac);
         for (int j=0; j<entropy_size[i]; j++)
         {
            J[primal_idx[i] + j] += (x_dual[j] - latent_k[i][j])/alpha;
            J[dual_idx[i] + j] = (x[primal_idx[i] + j] - jac[j])/alpha;
         }
      }
   }

   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &H) const override
   {
      ProcessParameters(Tr, ip);
      Vector x_primal(x.GetData(), f.n_input);
      Vector x_dual;
      H.SetSize(n_input, n_input);
      f.Hessian(x_primal, hess);
      H.SetSubMatrix(0, 0, hess);
      for (int i=0; i<dual_entropy.size(); i++)
      {
         x_dual.SetDataAndSize(x.GetData() + dual_idx[i],
                               entropy_size[i]);
         dual_entropy[i]->Hessian(x_dual, hess);
         hess *= -1.0 / alpha;
         H.SetSubMatrix(dual_idx[i], dual_idx[i], hess);
         for (int j=0; j<entropy_size[i]; j++)
         {
            H(dual_idx[i] + j, primal_idx[i] + j) = 1.0 / alpha;
            H(primal_idx[i] + j, dual_idx[i] + j) = 1.0 / alpha;
         }
      }
   }

   AD_IMPL(T, V, M, x_latent,
   {
      // variables
      const V x(x_latent.GetData(), f.n_input);
      std::vector<V> latent(entropy_size.size());
      for (int i=0; i<entropy_size.size(); i++)
      {
         latent[i].SetDataAndSize(x_latent.GetData() + f.n_input + primal_idx[i],
                                  entropy_size[i]);
      }

      // evaluate mixed value
      T cross_entropy = T();
      T dual_entropy_sum = T();
      for (int i=0; i<dual_entropy.size(); i++)
      {
         for (int j=0; j<dual_entropy[i]->n_input; j++)
         {
            cross_entropy += x[primal_idx[i] + j]*(latent[i][j] - latent_k[i][j]);
            dual_entropy_sum += (*dual_entropy[i])(latent[i]);
         }
      }
      return f(x) + (1.0 / alpha)*(cross_entropy - dual_entropy_sum);
   });
};

enum LatentType
{
   COEFFICIENT,
   GF,
   QF
};


// Dual entropy for (negative) Shannon entropy (xlogx - x) with half bound
// when bound[1] = 1, [lower, inf[
// when bound[1] = -1, ]-inf, upper]
//
// The resulting dual is (f(pm1*(x - shift)))^*
// = f^*(pm1*x^*) + shift*pm1*x^*
struct ShannonEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t shift;
   int sign;
   ShannonEntropy(Coefficient &bound, int sign=1)
      : ADEntropy(1)
      , bound(bound)
      , sign(sign)
   {
      MFEM_VERIFY(sign == 1 || sign == -1,
                  "ShannonEntropy: sign must be 1 or -1");
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      shift = bound.Eval(Tr, ip);
   }
   AD_IMPL(T, V, M, x, return sign*(exp(x[0]*sign)) + shift*x[0]; );
};

// Dual entropy for (negative) Fermi-Dirac with [lower, upper] bounds
struct FermiDiracEntropy : public ADEntropy
{
   Coefficient *upper_bound;
   Coefficient *lower_bound;
   std::vector<std::unique_ptr<Coefficient>> owned_cf;
   mutable real_t shift;
   mutable real_t scale;

   int sign;
   FermiDiracEntropy(Coefficient &lower_bound, Coefficient &upper_bound)
      : ADEntropy(1)
      , lower_bound(&lower_bound)
      , upper_bound(&upper_bound)
   { }
   FermiDiracEntropy(GridFunction &lower_bound, GridFunction &upper_bound)
      : ADEntropy(1)
   {
      MFEM_VERIFY(lower_bound.FESpace()->GetVDim() == 1 &&
                  upper_bound.FESpace()->GetVDim() == 1,
                  "FermiDiracEntropy: lower_bound and upper_bound must be scalar GridFunctions");
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<GridFunctionCoefficient>(&lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<GridFunctionCoefficient>(&upper_bound);
      this->upper_bound = owned_cf[1].get();
   }
   FermiDiracEntropy(QuadratureFunction &lower_bound,
                     QuadratureFunction &upper_bound)
      : ADEntropy(1)
   {
      MFEM_VERIFY(lower_bound.GetVDim() == 1 &&
                  upper_bound.GetVDim() == 1,
                  "FermiDiracEntropy: lower_bound and upper_bound must be scalar GridFunctions");
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<QuadratureFunctionCoefficient>(lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<QuadratureFunctionCoefficient>(upper_bound);
      this->upper_bound = owned_cf[1].get();
   }
   FermiDiracEntropy(real_t lower_bound, real_t upper_bound)
      : ADEntropy(1)
   {
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<ConstantCoefficient>(lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<ConstantCoefficient>(upper_bound);
      this->upper_bound = owned_cf[1].get();
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      shift = lower_bound->Eval(Tr, ip);
      scale = upper_bound->Eval(Tr, ip) - shift;
   }
   AD_IMPL(T, V, M, x,
   {
      T z = x[0]*scale;

      // Use a numerically stable implementation of log(1+exp(z))
      if (z > 0)
      {
         return z + log(1.0 + exp(-z)) + shift*x[0];
      }
      else
      {
         return log(1.0 + exp(z)) + shift*x[0];
      }
   });
};
// Dual entropy for (negative) Hellinger entropy with bound > 0
struct HellingerEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t scale;

   HellingerEntropy(Coefficient &bound)
      : ADEntropy(1), bound(bound)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      scale = bound.Eval(Tr, ip);
      MFEM_ASSERT(scale > 0, "HellingerEntropy: bound must be positive");
   }
   AD_IMPL(T, V, M, x, return sqrt(1 + (x*x)*(scale*scale)););
};

// Dual entropy for (negative) Simplex entropy with
// x_i >= 0 sum_i x_i = bound
// Also known as cateborical entropy or multinomial Shannon entropy
struct SimplexEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t scale;

   SimplexEntropy(Coefficient &bound)
      : ADEntropy(1), bound(bound)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      scale = bound.Eval(Tr, ip);
      MFEM_ASSERT(scale >= 0, "SimplexEntropy: bound must be non-negative");
      if (scale == 0)
      {
         MFEM_WARNING("SimplexEntropy: bound is zero, entropy is undefined");
      }
   }
   AD_IMPL(T, V, M, x,
   {
      T maxval = x[0];
      for (int i=1; i<x.Size(); i++)
      {
         maxval = max(maxval, x[i]);
      }

      T sum_exp = T();
      for (int i=0; i<x.Size(); i++)
      {
         sum_exp += exp(x[i]);
      }
      return scale*log(sum_exp);
   });
};
} // namespace mfem
