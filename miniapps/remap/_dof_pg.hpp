#pragma once
#include "mfem.hpp"
#include "ad_intg.hpp"
#include "pg.hpp"

namespace mfem
{

struct EmptyEnergy : public ADFunction
{
   EmptyEnergy(int n_var)
      : ADFunction(n_var)
   {}
   AD_IMPL(T, V, M, x, return T(););
};

template <ADEval... modes>
class ADDofPGNonlinearFormIntegrator : public
   ADBlockNonlinearFormIntegrator<modes...>
{
   const ADPGFunctional &pg_functional;
   const std::vector<ADEntropy*> &entropies;
public:
   ADDofPGNonlinearFormIntegrator(ADPGFunctional &f,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator<modes...>(f.GetObjective(), ir)
      , pg_functional(f)
      , entropies(f.GetEntropies())
   {}

   ADDofPGNonlinearFormIntegrator(ADPGFunctional &f,
                                  std::initializer_list<int> vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator<modes...>(f.GetObjective(), vdim, ir)
      , pg_functional(f)
      , entropies(f.GetEntropies())
   {}

   ADDofPGNonlinearFormIntegrator(ADPGFunctional &f, const Array<int> &vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator<modes...>(f.GetObjective(), vdim, ir)
      , pg_functional(f)
      , entropies(f.GetEntropies())
   {}

   /// Compute the local energy
   real_t GetElementEnergy(const Array<const FiniteElement *> &el,
                           ElementTransformation &Tr,
                           const Array<const Vector*> &elfun) override;

   /// Perform the local action of the NonlinearFormIntegrator
   void AssembleElementVector(const Array<const FiniteElement *>&el,
                              ElementTransformation &Tr,
                              const Array<const Vector *>&elfun,
                              const Array<Vector *>&elvect) override;

   /// Assemble the local gradient matrix
   void AssembleElementGrad(const Array<const FiniteElement *>&el,
                            ElementTransformation &Tr,
                            const Array<const Vector *>&elfun,
                            const Array2D<DenseMatrix *>&elmat) override;

};

}
