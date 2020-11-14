#include "mfem.hpp"

#pragma once

namespace mfem
{
class GenericIntegrator
{
public:
   GenericIntegrator(const IntegrationRule *ir = nullptr) : IntRule(ir) {}

   virtual void Setup(const FiniteElementSpace &) = 0;

   virtual void Apply(const Vector &x, Vector &y) const = 0;

   virtual void ApplyGradient(const Vector &x,
                              const Vector &v,
                              Vector &y) const = 0;

protected:
   const IntegrationRule *IntRule;
};
} // namespace mfem