#include "mfem.hpp"
#include "genericintegrator.hpp"

#pragma once

namespace mfem
{
class GenericForm : public Operator
{
public:
   GenericForm(FiniteElementSpace *f);

   void Mult(const Vector &x, Vector &y) const override;

   void AddDomainIntegrator(GenericIntegrator *i)
   {
      domain_integrators.Append(i);
      i->Setup(*fes);
   }

protected:
   FiniteElementSpace *fes;
   Array<GenericIntegrator *> domain_integrators;
   Array<int> ess_tdof_list;

   // T -> L
   const Operator *P;
   // L -> E
   const Operator *G;

   mutable Vector x_local, y_local, z1;
};
} // namespace mfem