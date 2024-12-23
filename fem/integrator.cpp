#include "integrator.hpp"
#include "fem.hpp"
#include "intrules.hpp"

namespace mfem
{
const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement* trial_fe, const FiniteElement* test_fe,
   const ElementTransformation* trans) const
{
   const NURBSFiniteElement *NURBSFE = dynamic_cast<const NURBSFiniteElement *>
                                       (test_fe);
   const IntegrationRule* result;
   if (NURBSFE && patchRules)
   {
      const int patch = NURBSFE->GetPatch();
      const int* ijk = NURBSFE->GetIJK();
      Array<const KnotVector*>& kv = NURBSFE->KnotVectors();
      result = &patchRules->GetElementRule(NURBSFE->GetElement(), patch, ijk,
                                           kv);
   }
   else if (IntRule)
   {
      result = IntRule;
   }
   else
   {
      result = GetDefaultIntegrationRule(trial_fe, test_fe, trans);
   }
   return result;
}
const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement* trial_fe, const FiniteElement* test_fe) const
{
   return GetIntegrationRule(trial_fe, test_fe, NULL);
}
const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement* el,
   const ElementTransformation* trans) const
{
   return GetIntegrationRule(el, el, trans);
}
const IntegrationRule* Integrator::GetIntegrationRule(const FiniteElement* el)
const
{
   return GetIntegrationRule(el, el, NULL);
}
}
