#include "integrator.hpp"
#include "fem.hpp"
#include "intrules.hpp"

namespace mfem
{
const IntegrationRule* Integrator::GetIntegrationRule(const FiniteElement*
                                                      trial_fe, const FiniteElement* test_fe, const ElementTransformation* trans,
                                                      bool& deleteRule) const
{
   const NURBSFiniteElement *NURBSFE = dynamic_cast<const NURBSFiniteElement *>
                                       (test_fe);
   deleteRule = false;
   if (NURBSFE && patchRules)
   {
      const int patch = NURBSFE->GetPatch();
      const int* ijk = NURBSFE->GetIJK();
      Array<const KnotVector*>& kv = NURBSFE->KnotVectors();
      return &patchRules->GetElementRule(NURBSFE->GetElement(), patch, ijk, kv,
                                         deleteRule);
   }
   else if (IntRule)
   {
      return IntRule;
   }
   else
   {
      return GetDefaultIntegrationRule(trial_fe, test_fe, trans);
   }
}
const IntegrationRule* Integrator::GetIntegrationRule(const FiniteElement*
                                                      trial_fe, const FiniteElement* test_fe, bool& deleteRule) const
{
   return GetIntegrationRule(trial_fe, test_fe, NULL, deleteRule);
}
const IntegrationRule* Integrator::GetIntegrationRule(const FiniteElement* el,
                                                      const ElementTransformation* trans, bool& deleteRule) const
{
   return GetIntegrationRule(el, el, trans, deleteRule);
}
const IntegrationRule* Integrator::GetIntegrationRule(const FiniteElement* el,
                                                      bool& deleteRule) const
{
   return GetIntegrationRule(el, el, NULL, deleteRule);
}
}
