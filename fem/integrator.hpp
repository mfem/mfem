#ifndef MFEM_INTEGRATOR
#define MFEM_INTEGRATOR

#include "../config/config.hpp"
#include "fe.hpp"

namespace mfem
{

/** @brief This wraps a pointer to an integration rule, while allowing it to be designated as temporary, to be deleted after passing out of scope */
class IntegrationRulePtr
{
private:
   const IntegrationRule *IntRule;
   bool DeleteRule;
public:
   IntegrationRulePtr(const IntegrationRule* intRule,
                      const bool deleteRule = false) :
      IntRule(intRule), DeleteRule(deleteRule) {}
   // Responsibility for deletion is trasnferred by copy or assignment, so const references cannot be copied/assigned.
   IntegrationRulePtr(IntegrationRulePtr& other) : IntRule(other.IntRule),
      DeleteRule(other.DeleteRule) {other.DeleteRule = false;}
   IntegrationRulePtr& operator=(IntegrationRulePtr& other)
   {
      if (this == &other) { return *this; }
      this->IntRule = other.IntRule;
      this->DeleteRule = other.DeleteRule;
      other.DeleteRule = false;
      return *this;
   }
   IntegrationRulePtr(const IntegrationRulePtr& other) = delete;
   IntegrationRulePtr& operator=(const IntegrationRulePtr& other) = delete;
   ~IntegrationRulePtr() {if (DeleteRule) {delete IntRule;}}
   const IntegrationRule* Get() const {return IntRule;}
};

/** @brief This base class implements some shared functionality between linear and nonlinear form integrators. */
class Integrator
{
protected:
   const IntegrationRule *IntRule;
   // Prescribed integration rules (not reduced approximate rules).
   NURBSMeshRules *patchRules = nullptr;

   // This will pick the NURBS rule's restriction to `el` if given and applicable, and `IntRule` otherwise, prioritizing the NURBS rule if available.
   // If the NURBS rule is unspecified or inapplicable and `IntRule` is `NULL`, then forms will fall back on form-specific logic to choose a default
   // integration rule.
   IntegrationRulePtr GetIntegrationRule(const FiniteElement* trial_fe,
                                         const FiniteElement* test_fe, const ElementTransformation* trans) const;
   // Overloads for convenience:
   IntegrationRulePtr GetIntegrationRule(const FiniteElement* trial_fe,
                                         const FiniteElement* test_fe) const;
   IntegrationRulePtr GetIntegrationRule(const FiniteElement* el,
                                         const ElementTransformation* trans) const;
   IntegrationRulePtr GetIntegrationRule(const FiniteElement* el) const;

   // Method to be implemented by subclasses to define a default quadrature rule when neither NURBS rules nor `IntRule` are available.
   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement* trial_fe, const FiniteElement* test_fe,
      const ElementTransformation* trans) const
   // TODO: Move various `ir == NULL` branches from subclass' element matrix/vector assembly methods into implementations of this to consolidate logic. Ideally, it would be
   // pure virtual, but it may take some time and further refactoring to make that feasible. The current default of `return NULL` is compatible with how existing defaults
   // for forms are implemented (i.e., checking whether `IntRule` is `NULL`), so this is non-breaking in its current form and allows default schemes to be implemented
   // incrementally.
   {return NULL;}

public:
   Integrator(const IntegrationRule *ir = NULL) : IntRule(ir) {};

   /** @brief Prescribe a fixed IntegrationRule to use (when @a ir != NULL) or
       let the integrator choose (when @a ir == NULL). */
   virtual void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   /// Prescribe a fixed IntegrationRule to use.
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /// For patchwise integration, SetNURBSPatchIntRule must be called.
   void SetNURBSPatchIntRule(NURBSMeshRules *pr) { patchRules = pr; }
   bool HasNURBSPatchIntRule() const { return patchRules != nullptr; }

   /// Get the integration rule of the integrator (possibly NULL).
   const IntegrationRule *GetIntegrationRule() const { return IntRule; }
};
}

#endif
