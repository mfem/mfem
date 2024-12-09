#ifndef MFEM_INTEGRATOR
#define MFEM_INTEGRATOR

#include "../config/config.hpp"
#include "fe.hpp"

namespace mfem
{
/** @brief This base class implements some shared functionality between linear and nonlinear form integrators. */
class Integrator
{
protected:
   const IntegrationRule *IntRule;
   NURBSMeshRules *patchRules = nullptr;

   /** This will pick the NURBS rule's restriction to test_fe if given and applicable, and IntRule otherwise, prioritizing the NURBS rule if available.
       If the NURBS rule is unspecified or inapplicable and IntRule is null, then integrators will fall back on the virtual method GetDefaultIntegrationRule
       to choose a default integration rule. */
   const IntegrationRule* GetIntegrationRule(const FiniteElement* trial_fe,
                                             const FiniteElement* test_fe, const ElementTransformation* trans) const;
   /** Overload that is equivalent to passing NULL for the ElementTransformation pointer, if it is unavailable or unused in the default behavior. */
   const IntegrationRule* GetIntegrationRule(const FiniteElement* trial_fe,
                                             const FiniteElement* test_fe) const;
   /** Overload for cases where trial_fe and test_fe are the same. */
   const IntegrationRule* GetIntegrationRule(const FiniteElement* el,
                                             const ElementTransformation* trans) const;
   /** Overload that is equivalent to passing NULL for the ElementTransformation pointer, if it is unavailable or unused in the default behavior. */
   const IntegrationRule* GetIntegrationRule(const FiniteElement* el) const;

   /** This method is intended to be overriden by subclasses to choose an appropriate integration rule based on the finite element spaces and/or
       element transformation. The trial_fe and test_fe should be equal for linear forms. The default base-class implementation returns null, which
       assumes that an appropriate rule is provided by another means, or that null integration rules are handled appropriately outside of this method. */
   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement* trial_fe, const FiniteElement* test_fe,
      const ElementTransformation* trans) const
   {return NULL;}

public:
   /** Create a new Integrator, optionally providing a prescribed quadrature rule to use in assembly. */
   Integrator(const IntegrationRule *ir = NULL) : IntRule(ir) {};

   /** Prescribe a fixed IntegrationRule to use, or set to null to let the integrator choose an appropriate rule. */
   virtual void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   /** Prescribe a fixed IntegrationRule to use. */
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /** For patchwise integration, SetNURBSPatchIntRule must be called. This will override IntRule if both are non-null. */
   void SetNURBSPatchIntRule(NURBSMeshRules *pr) { patchRules = pr; }

   /** Check if a NURBS patch integration rule has been set. */
   bool HasNURBSPatchIntRule() const { return patchRules != nullptr; }

   /** Directly return the IntRule pointer (possibly null) without checking for NURBS patch rules or falling back on a default. */
   const IntegrationRule *GetIntegrationRule() const { return IntRule; }
};
}

#endif
