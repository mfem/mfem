// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_INTEGRATOR
#define MFEM_INTEGRATOR

#include "../config/config.hpp"
#include "fe.hpp"

namespace mfem
{
/** @brief This base class implements some shared functionality between
    linear and nonlinear form integrators. */
class Integrator
{
public:
   /** @brief Create a new Integrator, optionally providing a prescribed
       quadrature rule to use in assembly. */
   Integrator(const IntegrationRule *ir = NULL) : IntRule(ir) {}

   /** @brief Prescribe a fixed IntegrationRule to use, or set to null to let
              the integrator choose an appropriate rule.

       @details This method allows setting a custom integration rule to use
                on each element during assembly, overriding the default
                choice if it is non-null. Passing a non-null value will
                set the Integrator's NURBS patch integration rule to null
                to avoid ambiguity in GetIntegrationRule.
   */
   virtual void SetIntRule(const IntegrationRule *ir)
   { IntRule = ir; if (ir) { patchRules = nullptr; } }

   /** @brief Prescribe a fixed IntegrationRule to use. Sets the NURBS patch
              integration rule to null.

       @see SetIntRule(const IntegrationRule*)
   */
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /** @brief Sets an integration rule for use on NURBS patches.

       @details For patchwise integration, SetNURBSPatchIntRule
                must be called. Passing a non-null value will set the
                Integrator's standard element IntegrationRule to null
                to avoid ambiguity in GetIntegrationRule.
   */
   void SetNURBSPatchIntRule(NURBSMeshRules *pr)
   { patchRules = pr; if (pr) { IntRule = nullptr; } }

   /** @brief Check if a NURBS patch integration rule has been set. */
   bool HasNURBSPatchIntRule() const { return patchRules != nullptr; }

   /** @brief Directly return the IntRule pointer (possibly null) without
       checking for NURBS patch rules or falling back on a default. */
   const IntegrationRule *GetIntRule() const { return IntRule; }

   /** @brief Equivalent to GetIntRule, but retained for backward
       compatibility with applications. */
   const IntegrationRule *GetIntegrationRule() const { return GetIntRule(); }

protected:
   const IntegrationRule *IntRule;
   NURBSMeshRules *patchRules = nullptr;

   /** @brief Returns an integration rule based on the arguments and internal
              state of the Integrator object.

       @details This method returns an integration rule in a way that depends
                on the integrator's attributes. Attributes can specify an
                existing IntegrationRule, and/or a NURBSMeshRules object.
                This method will pick the NURBSMeshRules' restriction to the
                element if given and applicable, and IntRule otherwise,
                prioritizing the NURBS rule if available. If neither is
                valid, the integrator will fall back on the virtual method
                GetDefaultIntegrationRule to choose a default integration
                rule, where subclasses can override this in a problem-specific
                way.
   */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const;

   /** @brief Returns an integration rule based on the arguments and
              internal state. (Version for identical trial_fe and test_fe)

       @see GetIntegrationRule(const FiniteElement*, const FiniteElement*,
            const ElementTransformation*)
   */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& el,
      const ElementTransformation& trans) const;

   /** @brief Subclasses should override to choose a default integration rule.

       @details This method is intended to be overridden by subclasses to
                choose an appropriate integration rule based on the finite
                element spaces and/or element transformation. The trial_fe
                and test_fe should be equal for linear forms. The default
                base-class implementation returns null, which assumes that
                an appropriate rule is provided by another means, or that null
                integration rules are handled appropriately by the caller.
   */
   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const
   { return NULL; }
};
}

#endif
