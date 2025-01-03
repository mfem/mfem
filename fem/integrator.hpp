// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
   /** Create a new Integrator, optionally providing a prescribed quadrature
       rule to use in assembly. */
   Integrator(const IntegrationRule *ir = NULL) : IntRule(ir) {};

   /** Prescribe a fixed IntegrationRule to use, or set to null to let the
       integrator choose an appropriate rule. */
   virtual void SetIntRule(const IntegrationRule *ir) { IntRule = ir; }

   /** Prescribe a fixed IntegrationRule to use. */
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /** For patchwise integration, SetNURBSPatelementchIntRule must be called.
       This will override IntRule if both are non-null. */
   void SetNURBSPatchIntRule(NURBSMeshRules *pr) { patchRules = pr; }

   /** Check if a NURBS patch integration rule has been set. */
   bool HasNURBSPatchIntRule() const { return patchRules != nullptr; }

   /** Directly return the IntRule pointer (possibly null) without checking
       for NURBS patch rules or falling back on a default. */
   const IntegrationRule *GetIntegrationRule() const { return IntRule; }

  /** Equivalent to GetIntegrationRule, but retained for backward
      compatibility with applications. */
  const IntegrationRule *GetIntRule() const { return IntRule; }

protected:
   const IntegrationRule *IntRule;
   NURBSMeshRules *patchRules = nullptr;

   /** @brief Selects an integration rule based on the the arguments and
              internal state of the Integrator object.

       @details This method selects an integration rule in a way that depends
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
      const FiniteElement* trial_fe, const FiniteElement* test_fe,
      const ElementTransformation* trans) const;
   /** Overload that is equivalent to passing NULL for the
       ElementTransformation pointer, if it is unavailable or unused in the
       default behavior. */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement* trial_fe,
      const FiniteElement* test_fe) const;
   /** Overload for cases where trial_fe and test_fe are the same. */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement* el,
      const ElementTransformation* trans) const;
   /** Overload that is equivalent to passing NULL for the
       ElementTransformation pointer, if it is unavailable or unused in the
       default behavior. */
   const IntegrationRule* GetIntegrationRule(const FiniteElement* el) const;

   /** @brief Subclasses should override to choose a default integration rule.

       @details This method is intended to be overriden by subclasses to
                choose an appropriate integration rule based on the finite
                element spaces and/or element transformation. The trial_fe
                and test_fe should be equal for linear forms. The default
                base-class implementation returns null, which assumes that
                an appropriate rule is provided by another means, or that null
                integration rules are handled appropriately by the caller.
   */
   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement* trial_fe, const FiniteElement* test_fe,
      const ElementTransformation* trans) const
   { return NULL; }
};
}

#endif
