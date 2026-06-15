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

#include "integrator.hpp"
#include "fem.hpp"
#include "intrules.hpp"

namespace mfem
{
const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement& trial_fe, const FiniteElement& test_fe,
   const ElementTransformation& trans) const
{
   const IntegrationRule* result;
   const NURBSFiniteElement *NURBSFE;
   if (patchRules &&
       (NURBSFE = dynamic_cast<const NURBSFiniteElement *>(&test_fe)))
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
   const FiniteElement& el,
   const ElementTransformation& trans) const
{
   return GetIntegrationRule(el, el, trans);
}
}
