// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "estimators.hpp"

namespace mfem
{

void ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   // In parallel, 'flux' can be a GridFunction, as long as 'flux_space' is a
   // ParFiniteElementSpace and 'solution' is a ParGridFunction.
   GridFunction flux(flux_space);

   if (!anisotropic) { aniso_flags.SetSize(0); }
   total_error = ZZErrorEstimator(*integ, *solution, flux, error_estimates,
                                  anisotropic ? &aniso_flags : NULL,
                                  flux_averaging,
                                  with_coeff);

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();
}


#ifdef MFEM_USE_MPI

void L2ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   smooth_flux_space->Update(false);

   // TODO: move these parameters in the class, and add Set* methods.
   const double solver_tol = 1e-12;
   const int solver_max_it = 200;
   total_error = L2ZZErrorEstimator(*integ, *solution, *smooth_flux_space,
                                    *flux_space, error_estimates,
                                    local_norm_p, solver_tol, solver_max_it);

   current_sequence = solution->FESpace()->GetMesh()->GetSequence();
}

#endif // MFEM_USE_MPI

void LpErrorEstimator::ComputeEstimates()
{
   MFEM_VERIFY(coef != NULL || vcoef != NULL,
               "LpErrorEstimator has no coefficient!  Call SetCoef first.");

   error_estimates.SetSize(sol->FESpace()->GetMesh()->GetNE());
   if (coef)
   {
      sol->ComputeElementLpErrors(local_norm_p, *coef, error_estimates);
   }
   else
   {
      sol->ComputeElementLpErrors(local_norm_p, *vcoef, error_estimates);
   }
   current_sequence = sol->FESpace()->GetMesh()->GetSequence();
}

} // namespace mfem
