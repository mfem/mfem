// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "estimators.hpp"

namespace mfem
{

void ZienkiewiczZhuEstimator::ComputeEstimates()
{
   flux_space->Update(false);
   GridFunction flux(flux_space);

   if (!anisotropic) { aniso_flags.SetSize(0); }
   const int with_subdomains = 1;
   total_error = ZZErrorEstimator(*integ, *solution, flux, error_estimates,
                                  anisotropic ? &aniso_flags : NULL,
                                  with_subdomains);

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

} // namespace mfem
