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

#ifndef MFEM_BACKENDS_KERNELS_BILIN_INTEG_DIFFUSION_HPP
#define MFEM_BACKENDS_KERNELS_BILIN_INTEG_DIFFUSION_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsDiffusionIntegrator : public KernelsIntegrator
{
private:
   KernelsCoefficient coeff;
   Vector assembledOperator;
public:
   KernelsDiffusionIntegrator(const KernelsCoefficient &coeff_);
   virtual ~KernelsDiffusionIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   virtual void MultAdd(Vector &x, Vector &y);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_BILIN_INTEG_DIFFUSION_HPP
