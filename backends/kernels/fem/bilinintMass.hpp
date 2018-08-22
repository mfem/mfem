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

#ifndef MFEM_BACKENDS_KERNELS_BILIN_INTEG_MASS_HPP
#define MFEM_BACKENDS_KERNELS_BILIN_INTEG_MASS_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsMassIntegrator : public KernelsIntegrator
{
private:
   const mfem::Engine& engine;
   KernelsCoefficient coeff;
   kernels::Vector assembledOperator;
   mfem::Vector op;
public:
   KernelsMassIntegrator(const mfem::Engine&);
   KernelsMassIntegrator(const KernelsCoefficient&);
   virtual ~KernelsMassIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   void SetOperator(mfem::Vector &v);
   virtual void MultAdd(Vector &x, Vector &y);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_BILIN_INTEG_MASS_HPP
