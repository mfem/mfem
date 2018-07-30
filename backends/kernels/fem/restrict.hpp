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

#ifndef MFEM_BACKENDS_KERNELS_RESTRICT_HPP
#define MFEM_BACKENDS_KERNELS_RESTRICT_HPP
#warning RajaRestrictionOperator

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

   // ***************************************************************************
   // * RajaRestrictionOperator
   // ***************************************************************************
   class RajaRestrictionOperator : public mfem::Operator {
   protected:
      int entries;
      const kernels::array<int> *indices;
   public:
      RajaRestrictionOperator(const int h, const int w,
                              const kernels::array<int> *idx):
         mfem::Operator(h,w),
         entries(idx->size()>>1),
         indices(idx){} 
      void Mult(const kernels::Vector& x, kernels::Vector& y) const ;
   };

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_RESTRICT_HPP
