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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// **************************************************************************
RestrictionOperator::RestrictionOperator(kernels::Layout &in_layout,
                                         kernels::Layout &out_layout,
                                         const kernels::array<int> *indices) :
   kernels::Operator(in_layout, out_layout)
{
   push();
   entries = indices->size() >> 1;
   trueIndices = indices;
   pop();
}

// **************************************************************************
void RestrictionOperator::Mult_(const kernels::Vector &x,
                                kernels::Vector &y) const
{
   push();
   rExtractSubVector(entries, trueIndices->ptr(), x.GetData(), y.GetData());
   pop();
}

// **************************************************************************
void RestrictionOperator::MultTranspose_(const kernels::Vector &x,
                                         kernels::Vector &y) const
{
   push();
   assert(false);
   y.Fill<double>(0.0);
   rMapSubVector(entries, trueIndices->ptr(), x.GetData(), y.GetData());
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
