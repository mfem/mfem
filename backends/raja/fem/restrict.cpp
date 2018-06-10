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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"

namespace mfem
{

namespace raja
{

// **************************************************************************
RestrictionOperator::RestrictionOperator(Layout &in_layout, Layout &out_layout,
                                         raja::array<int> indices) :
   Operator(in_layout, out_layout)
{
   push();
   entries     = indices.size() / 2;
   trueIndices = indices;
   pop();
}

// **************************************************************************
void RestrictionOperator::Mult_(const Vector &x, Vector &y) const
{
   push();
   dbg("\033[34;1m[Mult_] entries=%ld, trueIndices size=%ld",entries,
       trueIndices.Size());
   rExtractSubVector(entries, trueIndices, (double*)x.RajaMem().ptr(),
                     (double*)y.RajaMem().ptr());
   //assert(false);
   pop();
}

// **************************************************************************
void RestrictionOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   push();
   y.Fill<double>(0.0);
   assert(false);
   rMapSubVector(entries, trueIndices,
                 (double*)x.RajaMem().ptr(),
                 (double*)y.RajaMem().ptr());
   pop();
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
