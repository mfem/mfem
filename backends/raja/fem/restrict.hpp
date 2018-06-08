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

#ifndef MFEM_BACKENDS_RAJA_RESTRICT_HPP
#define MFEM_BACKENDS_RAJA_RESTRICT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../raja.hpp"
#include "../linalg/vector.hpp"
#include "../engine/engine.hpp"
#include "../linalg/sparsemat.hpp"
#include "../../../fem/fem.hpp"

namespace mfem
{

namespace raja
{

// *****************************************************************************
class RestrictionOperator : public Operator
{
protected:
   int entries;
   raja::array<int> trueIndices;
public:
   RestrictionOperator(Layout &in_layout, Layout &out_layout,
                       raja::array<int> indices);
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_RESTRICT_HPP
