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

#ifndef MFEM_BACKENDS_RAJA_PROLONG_HPP
#define MFEM_BACKENDS_RAJA_PROLONG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

// *****************************************************************************
class ProlongationOperator : public Operator
{
protected:
   const mfem::Operator *pmat;
   RajaSparseMatrix multOp, multTransposeOp;
public:
   ProlongationOperator(RajaSparseMatrix &multOp_,
                        RajaSparseMatrix &multTransposeOp_);

   ProlongationOperator(Layout &in_layout, Layout &out_layout,
                        const mfem::Operator *pmat_);

   // overrides
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_PROLONG_HPP
