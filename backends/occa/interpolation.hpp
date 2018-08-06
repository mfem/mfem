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

#ifndef MFEM_BACKENDS_OCCA_INTERPOLATION_HPP
#define MFEM_BACKENDS_OCCA_INTERPOLATION_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include <occa.hpp>
#include "vector.hpp"
#include "engine.hpp"
#include "sparsemat.hpp"
#include "../../fem/fem.hpp"

namespace mfem
{

namespace occa
{

class RestrictionOperator : public Operator
{
protected:
   ::occa::array<int> trueIndices; // ldof = trueIndices[ltdof]
   ::occa::kernel multOp, multTransposeOp;

public:
   RestrictionOperator(Layout &in_layout, Layout &out_layout,
                       ::occa::array<int> indices);

   // overrides
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;
};

class ProlongationOperator : public Operator
{
protected:
   const mfem::Operator *pmat;
   OccaSparseMatrix multOp, multTransposeOp;

public:
   ProlongationOperator(OccaSparseMatrix &multOp_,
                        OccaSparseMatrix &multTransposeOp_);

   ProlongationOperator(Layout &in_layout, Layout &out_layout,
                        const mfem::Operator *pmat_);

   // overrides
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;

   // overrides
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
   virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const;
};

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_INTERPOLATION_HPP
