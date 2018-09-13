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

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "interpolation.hpp"

namespace mfem
{

namespace occa
{

RestrictionOperator::RestrictionOperator(Layout &in_layout, Layout &out_layout,
                                         ::occa::array<int> indices)
   : Operator(in_layout, out_layout)
{
   trueIndices = indices;

   ::occa::device device = in_layout.OccaEngine().GetDevice();
   const std::string &okl_path = in_layout.OccaEngine().GetOklPath();
   multOp = device.buildKernel(okl_path + "mappings.okl",
                               "ExtractSubVector",
                               "defines: { TILESIZE: 256 }");

   multTransposeOp = device.buildKernel(okl_path + "mappings.okl",
                                        "SetSubVector",
                                        "defines: { TILESIZE: 256 }");
}

void RestrictionOperator::Mult_(const Vector &x, Vector &y) const
{
   // y[i] = x[trueIndices[i]]
   multOp(height, trueIndices, x.OccaMem(), y.OccaMem());
}

void RestrictionOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   y.OccaFill<double>(0.0);
   // y[trueIndices[i]] = x[i]
   multTransposeOp(height, trueIndices, x.OccaMem(), y.OccaMem());
}

ProlongationOperator::ProlongationOperator(OccaSparseMatrix &multOp_,
                                           OccaSparseMatrix &multTransposeOp_)
   : Operator(multOp_),
     pmat(NULL),
     multOp(multOp_),
     multTransposeOp(multTransposeOp_)
{ }

ProlongationOperator::ProlongationOperator(Layout &in_layout,
                                           Layout &out_layout,
                                           const mfem::Operator *pmat_)
   : Operator(in_layout, out_layout),
     pmat(pmat_),
     multOp(*this),
     multTransposeOp(*this)
{ }

void ProlongationOperator::Mult_(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(pmat == NULL, "");
   multOp.Mult_(x, y);
}

void ProlongationOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(pmat == NULL, "");
   multTransposeOp.Mult_(x, y);
}

void ProlongationOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   if (pmat)
   {
      // FIXME: create an OCCA version of 'pmat'
      x.Pull();
      y.Pull(false);
      pmat->Mult(x, y);
      y.Push();
   }
   else
   {
      multOp.Mult(x, y);
   }
}

void ProlongationOperator::MultTranspose(const mfem::Vector &x,
                                         mfem::Vector &y) const
{
   if (pmat)
   {
      // FIXME: create an OCCA version of 'pmat'
      x.Pull();
      y.Pull(false);
      pmat->MultTranspose(x, y);
      y.Push();
   }
   else
   {
      multTransposeOp.Mult(x, y);
   }
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
