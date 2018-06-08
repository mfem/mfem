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
#include "../fem/prolong.hpp"
#include "../linalg/vector.hpp"
#include "../engine/engine.hpp"
#include "../linalg/sparsemat.hpp"
#include "../../../fem/fem.hpp"

namespace mfem
{

namespace raja
{
      
mfem::Vector& GetHostVector(const int id, const int64_t size) {
   static std::vector<mfem::Vector*> v;
   if (v.size() <= (size_t) id) {
      for (int i = (int) v.size(); i < (id + 1); ++i) {
         v.push_back(new mfem::Vector);
      }
   }
   if (size >= 0) {
      v[id]->SetSize(size);
   }
   return *(v[id]);
}
   
void RajaMult(const mfem::Operator &op,
              const Vector &x, Vector &y) {
   raja::device device = x.RajaLayout().RajaEngine().GetDevice();
   if (device.hasSeparateMemorySpace()) {
      assert(false);/*
                      mfem::Vector &hostX = GetHostVector(0, op.Width());
                      mfem::Vector &hostY = GetHostVector(1, op.Height());
                      x.RajaMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
                      op.Mult(hostX, hostY);
                      y.RajaMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   } else {
      mfem::Vector hostX((double*) x.RajaMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.RajaMem().ptr(), y.Size());
      op.Mult(hostX, hostY);
   }
}

void RajaMultTranspose(const mfem::Operator &op,
                       const Vector &x, Vector &y) {
   raja::device device = x.RajaLayout().RajaEngine().GetDevice();
   if (device.hasSeparateMemorySpace()) {
      assert(false);/*
      mfem::Vector &hostX = GetHostVector(1, op.Height());
      mfem::Vector &hostY = GetHostVector(0, op.Width());
      x.RajaMem().copyTo((void*)hostX.GetData(), hostX.Size() * sizeof(double));
      op.MultTranspose(hostX, hostY);
      y.RajaMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   } else {
      mfem::Vector hostX((double*) x.RajaMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.RajaMem().ptr(), y.Size());
      op.MultTranspose(hostX, hostY);
   }
}

   // **************************************************************************
   ProlongationOperator::ProlongationOperator(RajaSparseMatrix &multOp_,
                                              RajaSparseMatrix &multTransposeOp_) :
      Operator(multOp_),
      pmat(NULL),
      multOp(multOp_),
      multTransposeOp(multTransposeOp_) {}
   
   // **************************************************************************
   ProlongationOperator::ProlongationOperator(Layout &in_layout,
                                              Layout &out_layout,
                                              const mfem::Operator *pmat_) :
      Operator(in_layout, out_layout),
      pmat(pmat_),
      multOp(*this),
      multTransposeOp(*this)
   { }

   // **************************************************************************
   void ProlongationOperator::Mult_(const Vector &x, Vector &y) const
   {
      //assert(false);
      if (pmat)
      {
         RajaMult(*pmat, x, y);
      }
      else
      {
         // TODO: define 'ox' and 'oy'
         multOp.Mult_(x, y);
      }
   }

   // **************************************************************************
   void ProlongationOperator::MultTranspose_(const Vector &x, Vector &y) const
   {
      if (pmat)
      {
         RajaMultTranspose(*pmat, x, y);
      }
      else
      {
         multTransposeOp.Mult_(x, y);
      }
   }

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
