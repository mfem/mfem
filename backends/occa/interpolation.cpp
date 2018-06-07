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

// TODO this should be moved somewhere more useful
namespace {
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

void OccaMult(const mfem::Operator &op,
	      const Vector &x, Vector &y) {
   ::occa::device device = x.OccaLayout().OccaEngine().GetDevice();
   if (device.hasSeparateMemorySpace()) {
      mfem::Vector &hostX = GetHostVector(0, op.Width());
      mfem::Vector &hostY = GetHostVector(1, op.Height());
      x.OccaMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
      op.Mult(hostX, hostY);
      y.OccaMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));
   } else {
      mfem::Vector hostX((double*) x.OccaMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.OccaMem().ptr(), y.Size());
      op.Mult(hostX, hostY);
   }
}

void OccaMultTranspose(const mfem::Operator &op,
		       const Vector &x, Vector &y) {
   ::occa::device device = x.OccaLayout().OccaEngine().GetDevice();
   if (device.hasSeparateMemorySpace()) {
      mfem::Vector &hostX = GetHostVector(1, op.Height());
      mfem::Vector &hostY = GetHostVector(0, op.Width());
      x.OccaMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
      op.MultTranspose(hostX, hostY);
      y.OccaMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));
   } else {
      mfem::Vector hostX((double*) x.OccaMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.OccaMem().ptr(), y.Size());
      op.MultTranspose(hostX, hostY);
   }
}
}

void CreateRPOperators(Layout &v_layout, Layout &t_layout,
                       const mfem::SparseMatrix *R, const mfem::Operator *P,
                       mfem::Operator *&OccaR, mfem::Operator *&OccaP)
{
   if (!P)
   {
      OccaR = new IdentityOperator(t_layout);
      OccaP = new IdentityOperator(t_layout);
      return;
   }

   const mfem::SparseMatrix *pmat = dynamic_cast<const mfem::SparseMatrix*>(P);
   ::occa::device device = v_layout.OccaEngine().GetDevice();

   if (R)
   {
      OccaSparseMatrix *occaR =
         CreateMappedSparseMatrix(v_layout, t_layout, *R);
      ::occa::array<int> reorderIndices = occaR->reorderIndices;
      delete occaR;

      OccaR = new RestrictionOperator(v_layout, t_layout, reorderIndices);
   }

   if (pmat)
   {
      const mfem::SparseMatrix *pmatT = Transpose(*pmat);

      OccaSparseMatrix *occaP  =
         CreateMappedSparseMatrix(t_layout, v_layout, *pmat);
      OccaSparseMatrix *occaPT =
         CreateMappedSparseMatrix(v_layout, t_layout, *pmatT);

      OccaP = new ProlongationOperator(*occaP, *occaPT);
   }
   else
   {
      OccaP = new ProlongationOperator(t_layout, v_layout, P);
   }
}

RestrictionOperator::RestrictionOperator(Layout &in_layout, Layout &out_layout,
                                         ::occa::array<int> indices) :
   Operator(in_layout, out_layout)
{

   entries     = indices.size() / 2;
   trueIndices = indices;

   // FIXME: paths ...
   ::occa::device device = in_layout.OccaEngine().GetDevice();
   const std::string &okl_path = in_layout.OccaEngine().GetOklPath();
   const std::string &okl_defines = in_layout.OccaEngine().GetOklDefines();
   multOp = device.buildKernel(okl_path + "mappings.okl",
                               "ExtractSubVector",
                               "defines: { TILESIZE: 256 }" + okl_defines);

   multTransposeOp = device.buildKernel(okl_path + "mappings.okl",
                                        "SetSubVector",
                                        "defines: { TILESIZE: 256 }" +
                                        okl_defines);
}

void RestrictionOperator::Mult_(const Vector &x, Vector &y) const
{
   multOp(entries, trueIndices, x.OccaMem(), y.OccaMem());
}

void RestrictionOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   y.Fill<double>(0.0);
   multTransposeOp(entries, trueIndices, x.OccaMem(), y.OccaMem());
}

ProlongationOperator::ProlongationOperator(OccaSparseMatrix &multOp_,
                                           OccaSparseMatrix &multTransposeOp_) :
   Operator(multOp_),
   pmat(NULL),
   multOp(multOp_),
   multTransposeOp(multTransposeOp_) {}

ProlongationOperator::ProlongationOperator(Layout &in_layout,
                                           Layout &out_layout,
                                           const mfem::Operator *pmat_) :
   Operator(in_layout, out_layout),
   pmat(pmat_),
   multOp(*this),
   multTransposeOp(*this)
{ }

void ProlongationOperator::Mult_(const Vector &x, Vector &y) const
{
   if (pmat)
   {
      OccaMult(*pmat, x, y);
   }
   else
   {
      // TODO: define 'ox' and 'oy'
      multOp.Mult_(x, y);
   }
}

void ProlongationOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   if (pmat)
   {
      OccaMultTranspose(*pmat, x, y);
   }
   else
   {
      multTransposeOp.Mult_(x, y);
   }
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
