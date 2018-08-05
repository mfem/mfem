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

#include "operator.hpp"

namespace mfem
{

namespace occa
{

// FIXME: move this object to the Backend?
::occa::kernelBuilder OccaConstrainedOperator::mapDofBuilder =
   ::occa::linalg::customLinearMethod(
      "vector_map_dofs",

      "const int idx = v2[i];"
      "v0[idx] = v1[idx];",

      "defines: {"
      "  VTYPE0: 'double',"
      "  VTYPE1: 'double',"
      "  VTYPE2: 'int',"
      "  TILESIZE: 128,"
      "}");

// FIXME: move this object to the Backend?
::occa::kernelBuilder OccaConstrainedOperator::clearDofBuilder =
   ::occa::linalg::customLinearMethod(
      "vector_clear_dofs",

      "v0[v1[i]] = 0.0;",

      "defines: {"
      "  VTYPE0: 'double',"
      "  VTYPE1: 'int',"
      "  TILESIZE: 128,"
      "}");

OccaConstrainedOperator::OccaConstrainedOperator(
   mfem::Operator *A_,
   const mfem::Array<int> &constraintList_,
   bool own_A_)

   : Operator(A_->InLayout()->As<Layout>()),
     z(OutLayout_()),
     w(OutLayout_()),
     mfem_z((z.DontDelete(), z)),
     mfem_w((w.DontDelete(), w))
{
   Setup(OutLayout_().OccaEngine().GetDevice(), A_, constraintList_, own_A_);
}

void OccaConstrainedOperator::Setup(::occa::device device_,
                                    mfem::Operator *A_,
                                    const mfem::Array<int> &constraintList_,
                                    bool own_A_)
{
   device = device_;

   A = A_;
   own_A = own_A_;

   constraintIndices = constraintList_.Size();
   if (constraintList_.Size() > 0)
   {
      constraintList = constraintList_.Get_PArray()->As<Array>().OccaMem();
   }
   else
   {
      constraintList = ::occa::memory();
   }
}

void OccaConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   ::occa::kernel mapDofs = mapDofBuilder.build(device);

   w.Fill<double>(0.0);

   if (constraintIndices)
   {
      mapDofs(constraintIndices, w.OccaMem(), x.OccaMem(), constraintList);
   }

   A->Mult(mfem_w, mfem_z);

   b.Axpby<double>(1.0, b, -1.0, z);

   if (constraintIndices)
   {
      mapDofs(constraintIndices, b.OccaMem(), x.OccaMem(), constraintList);
   }
}

void OccaConstrainedOperator::Mult_(const Vector &x, Vector &y) const
{
   mfem::Vector mfem_y(y);
   if (constraintIndices == 0)
   {
      A->Mult(x.Wrap(), mfem_y);
      return;
   }

   ::occa::kernel mapDofs   = mapDofBuilder.build(device);
   ::occa::kernel clearDofs = clearDofBuilder.build(device);

   z.Assign<double>(x); // z = x

   clearDofs(constraintIndices, z.OccaMem(), constraintList);

   A->Mult(mfem_z, mfem_y);

   mapDofs(constraintIndices, y.OccaMem(), x.OccaMem(), constraintList);
}

OccaConstrainedOperator::~OccaConstrainedOperator()
{
   if (own_A)
   {
      delete A;
   }
}

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
