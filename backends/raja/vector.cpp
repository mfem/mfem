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
#include "raja.hpp"

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "vector.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

namespace raja
{


PVector *Vector::DoVectorClone(bool copy_data, void **buffer,
                               int buffer_type_id) const
{
   MFEM_ASSERT(buffer_type_id == ScalarId<double>::value, "");
   Vector *new_vector = new Vector(RajaLayout());
   if (copy_data)
   {
      new_vector->slice.copyFrom(slice);
   }
   if (buffer)
   {
      *buffer = new_vector->GetBuffer();
   }
   return new_vector;
}

void Vector::DoDotProduct(const PVector &x, void *result,
                          int result_type_id) const
{
   // called only when Size() != 0
   MFEM_ASSERT(result_type_id == ScalarId<double>::value, "");
   double *res = (double *)result;
   MFEM_ASSERT(dynamic_cast<const Vector *>(&x) != NULL, "invalid Vector type");
   const Vector *xp = static_cast<const Vector *>(&x);
   MFEM_ASSERT(this->Size() == xp->Size(), "");
   *res = raja::linalg::dot(this->slice, xp->slice);
   // FIXME: MPI*/
}

void Vector::DoAxpby(const void *a, const PVector &x,
                     const void *b, const PVector &y,
                     int ab_type_id)
{
   MFEM_ABORT("FIXME");
}

mfem::Vector Vector::Wrap()
{
   return mfem::Vector(*this);
}

const mfem::Vector Vector::Wrap() const
{
   return mfem::Vector(*const_cast<Vector*>(this));
}

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
