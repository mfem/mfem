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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "vector.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

namespace omp
{

PVector *Vector::DoVectorClone(bool copy_data, void **buffer,
                               int buffer_type_id) const
{
   MFEM_ASSERT(buffer_type_id == ScalarId<double>::value, "");
   Vector *new_vector = new Vector(OmpLayout());
   if (copy_data)
   {
      const std::size_t total_size = sizeof(double) * OmpLayout().Size();
      if (!ComputeOnDevice())
         std::memcpy(new_vector->GetBuffer(), data, total_size);
      else
      {
         char *new_data = new_vector->GetBuffer();
#pragma omp target teams distribute parallel for is_device_ptr(new_data)
         for (std::size_t i = 0; i < total_size; i++) new_data[i] = data[i];
      }
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
   // Can be called when Size() == 0, e.g. when an MPI-parallel vector has a
   // local size of 0.

   MFEM_ASSERT(result_type_id == ScalarId<double>::value, "");
   double *res = (double *)result;
   double local_dot = 0.;
   MFEM_ASSERT(dynamic_cast<const Vector *>(&x) != NULL, "invalid Vector type");
   const Vector *xp = static_cast<const Vector *>(&x);
   MFEM_ASSERT(this->Size() == xp->Size(), "");

   const double *ptr = GetData<double>();
   const double *xptr = xp->GetData<double>();
   const std::size_t size = Size();

   if (!ComputeOnDevice())
   {
      for (std::size_t i = 0; i < size; i++) local_dot += ptr[i] * xptr[i];
   }
   else
   {
#pragma omp target teams distribute parallel for map(to: ptr, xptr) reduction(+:local_dot)
      for (std::size_t i = 0; i < size; i++) local_dot += ptr[i] * xptr[i];
   }

   *res = local_dot;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = OmpLayout().OmpEngine().GetComm();
   if (comm != MPI_COMM_NULL)
   {
      MPI_Allreduce(&local_dot, res, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
#endif
}

void Vector::DoAxpby(const void *a, const PVector &x,
                     const void *b, const PVector &y,
                     int ab_type_id)
{
   // called only when Size() != 0

   MFEM_ASSERT(ab_type_id == ScalarId<double>::value, "");
   const double da = *static_cast<const double *>(a);
   const double db = *static_cast<const double *>(b);
   MFEM_ASSERT(da == 0.0 || dynamic_cast<const Vector *>(&x) != NULL,
               "invalid Vector x");
   MFEM_ASSERT(db == 0.0 || dynamic_cast<const Vector *>(&y) != NULL,
               "invalid Vector y");
   const Vector *xp = static_cast<const Vector *>(&x);
   const Vector *yp = static_cast<const Vector *>(&y);

   MFEM_ASSERT(da == 0.0 || this->Size() == xp->Size(), "");
   MFEM_ASSERT(db == 0.0 || this->Size() == yp->Size(), "");

   const std::size_t size = Size();
   const std::size_t critical_size = 1000;
   const double *xd = xp->GetData<double>();
   const double *yd = yp->GetData<double>();
   double *td = GetData<double>();

   const bool use_target = ComputeOnDevice();
   const bool use_parallel = (use_target || size > critical_size);

   if (da == 0.0)
   {
      if (db == 0.0)
      {
         OmpFill(&da);
      }
      else
      {
         if (td == yd)
         {
            // *this *= db
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: db)
            for (std::size_t i = 0; i < size; i++) td[i] *= db;
         }
         else
         {
            // *this = db * y
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: yd, db)
            for (std::size_t i = 0; i < size; i++) td[i] = yd[i] * db;
         }
      }
   }
   else
   {
      if (db == 0.0)
      {
         if (td == xd)
         {
            // *this *= da
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: da)
            for (std::size_t i = 0; i < size; i++) td[i] *= da;
         }
         else
         {
            // *this = da * x
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: xd, da)
            for (std::size_t i = 0; i < size; i++) td[i] = xd[i] * da;
         }
      }
      else
      {
         MFEM_ASSERT(xd != yd, "invalid input");
         if (td == xd)
         {
            // *this = da * (*this) + db * y
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: da, td, db, yd)
            for (std::size_t i = 0; i < size; i++) td[i] = da * td[i] + db * yd[i];
         }
         else if (td == yd)
         {
            // *this = da * x + db * (*this)
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: da, xd, db, td)
            for (std::size_t i = 0; i < size; i++) td[i] = da * xd[i] + db * td[i];
         }
         else
         {
            // *this = da * x + db * y
#pragma omp target teams distribute parallel for        \
   if (target: use_target)                              \
   if (parallel: use_parallel) map (to: da, xd, db, yd)
            for (std::size_t i = 0; i < size; i++) td[i] = da * xd[i] + db * yd[i];
         }
      }
   }
}

mfem::Vector Vector::Wrap()
{
   return mfem::Vector(*this);
}

const mfem::Vector Vector::Wrap() const
{
   return mfem::Vector(*const_cast<Vector*>(this));
}

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
