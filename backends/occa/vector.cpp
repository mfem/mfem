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

#include "vector.hpp"
#include "../../linalg/vector.hpp"

namespace mfem
{

namespace occa
{

PVector *Vector::DoVectorClone(bool copy_data, void **buffer,
                               int buffer_type_id) const
{
   MFEM_ASSERT(buffer_type_id == ScalarId<double>::value, "");
   Vector *new_vector = new Vector(OccaLayout());
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
   // Can be called when Size() == 0, e.g. when an MPI-parallel vector has a
   // local size of 0.

   MFEM_ASSERT(result_type_id == ScalarId<double>::value, "");
   double *res = (double *)result;
   double local_dot = 0.;
   MFEM_ASSERT(dynamic_cast<const Vector *>(&x) != NULL, "invalid Vector type");
   const Vector *xp = static_cast<const Vector *>(&x);
   MFEM_ASSERT(this->Size() == xp->Size(), "");
   local_dot = ::occa::linalg::dot<double, double, double>(this->slice, xp->slice);

   *res = local_dot;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = OccaLayout().OccaEngine().GetComm();
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
   const std::string &okl_defines = OccaLayout().OccaEngine().GetOklDefines();

   //
   // TODO: move all kernel builders to class mfem::occa::Backend
   //
   static ::occa::kernelBuilder axpby1_builder =
      ::occa::linalg::customLinearMethod(
         "mfem_occa_axpby1",
         "v0[i] = c0 * v1[i];",
         "defines: {"
         "  CTYPE0: 'double',"
         "  VTYPE0: 'double',"
         "  VTYPE1: 'double',"
         "  TILESIZE: '128',"
         "}");

   static ::occa::kernelBuilder axpby2_builder =
      ::occa::linalg::customLinearMethod(
         "mfem_occa_axpby2",
         "v0[i] = c0 * v0[i] + c1 * v1[i];",
         "defines: {"
         "  CTYPE0: 'double',"
         "  CTYPE1: 'double',"
         "  VTYPE0: 'double',"
         "  VTYPE1: 'double',"
         "  TILESIZE: '128',"
         "}");

   static ::occa::kernelBuilder axpby3_builder =
      ::occa::linalg::customLinearMethod(
         "mfem_occa_axpby3",
         "v0[i] = c0 * v1[i] + c1 * v2[i];",
         "defines: {"
         "  CTYPE0: 'double',"
         "  CTYPE1: 'double',"
         "  VTYPE0: 'double',"
         "  VTYPE1: 'double',"
         "  VTYPE2: 'double',"
         "  TILESIZE: '128',"
         "}");

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

   if (da == 0.0)
   {
      if (db == 0.0)
      {
         OccaFill(&da);
      }
      else
      {
         if (this->slice == yp->slice)
         {
            // *this *= db
            ::occa::linalg::operator_mult_eq(slice, db);
         }
         else
         {
            // *this = db * y
            ::occa::kernel kernel = axpby1_builder.build(slice.getDevice(),
                                                         okl_defines);
            kernel((int)Size(), db, slice, yp->slice);
         }
      }
   }
   else
   {
      if (db == 0.0)
      {
         if (this->slice == xp->slice)
         {
            // *this *= da
            ::occa::linalg::operator_mult_eq(slice, da);
         }
         else
         {
            // *this = da * x
            ::occa::kernel kernel = axpby1_builder.build(slice.getDevice(),
                                                         okl_defines);
            kernel((int)Size(), da, slice, xp->slice);
         }
      }
      else
      {
         MFEM_ASSERT(xp->slice != yp->slice, "invalid input");
         if (this->slice == xp->slice)
         {
            // *this = da * (*this) + db * y
            ::occa::kernel kernel = axpby2_builder.build(slice.getDevice(),
                                                         okl_defines);
            kernel((int)Size(), da, db, slice, yp->slice);
         }
         else if (this->slice == yp->slice)
         {
            // *this = da * x + db * (*this)
            ::occa::kernel kernel = axpby2_builder.build(slice.getDevice(),
                                                         okl_defines);
            kernel((int)Size(), db, da, slice, xp->slice);
         }
         else
         {
            // *this = da * x + db * y
            ::occa::kernel kernel = axpby3_builder.build(slice.getDevice(),
                                                         okl_defines);
            kernel((int)Size(), da, db, slice, xp->slice, yp->slice);
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

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)
