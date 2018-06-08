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

#include "../linalg/vector.hpp"
#include "../../../linalg/vector.hpp"
#include "../raja.hpp"

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
#ifdef MFEM_USE_MPI
   double local_dot = *res;
   if (IsParallel())
   {
      MPI_Allreduce(&local_dot, res, 1, MPI_DOUBLE, MPI_SUM, RajaLayout().RajaEngine().GetComm());
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
   
   /*for(size_t i=0;i<this->Size();i+=1){
      printf("\n\t\033[36m[DoAxpby] da=%f, db=%f this[%ld]=%f x:%f y:%f",da,db,i,
             ((double*)this->RajaMem().ptr())[i],
             ((double*)xp->RajaMem().ptr())[i],
             ((double*)yp->RajaMem().ptr())[i]);
             }*/
   
   if (da == 0.0)
   {
      if (db == 0.0)
      {
         RajaFill(&da);
      }
      else
      {
         if (this->slice == yp->slice)
         {
            // *this *= db
            assert(false);
            //raja::linalg::operator_mult_eq(slice, db);
         }
         else
         {
            // *this = db * y
            assert(false);/*
            raja::kernel kernel = axpby1_builder.build(slice.getDevice(),
                                                         okl_defines);
                                                         kernel((int)Size(), db, slice, yp->slice);*/
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
            assert(false);
            //raja::linalg::operator_mult_eq(slice, da);
         }
         else
         {
            // *this = da * x
            assert(false);
            /*
            ::raja::kernel kernel = axpby1_builder.build(slice.getDevice(),
                                                         okl_defines);
                                                         kernel((int)Size(), da, slice, xp->slice);*/
         }
      }
      else
      {
         //MFEM_ASSERT(xp->slice != yp->slice, "invalid input");
         if (this->slice == xp->slice)
         {
            dbg("\n[DoAxpby] 1");
            // *this = da * (*this) + db * y
            vector_axpby((int)Size(), da, db, slice, yp->slice);
         }
         else if (this->slice == yp->slice)
         {
            dbg("\n[DoAxpby] 2");
            // *this = da * x + db * (*this)
             vector_axpby((int)Size(), db, da, slice, xp->slice);
         }
         else
         {
            dbg("\n[DoAxpby] 3");
            // *this = da * x + db * y
            vector_axpby3((int)Size(), da, db, slice, xp->slice, yp->slice);
         }
      }
   }
   /*for(size_t i=0;i<this->Size();i+=1){
      printf("\n\t\033[36m[DoAxpby] da=%f, db=%f this[%ld]=%f x:%f y:%f",da,db,i,
             ((double*)this->RajaMem().ptr())[i],
             ((double*)xp->RajaMem().ptr())[i],
             ((double*)yp->RajaMem().ptr())[i]);
             }*/
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
