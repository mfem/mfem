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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

// *****************************************************************************
Vector::Vector(Layout &lt) : mfem::PArray(lt),
                             kernels::Array(lt, sizeof(double)),
                             mfem::PVector(lt)
{
   dbg("new kernels::Vector");
}

// *****************************************************************************
PVector *Vector::DoVectorClone(bool copy_data,
                               void **buffer,
                               int buffer_type_id) const
{
   push();
   MFEM_ASSERT(buffer_type_id == ScalarId<double>::value, "");
   Vector *new_vector = new Vector(KernelsLayout());
   if (copy_data)
   {
      new_vector->slice.copyFrom(slice);
   }
   if (buffer)
   {
      *buffer = new_vector->GetBuffer();
   }
   pop();
   return new_vector;
}

// *****************************************************************************
void Vector::DoDotProduct(const mfem::PVector &x,
                          void *result,
                          int result_type_id) const
{
   push();
   // called only when Size() != 0
   MFEM_ASSERT(result_type_id == ScalarId<double>::value, "");
   double *res = (double *)result;
   const Vector &xp = x.As<Vector>();
   MFEM_ASSERT(this->Size() == xp.Size(), "");
   *res = kernels::linalg::dot(this->slice, xp.slice);
#ifdef MFEM_USE_MPI
   double local_dot = *res;
   if (IsParallel())
   {
      MPI_Allreduce(&local_dot, res, 1, MPI_DOUBLE, MPI_SUM,
                    KernelsLayout().KernelsEngine().GetComm());
   }
   pop();
#endif
}

// *****************************************************************************
void Vector::DoAxpby(const void *a, const mfem::PVector &x,
                     const void *b, const mfem::PVector &y,
                     int ab_type_id)
{
   push();
   MFEM_ASSERT(ab_type_id == ScalarId<double>::value, "");
   const double da = *static_cast<const double *>(a);
   const double db = *static_cast<const double *>(b);
   MFEM_ASSERT(da == 0.0 ||
               dynamic_cast<const Vector *>(&x) != NULL,
               "\033[31minvalid Vector x\033[m");
   MFEM_ASSERT(db == 0.0 ||
               dynamic_cast<const Vector *>(&y) != NULL,
               "\033[31minvalid Vector y\033[m");
   const Vector *xp = static_cast<const Vector *>(&x);
   const Vector *yp = static_cast<const Vector *>(&y);
   MFEM_ASSERT(da == 0.0 || this->Size() == xp->Size(), "");
   MFEM_ASSERT(db == 0.0 || this->Size() == yp->Size(), "");
   if (da == 0.0)
   {
      if (db == 0.0)
      {
         KernelsFill(&da);
      }
      else
      {
         if (this->slice == yp->slice)
         {
            // *this *= db
            assert(false);
         }
         else
         {
            // *this = db * y
            assert(false);
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
            vector_vec_mul((int)Size(), slice, da);
         }
         else
         {
            // *this = da * x
            assert(false);
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
   pop();
}

// *****************************************************************************
void Vector::Print()
{
   for (size_t i=0; i<Size(); i+=1)
   {
      printf("%f ",((double*)KernelsMem().ptr())[i]);
   }
}

// *****************************************************************************
void Vector::SetSubVector(const mfem::Array<int> &ess_tdofs,
                          const double value,
                          const int N)
{
   push();
   vector_set_subvector_const(N, value, data, ess_tdofs.GetData());
   pop();
}
   
// *****************************************************************************
void Vector::MapSubVector(const mfem::Array<int> &ess_tdofs,
                          const kernels::Vector &v,
                          const int N)
{
   push();
   vector_map_dofs(N, data, v.data, ess_tdofs.GetData());
   pop();
}

// *****************************************************************************
mfem::Vector Vector::Wrap()
{
   dbg("Wrap");
   return mfem::Vector(*this);
}

// *****************************************************************************
const mfem::Vector Vector::Wrap() const
{
   dbg("const Wrap");
   return mfem::Vector(*const_cast<kernels::Vector*>(this));   
}

// *****************************************************************************
#if defined(MFEM_USE_MPI)
bool Vector::IsParallel() const
{
   dbg("IsParallel");
   return (KernelsLayout().KernelsEngine().GetComm() != MPI_COMM_NULL);
}
#endif

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
