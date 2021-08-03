// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "linalg.hpp"
#include "../fem/fem.hpp"
#include "../general/forall.hpp"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#ifdef MFEM_USE_SUNDIALS
#include <nvector/nvector_parallel.h>
#endif

using namespace std;

namespace mfem
{

template<typename TargetT, typename SourceT>
static TargetT *DuplicateAs(const SourceT *array, int size,
                            bool cplusplus = true)
{
   TargetT *target_array = cplusplus ? (TargetT*) Memory<TargetT>(size)
                           /*     */ : mfem_hypre_TAlloc_host(TargetT, size);
   for (int i = 0; i < size; i++)
   {
      target_array[i] = array[i];
   }
   return target_array;
}


/// Return true if the @a src Memory can be used with the MemoryClass @a mc.
/** If this function returns true then src.{Read,Write,ReadWrite} can be called
    safely with the MemoryClass @a mc. */
template <typename T>
bool CanShallowCopy(const Memory<T> &src, MemoryClass mc)
{
   MemoryType src_h_mt = src.GetHostMemoryType();
   MemoryType src_d_mt = src.GetDeviceMemoryType();
   if (src_d_mt == MemoryType::DEFAULT)
   {
      src_d_mt = MemoryManager::GetDualMemoryType(src_h_mt);
   }
   return (MemoryClassContainsType(mc, src_h_mt) ||
           MemoryClassContainsType(mc, src_d_mt));
}


inline void HypreParVector::_SetDataAndSize_()
{
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
#ifndef HYPRE_USING_CUDA
   SetDataAndSize(hypre_VectorData(x_loc),
                  internal::to_int(hypre_VectorSize(x_loc)));
#else
   size = internal::to_int(hypre_VectorSize(x_loc));
   MemoryType mt = (hypre_VectorMemoryLocation(x_loc) == HYPRE_MEMORY_HOST
                    ? MemoryType::HOST : GetHypreMemoryType());
   if (hypre_VectorData(x_loc) != NULL)
   {
      data.Wrap(hypre_VectorData(x_loc), size, mt, false);
   }
   else
   {
      data.Reset();
   }
#endif
}

HypreParVector::HypreParVector(MPI_Comm comm, HYPRE_BigInt glob_size,
                               HYPRE_BigInt *col) : Vector()
{
   x = hypre_ParVectorCreate(comm,glob_size,col);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(MPI_Comm comm, HYPRE_BigInt glob_size,
                               double *data_, HYPRE_BigInt *col,
                               bool is_device_ptr)
   : Vector()
{
   x = hypre_ParVectorCreate(comm,glob_size,col);
   hypre_ParVectorSetDataOwner(x,1); // owns the seq vector
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_SeqVectorSetDataOwner(x_loc,0);
   hypre_ParVectorSetPartitioningOwner(x,0);
   double tmp = 0.0;
   hypre_VectorData(x_loc) = &tmp;
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) =
      is_device_ptr ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
#else
   (void)is_device_ptr;
#endif
   // If hypre_ParVectorLocalVector(x) and &tmp are non-NULL,
   // hypre_ParVectorInitialize(x) does not allocate memory!
   hypre_ParVectorInitialize(x);
   // Set the internal data array to the one passed in
   hypre_VectorData(x_loc) = data_;
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParVector &y) : Vector()
{
   x = hypre_ParVectorCreate(y.x -> comm, y.x -> global_size,
                             y.x -> partitioning);
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(const HypreParMatrix &A,
                               int transpose) : Vector()
{
   if (!transpose)
   {
      x = hypre_ParVectorInDomainOf(const_cast<HypreParMatrix&>(A));
   }
   else
   {
      x = hypre_ParVectorInRangeOf(const_cast<HypreParMatrix&>(A));
   }
   _SetDataAndSize_();
   own_ParVector = 1;
}

HypreParVector::HypreParVector(HYPRE_ParVector y) : Vector()
{
   x = (hypre_ParVector *) y;
   _SetDataAndSize_();
   own_ParVector = 0;
}

HypreParVector::HypreParVector(ParFiniteElementSpace *pfes)
{
   x = hypre_ParVectorCreate(pfes->GetComm(), pfes->GlobalTrueVSize(),
                             pfes->GetTrueDofOffsets());
   hypre_ParVectorInitialize(x);
   hypre_ParVectorSetPartitioningOwner(x,0);
   // The data will be destroyed by hypre (this is the default)
   hypre_ParVectorSetDataOwner(x,1);
   hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x),1);
   _SetDataAndSize_();
   own_ParVector = 1;
}

void HypreParVector::WrapHypreParVector(hypre_ParVector *y, bool owner)
{
   if (own_ParVector) { hypre_ParVectorDestroy(x); }
   Destroy();
   x = y;
   _SetDataAndSize_();
   own_ParVector = owner;
}

Vector * HypreParVector::GlobalVector() const
{
   hypre_Vector *hv = hypre_ParVectorToVectorAll(*this);
   Vector *v = new Vector(hv->data, internal::to_int(hv->size));
   v->MakeDataOwner();
   hypre_SeqVectorSetDataOwner(hv,0);
   hypre_SeqVectorDestroy(hv);
   return v;
}

HypreParVector& HypreParVector::operator=(double d)
{
   Vector::operator=(d);
   return *this;
}

HypreParVector& HypreParVector::operator=(const HypreParVector &y)
{
#ifdef MFEM_DEBUG
   if (size != y.Size())
   {
      mfem_error("HypreParVector::operator=");
   }
#endif

   Vector::operator=(y);
   return *this;
}

void HypreParVector::SetData(double *data_)
{
   hypre_VectorData(hypre_ParVectorLocalVector(x)) = data_;
   Vector::SetData(data_);
}

void HypreParVector::HypreRead() const
{
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) =
      const_cast<double*>(data.Read(GetHypreMemoryClass(), size));
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
}

void HypreParVector::HypreReadWrite()
{
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) = data.ReadWrite(GetHypreMemoryClass(), size);
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
}

void HypreParVector::HypreWrite()
{
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) = data.Write(GetHypreMemoryClass(), size);
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
}

void HypreParVector::WrapMemoryRead(const Memory<double> &mem)
{
   MFEM_ASSERT(CanShallowCopy(mem, GetHypreMemoryClass()), "");
   MFEM_ASSERT(mem.Capacity() >= size, "");

   data.Delete();
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) =
      const_cast<double*>(mem.Read(GetHypreMemoryClass(), size));
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
   data.MakeAlias(mem, 0, size);
}

void HypreParVector::WrapMemoryReadWrite(Memory<double> &mem)
{
   MFEM_ASSERT(CanShallowCopy(mem, GetHypreMemoryClass()), "");
   MFEM_ASSERT(mem.Capacity() >= size, "");

   data.Delete();
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) = mem.ReadWrite(GetHypreMemoryClass(), size);
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
   data.MakeAlias(mem, 0, size);
}

void HypreParVector::WrapMemoryWrite(Memory<double> &mem)
{
   MFEM_ASSERT(CanShallowCopy(mem, GetHypreMemoryClass()), "");
   MFEM_ASSERT(mem.Capacity() >= size, "");

   data.Delete();
   hypre_Vector *x_loc = hypre_ParVectorLocalVector(x);
   hypre_VectorData(x_loc) = mem.Write(GetHypreMemoryClass(), size);
#ifdef HYPRE_USING_CUDA
   hypre_VectorMemoryLocation(x_loc) = HYPRE_MEMORY_DEVICE;
#endif
   data.MakeAlias(mem, 0, size);
}

HYPRE_Int HypreParVector::Randomize(HYPRE_Int seed)
{
   return hypre_ParVectorSetRandomValues(x,seed);
}

void HypreParVector::Print(const char *fname) const
{
   hypre_ParVectorPrint(x,fname);
}

HypreParVector::~HypreParVector()
{
   if (own_ParVector)
   {
      hypre_ParVectorDestroy(x);
   }
}

#ifdef MFEM_USE_SUNDIALS

N_Vector HypreParVector::ToNVector()
{
   return N_VMake_Parallel(GetComm(), Size(), GlobalSize(), GetData());
}

#endif // MFEM_USE_SUNDIALS


double InnerProduct(HypreParVector *x, HypreParVector *y)
{
   return hypre_ParVectorInnerProd(*x, *y);
}

double InnerProduct(HypreParVector &x, HypreParVector &y)
{
   return hypre_ParVectorInnerProd(x, y);
}


double ParNormlp(const Vector &vec, double p, MPI_Comm comm)
{
   double norm = 0.0;
   if (p == 1.0)
   {
      double loc_norm = vec.Norml1();
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
   if (p == 2.0)
   {
      double loc_norm = vec*vec;
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
      norm = sqrt(norm);
   }
   if (p < infinity())
   {
      double sum = 0.0;
      for (int i = 0; i < vec.Size(); i++)
      {
         sum += pow(fabs(vec(i)), p);
      }
      MPI_Allreduce(&sum, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
      norm = pow(norm, 1.0/p);
   }
   else
   {
      double loc_norm = vec.Normlinf();
      MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_MAX, comm);
   }
   return norm;
}

/** @brief Shallow or deep copy @a src to @a dst with the goal to make the
    array @a src accessible through @a dst with the MemoryClass @a dst_mc. If
    one of the host/device MemoryType%s of @a src is contained in @a dst_mc,
    then a shallow copy will be used and @a dst will simply be an alias of
    @a src. Otherwise, @a dst will be properly allocated and @a src will be deep
    copied to @a dst. */
/** If @a dst_owner is set to true and shallow copy is being used, then @a dst
    will not be an alias of @a src; instead, @a src is copied to @a dst and all
    ownership flags of @a src are reset.

    In both cases (deep or shallow copy), when @a dst is no longer needed,
    dst.Delete() must be called to ensure all associated memory allocations are
    freed.

    The input contents of @a dst, if any, is not used and it is overwritten by
    this function. In particular, @a dst should be empty or deleted before
    calling this function. */
template <typename T>
void CopyMemory(Memory<T> &src, Memory<T> &dst, MemoryClass dst_mc,
                bool dst_owner)
{
   if (CanShallowCopy(src, dst_mc))
   {
      // shallow copy
      if (!dst_owner)
      {
         src.Read(dst_mc, src.Capacity());  // Registers src if on host only
         dst.MakeAlias(src, 0, src.Capacity());
      }
      else
      {
         dst = src;
         src.ClearOwnerFlags();
      }
   }
   else
   {
      // deep copy
      dst.New(src.Capacity(), GetMemoryType(dst_mc));
      dst.CopyFrom(src, src.Capacity());
   }
}

/** @brief Deep copy and convert @a src to @a dst with the goal to make the
    array @a src accessible through @a dst with the MemoryClass @a dst_mc and
    convert it from type SrcT to type DstT. */
/** When @a dst is no longer needed, dst.Delete() must be called to ensure all
    associated memory allocations are freed.

    The input contents of @a dst, if any, is not used and it is overwritten by
    this function. In particular, @a dst should be empty or deleted before
    calling this function. */
template <typename SrcT, typename DstT>
void CopyConvertMemory(Memory<SrcT> &src, MemoryClass dst_mc, Memory<DstT> &dst)
{
   auto capacity = src.Capacity();
   dst.New(capacity, GetMemoryType(dst_mc));
   // Perform the copy using the configured mfem Device
   auto src_p = mfem::Read(src, capacity);
   auto dst_p = mfem::Write(dst, capacity);
   MFEM_FORALL(i, capacity, dst_p[i] = src_p[i];);
}


void HypreParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   auxX.Reset(); auxY.Reset();
   diagOwner = offdOwner = colMapOwner = -1;
   ParCSROwner = 1;
   mem_diag.I.Reset();
   mem_diag.J.Reset();
   mem_diag.data.Reset();
   mem_offd.I.Reset();
   mem_offd.J.Reset();
   mem_offd.data.Reset();
}

void HypreParMatrix::Read(MemoryClass mc) const
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   const int num_rows = NumRows();
   const int diag_nnz = internal::to_int(diag->num_nonzeros);
   const int offd_nnz = internal::to_int(offd->num_nonzeros);
   diag->i = const_cast<HYPRE_Int*>(mem_diag.I.Read(mc, num_rows+1));
   diag->j = const_cast<HYPRE_Int*>(mem_diag.J.Read(mc, diag_nnz));
   diag->data = const_cast<double*>(mem_diag.data.Read(mc, diag_nnz));
   offd->i = const_cast<HYPRE_Int*>(mem_offd.I.Read(mc, num_rows+1));
   offd->j = const_cast<HYPRE_Int*>(mem_offd.J.Read(mc, offd_nnz));
   offd->data = const_cast<double*>(mem_offd.data.Read(mc, offd_nnz));
#if MFEM_HYPRE_VERSION >= 21800
   decltype(diag->memory_location) ml =
      (mc != GetHypreMemoryClass() ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE);
   diag->memory_location = ml;
   offd->memory_location = ml;
#endif
}

void HypreParMatrix::ReadWrite(MemoryClass mc)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   const int num_rows = NumRows();
   const int diag_nnz = internal::to_int(diag->num_nonzeros);
   const int offd_nnz = internal::to_int(offd->num_nonzeros);
   diag->i = mem_diag.I.ReadWrite(mc, num_rows+1);
   diag->j = mem_diag.J.ReadWrite(mc, diag_nnz);
   diag->data = mem_diag.data.ReadWrite(mc, diag_nnz);
   offd->i = mem_offd.I.ReadWrite(mc, num_rows+1);
   offd->j = mem_offd.J.ReadWrite(mc, offd_nnz);
   offd->data = mem_offd.data.ReadWrite(mc, offd_nnz);
#if MFEM_HYPRE_VERSION >= 21800
   decltype(diag->memory_location) ml =
      (mc != GetHypreMemoryClass() ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE);
   diag->memory_location = ml;
   offd->memory_location = ml;
#endif
}

void HypreParMatrix::Write(MemoryClass mc, bool set_diag, bool set_offd)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   if (set_diag)
   {
      diag->i = mem_diag.I.Write(mc, mem_diag.I.Capacity());
      diag->j = mem_diag.J.Write(mc, mem_diag.J.Capacity());
      diag->data = mem_diag.data.Write(mc, mem_diag.data.Capacity());
   }
   if (set_offd)
   {
      offd->i = mem_offd.I.Write(mc, mem_offd.I.Capacity());
      offd->j = mem_offd.J.Write(mc, mem_offd.J.Capacity());
      offd->data = mem_offd.data.Write(mc, mem_offd.data.Capacity());
   }
#if MFEM_HYPRE_VERSION >= 21800
   decltype(diag->memory_location) ml =
      (mc != GetHypreMemoryClass() ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE);
   if (set_diag) { diag->memory_location = ml; }
   if (set_offd) { offd->memory_location = ml; }
#endif
}

HypreParMatrix::HypreParMatrix()
{
   Init();
   height = width = 0;
}

void HypreParMatrix::WrapHypreParCSRMatrix(hypre_ParCSRMatrix *a, bool owner)
{
   Destroy();
   Init();
   A = a;
   ParCSROwner = owner;
   height = GetNumRows();
   width = GetNumCols();
#if MFEM_HYPRE_VERSION >= 21800
   MemoryType diag_mt = (A->diag->memory_location == HYPRE_MEMORY_HOST
                         ? MemoryType::HOST : GetHypreMemoryType());
   MemoryType offd_mt = (A->offd->memory_location == HYPRE_MEMORY_HOST
                         ? MemoryType::HOST : GetHypreMemoryType());
#else
   const MemoryType diag_mt = MemoryType::HOST;
   const MemoryType offd_mt = MemoryType::HOST;
#endif
   diagOwner = HypreCsrToMem(A->diag, diag_mt, false, mem_diag);
   offdOwner = HypreCsrToMem(A->offd, offd_mt, false, mem_offd);
   HypreRead();
}

signed char HypreParMatrix::CopyCSR(SparseMatrix *csr,
                                    MemoryIJData &mem_csr,
                                    hypre_CSRMatrix *hypre_csr,
                                    bool mem_owner)
{
   const MemoryClass hypre_mc = GetHypreMemoryClass();
#ifndef HYPRE_BIGINT
   // code for the case HYPRE_Int == int
   CopyMemory(csr->GetMemoryI(), mem_csr.I, hypre_mc, mem_owner);
   CopyMemory(csr->GetMemoryJ(), mem_csr.J, hypre_mc, mem_owner);
#else
   // code for the case HYPRE_Int == long long int
   CopyConvertMemory(csr->GetMemoryI(), hypre_mc, mem_csr.I);
   CopyConvertMemory(csr->GetMemoryJ(), hypre_mc, mem_csr.J);
#endif
   CopyMemory(csr->GetMemoryData(), mem_csr.data, hypre_mc, mem_owner);

   const int num_rows = csr->Height();
   const int nnz = csr->NumNonZeroElems();
   hypre_csr->i = const_cast<HYPRE_Int*>(mem_csr.I.Read(hypre_mc, num_rows+1));
   hypre_csr->j = const_cast<HYPRE_Int*>(mem_csr.J.Read(hypre_mc, nnz));
   hypre_csr->data = const_cast<double*>(mem_csr.data.Read(hypre_mc, nnz));

   MFEM_ASSERT(mem_csr.I.OwnsHostPtr() == mem_csr.J.OwnsHostPtr(),
               "invalid state: host ownership for I and J differ!");
   return (mem_csr.I.OwnsHostPtr()    ? 1 : 0) +
          (mem_csr.data.OwnsHostPtr() ? 2 : 0);
}

signed char HypreParMatrix::CopyBoolCSR(Table *bool_csr,
                                        MemoryIJData &mem_csr,
                                        hypre_CSRMatrix *hypre_csr)
{
   const MemoryClass hypre_mc = GetHypreMemoryClass();
#ifndef HYPRE_BIGINT
   // code for the case HYPRE_Int == int
   CopyMemory(bool_csr->GetIMemory(), mem_csr.I, hypre_mc, false);
   CopyMemory(bool_csr->GetJMemory(), mem_csr.J, hypre_mc, false);
#else
   // code for the case HYPRE_Int == long long int
   CopyConvertMemory(bool_csr->GetIMemory(), hypre_mc, mem_csr.I);
   CopyConvertMemory(bool_csr->GetJMemory(), hypre_mc, mem_csr.J);
#endif
   const int num_rows = bool_csr->Size();
   const int nnz = bool_csr->Size_of_connections();
   mem_csr.data.New(nnz, GetHypreMemoryType());
   double *data = mfem::HostWrite(mem_csr.data, nnz);
   for (int i = 0; i < nnz; i++)
   {
      data[i] = 1.0;
   }
   hypre_csr->i = const_cast<HYPRE_Int*>(mem_csr.I.Read(hypre_mc, num_rows+1));
   hypre_csr->j = const_cast<HYPRE_Int*>(mem_csr.J.Read(hypre_mc, nnz));
   hypre_csr->data = const_cast<double*>(mem_csr.data.Read(hypre_mc, nnz));

   MFEM_ASSERT(mem_csr.I.OwnsHostPtr() == mem_csr.J.OwnsHostPtr(),
               "invalid state: host ownership for I and J differ!");
   return (mem_csr.I.OwnsHostPtr()    ? 1 : 0) +
          (mem_csr.data.OwnsHostPtr() ? 2 : 0);
}

void HypreParMatrix::CopyCSR_J(hypre_CSRMatrix *hypre_csr, int *J)
{
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(hypre_csr);
#if MFEM_HYPRE_VERSION >= 21600
   if (hypre_CSRMatrixBigJ(hypre_csr))
   {
      for (HYPRE_Int j = 0; j < nnz; j++)
      {
         J[j] = internal::to_int(hypre_CSRMatrixBigJ(hypre_csr)[j]);
      }
      return;
   }
#endif
   for (HYPRE_Int j = 0; j < nnz; j++)
   {
      J[j] = internal::to_int(hypre_CSRMatrixJ(hypre_csr)[j]);
   }
}

// static method
signed char HypreParMatrix::HypreCsrToMem(hypre_CSRMatrix *h_mat,
                                          MemoryType h_mat_mt,
                                          bool own_ija,
                                          MemoryIJData &mem)
{
   const int nr1 = internal::to_int(h_mat->num_rows) + 1;
   const int nnz = internal::to_int(h_mat->num_nonzeros);
   mem.I.Wrap(h_mat->i, nr1, h_mat_mt, own_ija);
   mem.J.Wrap(h_mat->j, nnz, h_mat_mt, own_ija);
   mem.data.Wrap(h_mat->data, nnz, h_mat_mt, own_ija);
   const MemoryClass hypre_mc = GetHypreMemoryClass();
   if (!CanShallowCopy(mem.I, hypre_mc))
   {
      const MemoryType hypre_mt = GetHypreMemoryType();
      MemoryIJData h_mem;
      h_mem.I.New(nr1, hypre_mt);
      h_mem.I.CopyFrom(mem.I, nr1);
      mem.I.Delete();
      h_mem.J.New(nnz, hypre_mt);
      h_mem.J.CopyFrom(mem.J, nnz);
      mem.J.Delete();
      h_mem.data.New(nnz, hypre_mt);
      h_mem.data.CopyFrom(mem.data, nnz);
      mem.data.Delete();
      mem = h_mem;
      if (!own_ija)
      {
         // FIXME: Even if own_ija == false, it does not necessarily mean we
         // need to delete h_mat->{i,j,data} even if h_mat->owns_data == true.

         // h_mat owns i; owns j,data if h_mat->owns_data
#if MFEM_HYPRE_VERSION < 21400
         hypre_TFree(h_mat->i);
#elif MFEM_HYPRE_VERSION < 21800
         hypre_TFree(h_mat->i, HYPRE_MEMORY_SHARED);
#else
         hypre_TFree(h_mat->i, h_mat->memory_location);
#endif
         if (h_mat->owns_data)
         {
#if MFEM_HYPRE_VERSION < 21400
            hypre_TFree(h_mat->j);
            hypre_TFree(h_mat->data);
#elif MFEM_HYPRE_VERSION < 21800
            hypre_TFree(h_mat->j, HYPRE_MEMORY_SHARED);
            hypre_TFree(h_mat->data, HYPRE_MEMORY_SHARED);
#else
            hypre_TFree(h_mat->j, h_mat->memory_location);
            hypre_TFree(h_mat->data, h_mat->memory_location);
#endif
         }
      }
      h_mat->i = mem.I.ReadWrite(hypre_mc, nr1);
      h_mat->j = mem.J.ReadWrite(hypre_mc, nnz);
      h_mat->data = mem.data.ReadWrite(hypre_mc, nnz);
      h_mat->owns_data = 0;
#if MFEM_HYPRE_VERSION >= 21800
      h_mat->memory_location = HYPRE_MEMORY_DEVICE;
#endif
      return 3;
   }
   return own_ija ? 3 : (h_mat_mt == GetHypreMemoryType() ? -2 : -1);
}

// Square block-diagonal constructor (4 arguments, v1)
HypreParMatrix::HypreParMatrix(MPI_Comm comm, HYPRE_BigInt glob_size,
                               HYPRE_BigInt *row_starts, SparseMatrix *diag)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, glob_size, glob_size, row_starts,
                                row_starts, 0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, mem_diag, A->diag, false);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = mfem_hypre_CTAlloc(HYPRE_Int, diag->Height()+1);
   offdOwner = HypreCsrToMem(A->offd, GetHypreMemoryType(), false, mem_offd);

   /* Don't need to call these, since they allocate memory only
      if it was not already allocated */
   // hypre_CSRMatrixInitialize(A->diag);
   // hypre_ParCSRMatrixInitialize(A);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));

   // FIXME:
#ifdef HYPRE_BIGINT
   CopyCSR_J(A->diag, diag->GetJ());
#endif

   hypre_MatvecCommPkgCreate(A);
}

// Rectangular block-diagonal constructor (6 arguments, v1)
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_BigInt global_num_rows,
                               HYPRE_BigInt global_num_cols,
                               HYPRE_BigInt *row_starts,
                               HYPRE_BigInt *col_starts,
                               SparseMatrix *diag)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts,
                                0, diag->NumNonZeroElems(), 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, mem_diag, A->diag, false);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = mfem_hypre_CTAlloc(HYPRE_Int, diag->Height()+1);
   offdOwner = HypreCsrToMem(A->offd, GetHypreMemoryType(), false, mem_offd);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
      // FIXME:
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);
}

// General rectangular constructor with diagonal and off-diagonal (8+1
// arguments)
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_BigInt global_num_rows,
                               HYPRE_BigInt global_num_cols,
                               HYPRE_BigInt *row_starts,
                               HYPRE_BigInt *col_starts,
                               SparseMatrix *diag, SparseMatrix *offd,
                               HYPRE_BigInt *cmap,
                               bool own_diag_offd)
   : Operator(diag->Height(), diag->Width())
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts,
                                offd->Width(), diag->NumNonZeroElems(),
                                offd->NumNonZeroElems());
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyCSR(diag, mem_diag, A->diag, own_diag_offd);
   if (own_diag_offd) { delete diag; }
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   offdOwner = CopyCSR(offd, mem_offd, A->offd, own_diag_offd);
   if (own_diag_offd) { delete offd; }
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;
   // Prevent hypre from destroying A->col_map_offd
   colMapOwner = 0;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
      // FIXME:
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);
}

// General rectangular constructor with diagonal and off-diagonal (13+1
// arguments)
HypreParMatrix::HypreParMatrix(
   MPI_Comm comm,
   HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols,
   HYPRE_BigInt *row_starts, HYPRE_BigInt *col_starts,
   HYPRE_Int *diag_i, HYPRE_Int *diag_j, double *diag_data,
   HYPRE_Int *offd_i, HYPRE_Int *offd_j, double *offd_data,
   HYPRE_Int offd_num_cols, HYPRE_BigInt *offd_col_map,
   bool hypre_arrays)
{
   Init();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, offd_num_cols, 0, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   HYPRE_Int local_num_rows = hypre_CSRMatrixNumRows(A->diag);

   hypre_CSRMatrixSetDataOwner(A->diag, hypre_arrays);
   hypre_CSRMatrixI(A->diag) = diag_i;
   hypre_CSRMatrixJ(A->diag) = diag_j;
   hypre_CSRMatrixData(A->diag) = diag_data;
   hypre_CSRMatrixNumNonzeros(A->diag) = diag_i[local_num_rows];
#ifdef HYPRE_USING_CUDA
   hypre_CSRMatrixMemoryLocation(A->diag) = HYPRE_MEMORY_HOST;
#endif
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd, hypre_arrays);
   hypre_CSRMatrixI(A->offd) = offd_i;
   hypre_CSRMatrixJ(A->offd) = offd_j;
   hypre_CSRMatrixData(A->offd) = offd_data;
   hypre_CSRMatrixNumNonzeros(A->offd) = offd_i[local_num_rows];
#ifdef HYPRE_USING_CUDA
   hypre_CSRMatrixMemoryLocation(A->offd) = HYPRE_MEMORY_HOST;
#endif
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = offd_col_map;
   // Prevent hypre from destroying A->col_map_offd, own A->col_map_offd
   colMapOwner = hypre_arrays ? -1 : 1;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();

   if (!hypre_arrays)
   {
      const MemoryType host_mt = Device::GetHostMemoryType();
      diagOwner = HypreCsrToMem(A->diag, host_mt, true, mem_diag);
      offdOwner = HypreCsrToMem(A->offd, host_mt, true, mem_offd);
   }
   else
   {
      const MemoryType host_mt = MemoryType::HOST;
      diagOwner = HypreCsrToMem(A->diag, host_mt, false, mem_diag);
      offdOwner = HypreCsrToMem(A->offd, host_mt, false, mem_offd);
   }
   HypreRead();
}

// Constructor from a CSR matrix on rank 0 (4 arguments, v2)
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_BigInt *row_starts,
                               HYPRE_BigInt *col_starts,
                               SparseMatrix *sm_a)
{
   MFEM_ASSERT(sm_a != NULL, "invalid input");
   MFEM_VERIFY(!HYPRE_AssumedPartitionCheck(),
               "this method can not be used with assumed partition");

   Init();

   hypre_CSRMatrix *csr_a;
   csr_a = hypre_CSRMatrixCreate(sm_a -> Height(), sm_a -> Width(),
                                 sm_a -> NumNonZeroElems());

   hypre_CSRMatrixSetDataOwner(csr_a,0);
   MemoryIJData mem_a;
   CopyCSR(sm_a, mem_a, csr_a, false);
   hypre_CSRMatrixSetRownnz(csr_a);

   // NOTE: this call creates a matrix on host even when device support is
   // enabled in hypre.
   hypre_ParCSRMatrix *new_A =
      hypre_CSRMatrixToParCSRMatrix(comm, csr_a, row_starts, col_starts);

   mem_a.I.Delete();
   mem_a.J.Delete();
   mem_a.data.Delete();

   hypre_CSRMatrixI(csr_a) = NULL;
   hypre_CSRMatrixDestroy(csr_a);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(new_A));
   }

   hypre_MatvecCommPkgCreate(A);

   WrapHypreParCSRMatrix(new_A);
}

// Boolean, rectangular, block-diagonal constructor (6 arguments, v2)
HypreParMatrix::HypreParMatrix(MPI_Comm comm,
                               HYPRE_BigInt global_num_rows,
                               HYPRE_BigInt global_num_cols,
                               HYPRE_BigInt *row_starts,
                               HYPRE_BigInt *col_starts,
                               Table *diag)
{
   Init();
   int nnz = diag->Size_of_connections();
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, 0, nnz, 0);
   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   diagOwner = CopyBoolCSR(diag, mem_diag, A->diag);
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,1);
   hypre_CSRMatrixI(A->offd) = mfem_hypre_CTAlloc(HYPRE_Int, diag->Size()+1);
   offdOwner = HypreCsrToMem(A->offd, GetHypreMemoryType(), false, mem_offd);

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
      // FIXME:
#ifdef HYPRE_BIGINT
      CopyCSR_J(A->diag, diag->GetJ());
#endif
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

// Boolean, general rectangular constructor with diagonal and off-diagonal
// (11 arguments)
HypreParMatrix::HypreParMatrix(MPI_Comm comm, int id, int np,
                               HYPRE_BigInt *row, HYPRE_BigInt *col,
                               HYPRE_Int *i_diag, HYPRE_Int *j_diag,
                               HYPRE_Int *i_offd, HYPRE_Int *j_offd,
                               HYPRE_BigInt *cmap, HYPRE_Int cmap_size)
{
   HYPRE_Int diag_nnz, offd_nnz;

   Init();
   if (HYPRE_AssumedPartitionCheck())
   {
      diag_nnz = i_diag[row[1]-row[0]];
      offd_nnz = i_offd[row[1]-row[0]];

      A = hypre_ParCSRMatrixCreate(comm, row[2], col[2], row, col,
                                   cmap_size, diag_nnz, offd_nnz);
   }
   else
   {
      diag_nnz = i_diag[row[id+1]-row[id]];
      offd_nnz = i_offd[row[id+1]-row[id]];

      A = hypre_ParCSRMatrixCreate(comm, row[np], col[np], row, col,
                                   cmap_size, diag_nnz, offd_nnz);
   }

   hypre_ParCSRMatrixSetDataOwner(A,1);
   hypre_ParCSRMatrixSetRowStartsOwner(A,0);
   hypre_ParCSRMatrixSetColStartsOwner(A,0);

   mem_diag.data.New(diag_nnz);
   for (HYPRE_Int i = 0; i < diag_nnz; i++)
   {
      mem_diag.data[i] = 1.0;
   }

   mem_offd.data.New(offd_nnz);
   for (HYPRE_Int i = 0; i < offd_nnz; i++)
   {
      mem_offd.data[i] = 1.0;
   }

   hypre_CSRMatrixSetDataOwner(A->diag,0);
   hypre_CSRMatrixI(A->diag)    = i_diag;
   hypre_CSRMatrixJ(A->diag)    = j_diag;
   hypre_CSRMatrixData(A->diag) = mem_diag.data;
#ifdef HYPRE_USING_CUDA
   hypre_CSRMatrixMemoryLocation(A->diag) = HYPRE_MEMORY_HOST;
#endif
   hypre_CSRMatrixSetRownnz(A->diag);

   hypre_CSRMatrixSetDataOwner(A->offd,0);
   hypre_CSRMatrixI(A->offd)    = i_offd;
   hypre_CSRMatrixJ(A->offd)    = j_offd;
   hypre_CSRMatrixData(A->offd) = mem_offd.data;
#ifdef HYPRE_USING_CUDA
   hypre_CSRMatrixMemoryLocation(A->offd) = HYPRE_MEMORY_HOST;
#endif
   hypre_CSRMatrixSetRownnz(A->offd);

   hypre_ParCSRMatrixColMapOffd(A) = cmap;
   // Prevent hypre from destroying A->col_map_offd, own A->col_map_offd
   colMapOwner = 1;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   /* Make sure that the first entry in each row is the diagonal one. */
   if (row == col)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();

   const MemoryType host_mt = Device::GetHostMemoryType();
   diagOwner = HypreCsrToMem(A->diag, host_mt, true, mem_diag);
   offdOwner = HypreCsrToMem(A->offd, host_mt, true, mem_offd);
   HypreRead();
}

// General rectangular constructor with diagonal and off-diagonal constructed
// from a CSR matrix that contains both diagonal and off-diagonal blocks
// (9 arguments)
HypreParMatrix::HypreParMatrix(MPI_Comm comm, int nrows,
                               HYPRE_BigInt glob_nrows,
                               HYPRE_BigInt glob_ncols,
                               int *I, HYPRE_BigInt *J,
                               double *data,
                               HYPRE_BigInt *rows,
                               HYPRE_BigInt *cols)
{
   Init();

   // Determine partitioning size, and my column start and end
   int part_size;
   HYPRE_BigInt my_col_start, my_col_end; // my range: [my_col_start, my_col_end)
   if (HYPRE_AssumedPartitionCheck())
   {
      part_size = 2;
      my_col_start = cols[0];
      my_col_end = cols[1];
   }
   else
   {
      int myid;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &part_size);
      part_size++;
      my_col_start = cols[myid];
      my_col_end = cols[myid+1];
   }

   // Copy in the row and column partitionings
   HYPRE_BigInt *row_starts, *col_starts;
   if (rows == cols)
   {
      row_starts = col_starts = mfem_hypre_TAlloc_host(HYPRE_BigInt, part_size);
      for (int i = 0; i < part_size; i++)
      {
         row_starts[i] = rows[i];
      }
   }
   else
   {
      row_starts = mfem_hypre_TAlloc_host(HYPRE_BigInt, part_size);
      col_starts = mfem_hypre_TAlloc_host(HYPRE_BigInt, part_size);
      for (int i = 0; i < part_size; i++)
      {
         row_starts[i] = rows[i];
         col_starts[i] = cols[i];
      }
   }

   // Create a map for the off-diagonal indices - global to local. Count the
   // number of diagonal and off-diagonal entries.
   HYPRE_Int diag_nnz = 0, offd_nnz = 0, offd_num_cols = 0;
   map<HYPRE_BigInt, HYPRE_Int> offd_map;
   for (HYPRE_Int j = 0, loc_nnz = I[nrows]; j < loc_nnz; j++)
   {
      HYPRE_BigInt glob_col = J[j];
      if (my_col_start <= glob_col && glob_col < my_col_end)
      {
         diag_nnz++;
      }
      else
      {
         offd_map.insert(pair<const HYPRE_BigInt, HYPRE_Int>(glob_col, -1));
         offd_nnz++;
      }
   }
   // count the number of columns in the off-diagonal and set the local indices
   for (auto it = offd_map.begin(); it != offd_map.end(); ++it)
   {
      it->second = offd_num_cols++;
   }

   // construct the global ParCSR matrix
   A = hypre_ParCSRMatrixCreate(comm, glob_nrows, glob_ncols,
                                row_starts, col_starts, offd_num_cols,
                                diag_nnz, offd_nnz);
   hypre_ParCSRMatrixInitialize(A);

   diagOwner = HypreCsrToMem(A->diag, GetHypreMemoryType(), false, mem_diag);
   offdOwner = HypreCsrToMem(A->offd, GetHypreMemoryType(), false, mem_offd);
   HostWrite();

   HYPRE_Int *diag_i, *diag_j, *offd_i, *offd_j;
   HYPRE_BigInt *offd_col_map;
   double *diag_data, *offd_data;
   diag_i = A->diag->i;
   diag_j = A->diag->j;
   diag_data = A->diag->data;
   offd_i = A->offd->i;
   offd_j = A->offd->j;
   offd_data = A->offd->data;
   offd_col_map = A->col_map_offd;

   diag_nnz = offd_nnz = 0;
   for (HYPRE_Int i = 0, j = 0; i < nrows; i++)
   {
      diag_i[i] = diag_nnz;
      offd_i[i] = offd_nnz;
      for (HYPRE_Int j_end = I[i+1]; j < j_end; j++)
      {
         HYPRE_BigInt glob_col = J[j];
         if (my_col_start <= glob_col && glob_col < my_col_end)
         {
            diag_j[diag_nnz] = glob_col - my_col_start;
            diag_data[diag_nnz] = data[j];
            diag_nnz++;
         }
         else
         {
            offd_j[offd_nnz] = offd_map[glob_col];
            offd_data[offd_nnz] = data[j];
            offd_nnz++;
         }
      }
   }
   diag_i[nrows] = diag_nnz;
   offd_i[nrows] = offd_nnz;
   for (auto it = offd_map.begin(); it != offd_map.end(); ++it)
   {
      offd_col_map[it->second] = it->first;
   }

   hypre_ParCSRMatrixSetNumNonzeros(A);
   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }
   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();

   HypreRead();
}

HypreParMatrix::HypreParMatrix(const HypreParMatrix &P)
{
   hypre_ParCSRMatrix *Ph = static_cast<hypre_ParCSRMatrix *>(P);

   Init();

   // Clone the structure
   A = hypre_ParCSRMatrixCompleteClone(Ph);
   // Make a deep copy of the data from the source
   hypre_ParCSRMatrixCopy(Ph, A, 1);

   height = GetNumRows();
   width = GetNumCols();

   CopyRowStarts();
   CopyColStarts();

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   diagOwner = HypreCsrToMem(A->diag, GetHypreMemoryType(), false, mem_diag);
   offdOwner = HypreCsrToMem(A->offd, GetHypreMemoryType(), false, mem_offd);
}

void HypreParMatrix::MakeRef(const HypreParMatrix &master)
{
   Destroy();
   Init();
   A = master.A;
   ParCSROwner = 0;
   height = master.GetNumRows();
   width = master.GetNumCols();
   mem_diag.I.MakeAlias(master.mem_diag.I, 0, master.mem_diag.I.Capacity());
   mem_diag.J.MakeAlias(master.mem_diag.J, 0, master.mem_diag.J.Capacity());
   mem_diag.data.MakeAlias(master.mem_diag.data, 0,
                           master.mem_diag.data.Capacity());
   mem_offd.I.MakeAlias(master.mem_offd.I, 0, master.mem_offd.I.Capacity());
   mem_offd.J.MakeAlias(master.mem_offd.J, 0, master.mem_offd.J.Capacity());
   mem_offd.data.MakeAlias(master.mem_offd.data, 0,
                           master.mem_offd.data.Capacity());
}

hypre_ParCSRMatrix* HypreParMatrix::StealData()
{
   // Only safe when (diagOwner < 0 && offdOwner < 0 && colMapOwner == -1)
   // Otherwise, there may be memory leaks or hypre may destroy arrays allocated
   // with operator new.
   MFEM_ASSERT(diagOwner < 0 && offdOwner < 0 && colMapOwner == -1, "");
   MFEM_ASSERT(diagOwner == offdOwner, "");
   MFEM_ASSERT(ParCSROwner, "");
   hypre_ParCSRMatrix *R = A;
#ifdef HYPRE_USING_CUDA
   if (diagOwner == -1) { HostReadWrite(); }
   else { HypreReadWrite(); }
#endif
   ParCSROwner = false;
   Destroy();
   Init();
   return R;
}

void HypreParMatrix::SetOwnerFlags(signed char diag, signed char offd,
                                   signed char colmap)
{
   diagOwner = diag;
   mem_diag.I.SetHostPtrOwner((diag >= 0) && (diag & 1));
   mem_diag.J.SetHostPtrOwner((diag >= 0) && (diag & 1));
   mem_diag.data.SetHostPtrOwner((diag >= 0) && (diag & 2));
   offdOwner = offd;
   mem_offd.I.SetHostPtrOwner((offd >= 0) && (offd & 1));
   mem_offd.J.SetHostPtrOwner((offd >= 0) && (offd & 1));
   mem_offd.data.SetHostPtrOwner((offd >= 0) && (offd & 2));
   colMapOwner = colmap;
}

void HypreParMatrix::CopyRowStarts()
{
   if (!A || hypre_ParCSRMatrixOwnsRowStarts(A) ||
       (hypre_ParCSRMatrixRowStarts(A) == hypre_ParCSRMatrixColStarts(A) &&
        hypre_ParCSRMatrixOwnsColStarts(A)))
   {
      return;
   }

   int row_starts_size;
   if (HYPRE_AssumedPartitionCheck())
   {
      row_starts_size = 2;
   }
   else
   {
      MPI_Comm_size(hypre_ParCSRMatrixComm(A), &row_starts_size);
      row_starts_size++; // num_proc + 1
   }

   HYPRE_BigInt *old_row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt *new_row_starts = mfem_hypre_CTAlloc_host(HYPRE_BigInt,
                                                          row_starts_size);
   for (int i = 0; i < row_starts_size; i++)
   {
      new_row_starts[i] = old_row_starts[i];
   }

   hypre_ParCSRMatrixRowStarts(A) = new_row_starts;
   hypre_ParCSRMatrixOwnsRowStarts(A) = 1;

   if (hypre_ParCSRMatrixColStarts(A) == old_row_starts)
   {
      hypre_ParCSRMatrixColStarts(A) = new_row_starts;
      hypre_ParCSRMatrixOwnsColStarts(A) = 0;
   }
}

void HypreParMatrix::CopyColStarts()
{
   if (!A || hypre_ParCSRMatrixOwnsColStarts(A) ||
       (hypre_ParCSRMatrixRowStarts(A) == hypre_ParCSRMatrixColStarts(A) &&
        hypre_ParCSRMatrixOwnsRowStarts(A)))
   {
      return;
   }

   int col_starts_size;
   if (HYPRE_AssumedPartitionCheck())
   {
      col_starts_size = 2;
   }
   else
   {
      MPI_Comm_size(hypre_ParCSRMatrixComm(A), &col_starts_size);
      col_starts_size++; // num_proc + 1
   }

   HYPRE_BigInt *old_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt *new_col_starts = mfem_hypre_CTAlloc_host(HYPRE_BigInt,
                                                          col_starts_size);
   for (int i = 0; i < col_starts_size; i++)
   {
      new_col_starts[i] = old_col_starts[i];
   }

   hypre_ParCSRMatrixColStarts(A) = new_col_starts;

   if (hypre_ParCSRMatrixRowStarts(A) == old_col_starts)
   {
      hypre_ParCSRMatrixRowStarts(A) = new_col_starts;
      hypre_ParCSRMatrixOwnsRowStarts(A) = 1;
      hypre_ParCSRMatrixOwnsColStarts(A) = 0;
   }
   else
   {
      hypre_ParCSRMatrixOwnsColStarts(A) = 1;
   }
}

void HypreParMatrix::GetDiag(Vector &diag) const
{
   const int size = Height();
   diag.SetSize(size);
#ifdef HYPRE_USING_CUDA
   if (Device::Allows(Backend::CUDA_MASK))
   {
      MFEM_ASSERT(A->diag->memory_location == HYPRE_MEMORY_DEVICE, "");
      double *d_diag = diag.Write();
      const HYPRE_Int *A_diag_i = A->diag->i;
      const double *A_diag_d = A->diag->data;
      MFEM_FORALL(i, size, d_diag[i] = A_diag_d[A_diag_i[i]];);
   }
   else
#endif
   {
      diag.HostWrite();
      HostRead();
      for (int j = 0; j < size; j++)
      {
         diag(j) = A->diag->data[A->diag->i[j]];
         MFEM_ASSERT(A->diag->j[A->diag->i[j]] == j,
                     "the first entry in each row must be the diagonal one");
      }
      HypreRead();
   }
}

static void MakeSparseMatrixWrapper(int nrows, int ncols,
                                    HYPRE_Int *I, HYPRE_Int *J, double *data,
                                    SparseMatrix &wrapper)
{
#ifndef HYPRE_BIGINT
   SparseMatrix tmp(I, J, data, nrows, ncols, false, false, false);
#else
   int *mI = Memory<int>(nrows + 1);
   for (int i = 0; i <= nrows; i++)
   {
      mI[i] = internal::to_int(I[i]); // checks for overflow in debug mode
   }
   const int nnz = mI[nrows];
   int *mJ = Memory<int>(nnz);
   for (int j = 0; j < nnz; j++)
   {
      mJ[j] = internal::to_int(J[j]); // checks for overflow in debug mode
   }
   SparseMatrix tmp(mI, mJ, data, nrows, ncols, true, false, false);
#endif
   wrapper.Swap(tmp);
}

static void MakeWrapper(const hypre_CSRMatrix *mat,
                        const MemoryIJData &mem,
                        SparseMatrix &wrapper)
{
   const int nrows = internal::to_int(hypre_CSRMatrixNumRows(mat));
   const int ncols = internal::to_int(hypre_CSRMatrixNumCols(mat));
   const int nnz = internal::to_int(mat->num_nonzeros);
   const HYPRE_Int *I = mfem::HostRead(mem.I, nrows + 1);
   const HYPRE_Int *J = mfem::HostRead(mem.J, nnz);
   const double *data = mfem::HostRead(mem.data, nnz);
   MakeSparseMatrixWrapper(nrows, ncols,
                           const_cast<HYPRE_Int*>(I),
                           const_cast<HYPRE_Int*>(J),
                           const_cast<double*>(data),
                           wrapper);
}

void HypreParMatrix::GetDiag(SparseMatrix &diag) const
{
   MakeWrapper(A->diag, mem_diag, diag);
}

void HypreParMatrix::GetOffd(SparseMatrix &offd, HYPRE_BigInt* &cmap) const
{
   MakeWrapper(A->offd, mem_offd, offd);
   cmap = A->col_map_offd;
}

void HypreParMatrix::MergeDiagAndOffd(SparseMatrix &merged)
{
   HostRead();
   hypre_CSRMatrix *hypre_merged = hypre_MergeDiagAndOffd(A);
   HypreRead();
   // Wrap 'hypre_merged' as a SparseMatrix 'merged_tmp'
   SparseMatrix merged_tmp;
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(hypre_merged);
#endif
   MakeSparseMatrixWrapper(
      internal::to_int(hypre_merged->num_rows),
      internal::to_int(hypre_merged->num_cols),
      hypre_merged->i,
      hypre_merged->j,
      hypre_merged->data,
      merged_tmp);
   // Deep copy 'merged_tmp' to 'merged' so that 'merged' does not need
   // 'hypre_merged'
   merged = merged_tmp;
   merged_tmp.Clear();
   hypre_CSRMatrixDestroy(hypre_merged);
}

void HypreParMatrix::GetBlocks(Array2D<HypreParMatrix*> &blocks,
                               bool interleaved_rows,
                               bool interleaved_cols) const
{
   int nr = blocks.NumRows();
   int nc = blocks.NumCols();

   hypre_ParCSRMatrix **hypre_blocks = new hypre_ParCSRMatrix*[nr * nc];
   HostRead();
   internal::hypre_ParCSRMatrixSplit(A, nr, nc, hypre_blocks,
                                     interleaved_rows, interleaved_cols);
   HypreRead();

   for (int i = 0; i < nr; i++)
   {
      for (int j = 0; j < nc; j++)
      {
         blocks[i][j] = new HypreParMatrix(hypre_blocks[i*nc + j]);
      }
   }

   delete [] hypre_blocks;
}

HypreParMatrix * HypreParMatrix::Transpose() const
{
   hypre_ParCSRMatrix * At;
   hypre_ParCSRMatrixTranspose(A, &At, 1);
   hypre_ParCSRMatrixSetNumNonzeros(At);

   hypre_MatvecCommPkgCreate(At);

   if ( M() == N() )
   {
      /* If the matrix is square, make sure that the first entry in each
         row is the diagonal one. */
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(At));
   }

   return new HypreParMatrix(At);
}

#if MFEM_HYPRE_VERSION >= 21800
HypreParMatrix *HypreParMatrix::ExtractSubmatrix(const Array<int> &indices,
                                                 double threshold) const
{
   if (!(A->comm))
   {
      hypre_MatvecCommPkgCreate(A);
   }

   hypre_ParCSRMatrix *submat;

   // Get number of rows stored on this processor
   int local_num_vars = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   // Form hypre CF-splitting array designating submatrix as F-points (-1)
   Array<HYPRE_Int> CF_marker(local_num_vars);
   CF_marker = 1;
   for (int j=0; j<indices.Size(); j++)
   {
      if (indices[j] > local_num_vars)
      {
         MFEM_WARNING("WARNING : " << indices[j] << " > " << local_num_vars);
      }
      CF_marker[indices[j]] = -1;
   }

   // Construct cpts_global array on hypre matrix structure
   HYPRE_BigInt *cpts_global;
   hypre_BoomerAMGCoarseParms(MPI_COMM_WORLD, local_num_vars, 1, NULL,
                              CF_marker, NULL, &cpts_global);

   // Extract submatrix into *submat
   hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, cpts_global,
                                        "FF", &submat, threshold);

   mfem_hypre_TFree(cpts_global);
   return new HypreParMatrix(submat);
}
#endif

HYPRE_Int HypreParMatrix::Mult(HypreParVector &x, HypreParVector &y,
                               double a, double b) const
{
   x.HypreRead();
   (b == 0.0) ? y.HypreWrite() : y.HypreReadWrite();
   return hypre_ParCSRMatrixMatvec(a, A, x, b, y);
}

void HypreParMatrix::Mult(double a, const Vector &x, double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   if (X == NULL)
   {
      X = new HypreParVector(A->comm,
                             GetGlobalNumCols(),
                             nullptr,
                             GetColStarts());
      Y = new HypreParVector(A->comm,
                             GetGlobalNumRows(),
                             nullptr,
                             GetRowStarts());
   }

   const bool xshallow = CanShallowCopy(x.GetMemory(), GetHypreMemoryClass());
   const bool yshallow = CanShallowCopy(y.GetMemory(), GetHypreMemoryClass());

   if (xshallow)
   {
      X->WrapMemoryRead(x.GetMemory());
   }
   else
   {
      if (auxX.Empty()) { auxX.New(NumCols(), GetHypreMemoryType()); }
      auxX.CopyFrom(x.GetMemory(), auxX.Capacity());  // Deep copy
      X->WrapMemoryRead(auxX);
   }

   if (yshallow)
   {
      if (b != 0.0) { Y->WrapMemoryReadWrite(y.GetMemory()); }
      else { Y->WrapMemoryWrite(y.GetMemory()); }
   }
   else
   {
      if (auxY.Empty()) { auxY.New(NumRows(), GetHypreMemoryType()); }
      if (b != 0.0)
      {
         auxY.CopyFrom(y.GetMemory(), auxY.Capacity());  // Deep copy
         Y->WrapMemoryReadWrite(auxY);
      }
      else
      {
         Y->WrapMemoryWrite(auxY);
      }
   }

   hypre_ParCSRMatrixMatvec(a, A, *X, b, *Y);

   if (!yshallow) { y = *Y; }  // Deep copy
}

void HypreParMatrix::MultTranspose(double a, const Vector &x,
                                   double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   // Note: x has the dimensions of Y (height), and
   //       y has the dimensions of X (width)
   if (X == NULL)
   {
      X = new HypreParVector(A->comm,
                             GetGlobalNumCols(),
                             nullptr,
                             GetColStarts());
      Y = new HypreParVector(A->comm,
                             GetGlobalNumRows(),
                             nullptr,
                             GetRowStarts());
   }

   const bool xshallow = CanShallowCopy(x.GetMemory(), GetHypreMemoryClass());
   const bool yshallow = CanShallowCopy(y.GetMemory(), GetHypreMemoryClass());

   // x <--> Y
   if (xshallow)
   {
      Y->WrapMemoryRead(x.GetMemory());
   }
   else
   {
      if (auxY.Empty()) { auxY.New(NumRows(), GetHypreMemoryType()); }
      auxY.CopyFrom(x.GetMemory(), auxY.Capacity());  // Deep copy
      Y->WrapMemoryRead(auxY);
   }

   // y <--> X
   if (yshallow)
   {
      if (b != 0.0) { X->WrapMemoryReadWrite(y.GetMemory()); }
      else { X->WrapMemoryWrite(y.GetMemory()); }
   }
   else
   {
      if (auxX.Empty()) { auxX.New(NumCols(), GetHypreMemoryType()); }
      if (b != 0.0)
      {
         auxX.CopyFrom(y.GetMemory(), auxX.Capacity());  // Deep copy
         X->WrapMemoryReadWrite(auxX);
      }
      else
      {
         X->WrapMemoryWrite(auxX);
      }
   }

   hypre_ParCSRMatrixMatvecT(a, A, *Y, b, *X);

   if (!yshallow) { y = *X; }  // Deep copy
}

HYPRE_Int HypreParMatrix::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                               double a, double b) const
{
   return hypre_ParCSRMatrixMatvec(a, A, (hypre_ParVector *) x, b,
                                   (hypre_ParVector *) y);
}

HYPRE_Int HypreParMatrix::MultTranspose(HypreParVector & x, HypreParVector & y,
                                        double a, double b) const
{
   x.HypreRead();
   (b == 0.0) ? y.HypreWrite() : y.HypreReadWrite();
   return hypre_ParCSRMatrixMatvecT(a, A, x, b, y);
}

void HypreParMatrix::AbsMult(double a, const Vector &x,
                             double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   auto x_data = x.HostRead();
   auto y_data = (b == 0.0) ? y.HostWrite() : y.HostReadWrite();

   HostRead();
   internal::hypre_ParCSRMatrixAbsMatvec(A, a, const_cast<double*>(x_data),
                                         b, y_data);
   HypreRead();
}

void HypreParMatrix::AbsMultTranspose(double a, const Vector &x,
                                      double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   auto x_data = x.HostRead();
   auto y_data = (b == 0.0) ? y.HostWrite() : y.HostReadWrite();

   HostRead();
   internal::hypre_ParCSRMatrixAbsMatvecT(A, a, const_cast<double*>(x_data),
                                          b, y_data);
   HypreRead();
}

HypreParMatrix* HypreParMatrix::LeftDiagMult(const SparseMatrix &D,
                                             HYPRE_BigInt* row_starts) const
{
   const bool assumed_partition = HYPRE_AssumedPartitionCheck();
   const bool row_starts_given = (row_starts != NULL);
   if (!row_starts_given)
   {
      row_starts = hypre_ParCSRMatrixRowStarts(A);
      MFEM_VERIFY(D.Height() == hypre_CSRMatrixNumRows(A->diag),
                  "the matrix D is NOT compatible with the row starts of"
                  " this HypreParMatrix, row_starts must be given.");
   }
   else
   {
      int offset;
      if (assumed_partition)
      {
         offset = 0;
      }
      else
      {
         MPI_Comm_rank(GetComm(), &offset);
      }
      int local_num_rows = row_starts[offset+1]-row_starts[offset];
      MFEM_VERIFY(local_num_rows == D.Height(), "the number of rows in D is "
                  " not compatible with the given row_starts");
   }
   // D.Width() will be checked for compatibility by the SparseMatrix
   // multiplication function, mfem::Mult(), called below.

   int part_size;
   HYPRE_BigInt global_num_rows;
   if (assumed_partition)
   {
      part_size = 2;
      if (row_starts_given)
      {
         global_num_rows = row_starts[2];
         // Here, we use row_starts[2], so row_starts must come from the
         // methods GetDofOffsets/GetTrueDofOffsets of ParFiniteElementSpace
         // (HYPRE's partitions have only 2 entries).
      }
      else
      {
         global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
      }
   }
   else
   {
      MPI_Comm_size(GetComm(), &part_size);
      global_num_rows = row_starts[part_size];
      part_size++;
   }

   HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt *col_map_offd;

   // get the diag and offd blocks as SparseMatrix wrappers
   SparseMatrix A_diag, A_offd;
   GetDiag(A_diag);
   GetOffd(A_offd, col_map_offd);

   // Multiply the diag and offd blocks with D -- these products will be the
   // diag and offd blocks of the output HypreParMatrix, DA.
   SparseMatrix* DA_diag = mfem::Mult(D, A_diag);
   SparseMatrix* DA_offd = mfem::Mult(D, A_offd);

   // Copy row_starts, col_starts, and col_map_offd; ownership of these arrays
   // will be given to the newly constructed output HypreParMatrix, DA.
   HYPRE_BigInt *new_row_starts =
      DuplicateAs<HYPRE_BigInt>(row_starts, part_size, false);
   HYPRE_BigInt *new_col_starts =
      (row_starts == col_starts ? new_row_starts :
       DuplicateAs<HYPRE_BigInt>(col_starts, part_size, false));
   HYPRE_BigInt *new_col_map_offd =
      DuplicateAs<HYPRE_BigInt>(col_map_offd, A_offd.Width());

   // Ownership of DA_diag and DA_offd is transfered to the HypreParMatrix
   // constructor.
   const bool own_diag_offd = true;

   // Create the output HypreParMatrix, DA, from DA_diag and DA_offd
   HypreParMatrix* DA =
      new HypreParMatrix(GetComm(),
                         global_num_rows, hypre_ParCSRMatrixGlobalNumCols(A),
                         new_row_starts, new_col_starts,
                         DA_diag, DA_offd, new_col_map_offd,
                         own_diag_offd);

   // Give ownership of row_starts, col_starts, and col_map_offd to DA
   hypre_ParCSRMatrixSetRowStartsOwner(DA->A, 1);
   hypre_ParCSRMatrixSetColStartsOwner(DA->A, 1);
   DA->colMapOwner = 1;

   return DA;
}

void HypreParMatrix::ScaleRows(const Vector &diag)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   if (hypre_CSRMatrixNumRows(A->diag) != diag.Size())
   {
      mfem_error("Note the Vector diag is not of compatible dimensions with A\n");
   }

   HostReadWrite();
   diag.HostRead();

   int size = Height();
   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);

   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   double val;
   HYPRE_Int jj;
   for (int i(0); i < size; ++i)
   {
      val = diag[i];
      for (jj = Adiag_i[i]; jj < Adiag_i[i+1]; ++jj)
      {
         Adiag_data[jj] *= val;
      }
      for (jj = Aoffd_i[i]; jj < Aoffd_i[i+1]; ++jj)
      {
         Aoffd_data[jj] *= val;
      }
   }

   HypreRead();
}

void HypreParMatrix::InvScaleRows(const Vector &diag)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   if (hypre_CSRMatrixNumRows(A->diag) != diag.Size())
   {
      mfem_error("Note the Vector diag is not of compatible dimensions with A\n");
   }

   HostReadWrite();
   diag.HostRead();

   int size = Height();
   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);


   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   double val;
   HYPRE_Int jj;
   for (int i(0); i < size; ++i)
   {
#ifdef MFEM_DEBUG
      if (0.0 == diag(i))
      {
         mfem_error("HypreParMatrix::InvDiagScale : Division by 0");
      }
#endif
      val = 1./diag(i);
      for (jj = Adiag_i[i]; jj < Adiag_i[i+1]; ++jj)
      {
         Adiag_data[jj] *= val;
      }
      for (jj = Aoffd_i[i]; jj < Aoffd_i[i+1]; ++jj)
      {
         Aoffd_data[jj] *= val;
      }
   }

   HypreRead();
}

void HypreParMatrix::operator*=(double s)
{
   if (hypre_CSRMatrixNumRows(A->diag) != hypre_CSRMatrixNumRows(A->offd))
   {
      mfem_error("Row does not match");
   }

   HostReadWrite();

   HYPRE_Int size=hypre_CSRMatrixNumRows(A->diag);
   HYPRE_Int jj;

   double     *Adiag_data   = hypre_CSRMatrixData(A->diag);
   HYPRE_Int  *Adiag_i      = hypre_CSRMatrixI(A->diag);
   for (jj = 0; jj < Adiag_i[size]; ++jj)
   {
      Adiag_data[jj] *= s;
   }

   double     *Aoffd_data   = hypre_CSRMatrixData(A->offd);
   HYPRE_Int  *Aoffd_i      = hypre_CSRMatrixI(A->offd);
   for (jj = 0; jj < Aoffd_i[size]; ++jj)
   {
      Aoffd_data[jj] *= s;
   }

   HypreRead();
}

static void get_sorted_rows_cols(const Array<int> &rows_cols,
                                 Array<HYPRE_Int> &hypre_sorted)
{
   rows_cols.HostRead();
   hypre_sorted.SetSize(rows_cols.Size());
   bool sorted = true;
   for (int i = 0; i < rows_cols.Size(); i++)
   {
      hypre_sorted[i] = rows_cols[i];
      if (i && rows_cols[i-1] > rows_cols[i]) { sorted = false; }
   }
   if (!sorted) { hypre_sorted.Sort(); }
}

void HypreParMatrix::Threshold(double threshold)
{
   int ierr = 0;

   MPI_Comm comm;
   hypre_CSRMatrix * csr_A;
   hypre_CSRMatrix * csr_A_wo_z;
   hypre_ParCSRMatrix * parcsr_A_ptr;
   HYPRE_BigInt * row_starts = NULL; HYPRE_BigInt * col_starts = NULL;
   HYPRE_BigInt row_start = -1;   HYPRE_BigInt row_end = -1;
   HYPRE_BigInt col_start = -1;   HYPRE_BigInt col_end = -1;

   comm = hypre_ParCSRMatrixComm(A);

   ierr += hypre_ParCSRMatrixGetLocalRange(A,
                                           &row_start,&row_end,
                                           &col_start,&col_end );

   row_starts = hypre_ParCSRMatrixRowStarts(A);
   col_starts = hypre_ParCSRMatrixColStarts(A);

   bool old_owns_row = hypre_ParCSRMatrixOwnsRowStarts(A);
   bool old_owns_col = hypre_ParCSRMatrixOwnsColStarts(A);
   HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   parcsr_A_ptr = hypre_ParCSRMatrixCreate(comm, global_num_rows,
                                           global_num_cols,
                                           row_starts, col_starts,
                                           0, 0, 0);
   hypre_ParCSRMatrixOwnsRowStarts(parcsr_A_ptr) = old_owns_row;
   hypre_ParCSRMatrixOwnsColStarts(parcsr_A_ptr) = old_owns_col;
   hypre_ParCSRMatrixOwnsRowStarts(A) = 0;
   hypre_ParCSRMatrixOwnsColStarts(A) = 0;

   csr_A = hypre_MergeDiagAndOffd(A);

   // Free A, if owned
   Destroy();
   Init();

   csr_A_wo_z = hypre_CSRMatrixDeleteZeros(csr_A,threshold);

   /* hypre_CSRMatrixDeleteZeros will return a NULL pointer rather than a usable
      CSR matrix if it finds no non-zeros */
   if (csr_A_wo_z == NULL)
   {
      csr_A_wo_z = csr_A;
   }
   else
   {
      ierr += hypre_CSRMatrixDestroy(csr_A);
   }

   /* TODO: GenerateDiagAndOffd() uses an int array of size equal to the number
      of columns in csr_A_wo_z which is the global number of columns in A. This
      does not scale well. */
   ierr += GenerateDiagAndOffd(csr_A_wo_z,parcsr_A_ptr,
                               col_start,col_end);

   ierr += hypre_CSRMatrixDestroy(csr_A_wo_z);

   MFEM_VERIFY(ierr == 0, "");

   A = parcsr_A_ptr;

   hypre_ParCSRMatrixSetNumNonzeros(A);
   /* Make sure that the first entry in each row is the diagonal one. */
   if (row_starts == col_starts)
   {
      hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A));
   }
   hypre_MatvecCommPkgCreate(A);
   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::DropSmallEntries(double tol)
{
   HYPRE_Int err = 0, old_err = hypre_error_flag;
   hypre_error_flag = 0;

#if MFEM_HYPRE_VERSION < 21400

   double threshold = 0.0;
   if (tol > 0.0)
   {
      HYPRE_Int *diag_I = A->diag->i,    *offd_I = A->offd->i;
      double    *diag_d = A->diag->data, *offd_d = A->offd->data;
      HYPRE_Int local_num_rows = A->diag->num_rows;
      double max_l2_row_norm = 0.0;
      Vector row;
      for (HYPRE_Int r = 0; r < local_num_rows; r++)
      {
         row.SetDataAndSize(diag_d + diag_I[r], diag_I[r+1]-diag_I[r]);
         double l2_row_norm = row.Norml2();
         row.SetDataAndSize(offd_d + offd_I[r], offd_I[r+1]-offd_I[r]);
         l2_row_norm = std::hypot(l2_row_norm, row.Norml2());
         max_l2_row_norm = std::max(max_l2_row_norm, l2_row_norm);
      }
      double loc_max_l2_row_norm = max_l2_row_norm;
      MPI_Allreduce(&loc_max_l2_row_norm, &max_l2_row_norm, 1, MPI_DOUBLE,
                    MPI_MAX, A->comm);
      threshold = tol * max_l2_row_norm;
   }

   Threshold(threshold);

#elif MFEM_HYPRE_VERSION < 21800

   err = hypre_ParCSRMatrixDropSmallEntries(A, tol);

#else

   err = hypre_ParCSRMatrixDropSmallEntries(A, tol, 2);

#endif

   MFEM_VERIFY(!err, "error encountered: error code = " << err);

   hypre_error_flag = old_err;
}

void HypreParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B)
{
   Array<HYPRE_Int> rc_sorted;
   get_sorted_rows_cols(rows_cols, rc_sorted);

   internal::hypre_ParCSRMatrixEliminateAXB(
      A, rc_sorted.Size(), rc_sorted.GetData(), X, B);
}

HypreParMatrix* HypreParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Array<HYPRE_Int> rc_sorted;
   get_sorted_rows_cols(rows_cols, rc_sorted);

   hypre_ParCSRMatrix* Ae;
   HostReadWrite();
   internal::hypre_ParCSRMatrixEliminateAAe(
      A, &Ae, rc_sorted.Size(), rc_sorted.GetData());
   HypreRead();

   return new HypreParMatrix(Ae, true);
}

HypreParMatrix* HypreParMatrix::EliminateCols(const Array<int> &cols)
{
   Array<HYPRE_Int> rc_sorted;
   get_sorted_rows_cols(cols, rc_sorted);

   hypre_ParCSRMatrix* Ae;
   HostReadWrite();
   internal::hypre_ParCSRMatrixEliminateAAe(
      A, &Ae, rc_sorted.Size(), rc_sorted.GetData(), 1);
   HypreRead();

   return new HypreParMatrix(Ae, true);
}

void HypreParMatrix::EliminateRows(const Array<int> &rows)
{
   if (rows.Size() > 0)
   {
      Array<HYPRE_Int> r_sorted;
      get_sorted_rows_cols(rows, r_sorted);
      HostReadWrite();
      internal::hypre_ParCSRMatrixEliminateRows(A, r_sorted.Size(),
                                                r_sorted.GetData());
      HypreRead();
   }
}

void HypreParMatrix::EliminateBC(const HypreParMatrix &Ae,
                                 const Array<int> &ess_dof_list,
                                 const Vector &X, Vector &B) const
{
   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   // All operations below are local, so we can skip them if ess_dof_list is
   // empty on this processor to avoid potential host <--> device transfers.
   if (ess_dof_list.Size() == 0) { return; }

   HostRead();
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double *data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *I = hypre_CSRMatrixI(A_diag);
#ifdef MFEM_DEBUG
   HYPRE_Int    *J   = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *I_offd = hypre_CSRMatrixI(A_offd);
   double *data_offd = hypre_CSRMatrixData(A_offd);
#endif

   ess_dof_list.HostRead();
   X.HostRead();
   B.HostReadWrite();

   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = data[I[r]] * X(r);
#ifdef MFEM_DEBUG
      MFEM_ASSERT(I[r] < I[r+1], "empty row found!");
      // Check that in the rows specified by the ess_dof_list, the matrix A has
      // only one entry -- the diagonal.
      // if (I[r+1] != I[r]+1 || J[I[r]] != r || I_offd[r] != I_offd[r+1])
      if (J[I[r]] != r)
      {
         MFEM_ABORT("the diagonal entry must be the first entry in the row!");
      }
      for (int j = I[r]+1; j < I[r+1]; j++)
      {
         if (data[j] != 0.0)
         {
            MFEM_ABORT("all off-diagonal entries must be zero!");
         }
      }
      for (int j = I_offd[r]; j < I_offd[r+1]; j++)
      {
         if (data_offd[j] != 0.0)
         {
            MFEM_ABORT("all off-diagonal entries must be zero!");
         }
      }
#endif
   }
   HypreRead();
}

void HypreParMatrix::Print(const char *fname, HYPRE_Int offi,
                           HYPRE_Int offj) const
{
   HostRead();
   hypre_ParCSRMatrixPrintIJ(A,offi,offj,fname);
   HypreRead();
}

void HypreParMatrix::Read(MPI_Comm comm, const char *fname)
{
   Destroy();
   Init();

   HYPRE_Int base_i, base_j;
   hypre_ParCSRMatrixReadIJ(comm, fname, &base_i, &base_j, &A);
   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::Read_IJMatrix(MPI_Comm comm, const char *fname)
{
   Destroy();
   Init();

   HYPRE_IJMatrix A_ij;
   HYPRE_IJMatrixRead(fname, comm, 5555, &A_ij); // HYPRE_PARCSR = 5555

   HYPRE_ParCSRMatrix A_parcsr;
   HYPRE_IJMatrixGetObject(A_ij, (void**) &A_parcsr);

   A = (hypre_ParCSRMatrix*)A_parcsr;

   hypre_ParCSRMatrixSetNumNonzeros(A);

   hypre_MatvecCommPkgCreate(A);

   height = GetNumRows();
   width = GetNumCols();
}

void HypreParMatrix::PrintCommPkg(std::ostream &out) const
{
   hypre_ParCSRCommPkg *comm_pkg = A->comm_pkg;
   MPI_Comm comm = A->comm;
   char c = '\0';
   const int tag = 46801;
   int myid, nproc;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nproc);

   if (myid != 0)
   {
      MPI_Recv(&c, 1, MPI_CHAR, myid-1, tag, comm, MPI_STATUS_IGNORE);
   }
   else
   {
      out << "\nHypreParMatrix: hypre_ParCSRCommPkg:\n";
   }
   out << "Rank " << myid << ":\n"
       "   number of sends  = " << comm_pkg->num_sends <<
       " (" << sizeof(double)*comm_pkg->send_map_starts[comm_pkg->num_sends] <<
       " bytes)\n"
       "   number of recvs  = " << comm_pkg->num_recvs <<
       " (" << sizeof(double)*comm_pkg->recv_vec_starts[comm_pkg->num_recvs] <<
       " bytes)\n";
   if (myid != nproc-1)
   {
      out << std::flush;
      MPI_Send(&c, 1, MPI_CHAR, myid+1, tag, comm);
   }
   else
   {
      out << std::endl;
   }
   MPI_Barrier(comm);
}

void HypreParMatrix::PrintHash(std::ostream &out) const
{
   HashFunction hf;

   out << "global number of rows    : " << A->global_num_rows << '\n'
       << "global number of columns : " << A->global_num_cols << '\n'
       << "first row index : " << A->first_row_index << '\n'
       << " last row index : " << A->last_row_index << '\n'
       << "first col diag  : " << A->first_col_diag << '\n'
       << " last col diag  : " << A->last_col_diag << '\n'
       << "number of nonzeros : " << A->num_nonzeros << '\n';
   // diagonal, off-diagonal
   hypre_CSRMatrix *csr = A->diag;
   const char *csr_name = "diag";
   for (int m = 0; m < 2; m++)
   {
      auto csr_nnz = csr->i[csr->num_rows];
      out << csr_name << " num rows : " << csr->num_rows << '\n'
          << csr_name << " num cols : " << csr->num_cols << '\n'
          << csr_name << " num nnz  : " << csr->num_nonzeros << '\n'
          << csr_name << " i last   : " << csr_nnz
          << (csr_nnz == csr->num_nonzeros ?
              " [good]" : " [** BAD **]") << '\n';
      hf.AppendInts(csr->i, csr->num_rows + 1);
      out << csr_name << " i     hash : " << hf.GetHash() << '\n';
      out << csr_name << " j     hash : ";
      if (csr->j == nullptr)
      {
         out << "(null)\n";
      }
      else
      {
         hf.AppendInts(csr->j, csr_nnz);
         out << hf.GetHash() << '\n';
      }
#if MFEM_HYPRE_VERSION >= 21600
      out << csr_name << " big j hash : ";
      if (csr->big_j == nullptr)
      {
         out << "(null)\n";
      }
      else
      {
         hf.AppendInts(csr->big_j, csr_nnz);
         out << hf.GetHash() << '\n';
      }
#endif
      out << csr_name << " data  hash : ";
      if (csr->data == nullptr)
      {
         out << "(null)\n";
      }
      else
      {
         hf.AppendDoubles(csr->data, csr_nnz);
         out << hf.GetHash() << '\n';
      }

      csr = A->offd;
      csr_name = "offd";
   }

   hf.AppendInts(A->col_map_offd, A->offd->num_cols);
   out << "col map offd hash : " << hf.GetHash() << '\n';
}

inline void delete_hypre_ParCSRMatrixColMapOffd(hypre_ParCSRMatrix *A)
{
   HYPRE_BigInt  *A_col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   int size = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   Memory<HYPRE_Int>(A_col_map_offd, size, true).Delete();
}

void HypreParMatrix::Destroy()
{
   if ( X != NULL ) { delete X; }
   if ( Y != NULL ) { delete Y; }
   auxX.Delete();
   auxY.Delete();

   if (A == NULL) { return; }

#ifdef HYPRE_USING_CUDA
   if (ParCSROwner && (diagOwner < 0 || offdOwner < 0))
   {
      // Put the "host" or "hypre" pointers in {i,j,data} of A->{diag,offd}, so
      // that they can be destroyed by hypre when hypre_ParCSRMatrixDestroy(A)
      // is called below.

      // Check that if both diagOwner and offdOwner are negative then they have
      // the same value.
      MFEM_VERIFY(!(diagOwner < 0 && offdOwner < 0) || diagOwner == offdOwner,
                  "invalid state");

      MemoryClass mc = (diagOwner == -1 || offdOwner == -1) ?
                       Device::GetHostMemoryClass() : GetHypreMemoryClass();
      Write(mc, diagOwner < 0, offdOwner <0);
   }
#endif

   mem_diag.I.Delete();
   mem_diag.J.Delete();
   mem_diag.data.Delete();
   if (diagOwner >= 0)
   {
      hypre_CSRMatrixI(A->diag) = NULL;
      hypre_CSRMatrixJ(A->diag) = NULL;
      hypre_CSRMatrixData(A->diag) = NULL;
   }
   mem_offd.I.Delete();
   mem_offd.J.Delete();
   mem_offd.data.Delete();
   if (offdOwner >= 0)
   {
      hypre_CSRMatrixI(A->offd) = NULL;
      hypre_CSRMatrixJ(A->offd) = NULL;
      hypre_CSRMatrixData(A->offd) = NULL;
   }
   if (colMapOwner >= 0)
   {
      if (colMapOwner & 1)
      {
         delete_hypre_ParCSRMatrixColMapOffd(A);
      }
      hypre_ParCSRMatrixColMapOffd(A) = NULL;
   }

   if (ParCSROwner)
   {
      hypre_ParCSRMatrixDestroy(A);
   }
}

#if MFEM_HYPRE_VERSION >= 21800

void BlockInverseScale(const HypreParMatrix *A, HypreParMatrix *C,
                       const Vector *b, HypreParVector *d,
                       int blocksize, BlockInverseScaleJob job)
{
   if (job == BlockInverseScaleJob::MATRIX_ONLY ||
       job == BlockInverseScaleJob::MATRIX_AND_RHS)
   {
      hypre_ParCSRMatrix *C_hypre;
      hypre_ParcsrBdiagInvScal(*A, blocksize, &C_hypre);
      hypre_ParCSRMatrixDropSmallEntries(C_hypre, 1e-15, 1);
      C->WrapHypreParCSRMatrix(C_hypre);
   }

   if (job == BlockInverseScaleJob::RHS_ONLY ||
       job == BlockInverseScaleJob::MATRIX_AND_RHS)
   {
      HypreParVector b_Hypre(A->GetComm(),
                             A->GetGlobalNumRows(),
                             b->GetData(), A->GetRowStarts());
      hypre_ParVector *d_hypre;
      hypre_ParvecBdiagInvScal(b_Hypre, blocksize, &d_hypre, *A);

      d->WrapHypreParVector(d_hypre, true);
   }
}

#endif

#if MFEM_HYPRE_VERSION < 21400

HypreParMatrix *Add(double alpha, const HypreParMatrix &A,
                    double beta,  const HypreParMatrix &B)
{
   hypre_ParCSRMatrix *C_hypre =
      internal::hypre_ParCSRMatrixAdd(const_cast<HypreParMatrix &>(A),
                                      const_cast<HypreParMatrix &>(B));
   MFEM_VERIFY(C_hypre, "error in hypre_ParCSRMatrixAdd");

   hypre_MatvecCommPkgCreate(C_hypre);
   HypreParMatrix *C = new HypreParMatrix(C_hypre);
   *C = 0.0;
   C->Add(alpha, A);
   C->Add(beta, B);

   return C;
}

HypreParMatrix * ParAdd(const HypreParMatrix *A, const HypreParMatrix *B)
{
   hypre_ParCSRMatrix * C = internal::hypre_ParCSRMatrixAdd(*A,*B);

   hypre_MatvecCommPkgCreate(C);

   return new HypreParMatrix(C);
}

#else

HypreParMatrix *Add(double alpha, const HypreParMatrix &A,
                    double beta,  const HypreParMatrix &B)
{
   hypre_ParCSRMatrix *C;
#if MFEM_HYPRE_VERSION <= 22000
   hypre_ParcsrAdd(alpha, A, beta, B, &C);
#else
   hypre_ParCSRMatrixAdd(alpha, A, beta, B, &C);
#endif
   hypre_MatvecCommPkgCreate(C);

   return new HypreParMatrix(C);
}

HypreParMatrix * ParAdd(const HypreParMatrix *A, const HypreParMatrix *B)
{
   hypre_ParCSRMatrix *C;
#if MFEM_HYPRE_VERSION <= 22000
   hypre_ParcsrAdd(1.0, *A, 1.0, *B, &C);
#else
   hypre_ParCSRMatrixAdd(1.0, *A, 1.0, *B, &C);
#endif

   return new HypreParMatrix(C);
}

#endif

HypreParMatrix * ParMult(const HypreParMatrix *A, const HypreParMatrix *B,
                         bool own_matrix)
{
   hypre_ParCSRMatrix * ab;
#ifdef HYPRE_USING_CUDA
   ab = hypre_ParCSRMatMat(*A, *B);
#else
   ab = hypre_ParMatmul(*A,*B);
#endif
   hypre_ParCSRMatrixSetNumNonzeros(ab);

   hypre_MatvecCommPkgCreate(ab);
   HypreParMatrix *C = new HypreParMatrix(ab);
   if (own_matrix)
   {
      C->CopyRowStarts();
      C->CopyColStarts();
   }
   return C;
}

HypreParMatrix * RAP(const HypreParMatrix *A, const HypreParMatrix *P)
{
   hypre_ParCSRMatrix * rap;

#ifdef HYPRE_USING_CUDA
   // FIXME: this way of computing Pt A P can completely eliminate zero rows
   //        from the sparsity pattern of the product which prevents
   //        EliminateZeroRows() from working correctly. This issue is observed
   //        in ex28p.
   // Quick fix: add a diagonal matrix with 0 diagonal.
   // Maybe use hypre_CSRMatrixCheckDiagFirst to see if we need the fix.
   {
      hypre_ParCSRMatrix *Q = hypre_ParCSRMatMat(*A,*P);
      const bool keepTranspose = false;
      rap = hypre_ParCSRTMatMatKT(*P,Q,keepTranspose);
      hypre_ParCSRMatrixDestroy(Q);

      // alternative:
      // hypre_ParCSRMatrixRAPKT
   }
#else
   HYPRE_Int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));

   hypre_BoomerAMGBuildCoarseOperator(*P,*A,*P,&rap);

   /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
      from P (even if it does not own them)! */
   hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
   hypre_ParCSRMatrixSetColStartsOwner(rap,0);
   if (P_owns_its_col_starts)
   {
      hypre_ParCSRMatrixSetColStartsOwner(*P, 1);
   }
#endif

   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);

   return new HypreParMatrix(rap);
}

HypreParMatrix * RAP(const HypreParMatrix * Rt, const HypreParMatrix *A,
                     const HypreParMatrix *P)
{
   hypre_ParCSRMatrix * rap;

#ifdef HYPRE_USING_CUDA
   {
      hypre_ParCSRMatrix *Q = hypre_ParCSRMatMat(*A,*P);
      rap = hypre_ParCSRTMatMat(*Rt,Q);
      hypre_ParCSRMatrixDestroy(Q);
   }
#else
   HYPRE_Int P_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
   HYPRE_Int Rt_owns_its_col_starts =
      hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*Rt));

   hypre_BoomerAMGBuildCoarseOperator(*Rt,*A,*P,&rap);

   /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
      from Rt and P (even if they do not own them)! */
   hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
   hypre_ParCSRMatrixSetColStartsOwner(rap,0);
   if (P_owns_its_col_starts)
   {
      hypre_ParCSRMatrixSetColStartsOwner(*P, 1);
   }
   if (Rt_owns_its_col_starts)
   {
      hypre_ParCSRMatrixSetColStartsOwner(*Rt, 1);
   }
#endif

   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);

   return new HypreParMatrix(rap);
}

// Helper function for HypreParMatrixFromBlocks. Note that scalability to
// extremely large processor counts is limited by the use of MPI_Allgather.
void GatherBlockOffsetData(MPI_Comm comm, const int rank, const int nprocs,
                           const int num_loc, const Array<int> &offsets,
                           std::vector<int> &all_num_loc, const int numBlocks,
                           std::vector<std::vector<HYPRE_BigInt>> &blockProcOffsets,
                           std::vector<HYPRE_BigInt> &procOffsets,
                           std::vector<std::vector<int>> &procBlockOffsets,
                           HYPRE_BigInt &firstLocal, HYPRE_BigInt &globalNum)
{
   std::vector<std::vector<int>> all_block_num_loc(numBlocks);

   MPI_Allgather(&num_loc, 1, MPI_INT, all_num_loc.data(), 1, MPI_INT, comm);

   for (int j = 0; j < numBlocks; ++j)
   {
      all_block_num_loc[j].resize(nprocs);
      blockProcOffsets[j].resize(nprocs);

      const int blockNumRows = offsets[j + 1] - offsets[j];
      MPI_Allgather(&blockNumRows, 1, MPI_INT, all_block_num_loc[j].data(), 1,
                    MPI_INT, comm);
      blockProcOffsets[j][0] = 0;
      for (int i = 0; i < nprocs - 1; ++i)
      {
         blockProcOffsets[j][i + 1] = blockProcOffsets[j][i]
                                      + all_block_num_loc[j][i];
      }
   }

   firstLocal = 0;
   globalNum = 0;
   procOffsets[0] = 0;
   for (int i = 0; i < nprocs; ++i)
   {
      globalNum += all_num_loc[i];
      if (rank == 0)
      {
         MFEM_VERIFY(globalNum >= 0, "overflow in global size");
      }
      if (i < rank)
      {
         firstLocal += all_num_loc[i];
      }

      if (i < nprocs - 1)
      {
         procOffsets[i + 1] = procOffsets[i] + all_num_loc[i];
      }

      procBlockOffsets[i].resize(numBlocks);
      procBlockOffsets[i][0] = 0;
      for (int j = 1; j < numBlocks; ++j)
      {
         procBlockOffsets[i][j] = procBlockOffsets[i][j - 1]
                                  + all_block_num_loc[j - 1][i];
      }
   }
}

HypreParMatrix * HypreParMatrixFromBlocks(Array2D<HypreParMatrix*> &blocks,
                                          Array2D<double> *blockCoeff)
{
   const int numBlockRows = blocks.NumRows();
   const int numBlockCols = blocks.NumCols();

   MFEM_VERIFY(numBlockRows > 0 &&
               numBlockCols > 0, "Invalid input to HypreParMatrixFromBlocks");

   if (blockCoeff != NULL)
   {
      MFEM_VERIFY(numBlockRows == blockCoeff->NumRows() &&
                  numBlockCols == blockCoeff->NumCols(),
                  "Invalid input to HypreParMatrixFromBlocks");
   }

   Array<int> rowOffsets(numBlockRows+1);
   Array<int> colOffsets(numBlockCols+1);

   int nonNullBlockRow0 = -1;
   for (int j=0; j<numBlockCols; ++j)
   {
      if (blocks(0,j) != NULL)
      {
         nonNullBlockRow0 = j;
         break;
      }
   }

   MFEM_VERIFY(nonNullBlockRow0 >= 0, "Null row of blocks");
   MPI_Comm comm = blocks(0,nonNullBlockRow0)->GetComm();

   // Set offsets based on the number of rows or columns in each block.
   rowOffsets = 0;
   colOffsets = 0;
   for (int i=0; i<numBlockRows; ++i)
   {
      for (int j=0; j<numBlockCols; ++j)
      {
         if (blocks(i,j) != NULL)
         {
            const int nrows = blocks(i,j)->NumRows();
            const int ncols = blocks(i,j)->NumCols();

            MFEM_VERIFY(nrows > 0 &&
                        ncols > 0, "Invalid block in HypreParMatrixFromBlocks");

            if (rowOffsets[i+1] == 0)
            {
               rowOffsets[i+1] = nrows;
            }
            else
            {
               MFEM_VERIFY(rowOffsets[i+1] == nrows,
                           "Inconsistent blocks in HypreParMatrixFromBlocks");
            }

            if (colOffsets[j+1] == 0)
            {
               colOffsets[j+1] = ncols;
            }
            else
            {
               MFEM_VERIFY(colOffsets[j+1] == ncols,
                           "Inconsistent blocks in HypreParMatrixFromBlocks");
            }
         }
      }

      MFEM_VERIFY(rowOffsets[i+1] > 0, "Invalid input blocks");
      rowOffsets[i+1] += rowOffsets[i];
   }

   for (int j=0; j<numBlockCols; ++j)
   {
      MFEM_VERIFY(colOffsets[j+1] > 0, "Invalid input blocks");
      colOffsets[j+1] += colOffsets[j];
   }

   const int num_loc_rows = rowOffsets[numBlockRows];
   const int num_loc_cols = colOffsets[numBlockCols];

   int nprocs, rank;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   std::vector<int> all_num_loc_rows(nprocs);
   std::vector<int> all_num_loc_cols(nprocs);
   std::vector<HYPRE_BigInt> procRowOffsets(nprocs);
   std::vector<HYPRE_BigInt> procColOffsets(nprocs);
   std::vector<std::vector<HYPRE_BigInt>> blockRowProcOffsets(numBlockRows);
   std::vector<std::vector<HYPRE_BigInt>> blockColProcOffsets(numBlockCols);
   std::vector<std::vector<int>> procBlockRowOffsets(nprocs);
   std::vector<std::vector<int>> procBlockColOffsets(nprocs);

   HYPRE_BigInt first_loc_row, glob_nrows, first_loc_col, glob_ncols;
   GatherBlockOffsetData(comm, rank, nprocs, num_loc_rows, rowOffsets,
                         all_num_loc_rows, numBlockRows, blockRowProcOffsets,
                         procRowOffsets, procBlockRowOffsets, first_loc_row,
                         glob_nrows);

   GatherBlockOffsetData(comm, rank, nprocs, num_loc_cols, colOffsets,
                         all_num_loc_cols, numBlockCols, blockColProcOffsets,
                         procColOffsets, procBlockColOffsets, first_loc_col,
                         glob_ncols);

   std::vector<int> opI(num_loc_rows + 1);
   std::vector<int> cnt(num_loc_rows);

   for (int i = 0; i < num_loc_rows; ++i)
   {
      opI[i] = 0;
      cnt[i] = 0;
   }

   opI[num_loc_rows] = 0;

   Array2D<hypre_CSRMatrix *> csr_blocks(numBlockRows, numBlockCols);

   // Loop over all blocks, to determine nnz for each row.
   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (blocks(i, j) == NULL)
         {
            csr_blocks(i, j) = NULL;
         }
         else
         {
            blocks(i, j)->HostRead();
            csr_blocks(i, j) = hypre_MergeDiagAndOffd(*blocks(i, j));
            blocks(i, j)->HypreRead();

            for (int k = 0; k < csr_blocks(i, j)->num_rows; ++k)
            {
               opI[rowOffsets[i] + k + 1] +=
                  csr_blocks(i, j)->i[k + 1] - csr_blocks(i, j)->i[k];
            }
         }
      }
   }

   // Now opI[i] is nnz for row i-1. Do a partial sum to get offsets.
   for (int i = 0; i < num_loc_rows; ++i)
   {
      opI[i + 1] += opI[i];
   }

   const int nnz = opI[num_loc_rows];

   std::vector<HYPRE_BigInt> opJ(nnz);
   std::vector<double> data(nnz);

   // Loop over all blocks, to set matrix data.
   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (csr_blocks(i, j) != NULL)
         {
            const int nrows = csr_blocks(i, j)->num_rows;
            const double cij = blockCoeff ? (*blockCoeff)(i, j) : 1.0;
#if MFEM_HYPRE_VERSION >= 21600
            const bool usingBigJ = (csr_blocks(i, j)->big_j != NULL);
#endif

            for (int k = 0; k < nrows; ++k)
            {
               const int rowg = rowOffsets[i] + k; // process-local row
               const int nnz_k = csr_blocks(i,j)->i[k+1]-csr_blocks(i,j)->i[k];
               const int osk = csr_blocks(i, j)->i[k];

               for (int l = 0; l < nnz_k; ++l)
               {
                  // Find the column process offset for the block.
#if MFEM_HYPRE_VERSION >= 21600
                  const HYPRE_Int bcol = usingBigJ ?
                                         csr_blocks(i, j)->big_j[osk + l] :
                                         csr_blocks(i, j)->j[osk + l];
#else
                  const HYPRE_Int bcol = csr_blocks(i, j)->j[osk + l];
#endif

                  // find the processor 'bcolproc' that holds column 'bcol':
                  const auto &offs = blockColProcOffsets[j];
                  const int bcolproc =
                     std::upper_bound(offs.begin() + 1, offs.end(), bcol)
                     - offs.begin() - 1;

                  opJ[opI[rowg] + cnt[rowg]] = procColOffsets[bcolproc] +
                                               procBlockColOffsets[bcolproc][j]
                                               + bcol
                                               - blockColProcOffsets[j][bcolproc];
                  data[opI[rowg] + cnt[rowg]] = cij * csr_blocks(i, j)->data[osk + l];
                  cnt[rowg]++;
               }
            }
         }
      }
   }

   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (csr_blocks(i, j) != NULL)
         {
            hypre_CSRMatrixDestroy(csr_blocks(i, j));
         }
      }
   }

   std::vector<HYPRE_BigInt> rowStarts2(2);
   rowStarts2[0] = first_loc_row;
   rowStarts2[1] = first_loc_row + all_num_loc_rows[rank];

   std::vector<HYPRE_BigInt> colStarts2(2);
   colStarts2[0] = first_loc_col;
   colStarts2[1] = first_loc_col + all_num_loc_cols[rank];

   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(),
               "only 'assumed partition' mode is supported");

   return new HypreParMatrix(comm, num_loc_rows, glob_nrows, glob_ncols,
                             opI.data(), opJ.data(),
                             data.data(),
                             rowStarts2.data(),
                             colStarts2.data());
}

void EliminateBC(const HypreParMatrix &A, const HypreParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   A.EliminateBC(Ae, ess_dof_list, X, B);
}

// Taubin or "lambda-mu" scheme, which alternates between positive and
// negative step sizes to approximate low-pass filter effect.

int ParCSRRelax_Taubin(hypre_ParCSRMatrix *A, // matrix to relax with
                       hypre_ParVector *f,    // right-hand side
                       double lambda,
                       double mu,
                       int N,
                       double max_eig,
                       hypre_ParVector *u,    // initial/updated approximation
                       hypre_ParVector *r     // another temp vector
                      )
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   double *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   for (int i = 0; i < N; i++)
   {
      // get residual: r = f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      double coef;
      (0 == (i % 2)) ? coef = lambda : coef = mu;

      for (HYPRE_Int j = 0; j < num_rows; j++)
      {
         u_data[j] += coef*r_data[j] / max_eig;
      }
   }

   return 0;
}

// FIR scheme, which uses Chebyshev polynomials and a window function
// to approximate a low-pass step filter.

int ParCSRRelax_FIR(hypre_ParCSRMatrix *A, // matrix to relax with
                    hypre_ParVector *f,    // right-hand side
                    double max_eig,
                    int poly_order,
                    double* fir_coeffs,
                    hypre_ParVector *u,    // initial/updated approximation
                    hypre_ParVector *x0,   // temporaries
                    hypre_ParVector *x1,
                    hypre_ParVector *x2,
                    hypre_ParVector *x3)

{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   double *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

   double *x0_data = hypre_VectorData(hypre_ParVectorLocalVector(x0));
   double *x1_data = hypre_VectorData(hypre_ParVectorLocalVector(x1));
   double *x2_data = hypre_VectorData(hypre_ParVectorLocalVector(x2));
   double *x3_data = hypre_VectorData(hypre_ParVectorLocalVector(x3));

   hypre_ParVectorCopy(u, x0);

   // x1 = f -A*x0/max_eig
   hypre_ParVectorCopy(f, x1);
   hypre_ParCSRMatrixMatvec(-1.0, A, x0, 1.0, x1);

   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x1_data[i] /= -max_eig;
   }

   // x1 = x0 -x1
   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x1_data[i] = x0_data[i] -x1_data[i];
   }

   // x3 = f0*x0 +f1*x1
   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      x3_data[i] = fir_coeffs[0]*x0_data[i] +fir_coeffs[1]*x1_data[i];
   }

   for (int n = 2; n <= poly_order; n++)
   {
      // x2 = f - A*x1/max_eig
      hypre_ParVectorCopy(f, x2);
      hypre_ParCSRMatrixMatvec(-1.0, A, x1, 1.0, x2);

      for (HYPRE_Int i = 0; i < num_rows; i++)
      {
         x2_data[i] /= -max_eig;
      }

      // x2 = (x1-x0) +(x1-2*x2)
      // x3 = x3 +f[n]*x2
      // x0 = x1
      // x1 = x2

      for (HYPRE_Int i = 0; i < num_rows; i++)
      {
         x2_data[i] = (x1_data[i]-x0_data[i]) +(x1_data[i]-2*x2_data[i]);
         x3_data[i] += fir_coeffs[n]*x2_data[i];
         x0_data[i] = x1_data[i];
         x1_data[i] = x2_data[i];
      }
   }

   for (HYPRE_Int i = 0; i < num_rows; i++)
   {
      u_data[i] = x3_data[i];
   }

   return 0;
}

HypreSmoother::HypreSmoother() : Solver()
{
   type = default_type;
   relax_times = 1;
   relax_weight = 1.0;
   omega = 1.0;
   poly_order = 2;
   poly_fraction = .3;
   lambda = 0.5;
   mu = -0.5;
   taubin_iter = 40;

   l1_norms = NULL;
   pos_l1_norms = false;
   eig_est_cg_iter = 10;
   B = X = V = Z = P = R = NULL;
   auxB.Reset(); auxX.Reset();
   X0 = X1 = NULL;
   fir_coeffs = NULL;
   A_is_symmetric = false;
}

HypreSmoother::HypreSmoother(const HypreParMatrix &A_, int type_,
                             int relax_times_, double relax_weight_,
                             double omega_, int poly_order_,
                             double poly_fraction_, int eig_est_cg_iter_)
{
   type = type_;
   relax_times = relax_times_;
   relax_weight = relax_weight_;
   omega = omega_;
   poly_order = poly_order_;
   poly_fraction = poly_fraction_;
   eig_est_cg_iter = eig_est_cg_iter_;

   l1_norms = NULL;
   pos_l1_norms = false;
   B = X = V = Z = P = R = NULL;
   auxB.Reset(); auxX.Reset();
   X0 = X1 = NULL;
   fir_coeffs = NULL;
   A_is_symmetric = false;

   SetOperator(A_);
}

void HypreSmoother::SetType(HypreSmoother::Type type_, int relax_times_)
{
   type = static_cast<int>(type_);
   relax_times = relax_times_;
}

void HypreSmoother::SetSOROptions(double relax_weight_, double omega_)
{
   relax_weight = relax_weight_;
   omega = omega_;
}

void HypreSmoother::SetPolyOptions(int poly_order_, double poly_fraction_,
                                   int eig_est_cg_iter_)
{
   poly_order = poly_order_;
   poly_fraction = poly_fraction_;
   eig_est_cg_iter = eig_est_cg_iter_;
}

void HypreSmoother::SetTaubinOptions(double lambda_, double mu_,
                                     int taubin_iter_)
{
   lambda = lambda_;
   mu = mu_;
   taubin_iter = taubin_iter_;
}

void HypreSmoother::SetWindowByName(const char* name)
{
   double a = -1, b, c;
   if (!strcmp(name,"Rectangular")) { a = 1.0,  b = 0.0,  c = 0.0; }
   if (!strcmp(name,"Hanning")) { a = 0.5,  b = 0.5,  c = 0.0; }
   if (!strcmp(name,"Hamming")) { a = 0.54, b = 0.46, c = 0.0; }
   if (!strcmp(name,"Blackman")) { a = 0.42, b = 0.50, c = 0.08; }
   if (a < 0)
   {
      mfem_error("HypreSmoother::SetWindowByName : name not recognized!");
   }

   SetWindowParameters(a, b, c);
}

void HypreSmoother::SetWindowParameters(double a, double b, double c)
{
   window_params[0] = a;
   window_params[1] = b;
   window_params[2] = c;
}

void HypreSmoother::SetOperator(const Operator &op)
{
   A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
   if (A == NULL)
   {
      mfem_error("HypreSmoother::SetOperator : not HypreParMatrix!");
   }

   height = A->Height();
   width = A->Width();

   auxX.Delete(); auxB.Delete();
   if (B) { delete B; }
   if (X) { delete X; }
   if (V) { delete V; }
   if (Z) { delete Z; }
   if (P) { delete P; }
   if (R) { delete R; }
   if (l1_norms)
   {
      mfem_hypre_TFree(l1_norms);
   }
   delete X0;
   delete X1;

   X1 = X0 = Z = V = B = X = P = R = NULL;
   auxB.Reset(); auxX.Reset();

   if (type >= 1 && type <= 4)
   {
      hypre_ParCSRComputeL1Norms(*A, type, NULL, &l1_norms);
      // The above call will set the hypre_error_flag when it encounters zero
      // rows in A.
   }
   else if (type == 5)
   {
      l1_norms = mfem_hypre_CTAlloc(double, height);
      Vector ones(height), diag(l1_norms, height);
      ones = 1.0;
      A->Mult(ones, diag);
   }
   else
   {
      l1_norms = NULL;
   }
   if (l1_norms && pos_l1_norms)
   {
#ifdef HYPRE_USING_CUDA
      double *d_l1_norms = l1_norms;  // avoid *this capture
      CuWrap1D(height, [=] MFEM_DEVICE (int i)
      {
         d_l1_norms[i] = std::abs(d_l1_norms[i]);
      });
#else
      for (int i = 0; i < height; i++)
      {
         l1_norms[i] = std::abs(l1_norms[i]);
      }
#endif
   }

   if (type == 16)
   {
      poly_scale = 1;
      if (eig_est_cg_iter > 0)
      {
         hypre_ParCSRMaxEigEstimateCG(*A, poly_scale, eig_est_cg_iter,
                                      &max_eig_est, &min_eig_est);
      }
      else
      {
         min_eig_est = 0;
         hypre_ParCSRMaxEigEstimate(*A, poly_scale, &max_eig_est);
      }

      Z = new HypreParVector(*A);
      P = new HypreParVector(*A);
      R = new HypreParVector(*A);

      hypre_ParCSRRelax_Cheby_Setup(*A, max_eig_est, min_eig_est, poly_fraction, poly_order, 1, 0, poly_coeffs, diags);

   }
   else if (type == 1001 || type == 1002)
   {
      poly_scale = 0;
      if (eig_est_cg_iter > 0)
      {
         hypre_ParCSRMaxEigEstimateCG(*A, poly_scale, eig_est_cg_iter,
                                      &max_eig_est, &min_eig_est);
      }
      else
      {
         min_eig_est = 0;
         hypre_ParCSRMaxEigEstimate(*A, poly_scale, &max_eig_est);
      }

      // The Taubin and FIR polynomials are defined on [0, 2]
      max_eig_est /= 2;

      // Compute window function, Chebyshev coefficients, and allocate temps.
      if (type == 1002)
      {
         // Temporaries for Chebyshev recursive evaluation
         Z = new HypreParVector(*A);
         X0 = new HypreParVector(*A);
         X1 = new HypreParVector(*A);

         SetFIRCoefficients(max_eig_est);
      }
   }
}

void HypreSmoother::SetFIRCoefficients(double max_eig)
{
   if (fir_coeffs)
   {
      delete [] fir_coeffs;
   }

   fir_coeffs = new double[poly_order+1];

   double* window_coeffs = new double[poly_order+1];
   double* cheby_coeffs = new double[poly_order+1];

   double a = window_params[0];
   double b = window_params[1];
   double c = window_params[2];
   for (int i = 0; i <= poly_order; i++)
   {
      double t = (i*M_PI)/(poly_order+1);
      window_coeffs[i] = a + b*cos(t) +c*cos(2*t);
   }

   double k_pb = poly_fraction*max_eig;
   double theta_pb = acos(1.0 -0.5*k_pb);
   double sigma = 0.0;
   cheby_coeffs[0] = (theta_pb +sigma)/M_PI;
   for (int i = 1; i <= poly_order; i++)
   {
      double t = i*(theta_pb+sigma);
      cheby_coeffs[i] = 2.0*sin(t)/(i*M_PI);
   }

   for (int i = 0; i <= poly_order; i++)
   {
      fir_coeffs[i] = window_coeffs[i]*cheby_coeffs[i];
   }

   delete[] window_coeffs;
   delete[] cheby_coeffs;
}

void HypreSmoother::Mult(const HypreParVector &b, HypreParVector &x) const
{
   if (A == NULL)
   {
      mfem_error("HypreSmoother::Mult (...) : HypreParMatrix A is missing");
      return;
   }

   // TODO: figure out where each function needs A, b, and x ...

   b.HypreRead();
   if (!iterative_mode)
   {
      x.HypreWrite();
      if (type == 0 && relax_times == 1)
      {
         // Note: hypre_ParCSRDiagScale() is not exposed in older versions
         HYPRE_ParCSRDiagScale(NULL, *A, b, x);
         if (relax_weight != 1.0)
         {
            hypre_ParVectorScale(relax_weight, x);
         }
         return;
      }
      hypre_ParVectorSetConstantValues(x, 0.0);
   }
   else
   {
      x.HypreReadWrite();
   }

   if (V == NULL)
   {
      V = new HypreParVector(*A);
   }

   if (type == 1001)
   {
      for (int sweep = 0; sweep < relax_times; sweep++)
      {
         ParCSRRelax_Taubin(*A, b, lambda, mu, taubin_iter,
                            max_eig_est,
                            x, *V);
      }
   }
   else if (type == 1002)
   {
      for (int sweep = 0; sweep < relax_times; sweep++)
      {
         ParCSRRelax_FIR(*A, b,
                         max_eig_est,
                         poly_order,
                         fir_coeffs,
                         x,
                         *X0, *X1, *V, *Z);
      }
   }
   else
   {
      int hypre_type = type;
      // hypre doesn't have lumped Jacobi, so treat the action as l1-Jacobi
      if (type == 5) { hypre_type = 1; }

      if (Z == NULL)
      {
         hypre_ParCSRRelax(*A, b, hypre_type,
                           relax_times, l1_norms, relax_weight, omega,
                           max_eig_est, min_eig_est, poly_order, poly_fraction,
                           x, *V, NULL, (P?*P:NULL), (R?*R:NULL), poly_coeffs, diags);
      }
      else
      {
         hypre_ParCSRRelax(*A, b, hypre_type,
                           relax_times, l1_norms, relax_weight, omega,
                           max_eig_est, min_eig_est, poly_order, poly_fraction,
                           x, *V, *Z, (P?*P:NULL), (R?*R:NULL), poly_coeffs, diags);
      }
   }
}

void HypreSmoother::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(b.Size() == NumCols(), "");
   MFEM_ASSERT(x.Size() == NumRows(), "");

   if (A == NULL)
   {
      mfem_error("HypreSmoother::Mult (...) : HypreParMatrix A is missing");
      return;
   }

   if (B == NULL)
   {
      B = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumRows(),
                             nullptr,
                             A -> GetRowStarts());
      X = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumCols(),
                             nullptr,
                             A -> GetColStarts());
   }

   const bool bshallow = CanShallowCopy(b.GetMemory(), GetHypreMemoryClass());
   const bool xshallow = CanShallowCopy(x.GetMemory(), GetHypreMemoryClass());

   if (bshallow)
   {
      B->WrapMemoryRead(b.GetMemory());
   }
   else
   {
      if (auxB.Empty()) { auxB.New(NumCols(), GetHypreMemoryType()); }
      auxB.CopyFrom(b.GetMemory(), auxB.Capacity());  // Deep copy
      B->WrapMemoryRead(auxB);
   }

   if (xshallow)
   {
      if (iterative_mode) { X->WrapMemoryReadWrite(x.GetMemory()); }
      else { X->WrapMemoryWrite(x.GetMemory()); }
   }
   else
   {
      if (auxX.Empty()) { auxX.New(NumRows(), GetHypreMemoryType()); }
      if (iterative_mode)
      {
         auxX.CopyFrom(x.GetMemory(), x.Size());  // Deep copy
         X->WrapMemoryReadWrite(auxX);
      }
      else
      {
         X->WrapMemoryWrite(auxX);
      }
   }

   Mult(*B, *X);

   if (!xshallow) { x = *X; }  // Deep copy
}

void HypreSmoother::MultTranspose(const Vector &b, Vector &x) const
{
   if (A_is_symmetric || type == 0 || type == 1 || type == 5)
   {
      Mult(b, x);
      return;
   }
   mfem_error("HypreSmoother::MultTranspose (...) : undefined!\n");
}

HypreSmoother::~HypreSmoother()
{
   auxX.Delete(); auxB.Delete();
   if (B) { delete B; }
   if (X) { delete X; }
   if (V) { delete V; }
   if (Z) { delete Z; }
   if (P) { delete P; }
   if (R) { delete R; }
   if (l1_norms)
   {
      mfem_hypre_TFree(l1_norms);
   }
   if (fir_coeffs)
   {
      delete [] fir_coeffs;
   }
   if (X0) { delete X0; }
   if (X1) { delete X1; }
   if (poly_coeffs) { mfem_hypre_TFree(poly_coeffs); };
   if (diags) { mfem_hypre_TFree(diags); };
}


HypreSolver::HypreSolver()
{
   A = NULL;
   setup_called = 0;
   B = X = NULL;
   auxB.Reset();
   auxX.Reset();
   error_mode = ABORT_HYPRE_ERRORS;
}

HypreSolver::HypreSolver(const HypreParMatrix *A_)
   : Solver(A_->Height(), A_->Width())
{
   A = A_;
   setup_called = 0;
   B = X = NULL;
   auxB.Reset();
   auxX.Reset();
   error_mode = ABORT_HYPRE_ERRORS;
}

void HypreSolver::Mult(const HypreParVector &b, HypreParVector &x) const
{
   HYPRE_Int err;
   if (A == NULL)
   {
      mfem_error("HypreSolver::Mult (...) : HypreParMatrix A is missing");
      return;
   }

   if (!iterative_mode)
   {
      x.HypreWrite();
      hypre_ParVectorSetConstantValues(x, 0.0);
   }

   b.HypreRead();
   x.HypreReadWrite();

   if (!setup_called)
   {
      err = SetupFcn()(*this, *A, b, x);
      if (error_mode == WARN_HYPRE_ERRORS)
      {
         if (err) { MFEM_WARNING("Error during setup! Error code: " << err); }
      }
      else if (error_mode == ABORT_HYPRE_ERRORS)
      {
         MFEM_VERIFY(!err, "Error during setup! Error code: " << err);
      }
      hypre_error_flag = 0;
      setup_called = 1;
   }

   err = SolveFcn()(*this, *A, b, x);
   if (error_mode == WARN_HYPRE_ERRORS)
   {
      if (err) { MFEM_WARNING("Error during solve! Error code: " << err); }
   }
   else if (error_mode == ABORT_HYPRE_ERRORS)
   {
      MFEM_VERIFY(!err, "Error during solve! Error code: " << err);
   }
   hypre_error_flag = 0;
}

void HypreSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(b.Size() == NumCols(), "");
   MFEM_ASSERT(x.Size() == NumRows(), "");

   if (A == NULL)
   {
      mfem_error("HypreSolver::Mult (...) : HypreParMatrix A is missing");
      return;
   }

   if (B == NULL)
   {
      B = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumRows(),
                             nullptr,
                             A -> GetRowStarts());
      X = new HypreParVector(A->GetComm(),
                             A -> GetGlobalNumCols(),
                             nullptr,
                             A -> GetColStarts());
   }

   const bool bshallow = CanShallowCopy(b.GetMemory(), GetHypreMemoryClass());
   const bool xshallow = CanShallowCopy(x.GetMemory(), GetHypreMemoryClass());

   if (bshallow)
   {
      B->WrapMemoryRead(b.GetMemory());
   }
   else
   {
      if (auxB.Empty()) { auxB.New(NumCols(), GetHypreMemoryType()); }
      auxB.CopyFrom(b.GetMemory(), auxB.Capacity());  // Deep copy
      B->WrapMemoryRead(auxB);
   }

   if (xshallow)
   {
      if (iterative_mode) { X->WrapMemoryReadWrite(x.GetMemory()); }
      else { X->WrapMemoryWrite(x.GetMemory()); }
   }
   else
   {
      if (auxX.Empty()) { auxX.New(NumRows(), GetHypreMemoryType()); }
      if (iterative_mode)
      {
         auxX.CopyFrom(x.GetMemory(), x.Size());  // Deep copy
         X->WrapMemoryReadWrite(auxX);
      }
      else
      {
         X->WrapMemoryWrite(auxX);
      }
   }

   Mult(*B, *X);

   if (!xshallow) { x = *X; }  // Deep copy
}

HypreSolver::~HypreSolver()
{
   if (B) { delete B; }
   if (X) { delete X; }
   auxB.Delete();
   auxX.Delete();
}


HyprePCG::HyprePCG(MPI_Comm comm) : precond(NULL)
{
   iterative_mode = true;

   HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
}

HyprePCG::HyprePCG(const HypreParMatrix &A_) : HypreSolver(&A_), precond(NULL)
{
   MPI_Comm comm;

   iterative_mode = true;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRPCGCreate(comm, &pcg_solver);
}

void HyprePCG::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   if (precond)
   {
      precond->SetOperator(*A);
      this->SetPreconditioner(*precond);
   }
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

void HyprePCG::SetTol(double tol)
{
   HYPRE_PCGSetTol(pcg_solver, tol);
}

void HyprePCG::SetAbsTol(double atol)
{
   HYPRE_PCGSetAbsoluteTol(pcg_solver, atol);
}

void HyprePCG::SetMaxIter(int max_iter)
{
   HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
}

void HyprePCG::SetLogging(int logging)
{
   HYPRE_PCGSetLogging(pcg_solver, logging);
}

void HyprePCG::SetPrintLevel(int print_lvl)
{
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_lvl);
}

void HyprePCG::SetPreconditioner(HypreSolver &precond_)
{
   precond = &precond_;

   HYPRE_ParCSRPCGSetPrecond(pcg_solver,
                             precond_.SolveFcn(),
                             precond_.SetupFcn(),
                             precond_);
}

void HyprePCG::SetResidualConvergenceOptions(int res_frequency, double rtol)
{
   HYPRE_PCGSetTwoNorm(pcg_solver, 1);
   if (res_frequency > 0)
   {
      HYPRE_PCGSetRecomputeResidualP(pcg_solver, res_frequency);
   }
   if (rtol > 0.0)
   {
      HYPRE_PCGSetResidualTol(pcg_solver, rtol);
   }
}

void HyprePCG::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   HYPRE_Int time_index = 0;
   HYPRE_Int num_iterations;
   double final_res_norm;
   MPI_Comm comm;
   HYPRE_Int print_level;

   HYPRE_PCGGetPrintLevel(pcg_solver, &print_level);
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level%3);

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!iterative_mode)
   {
      x.HypreWrite();
      hypre_ParVectorSetConstantValues(x, 0.0);
   }

   b.HypreRead();
   x.HypreReadWrite();

   if (!setup_called)
   {
      if (print_level > 0 && print_level < 3)
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRPCGSetup(pcg_solver, *A, b, x);
      setup_called = 1;

      if (print_level > 0 && print_level < 3)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
   }

   if (print_level > 0 && print_level < 3)
   {
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
   }

   HYPRE_ParCSRPCGSolve(pcg_solver, *A, b, x);

   if (print_level > 0)
   {
      if (print_level < 3)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }

      HYPRE_ParCSRPCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg_solver,
                                                  &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         mfem::out << "PCG Iterations = " << num_iterations << endl
                   << "Final PCG Relative Residual Norm = " << final_res_norm
                   << endl;
      }
   }
   HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, print_level);
}

HyprePCG::~HyprePCG()
{
   HYPRE_ParCSRPCGDestroy(pcg_solver);
}


HypreGMRES::HypreGMRES(MPI_Comm comm) : precond(NULL)
{
   iterative_mode = true;

   HYPRE_ParCSRGMRESCreate(comm, &gmres_solver);
   SetDefaultOptions();
}

HypreGMRES::HypreGMRES(const HypreParMatrix &A_)
   : HypreSolver(&A_), precond(NULL)
{
   MPI_Comm comm;

   iterative_mode = true;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRGMRESCreate(comm, &gmres_solver);
   SetDefaultOptions();
}

void HypreGMRES::SetDefaultOptions()
{
   int k_dim    = 50;
   int max_iter = 100;
   double tol   = 1e-6;

   HYPRE_ParCSRGMRESSetKDim(gmres_solver, k_dim);
   HYPRE_ParCSRGMRESSetMaxIter(gmres_solver, max_iter);
   HYPRE_ParCSRGMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   if (precond)
   {
      precond->SetOperator(*A);
      this->SetPreconditioner(*precond);
   }
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

void HypreGMRES::SetTol(double tol)
{
   HYPRE_GMRESSetTol(gmres_solver, tol);
}

void HypreGMRES::SetAbsTol(double tol)
{
   HYPRE_GMRESSetAbsoluteTol(gmres_solver, tol);
}

void HypreGMRES::SetMaxIter(int max_iter)
{
   HYPRE_GMRESSetMaxIter(gmres_solver, max_iter);
}

void HypreGMRES::SetKDim(int k_dim)
{
   HYPRE_GMRESSetKDim(gmres_solver, k_dim);
}

void HypreGMRES::SetLogging(int logging)
{
   HYPRE_GMRESSetLogging(gmres_solver, logging);
}

void HypreGMRES::SetPrintLevel(int print_lvl)
{
   HYPRE_GMRESSetPrintLevel(gmres_solver, print_lvl);
}

void HypreGMRES::SetPreconditioner(HypreSolver &precond_)
{
   precond = &precond_;

   HYPRE_ParCSRGMRESSetPrecond(gmres_solver,
                               precond_.SolveFcn(),
                               precond_.SetupFcn(),
                               precond_);
}

void HypreGMRES::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   HYPRE_Int time_index = 0;
   HYPRE_Int num_iterations;
   double final_res_norm;
   MPI_Comm comm;
   HYPRE_Int print_level;

   HYPRE_GMRESGetPrintLevel(gmres_solver, &print_level);

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!iterative_mode)
   {
      x.HypreWrite();
      hypre_ParVectorSetConstantValues(x, 0.0);
   }

   b.HypreRead();
   x.HypreReadWrite();

   if (!setup_called)
   {
      if (print_level > 0)
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRGMRESSetup(gmres_solver, *A, b, x);
      setup_called = 1;

      if (print_level > 0)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
   }

   if (print_level > 0)
   {
      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);
   }

   HYPRE_ParCSRGMRESSolve(gmres_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRGMRESGetNumIterations(gmres_solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(gmres_solver,
                                                    &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         mfem::out << "GMRES Iterations = " << num_iterations << endl
                   << "Final GMRES Relative Residual Norm = " << final_res_norm
                   << endl;
      }
   }
}

HypreGMRES::~HypreGMRES()
{
   HYPRE_ParCSRGMRESDestroy(gmres_solver);
}


HypreFGMRES::HypreFGMRES(MPI_Comm comm) : precond(NULL)
{
   iterative_mode = true;

   HYPRE_ParCSRFlexGMRESCreate(comm, &fgmres_solver);
   SetDefaultOptions();
}

HypreFGMRES::HypreFGMRES(const HypreParMatrix &A_)
   : HypreSolver(&A_), precond(NULL)
{
   MPI_Comm comm;

   iterative_mode = true;

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   HYPRE_ParCSRFlexGMRESCreate(comm, &fgmres_solver);
   SetDefaultOptions();
}

void HypreFGMRES::SetDefaultOptions()
{
   int k_dim    = 50;
   int max_iter = 100;
   double tol   = 1e-6;

   HYPRE_ParCSRFlexGMRESSetKDim(fgmres_solver, k_dim);
   HYPRE_ParCSRFlexGMRESSetMaxIter(fgmres_solver, max_iter);
   HYPRE_ParCSRFlexGMRESSetTol(fgmres_solver, tol);
}

void HypreFGMRES::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   if (precond)
   {
      precond->SetOperator(*A);
      this->SetPreconditioner(*precond);
   }
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

void HypreFGMRES::SetTol(double tol)
{
   HYPRE_ParCSRFlexGMRESSetTol(fgmres_solver, tol);
}

void HypreFGMRES::SetMaxIter(int max_iter)
{
   HYPRE_ParCSRFlexGMRESSetMaxIter(fgmres_solver, max_iter);
}

void HypreFGMRES::SetKDim(int k_dim)
{
   HYPRE_ParCSRFlexGMRESSetKDim(fgmres_solver, k_dim);
}

void HypreFGMRES::SetLogging(int logging)
{
   HYPRE_ParCSRFlexGMRESSetLogging(fgmres_solver, logging);
}

void HypreFGMRES::SetPrintLevel(int print_lvl)
{
   HYPRE_ParCSRFlexGMRESSetPrintLevel(fgmres_solver, print_lvl);
}

void HypreFGMRES::SetPreconditioner(HypreSolver &precond_)
{
   precond = &precond_;
   HYPRE_ParCSRFlexGMRESSetPrecond(fgmres_solver,
                                   precond_.SolveFcn(),
                                   precond_.SetupFcn(),
                                   precond_);
}

void HypreFGMRES::Mult(const HypreParVector &b, HypreParVector &x) const
{
   int myid;
   HYPRE_Int time_index = 0;
   HYPRE_Int num_iterations;
   double final_res_norm;
   MPI_Comm comm;
   HYPRE_Int print_level;

   HYPRE_FlexGMRESGetPrintLevel(fgmres_solver, &print_level);

   HYPRE_ParCSRMatrixGetComm(*A, &comm);

   if (!iterative_mode)
   {
      x.HypreWrite();
      hypre_ParVectorSetConstantValues(x, 0.0);
   }

   b.HypreRead();
   x.HypreReadWrite();

   if (!setup_called)
   {
      if (print_level > 0)
      {
         time_index = hypre_InitializeTiming("FGMRES Setup");
         hypre_BeginTiming(time_index);
      }

      HYPRE_ParCSRFlexGMRESSetup(fgmres_solver, *A, b, x);
      setup_called = 1;

      if (print_level > 0)
      {
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
   }

   if (print_level > 0)
   {
      time_index = hypre_InitializeTiming("FGMRES Solve");
      hypre_BeginTiming(time_index);
   }

   HYPRE_ParCSRFlexGMRESSolve(fgmres_solver, *A, b, x);

   if (print_level > 0)
   {
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", comm);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_ParCSRFlexGMRESGetNumIterations(fgmres_solver, &num_iterations);
      HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(fgmres_solver,
                                                        &final_res_norm);

      MPI_Comm_rank(comm, &myid);

      if (myid == 0)
      {
         mfem::out << "FGMRES Iterations = " << num_iterations << endl
                   << "Final FGMRES Relative Residual Norm = " << final_res_norm
                   << endl;
      }
   }
}

HypreFGMRES::~HypreFGMRES()
{
   HYPRE_ParCSRFlexGMRESDestroy(fgmres_solver);
}


void HypreDiagScale::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}


HypreParaSails::HypreParaSails(MPI_Comm comm)
{
   HYPRE_ParaSailsCreate(comm, &sai_precond);
   SetDefaultOptions();
}

HypreParaSails::HypreParaSails(const HypreParMatrix &A) : HypreSolver(&A)
{
   MPI_Comm comm;

   HYPRE_ParCSRMatrixGetComm(A, &comm);

   HYPRE_ParaSailsCreate(comm, &sai_precond);
   SetDefaultOptions();
}

void HypreParaSails::SetDefaultOptions()
{
   int    sai_max_levels = 1;
   double sai_threshold  = 0.1;
   double sai_filter     = 0.1;
   int    sai_sym        = 0;
   double sai_loadbal    = 0.0;
   int    sai_reuse      = 0;
   int    sai_logging    = 1;

   HYPRE_ParaSailsSetParams(sai_precond, sai_threshold, sai_max_levels);
   HYPRE_ParaSailsSetFilter(sai_precond, sai_filter);
   HYPRE_ParaSailsSetSym(sai_precond, sai_sym);
   HYPRE_ParaSailsSetLoadbal(sai_precond, sai_loadbal);
   HYPRE_ParaSailsSetReuse(sai_precond, sai_reuse);
   HYPRE_ParaSailsSetLogging(sai_precond, sai_logging);
}

void HypreParaSails::ResetSAIPrecond(MPI_Comm comm)
{
   HYPRE_Int  sai_max_levels;
   HYPRE_Real sai_threshold;
   HYPRE_Real sai_filter;
   HYPRE_Int  sai_sym;
   HYPRE_Real sai_loadbal;
   HYPRE_Int  sai_reuse;
   HYPRE_Int  sai_logging;

   // hypre_ParAMGData *amg_data = (hypre_ParAMGData *)sai_precond;
   HYPRE_ParaSailsGetNlevels(sai_precond, &sai_max_levels);
   HYPRE_ParaSailsGetThresh(sai_precond, &sai_threshold);
   HYPRE_ParaSailsGetFilter(sai_precond, &sai_filter);
   HYPRE_ParaSailsGetSym(sai_precond, &sai_sym);
   HYPRE_ParaSailsGetLoadbal(sai_precond, &sai_loadbal);
   HYPRE_ParaSailsGetReuse(sai_precond, &sai_reuse);
   HYPRE_ParaSailsGetLogging(sai_precond, &sai_logging);

   HYPRE_ParaSailsDestroy(sai_precond);
   HYPRE_ParaSailsCreate(comm, &sai_precond);

   HYPRE_ParaSailsSetParams(sai_precond, sai_threshold, sai_max_levels);
   HYPRE_ParaSailsSetFilter(sai_precond, sai_filter);
   HYPRE_ParaSailsSetSym(sai_precond, sai_sym);
   HYPRE_ParaSailsSetLoadbal(sai_precond, sai_loadbal);
   HYPRE_ParaSailsSetReuse(sai_precond, sai_reuse);
   HYPRE_ParaSailsSetLogging(sai_precond, sai_logging);
}

void HypreParaSails::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   if (A)
   {
      MPI_Comm comm;
      HYPRE_ParCSRMatrixGetComm(*A, &comm);
      ResetSAIPrecond(comm);
   }

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

void HypreParaSails::SetSymmetry(int sym)
{
   HYPRE_ParaSailsSetSym(sai_precond, sym);
}

HypreParaSails::~HypreParaSails()
{
   HYPRE_ParaSailsDestroy(sai_precond);
}


HypreEuclid::HypreEuclid(MPI_Comm comm)
{
   HYPRE_EuclidCreate(comm, &euc_precond);
   SetDefaultOptions();
}

HypreEuclid::HypreEuclid(const HypreParMatrix &A) : HypreSolver(&A)
{
   MPI_Comm comm;

   HYPRE_ParCSRMatrixGetComm(A, &comm);

   HYPRE_EuclidCreate(comm, &euc_precond);
   SetDefaultOptions();
}

void HypreEuclid::SetDefaultOptions()
{
   int    euc_level = 1; // We use ILU(1)
   int    euc_stats = 0; // No logging
   int    euc_mem   = 0; // No memory logging
   int    euc_bj    = 0; // 1: Use Block Jacobi
   int    euc_ro_sc = 0; // 1: Use Row scaling

   HYPRE_EuclidSetLevel(euc_precond, euc_level);
   HYPRE_EuclidSetStats(euc_precond, euc_stats);
   HYPRE_EuclidSetMem(euc_precond, euc_mem);
   HYPRE_EuclidSetBJ(euc_precond, euc_bj);
   HYPRE_EuclidSetRowScale(euc_precond, euc_ro_sc);
}

void HypreEuclid::ResetEuclidPrecond(MPI_Comm comm)
{
   // Euclid does not seem to offer access to its current configuration, so we
   // simply reset it to its default options.
   HYPRE_EuclidDestroy(euc_precond);
   HYPRE_EuclidCreate(comm, &euc_precond);

   SetDefaultOptions();
}

void HypreEuclid::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   if (A)
   {
      MPI_Comm comm;
      HYPRE_ParCSRMatrixGetComm(*new_A, &comm);
      ResetEuclidPrecond(comm);
   }

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

HypreEuclid::~HypreEuclid()
{
   HYPRE_EuclidDestroy(euc_precond);
}


#if MFEM_HYPRE_VERSION >= 21900
HypreILU::HypreILU()
{
   HYPRE_ILUCreate(&ilu_precond);
   SetDefaultOptions();
}

void HypreILU::SetDefaultOptions()
{
   // The type of incomplete LU used locally and globally (see class doc)
   HYPRE_Int ilu_type = 0; // ILU(k) locally and block Jacobi globally
   HYPRE_ILUSetType(ilu_precond, ilu_type);

   // Maximum iterations; 1 iter for preconditioning
   HYPRE_Int max_iter = 1;
   HYPRE_ILUSetMaxIter(ilu_precond, max_iter);

   // The tolerance when used as a smoother; set to 0.0 for preconditioner
   HYPRE_Real tol = 0.0;
   HYPRE_ILUSetTol(ilu_precond, tol);

   // Fill level for ILU(k)
   HYPRE_Int lev_fill = 1;
   HYPRE_ILUSetLevelOfFill(ilu_precond, lev_fill);

   // Local reordering scheme; 0 = no reordering, 1 = reverse Cuthill-McKee
   HYPRE_Int reorder_type = 1;
   HYPRE_ILUSetLocalReordering(ilu_precond, reorder_type);

   // Information print level; 0 = none, 1 = setup, 2 = solve, 3 = setup+solve
   HYPRE_Int print_level = 0;
   HYPRE_ILUSetPrintLevel(ilu_precond, print_level);
}

void HypreILU::ResetILUPrecond()
{
   if (ilu_precond)
   {
      HYPRE_ILUDestroy(ilu_precond);
   }
   HYPRE_ILUCreate(&ilu_precond);
   SetDefaultOptions();
}

void HypreILU::SetLevelOfFill(HYPRE_Int lev_fill)
{
   HYPRE_ILUSetLevelOfFill(ilu_precond, lev_fill);
}

void HypreILU::SetPrintLevel(HYPRE_Int print_level)
{
   HYPRE_ILUSetPrintLevel(ilu_precond, print_level);
}

void HypreILU::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   if (A) { ResetILUPrecond(); }

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

HypreILU::~HypreILU()
{
   HYPRE_ILUDestroy(ilu_precond);
}
#endif


HypreBoomerAMG::HypreBoomerAMG()
{
   HYPRE_BoomerAMGCreate(&amg_precond);
   SetDefaultOptions();
}

HypreBoomerAMG::HypreBoomerAMG(const HypreParMatrix &A) : HypreSolver(&A)
{
   HYPRE_BoomerAMGCreate(&amg_precond);
   SetDefaultOptions();
}

void HypreBoomerAMG::SetDefaultOptions()
{
#ifndef HYPRE_USING_CUDA
   // AMG coarsening options:
   int coarsen_type = 10;   // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
   int agg_levels   = 1;    // number of aggressive coarsening levels
   double theta     = 0.25; // strength threshold: 0.25, 0.5, 0.8

   // AMG interpolation options:
   int interp_type  = 6;    // 6 = extended+i, 0 = classical
   int Pmax         = 4;    // max number of elements per row in P

   // AMG relaxation options:
   int relax_type   = 8;    // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
   int relax_sweeps = 1;    // relaxation sweeps on each level

   // Additional options:
   int print_level  = 1;    // print AMG iterations? 1 = no, 2 = yes
   int max_levels   = 25;   // max number of levels in AMG hierarchy
#else
   // AMG coarsening options:
   int coarsen_type = 8;    // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
   int agg_levels   = 0;    // number of aggressive coarsening levels
   double theta     = 0.25; // strength threshold: 0.25, 0.5, 0.8

   // AMG interpolation options:
   int interp_type  = 6;   // or 3 = direct
   int Pmax         = 4;    // max number of elements per row in P

   // AMG relaxation options:
   int relax_type   = 18;    // or 18 = l1-Jacobi
   int relax_sweeps = 1;    // relaxation sweeps on each level

   // Additional options:
   int print_level  = 1;    // print AMG iterations? 1 = no, 2 = yes
   int max_levels   = 25;   // max number of levels in AMG hierarchy
#endif

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   // default in hypre is 1.0 with some exceptions, e.g. for relax_type = 7
   // HYPRE_BoomerAMGSetRelaxWt(amg_precond, 1.0);
   HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
   HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
   HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

   // Use as a preconditioner (one V-cycle, zero tolerance)
   HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
   HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
}

void HypreBoomerAMG::ResetAMGPrecond()
{
   HYPRE_Int coarsen_type;
   HYPRE_Int agg_levels;
   HYPRE_Int relax_type;
   HYPRE_Int relax_sweeps;
   HYPRE_Real theta;
   HYPRE_Int interp_type;
   HYPRE_Int Pmax;
   HYPRE_Int print_level;
   HYPRE_Int max_levels;
   HYPRE_Int dim;
   HYPRE_Int nrbms = rbms.Size();
   HYPRE_Int nodal;
   HYPRE_Int nodal_diag;
   HYPRE_Int relax_coarse;
   HYPRE_Int interp_vec_variant;
   HYPRE_Int q_max;
   HYPRE_Int smooth_interp_vectors;
   HYPRE_Int interp_refine;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)amg_precond;

   // read options from amg_precond
   HYPRE_BoomerAMGGetCoarsenType(amg_precond, &coarsen_type);
   agg_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   relax_type = hypre_ParAMGDataUserRelaxType(amg_data);
   relax_sweeps = hypre_ParAMGDataUserNumSweeps(amg_data);
   HYPRE_BoomerAMGGetStrongThreshold(amg_precond, &theta);
   hypre_BoomerAMGGetInterpType(amg_precond, &interp_type);
   HYPRE_BoomerAMGGetPMaxElmts(amg_precond, &Pmax);
   HYPRE_BoomerAMGGetPrintLevel(amg_precond, &print_level);
   HYPRE_BoomerAMGGetMaxLevels(amg_precond, &max_levels);
   HYPRE_BoomerAMGGetNumFunctions(amg_precond, &dim);
   if (nrbms) // elasticity solver options
   {
      nodal = hypre_ParAMGDataNodal(amg_data);
      nodal_diag = hypre_ParAMGDataNodalDiag(amg_data);
      HYPRE_BoomerAMGGetCycleRelaxType(amg_precond, &relax_coarse, 3);
      interp_vec_variant = hypre_ParAMGInterpVecVariant(amg_data);
      q_max = hypre_ParAMGInterpVecQMax(amg_data);
      smooth_interp_vectors = hypre_ParAMGSmoothInterpVectors(amg_data);
      interp_refine = hypre_ParAMGInterpRefine(amg_data);
   }

   HYPRE_BoomerAMGDestroy(amg_precond);
   HYPRE_BoomerAMGCreate(&amg_precond);

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
   HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);
   HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
   HYPRE_BoomerAMGSetMaxIter(amg_precond, 1); // one V-cycle
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
   HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
   HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);
   if (nrbms)
   {
      HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
      HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
      HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
      HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
      HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
      HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
      HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);
      RecomputeRBMs();
      HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());
   }
}

void HypreBoomerAMG::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   if (A) { ResetAMGPrecond(); }

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);
   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

void HypreBoomerAMG::SetSystemsOptions(int dim, bool order_bynodes)
{
   HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

   // The default "system" ordering in hypre is Ordering::byVDIM. When we are
   // using Ordering::byNODES, we have to specify the ordering explicitly with
   // HYPRE_BoomerAMGSetDofFunc as in the following code.
   if (order_bynodes)
   {
      // hypre actually deletes the following pointer in HYPRE_BoomerAMGDestroy,
      // so we don't need to track it
      HYPRE_Int *mapping = mfem_hypre_CTAlloc_host(HYPRE_Int, height);
      int h_nnodes = height / dim; // nodes owned in linear algebra (not fem)
      MFEM_VERIFY(height % dim == 0, "Ordering does not work as claimed!");
      int k = 0;
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < h_nnodes; ++j)
         {
            mapping[k++] = i;
         }
      }
      HYPRE_BoomerAMGSetDofFunc(amg_precond, mapping);
   }

   // More robust options with respect to convergence
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);
}

// Rotational rigid-body mode functions, used in SetElasticityOptions()
static void func_rxy(const Vector &x, Vector &y)
{
   y = 0.0; y(0) = x(1); y(1) = -x(0);
}
static void func_ryz(const Vector &x, Vector &y)
{
   y = 0.0; y(1) = x(2); y(2) = -x(1);
}
static void func_rzx(const Vector &x, Vector &y)
{
   y = 0.0; y(2) = x(0); y(0) = -x(2);
}

void HypreBoomerAMG::RecomputeRBMs()
{
   int nrbms;
   Array<HypreParVector*> gf_rbms;
   int dim = fespace->GetParMesh()->Dimension();

   for (int i = 0; i < rbms.Size(); i++)
   {
      HYPRE_ParVectorDestroy(rbms[i]);
   }

   if (dim == 2)
   {
      nrbms = 1;

      VectorFunctionCoefficient coeff_rxy(2, func_rxy);

      ParGridFunction rbms_rxy(fespace);
      rbms_rxy.ProjectCoefficient(coeff_rxy);

      rbms.SetSize(nrbms);
      gf_rbms.SetSize(nrbms);
      gf_rbms[0] = rbms_rxy.ParallelAverage();
   }
   else if (dim == 3)
   {
      nrbms = 3;

      VectorFunctionCoefficient coeff_rxy(3, func_rxy);
      VectorFunctionCoefficient coeff_ryz(3, func_ryz);
      VectorFunctionCoefficient coeff_rzx(3, func_rzx);

      ParGridFunction rbms_rxy(fespace);
      ParGridFunction rbms_ryz(fespace);
      ParGridFunction rbms_rzx(fespace);
      rbms_rxy.ProjectCoefficient(coeff_rxy);
      rbms_ryz.ProjectCoefficient(coeff_ryz);
      rbms_rzx.ProjectCoefficient(coeff_rzx);

      rbms.SetSize(nrbms);
      gf_rbms.SetSize(nrbms);
      gf_rbms[0] = rbms_rxy.ParallelAverage();
      gf_rbms[1] = rbms_ryz.ParallelAverage();
      gf_rbms[2] = rbms_rzx.ParallelAverage();
   }
   else
   {
      nrbms = 0;
      rbms.SetSize(nrbms);
   }

   // Transfer the RBMs from the ParGridFunction to the HYPRE_ParVector objects
   for (int i = 0; i < nrbms; i++)
   {
      rbms[i] = gf_rbms[i]->StealParVector();
      delete gf_rbms[i];
   }
}

void HypreBoomerAMG::SetElasticityOptions(ParFiniteElementSpace *fespace)
{
#ifdef HYPRE_USING_CUDA
   MFEM_ABORT("this method is not supported in hypre built with CUDA");
#endif

   // Save the finite element space to support multiple calls to SetOperator()
   this->fespace = fespace;

   // Make sure the systems AMG options are set
   int dim = fespace->GetParMesh()->Dimension();
   SetSystemsOptions(dim);

   // Nodal coarsening options (nodal coarsening is required for this solver)
   // See hypre's new_ij driver and the paper for descriptions.
   int nodal                 = 4; // strength reduction norm: 1, 3 or 4
   int nodal_diag            = 1; // diagonal in strength matrix: 0, 1 or 2
   int relax_coarse          = 8; // smoother on the coarsest grid: 8, 99 or 29

   // Elasticity interpolation options
   int interp_vec_variant    = 2; // 1 = GM-1, 2 = GM-2, 3 = LN
   int q_max                 = 4; // max elements per row for each Q
   int smooth_interp_vectors = 1; // smooth the rigid-body modes?

   // Optionally pre-process the interpolation matrix through iterative weight
   // refinement (this is generally applicable for any system)
   int interp_refine         = 1;

   HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
   HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
   HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
   HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
   HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
   HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
   HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

   RecomputeRBMs();
   HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());

   // The above BoomerAMG options may result in singular matrices on the coarse
   // grids, which are handled correctly in hypre's Solve method, but can produce
   // hypre errors in the Setup (specifically in the l1 row norm computation).
   // See the documentation of SetErrorMode() for more details.
   error_mode = IGNORE_HYPRE_ERRORS;
}

#if MFEM_HYPRE_VERSION >= 21800

void HypreBoomerAMG::SetAdvectiveOptions(int distanceR,
                                         const std::string &prerelax,
                                         const std::string &postrelax)
{
   // Hypre parameters
   int Sabs = 0;
   int interp_type = 100;
   int relax_type = 10;
   int coarsen_type = 6;
   double strength_tolC = 0.1;
   double strength_tolR = 0.01;
   double filter_tolR = 0.0;
   double filterA_tol = 0.0;

   // Set relaxation on specified grid points
   int ns_down, ns_up, ns_coarse;
   if (distanceR > 0)
   {
      ns_down = prerelax.length();
      ns_up = postrelax.length();
      ns_coarse = 1;

      // Array to store relaxation scheme and pass to Hypre
      HYPRE_Int **grid_relax_points = mfem_hypre_TAlloc(HYPRE_Int*, 4);
      grid_relax_points[0] = NULL;
      grid_relax_points[1] = mfem_hypre_TAlloc(HYPRE_Int, ns_down);
      grid_relax_points[2] = mfem_hypre_TAlloc(HYPRE_Int, ns_up);
      grid_relax_points[3] = mfem_hypre_TAlloc(HYPRE_Int, 1);
      grid_relax_points[3][0] = 0;

      // set down relax scheme
      for (int i = 0; i<ns_down; i++)
      {
         if (prerelax[i] == 'F')
         {
            grid_relax_points[1][i] = -1;
         }
         else if (prerelax[i] == 'C')
         {
            grid_relax_points[1][i] = 1;
         }
         else if (prerelax[i] == 'A')
         {
            grid_relax_points[1][i] = 0;
         }
      }

      // set up relax scheme
      for (int i = 0; i<ns_up; i++)
      {
         if (postrelax[i] == 'F')
         {
            grid_relax_points[2][i] = -1;
         }
         else if (postrelax[i] == 'C')
         {
            grid_relax_points[2][i] = 1;
         }
         else if (postrelax[i] == 'A')
         {
            grid_relax_points[2][i] = 0;
         }
      }

      HYPRE_BoomerAMGSetRestriction(amg_precond, distanceR);

      HYPRE_BoomerAMGSetGridRelaxPoints(amg_precond, grid_relax_points);

      HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
   }

   if (Sabs)
   {
      HYPRE_BoomerAMGSetSabs(amg_precond, Sabs);
   }

   HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);

   // does not support aggressive coarsening
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);

   HYPRE_BoomerAMGSetStrongThreshold(amg_precond, strength_tolC);

   if (distanceR > 0)
   {
      HYPRE_BoomerAMGSetStrongThresholdR(amg_precond, strength_tolR);
      HYPRE_BoomerAMGSetFilterThresholdR(amg_precond, filter_tolR);
   }

   if (relax_type > -1)
   {
      HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
   }

   if (distanceR > 0)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_coarse, 3);
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_down,   1);
      HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_up,     2);

      HYPRE_BoomerAMGSetADropTol(amg_precond, filterA_tol);
      // type = -1: drop based on row inf-norm
      HYPRE_BoomerAMGSetADropType(amg_precond, -1);
   }
}

#endif

HypreBoomerAMG::~HypreBoomerAMG()
{
   for (int i = 0; i < rbms.Size(); i++)
   {
      HYPRE_ParVectorDestroy(rbms[i]);
   }

   HYPRE_BoomerAMGDestroy(amg_precond);
}

HypreAMS::HypreAMS(ParFiniteElementSpace *edge_fespace)
{
   Init(edge_fespace);
}

HypreAMS::HypreAMS(const HypreParMatrix &A, ParFiniteElementSpace *edge_fespace)
   : HypreSolver(&A)
{
   Init(edge_fespace);
}

void HypreAMS::Init(ParFiniteElementSpace *edge_fespace)
{
   int cycle_type       = 13;
   int rlx_sweeps       = 1;
   double rlx_weight    = 1.0;
   double rlx_omega     = 1.0;
#ifndef HYPRE_USING_CUDA
   int amg_coarsen_type = 10;
   int amg_agg_levels   = 1;
   int amg_rlx_type     = 8;
   int rlx_type         = 2;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
#else
   int amg_coarsen_type = 8;
   int amg_agg_levels   = 0;
   int amg_rlx_type     = 18;
   int rlx_type         = 1;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
#endif

   int dim = edge_fespace->GetMesh()->Dimension();
   int sdim = edge_fespace->GetMesh()->SpaceDimension();
   const FiniteElementCollection *edge_fec = edge_fespace->FEColl();

   bool trace_space, rt_trace_space;
   ND_Trace_FECollection *nd_tr_fec = NULL;
   trace_space = dynamic_cast<const ND_Trace_FECollection*>(edge_fec);
   rt_trace_space = dynamic_cast<const RT_Trace_FECollection*>(edge_fec);
   trace_space = trace_space || rt_trace_space;

   int p = 1;
   if (edge_fespace->GetNE() > 0)
   {
      MFEM_VERIFY(!edge_fespace->IsVariableOrder(), "");
      if (trace_space)
      {
         p = edge_fespace->GetFaceOrder(0);
         if (dim == 2) { p++; }
      }
      else
      {
         p = edge_fespace->GetElementOrder(0);
      }
   }

   ParMesh *pmesh = edge_fespace->GetParMesh();
   if (rt_trace_space)
   {
      nd_tr_fec = new ND_Trace_FECollection(p, dim);
      edge_fespace = new ParFiniteElementSpace(pmesh, nd_tr_fec);
   }

   HYPRE_AMSCreate(&ams);

   HYPRE_AMSSetDimension(ams, sdim); // 2D H(div) and 3D H(curl) problems
   HYPRE_AMSSetTol(ams, 0.0);
   HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
   HYPRE_AMSSetCycleType(ams, cycle_type);
   HYPRE_AMSSetPrintLevel(ams, 1);

   // define the nodal linear finite element space associated with edge_fespace
   FiniteElementCollection *vert_fec;
   if (trace_space)
   {
      vert_fec = new H1_Trace_FECollection(p, dim);
   }
   else
   {
      vert_fec = new H1_FECollection(p, dim);
   }
   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,
                                                                   vert_fec);

   // generate and set the vertex coordinates
   if (p == 1 && pmesh->GetNodes() == NULL)
   {
      ParGridFunction x_coord(vert_fespace);
      ParGridFunction y_coord(vert_fespace);
      ParGridFunction z_coord(vert_fespace);
      double *coord;
      for (int i = 0; i < pmesh->GetNV(); i++)
      {
         coord = pmesh -> GetVertex(i);
         x_coord(i) = coord[0];
         y_coord(i) = coord[1];
         if (sdim == 3) { z_coord(i) = coord[2]; }
      }
      x = x_coord.ParallelProject();
      y = y_coord.ParallelProject();

      x->HypreReadWrite();
      y->HypreReadWrite();
      if (sdim == 2)
      {
         z = NULL;
         HYPRE_AMSSetCoordinateVectors(ams, *x, *y, NULL);
      }
      else
      {
         z = z_coord.ParallelProject();
         z->HypreReadWrite();
         HYPRE_AMSSetCoordinateVectors(ams, *x, *y, *z);
      }
   }
   else
   {
      x = NULL;
      y = NULL;
      z = NULL;
   }

   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
   if (trace_space)
   {
      grad->AddTraceFaceInterpolator(new GradientInterpolator);
   }
   else
   {
      grad->AddDomainInterpolator(new GradientInterpolator);
   }
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   HYPRE_AMSSetDiscreteGradient(ams, *G);
   delete grad;

   // generate and set the Nedelec interpolation matrices
   Pi = Pix = Piy = Piz = NULL;
   if (p > 1 || pmesh->GetNodes() != NULL)
   {
      ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, sdim, Ordering::byVDIM);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      if (trace_space)
      {
         id_ND->AddTraceFaceInterpolator(new IdentityInterpolator);
      }
      else
      {
         id_ND->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_ND->Assemble();
      id_ND->Finalize();

      if (cycle_type < 10)
      {
         Pi = id_ND->ParallelAssemble();
      }
      else
      {
         Array2D<HypreParMatrix *> Pi_blocks;
         id_ND->GetParBlocks(Pi_blocks);
         Pix = Pi_blocks(0,0);
         Piy = Pi_blocks(0,1);
         if (sdim == 3) { Piz = Pi_blocks(0,2); }
      }

      delete id_ND;

      HYPRE_ParCSRMatrix HY_Pi  = (Pi)  ? (HYPRE_ParCSRMatrix) *Pi  : NULL;
      HYPRE_ParCSRMatrix HY_Pix = (Pix) ? (HYPRE_ParCSRMatrix) *Pix : NULL;
      HYPRE_ParCSRMatrix HY_Piy = (Piy) ? (HYPRE_ParCSRMatrix) *Piy : NULL;
      HYPRE_ParCSRMatrix HY_Piz = (Piz) ? (HYPRE_ParCSRMatrix) *Piz : NULL;
      HYPRE_AMSSetInterpolations(ams, HY_Pi, HY_Pix, HY_Piy, HY_Piz);

      delete vert_fespace_d;
   }

   delete vert_fespace;
   delete vert_fec;

   if (rt_trace_space)
   {
      delete edge_fespace;
      delete nd_tr_fec;
   }

   // set additional AMS options
   HYPRE_AMSSetSmoothingOptions(ams, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
   HYPRE_AMSSetAlphaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                               theta, amg_interp_type, amg_Pmax);
   HYPRE_AMSSetBetaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                              theta, amg_interp_type, amg_Pmax);

   HYPRE_AMSSetAlphaAMGCoarseRelaxType(ams, amg_rlx_type);
   HYPRE_AMSSetBetaAMGCoarseRelaxType(ams, amg_rlx_type);

   // The AMS preconditioner may sometimes require inverting singular matrices
   // with BoomerAMG, which are handled correctly in hypre's Solve method, but
   // can produce hypre errors in the Setup (specifically in the l1 row norm
   // computation). See the documentation of SetErrorMode() for more details.
   error_mode = IGNORE_HYPRE_ERRORS;
}

void HypreAMS::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);

   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

HypreAMS::~HypreAMS()
{
   HYPRE_AMSDestroy(ams);

   delete x;
   delete y;
   delete z;

   delete G;
   delete Pi;
   delete Pix;
   delete Piy;
   delete Piz;
}

void HypreAMS::SetPrintLevel(int print_lvl)
{
   HYPRE_AMSSetPrintLevel(ams, print_lvl);
}

HypreADS::HypreADS(ParFiniteElementSpace *face_fespace)
{
   Init(face_fespace);
}

HypreADS::HypreADS(const HypreParMatrix &A, ParFiniteElementSpace *face_fespace)
   : HypreSolver(&A)
{
   Init(face_fespace);
}

void HypreADS::Init(ParFiniteElementSpace *face_fespace)
{
   int cycle_type       = 11;
   int rlx_type         = 2;
   int rlx_sweeps       = 1;
   double rlx_weight    = 1.0;
   double rlx_omega     = 1.0;
#ifndef HYPRE_USING_CUDA
   int amg_coarsen_type = 10;
   int amg_agg_levels   = 1;
   int amg_rlx_type     = 8;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
#else
   int amg_coarsen_type = 8;
   int amg_agg_levels   = 0;
   int amg_rlx_type     = 18;
   double theta         = 0.25;
   int amg_interp_type  = 6;
   int amg_Pmax         = 4;
#endif
   int ams_cycle_type   = 14;

   const FiniteElementCollection *face_fec = face_fespace->FEColl();
   bool trace_space =
      (dynamic_cast<const RT_Trace_FECollection*>(face_fec) != NULL);
   int p = 1;
   if (face_fespace->GetNE() > 0)
   {
      MFEM_VERIFY(!face_fespace->IsVariableOrder(), "");
      if (trace_space)
      {
         p = face_fespace->GetFaceOrder(0) + 1;
      }
      else
      {
         p = face_fespace->GetElementOrder(0);
      }
   }

   HYPRE_ADSCreate(&ads);

   HYPRE_ADSSetTol(ads, 0.0);
   HYPRE_ADSSetMaxIter(ads, 1); // use as a preconditioner
   HYPRE_ADSSetCycleType(ads, cycle_type);
   HYPRE_ADSSetPrintLevel(ads, 1);

   // define the nodal and edge finite element spaces associated with face_fespace
   ParMesh *pmesh = (ParMesh *) face_fespace->GetMesh();
   FiniteElementCollection *vert_fec, *edge_fec;
   if (trace_space)
   {
      vert_fec = new H1_Trace_FECollection(p, 3);
      edge_fec = new ND_Trace_FECollection(p, 3);
   }
   else
   {
      vert_fec = new H1_FECollection(p, 3);
      edge_fec = new ND_FECollection(p, 3);
   }

   ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh,
                                                                   vert_fec);
   ParFiniteElementSpace *edge_fespace = new ParFiniteElementSpace(pmesh,
                                                                   edge_fec);

   // generate and set the vertex coordinates
   if (p == 1 && pmesh->GetNodes() == NULL)
   {
      ParGridFunction x_coord(vert_fespace);
      ParGridFunction y_coord(vert_fespace);
      ParGridFunction z_coord(vert_fespace);
      double *coord;
      for (int i = 0; i < pmesh->GetNV(); i++)
      {
         coord = pmesh -> GetVertex(i);
         x_coord(i) = coord[0];
         y_coord(i) = coord[1];
         z_coord(i) = coord[2];
      }
      x = x_coord.ParallelProject();
      y = y_coord.ParallelProject();
      z = z_coord.ParallelProject();
      x->HypreReadWrite();
      y->HypreReadWrite();
      z->HypreReadWrite();
      HYPRE_ADSSetCoordinateVectors(ads, *x, *y, *z);
   }
   else
   {
      x = NULL;
      y = NULL;
      z = NULL;
   }

   // generate and set the discrete curl
   ParDiscreteLinearOperator *curl;
   curl = new ParDiscreteLinearOperator(edge_fespace, face_fespace);
   if (trace_space)
   {
      curl->AddTraceFaceInterpolator(new CurlInterpolator);
   }
   else
   {
      curl->AddDomainInterpolator(new CurlInterpolator);
   }
   curl->Assemble();
   curl->Finalize();
   C = curl->ParallelAssemble();
   C->CopyColStarts(); // since we'll delete edge_fespace
   HYPRE_ADSSetDiscreteCurl(ads, *C);
   delete curl;

   // generate and set the discrete gradient
   ParDiscreteLinearOperator *grad;
   grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
   if (trace_space)
   {
      grad->AddTraceFaceInterpolator(new GradientInterpolator);
   }
   else
   {
      grad->AddDomainInterpolator(new GradientInterpolator);
   }
   grad->Assemble();
   grad->Finalize();
   G = grad->ParallelAssemble();
   G->CopyColStarts(); // since we'll delete vert_fespace
   G->CopyRowStarts(); // since we'll delete edge_fespace
   HYPRE_ADSSetDiscreteGradient(ads, *G);
   delete grad;

   // generate and set the Nedelec and Raviart-Thomas interpolation matrices
   RT_Pi = RT_Pix = RT_Piy = RT_Piz = NULL;
   ND_Pi = ND_Pix = ND_Piy = ND_Piz = NULL;
   if (p > 1 || pmesh->GetNodes() != NULL)
   {
      ParFiniteElementSpace *vert_fespace_d
         = new ParFiniteElementSpace(pmesh, vert_fec, 3, Ordering::byVDIM);

      ParDiscreteLinearOperator *id_ND;
      id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
      if (trace_space)
      {
         id_ND->AddTraceFaceInterpolator(new IdentityInterpolator);
      }
      else
      {
         id_ND->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_ND->Assemble();
      id_ND->Finalize();

      if (ams_cycle_type < 10)
      {
         ND_Pi = id_ND->ParallelAssemble();
         ND_Pi->CopyColStarts(); // since we'll delete vert_fespace_d
         ND_Pi->CopyRowStarts(); // since we'll delete edge_fespace
      }
      else
      {
         Array2D<HypreParMatrix *> ND_Pi_blocks;
         id_ND->GetParBlocks(ND_Pi_blocks);
         ND_Pix = ND_Pi_blocks(0,0);
         ND_Piy = ND_Pi_blocks(0,1);
         ND_Piz = ND_Pi_blocks(0,2);
      }

      delete id_ND;

      ParDiscreteLinearOperator *id_RT;
      id_RT = new ParDiscreteLinearOperator(vert_fespace_d, face_fespace);
      if (trace_space)
      {
         id_RT->AddTraceFaceInterpolator(new NormalInterpolator);
      }
      else
      {
         id_RT->AddDomainInterpolator(new IdentityInterpolator);
      }
      id_RT->Assemble();
      id_RT->Finalize();

      if (cycle_type < 10)
      {
         RT_Pi = id_RT->ParallelAssemble();
         RT_Pi->CopyColStarts(); // since we'll delete vert_fespace_d
      }
      else
      {
         Array2D<HypreParMatrix *> RT_Pi_blocks;
         id_RT->GetParBlocks(RT_Pi_blocks);
         RT_Pix = RT_Pi_blocks(0,0);
         RT_Piy = RT_Pi_blocks(0,1);
         RT_Piz = RT_Pi_blocks(0,2);
      }

      delete id_RT;

      HYPRE_ParCSRMatrix HY_RT_Pi, HY_RT_Pix, HY_RT_Piy, HY_RT_Piz;
      HY_RT_Pi  = (RT_Pi)  ? (HYPRE_ParCSRMatrix) *RT_Pi  : NULL;
      HY_RT_Pix = (RT_Pix) ? (HYPRE_ParCSRMatrix) *RT_Pix : NULL;
      HY_RT_Piy = (RT_Piy) ? (HYPRE_ParCSRMatrix) *RT_Piy : NULL;
      HY_RT_Piz = (RT_Piz) ? (HYPRE_ParCSRMatrix) *RT_Piz : NULL;
      HYPRE_ParCSRMatrix HY_ND_Pi, HY_ND_Pix, HY_ND_Piy, HY_ND_Piz;
      HY_ND_Pi  = (ND_Pi)  ? (HYPRE_ParCSRMatrix) *ND_Pi  : NULL;
      HY_ND_Pix = (ND_Pix) ? (HYPRE_ParCSRMatrix) *ND_Pix : NULL;
      HY_ND_Piy = (ND_Piy) ? (HYPRE_ParCSRMatrix) *ND_Piy : NULL;
      HY_ND_Piz = (ND_Piz) ? (HYPRE_ParCSRMatrix) *ND_Piz : NULL;
      HYPRE_ADSSetInterpolations(ads,
                                 HY_RT_Pi, HY_RT_Pix, HY_RT_Piy, HY_RT_Piz,
                                 HY_ND_Pi, HY_ND_Pix, HY_ND_Piy, HY_ND_Piz);

      delete vert_fespace_d;
   }

   delete vert_fec;
   delete vert_fespace;
   delete edge_fec;
   delete edge_fespace;

   // set additional ADS options
   HYPRE_ADSSetSmoothingOptions(ads, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
   HYPRE_ADSSetAMGOptions(ads, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                          theta, amg_interp_type, amg_Pmax);
   HYPRE_ADSSetAMSOptions(ads, ams_cycle_type, amg_coarsen_type, amg_agg_levels,
                          amg_rlx_type, theta, amg_interp_type, amg_Pmax);

   // The ADS preconditioner requires inverting singular matrices with BoomerAMG,
   // which are handled correctly in hypre's Solve method, but can produce hypre
   // errors in the Setup (specifically in the l1 row norm computation). See the
   // documentation of SetErrorMode() for more details.
   error_mode = IGNORE_HYPRE_ERRORS;
}

void HypreADS::SetOperator(const Operator &op)
{
   const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

   // update base classes: Operator, Solver, HypreSolver
   height = new_A->Height();
   width  = new_A->Width();
   A = const_cast<HypreParMatrix *>(new_A);

   setup_called = 0;
   delete X;
   delete B;
   B = X = NULL;
   auxB.Delete(); auxB.Reset();
   auxX.Delete(); auxX.Reset();
}

HypreADS::~HypreADS()
{
   HYPRE_ADSDestroy(ads);

   delete x;
   delete y;
   delete z;

   delete G;
   delete C;

   delete RT_Pi;
   delete RT_Pix;
   delete RT_Piy;
   delete RT_Piz;

   delete ND_Pi;
   delete ND_Pix;
   delete ND_Piy;
   delete ND_Piz;
}

void HypreADS::SetPrintLevel(int print_lvl)
{
   HYPRE_ADSSetPrintLevel(ads, print_lvl);
}

HypreLOBPCG::HypreMultiVector::HypreMultiVector(int n, HypreParVector & v,
                                                mv_InterfaceInterpreter & interpreter)
   : hpv(NULL),
     nv(n)
{
   mv_ptr = mv_MultiVectorCreateFromSampleVector(&interpreter, nv,
                                                 (HYPRE_ParVector)v);

   HYPRE_ParVector* vecs = NULL;
   {
      mv_TempMultiVector* tmp =
         (mv_TempMultiVector*)mv_MultiVectorGetData(mv_ptr);
      vecs = (HYPRE_ParVector*)(tmp -> vector);
   }

   hpv = new HypreParVector*[nv];
   for (int i=0; i<nv; i++)
   {
      hpv[i] = new HypreParVector(vecs[i]);
   }
}

HypreLOBPCG::HypreMultiVector::~HypreMultiVector()
{
   if ( hpv != NULL )
   {
      for (int i=0; i<nv; i++)
      {
         delete hpv[i];
      }
      delete [] hpv;
   }

   mv_MultiVectorDestroy(mv_ptr);
}

void
HypreLOBPCG::HypreMultiVector::Randomize(HYPRE_Int seed)
{
   mv_MultiVectorSetRandom(mv_ptr, seed);
}

HypreParVector &
HypreLOBPCG::HypreMultiVector::GetVector(unsigned int i)
{
   MFEM_ASSERT((int)i < nv, "index out of range");

   return ( *hpv[i] );
}

HypreParVector **
HypreLOBPCG::HypreMultiVector::StealVectors()
{
   HypreParVector ** hpv_ret = hpv;

   hpv = NULL;

   mv_TempMultiVector * mv_tmp =
      (mv_TempMultiVector*)mv_MultiVectorGetData(mv_ptr);

   mv_tmp->ownsVectors = 0;

   for (int i=0; i<nv; i++)
   {
      hpv_ret[i]->SetOwnership(1);
   }

   return hpv_ret;
}

HypreLOBPCG::HypreLOBPCG(MPI_Comm c)
   : comm(c),
     myid(0),
     numProcs(1),
     nev(10),
     seed(75),
     glbSize(-1),
     part(NULL),
     multi_vec(NULL),
     x(NULL),
     subSpaceProj(NULL)
{
   MPI_Comm_size(comm,&numProcs);
   MPI_Comm_rank(comm,&myid);

   HYPRE_ParCSRSetupInterpreter(&interpreter);
   HYPRE_ParCSRSetupMatvec(&matvec_fn);
   HYPRE_LOBPCGCreate(&interpreter, &matvec_fn, &lobpcg_solver);
}

HypreLOBPCG::~HypreLOBPCG()
{
   delete multi_vec;
   delete x;
   delete [] part;

   HYPRE_LOBPCGDestroy(lobpcg_solver);
}

void
HypreLOBPCG::SetTol(double tol)
{
   HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
}

void
HypreLOBPCG::SetRelTol(double rel_tol)
{
#if MFEM_HYPRE_VERSION >= 21101
   HYPRE_LOBPCGSetRTol(lobpcg_solver, rel_tol);
#else
   MFEM_ABORT("This method requires HYPRE version >= 2.11.1");
#endif
}

void
HypreLOBPCG::SetMaxIter(int max_iter)
{
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver, max_iter);
}

void
HypreLOBPCG::SetPrintLevel(int logging)
{
   if (myid == 0)
   {
      HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, logging);
   }
}

void
HypreLOBPCG::SetPrecondUsageMode(int pcg_mode)
{
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcg_mode);
}

void
HypreLOBPCG::SetPreconditioner(Solver & precond)
{
   HYPRE_LOBPCGSetPrecond(lobpcg_solver,
                          (HYPRE_PtrToSolverFcn)this->PrecondSolve,
                          (HYPRE_PtrToSolverFcn)this->PrecondSetup,
                          (HYPRE_Solver)&precond);
}

void
HypreLOBPCG::SetOperator(Operator & A)
{
   HYPRE_BigInt locSize = A.Width();

   if (HYPRE_AssumedPartitionCheck())
   {
      part = new HYPRE_BigInt[2];

      MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);

      part[0] = part[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
   }
   else
   {
      part = new HYPRE_BigInt[numProcs+1];

      MPI_Allgather(&locSize, 1, HYPRE_MPI_BIG_INT,
                    &part[1], 1, HYPRE_MPI_BIG_INT, comm);

      part[0] = 0;
      for (int i=0; i<numProcs; i++)
      {
         part[i+1] += part[i];
      }

      glbSize = part[numProcs];
   }

   if ( x != NULL )
   {
      delete x;
   }

   // Create a distributed vector without a data array.
   const bool is_device_ptr = true;
   x = new HypreParVector(comm,glbSize,NULL,part,is_device_ptr);

   matvec_fn.MatvecCreate  = this->OperatorMatvecCreate;
   matvec_fn.Matvec        = this->OperatorMatvec;
   matvec_fn.MatvecDestroy = this->OperatorMatvecDestroy;

   HYPRE_LOBPCGSetup(lobpcg_solver,(HYPRE_Matrix)&A,NULL,NULL);
}

void
HypreLOBPCG::SetMassMatrix(Operator & M)
{
   matvec_fn.MatvecCreate  = this->OperatorMatvecCreate;
   matvec_fn.Matvec        = this->OperatorMatvec;
   matvec_fn.MatvecDestroy = this->OperatorMatvecDestroy;

   HYPRE_LOBPCGSetupB(lobpcg_solver,(HYPRE_Matrix)&M,NULL);
}

void
HypreLOBPCG::GetEigenvalues(Array<double> & eigs) const
{
   // Initialize eigenvalues array with marker values
   eigs.SetSize(nev);

   for (int i=0; i<nev; i++)
   {
      eigs[i] = eigenvalues[i];
   }
}

const HypreParVector &
HypreLOBPCG::GetEigenvector(unsigned int i) const
{
   return multi_vec->GetVector(i);
}

void
HypreLOBPCG::SetInitialVectors(int num_vecs, HypreParVector ** vecs)
{
   // Initialize HypreMultiVector object if necessary
   if ( multi_vec == NULL )
   {
      MFEM_ASSERT(x != NULL, "In HypreLOBPCG::SetInitialVectors()");

      multi_vec = new HypreMultiVector(nev, *x, interpreter);
   }

   // Copy the vectors provided
   for (int i=0; i < min(num_vecs,nev); i++)
   {
      multi_vec->GetVector(i) = *vecs[i];
   }

   // Randomize any remaining vectors
   for (int i=min(num_vecs,nev); i < nev; i++)
   {
      multi_vec->GetVector(i).Randomize(seed);
   }

   // Ensure all vectors are in the proper subspace
   if ( subSpaceProj != NULL )
   {
      HypreParVector y(*x);
      y = multi_vec->GetVector(0);

      for (int i=1; i<nev; i++)
      {
         subSpaceProj->Mult(multi_vec->GetVector(i),
                            multi_vec->GetVector(i-1));
      }
      subSpaceProj->Mult(y,
                         multi_vec->GetVector(nev-1));
   }
}

void
HypreLOBPCG::Solve()
{
   // Initialize HypreMultiVector object if necessary
   if ( multi_vec == NULL )
   {
      MFEM_ASSERT(x != NULL, "In HypreLOBPCG::Solve()");

      multi_vec = new HypreMultiVector(nev, *x, interpreter);
      multi_vec->Randomize(seed);

      if ( subSpaceProj != NULL )
      {
         HypreParVector y(*x);
         y = multi_vec->GetVector(0);

         for (int i=1; i<nev; i++)
         {
            subSpaceProj->Mult(multi_vec->GetVector(i),
                               multi_vec->GetVector(i-1));
         }
         subSpaceProj->Mult(y, multi_vec->GetVector(nev-1));
      }
   }

   eigenvalues.SetSize(nev);
   eigenvalues = NAN;

   // Perform eigenmode calculation
   //
   // The eigenvalues are computed in ascending order (internally the
   // order is determined by the LAPACK routine 'dsydv'.)
   HYPRE_LOBPCGSolve(lobpcg_solver, NULL, *multi_vec, eigenvalues);
}

void *
HypreLOBPCG::OperatorMatvecCreate( void *A,
                                   void *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

HYPRE_Int
HypreLOBPCG::OperatorMatvec( void *matvec_data,
                             HYPRE_Complex alpha,
                             void *A,
                             void *x,
                             HYPRE_Complex beta,
                             void *y )
{
   MFEM_VERIFY(alpha == 1.0 && beta == 0.0, "values not supported");

   Operator *Aop = (Operator*)A;

   int width = Aop->Width();

   hypre_ParVector * xPar = (hypre_ParVector *)x;
   hypre_ParVector * yPar = (hypre_ParVector *)y;

   Vector xVec(xPar->local_vector->data, width);
   Vector yVec(yPar->local_vector->data, width);

   Aop->Mult( xVec, yVec );

   return 0;
}

HYPRE_Int
HypreLOBPCG::OperatorMatvecDestroy( void *matvec_data )
{
   return 0;
}

HYPRE_Int
HypreLOBPCG::PrecondSolve(void *solver,
                          void *A,
                          void *b,
                          void *x)
{
   Solver   *PC = (Solver*)solver;
   Operator *OP = (Operator*)A;

   int width = OP->Width();

   hypre_ParVector * bPar = (hypre_ParVector *)b;
   hypre_ParVector * xPar = (hypre_ParVector *)x;

   Vector bVec(bPar->local_vector->data, width);
   Vector xVec(xPar->local_vector->data, width);

   PC->Mult( bVec, xVec );

   return 0;
}

HYPRE_Int
HypreLOBPCG::PrecondSetup(void *solver,
                          void *A,
                          void *b,
                          void *x)
{
   return 0;
}

HypreAME::HypreAME(MPI_Comm comm)
   : myid(0),
     numProcs(1),
     nev(10),
     setT(false),
     ams_precond(NULL),
     eigenvalues(NULL),
     multi_vec(NULL),
     eigenvectors(NULL)
{
   MPI_Comm_size(comm,&numProcs);
   MPI_Comm_rank(comm,&myid);

   HYPRE_AMECreate(&ame_solver);
   HYPRE_AMESetPrintLevel(ame_solver, 0);
}

HypreAME::~HypreAME()
{
   if ( multi_vec )
   {
      mfem_hypre_TFree_host(multi_vec);
   }

   if ( eigenvectors )
   {
      for (int i=0; i<nev; i++)
      {
         delete eigenvectors[i];
      }
   }
   delete [] eigenvectors;

   if ( eigenvalues )
   {
      mfem_hypre_TFree_host(eigenvalues);
   }

   HYPRE_AMEDestroy(ame_solver);
}

void
HypreAME::SetNumModes(int num_eigs)
{
   nev = num_eigs;

   HYPRE_AMESetBlockSize(ame_solver, nev);
}

void
HypreAME::SetTol(double tol)
{
   HYPRE_AMESetTol(ame_solver, tol);
}

void
HypreAME::SetRelTol(double rel_tol)
{
#if MFEM_HYPRE_VERSION >= 21101
   HYPRE_AMESetRTol(ame_solver, rel_tol);
#else
   MFEM_ABORT("This method requires HYPRE version >= 2.11.1");
#endif
}

void
HypreAME::SetMaxIter(int max_iter)
{
   HYPRE_AMESetMaxIter(ame_solver, max_iter);
}

void
HypreAME::SetPrintLevel(int logging)
{
   if (myid == 0)
   {
      HYPRE_AMESetPrintLevel(ame_solver, logging);
   }
}

void
HypreAME::SetPreconditioner(HypreSolver & precond)
{
   ams_precond = &precond;
}

void
HypreAME::SetOperator(const HypreParMatrix & A)
{
   if ( !setT )
   {
      HYPRE_Solver ams_precond_ptr = (HYPRE_Solver)*ams_precond;

      ams_precond->SetupFcn()(*ams_precond,A,NULL,NULL);

      HYPRE_AMESetAMSSolver(ame_solver, ams_precond_ptr);
   }

   HYPRE_AMESetup(ame_solver);
}

void
HypreAME::SetMassMatrix(const HypreParMatrix & M)
{
   HYPRE_ParCSRMatrix parcsr_M = M;
   HYPRE_AMESetMassMatrix(ame_solver,(HYPRE_ParCSRMatrix)parcsr_M);
}

void
HypreAME::Solve()
{
   HYPRE_AMESolve(ame_solver);

   // Grab a pointer to the eigenvalues from AME
   HYPRE_AMEGetEigenvalues(ame_solver,&eigenvalues);

   // Grad a pointer to the eigenvectors from AME
   HYPRE_AMEGetEigenvectors(ame_solver,&multi_vec);
}

void
HypreAME::GetEigenvalues(Array<double> & eigs) const
{
   // Initialize eigenvalues array with marker values
   eigs.SetSize(nev); eigs = -1.0;

   // Copy eigenvalues to eigs array
   for (int i=0; i<nev; i++)
   {
      eigs[i] = eigenvalues[i];
   }
}

void
HypreAME::createDummyVectors() const
{
   eigenvectors = new HypreParVector*[nev];
   for (int i=0; i<nev; i++)
   {
      eigenvectors[i] = new HypreParVector(multi_vec[i]);
      eigenvectors[i]->SetOwnership(1);
   }
}

const HypreParVector &
HypreAME::GetEigenvector(unsigned int i) const
{
   if ( eigenvectors == NULL )
   {
      this->createDummyVectors();
   }

   return *eigenvectors[i];
}

HypreParVector **
HypreAME::StealEigenvectors()
{
   if ( eigenvectors == NULL )
   {
      this->createDummyVectors();
   }

   // Set the local pointers to NULL so that they won't be deleted later
   HypreParVector ** vecs = eigenvectors;
   eigenvectors = NULL;
   multi_vec = NULL;

   return vecs;
}

}

#endif
