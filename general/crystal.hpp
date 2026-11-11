// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>
#include "device.hpp"
#include "forall.hpp"
#include "../linalg/ordering.hpp"
#include "../linalg/vector.hpp"
#include "../linalg/particlevector.hpp"
#include "array.hpp"

namespace mfem
{


namespace CR
{
namespace internal
{

/** @brief Abstract interface for one data column routed by CrystalRouter

    A Field is a column of a particle attribute (coordinates, id, tag, etc.)
    stored for all particles. It wraps over an MFEM container to be used for
    routing. Subclasses implement specific functionalities for routing.
    Each row slice of a field is the attribute that belongs to one particle.

    Each Field has six functions:
    - GetNumParticles(): number of particles in the column.
    - GetParticleBytes(): bytes per particle, vdim * sizeof(T).
    - GetAlign(): alignment in bytes of one component.
    - OnDevice(): true if the column uses the device (a Device is enabled and
      the container's UseDevice() flag is set).
    - Pack(): copy leaving particles into the rows of a row buffer.
    - Unpack(): rebuild the column from the arrival rows and the old tail.

    @note currently lots of repeated code and functions, attempting to cut down
          and refactor shared methods in crystal-routing-testing branch
*/
class Field
{
public:
   virtual ~Field() = default;
   virtual int GetNumParticles() const = 0;
   virtual std::size_t GetParticleBytes() const = 0;
   virtual std::size_t GetAlign() const = 0;
   virtual bool OnDevice() const = 0;

   /** @brief Copy the particles @a leaving_idx into the rows of buffer

       @param[in]  leaving_idx  Indices of the particles to copy
       @param[out] buf          Row buffer
       @param[in]  stride       Size in bytes of one row
       @param[in]  off          Offset in bytes of this field in a row */
   virtual void Pack(const mfem::Array<int> &leaving_idx, char *buf,
                     std::size_t stride, std::size_t off) const = 0;

   /** @brief Rebuilds a field with @a n_new particles from buffer

       @param[in] dst         Particles to write
       @param[in] tail        Old particles to move,
                              dst.Size() = n_arrivals + tail.Size()
       @param[in] buf         Row buffer, same side as OnDevice(), read only
                              for k < n_arrivals
       @param[in] n_arrivals  Number of rows in @a buf to read
       @param[in] n_new       New number of particles
       @param[in] stride      Size in bytes of one row
       @param[in] off         Offset in bytes of this field in a row */
   virtual void Unpack(const mfem::Array<int> &dst,
                       const mfem::Array<int> &tail,
                       const char *buf, int n_arrivals, int n_new,
                       std::size_t stride, std::size_t off) = 0;
};

/** @brief Return the position of one component of a particle in a flat array,
    used internally for indexing

    @param[in] particle     Particle index
    @param[in] comp         Component index, in [0, @a vdim)
    @param[in] n_particles  Total number of particles stored
    @param[in] vdim         Number of components per particle
    @param[in] ordering     Ordering::byNODES or Ordering::byVDIM */
MFEM_HOST_DEVICE inline int ElemIndex(int particle, int comp, int n_particles,
                                      int vdim, Ordering::Type ordering)
{
   if (ordering == Ordering::byNODES) { return (comp*n_particles) + particle; }
   else { return particle*vdim + comp; }
}

//----------------------------------------------------------------------------

/** @brief Field Class for mfem::Array

    @tparam T  Element type */
template <typename T>
class ArrayField : public Field
{
   mfem::Array<T> *arr;
   int vdim;
   Ordering::Type ordering;

public:
   /** @brief Array -> ArrayField constructor

       @param[in,out] a          Array to wrap
       @param[in]     vdim_      Number of values per particle
       @param[in]     ordering_  Layout of the values in @a a */
   ArrayField(mfem::Array<T> &a, int vdim_ = 1,
              Ordering::Type ordering_ = Ordering::byNODES)
      : arr(&a), vdim(vdim_), ordering(ordering_) { }

   /// @copydoc Field::GetNumParticles
   int GetNumParticles() const override  { return arr->Size() / vdim; }
   /// @copydoc Field::GetParticleBytes
   std::size_t GetParticleBytes() const override
   { return (std::size_t)vdim * sizeof(T); }
   /// @copydoc Field::GetAlign
   std::size_t GetAlign() const override { return alignof(T); }
   /// @copydoc Field::OnDevice
   bool OnDevice() const override
   { return Device::IsEnabled() && arr->UseDevice(); }

   /// @copydoc Field::Pack
   void Pack(const mfem::Array<int> &leaving_idx, char *buf, std::size_t stride,
             std::size_t off) const override
   {
      const int n_leaving = leaving_idx.Size();
      // if no particles leave, nothing to pack into movement buffer
      if (n_leaving == 0) { return; }
      const bool use_dev = OnDevice();
      const int n_particles = GetNumParticles();
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;
      const T *old_data = arr->Read(use_dev);
      const int *leaving_idx_data = leaving_idx.Read(use_dev);
      mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k)
      {
         T *row = reinterpret_cast<T*>(buf + (std::size_t)k*stride + off);
         const int particle = leaving_idx_data[k];
         for (int comp = 0; comp < vd; comp++)
         {
            row[comp] = old_data[ElemIndex(particle, comp, n_particles, vd,
                                           col_ordering)];
         }
      });
   }

   /// @copydoc Field::Unpack
   void Unpack(const mfem::Array<int> &dst, const mfem::Array<int> &tail,
               const char *buf, int n_arrivals, int n_new,
               std::size_t stride, std::size_t off) override
   {
      const bool use_dev = OnDevice();
      const int n = GetNumParticles();
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;
      const std::size_t need = (std::size_t)n_new*vd;
      MFEM_VERIFY(need <= (std::size_t)INT_MAX,
                  "CrystalRouter: column too large");
      const std::size_t old_cap = (std::size_t)arr->Capacity();
      const bool in_place = (ordering == Ordering::byVDIM || vd == 1 ||
                             n_new == n) && need <= old_cap;

      T *out = nullptr;
      const T *src = nullptr;

      mfem::Array<T> new_col;
      if (in_place)
      {
         const int max_n = std::max(n, n_new);
         arr->SetSize(max_n*vd);
         if (max_n*vd > 0) { out = arr->ReadWrite(use_dev); src = out; }
      }
      else
      {
         // capacity never shrinks
         std::size_t cap = old_cap;
         if (need > old_cap)
         {
            cap = std::min(std::max(need, 2*old_cap), (std::size_t)INT_MAX);
         }
         new_col.SetSize((int)cap, arr->GetMemory().GetMemoryType());
         new_col.SetSize(n_new*vd);
         new_col.GetMemory().UseDevice(arr->UseDevice());
         if (n > 0) { src = arr->Read(use_dev); }
         if (n_new > 0) { out = new_col.Write(use_dev); }
         const int n_keep = std::min(n, n_new);
         if (n_keep > 0)
         {
            mfem::forall_switch(use_dev, n_keep, [=] MFEM_HOST_DEVICE (int i)
            {
               for (int comp = 0; comp < vd; comp++)
               {
                  out[ElemIndex(i, comp, n_new, vd, col_ordering)] =
                     src[ElemIndex(i, comp, n, vd, col_ordering)];
               }
            });
         }
      }

      const int n_fill = dst.Size();
      if (n_fill > 0)
      {
         const int *dst_data = dst.Read(use_dev);
         const int *tail_data = tail.Size() > 0 ? tail.Read(use_dev) : nullptr;
         mfem::forall_switch(use_dev, n_fill, [=] MFEM_HOST_DEVICE (int m)
         {
            for (int comp = 0; comp < vd; comp++)
            {
               T v;
               if (m < n_arrivals)
               {
                  const char *row = buf + (std::size_t)m*stride + off;
                  v = reinterpret_cast<const T*>(row)[comp];
               }
               else
               {
                  const int t = tail_data[m - n_arrivals];
                  v = src[ElemIndex(t, comp, n, vd, col_ordering)];
               }
               out[ElemIndex(dst_data[m], comp, n_new, vd, col_ordering)] = v;
            }
         });
      }

      if (in_place) { arr->SetSize((int)need); }
      else { arr->Swap(new_col); }
   }
};



//----------------------------------------------------------------------------



/** @brief Field Class for mfem::Vector */
class VectorField : public Field
{
   mfem::Vector *vec;
   int vdim;
   Ordering::Type ordering;

public:
   /** @brief mfem::Vector -> CR::VectorField constructor

       @param[in,out] v          Vector to wrap
       @param[in]     vdim_      Number of values per particle
       @param[in]     ordering_  Layout of the values in @a v */
   VectorField(mfem::Vector &v, int vdim_ = 1,
               Ordering::Type ordering_ = Ordering::byNODES)
      : vec(&v), vdim(vdim_), ordering(ordering_) { }

   /// @copydoc Field::GetNumParticles
   int GetNumParticles() const override { return vec->Size() / vdim; }
   /// @copydoc Field::GetParticleBytes
   std::size_t GetParticleBytes() const override
   { return (std::size_t)vdim * sizeof(real_t); }
   /// @copydoc Field::GetAlign
   std::size_t GetAlign() const override { return alignof(real_t); }
   /// @copydoc Field::OnDevice
   bool OnDevice() const override
   { return Device::IsEnabled() && vec->UseDevice(); }


   /// @copydoc Field::Pack
   void Pack(const mfem::Array<int> &leaving_idx, char *buf,
             std::size_t stride, std::size_t off) const override
   {
      const int n_leaving = leaving_idx.Size();
      if (n_leaving == 0) { return; }
      const bool use_dev = OnDevice();
      const int n_particles = GetNumParticles();
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      const real_t *old_data = vec->Read(use_dev);
      const int *leaving_idx_data = leaving_idx.Read(use_dev);

      mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k)
      {
         real_t *row =
            reinterpret_cast<real_t*>(buf + (std::size_t)k*stride + off);
         const int particle = leaving_idx_data[k];
         for (int comp = 0; comp < vd; comp++)
         {
            row[comp] = old_data[ElemIndex(particle, comp, n_particles, vd,
                                           col_ordering)];
         }
      });
   }

   /// @copydoc Field::Unpack
   void Unpack(const mfem::Array<int> &dst, const mfem::Array<int> &tail,
               const char *buf, int n_arrivals, int n_new,
               std::size_t stride, std::size_t off) override
   {
      const bool use_dev = OnDevice();
      const int n = GetNumParticles();
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      const std::size_t need = (std::size_t)n_new*vd;
      MFEM_VERIFY(need <= (std::size_t)INT_MAX,
                  "CrystalRouter: column too large");
      const std::size_t old_cap = (std::size_t)vec->Capacity();
      const bool in_place = (ordering == Ordering::byVDIM || vd == 1 ||
                             n_new == n) && need <= old_cap;

      real_t *out = nullptr;
      const real_t *src = nullptr;

      mfem::Vector new_col;
      if (in_place)
      {
         const int max_n = std::max(n, n_new);
         vec->SetSize(max_n*vd);
         if (max_n*vd > 0) { out = vec->ReadWrite(use_dev); src = out; }
      }
      else
      {
         // capacity never shrinks
         std::size_t cap = old_cap;
         if (need > old_cap)
         {
            cap = std::min(std::max(need, 2*old_cap), (std::size_t)INT_MAX);
         }
         new_col.SetSize((int)cap, vec->GetMemory().GetMemoryType());
         new_col.SetSize(n_new*vd);
         new_col.UseDevice(vec->UseDevice());
         if (n > 0) { src = vec->Read(use_dev); }
         if (n_new > 0) { out = new_col.Write(use_dev); }
         const int n_keep = std::min(n, n_new);
         if (n_keep > 0)
         {
            mfem::forall_switch(use_dev, n_keep, [=] MFEM_HOST_DEVICE (int i)
            {
               for (int comp = 0; comp < vd; comp++)
               {
                  out[ElemIndex(i, comp, n_new, vd, col_ordering)] =
                     src[ElemIndex(i, comp, n, vd, col_ordering)];
               }
            });
         }
      }

      const int n_fill = dst.Size();
      if (n_fill > 0)
      {
         const int *dst_data = dst.Read(use_dev);
         const int *tail_data = tail.Size() > 0 ? tail.Read(use_dev) : nullptr;
         mfem::forall_switch(use_dev, n_fill, [=] MFEM_HOST_DEVICE (int m)
         {
            for (int comp = 0; comp < vd; comp++)
            {
               real_t v;
               if (m < n_arrivals)
               {
                  const char *row = buf + (std::size_t)m*stride + off;
                  v = reinterpret_cast<const real_t*>(row)[comp];
               }
               else
               {
                  const int t = tail_data[m - n_arrivals];
                  v = src[ElemIndex(t, comp, n, vd, col_ordering)];
               }
               out[ElemIndex(dst_data[m], comp, n_new, vd, col_ordering)] = v;
            }
         });
      }

      if (in_place) { vec->SetSize((int)need); }
      else { vec->Swap(new_col); }
   }
};





//----------------------------------------------------------------------------





/** @brief One or more Fields grouped together as a single argument to
    CrystalRouter::Route()
*/
struct FieldGroup
{
   std::vector<std::shared_ptr<Field>> cols;
};

} // namespace internal



/** @name Public wrappers for mfem type -> Field for CrystalRouter::Route()

    External wrappers mfem::Type -> CR::TypeField -> CR::FieldGroup
    Route() uses one type, FieldGroup */
///@{
template <typename T>
internal::FieldGroup Array(mfem::Array<T> &a, int vdim = 1,
                           Ordering::Type ordering = Ordering::byNODES)
{
   return {{std::make_shared<internal::ArrayField<T>>(a, vdim, ordering)}};
}

inline internal::FieldGroup Vector(mfem::Vector &v, int vdim = 1,
                                   Ordering::Type ordering = Ordering::byNODES)
{
   return {{std::make_shared<internal::VectorField>(v, vdim, ordering)}};
}

inline internal::FieldGroup ParticleVector(mfem::ParticleVector &pv)
{
   return Vector(pv, pv.GetVDim(), pv.GetOrdering());
}

inline internal::FieldGroup ParticleVector(
   const std::vector<std::unique_ptr<mfem::ParticleVector>> &pvs)
{
   internal::FieldGroup g;
   for (const auto &p : pvs)
   {
      g.cols.push_back(std::make_shared<internal::VectorField>(
                          *p, p->GetVDim(), p->GetOrdering()));
   }
   return g;
}

/** @brief Wrap a list of int arrays as one FieldGroup for ParticleSet

    ParticleSet Tags are `std::vector<std::unique_ptr<Array<int>>>`,
    overloader function to convert it into FieldGroup

    @param[in] arrs  Arrays to wrap, one value per particle each
    @return The FieldGroup holding one field per array */
inline internal::FieldGroup Array(
   const std::vector<std::unique_ptr<mfem::Array<int>>> &arrs)
{
   internal::FieldGroup g;
   for (const auto &a : arrs)
   {
      g.cols.push_back(std::make_shared<internal::ArrayField<int>>(*a));
   }
   return g;
}
///@}

} // namespace CR



/** @brief Route particle data to destination MPI ranks with the crystal router
    algorithm

    Each particle carries a destination rank. Route() sends every particle, with
    its data from all given fields, to that rank in O(log P) communication
    rounds, where P is the number of ranks. After the call, each rank holds
    exactly the particles addressed to it.

    The order of the particles after routing is not specified

    TODO(docs): order changed in the single-buffer router. Staying particles
    below the new count keep their index. Arrivals fill the holes of leaving
    particles, and staying particles past the new count move into the other
    holes. Replace this note with the real documentation.

    @note Route() is collective: every rank in the communicator must call it */
class CrystalRouter
{
public:
   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();

   /// Copying is not allowed, the router owns an MPI communicator
   CrystalRouter(const CrystalRouter&) = delete;
   CrystalRouter& operator=(const CrystalRouter&) = delete;

   /** @brief Send each particle to its destination rank

       @param[in]     rank_list     Destination rank of each particle
       @param[in,out] field_groups  Data to route, as a brace-enclosed list of
                                    CR::Array(), CR::Vector() and
                                    CR::ParticleVector() wrappers. Each field
                                    must have rank_list.Size() particles.

       @note Collective on the communicator of the router */
   void Route(const Array<unsigned int> &rank_list,
              std::initializer_list<CR::internal::FieldGroup> field_groups);

private:
   MPI_Comm comm;
   int rank;
   int nprocs;

   /** Packed rows, one row per particle: the packed columns, then the
       destination rank (unsigned int). Its size is in 8-byte words. */
   Array<std::uint64_t> buf;

   /** @brief Run the crystal router rounds on the rows in #buf

       @param[in] nrows   Number of rows in #buf
       @param[in] stride  Size of one row in bytes, a multiple of 8
       @param[in] hdr     Byte offset of the destination rank in a row
       @param[in] row_t   MPI type of one row (@a stride bytes)
       @return Number of rows in #buf after routing. All of them are for this
               rank. */
   int RouteInternal(int nrows, std::size_t stride, std::size_t hdr,
                     MPI_Datatype row_t);

   /** @brief Partition the rows of #buf in place. The rows that cross the
       cutoff go to the end.

       @param[in] nrows   Number of rows in #buf
       @param[in] stride  Size of one row in bytes
       @param[in] hdr     Byte offset of the destination rank in a row
       @param[in] cutoff  First rank of the upper half of the current group of
                          ranks
       @param[in] send_hi If true, send rows with rank >= @a cutoff, otherwise
                          send those with rank < @a cutoff
       @return Number of rows that stay (they are the first rows of #buf) */
   int Move(int nrows, std::size_t stride, std::size_t hdr,
            unsigned int cutoff, bool send_hi);

   /** @brief Send the last @a nsend rows of #buf to a partner rank and
       receive rows from it

       The received rows replace the sent rows. If more rows arrive than were
       sent, #buf grows.

       @param[in] nkeep    Number of kept rows at the front of #buf
       @param[in] nsend    Number of rows to send, they follow the kept rows
       @param[in] stride   Size of one row in bytes
       @param[in] row_t    MPI type of one row
       @param[in] target   Rank to send to, and to receive from
       @param[in] recvn    Number of messages to receive: 0, 1, or 2 (the
                           second one comes from rank - 1)
       @param[in] msg_tag  Tag of the count message. The data message uses
                           @a msg_tag + 1.
       @return Number of rows in #buf: the kept rows, then the received rows */
   int Exchange(int nkeep, int nsend, std::size_t stride, MPI_Datatype row_t,
                int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
