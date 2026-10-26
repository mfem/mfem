#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
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
/** @brief Abstract interface for one data column routed by CrystalRouter

    A Field is a column of a particle attribute (coordinates, id, tag, etc.)
    stored for all particles. It wraps over an MFEM container to be used for
    routing. Subclasses implement specific functionalities for routing.
    Each row slice of a field is the attribute that belongs to one particle.
*/
class Field
{
public:
   virtual ~Field() = default;
   virtual int GetNumParticles() const = 0;
   virtual std::size_t GetParticleBytes() const = 0;

   /** @brief Copy the data of particle @a i into a byte buffer

       @param[in]  i    Index of the particle to copy
       @param[out] dst  Destination buffer */
   virtual void Pack(int i, char *dst) const = 0;

   /** @brief Copy a byte buffer into the data of particle @a i

       @param[in] i    Index of the particle to overwrite
       @param[in] src  Source buffer */
   virtual void Unpack(int i, const char *src) = 0;

   /** @brief Copy the data of particle @a src onto particle @a dst

       @param[in] src  Index of the particle to read
       @param[in] dst  Index of the particle to overwrite */
   virtual void Copy(int src, int dst) = 0;

   /** @brief Change the number of particles to @a n

       Existing particles keep their data and their order,
       new particle slots are initialized to zero.

       @param[in] n  New number of particles */
   virtual void SetNumParticles(int n) = 0;

   /** @brief Copy the leaving device particles into a new field on host

       @param[in] leaving_idx  Indices of the particles to copy
       @return A new field with leaving_idx.Size() particles */
   virtual std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) = 0;

   /** @brief Rebuild this field from the staying particles and the arrivals

       @param[in]     staying_idx  Indices of the particles that did not leave
       @param[in,out] arrivals     Field of received particles, must have the
                                   same type as this field */
   virtual void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) = 0;
};

namespace internal
{

/** @brief Return the position of one component of a particle in a flat array,
    used internally for indexing

    @param[in] particle     Particle index
    @param[in] comp         Component index, in [0, @a vdim)
    @param[in] n_particles  Total number of particles stored
    @param[in] vdim         Number of components per particle
    @param[in] ordering     Ordering::byNODES or Ordering::byVDIM */
MFEM_HOST_DEVICE inline int ElemIndex(int particle, int comp, int n_particles,
                                      int vdim, Ordering::Type ordering) {
   if(ordering == Ordering::byNODES){ return (comp*n_particles) + particle; }
   else { return particle*vdim + comp; }
}

/** @brief Shrink an Ordering::byNODES container in place from @a old_n to @a new_n

    @tparam C  Container with operator[] (Array, Vector, ...)
    @param[in,out] cont   Container to compact
    @param[in]     old_n  Current number of particles
    @param[in]     new_n  New number of particles, at most @a old_n
    @param[in]     vdim   Number of components per particle */
template <typename C>
void CompactNodes(C &cont, int old_n, int new_n, int vdim) {
   for (int c = 1; c < vdim; c++) {
      for (int i = 0; i < new_n; i++) {
         cont[c*new_n + i] = cont[c*old_n + i];
      }
   }
}

/** @brief Grow an Ordering::byNODES container in place from @a old_n to @a new_n

    @tparam C  Container with operator[] (Array, Vector, ...)
    @tparam E  Type of the fill value
    @param[in,out] cont   Container to spread
    @param[in]     old_n  Current number of particles
    @param[in]     new_n  New number of particles, at least @a old_n
    @param[in]     vdim   Number of components per particle
    @param[in]     fill   Value written into the new slots */
template <typename C, typename E>
void SpreadNodes(C &cont, int old_n, int new_n, int vdim, const E &fill) {
   for (int c = vdim-1; c > 0; c--) {
      for (int i = old_n-1; i >= 0; i--) {
         cont[c*new_n + i] = cont[c*old_n + i];
      }
   }
   for (int c = 0; c < vdim; c++) {
      for (int i = old_n; i < new_n; i++) {
         cont[c*new_n + i] = fill;
      }
   }
}

} // namespace internal

///----------------------------------------------------------------------------------------------

/** @brief Field that wraps an mfem::Array

    @tparam T  Element type */
template <typename T>
class ArrayField : public Field
{
   mfem::Array<T> *arr;
   int vdim;
   Ordering::Type ordering;
   std::shared_ptr<mfem::Array<T>> own;

public:
   /** @brief Wrap an array as a Field

       @param[in,out] a          Array to wrap
       @param[in]     vdim_      Number of values per particle
       @param[in]     ordering_  Layout of the values in @a a */
   ArrayField(mfem::Array<T> &a, int vdim_ = 1,
              Ordering::Type ordering_ = Ordering::byNODES)
      : arr(&a), vdim(vdim_), ordering(ordering_) { }

   /// @copydoc Field::GetNumParticles
   int              GetNumParticles() const override  { return arr->Size() / vdim; }
   /// @copydoc Field::GetParticleBytes
   std::size_t      GetParticleBytes() const override { return (std::size_t)vdim * sizeof(T); }

   /// @copydoc Field::ToHost
   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && arr->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_leaving = leaving_idx.Size();
      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<mfem::Array<T>>(n_leaving*n_comps);
      host_buf->GetMemory().UseDevice(use_dev);
      if (n_leaving > 0) {
         const T *col_data = arr->Read(use_dev);
         const int *leaving = leaving_idx.Read(use_dev);
         T *host_buf_data = host_buf->Write(use_dev);

         /// Loops to copy every device particle col_data into host buffer host_buf_data
         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving[k];
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, n_comps, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, n_comps, col_ordering);
               host_buf_data[to] = col_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->GetMemory().UseDevice(false);

      auto host_field = std::make_unique<ArrayField<T>>(*host_buf, n_comps, col_ordering);
      host_field->own = host_buf;
      return host_field;
   }


   /// @copydoc Field::ToDevice
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Array<T> &arrival_arr = *static_cast<ArrayField<T>&>(arrivals).arr;
      const bool use_dev = Device::IsEnabled() && arr->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_arr.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      mfem::Array<T> new_col(std::max(arr->Capacity(), new_n_particles*n_comps),
                             arr->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*n_comps);
      new_col.GetMemory().UseDevice(arr->UseDevice());
      if (new_n_particles > 0) {
         const T *col_data = n_staying > 0 ? arr->Read(use_dev) : nullptr;
         const int *staying = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const T *arrival_data = n_arrivals > 0 ? arrival_arr.Read(use_dev) : nullptr;
         T *new_data = new_col.Write(use_dev);
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles,
                                                  n_comps, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying[slot], comp,
                                                       n_particles, n_comps,
                                                       col_ordering);
                  new_data[to] = col_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, n_comps, col_ordering);
                  new_data[to] = arrival_data[from];
               }
            }
         });
      }
      arr->Swap(new_col);
   }

   /// @copydoc Field::Pack
   void Pack(int i, char *dst) const override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         const T v = (*arr)[internal::ElemIndex(i, c, n, vdim, ordering)];
         std::memcpy(dst + (std::size_t)c*sizeof(T), &v, sizeof(T));
      }
   }
   /// @copydoc Field::Unpack
   void Unpack(int i, const char *src) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         T v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(T), sizeof(T));
         (*arr)[internal::ElemIndex(i, c, n, vdim, ordering)] = v;
      }
   }
   /// @copydoc Field::Copy
   void Copy(int src, int dst) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         (*arr)[internal::ElemIndex(dst, c, n, vdim, ordering)] =
            (*arr)[internal::ElemIndex(src, c, n, vdim, ordering)];
      }
   }
   /// @copydoc Field::SetNumParticles
   void SetNumParticles(int n) override {
      const int old_n = GetNumParticles();
      if (n == old_n) { return; }
      if (n < old_n) {
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::CompactNodes(*arr, old_n, n, vdim);
         }
         arr->SetSize(n*vdim);
      }
      else {
         arr->SetSize(n*vdim, T());
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::SpreadNodes(*arr, old_n, n, vdim, T());
         }
      }
   }
};




/** @brief Field that wraps an mfem::Vector of real_t values

    The vector holds @a vdim consecutive values per particle, laid out
    according to @a ordering. The vector is not owned (unless this field was
    created by ToHost()) and must outlive the field. */
class VectorField : public Field
{
   /// Wrapped vector, not owned
   mfem::Vector *vec;
   /// Number of values per particle
   int vdim;
   /// Layout of the values, byNODES or byVDIM
   Ordering::Type ordering;
   /// Keeps the storage alive when this field wraps a ToHost() buffer
   std::shared_ptr<mfem::Vector> own;

public:
   /** @brief Wrap a vector as a Field

       @param[in,out] v          Vector to wrap. Must outlive this field.
       @param[in]     vdim_      Number of values per particle
       @param[in]     ordering_  Layout of the values in @a v */
   VectorField(mfem::Vector &v, int vdim_ = 1,
               Ordering::Type ordering_ = Ordering::byNODES)
      : vec(&v), vdim(vdim_), ordering(ordering_) { }

   /// @copydoc Field::GetNumParticles
   int              GetNumParticles() const override           { return vec->Size() / vdim; }
   /// @copydoc Field::GetParticleBytes
   std::size_t      GetParticleBytes() const override       { return (std::size_t)vdim * sizeof(real_t); }

   /// @copydoc Field::ToHost
   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && vec->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_leaving = leaving_idx.Size();
      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<mfem::Vector>(n_leaving*n_comps);
      host_buf->UseDevice(use_dev);
      if (n_leaving > 0) {
         const real_t *col_data = vec->Read(use_dev);
         const int *leaving = leaving_idx.Read(use_dev);
         real_t *host_buf_data = host_buf->Write(use_dev);

         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving[k];
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, n_comps, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, n_comps, col_ordering);
               host_buf_data[to] = col_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->UseDevice(false);

      auto host_field = std::make_unique<VectorField>(*host_buf, n_comps, col_ordering);
      host_field->own = host_buf;
      return host_field;
   }


   /// @copydoc Field::ToDevice
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Vector &arrival_vec = *static_cast<VectorField&>(arrivals).vec;
      const bool use_dev = Device::IsEnabled() && vec->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_vec.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      mfem::Vector new_col;
      // keep the old capacity, only the size shrinks
      new_col.SetSize(std::max(vec->Capacity(), new_n_particles*n_comps),
                      vec->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*n_comps);
      new_col.UseDevice(vec->UseDevice());
      if (new_n_particles > 0) {
         const real_t *col_data = n_staying > 0 ? vec->Read(use_dev) : nullptr;
         const int *staying = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const real_t *arrival_data = n_arrivals > 0 ? arrival_vec.Read(use_dev) : nullptr;
         real_t *new_data = new_col.Write(use_dev);
         // staying -> kept in original order; arrivals -> appended to the end
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles, n_comps, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying[slot], comp, n_particles, n_comps, col_ordering);
                  new_data[to] = col_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, n_comps, col_ordering);
                  new_data[to] = arrival_data[from];
               }
            }
         });
      }
      vec->Swap(new_col);
   }

   /// @copydoc Field::Pack
   void Pack(int i, char *dst) const override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         const real_t v = (*vec)[internal::ElemIndex(i, c, n, vdim, ordering)];
         std::memcpy(dst + (std::size_t)c*sizeof(real_t), &v, sizeof(real_t));
      }
   }
   /// @copydoc Field::Unpack
   void Unpack(int i, const char *src) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         real_t v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(real_t), sizeof(real_t));
         (*vec)[internal::ElemIndex(i, c, n, vdim, ordering)] = v;
      }
   }
   /// @copydoc Field::Copy
   void Copy(int src, int dst) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         (*vec)[internal::ElemIndex(dst, c, n, vdim, ordering)] =
            (*vec)[internal::ElemIndex(src, c, n, vdim, ordering)];
      }
   }
   /// @copydoc Field::SetNumParticles
   void SetNumParticles(int n) override {
      const int old_n = GetNumParticles();
      if (n == old_n) { return; }
      if (n < old_n) {
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::CompactNodes(*vec, old_n, n, vdim);
         }
         vec->SetSize(n*vdim);
      }
      else {
         const int old_sz = old_n*vdim, new_sz = n*vdim;
         if (new_sz > vec->Capacity()) {
            mfem::Vector tmp;
            tmp.SetSize(new_sz, vec->GetMemory().GetMemoryType());
            tmp.UseDevice(vec->UseDevice());
            if (old_sz > 0) {
               std::memcpy(tmp.GetData(), vec->GetData(),
                           (std::size_t)old_sz*sizeof(real_t));
            }
            for (int k = old_sz; k < new_sz; k++) { tmp[k] = 0.0; }
            vec->Swap(tmp);
         }
         else {
            vec->SetSize(new_sz);
            for (int k = old_sz; k < new_sz; k++) { (*vec)[k] = 0.0; }
         }
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::SpreadNodes(*vec, old_n, n, vdim, (real_t)0.0);
         }
      }
   }
};





/** @brief Field that wraps an mfem::ParticleVector

    The vdim and ordering are read from the ParticleVector itself. The
    ParticleVector is not owned (unless this field was created by ToHost()) and
    must outlive the field. */
class ParticleVectorField : public Field
{
   /// Wrapped particle vector, not owned
   mfem::ParticleVector *pv;
   /// Keeps the storage alive when this field wraps a ToHost() buffer
   std::shared_ptr<mfem::ParticleVector> own;

public:
   /** @brief Wrap a ParticleVector as a Field

       @param[in,out] pv_  ParticleVector to wrap. Must outlive this field. */
   ParticleVectorField(mfem::ParticleVector &pv_) : pv(&pv_) { }

   /// @copydoc Field::GetNumParticles
   int              GetNumParticles() const override           { return pv->GetNumParticles(); }
   /// @copydoc Field::GetParticleBytes
   std::size_t      GetParticleBytes() const override       { return (std::size_t)pv->GetVDim() * sizeof(real_t); }

   /// @copydoc Field::ToHost
   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && pv->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_leaving = leaving_idx.Size();
      const int n_comps = pv->GetVDim();
      const Ordering::Type col_ordering = pv->GetOrdering();

      auto host_buf = std::make_shared<mfem::ParticleVector>(n_comps, col_ordering, n_leaving);
      host_buf->UseDevice(use_dev);
      if (n_leaving > 0) {
         const real_t *col_data = pv->Read(use_dev);
         const int *leaving = leaving_idx.Read(use_dev);
         real_t *host_buf_data = host_buf->Write(use_dev);

         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving[k];
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, n_comps, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, n_comps, col_ordering);
               host_buf_data[to] = col_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->UseDevice(false);

      auto host_field = std::make_unique<ParticleVectorField>(*host_buf);
      host_field->own = host_buf;
      return host_field;
   }


   /// @copydoc Field::ToDevice
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::ParticleVector &arrival_pv = *static_cast<ParticleVectorField&>(arrivals).pv;
      const bool use_dev = Device::IsEnabled() && pv->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_pv.GetNumParticles();
      const int new_n_particles = n_staying + n_arrivals;

      const int n_comps = pv->GetVDim();
      const Ordering::Type col_ordering = pv->GetOrdering();

      mfem::Vector new_col;
      new_col.SetSize(std::max(pv->Capacity(), new_n_particles*n_comps),
                      pv->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*n_comps);
      new_col.UseDevice(pv->UseDevice());
      if (new_n_particles > 0) {
         const real_t *col_data = n_staying > 0 ? pv->Read(use_dev) : nullptr;
         const int *staying = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const real_t *arrival_data = n_arrivals > 0 ? arrival_pv.Read(use_dev) : nullptr;
         real_t *new_data = new_col.Write(use_dev);
         // staying -> kept in original order; arrivals -> appended to the end
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < n_comps; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles, n_comps, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying[slot], comp, n_particles, n_comps, col_ordering);
                  new_data[to] = col_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, n_comps, col_ordering);
                  new_data[to] = arrival_data[from];
               }
            }
         });
      }
      // swaps data + size only; vdim/ordering stay on pv
      pv->Swap(new_col);
   }

   /// @copydoc Field::Pack
   void Pack(int i, char *dst) const override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) {
         const real_t v = (*pv)(i, c);
         std::memcpy(dst + (std::size_t)c*sizeof(real_t), &v, sizeof(real_t));
      }
   }
   /// @copydoc Field::Unpack
   void Unpack(int i, const char *src) override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) {
         real_t v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(real_t), sizeof(real_t));
         (*pv)(i, c) = v;
      }
   }
   /// @copydoc Field::Copy
   void Copy(int src, int dst) override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) { (*pv)(dst, c) = (*pv)(src, c); }
   }
   /// @copydoc Field::SetNumParticles
   void SetNumParticles(int n) override { pv->SetNumParticles(n, true); }
};





/** @brief Wrap an mfem::Array as a Field for CrystalRouter::Route()

    @tparam T  Element type. Must be trivially copyable.
    @param[in,out] a         Array to wrap. Must outlive the returned field.
    @param[in]     vdim      Number of values per particle
    @param[in]     ordering  Layout of the values in @a a
    @return The wrapping ArrayField */
template <typename T>
ArrayField<T> Array(mfem::Array<T> &a, int vdim = 1,
                    Ordering::Type ordering = Ordering::byNODES) {
   return ArrayField<T>(a, vdim, ordering);
}

/** @brief Wrap an mfem::Vector as a Field for CrystalRouter::Route()

    @param[in,out] v         Vector to wrap. Must outlive the returned field.
    @param[in]     vdim      Number of values per particle
    @param[in]     ordering  Layout of the values in @a v
    @return The wrapping VectorField */
inline VectorField Vector(mfem::Vector &v, int vdim = 1,
                          Ordering::Type ordering = Ordering::byNODES) {
   return VectorField(v, vdim, ordering);
}

/** @brief Wrap an mfem::ParticleVector as a Field for CrystalRouter::Route()

    @param[in,out] pv  ParticleVector to wrap. Must outlive the returned field.
    @return The wrapping ParticleVectorField */
inline ParticleVectorField ParticleVector(mfem::ParticleVector &pv) {
   return ParticleVectorField(pv);
}



/** @brief One or more Fields passed together as a single argument to
    CrystalRouter::Route()

    FieldGroup lets Route() take a mixed brace-enclosed list of single fields
    and lists of fields. All constructors are implicit on purpose. Every field
    in a group must have the same number of particles. */
class FieldGroup
{
public:
   /// The fields in this group, in routing order
   std::vector<std::shared_ptr<Field>> cols;

   /// Create a group holding a copy of one ArrayField
   template <typename T>
   FieldGroup(const ArrayField<T> &f) : cols{std::make_shared<ArrayField<T>>(f)} { }
   /// Create a group holding a copy of one VectorField
   FieldGroup(const VectorField &f) : cols{std::make_shared<VectorField>(f)} { }
   /// Create a group holding a copy of one ParticleVectorField
   FieldGroup(const ParticleVectorField &f) : cols{std::make_shared<ParticleVectorField>(f)} { }
   /** @brief Create a group with one ParticleVectorField per ParticleVector

       @param[in] v  ParticleVectors to wrap. Must outlive the group. */
   FieldGroup(const std::vector<std::unique_ptr<mfem::ParticleVector>> &v) {
      cols.reserve(v.size());
      for (const auto &p : v) {
         cols.push_back(std::make_shared<ParticleVectorField>(*p));
      }
   }
   /** @brief Create a group with one single-component int ArrayField per
       array

       @param[in] v  Arrays to wrap. Must outlive the group. */
   FieldGroup(const std::vector<std::unique_ptr<mfem::Array<int>>> &v) {
      cols.reserve(v.size());
      for (const auto &p : v) {
         cols.push_back(std::make_shared<ArrayField<int>>(*p, 1, Ordering::byNODES));
      }
   }
};


/** @brief Wrap a list of ParticleVectors as one FieldGroup

    @param[in] pvs  ParticleVectors to wrap. Must outlive the returned group.
    @return The FieldGroup holding one field per ParticleVector */
inline FieldGroup ParticleVector(
   const std::vector<std::unique_ptr<mfem::ParticleVector>> &pvs) {
   return FieldGroup(pvs);
}

/** @brief Wrap a list of int arrays as one FieldGroup

    @param[in] arrs  Arrays to wrap, one value per particle each. Must outlive
                     the returned group.
    @return The FieldGroup holding one field per array */
inline FieldGroup Array(
   const std::vector<std::unique_ptr<mfem::Array<int>>> &arrs) {
   return FieldGroup(arrs);
}

} // namespace CR



/** @brief Route particle data to destination MPI ranks with the crystal router
    algorithm

    Each particle carries a destination rank. Route() sends every particle, with
    its data from all given fields, to that rank in O(log P) communication
    rounds, where P is the number of ranks. After the call, each rank holds
    exactly the particles addressed to it.

    The order of the particles after routing is not specified

    @note Route() is collective: every rank in the communicator must call it */
class CrystalRouter
{
public:
   /** @brief Create a router on a communicator

       The communicator is duplicated, so messages of the router cannot clash
       with other traffic

       @param[in] comm  MPI communicator. Collective call. */
   CrystalRouter(MPI_Comm comm);

   /// Free the duplicated communicator (skipped if MPI is already finalized)
   ~CrystalRouter();

   /// Copying is not allowed, the router owns an MPI communicator
   CrystalRouter(const CrystalRouter&) = delete;
   /// Copying is not allowed, the router owns an MPI communicator
   CrystalRouter& operator=(const CrystalRouter&) = delete;

   /** @brief Send each particle to its destination rank

       All fields are resized to the new number of particles. When a mfem::Device
       is enabled, only the leaving particles are copied to the host and only
       the arrivals are copied back.

       @param[in]     rank_list  Destination rank of each particle
       @param[in,out] fields     Data to route, as a brace-enclosed list of
                                 CR::Array(), CR::Vector() and
                                 CR::ParticleVector() wrappers. Each field
                                 must have rank_list.Size() particles.

       @note Collective on the communicator of the router */
   void Route(const Array<unsigned int> &rank_list,
              std::initializer_list<CR::FieldGroup> fields);

private:
   /// Private duplicate of the user's communicator
   MPI_Comm comm;
   /// Rank of this process in #comm
   int rank;
   /// Number of processes in #comm
   int nprocs;

   /// Byte buffer holding the rows that leave in the current round
   Array<char> send_buf;
   /// Byte buffer holding the rows received in the current round
   Array<char> recv_buf;

   /** @brief Run the crystal router rounds on host data

       @param[in,out] ranks   Destination rank of each particle. Updated to
                              match the fields after routing.
       @param[in,out] fields  Fields to route, all of size ranks.Size() */
   void RouteInternal(Array<unsigned int> &ranks,
                      const std::vector<CR::Field*> &fields);

   /** @brief Route fields that may live on a device

       Moves only the leaving particles to the host, routes them with
       RouteInternal(), and moves the arrivals back to the device

       @param[in]     rank_list  Destination rank of each particle
       @param[in,out] fields     Fields to route */
   void RouteDevice(const Array<unsigned int> &rank_list,
                    const std::vector<CR::Field*> &fields);

   /** @brief Pack the particles that cross the cutoff into #send_buf and
       remove them from the fields

       The remaining particles are compacted in place, keeping their order

       @param[in,out] ranks    Destination ranks. Reduced to the kept particles.
       @param[in,out] fields   Fields to split. Reduced to the kept particles.
       @param[in]     cutoff   First rank of the upper half of the current
                               group of ranks
       @param[in]     send_hi  If true, send particles with rank >= @a cutoff,
                               otherwise send those with rank < @a cutoff */
   void Move(Array<unsigned int> &ranks,
             const std::vector<CR::Field*> &fields,
             unsigned int cutoff, bool send_hi);

   /** @brief Send #send_buf to a partner rank and append the received rows to
       the fields

       Counts are exchanged first, then the data. Each row holds the
       destination rank followed by one row per field.

       @param[in,out] ranks    Destination ranks. Received ranks are appended.
       @param[in,out] fields   Fields. Received particles are appended.
       @param[in]     target   Rank to send to, and to receive from
       @param[in]     recvn    Number of messages to receive: 0, 1, or 2 (the
                               second one comes from rank - 1)
       @param[in]     msg_tag  Tag of the count message. The data message uses
                               @a msg_tag + 1. */
   void Exchange(Array<unsigned int> &ranks,
                 const std::vector<CR::Field*> &fields,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
