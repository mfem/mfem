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

    @note currently lots of repeated code and functions, attempting to cut down
          and refactor shared methods in crystal-routing-testing branch
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

//----------------------------------------------------------------------------------------------

/** @brief Field Class for mfem::Array

    @tparam T  Element type */
template <typename T>
class ArrayField : public Field
{
   mfem::Array<T> *arr;
   int vdim;
   Ordering::Type ordering;
   /// keeps a reference to local host_buf alive, if using device
   std::shared_ptr<mfem::Array<T>> own;

public:
   /** @brief Array -> ArrayField constructor

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
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<mfem::Array<T>>(n_leaving*vd);
      host_buf->GetMemory().UseDevice(use_dev);
      if (n_leaving > 0) {
         const T *old_data = arr->Read(use_dev);
         const int *leaving_idx_data = leaving_idx.Read(use_dev);
         T *host_buf_data = host_buf->Write(use_dev);

         // Loops to copy every device particle old_data into host buffer host_buf_data
         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving_idx_data[k];
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, vd, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, vd, col_ordering);
               host_buf_data[to] = old_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->GetMemory().UseDevice(false);

      // return new field, shared own pointer to local host buffer to keep it "alive"
      auto host_field = std::make_unique<ArrayField<T>>(*host_buf, vd, col_ordering);
      host_field->own = host_buf;
      return host_field;
   }


   /** @copydoc Field::ToDevice

       @note n_particles is the *old* count of particles pre routing,
             required for indexing
       @note new_n_particles is the new count of particles including those
             that arrived from router and didnt get moved */
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Array<T> &arrival_arr = *static_cast<ArrayField<T>&>(arrivals).arr;
      const bool use_dev = Device::IsEnabled() && arr->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_arr.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      // keep the old capacity, only the size shrinks
      mfem::Array<T> new_col(std::max(arr->Capacity(), new_n_particles*vd),
                             arr->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*vd);
      new_col.GetMemory().UseDevice(arr->UseDevice());

      // if nothing stays -> local buffer + indices should be empty
      // if nothing arrived -> arriving appending buffer should be empty
      if (new_n_particles > 0) {
         const T *old_data = n_staying > 0 ? arr->Read(use_dev) : nullptr;
         const int *staying_idx_data = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const T *arrival_data = n_arrivals > 0 ? arrival_arr.Read(use_dev) : nullptr;
         T *new_data = new_col.Write(use_dev);
         // staying -> kept in original order
         // arrivals -> appended to the end
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles,
                                                  vd, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying_idx_data[slot], comp,
                                                       n_particles, vd,
                                                       col_ordering);
                  new_data[to] = old_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, vd, col_ordering);
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




/** @brief Field Class for mfem::Vector */
class VectorField : public Field
{
   mfem::Vector *vec;
   int vdim;
   Ordering::Type ordering;
   /// keeps a reference to local host_buf alive, if using device
   std::shared_ptr<mfem::Vector> own;

public:
   /** @brief mfem::Vector -> CR::VectorField constructor

       @param[in,out] v          Vector to wrap
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
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<mfem::Vector>(n_leaving*vd);
      host_buf->UseDevice(use_dev);
      if (n_leaving > 0) {
         const real_t *old_data = vec->Read(use_dev);
         const int *leaving_idx_data = leaving_idx.Read(use_dev);
         real_t *host_buf_data = host_buf->Write(use_dev);

         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving_idx_data[k];
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, vd, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, vd, col_ordering);
               host_buf_data[to] = old_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->UseDevice(false);

      // return new field, shared own pointer to local host buffer to keep it "alive"
      auto host_field = std::make_unique<VectorField>(*host_buf, vd, col_ordering);
      host_field->own = host_buf;
      return host_field;
   }


   /** @copydoc Field::ToDevice

       @note n_particles is the *old* count of particles pre routing,
             required for indexing
       @note new_n_particles is the new count of particles including those
             that arrived from router and didnt get moved */
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Vector &arrival_vec = *static_cast<VectorField&>(arrivals).vec;
      const bool use_dev = Device::IsEnabled() && vec->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_vec.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      mfem::Vector new_col;
      // keep the old capacity, only the size shrinks
      new_col.SetSize(std::max(vec->Capacity(), new_n_particles*vd),
                      vec->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*vd);
      new_col.UseDevice(vec->UseDevice());

      // if nothing stays -> local buffer + indices should be empty
      // if nothing arrived -> arriving appending buffer should be empty
      if (new_n_particles > 0) {
         const real_t *old_data = n_staying > 0 ? vec->Read(use_dev) : nullptr;
         const int *staying_idx_data = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const real_t *arrival_data = n_arrivals > 0 ? arrival_vec.Read(use_dev) : nullptr;
         real_t *new_data = new_col.Write(use_dev);
         // staying -> kept in original order
         // arrivals -> appended to the end
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles, vd, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying_idx_data[slot], comp, n_particles, vd, col_ordering);
                  new_data[to] = old_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, vd, col_ordering);
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





/** @brief Field Class for mfem::ParticleVector  */
class ParticleVectorField : public Field
{
   mfem::ParticleVector *pv;
   /// keeps a reference to local host_buf alive, if using device
   std::shared_ptr<mfem::ParticleVector> own;
redistribute
public:
   /** @brief mfem::PV -> CR::PVField constructor

       @param[in,out] pv_  ParticleVector to wrap */
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
      const int vd = pv->GetVDim();
      const Ordering::Type col_ordering = pv->GetOrdering();

      auto host_buf = std::make_shared<mfem::ParticleVector>(vd, col_ordering, n_leaving);
      host_buf->UseDevice(use_dev);
      if (n_leaving > 0) {
         const real_t *old_data = pv->Read(use_dev);
         const int *leaving_idx_data = leaving_idx.Read(use_dev);
         real_t *host_buf_data = host_buf->Write(use_dev);

         mfem::forall_switch(use_dev, n_leaving, [=] MFEM_HOST_DEVICE (int k) {
            const int particle = leaving_idx_data[k];
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(k, comp, n_leaving, vd, col_ordering);
               const int from = internal::ElemIndex(particle, comp, n_particles, vd, col_ordering);
               host_buf_data[to] = old_data[from];
            }
         });
      }

      host_buf->HostReadWrite();
      host_buf->UseDevice(false);

      // return new field, shared own pointer to local host buffer to keep it "alive"
      auto host_field = std::make_unique<ParticleVectorField>(*host_buf);
      host_field->own = host_buf;
      return host_field;
   }


   /** @copydoc Field::ToDevice

       @note n_particles is the *old* count of particles pre routing,
             required for indexing
       @note new_n_particles is the new count of particles including those
             that arrived from router and didnt get moved */
   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::ParticleVector &arrival_pv = *static_cast<ParticleVectorField&>(arrivals).pv;
      const bool use_dev = Device::IsEnabled() && pv->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_pv.GetNumParticles();
      const int new_n_particles = n_staying + n_arrivals;

      const int vd = pv->GetVDim();
      const Ordering::Type col_ordering = pv->GetOrdering();

      mfem::Vector new_col;
      // keep the old capacity, only the size shrinks
      new_col.SetSize(std::max(pv->Capacity(), new_n_particles*vd),
                      pv->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*vd);
      new_col.UseDevice(pv->UseDevice());

      // if nothing stays -> local buffer + indices should be empty
      // if nothing arrived -> arriving appending buffer should be empty
      if (new_n_particles > 0) {
         const real_t *old_data = n_staying > 0 ? pv->Read(use_dev) : nullptr;
         const int *staying_idx_data = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const real_t *arrival_data = n_arrivals > 0 ? arrival_pv.Read(use_dev) : nullptr;
         real_t *new_data = new_col.Write(use_dev);
         // staying -> kept in original order
         // arrivals -> appended to the end
         mfem::forall_switch(use_dev, new_n_particles,
                             [=] MFEM_HOST_DEVICE (int slot) {
            for (int comp = 0; comp < vd; comp++) {
               const int to = internal::ElemIndex(slot, comp, new_n_particles, vd, col_ordering);
               if (slot < n_staying) {
                  const int from = internal::ElemIndex(staying_idx_data[slot], comp, n_particles, vd, col_ordering);
                  new_data[to] = old_data[from];
               }
               else {
                  const int from = internal::ElemIndex(slot - n_staying, comp, n_arrivals, vd, col_ordering);
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
                           Ordering::Type ordering = Ordering::byNODES) {
   return {{std::make_shared<internal::ArrayField<T>>(a, vdim, ordering)}};
}

inline internal::FieldGroup Vector(mfem::Vector &v, int vdim = 1,
                                   Ordering::Type ordering = Ordering::byNODES) {
   return {{std::make_shared<internal::VectorField>(v, vdim, ordering)}};
}

inline internal::FieldGroup ParticleVector(mfem::ParticleVector &pv) {
   return {{std::make_shared<internal::ParticleVectorField>(pv)}};
}

inline internal::FieldGroup ParticleVector(
   const std::vector<std::unique_ptr<mfem::ParticleVector>> &pvs) {
   internal::FieldGroup g;
   for (const auto &p : pvs) {
      g.cols.push_back(std::make_shared<internal::ParticleVectorField>(*p));
   }
   return g;
}

/** @brief Wrap a list of int arrays as one FieldGroup for ParticleSet

    ParticleSet Tags are `std::vector<std::unique_ptr<Array<int>>>`,
    overloader function to convert it into FieldGroup

    @param[in] arrs  Arrays to wrap, one value per particle each
    @return The FieldGroup holding one field per array */
inline internal::FieldGroup Array(const std::vector<std::unique_ptr<mfem::Array<int>>> &arrs) {
   internal::FieldGroup g;
   for (const auto &a : arrs) {
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

   Array<char> send_buf;
   Array<char> recv_buf;

   /** @brief Run the crystal router rounds on host data

       @param[in,out] ranks   Destination rank of each particle. Updated to
                              match the fields after routing.
       @param[in,out] fields  Fields to route, all of size ranks.Size() */
   void RouteInternal(Array<unsigned int> &ranks,
                      const std::vector<CR::internal::Field*> &fields);

   /** @brief Route fields that may live on a device

       Moves only the leaving particles to the host, routes them with
       RouteInternal(), and moves the arrivals back to the device

       @param[in]     rank_list  Destination rank of each particle
       @param[in,out] fields     Fields to route */
   void RouteDevice(const Array<unsigned int> &rank_list,
                    const std::vector<CR::internal::Field*> &fields);

   /** @brief Pack the particles that cross the cutoff into #send_buf and
       remove them from the fields

       @param[in,out] ranks    Destination ranks. Reduced to the kept particles.
       @param[in,out] fields   Fields to split. Reduced to the kept particles.
       @param[in]     cutoff   First rank of the upper half of the current
                               group of ranks
       @param[in]     send_hi  If true, send particles with rank >= @a cutoff,
                               otherwise send those with rank < @a cutoff */
   void Move(Array<unsigned int> &ranks,
             const std::vector<CR::internal::Field*> &fields,
             unsigned int cutoff, bool send_hi);

   /** @brief Send #send_buf to a partner rank and append the received rows to
       the fields

       @param[in,out] ranks    Destination ranks. Received ranks are appended.
       @param[in,out] fields   Fields. Received particles are appended.
       @param[in]     target   Rank to send to, and to receive froconvertm
       @param[in]     recvn    Number of messages to receive: 0, 1, or 2 (the
                               second one comes from rank - 1)
       @param[in]     msg_tag  Tag of the count message. The data message uses
                               @a msg_tag + 1. */
   void Exchange(Array<unsigned int> &ranks,
                 const std::vector<CR::internal::Field*> &fields,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
