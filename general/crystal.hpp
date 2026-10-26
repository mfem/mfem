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

/** @brief Field that wraps an mfem::Array or an mfem::Vector

    The container holds @a vdim values per particle, laid out by @a ordering.
    Field <- ContainerField <- mfem::Array or mfem::Vector

    @tparam C  Container type
    @tparam T  Element type */
template <typename C, typename T>
class ContainerField : public Field
{
   C *cont;
   int vdim;
   Ordering::Type ordering;
   /// keeps a reference to local host_buf alive, if using device
   std::shared_ptr<C> own;

public:
   /** @brief Wrap a container as a Field

       @param[in,out] c          Container to wrap. Must outlive this field.
       @param[in]     vdim_      Number of values per particle
       @param[in]     ordering_  Layout of the values in @a c */
   ContainerField(C &c, int vdim_ = 1,
                  Ordering::Type ordering_ = Ordering::byNODES)
      : cont(&c), vdim(vdim_), ordering(ordering_) { }

   /// @copydoc Field::GetNumParticles
   int              GetNumParticles() const override  { return cont->Size() / vdim; }
   /// @copydoc Field::GetParticleBytes
   std::size_t      GetParticleBytes() const override { return (std::size_t)vdim * sizeof(T); }

   /// @copydoc Field::ToHost
   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && cont->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_leaving = leaving_idx.Size();
      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<C>(n_leaving*vd);
      host_buf->GetMemory().UseDevice(use_dev);
      if (n_leaving > 0) {
         const T *old_data = cont->Read(use_dev);
         const int *leaving_idx_data = leaving_idx.Read(use_dev);
         T *host_buf_data = host_buf->Write(use_dev);

         /// loops to copy every device particle old_data into host buffer host_buf_data
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

      /// return new field, shared own pointer to local host buffer to keep it "alive"
      auto host_field = std::make_unique<ContainerField>(*host_buf, vd, col_ordering);
      host_field->own = host_buf;
      return host_field;
   }


   /// @copydoc Field::ToDevice

   /*
    *  @note
    *  n_particles is the *old* count of particles pre routing,
    *  required for indexing
    *
    *  new_n_particles is the new count of particles including those
    *  that arrived from router and didnt get moved
    */

   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const C &arrival_cont = *static_cast<ContainerField&>(arrivals).cont;
      const bool use_dev = Device::IsEnabled() && cont->UseDevice();
      const int n_particles = GetNumParticles();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_cont.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int vd = vdim;
      const Ordering::Type col_ordering = ordering;

      /// keep the old capacity, only the size shrinks
      C new_col(std::max(cont->Capacity(), new_n_particles*vd),
                cont->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*vd);
      new_col.GetMemory().UseDevice(cont->UseDevice());


      /// if nothing stays -> local buffer + indices should be empty
      /// if nothing arrived -> arriving appending buffer should be empty
      if (new_n_particles > 0) {
         const T *old_data = n_staying > 0 ? cont->Read(use_dev) : nullptr;
         const int *staying_idx_data = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;

         const T *arrival_data = n_arrivals > 0 ? arrival_cont.Read(use_dev) : nullptr;
         T *new_data = new_col.Write(use_dev);


         /// staying -> kept in original order
         /// arrivals -> appended to the end
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

      cont->Swap(new_col);
   }

   /// @copydoc Field::Pack
   void Pack(int i, char *dst) const override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         const T v = (*cont)[internal::ElemIndex(i, c, n, vdim, ordering)];
         std::memcpy(dst + (std::size_t)c*sizeof(T), &v, sizeof(T));
      }
   }
   /// @copydoc Field::Unpack
   void Unpack(int i, const char *src) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         T v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(T), sizeof(T));
         (*cont)[internal::ElemIndex(i, c, n, vdim, ordering)] = v;
      }
   }
   /// @copydoc Field::Copy
   void Copy(int src, int dst) override {
      const int n = GetNumParticles();
      for (int c = 0; c < vdim; c++) {
         (*cont)[internal::ElemIndex(dst, c, n, vdim, ordering)] =
            (*cont)[internal::ElemIndex(src, c, n, vdim, ordering)];
      }
   }
   /// @copydoc Field::SetNumParticles
   void SetNumParticles(int n) override {
      const int old_n = GetNumParticles();
      if (n == old_n) { return; }
      if (n < old_n) {
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::CompactNodes(*cont, old_n, n, vdim);
         }
         cont->SetSize(n*vdim);
      }
      else {
         const int old_sz = old_n*vdim, new_sz = n*vdim;
         if (new_sz > cont->Capacity()) {
            C tmp(new_sz, cont->GetMemory().GetMemoryType());
            tmp.GetMemory().UseDevice(cont->UseDevice());
            if (old_sz > 0) {
               std::memcpy(tmp.GetData(), cont->GetData(),
                           (std::size_t)old_sz*sizeof(T));
            }
            cont->Swap(tmp);
         }
         else {
            cont->SetSize(new_sz);
         }
         for (int k = old_sz; k < new_sz; k++) { (*cont)[k] = T(); }
         if (ordering == Ordering::byNODES && vdim > 1) {
            internal::SpreadNodes(*cont, old_n, n, vdim, T());
         }
      }
   }
};

/// ----------------------------------------------------------------------------


/** @brief Aliases for Array, Vector, PV

    @tparam T  Element type */
template <typename T>
using ArrayField = ContainerField<mfem::Array<T>, T>;
using VectorField = ContainerField<mfem::Vector, real_t>;

class ParticleVectorField : public VectorField
{
   mfem::ParticleVector *pv;

public:
   ParticleVectorField(mfem::ParticleVector &pv_)
      : VectorField(pv_, pv_.GetVDim(), pv_.GetOrdering()), pv(&pv_) { }

   void SetNumParticles(int n) override { pv->SetNumParticles(n, true); }
};


/// ----------------------------------------------------------------------------


/// externally callable wrappers for Array, Vector, PV
template <typename T>
ArrayField<T> Array(mfem::Array<T> &a, int vdim = 1,
                    Ordering::Type ordering = Ordering::byNODES) {
   return ArrayField<T>(a, vdim, ordering);
}
inline VectorField Vector(mfem::Vector &v, int vdim = 1,
                          Ordering::Type ordering = Ordering::byNODES) {
   return VectorField(v, vdim, ordering);
}
inline ParticleVectorField ParticleVector(mfem::ParticleVector &pv) {
   return ParticleVectorField(pv);
}


/// ----------------------------------------------------------------------------


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

   FieldGroup() = default;

   template <typename F,
             std::enable_if_t<std::is_base_of<Field, F>::value, int> = 0>
   FieldGroup(const F &f) : cols{std::make_shared<F>(f)} { }
};


/** @brief Wrap a list of ParticleVectors as one FieldGroup

    @param[in] pvs  ParticleVectors to wrap. Must outlive the returned group.
    @return The FieldGroup holding one field per ParticleVector */
inline FieldGroup ParticleVector(
   const std::vector<std::unique_ptr<mfem::ParticleVector>> &pvs) {
   FieldGroup g;
   for (const auto &p : pvs) {
      g.cols.push_back(std::make_shared<ParticleVectorField>(*p));
   }
   return g;
}

/** @brief Wrap a list of int arrays as one FieldGroup

    @param[in] arrs  Arrays to wrap, one value per particle each. Must outlive
                     the returned group.
    @return The FieldGroup holding one field per array */
inline FieldGroup Array(
   const std::vector<std::unique_ptr<mfem::Array<int>>> &arrs) {
   FieldGroup g;
   for (const auto &a : arrs) {
      g.cols.push_back(std::make_shared<ArrayField<int>>(*a));
   }
   return g;
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
   void Route(const Array<unsigned int> &rank_list,
              std::initializer_list<CR::FieldGroup> fields);

private:
   MPI_Comm comm;
   int rank;
   int nprocs;

   Array<char> send_buf;
   Array<char> recv_buf;

   void RouteInternal(Array<unsigned int> &ranks,
                      const std::vector<CR::Field*> &fields);

   void RouteDevice(const Array<unsigned int> &rank_list,
                    const std::vector<CR::Field*> &fields);

   void Move(Array<unsigned int> &ranks,
             const std::vector<CR::Field*> &fields,
             unsigned int cutoff, bool send_hi);

   void Exchange(Array<unsigned int> &ranks,
                 const std::vector<CR::Field*> &fields,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
