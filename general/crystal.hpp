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
// Generic Abstract Field Class
// Different datatypes replace these functions
class Field {
public:
   virtual ~Field() = default;
   virtual int Size() const = 0;
   virtual std::size_t RowBytes() const = 0;
   virtual void Pack(int i, char *dst) const = 0;
   virtual void Unpack(int i, const char *src) = 0;
   virtual void Copy(int src, int dst) = 0;
   virtual void Resize(int n) = 0;

   virtual std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) = 0;
   virtual void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) = 0;
};

namespace internal
{

// helper function for nodes vs vdim indexing:
MFEM_HOST_DEVICE inline int ElemIndex(int particle, int comp, int n_particles,
                                      int vdim, Ordering::Type ordering) {
   if(ordering == Ordering::byNODES)
       return (comp*n_particles) + particle;
   return particle*vdim + comp;
}

// helper function for compacting and spreading nodes
template <typename C>
void CompactNodes(C &cont, int old_n, int new_n, int vdim) {
   for (int c = 1; c < vdim; c++) {
      for (int i = 0; i < new_n; i++) {
         cont[c*new_n + i] = cont[c*old_n + i];
      }
   }
}

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


// mfem::Array
// @param *arr: pointer to the array
// @param vdim: # of dimensions
// @param ordering: ordering type (nodes or vdim)
template <typename T>
class ArrayField : public Field {
   mfem::Array<T> *arr;
   int vdim;
   Ordering::Type ordering;
   std::shared_ptr<mfem::Array<T>> own; // set only when wrapping a host_buf

public:
   ArrayField(mfem::Array<T> &a, int vdim_ = 1,
              Ordering::Type ordering_ = Ordering::byNODES)
      : arr(&a), vdim(vdim_), ordering(ordering_) { }

   int              Size() const override           { return arr->Size() / vdim; }
   std::size_t      RowBytes() const override       { return (std::size_t)vdim * sizeof(T); }

   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && arr->UseDevice();
      const int n_particles = Size();
      const int n_leaving = leaving_idx.Size();
      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      auto host_buf = std::make_shared<mfem::Array<T>>(n_leaving*n_comps);
      host_buf->GetMemory().UseDevice(use_dev);
      if (n_leaving > 0) {
         const T *col_data = arr->Read(use_dev);
         const int *leaving = leaving_idx.Read(use_dev);
         T *host_buf_data = host_buf->Write(use_dev);

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


   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Array<T> &arrival_arr = *static_cast<ArrayField<T>&>(arrivals).arr;
      const bool use_dev = Device::IsEnabled() && arr->UseDevice();
      const int n_particles = Size();
      const int n_staying = staying_idx.Size();
      const int n_arrivals = arrival_arr.Size() / vdim;
      const int new_n_particles = n_staying + n_arrivals;

      const int n_comps = vdim;
      const Ordering::Type col_ordering = ordering;

      // keep the old capacity, only the size shrinks
      mfem::Array<T> new_col(std::max(arr->Capacity(), new_n_particles*n_comps),
                             arr->GetMemory().GetMemoryType());
      new_col.SetSize(new_n_particles*n_comps);
      new_col.GetMemory().UseDevice(arr->UseDevice());
      if (new_n_particles > 0) {
         const T *col_data = n_staying > 0 ? arr->Read(use_dev) : nullptr;
         const int *staying = n_staying > 0 ? staying_idx.Read(use_dev) : nullptr;
         const T *arrival_data = n_arrivals > 0 ? arrival_arr.Read(use_dev) : nullptr;
         T *new_data = new_col.Write(use_dev);
         // staying -> kept in original order; arrivals -> appended to the end
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

   void Pack(int i, char *dst) const override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         const T v = (*arr)[internal::ElemIndex(i, c, n, vdim, ordering)];
         std::memcpy(dst + (std::size_t)c*sizeof(T), &v, sizeof(T));
      }
   }
   void Unpack(int i, const char *src) override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         T v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(T), sizeof(T));
         (*arr)[internal::ElemIndex(i, c, n, vdim, ordering)] = v;
      }
   }
   void Copy(int src, int dst) override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         (*arr)[internal::ElemIndex(dst, c, n, vdim, ordering)] =
            (*arr)[internal::ElemIndex(src, c, n, vdim, ordering)];
      }
   }
   void Resize(int n) override {
      const int old_n = Size();
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




// mfem::Vector
class VectorField : public Field {
   mfem::Vector *vec;
   int vdim;
   Ordering::Type ordering;
   std::shared_ptr<mfem::Vector> own; // set only when wrapping a host_buf

public:
   VectorField(mfem::Vector &v, int vdim_ = 1,
               Ordering::Type ordering_ = Ordering::byNODES)
      : vec(&v), vdim(vdim_), ordering(ordering_) { }

   int              Size() const override           { return vec->Size() / vdim; }
   std::size_t      RowBytes() const override       { return (std::size_t)vdim * sizeof(real_t); }

   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && vec->UseDevice();
      const int n_particles = Size();
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


   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::Vector &arrival_vec = *static_cast<VectorField&>(arrivals).vec;
      const bool use_dev = Device::IsEnabled() && vec->UseDevice();
      const int n_particles = Size();
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

   void Pack(int i, char *dst) const override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         const real_t v = (*vec)[internal::ElemIndex(i, c, n, vdim, ordering)];
         std::memcpy(dst + (std::size_t)c*sizeof(real_t), &v, sizeof(real_t));
      }
   }
   void Unpack(int i, const char *src) override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         real_t v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(real_t), sizeof(real_t));
         (*vec)[internal::ElemIndex(i, c, n, vdim, ordering)] = v;
      }
   }
   void Copy(int src, int dst) override {
      const int n = Size();
      for (int c = 0; c < vdim; c++) {
         (*vec)[internal::ElemIndex(dst, c, n, vdim, ordering)] =
            (*vec)[internal::ElemIndex(src, c, n, vdim, ordering)];
      }
   }
   void Resize(int n) override {
      const int old_n = Size();
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





// mfem::ParticleVector
class ParticleVectorField : public Field {
   mfem::ParticleVector *pv;
   std::shared_ptr<mfem::ParticleVector> own; // set only when wrapping a host_buf

public:
   ParticleVectorField(mfem::ParticleVector &pv_) : pv(&pv_) { }

   int              Size() const override           { return pv->GetNumParticles(); }
   std::size_t      RowBytes() const override       { return (std::size_t)pv->GetVDim() * sizeof(real_t); }

   std::unique_ptr<Field> ToHost(const mfem::Array<int> &leaving_idx) override {
      const bool use_dev = Device::IsEnabled() && pv->UseDevice();
      const int n_particles = Size();
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


   void ToDevice(const mfem::Array<int> &staying_idx, Field &arrivals) override {
      const mfem::ParticleVector &arrival_pv = *static_cast<ParticleVectorField&>(arrivals).pv;
      const bool use_dev = Device::IsEnabled() && pv->UseDevice();
      const int n_particles = Size();
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

   void Pack(int i, char *dst) const override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) {
         const real_t v = (*pv)(i, c);
         std::memcpy(dst + (std::size_t)c*sizeof(real_t), &v, sizeof(real_t));
      }
   }
   void Unpack(int i, const char *src) override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) {
         real_t v;
         std::memcpy(&v, src + (std::size_t)c*sizeof(real_t), sizeof(real_t));
         (*pv)(i, c) = v;
      }
   }
   void Copy(int src, int dst) override {
      const int vd = pv->GetVDim();
      for (int c = 0; c < vd; c++) { (*pv)(dst, c) = (*pv)(src, c); }
   }
   void Resize(int n) override { pv->SetNumParticles(n, true); }
};





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



// Sort the variables into the appropriate groups of pointers
// In: Initializer List of Fields
// Sort: Fields into the appropriate CR type
// Out: std::vector of columns used by internal router
class FieldGroup {
public:
   std::vector<std::shared_ptr<Field>> cols;

   template <typename T>
   FieldGroup(const ArrayField<T> &f) : cols{std::make_shared<ArrayField<T>>(f)} { }
   FieldGroup(const VectorField &f) : cols{std::make_shared<VectorField>(f)} { }
   FieldGroup(const ParticleVectorField &f) : cols{std::make_shared<ParticleVectorField>(f)} { }
   FieldGroup(const std::vector<std::unique_ptr<mfem::ParticleVector>> &v) {
      cols.reserve(v.size());
      for (const auto &p : v) {
         cols.push_back(std::make_shared<ParticleVectorField>(*p));
      }
   }
   FieldGroup(const std::vector<std::unique_ptr<mfem::Array<int>>> &v) {
      cols.reserve(v.size());
      for (const auto &p : v) {
         cols.push_back(std::make_shared<ArrayField<int>>(*p, 1, Ordering::byNODES));
      }
   }
};


inline FieldGroup ParticleVector(
   const std::vector<std::unique_ptr<mfem::ParticleVector>> &pvs) {
   return FieldGroup(pvs);
}

inline FieldGroup Array(
   const std::vector<std::unique_ptr<mfem::Array<int>>> &arrs) {
   return FieldGroup(arrs);
}

} // namespace CR



class CrystalRouter {
public:
   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();
   CrystalRouter(const CrystalRouter&) = delete;
   CrystalRouter& operator=(const CrystalRouter&) = delete;

   // external route, {field, field, field} call
   void Route(const Array<unsigned int> &rank_list,
              std::initializer_list<CR::FieldGroup> fields);

private:
   MPI_Comm comm;
   int rank, nprocs;

   Array<char> send_buf;
   Array<char> recv_buf;

   // internal route, fields get sorted into FieldGroup -> FieldGroup call
   void RouteInternal(Array<unsigned int> &ranks,
                      const std::vector<CR::Field*> &fields);

   // device path: move only leaving particles to host, route, move arrivals back
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
