#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>
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
   virtual int GetVDim() const = 0;
   virtual Ordering::Type GetOrdering() const = 0;
   virtual int Size() const = 0;
   virtual std::size_t RowBytes() const = 0;
   virtual void Pack(int i, char *dst) const = 0;
   virtual void Unpack(int i, const char *src) = 0;
   virtual void Copy(int src, int dst) = 0;
   virtual void Resize(int n) = 0;
};

namespace internal
{

// helper function for nodes vs vdim indexing
inline int ElemIndex(int i, int c, int n, int vdim, Ordering::Type ordering) {
   if(ordering == Ordering::byNODES)
       return (c*n) + i;
   return i*vdim + c;
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

public:
   ArrayField(mfem::Array<T> &a, int vdim_ = 1,
              Ordering::Type ordering_ = Ordering::byNODES)
      : arr(&a), vdim(vdim_), ordering(ordering_) { }

   int GetVDim() const override { return vdim; }
   Ordering::Type GetOrdering() const override { return ordering; }
   int Size() const override { return arr->Size() / vdim; }
   std::size_t RowBytes() const override { return (std::size_t)vdim * sizeof(T); }

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
public:
   VectorField(mfem::Vector &v, int vdim_ = 1,
               Ordering::Type ordering_ = Ordering::byNODES)
      : vec(&v), vdim(vdim_), ordering(ordering_) { }

   int GetVDim() const override { return vdim; }
   Ordering::Type GetOrdering() const override { return ordering; }
   int Size() const override { return vec->Size() / vdim; }
   std::size_t RowBytes() const override{ return (std::size_t)vdim * sizeof(real_t); }

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
public:
   ParticleVectorField(mfem::ParticleVector &pv_) : pv(&pv_) { }

   int GetVDim() const override { return pv->GetVDim(); }
   Ordering::Type GetOrdering() const override { return pv->GetOrdering(); }
   int Size() const override { return pv->GetNumParticles(); }
   std::size_t RowBytes() const override{ return (std::size_t)pv->GetVDim() * sizeof(real_t); }

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
