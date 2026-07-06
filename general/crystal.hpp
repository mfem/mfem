#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include <memory>
#include <initializer_list>
#include "../linalg/vector.hpp"
#include "../linalg/particlevector.hpp"
#include "array.hpp"

namespace mfem
{

class CrystalRouter {
public:
   template <typename T>
   struct Group {
      std::vector<T*> cols;
      Group(T &col) : cols{&col} {}                                 // one pointer
      Group(T *col) : cols{col} {}                                  // one value
      Group(const std::vector<T*> &v) : cols(v) {}                  // many pointers (raw)
      Group(const std::vector<std::unique_ptr<T>> &v){              // many pointers (unique)
         cols.reserve(v.size());
         for (const auto &p : v) { cols.push_back(p.get()); }
      }
   };

   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();

   void Route(const Array<unsigned int> &rank_list,
              Array<unsigned long long> &ids,
              std::initializer_list<Group<Array<int>>> tags,
              std::initializer_list<Group<ParticleVector>> data);

private:
   MPI_Comm comm;
   int rank, nprocs;

   Array<char> send_buf;
   Array<char> recv_buf;

   void RouteInternal(Array<unsigned int> &ranks,
                      Array<unsigned long long> &ids,
                      const std::vector<Array<int>*> &tags,
                      const std::vector<ParticleVector*> &data);

   void Move(Array<unsigned int> &ranks,
             Array<unsigned long long> &ids,
             const std::vector<Array<int>*> &tags,
             const std::vector<ParticleVector*> &data,
             unsigned int cutoff, bool send_hi);

   void Exchange(Array<unsigned int> &ranks,
                 Array<unsigned long long> &ids,
                 const std::vector<Array<int>*> &tags,
                 const std::vector<ParticleVector*> &data,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
