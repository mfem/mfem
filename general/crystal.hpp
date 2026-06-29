#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include <type_traits>
#include "../linalg/vector.hpp"
#include "../linalg/particlevector.hpp"
#include "array.hpp"

namespace mfem
{

template <typename T>
// true if T is a ParticleVector or an Array<int> (tags)
inline constexpr bool Routable =
   std::is_base_of<ParticleVector, std::decay_t<T>>::value ||
   std::is_same<std::decay_t<T>, Array<int>>::value;

class CrystalRouter
{
public:
   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();

   // enable this overload only if all the input columns are routable (ParticleVector or tags Array<int>)
   template <typename... Cols, std::enable_if_t<(Routable<Cols> && ...), int> = 0>

   void Route(Array<unsigned int> &ranks, Array<unsigned long long> &ids, Cols &... cols)
   {
      std::vector<Array<int>*>     tags;
      std::vector<ParticleVector*> data;
      (Bucket(cols, tags, data), ...);

      Route(ranks, ids, tags, data);
   }

   void Route(Array<unsigned int> &ranks, Array<unsigned long long> &ids,
              const std::vector<Array<int>*> &tags,
              const std::vector<ParticleVector*> &data);

private:
   MPI_Comm comm;
   int rank, nprocs;

   Array<char> send_buf;
   Array<char> recv_buf;

   template <typename T>
   static void Bucket(T &col, std::vector<Array<int>*> &tags,
                      std::vector<ParticleVector*> &data)
   {
      if constexpr (std::is_base_of<ParticleVector, T>::value) {
         data.push_back(&col);
      }
      else {
         tags.push_back(&col);
      }
   }

   void Move(Array<unsigned int> &ranks, Array<unsigned long long> &ids,
             const std::vector<Array<int>*> &tags,
             const std::vector<ParticleVector*> &data,
             unsigned int cutoff, bool send_hi);

   void Exchange(Array<unsigned int> &ranks, Array<unsigned long long> &ids,
                 const std::vector<Array<int>*> &tags,
                 const std::vector<ParticleVector*> &data,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
