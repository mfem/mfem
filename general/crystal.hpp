#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include "../linalg/vector.hpp"
#include "../linalg/particlevector.hpp"
#include "array.hpp"

namespace mfem
{

class CrystalRouter
{
public:
   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();

   // Route ids/tags/ranks plus an arbitrary number of ParticleVectors
   // Each ParticleVector keeps its own vdim/ordering.
   // Template collapses ParticleVectors into std::vector<mfem::ParticleVector*> for internal use [RouteInternal].
   template <typename... PVs>
   void Route(Array<unsigned long long> &ids, Array<int> &tags,
              Array<int> &ranks, PVs &... pvs)
   {
      std::vector<ParticleVector*> data{ &pvs... };
      RouteInternal(ids, tags, ranks, data);
   }

private:
   MPI_Comm comm;
   int rank, nprocs;

   int tag_vdim;

   Array<char> send_buf;
   Array<char> recv_buf;

   void RouteInternal(Array<unsigned long long> &ids, Array<int> &tags,
                  Array<int> &ranks, const std::vector<ParticleVector*> &data);

   void Move(const std::vector<ParticleVector*> &data,
             Array<unsigned long long> &ids, Array<int> &tags,
             Array<int> &ranks, int cutoff, bool send_hi);

   void Exchange(const std::vector<ParticleVector*> &data,
                 Array<unsigned long long> &ids, Array<int> &tags,
                 Array<int> &ranks, int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
