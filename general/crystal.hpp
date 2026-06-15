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

   void Route(ParticleVector &data, Array<unsigned long long> &ids,
              Array<int> &tags, Array<int> &ranks);

private:
   MPI_Comm comm;
   int rank, nprocs;

   int tag_vdim;

   Array<char> send_buf;
   Array<char> recv_buf;

   void Move(ParticleVector &data, Array<unsigned long long> &ids,
             Array<int> &tags, Array<int> &ranks, int cutoff, bool send_hi);

   void Exchange(ParticleVector &data, Array<unsigned long long> &ids,
                 Array<int> &tags, Array<int> &ranks,
                 int target, int recvn, int msg_tag);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL
