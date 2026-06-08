#ifndef MFEM_CRYSTAL_OLD
#define MFEM_CRYSTAL_OLD

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include "../linalg/vector.hpp"
#include "array.hpp"

namespace mfem
{

// Original (non-templated) crystal router, kept for reference and as the
// backing implementation of ParticleSet::Redistribute. The new, templated
// CrystalRouter lives in crystal.hpp.
class CrystalRouterOld
{
public:
   CrystalRouterOld(MPI_Comm comm);
   ~CrystalRouterOld();

   void Route(Array<int> &ranks, std::vector<Array<int>*> &int_data, std::vector<Array<real_t>*> &real_data);

private:
   MPI_Comm comm;
   int rank, nprocs;

   // partition items into keep vs send based on rank
   void Move(Array<int> &ranks, std::vector<Array<int>*> &int_data, std::vector<Array<real_t>*> &real_data,
            int cutoff, bool send_hi,
            Array<int> &send_ranks, std::vector<Array<int>> &send_int_data, std::vector<Array<real_t>> &send_real_data);


   // exchange send buffers with partner ranks and append received items to ranks and data
   void Exchange(Array<int> &ranks, std::vector<Array<int>*> &int_data, std::vector<Array<real_t>*> &real_data,
                 int target, int recvn, int tag,
                 Array<int> &send_ranks, std::vector<Array<int>> &send_int_data, std::vector<Array<real_t>> &send_real_data);
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL_OLD
