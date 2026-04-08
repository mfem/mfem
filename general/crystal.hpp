#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

//#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include "array.hpp"

namespace mfem
{

class CrystalRouter
{
public:
   /// @param[in,out] ranks     Destination rank per item
   /// @param[in,out] data      Vector of pointers to Array<int>, each of the same length as ranks

   CrystalRouter(MPI_Comm comm);
   ~CrystalRouter();

   void Route(Array<int> &ranks, std::vector<Array<int>*> &data);

private:
   MPI_Comm comm;
   int rank, nprocs;

   // partition items into keep vs send based on rank
   int Move(Array<int> &ranks, std::vector<Array<int>*> &data,
            int cutoff, bool send_hi,
            Array<int> &send_ranks, std::vector<Array<int>> &send_data);


   // exchange send buffers with partner ranks and append received items to ranks and data
   void Exchange(Array<int> &ranks, std::vector<Array<int>*> &data,
                 int target, int recvn, int tag,
                 Array<int> &send_ranks, std::vector<Array<int>> &send_data);
};

}

//#endif // MFEM_USE_MPI
#endif // MFEM_CRYSTAL