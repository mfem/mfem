#ifndef MFEM_CRYSTAL
#define MFEM_CRYSTAL

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include <vector>
#include <cstdint>

namespace mfem
{
    class CrystalRouter{
    public:
        // crystal.c/crystal_init
        // crystal.c/crystal_free 
        // equivalents
        CrystalRouter(MPI_Comm comm);
        ~CrystalRouter();

        // crystal.c/crystal_router
        void Route();

        // access to message buffer
        std::vector<uint32_t> &GetData() { return data; }

    private:
        MPI_Comm comm;
        int rank, nprocs;
        std::vector<uint32_t> data;  // gslib's buffer data
        std::vector<uint32_t> work;  // gslib's buffer work

        // crystal.c/crystal_move
        uint32_t Move(uint32_t cutoff, bool send_hi);

        // crystal.c/crystal_exchange
        void Exchange(uint32_t send_n, int target, int recvn, int tag);
    };
}

#endif
#endif