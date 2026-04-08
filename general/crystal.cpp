#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <memory>
#include <print>
#include <cstdio>

namespace mfem{
    CrystalRouter::CrystalRouter(MPI_Comm comm){
        MPI_Comm_dup(comm, &this->comm);
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &nprocs);
    }
    
    ~CrystalRouter(){
        MPI_Comm_free(&comm);
    }

} //::mfem

#endif // MFEM_USE_MPI