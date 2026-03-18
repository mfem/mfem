// WIP

#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <memory>

namespace mfem{

// crystal.c/crystal_init
CrystalRouter::CrystalRouter(MPI_Comm comm){
    MPI_Comm_dup(comm, &this->comm);
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &nprocs);
}

// crystal.c/crystal_free
CrystalRouter::~CrystalRouter(){
    MPI_Comm_free(&comm);
}

// crystal.c/crystal_router
void CrystalRouter::Route(){

}

// crystal.c/crystal_move
uint32_t CrystalRouter::Move(uint32_t cutoff, bool send_hi){
    size_t src = 0, keep = 0, send = 0, len = 0;
    work.resize(data.size());

    if(send_hi){ // send hi keep lo
        while(src < data.size()){
            // notes for myself:
            // |target|source|length|payload ...| repeated
            // offset length for 3 values + size of payload for next msg

            len = 3 + data[src + 2];

            if(data[src] >= cutoff){
                //send
                std::copy(data.begin() + src, data.begin() + src + len, work.begin() + send);
                send += len;
            }
            else{
                //keep
                std::copy(data.begin() + src, data.begin() + src + len, data.begin() + keep);
                keep += len;
            }
            src += len;
        }
    }
    else{ // send lo keep hi
        while(src < data.size()){
            len = 3 + data[src + 2];

            if(data[src] >= cutoff){
                //keep
                std::copy(data.begin() + src, data.begin() + src + len, data.begin() + keep);
                keep += len;
            }
            else{
                //send
                std::copy(data.begin() + src, data.begin() + src + len, work.begin() + send);
                send += len;
            }
            src += len;
        }
    }
    data.resize(keep);
    return send;
}


// crystal.c/crystal_exchange
void CrystalRouter::Exchange(uint32_t send_n, int target, int recvn, int tag){
}


}

#endif