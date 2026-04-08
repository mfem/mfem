#include "crystal.hpp"

//#ifdef MFEM_USE_MPI
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

    CrystalRouter::~CrystalRouter(){
        MPI_Comm_free(&comm);
    }


    // PRIVATE METHODS

    // moves and adjust ranks and data buffers based on cutoff and send_hi
    // return: number of items to send
    int CrystalRouter::Move(Array<int> &ranks, std::vector<Array<int>*> &data,
            int cutoff, bool send_hi,
            Array<int> &send_ranks, std::vector<Array<int>> &send_data){

            int n = ranks.Size();
            int ndata = data.size();


            std::vector<int> send_indices;
            std::vector<int> keep_indices;
            send_indices.reserve(n);
            keep_indices.reserve(n);

            // classify items into send vs keep
            for (int i = 0; i < n; i++){
                int r = ranks[i];
                if ((send_hi && r >= cutoff) || (!send_hi && r < cutoff)){
                    send_indices.push_back(i);
                }
                else{
                    keep_indices.push_back(i);
                }
            }
            int nsend = send_indices.size();
            int nkeep = keep_indices.size();


            // resize send buffer
            // send nsend items to every target
            send_ranks.SetSize(nsend);
            send_data.resize(ndata);
            for(int j = 0; j < ndata; j++){
                send_data[j].SetSize(nsend);
            }

            // actually fill the send buffer with data
            for(int i = 0; i < nsend; i++){
                int idx = send_indices[i];
                send_ranks[i] = ranks[idx];
                for (int j = 0; j < ndata; j++){
                    send_data[j][i] = (*data[j])[idx];
                }
            }

            // compact ranks arr to be the kept data
            // also update data arrays to be the kept data
            for(int k = 0; k < nkeep; k++){
                int idx = keep_indices[k];
                if (k != idx){
                    ranks[k] = ranks[idx];
                    for (int j = 0; j < ndata; j++){
                        (*data[j])[k] = (*data[j])[idx];
                    }
                }
            }
            // update kept sizes
            ranks.SetSize(nkeep);
            for(int j = 0; j < ndata; j++){
                data[j]->SetSize(nkeep);
            }

            return nsend;
    }

} //mfem

//#endif // MFEM_USE_MPI