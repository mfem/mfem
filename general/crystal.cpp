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
    
    CrystalRouter::~CrystalRouter(){
        MPI_Comm_free(&comm);
    }
    
    void CrystalRouter::Route(Array<int> &ranks, std::vector<Array<int>*> &data){
    
    uint32_t bl = 0, bh, nl;
    uint32_t id = rank, n = nprocs;
    uint32_t targ, tag = 0;
    bool send_hi;
    int recvn;
    Array<int> send_ranks;
    std::vector<Array<int>> send_data;

    while(n > 1){
        nl = (n+1)/2;
        bh = bl + nl;
        send_hi = (id < nl) ? true : false;
        Move(ranks, data, bh, send_hi, send_ranks, send_data);

        /* overflow check, deal with later
        long long send_n_long = send_n;
        send_n_long *= sizeof(uint32_t);
        bool overflow = send_n_long > INT_MAX;
        
        if(overflow){
            std::print(stderr, "Error in crystal_router: rank = %d send_n = %lld (> "
            "INT_MAX)\n", id, send_n_long);
            MPI_Abort(comm, 1);
            }


        Move() doesn't return anything but in the other implementation it returns send_n which is used for this overflow check
        Once the overflow check gets reimplemented, need to make sure send_n is correctly set in Move() and returned here
        */
           
           
           recvn = 1, targ = n-1-(id-bl)+bl;
           if(id == targ){
               targ = bh;
               recvn = 0;
            }
            if(n&1 && id==bh){
                recvn = 2;
            }
            
            Exchange(ranks, data, targ, recvn, tag, send_ranks, send_data);
            if(id < bh){
                n = nl;
            }
        else{
            n -= nl;
            bl = bh;
        }
        tag += 2;
    }
}

// PRIVATE METHODS

    // moves and adjust ranks and data buffers based on cutoff and send_hi
    // return: number of items to send
    void CrystalRouter::Move(Array<int> &ranks, std::vector<Array<int>*> &data,
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
    }


    void CrystalRouter::Exchange(Array<int> &ranks, std::vector<Array<int>*> &data,
                 int target, int recvn, int tag,
                 Array<int> &send_ranks, std::vector<Array<int>> &send_data){
        
        
        // just like crystal.c implementation, need to first exchange counts
        MPI_Request reqs[3];
        int count[2] = {0, 0};
        
        int send_n = send_ranks.Size();
        
        // if recieving (recvn >= 1) Irecv for count of incoming items
        if(recvn >= 1){
            MPI_Irecv(&count[0], 1, MPI_INT32_T, target, tag, comm, &reqs[1]);
        }
        // middle rank will recieve two messages, so Irecv for second count
        if(recvn == 2){
            MPI_Irecv(&count[1], 1, MPI_INT32_T, rank - 1, tag, comm, &reqs[2]);
        }
        // otherwise just send count
        MPI_Isend(&send_n, 1, MPI_INT32_T, target, tag, comm, &reqs[0]);
        MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);
        
        
        // pack the data into a single buffer to use for MPI send/recv
        // 1 single MPI communication for all the data at once
        int packet = 1 + data.size();
        // size of 1 packet is 1 int for rank + data.size() for the rest
        Array<int> send_buffer;
        send_buffer.SetSize(send_n * packet);                    // each item has 1 rank + data.size() data entries
        Array<int> recv_buffer;
        recv_buffer.SetSize((count[0] + count[1]) * packet);     // worse case it recieves max data size from both sides
                                                                            // this is something that can probably be optimized later

        // pack ranks and data (packets) into send buffer
        // |packet0 | packet1 | ... | packetN |
        // |rank| data0 | data1 | ... | dataN | rank | data0 | data1 | ... |
        for(int i = 0; i < send_n; i++){
            send_buffer[i*packet] = send_ranks[i];
            for(int j = 0; j < static_cast<int>(data.size()) ; j++){
                send_buffer[i * packet + 1 + j] = send_data[j][i];
            }
        }

        // exchange data
        if(recvn >= 1){
            MPI_Irecv(recv_buffer.GetData(),
                      count[0] * packet,
                      MPI_INT32_T, target, tag+1, comm, &reqs[1]);
        }
        if(recvn == 2){
            MPI_Irecv(recv_buffer.GetData() + (count[0] * packet), //offset for a second recv
                      count[1] * packet,
                      MPI_INT32_T, rank - 1, tag+1, comm, &reqs[2]);
        }
        MPI_Isend(send_buffer.GetData(), send_n * packet, MPI_INT32_T, target, tag+1, comm, &reqs[0]);
        MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);


        // append recieved data to ranks and data
        int nrecv = count[0] + count[1];
        for(int i = 0; i < nrecv; i++){
            ranks.Append(recv_buffer[i*packet]);
            for(int j = 0; j < static_cast<int>(data.size()); j++){
                (*data[j]).Append(recv_buffer[i*packet + 1 + j]);
            }
        }

    }
} //mfem

#endif // MFEM_USE_MPI