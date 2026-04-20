#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <memory>
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
    
    void CrystalRouter::Route(Array<int> &ranks, std::vector<Array<int>*> &int_data, std::vector<Array<real_t>*> &real_data){
    
    uint32_t bl = 0, bh, nl;
    uint32_t id = rank, n = nprocs;
    uint32_t targ, tag = 0;
    bool send_hi;
    int recvn;
    Array<int> send_ranks;
    std::vector<Array<int>> send_int_data;
    std::vector<Array<real_t>> send_real_data;

    while(n > 1){
        nl = (n+1)/2;
        bh = bl + nl;
        send_hi = (id < nl) ? true : false;
        Move(ranks, int_data, real_data, bh, send_hi, send_ranks, send_int_data, send_real_data);

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
            
            Exchange(ranks, int_data, real_data, targ, recvn, tag, send_ranks, send_int_data, send_real_data);
            if(id < bh){
                n = nl;
            }
        else{
            n -= nl;
            bl = bh;
        }
        tag += 3;
    }
}

// PRIVATE METHODS

    // moves and adjust ranks and data buffers based on cutoff and send_hi
    // return: number of items to send
    void CrystalRouter::Move(Array<int> &ranks, std::vector<Array<int>*> &int_data, std::vector<Array<real_t>*> &real_data,
            int cutoff, bool send_hi,
            Array<int> &send_ranks, std::vector<Array<int>> &send_int_data, std::vector<Array<real_t>> &send_real_data){

            int n = ranks.Size();
            int ndata = int_data.size();


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


            int nrdata = real_data.size();

            // resize send buffers
            send_ranks.SetSize(nsend);
            send_int_data.resize(ndata);
            for(int j = 0; j < ndata; j++){
                send_int_data[j].SetSize(nsend);
            }
            send_real_data.resize(nrdata);
            for(int j = 0; j < nrdata; j++){
                send_real_data[j].SetSize(nsend);
            }

            // fill send buffers with data
            for(int i = 0; i < nsend; i++){
                int idx = send_indices[i];
                send_ranks[i] = ranks[idx];
                for (int j = 0; j < ndata; j++){
                    send_int_data[j][i] = (*int_data[j])[idx];
                }
                for (int j = 0; j < nrdata; j++){
                    send_real_data[j][i] = (*real_data[j])[idx];
                }
            }

            // compact kept data
            for(int k = 0; k < nkeep; k++){
                int idx = keep_indices[k];
                if (k != idx){
                    ranks[k] = ranks[idx];
                    for (int j = 0; j < ndata; j++){
                        (*int_data[j])[k] = (*int_data[j])[idx];
                    }
                    for (int j = 0; j < nrdata; j++){
                        (*real_data[j])[k] = (*real_data[j])[idx];
                    }
                }
            }
            // update kept sizes
            ranks.SetSize(nkeep);
            for(int j = 0; j < ndata; j++){
               int_data[j]->SetSize(nkeep);
            }
            for(int j = 0; j < nrdata; j++){
               real_data[j]->SetSize(nkeep);
            }
    }


    void CrystalRouter::Exchange(Array<int> &ranks,
                 std::vector<Array<int>*> &int_data,
                 std::vector<Array<real_t>*> &real_data,
                 int target, int recvn, int tag,
                 Array<int> &send_ranks,
                 std::vector<Array<int>> &send_int_data,
                 std::vector<Array<real_t>> &send_real_data){

        // exchange counts
        MPI_Request reqs[3];
        int count[2] = {0, 0};
        int send_n = send_ranks.Size();

        if(recvn >= 1){
            MPI_Irecv(&count[0], 1, MPI_INT32_T, target, tag, comm, &reqs[1]);
        }
        if(recvn == 2){
            MPI_Irecv(&count[1], 1, MPI_INT32_T, rank - 1, tag, comm, &reqs[2]);
        }
        MPI_Isend(&send_n, 1, MPI_INT32_T, target, tag, comm, &reqs[0]);
        MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

        int nrecv = count[0] + count[1];

        // pack and exchange int data (rank + int fields)
        int int_packet = 1 + int_data.size();
        Array<int> send_buffer(send_n * int_packet);
        Array<int> recv_buffer(nrecv * int_packet);

        for(int i = 0; i < send_n; i++){
            send_buffer[i * int_packet] = send_ranks[i];
            for(int j = 0; j < static_cast<int>(int_data.size()); j++){
                send_buffer[i * int_packet + 1 + j] = send_int_data[j][i];
            }
        }

        if(recvn >= 1){
            MPI_Irecv(recv_buffer.GetData(),
                      count[0] * int_packet,
                      MPI_INT32_T, target, tag+1, comm, &reqs[1]);
        }
        if(recvn == 2){
            MPI_Irecv(recv_buffer.GetData() + (count[0] * int_packet),
                      count[1] * int_packet,
                      MPI_INT32_T, rank - 1, tag+1, comm, &reqs[2]);
        }
        MPI_Isend(send_buffer.GetData(), send_n * int_packet, MPI_INT32_T, target, tag+1, comm, &reqs[0]);
        MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);




        // pack and exchange real data
        int nrdata = real_data.size();
        int real_packet = nrdata;
        Array<real_t> send_real_buf(send_n * real_packet);
        Array<real_t> recv_real_buf(nrecv * real_packet);

        for(int i = 0; i < send_n; i++){
            for(int j = 0; j < nrdata; j++){
                send_real_buf[i * real_packet + j] = send_real_data[j][i];
            }
        }

        if(recvn >= 1){
            MPI_Irecv(recv_real_buf.GetData(),
                      count[0] * real_packet,
                      MPI_DOUBLE, target, tag+2, comm, &reqs[1]);
        }
        if(recvn == 2){
            MPI_Irecv(recv_real_buf.GetData() + (count[0] * real_packet),
                      count[1] * real_packet,
                      MPI_DOUBLE, rank - 1, tag+2, comm, &reqs[2]);
        }
        MPI_Isend(send_real_buf.GetData(), send_n * real_packet, MPI_DOUBLE, target, tag+2, comm, &reqs[0]);
        MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);



        
        // append received data
        for(int i = 0; i < nrecv; i++){
            ranks.Append(recv_buffer[i * int_packet]);
            for(int j = 0; j < static_cast<int>(int_data.size()); j++){
                (*int_data[j]).Append(recv_buffer[i * int_packet + 1 + j]);
            }
            for(int j = 0; j < nrdata; j++){
                (*real_data[j]).Append(recv_real_buf[i * real_packet + j]);
            }
        }

    }
} //mfem

#endif // MFEM_USE_MPI