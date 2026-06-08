#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <cstdint>
#include <memory>
#include <cstdio>

namespace mfem
{

CrystalRouter::CrystalRouter(MPI_Comm comm)
{
   MPI_Comm_dup(comm, &this->comm);
   MPI_Comm_rank(this->comm, &rank);
   MPI_Comm_size(this->comm, &nprocs);
}

CrystalRouter::~CrystalRouter()
{
   MPI_Comm_free(&comm);
}

void CrystalRouter::Route(ParticleVector &data, Array<int> &ranks)
{
   uint32_t bl = 0, bh, nl;
   uint32_t id = rank, n = nprocs;
   uint32_t targ, tag = 0;
   bool send_hi;
   int recvn;

   while (n > 1)
   {
      nl = (n+1)/2;
      bh = bl + nl;
      send_hi = (id < bh);
      Move(data, ranks, bh, send_hi);

      recvn = 1, targ = n-1-(id-bl)+bl;
      if (id == targ)
      {
         targ = bh;
         recvn = 0;
      }
      if (n&1 && id==bh)
      {
         recvn = 2;
      }

      Exchange(data, ranks, targ, recvn, tag);

      if (id < bh)
      {
         n = nl;
      }
      else
      {
         n -= nl;
         bl = bh;
      }
      tag += 3;
   }
}

// PRIVATE METHODS

void CrystalRouter::Move(ParticleVector &data, Array<int> &ranks,
                         int cutoff, bool send_hi)
{
   int n = ranks.Size();

   std::vector<int> send_indices;
   std::vector<int> keep_indices;
   send_indices.reserve(n);
   keep_indices.reserve(n);

   // classify particles into send vs keep
   for (int i = 0; i < n; i++)
   {
      int r = ranks[i];
      if ((send_hi && r >= cutoff) || (!send_hi && r < cutoff))
      {
         send_indices.push_back(i);
      }
      else
      {
         keep_indices.push_back(i);
      }
   }
   int nsend = send_indices.size();
   int nkeep = keep_indices.size();
   int vdim = data.GetVDim();

   send_rank_buf.SetSize(nsend);
   send_data_buf.SetSize(nsend * vdim);

   // gather sends (rank + particle slice, straight into the flat buffer)
   Vector slice;
   for (int i = 0; i < nsend; i++)
   {
      int idx = send_indices[i];
      send_rank_buf[i] = ranks[idx];
      data.GetValues(idx, slice);        // ordering-aware read
      for (int c = 0; c < vdim; c++)
      {
         send_data_buf[i*vdim + c] = slice[c];
      }
   }

   // compact kept particles in place (shared index across rank + data)
   for (int k = 0; k < nkeep; k++)
   {
      int idx = keep_indices[k];
      if (k != idx)
      {
         ranks[k] = ranks[idx];
         data.GetValues(idx, slice);
         data.SetValues(k, slice);
      }
   }

   // update kept sizes (ordering-aware truncate for the real column)
   ranks.SetSize(nkeep);
   data.SetNumParticles(nkeep, true);
}

void CrystalRouter::Exchange(ParticleVector &data, Array<int> &ranks,
                             int target, int recvn, int tag)
{
   // exchange counts
   MPI_Request reqs[3];
   int count[2] = {0, 0};
   int send_n = send_rank_buf.Size();

   if (recvn >= 1)
   {
      MPI_Irecv(&count[0], 1, MPI_INT32_T, target, tag, comm, &reqs[1]);
   }
   if (recvn == 2)
   {
      MPI_Irecv(&count[1], 1, MPI_INT32_T, rank - 1, tag, comm, &reqs[2]);
   }
   MPI_Isend(&send_n, 1, MPI_INT32_T, target, tag, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   int nrecv = count[0] + count[1];

   // exchange ranks
   Array<int> recv_ranks(nrecv);
   if (recvn >= 1)
   {
      MPI_Irecv(recv_ranks.GetData(), count[0],
                MPI_INT32_T, target, tag+1, comm, &reqs[1]);
   }
   if (recvn == 2)
   {
      MPI_Irecv(recv_ranks.GetData() + count[0], count[1],
                MPI_INT32_T, rank - 1, tag+1, comm, &reqs[2]);
   }
   MPI_Isend(send_rank_buf.GetData(), send_n, MPI_INT32_T, target, tag+1, comm,
             &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);




   // exchange real data: send_data_buf is already flat (vdim reals/particle)
   int vd = data.GetVDim();
   Array<real_t> recv_buf(nrecv * vd);

   if (recvn >= 1)
   {
      MPI_Irecv(recv_buf.GetData(), count[0] * vd,
                MPI_DOUBLE, target, tag+2, comm, &reqs[1]);
   }
   if (recvn == 2)
   {
      MPI_Irecv(recv_buf.GetData() + (count[0] * vd), count[1] * vd,
                MPI_DOUBLE, rank - 1, tag+2, comm, &reqs[2]);
   }
   MPI_Isend(send_data_buf.GetData(), send_n * vd, MPI_DOUBLE, target, tag+2,
             comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);




   // append received data. Grow the real column once, then write per-particle
   // slices (ordering-aware) so the destination layout is respected.
   int old_n = ranks.Size();
   data.SetNumParticles(old_n + nrecv, true);

   Vector slice(vd);
   for (int i = 0; i < nrecv; i++)
   {
      ranks.Append(recv_ranks[i]);
      for (int c = 0; c < vd; c++)
      {
         slice[c] = recv_buf[i * vd + c];
      }
      data.SetValues(old_n + i, slice);
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
