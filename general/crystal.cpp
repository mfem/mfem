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

void CrystalRouter::Route(ParticleVector &data, Array<unsigned long long> &ids,
                          Array<int> &tags, Array<int> &ranks)
{
   
   // edge cases where no particles to route, still need to derive tag_vdim for packing/unpacking
   int local_tag_vdim = 0;
   if(ranks.Size() > 0){
      local_tag_vdim = tags.Size() / ranks.Size();
   }
   MPI_Allreduce(&local_tag_vdim, &tag_vdim, 1, MPI_INT, MPI_MAX, comm);

   uint32_t bl = 0, bh, nl;
   uint32_t id = rank, n = nprocs;
   uint32_t targ, msg_tag = 0;
   bool send_hi;
   int recvn;

   while (n > 1)
   {
      nl = (n+1)/2;
      bh = bl + nl;
      send_hi = (id < bh);
      Move(data, ids, tags, ranks, bh, send_hi);

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

      Exchange(data, ids, tags, ranks, targ, recvn, msg_tag);

      if (id < bh)
      {
         n = nl;
      }
      else
      {
         n -= nl;
         bl = bh;
      }
      // 2 message tags consumed per round: count, packed-byte payload
      msg_tag += 2;
   }
}

// PRIVATE METHODS

void CrystalRouter::Move(ParticleVector &data, Array<unsigned long long> &ids,
                         Array<int> &tags, Array<int> &ranks,
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

   const size_t tag_bytes  = (size_t)tag_vdim * sizeof(int);
   const size_t id_bytes   = sizeof(unsigned long long);
   const size_t data_bytes = (size_t)vdim * sizeof(real_t);
   // rank | tags | id | data
   const size_t total_bytes = sizeof(int) + tag_bytes + id_bytes + data_bytes;

   send_buf.SetSize(nsend * total_bytes);

   // gather sends: pack each particle's columns into one contiguous byte row
   Vector slice;
   for (int i = 0; i < nsend; i++)
   {
      int idx = send_indices[i];
      char *row = send_buf.GetData() + i * total_bytes;
      size_t off = 0;
      int r = ranks[idx];
      std::memcpy(row + off, &r, sizeof(int));                  off += sizeof(int);
      std::memcpy(row + off, tags.GetData() + idx*tag_vdim, tag_bytes);
      off += tag_bytes;
      unsigned long long gid = ids[idx];
      std::memcpy(row + off, &gid, id_bytes);                   off += id_bytes;
      data.GetValues(idx, slice);                               // ordering-aware read
      std::memcpy(row + off, slice.GetData(), data_bytes);      off += data_bytes;
   }

   // compact kept particles in place (shared index across every column)
   for (int k = 0; k < nkeep; k++)
   {
      int idx = keep_indices[k];
      if (k != idx)
      {
         ranks[k] = ranks[idx];
         ids[k] = ids[idx];
         for (int c = 0; c < tag_vdim; c++)
         {
            tags[k*tag_vdim + c] = tags[idx*tag_vdim + c];
         }
         data.GetValues(idx, slice);
         data.SetValues(k, slice);
      }
   }

   // update kept sizes (ordering-aware truncate for the real column)
   ranks.SetSize(nkeep);
   ids.SetSize(nkeep);
   tags.SetSize(nkeep * tag_vdim);
   data.SetNumParticles(nkeep, true);
}

void CrystalRouter::Exchange(ParticleVector &data, Array<unsigned long long> &ids,
                             Array<int> &tags, Array<int> &ranks,
                             int target, int recvn, int msg_tag)
{
   int vdim = data.GetVDim();
   const size_t tag_bytes  = (size_t)tag_vdim * sizeof(int);
   const size_t id_bytes   = sizeof(unsigned long long);
   const size_t data_bytes = (size_t)vdim * sizeof(real_t);
   const size_t total_bytes = sizeof(int) + tag_bytes + id_bytes + data_bytes;

   // exchange counts
   MPI_Request reqs[3];
   int count[2] = {0, 0};
   int send_n = send_buf.Size() / total_bytes;

   if (recvn >= 1)
   {
      MPI_Irecv(&count[0], 1, MPI_INT32_T, target, msg_tag, comm, &reqs[1]);
   }
   if (recvn == 2)
   {
      MPI_Irecv(&count[1], 1, MPI_INT32_T, rank - 1, msg_tag, comm, &reqs[2]);
   }
   MPI_Isend(&send_n, 1, MPI_INT32_T, target, msg_tag, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   int nrecv = count[0] + count[1];

   MFEM_VERIFY((size_t)send_n * total_bytes <= INT_MAX &&
               (size_t)nrecv * total_bytes <= INT_MAX,
               "CrystalRouter payload exceeds MPI int-count limit");
   recv_buf.SetSize(nrecv * total_bytes);

   if (recvn >= 1)
   {
      MPI_Irecv(recv_buf.GetData(), count[0] * total_bytes,
                MPI_BYTE, target, msg_tag+1, comm, &reqs[1]);
   }
   if (recvn == 2)
   {
      MPI_Irecv(recv_buf.GetData() + count[0] * total_bytes, count[1] * total_bytes,
                MPI_BYTE, rank - 1, msg_tag+1, comm, &reqs[2]);
   }
   MPI_Isend(send_buf.GetData(), send_n * total_bytes, MPI_BYTE, target,
             msg_tag+1, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   // append received rows. Grow the real column once, then unpack each row and
   // write per-particle slices (ordering-aware) so the layout is respected.
   int old_n = ranks.Size();
   data.SetNumParticles(old_n + nrecv, true);

   Vector slice(vdim);
   for (int i = 0; i < nrecv; i++)
   {
      const char *row = recv_buf.GetData() + i * total_bytes;
      size_t off = 0;
      int r;
      std::memcpy(&r, row + off, sizeof(int));
      off += sizeof(int);
      ranks.Append(r);
      for (int c = 0; c < tag_vdim; c++)
      {
         int t;
         std::memcpy(&t, row + off, sizeof(int));
         off += sizeof(int);
         tags.Append(t);
      }
      unsigned long long gid;
      std::memcpy(&gid, row + off, id_bytes);
      off += id_bytes;
      ids.Append(gid);
      std::memcpy(slice.GetData(), row + off, data_bytes);
      off += data_bytes;
      data.SetValues(old_n + i, slice);
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
