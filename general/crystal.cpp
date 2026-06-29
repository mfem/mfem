#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <cstdint>
#include <memory>
#include <cstdio>

namespace mfem
{

CrystalRouter::CrystalRouter(MPI_Comm comm) {
   MPI_Comm_dup(comm, &this->comm);
   MPI_Comm_rank(this->comm, &rank);
   MPI_Comm_size(this->comm, &nprocs);
}

CrystalRouter::~CrystalRouter() {
   MPI_Comm_free(&comm);
}

void CrystalRouter::Route(Array<unsigned int> &ranks,
                          Array<unsigned long long> &ids,
                          const std::vector<Array<int>*> &tags,
                          const std::vector<ParticleVector*> &data)
{

   uint32_t bl = 0, bh, nl;
   uint32_t id = rank, n = nprocs;
   uint32_t targ, msg_tag = 0;
   bool send_hi;
   int recvn;

   while (n > 1) {
      nl = (n+1)/2;
      bh = bl + nl;
      send_hi = (id < bh);
      Move(ranks, ids, tags, data, bh, send_hi);

      recvn = 1, targ = n-1-(id-bl)+bl;
      if (id == targ) {
         targ = bh;
         recvn = 0;
      }
      if (n&1 && id==bh) {
         recvn = 2;
      }

      Exchange(ranks, ids, tags, data, targ, recvn, msg_tag);

      if (id < bh) {
         n = nl;
      }
      else {
         n -= nl;
         bl = bh;
      }
      msg_tag += 2;
   }
}

// PRIVATE METHODS

void CrystalRouter::Move(Array<unsigned int> &ranks,
                         Array<unsigned long long> &ids,
                         const std::vector<Array<int>*> &tags,
                         const std::vector<ParticleVector*> &data,
                         unsigned int cutoff, bool send_hi)
{
   int n = ranks.Size();

   std::vector<int> send_indices;
   std::vector<int> keep_indices;
   send_indices.reserve(n);
   keep_indices.reserve(n);

   // classify particles into send vs keep
   for (int i = 0; i < n; i++) {
      unsigned int r = ranks[i];
      if ((send_hi && r >= cutoff) || (!send_hi && r < cutoff)) {
         send_indices.push_back(i);
      }
      else {
         keep_indices.push_back(i);
      }
   }
   int nsend = send_indices.size();
   int nkeep = keep_indices.size();

   // one int32 per tag column, fixed across the route
   const size_t tag_bytes  = tags.size() * sizeof(int);
   const size_t id_bytes   = sizeof(unsigned long long);

   // data block is all ParticleVectors concatenated
   size_t data_bytes = 0;
   for (auto *pv : data) { data_bytes += (size_t)pv->GetVDim() * sizeof(real_t); }

   // rank | tags | id | data
   const size_t total_bytes = sizeof(unsigned int) + tag_bytes + id_bytes + data_bytes;

   send_buf.SetSize(nsend * total_bytes);

   // gather sends: pack each particle's columns into one contiguous byte row
   Vector slice;
   for (int i = 0; i < nsend; i++) {
      int idx = send_indices[i];
      char *row = send_buf.GetData() + i * total_bytes;
      size_t off = 0;
      unsigned int r = ranks[idx];
      unsigned long long gid = ids[idx];

      // pack rank -> tags -> id -> data
      std::memcpy(row + off, &r, sizeof(unsigned int));                    off += sizeof(unsigned int);
      for (auto *tag : tags) {
         int t = (*tag)[idx];
         std::memcpy(row + off, &t, sizeof(int));                          off += sizeof(int);
      }
      std::memcpy(row + off, &gid, id_bytes);                              off += id_bytes;
      for (auto *pv : data) {
         const size_t b = (size_t)pv->GetVDim() * sizeof(real_t);
         pv->GetValues(idx, slice);
         std::memcpy(row + off, slice.GetData(), b);                       off += b;
      }
   }

   // compact kept particles in place (shared index across every column)
   for (int k = 0; k < nkeep; k++) {
      int idx = keep_indices[k];
      if (k != idx) {
         ranks[k] = ranks[idx];
         ids[k] = ids[idx];
         for (auto *tag : tags) { 
            (*tag)[k] = (*tag)[idx];
         }
         for (auto *pv : data) {
            pv->GetValues(idx, slice);
            pv->SetValues(k, slice);
         }
      }
   }

   // update kept sizes (ordering-aware truncate for the real columns)
   ranks.SetSize(nkeep);
   ids.SetSize(nkeep);
   for (auto *tag : tags) { tag->SetSize(nkeep); }
   for (auto *pv : data) { pv->SetNumParticles(nkeep, true); }
}

void CrystalRouter::Exchange(Array<unsigned int> &ranks,
                             Array<unsigned long long> &ids,
                             const std::vector<Array<int>*> &tags,
                             const std::vector<ParticleVector*> &data,
                             int target, int recvn, int msg_tag)
{
   const size_t tag_bytes     = tags.size() * sizeof(int);
   const size_t id_bytes      = sizeof(unsigned long long);
   size_t data_bytes = 0;
   for (auto *pv : data) { data_bytes += (size_t)pv->GetVDim() * sizeof(real_t); }
   const size_t total_bytes   = sizeof(unsigned int) + tag_bytes + id_bytes + data_bytes;

   // exchange counts
   MPI_Request reqs[3];
   int count[2] = {0, 0};
   int send_n = send_buf.Size() / total_bytes;

   if (recvn >= 1) {
      MPI_Irecv(&count[0], 1, MPI_INT32_T, target, msg_tag, comm, &reqs[1]);
   }
   if (recvn == 2) {
      MPI_Irecv(&count[1], 1, MPI_INT32_T, rank - 1, msg_tag, comm, &reqs[2]);
   }
   MPI_Isend(&send_n, 1, MPI_INT32_T, target, msg_tag, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   int nrecv = count[0] + count[1];

   MFEM_VERIFY((size_t)send_n * total_bytes <= INT_MAX &&
               (size_t)nrecv * total_bytes <= INT_MAX,
               "CrystalRouter payload exceeds MPI int-count limit");
   recv_buf.SetSize(nrecv * total_bytes);

   // exchange buffers
   if (recvn >= 1) {
      MPI_Irecv(recv_buf.GetData(), count[0] * total_bytes,
                MPI_BYTE, target, msg_tag+1, comm, &reqs[1]);
   }
   if (recvn == 2) {
      MPI_Irecv(recv_buf.GetData() + count[0] * total_bytes, count[1] * total_bytes,
                MPI_BYTE, rank - 1, msg_tag+1, comm, &reqs[2]);
   }
   MPI_Isend(send_buf.GetData(), send_n * total_bytes, MPI_BYTE, target,
             msg_tag+1, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   // append received rows
   int old_n = ranks.Size();
   for (auto *pv : data) { pv->SetNumParticles(old_n + nrecv, true); }

   Vector slice;
   for (int i = 0; i < nrecv; i++) {
      const char *row = recv_buf.GetData() + i * total_bytes;
      size_t off = 0;
      unsigned int r;
      std::memcpy(&r, row + off, sizeof(unsigned int));
      off += sizeof(unsigned int);
      ranks.Append(r);
      for (auto *tag : tags) {
         int t;
         std::memcpy(&t, row + off, sizeof(int));
         off += sizeof(int);
         tag->Append(t);
      }
      unsigned long long gid;
      std::memcpy(&gid, row + off, id_bytes);
      off += id_bytes;
      ids.Append(gid);
      for (auto *pv : data) {
         const int vd = pv->GetVDim();
         slice.SetSize(vd);
         std::memcpy(slice.GetData(), row + off, (size_t)vd * sizeof(real_t));
         off += (size_t)vd * sizeof(real_t);
         pv->SetValues(old_n + i, slice);
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
