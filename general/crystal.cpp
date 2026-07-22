#include "crystal.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <cstdint>
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

void CrystalRouter::Route(const Array<unsigned int> &rank_list,
                          std::initializer_list<CR::FieldGroup> field_groups)
{
   std::vector<CR::Field*> fields;
   for (const auto &g : field_groups) {
      for (const auto &f : g.cols) { fields.push_back(f.get()); }
   }
   for (const auto *f : fields) {
      MFEM_VERIFY(f->Size() == rank_list.Size(),
                  "CrystalRouter: field size does not match rank_list size");
   }

   // copy to preserve rank list
   Array<unsigned int> ranks(rank_list);
   RouteInternal(ranks, fields);
}

// PRIVATE METHODS
void CrystalRouter::RouteInternal(Array<unsigned int> &ranks,
                                  const std::vector<CR::Field*> &fields)
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
      Move(ranks, fields, bh, send_hi);

      recvn = 1, targ = n-1-(id-bl)+bl;
      if (id == targ) {
         targ = bh;
         recvn = 0;
      }
      if (n&1 && id==bh) {
         recvn = 2;
      }

      Exchange(ranks, fields, targ, recvn, msg_tag);

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

void CrystalRouter::Move(Array<unsigned int> &ranks,
                         const std::vector<CR::Field*> &fields,
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

   // rank | field blocks in list order
   size_t field_bytes = 0;
   for (const auto *f : fields) { field_bytes += f->RowBytes(); }
   const size_t total_bytes = sizeof(unsigned int) + field_bytes;

   send_buf.SetSize(nsend * total_bytes);

   // gather sends: pack each particle's columns into one contiguous byte row
   for (int i = 0; i < nsend; i++) {
      int idx = send_indices[i];
      char *row = send_buf.GetData() + i * total_bytes;
      size_t off = 0;
      unsigned int r = ranks[idx];

      std::memcpy(row + off, &r, sizeof(unsigned int));
      off += sizeof(unsigned int);
      for (auto *f : fields) {
         f->Pack(idx, row + off);
         off += f->RowBytes();
      }
   }

   // compact kept particles in place (shared index across every column)
   for (int k = 0; k < nkeep; k++) {
      int idx = keep_indices[k];
      if (k != idx) {
         ranks[k] = ranks[idx];
         for (auto *f : fields) { f->Copy(idx, k); }
      }
   }

   // update kept sizes (each field resizes ordering-aware)
   ranks.SetSize(nkeep);
   for (auto *f : fields) { f->Resize(nkeep); }
}

void CrystalRouter::Exchange(Array<unsigned int> &ranks,
                             const std::vector<CR::Field*> &fields,
                             int target, int recvn, int msg_tag)
{
   size_t field_bytes = 0;
   for (const auto *f : fields) { field_bytes += f->RowBytes(); }
   const size_t total_bytes = sizeof(unsigned int) + field_bytes;

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

   int old_n = ranks.Size();
   for (auto *f : fields) { f->Resize(old_n + nrecv); }

   for (int i = 0; i < nrecv; i++) {
      const char *row = recv_buf.GetData() + i * total_bytes;
      size_t off = 0;
      unsigned int r;
      std::memcpy(&r, row + off, sizeof(unsigned int));
      off += sizeof(unsigned int);
      ranks.Append(r);
      for (auto *f : fields) {
         f->Unpack(old_n + i, row + off);
         off += f->RowBytes();
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
