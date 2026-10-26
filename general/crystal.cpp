#include "crystal.hpp"
#include "communication.hpp"

#ifdef MFEM_USE_MPI
#include <cstring>
#include <climits>
#include <cstdint>
#include <cstdio>

namespace mfem
{

/// @details Duplicates @a comm and caches the rank and size of the duplicate
CrystalRouter::CrystalRouter(MPI_Comm comm) {
   MPI_Comm_dup(comm, &this->comm);
   MPI_Comm_rank(this->comm, &rank);
   MPI_Comm_size(this->comm, &nprocs);
}

/// @details The communicator is not freed after MPI_Finalize, which would be
/// an error. This happens when a router outlives the Mpi session.
CrystalRouter::~CrystalRouter() {
   if (!Mpi::IsFinalized()) { MPI_Comm_free(&comm); }
}

/** @details Flattens the field groups into one list of Field pointers, checks
    that every field has one entry per rank in @a rank_list, then picks the
    path: RouteDevice() if a Device is enabled, else RouteInternal() on a copy
    of @a rank_list */
void CrystalRouter::Route(const Array<unsigned int> &rank_list,
                          std::initializer_list<CR::FieldGroup> field_groups)
{
   std::vector<CR::Field*> fields;
   for (const auto &g : field_groups) {
      for (const auto &f : g.cols) { fields.push_back(f.get()); }
   }
   for (std::size_t i = 0; i < fields.size(); i++) {
      MFEM_VERIFY(fields[i]->GetNumParticles() == rank_list.Size(),
                  "CrystalRouter: field " << i << " has size "
                  << fields[i]->GetNumParticles() << " but rank_list has size "
                  << rank_list.Size());
   }

   if (Device::IsEnabled()) {
      RouteDevice(rank_list, fields);
   }
   else {
      Array<unsigned int> ranks(rank_list);
      RouteInternal(ranks, fields);
   }
}

// PRIVATE METHODS

/** @details Steps:
    -# Split the particle indices into leaving (rank differs from this rank)
       and staying
    -# Copy only the leaving particles of each field to the host (Field::ToHost)
    -# Route the host copies with RouteInternal()
    -# Rebuild each device field from the staying particles plus the arrivals
       (Field::ToDevice)

    If nothing left and nothing arrived, the fields are not touched */
void CrystalRouter::RouteDevice(const Array<unsigned int> &rank_list,
                                const std::vector<CR::Field*> &fields)
{
   // split into leaving/staying
   const int n = rank_list.Size();
   const unsigned int *h_ranks = rank_list.HostRead();
   Array<int> leaving_idx, staying_idx;
   Array<unsigned int> ranks;
   for (int i = 0; i < n; i++) {
      if (h_ranks[i] != (unsigned int)rank) {
         leaving_idx.Append(i);
         ranks.Append(h_ranks[i]);
      }
      else {
         staying_idx.Append(i);
      }
   }

   // pull only the leaving particles into host buffers
   std::vector<std::unique_ptr<CR::Field>> host_fields;
   std::vector<CR::Field*> host_field_ptrs;
   host_fields.reserve(fields.size());
   host_field_ptrs.reserve(fields.size());
   for (auto *f : fields) {
      host_fields.push_back(f->ToHost(leaving_idx));
      host_field_ptrs.push_back(host_fields.back().get());
   }

   // route
   RouteInternal(ranks, host_field_ptrs);

   // nothing left or arrived: fields are already correct
   if (leaving_idx.Size() == 0 && ranks.Size() == 0) { return; }

   // return to device
   for (std::size_t k = 0; k < fields.size(); k++) {
      fields[k]->ToDevice(staying_idx, *host_fields[k]);
   }
}

/** @details Recursive halving over the ranks. The active group of @c n ranks
    starts at rank @c bl. In each round:
    -# The group is split at @c bh = @c bl + ceil(n/2). Ranks below @c bh are
       the lower half.
    -# Move() packs the particles whose destination is in the other half
    -# Exchange() sends them to the partner @c targ, the mirror rank in the
       other half. If @c n is odd, the middle rank has no mirror: it sends to
       @c bh and receives nothing, and @c bh receives from both its mirror and
       the middle rank (@c recvn = 2).
    -# Each rank continues in its own half

    After about log2(P) rounds, every particle is on its destination rank.
    @c msg_tag grows by 2 per round (one tag for counts, one for data). */
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

/** @details Each sent row in #send_buf is the destination rank
    (unsigned int) followed by the packed row of every field, in list order. The
    rows that stay are compacted to the front of the fields in place, then all
    fields and @a ranks are shrunk to the kept size. Fails if the send buffer
    would exceed the size limit of mfem::Array (INT_MAX bytes). */
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
   for (const auto *f : fields) { field_bytes += f->GetParticleBytes(); }
   const size_t total_bytes = sizeof(unsigned int) + field_bytes;

   const size_t send_bytes = (size_t)nsend * total_bytes;
   MFEM_VERIFY(send_bytes <= (size_t)INT_MAX,
               "CrystalRouter send buffer exceeds Array size limit");
   send_buf.SetSize((int)send_bytes);

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
         off += f->GetParticleBytes();
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
   for (auto *f : fields) { f->SetNumParticles(nkeep); }
}

/** @details Two non-blocking phases, each completed with MPI_Waitall:
    -# Exchange the row counts (tag @a msg_tag)
    -# Exchange the packed rows (tag @a msg_tag + 1). A second receive from
       rank - 1 is posted when @a recvn is 2, and its rows land after those of
       @a target in #recv_buf.

    The fields are then grown by the number of received rows, and each row is
    unpacked into the new slots. Fails if a payload exceeds the MPI int-count
    limit (INT_MAX bytes). */
void CrystalRouter::Exchange(Array<unsigned int> &ranks,
                             const std::vector<CR::Field*> &fields,
                             int target, int recvn, int msg_tag)
{
   size_t field_bytes = 0;
   for (const auto *f : fields) { field_bytes += f->GetParticleBytes(); }
   const size_t total_bytes = sizeof(unsigned int) + field_bytes;

   // exchange counts
   MPI_Request reqs[3];
   int count[2] = {0, 0};
   int send_n = send_buf.Size() / total_bytes;

   if (recvn >= 1) {
      MPI_Irecv(&count[0], 1, MPI_INT, target, msg_tag, comm, &reqs[1]);
   }
   if (recvn == 2) {
      MPI_Irecv(&count[1], 1, MPI_INT, rank - 1, msg_tag, comm, &reqs[2]);
   }
   MPI_Isend(&send_n, 1, MPI_INT, target, msg_tag, comm, &reqs[0]);
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
   for (auto *f : fields) { f->SetNumParticles(old_n + nrecv); }

   for (int i = 0; i < nrecv; i++) {
      const char *row = recv_buf.GetData() + i * total_bytes;
      size_t off = 0;
      unsigned int r;
      std::memcpy(&r, row + off, sizeof(unsigned int));
      off += sizeof(unsigned int);
      ranks.Append(r);
      for (auto *f : fields) {
         f->Unpack(old_n + i, row + off);
         off += f->GetParticleBytes();
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
