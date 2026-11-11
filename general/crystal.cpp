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

CrystalRouter::~CrystalRouter() {
   if (!Mpi::IsFinalized()) { MPI_Comm_free(&comm); }
}

void CrystalRouter::Route(const Array<unsigned int> &rank_list,
                          std::initializer_list<CR::internal::FieldGroup> field_groups)
{
   std::vector<CR::internal::Field*> fields;
   for (const auto &g : field_groups) {
      for (const auto &f : g.cols) { fields.push_back(f.get()); }
   }
   for (std::size_t i = 0; i < fields.size(); i++) {
      MFEM_VERIFY(fields[i]->GetNumParticles() == rank_list.Size(),
                  "CrystalRouter: field " << i << " has size "
                  << fields[i]->GetNumParticles() << " but rank_list has size "
                  << rank_list.Size());
   }

   // row layout: columns (aligned), destination rank, padding
   auto align_up = [](std::size_t x, std::size_t a) { return (x + a - 1) / a * a; };
   std::vector<std::size_t> off(fields.size());
   // starts at 8: a row must be a whole number of the 8-byte words of buf
   std::size_t pos = 0, max_align = sizeof(std::uint64_t);
   for (std::size_t k = 0; k < fields.size(); k++) {
      const std::size_t a = fields[k]->GetAlign();
      pos = align_up(pos, a);
      off[k] = pos;
      pos += fields[k]->GetParticleBytes();
      max_align = std::max(max_align, a);
   }
   const std::size_t hdr = align_up(pos, alignof(unsigned int));
   const std::size_t stride = align_up(hdr + sizeof(unsigned int), max_align);

   MPI_Datatype row_t;
   MPI_Type_contiguous((int)stride, MPI_BYTE, &row_t);
   MPI_Type_commit(&row_t);

   // split into leaving/staying
   const int n = rank_list.Size();
   const unsigned int *h_ranks = n > 0 ? rank_list.HostRead() : nullptr;
   Array<int> leaving_idx;
   for (int i = 0; i < n; i++) {
      if (h_ranks[i] != (unsigned int)rank) { leaving_idx.Append(i); }
   }
   const int n_leave = leaving_idx.Size();

   bool any_dev = false;
   for (const auto *f : fields) { any_dev = any_dev || f->OnDevice(); }

   // pack the leaving rows
   buf.SetSize(0);
   const std::size_t words = (std::size_t)n_leave * stride / sizeof(std::uint64_t);
   MFEM_VERIFY(words <= (std::size_t)INT_MAX / 2,
               "CrystalRouter send buffer exceeds Array size limit");
   buf.SetSize((int)words);

   if (n_leave > 0) {
      if (any_dev) {
         char *d = reinterpret_cast<char*>(buf.Write(true));
         for (std::size_t k = 0; k < fields.size(); k++) {
            if (fields[k]->OnDevice()) { fields[k]->Pack(leaving_idx, d, stride, off[k]); }
         }
      }
      // the only device to host copy of buf, after the device packs
      char *h = reinterpret_cast<char*>(buf.HostReadWrite());
      for (std::size_t k = 0; k < fields.size(); k++) {
         if (!fields[k]->OnDevice()) { fields[k]->Pack(leaving_idx, h, stride, off[k]); }
      }
      for (int k = 0; k < n_leave; k++) {
         std::memcpy(h + (std::size_t)k * stride + hdr, &h_ranks[leaving_idx[k]],
                     sizeof(unsigned int));
      }
   }

   // route: afterwards buf holds the n_arr arrivals
   const int n_arr = RouteInternal(n_leave, stride, hdr, row_t);
   MPI_Type_free(&row_t);

   // nothing left or arrived: fields are already correct
   if (n_leave == 0 && n_arr == 0) { return; }

   // fill list: dst[m] is the slot that gets arrival m, then tail[m - n_arr]
   const int n_new = n - n_leave + n_arr;
   Array<int> dst, tail;
   int k = 0;
   for (; k < n_leave && leaving_idx[k] < n_new; k++) { dst.Append(leaving_idx[k]); } // holes
   for (int p = n; p < n_new; p++) { dst.Append(p); }                                  // new slots
   for (int p = n_new; p < n; p++) {                                                   // stayers past the end
      if (k < n_leave && leaving_idx[k] == p) { k++; } else { tail.Append(p); }
   }
   MFEM_ASSERT(dst.Size() == n_arr + tail.Size(), "CrystalRouter: bad fill list");

   // unpack: host columns first, then device columns
   const char *h = n_arr > 0 ? reinterpret_cast<const char*>(buf.HostRead()) : nullptr;
   for (std::size_t j = 0; j < fields.size(); j++) {
      if (!fields[j]->OnDevice()) {
         fields[j]->Unpack(dst, tail, h, n_arr, n_new, stride, off[j]);
      }
   }
   if (any_dev) {
      // the only host to device copy of buf
      const char *d = n_arr > 0 ? reinterpret_cast<const char*>(buf.Read(true)) : nullptr;
      for (std::size_t j = 0; j < fields.size(); j++) {
         if (fields[j]->OnDevice()) {
            fields[j]->Unpack(dst, tail, d, n_arr, n_new, stride, off[j]);
         }
      }
   }
}

// PRIVATE METHODS

/** @details Recursive halving over the ranks. In each round:
    -# The group is split at @c bh = @c bl + ceil(n/2). Ranks below @c bh are
       the lower half.
    -# Move() puts the rows whose destination is in the other half at the end
       of #buf
    -# Exchange() sends them to the partner @c targ, the mirror rank in the
       other half, and appends the rows it receives
    -# Each rank continues in its own half

    @c n here is the number of ranks in the group. The number of rows is
    @a nrows. */
int CrystalRouter::RouteInternal(int nrows, std::size_t stride, std::size_t hdr,
                                 MPI_Datatype row_t)
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
      const int nkeep = Move(nrows, stride, hdr, bh, send_hi);

      recvn = 1, targ = n-1-(id-bl)+bl;
      if (id == targ) {
         targ = bh;
         recvn = 0;
      }
      if (n&1 && id==bh) {
         recvn = 2;
      }

      nrows = Exchange(nkeep, nrows - nkeep, stride, row_t, targ, recvn, msg_tag);

      if (id < bh) {
         n = nl;
      }
      else {
         n -= nl;
         bl = bh;
      }
      msg_tag += 2;
   }
   return nrows;
}

/** @details Two-pointer partition of the rows of #buf, on the host. Each
    row ends with its destination rank at byte @a hdr. Rows that stay move to
    the front, rows that leave move to the end. The order inside each part is
    not kept. */
int CrystalRouter::Move(int nrows, std::size_t stride, std::size_t hdr,
                        unsigned int cutoff, bool send_hi)
{
   if (nrows == 0) { return 0; }
   char *b = reinterpret_cast<char*>(buf.HostReadWrite());
   auto sends = [&](int i) {
      unsigned int r;
      std::memcpy(&r, b + (std::size_t)i * stride + hdr, sizeof r);
      return send_hi ? r >= cutoff : r < cutoff;
   };
   int lo = 0, hi = nrows;   // [0,lo) keep, [hi,nrows) send
   while (true) {
      while (lo < hi && !sends(lo)) { lo++; }
      while (lo < hi && sends(hi-1)) { hi--; }
      if (lo >= hi) { break; }
      std::swap_ranges(b + (std::size_t)lo * stride, b + (std::size_t)(lo+1) * stride,
                       b + (std::size_t)(hi-1) * stride);
      lo++; hi--;
   }
   return lo;
}

/** @details
    -# Exchange the row counts (tag @a msg_tag)
    -# Resize #buf to hold the kept, sent and received rows
    -# Exchange the rows (tag @a msg_tag + 1), second receive from rank - 1
       when @a recvn is 2. The sent rows are read from the rows after the kept
       rows, the received rows are written after the sent rows.
    -# Move the last received rows over the sent rows, then shrink #buf to the
       kept and received rows. The received rows now follow the kept rows. */
int CrystalRouter::Exchange(int nkeep, int nsend, std::size_t stride,
                            MPI_Datatype row_t, int target, int recvn,
                            int msg_tag)
{
   // exchange counts
   MPI_Request reqs[3];
   int count[2] = {0, 0};
   int send_n = nsend;

   if (recvn >= 1) {
      MPI_Irecv(&count[0], 1, MPI_INT, target, msg_tag, comm, &reqs[1]);
   }
   if (recvn == 2) {
      MPI_Irecv(&count[1], 1, MPI_INT, rank - 1, msg_tag, comm, &reqs[2]);
   }
   MPI_Isend(&send_n, 1, MPI_INT, target, msg_tag, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   const int nrecv = count[0] + count[1];

   const std::size_t words = ((std::size_t)nkeep + nsend + nrecv) * stride
                             / sizeof(std::uint64_t);
   MFEM_VERIFY(words <= (std::size_t)INT_MAX / 2,
               "CrystalRouter buffer exceeds Array size limit");
   buf.SetSize((int)words);
   // after SetSize, which can reallocate
   char *b = words > 0 ? reinterpret_cast<char*>(buf.HostReadWrite()) : nullptr;
   char *send = b ? b + (std::size_t)nkeep * stride : nullptr;
   char *recv = b ? send + (std::size_t)nsend * stride : nullptr;

   // exchange rows (counts are in rows of type row_t)
   if (recvn >= 1) {
      MPI_Irecv(recv, count[0], row_t, target, msg_tag+1, comm, &reqs[1]);
   }
   if (recvn == 2) {
      MPI_Irecv(recv ? recv + (std::size_t)count[0] * stride : nullptr, count[1],
                row_t, rank - 1, msg_tag+1, comm, &reqs[2]);
   }
   MPI_Isend(send, nsend, row_t, target, msg_tag+1, comm, &reqs[0]);
   MPI_Waitall(recvn+1, reqs, MPI_STATUSES_IGNORE);

   // move received rows over the sent rows
   const int m = std::min(nsend, nrecv);
   if (m > 0) {
      std::memcpy(send, b + ((std::size_t)nkeep + nsend + nrecv - m) * stride,
                  (std::size_t)m * stride);
   }

   buf.SetSize((int)(((std::size_t)nkeep + nrecv) * stride / sizeof(std::uint64_t)));
   return nkeep + nrecv;
}

} // namespace mfem

#endif // MFEM_USE_MPI
