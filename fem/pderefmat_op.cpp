// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "pderefmat_op.hpp"

#ifdef MFEM_USE_MPI

#include "fes_kernels.hpp"
/// \cond DO_NOT_DOCUMENT
namespace mfem
{
namespace internal
{
template <Ordering::Type Order, bool Atomic>
static void ParDerefMultKernelImpl(const ParDerefineMatrixOp &op,
                                   const Vector &x, Vector &y)
{
   // pack sends
   if (op.xghost_send.Size())
   {
      auto src = x.Read();
      auto idcs = op.send_permutations.Read();
      auto dst = Device::GetGPUAwareMPI() ? op.xghost_send.Write()
                 : op.xghost_send.HostWrite();
      auto vdims = op.fespace->GetVDim();
      auto sptr = op.send_segment_idcs.Read();
      auto lptr = op.send_segments.Read();
      auto old_ndofs = x.Size() / vdims;

      forall(op.send_permutations.Size(), [=] MFEM_HOST_DEVICE(int i)
      {
         int seg = sptr[i];
         int width = lptr[seg + 1] - lptr[seg];
         auto tdst = dst + i + lptr[seg] * vdims;
         int sign = 1;
         int col = idcs[i];
         if (col < 0)
         {
            sign = -1;
            col = -1 - col;
         }
         for (int vdim = 0; vdim < vdims; ++vdim)
         {
            tdst[vdim * width] =
               sign
               * src[Order == Ordering::byNODES ? (col + vdim * old_ndofs)
                           : (col * vdims + vdim)];
         }
      });
      // TODO: is this needed so we can send the packed data correctly?
      // unclear for GPU-aware MPI, definitely required otherwise
      MFEM_DEVICE_SYNC;
   }
   // initialize off-diagonal receive and send
   op.requests.clear();
   if (op.xghost_recv.Size())
   {
      auto vdims = op.fespace->GetVDim();
      auto rcv = Device::GetGPUAwareMPI() ? op.xghost_recv.Write()
                 : op.xghost_recv.HostWrite();
      for (int i = 0; i < op.recv_ranks.Size(); ++i)
      {
         op.requests.emplace_back();
         MPI_Irecv(rcv + op.recv_segments[i] * vdims,
                   (op.recv_segments[i + 1] - op.recv_segments[i]) * vdims,
                   MPITypeMap<real_t>::mpi_type, op.recv_ranks[i],
                   MessageTag::DEREFINEMENT_MATRIX_CONSTRUCTION_DATA,
                   op.fespace->GetComm(), &op.requests.back());
      }
   }
   if (op.xghost_send.Size())
   {
      auto vdims = op.fespace->GetVDim();
      // only is a GPU mem ptr if GPU-aware MPI is enabled
      auto dst = Device::GetGPUAwareMPI() ? op.xghost_send.Write()
                 : op.xghost_send.HostWrite();
      for (int i = 0; i < op.send_ranks.Size(); ++i)
      {
         op.requests.emplace_back();
         MPI_Isend(dst + op.send_segments[i] * vdims,
                   (op.send_segments[i + 1] - op.send_segments[i]) * vdims,
                   MPITypeMap<real_t>::mpi_type, op.send_ranks[i],
                   MessageTag::DEREFINEMENT_MATRIX_CONSTRUCTION_DATA,
                   op.fespace->GetComm(), &op.requests.back());
      }
   }
   {
      // diagonal
      DerefineMatrixOpMultFunctor<Order, Atomic, true> func;
      func.xptr = x.Read();
      y.UseDevice();
      y = 0.;
      func.yptr = y.ReadWrite();
      func.bsptr = op.block_storage.Read();
      func.boptr = op.block_offsets.Read();
      func.brptr = op.block_row_idcs_offsets.Read();
      func.bcptr = op.block_col_idcs_offsets.Read();
      func.rptr = op.row_idcs.Read();
      func.cptr = op.col_idcs.Read();
      func.vdims = op.fespace->GetVDim();
      func.nblocks = op.block_offsets.Size();
      func.width = op.Width() / func.vdims;
      func.height = op.Height() / func.vdims;
      func.Run(op.max_rows);
   }
   // wait for comm to finish, if any
   if (op.requests.size())
   {
      MPI_Waitall(op.requests.size(), op.requests.data(), MPI_STATUSES_IGNORE);
      if (op.xghost_recv.Size())
      {
         // off-diagonal kernel
         DerefineMatrixOpMultFunctor<Order, Atomic, false> func;
         // directly read from host-pinned memory if not using GPU-aware MPI
         func.xptr = Device::GetGPUAwareMPI() ? op.xghost_recv.Read()
                     : op.xghost_recv.HostRead();
         func.yptr = y.ReadWrite();
         func.bsptr = op.block_storage.Read();
         func.boptr = op.off_diag_block_offsets.Read();
         func.brptr = op.block_off_diag_row_idcs_offsets.Read();
         func.rsptr = op.recv_segment_idcs.Read();
         func.segptr = op.recv_segments.Read();
         func.coptr = op.block_off_diag_col_offsets.Read();
         func.bwptr = op.block_off_diag_widths.Read();
         func.rptr = op.row_off_diag_idcs.Read();
         func.vdims = op.fespace->GetVDim();
         func.nblocks = op.off_diag_block_offsets.Size();
         func.width = op.xghost_recv.Size() / func.vdims;
         func.height = op.Height() / func.vdims;
         func.Run(op.max_rows);
      }
   }
}
} // namespace internal

template <Ordering::Type Order, bool Atomic>
ParDerefineMatrixOp::MultKernelType ParDerefineMatrixOp::MultKernel::Kernel()
{
   return internal::ParDerefMultKernelImpl<Order, Atomic>;
}

ParDerefineMatrixOp::MultKernelType
ParDerefineMatrixOp::MultKernel::Fallback(Ordering::Type, bool)
{
   MFEM_ABORT("invalid MultKernel parameters");
}

ParDerefineMatrixOp::Kernels::Kernels()
{
   MultKernel::Specialization<Ordering::byNODES, false>::Add();
   MultKernel::Specialization<Ordering::byVDIM, false>::Add();
   MultKernel::Specialization<Ordering::byNODES, true>::Add();
   MultKernel::Specialization<Ordering::byVDIM, true>::Add();
}

void ParDerefineMatrixOp::Mult(const Vector &x, Vector &y) const
{
   const bool is_dg = fespace->FEColl()->GetContType()
                      == FiniteElementCollection::DISCONTINUOUS;
   // DG needs atomic summation
   MultKernel::Run(fespace->GetOrdering(), is_dg, *this, x, y);
   // use this to prevent xghost* from being re-purposed for subsequent Mult
   // calls
   MFEM_DEVICE_SYNC;
}

ParDerefineMatrixOp::ParDerefineMatrixOp(ParFiniteElementSpace &fespace_,
                                         int old_ndofs,
                                         const Table *old_elem_dof,
                                         const Table *old_elem_fos)
   : Operator(fespace_.GetVSize(), old_ndofs * fespace_.GetVDim()),
     fespace(&fespace_)
{
   static Kernels kernels;
   constexpr int max_team_size = 256;

   const int NRanks = fespace->GetNRanks();

   const int nrk = HYPRE_AssumedPartitionCheck() ? 2 : NRanks;

   MFEM_VERIFY(fespace->Nonconforming(),
               "Not implemented for conforming meshes.");
   MFEM_VERIFY(fespace->old_dof_offsets[nrk],
               "Missing previous (finer) space.");

   const int MyRank = fespace->GetMyRank();
   ParNCMesh *old_pncmesh = fespace->GetParMesh()->pncmesh;
   const CoarseFineTransformations &dtrans =
      old_pncmesh->GetDerefinementTransforms();
   const Array<int> &old_ranks = old_pncmesh->GetDerefineOldRanks();

   const bool is_dg = fespace->FEColl()->GetContType()
                      == FiniteElementCollection::DISCONTINUOUS;
   DenseMatrix localRVO; // for variable-order only

   DenseTensor localR[Geometry::NumGeom];
   int diag_rows = 0;
   int off_diag_rows = 0;
   int diag_cols = 0;

   auto get_ldofs = [&](int k) -> int
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (fespace->IsVariableOrder())
      {
         const FiniteElement *fe = fespace->GetFE(emb.parent);
         return fe->GetDof();
      }
      else
      {
         Geometry::Type geom =
         fespace->GetParMesh()->GetElementBaseGeometry(emb.parent);
         return fespace->FEColl()->FiniteElementForGeometry(geom)->GetDof();
      }
   };
   Array<int> dofs, old_dofs;
   max_rows = 1;
   // first pass:
   // - determine memory block lengths
   // - identify dofs in x we need to send/receive
   // don't need to send the indices, fine rank will re-arrange and sign
   // change x before transmitting the ghost data

   // key: coarse rank to send to
   // value: old dofs to send (with sign)
   std::map<int, std::vector<int>> to_send;
   // key: fine rank
   // value: indices into dtrans.embeddings
   std::map<int, std::vector<int>> od_ks;
   // key: fine rank
   // value: recv segment length
   std::map<int, int> od_seg_lens;
   int send_len = 0;
   int recv_len = 0;
   // size of block_storage, if fespace->IsVariableOrder()
   // otherwise unused
   int total_size = 0;
   int num_diagonal_blocks = 0;
   int num_offdiagonal_blocks = 0;
   for (int k = 0; k < dtrans.embeddings.Size(); ++k)
   {
      const Embedding &emb = dtrans.embeddings[k];
      int fine_rank = old_ranks[k];
      int coarse_rank = (emb.parent < 0) ? (-1 - emb.parent)
                        : old_pncmesh->ElementRank(emb.parent);
      if (coarse_rank != MyRank && fine_rank == MyRank)
      {
         // this rank needs to send data in x to course_rank
         old_elem_dof->GetRow(k, old_dofs);
         auto &tmp = to_send[coarse_rank];
         send_len += old_dofs.Size();
         for (int i = 0; i < old_dofs.Size(); ++i)
         {
            tmp.emplace_back(old_dofs[i]);
         }
      }
      else if (coarse_rank == MyRank && fine_rank != MyRank)
      {
         // this rank needs to receive data in x from fine_rank
         MFEM_ASSERT(emb.parent >= 0, "");
         auto ldofs = get_ldofs(k);
         off_diag_rows += ldofs;
         recv_len += ldofs;
         od_ks[fine_rank].emplace_back(k);
         od_seg_lens[fine_rank] += ldofs;
         ++num_offdiagonal_blocks;
         if (fespace->IsVariableOrder())
         {
            total_size += ldofs * ldofs;
         }
      }
      else if (coarse_rank == MyRank && fine_rank == MyRank)
      {
         MFEM_ASSERT(emb.parent >= 0, "");
         // diagonal
         ++num_diagonal_blocks;
         auto ldofs = get_ldofs(k);
         diag_rows += ldofs;
         diag_cols += ldofs;
         if (fespace->IsVariableOrder())
         {
            total_size += ldofs * ldofs;
         }
      }
   }
   send_segments.SetSize(to_send.size() + 1);
   send_segments.HostWrite();
   send_ranks.SetSize(to_send.size());
   send_ranks.HostWrite();
   {
      int idx = 0;
      send_segments[0] = 0;
      for (auto &tmp : to_send)
      {
         send_ranks[idx] = tmp.first;
         send_segments[idx + 1] = send_segments[idx] + tmp.second.size();
         ++idx;
      }
   }
   recv_segment_idcs.SetSize(off_diag_rows);
   recv_segment_idcs.HostWrite();
   recv_segments.SetSize(od_ks.size() + 1);
   recv_segments.HostWrite();
   recv_ranks.SetSize(od_ks.size());
   recv_ranks.HostWrite();

   // set sizes
   row_idcs.SetSize(diag_rows);
   row_idcs.HostWrite();
   row_off_diag_idcs.SetSize(off_diag_rows);
   row_off_diag_idcs.HostWrite();
   col_idcs.SetSize(diag_cols);
   col_idcs.HostWrite();
   block_row_idcs_offsets.SetSize(num_diagonal_blocks + 1);
   block_row_idcs_offsets.HostWrite();
   block_col_idcs_offsets.SetSize(num_diagonal_blocks + 1);
   block_col_idcs_offsets.HostWrite();
   block_off_diag_row_idcs_offsets.SetSize(num_offdiagonal_blocks + 1);
   block_off_diag_row_idcs_offsets.HostWrite();
   block_off_diag_col_offsets.SetSize(num_offdiagonal_blocks);
   block_off_diag_col_offsets.HostWrite();
   block_off_diag_widths.SetSize(num_offdiagonal_blocks);
   block_off_diag_widths.HostWrite();
   pack_col_idcs.SetSize(send_len);
   // memory manager doesn't appear to have a graceful fallback for
   // HOST_PINNED if not built with CUDA or HIP
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
   xghost_send.SetSize(send_len * fespace->GetVDim(),
                       Device::GetGPUAwareMPI() ? MemoryType::DEFAULT
                       : MemoryType::HOST_PINNED);
   xghost_recv.SetSize(recv_len * fespace->GetVDim(),
                       Device::GetGPUAwareMPI() ? MemoryType::DEFAULT
                       : MemoryType::HOST_PINNED);
#else
   xghost_send.SetSize(send_len * fespace->GetVDim());
   xghost_recv.SetSize(recv_len * fespace->GetVDim());
#endif
   send_permutations.SetSize(send_len);
   send_segment_idcs.SetSize(send_len);
   block_offsets.SetSize(num_diagonal_blocks);
   block_offsets.HostWrite();
   off_diag_block_offsets.SetSize(num_offdiagonal_blocks);
   off_diag_block_offsets.HostWrite();
   int geom_offsets[Geometry::NumGeom];
   real_t *bs_ptr;

   if (fespace->IsVariableOrder())
   {
      block_storage.SetSize(total_size);
      bs_ptr = block_storage.HostWrite();
      // compute block data later
   }
   else
   {
      // compression scheme:
      // block_offsets is the start of each block, potentially repeated
      // only need to store localR for used shapes
      Mesh::GeometryList elem_geoms(*fespace->GetMesh());

      int size = 0;
      for (int i = 0; i < elem_geoms.Size(); ++i)
      {
         fespace->GetLocalDerefinementMatrices(elem_geoms[i],
                                               localR[elem_geoms[i]]);
         geom_offsets[elem_geoms[i]] = size;
         size += localR[elem_geoms[i]].TotalSize();
      }
      block_storage.SetSize(size);
      bs_ptr = block_storage.HostWrite();
      // copy blocks into block_storage
      for (int i = 0; i < elem_geoms.Size(); ++i)
      {
         std::copy(localR[elem_geoms[i]].Data(),
                   localR[elem_geoms[i]].Data()
                   + localR[elem_geoms[i]].TotalSize(),
                   bs_ptr);
         bs_ptr += localR[elem_geoms[i]].TotalSize();
      }
   }

   // second pass:
   // - initialize buffers

   {
      auto ptr = send_permutations.HostWrite();
      auto ptr2 = send_segment_idcs.HostWrite();
      int i = 0;
      for (auto &v : to_send)
      {
         ptr = std::copy(v.second.begin(), v.second.end(), ptr);
         for (size_t idx = 0; idx < v.second.size(); ++idx)
         {
            *ptr2 = i;
            ++ptr2;
         }
         ++i;
      }
   }

   block_row_idcs_offsets[0] = 0;
   block_col_idcs_offsets[0] = 0;
   block_off_diag_row_idcs_offsets[0] = 0;
   Array<int> mark(fespace->GetNDofs());
   mark = 0;
   {
      int idx = 0;
      recv_segments[0] = 0;
      for (auto &v : od_seg_lens)
      {
         recv_ranks[idx] = v.first;
         recv_segments[idx + 1] = recv_segments[idx] + v.second;
         ++idx;
      }
   }
   // key: index into dtrans.embeddings
   // value: off-diagonal block offset, od_ridx, seg id
   std::unordered_map<int, std::array<int, 3>> ks_map;
   {
      int od_ridx = 0;
      int seg_id = 0;
      for (auto &v1 : od_ks)
      {
         for (auto k : v1.second)
         {
            auto &tmp = ks_map[k];
            tmp[0] = ks_map.size() - 1;
            tmp[1] = od_ridx;
            tmp[2] = seg_id;
            od_ridx += get_ldofs(k);
         }
         ++seg_id;
      }
   }
   int diag_idx = 0;
   int var_offset = 0;
   int ridx = 0;
   int cidx = 0;
   // can't break this up into separate diagonals/off-diagonals loops because
   // of mark
   for (int k = 0; k < dtrans.embeddings.Size(); ++k)
   {
      const Embedding &emb = dtrans.embeddings[k];
      if (emb.parent < 0)
      {
         continue;
      }
      int fine_rank = old_ranks[k];
      int coarse_rank = (emb.parent < 0) ? (-1 - emb.parent)
                        : old_pncmesh->ElementRank(emb.parent);
      if (coarse_rank == MyRank)
      {
         // either diagonal or off-diagonal
         Geometry::Type geom =
            fespace->GetMesh()->GetElementBaseGeometry(emb.parent);
         if (fespace->IsVariableOrder())
         {
            const FiniteElement *fe = fespace->GetFE(emb.parent);
            const DenseTensor &pmats = dtrans.point_matrices[geom];
            const int ldof = fe->GetDof();

            IsoparametricTransformation isotr;
            isotr.SetIdentityTransformation(geom);

            localRVO.SetSize(ldof, ldof);
            isotr.SetPointMat(pmats(emb.matrix));
            // Local restriction is size ldofxldof assuming that the parent
            // and child are of same polynomial order.
            fe->GetLocalRestriction(isotr, localRVO);
            // copy block
            auto s = localRVO.Height() * localRVO.Width();
            std::copy(localRVO.Data(), localRVO.Data() + s, bs_ptr);
            bs_ptr += s;
         }
         DenseMatrix &lR =
            fespace->IsVariableOrder() ? localRVO : localR[geom](emb.matrix);
         max_rows = std::max(lR.Height(), max_rows);
         auto size = lR.Height() * lR.Width();
         fespace->elem_dof->GetRow(emb.parent, dofs);
         if (fine_rank == MyRank)
         {
            // diagonal
            old_elem_dof->GetRow(k, old_dofs);
            MFEM_VERIFY(old_dofs.Size() == dofs.Size(),
                        "Parent and child must have same #dofs.");
            block_row_idcs_offsets[diag_idx + 1] =
               block_row_idcs_offsets[diag_idx] + lR.Height();
            block_col_idcs_offsets[diag_idx + 1] =
               block_col_idcs_offsets[diag_idx] + lR.Width();

            if (fespace->IsVariableOrder())
            {
               block_offsets[diag_idx] = var_offset;
               var_offset += size;
            }
            else
            {
               block_offsets[diag_idx] = geom_offsets[geom] + size * emb.matrix;
            }
            for (int i = 0; i < lR.Height(); ++i, ++ridx)
            {
               if (!std::isfinite(lR(i, 0)))
               {
                  row_idcs[ridx] = INT_MAX;
                  continue;
               }
               int r = dofs[i];
               int m = (r >= 0) ? r : (-1 - r);
               if (is_dg || !mark[m])
               {
                  row_idcs[ridx] = r;
                  mark[m] = 1;
               }
               else
               {
                  row_idcs[ridx] = INT_MAX;
               }
            }
            for (int i = 0; i < lR.Width(); ++i, ++cidx)
            {
               col_idcs[cidx] = old_dofs[i];
            }
            ++diag_idx;
         }
         else
         {
            // off-diagonal
            auto &tmp = ks_map.at(k);
            auto od_idx = tmp[0];
            auto od_ridx = tmp[1];
            block_off_diag_row_idcs_offsets[od_idx + 1] =
               block_off_diag_row_idcs_offsets[od_idx] + lR.Height();
            block_off_diag_col_offsets[od_idx] = od_ridx;
            block_off_diag_widths[od_idx] = lR.Width();
            recv_segment_idcs[od_idx] = tmp[2];

            if (fespace->IsVariableOrder())
            {
               off_diag_block_offsets[od_idx] = var_offset;
               var_offset += size;
            }
            else
            {
               off_diag_block_offsets[od_idx] =
                  geom_offsets[geom] + size * emb.matrix;
            }
            for (int i = 0; i < lR.Height(); ++i, ++od_ridx)
            {
               if (!std::isfinite(lR(i, 0)))
               {
                  row_off_diag_idcs[od_ridx] = INT_MAX;
                  continue;
               }
               int r = dofs[i];
               int m = (r >= 0) ? r : (-1 - r);
               if (is_dg || !mark[m])
               {
                  row_off_diag_idcs[od_ridx] = r;
                  mark[m] = 1;
               }
               else
               {
                  row_off_diag_idcs[od_ridx] = INT_MAX;
               }
            }
            ++od_idx;
         }
      }
   }
   // if not using GPU, set max_rows/max_cols to zero
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      max_rows = std::min(max_rows, max_team_size);
   }
   else
   {
      max_rows = 1;
   }
   requests.reserve(recv_ranks.Size() + send_ranks.Size());
}
} // namespace mfem
/// \endcond DO_NOT_DOCUMENT

#endif
