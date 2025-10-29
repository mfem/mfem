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

#include "hybridization_ext.hpp"
#include "hybridization.hpp"
#include "pfespace.hpp"
#include "../general/forall.hpp"
#include "../linalg/batched/batched.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

HybridizationExtension::HybridizationExtension(Hybridization &hybridization_)
   : h(hybridization_)
{ }

static int GetNFacesPerElement(const Mesh &mesh)
{
   const int dim = mesh.Dimension();
   switch (dim)
   {
      case 2: return mesh.GetElement(0)->GetNEdges();
      case 3: return mesh.GetElement(0)->GetNFaces();
      default: MFEM_ABORT("Invalid dimension.");
   }
}

static bool IsParFESpace(const FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   return dynamic_cast<const ParFiniteElementSpace *>(&fes) != nullptr;
#else
   return false;
#endif
}

const Operator &HybridizationExtension::GetProlongation() const
{
   return P_pc ? *P_pc : *h.c_fes.GetProlongationMatrix();
}

void HybridizationExtension::ConstructC()
{
   Mesh &mesh = *h.fes.GetMesh();
   const int nf = mesh.GetNFbyType(FaceType::Interior);
   const int m = h.fes.GetFE(0)->GetDof(); // num hat dofs per el
   const int n = h.c_fes.GetFaceElement(0)->GetDof(); // num c dofs per face

   // Assemble Ct_mat using EA
   Vector emat(m * n * 2 * nf);
   h.c_bfi->AssembleEAInteriorFaces(h.c_fes, h.fes, emat, false);

   const auto *tbe = dynamic_cast<const TensorBasisElement*>(h.fes.GetFE(0));
   MFEM_VERIFY(tbe, "");
   // Note: copying the DOF map here (instead of using a reference) because
   // reading it on GPU can cause issues in other parts of the code when using
   // the debug device. The DOF map is accessed in other places without
   // explicitly calling HostRead, which fails on non-const access if the device
   // pointer is valid.
   Array<int> dof_map = tbe->GetDofMap();

   Ct_mat.SetSize(m * n * n_el_face);
   const auto d_emat = Reshape(emat.Read(), m, n, 2, nf);
   const int *d_dof_map = dof_map.Read();
   const auto d_face_to_el = Reshape(face_to_el.Read(), 2, 2, nf);
   auto d_Ct_mat = Reshape(Ct_mat.Write(), m, n, n_el_face);

   Ct_mat = 0.0; // On device, since previous call to Write()

   mfem::forall(m*n*2*nf, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i_lex = idx % m;
      const int j = (idx / m) % n;
      const int ie = (idx / m / n) % 2;
      const int f = idx / m / n / 2;

      const int fi = d_face_to_el(1, ie, f);

      // Skip elements belonging to face neighbors of shared faces
      if (fi >= 0)
      {
         // Convert to back to native MFEM ordering in the volume
         const int i_s = d_dof_map[i_lex];
         const int i = (i_s >= 0) ? i_s : -1 - i_s;
         d_Ct_mat(i, j, fi) = d_emat(i_lex, j, ie, f);
      }
   });

#ifdef MFEM_USE_MPI
   if (auto pc_fes = dynamic_cast<ParFiniteElementSpace*>(&h.c_fes))
   {
      if (pc_fes->Nonconforming())
      {
         P_pc.reset(pc_fes->GetPartialConformingInterpolation());
      }
   }
#endif
}

namespace internal
{
template <typename T, int SIZE>
struct LocalMemory
{
   T data[SIZE];
   MFEM_HOST_DEVICE inline operator T *() const { return (T*)data; }
};

template <typename T>
struct LocalMemory<T,0>
{
   MFEM_HOST_DEVICE inline operator T *() const { return (T*)nullptr; }
};
};

template <int MID, int MBD>
void HybridizationExtension::FactorElementMatrices(Vector &AhatInvCt_mat)
{
   const Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int m = h.fes.GetFE(0)->GetDof();
   const int n = h.c_fes.GetFaceElement(0)->GetDof();

   AhatInvCt_mat.SetSize(Ct_mat.Size());
   auto d_AhatInvCt = Reshape(AhatInvCt_mat.Write(), m, n, n_el_face);

   const int nidofs = idofs.Size();
   const int nbdofs = bdofs.Size();

   MFEM_VERIFY(nidofs <= MID, "");
   MFEM_VERIFY(nbdofs <= MBD, "");

   Ahat_ii.SetSize(nidofs*nidofs*ne);
   Ahat_ib.SetSize(nidofs*nbdofs*ne);
   Ahat_bi.SetSize(nbdofs*nidofs*ne);
   Ahat_bb.SetSize(nbdofs*nbdofs*ne);

   Ahat_ii_piv.SetSize(nidofs*ne);
   Ahat_bb_piv.SetSize(nbdofs*ne);

   const auto *d_idofs = idofs.Read();
   const auto *d_bdofs = bdofs.Read();

   const auto d_hat_dof_marker = Reshape(hat_dof_marker.Read(), m, ne);
   auto d_Ahat = Reshape(Ahat.Read(), m, m, ne);

   auto d_A_ii = Reshape(Ahat_ii.Write(), nidofs, nidofs, ne);
   auto d_A_ib_all = Reshape(Ahat_ib.Write(), nidofs*nbdofs, ne);
   auto d_A_bi_all = Reshape(Ahat_bi.Write(), nbdofs*nidofs, ne);
   auto d_A_bb_all = Reshape(Ahat_bb.Write(), nbdofs*nbdofs, ne);

   auto d_ipiv_ii = Reshape(Ahat_ii_piv.Write(), nidofs, ne);
   auto d_ipiv_bb = Reshape(Ahat_bb_piv.Write(), nbdofs, ne);

   const auto d_Ct_mat = Reshape(Ct_mat.Read(), m, n, n_el_face);
   const auto d_el_face_offsets = el_face_offsets.Read();

   static constexpr bool GLOBAL = (MID == 0 && MBD == 0);

   using internal::LocalMemory;

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MAX_DOFS = 3*MD1D*(MD1D-1)*(MD1D-1);
      constexpr int MAX_IDOFS = (MID == 0 && MBD == 0) ? MAX_DOFS : MID;
      constexpr int MAX_BDOFS = (MID == 0 && MBD == 0) ? MAX_DOFS : MBD;

      LocalMemory<int,MAX_IDOFS> idofs_loc;
      LocalMemory<int,MAX_BDOFS> bdofs_loc;
      for (int i = 0; i < nidofs; i++) { idofs_loc[i] = d_idofs[i]; }
      for (int i = 0; i < nbdofs; i++) { bdofs_loc[i] = d_bdofs[i]; }

      LocalMemory<int,MAX_BDOFS> essdofs_loc;
      int nbfdofs = 0;
      int nessdofs = 0;
      for (int i = 0; i < nbdofs; i++)
      {
         const int dof_idx = bdofs_loc[i];
         if (d_hat_dof_marker(dof_idx, e) == ESSENTIAL)
         {
            essdofs_loc[nessdofs] = dof_idx;
            nessdofs += 1;
         }
         else
         {
            bdofs_loc[nbfdofs] = dof_idx;
            nbfdofs += 1;
         }
      }

      LocalMemory<real_t, MID*MID> A_ii_loc;
      LocalMemory<real_t, MBD*MID> A_bi_loc;
      LocalMemory<real_t, MID*MBD> A_ib_loc;
      LocalMemory<real_t, MBD*MBD> A_bb_loc;

      DeviceMatrix A_ii(GLOBAL ? &d_A_ii(0,0,e) : A_ii_loc, nidofs, nidofs);
      DeviceMatrix A_ib(GLOBAL ? &d_A_ib_all(0,e) : A_ib_loc, nidofs, nbfdofs);
      DeviceMatrix A_bi(GLOBAL ? &d_A_bi_all(0,e) : A_bi_loc, nbfdofs, nidofs);
      DeviceMatrix A_bb(GLOBAL ? &d_A_bb_all(0,e) : A_bb_loc, nbfdofs, nbfdofs);

      for (int j = 0; j < nidofs; j++)
      {
         const int jj = idofs_loc[j];
         for (int i = 0; i < nidofs; i++)
         {
            A_ii(i,j) = d_Ahat(idofs_loc[i], jj, e);
         }
         for (int i = 0; i < nbfdofs; i++)
         {
            A_bi(i,j) = d_Ahat(bdofs_loc[i], jj, e);
         }
      }
      for (int j = 0; j < nbfdofs; j++)
      {
         const int jj = bdofs_loc[j];
         for (int i = 0; i < nidofs; i++)
         {
            A_ib(i,j) = d_Ahat(idofs_loc[i], jj, e);
         }
         for (int i = 0; i < nbfdofs; i++)
         {
            A_bb(i,j) = d_Ahat(bdofs_loc[i], jj, e);
         }
      }

      LocalMemory<int,MID> ipiv_ii_loc;
      LocalMemory<int,MBD> ipiv_bb_loc;

      auto ipiv_ii = GLOBAL ? &d_ipiv_ii(0,e) : ipiv_ii_loc;
      auto ipiv_bb = GLOBAL ? &d_ipiv_ii(0,e) : ipiv_bb_loc;

      kernels::LUFactor(A_ii, nidofs, ipiv_ii);
      kernels::BlockFactor(A_ii, nidofs, ipiv_ii, nbfdofs, A_ib, A_bi, A_bb);
      kernels::LUFactor(A_bb, nbfdofs, ipiv_bb);

      const int begin = d_el_face_offsets[e];
      const int end = d_el_face_offsets[e + 1];
      for (int f = begin; f < end; ++f)
      {
         for (int j = 0; j < n; ++j)
         {
            LocalMemory<real_t,MAX_BDOFS> Sb_inv_Cb_t;
            for (int i = 0; i < nbfdofs; ++i)
            {
               Sb_inv_Cb_t[i] = d_Ct_mat(bdofs_loc[i], j, f);
            }
            kernels::LUSolve(A_bb, nbfdofs, ipiv_bb, Sb_inv_Cb_t);
            for (int i = 0; i < nbfdofs; ++i)
            {
               const int b_i = bdofs_loc[i];
               d_AhatInvCt(b_i, j, f) = Sb_inv_Cb_t[i];
            }
            for (int i = 0; i < nidofs; ++i)
            {
               d_AhatInvCt(idofs_loc[i], j, f) = 0.0;
            }
            for (int i = 0; i < nessdofs; ++i)
            {
               d_AhatInvCt(essdofs_loc[i], j, f) = 0.0;
            }
         }
      }

      // Write out to global memory
      if (!GLOBAL)
      {
         // Note: in the following constructors, avoid using index 0 in
         //       d_A_{bi,ib,bb}_all when their size is 0.
         DeviceMatrix d_A_bi((nbfdofs && nidofs) ?
                             &d_A_bi_all(0,e) : nullptr,
                             nbfdofs, nidofs);
         DeviceMatrix d_A_ib((nbfdofs && nidofs) ?
                             &d_A_ib_all(0,e) : nullptr,
                             nidofs, nbfdofs);
         DeviceMatrix d_A_bb((nbfdofs) ? &d_A_bb_all(0,e) : nullptr,
                             nbfdofs, nbfdofs);

         for (int j = 0; j < nidofs; j++)
         {
            d_ipiv_ii(j,e) = ipiv_ii[j];
            for (int i = 0; i < nidofs; i++)
            {
               d_A_ii(i,j,e) = A_ii(i,j);
            }
            for (int i = 0; i < nbfdofs; i++)
            {
               d_A_bi(i,j) = A_bi(i,j);
            }
         }
         for (int j = 0; j < nbfdofs; j++)
         {
            d_ipiv_bb(j,e) = ipiv_bb[j];
            for (int i = 0; i < nidofs; i++)
            {
               d_A_ib(i,j) = A_ib(i,j);
            }
            for (int i = 0; i < nbfdofs; i++)
            {
               d_A_bb(i,j) = A_bb(i,j);
            }
         }
      }
   });
}

void HybridizationExtension::ConstructH()
{
   const Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int m = h.fes.GetFE(0)->GetDof();
   const int n = h.c_fes.GetFaceElement(0)->GetDof();

   Vector AhatInvCt_mat;

   {
      // The dispatch below is based on the following sizes, sorted
      // appropriately.
      //
      // RT(k) in 2D (quads): (interior,boundary) dofs:
      // - arbitrary k: 2*(k+1)*(k+2)-4*(k+1), 4*(k+1)
      // - k=0: (0,4)
      // - k=1: (4,8)
      // - k=2: (12,12)
      // - k=3: (24,16)
      // RT(k) in 3D (hexes): (interior,boundary) dofs:
      // - arbitrary k: 3*(k+1)^2*(k+2)-6*(k+1)^2, 6*(k+1)^2
      // - k=0: (0,6)
      // - k=1: (12,24)
      // - k=2: (54,54)
      const int NI = idofs.Size();
      const int NB = bdofs.Size();
      if (NI == 0 && NB <= 4) { FactorElementMatrices<0,4>(AhatInvCt_mat); }
      else if (NI == 0 && NB <= 6) { FactorElementMatrices<0,6>(AhatInvCt_mat); }
      else if (NI <= 4 && NB <= 8) { FactorElementMatrices<4,8>(AhatInvCt_mat); }
      else if (NI <= 12 && NB <= 12) { FactorElementMatrices<12,12>(AhatInvCt_mat); }
      else if (NI <= 12 && NB <= 24) { FactorElementMatrices<12,24>(AhatInvCt_mat); }
      else if (NI <= 24 && NB <= 16) { FactorElementMatrices<24,16>(AhatInvCt_mat); }
      else if (NI <= 54 && NB <= 54) { FactorElementMatrices<54,54>(AhatInvCt_mat); }
      // Fallback
      else { FactorElementMatrices<0,0>(AhatInvCt_mat); }
   }

   const auto d_AhatInvCt =
      Reshape(AhatInvCt_mat.Read(), m, n, n_el_face);

   const int nf = h.fes.GetNFbyType(FaceType::Interior);
   Array<int> face_to_face(n_face_face);

   Vector CAhatInvCt(n_face_face*n*n);

   const auto d_Ct = Reshape(Ct_mat.Read(), m, n, n_el_face);
   const auto d_face_to_el = Reshape(face_to_el.Read(), 2, 2, nf);
   const auto d_el_to_face = el_to_face.Read();
   const auto d_el_face_offsets = el_face_offsets.Read();
   auto d_CAhatInvCt = Reshape(CAhatInvCt.Write(), n, n, n_face_face);
   auto d_face_to_face = Reshape(face_to_face.Write(), n_face_face);
   auto d_face_face_offsets = face_face_offsets.Read();

   CAhatInvCt = 0.0;

   mfem::forall(nf, [=] MFEM_HOST_DEVICE (int fi)
   {
      const int begin_f = d_face_face_offsets[fi];

      int idx = 0;
      for (int ei = 0; ei < 2; ++ei)
      {
         const int e = d_face_to_el(0, ei, fi);
         if (e < 0 || e >= ne) { continue; }
         const int begin_i = d_el_face_offsets[e];
         const int end_i = d_el_face_offsets[e + 1];
         for (int fj_i = begin_i; fj_i < end_i; ++fj_i)
         {
            const int fj = d_el_to_face[fj_i];
            // Allow fi == fj (self-connections)

            // Have we seen this face before? It is possible in some
            // configurations to encounter the same neighboring face twice
            int idx_j = idx;
            for (int i = 0; i < idx; ++i)
            {
               if (d_face_to_face[begin_f + i] == fj)
               {
                  idx_j = i;
                  break;
               }
            }
            // This is a new face, record it and increment the counter
            if (idx_j == idx)
            {
               d_face_to_face[begin_f + idx] = fj;
               idx++;
            }
         }
      }
   });

   mfem::forall(nf, [=] MFEM_HOST_DEVICE (int fi)
   {
      const int begin = d_face_face_offsets[fi];
      const int end = d_face_face_offsets[fi + 1];
      for (int idx_j = begin; idx_j < end; ++idx_j)
      {
         const int fj = d_face_to_face[idx_j];
         for (int ei = 0; ei < 2; ++ei)
         {
            const int e = d_face_to_el(0, ei, fi);
            if (e < 0 || e >= ne) { continue; }
            const int fi_i = d_face_to_el(1, ei, fi);

            int fj_i = -1;
            for (int ej = 0; ej < 2; ++ej)
            {
               if (d_face_to_el(0, ej, fj) == e)
               {
                  fj_i = d_face_to_el(1, ej, fj);
                  break;
               }
            }
            if (fj_i >= 0)
            {
               const real_t *Ct_i = &d_Ct(0, 0, fi_i);
               const real_t *AhatInvCt_i = &d_AhatInvCt(0, 0, fj_i);
               real_t *CAhatInvCt_i = &d_CAhatInvCt(0, 0, idx_j);
               kernels::AddMultAtB(m, n, n, Ct_i, AhatInvCt_i, CAhatInvCt_i);
            }
         }
      }
   });


#ifdef MFEM_USE_MPI
   auto *c_pfes = dynamic_cast<ParFiniteElementSpace*>(&h.c_fes);
#endif

   const int ncdofs_face_nbr = [&]()
   {
#ifdef MFEM_USE_MPI
      // Only need to handle face neighbor DOFs when there are nonconforming
      // (ghost) faces.
      if (c_pfes && c_pfes->Nonconforming())
      {
         c_pfes->ExchangeFaceNbrData();
         return c_pfes->GetFaceNbrVSize();
      }
#endif
      return 0;
   }();

   const int ncdofs_local = h.c_fes.GetVSize();
   const int ncdofs = ncdofs_local + ncdofs_face_nbr;
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr =
      h.c_fes.GetFaceRestriction(ordering, FaceType::Interior);
   const auto *l2_face_restr =
      dynamic_cast<const L2InterfaceFaceRestriction*>(face_restr);
   MFEM_ASSERT(l2_face_restr, "");
   const auto c_scatter_map = Reshape(l2_face_restr->ScatterMap().Read(), n, nf);

   h.H.reset(new SparseMatrix);
   h.H->OverrideSize(ncdofs, ncdofs);

   h.H->GetMemoryI().New(ncdofs + 1, h.H->GetMemoryI().GetMemoryType());

   {
      int *I = h.H->WriteI();

      mfem::forall(ncdofs, [=] MFEM_HOST_DEVICE (int i) { I[i] = 0; });

      mfem::forall(nf*n, [=] MFEM_HOST_DEVICE (int idx_i)
      {
         const int i = idx_i % n;
         const int fi = idx_i / n;
         const int ii = c_scatter_map(i, fi);

         const int begin = d_face_face_offsets[fi];
         const int end = d_face_face_offsets[fi + 1];
         for (int idx = begin; idx < end; ++idx)
         {
            for (int j = 0; j < n; ++j)
            {
               if (d_CAhatInvCt(i, j, idx) != 0)
               {
                  I[ii]++;
               }
            }
         }
      });
   }

   // At this point, I[i] contains the number of nonzeros in row I. Perform a
   // partial sum to get I in CSR format. This is serial, so perform on host.
   //
   // At the same time, we find any empty rows (corresponding to non-ghost DOFs)
   // and add a single nonzero (we will put 1 on the diagonal) and record the
   // row index.
   Array<int> empty_rows;
   {
      int *I = h.H->HostReadWriteI();
      int empty_row_count = 0;
      for (int i = 0; i < ncdofs_local; i++)
      {
         if (I[i] == 0) { empty_row_count++; }
      }
      empty_rows.SetSize(empty_row_count);

      int empty_row_idx = 0;
      int sum = 0;
      for (int i = 0; i < ncdofs; i++)
      {
         int nnz = I[i];
         if (nnz == 0 && i < ncdofs_local)
         {
            empty_rows[empty_row_idx] = i;
            empty_row_idx++;
            nnz = 1;
         }
         I[i] = sum;
         sum += nnz;
      }
      I[ncdofs] = sum;
   }

   const int nnz = h.H->HostReadI()[ncdofs];
   h.H->GetMemoryJ().New(nnz, h.H->GetMemoryJ().GetMemoryType());
   h.H->GetMemoryData().New(nnz, h.H->GetMemoryData().GetMemoryType());

   {
      int *I = h.H->ReadWriteI();
      int *J = h.H->WriteJ();
      real_t *V = h.H->WriteData();

      mfem::forall(nf*n, [=] MFEM_HOST_DEVICE (int idx_i)
      {
         const int i = idx_i % n;
         const int fi = idx_i / n;
         const int ii = c_scatter_map[i + fi*n];
         const int begin = d_face_face_offsets[fi];
         const int end = d_face_face_offsets[fi + 1];
         for (int idx = begin; idx < end; ++idx)
         {
            const int fj = d_face_to_face[idx];
            for (int j = 0; j < n; ++j)
            {
               const real_t val = d_CAhatInvCt(i, j, idx);
               if (val != 0)
               {
                  const int k = I[ii];
                  const int jj = c_scatter_map(j, fj);
                  I[ii]++;
                  J[k] = jj;
                  V[k] = val;
               }
            }
         }
      });

      const int *d_empty_rows = empty_rows.Read();
      mfem::forall(empty_rows.Size(), [=] MFEM_HOST_DEVICE (int idx)
      {
         const int i = d_empty_rows[idx];
         const int k = I[i];
         I[i]++;
         J[k] = i;
         V[k] = 1.0;
      });
   }

   // Shift back down (serial, done on host)
   {
      int *I = h.H->HostReadWriteI();
      for (int i = ncdofs - 1; i > 0; --i)
      {
         I[i] = I[i-1];
      }
      I[0] = 0;
   }

#ifdef MFEM_USE_MPI
   if (c_pfes)
   {
      OperatorHandle dH(h.pH.Type());

      if (ncdofs_face_nbr > 0)
      {
         // Build the "face neighbor prolongation matrix" P_nbr, which maps from
         // VDOFs (i.e. L-vector) to L-vectors with face neighbor DOFs. The
         // action of P_nbr is equivalent to calling ExchangeFaceNbrData on a
         // ParGridFunction. We compute P^t A P with P = P_nbr to assemble the
         // face neighbor contributions into a parallel matrix.
         ParMesh &pmesh = *c_pfes->GetParMesh();

         HYPRE_BigInt ncdofs_bigint = ncdofs;
         const HYPRE_BigInt global_ncdofs = pmesh.ReduceInt(ncdofs);

         Array<HYPRE_BigInt> rows;
         Array<HYPRE_BigInt> *offsets[1] = { &rows };
         pmesh.GenerateOffsets(1, &ncdofs_bigint, offsets);

         Array<int> I(ncdofs + 1);
         auto d_I = I.Write();
         mfem::forall(ncdofs + 1, [=] MFEM_HOST_DEVICE (int i) { d_I[i] = i; });

         HYPRE_BigInt offset = c_pfes->GetMyDofOffset();
         Array<HYPRE_BigInt> J(ncdofs);
         auto d_J = J.Write();
         mfem::forall(ncdofs_local, [=] MFEM_HOST_DEVICE (int i)
         {
            d_J[i] = offset + i;
         });
         const HYPRE_BigInt *map = c_pfes->GetFaceNbrGlobalDofMapArray().Read();
         mfem::forall(ncdofs_face_nbr, [=] MFEM_HOST_DEVICE (int i)
         {
            d_J[ncdofs_local + i] = map[i];
         });

         Vector V(ncdofs);
         V.UseDevice();
         V = 1.0;

         auto P_face_nbr =
            std::make_unique<HypreParMatrix>(
               c_pfes->GetComm(), ncdofs, global_ncdofs, c_pfes->GlobalVSize(),
               I.HostReadWrite(), J.HostReadWrite(), V.HostReadWrite(), rows,
               c_pfes->GetDofOffsets());
         HypreParMatrix H_diag(c_pfes->GetComm(), global_ncdofs, rows, h.H.get());

         dH.Reset(RAP(&H_diag, P_face_nbr.get()));

         P_nbr = std::move(P_face_nbr);
      }
      else
      {
         dH.MakeSquareBlockDiag(c_pfes->GetComm(),c_pfes->GlobalVSize(),
                                c_pfes->GetDofOffsets(), h.H.get());
      }

      OperatorHandle pP(h.pH.Type());
      auto P_hyp = static_cast<HypreParMatrix*>(
                      P_pc ? P_pc.get() : c_pfes->Dof_TrueDof_Matrix());
      pP.ConvertFrom(P_hyp);
      h.pH.MakePtAP(dH, pP);
      h.H.reset();
   }
#endif
}

void HybridizationExtension::MultCt(const Vector &x, Vector &y) const
{
   Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int nf = mesh.GetNFbyType(FaceType::Interior);

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr =
      h.c_fes.GetFaceRestriction(ordering, FaceType::Interior);

   Vector x_evec(face_restr->Height());
   face_restr->Mult(x, x_evec);

   const int *d_el_to_face = el_to_face.Read();
   const int *d_el_face_offsets = el_face_offsets.Read();
   const auto d_Ct = Reshape(Ct_mat.Read(), n_hat_dof_per_el, n_c_dof_per_face,
                             n_el_face);
   const auto d_x_evec = Reshape(x_evec.Read(), n_c_dof_per_face, nf);
   auto d_y = Reshape(y.Write(), n_hat_dof_per_el, ne);

   mfem::forall(ne * n_hat_dof_per_el, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / n_hat_dof_per_el;
      const int i = idx % n_hat_dof_per_el;
      d_y(i, e) = 0.0;
      const int begin = d_el_face_offsets[e];
      const int end = d_el_face_offsets[e+1];
      for (int fi = begin; fi < end; ++fi)
      {
         const int f = d_el_to_face[fi];
         for (int j = 0; j < n_c_dof_per_face; ++j)
         {
            d_y(i, e) += d_Ct(i, j, fi)*d_x_evec(j, f);
         }
      }
   });
}

void HybridizationExtension::MultC(const Vector &x, Vector &y) const
{
   Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int nf = mesh.GetNFbyType(FaceType::Interior);

   const int n_hat_dof_per_el = h.fes.GetTypicalFE()->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetTypicalTraceElement()->GetDof();

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr = h.c_fes.GetFaceRestriction(
                                          ordering, FaceType::Interior);

   Vector y_evec(face_restr->Height());
   const auto d_face_to_el = Reshape(face_to_el.Read(), 2, 2, nf);
   const auto d_Ct = Reshape(Ct_mat.Read(), n_hat_dof_per_el, n_c_dof_per_face,
                             n_el_face);
   auto d_x = Reshape(x.Read(), n_hat_dof_per_el, ne);
   auto d_y_evec = Reshape(y_evec.Write(), n_c_dof_per_face, nf);

   mfem::forall(nf * n_c_dof_per_face, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int f = idx / n_c_dof_per_face;
      const int j = idx % n_c_dof_per_face;
      d_y_evec(j, f) = 0.0;
      for (int el_i = 0; el_i < 2; ++el_i)
      {
         const int e = d_face_to_el(0, el_i, f);
         const int fi = d_face_to_el(1, el_i, f);

         // Skip face neighbor elements of shared faces
         if (e >= ne) { continue; }

         for (int i = 0; i < n_hat_dof_per_el; ++i)
         {
            d_y_evec(j, f) += d_Ct(i, j, fi)*d_x(i, e);
         }
      }
   });

   y.SetSize(face_restr->Width());

   if (P_nbr)
   {
      auto l2_face_restr =
         dynamic_cast<const L2InterfaceFaceRestriction*>(face_restr);
      MFEM_ASSERT(l2_face_restr != nullptr, "");

      Vector y_s(P_nbr->Height());
      l2_face_restr->MultTransposeShared(y_evec, y_s);
      P_nbr->MultTranspose(y_s, y);
   }
   else
   {
      face_restr->MultTranspose(y_evec, y);
   }
}

void HybridizationExtension::AssembleMatrix(int el, const DenseMatrix &elmat)
{
   const int n = elmat.Width();
   const real_t *d_elmat = elmat.Read();
   real_t *d_Ahat = Ahat.ReadWrite();
   const int offset = el*n*n;
   mfem::forall(n*n, [=] MFEM_HOST_DEVICE (int i)
   {
      d_Ahat[offset + i] += d_elmat[i];
   });
}

void HybridizationExtension::AssembleBdrMatrix(int bdr_el,
                                               const DenseMatrix &elmat)
{
   DenseMatrix B = elmat; // deep copy
   const int n = h.fes.GetFE(0)->GetDof();
   // Create mapping e2f from element DOF indices to face DOF indices
   Array<int> e2f(n);
   e2f = -1;
   int el;
   {
      Mesh &mesh = *h.fes.GetMesh();
      int info;
      mesh.GetBdrElementAdjacentElement(bdr_el, el, info);
      Array<int> lvdofs;
      lvdofs.Reserve(elmat.Height());
      h.fes.FEColl()->SubDofOrder(mesh.GetElementGeometry(el),
                                  mesh.Dimension() - 1, info, lvdofs);
      // Convert local element dofs to local element vdofs.
      const int vdim = h.fes.GetVDim();
      Ordering::DofsToVDofs<Ordering::byNODES>(n/vdim, vdim, lvdofs);
      MFEM_ASSERT(lvdofs.Size() == elmat.Height(), "internal error");

      B.AdjustDofDirection(lvdofs);
      FiniteElementSpace::AdjustVDofs(lvdofs);
      // Create a map from local element vdofs to local boundary (face) vdofs.
      for (int i = 0; i < lvdofs.Size(); i++)
      {
         e2f[lvdofs[i]] = i;
      }
   }

   const int offset = el*n*n;
   Ahat.HostReadWrite();
   for (int j = 0; j < n; ++j)
   {
      const int j_f = e2f[j];
      if (j_f < 0) { continue; }
      for (int i = 0; i < n; ++i)
      {
         const int i_f = e2f[i];
         if (i_f < 0) { continue; }
         Ahat[offset + i + j*n] += B(i_f, j_f);
      }
   }
}

void HybridizationExtension::AssembleElementMatrices(const DenseTensor &elmats)
{
   const real_t *d_elmats = elmats.Read();
   real_t *d_Ahat = Ahat.ReadWrite();
   mfem::forall(elmats.TotalSize(), [=] MFEM_HOST_DEVICE (int i)
   {
      d_Ahat[i] += d_elmats[i];
   });
}

void HybridizationExtension::Init(const Array<int> &ess_tdof_list)
{
   // Verify that preconditions for the extension are met
   const Mesh &mesh = *h.fes.GetMesh();
   const int dim = mesh.Dimension();
   const int ne = h.fes.GetNE();
   const int nf = mesh.GetNFbyType(FaceType::Interior);
   const int ndof_per_el = h.fes.GetFE(0)->GetDof();
   const int ndof_per_face = h.c_fes.GetFaceElement(0)->GetDof();

   MFEM_VERIFY(!h.fes.IsVariableOrder(), "");
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(UsesTensorBasis(h.fes), "");

   // Set up array for idofs and bdofs
   {
      const TensorBasisElement* tbe =
         dynamic_cast<const TensorBasisElement*>(h.fes.GetFE(0));
      MFEM_VERIFY(tbe != nullptr, "");
      const Array<int> &dof_map = tbe->GetDofMap();

      const int n_faces_per_el = GetNFacesPerElement(mesh);

      Array<int> all_face_dofs;
      for (int f = 0; f < n_faces_per_el; ++f)
      {
         Array<int> face_map(ndof_per_face);
         h.fes.GetFE(0)->GetFaceMap(f, face_map);
         all_face_dofs.Append(face_map);
      }

      Array<bool> b_marker(ndof_per_el);
      b_marker = false;
      for (int i = 0; i < all_face_dofs.Size(); ++i)
      {
         const int j_s = all_face_dofs[i];
         const int j = (j_s >= 0) ? j_s : -1 - j_s;
         const int j_nat_s = dof_map[j];
         const int j_nat = (j_nat_s >= 0) ? j_nat_s : -1 - j_nat_s;
         b_marker[j_nat] = true;
      }

      for (int i = 0; i < ndof_per_el; ++i)
      {
         if (b_marker[i]) { bdofs.Append(i); }
         else { idofs.Append(i); }
      }
   }

   // Set up face info arrays
   el_face_offsets.SetSize(ne + 1);
   el_face_offsets = 0;
   // Count faces per element
   for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
   {
      const Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (!info.IsInterior() || info.IsNonconformingCoarse()) { continue; }
      el_face_offsets[info.element[0].index + 1] += 1;
      if (!info.IsShared())
      {
         el_face_offsets[info.element[1].index + 1] += 1;
      }
   }
   el_face_offsets.PartialSum();
   // Set up element-to-face and face-to-element arrays
   n_el_face = el_face_offsets.Last();
   el_to_face.SetSize(n_el_face);
   face_to_el.SetSize(4 * nf);

   {
      Array<int> el_face_counter(ne);
      el_face_counter = 0;

      int face_idx = 0;
      for (int f = 0; f < mesh.GetNumFacesWithGhost(); ++f)
      {
         const Mesh::FaceInformation info = mesh.GetFaceInformation(f);
         if (!info.IsInterior() || info.IsNonconformingCoarse()) { continue; }

         const int el1 = info.element[0].index;
         int &offset1 = el_face_offsets[el1];
         el_to_face[offset1] = face_idx;
         face_to_el[0 + 4*face_idx] = el1;
         face_to_el[1 + 4*face_idx] = offset1;

         offset1 += 1;

         const int el2 = info.element[1].index;
         if (!info.IsShared())
         {
            int &offset2 = el_face_offsets[el2];
            el_to_face[offset2] = face_idx;
            face_to_el[2 + 4*face_idx] = el2;
            face_to_el[3 + 4*face_idx] = offset2;
            offset2 += 1;
         }
         else
         {
            face_to_el[2 + 4*face_idx] = ne + el2;
            face_to_el[3 + 4*face_idx] = -1;
         }

         ++face_idx;
      }

      for (int i = ne; i > 0; i--)
      {
         el_face_offsets[i] = el_face_offsets[i-1];
      }
      el_face_offsets[0] = 0;
   }

   // Create the face-to-face connectivity
   {
      face_face_offsets.SetSize(nf + 1);
      const auto d_face_to_el = Reshape(face_to_el.Read(), 2, 2, nf);
      const auto d_el_face_offsets = el_face_offsets.Read();
      auto d_face_face_offsets = face_face_offsets.Write();
      mfem::forall(nf + 1, [=] MFEM_HOST_DEVICE (int i) { d_face_face_offsets[i] = 0; });
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int f)
      {
         int n_connections = 0;
         for (int ie = 0; ie < 2; ++ie)
         {
            // Number of faces adjacent to e
            const int e = d_face_to_el(0, ie, f);
            if (e < ne)
            {
               n_connections += d_el_face_offsets[e + 1] - d_el_face_offsets[e];
               // Subtract 1 since we are double-counting the face 'f' (it
               // belongs to both adjacent elements).
               if (ie > 0) { n_connections -= 1; }
            }
         }
         d_face_face_offsets[f + 1] = n_connections;
      });
      // TODO: parallel scan on device?
      face_face_offsets.HostReadWrite();
      face_face_offsets.PartialSum();
      n_face_face = face_face_offsets.Last();
   }

   // Count the number of dofs in the discontinuous version of fes:
   num_hat_dofs = ne*ndof_per_el;
   {
      h.hat_offsets.SetSize(ne + 1);
      int *d_hat_offsets = h.hat_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         d_hat_offsets[i] = i*ndof_per_el;
      });
   }

   Ahat.SetSize(ne*ndof_per_el*ndof_per_el);
   Ahat.UseDevice(true);
   Ahat = 0.0;

   ConstructC();

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const Operator *R_op = h.fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);
   MFEM_VERIFY(R, "");

   // Find out which "hat DOFs" are essential (depend only on essential Lagrange
   // multiplier DOFs).
   {
      const int ntdofs = h.fes.GetTrueVSize();
      // free_tdof_marker is 1 if the DOF is free, 0 if the DOF is essential
      Array<int> free_tdof_marker(ntdofs);
      {
         int *d_free_tdof_marker = free_tdof_marker.Write();
         mfem::forall(ntdofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[i] = 1;
         });
         const int n_ess_dofs = ess_tdof_list.Size();
         const int *d_ess_tdof_list = ess_tdof_list.Read();
         mfem::forall(n_ess_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            d_free_tdof_marker[d_ess_tdof_list[i]] = 0;
         });
      }

      Array<int> free_vdofs_marker;
#ifdef MFEM_USE_MPI
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(&h.fes);
      if (pfes)
      {
         HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
         free_vdofs_marker.SetSize(h.fes.GetVSize());
         // TODO: would be nice to do this on device
         P->BooleanMult(1, free_tdof_marker.HostRead(),
                        0, free_vdofs_marker.HostWrite());
      }
      else
#endif
      {
         const SparseMatrix *cP = h.fes.GetConformingProlongation();
         if (cP)
         {
            free_vdofs_marker.SetSize(cP->Height());
            cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
         }
         else
         {
            free_vdofs_marker.MakeRef(free_tdof_marker);
         }
      }

      hat_dof_marker.SetSize(num_hat_dofs);
      {
         // The gather map from the ElementRestriction operator gives us the
         // index of the L-dof corresponding to a given (element, local DOF)
         // index pair.
         const int *gather_map = R->GatherMap().Read();
         const int *d_free_vdofs_marker = free_vdofs_marker.Read();
         const auto d_Ct_mat = Reshape(Ct_mat.Read(), ndof_per_el,
                                       ndof_per_face, n_el_face);
         const int *d_el_face_offsets = el_face_offsets.Read();
         DofType *d_hat_dof_marker = hat_dof_marker.Write();

         // Set the hat_dofs_marker to 1 or 0 according to whether the DOF is
         // "free" or "essential". (For now, we mark all free DOFs as free
         // interior as a placeholder). Then, as a later step, the "free" DOFs
         // will be further classified as "interior free" or "boundary free".
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int j_s = gather_map[i];
            const int j = (j_s >= 0) ? j_s : -1 - j_s;
            if (d_free_vdofs_marker[j])
            {
               const int i_loc = i % ndof_per_el;
               const int e = i / ndof_per_el;
               d_hat_dof_marker[i] = INTERIOR;
               const int begin = d_el_face_offsets[e];
               const int end = d_el_face_offsets[e + 1];
               for (int f = begin; f < end; ++f)
               {
                  for (int k = 0; k < ndof_per_face; ++k)
                  {
                     if (d_Ct_mat(i_loc, k, f) != 0.0)
                     {
                        d_hat_dof_marker[i] = BOUNDARY;
                        break;
                     }
                  }
               }
            }
            else
            {
               d_hat_dof_marker[i] = ESSENTIAL;
            }
         });
      }
   }

   // Create the hat DOF gather map. This is used to apply the action of R and
   // R^T
   {
      const int vsize = h.fes.GetVSize();
      hat_dof_gather_map.SetSize(num_hat_dofs);
      const int *d_offsets = R->Offsets().Read();
      const int *d_indices = R->Indices().Read();
      int *d_hat_dof_gather_map = hat_dof_gather_map.Write();
      mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
      {
         d_hat_dof_gather_map[i] = -1;
      });
      mfem::forall(vsize, [=] MFEM_HOST_DEVICE (int i)
      {
         const int offset = d_offsets[i];
         const int j_s = d_indices[offset];
         const int hat_dof_index = (j_s >= 0) ? j_s : -1 - j_s;
         // Note: -1 is used as a special value (invalid), so the negative
         // DOF indices start at -2.
         d_hat_dof_gather_map[hat_dof_index] = (j_s >= 0) ? i : (-2 - i);
      });
   }
}

void HybridizationExtension::MultR(const Vector &x_hat, Vector &x) const
{
   const Operator *R = h.fes.GetRestrictionOperator();

   // If R is null, then L-vector and T-vector are the same, and we don't need
   // an intermediate temporary variable.
   //
   // If R is not null, we first convert to intermediate L-vector (with the
   // correct BCs), and then from L-vector to T-vector.
   if (!R)
   {
      MFEM_ASSERT(x.Size() == h.fes.GetVSize(), "");
      tmp2.MakeRef(x, 0);
   }
   else
   {
      tmp2.SetSize(R->Width());
      R->MultTranspose(x, tmp2);
   }

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const auto *restr = static_cast<const ElementRestriction*>(
                          h.fes.GetElementRestriction(ordering));
   const int *gather_map = restr->GatherMap().Read();
   const DofType *d_hat_dof_marker = hat_dof_marker.Read();
   const real_t *d_evec = x_hat.Read();
   real_t *d_lvec = tmp2.ReadWrite();
   mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
   {
      // Skip essential DOFs
      if (d_hat_dof_marker[i] == ESSENTIAL) { return; }

      const int j_s = gather_map[i];
      const int sgn = (j_s >= 0) ? 1 : -1;
      const int j = (j_s >= 0) ? j_s : -1 - j_s;

      d_lvec[j] = sgn*d_evec[i];
   });

   // Convert from L-vector to T-vector.
   if (R) { R->Mult(tmp2, x); }
}

void HybridizationExtension::MultRt(const Vector &b, Vector &b_hat) const
{
   Vector b_lvec;
   const Operator *R = h.fes.GetRestrictionOperator();
   if (!R)
   {
      b_lvec.MakeRef(const_cast<Vector&>(b), 0, b.Size());
   }
   else
   {
      tmp1.SetSize(h.fes.GetVSize());
      b_lvec.MakeRef(tmp1, 0, tmp1.Size());
      R->MultTranspose(b, b_lvec);
   }

   b_hat.SetSize(num_hat_dofs);
   const int *d_hat_dof_gather_map = hat_dof_gather_map.Read();
   const real_t *d_b_lvec = b_lvec.Read();
   real_t *d_b_hat = b_hat.Write();
   mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
   {
      const int j_s = d_hat_dof_gather_map[i];
      if (j_s == -1) // invalid
      {
         d_b_hat[i] = 0.0;
      }
      else
      {
         const int sgn = (j_s >= 0) ? 1 : -1;
         const int j = (j_s >= 0) ? j_s : -2 - j_s;
         d_b_hat[i] = sgn*d_b_lvec[j];
      }
   });
}

void HybridizationExtension::MultAhatInv(Vector &x) const
{
   const int ne = h.fes.GetMesh()->GetNE();
   const int n = h.fes.GetFE(0)->GetDof();

   const int nidofs = idofs.Size();
   const int nbdofs = bdofs.Size();

   const auto d_hat_dof_marker = Reshape(hat_dof_marker.Read(), n, ne);

   const auto d_A_ii = Reshape(Ahat_ii.Read(), nidofs, nidofs, ne);
   const auto d_A_ib = Reshape(Ahat_ib.Read(), nidofs*nbdofs, ne);
   const auto d_A_bi = Reshape(Ahat_bi.Read(), nbdofs*nidofs, ne);
   const auto d_A_bb = Reshape(Ahat_bb.Read(), nbdofs*nbdofs, ne);

   const auto d_ipiv_ii = Reshape(Ahat_ii_piv.Read(), nidofs, ne);
   const auto d_ipiv_bb = Reshape(Ahat_bb_piv.Read(), nbdofs, ne);

   const auto *d_idofs = idofs.Read();
   const auto *d_bdofs = bdofs.Read();

   Vector ivals(nidofs*ne);
   Vector bvals(nbdofs*ne);
   auto d_ivals = Reshape(ivals.Write(), nidofs, ne);
   auto d_bvals = Reshape(bvals.Write(), nbdofs, ne);

   auto d_x = Reshape(x.ReadWrite(), n, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MAX_DOFS = 3*MD1D*(MD1D-1)*(MD1D-1);
      internal::LocalMemory<int,MAX_DOFS> bdofs_loc;

      int nbfdofs = 0;
      for (int i = 0; i < nbdofs; i++)
      {
         const int dof_idx = d_bdofs[i];
         if (d_hat_dof_marker(dof_idx, e) != ESSENTIAL)
         {
            bdofs_loc[nbfdofs] = dof_idx;
            nbfdofs += 1;
         }
      }

      for (int i = 0; i < nidofs; ++i)
      {
         d_ivals(i, e) = d_x(d_idofs[i], e);
      }
      for (int i = 0; i < nbfdofs; ++i)
      {
         d_bvals(i, e) = d_x(bdofs_loc[i], e);
      }

      if (nidofs > 0)
      {
         // Block forward substitution:
         // B1 <- L^{-1} P B1
         kernels::LSolve(&d_A_ii(0,0,e), nidofs, &d_ipiv_ii(0,e), &d_ivals(0,e));
         // B2 <- B2 - L21 B1
         kernels::SubMult(
            nidofs, nbfdofs, 1, &d_A_bi(0,e), &d_ivals(0,e), &d_bvals(0, e));
      }

      // Schur complement solve
      kernels::LUSolve(&d_A_bb(0,e), nbfdofs, &d_ipiv_bb(0,e), &d_bvals(0,e));

      if (nidofs > 0)
      {
         // Block backward substitution
         // Y1 <- Y1 - U12 X2
         kernels::SubMult(
            nbfdofs, nidofs, 1, &d_A_ib(0,e), &d_bvals(0,e), &d_ivals(0, e));
         // Y1 <- U^{-1} Y1
         kernels::USolve(&d_A_ii(0,0,e), nidofs, &d_ivals(0,e));
      }

      for (int i = 0; i < nidofs; ++i)
      {
         d_x(d_idofs[i], e) = d_ivals(i, e);
      }
      for (int i = 0; i < nbfdofs; ++i)
      {
         d_x(bdofs_loc[i], e) = d_bvals(i, e);
      }
   });
}

void HybridizationExtension::ReduceRHS(const Vector &b, Vector &b_r) const
{
   Vector b_hat(num_hat_dofs);
   MultRt(b, b_hat);
   {
      const auto *d_hat_dof_marker = hat_dof_marker.Read();
      auto *d_b_hat = b_hat.ReadWrite();
      mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
      {
         if (d_hat_dof_marker[i] == ESSENTIAL) { d_b_hat[i] = 0.0; }
      });
   }
   MultAhatInv(b_hat);

   if (IsParFESpace(h.c_fes))
   {
      const Operator &P = GetProlongation();
      Vector bl(P.Height());
      b_r.SetSize(P.Width());
      MultC(b_hat, bl);
      P.MultTranspose(bl, b_r);
   }
   else
   {
      MultC(b_hat, b_r);
   }
}

void HybridizationExtension::ComputeSolution(
   const Vector &b, const Vector &sol_r, Vector &sol) const
{
   // tmp1 = A_hat^{-1} ( R^T b - C^T lambda )
   Vector b_hat(num_hat_dofs);
   MultRt(b, b_hat);

   tmp1.SetSize(num_hat_dofs);

   if (IsParFESpace(h.c_fes))
   {
      const Operator &P = GetProlongation();
      Vector sol_l(P.Height());
      P.Mult(sol_r, sol_l);
      MultCt(sol_l, tmp1);
   }
   else
   {
      MultCt(sol_r, tmp1);
   }

   add(b_hat, -1.0, tmp1, tmp1);
   // Eliminate essential DOFs
   const auto *d_hat_dof_marker = hat_dof_marker.Read();
   real_t *d_tmp1 = tmp1.ReadWrite();
   mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
   {
      if (d_hat_dof_marker[i] == ESSENTIAL) { d_tmp1[i] = 0.0; }
   });
   MultAhatInv(tmp1);
   MultR(tmp1, sol);
}

}
