// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "../general/forall.hpp"

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

void HybridizationExtension::ConstructC()
{
   Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();
   const int n_faces_per_el = GetNFacesPerElement(mesh);

   el_to_face.SetSize(ne * n_faces_per_el);
   face_to_el.SetSize(4 * mesh.GetNFbyType(FaceType::Interior));
   Ct_mat.SetSize(ne * n_hat_dof_per_el * n_c_dof_per_face * n_faces_per_el);

   // Assemble Ct_mat using EA
   Vector emat;
   emat.NewMemoryAndSize(Ct_mat.GetMemory(), Ct_mat.Size(), false);
   h.c_bfi->AssembleEAInteriorFaces(h.c_fes, h.fes, emat, false);

   el_to_face = -1;
   int face_idx = 0;

   // Set up el_to_face and face_to_el arrays
   for (int f = 0; f < mesh.GetNumFaces(); ++f)
   {
      const Mesh::FaceInformation info = mesh.GetFaceInformation(f);
      if (!info.IsInterior()) { continue; }

      const int el1 = info.element[0].index;
      const int fi1 = info.element[0].local_face_id;
      el_to_face[el1 * n_faces_per_el + fi1] = face_idx;

      const int el2 = info.element[1].index;
      const int fi2 = info.element[1].local_face_id;
      el_to_face[el2 * n_faces_per_el + fi2] = face_idx;

      face_to_el[0 + 4*face_idx] = el1;
      face_to_el[1 + 4*face_idx] = fi1;
      face_to_el[2 + 4*face_idx] = el2;
      face_to_el[3 + 4*face_idx] = fi2;

      ++face_idx;
   }
}

void HybridizationExtension::ConstructH()
{
   const Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int n_faces_per_el = GetNFacesPerElement(mesh);
   const int m = h.fes.GetFE(0)->GetDof();
   const int n = h.c_fes.GetFaceElement(0)->GetDof();

   Vector AhatInvCt_mat(Ct_mat);
   for (int e = 0; e < ne; ++e)
   {
      for (int fi = 0; fi < n_faces_per_el; ++fi)
      {
         const int offset = (fi + e*n_faces_per_el)*n*m;
         real_t *data = const_cast<real_t*>(Ahat_inv.GetData() + e*m*m);
         int *ipiv = const_cast<int*>(Ahat_piv.GetData() + e*m);
         LUFactors lu(data, ipiv);
         lu.Solve(m, n, AhatInvCt_mat.GetData() + offset);
      }
   }

   const int nf = h.fes.GetNFbyType(FaceType::Interior);
   const int n_face_connections = 2*n_faces_per_el - 1;
   Array<int> face_to_face(nf * n_face_connections);

   Array<double> CAhatInvCt_mat(nf*n_face_connections*n*n);
   CAhatInvCt_mat = 0.0;

   for (int fi = 0; fi < nf; ++fi)
   {
      int idx = 0;
      for (int ei = 0; ei < 2; ++ei)
      {
         const int e = face_to_el[2*ei + 4*fi];
         if (e < 0) { continue; }
         const int fi_i = face_to_el[1 + 2*ei + 4*fi];
         for (int fj_i = 0; fj_i < n_faces_per_el; ++fj_i)
         {
            const int fj = el_to_face[fj_i + e*n_faces_per_el];
            // if (fj < 0 || fi == fj) { continue; }
            // Explicitly allow fi == fj (self-connections)
            if (fj < 0) { continue; }

            // Have we seen this face before? It is possible in some
            // configurations to encounter the same neighboring face twice
            int idx_j = idx;
            for (int i = 0; i < idx; ++i)
            {
               if (face_to_face[i + fi*n_face_connections] == fj)
               {
                  idx_j = i;
                  break;
               }
            }
            // This is a new face, record it and increment the counter
            if (idx_j == idx)
            {
               face_to_face[idx + fi*n_face_connections] = fj;
               idx++;
            }

            const int offset_1 = (idx_j + fi*n_face_connections)*n*n;
            DenseMatrix CAhatInvCt(CAhatInvCt_mat.GetData() + offset_1, n, n);
            const int offset_2 = (fi_i + e*n_faces_per_el)*n*m;
            DenseMatrix Ct(Ct_mat.GetData() + offset_2, m, n);
            const int offset_3 = (fj_i + e*n_faces_per_el)*n*m;
            DenseMatrix AhatInvCt(AhatInvCt_mat.GetData() + offset_3, m, n);
            AddMultAtB(Ct, AhatInvCt, CAhatInvCt);
         }
      }
      // Fill unused entries with -1 to indicate invalid
      for (; idx < n_face_connections; ++idx)
      {
         face_to_face[idx + fi*n_face_connections] = -1;
      }
   }

   const int ncdofs = h.c_fes.GetTrueVSize();
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr =
      h.c_fes.GetFaceRestriction( ordering, FaceType::Interior);
   const auto c_gather_map = Reshape(face_restr->GatherMap().Read(), n, nf);

   const auto d_f2f = Reshape(face_to_face.Read(), n_face_connections, nf);
   const auto d_CAhatInvCt_mat = Reshape(CAhatInvCt_mat.Read(),
                                         n, n, n_face_connections, nf);

   h.H.reset(new SparseMatrix);
   h.H->OverrideSize(ncdofs, ncdofs);

   h.H->GetMemoryI().New(ncdofs + 1, h.H->GetMemoryI().GetMemoryType());

   {
      int *I = h.H->WriteI();

      mfem::forall(ncdofs, [=] MFEM_HOST_DEVICE (int i) { I[i] = 0; });

      // TODO: to expose more parallelism, should be able to make this forall
      // loop over nf*n (for indices fi and i)
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int fi)
      {
         for (int idx = 0; idx < n_face_connections; ++idx)
         {
            const int fj = d_f2f(idx, fi);
            if (fj < 0) { break; }
            for (int i = 0; i < n; ++ i)
            {
               const int ii = c_gather_map(i, fi);
               for (int j = 0; j < n; ++j)
               {
                  if (d_CAhatInvCt_mat(i, j, idx, fi) != 0)
                  {
                     I[ii]++;
                  }
               }
            }
         }
      });
   }

   // At this point, I[i] contains the number of nonzeros in row I. Perform a
   // partial sum to get I in CSR format. This is serial, so perform on host.
   //
   // At the same time, we find any empty rows and add a single nonzero (we will
   // put 1 on the diagonal) and record the row index.
   Array<int> empty_rows;
   {
      int *I = h.H->HostReadWriteI();
      int empty_row_count = 0;
      for (int i = 0; i < ncdofs; i++)
      {
         if (I[i] == 0) { empty_row_count++; }
      }
      empty_rows.SetSize(empty_row_count);

      int empty_row_idx = 0;
      int sum = 0;
      for (int i = 0; i < ncdofs; i++)
      {
         int nnz = I[i];
         if (nnz == 0)
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

      // TODO: to expose more parallelism, should be able to make this forall
      // loop over nf*n (for indices fi and i)
      mfem::forall(nf, [=] MFEM_HOST_DEVICE (int fi)
      {
         for (int idx = 0; idx < n_face_connections; ++idx)
         {
            const int fj = d_f2f(idx, fi);
            if (fj < 0) { break; }
            for (int i = 0; i < n; ++ i)
            {
               const int ii = c_gather_map[i + fi*n];
               for (int j = 0; j < n; ++j)
               {
                  const real_t val = d_CAhatInvCt_mat(i, j, idx, fi);
                  if (val != 0)
                  {
                     const int k = I[ii];
                     const int jj = c_gather_map(j, fj);
                     I[ii]++;
                     J[k] = jj;
                     V[k] = val;
                  }
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

   // Shift back down
   {
      int *I = h.H->HostReadWriteI();
      for (int i = ncdofs - 1; i > 0; --i)
      {
         I[i] = I[i-1];
      }
      I[0] = 0;
   }
}

void HybridizationExtension::MultCt(const Vector &x, Vector &y) const
{
   Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int nf = mesh.GetNFbyType(FaceType::Interior);

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();
   const int n_faces_per_el = GetNFacesPerElement(mesh);

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr =
      h.c_fes.GetFaceRestriction(ordering, FaceType::Interior);

   Vector x_evec(face_restr->Height());
   face_restr->Mult(x, x_evec);

   const int *d_el_to_face = el_to_face.Read();
   const auto d_Ct = Reshape(Ct_mat.Read(), n_hat_dof_per_el, n_c_dof_per_face,
                             n_faces_per_el, ne);
   const auto d_x_evec = Reshape(x_evec.Read(), n_c_dof_per_face, nf);
   auto d_y = Reshape(y.Write(), n_hat_dof_per_el, ne);

   mfem::forall(ne * n_hat_dof_per_el, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / n_hat_dof_per_el;
      const int i = idx % n_hat_dof_per_el;
      d_y(i, e) = 0.0;
      for (int fi = 0; fi < n_faces_per_el; ++fi)
      {
         const int f = d_el_to_face[e*n_faces_per_el + fi];
         if (f < 0) { continue; }
         for (int j = 0; j < n_c_dof_per_face; ++j)
         {
            d_y(i, e) += d_Ct(i, j, fi, e)*d_x_evec(j, f);
         }
      }
   });
}

void HybridizationExtension::MultC(const Vector &x, Vector &y) const
{
   Mesh &mesh = *h.fes.GetMesh();
   const int ne = mesh.GetNE();
   const int nf = mesh.GetNFbyType(FaceType::Interior);

   const int n_hat_dof_per_el = h.fes.GetFE(0)->GetDof();
   const int n_c_dof_per_face = h.c_fes.GetFaceElement(0)->GetDof();
   const int n_faces_per_el = GetNFacesPerElement(mesh);

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const FaceRestriction *face_restr = h.c_fes.GetFaceRestriction(
                                          ordering, FaceType::Interior);

   Vector y_evec(face_restr->Height());
   const auto d_face_to_el = Reshape(face_to_el.Read(), 2, 2, nf);
   const auto d_Ct = Reshape(Ct_mat.Read(), n_hat_dof_per_el, n_c_dof_per_face,
                             n_faces_per_el, ne);
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

         for (int i = 0; i < n_hat_dof_per_el; ++i)
         {
            d_y_evec(j, f) += d_Ct(i, j, fi, e)*d_x(i, e);
         }
      }
   });

   y.SetSize(face_restr->Width());
   face_restr->MultTranspose(y_evec, y);
}

void HybridizationExtension::AssembleMatrix(int el, const DenseMatrix &elmat)
{
   const int n = elmat.Width();
   double *Ainv = Ahat_inv.GetData() + el*n*n;
   int *ipiv = Ahat_piv.GetData() + el*n;

   std::copy(elmat.GetData(), elmat.GetData() + n*n, Ainv);

   // Eliminate essential DOFs from the local matrix
   for (int i = 0; i < n; ++i)
   {
      const int idx = i + el*n;
      if (h.hat_dofs_marker[idx] == ESSENTIAL)
      {
         for (int j = 0; j < n; ++j)
         {
            Ainv[i + j*n] = 0.0;
            Ainv[j + i*n] = 0.0;
         }
         Ainv[i + i*n] = 1.0;
      }
   }

   LUFactors lu(Ainv, ipiv);
   lu.Factor(n);
}

void HybridizationExtension::AssembleElementMatrices(
   const class DenseTensor &el_mats)
{
   Ahat_inv.GetMemory().CopyFrom(el_mats.GetMemory(), el_mats.TotalSize());

   const int ne = h.fes.GetNE();
   const int n = el_mats.SizeI();
   for (int e = 0; e < ne; ++e)
   {
      double *Ainv = Ahat_inv.GetData() + e*n*n;
      int *ipiv = Ahat_piv.GetData() + e*n;
      LUFactors lu(Ainv, ipiv);
      lu.Factor(n);
   }
}

void HybridizationExtension::Init(const Array<int> &ess_tdof_list)
{
   // Verify that preconditions for the extension are met
   const Mesh &mesh = *h.fes.GetMesh();
   const int dim = mesh.Dimension();
   MFEM_VERIFY(!h.fes.IsVariableOrder(), "");
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(mesh.Conforming(), "");
   MFEM_VERIFY(UsesTensorBasis(h.fes), "");

   // Count the number of dofs in the discontinuous version of fes:
   const int ne = h.fes.GetNE();
   const int ndof_per_el = h.fes.GetFE(0)->GetDof();
   num_hat_dofs = ne*ndof_per_el;
   {
      h.hat_offsets.SetSize(ne + 1);
      int *d_hat_offsets = h.hat_offsets.Write();
      mfem::forall(ne + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         d_hat_offsets[i] = i*ndof_per_el;
      });
   }

   Ahat_inv.SetSize(ne*ndof_per_el*ndof_per_el);
   Ahat_piv.SetSize(ne*ndof_per_el);

   ConstructC();
   h.ConstructC(); // TODO: delete me

   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   const Operator *R_op = h.fes.GetElementRestriction(ordering);
   const auto *R = dynamic_cast<const ElementRestriction*>(R_op);
   MFEM_VERIFY(R, "");

   // We now split the "hat DOFs" (broken DOFs) into three classes of DOFs, each
   // marked with an integer:
   //
   //  1: essential DOFs (depending only on essential Lagrange multiplier DOFs)
   //  0: free interior DOFs (the corresponding column in the C matrix is zero,
   //     this happens when the DOF is in the interior on an element)
   // -1: free boundary DOFs (free DOFs that lie on the interface between two
   //     elements)
   //
   // These integers are used to define the values in HatDofType enum.
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
      const SparseMatrix *cP = h.fes.GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(h.fes.GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }

      h.hat_dofs_marker.SetSize(num_hat_dofs);
      {
         // The gather map from the ElementRestriction operator gives us the
         // index of the L-dof corresponding to a given (element, local DOF)
         // index pair.
         const int *gather_map = R->GatherMap().Read();
         const int *d_free_vdofs_marker = free_vdofs_marker.Read();
         int *d_hat_dofs_marker = h.hat_dofs_marker.Write();

         // Set the hat_dofs_marker to 1 or 0 according to whether the DOF is
         // "free" or "essential". (For now, we mark all free DOFs as free
         // interior as a placeholder). Then, as a later step, the "free" DOFs
         // will be further classified as "interior free" or "boundary free".
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int j_s = gather_map[i];
            const int j = (j_s >= 0) ? j_s : -1 - j_s;
            d_hat_dofs_marker[i] = d_free_vdofs_marker[j] ? FREE_INTERIOR : ESSENTIAL;
         });

         // A free DOF is "interior" or "internal" if the corresponding column
         // of C (row of C^T) is zero. We mark the free DOFs with non-zero
         // columns of C as boundary free DOFs.
         const int *d_I = h.Ct->ReadI();
         mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
         {
            const int row_size = d_I[i+1] - d_I[i];
            if (d_hat_dofs_marker[i] == 0 && row_size > 0)
            {
               d_hat_dofs_marker[i] = FREE_BOUNDARY;
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
   const int *d_hat_dofs_marker = h.hat_dofs_marker.Read();
   const double *d_evec = x_hat.Read();
   double *d_lvec = tmp2.ReadWrite();
   mfem::forall(num_hat_dofs, [=] MFEM_HOST_DEVICE (int i)
   {
      // Skip essential DOFs
      if (d_hat_dofs_marker[i] == ESSENTIAL) { return; }

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
   const double *d_b_lvec = b_lvec.Read();
   double *d_b_hat = b_hat.Write();
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

   for (int i = 0; i < ne; ++i)
   {
      real_t *data = const_cast<real_t*>(Ahat_inv.GetData() + i*n*n);
      int *ipiv = const_cast<int*>(Ahat_piv.GetData() + i*n);
      LUFactors lu(data, ipiv);
      lu.Solve(n, 1, x.GetData() + i*n);
   }
}

void HybridizationExtension::ReduceRHS(const Vector &b, Vector &b_r) const
{
   Vector b_hat(num_hat_dofs);
   MultRt(b, b_hat);
   MultAhatInv(b_hat);
   MultC(b_hat, b_r);
}

void HybridizationExtension::ComputeSolution(
   const Vector &b, const Vector &sol_r, Vector &sol) const
{
   // tmp1 = A_hat^{-1} ( R^T b - C^T lambda )
   Vector b_hat(num_hat_dofs);
   MultRt(b, b_hat);

   tmp1.SetSize(num_hat_dofs);
   MultCt(sol_r, tmp1);
   add(b_hat, -1.0, tmp1, tmp1);
   // Eliminate essential DOFs
   for (int i = 0; i < num_hat_dofs; ++i)
   {
      if (h.hat_dofs_marker[i] == ESSENTIAL)
      {
         tmp1[i] = 0.0;
      }
   }
   MultAhatInv(tmp1);
   MultR(tmp1, sol);
}

}
