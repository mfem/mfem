// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lor.hpp"
#include "lor_sparsity.hpp"
#include "../pbilinearform.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

static MFEM_HOST_DEVICE int GetMinElt(const int *my_elts, const int nbElts,
                                      const int *nbr_elts, const int nbrNbElts)
{
   // Find the minimal element index found in both my_elts[] and nbr_elts[]
   int min_el = INT_MAX;
   for (int i = 0; i < nbElts; i++)
   {
      const int e_i = my_elts[i];
      if (e_i >= min_el) { continue; }
      for (int j = 0; j < nbrNbElts; j++)
      {
         if (e_i==nbr_elts[j])
         {
            min_el = e_i; // we already know e_i < min_el
            break;
         }
      }
   }
   return min_el;
}

int LORSparsity::FillI(SparseMatrix &A, const DenseMatrix &sparse_mapping) const
{
   static constexpr int Max = 16;

   const int nvdof = fes_ho.GetVSize();

   const int ndof_per_el = fes_ho.GetFE(0)->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_mapping.Height();

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = fes_ho.GetElementRestriction(ordering);
   const ElementRestriction *el_restr =
      dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != nullptr, "");

   const Array<int> &el_dof_lex_ = el_restr->GatherMap();
   const Array<int> &dof_glob2loc_ = el_restr->Indices();
   const Array<int> &dof_glob2loc_offsets_ = el_restr->Offsets();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto dof_glob2loc = dof_glob2loc_.Read();
   const auto K = dof_glob2loc_offsets_.Read();
   const auto map = Reshape(sparse_mapping.Read(), nnz_per_row, ndof_per_el);

   auto I = A.WriteI();

   MFEM_FORALL(ii, nvdof + 1, I[ii] = 0;);
   MFEM_FORALL(iel_ho, nel_ho,
   {
      for (int ii_el = 0; ii_el < ndof_per_el; ++ii_el)
      {
         // LDOF index of current row
         const int sii = el_dof_lex(ii_el, iel_ho); // signed
         const int ii = (sii >= 0) ? sii : -1 - sii;
         // Get number and list of elements containing this DOF
         int i_elts[Max];
         const int i_offset = K[ii];
         const int i_next_offset = K[ii+1];
         const int i_ne = i_next_offset - i_offset;
         for (int e_i = 0; e_i < i_ne; ++e_i)
         {
            const int si_E = dof_glob2loc[i_offset+e_i]; // signed
            const int i_E = (si_E >= 0) ? si_E : -1 - si_E;
            i_elts[e_i] = i_E/ndof_per_el;
         }

         for (int j = 0; j < nnz_per_row; ++j)
         {
            int jj_el = map(j, ii_el);
            if (jj_el < 0) { continue; }
            // LDOF index of column
            const int sjj = el_dof_lex(jj_el, iel_ho); // signed
            const int jj = (sjj >= 0) ? sjj : -1 - sjj;
            const int j_offset = K[jj];
            const int j_next_offset = K[jj+1];
            const int j_ne = j_next_offset - j_offset;
            if (i_ne == 1 || j_ne == 1) // no assembly required
            {
               AtomicAdd(I[ii], 1);
            }
            else // assembly required
            {
               int j_elts[Max];
               for (int e_j = 0; e_j < j_ne; ++e_j)
               {
                  const int sj_E = dof_glob2loc[j_offset+e_j]; // signed
                  const int j_E = (sj_E >= 0) ? sj_E : -1 - sj_E;
                  const int elt = j_E/ndof_per_el;
                  j_elts[e_j] = elt;
               }
               const int min_e = GetMinElt(i_elts, i_ne, j_elts, j_ne);
               if (iel_ho == min_e) // add the nnz only once
               {
                  AtomicAdd(I[ii], 1);
               }
            }
         }
      }
   });
   // TODO: on device
   // We need to sum the entries of I, we do it on CPU as it is very sequential.
   auto h_I = A.HostReadWriteI();
   int sum = 0;
   for (int i = 0; i < nvdof; i++)
   {
      const int nnz = h_I[i];
      h_I[i] = sum;
      sum+=nnz;
   }
   h_I[nvdof] = sum;

   // Return the number of nnz
   return h_I[nvdof];
}

/** Returns the index where a non-zero entry should be added and increment the
    number of non-zeros for the row i_L. */
static MFEM_HOST_DEVICE int GetAndIncrementNnzIndex(const int i_L, int* I)
{
   int ind = AtomicAdd(I[i_L],1);
   return ind;
}

void LORSparsity::FillJAndData(SparseMatrix &A, const Vector &sparse_ij,
                               const DenseMatrix &sparse_mapping) const
{
   const int nvdof = fes_ho.GetVSize();
   const int ndof_per_el = fes_ho.GetFE(0)->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_mapping.Height();

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = fes_ho.GetElementRestriction(ordering);
   const ElementRestriction *el_restr =
      dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != nullptr, "");

   const Array<int> &el_dof_lex_ = el_restr->GatherMap();
   const Array<int> &dof_glob2loc_ = el_restr->Indices();
   const Array<int> &dof_glob2loc_offsets_ = el_restr->Offsets();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto dof_glob2loc = dof_glob2loc_.Read();
   const auto K = dof_glob2loc_offsets_.Read();

   const auto V = Reshape(sparse_ij.Read(), nnz_per_row, ndof_per_el, nel_ho);
   const auto map = Reshape(sparse_mapping.Read(), nnz_per_row, ndof_per_el);

   Array<int> I_(nvdof + 1);
   const auto I = I_.Write();
   const auto J = A.WriteJ();
   auto AV = A.WriteData();

   // Copy A.I into I, use it as a temporary buffer
   {
      const auto I2 = A.ReadI();
      MFEM_FORALL(i, nvdof + 1, I[i] = I2[i];);
   }

   static constexpr int Max = 16;

   // MFEM_FORALL(iel_ho, nel_ho,
   for (int iel_ho=0; iel_ho<nel_ho; ++iel_ho)
   {
      for (int ii_el = 0; ii_el < ndof_per_el; ++ii_el)
      {
         // LDOF index of current row
         const int sii = el_dof_lex(ii_el, iel_ho); // signed
         const int ii = (sii >= 0) ? sii : -1 - sii;
         // Get number and list of elements containing this DOF
         int i_elts[Max];
         int i_B[Max];
         const int i_offset = K[ii];
         const int i_next_offset = K[ii+1];
         const int i_ne = i_next_offset - i_offset;
         for (int e_i = 0; e_i < i_ne; ++e_i)
         {
            const int si_E = dof_glob2loc[i_offset+e_i]; // signed
            const bool plus = si_E >= 0;
            const int i_E = plus ? si_E : -1 - si_E;
            i_elts[e_i] = i_E/ndof_per_el;
            const double i_Bi = i_E%ndof_per_el;
            i_B[e_i] = plus ? i_Bi : -1 - i_Bi; // encode with sign
         }
         for (int j=0; j<nnz_per_row; ++j)
         {
            int jj_el = map(j, ii_el);
            if (jj_el < 0) { continue; }
            // LDOF index of column
            const int sjj = el_dof_lex(jj_el, iel_ho); // signed
            const int jj = (sjj >= 0) ? sjj : -1 - sjj;
            const int sgn = ((sjj >=0 && sii >= 0) || (sjj < 0 && sii <0)) ? 1 : -1;
            const int j_offset = K[jj];
            const int j_next_offset = K[jj+1];
            const int j_ne = j_next_offset - j_offset;
            if (i_ne == 1 || j_ne == 1) // no assembly required
            {
               const int nnz = GetAndIncrementNnzIndex(ii, I);
               J[nnz] = jj;
               AV[nnz] = sgn*V(j, ii_el, iel_ho);
            }
            else // assembly required
            {
               int j_elts[Max];
               int j_B[Max];
               for (int e_j = 0; e_j < j_ne; ++e_j)
               {
                  const int sj_E = dof_glob2loc[j_offset+e_j]; // signed
                  const bool plus = sj_E >= 0;
                  const int j_E = plus ? sj_E : -1 - sj_E;
                  j_elts[e_j] = j_E/ndof_per_el;
                  const double j_Bj = j_E%ndof_per_el;
                  j_B[e_j] = plus ? j_Bj : -1 - j_Bj; // encode with sign
               }
               const int min_e = GetMinElt(i_elts, i_ne, j_elts, j_ne);
               if (iel_ho == min_e) // add the nnz only once
               {
                  double val = 0.0;
                  for (int k = 0; k < i_ne; k++)
                  {
                     const int iel_ho_2 = i_elts[k];
                     const int sii_el_2 = i_B[k]; // signed
                     const int ii_el_2 = (sii_el_2 >= 0) ? sii_el_2 : -1 -sii_el_2;
                     for (int l = 0; l < j_ne; l++)
                     {
                        const int jel_ho_2 = j_elts[l];
                        if (iel_ho_2 == jel_ho_2)
                        {
                           const int sjj_el_2 = j_B[l]; // signed
                           const int jj_el_2 = (sjj_el_2 >= 0) ? sjj_el_2 : -1 -sjj_el_2;
                           const int sgn_2 = ((sjj_el_2 >=0 && sii_el_2 >= 0)
                                              || (sjj_el_2 < 0 && sii_el_2 <0)) ? 1 : -1;
                           int j2 = -1;
                           // find nonzero in matrix of other element
                           for (int m = 0; m < nnz_per_row; ++m)
                           {
                              if (map(m, ii_el_2) == jj_el_2)
                              {
                                 j2 = m;
                                 break;
                              }
                           }
                           MFEM_ASSERT_KERNEL(j >= 0, "");
                           val += sgn_2*V(j2, ii_el_2, iel_ho_2);
                        }
                     }
                  }
                  const int nnz = GetAndIncrementNnzIndex(ii, I);
                  J[nnz] = jj;
                  AV[nnz] = val;
               }
            }
         }
      }
   }//);
}

SparseMatrix *LORSparsity::FormCSR(const Vector &sparse_ij,
                                   const DenseMatrix &sparse_mapping) const
{
   const int nvdof = fes_ho.GetVSize();

   SparseMatrix *A = new SparseMatrix(nvdof, nvdof, 0);

   A->GetMemoryI().New(nvdof+1, A->GetMemoryI().GetMemoryType());
   int nnz = FillI(*A, sparse_mapping);

   A->GetMemoryJ().New(nnz, A->GetMemoryJ().GetMemoryType());
   A->GetMemoryData().New(nnz, A->GetMemoryData().GetMemoryType());
   FillJAndData(*A, sparse_ij, sparse_mapping);

   return A;
}

} // namespace mfem
