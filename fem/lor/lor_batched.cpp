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

#include "lor_batched.hpp"
#include "../../fem/quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include <climits>
#include "../pbilinearform.hpp"
#include "../../fem/fe/face_map_utils.hpp"

// Specializations
#include "lor_h1.hpp"
#include "lor_dg.hpp"
#include "lor_nd.hpp"
#include "lor_rt.hpp"

namespace mfem
{

template <typename T1, typename T2>
bool HasIntegrators(BilinearForm &a)
{
   Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
   if (integs == NULL) { return false; }
   if (integs->Size() == 1)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      if (dynamic_cast<T1*>(i0) || dynamic_cast<T2*>(i0)) { return true; }
   }
   else if (integs->Size() == 2)
   {
      BilinearFormIntegrator *i0 = (*integs)[0];
      BilinearFormIntegrator *i1 = (*integs)[1];
      if ((dynamic_cast<T1*>(i0) && dynamic_cast<T2*>(i1)) ||
          (dynamic_cast<T2*>(i0) && dynamic_cast<T1*>(i1)))
      {
         return true;
      }
   }
   return false;
}

bool BatchedLORAssembly::FormIsSupported(BilinearForm &a)
{
   const FiniteElementCollection *fec = a.FESpace()->FEColl();
   // TODO: check for maximum supported orders

   // Batched LOR requires all tensor elements
   if (!UsesTensorBasis(*a.FESpace())) { return false; }

   if (dynamic_cast<const H1_FECollection*>(fec) ||
       dynamic_cast<const DG_FECollection*>(fec))
   {
      return HasIntegrators<DiffusionIntegrator, MassIntegrator>(a);
   }
   else if (dynamic_cast<const ND_FECollection*>(fec))
   {
      return HasIntegrators<CurlCurlIntegrator, VectorFEMassIntegrator>(a);
   }
   else if (dynamic_cast<const RT_FECollection*>(fec))
   {
      return HasIntegrators<DivDivIntegrator, VectorFEMassIntegrator>(a);
   }
   return false;
}

void BatchedLORAssembly::FormLORVertexCoordinates(FiniteElementSpace &fes_ho,
                                                  Vector &X_vert)
{
   Mesh &mesh_ho = *fes_ho.GetMesh();
   mesh_ho.EnsureNodes();

   // Get nodal points at the LOR vertices
   const int dim = mesh_ho.Dimension();
   const int sdim = mesh_ho.SpaceDimension();
   const int nel_ho = mesh_ho.GetNE();
   const int order = fes_ho.GetMaxElementOrder();
   const int nd1d = order + 1;
   const int ndof_per_el = static_cast<int>(pow(nd1d, dim));

   const GridFunction *nodal_gf = mesh_ho.GetNodes();
   const FiniteElementSpace *nodal_fes = nodal_gf->FESpace();
   const Operator *nodal_restriction =
      nodal_fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   // Map from nodal L-vector to E-vector
   Vector nodal_evec(nodal_restriction->Height());
   nodal_restriction->Mult(*nodal_gf, nodal_evec);

   IntegrationRule ir = GetCollocatedIntRule(fes_ho);

   // Map from nodal E-vector to Q-vector at the LOR vertex points
   X_vert.SetSize(sdim*ndof_per_el*nel_ho);
   const QuadratureInterpolator *quad_interp =
      nodal_fes->GetQuadratureInterpolator(ir);
   quad_interp->SetOutputLayout(QVectorLayout::byVDIM);
   quad_interp->Values(nodal_evec, X_vert);
}

// The following two functions (GetMinElt and GetAndIncrementNnzIndex) are
// copied from restriction.cpp. Should they be factored out?

// Return the minimal value found in both my_elts and nbr_elts
static MFEM_HOST_DEVICE int GetMinElt(const int *my_elts, const int n_my_elts,
                                      const int *nbr_elts, const int n_nbr_elts)
{
   int min_el = INT_MAX;
   for (int i = 0; i < n_my_elts; i++)
   {
      const int e_i = my_elts[i];
      if (e_i >= min_el) { continue; }
      for (int j = 0; j < n_nbr_elts; j++)
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

// Returns the index where a non-zero entry should be added and increment the
// number of non-zeros for the row i_L.
static MFEM_HOST_DEVICE int GetAndIncrementNnzIndex(const int i_L, int* I)
{
   int ind = AtomicAdd(I[i_L],1);
   return ind;
}

int BatchedLORAssembly::FillI(SparseMatrix &A) const
{
   static constexpr int Max = 16;

   const int nvdof = fes_ho.GetVSize();

   const int ndof_per_el = fes_ho.GetTypicalFE()->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_mapping.Size()/ndof_per_el;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = fes_ho.GetElementRestriction(ordering);
   const ElementRestriction *el_restr =
      dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != nullptr, "Bad element restriction");

   const Array<int> &el_dof_lex_ = el_restr->GatherMap();
   const Array<int> &dof_glob2loc_ = el_restr->Indices();
   const Array<int> &dof_glob2loc_offsets_ = el_restr->Offsets();

   const auto el_dof_lex = Reshape(el_dof_lex_.Read(), ndof_per_el, nel_ho);
   const auto dof_glob2loc = dof_glob2loc_.Read();
   const auto K = dof_glob2loc_offsets_.Read();
   const auto map = Reshape(sparse_mapping.Read(), nnz_per_row, ndof_per_el);


   auto I = A.WriteI();

   mfem::forall(nvdof + 1, [=] MFEM_HOST_DEVICE (int ii) { I[ii] = 0; });
   mfem::forall(ndof_per_el*nel_ho, [=] MFEM_HOST_DEVICE (int i)
   {
      const int ii_el = i%ndof_per_el;
      const int iel_ho = i/ndof_per_el;
      const int sii = el_dof_lex(ii_el, iel_ho);
      const int ii = (sii >= 0) ? sii : -1 -sii;
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
   });
   // TODO: on device, this is a scan operation
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

void BatchedLORAssembly::FillJAndData(SparseMatrix &A) const
{
   const int nvdof = fes_ho.GetVSize();
   const int ndof_per_el = fes_ho.GetTypicalFE()->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_mapping.Size()/ndof_per_el;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const Operator *op = fes_ho.GetElementRestriction(ordering);
   const ElementRestriction *el_restr =
      dynamic_cast<const ElementRestriction*>(op);
   MFEM_VERIFY(el_restr != nullptr, "Bad element restriction");

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
      mfem::forall(nvdof + 1, [=] MFEM_HOST_DEVICE (int i) { I[i] = I2[i]; });
   }

   static constexpr int Max = 16;

   mfem::forall(ndof_per_el*nel_ho, [=] MFEM_HOST_DEVICE (int i)
   {
      const int ii_el = i%ndof_per_el;
      const int iel_ho = i/ndof_per_el;
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
         const int i_Bi = i_E % ndof_per_el;
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
               const int j_Bj = j_E % ndof_per_el;
               j_B[e_j] = plus ? j_Bj : -1 - j_Bj; // encode with sign
            }
            const int min_e = GetMinElt(i_elts, i_ne, j_elts, j_ne);
            if (iel_ho == min_e) // add the nnz only once
            {
               real_t val = 0.0;
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
                        MFEM_ASSERT_KERNEL(j >= 0, "Can't find nonzero");
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
   });
}

void BatchedLORAssembly::SparseIJToCSR_DG(SparseMatrix &A) const
{
   const int nvdof = fes_ho.GetVSize();
   const int ndof_per_el = fes_ho.GetFE(0)->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_ij.Size()/ndof_per_el/nel_ho;
   const int num_rows = nel_ho*ndof_per_el;
   const int p = fes_ho.GetMaxElementOrder();
   const int nnz = num_rows*nnz_per_row;
   auto I = A.WriteI();

   //mfem::out << "nnz per row " << nnz_per_row << std::endl;

   //mfem::forall(num_rows+ 1, [=] MFEM_HOST_DEVICE (int i)
   //{
   //   I[i] = nnz_per_row*i;
   //});

   //const int nnz = num_rows*nnz_per_row;

   EnsureCapacity(A.GetMemoryJ(), nnz);
   EnsureCapacity(A.GetMemoryData(), nnz);

   const auto V = Reshape(sparse_ij.Read(), nnz_per_row, ndof_per_el, nel_ho);

   auto J = A.WriteJ();
   auto AV = A.WriteData();


   Vector neighbor_info_init(nel_ho*4*3);
   auto neighbor_info_arr = Reshape(neighbor_info_init.Write(),nel_ho, 4, 3);

   int num_faces = fes_ho.GetMesh()->GetNumFaces();
   int global_border_counter = 0;
   for (int f = 0; f<num_faces; ++f)
   {
      Mesh::FaceInformation face = fes_ho.GetMesh()->GetFaceInformation(f);
      int i = face.element[0].index;
      int k = face.element[0].local_face_id;
      if (face.IsBoundary())
      {
         neighbor_info_arr(i,k,0) = -1;
         neighbor_info_arr(i,k,1)= -1;
         neighbor_info_arr(i,k,2)= -1;
         global_border_counter = global_border_counter + (p+1);
      }
      else
      {
         int j = face.element[1].index;
         int l = face.element[1].local_face_id;
         neighbor_info_arr(i,k,0) = j;
         neighbor_info_arr(i,k,1)= face.element[1].orientation;
         neighbor_info_arr(i,k,2)= l;
         neighbor_info_arr(j,l,0) = i;
         neighbor_info_arr(j,l,1) = face.element[0].orientation;
         neighbor_info_arr(j,l,2) = k;
      }
   }
   Vector actual_nnz_per_row(num_rows);
   //auto actual_nnz_per_row =  actual_nnz_per_row_init.Write();
   //Vector I(num_rows+1);
   //auto nnz_so_far =  nnz_so_far_init.Write();
   I[0] = 0;
   for (int i=0; i<num_rows; ++i)
   {
      int loc_border_counter = 0;
      const int iel_ho = i / ndof_per_el;
      const int iloc = i % ndof_per_el;
      const int local_x = iloc % (p+1);
      const int local_y = iloc/(p+1);
      for (int j = 1; j < nnz_per_row; ++j)
      {
         bool boundary = (local_x == 0 && j == 4) || (local_x == p && j == 2) ||
                         (local_y == 0 && j == 1) || (local_y == p && j == 3);
         if (boundary)
         {
            int neighbor_idx = neighbor_info_arr(iel_ho, j-1, 0);
            if (neighbor_idx == -1)
            {
               loc_border_counter = loc_border_counter + 1;
            }
         }
      }
      I[i+1] = I[i] + (nnz_per_row - loc_border_counter);
      mfem::out << "loc_border_counter" << loc_border_counter << std::endl;

      actual_nnz_per_row[i] = nnz_per_row - loc_border_counter;
   }

   mfem::forall(num_rows, [=] MFEM_HOST_DEVICE (int i)
   {
      const int iel_ho = i / ndof_per_el;
      const int iloc = i % ndof_per_el;
      const int local_x = iloc % (p+1);
      const int local_y = iloc/(p+1);
      int nnz_per_current_row = actual_nnz_per_row[i];
      int nnz_so_far_current = I[i];
      AV[nnz_so_far_current] = V(0, iloc, iel_ho);
      J[nnz_so_far_current] = i;
      int k = 1;
      for (int j = 1; j < nnz_per_row; ++j)
      {
         bool boundary = (local_x == 0 && j == 4) || (local_x == p && j == 2) ||
                         (local_y == 0 && j == 1) || (local_y == p && j == 3);
         if (boundary)
         {
            int neighbor_idx = neighbor_info_arr(iel_ho, j-1, 0);
            int neighbor_face = neighbor_info_arr(iel_ho, j-1, 2);
            int neighbor_orientation = neighbor_info_arr(iel_ho, j-1, 1);
            if (neighbor_idx != -1)
            {
               int x_n; int y_n;
               if (j == 4 || j == 2)
               {
                  internal::FaceIdxToVolIdx2D(local_y, p+1, j-1, neighbor_face, 1, x_n, y_n);
               }
               else
               {
                  internal::FaceIdxToVolIdx2D(local_x, p+1, j-1, neighbor_face, 1, x_n, y_n);
               }
               int neighbor_loc_idx = x_n + (p+1)*y_n;
               int neighbor_lor_entry = neighbor_idx*ndof_per_el + neighbor_loc_idx;
               J[nnz_so_far_current + k] = neighbor_lor_entry;
               AV[nnz_so_far_current + k] = V(j, iloc, iel_ho);
               k = k+1;
            }
         }
         else
         {
            if (j == 4) {J[nnz_so_far_current + k] = i - 1;}
            if (j == 2) {J[nnz_so_far_current + k] = i + 1;}
            if (j == 1) {J[nnz_so_far_current + k] = i - (p+1);}
            if (j == 3) {J[nnz_so_far_current + k] = i + (p+1);}
            AV[nnz_so_far_current + k] = V(j, iloc, iel_ho);
            k = k+1;
         }
      }
   });
   I[num_rows] = nnz - global_border_counter;
}

void BatchedLORAssembly::SparseIJToCSR(OperatorHandle &A) const
{
   const int nvdof = fes_ho.GetVSize();

   // If A contains an existing SparseMatrix, reuse it (and try to reuse its
   // I, J, A arrays if they are big enough)
   SparseMatrix *A_mat = A.Is<SparseMatrix>();
   if (!A_mat)
   {
      A_mat = new SparseMatrix;
      A.Reset(A_mat);
   }

   A_mat->OverrideSize(nvdof, nvdof);
   EnsureCapacity(A_mat->GetMemoryI(), nvdof + 1);

   // Assembling the CSR matrix for DG spaces uses a different algorithm
   if (dynamic_cast<const DG_FECollection*>(fes_ho.FEColl()))
   {
      SparseIJToCSR_DG(*A_mat);
   }
   else
   {
      const int nnz = FillI(*A_mat);
      EnsureCapacity(A_mat->GetMemoryJ(), nnz);
      EnsureCapacity(A_mat->GetMemoryData(), nnz);
      FillJAndData(*A_mat);
   }
}

template <int ORDER, int SDIM, typename LOR_KERNEL>
static void Assemble_(LOR_KERNEL &kernel, int dim)
{
   if (dim == 2) { kernel.template Assemble2D<ORDER,SDIM>(); }
   else if (dim == 3) { kernel.template Assemble3D<ORDER>(); }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <int ORDER, typename LOR_KERNEL>
static void Assemble_(LOR_KERNEL &kernel, int dim, int sdim)
{
   if (sdim == 2) { Assemble_<ORDER,2>(kernel, dim); }
   else if (sdim == 3) { Assemble_<ORDER,3>(kernel, dim); }
   else { MFEM_ABORT("Unsupported space dimension."); }
}

template <typename LOR_KERNEL>
static void Assemble_(LOR_KERNEL &kernel, int dim, int sdim, int order)
{
   switch (order)
   {
      case 1: Assemble_<1>(kernel, dim, sdim); break;
      case 2: Assemble_<2>(kernel, dim, sdim); break;
      case 3: Assemble_<3>(kernel, dim, sdim); break;
      case 4: Assemble_<4>(kernel, dim, sdim); break;
      case 5: Assemble_<5>(kernel, dim, sdim); break;
      case 6: Assemble_<6>(kernel, dim, sdim); break;
      case 7: Assemble_<7>(kernel, dim, sdim); break;
      case 8: Assemble_<8>(kernel, dim, sdim); break;
      default: MFEM_ABORT("No kernel order " << order << "!");
   }
}

template <typename LOR_KERNEL>
void BatchedLORAssembly::AssemblyKernel(BilinearForm &a)
{
   LOR_KERNEL kernel(a, fes_ho, X_vert, sparse_ij, sparse_mapping);

   const int dim = fes_ho.GetMesh()->Dimension();
   const int sdim = fes_ho.GetMesh()->SpaceDimension();
   const int order = fes_ho.GetMaxElementOrder();

   Assemble_(kernel, dim, sdim, order);
}

void BatchedLORAssembly::AssembleWithoutBC(BilinearForm &a, OperatorHandle &A)
{
   // Assemble the matrix, depending on what the form is.
   // This fills in the arrays sparse_ij and sparse_mapping.
   const FiniteElementCollection *fec = fes_ho.FEColl();
   if (dynamic_cast<const H1_FECollection*>(fec))
   {
      if (HasIntegrators<DiffusionIntegrator, MassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_H1>(a);
      }
   }
   else if (dynamic_cast<const DG_FECollection*>(fec))
   {
      if (HasIntegrators<DiffusionIntegrator, MassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_DG>(a);
      }
   }
   else if (dynamic_cast<const ND_FECollection*>(fec))
   {
      if (HasIntegrators<CurlCurlIntegrator, VectorFEMassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_ND>(a);
      }
   }
   else if (dynamic_cast<const RT_FECollection*>(fec))
   {
      if (HasIntegrators<DivDivIntegrator, VectorFEMassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_RT>(a);
      }
   }

   return SparseIJToCSR(A);
}

#ifdef MFEM_USE_MPI
void BatchedLORAssembly::ParAssemble(
   BilinearForm &a, const Array<int> &ess_dofs, OperatorHandle &A)
{
   // Assemble the system matrix local to this partition
   OperatorHandle A_local;
   AssembleWithoutBC(a, A_local);

   ParBilinearForm *pa =
      dynamic_cast<ParBilinearForm*>(&a);

   pa->ParallelRAP(*A_local.As<SparseMatrix>(), A, true);

   A.As<HypreParMatrix>()->EliminateBC(ess_dofs,
                                       Operator::DiagonalPolicy::DIAG_ONE);
}
#endif

void BatchedLORAssembly::Assemble(
   BilinearForm &a, const Array<int> ess_dofs, OperatorHandle &A)
{
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParFiniteElementSpace*>(&fes_ho))
   {
      return ParAssemble(a, ess_dofs, A);
   }
#endif

   AssembleWithoutBC(a, A);
   SparseMatrix *A_mat = A.As<SparseMatrix>();

   A_mat->EliminateBC(ess_dofs,
                      Operator::DiagonalPolicy::DIAG_KEEP);
}

BatchedLORAssembly::BatchedLORAssembly(FiniteElementSpace &fes_ho_)
   : fes_ho(fes_ho_)
{
   FormLORVertexCoordinates(fes_ho, X_vert);
}

IntegrationRule GetCollocatedIntRule(FiniteElementSpace &fes)
{
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const Geometry::Type geom = fes.GetMesh()->GetTypicalElementGeometry();
   const int nd1d = fes.GetMaxElementOrder() + 1;
   return irs.Get(geom, 2*nd1d - 3);
}

} // namespace mfem
