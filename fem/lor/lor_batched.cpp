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

   const bool dg = fes_ho.IsDGSpace();

   // Get nodal points at the LOR vertices
   const int dim = mesh_ho.Dimension();
   const int sdim = mesh_ho.SpaceDimension();
   const int nel_ho = mesh_ho.GetNE();
   const int order = fes_ho.GetMaxElementOrder();
   const int nd1d = dg ? order + 2 : order + 1;
   const int ndof_per_el = static_cast<int>(pow(nd1d, dim));

   const GridFunction *nodal_gf = mesh_ho.GetNodes();
   const FiniteElementSpace *nodal_fes = nodal_gf->FESpace();
   const Operator *nodal_restriction =
      nodal_fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   // Map from nodal L-vector to E-vector
   Vector nodal_evec(nodal_restriction->Height());
   nodal_restriction->Mult(*nodal_gf, nodal_evec);

   const IntegrationRule ir = GetLobattoIntRule(
                                 mesh_ho.GetTypicalElementGeometry(), nd1d);

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

void BatchedLORAssembly::SparseIJToCSR_DG(OperatorHandle &A) const
{
   const int ndof_per_el = fes_ho.GetFE(0)->GetDof();
   const int nel_ho = fes_ho.GetNE();
   const int nnz_per_row = sparse_ij.Size()/ndof_per_el/nel_ho;
   const int dim = fes_ho.GetMesh()->Dimension();
   const int nrows = nel_ho*ndof_per_el;
   const int p = fes_ho.GetMaxElementOrder();
   const int pp1 = p + 1;
   const int nnz = nrows*nnz_per_row;

   const int face_nbr_vsize = [&]()
   {
#ifdef MFEM_USE_MPI
      if (auto *par_fes = dynamic_cast<ParFiniteElementSpace*>(&fes_ho))
      {
         return par_fes->GetFaceNbrVSize();
      }
#endif
      return 0;
   }();

   // If A contains an existing SparseMatrix, reuse it (and try to reuse its
   // I, J, A arrays if they are big enough)
   SparseMatrix *A_mat = A.Is<SparseMatrix>();
   if (!A_mat)
   {
      A_mat = new SparseMatrix;
      A.Reset(A_mat);
   }

   // The second argument (nrows + face_nbr_vsize) accounts for additional
   // columns contributed by DG face neighbors in parallel finite element
   // spaces. In serial, face_nbr_vsize is set to 0.
   A_mat->OverrideSize(nrows, nrows + face_nbr_vsize);

   EnsureCapacity(A_mat->GetMemoryI(), nrows + 1);
   EnsureCapacity(A_mat->GetMemoryJ(), nnz);
   EnsureCapacity(A_mat->GetMemoryData(), nnz);

   Array<int> nbr_info(nel_ho*3*2*dim);
   auto h_nbr_info = Reshape(nbr_info.HostWrite(), nel_ho, 2*dim, 3);
   const int num_faces = fes_ho.GetMesh()->GetNumFaces();
   for (int f = 0; f < num_faces; f++)
   {
      Mesh::FaceInformation finfo = fes_ho.GetMesh()->GetFaceInformation(f);
      int e0 = finfo.element[0].index;
      int f0 = finfo.element[0].local_face_id;
      if (finfo.IsBoundary())
      {
         h_nbr_info(e0,f0,0) = -1;
         h_nbr_info(e0,f0,1)= -1;
         h_nbr_info(e0,f0,2)= -1;
      }
      else if (finfo.IsShared())
      {
         // Face neighbors elements are indexed after the last local element
         h_nbr_info(e0,f0,0) = nel_ho + finfo.element[1].index;
         h_nbr_info(e0,f0,1)= finfo.element[1].orientation;
         h_nbr_info(e0,f0,2)= finfo.element[1].local_face_id;
      }
      else if (finfo.IsInterior())
      {
         int e1 = finfo.element[1].index;
         int f1 = finfo.element[1].local_face_id;
         h_nbr_info(e0,f0,0) = e1;
         h_nbr_info(e0,f0,1)= finfo.element[1].orientation;
         h_nbr_info(e0,f0,2)= f1;
         h_nbr_info(e1,f1,0) = e0;
         h_nbr_info(e1,f1,1) = finfo.element[1].orientation;
         h_nbr_info(e1,f1,2) = f0;
      }
   };

   auto h_I = A_mat->HostWriteI();
   h_I[0] = 0;
   for (int i = 0; i < nrows; ++i)
   {
      const int iel_ho = i / ndof_per_el;
      const int iloc = i % ndof_per_el;
      static const int lex_map_2[4] = {3, 1, 0, 2};
      static const int lex_map_3[6] = {4, 2, 1, 3, 0, 5};
      const int local_i[3] = {iloc % pp1, (iloc/pp1)%pp1, iloc/pp1/pp1};
      int bdr_count = 0;
      for (int n_idx = 0; n_idx < dim; ++n_idx)
      {
         for (int e_i = 0; e_i < 2; ++e_i)
         {
            const int j_lex = e_i + n_idx*2;
            const int f = (dim == 3) ? lex_map_3[j_lex]:lex_map_2[j_lex];
            const bool boundary = (local_i[n_idx] == e_i * p);
            if (boundary)
            {
               int neighbor_idx = h_nbr_info(iel_ho, f, 0);
               if (neighbor_idx == -1)
               {
                  ++bdr_count;
               }
            }
         }
      }
      h_I[i+1] = h_I[i] + (nnz_per_row - bdr_count);
   }

   const auto V = Reshape(sparse_ij.Read(), nnz_per_row, ndof_per_el, nel_ho);
   auto J = A_mat->WriteJ();
   auto AV = A_mat->WriteData();
   auto I = A_mat->ReadI();

   auto d_nbr_info = Reshape(nbr_info.Read(), nel_ho, 2*dim, 3);
   mfem::forall(nrows, [=] MFEM_HOST_DEVICE (int i)
   {
      const int e = i / ndof_per_el;
      const int iloc = i % ndof_per_el;
      const int local_x = iloc % pp1;
      const int local_y = (iloc/pp1)%pp1;
      const int local_z = iloc/pp1/pp1;
      const int local_i[3] = {local_x, local_y, local_z};
      int offset = I[i];
      static const int lex_map_2[4] = {3, 1, 0, 2};
      static const int lex_map_3[6] = {4,2,1,3,0,5};
      const int *lex_map = (dim == 2) ? lex_map_2 : lex_map_3;
      AV[offset] = V(0, iloc, e);
      J[offset] = i;
      ++offset;
      for (int n_idx = 0; n_idx < dim; ++n_idx)
      {
         // qi is the face lexicographic index, obtained by taking the
         // lexicographic index of the coordinates ommiting n_idx.
         int qi = 0;
         int stride = 1;
         for (int d = 0; d < dim; ++d)
         {
            if (d != n_idx)
            {
               qi += local_i[d]*stride;
               stride *= pp1;
            }
         }
         for (int e_i = 0; e_i < 2; ++e_i)
         {
            const int j_lex = e_i + n_idx*2;
            const int f = lex_map[j_lex];
            const bool bdr = (local_i[n_idx] == e_i * p);
            if (bdr)
            {
               const int nbr_e = d_nbr_info(e, f, 0);
               const int nbr_ori = d_nbr_info(e, f, 1);
               const int nbr_f = d_nbr_info(e, f, 2);
               if (nbr_e != -1)
               {
                  const int nbr_loc_idx = internal::FaceIdxToVolIdx(
                                             dim, qi, pp1, f, nbr_f, 1, nbr_ori);
                  J[offset] = nbr_e*ndof_per_el + nbr_loc_idx;
                  AV[offset] = V(f+1, iloc, e);
                  ++offset;
               }
            }
            else
            {
               int shift = (e_i == 0) ? -1 : 1;
               for (int n = 0; n < n_idx; ++n) { shift *= pp1; }
               J[offset] = i + shift;
               AV[offset] = V(f+1, iloc, e);
               ++offset;
            }
         }
      }
   });
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

   const int nnz = FillI(*A_mat);
   EnsureCapacity(A_mat->GetMemoryJ(), nnz);
   EnsureCapacity(A_mat->GetMemoryData(), nnz);
   FillJAndData(*A_mat);
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

   // Handle DG case separately, because assembly of CSR matrix requires
   // handling face terms.
   if (dynamic_cast<const DG_FECollection*>(fec))
   {
      if (HasIntegrators<DiffusionIntegrator, MassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_DG>(a);
      }
      SparseIJToCSR_DG(A);
      return;
   }

   if (dynamic_cast<const H1_FECollection*>(fec))
   {
      if (HasIntegrators<DiffusionIntegrator, MassIntegrator>(a))
      {
         AssemblyKernel<BatchedLOR_H1>(a);
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

   SparseIJToCSR(A);
}

#ifdef MFEM_USE_MPI
void BatchedLORAssembly::ParAssemble_DG(SparseMatrix &A_local,
                                        OperatorHandle &A)
{
   auto &par_fes = static_cast<ParFiniteElementSpace&>(fes_ho);

   // handle the case when 'a' contains off-diagonal
   const int lvsize = par_fes.GetVSize();
   const Array<HYPRE_BigInt> &face_nbr_glob_ldof =
      par_fes.GetFaceNbrGlobalDofMapArray();
   const HYPRE_BigInt ldof_offset = par_fes.GetMyDofOffset();

   const int nnz_local = A_local.NumNonZeroElems();
   Array<HYPRE_BigInt> glob_J(nnz_local);

   const HYPRE_BigInt *d_face_nbr_glob_ldof = face_nbr_glob_ldof.Read();
   const int *d_J = A_local.ReadJ();
   HYPRE_BigInt *d_glob_J = glob_J.Write();

   mfem::forall(nnz_local, [=] MFEM_HOST_DEVICE (int i)
   {
      if (d_J[i] < lvsize)
      {
         d_glob_J[i] = d_J[i] + ldof_offset;
      }
      else
      {
         d_glob_J[i] = d_face_nbr_glob_ldof[d_J[i] - lvsize];
      }
   });

   A.Reset(new HypreParMatrix(
              par_fes.GetComm(), lvsize, par_fes.GlobalVSize(),
              par_fes.GlobalVSize(), A_local.HostReadWriteI(),
              glob_J.HostReadWrite(), A_local.HostReadWriteData(),
              par_fes.GetDofOffsets(), par_fes.GetDofOffsets()));
}

void BatchedLORAssembly::ParAssemble(
   BilinearForm &a, const Array<int> &ess_dofs, OperatorHandle &A)
{
   // Assemble the system matrix local to this partition
   OperatorHandle A_local;
   AssembleWithoutBC(a, A_local);

   if (dynamic_cast<const DG_FECollection*>(fes_ho.FEColl()))
   {
      ParAssemble_DG(*A_local.As<SparseMatrix>(), A);
   }
   else
   {
      ParBilinearForm *pa =
         dynamic_cast<ParBilinearForm*>(&a);
      pa->ParallelRAP(*A_local.As<SparseMatrix>(), A, true);
      A.As<HypreParMatrix>()->EliminateBC(ess_dofs,
                                          Operator::DiagonalPolicy::DIAG_ONE);
   }
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

   const SparseMatrix *P = fes_ho.GetConformingProlongation();
   if (P)
   {
      std::unique_ptr<SparseMatrix> R(Transpose(*P));
      std::unique_ptr<SparseMatrix> RA(mfem::Mult(*R, *A.As<SparseMatrix>()));
      A.Reset(mfem::Mult(*RA, *P));
   }

   A.As<SparseMatrix>()->EliminateBC(ess_dofs,
                                     Operator::DiagonalPolicy::DIAG_KEEP);
}

BatchedLORAssembly::BatchedLORAssembly(FiniteElementSpace &fes_ho_)
   : fes_ho(fes_ho_)
{
   FormLORVertexCoordinates(fes_ho, X_vert);
}

IntegrationRule GetLobattoIntRule(Geometry::Type geom, int nd1d)
{
   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   return irs.Get(geom, 2*nd1d - 3);
}

IntegrationRule GetCollocatedIntRule(FiniteElementSpace &fes)
{
   const Geometry::Type geom = fes.GetMesh()->GetTypicalElementGeometry();
   return GetLobattoIntRule(geom, fes.GetMaxElementOrder() + 1);
}

IntegrationRule GetCollocatedFaceIntRule(FiniteElementSpace &fes)
{
   const Geometry::Type geom = fes.GetMesh()->GetTypicalFaceGeometry();
   return GetLobattoIntRule(geom, fes.GetMaxElementOrder() + 1);
}

} // namespace mfem
