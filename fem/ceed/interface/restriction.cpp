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

#include "../../../fem/gridfunc.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

static void InitNativeRestr(const mfem::FiniteElementSpace &fes,
                            Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetTypicalFE();
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   const mfem::Array<int>& dof_map = tfe->GetDofMap();

   for (int i = 0; i < fes.GetNE(); i++)
   {
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j+el_offset] = stride*el_dof.GetJ()[dof_map[j]+el_offset];
      }
   }

   CeedElemRestrictionCreate(ceed, fes.GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitLexicoRestr(const mfem::FiniteElementSpace &fes,
                            Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetTypicalFE();
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   const mfem::Table &el_dof = fes.GetElementToDofTable();
   mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
   const int stride = compstride == 1 ? fes.GetVDim() : 1;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      for (int i = 0; i < P; i++)
      {
         tp_el_dof[i + e*P] = stride*el_dof.GetJ()[i + e*P];
      }
   }

   CeedElemRestrictionCreate(ceed, fes.GetNE(), P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitRestrictionImpl(const mfem::FiniteElementSpace &fes,
                                Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetTypicalFE();
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   if ( tfe && tfe->GetDofMap().Size()>0 ) // Native ordering using dof_map
   {
      InitNativeRestr(fes, ceed, restr);
   }
   else  // Lexicographic ordering
   {
      InitLexicoRestr(fes, ceed, restr);
   }
}

static void InitNativeRestrWithIndices(
   const mfem::FiniteElementSpace &fes,
   int nelem,
   const int* indices,
   Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   mfem::Array<int> tp_el_dof(nelem*P);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   Array<int> dofs;
   const int stride = compstride == 1 ? fes.GetVDim() : 1;
   const mfem::Array<int>& dof_map = tfe->GetDofMap();

   for (int i = 0; i < nelem; i++)
   {
      const int elem_index = indices[i];
      fes.GetElementDofs(elem_index, dofs);
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j + el_offset] = stride*dofs[dof_map[j]];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitLexicoRestrWithIndices(
   const mfem::FiniteElementSpace &fes,
   int nelem,
   const int* indices,
   Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   CeedInt compstride = fes.GetOrdering()==Ordering::byVDIM ? 1 : fes.GetNDofs();
   mfem::Array<int> tp_el_dof(nelem*P);
   Array<int> dofs;
   const int stride = compstride == 1 ? fes.GetVDim() : 1;

   for (int i = 0; i < nelem; i++)
   {
      const int elem_index = indices[i];
      fes.GetElementDofs(elem_index, dofs);
      const int el_offset = P * i;
      for (int j = 0; j < P; j++)
      {
         tp_el_dof[j + el_offset] = stride*dofs[j];
      }
   }

   CeedElemRestrictionCreate(ceed, nelem, P, fes.GetVDim(),
                             compstride, (fes.GetVDim())*(fes.GetNDofs()),
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

static void InitRestrictionWithIndicesImpl(
   const mfem::FiniteElementSpace &fes,
   int nelem,
   const int* indices,
   Ceed ceed, CeedElemRestriction *restr)
{
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const mfem::TensorBasisElement * tfe =
      dynamic_cast<const mfem::TensorBasisElement *>(fe);
   if ( tfe && tfe->GetDofMap().Size()>0 ) // Native ordering using dof_map
   {
      InitNativeRestrWithIndices(fes, nelem, indices, ceed, restr);
   }
   else  // Lexicographic ordering
   {
      InitLexicoRestrWithIndices(fes, nelem, indices, ceed, restr);
   }
}

static void InitCoeffRestrictionWithIndicesImpl(
   const mfem::FiniteElementSpace &fes,
   int nelem,
   const int* indices,
   int nquads,
   int ncomp,
   Ceed ceed,
   CeedElemRestriction *restr)
{
   mfem::Array<int> tp_el_dof(nelem*nquads);
   const int stride_quad = ncomp;
   const int stride_elem = ncomp*nquads;
   // TODO generalize to support different #quads
   for (int i = 0; i < nelem; i++)
   {
      const int elem_index = indices[i];
      const int el_offset = elem_index * stride_elem;
      for (int j = 0; j < nquads; j++)
      {
         tp_el_dof[j + nquads * i] = j * stride_quad + el_offset;
      }
   }
   CeedElemRestrictionCreate(ceed, nelem, nquads, ncomp, 1,
                             ncomp*fes.GetNE()*nquads,
                             CEED_MEM_HOST, CEED_COPY_VALUES,
                             tp_el_dof.GetData(), restr);
}

void InitStridedRestriction(const mfem::FiniteElementSpace &fes,
                            CeedInt nelem, CeedInt nqpts, CeedInt qdatasize,
                            const CeedInt *strides,
                            CeedElemRestriction *restr)
{
   RestrKey restr_key(&fes, nelem, nqpts, qdatasize, restr_type::Strided);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      CeedElemRestrictionCreateStrided(mfem::internal::ceed, nelem, nqpts, qdatasize,
                                       nelem*nqpts*qdatasize,
                                       strides,
                                       restr);
      // Will be automatically destroyed when @a fes gets destroyed.
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestriction(const FiniteElementSpace &fes,
                     Ceed ceed,
                     CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const mfem::FiniteElement *fe = fes.GetTypicalFE();
   const int P = fe->GetDof();
   const int nelem = fes.GetNE();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, restr_type::Standard);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionImpl(fes, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitRestrictionWithIndices(const FiniteElementSpace &fes,
                                int nelem,
                                const int* indices,
                                Ceed ceed,
                                CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   const mfem::FiniteElement *fe = fes.GetFE(indices[0]);
   const int P = fe->GetDof();
   const int ncomp = fes.GetVDim();
   RestrKey restr_key(&fes, nelem, P, ncomp, restr_type::Standard);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitRestrictionWithIndicesImpl(fes, nelem, indices, ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

void InitCoeffRestrictionWithIndices(const FiniteElementSpace &fes,
                                     int nelem,
                                     const int* indices,
                                     int nquads,
                                     int ncomp,
                                     Ceed ceed,
                                     CeedElemRestriction *restr)
{
   // Check for FES -> basis, restriction in hash tables
   RestrKey restr_key(&fes, nelem, nquads, ncomp, restr_type::Coeff);
   auto restr_itr = mfem::internal::ceed_restr_map.find(restr_key);

   // Init or retrieve key values
   if (restr_itr == mfem::internal::ceed_restr_map.end())
   {
      InitCoeffRestrictionWithIndicesImpl(fes, nelem, indices, nquads, ncomp,
                                          ceed, restr);
      mfem::internal::ceed_restr_map[restr_key] = *restr;
   }
   else
   {
      *restr = restr_itr->second;
   }
}

#endif

} // namespace ceed

} // namespace mfem
