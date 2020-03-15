// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEMPLATE_FESPACE
#define MFEM_TEMPLATE_FESPACE

#include "../config/tconfig.hpp"
#include "../general/tassign.hpp"
#include "../linalg/ttensor.hpp"
#include "tfe.hpp" // for TFiniteElementSpace_simple
#include "fespace.hpp"

namespace mfem
{

// Templated finite element space classes, cf. fespace.?pp and fe_coll.?pp

// Index types

// IndexType must define:
// - constructor IndexType(const FE &fe, const FiniteElementSpace &fes)
// - copy constructor
// - void SetElement(int elem_idx)
// - int map(int loc_dof_idx, int elem_offset) const --> glob_dof_idx, for
//   single component; elem_offset is relative to the currently set element.

// Index type based on an array listing the dofs for each element where all
// elements are assumed to have the same number of dofs. Such an array is
// constructed from the J array of an element-to-dof Table with optional local
// renumbering to ensure tensor-product local dof ordering when needed.
template <typename FE>
class ElementDofIndexer
{
protected:
   const int *el_dof_list, *loc_dof_list;
   bool own_list;

public:
   typedef FE FE_type;

   ElementDofIndexer(const FE &fe, const FiniteElementSpace &fes)
   {
      const Array<int> *loc_dof_map = fe.GetDofMap();
      // fes.BuildElementToDofTable();
      const Table &el_dof = fes.GetElementToDofTable();
      MFEM_ASSERT(el_dof.Size_of_connections() == el_dof.Size() * FE::dofs,
                  "the element-to-dof Table is not compatible with this FE!");
      int num_dofs = el_dof.Size() * FE::dofs;
      if (!loc_dof_map)
      {
         // no local dof reordering
         el_dof_list = el_dof.GetJ();
         own_list = false;
      }
      else
      {
         // reorder the local dofs according to loc_dof_map
         int *el_dof_list_ = new int[num_dofs];
         const int *loc_dof_map_ = loc_dof_map->GetData();
         for (int i = 0; i < el_dof.Size(); i++)
         {
            MFEM_ASSERT(el_dof.RowSize(i) == FE::dofs,
                        "incompatible element-to-dof Table!");
            for (int j = 0; j < FE::dofs; j++)
            {
               el_dof_list_[j+FE::dofs*i] =
                  el_dof.GetJ()[loc_dof_map_[j]+FE::dofs*i];
            }
         }
         el_dof_list = el_dof_list_;
         own_list = true;
      }
      loc_dof_list = el_dof_list; // point to element 0
   }

   // Shallow copy constructor
   inline MFEM_ALWAYS_INLINE
   ElementDofIndexer(const ElementDofIndexer &orig)
      : el_dof_list(orig.el_dof_list),
        loc_dof_list(orig.loc_dof_list),
        own_list(false)
   { }

   inline MFEM_ALWAYS_INLINE
   ~ElementDofIndexer() { if (own_list) { delete [] el_dof_list; } }

   inline MFEM_ALWAYS_INLINE
   void SetElement(int elem_idx)
   {
      loc_dof_list = el_dof_list + elem_idx * FE::dofs;
   }

   inline MFEM_ALWAYS_INLINE
   int map(int loc_dof_idx, int elem_offset) const
   {
      return loc_dof_list[loc_dof_idx + elem_offset * FE::dofs];
   }
};


// Simple template Finite Element Space, built using an IndexType. For a
// description of the requirements on IndexType, see above.
template <typename FE, typename IndexType>
class TFiniteElementSpace_simple
{
public:
   typedef FE        FE_type;
   typedef IndexType index_type;

protected:
   index_type ind;

public:
   TFiniteElementSpace_simple(const FE &fe, const FiniteElementSpace &fes)
      : ind(fe, fes) { }

   // default copy constructor

   void SetElement(int el) { ind.SetElement(el); }

   // Multi-element Extract:
   // Extract dofs for multiple elements starting with the current element.
   // The number of elements to extract is given by the second dimension of
   // dof_layout_t: dof_layout is (DOFS x NumElems).
   template <AssignOp::Type Op, typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t    &dof_layout,
                dof_data_t            &dof_data) const
   {
      const int NE = dof_layout_t::dim_2;
      MFEM_STATIC_ASSERT(FE::dofs == dof_layout_t::dim_1,
                         "invalid number of dofs");
      for (int j = 0; j < NE; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            Assign<Op>(dof_data[dof_layout.ind(i,j)],
                       glob_dof_data[ind.map(i,j)]);
         }
      }
   }

   template <typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t    &dof_layout,
                dof_data_t            &dof_data) const
   {
      Extract<AssignOp::Set>(glob_dof_data, dof_layout, dof_data);
   }

   // Multi-element assemble.
   template <AssignOp::Type Op,
             typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const dof_layout_t &dof_layout,
                 const dof_data_t   &dof_data,
                 glob_dof_data_t    &glob_dof_data) const
   {
      const int NE = dof_layout_t::dim_2;
      MFEM_STATIC_ASSERT(FE::dofs == dof_layout_t::dim_1,
                         "invalid number of dofs");
      for (int j = 0; j < NE; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            Assign<Op>(glob_dof_data[ind.map(i,j)],
                       dof_data[dof_layout.ind(i,j)]);
         }
      }
   }

   template <typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   inline MFEM_ALWAYS_INLINE
   void Assemble(const dof_layout_t &dof_layout,
                 const dof_data_t   &dof_data,
                 glob_dof_data_t    &glob_dof_data) const
   {
      Assemble<AssignOp::Add>(dof_layout, dof_data, glob_dof_data);
   }

   // Multi-element VectorExtract: vdof_layout is (DOFS x NumComp x NumElems).
   template <AssignOp::Type Op,
             typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void VectorExtract(const vec_layout_t     &vl,
                      const glob_vdof_data_t &glob_vdof_data,
                      const vdof_layout_t    &vdof_layout,
                      vdof_data_t            &vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(NC == vl.NumComponents(), "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<Op>(vdof_data[vdof_layout.ind(i,k,j)],
                          glob_vdof_data[vl.ind(ind.map(i,j), k)]);
            }
         }
      }
   }

   template <typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void VectorExtract(const vec_layout_t     &vl,
                      const glob_vdof_data_t &glob_vdof_data,
                      const vdof_layout_t    &vdof_layout,
                      vdof_data_t            &vdof_data) const
   {
      VectorExtract<AssignOp::Set>(vl, glob_vdof_data, vdof_layout, vdof_data);
   }

   // Multi-element VectorAssemble: vdof_layout is (DOFS x NumComp x NumElems).
   template <AssignOp::Type Op,
             typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void VectorAssemble(const vdof_layout_t &vdof_layout,
                       const vdof_data_t   &vdof_data,
                       const vec_layout_t  &vl,
                       glob_vdof_data_t    &glob_vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(NC == vl.NumComponents(), "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<Op>(glob_vdof_data[vl.ind(ind.map(i,j), k)],
                          vdof_data[vdof_layout.ind(i,k,j)]);
            }
         }
      }
   }

   template <typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void VectorAssemble(const vdof_layout_t &vdof_layout,
                       const vdof_data_t   &vdof_data,
                       const vec_layout_t  &vl,
                       glob_vdof_data_t    &glob_vdof_data) const
   {
      VectorAssemble<AssignOp::Add>(vdof_layout, vdof_data, vl, glob_vdof_data);
   }

   // Extract a static number of consecutive components; vdof_layout is
   // (dofs x NC x NE), where NC is the number of components to extract. It is
   // assumed that: first_comp + NC <= vl.NumComponents().
   template <typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void ExtractComponents(int                     first_comp,
                          const vec_layout_t     &vl,
                          const glob_vdof_data_t &glob_vdof_data,
                          const vdof_layout_t    &vdof_layout,
                          vdof_data_t            &vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(first_comp + NC <= vl.NumComponents(),
                  "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<AssignOp::Set>(
                  vdof_data[vdof_layout.ind(i,k,j)],
                  glob_vdof_data[vl.ind(ind.map(i,j), first_comp+k)]);
            }
         }
      }
   }

   // Assemble a static number of consecutive components; vdof_layout is
   // (dofs x NC x NE), where NC is the number of components to add. It is
   // assumed that: first_comp + NC <= vl.NumComponents().
   template <typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   inline MFEM_ALWAYS_INLINE
   void AssembleComponents(int                  first_comp,
                           const vdof_layout_t &vdof_layout,
                           const vdof_data_t   &vdof_data,
                           const vec_layout_t  &vl,
                           glob_vdof_data_t    &glob_vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(first_comp + NC <= vl.NumComponents(),
                  "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<AssignOp::Add>(
                  glob_vdof_data[vl.ind(ind.map(i,j), first_comp+k)],
                  vdof_data[vdof_layout.ind(i,k,j)]);
            }
         }
      }
   }

   void Assemble(const TMatrix<FE::dofs,FE::dofs,double> &m,
                 SparseMatrix &M) const
   {
      MFEM_FLOPS_ADD(FE::dofs*FE::dofs);
      for (int i = 0; i < FE::dofs; i++)
      {
         M.SetColPtr(ind.map(i,0));
         for (int j = 0; j < FE::dofs; j++)
         {
            M._Add_(ind.map(j,0), m(i,j));
         }
         M.ClearColPtr();
      }
   }

   template <typename vec_layout_t>
   void AssembleBlock(int block_i, int block_j, const vec_layout_t &vl,
                      const TMatrix<FE::dofs,FE::dofs,double> &m,
                      SparseMatrix &M) const
   {
      MFEM_FLOPS_ADD(FE::dofs*FE::dofs);
      for (int i = 0; i < FE::dofs; i++)
      {
         M.SetColPtr(vl.ind(ind.map(i,0), block_i));
         for (int j = 0; j < FE::dofs; j++)
         {
            M._Add_(vl.ind(ind.map(j,0), block_j), m(i,j));
         }
         M.ClearColPtr();
      }
   }
};

// H1 Finite Element Space

template <typename FE>
class H1_FiniteElementSpace
   : public TFiniteElementSpace_simple<FE,ElementDofIndexer<FE> >
{
public:
   typedef FE FE_type;
   typedef TFiniteElementSpace_simple<FE,ElementDofIndexer<FE> > base_class;

   H1_FiniteElementSpace(const FE &fe, const FiniteElementSpace &fes)
      : base_class(fe, fes)
   { }

   // default copy constructor

   static bool Matches(const FiniteElementSpace &fes)
   {
      const FiniteElementCollection *fec = fes.FEColl();
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(fec);
      if (!h1_fec) { return false; }
      const FiniteElement *fe = h1_fec->FiniteElementForGeometry(FE_type::geom);
      if (fe->GetOrder() != FE_type::degree) { return false; }
      return true;
   }

   template <typename vec_layout_t>
   static bool VectorMatches(const FiniteElementSpace &fes)
   {
      return Matches(fes) && vec_layout_t::Matches(fes);
   }
};


// Simple index type for DG spaces, where the map method is given by:
// glob_dof_idx = loc_dof_idx + elem_idx * num_dofs.
template <typename FE>
class DGIndexer
{
protected:
   int offset;

public:
   typedef FE FE_type;

   DGIndexer(const FE &fe, const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetNDofs() == fes.GetNE() * FE::dofs,
                  "the FE space is not compatible with this FE!");
      offset = 0;
   }

   // default copy constructor

   inline MFEM_ALWAYS_INLINE
   void SetElement(int elem_idx)
   {
      offset = FE::dofs * elem_idx;
   }

   inline MFEM_ALWAYS_INLINE
   int map(int loc_dof_idx, int elem_offset) const
   {
      return offset + loc_dof_idx + elem_offset * FE::dofs;
   }
};


// L2 Finite Element Space

template <typename FE>
class L2_FiniteElementSpace
   : public TFiniteElementSpace_simple<FE,DGIndexer<FE> >
{
public:
   typedef FE FE_type;
   typedef TFiniteElementSpace_simple<FE,DGIndexer<FE> > base_class;

   L2_FiniteElementSpace(const FE &fe, const FiniteElementSpace &fes)
      : base_class(fe, fes)
   { }

   // default copy constructor

   static bool Matches(const FiniteElementSpace &fes)
   {
      const FiniteElementCollection *fec = fes.FEColl();
      const L2_FECollection *l2_fec =
         dynamic_cast<const L2_FECollection *>(fec);
      if (!l2_fec) { return false; }
      const FiniteElement *fe = l2_fec->FiniteElementForGeometry(FE_type::geom);
      if (fe->GetOrder() != FE_type::degree) { return false; }
      return true;
   }

   template <typename vec_layout_t>
   static bool VectorMatches(const FiniteElementSpace &fes)
   {
      return Matches(fes) && vec_layout_t::Matches(fes);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FESPACE
