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

#ifndef MFEM_TEMPLATE_MESH
#define MFEM_TEMPLATE_MESH

#include "../config/tconfig.hpp"
#include "../linalg/tlayout.hpp"

#include "mesh.hpp"

namespace mfem
{

template <typename FESpace,
          typename nodeLayout =
          VectorLayout<Ordering::byNODES,FESpace::FE_type::dim> >
class TMesh
{
public:
   typedef FESpace                   FESpace_type;
   typedef typename FESpace::FE_type FE_type;
   typedef nodeLayout                nodeLayout_type;

   static const int dim       = FE_type::dim;
   static const int space_dim = nodeLayout::vec_dim;

   const Mesh &m_mesh;
   const FiniteElementSpace &fes;
   const GridFunction &Nodes;

   FE_type fe;
   FESpace t_fes;

   nodeLayout node_layout;

public:
   TMesh(const Mesh &mesh)
      : m_mesh(mesh), fes(*mesh.GetNodalFESpace()), Nodes(*mesh.GetNodes()),
        fe(*fes.FEColl()), t_fes(fe, fes), node_layout(fes)
   {
      MFEM_STATIC_ASSERT(space_dim != 0, "dynamic space dim is not allowed");
   }

   int GetNE() const { return m_mesh.GetNE(); }

   static bool MatchesGeometry(const Mesh &mesh)
   {
      if (mesh.Dimension() != dim) { return false; }
      if (mesh.SpaceDimension() != space_dim) { return false; }
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (mesh.GetElementBaseGeometry(i) != FE_type::geom) { return false; }
      }
      return true;
   }

   static bool MatchesNodes(const Mesh &mesh)
   {
      if (!mesh.GetNodes()) { return false; }
      return FESpace::template VectorMatches<nodeLayout>(
         *mesh.GetNodalFESpace());
   }

   static bool Matches(const Mesh &mesh)
   {
      return MatchesGeometry(mesh) && MatchesNodes(mesh);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_MESH
