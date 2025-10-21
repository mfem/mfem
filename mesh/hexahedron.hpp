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

#ifndef MFEM_HEXAHEDRON
#define MFEM_HEXAHEDRON

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type hexahedron element
class Hexahedron : public Element
{
protected:
   int indices[8];

public:
   typedef Geometry::Constants<Geometry::CUBE> geom_t;

   Hexahedron() : Element(Geometry::CUBE) { }

   /// Constructs hexahedron by specifying the indices and the attribute.
   Hexahedron(const int *ind, int attr = 1);

   /// Constructs hexahedron by specifying the indices and the attribute.
   Hexahedron(int ind1, int ind2, int ind3, int ind4,
              int ind5, int ind6, int ind7, int ind8, int attr = 1);

   /// Return element's type
   Type GetType() const override { return Element::HEXAHEDRON; }

   /// Get the indices defining the vertices.
   void GetVertices(Array<int> &v) const override;

   /// Set the indices defining the vertices.
   void SetVertices(const Array<int> &v) override;

   /// @note The returned array should NOT be deleted by the caller.
   int * GetVertices () override { return indices; }

   /// Set the indices defining the vertices.
   void SetVertices(const int *ind) override;

   int GetNVertices() const override { return 8; }

   int GetNEdges() const override { return 12; }

   const int *GetEdgeVertices(int ei) const override
   { return geom_t::Edges[ei]; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED int GetNFaces(int &nFaceVertices) const override
   { nFaceVertices = 4; return 6; }

   int GetNFaces() const override { return 6; }

   int GetNFaceVertices(int) const override { return 4; }

   const int *GetFaceVertices(int fi) const override
   { return geom_t::FaceVert[fi]; }

   Element *Duplicate(Mesh *m) const override
   { return new Hexahedron(indices, attribute); }

   virtual ~Hexahedron() = default;
};

extern MFEM_EXPORT class TriLinear3DFiniteElement HexahedronFE;

}

#endif
