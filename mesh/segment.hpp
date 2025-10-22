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

#ifndef MFEM_SEGMENT
#define MFEM_SEGMENT

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type line segment element
class Segment : public Element
{
protected:
   int indices[2];

public:
   typedef Geometry::Constants<Geometry::SEGMENT> geom_t;

   Segment() : Element(Geometry::SEGMENT) {}

   /// Constructs triangle by specifying the indices and the attribute.
   Segment(const int *ind, int attr = 1);

   /// Constructs triangle by specifying the indices and the attribute.
   Segment(int ind1, int ind2, int attr = 1);

   /// Return element's type.
   Type GetType() const override { return Element::SEGMENT; }

   /// Get the indices defining the vertices.
   void GetVertices(Array<int> &v) const override;

   /// Set the indices defining the vertices.
   void SetVertices(const Array<int> &v) override;

   /// @note The returned array should NOT be deleted by the caller.
   int * GetVertices () override { return indices; }

   /// Set the indices defining the vertices.
   void SetVertices(const int *ind) override;

   int GetNVertices() const override { return 2; }

   int GetNEdges() const override { return 0; }

   const int *GetEdgeVertices(int ei) const override { return NULL; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED int GetNFaces(int &nFaceVertices) const override
   { nFaceVertices = 0; return 0; }

   int GetNFaces() const override { return 0; }

   int GetNFaceVertices(int) const override { return 0; }

   const int *GetFaceVertices(int fi) const override { return NULL; }

   Element *Duplicate(Mesh *m) const override
   { return new Segment(indices, attribute); }

   virtual ~Segment() = default;
};

class Linear1DFiniteElement;
extern MFEM_EXPORT Linear1DFiniteElement SegmentFE;

}

#endif
