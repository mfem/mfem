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

   /// Set the indices the element according to the input.
   virtual void SetVertices(const int *ind);

   /// Return element's type.
   virtual Type GetType() const { return Element::SEGMENT; }

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 2; }

   virtual int GetNEdges() const { return (0); }

   virtual const int *GetEdgeVertices(int ei) const { return NULL; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 0; return 0; }

   virtual int GetNFaces() const { return 0; }

   virtual int GetNFaceVertices(int) const { return 0; }

   virtual const int *GetFaceVertices(int fi) const { return NULL; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Segment(indices, attribute); }

   virtual ~Segment() { }
};

class Linear1DFiniteElement;
extern Linear1DFiniteElement SegmentFE;

}

#endif
