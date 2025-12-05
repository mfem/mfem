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

#ifndef MFEM_POINT
#define MFEM_POINT

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type point element
class Point : public Element
{
protected:
   int indices[1];

public:
   typedef Geometry::Constants<Geometry::POINT> geom_t;

   Point() : Element(Geometry::POINT) {}

   /// Constructs point by specifying the indices and the attribute.
   Point(const int *ind, int attr = 1);

   /// Return element's type.
   Type GetType() const override { return Element::POINT; }

   /// Get the indices defining the vertices
   void GetVertices(Array<int> &v) const override;

   /// Set the indices defining the vertices
   void SetVertices(const Array<int> &v) override;

   /// @note The returned array should NOT be deleted by the caller.
   int *GetVertices() override { return indices; }

   /// Set the vertices according to the given input.
   void SetVertices(const int *ind) override;

   int GetNVertices() const override { return 1; }

   int GetNEdges() const override { return (0); }

   const int *GetEdgeVertices(int ei) const override { return NULL; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED int GetNFaces(int &nFaceVertices) const override
   { nFaceVertices = 0; return 0; }

   int GetNFaces() const override { return 0; }

   int GetNFaceVertices(int) const override { return 0; }

   const int *GetFaceVertices(int fi) const override { return NULL; }

   Element *Duplicate(Mesh *m) const override
   { return new Point(indices, attribute); }

   virtual ~Point() = default;
};

class PointFiniteElement;
extern MFEM_EXPORT PointFiniteElement PointFE;

}

#endif
