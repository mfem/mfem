// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

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
   Point( const int *ind, int attr = -1 );

   /// Return element's type.
   virtual Type GetType() const { return Element::POINT; }

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices( Array<int> &v ) const;

   virtual int * GetVertices () { return indices; }

   virtual int GetNVertices() const { return 1; }

   virtual int GetNEdges() const { return (0); }

   virtual const int *GetEdgeVertices(int ei) const { return NULL; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 0; return 0; }

   virtual int GetNFaces() const { return 0; }

   virtual int GetNFaceVertices(int) const { return 0; }

   virtual const int *GetFaceVertices(int fi) const { return NULL; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Point (indices, attribute); }

   virtual ~Point() { }
};

class PointFiniteElement;
extern PointFiniteElement PointFE;

}

#endif
