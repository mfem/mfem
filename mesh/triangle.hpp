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

#ifndef MFEM_TRIANGLE
#define MFEM_TRIANGLE

#include "../config/config.hpp"
#include "../fem/fe.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type triangle element
class Triangle : public Element
{
protected:
   int indices[3];
   static const int edges[3][2];

public:

   Triangle() : Element(Geometry::TRIANGLE) { }

   /// Constructs triangle by specifying the indices and the attribute.
   Triangle(const int *ind, int attr = 1);

   /// Constructs triangle by specifying the indices and the attribute.
   Triangle(int ind1, int ind2, int ind3, int attr = 1);

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   int NeedRefinement(DSTable &v_to_v, int *middle) const;

   /// Set the vertices according to the given input.
   virtual void SetVertices(const int *ind);

   /** Reorder the vertices so that the longest edge is from vertex 0
       to vertex 0. If called it should be once from the mesh constructor,
       because the order may be used later for setting the edges. **/
   virtual void MarkEdge(DenseMatrix & pmat);

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(const DSTable &v_to_v, const int *length);

   /// Return element's type.
   virtual int GetType() const { return Element::TRIANGLE; }

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 3; }

   virtual int GetNEdges() const { return (3); }

   virtual const int *GetEdgeVertices(int ei) const { return (edges[ei]); }

   virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 0; return 0; }

   virtual const int *GetFaceVertices(int fi) const
   { MFEM_ABORT("not implemented"); return NULL; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Triangle(indices, attribute); }

   virtual ~Triangle() { }
};

extern Linear2DFiniteElement TriangleFE;

}

#endif
