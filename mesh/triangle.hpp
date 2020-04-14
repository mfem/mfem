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

   unsigned transform;

public:
   typedef Geometry::Constants<Geometry::TRIANGLE> geom_t;

   Triangle() : Element(Geometry::TRIANGLE) { transform = 0; }

   /// Constructs triangle by specifying the indices and the attribute.
   Triangle(const int *ind, int attr = 1);

   /// Constructs triangle by specifying the indices and the attribute.
   Triangle(int ind1, int ind2, int ind3, int attr = 1);

   /// Return element's type.
   virtual Type GetType() const { return Element::TRIANGLE; }

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   virtual int NeedRefinement(HashTable<Hashed2> &v_to_v) const;

   /// Set the vertices according to the given input.
   virtual void SetVertices(const int *ind);

   /** Reorder the vertices so that the longest edge is from vertex 0
       to vertex 1. If called it should be once from the mesh constructor,
       because the order may be used later for setting the edges. **/
   void MarkEdge(DenseMatrix & pmat);

   static void MarkEdge(int *indices, const DSTable &v_to_v, const int *length);

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(const DSTable &v_to_v, const int *length)
   { MarkEdge(indices, v_to_v, length); }

   virtual void ResetTransform(int tr) { transform = tr; }
   virtual unsigned GetTransform() const { return transform; }

   /// Add 'tr' to the current chain of coarse-fine transformations.
   virtual void PushTransform(int tr)
   { transform = (transform << 3) | (tr + 1); }

   /// Calculate point matrix corresponding to a chain of transformations.
   static void GetPointMatrix(unsigned transform, DenseMatrix &pm);

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 3; }

   virtual int GetNEdges() const { return (3); }

   virtual const int *GetEdgeVertices(int ei) const
   { return geom_t::Edges[ei]; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 0; return 0; }

   virtual int GetNFaces() const { return 0; }

   virtual int GetNFaceVertices(int) const { return 0; }

   virtual const int *GetFaceVertices(int fi) const
   { MFEM_ABORT("not implemented"); return NULL; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Triangle(indices, attribute); }

   virtual ~Triangle() { }
};

// Defined in fe.cpp to ensure construction before 'mfem::Geometries'.
extern Linear2DFiniteElement TriangleFE;

}

#endif
