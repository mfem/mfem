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
   Type GetType() const override { return Element::TRIANGLE; }

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   int NeedRefinement(HashTable<Hashed2> &v_to_v) const override;

   /** Reorder the vertices so that the longest edge is from vertex 0
       to vertex 1. If called it should be once from the mesh constructor,
       because the order may be used later for setting the edges. **/
   void MarkEdge(DenseMatrix & pmat);

   static void MarkEdge(int *indices, const DSTable &v_to_v, const int *length);

   /// Mark the longest edge by assuming/changing the order of the vertices.
   void MarkEdge(const DSTable &v_to_v, const int *length) override
   { MarkEdge(indices, v_to_v, length); }

   void ResetTransform(int tr) override { transform = tr; }
   unsigned GetTransform() const override { return transform; }

   /// Add 'tr' to the current chain of coarse-fine transformations.
   void PushTransform(int tr) override
   { transform = (transform << 3) | (tr + 1); }

   /// Calculate point matrix corresponding to a chain of transformations.
   static void GetPointMatrix(unsigned transform, DenseMatrix &pm);

   /// Get the indices defining the vertices.
   void GetVertices(Array<int> &v) const override;

   /// Set the indices defining the vertices.
   void SetVertices(const Array<int> &v) override;

   /// @note The returned array should NOT be deleted by the caller.
   int * GetVertices () override { return indices; }

   /// Set the indices defining the vertices.
   void SetVertices(const int *ind) override;


   int GetNVertices() const override { return 3; }

   int GetNEdges() const override { return (3); }

   const int *GetEdgeVertices(int ei) const override
   { return geom_t::Edges[ei]; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED int GetNFaces(int &nFaceVertices) const override
   { nFaceVertices = 0; return 0; }

   int GetNFaces() const override { return 0; }

   int GetNFaceVertices(int) const override { return 0; }

   const int *GetFaceVertices(int fi) const override
   { MFEM_ABORT("not implemented"); return NULL; }

   Element *Duplicate(Mesh *m) const override
   { return new Triangle(indices, attribute); }

   virtual ~Triangle() = default;
};

// Defined in fe.cpp to ensure construction before 'mfem::Geometries'.
extern MFEM_EXPORT Linear2DFiniteElement TriangleFE;

}

#endif
