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

#ifndef MFEM_ELEMENT
#define MFEM_ELEMENT

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../general/table.hpp"
#include "../linalg/densemat.hpp"
#include "../fem/geom.hpp"
#include "../general/hash.hpp"

namespace mfem
{

class Mesh;

/// Abstract data type element
class Element
{
protected:

   /// Element's attribute (specifying material property, etc).
   int attribute;

   /// Element's type from the Finite Element's perspective
   Geometry::Type base_geom;

public:

   /// Constants for the classes derived from Element.
   enum Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL,
               TETRAHEDRON, HEXAHEDRON, WEDGE
             };

   /// Default element constructor.
   explicit Element(Geometry::Type bg = Geometry::POINT)
   { attribute = -1; base_geom = bg; }

   /// Returns element's type
   virtual Type GetType() const = 0;

   Geometry::Type GetGeometryType() const { return base_geom; }

   /// Return element's attribute.
   inline int GetAttribute() const { return attribute; }

   /// Set element's attribute.
   inline void SetAttribute(const int attr) { attribute = attr; }

   /// Set the indices the element according to the input.
   virtual void SetVertices(const int *ind);

   /// Returns element's vertices.
   virtual void GetVertices(Array<int> &v) const = 0;

   virtual int *GetVertices() = 0;

   const int *GetVertices() const
   { return const_cast<Element *>(this)->GetVertices(); }

   virtual int GetNVertices() const = 0;

   virtual int GetNEdges() const = 0;

   virtual const int *GetEdgeVertices(int) const = 0;

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   MFEM_DEPRECATED virtual int GetNFaces(int &nFaceVertices) const = 0;

   virtual int GetNFaces() const = 0;

   virtual int GetNFaceVertices(int fi) const = 0;

   virtual const int *GetFaceVertices(int fi) const = 0;

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(const DSTable &v_to_v, const int *length) {}

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   virtual int NeedRefinement(HashTable<Hashed2> &v_to_v) const { return 0; }

   /// Set current coarse-fine transformation number.
   virtual void ResetTransform(int tr) {}

   /// Add 'tr' to the current chain of coarse-fine transformations.
   virtual void PushTransform(int tr) {}

   /// Return current coarse-fine transformation.
   virtual unsigned GetTransform() const { return 0; }

   virtual Element *Duplicate(Mesh *m) const = 0;

   /// Destroys element.
   virtual ~Element() { }
};

}

#endif
