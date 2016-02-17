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

#ifndef MFEM_ELEMENT
#define MFEM_ELEMENT

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../general/table.hpp"
#include "../linalg/densemat.hpp"
#include "../fem/geom.hpp"

namespace mfem
{

class Mesh;

/// Abstract data type element
class Element
{
protected:

   /// Element's attribute (specifying material property, etc).
   int attribute, base_geom;

public:

   /// Constants for the classes derived from Element.
   enum Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON,
               HEXAHEDRON, BISECTED, QUADRISECTED, OCTASECTED
             };

   /// Default element constructor.
   explicit Element(int bg = Geometry::POINT) { attribute = -1; base_geom = bg; }

   /// Set the indices the element according to the input.
   virtual void SetVertices(const int *ind);

   /// Returns element's type
   virtual int GetType() const = 0;

   int GetGeometryType() const { return base_geom; }

   /// Returns element's vertices.
   virtual void GetVertices(Array<int> &v) const = 0;

   virtual int *GetVertices() = 0;

   const int *GetVertices() const
   { return const_cast<Element *>(this)->GetVertices(); }

   virtual int GetNVertices() const = 0;

   virtual int GetNEdges() const = 0;

   virtual const int *GetEdgeVertices(int) const = 0;

   virtual int GetNFaces(int &nFaceVertices) const = 0;

   virtual const int *GetFaceVertices(int fi) const = 0;

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(DenseMatrix &pmat) {}

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(const DSTable &v_to_v, const int *length) {}

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   virtual int NeedRefinement(DSTable &v_to_v, int *middle) const { return 0; }

   /// Return element's attribute.
   inline int GetAttribute() const { return attribute; }

   /// Set element's attribute.
   inline void SetAttribute(const int attr) { attribute = attr; }

   virtual int GetRefinementFlag() { return 0; }

   virtual Element *Duplicate(Mesh *m) const = 0;

   /// Destroys element.
   virtual ~Element() { }
};

class RefinedElement : public Element
{
public:

   enum { COARSE = 0, FINE = 1 };

   // The default value is set in the file 'point.cpp'
   static int State;

   Element *CoarseElem, *FirstChild;

   RefinedElement() { }

   RefinedElement(Element *ce) : Element(ce->GetGeometryType())
   { attribute = ce->GetAttribute(); CoarseElem = ce; }
   // Assumes that the coarse element and its first child have the
   // same attribute and the same base geometry ...

   void SetCoarseElem(Element *ce)
   {
      base_geom = ce->GetGeometryType();
      attribute = ce->GetAttribute();
      CoarseElem = ce;
   }

   Element *IAm()
   {
      if (State == RefinedElement::COARSE) { return CoarseElem; }
      return FirstChild;
   }
   const Element *IAm() const
   {
      if (State == RefinedElement::COARSE) { return CoarseElem; }
      return FirstChild;
   }

   virtual void SetVertices(const int *ind) { IAm()->SetVertices(ind); }

   virtual void GetVertices(Array<int> &v) const { IAm()->GetVertices(v); }

   virtual int *GetVertices() { return IAm()->GetVertices(); }

   virtual int GetNVertices() const { return IAm()->GetNVertices(); }

   virtual int GetNEdges() const { return (IAm()->GetNEdges()); }

   virtual const int *GetEdgeVertices(int ei) const
   { return (IAm()->GetEdgeVertices(ei)); }

   virtual int GetNFaces(int &nFaceVertices) const
   { return IAm()->GetNFaces(nFaceVertices); }

   virtual const int *GetFaceVertices(int fi) const
   { return IAm()->GetFaceVertices(fi); }

   virtual void MarkEdge(DenseMatrix &pmat) { IAm()->MarkEdge(pmat); }

   virtual void MarkEdge(const DSTable &v_to_v, const int *length)
   { IAm()->MarkEdge(v_to_v, length); }

   virtual int NeedRefinement(DSTable &v_to_v, int *middle) const
   { return IAm()->NeedRefinement(v_to_v, middle); }
};

class BisectedElement : public RefinedElement
{
public:
   int SecondChild;

   BisectedElement() { }
   BisectedElement(Element *ce) : RefinedElement(ce) { }

   virtual int GetType() const { return Element::BISECTED; }

   virtual Element *Duplicate(Mesh *m) const
   { mfem_error("BisectedElement::Duplicate()"); return NULL; }
};

class QuadrisectedElement : public RefinedElement
{
public:
   int Child2, Child3, Child4;

   QuadrisectedElement(Element *ce) : RefinedElement(ce) { }

   virtual int GetType() const { return Element::QUADRISECTED; }

   virtual Element *Duplicate(Mesh *m) const
   { mfem_error("QuadrisectedElement::Duplicate()"); return NULL; }
};

class OctasectedElement : public RefinedElement
{
public:
   int Child[7];

   OctasectedElement(Element *ce) : RefinedElement(ce) { }

   virtual int GetType() const { return Element::OCTASECTED; }

   virtual Element *Duplicate(Mesh *m) const
   { mfem_error("OctasectedElement::Duplicate()"); return NULL; }
};

#ifdef MFEM_USE_MEMALLOC
// defined in tetrahedron.cpp
extern MemAlloc <BisectedElement, 1024> BEMemory;
#endif

}

#endif
