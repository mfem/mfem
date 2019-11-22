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

#ifndef MFEM_WEDGE
#define MFEM_WEDGE

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type Wedge element
class Wedge : public Element
{
protected:
   int indices[6];

public:
   typedef Geometry::Constants<Geometry::PRISM> geom_t;

   Wedge() : Element(Geometry::PRISM) { }

   /// Constructs wedge by specifying the indices and the attribute.
   Wedge(const int *ind, int attr = 1);

   /// Constructs wedge by specifying the indices and the attribute.
   Wedge(int ind1, int ind2, int ind3, int ind4, int ind5, int ind6,
         int attr = 1);

   /// Return element's type.
   virtual Type GetType() const { return Element::WEDGE; }

   /// Set the vertices according to the given input.
   virtual void SetVertices(const int *ind);

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 6; }

   virtual int GetNEdges() const { return 9; }

   virtual const int *GetEdgeVertices(int ei) const
   { return geom_t::Edges[ei]; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   virtual int GetNFaces(int &nFaceVertices) const;

   virtual int GetNFaces() const { return 5; }

   virtual int GetNFaceVertices(int fi) const
   { return (fi < 2) ? 3 : 4; }

   virtual const int *GetFaceVertices(int fi) const
   { return geom_t::FaceVert[fi]; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Wedge(indices, attribute); }

   virtual ~Wedge() { }
};

// Defined in fe.cpp to ensure construction after 'mfem::poly1d'.
extern H1_WedgeElement WedgeFE;

}

#endif
