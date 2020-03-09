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

#ifndef MFEM_HEXAHEDRON
#define MFEM_HEXAHEDRON

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type hexahedron element
class Hexahedron : public Element
{
protected:
   int indices[8];

public:
   typedef Geometry::Constants<Geometry::CUBE> geom_t;

   Hexahedron() : Element(Geometry::CUBE) { }

   /// Constructs hexahedron by specifying the indices and the attribute.
   Hexahedron(const int *ind, int attr = 1);

   /// Constructs hexahedron by specifying the indices and the attribute.
   Hexahedron(int ind1, int ind2, int ind3, int ind4,
              int ind5, int ind6, int ind7, int ind8, int attr = 1);

   /// Return element's type
   Type GetType() const { return Element::HEXAHEDRON; }

   /// Returns the indices of the element's vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 8; }

   virtual int GetNEdges() const { return 12; }

   virtual const int *GetEdgeVertices(int ei) const
   { return geom_t::Edges[ei]; }

   /// @deprecated Use GetNFaces(void) and GetNFaceVertices(int) instead.
   virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 4; return 6; }

   virtual int GetNFaces() const { return 6; }

   virtual int GetNFaceVertices(int) const { return 4; }

   virtual const int *GetFaceVertices(int fi) const
   { return geom_t::FaceVert[fi]; }

   virtual Element *Duplicate(Mesh *m) const
   { return new Hexahedron(indices, attribute); }

   virtual ~Hexahedron() { }
};

extern TriLinear3DFiniteElement HexahedronFE;

}

#endif
