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

#ifndef MFEM_PRISM
#define MFEM_PRISM

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type Prism element
class Prism : public Element
{
protected:
   int indices[6];

   // Rrefinement not yet supported
   // int refinement_flag;

   // Not sure what this might be
   // unsigned transform;

public:
   typedef Geometry::Constants<Geometry::PRISM> geom_t;

   Prism() : Element(Geometry::PRISM) { }

   /// Constructs prism by specifying the indices and the attribute.
   Prism(const int *ind, int attr = 1);

   /// Constructs prism by specifying the indices and the attribute.
   Prism(int ind1, int ind2, int ind3, int ind4, int ind5, int ind6,
         int attr = 1);

   /// Return element's type.
   virtual Type GetType() const { return Element::PRISM; }

   // void  ParseRefinementFlag(int refinement_edges[2], int &type, int &flag);
   // void CreateRefinementFlag(int refinement_edges[2], int  type, int  flag = 0);

   // void GetMarkedFace(const int face, int *fv);

   // virtual int GetRefinementFlag() { return refinement_flag; }

   // void SetRefinementFlag(int rf) { refinement_flag = rf; }

   /// Return 1 if the element needs refinement in order to get conforming mesh.
   // virtual int NeedRefinement(DSTable &v_to_v, int *middle) const;

   /// Set the vertices according to the given input.
   virtual void SetVertices(const int *ind);

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(DenseMatrix &pmat) { }

   /** Reorder the vertices so that the longest edge is from vertex 0
       to vertex 1. If called it should be once from the mesh constructor,
       because the order may be used later for setting the edges. **/
   // virtual void MarkEdge(const DSTable &v_to_v, const int *length);

   // virtual void ResetTransform(int tr) { transform = tr; }
   // virtual unsigned GetTransform() const { return transform; }

   /// Add 'tr' to the current chain of coarse-fine transformations.
   // virtual void PushTransform(int tr)
   // { transform = (transform << 3) | (tr + 1); }

   /// Calculate point matrix corresponding to a chain of transformations.
   // static void GetPointMatrix(unsigned transform, DenseMatrix &pm);

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 6; }

   virtual int GetNEdges() const { return 9; }

   virtual const int *GetEdgeVertices(int ei) const
   { return geom_t::Edges[ei]; }

   virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 4; return 5; }

   virtual int GetNFaceVerticess(int fi) const
   { return ( ( fi < 2 ) ? 3 : 4); }

   virtual const int *GetFaceVertices(int fi) const
   { return geom_t::FaceVert[fi]; }
  
   virtual Element *Duplicate(Mesh *m) const
   { return new Prism(indices, attribute); }

   virtual ~Prism() { }
};

extern BiLinear3DFiniteElement PrismFE;

}

#endif
