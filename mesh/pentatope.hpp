// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#ifndef MFEM_PENTATOPE
#define MFEM_PENTATOPE

#include "../config/config.hpp"
#include "element.hpp"

namespace mfem
{

/// Data type pentatope element
class Pentatope : public Element
{
protected:
   int indices[5];

   unsigned transform;

   /* Flag holds currently
    *    One bit indicating if the element has been swapped.
    *    Two bits indicating the tag of the simplex (0, 1, 2, 3)
    */
   unsigned char flag;

public:

   typedef Geometry::Constants<Geometry::PENTATOPE> geom_p;

   Pentatope() : Element(Geometry::PENTATOPE) { transform = 0; flag = 0;};

   /// Constructs pentatope by specifying the indices and the attribute.
   Pentatope(const int *ind, int attr = 1, unsigned char type = 0);

   /// Constructs pentatope by specifying the indices and the attribute.
   Pentatope(int ind1, int ind2, int ind3, int ind4, int ind5, int attr = 1, unsigned char type = 0);


   virtual int GetRefinementFlag()
   { MFEM_ABORT("PENTATOPE:: GetRefinementFlag not implemented"); return 0; }


   /// Return 1 if the element needs refinement in order to get conforming mesh.
   virtual int NeedRefinement(HashTable<Hashed2> &v_to_v) const;

   /// Set the vertices according to the given input.
   virtual void SetVertices(const int *ind);

   /// Mark the longest edge by assuming/changing the order of the vertices.
   virtual void MarkEdge(DenseMatrix &pmat)
   { MFEM_ABORT("PENTATOPE:: MarkEdge not implemented"); }

   /** Reorder the vertices so that the longest edge is from vertex 0
       to vertex 1. If called it should be once from the mesh constructor,
       because the order may be used later for setting the edges. **/
   virtual void MarkEdge(const DSTable &v_to_v, const int *length)
   { MFEM_ABORT("PENTATOPE:: MarkEdge not implemented"); }

   virtual void GetFace(int fi, int *fv);

   /// Return element's type.
   virtual Type GetType() const { return Element::PENTATOPE; }

   virtual void CreateFlag(char  t, bool  swap);
   virtual void  ParseFlag(char &t, bool &swap);

   virtual void SetFlag(const unsigned char t) { flag = t; }

   /// Return flag of element.
   virtual unsigned char GetFlag() const { return flag; }

   /// Returns the indices of the element's  vertices.
   virtual void GetVertices(Array<int> &v) const;

   virtual int *GetVertices() { return indices; }

   virtual int GetNVertices() const { return 5; }

   virtual int GetNEdges() const { return 10; }

   virtual int GetNPlanars() const { return 10; }

   virtual const int *GetEdgeVertices(int ei) const { return (geom_p::Edges[ei]); }

   virtual const int *GetPlanarsVertices(int pi) const { return (geom_p::PlanarVert[pi]); }

   MFEM_DEPRECATED virtual int GetNFaces(int &nFaceVertices) const
   { nFaceVertices = 4; return 5; }

   virtual int GetNFaces() const { return 5; };

   virtual int GetNFaceVertices(int fi) const { return 4; };

   virtual const int *GetFaceVertices(int fi) const
   { return geom_p::FaceVert[fi]; }

   /// Calculate point matrix corresponding to a chain of transformations.
   static void GetPointMatrix(unsigned transform, DenseMatrix &pm);

   virtual void ResetTransform(int tr) { transform = tr; }
   virtual unsigned GetTransform() const { return transform; }

   virtual void PushTransform(int tr)
   { transform = (transform << 5) | (tr + 1); }

   virtual Element *Duplicate(Mesh *m) const;

   virtual ~Pentatope() { }

};

extern Linear4DFiniteElement PentatopeFE;

}





#endif
