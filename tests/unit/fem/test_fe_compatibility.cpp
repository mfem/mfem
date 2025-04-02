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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

Element * GetElement(Geometry::Type type)
{
   Element *el = NULL;
   switch (type)
   {
      case Geometry::POINT:
         el = new Point;
         break;
      case Geometry::SEGMENT:
         el = new Segment;
         break;
      case Geometry::TRIANGLE:
         el = new Triangle;
         break;
      case Geometry::SQUARE:
         el = new Quadrilateral;
         break;
      case Geometry::TETRAHEDRON:
         el = new Tetrahedron;
         break;
      case Geometry::CUBE:
         el = new Hexahedron;
         break;
      case Geometry::PRISM:
         el = new Wedge;
         break;
      case Geometry::PYRAMID:
         el = new Pyramid;
         break;
      default:
         break;
   }
   return el;
}

// Build a mesh containing a single element
Mesh MakeElementMesh(Geometry::Type type, real_t * vertices)
{
   Element *elem = GetElement(type);

   int nvert = elem->GetNVertices();

   Array<int> el_inds(nvert), el_attr(1);
   for (int i=0; i<nvert; i++) { el_inds[i] = i; }
   el_attr[0] = 1;

   int dim = 0, sdim = -1;
   Geometry::Type bdr_type = Geometry::INVALID;
   if (type == Geometry::SEGMENT)
   {
      dim = 1;
      sdim = 1;
   }
   else if (type >= Geometry::TRIANGLE && type <= Geometry::SQUARE)
   {
      dim = 2;
      sdim = 2;
   }
   else if (type >= Geometry::TETRAHEDRON)
   {
      dim = 3;
      sdim = 3;
   }

   Mesh mesh(vertices, nvert, &el_inds[0], type, &el_attr[0], 1, NULL,
             bdr_type, NULL, 0, dim, sdim);

   mesh.Finalize();

   delete elem;

   return mesh;
}

// Build a mesh containing two copies of the edges of a single element.
// This creates a group of disconnected edges with both possible orientations
// which are aligned with the edges of the parent element.
Mesh MakeElementEdgeMesh(Geometry::Type type, real_t * vertices)
{
   Element *elem = GetElement(type);

   int dim = 1, sdim = 3;

   int nedge = elem->GetNEdges();

   int neelem = 2 * nedge;
   int nevert = 2 * neelem;

   Mesh mesh(dim, nevert, neelem, nevert, sdim);

   int v = 0;

   for (int i=0; i<nedge; i++)
   {
      const int *everts = elem->GetEdgeVertices(i);

      mesh.AddVertex(vertices[3*everts[0]+0],
                     vertices[3*everts[0]+1],
                     vertices[3*everts[0]+2]);
      mesh.AddVertex(vertices[3*everts[1]+0],
                     vertices[3*everts[1]+1],
                     vertices[3*everts[1]+2]);
      mesh.AddSegment(v, v + 1, 1);
      mesh.AddBdrPoint(v, everts[0]+1);
      mesh.AddBdrPoint(v + 1, everts[1]+1);
      v += 2;

      mesh.AddVertex(vertices[3*everts[1]+0],
                     vertices[3*everts[1]+1],
                     vertices[3*everts[1]+2]);
      mesh.AddVertex(vertices[3*everts[0]+0],
                     vertices[3*everts[0]+1],
                     vertices[3*everts[0]+2]);
      mesh.AddSegment(v, v + 1, 2);
      mesh.AddBdrPoint(v, everts[1]+1);
      mesh.AddBdrPoint(v + 1, everts[0]+1);
      v += 2;
   }
   mesh.FinalizeMesh();

   delete elem;

   return mesh;
}

// Build a mesh containing multiple copies of the faces of a single element.
// This creates a group of disconnected faces with all possible orientations
// which are aligned with the faces of the parent element. Specifically, this
// produces six copies of triangular faces and eight copies of quadrilateral
// faces.
Mesh MakeElementFaceMesh(Geometry::Type type, real_t * vertices)
{
   Element *elem = GetElement(type);

   int dim = 2, sdim = 3;

   int nface = elem->GetNFaces();

   int nfelem = 0;
   int nfvert = 0;

   Array<int> nfv(nface);
   for (int i=0; i<nface; i++)
   {
      nfv[i] = elem->GetNFaceVertices(i);

      nfelem += 2 * nfv[i];
      nfvert += 2 * nfv[i] * nfv[i];
   }

   Mesh mesh(dim, nfvert, nfelem, 0, sdim);

   int v = 0;

   for (int i=0; i<nface; i++)
   {
      const int *fverts = elem->GetFaceVertices(i);

      for (int p=0; p < 2; p++)
      {
         for (int o=0; o<nfv[i]; o++)
         {
            for (int j=0; j<nfv[i]; j++)
            {
               mesh.AddVertex(vertices[3*fverts[j]+0],
                              vertices[3*fverts[j]+1],
                              vertices[3*fverts[j]+2]);
            }
            if (nfv[i] == 3)
            {
               if (p == 0)
               {
                  mesh.AddTriangle(v + o%3, v + (o + 1)%3, v + (o + 2)%3, 1);
               }
               else
               {
                  mesh.AddTriangle(v + (o + 2)%3, v + (o + 1)%3, v + o%3, 1);
               }
            }
            else
            {
               if (p == 0)
               {
                  mesh.AddQuad(v + (o + 0)%4, v + (o + 1)%4,
                               v + (o + 2)%4, v + (o + 3)%4, 1);
               }
               else
               {
                  mesh.AddQuad(v + (o + 3)%4, v + (o + 2)%4,
                               v + (o + 1)%4, v + (o + 0)%4, 1);
               }
            }
            v += nfv[i];
         }
      }
   }
   mesh.FinalizeMesh();

   delete elem;

   return mesh;
}

// For a given element geometry, order, and dof index this function returns
// the geometry type of the entity associated with that particular H1 index.
// Additionally finfo returns the numbers of triangular and quadrilateral faces
// touching this dof index (ntri = finfo % 8, nquad = finfo / 8).
Geometry::Type GetH1DofType(Geometry::Type geom, int p, int index, int &finfo)
{
   finfo = 0;
   if (geom == Geometry::TETRAHEDRON)
   {
      if (index < 4)
      {
         finfo = 3;
         return Geometry::POINT;
      }
      if (index < 4 + 6 * (p - 1))
      {
         finfo = 2;
         return Geometry::SEGMENT;
      }
      if (index < 4 + 6 * (p - 1) + 2 * (p - 1) * (p - 2))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::TETRAHEDRON;
   }
   if (geom == Geometry::CUBE)
   {
      if (index < 8)
      {
         finfo = 8 * 3;
         return Geometry::POINT;
      }
      if (index < 8 + 12 * (p - 1))
      {
         finfo = 8 * 2;
         return Geometry::SEGMENT;
      }
      if (index < 8 + 12 * (p - 1) + 6 * (p - 1) * (p - 1))
      {
         return Geometry::SQUARE;
      }
      return Geometry::CUBE;
   }
   if (geom == Geometry::PRISM)
   {
      if (index < 6)
      {
         finfo = 1 + 8 * 2;
         return Geometry::POINT;
      }
      if (index < 6 + 9 * (p - 1))
      {
         finfo = (index < 6 + 6 * (p - 1)) ? (1 + 8 * 1) : (8 * 2);
         return Geometry::SEGMENT;
      }
      if (index < 6 + 9 * (p - 1) + (p - 1) * (p - 2))
      {
         return Geometry::TRIANGLE;
      }
      if (index < 6 + 9 * (p - 1) + (p - 1) * (p - 2) + 3 * (p - 1) * (p - 1))
      {
         return Geometry::SQUARE;
      }
      return Geometry::PRISM;
   }
   if (geom == Geometry::PYRAMID)
   {
      if (index < 5)
      {
         finfo = (index < 4) ? (2 + 8 * 1) : 4;
         return Geometry::POINT;
      }
      if (index < 5 + 8 * (p - 1))
      {
         finfo = (index < 5 + 4 * (p - 1)) ? (1 + 8 * 1) : 2;
         return Geometry::SEGMENT;
      }
      if (index < 5 + 8 * (p - 1) + (p - 1) * (p - 1))
      {
         return Geometry::SQUARE;
      }
      if (index < 5 + 8 * (p - 1) + (p - 1) * (p - 1) + 2 * (p - 1) * (p - 2))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::PYRAMID;
   }
   return Geometry::INVALID;
}

// For a given element geometry, order, and dof index this function returns
// the geometry type of the entity associated with that particular Nedelec
// index.
// Additionally finfo returns the numbers of triangular and quadrilateral
// faces touching this dof index (ntri = finfo % 8, nquad = finfo / 8).
Geometry::Type GetNDDofType(Geometry::Type geom, int p, int index, int &finfo)
{
   finfo = 0;
   if (geom == Geometry::TETRAHEDRON)
   {
      if (index < p * 6)
      {
         finfo = 2;
         return Geometry::SEGMENT;
      }
      if (index < p * 6 + 4 * p * (p - 1))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::TETRAHEDRON;
   }
   if (geom == Geometry::CUBE)
   {
      if (index < p * 12)
      {
         finfo = 8 * 2;
         return Geometry::SEGMENT;
      }
      if (index < p * 12 + 12 * p * (p - 1))
      {
         return Geometry::SQUARE;
      }
      return Geometry::CUBE;
   }
   if (geom == Geometry::PRISM)
   {
      if (index < p * 9)
      {
         finfo = (index < 6 * p) ? (1 + 8 * 1) : (8 * 2);
         return Geometry::SEGMENT;
      }
      if (index < p * 9 + 2 * p * (p - 1))
      {
         return Geometry::TRIANGLE;
      }
      if (index < p * 9 + 2 * p * (p - 1) + 6 * p * (p - 1))
      {
         return Geometry::SQUARE;
      }
      return Geometry::PRISM;
   }
   if (geom == Geometry::PYRAMID)
   {
      if (index < p * 8)
      {
         finfo = (index < 4 * p) ? (1 + 8 * 1) : 2;
         return Geometry::SEGMENT;
      }
      if (index < p * 8 + 2 * p * (p - 1))
      {
         return Geometry::SQUARE;
      }
      if (index < p * 8 + 2 * p * (p - 1) + 4 * p * (p - 1))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::PYRAMID;
   }
   return Geometry::INVALID;
}

// For a given element geometry, order, and dof index this function returns
// the geometry type of the entity associated with that particular
// Raviart-Thomas index.
Geometry::Type GetRTDofType(Geometry::Type geom, int p, int index)
{
   if (geom == Geometry::TETRAHEDRON)
   {
      if (index < 2 * p * (p + 1))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::TETRAHEDRON;
   }
   if (geom == Geometry::CUBE)
   {
      if (index < 6 * p * p)
      {
         return Geometry::SQUARE;
      }
      return Geometry::CUBE;
   }
   if (geom == Geometry::PRISM)
   {
      if (index < p * (p + 1))
      {
         return Geometry::TRIANGLE;
      }
      if (index < p * (p + 1) + 3 * p * p)
      {
         return Geometry::SQUARE;
      }
      return Geometry::PRISM;
   }
   if (geom == Geometry::PYRAMID)
   {
      if (index < p * p)
      {
         return Geometry::SQUARE;
      }
      if (index < p * p + 2 * p * (p + 1))
      {
         return Geometry::TRIANGLE;
      }
      return Geometry::PYRAMID;
   }
   return Geometry::INVALID;
}

static real_t ref_tet_vert[] = {0.,0.,0., 1.,0.,0., 0.,1.,0., 0.,0.,1.};
static real_t equ_tet_vert[] = {0.,0.,0., 1.,0.,0., 0.5,0.8660254037844386,0.,
                                0.5,0.2886751345948129,0.816496580927726
                               };

static real_t ref_cub_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0.,
                                0.,0.,1., 1.,0.,1., 1.,1.,1., 0.,1.,1.
                               };
static real_t *equ_cub_vert = ref_cub_vert;

static real_t ref_pri_vert[] = {0.,0.,0., 1.,0.,0., 0.,1.,0.,
                                0.,0.,1., 1.,0.,1., 0.,1.,1.
                               };
static real_t equ_pri_vert[] = {0.,0.,0., 1.,0.,0., 0.5,0.8660254037844386,0.,
                                0.,0.,1., 1.,0.,1., 0.5,0.8660254037844386,1.
                               };

static real_t ref_pyr_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0., 0.,0.,1.};
static real_t equ_pyr_vert[] = {0.,0.,0., 1.,0.,0., 1.,1.,0., 0.,1.,0.,
                                0.5,0.5,0.7071067811865475
                               };

// Coordinate transformation for 3D elements with planar sides
class VectorTransformation
{
protected:
   Vector a, b, c, d;
   Vector axb, bxc, cxa;
   mutable Vector pt0;
   real_t abc;
   bool init;

   VectorTransformation() : a(3), b(3), c(3), d(3), axb(3), bxc(3), cxa(3),
      pt0(3), abc(-1.0), init(false) {}

   void Init()
   {
      abc = (a[0] * b[1] * c[2] +
             a[1] * b[2] * c[0] +
             a[2] * b[0] * c[1] -
             a[2] * b[1] * c[0] -
             a[1] * b[0] * c[2] -
             a[0] * b[2] * c[1]);

      a.cross3D(b, axb);
      b.cross3D(c, bxc);
      c.cross3D(a, cxa);

      init = true;
   }

   bool CheckPositiveVolume()
   {
      if (!init) { Init(); }
      return abc > 0.0;
   }

public:
   VectorTransformation(const Vector &a_, const Vector &b_,
                        const Vector &c_, const Vector &d_)
      : a(a_), b(b_), c(c_), d(d_),
        axb(3), bxc(3), cxa(3),
        pt0(3), abc(-1.0)
   {
      Init();

      if (!CheckPositiveVolume())
      {
         mfem::err << "VectorTransformation given invalid vectors\n";
      }
   }

   virtual ~VectorTransformation() {}

   virtual void RefToPhys(const Vector &pt, Vector &V) const
   {
      V.SetSize(3);

      V = d;
      V.Add(pt[0], a);
      V.Add(pt[1], d);
      V.Add(pt[2], c);
   }

   virtual void PhysToRef(const Vector &pt, Vector &V) const
   {
      V.SetSize(3);

      add(pt, -1.0, d, pt0);

      V[0] = bxc * pt0;
      V[1] = cxa * pt0;
      V[2] = axb * pt0;
      V /= abc;
   }

   virtual void EvalJ(const Vector &pt, DenseMatrix &J) const
   {
      J.SetSize(3);

      J.SetCol(0, a);
      J.SetCol(1, b);
      J.SetCol(2, c);
   }

   virtual real_t DetJ(const Vector &pt) const { return abc; }
};

class TetrahedronTrans : public VectorTransformation
{
public:
   TetrahedronTrans(real_t *verts)
   {
      a[0] = verts[ 3] - verts[0];
      a[1] = verts[ 4] - verts[1];
      a[2] = verts[ 5] - verts[2];

      b[0] = verts[ 6] - verts[0];
      b[1] = verts[ 7] - verts[1];
      b[2] = verts[ 8] - verts[2];

      c[0] = verts[ 9] - verts[0];
      c[1] = verts[10] - verts[1];
      c[2] = verts[11] - verts[2];

      d[0] = verts[0];
      d[1] = verts[1];
      d[2] = verts[2];

      if (!CheckPositiveVolume())
      {
         mfem::err << "TetrahedronTrans given invalid vertices\n";
      }
   }
};

// Limited to parallelepipeds
class CubeTrans : public VectorTransformation
{
public:
   CubeTrans(real_t * verts)
   {
      a[0] = verts[ 3] - verts[0];
      a[1] = verts[ 4] - verts[1];
      a[2] = verts[ 5] - verts[2];

      b[0] = verts[ 9] - verts[0];
      b[1] = verts[10] - verts[1];
      b[2] = verts[11] - verts[2];

      c[0] = verts[12] - verts[0];
      c[1] = verts[13] - verts[1];
      c[2] = verts[14] - verts[2];

      d[0] = verts[0];
      d[1] = verts[1];
      d[2] = verts[2];

      if (!CheckPositiveVolume())
      {
         mfem::err << "CubeTrans given invalid vertices\n";
      }

   }
};

// Limited to prisms with parallel triangular faces
class PrismTrans : public VectorTransformation
{
public:
   PrismTrans(real_t * verts)
   {
      a[0] = verts[ 3] - verts[0];
      a[1] = verts[ 4] - verts[1];
      a[2] = verts[ 5] - verts[2];

      b[0] = verts[ 6] - verts[0];
      b[1] = verts[ 7] - verts[1];
      b[2] = verts[ 8] - verts[2];

      c[0] = verts[ 9] - verts[0];
      c[1] = verts[10] - verts[1];
      c[2] = verts[11] - verts[2];

      d[0] = verts[0];
      d[1] = verts[1];
      d[2] = verts[2];

      if (!CheckPositiveVolume())
      {
         mfem::err << "PrismTrans given invalid vertices\n";
      }
   }
};

// Limited to pyramids with parallelogram bases
class PyramidTrans : public VectorTransformation
{
public:
   PyramidTrans(real_t * verts)
   {
      a[0] = verts[ 3] - verts[0];
      a[1] = verts[ 4] - verts[1];
      a[2] = verts[ 5] - verts[2];

      b[0] = verts[ 9] - verts[0];
      b[1] = verts[10] - verts[1];
      b[2] = verts[11] - verts[2];

      c[0] = verts[12] - verts[0];
      c[1] = verts[13] - verts[1];
      c[2] = verts[14] - verts[2];

      d[0] = verts[0];
      d[1] = verts[1];
      d[2] = verts[2];

      if (!CheckPositiveVolume())
      {
         mfem::err << "PyramidTrans given invalid vertices\n";
      }
   }
};

VectorTransformation *GetVectorTransformation(Geometry::Type geom,
                                              real_t * verts)
{
   if (geom == Geometry::TETRAHEDRON)
   {
      return new TetrahedronTrans(verts);
   }
   else if (geom == Geometry::CUBE)
   {
      return new CubeTrans(verts);
   }
   else if (geom == Geometry::PRISM)
   {
      return new PrismTrans(verts);
   }
   else if (geom == Geometry::PYRAMID)
   {
      return new PyramidTrans(verts);
   }
   return NULL;
}

class H1BasisCoef : public Coefficient
{
private:
   int p, ndof;

   Geometry::Type geom;
   FiniteElement * elem;
   H1_TetrahedronElement    tet;
   H1_HexahedronElement     cub;
   H1_WedgeElement          pri;
   H1_FuentesPyramidElement pyr;

   VectorTransformation &vtrans;

   Vector dofs;
   mutable Vector shape;

public:
   H1BasisCoef(Geometry::Type g_, int p_, VectorTransformation &vtrans_)
      : p(p_), ndof(-1), geom(g_),
        tet(p), cub(p), pri(p), pyr(p), vtrans(vtrans_)
   {
      dofs = 0.0;

      if (geom == Geometry::TETRAHEDRON)
      {
         elem = &tet;
      }
      else if (geom == Geometry::CUBE)
      {
         elem = &cub;
      }
      else if (geom == Geometry::PRISM)
      {
         elem = &pri;
      }
      else if (geom == Geometry::PYRAMID)
      {
         elem = &pyr;
      }

      ndof = elem->GetDof();
      dofs.SetSize(ndof);
      shape.SetSize(ndof);
   }

   void SetDoF(int dof) { dofs = 0.0; dofs(dof) = 1.0; }
   int GetNDoF() const { return ndof; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip2d)
   {
      real_t pt3d_data[3];
      real_t ip3d_data[3];
      Vector pt3d(pt3d_data, 3);
      Vector ip3d_vec(ip3d_data, 3);
      T.Transform(ip2d, pt3d);

      vtrans.PhysToRef(pt3d, ip3d_vec);

      IntegrationPoint ip3d; ip3d.Set(ip3d_data, 3);
      elem->CalcShape(ip3d, shape);

      return dofs * shape;
   }
};

class HCurlBasisCoef : public VectorCoefficient
{
private:
   int p, ndof;

   Geometry::Type geom;
   FiniteElement * elem;
   ND_TetrahedronElement    tet;
   ND_HexahedronElement     cub;
   ND_WedgeElement          pri;
   ND_FuentesPyramidElement pyr;

   VectorTransformation &vtrans;

   Vector dofs;
   Vector nor;
   Vector tng;
   mutable DenseMatrix jac;
   mutable DenseMatrix jacInv;
   mutable DenseMatrix shape;
   mutable DenseMatrix tshape;

   bool restricted;

public:
   HCurlBasisCoef(Geometry::Type g_, int p_, VectorTransformation &vtrans_,
                  bool restricted_ = false)
      : VectorCoefficient(3),
        p(p_), ndof(-1), geom(g_),
        tet(p), cub(p), pri(p), pyr(p), vtrans(vtrans_),
        nor(3), tng(3), jacInv(3),
        restricted(restricted_)
   {
      dofs = 0.0;

      if (geom == Geometry::TETRAHEDRON)
      {
         elem = &tet;
      }
      else if (geom == Geometry::CUBE)
      {
         elem = &cub;
      }
      else if (geom == Geometry::PRISM)
      {
         elem = &pri;
      }
      else if (geom == Geometry::PYRAMID)
      {
         elem = &pyr;
      }

      ndof = elem->GetDof();
      dofs.SetSize(ndof);
      shape.SetSize(ndof, 3);
      tshape.SetSize(ndof, 3);
   }

   void SetDoF(int dof) { dofs = 0.0; dofs(dof) = 1.0; }
   int GetNDoF() const { return ndof; }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip2d)
   {
      V.SetSize(3);

      real_t pt3d_data[3];
      real_t ip3d_data[3];
      Vector pt3d(pt3d_data, 3);
      Vector ip3d_vec(ip3d_data, 3);
      T.Transform(ip2d, pt3d);

      vtrans.PhysToRef(pt3d, ip3d_vec);
      vtrans.EvalJ(ip3d_vec, jac);
      CalcInverse(jac, jacInv);

      IntegrationPoint ip3d; ip3d.Set(ip3d_data, 3);

      elem->CalcVShape(ip3d, shape);
      Mult(shape, jacInv, tshape);

      tshape.MultTranspose(dofs, V);

      if (restricted)
      {
         if (T.Jacobian().Width() == 1)
         {
            tng[0] = T.Jacobian()(0,0);
            tng[1] = T.Jacobian()(1,0);
            tng[2] = T.Jacobian()(2,0);
            tng /= tng.Norml2();
            real_t tV = tng * V;
            V.Set(tV, tng);
         }
         else if (T.Jacobian().Width() == 2)
         {
            CalcOrtho(T.Jacobian(), nor);
            nor /= nor.Norml2();

            real_t nV = nor * V;
            V.Add(-nV, nor);
         }
      }
   }
};

class HDivBasisCoef : public VectorCoefficient
{
private:
   int p, ndof;

   Geometry::Type geom;
   FiniteElement * elem;
   RT_TetrahedronElement    tet;
   RT_HexahedronElement     cub;
   RT_WedgeElement          pri;
   RT_FuentesPyramidElement pyr;

   VectorTransformation &vtrans;

   Vector dofs;
   mutable DenseMatrix jac;
   mutable DenseMatrix shape;
   mutable DenseMatrix tshape;

public:
   HDivBasisCoef(Geometry::Type g_, int p_, VectorTransformation &vtrans_)
      : VectorCoefficient(3),
        p(p_), ndof(-1), geom(g_),
        tet(p), cub(p), pri(p), pyr(p), vtrans(vtrans_)
   {
      dofs = 0.0;

      if (geom == Geometry::TETRAHEDRON)
      {
         elem = &tet;
      }
      else if (geom == Geometry::CUBE)
      {
         elem = &cub;
      }
      else if (geom == Geometry::PRISM)
      {
         elem = &pri;
      }
      else if (geom == Geometry::PYRAMID)
      {
         elem = &pyr;
      }

      ndof = elem->GetDof();
      dofs.SetSize(ndof);
      shape.SetSize(ndof, 3);
      tshape.SetSize(ndof, 3);
   }

   void SetDoF(int dof) { dofs = 0.0; dofs(dof) = 1.0; }
   int GetNDoF() const { return ndof; }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip2d)
   {
      V.SetSize(3);

      real_t pt3d_data[3];
      real_t ip3d_data[3];
      Vector pt3d(pt3d_data, 3);
      Vector ip3d_vec(ip3d_data, 3);
      T.Transform(ip2d, pt3d);

      vtrans.PhysToRef(pt3d, ip3d_vec);
      vtrans.EvalJ(ip3d_vec, jac);

      IntegrationPoint ip3d; ip3d.Set(ip3d_data, 3);

      elem->CalcVShape(ip3d, shape);
      MultABt(shape, jac, tshape);
      tshape *= 1/jac.Det();

      tshape.MultTranspose(dofs, V);
   }
};

class HDivTraceBasisCoef : public Coefficient
{
private:
   int p, ndof;

   Geometry::Type geom;
   FiniteElement * elem;
   RT_TetrahedronElement    tet;
   RT_HexahedronElement     cub;
   RT_WedgeElement          pri;
   RT_FuentesPyramidElement pyr;

   VectorTransformation &vtrans;

   Vector dofs;
   Vector V;
   Vector nor;
   mutable DenseMatrix jac;
   mutable DenseMatrix shape;
   mutable DenseMatrix tshape;

public:
   HDivTraceBasisCoef(Geometry::Type g_, int p_, VectorTransformation &vtrans_)
      : p(p_), ndof(-1), geom(g_),
        tet(p), cub(p), pri(p), pyr(p), vtrans(vtrans_), V(3), nor(3)
   {
      dofs = 0.0;

      if (geom == Geometry::TETRAHEDRON)
      {
         elem = &tet;
      }
      else if (geom == Geometry::CUBE)
      {
         elem = &cub;
      }
      else if (geom == Geometry::PRISM)
      {
         elem = &pri;
      }
      else if (geom == Geometry::PYRAMID)
      {
         elem = &pyr;
      }

      ndof = elem->GetDof();
      dofs.SetSize(ndof);
      shape.SetSize(ndof, 3);
      tshape.SetSize(ndof, 3);
   }

   void SetDoF(int dof) { dofs = 0.0; dofs(dof) = 1.0; }
   int GetNDoF() const { return ndof; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip2d)
   {
      real_t pt3d_data[3];
      real_t ip3d_data[3];
      Vector pt3d(pt3d_data, 3);
      Vector ip3d_vec(ip3d_data, 3);
      T.Transform(ip2d, pt3d);

      vtrans.PhysToRef(pt3d, ip3d_vec);
      vtrans.EvalJ(ip3d_vec, jac);

      IntegrationPoint ip3d; ip3d.Set(ip3d_data, 3);

      elem->CalcVShape(ip3d, shape);
      MultABt(shape, jac, tshape);
      tshape *= 1/jac.Det();

      tshape.MultTranspose(dofs, V);

      CalcOrtho(T.Jacobian(), nor);
      nor /= nor.Norml2();

      return nor * V;
   }
};

TEST_CASE("FE Compatibility",
          "[H1_TetrahedronElement]"
          "[H1_HexahedronElement]"
          "[H1_WedgeElement]"
          "[H1_PyramidElement]"
          "[ND_TetrahedronElement]"
          "[ND_HexahedronElement]"
          "[ND_WedgeElement]"
          "[ND_PyramidElement]"
          "[RT_TetrahedronElement]"
          "[RT_HexahedronElement]"
          "[RT_WedgeElement]"
          "[RT_PyramidElement]")
{
   auto geom = GENERATE(Geometry::TETRAHEDRON, Geometry::CUBE,
                        Geometry::PRISM, Geometry::PYRAMID);
   auto ref = GENERATE(false);
   auto p = GENERATE(3);

   CAPTURE(geom);
   CAPTURE(ref);
   CAPTURE(p);

   real_t *geom_vert = NULL;
   switch (geom)
   {
      case Geometry::TETRAHEDRON:
         geom_vert = ref ? ref_tet_vert : equ_tet_vert;
         break;
      case Geometry::CUBE:
         geom_vert = ref ? ref_cub_vert : equ_cub_vert;
         break;
      case Geometry::PRISM:
         geom_vert = ref ? ref_pri_vert : equ_pri_vert;
         break;
      case Geometry::PYRAMID:
         geom_vert = ref ? ref_pyr_vert : equ_pyr_vert;
         break;
      default:
         break;
   };
   VectorTransformation *vtrans = GetVectorTransformation(geom, geom_vert);

   Mesh elem_mesh = MakeElementMesh(geom, geom_vert);
   Mesh edge_mesh = MakeElementEdgeMesh(geom, geom_vert);
   Mesh face_mesh = MakeElementFaceMesh(geom, geom_vert);

   real_t tol = 1e-10;

   SECTION("H1 Trace")
   {
      H1_FECollection h1_fec_1d(p, 1);
      H1_FECollection h1_fec_2d(p, 2);
      H1_FECollection h1_fec_3d(p, 3);

      FiniteElementSpace h1_fes_1d(&edge_mesh, &h1_fec_1d);
      FiniteElementSpace h1_fes_2d(&face_mesh, &h1_fec_2d);
      FiniteElementSpace h1_fes_3d(&elem_mesh, &h1_fec_3d);

      GridFunction h1_gf_1d(&h1_fes_1d);
      GridFunction h1_gf_2d(&h1_fes_2d);
      GridFunction h1_gf_3d(&h1_fes_3d);

      H1BasisCoef H1Coef(geom, p, *vtrans);

      for (int i=0; i<H1Coef.GetNDoF(); i++)
      {
         CAPTURE(i);

         H1Coef.SetDoF(i);

         h1_gf_1d.ProjectCoefficient(H1Coef);
         h1_gf_2d.ProjectCoefficient(H1Coef);
         h1_gf_3d.ProjectCoefficient(H1Coef);

         real_t nrmlinf_1d = h1_gf_1d.Normlinf();
         real_t nrmlinf_2d = h1_gf_2d.Normlinf();
         real_t nrmlinf_3d = h1_gf_3d.Normlinf();

         real_t nrml1_1d   = h1_gf_1d.Norml1();
         real_t nrml1_2d   = h1_gf_2d.Norml1();
         real_t nrml1_3d   = h1_gf_3d.Norml1();

         // Should produce exactly one non-zero in 3D
         REQUIRE(fabs(nrml1_3d - nrmlinf_3d) < tol * nrmlinf_3d);

         int finfo = 0;
         Geometry::Type dofType = GetH1DofType(geom, p, i, finfo);

         int ntri = finfo % 8;
         int nsqr = finfo / 8;

         if (dofType == Geometry::POINT)
         {
            real_t a_1d = (geom == Geometry::PYRAMID && i == 4) ? 8.0 : 6.0;
            real_t a_2d = 6.0 * ntri + 8.0 * nsqr;

            // In most case this should find exactly six non-zeros with equal
            // values in 1D trace
            // - Two dofs for every edge which meets at the vertex
            //
            // The apex of a pyramid is a special case where four edges meet
            // producing exactly 8 non-zeros with equal values
            REQUIRE(fabs(nrml1_1d - a_1d * nrmlinf_1d) < tol * nrmlinf_1d);

            // The number of non-zeros in the 2D trace will depend on the
            // types of faces meeting at each vertex.
            REQUIRE(fabs(nrml1_2d - a_2d * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::SEGMENT)
         {
            real_t a_2d = 6.0 * ntri + 8.0 * nsqr;

            // Should find exactly two non-zeros with equal values in 1D trace
            REQUIRE(fabs(nrml1_1d - 2.0 * nrmlinf_1d) < tol * nrmlinf_1d);

            // The number of non-zeros in the 2D trace will depend on the
            // types of faces meeting at each edge.
            REQUIRE(fabs(nrml1_2d - a_2d * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::TRIANGLE)
         {
            // Should find exactly zero non-zeros in 1D trace
            REQUIRE(nrmlinf_1d < tol);

            // Should find exactly six non-zeros with equal values in 2D trace
            // - One non-zero in each of the six possible triangle
            //   orientations
            REQUIRE(fabs(nrml1_2d - 6 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::SQUARE)
         {
            // Should find exactly zero non-zeros in 1D trace
            REQUIRE(nrmlinf_1d < tol);

            // Should find exactly eight non-zeros with equal values in 2D trace
            // - One non-zero in each of the eight possible quadrilateral
            //   orientations
            REQUIRE(fabs(nrml1_2d - 8 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else
         {
            // Should find exactly zero non-zeros in 1D and 2D traces
            REQUIRE(nrmlinf_1d < tol);
            REQUIRE(nrmlinf_2d < tol);
         }
      }
   }

   SECTION("ND Trace")
   {
      ND_FECollection nd_fec_1d(p, 1);
      ND_FECollection nd_fec_2d(p, 2);
      ND_FECollection nd_fec_3d(p, 3);

      FiniteElementSpace nd_fes_1d(&edge_mesh, &nd_fec_1d);
      FiniteElementSpace nd_fes_2d(&face_mesh, &nd_fec_2d);
      FiniteElementSpace nd_fes_3d(&elem_mesh, &nd_fec_3d);

      GridFunction nd_gf_1d(&nd_fes_1d);
      GridFunction nd_gf_2d(&nd_fes_2d);
      GridFunction nd_gf_3d(&nd_fes_3d);

      HCurlBasisCoef HCurlFullCoef(geom, p, *vtrans, false);
      HCurlBasisCoef HCurlTraceCoef(geom, p, *vtrans, true);

      for (int i=0; i<HCurlFullCoef.GetNDoF(); i++)
      {
         HCurlFullCoef.SetDoF(i);
         HCurlTraceCoef.SetDoF(i);

         nd_gf_1d.ProjectCoefficient(HCurlTraceCoef);
         nd_gf_2d.ProjectCoefficient(HCurlTraceCoef);
         nd_gf_3d.ProjectCoefficient(HCurlFullCoef);

         real_t nrmlinf_1d = nd_gf_1d.Normlinf();
         real_t nrmlinf_2d = nd_gf_2d.Normlinf();
         real_t nrmlinf_3d = nd_gf_3d.Normlinf();

         real_t nrml1_1d   = nd_gf_1d.Norml1();
         real_t nrml1_2d   = nd_gf_2d.Norml1();
         real_t nrml1_3d   = nd_gf_3d.Norml1();

         // Should produce exactly one non-zero in 3D
         REQUIRE(fabs(nrml1_3d - nrmlinf_3d) < tol * nrmlinf_3d);

         int finfo = 0;
         Geometry::Type dofType = GetNDDofType(geom, p, i, finfo);

         int ntri = finfo % 8;
         int nsqr = finfo / 8;

         if (dofType == Geometry::SEGMENT)
         {
            real_t a_2d = 6.0 * ntri + 8.0 * nsqr;

            // Should find exactly two non-zeros with equal values in 1D trace
            REQUIRE(fabs(nrml1_1d - 2.0 * nrmlinf_1d) < tol * nrmlinf_1d);

            // The number of non-zeros in the 2D trace will depend on the
            // types of faces meeting at each edge.
            REQUIRE(fabs(nrml1_2d - a_2d * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::TRIANGLE)
         {
            // Should find exactly zero non-zeros in 1D trace
            REQUIRE(nrmlinf_1d < tol);

            // Should find exactly eight non-zeros with equal values in 2D trace
            // - Four values come from orientations in which the x or y axis of
            //   the reference triangle aligns with those of the 3D element.
            // - The other four values correspond to the two alignments where
            //   the 3D basis function aligns with the third edge of the
            //   reference triangle.
            REQUIRE(fabs(nrml1_2d - 8 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::SQUARE)
         {
            // Should find exactly zero non-zeros in 1D trace
            REQUIRE(nrmlinf_1d < tol);

            // Should find exactly eight non-zeros with equal values in 2D trace
            // - One non-zero in each of the eight possible quadrilateral
            //   orientations
            REQUIRE(fabs(nrml1_2d - 8 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else
         {
            // Should find exactly zero non-zeros in 1D and 2D traces
            REQUIRE(nrmlinf_1d < tol);
            REQUIRE(nrmlinf_2d < tol);
         }
      }
   }

   SECTION("RT Trace")
   {
      L2_FECollection rt_fec_2d(p - 1, 2,
                                BasisType::GaussLegendre,
                                FiniteElement::INTEGRAL);
      RT_FECollection rt_fec_3d(p - 1, 3);

      FiniteElementSpace rt_fes_2d(&face_mesh, &rt_fec_2d);
      FiniteElementSpace rt_fes_3d(&elem_mesh, &rt_fec_3d);

      GridFunction rt_gf_2d(&rt_fes_2d);
      GridFunction rt_gf_3d(&rt_fes_3d);

      HDivBasisCoef HDivFullCoef(geom, p - 1, *vtrans);
      HDivTraceBasisCoef HDivTraceCoef(geom, p - 1, *vtrans);

      for (int i=0; i<HDivFullCoef.GetNDoF(); i++)
      {
         CAPTURE(i);

         HDivFullCoef.SetDoF(i);
         HDivTraceCoef.SetDoF(i);
         rt_gf_2d.ProjectCoefficient(HDivTraceCoef);
         rt_gf_3d.ProjectCoefficient(HDivFullCoef);

         real_t nrmlinf_2d = rt_gf_2d.Normlinf();
         real_t nrmlinf_3d = rt_gf_3d.Normlinf();

         real_t nrml1_2d   = rt_gf_2d.Norml1();
         real_t nrml1_3d   = rt_gf_3d.Norml1();

         // Should produce exactly one non-zero in 3D
         REQUIRE(fabs(nrml1_3d - nrmlinf_3d) < tol * nrmlinf_3d);

         Geometry::Type dofType = GetRTDofType(geom, p, i);

         if (dofType == Geometry::TRIANGLE)
         {
            // Should find exactly six non-zeros with equal values in 2D trace
            // - One non-zero in each of the six possible triangle orientations
            REQUIRE(fabs(nrml1_2d - 6 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else if (dofType == Geometry::SQUARE)
         {
            // Should find exactly eight non-zeros with equal values in 2D trace
            // - One non-zero in each of the eight possible quadrilateral
            //   orientations
            REQUIRE(fabs(nrml1_2d - 8 * nrmlinf_2d) < tol * nrmlinf_2d);
         }
         else
         {
            // Should find exactly zero non-zeros in 2D trace
            REQUIRE(nrmlinf_2d < tol);
         }
      }
   }

   delete vtrans;
}
