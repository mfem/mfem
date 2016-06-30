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

#ifndef MFEM_FE
#define MFEM_FE

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "geom.hpp"

namespace mfem
{

// Base and derived classes for finite elements

/// Describes the space on each element
class FunctionSpace
{
public:
   enum
   {
      Pk, // polynomials of order k
      Qk, // tensor products of polynomials of order k
      rQk // refined tensor products of polynomials of order k
   };
};

class ElementTransformation;
class Coefficient;
class VectorCoefficient;
class KnotVector;

/// Abstract class for Finite Elements
class FiniteElement
{
protected:
   int Dim, GeomType, Dof, Order, FuncSpace, RangeType, MapType;
   IntegrationRule Nodes;

public:
   /// Enumeration for RangeType
   enum { SCALAR, VECTOR };

   /** Enumeration for MapType: defines how reference functions, uh(xh), are
       mapped to functions on a general element, u(x):

       VALUE       u(x) = uh(xh)
       INTEGRAL    u(x) = (1/w) * uh(xh)
       H_DIV       u(x) = (J/w) * uh(xh)
       H_CURL      u(x) = J^{-t} * uh(xh)           (square J)
       H_CURL      u(x) = J*(J^t*J)^{-1} * uh(xh)   (general J)

       where

       x = T(xh) is the image of the reference point xh ("x hat"),
       J = J(xh) is the Jacobian matrix of the transformation T, and
       w = w(xh) = / det(J),           for square J,
       .           \ det(J^t*J)^{1/2}, for general J,
       is the transformation weight factor.
   */
   enum { VALUE, INTEGRAL, H_DIV, H_CURL };

   /** Construct Finite Element with given
       (D)im      - space dimension,
       (G)eomType - geometry type (of type Geometry::Type),
       (Do)f      - degrees of freedom in the FE space, and
       (O)rder    - order of the FE space
       (F)uncSpace- type of space on each element */
   FiniteElement(int D, int G, int Do, int O, int F = FunctionSpace::Pk);

   /// Returns the space dimension for the finite element
   int GetDim() const { return Dim; }

   /// Returns the geometry type:
   int GetGeomType() const { return GeomType; }

   /// Returns the degrees of freedom in the FE space
   int GetDof() const { return Dof; }

   /// Returns the order of the finite element
   int GetOrder() const { return Order; }

   /// Returns the type of space on each element
   int Space() const { return FuncSpace; }

   int GetRangeType() const { return RangeType; }

   int GetMapType() const { return MapType; }

   /** pure virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof */
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const = 0;

   /** pure virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const = 0;

   const IntegrationRule & GetNodes() const { return Nodes; }

   // virtual functions for finite elements on vector spaces

   /** This virtual function evaluates the values of all components of
       all shape functions at the given IntegrationPoint.
       The result is stored in the DenseMatrix shape (Dof x Dim)
       so that each row contains the components of one shape function.  */
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   /** This virtual function evaluates the divergence of all shape
       functions at the given IntegrationPoint.
       The result is stored in the Vector divshape (of size Dof).  */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** pure virtual function which evaluates the values of the curl
       all shape functions at a given point ip and stores them in
       the matrix curl_shape (Dof x Dim) so that each row contains
       the curl of one shape function */
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;

   /** each row of h contains the upper triangular part of the hessian
       of one shape function; the order in 2D is {u_xx, u_xy, u_yy} */
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;

   /** Return the local interpolation matrix I (Dof x Dof) where the
       fine element is the image of the base geometry under the given
       transformation. */
   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   /** Given a coefficient and a transformation, compute its projection
       (approximation) in the local finite dimensional space in terms
       of the degrees of freedom. */
   virtual void Project (Coefficient &coeff,
                         ElementTransformation &Trans, Vector &dofs) const;

   /** Given a vector coefficient and a transformation, compute its
       projection (approximation) in the local finite dimensional space
       in terms of the degrees of freedom. (VectorFiniteElements) */
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   /** Compute a representation (up to multiplicative constant) for
       the delta function at the vertex with the given index. */
   virtual void ProjectDelta(int vertex, Vector &dofs) const;

   /** Compute the embedding/projection matrix from the given FiniteElement
       onto 'this' FiniteElement. The ElementTransformation is included to
       support cases when the projection depends on it. */
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   /** Compute the discrete gradient matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   /** Compute the discrete curl matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const;

   /** Compute the discrete divergence matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;

   virtual ~FiniteElement () { }
};

class NodalFiniteElement : public FiniteElement
{
protected:
   void NodalLocalInterpolation (ElementTransformation &Trans,
                                 DenseMatrix &I,
                                 const NodalFiniteElement &fine_fe) const;

   void ProjectCurl_2D(const FiniteElement &fe,
                       ElementTransformation &Trans,
                       DenseMatrix &curl) const;

#ifndef MFEM_THREAD_SAFE
   mutable Vector c_shape;
#endif

public:
   NodalFiniteElement(int D, int G, int Do, int O,
                      int F = FunctionSpace::Pk) :
#ifdef MFEM_THREAD_SAFE
      FiniteElement(D, G, Do, O, F)
#else
      FiniteElement(D, G, Do, O, F), c_shape(Do)
#endif
   { }

   void SetMapType(int M)
   {
      MFEM_VERIFY(M == VALUE || M == INTEGRAL, "unknown MapType");
      MapType = M;
   }

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const
   { NodalLocalInterpolation (Trans, I, *this); }

   virtual void Project (Coefficient &coeff,
                         ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;
};


class PositiveFiniteElement : public FiniteElement
{
public:
   PositiveFiniteElement(int D, int G, int Do, int O, int F = FunctionSpace::Pk)
      : FiniteElement(D, G, Do, O, F) { }
   using FiniteElement::Project;
   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;
};

class VectorFiniteElement : public FiniteElement
{
   // Hide the scalar functions CalcShape and CalcDShape.
private:
   /// Overrides the scalar CalcShape function to print an error.
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;

   /// Overrides the scalar CalcDShape function to print an error.
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix J, Jinv;
   mutable DenseMatrix vshape, curlshape, curlshape_J;
#endif

   void CalcVShape_RT(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void CalcVShape_ND(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   // rotated gradient in 2D
   void ProjectGrad_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   void ProjectCurl_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void ProjectCurl_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   void ProjectGrad_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   void LocalInterpolation_RT(const double *nk, const Array<int> &d2n,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalInterpolation_ND(const double *tk, const Array<int> &d2t,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

public:
   VectorFiniteElement (int D, int G, int Do, int O, int M,
                        int F = FunctionSpace::Pk) :
#ifdef MFEM_THREAD_SAFE
      FiniteElement(D, G, Do, O, F)
   { RangeType = VECTOR; MapType = M; }
#else
      FiniteElement(D, G, Do, O, F), Jinv(D), vshape(Do, D)
   { RangeType = VECTOR; MapType = M; }
#endif
};

class PointFiniteElement : public NodalFiniteElement
{
public:
   PointFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for linear FE on interval
class Linear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a linear FE on interval
   Linear1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (2) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (2 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for linear FE on triangle
class Linear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a linear FE on triangle
   Linear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (3 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }
};

/// Class for bilinear FE on quadrilateral
class BiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a bilinear FE on quadrilateral
   BiLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; } // { dofs = 1.0; }
};

/// Class for linear FE on triangle with nodes at the 3 "Gaussian" points
class GaussLinear2DFiniteElement : public NodalFiniteElement
{
public:
   GaussLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Class for bilinear FE on quad with nodes at the 4 Gaussian points
class GaussBiLinear2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];

public:
   GaussBiLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

class P1OnQuadFiniteElement : public NodalFiniteElement
{
public:
   P1OnQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// Class for quadratic FE on interval
class Quad1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on interval
   Quad1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class QuadPos1DFiniteElement : public FiniteElement
{
public:
   QuadPos1DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for quadratic FE on triangle
class Quad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on triangle
   Quad2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (6) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Class for quadratic FE on triangle with nodes at the "Gaussian" points
class GaussQuad2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];
   DenseMatrix A;
   mutable DenseMatrix D;
   mutable Vector pol;
public:
   GaussQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// Class for bi-quadratic FE on quadrilateral
class BiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a biquadratic FE on quadrilateral
   BiQuad2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

class BiQuadPos2DFiniteElement : public FiniteElement
{
public:
   BiQuadPos2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project(Coefficient &coeff, ElementTransformation &Trans,
                        Vector &dofs) const;
   virtual void Project(VectorCoefficient &vc, ElementTransformation &Trans,
                        Vector &dofs) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.; dofs(vertex) = 1.; }
};

/// Bi-quadratic element on quad with nodes at the 9 Gaussian points
class GaussBiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   GaussBiQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const { dofs = 1.; }
};

class BiCubic2DFiniteElement : public NodalFiniteElement
{
public:
   BiCubic2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

class Cubic1DFiniteElement : public NodalFiniteElement
{
public:
   Cubic1DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class Cubic2DFiniteElement : public NodalFiniteElement
{
public:
   Cubic2DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

/// Class for cubic FE on tetrahedron
class Cubic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a cubic FE on tetrahedron
   Cubic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for constant FE on triangle
class P0TriangleFiniteElement : public NodalFiniteElement
{
public:
   /// Construct P0 triangle finite element
   P0TriangleFiniteElement();

   /// evaluate shape function - constant 1
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /// evaluate derivatives of shape function - constant 0
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


class P0QuadFiniteElement : public NodalFiniteElement
{
public:
   P0QuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


/// Class for linear FE on tetrahedron
class Linear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a linear FE on tetrahedron
   Linear3DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }

   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;
};

/// Class for quadratic FE on tetrahedron
class Quadratic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on tetrahedron
   Quadratic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for tri-linear FE on cube
class TriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a tri-linear FE on cube
   TriLinear3DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (8) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (8 x 3)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }
};

/// Crouzeix-Raviart finite element on triangle
class CrouzeixRaviartFiniteElement : public NodalFiniteElement
{
public:
   CrouzeixRaviartFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// Crouzeix-Raviart finite element on quadrilateral
class CrouzeixRaviartQuadFiniteElement : public NodalFiniteElement
{
public:
   CrouzeixRaviartQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class P0SegmentFiniteElement : public NodalFiniteElement
{
public:
   P0SegmentFiniteElement(int Ord = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class RT0TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[3][2];

public:
   RT0TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

class RT0QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][2];

public:
   RT0QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

class RT1TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[8][2];

public:
   RT1TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

class RT1QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[12][2];

public:
   RT1QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

class RT2TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double M[15][15];
public:
   RT2TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
};

class RT2QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[24][2];
   static const double pt[4];
   static const double dpt[3];

public:
   RT2QuadFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};

/// Linear 1D element with nodes 1/3 and 2/3 (trace of RT1)
class P1SegmentFiniteElement : public NodalFiniteElement
{
public:
   P1SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Quadratic 1D element with nodes the Gaussian points in [0,1] (trace of RT2)
class P2SegmentFiniteElement : public NodalFiniteElement
{
public:
   P2SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class Lagrange1DFiniteElement : public NodalFiniteElement
{
private:
   Vector rwk;
#ifndef MFEM_THREAD_SAFE
   mutable Vector rxxk;
#endif
public:
   Lagrange1DFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class P1TetNonConfFiniteElement : public NodalFiniteElement
{
public:
   P1TetNonConfFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class P0TetFiniteElement : public NodalFiniteElement
{
public:
   P0TetFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

class P0HexFiniteElement : public NodalFiniteElement
{
public:
   P0HexFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

/// Tensor products of 1D FEs (only degree 2 is functional)
class LagrangeHexFiniteElement : public NodalFiniteElement
{
private:
   Lagrange1DFiniteElement * fe1d;
   int dof1d;
   int *I, *J, *K;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape1dx, shape1dy, shape1dz;
   mutable DenseMatrix dshape1dx, dshape1dy, dshape1dz;
#endif

public:
   LagrangeHexFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   ~LagrangeHexFiniteElement ();
};


/// Class for refined linear FE on interval
class RefinedLinear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on interval
   RefinedLinear1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for refined linear FE on triangle
class RefinedLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on triangle
   RefinedLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (6) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for refined linear FE on tetrahedron
class RefinedLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a quadratic FE on tetrahedron
   RefinedLinear3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for refined bi-linear FE on quadrilateral
class RefinedBiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a biquadratic FE on quadrilateral
   RefinedBiLinear2DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// Class for refined trilinear FE on a hexahedron
class RefinedTriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct a biquadratic FE on quadrilateral
   RefinedTriLinear3DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (9) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class Nedelec1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[12][3];

public:
   Nedelec1HexFiniteElement();
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class Nedelec1TetFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[6][3];

public:
   Nedelec1TetFiniteElement();
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;
   using FiniteElement::Project;
   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class RT0HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[6][3];

public:
   RT0HexFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class RT1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[36][3];

public:
   RT1HexFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class RT0TetFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][3];

public:
   RT0TetFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   virtual void GetLocalInterpolation (ElementTransformation &Trans,
                                       DenseMatrix &I) const;

   using FiniteElement::Project;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;
};


class RotTriLinearHexFiniteElement : public NodalFiniteElement
{
public:
   RotTriLinearHexFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class Poly_1D
{
public:
   class Basis
   {
   private:
      int mode; // 0 - use change of basis, O(p^2) Evals
      // 1 - use barycentric Lagrangian interpolation, O(p) Evals
      DenseMatrixInverse Ai;
      mutable Vector x, w;

   public:
      Basis(const int p, const double *nodes, const int _mode = 1);
      void Eval(const double x, Vector &u) const;
      void Eval(const double x, Vector &u, Vector &d) const;
   };

private:
   Array<double *> open_pts, closed_pts;
   Array<Basis *> open_basis, closed_basis;

   static Array2D<int> binom;

   static void CalcMono(const int p, const double x, double *u);
   static void CalcMono(const int p, const double x, double *u, double *d);

   static void CalcLegendre(const int p, const double x, double *u);
   static void CalcLegendre(const int p, const double x, double *u, double *d);

   static void CalcChebyshev(const int p, const double x, double *u);
   static void CalcChebyshev(const int p, const double x, double *u, double *d);

public:
   Poly_1D() { }

   static const int *Binom(const int p);

   const double *OpenPoints(const int p);
   const double *ClosedPoints(const int p);

   Basis &OpenBasis(const int p);
   Basis &ClosedBasis(const int p);

   // Evaluate the values of a hierarchical 1D basis at point x
   // hierarchical = k-th basis function is degree k polynomial
   static void CalcBasis(const int p, const double x, double *u)
   // { CalcMono(p, x, u); }
   // Bernstein basis is not hierarchical --> does not work for triangles
   //  and tetrahedra
   // { CalcBernstein(p, x, u); }
   // { CalcLegendre(p, x, u); }
   { CalcChebyshev(p, x, u); }

   // Evaluate the values and derivatives of a hierarchical 1D basis at point x
   static void CalcBasis(const int p, const double x, double *u, double *d)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d); }

   // Evaluate a representation of a Delta function at point x
   static double CalcDelta(const int p, const double x)
   { return pow(x, (double) p); }

   static void UniformPoints(const int p, double *x);
   static void GaussPoints(const int p, double *x);
   static void GaussLobattoPoints(const int p, double *x);
   static void ChebyshevPoints(const int p, double *x);

   /// Compute the terms in the expansion of the binomial (x + y)^p
   static void CalcBinomTerms(const int p, const double x, const double y,
                              double *u);
   /** Compute the terms in the expansion of the binomial (x + y)^p and their
       derivatives with respect to x assuming that dy/dx = -1. */
   static void CalcBinomTerms(const int p, const double x, const double y,
                              double *u, double *d);
   /** Compute the derivatives (w.r.t. x) of the terms in the expansion of the
       binomial (x + y)^p assuming that dy/dx = -1. */
   static void CalcDBinomTerms(const int p, const double x, const double y,
                               double *d);
   static void CalcBernstein(const int p, const double x, double *u)
   { CalcBinomTerms(p, x, 1. - x, u); }
   static void CalcBernstein(const int p, const double x, double *u, double *d)
   { CalcBinomTerms(p, x, 1. - x, u, d); }

   ~Poly_1D();
};

extern Poly_1D poly1d;


class H1_SegmentElement : public NodalFiniteElement
{
private:
   Poly_1D::Basis &basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif
   Array<int> dof_map;

public:
   H1_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};


class H1_QuadrilateralElement : public NodalFiniteElement
{
private:
   Poly_1D::Basis &basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif
   Array<int> dof_map;

public:
   H1_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};


class H1_HexahedronElement : public NodalFiniteElement
{
private:
   Poly_1D::Basis &basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif
   Array<int> dof_map;

public:
   H1_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};

class H1Pos_SegmentElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // This is to share scratch space between invocations, which helps
   // speed things up, but with OpenMP, we need one copy per thread.
   // Right now, we solve this by allocating this space within each function
   // call every time we call it.  Alternatively, we should do some sort
   // thread private thing.  Brunner, Jan 2014
   mutable Vector shape_x, dshape_x;
#endif
   Array<int> dof_map;

public:
   H1Pos_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};


class H1Pos_QuadrilateralElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegmentElement
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif
   Array<int> dof_map;

public:
   H1Pos_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};


class H1Pos_HexahedronElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegementElement.
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif
   Array<int> dof_map;

public:
   H1Pos_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   const Array<int> &GetDofMap() const { return dof_map; }
};


class H1_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   H1_TriangleElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class H1_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   H1_TetrahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class H1Pos_TriangleElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   H1Pos_TriangleElement(const int p);

   // The size of shape is (p+1)(p+2)/2 (dof).
   static void CalcShape(const int p, const double x, const double y,
                         double *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const double x, const double y,
                          double *dshape_1d, double *dshape);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class H1Pos_TetrahedronElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector m_shape, dshape_1d;
   mutable DenseMatrix m_dshape;
#endif
   Array<int> dof_map;

public:
   H1Pos_TetrahedronElement(const int p);

   // The size of shape is (p+1)(p+2)(p+3)/6 (dof).
   static void CalcShape(const int p, const double x, const double y,
                         const double z, double *shape);

   // The size of dshape_1d is p+1; the size of dshape is (dof x dim).
   static void CalcDShape(const int p, const double x, const double y,
                          const double z, double *dshape_1d, double *dshape);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class L2_SegmentElement : public NodalFiniteElement
{
private:
   int type;
   Poly_1D::Basis *basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   L2_SegmentElement(const int p, const int _type = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2Pos_SegmentElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   L2Pos_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2_QuadrilateralElement : public NodalFiniteElement
{
private:
   int type;
   Poly_1D::Basis *basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   L2_QuadrilateralElement(const int p, const int _type = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_2D(fe, Trans, curl); }
};


class L2Pos_QuadrilateralElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   L2Pos_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2_HexahedronElement : public NodalFiniteElement
{
private:
   int type;
   Poly_1D::Basis *basis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   L2_HexahedronElement(const int p, const int _type = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2Pos_HexahedronElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   L2Pos_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2_TriangleElement : public NodalFiniteElement
{
private:
   int type;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   L2_TriangleElement(const int p, const int _type = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_2D(fe, Trans, curl); }
};


class L2Pos_TriangleElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   L2Pos_TriangleElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2_TetrahedronElement : public NodalFiniteElement
{
private:
   int type;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   L2_TetrahedronElement(const int p, const int _type = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2Pos_TetrahedronElement : public PositiveFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector dshape_1d;
#endif

public:
   L2Pos_TetrahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class RT_QuadrilateralElement : public VectorFiniteElement
{
private:
   static const double nk[8];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof_map, dof2nk;

public:
   RT_QuadrilateralElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
};


class RT_HexahedronElement : public VectorFiniteElement
{
   static const double nk[18];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof_map, dof2nk;

public:
   RT_HexahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


class RT_TriangleElement : public VectorFiniteElement
{
   static const double nk[6], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   RT_TriangleElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
};


class RT_TetrahedronElement : public VectorFiniteElement
{
   static const double nk[12], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif
   Array<int> dof2nk;
   DenseMatrixInverse Ti;

public:
   RT_TetrahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


class ND_HexahedronElement : public VectorFiniteElement
{
   static const double tk[18];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof_map, dof2tk;

public:
   ND_HexahedronElement(const int p);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }

   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};


class ND_QuadrilateralElement : public VectorFiniteElement
{
   static const double tk[8];

   Poly_1D::Basis &cbasis1d, &obasis1d;
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof_map, dof2tk;

public:
   ND_QuadrilateralElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


class ND_TetrahedronElement : public VectorFiniteElement
{
   static const double tk[18], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
   mutable DenseMatrix u;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   ND_TetrahedronElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }

   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_ND(tk, dof2tk, fe, Trans, curl); }
};

class ND_TriangleElement : public VectorFiniteElement
{
   static const double tk[8], c;

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector curlu;
#endif
   Array<int> dof2tk;
   DenseMatrixInverse Ti;

public:
   ND_TriangleElement(const int p);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


class ND_SegmentElement : public VectorFiniteElement
{
   static const double tk[1];

   Poly_1D::Basis &obasis1d;
   Array<int> dof2tk;

public:
   ND_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const
   { obasis1d.Eval(ip.x, shape); }
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   // virtual void CalcCurlShape(const IntegrationPoint &ip,
   //                            DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


class NURBSFiniteElement : public FiniteElement
{
protected:
   mutable Array <KnotVector*> kv;
   mutable int *ijk, patch, elem;
   mutable Vector weights;

public:
   NURBSFiniteElement(int D, int G, int Do, int O, int F)
      : FiniteElement(D, G, Do, O, F)
   {
      ijk = NULL;
      patch = elem = -1;
      kv.SetSize(Dim);
      weights.SetSize(Dof);
      weights = 1.0;
   }

   void                 Reset      ()         const { patch = elem = -1; }
   void                 SetIJK     (int *IJK) const { ijk = IJK; }
   int                  GetPatch   ()         const { return patch; }
   void                 SetPatch   (int p)    const { patch = p; }
   int                  GetElement ()         const { return elem; }
   void                 SetElement (int e)    const { elem = e; }
   Array <KnotVector*> &KnotVectors()         const { return kv; }
   Vector              &Weights    ()         const { return weights; }
};

class NURBS1DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector shape_x;

public:
   NURBS1DFiniteElement(int p)
      : NURBSFiniteElement(1, Geometry::SEGMENT, p + 1, p, FunctionSpace::Qk),
        shape_x(p + 1) { }

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class NURBS2DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y;

public:
   NURBS2DFiniteElement(int p)
      : NURBSFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                           FunctionSpace::Qk), u(Dof),
        shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1), dshape_y(p + 1) { }

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class NURBS3DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;

public:
   NURBS3DFiniteElement(int p)
      : NURBSFiniteElement(3, Geometry::CUBE, (p + 1)*(p + 1)*(p + 1), p,
                           FunctionSpace::Qk), u(Dof),
        shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1) { }

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

}

#endif
