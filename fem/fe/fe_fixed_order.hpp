// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_FIXED_ORDER
#define MFEM_FE_FIXED_ORDER

#include "fe_base.hpp"
#include "fe_h1.hpp"
#include "fe_l2.hpp"

namespace mfem
{

/// A 0D point finite element
class PointFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the PointFiniteElement
   PointFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D linear element with nodes on the endpoints
class Linear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Linear1DFiniteElement
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

/// A 2D linear element on triangle with nodes at the vertices of the triangle
class Linear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Linear2DFiniteElement
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

/// A 2D bi-linear element on a square with nodes at the vertices of the square
class BiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the BiLinear2DFiniteElement
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

/// A linear element on a triangle with nodes at the 3 "Gaussian" points
class GaussLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the GaussLinear2DFiniteElement
   GaussLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// A 2D bi-linear element on a square with nodes at the "Gaussian" points
class GaussBiLinear2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];

public:
   /// Construct the FiniteElement
   GaussBiLinear2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/** @brief A 2D linear element on a square with 3 nodes at the
    vertices of the lower left triangle */
class P1OnQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1OnQuadFiniteElement
   P1OnQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// A 1D quadractic finite element with uniformly spaced nodes
class Quad1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Quad1DFiniteElement
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

/** @brief A 2D quadratic element on triangle with nodes at the
    vertices and midpoints of the triangle. */
class Quad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Quad2DFiniteElement
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

/// A quadratic element on triangle with nodes at the "Gaussian" points
class GaussQuad2DFiniteElement : public NodalFiniteElement
{
private:
   static const double p[2];
   DenseMatrix A;
   mutable DenseMatrix D;
   mutable Vector pol;
public:
   /// Construct the GaussQuad2DFiniteElement
   GaussQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

/// A 2D bi-quadratic element on a square with uniformly spaced nodes
class BiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the BiQuad2DFiniteElement
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


/// A 2D bi-quadratic element on a square with nodes at the 9 "Gaussian" points
class GaussBiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the GaussBiQuad2DFiniteElement
   GaussBiQuad2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const { dofs = 1.; }
};


/// A 2D bi-cubic element on a square with uniformly spaces nodes
class BiCubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the BiCubic2DFiniteElement
   BiCubic2DFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   /// Compute the Hessian of second order partial derivatives at @a ip.
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

/// A 1D cubic element with uniformly spaced nodes
class Cubic1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic1DFiniteElement
   Cubic1DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D cubic element on a triangle with uniformly spaced nodes
class Cubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic2DFiniteElement
   Cubic2DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &h) const;
};

/// A 3D cubic element on a tetrahedron with 20 nodes at the thirds of the
/// tetrahedron
class Cubic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic3DFiniteElement
   Cubic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D constant element on a triangle
class P0TriangleFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0TriangleFiniteElement
   P0TriangleFiniteElement();

   /// evaluate shape function - constant 1
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /// evaluate derivatives of shape function - constant 0
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


/// A 2D constant element on a square
class P0QuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0QuadFiniteElement
   P0QuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};


/** @brief A 3D linear element on a tetrahedron with nodes at the
    vertices of the tetrahedron */
class Linear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Linear3DFiniteElement
   Linear3DFiniteElement();

   /** @brief virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   /** @brief virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 0.0; dofs(vertex) = 1.0; }

   /** @brief Get the dofs associated with the given @a face.
       @a *dofs is set to an internal array of the local dofc on the
       face, while *ndofs is set to the number of dofs on that face.
   */
   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;
};

/// A 3D quadratic element on a tetrahedron with uniformly spaced nodes
class Quadratic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Quadratic3DFiniteElement
   Quadratic3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D tri-linear element on a cube with nodes at the vertices of the cube
class TriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the TriLinear3DFiniteElement
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


/// A 2D Crouzeix-Raviart element on triangle
class CrouzeixRaviartFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the CrouzeixRaviartFiniteElement
   CrouzeixRaviartFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs = 1.0; }
};

/// A 2D Crouzeix-Raviart finite element on square
class CrouzeixRaviartQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the CrouzeixRaviartQuadFiniteElement
   CrouzeixRaviartQuadFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


/// A 1D constant element on a segment
class P0SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0SegmentFiniteElement with dummy order @a Ord
   P0SegmentFiniteElement(int Ord = 0);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/** @brief A 2D 1st order Raviart-Thomas vector element on a triangle */
class RT0TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[3][2];

public:
   /// Construct the RT0TriangleFiniteElement
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

/** @brief A 2D 1st order Raviart-Thomas vector element on a square*/
class RT0QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][2];

public:
   /// Construct the RT0QuadFiniteElement
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

/** @brief A 2D 2nd order Raviart-Thomas vector element on a triangle */
class RT1TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[8][2];

public:
   /// Construct the RT1TriangleFiniteElement
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

/** @brief A 2D 2nd order Raviart-Thomas vector element on a square */
class RT1QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[12][2];

public:
   /// Construct the RT1QuadFiniteElement
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

/** @brief A 2D 3rd order Raviart-Thomas vector element on a triangle */
class RT2TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const double M[15][15];
public:
   /// Construct the RT2TriangleFiniteElement
   RT2TriangleFiniteElement();

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }

   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
};

/** @brief A 2D 3rd order Raviart-Thomas vector element on a square */
class RT2QuadFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[24][2];
   static const double pt[4];
   static const double dpt[3];

public:
   /// Construct the RT2QuadFiniteElement
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

/// A 1D linear element with nodes at 1/3 and 2/3 (trace of RT1)
class P1SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1SegmentFiniteElement
   P1SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D quadratic element with nodes at the Gaussian points (trace of RT2)
class P2SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P2SegmentFiniteElement
   P2SegmentFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 1D element with uniform nodes
class Lagrange1DFiniteElement : public NodalFiniteElement
{
private:
   Vector rwk;
#ifndef MFEM_THREAD_SAFE
   mutable Vector rxxk;
#endif
public:
   /// Construct the Lagrange1DFiniteElement with the provided @a degree
   Lagrange1DFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D Crouzeix-Raviart element on the tetrahedron.
class P1TetNonConfFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1TetNonConfFiniteElement
   P1TetNonConfFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 3D constant element on a tetrahedron
class P0TetFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0TetFiniteElement
   P0TetFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

/// A 3D constant element on a cube
class P0HexFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0HexFiniteElement
   P0HexFiniteElement ();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const
   { dofs(0) = 1.0; }
};

/** @brief Tensor products of 1D Lagrange1DFiniteElement
    (only degree 2 is functional) */
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
   /// Construct the LagrangeHexFiniteElement with the provided @a degree
   LagrangeHexFiniteElement (int degree);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   ~LagrangeHexFiniteElement ();
};


/// A 1D refined linear element
class RefinedLinear1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedLinear1DFiniteElement
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

/// A 2D refined linear element on a triangle
class RefinedLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedLinear2DFiniteElement
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

/// A 2D refined linear element on a tetrahedron
class RefinedLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedLinear3DFiniteElement
   RefinedLinear3DFiniteElement();

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;

   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

/// A 2D refined bi-linear FE on a square
class RefinedBiLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedBiLinear2DFiniteElement
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

/// A 3D refined tri-linear element on a cube
class RefinedTriLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedTriLinear3DFiniteElement
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


/// Class for linear FE on wedge
class BiLinear3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a linear FE on wedge
   BiLinear3DFiniteElement() : H1_WedgeElement(1) {}
};

/// Class for quadratic FE on wedge
class BiQuadratic3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a quadratic FE on wedge
   BiQuadratic3DFiniteElement() : H1_WedgeElement(2) {}
};

/// Class for cubic FE on wedge
class BiCubic3DFiniteElement : public H1_WedgeElement
{
public:
   /// Construct a cubic FE on wedge
   BiCubic3DFiniteElement() : H1_WedgeElement(3) {}
};


/// A 0th order L2 element on a Wedge
class P0WedgeFiniteElement : public L2_WedgeElement
{
public:
   /// Construct the P0WedgeFiniteElement
   P0WedgeFiniteElement () : L2_WedgeElement(0) {}
};


/// A 3D 1st order Nedelec element on a cube
class Nedelec1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[12][3];

public:
   /// Construct the Nedelec1HexFiniteElement
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


/// A 3D 1st order Nedelec element on a tetrahedron
class Nedelec1TetFiniteElement : public VectorFiniteElement
{
private:
   static const double tk[6][3];

public:
   /// Construct the Nedelec1TetFiniteElement
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


/// A 3D 0th order Raviert-Thomas element on a cube
class RT0HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[6][3];

public:
   /// Construct the RT0HexFiniteElement
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


/// A 3D 1st order Raviert-Thomas element on a cube
class RT1HexFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[36][3];

public:
   /// Construct the RT1HexFiniteElement
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


/// A 3D 0th order Raviert-Thomas element on a tetrahedron
class RT0TetFiniteElement : public VectorFiniteElement
{
private:
   static const double nk[4][3];

public:
   /// Construct the RT0TetFiniteElement
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
   /// Construct the RotTriLinearHexFiniteElement
   RotTriLinearHexFiniteElement();
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


} // namespace mfem

#endif

