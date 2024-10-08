// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (2 x 1) so that each row
       contains the derivative of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (3 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &h) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.0; dofs(vertex) = 1.0; } // { dofs = 1.0; }
};

/// A linear element on a triangle with nodes at the 3 "Gaussian" points
class GaussLinear2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the GaussLinear2DFiniteElement
   GaussLinear2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};

/// A 2D bi-linear element on a square with nodes at the "Gaussian" points
class GaussBiLinear2DFiniteElement : public NodalFiniteElement
{
private:
   static const real_t p[2];

public:
   /// Construct the FiniteElement
   GaussBiLinear2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};

/** @brief A 2D linear element on a square with 3 nodes at the
    vertices of the lower left triangle */
class P1OnQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1OnQuadFiniteElement
   P1OnQuadFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 1.0; }
};

/// A 1D quadratic finite element with uniformly spaced nodes
class Quad1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Quad1DFiniteElement
   Quad1DFiniteElement();

   /** virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (3) */
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &h) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};

/// A quadratic element on triangle with nodes at the "Gaussian" points
class GaussQuad2DFiniteElement : public NodalFiniteElement
{
private:
   static const real_t p[2];
   DenseMatrix A;
   mutable DenseMatrix D;
   mutable Vector pol;
public:
   /// Construct the GaussQuad2DFiniteElement
   GaussQuad2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override;
};


/// A 2D bi-quadratic element on a square with nodes at the 9 "Gaussian" points
class GaussBiQuad2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the GaussBiQuad2DFiniteElement
   GaussBiQuad2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   // virtual void ProjectDelta(int vertex, Vector &dofs) const { dofs = 1.; }
};


/// A 2D bi-cubic element on a square with uniformly spaces nodes
class BiCubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the BiCubic2DFiniteElement
   BiCubic2DFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   /// Compute the Hessian of second order partial derivatives at @a ip.
   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &h) const override;
};

/// A 1D cubic element with uniformly spaced nodes
class Cubic1DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic1DFiniteElement
   Cubic1DFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A 2D cubic element on a triangle with uniformly spaced nodes
class Cubic2DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic2DFiniteElement
   Cubic2DFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void CalcHessian(const IntegrationPoint &ip,
                    DenseMatrix &h) const override;
};

/// A 3D cubic element on a tetrahedron with 20 nodes at the thirds of the
/// tetrahedron
class Cubic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Cubic3DFiniteElement
   Cubic3DFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A linear element defined on a triangular prism
class LinearWedgeFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the LinearWedgeFiniteElement
   LinearWedgeFiniteElement();

   /** @brief virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** @brief virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.0; dofs(vertex) = 1.0; }

   /** @brief Get the dofs associated with the given @a face.
       @a *dofs is set to an internal array of the local dofc on the
       face, while *ndofs is set to the number of dofs on that face.
   */
   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

/// A linear element defined on a square pyramid
class LinearPyramidFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the LinearPyramidFiniteElement
   LinearPyramidFiniteElement();

   /** @brief virtual function which evaluates the values of all
       shape functions at a given point ip and stores
       them in the vector shape of dimension Dof (4) */
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** @brief virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.0; dofs(vertex) = 1.0; }

   /** @brief Get the dofs associated with the given @a face.
       @a *dofs is set to an internal array of the local dofc on the
       face, while *ndofs is set to the number of dofs on that face.
   */
   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

/// A 2D constant element on a triangle
class P0TriangleFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0TriangleFiniteElement
   P0TriangleFiniteElement();

   /// evaluate shape function - constant 1
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /// evaluate derivatives of shape function - constant 0
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs(0) = 1.0; }
};


/// A 2D constant element on a square
class P0QuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0QuadFiniteElement
   P0QuadFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** @brief virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (4 x 3)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.0; dofs(vertex) = 1.0; }

   /** @brief Get the dofs associated with the given @a face.
       @a *dofs is set to an internal array of the local dofc on the
       face, while *ndofs is set to the number of dofs on that face.
   */
   void GetFaceDofs(int face, int **dofs, int *ndofs) const override;
};

/// A 3D quadratic element on a tetrahedron with uniformly spaced nodes
class Quadratic3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the Quadratic3DFiniteElement
   Quadratic3DFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (8 x 3)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 0.0; dofs(vertex) = 1.0; }
};


/// A 2D Crouzeix-Raviart element on triangle
class CrouzeixRaviartFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the CrouzeixRaviartFiniteElement
   CrouzeixRaviartFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs = 1.0; }
};

/// A 2D Crouzeix-Raviart finite element on square
class CrouzeixRaviartQuadFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the CrouzeixRaviartQuadFiniteElement
   CrouzeixRaviartQuadFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


/// A 1D constant element on a segment
class P0SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0SegmentFiniteElement with dummy order @a Ord
   P0SegmentFiniteElement(int Ord = 0);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/** @brief A 2D 1st order Raviart-Thomas vector element on a triangle */
class RT0TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[3][2];

public:
   /// Construct the RT0TriangleFiniteElement
   RT0TriangleFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

/** @brief A 2D 1st order Raviart-Thomas vector element on a square*/
class RT0QuadFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[4][2];

public:
   /// Construct the RT0QuadFiniteElement
   RT0QuadFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

/** @brief A 2D 2nd order Raviart-Thomas vector element on a triangle */
class RT1TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[8][2];

public:
   /// Construct the RT1TriangleFiniteElement
   RT1TriangleFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

/** @brief A 2D 2nd order Raviart-Thomas vector element on a square */
class RT1QuadFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[12][2];

public:
   /// Construct the RT1QuadFiniteElement
   RT1QuadFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

/** @brief A 2D 3rd order Raviart-Thomas vector element on a triangle */
class RT2TriangleFiniteElement : public VectorFiniteElement
{
private:
   static const real_t M[15][15];
public:
   /// Construct the RT2TriangleFiniteElement
   RT2TriangleFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;
};

/** @brief A 2D 3rd order Raviart-Thomas vector element on a square */
class RT2QuadFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[24][2];
   static const real_t pt[4];
   static const real_t dpt[3];

public:
   /// Construct the RT2QuadFiniteElement
   RT2QuadFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};

/// A 1D linear element with nodes at 1/3 and 2/3 (trace of RT1)
class P1SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1SegmentFiniteElement
   P1SegmentFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A 1D quadratic element with nodes at the Gaussian points (trace of RT2)
class P2SegmentFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P2SegmentFiniteElement
   P2SegmentFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   Lagrange1DFiniteElement(int degree);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A 3D Crouzeix-Raviart element on the tetrahedron.
class P1TetNonConfFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P1TetNonConfFiniteElement
   P1TetNonConfFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A 3D constant element on a tetrahedron
class P0TetFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0TetFiniteElement
   P0TetFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs(0) = 1.0; }
};

/// A 3D constant element on a cube
class P0HexFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0HexFiniteElement
   P0HexFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs(0) = 1.0; }
};

/// A 3D constant element on a wedge
class P0WdgFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0WdgFiniteElement
   P0WdgFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
   { dofs(0) = 1.0; }
};

/// A 3D constant element on a pyramid
class P0PyrFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the P0PyrFiniteElement
   P0PyrFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   void ProjectDelta(int vertex, Vector &dofs) const override
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
   LagrangeHexFiniteElement(int degree);
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
   ~LagrangeHexFiniteElement();
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the derivatives of all
       shape functions at a given point ip and stores them in
       the matrix dshape (Dof x Dim) (3 x 1) so that each row
       contains the derivative of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (6 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};

/// A 2D refined linear element on a tetrahedron
class RefinedLinear3DFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RefinedLinear3DFiniteElement
   RefinedLinear3DFiniteElement();

   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;

   /** virtual function which evaluates the values of all
       partial derivatives of all shape functions at a given
       point ip and stores them in the matrix dshape (Dof x Dim) (9 x 2)
       so that each row contains the derivatives of one shape function */
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
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
   P0WedgeFiniteElement() : L2_WedgeElement(0) {}
};


/// A 3D 1st order Nedelec element on a cube
class Nedelec1HexFiniteElement : public VectorFiniteElement
{
private:
   static const real_t tk[12][3];

public:
   /// Construct the Nedelec1HexFiniteElement
   Nedelec1HexFiniteElement();
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;
};


/// A 3D 1st order Nedelec element on a tetrahedron
class Nedelec1TetFiniteElement : public VectorFiniteElement
{
private:
   static const real_t tk[6][3];

public:
   /// Construct the Nedelec1TetFiniteElement
   Nedelec1TetFiniteElement();
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;
};


/// A 3D 1st order Nedelec element on a wedge
class Nedelec1WdgFiniteElement : public VectorFiniteElement
{
private:
   static const real_t tk[9][3];

public:
   /// Construct the Nedelec1WdgFiniteElement
   Nedelec1WdgFiniteElement();
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;
};


/// A 3D 1st order Nedelec element on a pyramid
class Nedelec1PyrFiniteElement : public VectorFiniteElement
{
private:
   static const real_t tk[8][3];

public:
   /// Construct the Nedelec1PyrFiniteElement
   Nedelec1PyrFiniteElement();
   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;
   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_ND(Trans, shape); }
   void CalcCurlShape(const IntegrationPoint &ip,
                      DenseMatrix &curl_shape) const override;
   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;
   using FiniteElement::Project;
   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;
};


/// A 3D 0th order Raviert-Thomas element on a cube
class RT0HexFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[6][3];

public:
   /// Construct the RT0HexFiniteElement
   RT0HexFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};


/// A 3D 1st order Raviert-Thomas element on a cube
class RT1HexFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[36][3];

public:
   /// Construct the RT1HexFiniteElement
   RT1HexFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};


/// A 3D 0th order Raviert-Thomas element on a tetrahedron
class RT0TetFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[4][3];

public:
   /// Construct the RT0TetFiniteElement
   RT0TetFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;
};


/// A 3D 0th order Raviert-Thomas element on a wedge
class RT0WdgFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[5][3];

public:
   /// Construct the RT0WdgFiniteElement
   RT0WdgFiniteElement();

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override;
};


/// A 3D 0th order Raviert-Thomas element on a pyramid
class RT0PyrFiniteElement : public VectorFiniteElement
{
private:
   static const real_t nk[5][3];

   // If true match RT0TetFiniteElement rather than RT_TetrahedronElement(0)
   bool rt0;

public:
   /// Construct the RT0PyrFiniteElement
   RT0PyrFiniteElement(bool rt0tets = true);

   void CalcVShape(const IntegrationPoint &ip,
                   DenseMatrix &shape) const override;

   void CalcVShape(ElementTransformation &Trans,
                   DenseMatrix &shape) const override
   { CalcVShape_RT(Trans, shape); }

   void CalcDivShape(const IntegrationPoint &ip,
                     Vector &divshape) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override;

   using FiniteElement::Project;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   void ProjectCurl(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &curl) const override;
};


class RotTriLinearHexFiniteElement : public NodalFiniteElement
{
public:
   /// Construct the RotTriLinearHexFiniteElement
   RotTriLinearHexFiniteElement();
   void CalcShape(const IntegrationPoint &ip, Vector &shape) const override;
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;
};


} // namespace mfem

#endif
