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

#ifndef MFEM_FE
#define MFEM_FE

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "geom.hpp"

#include <map>

namespace mfem
{

/// Possible basis types. Note that not all elements can use all BasisType(s).
class BasisType
{
public:
   enum
   {
      Invalid         = -1,
      GaussLegendre   = 0,  ///< Open type
      GaussLobatto    = 1,  ///< Closed type
      Positive        = 2,  ///< Bernstein polynomials
      OpenUniform     = 3,  ///< Nodes: x_i = (i+1)/(n+1), i=0,...,n-1
      ClosedUniform   = 4,  ///< Nodes: x_i = i/(n-1),     i=0,...,n-1
      OpenHalfUniform = 5,  ///< Nodes: x_i = (i+1/2)/n,   i=0,...,n-1
      Serendipity     = 6,  ///< Serendipity basis (squares / cubes)
      NumBasisTypes   = 7   /**< Keep track of maximum types to prevent
                                 hard-coding */
   };
   /** @brief If the input does not represents a valid BasisType, abort with an
       error; otherwise return the input. */
   static int Check(int b_type)
   {
      MFEM_VERIFY(0 <= b_type && b_type < NumBasisTypes,
                  "unknown BasisType: " << b_type);
      return b_type;
   }
   /** @brief If the input does not represents a valid nodal BasisType, abort
       with an error; otherwise return the input. */
   static int CheckNodal(int b_type)
   {
      MFEM_VERIFY(Check(b_type) != Positive,
                  "invalid nodal BasisType: " << Name(b_type));
      return b_type;
   }
   /** @brief Get the corresponding Quadrature1D constant, when that makes
       sense; otherwise return Quadrature1D::Invalid. */
   static int GetQuadrature1D(int b_type)
   {
      switch (b_type)
      {
         case GaussLegendre:   return Quadrature1D::GaussLegendre;
         case GaussLobatto:    return Quadrature1D::GaussLobatto;
         case Positive:        return Quadrature1D::ClosedUniform; // <-----
         case OpenUniform:     return Quadrature1D::OpenUniform;
         case ClosedUniform:   return Quadrature1D::ClosedUniform;
         case OpenHalfUniform: return Quadrature1D::OpenHalfUniform;
         case Serendipity:     return Quadrature1D::GaussLobatto;
      }
      return Quadrature1D::Invalid;
   }
   /// Return the nodal BasisType corresponding to the Quadrature1D type.
   static int GetNodalBasis(int qpt_type)
   {
      switch (qpt_type)
      {
         case Quadrature1D::GaussLegendre:   return GaussLegendre;
         case Quadrature1D::GaussLobatto:    return GaussLobatto;
         case Quadrature1D::OpenUniform:     return OpenUniform;
         case Quadrature1D::ClosedUniform:   return ClosedUniform;
         case Quadrature1D::OpenHalfUniform: return OpenHalfUniform;
      }
      return Invalid;
   }
   /// Check and convert a BasisType constant to a string identifier.
   static const char *Name(int b_type)
   {
      static const char *name[] =
      {
         "Gauss-Legendre", "Gauss-Lobatto", "Positive (Bernstein)",
         "Open uniform", "Closed uniform", "Open half uniform"
      };
      return name[Check(b_type)];
   }
   /// Check and convert a BasisType constant to a char basis identifier.
   static char GetChar(int b_type)
   {
      static const char ident[] = { 'g', 'G', 'P', 'u', 'U', 'o' };
      return ident[Check(b_type)];
   }
   /// Convert char basis identifier to a BasisType constant.
   static int GetType(char b_ident)
   {
      switch (b_ident)
      {
         case 'g': return GaussLegendre;
         case 'G': return GaussLobatto;
         case 'P': return Positive;
         case 'u': return OpenUniform;
         case 'U': return ClosedUniform;
         case 'o': return OpenHalfUniform;
         case 's': return GaussLobatto;
      }
      MFEM_ABORT("unknown BasisType identifier");
      return -1;
   }
};


/** @brief Structure representing the matrices/tensors needed to evaluate (in
    reference space) the values, gradients, divergences, or curls of a
    FiniteElement at a the quadrature points of a given IntegrationRule. */
/** Object of this type are typically created and owned by the respective
    FiniteElement object. */
class DofToQuad
{
public:
   /// The FiniteElement that created and owns this object.
   /** This pointer is not owned. */
   const class FiniteElement *FE;

   /** @brief IntegrationRule that defines the quadrature points at which the
       basis functions of the #FE are evaluated. */
   /** This pointer is not owned. */
   const IntegrationRule *IntRule;

   /// Type of data stored in the arrays #B, #Bt, #G, and #Gt.
   enum Mode
   {
      /** @brief Full multidimensional representation which does not use tensor
          product structure. The ordering of the degrees of freedom is as
          defined by #FE */
      FULL,

      /** @brief Tensor product representation using 1D matrices/tensors with
          dimensions using 1D number of quadrature points and degrees of
          freedom. */
      /** When representing a vector-valued FiniteElement, two DofToQuad objects
          are used to describe the "closed" and "open" 1D basis functions
          (TODO). */
      TENSOR
   };

   /// Describes the contents of the #B, #Bt, #G, and #Gt arrays, see #Mode.
   Mode mode;

   /** @brief Number of degrees of freedom = number of basis functions. When
       #mode is TENSOR, this is the 1D number. */
   int ndof;

   /** @brief Number of quadrature points. When #mode is TENSOR, this is the 1D
       number. */
   int nqpt;

   /// Basis functions evaluated at quadrature points.
   /** The storage layout is column-major with dimensions:
       - #nqpt x #ndof, for scalar elements, or
       - #nqpt x dim x #ndof, for vector elements, (TODO)

       where

       - dim = dimension of the finite element reference space when #mode is
         FULL, and dim = 1 when #mode is TENSOR. */
   Array<double> B;

   /// Transpose of #B.
   /** The storage layout is column-major with dimensions:
       - #ndof x #nqpt, for scalar elements, or
       - #ndof x #nqpt x dim, for vector elements (TODO). */
   Array<double> Bt;

   /** @brief Gradients/divergences/curls of basis functions evaluated at
       quadrature points. */
   /** The storage layout is column-major with dimensions:
       - #nqpt x dim x #ndof, for scalar elements, or
       - #nqpt x #ndof, for H(div) vector elements (TODO), or
       - #nqpt x cdim x #ndof, for H(curl) vector elements (TODO),

       where

       - dim = dimension of the finite element reference space when #mode is
         FULL, and 1 when #mode is TENSOR,
       - cdim = 1/1/3 in 1D/2D/3D, respectively, when #mode is FULL, and cdim =
         1 when #mode is TENSOR. */
   Array<double> G;

   /// Transpose of #G.
   /** The storage layout is column-major with dimensions:
       - #ndof x #nqpt x dim, for scalar elements, or
       - #ndof x #nqpt, for H(div) vector elements (TODO), or
       - #ndof x #nqpt x cdim, for H(curl) vector elements (TODO). */
   Array<double> Gt;
};


/// Describes the space on each element
class FunctionSpace
{
public:
   enum
   {
      Pk, ///< Polynomials of order k
      Qk, ///< Tensor products of polynomials of order k
      rQk, ///< Refined tensor products of polynomials of order k
      RBF, ///< Radial basis functions
      RK, ///< Reproducing kernels
   };
};

class ElementTransformation;
class Coefficient;
class VectorCoefficient;
class MatrixCoefficient;
class KnotVector;


// Base and derived classes for finite elements


/// Abstract class for Finite Elements
class FiniteElement
{
protected:
   int Dim;      ///< Dimension of reference space
   Geometry::Type GeomType; ///< Geometry::Type of the reference element
   int FuncSpace, RangeType, MapType,
       DerivType, DerivRangeType, DerivMapType;
   mutable
   int Dof,      ///< Number of degrees of freedom
       Order;    ///< Order/degree of the shape functions
   mutable int Orders[Geometry::MaxDim]; ///< Anisotropic orders
   IntegrationRule Nodes;
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix vshape; // Dof x Dim
#endif
   /// Container for all DofToQuad objects created by the FiniteElement.
   /** Multiple DofToQuad objects may be needed when different quadrature rules
       or different DofToQuad::Mode are used. */
   mutable Array<DofToQuad*> dof2quad_array;

public:
   /// Enumeration for RangeType and DerivRangeType
   enum { SCALAR, VECTOR };

   /** @brief Enumeration for MapType: defines how reference functions are
       mapped to physical space.

       A reference function, `uh(xh)`, can be mapped to a function, `u(x)`, on a
       general physical element in following ways:

           VALUE       u(x) = uh(xh)
           INTEGRAL    u(x) = (1/w) * uh(xh)
           H_DIV       u(x) = (J/w) * uh(xh)
           H_CURL      u(x) = J^{-t} * uh(xh)           (square J)
           H_CURL      u(x) = J*(J^t*J)^{-1} * uh(xh)   (general J)

       where

           x = T(xh) is the image of the reference point xh ("x hat"),
           J = J(xh) is the Jacobian matrix of the transformation T, and
           w = w(xh) = / det(J),           for square J,
                       \ det(J^t*J)^{1/2}, for general J,
                     is the transformation weight factor.
   */
   enum { VALUE,     ///< For scalar fields; preserves point values
          INTEGRAL,  ///< For scalar fields; preserves volume integrals
          H_DIV,     /**< For vector fields; preserves surface integrals of the
                          normal component */
          H_CURL     /**< For vector fields; preserves line integrals of the
                          tangential component */
        };

   /** @brief Enumeration for DerivType: defines which derivative method
       is implemented.

       Each FiniteElement class implements only one type of derivative.  The
       value returned by GetDerivType() indicates which derivative method is
       implemented.
   */
   enum { NONE, ///< No derivatives implemented
          GRAD, ///< Implements CalcDShape methods
          DIV,  ///< Implements CalcDivShape methods
          CURL  ///< Implements CalcCurlShape methods
        };

   /** Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */
   FiniteElement(int D, Geometry::Type G, int Do, int O,
                 int F = FunctionSpace::Pk);

   /// Returns the reference space dimension for the finite element
   int GetDim() const { return Dim; }

   /// Returns the Geometry::Type of the reference element
   Geometry::Type GetGeomType() const { return GeomType; }

   /// Returns the number of degrees of freedom in the finite element
   int GetDof() const { return Dof; }

   /** @brief Returns the order of the finite element. In the case of
       anisotropic orders, returns the maximum order. */
   int GetOrder() const { return Order; }

   /** @brief Returns true if the FiniteElement basis *may be using* different
       orders/degrees in different spatial directions. */
   bool HasAnisotropicOrders() const { return Orders[0] != -1; }

   /// Returns an array containing the anisotropic orders/degrees.
   const int *GetAnisotropicOrders() const { return Orders; }

   /// Returns the type of space on each element
   int Space() const { return FuncSpace; }

   int GetRangeType() const { return RangeType; }

   int GetDerivRangeType() const { return DerivRangeType; }

   int GetMapType() const { return MapType; }

   int GetDerivType() const { return DerivType; }

   int GetDerivMapType() const { return DerivMapType; }

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** The size (#Dof) of the result Vector @a shape must be set in advance. */
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const = 0;

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** The size (#Dof) of the result Vector @a shape must be set in advance. */
   void CalcPhysShape(ElementTransformation &Trans, Vector &shape) const;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#Dof x #Dim) of @a dshape must be set in
       advance.  */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const = 0;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#Dof x SDim) of @a dshape must be set in
       advance, where SDim >= #Dim is the physical space dimension as described
       by @a Trans. */
   void CalcPhysDShape(ElementTransformation &Trans, DenseMatrix &dshape) const;

   const IntegrationRule & GetNodes() const { return Nodes; }

   // virtual functions for finite elements on vector spaces

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#Dof x #Dim) of @a shape must be set
       in advance. */
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#Dof x SDim) of @a shape must be set
       in advance, where SDim >= #Dim is the physical space dimension as
       described by @a Trans. */
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   /// Equivalent to the CalcVShape() method with the same arguments.
   void CalcPhysVShape(ElementTransformation &Trans, DenseMatrix &shape) const
   { CalcVShape(Trans, shape); }

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#Dof) of the result Vector @a divshape must be set in advance.
    */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in physical space at the point described by @a Trans. */
   /** The size (#Dof) of the result Vector @a divshape must be set in advance.
    */
   void CalcPhysDivShape(ElementTransformation &Trans, Vector &divshape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#Dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #Dim = 3 and
       CDim = 1 for #Dim = 2. */
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#Dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #Dim = 3 and
       CDim = 1 for #Dim = 2. */
   void CalcPhysCurlShape(ElementTransformation &Trans,
                          DenseMatrix &curl_shape) const;

   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;

   /** @brief Evaluate the Hessians of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a Hessian contains upper triangular
       part of the Hessian of one shape function.
       The order in 2D is {u_xx, u_xy, u_yy}.
       The size (#Dof x (#Dim (#Dim-1)/2) of @a Hessian must be set in advance.*/
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &Hessian) const;

   /** @brief Evaluate the Hessian of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** The size (#Dof, #Dim*(#Dim+1)/2) of @a Hessian must be set in advance. */
   virtual void CalcPhysHessian(ElementTransformation &Trans,
                                DenseMatrix& Hessian) const;

   /** @brief Evaluate the Laplacian of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** The size (#Dof) of @a Laplacian must be set in advance. */
   virtual void CalcPhysLaplacian(ElementTransformation &Trans,
                                  Vector& Laplacian) const;

   virtual void CalcPhysLinLaplacian(ElementTransformation &Trans,
                                     Vector& Laplacian) const;

   /** @brief Return the local interpolation matrix @a I (Dof x Dof) where the
       fine element is the image of the base geometry under the given
       transformation. */
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const;

   /** @brief Return a local restriction matrix @a R (Dof x Dof) mapping fine
       dofs to coarse dofs.

       The fine element is the image of the base geometry under the given
       transformation, @a Trans.

       The assumption in this method is that a subset of the coarse dofs can be
       expressed only in terms of the dofs of the given fine element.

       Rows in @a R corresponding to coarse dofs that cannot be expressed in
       terms of the fine dofs will be marked as invalid by setting the first
       entry (column 0) in the row to infinity().

       This method assumes that the dimensions of @a R are set before it is
       called. */
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const;

   /** @brief Return interpolation matrix, @a I, which maps dofs from a coarse
       element, @a fe, to the fine dofs on @a this finite element. */
   /** @a Trans represents the mapping from the reference element of @a this
       element into a subset of the reference space of the element @a fe, thus
       allowing the "coarse" FiniteElement to be different from the "fine"
       FiniteElement as when h-refinement is combined with p-refinement or
       p-derefinement. It is assumed that both finite elements use the same
       MapType. */
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
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

   /** Given a matrix coefficient and a transformation, compute an approximation
       ("projection") in the local finite dimensional space in terms of the
       degrees of freedom. For VectorFiniteElements, the rows of the coefficient
       are projected in the vector space. */
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

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

   /** Return a DofToQuad structure corresponding to the given IntegrationRule
       using the given DofToQuad::Mode. */
   /** See the documentation for DofToQuad for more details. */
   virtual const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                         DofToQuad::Mode mode) const;

   virtual ~FiniteElement();

   static bool IsClosedType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckClosed(q_type) != Quadrature1D::Invalid));
   }

   static bool IsOpenType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckOpen(q_type) != Quadrature1D::Invalid));
   }

   static int VerifyClosed(int b_type)
   {
      MFEM_VERIFY(IsClosedType(b_type),
                  "invalid closed basis type: " << b_type);
      return b_type;
   }
   static int VerifyOpen(int b_type)
   {
      MFEM_VERIFY(IsOpenType(b_type), "invalid open basis type: " << b_type);
      return b_type;
   }
   static int VerifyNodal(int b_type)
   {
      return BasisType::CheckNodal(b_type);
   }
};

class ScalarFiniteElement : public FiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector c_shape;
#endif

   static const ScalarFiniteElement &CheckScalarFE(const FiniteElement &fe)
   {
      if (fe.GetRangeType() != SCALAR)
      { mfem_error("'fe' must be a ScalarFiniteElement"); }
      return static_cast<const ScalarFiniteElement &>(fe);
   }

   const DofToQuad &GetTensorDofToQuad(const class TensorBasisElement &tb,
                                       const IntegrationRule &ir,
                                       DofToQuad::Mode mode) const;

public:
   ScalarFiniteElement(int D, Geometry::Type G, int Do, int O,
                       int F = FunctionSpace::Pk)
#ifdef MFEM_THREAD_SAFE
      : FiniteElement(D, G, Do, O, F)
   { DerivType = GRAD; DerivRangeType = VECTOR; DerivMapType = H_CURL; }
#else
      : FiniteElement(D, G, Do, O, F), c_shape(Dof)
   { DerivType = GRAD; DerivRangeType = VECTOR; DerivMapType = H_CURL; }
#endif

   void SetMapType(int M)
   {
      MFEM_VERIFY(M == VALUE || M == INTEGRAL, "unknown MapType");
      MapType = M;
      DerivType = (M == VALUE) ? GRAD : NONE;
   }

   /// Nodal interpolation.
   void NodalLocalInterpolation(ElementTransformation &Trans,
                                DenseMatrix &I,
                                const ScalarFiniteElement &fine_fe) const;

   /// "Interpolation" defined through local L2-projection.
   /** If the "fine" elements cannot represent all basis functions of the
       "coarse" element, then boundary values from different sub-elements are
       generally different. */
   void ScalarLocalInterpolation(ElementTransformation &Trans,
                                 DenseMatrix &I,
                                 const ScalarFiniteElement &fine_fe) const;

   virtual const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                         DofToQuad::Mode mode) const;
};

class NodalFiniteElement : public ScalarFiniteElement
{
protected:
   void ProjectCurl_2D(const FiniteElement &fe,
                       ElementTransformation &Trans,
                       DenseMatrix &curl) const;

public:
   NodalFiniteElement(int D, Geometry::Type G, int Do, int O,
                      int F = FunctionSpace::Pk)
      : ScalarFiniteElement(D, G, Do, O, F) { }

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { NodalLocalInterpolation(Trans, I, *this); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const;

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).NodalLocalInterpolation(Trans, I, *this); }

   virtual void Project (Coefficient &coeff,
                         ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project (VectorCoefficient &vc,
                         ElementTransformation &Trans, Vector &dofs) const;

   // (mc.height x mc.width) @ DOFs -> (Dof x mc.width x mc.height) in dofs
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;
};


class PositiveFiniteElement : public ScalarFiniteElement
{
public:
   PositiveFiniteElement(int D, Geometry::Type G, int Do, int O,
                         int F = FunctionSpace::Pk) :
      ScalarFiniteElement(D, G, Do, O, F)
   { }

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { ScalarLocalInterpolation(Trans, I, *this); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this); }

   using FiniteElement::Project;

   // Low-order monotone "projection" (actually it is not a projection): the
   // dofs are set to be the Coefficient values at the nodes.
   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project (VectorCoefficient &vc,
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
   mutable DenseMatrix curlshape, curlshape_J;
#endif
   void SetDerivMembers();

   void CalcVShape_RT(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void CalcVShape_ND(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   // project the rows of the matrix coefficient in an RT space
   void ProjectMatrixCoefficient_RT(
      const double *nk, const Array<int> &d2n,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   void Project_RT(const double *nk, const Array<int> &d2n,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   // rotated gradient in 2D
   void ProjectGrad_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   // Compute the curl as a discrete operator from ND FE (fe) to ND FE (this).
   // The natural FE for the range is RT, so this is an approximation.
   void ProjectCurl_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void ProjectCurl_RT(const double *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   // project the rows of the matrix coefficient in an ND space
   void ProjectMatrixCoefficient_ND(
      const double *tk, const Array<int> &d2t,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   void Project_ND(const double *tk, const Array<int> &d2t,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   void ProjectGrad_ND(const double *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   void LocalInterpolation_RT(const VectorFiniteElement &cfe,
                              const double *nk, const Array<int> &d2n,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalInterpolation_ND(const VectorFiniteElement &cfe,
                              const double *tk, const Array<int> &d2t,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalRestriction_RT(const double *nk, const Array<int> &d2n,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   void LocalRestriction_ND(const double *tk, const Array<int> &d2t,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   static const VectorFiniteElement &CheckVectorFE(const FiniteElement &fe)
   {
      if (fe.GetRangeType() != VECTOR)
      { mfem_error("'fe' must be a VectorFiniteElement"); }
      return static_cast<const VectorFiniteElement &>(fe);
   }

public:
   VectorFiniteElement (int D, Geometry::Type G, int Do, int O, int M,
                        int F = FunctionSpace::Pk) :
#ifdef MFEM_THREAD_SAFE
      FiniteElement(D, G, Do, O, F)
   { RangeType = VECTOR; MapType = M; SetDerivMembers(); }
#else
      FiniteElement(D, G, Do, O, F), Jinv(D)
   { RangeType = VECTOR; MapType = M; SetDerivMembers(); }
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

class QuadPos1DFiniteElement : public PositiveFiniteElement
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

class BiQuadPos2DFiniteElement : public PositiveFiniteElement
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
   enum EvalType
   {
      ChangeOfBasis = 0, // Use change of basis, O(p^2) Evals
      Barycentric   = 1, // Use barycentric Lagrangian interpolation, O(p) Evals
      Positive      = 2, // Fast evaluation of Bernstein polynomials
      NumEvalTypes  = 3  // Keep count of the number of eval types
   };

   class Basis
   {
   private:
      int etype;
      DenseMatrixInverse Ai;
      mutable Vector x, w;

   public:
      /// Create a nodal or positive (Bernstein) basis
      Basis(const int p, const double *nodes, EvalType etype = Barycentric);
      void Eval(const double x, Vector &u) const;
      void Eval(const double x, Vector &u, Vector &d) const;
   };

private:
   typedef std::map< int, Array<double*>* > PointsMap;
   typedef std::map< int, Array<Basis*>* > BasisMap;

   MemoryType h_mt;
   PointsMap points_container;
   BasisMap  bases_container;

   static Array2D<int> binom;

   static void CalcChebyshev(const int p, const double x, double *u);
   static void CalcChebyshev(const int p, const double x, double *u, double *d);
   static void CalcChebyshev(const int p, const double x, double *u, double *d,
                             double *dd);

   QuadratureFunctions1D quad_func;

public:
   Poly_1D(): h_mt(MemoryType::HOST) { }

   /** @brief Get a pointer to an array containing the binomial coefficients "p
       choose k" for k=0,...,p for the given p. */
   static const int *Binom(const int p);

   /** @brief Get the coordinates of the points of the given BasisType,
       @a btype.

       @param[in] p      The polynomial degree; the number of points is `p+1`.
       @param[in] btype  The BasisType.

       @return A pointer to an array containing the `p+1` coordinates of the
               points. Returns NULL if the BasisType has no associated set of
               points. */
   const double *GetPoints(const int p, const int btype);
   const double *OpenPoints(const int p,
                            const int btype = BasisType::GaussLegendre)
   { return GetPoints(p, btype); }
   const double *ClosedPoints(const int p,
                              const int btype = BasisType::GaussLobatto)
   { return GetPoints(p, btype); }

   /** @brief Get a Poly_1D::Basis object of the given degree and BasisType,
       @a btype.

       @param[in] p      The polynomial degree of the basis.
       @param[in] btype  The BasisType.

       @return A reference to an object of type Poly_1D::Basis that represents
               the requested basis type. */
   Basis &GetBasis(const int p, const int btype);

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

   // Evaluate the values, derivatives and second derivatives of a hierarchical 1D basis at point x
   static void CalcBasis(const int p, const double x, double *u, double *d,
                         double *dd)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d, dd); }

   // Evaluate a representation of a Delta function at point x
   static double CalcDelta(const int p, const double x)
   { return pow(x, (double) p); }

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

   static void CalcMono(const int p, const double x, double *u);
   static void CalcMono(const int p, const double x, double *u, double *d);

   static void CalcLegendre(const int p, const double x, double *u);
   static void CalcLegendre(const int p, const double x, double *u, double *d);

   ~Poly_1D();
};

extern Poly_1D poly1d;

class TensorBasisElement
{
protected:
   int b_type;
   Array<int> dof_map;
   Poly_1D::Basis &basis1d;
   Array<int> inv_dof_map;

public:
   enum DofMapType
   {
      L2_DOF_MAP = 0,
      H1_DOF_MAP = 1,
      Sr_DOF_MAP = 2,  // Sr = Serendipity
   };

   TensorBasisElement(const int dims, const int p, const int btype,
                      const DofMapType dmtype);

   int GetBasisType() const { return b_type; }

   const Poly_1D::Basis& GetBasis1D() const { return basis1d; }

   /** @brief Get an Array<int> that maps lexicographically ordered indices to
       the indices of the respective nodes/dofs/basis functions. If the dofs are
       ordered lexicographically, i.e. the mapping is identity, the returned
       Array will be empty. */
   const Array<int> &GetDofMap() const { return dof_map; }

   static Geometry::Type GetTensorProductGeometry(int dim)
   {
      switch (dim)
      {
         case 1: return Geometry::SEGMENT;
         case 2: return Geometry::SQUARE;
         case 3: return Geometry::CUBE;
         default:
            MFEM_ABORT("invalid dimension: " << dim);
            return Geometry::INVALID;
      }
   }

   /// Return @a base raised to the power @a dim.
   static int Pow(int base, int dim)
   {
      switch (dim)
      {
         case 1: return base;
         case 2: return base*base;
         case 3: return base*base*base;
         default: MFEM_ABORT("invalid dimension: " << dim); return -1;
      }
   }
};

class NodalTensorFiniteElement : public NodalFiniteElement,
   public TensorBasisElement
{
public:
   NodalTensorFiniteElement(const int dims, const int p, const int btype,
                            const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const
   {
      return (mode == DofToQuad::FULL) ?
             ScalarFiniteElement::GetDofToQuad(ir, mode) :
             ScalarFiniteElement::GetTensorDofToQuad(*this, ir, mode);
   }
};

class PositiveTensorFiniteElement : public PositiveFiniteElement,
   public TensorBasisElement
{
public:
   PositiveTensorFiniteElement(const int dims, const int p,
                               const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const
   {
      return (mode == DofToQuad::FULL) ?
             ScalarFiniteElement::GetDofToQuad(ir, mode) :
             ScalarFiniteElement::GetTensorDofToQuad(*this, ir, mode);
   }
};

class VectorTensorFiniteElement : public VectorFiniteElement,
   public TensorBasisElement
{
private:
   mutable Array<DofToQuad*> dof2quad_array_open;

protected:
   Poly_1D::Basis &cbasis1d, &obasis1d;

public:
   VectorTensorFiniteElement(const int dims, const int d, const int p,
                             const int cbtype, const int obtype,
                             const int M, const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const;

   const DofToQuad &GetDofToQuadOpen(const IntegrationRule &ir,
                                     DofToQuad::Mode mode) const;

   const DofToQuad &GetTensorDofToQuad(const IntegrationRule &ir,
                                       DofToQuad::Mode mode,
                                       const bool closed) const;
};

class H1_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   H1_SegmentElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class H1_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   H1_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class H1_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   H1_HexahedronElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};

class H1Pos_SegmentElement : public PositiveTensorFiniteElement
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

public:
   H1Pos_SegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class H1Pos_QuadrilateralElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegmentElement
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   H1Pos_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class H1Ser_QuadrilateralElement : public ScalarFiniteElement
{
public:
   H1Ser_QuadrilateralElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const;
   using FiniteElement::Project;
};

class H1Pos_HexahedronElement : public PositiveTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   // See comment in H1Pos_SegementElement.
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   H1Pos_HexahedronElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class H1_TriangleElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   H1_TriangleElement(const int p, const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &ddshape) const;
};


class H1_TetrahedronElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_z, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   DenseMatrixInverse Ti;

public:
   H1_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &ddshape) const;
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


class H1_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1_TriangleElement TriangleFE;
   H1_SegmentElement  SegmentFE;

public:
   H1_WedgeElement(const int p,
                   const int btype = BasisType::GaussLobatto);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
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

class H1Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   H1Pos_TriangleElement TriangleFE;
   H1Pos_SegmentElement  SegmentFE;

public:
   H1Pos_WedgeElement(const int p);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};


class L2_SegmentElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif

public:
   L2_SegmentElement(const int p, const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2Pos_SegmentElement : public PositiveTensorFiniteElement
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


class L2_QuadrilateralElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, dshape_x, dshape_y;
#endif

public:
   L2_QuadrilateralElement(const int p,
                           const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_2D(fe, Trans, curl); }
};


class L2Pos_QuadrilateralElement : public PositiveTensorFiniteElement
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


class L2_HexahedronElement : public NodalTensorFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
#endif

public:
   L2_HexahedronElement(const int p,
                        const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void ProjectDelta(int vertex, Vector &dofs) const;
};


class L2Pos_HexahedronElement : public PositiveTensorFiniteElement
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
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   L2_TriangleElement(const int p,
                      const int btype = BasisType::GaussLegendre);
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
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_z, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
   mutable DenseMatrix du;
#endif
   DenseMatrixInverse Ti;

public:
   L2_TetrahedronElement(const int p,
                         const int btype = BasisType::GaussLegendre);
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


class L2_WedgeElement : public NodalFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2_TriangleElement TriangleFE;
   L2_SegmentElement  SegmentFE;

public:
   L2_WedgeElement(const int p,
                   const int btype = BasisType::GaussLegendre);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
};

class P0WedgeFiniteElement : public L2_WedgeElement
{
public:
   P0WedgeFiniteElement () : L2_WedgeElement(0) {}
};

class L2Pos_WedgeElement : public PositiveFiniteElement
{
protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector t_shape, s_shape;
   mutable DenseMatrix t_dshape, s_dshape;
#endif
   Array<int> t_dof, s_dof;

   L2Pos_TriangleElement TriangleFE;
   L2Pos_SegmentElement  SegmentFE;

public:
   L2Pos_WedgeElement(const int p);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
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
   RT_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }
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
   RT_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_RT(Trans, shape); }
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
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
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   // Gradient + rotation = Curl: H1 -> H(div)
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, grad); }
   // Curl = Gradient + rotation: H1 -> H(div)
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectGrad_RT(nk, dof2nk, fe, Trans, curl); }
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
   { LocalInterpolation_RT(*this, nk, dof2nk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_RT(nk, dof2nk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_RT(CheckVectorFE(fe), nk, dof2nk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_RT(nk, dof2nk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_RT(nk, dof2nk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_RT(nk, dof2nk, fe, Trans, I); }
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const
   { ProjectCurl_RT(nk, dof2nk, fe, Trans, curl); }
};


class ND_HexahedronElement : public VectorTensorFiniteElement
{
   static const double tk[18];
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
   mutable Vector dshape_cx, dshape_cy, dshape_cz;
#endif
   Array<int> dof2tk;

public:
   ND_HexahedronElement(const int p,
                        const int cb_type = BasisType::GaussLobatto,
                        const int ob_type = BasisType::GaussLegendre);

   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }

   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }

   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }

   using FiniteElement::Project;

   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }

   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }

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


class ND_QuadrilateralElement : public VectorTensorFiniteElement
{
   static const double tk[8];

#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy;
#endif
   Array<int> dof2tk;

public:
   ND_QuadrilateralElement(const int p,
                           const int cb_type = BasisType::GaussLobatto,
                           const int ob_type = BasisType::GaussLegendre);
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const
   { CalcVShape_ND(Trans, shape); }
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;
   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
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
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
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
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
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
   ND_SegmentElement(const int p, const int ob_type = BasisType::GaussLegendre);
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
   { LocalInterpolation_ND(*this, tk, dof2tk, Trans, I); }
   virtual void GetLocalRestriction(ElementTransformation &Trans,
                                    DenseMatrix &R) const
   { LocalRestriction_ND(tk, dof2tk, Trans, R); }
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { LocalInterpolation_ND(CheckVectorFE(fe), tk, dof2tk, Trans, I); }
   using FiniteElement::Project;
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const
   { Project_ND(tk, dof2tk, vc, Trans, dofs); }
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
   { ProjectMatrixCoefficient_ND(tk, dof2tk, mc, T, dofs); }
   virtual void Project(const FiniteElement &fe,
                        ElementTransformation &Trans,
                        DenseMatrix &I) const
   { Project_ND(tk, dof2tk, fe, Trans, I); }
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const
   { ProjectGrad_ND(tk, dof2tk, fe, Trans, grad); }
};


class NURBSFiniteElement : public ScalarFiniteElement
{
protected:
   mutable Array <const KnotVector*> kv;
   mutable const int *ijk;
   mutable int patch, elem;
   mutable Vector weights;

public:
   NURBSFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F)
   {
      ijk = NULL;
      patch = elem = -1;
      kv.SetSize(Dim);
      weights.SetSize(Dof);
      weights = 1.0;
   }

   void                 Reset      ()         const { patch = elem = -1; }
   void                 SetIJK     (const int *IJK) const { ijk = IJK; }
   int                  GetPatch   ()         const { return patch; }
   void                 SetPatch   (int p)    const { patch = p; }
   int                  GetElement ()         const { return elem; }
   void                 SetElement (int e)    const { elem = e; }
   Array <const KnotVector*> &KnotVectors()   const { return kv; }
   Vector              &Weights    ()         const { return weights; }
   /// Update the NURBSFiniteElement according to the currently set knot vectors
   virtual void         SetOrder   ()         const { }
};

class NURBS1DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector shape_x;

public:
   NURBS1DFiniteElement(int p)
      : NURBSFiniteElement(1, Geometry::SEGMENT, p + 1, p, FunctionSpace::Qk),
        shape_x(p + 1) { }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};

class NURBS2DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
   mutable DenseMatrix du;

public:
   NURBS2DFiniteElement(int p)
      : NURBSFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(Dof), shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1),
        dshape_y(p + 1), d2shape_x(p + 1), d2shape_y(p + 1), du(Dof,2)
   { Orders[0] = Orders[1] = p; }

   NURBS2DFiniteElement(int px, int py)
      : NURBSFiniteElement(2, Geometry::SQUARE, (px + 1)*(py + 1),
                           std::max(px, py), FunctionSpace::Qk),
        u(Dof), shape_x(px + 1), shape_y(py + 1), dshape_x(px + 1),
        dshape_y(py + 1), d2shape_x(px + 1), d2shape_y(py + 1), du(Dof,2)
   { Orders[0] = px; Orders[1] = py; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};

class NURBS3DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z;
   mutable Vector d2shape_x, d2shape_y, d2shape_z;
   mutable DenseMatrix du;

public:
   NURBS3DFiniteElement(int p)
      : NURBSFiniteElement(3, Geometry::CUBE, (p + 1)*(p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(Dof), shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1),
        d2shape_x(p + 1), d2shape_y(p + 1), d2shape_z(p + 1), du(Dof,3)
   { Orders[0] = Orders[1] = Orders[2] = p; }

   NURBS3DFiniteElement(int px, int py, int pz)
      : NURBSFiniteElement(3, Geometry::CUBE, (px + 1)*(py + 1)*(pz + 1),
                           std::max(std::max(px,py),pz), FunctionSpace::Qk),
        u(Dof), shape_x(px + 1), shape_y(py + 1), shape_z(pz + 1),
        dshape_x(px + 1), dshape_y(py + 1), dshape_z(pz + 1),
        d2shape_x(px + 1), d2shape_y(py + 1), d2shape_z(pz + 1), du(Dof,3)
   { Orders[0] = px; Orders[1] = py; Orders[2] = pz; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};

class RBFFunction
{
public:
   static const double GlobalRadius; // functions with r>=GR are considered global
   RBFFunction() { };
   virtual ~RBFFunction() { }

   // The r is a normalized distance
   virtual double BaseFunction(double r) const = 0;
   virtual double BaseDerivative(double r) const = 0;
   virtual double BaseDerivative2(double r) const = 0;

   // The support radius, outside of which the function is zero
   virtual double Radius() const { return GlobalRadius; }

   // Does function have compact support?
   virtual bool CompactSupport() const { return false; }
   
   // This makes the shape parameter consistent across kernels
   virtual double HNorm() const = 0;
};

class GaussianRBF : public RBFFunction
{
   // hNorm minimizes integral of Gaussian minus Wendland kernel over r=0,1
   static const double hNorm;
public:
   GaussianRBF() { };
   virtual ~GaussianRBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;
   
   virtual double HNorm() const { return hNorm; }
};

class MultiquadricRBF : public RBFFunction
{
   // Same as inverse multiquadric
   static const double hNorm;
public:
   MultiquadricRBF() { };
   virtual ~MultiquadricRBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;
   
   virtual double HNorm() const { return hNorm; }
};

class InvMultiquadricRBF : public RBFFunction
{
   // hNorm minimizes integral of Gaussian minus InvMQ kernel over r=0,0.5
   static const double hNorm;
public:
   InvMultiquadricRBF() { };
   virtual ~InvMultiquadricRBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;
   
   virtual double HNorm() const { return hNorm; }
};

// Shifted down to be exactly zero at radius
class CompactGaussianRBF : public RBFFunction
{
   static const double hNorm;
   const double radius;
   double multK, shiftK;
   
public:
   CompactGaussianRBF(const double rad = 5.0);
   virtual ~CompactGaussianRBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }
   
   virtual double HNorm() const { return hNorm; }
   virtual bool CompactSupport() const { return true; }
};

// Truncated at radius
class TruncatedGaussianRBF : public RBFFunction
{
   static const double hNorm;
   const double radius;
public:
   TruncatedGaussianRBF(const double rad = 5.0)
      : radius(rad) { }
   virtual ~TruncatedGaussianRBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }
   
   virtual double HNorm() const { return hNorm; }
   virtual bool CompactSupport() const { return true; }
};

class Wendland11RBF : public RBFFunction
{
   static const double radius;
   
public:
   Wendland11RBF() { }
   virtual ~Wendland11RBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }
   
   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

class Wendland31RBF : public RBFFunction
{
   static const double radius;
   
public:
   Wendland31RBF() { };
   virtual ~Wendland31RBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }
   
   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

class Wendland33RBF : public RBFFunction
{
   static const double radius;
   
public:
   Wendland33RBF() { };
   virtual ~Wendland33RBF() { }
   
   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }
   
   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

// Choose the type of RBF to use
class RBFType
{
public:
   enum
   {
      Gaussian = 0,
      Multiquadric = 1,
      InvMultiquadric = 2,
      TruncatedGaussian = 3,
      CompactGaussian = 4,
      Wendland11 = 5,
      Wendland31 = 6,
      Wendland33 = 7,
      NumRBFTypes = 8
   };

   // Return the requested RBF
   static RBFFunction *GetRBF(const int rbfType)
   {
      switch (rbfType)
      {
      case RBFType::Gaussian:
         return new GaussianRBF();
      case RBFType::Multiquadric:
         return new MultiquadricRBF();
      case RBFType::InvMultiquadric:
         return new InvMultiquadricRBF();
      case RBFType::TruncatedGaussian:
         return new TruncatedGaussianRBF();
      case RBFType::CompactGaussian:
         return new CompactGaussianRBF();
      case RBFType::Wendland11:
         return new Wendland11RBF();
      case RBFType::Wendland31:
         return new Wendland31RBF();
      case RBFType::Wendland33:
         return new Wendland33RBF();
      }
      MFEM_ABORT("unknown RBF type");
      return NULL;
   }
   
   // Abort if rbfType is invalid
   static int Check(const int rbfType)
   {
      MFEM_VERIFY(0 <= rbfType && rbfType < NumRBFTypes,
                  "unknown RBF type: " << rbfType);
      return rbfType;
   }

   // Convert rbf int to identifier
   static char GetChar(const int rbfType)
   {
      static const char ident[] = { 'G', 'M', 'I',
                                    'T', 'C',
                                    '1', '3', '6' };
      return ident[Check(rbfType)];
   }
   
   // Convert identifier to rbf int
   static int GetType(const char rbfIdent)
   {
      switch (rbfIdent)
      {
      case 'G': return Gaussian;
      case 'M': return Multiquadric;
      case 'I': return InvMultiquadric;
      case 'T': return TruncatedGaussian;
      case 'C': return CompactGaussian;
      case '1': return Wendland11;
      case '3': return Wendland31;
      case '6': return Wendland33;
      }
      MFEM_ABORT("unknown RBF identifier: " << rbfIdent);
      return -1;
   }
};

class DistanceMetric
{
protected:
   int Dim;
public:
   DistanceMetric(int D) { Dim = D; }
   virtual ~DistanceMetric() { }

   virtual void SetDim(int D) { Dim = D; }
   
   virtual void Distance(const Vector &x,
                         double &r) const = 0;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const = 0;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const = 0;
   
   static DistanceMetric *GetDistance(int Dim, int pnorm);
};

class L1Distance : public DistanceMetric
{
public:
   L1Distance(int D) : DistanceMetric(D) { };
   virtual ~L1Distance() { }

   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};

class L2Distance : public DistanceMetric
{
public:
   L2Distance(int D) : DistanceMetric(D) { };
   virtual ~L2Distance() { }

   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};

class LpDistance : public DistanceMetric
{
   const int p;
   const double pinv;
public:
   LpDistance(int D, int pnorm)
      : DistanceMetric(D),
        p(pnorm),
        pinv(1. / static_cast<double>(p))
   { };
   virtual ~LpDistance() { }
   
   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};

class KernelFiniteElement : public ScalarFiniteElement
{
public:
   KernelFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F) { }
   virtual ~KernelFiniteElement() { }

   // Converts integration rule to vector
   virtual void IntRuleToVec(const IntegrationPoint &ip,
                             Vector &vec) const;

   virtual bool IsCompact() const = 0;
   virtual const RBFFunction *Kernel() const = 0;

   // Get range of i,j,k indices that are nonzero for compact support
   virtual bool TensorIndexed() const { return false; }
   virtual void GetTensorIndices(const Vector &ip,
                                 int (&indices)[3][2]) const
   { MFEM_ABORT("GetTensorIndices(...)"); }
   virtual void GetTensorNumPoints(int (&tNumPoints)[3]) const
   { MFEM_ABORT("GetTensorNumPoints(...)"); }
   
   using FiniteElement::Project;
   
   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { ScalarLocalInterpolation(Trans, I, *this); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this); }
};

class RBFFiniteElement : public KernelFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable double r_scr, f_scr, df_scr, ddf_scr;
   mutable Vector x_scr, y_scr, dy_scr, dr_scr;
   mutable DenseMatrix ddr_scr;
   mutable int cInd[3][2];
#endif
   bool isCompact; // Is the RBF with the given h compact?
   int dimPoints[3];
   int numPointsD; // Number of points across the element in each D
   double delta; // Distance between points
   double h; // Shape parameter, approx number of points in 1d support radius
   double hPhys; // Shape parameter times distance between points times HNorm
   double hPhysInv; // Inverse hPhys
   double radPhys; // Radius adjusted by h
   const RBFFunction *rbf;
   const DistanceMetric *distance;
   void InitializeGeometry();
   
   virtual void DistanceVec(const int i,
                            const Vector &x,
                            Vector &y) const;
   
public:
   RBFFiniteElement(const int D,
                    const int numPointsD,
                    const double h,
                    const int rbfType,
                    const int distNorm,
                    const int intOrder);
   virtual ~RBFFiniteElement() { delete rbf; delete distance; }

   virtual bool TensorIndexed() const { return true; }
   virtual void GetCompactIndices(const Vector &ip,
                                  int (&indices)[3][2]) const;
   virtual void GetGlobalIndices(const Vector &ip,
                                 int (&indices)[3][2]) const;
   virtual void GetTensorIndices(const Vector &ip,
                                 int (&indices)[3][2]) const;
   virtual void GetTensorNumPoints(int (&tNumPoints)[3]) const
   {
      tNumPoints[0] = dimPoints[0];
      tNumPoints[1] = dimPoints[1];
      tNumPoints[2] = dimPoints[2];
   }
   
   virtual bool IsCompact() const { return isCompact; }
   virtual const RBFFunction *Kernel() const { return rbf; }
   
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &h) const;
};

class RKFiniteElement : public KernelFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable double f_scr;
   mutable Vector x_scr, y_scr, g_scr, c_scr, s_scr, p_scr, df_scr;
   mutable DenseMatrix q_scr, dq_scr, M_scr;
   mutable Vector dc_scr[3], dp_scr[3];
   mutable DenseMatrix dM_scr[3];
   mutable DenseMatrixInverse Minv_scr;
   mutable int cInd[3][2];
   mutable int dimPoints[3];
#endif
   int polyOrd, numPoly, numPoly1d;
   KernelFiniteElement *baseFE;

   virtual void GetPoly(const Vector &x,
                        Vector &p) const;
   virtual void GetDPoly(const Vector &x,
                         Vector &p,
                         Vector (&dp)[3]) const;
   virtual void GetG(Vector &g) const;
   virtual void GetM(const Vector &baseShape,
                     const IntegrationPoint &ip,
                     DenseMatrix &M) const;
   virtual void GetDM(const Vector &baseShape,
                      const DenseMatrix &baseDeriv,
                      const IntegrationPoint &ip,
                      DenseMatrix &M,
                      DenseMatrix (&dM)[3]) const;
   virtual void AddToM(const Vector &p,
                       const double &f,
                       DenseMatrix &M) const;
   virtual void AddToDM(const Vector &p,
                        const Vector (&dp)[3],
                        const double &f,
                        const Vector &df,
                        DenseMatrix (&dM)[3]) const;
   virtual void CalculateValues(const Vector &c,
                                const Vector &baseShape,
                                const IntegrationPoint &ip,
                                Vector &shape) const;
   virtual void CalculateDValues(const Vector &c,
                                 const Vector (&dc)[3],
                                 const Vector &baseShape,
                                 const DenseMatrix &baseDShape,
                                 const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const;
   virtual void DistanceVec(const int i,
                            const Vector &x,
                            Vector &y) const;
   
public:
   RKFiniteElement(const int D,
                   const int numPointsD,
                   const double h,
                   const int rbfType,
                   const int distNorm,
                   const int order,
                   const int intOrder);
   virtual ~RKFiniteElement() { delete baseFE; }
   
   virtual bool IsCompact() const { return baseFE->IsCompact(); }
   virtual const RBFFunction *Kernel() const { return baseFE->Kernel(); }
   
   static int GetNumPoly(int polyOrd, int dim);
   
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // Should put in a method to calculate shape and dshape simultaneously
};

} // namespace mfem

#endif
