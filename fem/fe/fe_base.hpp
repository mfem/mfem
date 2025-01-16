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

#ifndef MFEM_FE_BASE
#define MFEM_FE_BASE

#include "../intrules.hpp"
#include "../geom.hpp"
#include "../doftrans.hpp"

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
      ClosedGL        = 7,  ///< Closed GaussLegendre
      IntegratedGLL   = 8,  ///< Integrated GLL indicator functions
      NumBasisTypes   = 9   /**< Keep track of maximum types to prevent
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
      MFEM_VERIFY(Check(b_type) != Positive && b_type != IntegratedGLL,
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
         case ClosedGL:        return Quadrature1D::ClosedGL;
         case IntegratedGLL:   return Quadrature1D::GaussLegendre;
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
         case Quadrature1D::ClosedGL:        return ClosedGL;
      }
      return Invalid;
   }
   /// Check and convert a BasisType constant to a string identifier.
   static const char *Name(int b_type)
   {
      static const char *name[] =
      {
         "Gauss-Legendre", "Gauss-Lobatto", "Positive (Bernstein)",
         "Open uniform", "Closed uniform", "Open half uniform",
         "Serendipity", "Closed Gauss-Legendre",
         "Integrated Gauss-Lobatto indicator"
      };
      return name[Check(b_type)];
   }
   /// Check and convert a BasisType constant to a char basis identifier.
   static char GetChar(int b_type)
   {
      static const char ident[]
         = { 'g', 'G', 'P', 'u', 'U', 'o', 'S', 'c', 'i' };
      return ident[Check(b_type)];
   }
   /// Convert char basis identifier to a BasisType constant.
   static int GetType(char b_ident)
   {
      switch (b_ident)
      {
         case 'g': return GaussLegendre;
         case 'G': return GaussLobatto;
         case 's': return GaussLobatto;
         case 'P': return Positive;
         case 'u': return OpenUniform;
         case 'U': return ClosedUniform;
         case 'o': return OpenHalfUniform;
         case 'S': return Serendipity;
         case 'c': return ClosedGL;
         case 'i': return IntegratedGLL;
      }
      MFEM_ABORT("unknown BasisType identifier");
      return -1;
   }
};

/** @brief Structure representing the matrices/tensors needed to evaluate (in
    reference space) the values, gradients, divergences, or curls of a
    FiniteElement at the quadrature points of a given IntegrationRule. */
/** Objects of this type are typically created and owned by the respective
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
          are used to describe the "closed" and "open" 1D basis functions. */
      TENSOR,

      /** @brief Full multidimensional representation which does not use tensor
          product structure. The ordering of the degrees of freedom is the
          same as TENSOR, but the sizes of B and G are the same as FULL.*/
      LEXICOGRAPHIC_FULL
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
       - #nqpt x dim x #ndof, for vector elements,

       where

       - dim = dimension of the finite element reference space when #mode is
         FULL, and dim = 1 when #mode is TENSOR. */
   Array<real_t> B;

   /// Transpose of #B.
   /** The storage layout is column-major with dimensions:
       - #ndof x #nqpt, for scalar elements, or
       - #ndof x #nqpt x dim, for vector elements. */
   Array<real_t> Bt;

   /** @brief Gradients/divergences/curls of basis functions evaluated at
       quadrature points. */
   /** The storage layout is column-major with dimensions:
       - #nqpt x dim x #ndof, for scalar elements, or
       - #nqpt x #ndof, for H(div) vector elements, or
       - #nqpt x cdim x #ndof, for H(curl) vector elements,

       where

       - dim = dimension of the finite element reference space when #mode is
         FULL, and 1 when #mode is TENSOR,
       - cdim = 1/1/3 in 1D/2D/3D, respectively, when #mode is FULL, and cdim =
         1 when #mode is TENSOR. */
   Array<real_t> G;

   /// Transpose of #G.
   /** The storage layout is column-major with dimensions:
       - #ndof x #nqpt x dim, for scalar elements, or
       - #ndof x #nqpt, for H(div) vector elements, or
       - #ndof x #nqpt x cdim, for H(curl) vector elements. */
   Array<real_t> Gt;
};

/// Describes the function space on each element
class FunctionSpace
{
public:
   enum
   {
      Pk, ///< Polynomials of order k
      Qk, ///< Tensor products of polynomials of order k
      rQk ///< Refined tensor products of polynomials of order k
   };
};

class ElementTransformation;
class Coefficient;
class VectorCoefficient;
class MatrixCoefficient;

/// Abstract class for all finite elements.
class FiniteElement
{
protected:
   int dim;      ///< Dimension of reference space
   int vdim;     ///< Vector dimension of vector-valued basis functions
   int cdim;     ///< Dimension of curl for vector-valued basis functions
   Geometry::Type geom_type; ///< Geometry::Type of the reference element
   int func_space, range_type, map_type,
       deriv_type, deriv_range_type, deriv_map_type;
   mutable
   int dof,      ///< Number of degrees of freedom
       order;    ///< Order/degree of the shape functions
   mutable int orders[Geometry::MaxDim]; ///< Anisotropic orders
   IntegrationRule Nodes;
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix vshape; // Dof x Dim
#endif
   /// Container for all DofToQuad objects created by the FiniteElement.
   /** Multiple DofToQuad objects may be needed when different quadrature rules
       or different DofToQuad::Mode are used. */
   mutable Array<DofToQuad *> dof2quad_array;

public:
   /// Enumeration for range_type and deriv_range_type
   enum RangeType { UNKNOWN_RANGE_TYPE = -1, SCALAR, VECTOR };

   /** @brief Enumeration for MapType: defines how reference functions are
       mapped to physical space.

       A reference function $ \hat u(\hat x) $ can be mapped to a function
      $ u(x) $ on a general physical element in following ways:
       - $ x = T(\hat x) $ is the image of the reference point $ \hat x $
       - $ J = J(\hat x) $ is the Jacobian matrix of the transformation T
       - $ w = w(\hat x) = det(J) $ is the transformation weight factor for square J
       - $ w = w(\hat x) = det(J^t J)^{1/2} $ is the transformation weight factor in general
   */
   enum MapType
   {
      UNKNOWN_MAP_TYPE = -1, /**< Used to distinguish an unset MapType variable
                                  from the known values below. */
      VALUE,     /**< For scalar fields; preserves point values
                          $ u(x) = \hat u(\hat x) $ @anchor map_type_value */
      INTEGRAL,  /**< For scalar fields; preserves volume integrals
                          $ u(x) = (1/w) \hat u(\hat x) $ */
      H_DIV,     /**< For vector fields; preserves surface integrals of the
                          normal component $ u(x) = (J/w) \hat u(\hat x) $ */
      H_CURL     /**< For vector fields; preserves line integrals of the
                          tangential component
                          $ u(x) = J^{-t} \hat u(\hat x) $ (square J),
                          $ u(x) = J(J^t J)^{-1} \hat u(\hat x) $ (general J) */
   };

   /** @brief Enumeration for DerivType: defines which derivative method
       is implemented.

       Each FiniteElement class implements up to one type of derivative.  The
       value returned by GetDerivType() indicates which derivative method is
       implemented.
   */
   enum DerivType
   {
      NONE, ///< No derivatives implemented
      GRAD, ///< Implements CalcDShape methods
      DIV,  ///< Implements CalcDivShape methods
      CURL  ///< Implements CalcCurlShape methods
   };

   /** @brief Construct FiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */
   FiniteElement(int D, Geometry::Type G, int Do, int O,
                 int F = FunctionSpace::Pk);

   /// Returns the reference space dimension for the finite element.
   int GetDim() const { return dim; }

   /** @brief Returns the vector dimension for vector-valued finite elements,
       which is also the dimension of the interpolation operation. */
   int GetRangeDim() const { return vdim; }

   /// Returns the dimension of the curl for vector-valued finite elements.
   int GetCurlDim() const { return cdim; }

   /// Returns the Geometry::Type of the reference element.
   Geometry::Type GetGeomType() const { return geom_type; }

   /// Returns the number of degrees of freedom in the finite element.
   int GetDof() const { return dof; }

   /** @brief Returns the order of the finite element. In the case of
       anisotropic orders, returns the maximum order. */
   int GetOrder() const { return order; }

   /** @brief Returns true if the FiniteElement basis *may be using* different
       orders/degrees in different spatial directions. */
   bool HasAnisotropicOrders() const { return orders[0] != -1; }

   /// Returns an array containing the anisotropic orders/degrees.
   const int *GetAnisotropicOrders() const { return orders; }

   /// Returns the type of FunctionSpace on the element.
   int Space() const { return func_space; }

   /// Returns the FiniteElement::RangeType of the element, one of {SCALAR, VECTOR}.
   int GetRangeType() const { return range_type; }

   /** @brief Returns the FiniteElement::RangeType of the element derivative, either
       SCALAR or VECTOR. */
   int GetDerivRangeType() const { return deriv_range_type; }

   /** @brief Returns the FiniteElement::MapType of the element describing how reference
       functions are mapped to physical space, one of {VALUE, INTEGRAL
       H_DIV, H_CURL}. */
   int GetMapType() const { return map_type; }

   /** @brief Returns the FiniteElement::DerivType of the element describing the
       spatial derivative method implemented, one of {NONE, GRAD,
       DIV, CURL}. */
   int GetDerivType() const { return deriv_type; }

   /** @brief Returns the FiniteElement::DerivType of the element describing how
       reference function derivatives are mapped to physical space, one of {VALUE,
       INTEGRAL, H_DIV, H_CURL}. */
   int GetDerivMapType() const { return deriv_map_type; }

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** The size (#dof) of the result Vector @a shape must be set in advance. */
   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const = 0;

   /** @brief Evaluate the values of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** The size (#dof) of the result Vector @a shape must be set in advance. */
   void CalcPhysShape(ElementTransformation &Trans, Vector &shape) const;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#dof x #dim) of @a dshape must be set in
       advance.  */
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const = 0;

   /** @brief Evaluate the gradients of all shape functions of a scalar finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a dshape contains the derivatives of
       one shape function. The size (#dof x SDim) of @a dshape must be set in
       advance, where SDim >= #dim is the physical space dimension as described
       by @a Trans. */
   void CalcPhysDShape(ElementTransformation &Trans, DenseMatrix &dshape) const;

   /// Get a const reference to the nodes of the element
   const IntegrationRule & GetNodes() const { return Nodes; }

   /** @brief Evaluate the Hessians of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a Hessian contains upper triangular
       part of the Hessian of one shape function.
       The order in 2D is {u_xx, u_xy, u_yy}.
       The size (#dof x (#dim (#dim+1)/2) of @a Hessian must be set in advance.*/
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &Hessian) const;

   /** @brief Evaluate the Hessian of all shape functions of a scalar finite
       element in physical space at the given point @a ip. */
   /** The size (#dof, #dim*(#dim+1)/2) of @a Hessian must be set in advance. */
   void CalcPhysHessian(ElementTransformation &Trans,
                        DenseMatrix& Hessian) const;

   /** @brief Evaluate the Laplacian of all shape functions of a scalar finite
       element in physical space at the given point @a ip. */
   /** The size (#dof) of @a Laplacian must be set in advance. */
   void CalcPhysLaplacian(ElementTransformation &Trans,
                          Vector& Laplacian) const;

   /** @brief Evaluate the Laplacian of all shape functions of a scalar finite
       element in physical space at the given point @a ip. */
   /** The size (#dof) of @a Laplacian must be set in advance. */
   void CalcPhysLinLaplacian(ElementTransformation &Trans,
                             Vector& Laplacian) const;

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#dof x #dim) of @a shape must be set
       in advance. */
   virtual void CalcVShape(const IntegrationPoint &ip,
                           DenseMatrix &shape) const;

   /** @brief Evaluate the values of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a shape contains the components of
       one vector shape function. The size (#dof x SDim) of @a shape must be set
       in advance, where SDim >= #dim is the physical space dimension as
       described by @a Trans. */
   virtual void CalcVShape(ElementTransformation &Trans,
                           DenseMatrix &shape) const;

   /// Equivalent to the CalcVShape() method with the same arguments.
   void CalcPhysVShape(ElementTransformation &Trans, DenseMatrix &shape) const
   { CalcVShape(Trans, shape); }

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#dof) of the result Vector @a divshape must be set in advance.
    */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in physical space at the point described by @a Trans. */
   /** The size (#dof) of the result Vector @a divshape must be set in advance.
    */
   void CalcPhysDivShape(ElementTransformation &Trans, Vector &divshape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #dim = 3 and
       CDim = 1 for #dim = 2. */
   virtual void CalcCurlShape(const IntegrationPoint &ip,
                              DenseMatrix &curl_shape) const;

   /** @brief Evaluate the curl of all shape functions of a *vector* finite
       element in physical space at the point described by @a Trans. */
   /** Each row of the result DenseMatrix @a curl_shape contains the components
       of the curl of one vector shape function. The size (#dof x CDim) of
       @a curl_shape must be set in advance, where CDim = 3 for #dim = 3 and
       CDim = 1 for #dim = 2. */
   virtual void CalcPhysCurlShape(ElementTransformation &Trans,
                                  DenseMatrix &curl_shape) const;

   /** @brief Get the dofs associated with the given @a face.
       @a *dofs is set to an internal array of the local dofc on the
       face, while *ndofs is set to the number of dofs on that face.
   */
   virtual void GetFaceDofs(int face, int **dofs, int *ndofs) const;

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
       FiniteElement::MapType. */
   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const;

   /** @brief Given a coefficient and a transformation, compute its projection
       (approximation) in the local finite dimensional space in terms
       of the degrees of freedom. */
   /** The approximation used to project is usually local interpolation of
       degrees of freedom. The derived class could use other methods not
       implemented yet, e.g. local L2 projection. */
   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;

   /** @brief Given a vector coefficient and a transformation, compute its
       projection (approximation) in the local finite dimensional space
       in terms of the degrees of freedom. (VectorFiniteElements) */
   /** The approximation used to project is usually local interpolation of
       degrees of freedom. The derived class could use other methods not
       implemented yet, e.g. local L2 projection. */
   virtual void Project(VectorCoefficient &vc,
                        ElementTransformation &Trans, Vector &dofs) const;

   /** @brief Given a vector of values at the finite element nodes and a
       transformation, compute its projection (approximation) in the local
       finite dimensional space in terms of the degrees of freedom. Valid for
       VectorFiniteElements. */
   virtual void ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                 Vector &dofs) const;

   /** @brief Given a matrix coefficient and a transformation, compute an
       approximation ("projection") in the local finite dimensional space in
       terms of the degrees of freedom. For VectorFiniteElements, the rows of
       the coefficient are projected in the vector space. */
   virtual void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   /** @brief Project a delta function centered on the given @a vertex in
       the local finite dimensional space represented by the @a dofs. */
   virtual void ProjectDelta(int vertex, Vector &dofs) const;

   /** @brief Compute the embedding/projection matrix from the given
       FiniteElement onto 'this' FiniteElement. The ElementTransformation is
       included to support cases when the projection depends on it. */
   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   /** @brief Compute the discrete gradient matrix from the given FiniteElement
       onto 'this' FiniteElement. The ElementTransformation is included to
       support cases when the matrix depends on it. */
   virtual void ProjectGrad(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &grad) const;

   /** @brief Compute the discrete curl matrix from the given FiniteElement onto
       'this' FiniteElement. The ElementTransformation is included to support
       cases when the matrix depends on it. */
   virtual void ProjectCurl(const FiniteElement &fe,
                            ElementTransformation &Trans,
                            DenseMatrix &curl) const;

   /** @brief Compute the discrete divergence matrix from the given
       FiniteElement onto 'this' FiniteElement. The ElementTransformation is
       included to support cases when the matrix depends on it. */
   virtual void ProjectDiv(const FiniteElement &fe,
                           ElementTransformation &Trans,
                           DenseMatrix &div) const;

   /** @brief Return a DofToQuad structure corresponding to the given
       IntegrationRule using the given DofToQuad::Mode. */
   /** See the documentation for DofToQuad for more details. */
   virtual const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                         DofToQuad::Mode mode) const;


   /** @brief Return the mapping from lexicographic face DOFs to lexicographic
       element DOFs for the given local face @a face_id. */
   /** Given the @a ith DOF (lexicographically ordered) on the face referenced
       by @a face_id, face_map[i] gives the corresponding index of the DOF in
       the element (also lexicographically ordered).

       @note For L2 spaces, this is only well-defined for "closed" bases such as
       the Gauss-Lobatto or Bernstein (positive) bases.

       @warning GetFaceMap() is currently only implemented for tensor-product
       (quadrilateral and hexahedral) elements. Its functionality may change
       when simplex elements are supported in the future. */
   virtual void GetFaceMap(const int face_id, Array<int> &face_map) const;

   /** @brief Return a DoF transformation object for this particular type of
       basis.
   */
   virtual const StatelessDofTransformation *GetDofTransformation() const
   { return NULL; }

   /// Deconstruct the FiniteElement
   virtual ~FiniteElement();

   /** @brief Return true if the BasisType of @a b_type is closed
       (has Quadrature1D points on the boundary). */
   static bool IsClosedType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckClosed(q_type) != Quadrature1D::Invalid));
   }

   /** @brief Return true if the BasisType of @a b_type is open
       (doesn't have Quadrature1D points on the boundary). */
   static bool IsOpenType(int b_type)
   {
      const int q_type = BasisType::GetQuadrature1D(b_type);
      return ((q_type != Quadrature1D::Invalid) &&
              (Quadrature1D::CheckOpen(q_type) != Quadrature1D::Invalid));
   }

   /** @brief Ensure that the BasisType of @a b_type is closed
       (has Quadrature1D points on the boundary). */
   static int VerifyClosed(int b_type)
   {
      MFEM_VERIFY(IsClosedType(b_type),
                  "invalid closed basis type: " << b_type);
      return b_type;
   }

   /** @brief Ensure that the BasisType of @a b_type is open
       (doesn't have Quadrature1D points on the boundary). */
   static int VerifyOpen(int b_type)
   {
      MFEM_VERIFY(IsOpenType(b_type), "invalid open basis type: " << b_type);
      return b_type;
   }

   /** @brief Ensure that the BasisType of @a b_type nodal
       (satisfies the interpolation property). */
   static int VerifyNodal(int b_type)
   {
      return BasisType::CheckNodal(b_type);
   }
};

/** @brief Class for finite elements with basis functions
    that return scalar values. */
class ScalarFiniteElement : public FiniteElement
{
protected:
   static const ScalarFiniteElement &CheckScalarFE(const FiniteElement &fe)
   {
      MFEM_VERIFY(fe.GetRangeType() == SCALAR,
                  "'fe' must be a ScalarFiniteElement");
      return static_cast<const ScalarFiniteElement &>(fe);
   }

public:
   /** @brief Construct ScalarFiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */
   ScalarFiniteElement(int D, Geometry::Type G, int Do, int O,
                       int F = FunctionSpace::Pk)
      : FiniteElement(D, G, Do, O, F)
   { deriv_type = GRAD; deriv_range_type = VECTOR; deriv_map_type = H_CURL; }

   /** @brief Set the FiniteElement::MapType of the element to either VALUE or
       INTEGRAL. Also sets the FiniteElement::DerivType to GRAD if the
       FiniteElement::MapType is VALUE. */
   virtual void SetMapType(int M)
   {
      MFEM_VERIFY(M == VALUE || M == INTEGRAL, "unknown MapType");
      map_type = M;
      deriv_type = (M == VALUE) ? GRAD : NONE;
   }

   /** @brief Get the matrix @a I that defines nodal interpolation
       @a between this element and the refined element @a fine_fe. */
   void NodalLocalInterpolation(ElementTransformation &Trans,
                                DenseMatrix &I,
                                const ScalarFiniteElement &fine_fe) const;

   /** @brief Get matrix @a I "Interpolation" defined through local
       L2-projection in the space defined by the @a fine_fe. */
   /** If the "fine" elements cannot represent all basis functions of the
       "coarse" element, then boundary values from different sub-elements are
       generally different. */
   void ScalarLocalInterpolation(ElementTransformation &Trans,
                                 DenseMatrix &I,
                                 const ScalarFiniteElement &fine_fe) const;

   /** @brief Get restriction matrix @a R defined through local L2-projection
        in the space defined by the @a coarse_fe. */
   /** If the "fine" elements cannot represent all basis functions of the
       "coarse" element, then boundary values from different sub-elements are
       generally different. */
   void ScalarLocalL2Restriction(ElementTransformation &Trans,
                                 DenseMatrix &R,
                                 const ScalarFiniteElement &coarse_fe) const;
};

/// Class for standard nodal finite elements.
class NodalFiniteElement : public ScalarFiniteElement
{
private:
   /// Create and cache the LEXICOGRAPHIC_FULL DofToQuad maps.
   void CreateLexicographicFullMap(const IntegrationRule &ir) const;
protected:
   Array<int> lex_ordering;
   void ProjectCurl_2D(const FiniteElement &fe,
                       ElementTransformation &Trans,
                       DenseMatrix &curl) const;

public:
   /** @brief Construct NodalFiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
   */
   NodalFiniteElement(int D, Geometry::Type G, int Do, int O,
                      int F = FunctionSpace::Pk)
      : ScalarFiniteElement(D, G, Do, O, F) { }

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const override;

   void GetLocalInterpolation(ElementTransformation &Trans,
                              DenseMatrix &I) const override
   { NodalLocalInterpolation(Trans, I, *this); }

   void GetLocalRestriction(ElementTransformation &Trans,
                            DenseMatrix &R) const override;

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   { CheckScalarFE(fe).NodalLocalInterpolation(Trans, I, *this); }

   void Project(Coefficient &coeff,
                ElementTransformation &Trans, Vector &dofs) const override;

   void Project(VectorCoefficient &vc,
                ElementTransformation &Trans, Vector &dofs) const override;

   // (mc.height x mc.width) @ DOFs -> (Dof x mc.width x mc.height) in dofs
   void ProjectMatrixCoefficient(
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const override;

   void Project(const FiniteElement &fe, ElementTransformation &Trans,
                DenseMatrix &I) const override;

   void ProjectGrad(const FiniteElement &fe,
                    ElementTransformation &Trans,
                    DenseMatrix &grad) const override;

   void ProjectDiv(const FiniteElement &fe,
                   ElementTransformation &Trans,
                   DenseMatrix &div) const override;

   /** @brief Get an Array<int> that maps lexicographically ordered indices to
       the indices of the respective nodes/dofs/basis functions.

       Lexicographic ordering of nodes is defined in terms of reference-space
       coordinates (x,y,z). Lexicographically ordered nodes are listed first in
       order of increasing x-coordinate, and then in order of increasing
       y-coordinate, and finally in order of increasing z-coordinate.

       For example, the six nodes of a quadratic triangle are lexicographically
       ordered as follows:

       5
       |\
       3 4
       |  \
       0-1-2

       The resulting array may be empty if the DOFs are already ordered
       lexicographically, or if the finite element does not support creating
       this permutation. The array returned is the same as the array given by
       TensorBasisElement::GetDofMap, but it is also available for non-tensor
       elements. */
   const Array<int> &GetLexicographicOrdering() const { return lex_ordering; }
};

/** @brief Intermediate class for finite elements whose basis functions return
    vector values. */
class VectorFiniteElement : public FiniteElement
{
   // Hide the scalar functions CalcShape and CalcDShape.
private:
   /// Overrides the scalar CalcShape function to print an error.
   void CalcShape(const IntegrationPoint &ip,
                  Vector &shape) const override;

   /// Overrides the scalar CalcDShape function to print an error.
   void CalcDShape(const IntegrationPoint &ip,
                   DenseMatrix &dshape) const override;

protected:
   bool is_nodal;
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix JtJ;
   mutable DenseMatrix curlshape, curlshape_J;
#endif
   void SetDerivMembers();

   void CalcVShape_RT(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   void CalcVShape_ND(ElementTransformation &Trans,
                      DenseMatrix &shape) const;

   /** @brief Project a vector coefficient onto the RT basis functions
       @param nk    Face normal vectors for this element type
       @param d2n   Offset into nk for each degree of freedom
       @param vc    Vector coefficient to be projected
       @param Trans Transformation from reference to physical coordinates
       @param dofs  Expansion coefficients for the approximation of vc
   */
   void Project_RT(const real_t *nk, const Array<int> &d2n,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   /// Projects the vector of values given at FE nodes to RT space
   /** Project vector values onto the RT basis functions
       @param nk    Face normal vectors for this element type
       @param d2n   Offset into nk for each degree of freedom
       @param vc    Vector values at each interpolation point
       @param Trans Transformation from reference to physical coordinates
       @param dofs  Expansion coefficients for the approximation of vc
   */
   void Project_RT(const real_t *nk, const Array<int> &d2n,
                   Vector &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   /// Project the rows of the matrix coefficient in an RT space
   void ProjectMatrixCoefficient_RT(
      const real_t *nk, const Array<int> &d2n,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   /** @brief Project vector-valued basis functions onto the RT basis functions
       @param nk    Face normal vectors for this element type
       @param d2n   Offset into nk for each degree of freedom
       @param fe    Vector-valued finite element basis
       @param Trans Transformation from reference to physical coordinates
       @param I     Expansion coefficients for the approximation of each basis
                    function

       Note: If the FiniteElement, fe, is scalar-valued the projection will
             assume that a FiniteElementSpace is being used to define a vector
             field using the scalar basis functions for each component of the
             vector field.
   */
   void Project_RT(const real_t *nk, const Array<int> &d2n,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   // rotated gradient in 2D
   void ProjectGrad_RT(const real_t *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   // Compute the curl as a discrete operator from ND FE (fe) to ND FE (this).
   // The natural FE for the range is RT, so this is an approximation.
   void ProjectCurl_ND(const real_t *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   void ProjectCurl_RT(const real_t *nk, const Array<int> &d2n,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &curl) const;

   /** @brief Project a vector coefficient onto the ND basis functions
       @param tk    Edge tangent vectors for this element type
       @param d2t   Offset into tk for each degree of freedom
       @param vc    Vector coefficient to be projected
       @param Trans Transformation from reference to physical coordinates
       @param dofs  Expansion coefficients for the approximation of vc
   */
   void Project_ND(const real_t *tk, const Array<int> &d2t,
                   VectorCoefficient &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   /// Projects the vector of values given at FE nodes to ND space
   /** Project vector values onto the ND basis functions
       @param tk    Edge tangent vectors for this element type
       @param d2t   Offset into tk for each degree of freedom
       @param vc    Vector values at each interpolation point
       @param Trans Transformation from reference to physical coordinates
       @param dofs  Expansion coefficients for the approximation of vc
   */
   void Project_ND(const real_t *tk, const Array<int> &d2t,
                   Vector &vc, ElementTransformation &Trans,
                   Vector &dofs) const;

   /// Project the rows of the matrix coefficient in an ND space
   void ProjectMatrixCoefficient_ND(
      const real_t *tk, const Array<int> &d2t,
      MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const;

   /** @brief Project vector-valued basis functions onto the ND basis functions
       @param tk    Edge tangent vectors for this element type
       @param d2t   Offset into tk for each degree of freedom
       @param fe    Vector-valued finite element basis
       @param Trans Transformation from reference to physical coordinates
       @param I     Expansion coefficients for the approximation of each basis
                    function

       Note: If the FiniteElement, fe, is scalar-valued the projection will
             assume that a FiniteElementSpace is being used to define a vector
             field using the scalar basis functions for each component of the
             vector field.
   */
   void Project_ND(const real_t *tk, const Array<int> &d2t,
                   const FiniteElement &fe, ElementTransformation &Trans,
                   DenseMatrix &I) const;

   void ProjectGrad_ND(const real_t *tk, const Array<int> &d2t,
                       const FiniteElement &fe, ElementTransformation &Trans,
                       DenseMatrix &grad) const;

   void LocalL2Projection_RT(const VectorFiniteElement &cfe,
                             ElementTransformation &Trans,
                             DenseMatrix &I) const;

   void LocalInterpolation_RT(const VectorFiniteElement &cfe,
                              const real_t *nk, const Array<int> &d2n,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalL2Projection_ND(const VectorFiniteElement &cfe,
                             ElementTransformation &Trans,
                             DenseMatrix &I) const;

   void LocalInterpolation_ND(const VectorFiniteElement &cfe,
                              const real_t *tk, const Array<int> &d2t,
                              ElementTransformation &Trans,
                              DenseMatrix &I) const;

   void LocalRestriction_RT(const real_t *nk, const Array<int> &d2n,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   void LocalRestriction_ND(const real_t *tk, const Array<int> &d2t,
                            ElementTransformation &Trans,
                            DenseMatrix &R) const;

   static const VectorFiniteElement &CheckVectorFE(const FiniteElement &fe)
   {
      if (fe.GetRangeType() != VECTOR)
      { mfem_error("'fe' must be a VectorFiniteElement"); }
      return static_cast<const VectorFiniteElement &>(fe);
   }

public:
   VectorFiniteElement(int D, Geometry::Type G, int Do, int O, int M,
                       int F = FunctionSpace::Pk);
};

/// @brief Class for computing 1D special polynomials and their associated basis
/// functions
class Poly_1D
{
public:
   /// One-dimensional basis evaluation type
   enum EvalType
   {
      ChangeOfBasis = 0, ///< Use change of basis, O(p^2) Evals
      Barycentric   = 1, ///< Use barycentric Lagrangian interpolation, O(p) Evals
      Positive      = 2, ///< Fast evaluation of Bernstein polynomials
      Integrated    = 3, ///< Integrated indicator functions (cf. Gerritsma)
      NumEvalTypes  = 4  ///< Keep count of the number of eval types
   };

   /// @brief Class for evaluating 1D nodal, positive (Bernstein), or integrated
   /// (Gerritsma) bases.
   class Basis
   {
   private:
      EvalType etype; ///< Determines how the basis functions should be evaluated.
      DenseMatrixInverse Ai;
      mutable Vector x, w;
      /// The following data members are used for "integrated basis type", which
      /// is defined in terms of nodal basis of one degree higher.
      ///@{
      mutable Vector u_aux, d_aux, d2_aux;
      ///@}
      /// @brief An auxiliary nodal basis used to evaluate the integrated basis.
      /// This member variable is NULL whenever etype != Integrated.
      Basis *auxiliary_basis;
      /// Should the integrated basis functions be scaled? See ScaleIntegrated.
      bool scale_integrated;

   public:
      /// Create a nodal or positive (Bernstein) basis of degree @a p
      Basis(const int p, const real_t *nodes, EvalType etype = Barycentric);
      /// Evaluate the basis functions at point @a x in [0,1]
      void Eval(const real_t x, Vector &u) const;
      /// @brief Evaluate the basis functions and their derivatives at point @a
      /// x in [0,1]
      void Eval(const real_t x, Vector &u, Vector &d) const;
      /// @brief Evaluate the basis functions and their first two derivatives at
      /// point @a x in [0,1]
      void Eval(const real_t x, Vector &u, Vector &d, Vector &d2) const;
      /// @brief Evaluate the "integrated" basis type using pre-computed closed
      /// basis derivatives.
      ///
      /// This basis is given by the negative partial sum of the corresponding
      /// closed basis derivatives. The closed basis derivatives are given by @a
      /// d, and the result is stored in @a i.
      void EvalIntegrated(const Vector &d, Vector &i) const;
      /// @brief Set whether the "integrated" basis should be scaled by the
      /// subcell sizes. Has no effect for non-integrated bases.
      ///
      /// Generally, this should be true for mfem::FiniteElement::MapType VALUE
      /// and false for all other map types. If this option is enabled, the
      /// basis functions will be scaled by the widths of the subintervals, so
      /// that the basis functions represent mean values. Otherwise, the basis
      /// functions represent integrated values.
      void ScaleIntegrated(bool scale_integrated_);
      /// Returns true if the basis is "integrated", false otherwise.
      bool IsIntegratedType() const { return etype == Integrated; }
      ~Basis();
   };

private:
   typedef std::map<int, Array<real_t*>*> PointsMap;
   typedef std::map<int, Array<Basis*>*> BasisMap;

   MemoryType h_mt;
   PointsMap points_container;
   BasisMap  bases_container;

   static Array2D<int> binom;

   static void CalcMono(const int p, const real_t x, real_t *u);
   static void CalcMono(const int p, const real_t x, real_t *u, real_t *d);

   static void CalcChebyshev(const int p, const real_t x, real_t *u);
   static void CalcChebyshev(const int p, const real_t x, real_t *u, real_t *d);
   static void CalcChebyshev(const int p, const real_t x, real_t *u, real_t *d,
                             real_t *dd);

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
   const real_t *GetPoints(const int p, const int btype);

   /// Get coordinates of an open (GaussLegendre) set of points if degree @a p
   const real_t *OpenPoints(const int p,
                            const int btype = BasisType::GaussLegendre)
   { return GetPoints(p, btype); }

   /// Get coordinates of a closed (GaussLegendre) set of points if degree @a p
   const real_t *ClosedPoints(const int p,
                              const int btype = BasisType::GaussLobatto)
   { return GetPoints(p, btype); }

   /** @brief Get a Poly_1D::Basis object of the given degree and BasisType,
       @a btype.

       @param[in] p      The polynomial degree of the basis.
       @param[in] btype  The BasisType.

       @return A reference to an object of type Poly_1D::Basis that represents
               the requested basis type. */
   Basis &GetBasis(const int p, const int btype);

   /** @brief Evaluate the values of a hierarchical 1D basis at point x
       hierarchical = k-th basis function is degree k polynomial */
   static void CalcBasis(const int p, const real_t x, real_t *u)
   // { CalcMono(p, x, u); }
   // Bernstein basis is not hierarchical --> does not work for triangles
   //  and tetrahedra
   // { CalcBernstein(p, x, u); }
   // { CalcLegendre(p, x, u); }
   { CalcChebyshev(p, x, u); }

   /** @brief Evaluate the values of a hierarchical 1D basis at point x
       hierarchical = k-th basis function is degree k polynomial */
   static void CalcBasis(const int p, const real_t x, Vector &u)
   { CalcBasis(p, x, u.GetData()); }

   /// Evaluate the values and derivatives of a hierarchical 1D basis at point @a x
   static void CalcBasis(const int p, const real_t x, real_t *u, real_t *d)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d); }

   /** @brief Evaluate the values and derivatives of a hierarchical 1D basis at
       point @a x. */
   static void CalcBasis(const int p, const real_t x, Vector &u, Vector &d)
   { CalcBasis(p, x, u.GetData(), d.GetData()); }

   /// Evaluate the values, derivatives and second derivatives of a hierarchical 1D basis at point x
   static void CalcBasis(const int p, const real_t x, real_t *u, real_t *d,
                         real_t *dd)
   // { CalcMono(p, x, u, d); }
   // { CalcBernstein(p, x, u, d); }
   // { CalcLegendre(p, x, u, d); }
   { CalcChebyshev(p, x, u, d, dd); }

   /** @brief Evaluate the values, derivatives and second derivatives of a
       hierarchical 1D basis at point @a x. */
   static void CalcBasis(const int p, const real_t x, Vector &u, Vector &d,
                         Vector &dd)
   { CalcBasis(p, x, u.GetData(), d.GetData(), dd.GetData()); }

   /// Evaluate a representation of a Delta function at point x
   static real_t CalcDelta(const int p, const real_t x)
   { return pow(x, (real_t) p); }

   /** @brief Compute the points for the Chebyshev polynomials of order @a p
       and place them in the already allocated @a x array. */
   static void ChebyshevPoints(const int p, real_t *x);

   /** @brief Compute the @a p terms in the expansion of the binomial (x + y)^p
       and store them in the already allocated @a u array. */
   static void CalcBinomTerms(const int p, const real_t x, const real_t y,
                              real_t *u);
   /** @brief Compute the terms in the expansion of the binomial (x + y)^p and
       their derivatives with respect to x assuming that dy/dx = -1.  Store the
       results in the already allocated @a u and @a d arrays.*/
   static void CalcBinomTerms(const int p, const real_t x, const real_t y,
                              real_t *u, real_t *d);
   /** @brief Compute the derivatives (w.r.t. x) of the terms in the expansion
       of the binomial (x + y)^p assuming that dy/dx = -1.  Store the results
       in the already allocated @a d array.*/
   static void CalcDBinomTerms(const int p, const real_t x, const real_t y,
                               real_t *d);

   /** @brief Compute the values of the Bernstein basis functions of order
       @a p at coordinate @a x and store the results in the already allocated
       @a u array. */
   static void CalcBernstein(const int p, const real_t x, real_t *u)
   { CalcBinomTerms(p, x, 1. - x, u); }

   /** @brief Compute the values of the Bernstein basis functions of order
       @a p at coordinate @a x and store the results in the already allocated
       @a u array. */
   static void CalcBernstein(const int p, const real_t x, Vector &u)
   { CalcBernstein(p, x, u.GetData()); }

   /** @brief Compute the values and derivatives of the Bernstein basis functions
       of order @a p at coordinate @a x and store the results in the already allocated
       @a u and @a d arrays. */
   static void CalcBernstein(const int p, const real_t x, real_t *u, real_t *d)
   { CalcBinomTerms(p, x, 1. - x, u, d); }

   /** @brief Compute the values and derivatives of the Bernstein basis
       functions of order @a p at coordinate @a x and store the results in the
       already allocated @a u and @a d arrays. */
   static void CalcBernstein(const int p, const real_t x, Vector &u, Vector &d)
   { CalcBernstein(p, x, u.GetData(), d.GetData()); }

   static void CalcLegendre(const int p, const real_t x, real_t *u);
   static void CalcLegendre(const int p, const real_t x, real_t *u, real_t *d);

   ~Poly_1D();
};

extern MFEM_EXPORT Poly_1D poly1d;

/// An element defined as an ND tensor product of 1D elements on a segment,
/// square, or cube
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

   const Poly_1D::Basis &GetBasis1D() const { return basis1d; }

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

   static const DofToQuad &GetTensorDofToQuad(
      const FiniteElement &fe, const IntegrationRule &ir,
      DofToQuad::Mode mode, const Poly_1D::Basis &basis, bool closed,
      Array<DofToQuad*> &dof2quad_array);
};

class NodalTensorFiniteElement : public NodalFiniteElement,
   public TensorBasisElement
{
public:
   NodalTensorFiniteElement(const int dims, const int p, const int btype,
                            const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const override;

   void SetMapType(const int map_type_) override;

   void GetTransferMatrix(const FiniteElement &fe,
                          ElementTransformation &Trans,
                          DenseMatrix &I) const override
   {
      if (basis1d.IsIntegratedType())
      {
         CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this);
      }
      else
      {
         NodalFiniteElement::GetTransferMatrix(fe, Trans, I);
      }
   }

   void GetFaceMap(const int face_id, Array<int> &face_map) const override;
};

class VectorTensorFiniteElement : public VectorFiniteElement,
   public TensorBasisElement
{
private:
   mutable Array<DofToQuad*> dof2quad_array_open;

protected:
   Poly_1D::Basis &obasis1d;

public:
   VectorTensorFiniteElement(const int dims, const int d, const int p,
                             const int cbtype, const int obtype,
                             const int M, const DofMapType dmtype);

   // For 1D elements: there is only an "open basis", no "closed basis"
   VectorTensorFiniteElement(const int dims, const int d, const int p,
                             const int obtype, const int M,
                             const DofMapType dmtype);

   const DofToQuad &GetDofToQuad(const IntegrationRule &ir,
                                 DofToQuad::Mode mode) const override
   {
      return (mode == DofToQuad::TENSOR) ?
             GetTensorDofToQuad(*this, ir, mode, basis1d, true, dof2quad_array) :
             FiniteElement::GetDofToQuad(ir, mode);
   }

   const DofToQuad &GetDofToQuadOpen(const IntegrationRule &ir,
                                     DofToQuad::Mode mode) const
   {
      MFEM_VERIFY(mode == DofToQuad::TENSOR, "invalid mode requested");
      return GetTensorDofToQuad(*this, ir, mode, obasis1d, false,
                                dof2quad_array_open);
   }

   virtual ~VectorTensorFiniteElement();
};

void InvertLinearTrans(ElementTransformation &trans,
                       const IntegrationPoint &pt, Vector &x);

} // namespace mfem

#endif
