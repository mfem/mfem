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

#ifndef MFEM_ELEMENTTRANSFORM
#define MFEM_ELEMENTTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "fe.hpp"

namespace mfem
{

class ElementTransformation
{
protected:
   const IntegrationPoint *IntPoint;
   DenseMatrix dFdx, adjJ, invJ;
   DenseMatrix d2Fdx2;
   double Wght;
   int EvalState;
   enum StateMasks
   {
      JACOBIAN_MASK = 1,
      WEIGHT_MASK   = 2,
      ADJUGATE_MASK = 4,
      INVERSE_MASK  = 8,
      HESSIAN_MASK  = 16
   };
   Geometry::Type geom;

   /** @brief Evaluate the Jacobian of the transformation at the IntPoint and
       store it in dFdx. */
   virtual const DenseMatrix &EvalJacobian() = 0;

   /** @brief Evaluate the Hessian of the transformation at the IntPoint and
       store it in d2Fdx2. */
   virtual const DenseMatrix &EvalHessian() = 0;

   double EvalWeight();
   const DenseMatrix &EvalAdjugateJ();
   const DenseMatrix &EvalInverseJ();

public:

   /** This enumeration declares the values stored in
       ElementTransformation::ElementType and indicates which group of objects
       the index stored in ElementTransformation::ElementNo refers:

       | ElementType | Range of ElementNo
       +-------------+-------------------------
       | ELEMENT     | [0, Mesh::GetNE()     )
       | BDR_ELEMENT | [0, Mesh::GetNBE()    )
       | EDGE        | [0, Mesh::GetNEdges() )
       | FACE        | [0, Mesh::GetNFaces() )
       | BDR_FACE    | [0, Mesh::GetNBE()    )
   */
   enum
   {
      ELEMENT     = 1,
      BDR_ELEMENT = 2,
      EDGE        = 3,
      FACE        = 4,
      BDR_FACE    = 5
   };

   int Attribute, ElementNo, ElementType;

   ElementTransformation();

   /** @brief Set the integration point @a ip that weights and Jacobians will
       be evaluated at. */
   void SetIntPoint(const IntegrationPoint *ip)
   { IntPoint = ip; EvalState = 0; }

   /** @brief Get a const reference to the currently set integration point.  This
       will return NULL if no integration point is set. */
   const IntegrationPoint &GetIntPoint() { return *IntPoint; }

   /** @brief Transform integration point from reference coordinates to
       physical coordinates and store them in the vector. */
   virtual void Transform(const IntegrationPoint &, Vector &) = 0;

   /** @brief Transform all the integration points from the integration rule
       from reference coordinates to physical
       coordinates and store them as column vectors in the matrix. */
   virtual void Transform(const IntegrationRule &, DenseMatrix &) = 0;

   /** @brief Transform all the integration points from the column vectors
       of @a matrix from reference coordinates to physical
       coordinates and store them as column vectors in @a result. */
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result) = 0;

   /** @brief Return the Jacobian matrix of the transformation at the currently
       set IntegrationPoint, using the method SetIntPoint(). */
   /** The dimensions of the Jacobian matrix are physical-space-dim by
       reference-space-dim. The first column contains the x derivatives of the
       transformation, the second -- the y derivatives, etc. */
   const DenseMatrix &Jacobian()
   { return (EvalState & JACOBIAN_MASK) ? dFdx : EvalJacobian(); }


   /** @brief Return the Hessian matrix of the transformation at the currently
       set IntegrationPoint, using the method SetIntPoint(). */
   const DenseMatrix &Hessian()
   { return (EvalState & HESSIAN_MASK) ? d2Fdx2 : EvalHessian(); }

   /** @brief Return the weight of the Jacobian matrix of the transformation
       at the currently set IntegrationPoint.
       The Weight evaluates to \f$ \sqrt{\lvert J^T J \rvert} \f$. */
   double Weight() { return (EvalState & WEIGHT_MASK) ? Wght : EvalWeight(); }

   /** @brief Return the adjugate of the Jacobian matrix of the transformation
        at the currently set IntegrationPoint. */
   const DenseMatrix &AdjugateJacobian()
   { return (EvalState & ADJUGATE_MASK) ? adjJ : EvalAdjugateJ(); }

   /** @brief Return the inverse of the Jacobian matrix of the transformation
        at the currently set IntegrationPoint. */
   const DenseMatrix &InverseJacobian()
   { return (EvalState & INVERSE_MASK) ? invJ : EvalInverseJ(); }

   /// Return the order of the current element we are using for the transformation.
   virtual int Order() const = 0;

   /// Return the order of the elements of the Jacobian of the transformation.
   virtual int OrderJ() const = 0;

   /** @brief Return the order of the determinant of the Jacobian (weight)
       of the transformation. */
   virtual int OrderW() const = 0;

   /// Return the order of \f$ adj(J)^T \nabla fi \f$
   virtual int OrderGrad(const FiniteElement *fe) const = 0;

   /// Return the Geometry::Type of the reference element.
   Geometry::Type GetGeometryType() const { return geom; }

   /// Return the topological dimension of the reference element.
   int GetDimension() const { return Geometry::Dimension[geom]; }

   /// Get the dimension of the target (physical) space.
   /** We support 2D meshes embedded in 3D; in this case the function will
       return "3". */
   virtual int GetSpaceDim() const = 0;

   /** @brief Transform a point @a pt from physical space to a point @a ip in
       reference space. */
   /** Attempt to find the IntegrationPoint that is transformed into the given
       point in physical space. If the inversion fails a non-zero value is
       returned. This method is not 100 percent reliable for non-linear
       transformations. */
   virtual int TransformBack(const Vector &pt, IntegrationPoint &ip) = 0;

   // ADDED //
   virtual int TransformBack(const Vector &, IntegrationPoint &, 
                             IntegrationPoint &) = 0;
   // ADDED //

   virtual ~ElementTransformation() { }
};


/// The inverse transformation of a given ElementTransformation.
class InverseElementTransformation
{
public:
   /// Algorithms for selecting an initial guess.
   enum InitGuessType
   {
      Center = 0, ///< Use the center of the reference element.
      ClosestPhysNode = 1, /**<
         Use the point returned by FindClosestPhysPoint() from a reference-space
         grid of type and size controlled by SetInitGuessPointsType() and
         SetInitGuessRelOrder(), respectively. */
      ClosestRefNode = 2, /**<
         Use the point returned by FindClosestRefPoint() from a reference-space
         grid of type and size controlled by SetInitGuessPointsType() and
         SetInitGuessRelOrder(), respectively. */
      GivenPoint = 3 ///< Use a specific point, set with SetInitialGuess().
   };

   /// Solution strategy.
   enum SolverType
   {
      Newton = 0, /**<
         Use Newton's algorithm, without restricting the reference-space points
         (iterates) to the reference element. */
      NewtonSegmentProject = 1, /**<
         Use Newton's algorithm, restricting the reference-space points to the
         reference element by scaling back the Newton increments, i.e.
         projecting new iterates, x_new, lying outside the element, to the
         intersection of the line segment [x_old, x_new] with the boundary. */
      NewtonElementProject = 2 /**<
         Use Newton's algorithm, restricting the reference-space points to the
         reference element by projecting new iterates, x_new, lying outside the
         element, to the point on the boundary closest (in reference-space) to
         x_new. */
   };

   /// Values returned by Transform().
   enum TransformResult
   {
      Inside  = 0, ///< The point is inside the element
      Outside = 1, ///< The point is _probably_ outside the element
      Unknown = 2  ///< The algorithm failed to determine where the point is
   };

protected:
   // Pointer to the forward transformation. Not owned.
   ElementTransformation *T;

   // Parameters of the inversion algorithms:
   const IntegrationPoint *ip0;
   int init_guess_type; // algorithm to use
   int qpts_type; // Quadrature1D type for the initial guess type
   int rel_qpts_order; // num_1D_qpts = max(trans_order+rel_qpts_order,0)+1
   int solver_type; // solution strategy to use
   int max_iter; // max. number of Newton iterations
   double ref_tol; // reference space tolerance
   double phys_rtol; // physical space tolerance (relative)
   double ip_tol; // tolerance for checking if a point is inside the ref. elem.
   int print_level;

   void NewtonPrint(int mode, double val);
   void NewtonPrintPoint(const char *prefix, const Vector &pt,
                         const char *suffix);
   int NewtonSolve(const Vector &pt, IntegrationPoint &ip);

public:
   /// Construct the InverseElementTransformation with default parameters.
   /** Some practical considerations regarding the choice of initial guess type
       and solver type:
       1. The combination of #Center and #NewtonSegmentProject provides the
          fastest way to estimate if a point lies inside an element, assuming
          that most queried elements are not very deformed, e.g. if most
          elements are convex.
       2. [Default] The combination of #Center and #NewtonElementProject
          provides a somewhat slower alternative to 1 with the benefit of being
          more reliable in the case when the query point is inside the element
          but potentially slower in the case when the query point is outside the
          element.
       3. The combination of #ClosestPhysNode and #NewtonElementProject is
          slower than 1 and 2 but much more reliable, especially in the case of
          highly distorted elements which do not have very high aspect ratios.
       4. The combination of #ClosestRefNode and #NewtonElementProject should
          generally be the most reliable, coming at a bit higher computational
          cost than 3 while performing better on distorted meshes with elements
          having high aspect ratios.
       @note None of these choices provide a guarantee that if a query point is
       inside the element then it will be found. The only guarantee is that if
       the Transform() method returns #Inside then the point lies inside the
       element up to one of the specified physical- or reference-space
       tolerances. */
   InverseElementTransformation(ElementTransformation *Trans = NULL)
      : T(Trans),
        ip0(NULL),
        init_guess_type(Center),
        qpts_type(Quadrature1D::OpenHalfUniform),
        rel_qpts_order(-1),
        solver_type(NewtonElementProject),
        max_iter(16),
        ref_tol(1e-15),
        phys_rtol(1e-15),
        ip_tol(1e-8),
        print_level(-1)
   { }

   virtual ~InverseElementTransformation() { }

   /// Set a new forward ElementTransformation, @a Trans.
   void SetTransformation(ElementTransformation &Trans) { T = &Trans; }

   /** @brief Choose how the initial guesses for subsequent calls to Transform()
       will be selected. */
   void SetInitialGuessType(InitGuessType itype) { init_guess_type = itype; }

   /** @brief Set the initial guess for subsequent calls to Transform(),
       switching to the #GivenPoint #InitGuessType at the same time. */
   void SetInitialGuess(const IntegrationPoint &init_ip)
   { ip0 = &init_ip; SetInitialGuessType(GivenPoint); }

   /// Set the Quadrature1D type used for the `Closest*` initial guess types.
   void SetInitGuessPointsType(int q_type) { qpts_type = q_type; }

   /// Set the relative order used for the `Closest*` initial guess types.
   /** The number of points in each spatial direction is given by the formula
       max(trans_order+order,0)+1, where trans_order is the order of the current
       ElementTransformation. */
   void SetInitGuessRelOrder(int order) { rel_qpts_order = order; }

   /** @brief Specify which algorithm to use for solving the transformation
       equation, i.e. when calling the Transform() method. */
   void SetSolverType(SolverType stype) { solver_type = stype; }

   /// Set the maximum number of iterations when solving for a reference point.
   void SetMaxIter(int max_it) { max_iter = max_it; }

   /// Set the reference-space convergence tolerance.
   void SetReferenceTol(double ref_sp_tol) { ref_tol = ref_sp_tol; }

   /// Set the relative physical-space convergence tolerance.
   void SetPhysicalRelTol(double phys_rel_tol) { phys_rtol = phys_rel_tol; }

   /** @brief Set the tolerance used to determine if a point lies inside or
       outside of the reference element. */
   /** This tolerance is used only with the pure #Newton solver. */
   void SetElementTol(double el_tol) { ip_tol = el_tol; }

   /// Set the desired print level, useful for debugging.
   /** The valid options are: -1 - never print (default); 0 - print only errors;
       1 - print the first and last last iterations; 2 - print every iteration;
       and 3 - print every iteration including point coordinates. */
   void SetPrintLevel(int pr_level) { print_level = pr_level; }

   /** @brief Find the IntegrationPoint mapped closest to @a pt. */
   /** This function uses the given IntegrationRule, @a ir, maps its points to
       physical space and finds the one that is closest to the point @a pt.

       @param pt  The query point.
       @param ir  The IntegrationRule, i.e. the set of reference points to map
                  to physical space and check.
       @return The index of the IntegrationPoint in @a ir whose mapped point is
               closest to @a pt.
       @see FindClosestRefPoint(). */
   int FindClosestPhysPoint(const Vector& pt, const IntegrationRule &ir);

   /** @brief Find the IntegrationPoint mapped closest to @a pt, using a norm
       that approximates the (unknown) distance in reference coordinates. */
   /** @see FindClosestPhysPoint(). */
   int FindClosestRefPoint(const Vector& pt, const IntegrationRule &ir);

   /** @brief Given a point, @a pt, in physical space, find its reference
       coordinates, @a ip.

       @returns A value of type #TransformResult. */
   virtual int Transform(const Vector &pt, IntegrationPoint &ip);
};

/// A standard isoparametric element transformation
class IsoparametricTransformation : public ElementTransformation
{
private:
   DenseMatrix dshape,d2shape;
   Vector shape;

   const FiniteElement *FElem;
   DenseMatrix PointMat; // dim x dof

   /** @brief Evaluate the Jacobian of the transformation at the IntPoint and
       store it in dFdx. */
   virtual const DenseMatrix &EvalJacobian();
   // Evaluate the Hessian of the transformation at the IntPoint and store it
   // in d2Fdx2.
   virtual const DenseMatrix &EvalHessian();
public:
   /// Set the element that will be used to compute the transformations
   void SetFE(const FiniteElement *FE) { FElem = FE; geom = FE->GetGeomType(); }

   /// Get the current element used to compute the transformations
   const FiniteElement* GetFE() const { return FElem; }

   /// @brief Set the underlying point matrix describing the transformation.
   /** The dimensions of the matrix are space-dim x dof. The transformation is
       defined as
           \f$ x = F( \hat x ) = P \phi( \hat x ) \f$

       where \f$ \hat x \f$  is the reference point, @a x is the corresponding
       physical point, @a P is the point matrix, and \f$ \phi( \hat x ) \f$ is
       the column-vector of all basis functions evaluated at \f$ \hat x \f$ .
       The columns of @a P represent the control points in physical space
       defining the transformation. */
   void SetPointMat(const DenseMatrix &pm) { PointMat = pm; }

   /// Return the stored point matrix.
   const DenseMatrix &GetPointMat() const { return PointMat; }

   /// Write access to the stored point matrix. Use with caution.
   DenseMatrix &GetPointMat() { return PointMat; }

   /// Set the FiniteElement Geometry for the reference elements being used.
   void SetIdentityTransformation(Geometry::Type GeomType);

   /** @brief Transform integration point from reference coordinates to
       physical coordinates and store them in the vector. */
   virtual void Transform(const IntegrationPoint &, Vector &);

   /** @brief Transform all the integration points from the integration rule
       from reference coordinates to physical
      coordinates and store them as column vectors in the matrix. */
   virtual void Transform(const IntegrationRule &, DenseMatrix &);

   /** @brief Transform all the integration points from the column vectors
       of @a matrix from reference coordinates to physical
       coordinates and store them as column vectors in @a result. */
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result);

   /// Return the order of the current element we are using for the transformation.
   virtual int Order() const { return FElem->GetOrder(); }

   /// Return the order of the elements of the Jacobian of the transformation.
   virtual int OrderJ() const;

   /** @brief Return the order of the determinant of the Jacobian (weight)
       of the transformation. */
   virtual int OrderW() const;

   /// Return the order of \f$ adj(J)^T \nabla fi \f$
   virtual int OrderGrad(const FiniteElement *fe) const;

   virtual int GetSpaceDim() const { return PointMat.Height(); }

   /** @brief Transform a point @a pt from physical space to a point @a ip in
       reference space. */
   /** Attempt to find the IntegrationPoint that is transformed into the given
       point in physical space. If the inversion fails a non-zero value is
       returned. This method is not 100 percent reliable for non-linear
       transformations. */
   virtual int TransformBack(const Vector & v, IntegrationPoint & ip)
   {
      InverseElementTransformation inv_tr(this);
      return inv_tr.Transform(v, ip);
   }

   // ADDED //
   virtual int TransformBack(const Vector &pt, IntegrationPoint &ip,
                             IntegrationPoint &xip);
   // ADDED //

   virtual ~IsoparametricTransformation() { }

   MFEM_DEPRECATED void FinalizeTransformation() {}
};

class IntegrationPointTransformation
{
public:
   IsoparametricTransformation Transf;
   void Transform (const IntegrationPoint &, IntegrationPoint &);
   void Transform (const IntegrationRule  &, IntegrationRule  &);
};


class FaceElementTransformations : public IsoparametricTransformation
{
private:
   int mask;

   IntegrationPoint eip1, eip2;

public:
   int Elem1No, Elem2No;
   Geometry::Type &FaceGeom; ///< @deprecated Use GetGeometryType instead
   ElementTransformation *Elem1, *Elem2;
   ElementTransformation *Face; ///< @deprecated No longer necessary
   IntegrationPointTransformation Loc1, Loc2;

   FaceElementTransformations() : FaceGeom(geom), Face(this) {}

   /** @brief Method to set the geometry type of the face.

       @note This method is designed to be used when
       [Par]Mesh::GetFaceTransformation will not be called i.e. when the face
       transformation will not be needed but the neighboring element
       transformations will be.  Using this method to override the GeometryType
       should only be done with great care.
   */
   void SetGeometryType(Geometry::Type g) { geom = g; }

   /// Set the mask indicating which portions of the object have been setup
   /** The argument @a m is a bitmask used in
       Mesh::GetFaceElementTransformations to indicate which portions of the
       FaceElement Transformations object have been configured.

       mask &  1: Elem1 is configured
       mask &  2: Elem2 is configured
       mask &  4: Loc1 is configured
       mask &  8: Loc2 is configured
       mask & 16: The Face transformation itself is configured
   */
   void SetConfigurationMask(int m) { mask = m; }
   int  GetConfigurationMask() const { return mask; }

   /** @brief Set the integration point in the Face and the two neighboring
       elements, if present. */
   void SetIntPoint(const IntegrationPoint *ip);

   virtual void Transform(const IntegrationPoint &, Vector &);
   virtual void Transform(const IntegrationRule &, DenseMatrix &);
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result);

   ElementTransformation & GetElement1Transformation();
   ElementTransformation & GetElement2Transformation();
   IntegrationPointTransformation & GetIntPoint1Transformation();
   IntegrationPointTransformation & GetIntPoint2Transformation();
};

/**                Elem1(Loc1(x)) = Face(x) = Elem2(Loc2(x))


                                Physical Space

               *--------*             ^            *--------*
    Elem1No   /        / \           / \          / \        \   Elem2No
             /        /   \         /   \        /   \        \
            /        /  n  \       /     \      /     \        \
           *--------*   ==> *     (       )    *       *--------*
            \        \     /       \     /      \     /        /
             \        \   /         \   /        \   /        /
              \        \ /           \ /          \ /        /
               *--------*             v            *--------*

              ^                                              ^
              |                       ^                      |
        Elem1 |                       |                      | Elem2
              |                       | Face                 |
                                      |
        *--------*                                          *--------*
       /        /|                                         /        /|
    1 *--------* |              1 *--------*            1 *--------* |
      |        | |     Loc1       |        |     Loc2     |        | |
      |        | *    <-----      |    x   |    ----->    |        | *
      |        |/                 |        |              |        |/
      *--------*                  *--------*              *--------*
     0         1                 0         1             0         1

                               Reference Space
*/

}

#endif
