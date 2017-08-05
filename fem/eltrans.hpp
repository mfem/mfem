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
   int JacobianIsEvaluated;
   int WeightIsEvaluated;
   const IntegrationPoint *IntPoint;
   DenseMatrix dFdx, adjJ, invJ;
   double Wght;
   int EvalState;
   enum StateMasks
   {
      JACOBIAN_MASK = 1,
      WEIGHT_MASK   = 2,
      ADJUGATE_MASK = 4,
      INVERSE_MASK  = 8
   };

   // Evaluate the Jacobian of the transformation at the IntPoint and store it
   // in dFdx.
   virtual const DenseMatrix &EvalJacobian() = 0;

   double EvalWeight();
   const DenseMatrix &EvalAdjugateJ();
   const DenseMatrix &EvalInverseJ();

public:
   int Attribute, ElementNo;

   ElementTransformation();

   void SetIntPoint(const IntegrationPoint *ip)
   { IntPoint = ip; EvalState = 0; WeightIsEvaluated = JacobianIsEvaluated = 0; }
   const IntegrationPoint &GetIntPoint() { return *IntPoint; }

   virtual void Transform(const IntegrationPoint &, Vector &) = 0;
   virtual void Transform(const IntegrationRule &, DenseMatrix &) = 0;

   /// Transform columns of 'matrix', store result in 'result'.
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result) = 0;

   /** Return the Jacobian of the transformation at the IntPoint.
       The first column contains the x derivatives of the
       transformation, the second -- the y derivatives, etc.  */
   const DenseMatrix &Jacobian()
   { return (EvalState & JACOBIAN_MASK) ? dFdx : EvalJacobian(); }

   double Weight() { return (EvalState & WEIGHT_MASK) ? Wght : EvalWeight(); }
   //virtual double Weight() = 0;
   const DenseMatrix &AdjugateJacobian()
   { return (EvalState & ADJUGATE_MASK) ? adjJ : EvalAdjugateJ(); }

   const DenseMatrix &InverseJacobian()
   { return (EvalState & INVERSE_MASK) ? invJ : EvalInverseJ(); }

   virtual int Order() = 0;
   virtual int OrderJ() = 0;
   virtual int OrderW() = 0;
   /// order of adj(J)^t.grad(fi)
   virtual int OrderGrad(const FiniteElement *fe) = 0;

   /** Get dimension of target space (we support 2D meshes embedded in 3D; in
       this case the function should return "3"). */
   virtual int GetSpaceDim() = 0;

   /** Attempt to find the IntegrationPoint that is transformed into the given
       point in physical space. If the inversion fails a non-zero value is
       returned. This method is not 100 percent reliable for non-linear
       transformations. */
   virtual int TransformBack(const Vector &, IntegrationPoint &) = 0;

   virtual ~ElementTransformation() { }
};

class IsoparametricTransformation : public ElementTransformation
{
private:
   DenseMatrix dshape;
   Vector shape;

   const FiniteElement *FElem;
   DenseMatrix PointMat; // dim x dof

   // Evaluate the Jacobian of the transformation at the IntPoint and store it
   // in dFdx.
   virtual const DenseMatrix &EvalJacobian();

public:
   void SetFE(const FiniteElement *FE) { FElem = FE; }
   const FiniteElement* GetFE() const { return FElem; }

   /** @brief Read and write access to the underlying point matrix describing
       the transformation. */
   /** The dimensions of the matrix are space-dim x dof. The transformation is
       defined as

           x=F(xh)=P.phi(xh),

       where xh (x hat) is the reference point, x is the corresponding physical
       point, P is the point matrix, and phi(xh) is the column-vector of all
       basis functions evaluated at xh. The columns of P represent the control
       points in physical space defining the transformation. */
   DenseMatrix &GetPointMat() { return PointMat; }

   void SetIdentityTransformation(int GeomType);

   virtual void Transform(const IntegrationPoint &, Vector &);
   virtual void Transform(const IntegrationRule &, DenseMatrix &);
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result);

   virtual int Order() { return FElem->GetOrder(); }
   virtual int OrderJ();
   virtual int OrderW();
   virtual int OrderGrad(const FiniteElement *fe);

   virtual int GetSpaceDim()
   {
      // this function should only be called after PointMat is initialized
      return PointMat.Height();
   }

   virtual int TransformBack(const Vector &, IntegrationPoint &);

   virtual ~IsoparametricTransformation() { }
};

class IntegrationPointTransformation
{
public:
   IsoparametricTransformation Transf;
   void Transform (const IntegrationPoint &, IntegrationPoint &);
   void Transform (const IntegrationRule  &, IntegrationRule  &);
};

class FaceElementTransformations
{
public:
   int Elem1No, Elem2No, FaceGeom;
   ElementTransformation *Elem1, *Elem2, *Face;
   IntegrationPointTransformation Loc1, Loc2;
};

/*                 Elem1(Loc1(x)) = Face(x) = Elem2(Loc2(x))


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
