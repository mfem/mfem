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

#ifndef MFEM_ELEMENTTRANSFORM
#define MFEM_ELEMENTTRANSFORM

class ElementTransformation
{
protected:
   int JacobianIsEvaluated;
   int WeightIsEvaluated;
   const IntegrationPoint *IntPoint;

public:
   int Attribute, ElementNo;
   void SetIntPoint(const IntegrationPoint *ip)
   { IntPoint = ip; WeightIsEvaluated = JacobianIsEvaluated = 0; }
   const IntegrationPoint &GetIntPoint() { return *IntPoint; }

   virtual void Transform(const IntegrationPoint &, Vector &) = 0;
   virtual void Transform(const IntegrationRule &, DenseMatrix &) = 0;
   /** Return the jacobian of the transformation at the IntPoint.
       The first column contains the x derivatives of the
       transformation, the second -- the y derivatives, etc.  */
   virtual const DenseMatrix & Jacobian() = 0;
   virtual double Weight() = 0;
   virtual int OrderJ() = 0;
   virtual int OrderW() = 0;
   /// order of adj(J)^t.grad(fi)
   virtual int OrderGrad(const FiniteElement *fe) = 0;

   virtual ~ElementTransformation() { }
};

class IsoparametricTransformation : public ElementTransformation
{
private:
   DenseMatrix dshape, dFdx;
   double Wght;
   Vector shape;

   const FiniteElement *FElem;
   DenseMatrix PointMat;

public:
   void SetFE(const FiniteElement *FE) { FElem = FE; };
   DenseMatrix &GetPointMat () { return PointMat; };

   virtual void Transform(const IntegrationPoint &, Vector &);
   virtual void Transform(const IntegrationRule &, DenseMatrix &);
   virtual const DenseMatrix & Jacobian();
   virtual double Weight();
   virtual int OrderJ();
   virtual int OrderW();
   virtual int OrderGrad(const FiniteElement *fe);

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

#endif
