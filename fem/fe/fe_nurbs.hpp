// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_NURBS
#define MFEM_FE_NURBS

#include "fe_base.hpp"

namespace mfem
{

class KnotVector;

/// An arbitrary order and dimension NURBS element
class NURBSFiniteElement
{
protected:
   mutable Array <const KnotVector*> kv;
   mutable const int *ijk;
   mutable int patch, elem;
   mutable Vector weights;

public:
   /** @brief Construct NURBSFiniteElement with given
       @param D    Reference space dimension
    */
   NURBSFiniteElement(int dim,int dof)
   {
      ijk = NULL;
      patch = elem = -1;
      kv.SetSize(dim);
      weights.SetSize(dof);
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

   /// Returns the indices (i,j) in 2D or (i,j,k) in 3D of this element in the
   /// tensor product ordering of the patch.
   const int* GetIJK() const { return ijk; }
};


/// An arbitrary order 1D NURBS element on a segment
class NURBS1DFiniteElement : public ScalarFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector shape_x;

public:
   /// Construct the NURBS1DFiniteElement of order @a p
   NURBS1DFiniteElement(int p)
      : ScalarFiniteElement(1, Geometry::SEGMENT, p + 1, p, FunctionSpace::Qk),
        NURBSFiniteElement(1,(p + 1)),
        shape_x(p + 1) { }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};

/// An arbitrary order 2D NURBS element on a square
class NURBS2DFiniteElement : public ScalarFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
   mutable DenseMatrix du;

public:
   /// Construct the NURBS2DFiniteElement of order @a p
   NURBS2DFiniteElement(int p)
      : ScalarFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                            FunctionSpace::Qk),
        NURBSFiniteElement(2, (p + 1)*(p + 1)),
        u(dof), shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1),
        dshape_y(p + 1), d2shape_x(p + 1), d2shape_y(p + 1), du(dof,2)
   { orders[0] = orders[1] = p; }

   /// Construct the NURBS2DFiniteElement with x-order @a px and y-order @a py
   NURBS2DFiniteElement(int px, int py)
      : ScalarFiniteElement(2, Geometry::SQUARE, (px + 1)*(py + 1),
                            std::max(px, py), FunctionSpace::Qk),
        NURBSFiniteElement(2, (px + 1)*(py + 1)),
        u(dof), shape_x(px + 1), shape_y(py + 1), dshape_x(px + 1),
        dshape_y(py + 1), d2shape_x(px + 1), d2shape_y(py + 1), du(dof,2)
   { orders[0] = px; orders[1] = py; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};

/// An arbitrary order 3D NURBS element on a cube
class NURBS3DFiniteElement : public ScalarFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z;
   mutable Vector d2shape_x, d2shape_y, d2shape_z;
   mutable DenseMatrix du;

public:
   /// Construct the NURBS3DFiniteElement of order @a p
   NURBS3DFiniteElement(int p)
      : ScalarFiniteElement(3, Geometry::CUBE, (p + 1)*(p + 1)*(p + 1), p,
                            FunctionSpace::Qk),
        NURBSFiniteElement(3, (p + 1)*(p + 1)*(p + 1)),
        u(dof), shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1),
        d2shape_x(p + 1), d2shape_y(p + 1), d2shape_z(p + 1), du(dof,3)
   { orders[0] = orders[1] = orders[2] = p; }

   /// Construct the NURBS3DFiniteElement with x-order @a px and y-order @a py
   /// and z-order @a pz
   NURBS3DFiniteElement(int px, int py, int pz)
      : ScalarFiniteElement(3, Geometry::CUBE, (px + 1)*(py + 1)*(pz + 1),
                            std::max(std::max(px,py),pz), FunctionSpace::Qk),
        NURBSFiniteElement(2, (px + 1)*(py + 1)*(pz + 1)),
        u(dof), shape_x(px + 1), shape_y(py + 1), shape_z(pz + 1),
        dshape_x(px + 1), dshape_y(py + 1), dshape_z(pz + 1),
        d2shape_x(px + 1), d2shape_y(py + 1), d2shape_z(pz + 1), du(dof,3)
   { orders[0] = px; orders[1] = py; orders[2] = pz; }

   virtual void SetOrder() const;
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian (const IntegrationPoint &ip,
                             DenseMatrix &hessian) const;
};


/// An arbitrary order H(div)-conforming 2D NURBS element on a square
class NURBS_HDiv1DFiniteElement : public VectorFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, dshape_x, d2shape_x;
   mutable Vector shape1_x, dshape1_x, d2shape1_x;
   mutable DenseMatrix du;
   mutable Array <const KnotVector*> kv1;

public:
   /// Construct the NURBS_HDiv1DFiniteElement of order @a p
   NURBS_HDiv1DFiniteElement(int p, int vdim)
      : VectorFiniteElement(1, Geometry::SQUARE, (p + 1)*(p + 1), p,
                            H_DIV,FunctionSpace::Qk),
        NURBSFiniteElement(1, (p + 1)*(p + 1)),
        shape_x(p + 1), dshape_x(p + 1), d2shape_x(p + 1),
        shape1_x(p + 2),  dshape1_x(p + 2), d2shape1_x(p + 2),
        u(dof), du(dof,2)
   {
      orders[0] = orders[1] = p;
      kv1.SetSize(dim);
      kv1[0] = nullptr;
   }

   virtual void SetOrder() const;

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

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#dof) of the result Vector @a divshape must be set in advance.
    */
   //   virtual void CalcDivShape(const IntegrationPoint &ip,
   //                             Vector &divshape) const;

   /** @brief Evaluate the gradients of all shape functions of a vector finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DensTensor @a dshape contains the derivatives of
       one shape function. The size (#dof x #dim x #dim) of @a dshape must be set in
       advance.  */
   //   virtual void CalcVDShape(const IntegrationPoint &ip,
   //                            DenseTensor &dshape) const;

   /** @brief Evaluate the Hessians of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a Hessian contains upper triangular
       part of the Hessian of one shape function.
       The order in 2D is {u_xx, u_xy, u_yy}.
       The size (#dof x (#dim (#dim+1)/2 x #dim) of @a Hessian must be set in advance.*/
   //   virtual void CalcVHessian(const IntegrationPoint &ip,
   //                             DenseTensor &Hessian) const;
};


/// An arbitrary order H(div)-conforming 2D NURBS element on a square
class NURBS_HDiv2DFiniteElement : public VectorFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
   mutable Vector shape1_x, shape1_y, dshape1_x, dshape1_y, d2shape1_x, d2shape1_y;
   mutable DenseMatrix du;
   mutable Array <const KnotVector*> kv1;

public:
   /// Construct the NURBS_HDiv22DFiniteElement of order @a p
   NURBS_HDiv2DFiniteElement(int p, int vdim)
      : VectorFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                            H_DIV,FunctionSpace::Qk),
        NURBSFiniteElement(2, (p + 1)*(p + 1)),
        shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1),
        dshape_y(p + 1), d2shape_x(p + 1), d2shape_y(p + 1),
        shape1_x(p + 2), shape1_y(p + 2), dshape1_x(p + 2),
        dshape1_y(p + 2), d2shape1_x(p + 2), d2shape1_y(p + 2),
        u(dof), du(dof,2)
   {
      orders[0] = orders[1] = p;
      kv1.SetSize(dim);
      kv1[0] = nullptr;
      kv1[1] = nullptr;
   }

   /// Construct the NURBS_HDiv22DFiniteElement with x-order @a px and y-order @a py
   NURBS_HDiv2DFiniteElement(int px, int py, int vdim)
      : VectorFiniteElement(2, Geometry::SQUARE, (px + 1)*(py + 1),
                            std::max(px, py), H_DIV, FunctionSpace::Qk),
        NURBSFiniteElement(2, (px + 1)*(py + 1)),
        shape_x(px + 1), shape_y(py + 1), dshape_x(px + 1),
        dshape_y(py + 1), d2shape_x(px + 1), d2shape_y(py + 1),
        shape1_x(px + 2), shape1_y(py + 2), dshape1_x(px + 2),
        dshape1_y(py + 2), d2shape1_x(px + 2), d2shape1_y(py + 2),
        u(dof),  du(dof,2)
   {
      orders[0] = px; orders[1] = py;
      kv1.SetSize(dim);
      kv1[0] = nullptr;
      kv1[1] = nullptr;
      map_type = H_DIV;
   }

   virtual void SetOrder() const;

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

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#dof) of the result Vector @a divshape must be set in advance.
    */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** @brief Evaluate the gradients of all shape functions of a vector finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DensTensor @a dshape contains the derivatives of
       one shape function. The size (#dof x #dim x #dim) of @a dshape must be set in
       advance.  */
   virtual void CalcVDShape(const IntegrationPoint &ip,
                            DenseTensor &dshape) const;

   /** @brief Evaluate the Hessians of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a Hessian contains upper triangular
       part of the Hessian of one shape function.
       The order in 2D is {u_xx, u_xy, u_yy}.
       The size (#dof x (#dim (#dim+1)/2 x #dim) of @a Hessian must be set in advance.*/
   virtual void CalcVHessian(const IntegrationPoint &ip,
                             DenseTensor &Hessian) const;

   ~NURBS_HDiv2DFiniteElement();
};

/// An arbitrary order H(div)-conforming 3D NURBS element on a square
class NURBS_HDiv3DFiniteElement : public VectorFiniteElement,
   public NURBSFiniteElement
{
protected:
   mutable Vector u;
   mutable Vector shape_x, dshape_x, d2shape_x, shape1_x, dshape1_x, d2shape1_x;
   mutable Vector shape_y, dshape_y, d2shape_y, shape1_y, dshape1_y, d2shape1_y;
   mutable Vector shape_z, dshape_z, d2shape_z, shape1_z, dshape1_z, d2shape1_z;
   
   mutable DenseMatrix du;
   mutable Array <const KnotVector*> kv1;

public:
   /// Construct the NURBS_HDiv22DFiniteElement of order @a p
   NURBS_HDiv3DFiniteElement(int p, int vdim)
      : VectorFiniteElement(3, Geometry::SQUARE, (p + 1)*(p + 1)*(p + 1), p,
                            H_DIV,FunctionSpace::Qk),
        NURBSFiniteElement(3, (p + 1)*(p + 1)*(p + 1)),
        shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1),
        d2shape_x(p + 1), d2shape_y(p + 1), d2shape_z(p + 1),
        shape1_x(p + 2), shape1_y(p + 2), shape1_z(p + 2),
        dshape1_x(p + 2), dshape1_y(p + 2),dshape1_z(p + 2),
        d2shape1_x(p + 2), d2shape1_y(p + 2), d2shape1_z(p + 2),
        u(dof), du(dof,2)
   {
      orders[0] = orders[1] = orders[2] = p;
      kv1.SetSize(dim);
      kv1[0] = nullptr;
      kv1[1] = nullptr;
      kv1[2] = nullptr;
   }

   /// Construct the NURBS_HDiv22DFiniteElement with x-order @a px and y-order @a py
   NURBS_HDiv3DFiniteElement(int px, int py, int pz, int vdim)
      : VectorFiniteElement(3, Geometry::SQUARE, (px + 1)*(py + 1)*(pz + 1),
                            std::max(px, py), H_DIV, FunctionSpace::Qk),
        NURBSFiniteElement(3, (px + 1)*(py + 1)*(pz + 1)),
        shape_x(px + 1), shape_y(py + 1), shape_z(pz + 1),
        dshape_x(px + 1), dshape_y(py + 1), dshape_z(pz + 1),
        d2shape_x(px + 1), d2shape_y(py + 1), d2shape_z(pz + 1),
        shape1_x(px + 2), shape1_y(py + 2), shape1_z(pz + 2),
        dshape1_x(px + 2), dshape1_y(py + 2),dshape1_z(pz + 2),
        d2shape1_x(px + 2), d2shape1_y(py + 2), d2shape1_z(pz + 2),
        u(dof),  du(dof,2)
   {
      orders[0] = px; orders[1] = py; orders[1] = pz;
      kv1.SetSize(dim);
      kv1[0] = nullptr;
      kv1[1] = nullptr;
      kv1[2] = nullptr;
      map_type = H_DIV;
   }

   virtual void SetOrder() const;

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

   /** @brief Evaluate the divergence of all shape functions of a *vector*
       finite element in reference space at the given point @a ip. */
   /** The size (#dof) of the result Vector @a divshape must be set in advance.
    */
   virtual void CalcDivShape(const IntegrationPoint &ip,
                             Vector &divshape) const;

   /** @brief Evaluate the gradients of all shape functions of a vector finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DensTensor @a dshape contains the derivatives of
       one shape function. The size (#dof x #dim x #dim) of @a dshape must be set in
       advance.  */
   virtual void CalcVDShape(const IntegrationPoint &ip,
                            DenseTensor &dshape) const;

   /** @brief Evaluate the Hessians of all shape functions of a scalar finite
       element in reference space at the given point @a ip. */
   /** Each row of the result DenseMatrix @a Hessian contains upper triangular
       part of the Hessian of one shape function.
       The order in 2D is {u_xx, u_xy, u_yy}.
       The size (#dof x (#dim (#dim+1)/2 x #dim) of @a Hessian must be set in advance.*/
   virtual void CalcVHessian(const IntegrationPoint &ip,
                             DenseTensor &Hessian) const;

   ~NURBS_HDiv3DFiniteElement();
};




} // namespace mfem

#endif
