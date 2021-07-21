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

#ifndef MFEM_FE_NURBS
#define MFEM_FE_NURBS

#include "fe_base.hpp"

namespace mfem
{

class KnotVector;

/// An arbitrary order and dimension NURBS element
class NURBSFiniteElement : public ScalarFiniteElement
{
protected:
   mutable Array <const KnotVector*> kv;
   mutable const int *ijk;
   mutable int patch, elem;
   mutable Vector weights;

public:
   /** @brief Construct NURBSFiniteElement with given
       @param D    Reference space dimension
       @param G    Geometry type (of type Geometry::Type)
       @param Do   Number of degrees of freedom in the FiniteElement
       @param O    Order/degree of the FiniteElement
       @param F    FunctionSpace type of the FiniteElement
    */
   NURBSFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F)
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
};


/// An arbitrary order 1D NURBS element on a segment
class NURBS1DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector shape_x;

public:
   /// Construct the NURBS1DFiniteElement of order @a p
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

/// An arbitrary order 2D NURBS element on a square
class NURBS2DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, dshape_x, dshape_y, d2shape_x, d2shape_y;
   mutable DenseMatrix du;

public:
   /// Construct the NURBS2DFiniteElement of order @a p
   NURBS2DFiniteElement(int p)
      : NURBSFiniteElement(2, Geometry::SQUARE, (p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(dof), shape_x(p + 1), shape_y(p + 1), dshape_x(p + 1),
        dshape_y(p + 1), d2shape_x(p + 1), d2shape_y(p + 1), du(dof,2)
   { orders[0] = orders[1] = p; }

   /// Construct the NURBS2DFiniteElement with x-order @a px and y-order @a py
   NURBS2DFiniteElement(int px, int py)
      : NURBSFiniteElement(2, Geometry::SQUARE, (px + 1)*(py + 1),
                           std::max(px, py), FunctionSpace::Qk),
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
class NURBS3DFiniteElement : public NURBSFiniteElement
{
protected:
   mutable Vector u, shape_x, shape_y, shape_z;
   mutable Vector dshape_x, dshape_y, dshape_z;
   mutable Vector d2shape_x, d2shape_y, d2shape_z;
   mutable DenseMatrix du;

public:
   /// Construct the NURBS3DFiniteElement of order @a p
   NURBS3DFiniteElement(int p)
      : NURBSFiniteElement(3, Geometry::CUBE, (p + 1)*(p + 1)*(p + 1), p,
                           FunctionSpace::Qk),
        u(dof), shape_x(p + 1), shape_y(p + 1), shape_z(p + 1),
        dshape_x(p + 1), dshape_y(p + 1), dshape_z(p + 1),
        d2shape_x(p + 1), d2shape_y(p + 1), d2shape_z(p + 1), du(dof,3)
   { orders[0] = orders[1] = orders[2] = p; }

   /// Construct the NURBS3DFiniteElement with x-order @a px and y-order @a py
   /// and z-order @a pz
   NURBS3DFiniteElement(int px, int py, int pz)
      : NURBSFiniteElement(3, Geometry::CUBE, (px + 1)*(py + 1)*(pz + 1),
                           std::max(std::max(px,py),pz), FunctionSpace::Qk),
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

} // namespace mfem

#endif
