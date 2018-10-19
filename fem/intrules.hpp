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

#ifndef MFEM_INTRULES
#define MFEM_INTRULES

#include "../config/config.hpp"
#include "../general/array.hpp"

namespace mfem
{

/* Classes for IntegrationPoint, IntegrationRule, and container class
   IntegrationRules.  Declares the global variable IntRules */

/// Class for integration point with weight
class IntegrationPoint
{
public:
   double x, y, z, weight;

   void Init() { x = y = z = weight = 0.0; }

   void Set(const double *p, const int dim)
   {
      MFEM_ASSERT(1 <= dim && dim <= 3, "invalid dim: " << dim);
      x = p[0];
      if (dim > 1)
      {
         y = p[1];
         if (dim > 2)
         {
            z = p[2];
         }
      }
   }

   void Get(double *p, const int dim) const
   {
      MFEM_ASSERT(1 <= dim && dim <= 3, "invalid dim: " << dim);
      p[0] = x;
      if (dim > 1)
      {
         p[1] = y;
         if (dim > 2)
         {
            p[2] = z;
         }
      }
   }

   void Set(const double x1, const double x2, const double x3, const double w)
   { x = x1; y = x2; z = x3; weight = w; }

   void Set3w(const double *p) { x = p[0]; y = p[1]; z = p[2]; weight = p[3]; }

   void Set3(const double x1, const double x2, const double x3)
   { x = x1; y = x2; z = x3; }

   void Set3(const double *p) { x = p[0]; y = p[1]; z = p[2]; }

   void Set2w(const double x1, const double x2, const double w)
   { x = x1; y = x2; weight = w; }

   void Set2w(const double *p) { x = p[0]; y = p[1]; weight = p[2]; }

   void Set2(const double x1, const double x2) { x = x1; y = x2; }

   void Set2(const double *p) { x = p[0]; y = p[1]; }

   void Set1w(const double x1, const double w) { x = x1; weight = w; }

   void Set1w(const double *p) { x = p[0]; weight = p[1]; }
};

/// Class for an integration rule - an Array of IntegrationPoint.
class IntegrationRule : public Array<IntegrationPoint>
{
private:
   friend class IntegrationRules;
   int Order;

   /// Define n-simplex rule (triangle/tetrahedron for n=2/3) of order (2s+1)
   void GrundmannMollerSimplexRule(int s, int n = 3);

   void AddTriMidPoint(const int off, const double weight)
   { IntPoint(off).Set2w(1./3., 1./3., weight); }

   void AddTriPoints3(const int off, const double a, const double b,
                      const double weight)
   {
      IntPoint(off + 0).Set2w(a, a, weight);
      IntPoint(off + 1).Set2w(a, b, weight);
      IntPoint(off + 2).Set2w(b, a, weight);
   }

   void AddTriPoints3(const int off, const double a, const double weight)
   { AddTriPoints3(off, a, 1. - 2.*a, weight); }

   void AddTriPoints3b(const int off, const double b, const double weight)
   { AddTriPoints3(off, (1. - b)/2., b, weight); }

   void AddTriPoints3R(const int off, const double a, const double b,
                       const double c, const double weight)
   {
      IntPoint(off + 0).Set2w(a, b, weight);
      IntPoint(off + 1).Set2w(c, a, weight);
      IntPoint(off + 2).Set2w(b, c, weight);
   }

   void AddTriPoints3R(const int off, const double a, const double b,
                       const double weight)
   { AddTriPoints3R(off, a, b, 1. - a - b, weight); }

   void AddTriPoints6(const int off, const double a, const double b,
                      const double c, const double weight)
   {
      IntPoint(off + 0).Set2w(a, b, weight);
      IntPoint(off + 1).Set2w(b, a, weight);
      IntPoint(off + 2).Set2w(a, c, weight);
      IntPoint(off + 3).Set2w(c, a, weight);
      IntPoint(off + 4).Set2w(b, c, weight);
      IntPoint(off + 5).Set2w(c, b, weight);
   }

   void AddTriPoints6(const int off, const double a, const double b,
                      const double weight)
   { AddTriPoints6(off, a, b, 1. - a - b, weight); }

   // add the permutations of (a,a,b)
   void AddTetPoints3(const int off, const double a, const double b,
                      const double weight)
   {
      IntPoint(off + 0).Set(a, a, b, weight);
      IntPoint(off + 1).Set(a, b, a, weight);
      IntPoint(off + 2).Set(b, a, a, weight);
   }

   // add the permutations of (a,b,c)
   void AddTetPoints6(const int off, const double a, const double b,
                      const double c, const double weight)
   {
      IntPoint(off + 0).Set(a, b, c, weight);
      IntPoint(off + 1).Set(a, c, b, weight);
      IntPoint(off + 2).Set(b, c, a, weight);
      IntPoint(off + 3).Set(b, a, c, weight);
      IntPoint(off + 4).Set(c, a, b, weight);
      IntPoint(off + 5).Set(c, b, a, weight);
   }

   void AddTetMidPoint(const int off, const double weight)
   { IntPoint(off).Set(0.25, 0.25, 0.25, weight); }

   // given a, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4(const int off, const double a, const double weight)
   {
      IntPoint(off).Set(a, a, a, weight);
      AddTetPoints3(off + 1, a, 1. - 3.*a, weight);
   }

   // given b, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4b(const int off, const double b, const double weight)
   {
      const double a = (1. - b)/3.;
      IntPoint(off).Set(a, a, a, weight);
      AddTetPoints3(off + 1, a, b, weight);
   }

   // add the permutations of (a,a,b,b), 2*(a + b) = 1
   void AddTetPoints6(const int off, const double a, const double weight)
   {
      const double b = 0.5 - a;
      AddTetPoints3(off,     a, b, weight);
      AddTetPoints3(off + 3, b, a, weight);
   }

   // given (a,b) or (a,c), add the permutations of (a,a,b,c), 2*a + b + c = 1
   void AddTetPoints12(const int off, const double a, const double bc,
                       const double weight)
   {
      const double cb = 1. - 2*a - bc;
      AddTetPoints3(off,     a, bc, weight);
      AddTetPoints3(off + 3, a, cb, weight);
      AddTetPoints6(off + 6, a, bc, cb, weight);
   }

   // given (b,c), add the permutations of (a,a,b,c), 2*a + b + c = 1
   void AddTetPoints12bc(const int off, const double b, const double c,
                         const double weight)
   {
      const double a = (1. - b - c)/2.;
      AddTetPoints3(off,     a, b, weight);
      AddTetPoints3(off + 3, a, c, weight);
      AddTetPoints6(off + 6, a, b, c, weight);
   }

public:
   IntegrationRule() :
      Array<IntegrationPoint>(), Order(0) { }

   /// Construct an integration rule with given number of points
   explicit IntegrationRule(int NP) :
      Array<IntegrationPoint>(NP), Order(0)
   {
      for (int i = 0; i < this->Size(); i++)
      {
         (*this)[i].Init();
      }
   }

   /// Tensor product of two 1D integration rules
   IntegrationRule(IntegrationRule &irx, IntegrationRule &iry);

   /// Tensor product of three 1D integration rules
   IntegrationRule(IntegrationRule &irx, IntegrationRule &iry,
                   IntegrationRule &irz);

   /// Returns the order of the integration rule
   int GetOrder() const { return Order; }

   /** @brief Sets the order of the integration rule. This is only for keeping
       order information, it does not alter any data in the IntegrationRule. */
   void SetOrder(const int order) { Order = order; }

   /// Returns the number of the points in the integration rule
   int GetNPoints() const { return Size(); }

   /// Returns a reference to the i-th integration point
   IntegrationPoint &IntPoint(int i) { return (*this)[i]; }

   /// Returns a const reference to the i-th integration point
   const IntegrationPoint &IntPoint(int i) const { return (*this)[i]; }

   /// Destroys an IntegrationRule object
   ~IntegrationRule() { }
};

/// A Class that defines 1-D numerical quadrature rules on [0,1].
class QuadratureFunctions1D
{
public:
   /** @name Methods for calculating quadrature rules.
       These methods calculate the actual points and weights for the different
       types of quadrature rules. */
   ///@{
   void GaussLegendre(const int np, IntegrationRule* ir);
   void GaussLobatto(const int np, IntegrationRule *ir);
   void OpenUniform(const int np, IntegrationRule *ir);
   void ClosedUniform(const int np, IntegrationRule *ir);
   void OpenHalfUniform(const int np, IntegrationRule *ir);
   ///@}

   /// A helper function that will play nice with Poly_1D::OpenPoints and
   /// Poly_1D::ClosedPoints
   void GivePolyPoints(const int np, double *pts, const int type);

private:
   void CalculateUniformWeights(IntegrationRule *ir, const int type);
};

/// A class container for 1D quadrature type constants.
class Quadrature1D
{
public:
   enum
   {
      Invalid         = -1,
      GaussLegendre   = 0,
      GaussLobatto    = 1,
      OpenUniform     = 2,  ///< aka open Newton-Cotes
      ClosedUniform   = 3,  ///< aka closed Newton-Cotes
      OpenHalfUniform = 4   ///< aka "open half" Newton-Cotes
   };
   /** @brief If the Quadrature1D type is not closed return Invalid; otherwise
       return type. */
   static int CheckClosed(int type);
   /** @brief If the Quadrature1D type is not open return Invalid; otherwise
       return type. */
   static int CheckOpen(int type);
};

/// Container class for integration rules
class IntegrationRules
{
private:
   /// Taken from the Quadrature1D class anonymous enum
   /// Determines the type of numerical quadrature used for
   /// segment, square, and cube geometries
   const int quad_type;

   int own_rules, refined;

   /// Function that generates quadrature points and weights on [0,1]
   QuadratureFunctions1D quad_func;

   Array<IntegrationRule *> PointIntRules;
   Array<IntegrationRule *> SegmentIntRules;
   Array<IntegrationRule *> TriangleIntRules;
   Array<IntegrationRule *> SquareIntRules;
   Array<IntegrationRule *> TetrahedronIntRules;
   Array<IntegrationRule *> PrismIntRules;
   Array<IntegrationRule *> CubeIntRules;

   void AllocIntRule(Array<IntegrationRule *> &ir_array, int Order)
   {
      if (ir_array.Size() <= Order)
      {
         ir_array.SetSize(Order + 1, NULL);
      }
   }
   bool HaveIntRule(Array<IntegrationRule *> &ir_array, int Order)
   {
      return (ir_array.Size() > Order && ir_array[Order] != NULL);
   }
   int GetSegmentRealOrder(int Order) const
   {
      return Order | 1; // valid for all quad_type's
   }

   IntegrationRule *GenerateIntegrationRule(int GeomType, int Order);
   IntegrationRule *PointIntegrationRule(int Order);
   IntegrationRule *SegmentIntegrationRule(int Order);
   IntegrationRule *TriangleIntegrationRule(int Order);
   IntegrationRule *SquareIntegrationRule(int Order);
   IntegrationRule *TetrahedronIntegrationRule(int Order);
   IntegrationRule *PrismIntegrationRule(int Order);
   IntegrationRule *CubeIntegrationRule(int Order);

   void DeleteIntRuleArray(Array<IntegrationRule *> &ir_array);

public:
   /// Sets initial sizes for the integration rule arrays, but rules
   /// are defined the first time they are requested with the Get method.
   explicit IntegrationRules(int Ref = 0,
                             int type = Quadrature1D::GaussLegendre);

   /// Returns an integration rule for given GeomType and Order.
   const IntegrationRule &Get(int GeomType, int Order);

   void Set(int GeomType, int Order, IntegrationRule &IntRule);

   void SetOwnRules(int o) { own_rules = o; }

   /// Destroys an IntegrationRules object
   ~IntegrationRules();
};

/// A global object with all integration rules (defined in intrules.cpp)
extern IntegrationRules IntRules;

/// A global object with all refined integration rules
extern IntegrationRules RefinedIntRules;

}

#endif
