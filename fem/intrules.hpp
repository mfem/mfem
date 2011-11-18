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

#ifndef MFEM_INTRULES
#define MFEM_INTRULES

/* Classes for IntegrationPoint, IntegrationRule, and container class
   IntegrationRules.  Declares the global variable IntRules */

/// Class for integration point with weight
class IntegrationPoint
{
public:
   double x, y, z, weight;

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

/// Class for integration rule
class IntegrationRule
{
private:
   int NPoints;
   IntegrationPoint *IntPoints;

   friend class IntegrationRules;

   /// Computes Gaussian integration rule on (0,1) with NPoints
   void GaussianRule();

   /// Defines composite trapezoidal integration rule on [0,1]
   void UniformRule();

   /// Define tetrahedron rule of order (2s+1)
   void GrundmannMollerTetrahedronRule(int s);

   void AddTriMidPoint(const int off, const double weight)
   { IntPoints[off].Set2w(1./3., 1./3., weight); }

   void AddTriPoints3(const int off, const double a, const double b,
                      const double weight)
   {
      IntPoints[off + 0].Set2w(a, a, weight);
      IntPoints[off + 1].Set2w(a, b, weight);
      IntPoints[off + 2].Set2w(b, a, weight);
   }

   void AddTriPoints3(const int off, const double a, const double weight)
   { AddTriPoints3(off, a, 1. - 2.*a, weight); }

   void AddTriPoints3b(const int off, const double b, const double weight)
   { AddTriPoints3(off, (1. - b)/2., b, weight); }

   void AddTriPoints3R(const int off, const double a, const double b,
                       const double c, const double weight)
   {
      IntPoints[off + 0].Set2w(a, b, weight);
      IntPoints[off + 1].Set2w(c, a, weight);
      IntPoints[off + 2].Set2w(b, c, weight);
   }

   void AddTriPoints3R(const int off, const double a, const double b,
                       const double weight)
   { AddTriPoints3R(off, a, b, 1. - a - b, weight); }

   void AddTriPoints6(const int off, const double a, const double b,
                      const double c, const double weight)
   {
      IntPoints[off + 0].Set2w(a, b, weight);
      IntPoints[off + 1].Set2w(b, a, weight);
      IntPoints[off + 2].Set2w(a, c, weight);
      IntPoints[off + 3].Set2w(c, a, weight);
      IntPoints[off + 4].Set2w(b, c, weight);
      IntPoints[off + 5].Set2w(c, b, weight);
   }

   void AddTriPoints6(const int off, const double a, const double b,
                      const double weight)
   { AddTriPoints6(off, a, b, 1. - a - b, weight); }

   // add the permutations of (a,a,b)
   void AddTetPoints3(const int off, const double a, const double b,
                      const double weight)
   {
      IntPoints[off + 0].Set(a, a, b, weight);
      IntPoints[off + 1].Set(a, b, a, weight);
      IntPoints[off + 2].Set(b, a, a, weight);
   }

   // add the permutations of (a,b,c)
   void AddTetPoints6(const int off, const double a, const double b,
                      const double c, const double weight)
   {
      IntPoints[off + 0].Set(a, b, c, weight);
      IntPoints[off + 1].Set(a, c, b, weight);
      IntPoints[off + 2].Set(b, c, a, weight);
      IntPoints[off + 3].Set(b, a, c, weight);
      IntPoints[off + 4].Set(c, a, b, weight);
      IntPoints[off + 5].Set(c, b, a, weight);
   }

   void AddTetMidPoint(const int off, const double weight)
   { IntPoints[off].Set(0.25, 0.25, 0.25, weight); }

   // given a, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4(const int off, const double a, const double weight)
   {
      IntPoints[off].Set(a, a, a, weight);
      AddTetPoints3(off + 1, a, 1. - 3.*a, weight);
   }

   // given b, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4b(const int off, const double b, const double weight)
   {
      const double a = (1. - b)/3.;
      IntPoints[off].Set(a, a, a, weight);
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
   IntegrationRule() { NPoints = 0; IntPoints = NULL; }

   /// Construct an integration rule with given number of points
   explicit IntegrationRule(int NP);

   /// Tensor product of two 1D integration rules
   IntegrationRule(IntegrationRule &irx, IntegrationRule &iry);

   /// Returns the number of the points in the integration rule
   int GetNPoints() const { return NPoints; }

   /// Returns a reference to the i-th integration point
   IntegrationPoint &IntPoint(int i) { return IntPoints[i]; }

   /// Returns a const reference to the i-th integration point
   const IntegrationPoint &IntPoint(int i) const { return IntPoints[i]; }

   /// Destroys an IntegrationRule object
   ~IntegrationRule();
};

/// Container class for integration rules
class IntegrationRules
{
private:
   int own_rules;

   Array<IntegrationRule *> PointIntRules;
   Array<IntegrationRule *> SegmentIntRules;
   Array<IntegrationRule *> TriangleIntRules;
   Array<IntegrationRule *> SquareIntRules;
   Array<IntegrationRule *> TetrahedronIntRules;
   Array<IntegrationRule *> CubeIntRules;

   void PointIntegrationRules();
   void SegmentIntegrationRules(int refined);
   void TriangleIntegrationRules(int refined);
   void SquareIntegrationRules();
   void TetrahedronIntegrationRules(int refined);
   void CubeIntegrationRules();

   void DeleteIntRuleArray(Array<IntegrationRule *> &ir_array);

public:
   /// Defines all integration rules
   explicit IntegrationRules(int refined = 0);

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

#endif
