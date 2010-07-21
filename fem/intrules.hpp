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
};

// Class for integration rule
class IntegrationRule
{
private:
   int NPoints;
   IntegrationPoint * IntPoints;

   friend class IntegrationRules;

   /// Computes Gaussian integration rule on (0,1) with NPoints
   void GaussianRule();

   /// Defines composite trapezoidal integration rule on [0,1]
   void UniformRule();

public:
   IntegrationRule () {}

   /// Construct an integration rule with given number of points
   IntegrationRule (int NP);

   /// Tensor product of two 1D integration rules
   IntegrationRule (IntegrationRule &irx, IntegrationRule &iry);

   /// Returns the number of the points in the integration rule
   int GetNPoints() const { return NPoints; };

   /// Returns a reference to the i-th integration point
   IntegrationPoint & IntPoint (int i) { return IntPoints[i]; };

   /// Returns a const reference to the i-th integration point
   const IntegrationPoint & IntPoint (int i) const { return IntPoints[i]; };

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

public:
   /// Defines all integration rules
   IntegrationRules(int refined = 0);

   /// Returns an integration rule for given GeomType and Order.
   const IntegrationRule & Get (int GeomType, int Order);

   void Set (int GeomType, int Order, IntegrationRule &IntRule);

   void SetOwnRules (int o) { own_rules = o; };

   /// Destroys an IntegrationRules object
   ~IntegrationRules();
};

/// A global object with all integration rules (defined in intrules.cpp)
extern IntegrationRules IntRules;

/// A global object with all refined integration rules
extern IntegrationRules RefinedIntRules;

#endif
