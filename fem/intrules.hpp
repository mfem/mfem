// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_INTRULES
#define MFEM_INTRULES

#include "../config/config.hpp"
#include "../general/array.hpp"
#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
#include <omp.h>
#endif

#include <vector>
#include <map>

namespace mfem
{

class KnotVector;
class Mesh;

/* Classes for IntegrationPoint, IntegrationRule, and container class
   IntegrationRules.  Declares the global variable IntRules */

/// Class for integration point with weight
class IntegrationPoint
{
public:
   real_t x, y, z, weight;
   int index;

   void Init(int const i)
   {
      x = y = z = weight = 0.0;
      index = i;
   }

   void Set(const real_t *p, const int dim)
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

   void Get(real_t *p, const int dim) const
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

   void Set(const real_t x1, const real_t x2, const real_t x3, const real_t w)
   { x = x1; y = x2; z = x3; weight = w; }

   void Set3w(const real_t *p) { x = p[0]; y = p[1]; z = p[2]; weight = p[3]; }

   void Set3(const real_t x1, const real_t x2, const real_t x3)
   { x = x1; y = x2; z = x3; }

   void Set3(const real_t *p) { x = p[0]; y = p[1]; z = p[2]; }

   void Set2w(const real_t x1, const real_t x2, const real_t w)
   { x = x1; y = x2; weight = w; }

   void Set2w(const real_t *p) { x = p[0]; y = p[1]; weight = p[2]; }

   void Set2(const real_t x1, const real_t x2) { x = x1; y = x2; }

   void Set2(const real_t *p) { x = p[0]; y = p[1]; }

   void Set1w(const real_t x1, const real_t w) { x = x1; weight = w; }

   void Set1w(const real_t *p) { x = p[0]; weight = p[1]; }
};

/// Class for an integration rule - an Array of IntegrationPoint.
class IntegrationRule : public Array<IntegrationPoint>
{
private:
   friend class IntegrationRules;
   int Order = 0;
   /** @brief The quadrature weights gathered as a contiguous array. Created
       by request with the method GetWeights(). */
   mutable Array<real_t> weights;

   /// Define n-simplex rule (triangle/tetrahedron for n=2/3) of order (2s+1)
   void GrundmannMollerSimplexRule(int s, int n = 3);

   void AddTriMidPoint(const int off, const real_t weight)
   { IntPoint(off).Set2w(1./3., 1./3., weight); }

   void AddTriPoints3(const int off, const real_t a, const real_t b,
                      const real_t weight)
   {
      IntPoint(off + 0).Set2w(a, a, weight);
      IntPoint(off + 1).Set2w(a, b, weight);
      IntPoint(off + 2).Set2w(b, a, weight);
   }

   void AddTriPoints3(const int off, const real_t a, const real_t weight)
   { AddTriPoints3(off, a, 1. - 2.*a, weight); }

   void AddTriPoints3b(const int off, const real_t b, const real_t weight)
   { AddTriPoints3(off, (1. - b)/2., b, weight); }

   void AddTriPoints3R(const int off, const real_t a, const real_t b,
                       const real_t c, const real_t weight)
   {
      IntPoint(off + 0).Set2w(a, b, weight);
      IntPoint(off + 1).Set2w(c, a, weight);
      IntPoint(off + 2).Set2w(b, c, weight);
   }

   void AddTriPoints3R(const int off, const real_t a, const real_t b,
                       const real_t weight)
   { AddTriPoints3R(off, a, b, 1. - a - b, weight); }

   void AddTriPoints6(const int off, const real_t a, const real_t b,
                      const real_t c, const real_t weight)
   {
      IntPoint(off + 0).Set2w(a, b, weight);
      IntPoint(off + 1).Set2w(b, a, weight);
      IntPoint(off + 2).Set2w(a, c, weight);
      IntPoint(off + 3).Set2w(c, a, weight);
      IntPoint(off + 4).Set2w(b, c, weight);
      IntPoint(off + 5).Set2w(c, b, weight);
   }

   void AddTriPoints6(const int off, const real_t a, const real_t b,
                      const real_t weight)
   { AddTriPoints6(off, a, b, 1. - a - b, weight); }

   // add the permutations of (a,a,b)
   void AddTetPoints3(const int off, const real_t a, const real_t b,
                      const real_t weight)
   {
      IntPoint(off + 0).Set(a, a, b, weight);
      IntPoint(off + 1).Set(a, b, a, weight);
      IntPoint(off + 2).Set(b, a, a, weight);
   }

   // add the permutations of (a,b,c)
   void AddTetPoints6(const int off, const real_t a, const real_t b,
                      const real_t c, const real_t weight)
   {
      IntPoint(off + 0).Set(a, b, c, weight);
      IntPoint(off + 1).Set(a, c, b, weight);
      IntPoint(off + 2).Set(b, c, a, weight);
      IntPoint(off + 3).Set(b, a, c, weight);
      IntPoint(off + 4).Set(c, a, b, weight);
      IntPoint(off + 5).Set(c, b, a, weight);
   }

   void AddTetMidPoint(const int off, const real_t weight)
   { IntPoint(off).Set(0.25, 0.25, 0.25, weight); }

   // given a, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4(const int off, const real_t a, const real_t weight)
   {
      IntPoint(off).Set(a, a, a, weight);
      AddTetPoints3(off + 1, a, 1. - 3.*a, weight);
   }

   // given b, add the permutations of (a,a,a,b), where 3*a + b = 1
   void AddTetPoints4b(const int off, const real_t b, const real_t weight)
   {
      const real_t a = (1. - b)/3.;
      IntPoint(off).Set(a, a, a, weight);
      AddTetPoints3(off + 1, a, b, weight);
   }

   // add the permutations of (a,a,b,b), 2*(a + b) = 1
   void AddTetPoints6(const int off, const real_t a, const real_t weight)
   {
      const real_t b = 0.5 - a;
      AddTetPoints3(off,     a, b, weight);
      AddTetPoints3(off + 3, b, a, weight);
   }

   // given (a,b) or (a,c), add the permutations of (a,a,b,c), 2*a + b + c = 1
   void AddTetPoints12(const int off, const real_t a, const real_t bc,
                       const real_t weight)
   {
      const real_t cb = 1. - 2*a - bc;
      AddTetPoints3(off,     a, bc, weight);
      AddTetPoints3(off + 3, a, cb, weight);
      AddTetPoints6(off + 6, a, bc, cb, weight);
   }

   // given (b,c), add the permutations of (a,a,b,c), 2*a + b + c = 1
   void AddTetPoints12bc(const int off, const real_t b, const real_t c,
                         const real_t weight)
   {
      const real_t a = (1. - b - c)/2.;
      AddTetPoints3(off,     a, b, weight);
      AddTetPoints3(off + 3, a, c, weight);
      AddTetPoints6(off + 6, a, b, c, weight);
   }

public:
   IntegrationRule() :
      Array<IntegrationPoint>() { }

   /// Construct an integration rule with given number of points
   explicit IntegrationRule(int NP) :
      Array<IntegrationPoint>(NP)
   {
      for (int i = 0; i < this->Size(); i++)
      {
         (*this)[i].Init(i);
      }
   }

   /// Sets the indices of each quadrature point on initialization.
   /** Note that most calls to IntegrationRule::SetSize should be paired with a
       call to SetPointIndices in order for the indices to be set correctly. */
   void SetPointIndices();

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

   /// Return the quadrature weights in a contiguous array.
   /** If a contiguous array is not required, the weights can be accessed with
       a call like this: `IntPoint(i).weight`. */
   const Array<real_t> &GetWeights() const;

   /// @brief Return an integration rule for KnotVector @a kv, defined by
   /// applying this rule on each knot interval.
   IntegrationRule* ApplyToKnotIntervals(KnotVector const& kv) const;

   /// Destroys an IntegrationRule object
   ~IntegrationRule() { }
};

/// Class for defining different integration rules on each NURBS patch.
class NURBSMeshRules
{
public:
   /// Construct a rule for each patch, using SetPatchRules1D.
   NURBSMeshRules(const int numPatches, const int dim_) :
      patchRules1D(numPatches, dim_),
      npatches(numPatches), dim(dim_) { }

   /// Returns a rule for the element.
   IntegrationRule &GetElementRule(const int elem, const int patch,
                                   const int *ijk,
                                   Array<const KnotVector*> const& kv) const;

   /// Add a rule to be used for individual elements. Returns the rule index.
   std::size_t AddElementRule(IntegrationRule *ir_element)
   {
      elementRule.push_back(ir_element);
      return elementRule.size() - 1;
   }

   /// @brief Set the integration rule for the element of the given index. This
   /// rule is used instead of the rule for the patch containing the element.
   void SetElementRule(const std::size_t element,
                       const std::size_t elementRuleIndex)
   {
      elementToRule[element] = elementRuleIndex;
   }

   /// @brief Set 1D integration rules to be used as a tensor product rule on
   /// the patch with index @a patch. This class takes ownership of these rules.
   void SetPatchRules1D(const int patch,
                        std::vector<const IntegrationRule*> & ir1D);

   /// @brief For tensor product rules defined on each patch by
   /// SetPatchRules1D(), return a pointer to the 1D rule in the specified
   /// @a dimension.
   const IntegrationRule* GetPatchRule1D(const int patch,
                                         const int dimension) const
   {
      return patchRules1D(patch, dimension);
   }

   /// @brief For tensor product rules defined on each patch by
   /// SetPatchRules1D(), return the integration point with index (i,j,k).
   void GetIntegrationPointFrom1D(const int patch, int i, int j, int k,
                                  IntegrationPoint & ip);

   /// @brief Finalize() must be called before this class can be used for
   /// assembly. In particular, it defines data used by GetPointElement().
   void Finalize(Mesh const& mesh);

   /// @brief For tensor product rules defined on each patch by
   /// SetPatchRules1D(), returns the index of the element containing
   /// integration point (i,j,k) for patch index @a patch. Finalize() must be
   /// called first.
   int GetPointElement(int patch, int i, int j, int k) const
   {
      return pointToElem[patch](i,j,k);
   }

   int GetDim() const { return dim; }

   /// @brief For tensor product rules defined on each patch by
   /// SetPatchRules1D(), returns an array of knot span indices for each
   /// integration point in the specified @a dimension.
   const Array<int>& GetPatchRule1D_KnotSpan(const int patch,
                                             const int dimension) const
   {
      return patchRules1D_KnotSpan[patch][dimension];
   }

   ~NURBSMeshRules();

private:
   /// Tensor-product rules defined on all patches independently.
   Array2D<const IntegrationRule*> patchRules1D;

   /// Integration rules defined on elements.
   std::vector<IntegrationRule*> elementRule;

   std::map<std::size_t, std::size_t> elementToRule;

   std::vector<Array3D<int>> pointToElem;
   std::vector<std::vector<Array<int>>> patchRules1D_KnotSpan;

#ifndef MFEM_THREAD_SAFE
   // This is a temporary quadrature rule for integrating over the
   // current element in an assembly loop. It may be modified when
   // moving to a new element, and is therefore not thread-safe.
   mutable IntegrationRule temporaryElementRule;
#endif

   const int npatches;
   const int dim;
};

/// A Class that defines 1-D numerical quadrature rules on [0,1].
class QuadratureFunctions1D
{
public:
   /** @name Methods for calculating quadrature rules.
       These methods calculate the actual points and weights for the different
       types of quadrature rules. */
   ///@{
   static void GaussLegendre(const int np, IntegrationRule* ir);
   static void GaussLobatto(const int np, IntegrationRule *ir);
   static void OpenUniform(const int np, IntegrationRule *ir);
   static void ClosedUniform(const int np, IntegrationRule *ir);
   static void OpenHalfUniform(const int np, IntegrationRule *ir);
   static void ClosedGL(const int np, IntegrationRule *ir);
   ///@}

   /// A helper function that will play nice with Poly_1D::OpenPoints and
   /// Poly_1D::ClosedPoints
   static void GivePolyPoints(const int np, real_t *pts, const int type);

private:
   static void CalculateUniformWeights(IntegrationRule *ir, const int type);
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
      OpenHalfUniform = 4,  ///< aka "open half" Newton-Cotes
      ClosedGL        = 5   ///< aka closed Gauss Legendre
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

   Array<IntegrationRule *> PointIntRules;
   Array<IntegrationRule *> SegmentIntRules;
   Array<IntegrationRule *> TriangleIntRules;
   Array<IntegrationRule *> SquareIntRules;
   Array<IntegrationRule *> TetrahedronIntRules;
   Array<IntegrationRule *> PyramidIntRules;
   Array<IntegrationRule *> PrismIntRules;
   Array<IntegrationRule *> CubeIntRules;

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   Array<omp_lock_t> IntRuleLocks;
#endif

   void AllocIntRule(Array<IntegrationRule *> &ir_array, int Order) const
   {
      if (ir_array.Size() <= Order)
      {
         ir_array.SetSize(Order + 1, NULL);
      }
   }
   bool HaveIntRule(Array<IntegrationRule *> &ir_array, int Order) const
   {
      return (ir_array.Size() > Order && ir_array[Order] != NULL);
   }
   int GetSegmentRealOrder(int Order) const
   {
      return Order | 1; // valid for all quad_type's
   }
   void DeleteIntRuleArray(Array<IntegrationRule *> &ir_array) const;

   /// The following methods allocate new IntegrationRule objects without
   /// checking if they already exist.  To avoid memory leaks use
   /// IntegrationRules::Get(int GeomType, int Order) instead.
   IntegrationRule *GenerateIntegrationRule(int GeomType, int Order);
   IntegrationRule *PointIntegrationRule(int Order);
   IntegrationRule *SegmentIntegrationRule(int Order);
   IntegrationRule *TriangleIntegrationRule(int Order);
   IntegrationRule *SquareIntegrationRule(int Order);
   IntegrationRule *TetrahedronIntegrationRule(int Order);
   IntegrationRule *PyramidIntegrationRule(int Order);
   IntegrationRule *PrismIntegrationRule(int Order);
   IntegrationRule *CubeIntegrationRule(int Order);

public:
   /// Sets initial sizes for the integration rule arrays, but rules
   /// are defined the first time they are requested with the Get method.
   explicit IntegrationRules(int ref = 0,
                             int type = Quadrature1D::GaussLegendre);

   /// Returns an integration rule for given GeomType and Order.
   const IntegrationRule &Get(int GeomType, int Order);

   void Set(int GeomType, int Order, IntegrationRule &IntRule);

   void SetOwnRules(int o) { own_rules = o; }

   /// Destroys an IntegrationRules object
   ~IntegrationRules();
};

/// A global object with all integration rules (defined in intrules.cpp)
extern MFEM_EXPORT IntegrationRules IntRules;

/// A global object with all refined integration rules
extern MFEM_EXPORT IntegrationRules RefinedIntRules;

}

#endif
