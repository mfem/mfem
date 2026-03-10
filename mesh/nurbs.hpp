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

#ifndef MFEM_NURBS
#define MFEM_NURBS

#include "../config/config.hpp"
#include "../general/table.hpp"
#include "../linalg/vector.hpp"
#include "element.hpp"
#include "mesh.hpp"
#include "spacing.hpp"
#ifdef MFEM_USE_MPI
#include "../general/communication.hpp"
#endif
#include <iostream>

namespace mfem
{

class GridFunction;

/** @brief A vector of knots in one dimension, with B-spline basis functions of
    a prescribed order.

    @note Order is defined in the sense of "The NURBS book" - 2nd ed - Piegl and
    Tiller, cf. section 2.2.
*/
class KnotVector
{
protected:
   static const int MaxOrder;

   /// Stores the values of all knots.
   Vector knot;

   /// Order of the B-spline basis functions.
   int Order;

   /// Number of control points.
   int NumOfControlPoints;

   /// Number of elements, defined by distinct knots.
   int NumOfElements;

   // Stores the demko points
   mutable Vector demko;

   /// Compute all the Demko points
   void ComputeDemko() const;

#ifdef MFEM_USE_LAPACK
   // Data for reusing banded matrix factorization in FindInterpolant().
   mutable DenseMatrix fact_AB; /// Banded matrix factorization
   mutable Array<int> fact_ipiv; /// Row pivot indices
#else
   mutable DenseMatrix A_coll_inv; /// Collocation matrix inverse
#endif


public:
   /// Create an empty KnotVector.
   KnotVector() = default;

   /** @brief Create a KnotVector by reading data from stream @a input. Two
       integers are read, for order and number of control points. */
   KnotVector(std::istream &input);

   /** @brief Create a KnotVector with order @a order.
       When @a NCP is not provided  the number of control points is set to
       @a order + 1, and the first @a order + 1 knots are set to 0 and last
       @a order + 1 knots are set to 1.
       When @a NCP is given number of control points is @a NCP and
       the knots are initialized to -1) */
   KnotVector(int order, int NCP = -1);

   /** @brief Create a KnotVector with order @a order and knots @a knot.
       If @a k has the correct number of repeated knots at the begin and end,
       then this constructor will copy the knots as provided.
       Otherwise, the knot vector will be extended by repeating the end knots
       (order + 1) times. Internal knots will retain the multiplicity as given
       in the input. */
   KnotVector(int order, const Vector &k);

   /** @brief Create a KnotVector by passing in a degree, a Vector of interval
       lengths of length n, and a list of continuity of length n + 1.
       The intervals refer to spans between unique knot values (not counting
       zero-size intervals at repeated knots), and the continuity values should
       be >= -1 (discontinuous) and <= order-1 (maximally-smooth for the given
       polynomial degree). Periodicity is not supported.*/
   KnotVector(int order, const Vector& intervals,
              const Array<int>& continuity);

   /// Copy constructor.
   KnotVector(const KnotVector &kv) { (*this) = kv; }

   KnotVector &operator=(const KnotVector &kv);

   /// Return the number of elements, defined by distinct knots.
   int GetNE()    const { return NumOfElements; }

   /// Return the number of control points.
   int GetNCP()   const { return NumOfControlPoints; }

   /// Return the order.
   int GetOrder() const { return Order; }

   /// Return the number of knots, including multiplicities.
   int Size()     const { return knot.Size(); }

   /// Count the number of elements.
   void GetElements();

   /** @brief Return whether the knot index Order plus @a i is the beginning of
       an element. */
   bool isElement(int i) const { return (knot(Order+i) != knot(Order+i+1)); }

   /** @brief Return the number of control points minus the order. This is not
       the number of knot spans, but it gives the number of knots to be checked
       with @a isElement for non-empty knot spans (elements). */
   int GetNKS() const { return NumOfControlPoints - Order; }

   /// Return whether knot location @a u is in a given span @a ni.
   bool inSpan(real_t u, int ni) const
   {
      if ((u < knot(ni)) || (u > knot(ni+1))) { return false; }
      return true;
   }

   /// Return the index of the knot span containing parameter @a u.
   int GetSpan(real_t u) const;

   /** @brief Return the reference coordinate in [0,1] for parameter @a u
       in the element beginning at knot @a ni. */
   real_t GetRefPoint(real_t u, int ni) const
   { return (u-knot(ni))/(knot(ni+1)-knot(ni)); };

   /** @brief Return the knot location for element reference coordinate @a xi
       in [0,1], for the element beginning at knot @a ni. */
   real_t GetKnotLocation(real_t xi, int ni) const
   { return (xi*knot(ni+1) + (1. - xi)*knot(ni)); }

   /** @brief Return the parameter for element reference coordinate @a xi
       in [0,1], for the element beginning at knot @a ni. */
   MFEM_DEPRECATED real_t getKnotLocation(real_t xi, int ni) const
   { return (xi*knot(ni+1) + (1. - xi)*knot(ni)); } // Use GetKnotLocation instead

   /// Return the index of the knot span containing parameter @a u.
   MFEM_DEPRECATED int findKnotSpan(real_t u) const;  // Use GetSpan instead

   /** Gives the @a i average knot location. Average is taken over @a Order
       number of knots.*/
   real_t GetGreville(int i) const;

   void GetGreville(Vector &xi) const;

   /** Gives the knot location where the @a i shape function is maximum.
       Reverts to the Greville point if knot is repeated @a Order +1 times.
       For background see:

       Olivier Botella and Karim Shariff.
       "B-spline methods in fluid dynamics."
       International Journal of Computational Fluid Dynamics 17.2 (2003): 133-149.

       Points are found using Newton iteration, with the Greville point as the
       starting value. */
   real_t GetBotella(int i) const;

   void GetBotella(Vector &xi) const;

   /** Gives the knot location of the @a i extremum of the Chebyshev spline.
       For background see:

       Stephen Demko
       "On the existence of interpolating projections onto spline spaces."
       Journal of approximation theory 43.2 (1985): 151-156.

       Points are found using Remez iteration:
        - Find interpolant, given by a, through given points, given by Demko
        - Find extrema of this polynomial and update Demko points
        - Repeat until converged
        - Use the Greville point as starting point */
   real_t GetDemko(int i) const;

   void GetDemko(Vector &xi) const;

   // The following functions evaluate shape functions, which are B-spline basis
   // functions.

   /** @brief Calculate the nonvanishing shape function values in @a shape for
       the element corresponding to knot index @a i and element reference
       coordinate @a xi. */
   void CalcShape  (Vector &shape, int i, real_t xi) const;

   /** @brief Calculate derivatives of the nonvanishing shape function values in
       @a grad for the element corresponding to knot index @a i and element
       reference coordinate @a xi. */
   void CalcDShape (Vector &grad,  int i, real_t xi) const;

   /** @brief Calculate n-th derivatives (order @a n) of the nonvanishing shape
       function values in @a grad for the element corresponding to knot index
       @a i and element reference coordinate @a xi. */
   void CalcDnShape(Vector &gradn, int n, int i, real_t xi) const;

   /// Calculate second-order shape function derivatives, using CalcDnShape.
   void CalcD2Shape(Vector &grad2, int i, real_t xi) const
   { CalcDnShape(grad2, 2, i, xi); }

   /** @brief Gives the locations of the maxima of the KnotVector in reference
       space. The function gives the knot span @a ks, the coordinate in the
       knot span @a xi, and the coordinate of the maximum in parameter space
       @a u.
       The main purpose of this function is its use in FindInterpolant.
       Use GetBotella instead for each shape function separately, perhaps in
       conjuction with GetSpan and GetRefPoint.*/
   MFEM_DEPRECATED void FindMaxima(Array<int> &ks, Vector &xi, Vector &u) const;

   /** @brief Global curve interpolation through the points @a x (overwritten).
       @a x is an array with the length of the spatial dimension containing
       vectors with spatial coordinates. The control points of the interpolated
       curve are returned in @a x in the same form.
       Use GetInterpolant instead. For the knot location one can use either
       GetBotella, GetDemko or GetGreville. FindInterpolant uses the Botella
       points, however, the Demko points might be more appropriate. */
   MFEM_DEPRECATED void FindInterpolant(Array<Vector*> &x, bool reuse_inverse);

   /** @brief Global curve interpolation through the points @a x (overwritten)
       at the knot location @a u. The control points of the
       interpolated curve are returned in @a x in the same form.
       For the knot location one can use  for instance GetBotella, GetDemko or
       GetGreville. The Demko points might be most appropriate.*/
   void GetInterpolant(Array<Vector*> &x, const Vector &u,
                       bool reuse_inverse = false) const;

   /// Different interface to same routine
   void GetInterpolant(const Vector &x, const Vector &u,
                       Vector &a, bool reuse_inverse = false) const;

   /** Set @a diff, comprised of knots in @a kv not contained in this KnotVector.
       @a kv must be of the same order as this KnotVector. The current
       implementation is not well defined, and the function may have undefined
       behavior, as @a diff may have unset entries at the end. */
   void Difference(const KnotVector &kv, Vector &diff) const;

   /// Uniformly refine by factor @a rf, by inserting knots in each span.
   void UniformRefinement(Vector &new_knots, int rf) const;

   /// Refine with refinement factor @a rf.
   void Refinement(Vector &new_knots, int rf) const;

   /** Returns the coarsening factor needed for non-nested nonuniform spacing
       functions, to result in a single element from which refinement can be
       done. The return value is 1 if uniform or nested spacing is used. */
   int GetCoarseningFactor() const;

   /** For a given coarsening factor @a cf, find the fine knots between the
       coarse knots. */
   Vector GetFineKnots(const int cf) const;

   /** @brief Return a new KnotVector with elevated degree by repeating the
       endpoints of the KnotVector. */
   /// @note The returned object should be deleted by the caller.
   KnotVector *DegreeElevate(int t) const;

   /// Reverse the knots.
   void Flip();

   /** @brief Print the order, number of control points, and knots.

       The output is formatted for writing a mesh to file. This function is
       called by NURBSPatch::Print. */
   void Print(std::ostream &os) const;

   /** @brief Prints the non-zero shape functions and their first and second
       derivatives associated with the KnotVector per element. Use GetElements()
       to count the elements before using this function. @a samples is the
       number of samples of the shape functions per element.*/
   void PrintFunctions(std::ostream &os, int samples=11) const;

   /** Prints the function with basis function coefficient @a a, and its first
       and second derivatives associated with the KnotVector per element.
       Use GetElements() to count the elements before using this function.
       @a samples is the number of samples of the shape functions per element.*/
   void PrintFunction(std::ostream &os, const Vector &a, int samples=11) const;

   /** Prints the @a i-th function and its first and second
       derivatives associated with the KnotVector per element. Use GetElements()
       to count the elements before using this function. @a samples is the
       number of samples of the shape functions per element.*/
   void PrintFunction(std::ostream &os, int i, int samples=11) const;

   /// Destroys KnotVector
   ~KnotVector() { }

   /// Access function to knot @a i.
   real_t &operator[](int i) { return knot(i); }

   /// Const access function to knot @a i.
   const real_t &operator[](int i) const { return knot(i); }

   /// Coarsen to a single element.
   KnotVector* FullyCoarsen();

   /// Function to define the distribution of knots for any number of knot spans.
   std::shared_ptr<SpacingFunction> spacing;

   /** @brief Flag to indicate whether the KnotVector has been coarsened, which
       means it is ready for non-nested refinement. */
   bool coarse;
};


/** @brief A NURBS patch can be 1D, 2D, or 3D, and is defined as a tensor
    product of KnotVectors. */
class NURBSPatch
{
protected:

   /// B-NET dimensions
   int ni, nj, nk;

   /// Physical dimension plus 1
   int Dim;

   /// Data with the layout (Dim x ni x nj x nk)
   real_t *data;

   /// KnotVectors in each direction
   Array<KnotVector *> kv;

   // Special B-NET access functions
   //  - SetLoopDirection(int dir) flattens the multi-dimensional B-NET in the
   //    requested direction. It effectively creates a 1D net in homogeneous
   //    coordinates.
   //  - The slice(int, int) operator is the access function in that flattened
   //    structure. The first int gives the slice and the second int the element
   //    in that slice.
   //  - Both routines are used in 'KnotInsert', `KnotRemove`, 'DegreeElevate',
   //    and 'UniformRefinement'.
   //  - In older implementations, slice(int, int) was implemented as
   //    operator()(int, int).
   int nd; // Number of control points in flattened structure
   int ls; // Number of variables per control point in flattened structure
   int sd; // Stride for data access

   /** @brief Flattens the B-NET in direction @a dir, producing a 1D net.
       Returns the number of variables per knot in flattened structure. */
   int SetLoopDirection(int dir);

   /** @brief Access function for the effectively 1D flattened net, where @a i
       is a knot index, and @a j is an index of a variable per knot. */
   inline       real_t &slice(int i, int j);
   inline const real_t &slice(int i, int j) const;

   /// Copy constructor
   NURBSPatch(NURBSPatch *parent, int dir, int Order, int NCP);

   /// Deletes own data, takes data from @a np, and deletes np.
   void swap(NURBSPatch *np);

   /// Sets dimensions and allocates data, based on KnotVectors.
   /// @a dim is the physical dimension plus 1.
   void init(int dim);

public:
   /// Copy constructor
   NURBSPatch(const NURBSPatch &orig);

   /// Constructor using data read from stream @a input.
   NURBSPatch(std::istream &input);

   /// Constructor for a 2D patch. @a dim is the physical dimension plus 1.
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1, int dim);

   /// Constructor for a 3D patch.
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1,
              const KnotVector *kv2, int dim);

   /** Create a bivariate NURBS patch with given control points. See n-variate
       overload for additional notes. */
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1, int dim_,
              const real_t* control_points);
   /** Create a trivariate NURBS patch with given control points. See n-variate
       overload for additional notes. */
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1,
              const KnotVector *kv2, int dim_, const real_t* control_points);
   /** Create an n-variate NURBS patch with given control points of dimension
       dim_, where n is the length of the array of knot vectors and dim_
       includes the weight. The array of control point coordinates stores each
       point's coordinates contiguously, and points are ordered in a standard
       ijk grid ordering. */
   NURBSPatch(Array<const KnotVector *> &kv_, int dim_,
              const real_t* control_points);

   /// Constructor for a patch of dimension equal to the size of @a kv.
   NURBSPatch(Array<const KnotVector *> &kv, int dim);

   /// Copy assignment not supported.
   NURBSPatch& operator=(const NURBSPatch&) = delete;

   /// Deletes data and KnotVectors.
   ~NURBSPatch();

   /** @brief Writes KnotVectors and data to the stream @a os.

       The output is formatted for writing a mesh to file. This function is
       called by NURBSExtension::Print. */
   void Print(std::ostream &os) const;

   /// Increase the order in direction @a dir by @a t >= 0.
   void DegreeElevate(int dir, int t);

   /// Increase the order in all directions by @a t >= 0.
   void DegreeElevate(int t);

   /**  @brief Insert any new knots from @a knot in direction @a dir. If the
        order of @a knot is higher than the current order in direction
        @a dir, then the order is elevated in that direction to match. */
   void KnotInsert(int dir, const KnotVector &knot);

   /** @brief Insert knots from @a knot in direction @a dir. If a knot already
       exists, then it is still added, increasing its multiplicity. */
   void KnotInsert(int dir, const Vector &knot);

   /// Call KnotInsert for each direction with the corresponding @a knot entry.
   void KnotInsert(Array<Vector *> &knot);
   /// Insert knots from @a knot determined by @a Difference, in each direction.
   void KnotInsert(Array<KnotVector *> &knot);

   /** @brief Remove knot with value @a knot from direction @a dir.

       The optional input parameter @a ntimes specifies the number of times the
       knot should be removed, default 1. The knot is removed only if the new
       curve (in direction @a dir) deviates from the old curve by less than
       @a tol.

       @returns The number of times the knot was successfully removed. */
   int KnotRemove(int dir, real_t knot, int ntimes=1, real_t tol = 1.0e-12);

   /// Remove all knots in @a knot once.
   void KnotRemove(int dir, Vector const& knot, real_t tol = 1.0e-12);
   /// Remove all knots in @a knot once, for each direction.
   void KnotRemove(Array<Vector *> &knot, real_t tol = 1.0e-12);

   /** @brief Refine with optional refinement factor @a rf. Uniform means
       refinement is done everywhere by the same factor, although nonuniform
       spacing functions may be used.

       @param[in] rf Optional refinement factor. If scalar, the factor is used
                     for all dimensions. If an array, factors can be specified
                     for each dimension.
       @param[in] multiplicity Optional multiplicity for new knots inserted. */
   void UniformRefinement(int rf = 2, int multiplicity = 1);
   void UniformRefinement(const Array<int> &rf, int multiplicity = 1);

   /// Flag @a coarsened indicates whether the patch is a single element.
   void UniformRefinement(const std::vector<Array<int>> &rf,
                          bool coarsened = false, int multiplicity = 1);

   /** @brief Coarsen with optional coarsening factor @a cf which divides the
       number of elements in each dimension. Nonuniform spacing functions may be
       used in each direction.

       @param[in] cf  Optional coarsening factor. If scalar, the factor is used
                      for all dimensions. If an array, factors can be specified
                      for each dimension.
       @param[in] tol NURBS geometry deviation tolerance, cf. Algorithm A5.8 of
       "The NURBS Book", 2nd ed, Piegl and Tiller. */
   void Coarsen(int cf = 2, real_t tol = 1.0e-12);
   void Coarsen(const Array<int> &cf, real_t tol = 1.0e-12);

   /// Calls KnotVector::GetCoarseningFactor for each direction.
   void GetCoarseningFactors(Array<int> &f) const;

   /// Marks the KnotVector in each dimension as coarse.
   void SetKnotVectorsCoarse(bool c);

   /// Coarsen to a single element.
   void FullyCoarsen(const Array2D<double> &cp, int ncp1D);

   /// Update piecewise spacing function partitions to match refined @a pkv.
   void UpdateSpacingPartitions(const Array<KnotVector*> &pkv);

   /// Return the number of components stored in the NURBSPatch
   int GetNC() const { return Dim; }

   /// Return the number of KnotVectors, which is the patch dimension.
   int GetNKV() const { return kv.Size(); }

   /// Return a pointer to the KnotVector in direction @a dir.
   /// @note The returned object should NOT be deleted by the caller.
   KnotVector *GetKV(int dir) { return kv[dir]; }

   // Standard B-NET access functions

   /// 1D access function. @a i is a B-NET index, and @a l is a variable index.
   inline       real_t &operator()(int i, int l);
   inline const real_t &operator()(int i, int l) const;

   /** @brief 2D access function. @a i, @a j are B-NET indices, and @a l is a
       variable index. */
   inline       real_t &operator()(int i, int j, int l);
   inline const real_t &operator()(int i, int j, int l) const;

   /** @brief 3D access function. @a i, @a j, @a k are B-NET indices, and @a l
       is a variable index. */
   inline       real_t &operator()(int i, int j, int k, int l);
   inline const real_t &operator()(int i, int j, int k, int l) const;

   /// Compute the 2D rotation matrix @a T for angle @a angle.
   static void Get2DRotationMatrix(real_t angle, DenseMatrix &T);

   /** @brief Compute the 3D rotation matrix @a T for angle @a angle around
       axis @a n (a 3D vector, not necessarily normalized) and scalar factor
       @a r. */
   static void Get3DRotationMatrix(real_t n[], real_t angle, real_t r,
                                   DenseMatrix &T);

   /// Reverse data and knots in direction @a dir.
   void FlipDirection(int dir);

   /// Swap data and KnotVectors in directions @a dir1 and @a dir2.
   /** @note Direction pairs (0,2) and (2,0) are not supported, resulting in an
       error being thrown. */
   void SwapDirections(int dir1, int dir2);

   /// Rotate the NURBSPatch in 2D or 3D..
   /** A rotation of a 2D NURBS-patch requires an angle only. Rotating
       a 3D NURBS-patch requires a normal as well.*/
   void Rotate(real_t angle, real_t normal[] = NULL);

   /// Rotate the NURBSPatch, 2D case.
   void Rotate2D(real_t angle);

   /// Rotate the NURBSPatch, 3D case.
   void Rotate3D(real_t normal[], real_t angle);

   /** Elevate KnotVectors in all directions to degree @a degree if given,
       otherwise to the maximum current degree among all directions. */
   int MakeUniformDegree(int degree = -1);

   /** @brief Given two patches @a p1 and @a p2 of the same dimensions, create
       and return a new patch by merging their knots and data. */
   /// @note The returned object should be deleted by the caller.
   friend NURBSPatch *Interpolate(NURBSPatch &p1, NURBSPatch &p2);

   /// Create and return a new patch by revolving @a patch in 3D.
   /// @note The returned object should be deleted by the caller.
   friend NURBSPatch *Revolve3D(NURBSPatch &patch, real_t n[], real_t ang,
                                int times);
};


#ifdef MFEM_USE_MPI
class ParNURBSExtension;
#endif

class NURBSPatchMap;

/** @brief NURBSExtension generally contains multiple NURBSPatch objects
    spanning an entire Mesh. It also defines and manages DOFs in NURBS finite
    element spaces. */
class NURBSExtension
{
#ifdef MFEM_USE_MPI
   friend class ParNURBSExtension;
#endif
   friend class NURBSPatchMap;

protected:

   /// Flag for indicating what type of NURBS fespace this extension is used for.
   enum class Mode
   {
      H_1,    ///> Extension for a standard scalar-valued space
      H_DIV,  ///> Extension for a divergence conforming vector-valued space
      H_CURL, ///> Extension for a curl conforming vector-valued space
   };
   Mode mode = Mode::H_1;

   /// Order of KnotVectors, see GetOrder() for description.
   int mOrder;

   /// Orders of all KnotVectors
   Array<int> mOrders;

   /// Number of unique (not comprehensive) KnotVectors
   int NumOfKnotVectors;

   /// Global entity counts
   int NumOfVertices, NumOfElements, NumOfBdrElements, NumOfDofs;

   /// Local entity counts
   int NumOfActiveVertices, NumOfActiveElems, NumOfActiveBdrElems;
   int NumOfActiveDofs;

   Array<int>  activeVert; // activeVert[glob_vert] = loc_vert or -1
   Array<bool> activeElem;
   Array<bool> activeBdrElem;
   Array<int>  activeDof; // activeDof[glob_dof] = loc_dof + 1 or 0

   /// Patch topology mesh
   Mesh *patchTopo;

   /// Whether this object owns patchTopo
   bool own_topo;

   /// Map from patchTopo edge indices to unique KnotVector indices
   Array<int> edge_to_ukv;

   /// Set of unique KnotVectors
   Array<KnotVector *> knotVectors;

   /// Comprehensive set of all KnotVectors, one for every edge.
   Array<KnotVector *> knotVectorsCompr;

   /// Weights for each control point or DOF
   Vector weights;

   /** @brief Periodic BC info:
       - dof 2 dof map
       - master and slave boundary indices */
   Array<int> d_to_d;
   Array<int> master;
   Array<int> slave;

   /// Global mesh offsets, meshOffsets == meshVertexOffsets
   Array<int> v_meshOffsets;
   Array<int> e_meshOffsets;
   Array<int> f_meshOffsets;
   Array<int> p_meshOffsets;

   /// Global space offsets, spaceOffsets == dofOffsets
   Array<int> v_spaceOffsets;
   Array<int> e_spaceOffsets;
   Array<int> f_spaceOffsets;
   Array<int> p_spaceOffsets;

   /// Table of DOFs for each element (el_dof) or boundary element (bel_dof).
   Table *el_dof, *bel_dof;

   /// Map from element indices to patch indices
   Array<int> el_to_patch;
   /// Map from boundary element indices to patch indices
   Array<int> bel_to_patch;

   /// Map from element indices to IJK knot span indices
   Array2D<int> el_to_IJK;  // IJK are "knot-span" indices!
   Array2D<int> bel_to_IJK; // they are NOT element indices!

   /// For each patch p, @a patch_to_el[p] lists all elements in the patch.
   std::vector<Array<int>> patch_to_el;
   /// For each patch p, @a patch_to_bel[p] lists all boundary elements in the patch.
   std::vector<Array<int>> patch_to_bel;

   /// Array of all patches in the mesh.
   Array<NURBSPatch*> patches;

   /// Return the unsigned index of the KnotVector for edge @a edge.
   inline int KnotInd(int edge) const;
   /// Return the sign (orientation) of the KnotVector for edge @a edge.
   inline int KnotSign(int edge) const;

   bool nonconformingPT = false; /// Whether patchTopo is a nonconforming mesh.

   int num_structured_patches = 0; /// Number of structured patches

   Array3D<double> patchCP; /// Control points for coarse structured patches

   std::vector<Array<int>> kvf, kvf_coarse; /// Knotvector refinement factors

   Array<int> ref_factors; /// Refinement factors in each dimension.

   static constexpr int unsetFactor = 0; /// Unset refinement factor value

   Array<int> dof2patch; /// DOF to owning patch map in @a SetSolutionVector()

   /// Access function for the KnotVector associated with edge @a edge.
   /// @note The returned object should NOT be deleted by the caller.
   inline KnotVector *KnotVec(int edge);
   /// Const access function for the KnotVector associated with edge @a edge.
   /// @note The returned object should NOT be deleted by the caller.
   inline const KnotVector *KnotVec(int edge) const;
   /** @brief Const access function for the KnotVector associated with edge
       @a edge. The output orientation @a okv is set to @a oedge with sign flipped
       if the KnotVector index associated with edge @a edge is negative. */
   inline const KnotVector *KnotVec(int edge, int oedge, int *okv) const;

   /// Throw an error if any boundary patch has invalid KnotVector orientation.
   MFEM_DEPRECATED void CheckBdrPatches();

   /// Return the patch-topology edge indices that define the KnotVectors for
   /// patch @a p in each parametric direction.
   void GetPatchDirectionEdges(int p, Array<int> &edges);

   /** @brief Return the directions in @a kvdir of the KnotVectors in patch @a p
       based on the patch edge orientations. Each entry of @a kvdir is -1 if the
       KnotVector direction is flipped, +1 otherwise. */
   void CheckKVDirection(int p, Array <int> &kvdir);

   /** @brief Create the comprehensive set of KnotVectors, one per patch and
       parametric direction, accounting for the edge orientations. */
   void CreateComprehensiveKV();

   /** @brief Update the unique set of KnotVectors from the comprehensive set
       of KnotVectors. */
   void UpdateUniqueKV();

   /** @brief Check if the comprehensive array of KnotVectors agrees with the
       unique set of KnotVectors, on each patch. Return false if there is a
       difference, true otherwise. */
   bool ConsistentKVSets();

   /// Return KnotVectors in @a kv in each dimension for patch @a p.
   void GetPatchKnotVectors   (int p, Array<KnotVector *> &kv);
   /// Return KnotVectors in @a kv in each dimension for boundary patch @a bp.
   void GetBdrPatchKnotVectors(int bp, Array<KnotVector *> &kv);

   /// Set overall order @a mOrder based on KnotVector orders.
   void SetOrderFromOrders();

   /// Set orders from KnotVector orders.
   void SetOrdersFromKnotVectors();

   // Periodic BC helper functions

   /// Set DOF map sizes to 0.
   void InitDofMap();

   /// Set DOF maps for periodic BC.
   void ConnectBoundaries();
   void ConnectBoundaries1D(int bnd0, int bnd1);
   void ConnectBoundaries2D(int bnd0, int bnd1);
   void ConnectBoundaries3D(int bnd0, int bnd1);

   /** @brief Set the mesh and space offsets, and also count the global
   @a NumOfVertices and the global @a NumOfDofs. */
   virtual void GenerateOffsets();

   /// Count the global @a NumOfElements.
   void CountElements();
   /// Count the global @a NumOfBdrElements.
   void CountBdrElements();

   /// Generate the active mesh elements and return them in @a elements.
   void Get1DElementTopo(Array<Element *> &elements) const;
   void Get2DElementTopo(Array<Element *> &elements) const;
   void Get3DElementTopo(Array<Element *> &elements) const;

   /// Generate the active mesh boundary elements and return them in @a boundary.
   void Get1DBdrElementTopo(Array<Element *> &boundary) const;
   void Get2DBdrElementTopo(Array<Element *> &boundary) const;
   void Get3DBdrElementTopo(Array<Element *> &boundary) const;

   // FE space generation functions

   /** @brief Based on activeElem, count NumOfActiveDofs and generate el_dof,
       el_to_patch, el_to_IJK, activeDof map (global-to-local). */
   void GenerateElementDofTable();

   /** @brief Generate elem_to_global-dof table for the active elements, and
       define el_to_patch, el_to_IJK, activeDof (as bool). */
   void Generate1DElementDofTable();
   void Generate2DElementDofTable();
   void Generate3DElementDofTable();

   /// Call after GenerateElementDofTable to set boundary element DOF table.
   void GenerateBdrElementDofTable();

   /** @brief Generate the table of global DOFs for active boundary elements,
       and define bel_to_patch, bel_to_IJK. */
   void Generate1DBdrElementDofTable();
   void Generate2DBdrElementDofTable();
   void Generate3DBdrElementDofTable();

   // FE --> Patch translation functions

   /// Set the B-NET on each patch using values from @a coords.
   void GetPatchNets  (const Vector &coords, int vdim);
   void Get1DPatchNets(const Vector &coords, int vdim);
   void Get2DPatchNets(const Vector &coords, int vdim);
   void Get3DPatchNets(const Vector &coords, int vdim);

   // Patch --> FE translation functions

   /** @brief Return in @a coords the coordinates from each patch. Side effects:
       delete the patches and update the weights from the patches. */
   void SetSolutionVector  (Vector &coords, int vdim);
   void Set1DSolutionVector(Vector &coords, int vdim);
   void Set2DSolutionVector(Vector &coords, int vdim);
   void Set3DSolutionVector(Vector &coords, int vdim);

   /// Determine activeVert, NumOfActiveVertices from the activeElem array.
   void GenerateActiveVertices();

   /// Determine activeBdrElem, NumOfActiveBdrElems.
   void GenerateActiveBdrElems();

   /** @brief Set the weights in this object to values from active elements in
       @a num_pieces meshes in @a mesh_array. */
   void MergeWeights(Mesh *mesh_array[], int num_pieces);

   /// Set @a patch_to_el.
   void SetPatchToElements();
   /// Set @a patch_to_bel.
   void SetPatchToBdrElements();

   /// Load data from file (used by constructor).
   void Load(std::istream &input, bool spacing);

   /// Return true if @a edge is a master NC-patch edge.
   virtual bool IsMasterEdge(int edge) const { return false; }

   /// Return true if @a face is a master NC-patch face.
   virtual bool IsMasterFace(int face) const { return false; }

   /// Given a pair of vertices, return the corresponding edge.
   virtual int VertexPairToEdge(const std::pair<int, int> &vertices) const;

   /** @brief Get the DOFs (dof = true) or vertices (dof = false) for
        master edge @a me. */
   virtual void GetMasterEdgeDofs(bool dof, int me, Array<int> &dofs) const;

   /** @brief Get the DOFs (dof = true) or vertices (dof = false) for
       master face @a mf. */
   virtual void GetMasterFaceDofs(bool dof, int mf, Array2D<int> &dofs) const;

   /// Helper function for @a GenerateOffsets().
   void GetPatchOffsets(int &meshCounter, int &spaceCounter);

   /// Return NURBSPatch object; returned object should NOT be deleted.
   const NURBSPatch* GetPatch(int patch) const { return patches[patch]; }

   /// To be used by ParNURBSExtension constructor(s)
   NURBSExtension() : el_dof(nullptr), bel_dof(nullptr) { }

private:
   /// Get the degrees of freedom for the vertex @a vertex in @a dofs.
   void GetVertexDofs(int vertex, Array<int> &dofs) const;

   /// Get the degrees of freedom for the edge @a edge in @a dofs.
   void GetEdgeDofs(int edge, Array<int> &dofs) const;

   // TODO: does this still need to be virtual?
   /// Helper function for @a GenerateOffsets().
   virtual void SetDofToPatch() { };

public:
   /// Copy constructor: deep copy
   NURBSExtension(const NURBSExtension &orig);
   /// Read-in a NURBSExtension from a stream @a input.
   NURBSExtension(std::istream &input, bool spacing=false);
   /** @brief Create a NURBSExtension with elevated order by repeating the
       endpoints of the KnotVectors and using uniform weights of 1. */
   /** @note If a KnotVector in @a parent already has order greater than or
       equal to @a newOrder, it will be used unmodified. */
   NURBSExtension(NURBSExtension *parent, int newOrder);
   /** @brief Create a NURBSExtension with elevated KnotVector orders (by
       repeating the endpoints of the KnotVectors and using uniform weights of
       1) as given by the array @a newOrders. */
   /** @a note If a KnotVector in @a parent already has order greater than or
       equal to the corresponding entry in @a newOrder, it will be used
       unmodified. */
   NURBSExtension(NURBSExtension *parent, const Array<int> &newOrders,
                  Mode mode = Mode::H_1);
   /// Construct a NURBSExtension by merging a partitioned NURBS mesh.

   NURBSExtension(Mesh *mesh_array[], int num_pieces);

   NURBSExtension(const Mesh *patch_topology,
                  const Array<const NURBSPatch*> &patches_);

   /// Copy assignment not supported.
   NURBSExtension& operator=(const NURBSExtension&) = delete;

   /// Generate connections between boundaries, such as periodic BCs.
   void ConnectBoundaries(Array<int> &master, Array<int> &slave);
   const Array<int> &GetMaster() const { return master; };
   Array<int> &GetMaster()  { return master; };
   const Array<int> &GetSlave() const { return slave; };
   Array<int> &GetSlave()  { return slave; };

   /** @brief Set the DOFs of @a merged to values from active elements in
       @a num_pieces of Gridfunctions @a gf_array. */
   void MergeGridFunctions(GridFunction *gf_array[], int num_pieces,
                           GridFunction &merged);

   /// Returns false if any patch has an inconsistent edge_to_ukv mapping.
   bool CheckPatches();

   /// Destroy a NURBSExtension.
   virtual ~NURBSExtension();

   // Print functions

   /** @brief Writes all patch data to the stream @a os.

       The optional input argument @a comments is a string of comments to be
       printed after the first line (containing version number) of a mesh file.
       The output is formatted for writing a mesh to file. This function is
       called by Mesh::Printer. */
   void Print(std::ostream &os, const std::string &comments = "") const;

   /// Print various mesh characteristics to the stream @a os.
   void PrintCharacteristics(std::ostream &os) const;

   /** @brief Call @a KnotVector::PrintFunctions for all KnotVectors, using a
       separate, newly created ofstream with filename "basename_i.dat" for
       KnotVector i. */
   void PrintFunctions(const char *basename, int samples=11) const;

   // Meta data functions

   /// Return the dimension of the reference space (not physical space).
   int Dimension() const { return patchTopo->Dimension(); }

   /** @brief Return the physical dimension of the NURBS geometry

       The physical dimension is inferred from the first patch,
       i.e. number of coordinates per control point minus one (for the weight).
       This method requires patch data to be present, i.e. HavePatches() == true */
   int GetPatchSpaceDimension() const;

   /// Return the number of patches.
   int GetNP()     const { return patchTopo->GetNE(); }

   /// Return the number of boundary patches.
   int GetNBP()    const { return patchTopo->GetNBE(); }

   /// Read-only access to the orders of all KnotVectors.
   const Array<int> &GetOrders() const { return mOrders; }

   /** @brief If all KnotVector orders are identical, return that number.
        Otherwise, return NURBSFECollection::VariableOrder. */
   int GetOrder() const { return mOrder; }

   /// Return the number of KnotVectors.
   int GetNKV()  const { return NumOfKnotVectors; }

   /// Return the global number of vertices.
   int GetGNV()  const { return NumOfVertices; }
   /// Return the local number of active vertices.
   int GetNV()   const { return NumOfActiveVertices; }
   /// Return the global number of elements.
   int GetGNE()  const { return NumOfElements; }
   /// Return the number of active elements.
   int GetNE()   const { return NumOfActiveElems; }
   /// Return the global number of boundary elements.
   int GetGNBE() const { return NumOfBdrElements; }
   /// Return the number of active boundary elements.
   int GetNBE()  const { return NumOfActiveBdrElems; }

   /// Return the total number of DOFs.
   int GetNTotalDof() const { return NumOfDofs; }
   /// Return the number of active DOFs.
   int GetNDof()      const { return NumOfActiveDofs; }

   /// Return the local DOF number for a given global DOF number @a glob.
   int GetActiveDof(int glob) const { return activeDof[glob]; };

   /// Return the dof index whilst accounting for periodic boundaries.
   int DofMap(int dof) const
   {
      return (d_to_d.Size() > 0) ? d_to_d[dof] : dof;
   };

   /// Return KnotVectors in @a kv in each dimension for patch @a p.
   void GetPatchKnotVectors(int p, Array<const KnotVector *> &kv) const;

   /// Return KnotVectors in @a kv in each dimension for boundary patch @a bp.
   void GetBdrPatchKnotVectors(int bp, Array<const KnotVector *> &kv) const;

   /// KnotVector read-only access function.
   const KnotVector *GetKnotVector(int i) const { return knotVectors[i]; }

   // Mesh generation functions

   /// Generate the active mesh elements and return them in @a elements.
   void GetElementTopo   (Array<Element *> &elements) const;
   /// Generate the active mesh boundary elements and return them in @a boundary.
   void GetBdrElementTopo(Array<Element *> &boundary) const;

   /// Return true if at least 1 patch is defined, false otherwise.
   bool HavePatches() const { return (patches.Size() != 0); }

   /// Access function for the element DOF table @a el_dof.
   /// @note The returned object should NOT be deleted by the caller.
   Table *GetElementDofTable() { return el_dof; }

   /// Access function for the boundary element DOF table @a bel_dof.
   /// @note The returned object should NOT be deleted by the caller.
   Table *GetBdrElementDofTable() { return bel_dof; }

   /// Get the local to global vertex index map @a lvert_vert.
   void GetVertexLocalToGlobal(Array<int> &lvert_vert);
   /// Get the local to global element index map @a lelem_elem.
   void GetElementLocalToGlobal(Array<int> &lelem_elem);

   /** @brief Set the attribute for patch @a i, which is set to all elements in
       the patch. */
   void SetPatchAttribute(int i, int attr) { patchTopo->SetAttribute(i, attr); }

   /** @brief Get the attribute for patch @a i, which is set to all elements in
       the patch. */
   int GetPatchAttribute(int i) const { return patchTopo->GetAttribute(i); }

   /** @brief Set the attribute for patch boundary element @a i to @a attr, which
       is set to all boundary elements in the patch. */
   void SetPatchBdrAttribute(int i, int attr)
   { patchTopo->SetBdrAttribute(i, attr); }

   /** @brief Get the attribute for boundary patch element @a i, which is set to
       all boundary elements in the patch. */
   int GetPatchBdrAttribute(int i) const
   { return patchTopo->GetBdrAttribute(i); }

   /// Return the number of knotvector elements for edge @a edge.
   inline int KnotVecNE(int edge) const;

   // Load functions

   /// Load element @a i into @a FE.
   void LoadFE(int i, const FiniteElement *FE) const;
   /// Load boundary element @a i into @a BE.
   void LoadBE(int i, const FiniteElement *BE) const;

   /// Access function to the vector of weights @a weights.
   const Vector &GetWeights() const { return  weights; }
   Vector       &GetWeights()       { return  weights; }

   // Translation functions between FE coordinates and IJK patch format.

   /// Define patches in IKJ (B-net) format, using FE coordinates in @a Nodes.
   void ConvertToPatches(const Vector &Nodes);
   /// Set KnotVectors from @a patches and construct mesh and space data.
   void SetKnotsFromPatches();
   /** @brief Set FE coordinates in @a Nodes, using data from @a patches,
       with physical vector dimension @a vdim, and erase @a patches. */
   void SetCoordsFromPatches(Vector &Nodes, int vdim);

   /** @brief Read a GridFunction @a sol from stream @a input, written
       patch-by-patch, e.g. with PrintSolution(). */
   void LoadSolution(std::istream &input, GridFunction &sol) const;
   /// Write a GridFunction @a sol patch-by-patch to stream @a os.
   void PrintSolution(const GridFunction &sol, std::ostream &os) const;

   // Refinement methods

   /** @brief Call @a DegreeElevate for all KnotVectors of all patches. For each
       KnotVector, the new degree is
       max(old_degree, min(old_degree + rel_degree, degree)). */
   void DegreeElevate(int rel_degree, int degree = 16);

   /** @brief Refine with optional refinement factor @a rf. Uniform means
       refinement is done everywhere by the same factor, although nonuniform
       spacing functions may be used. */
   void UniformRefinement(int rf = 2);
   virtual void UniformRefinement(const Array<int> &rf);

   /// Refine with refinement factors loaded for some knotvectors specified in
   /// the given file, with default refinement factor @a rf elsewhere. The flag
   /// @a coarsened indicates whether each patch is a single element.
   virtual void RefineWithKVFactors(int rf, const std::string &kvf_filename,
                                    bool coarsened);

   /// Coarsen with optional coarsening factor @a cf.
   void Coarsen(int cf = 2, real_t tol = 1.0e-12);
   void Coarsen(Array<int> const& cf, real_t tol = 1.0e-12);

   /** @brief Insert knots from @a kv into all KnotVectors in all patches. The
       size of @a kv should be the same as @a knotVectors. */
   void KnotInsert(Array<KnotVector *> &kv);
   void KnotInsert(Array<Vector *> &kv);

   /** Returns the NURBSExtension to be used for @a component of
       an H(div) conforming NURBS space. Caller gets ownership of
       the returned object, and is responsible for deletion.*/
   NURBSExtension* GetDivExtension(int component);

   /** Returns the NURBSExtension to be used for @a component of
       an H(curl) conforming NURBS space. Caller gets ownership of
       the returned object, and is responsible for deletion.*/
   NURBSExtension* GetCurlExtension(int component);

   void KnotRemove(Array<Vector *> &kv, real_t tol = 1.0e-12);

   /** Calls GetCoarseningFactors for each patch and finds the minimum factor
       for each direction that ensures refinement will work in the case of
       non-nested spacing functions. */
   void GetCoarseningFactors(Array<int> &f) const;

   /// Returns the index of the patch containing element @a elem.
   int GetElementPatch(int elem) const { return el_to_patch[elem]; }

   /** @brief Return Cartesian indices (i,j) in 2D or (i,j,k) in 3D of element
       @a elem, in the knot-span tensor product ordering for its patch. */
   void GetElementIJK(int elem, Array<int> & ijk);

   /** @brief Return the degrees of freedom in @a dofs on patch @a patch, in
       Cartesian order. */
   void GetPatchDofs(const int patch, Array<int> &dofs);

   /// Returns a deep copy of the patch topology mesh
   Mesh GetPatchTopology() const { return Mesh(*patchTopo); }

   /** Returns a deep copy of all instantiated patches. To ensure that patches
       are instantiated, use Mesh::GetNURBSPatches() instead. Caller gets
       ownership of the returned object, and is responsible for deletion.*/
   void GetPatches(Array<NURBSPatch*> &patches);

   /// Return the array of indices of all elements in patch @a patch.
   const Array<int>& GetPatchElements(int patch);
   /// Return the array of indices of all boundary elements in patch @a patch.
   const Array<int>& GetPatchBdrElements(int patch);

   /// Return true if the patch topology mesh is nonconforming.
   bool NonconformingPatches() const { return nonconformingPT; }

   /// Return a pointer to the NCMesh of a nonconforming patch topology mesh.
   NCMesh *GetNCMesh() const { return patchTopo->ncmesh; }

   /// Read the control points for coarse patches.
   virtual void ReadCoarsePatchCP(std::istream &input);

   /** @brief Fully coarsen all structured patches, for non-nested refinement of
       a mesh with a nonconforming patch topology. */
   void FullyCoarsen();

   /// Print control points for coarse patches.
   virtual void PrintCoarsePatches(std::ostream &os);
};


#ifdef MFEM_USE_MPI
/** @brief Parallel version of NURBSExtension. */
class ParNURBSExtension : public NURBSExtension
{
private:
   /// Partitioning of the global elements by MPI rank
   mfem::Array<int> partitioning;

   /// Construct and return a table of DOFs for each global element.
   Table *GetGlobalElementDofTable();
   Table *Get1DGlobalElementDofTable();
   Table *Get2DGlobalElementDofTable();
   Table *Get3DGlobalElementDofTable();

   /** @brief Set active global elements and boundary elements based on MPI
       ranks in @a partition and the array @a active_bel. */
   void SetActive(const int *partitioning_, const Array<bool> &active_bel);

   /// Set up GroupTopology @a gtopo for MPI communication.
   void BuildGroups(const int *partitioning_, const Table &elem_dof);

public:
   GroupTopology gtopo;

   Array<int> ldof_group;

   /// Copy constructor
   ParNURBSExtension(const ParNURBSExtension &orig);

   /** @brief Constructor for an MPI communicator @a comm, a global
       NURBSExtension @a parent, a partitioning @a partitioning_ of the global
       elements by MPI rank, and a marker @a active_bel of active global
       boundary elements on this rank. The partitioning is deep-copied and will
       not be deleted by this object. */
   ParNURBSExtension(MPI_Comm comm, NURBSExtension *parent,
                     const int *partitioning_,
                     const Array<bool> &active_bel);

   /** @brief Create a parallel version of @a parent with partitioning as in
       @a par_parent; the @a parent object is destroyed.
       The @a parent can be either a local NURBSExtension or a global one. */
   ParNURBSExtension(NURBSExtension *parent,
                     const ParNURBSExtension *par_parent);
};
#endif


/** @brief Mapping for mesh vertices and NURBS space DOFs on a patch.

    This class has two modes, for vertices or DOFs, depending on whether
    @a SetPatchVertexMap or @a SetPatchDofMap is called.
 */
class NURBSPatchMap
{
private:
   /// This object must be associated with exactly one NURBSExtension.
   const NURBSExtension *Ext;

   /// Vertex mode: Number of elements in each direction, minus 1.
   /// DOF mode: Number of control points in each direction, minus 2.
   int I, J, K;

   /// Vertex/DOF offset for this patch, among all patches.
   int pOffset;
   /// Orientation for this boundary patch (0 in the patch case).
   int opatch;

   /// Patch topology entities for this patch or boundary patch.
   Array<int> verts, edges, faces, oedge, oface;
   Array<bool> edgeMaster, faceMaster;
   Array<int> edgeMasterOffset, faceMasterOffset;
   Array<int> masterDofs;

   inline static int F(const int n, const int N)
   { return (n < 0) ? 0 : ((n >= N) ? 2 : 1); }

   inline static int Or1D(const int n, const int N, const int Or)
   { return (Or > 0) ? n : (N - 1 - n); }

   inline static int Or2D(const int n1, const int n2,
                          const int N1, const int N2, const int Or);

   inline int EC(const int e, const int n, const int N, const int s=1) const
   {
      return !edgeMaster[e] ? edges[e] + Or1D(n, N, s*oedge[e]) :
             GetMasterEdgeDof(e, Or1D(n, N, s*oedge[e]));
   }

   inline int FC(const int f, const int m, const int n,
                 const int M, const int N) const
   {
      return !faceMaster[f] ? faces[f] + Or2D(m, n, M, N, oface[f]) :
             GetMasterFaceDof(f, Or2D(m, n, M, N, oface[f]));
   }

   inline int FCP(const int f, const int m, const int n,
                  const int M, const int N) const
   {
      return (faceMaster.Size() == 0 || !faceMaster[f]) ?
             pOffset + Or2D(m, n, M, N, opatch) :
             GetMasterFaceDof(f, Or2D(m, n, M, N, opatch));
   }

   // The following 2 functions also set verts, edges, faces, orientations etc.

   /// Get the KnotVectors for patch @a p in @a kv.
   void GetPatchKnotVectors   (int p, const KnotVector *kv[]);
   /** @brief Get the KnotVectors for boundary patch @a bp in @a kv, with
       orientations output in @a okv. */
   void GetBdrPatchKnotVectors(int bp, const KnotVector *kv[], int *okv);

   void SetMasterEdges(bool dof, const KnotVector *kv[] = nullptr);
   void SetMasterFaces(bool dof);
   int GetMasterEdgeDof(const int e, const int i) const;
   int GetMasterFaceDof(const int f, const int i) const;

public:
   /// Constructor for an object associated with NURBSExtension @a ext.
   NURBSPatchMap(const NURBSExtension *ext) { Ext = ext; }

   /// Vertex mode: Return the number of elements in the first direction.
   /// DOF mode: Return the number of control points - 1 in the first direction.
   inline int nx() const { return I + 1; }

   /// Vertex mode: Return the number of elements in the second direction (2D or 3D).
   /// DOF mode: Return the number of control points - 1 in the second direction (2D or 3D).
   inline int ny() const { return J + 1; }

   /// Vertex mode: Return the number of elements in the third direction (3D).
   /// DOF mode: Return the number of control points - 1 in the third direction (3D).
   inline int nz() const { return K + 1; }

   /// Set mesh vertex map for patch @a p with KnotVectors @a kv.
   void SetPatchVertexMap(int p, const KnotVector *kv[]);
   /// Set NURBS space DOF map for patch @a p with KnotVectors @a kv.
   void SetPatchDofMap   (int p, const KnotVector *kv[]);

   /// Set mesh vertex map for boundary patch @a bp with KnotVectors @a kv.
   void SetBdrPatchVertexMap(int bp, const KnotVector *kv[], int *okv);
   /// Set NURBS space DOF map for boundary patch @a bp with KnotVectors @a kv.
   void SetBdrPatchDofMap   (int bp, const KnotVector *kv[], int *okv);

   /// For 1D, return the vertex or DOF at index @a i.
   inline int operator()(const int i) const;
   inline int operator[](const int i) const { return (*this)(i); }

   /// For 2D, return the vertex or DOF at indices @a i, @a j.
   inline int operator()(const int i, const int j) const;

   /// For 3D, return the vertex or DOF at indices @a i, @a j, @a k.
   inline int operator()(const int i, const int j, const int k) const;
};


// Inline function implementations

inline real_t &NURBSPatch::slice(int i, int j)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= nd || j < 0 || j > ls)
   {
      mfem_error("NURBSPatch::slice()");
   }
#endif
   return data[j%sd + sd*(i + (j/sd)*nd)];
}

inline const real_t &NURBSPatch::slice(int i, int j) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= nd || j < 0 || j > ls)
   {
      mfem_error("NURBSPatch::slice()");
   }
#endif
   return data[j%sd + sd*(i + (j/sd)*nd)];
}


inline real_t &NURBSPatch::operator()(int i, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || nj > 0 || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 1D");
   }
#endif

   return data[i*Dim+l];
}

inline const real_t &NURBSPatch::operator()(int i, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni ||  nj > 0 || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 1D");
   }
#endif

   return data[i*Dim+l];
}

inline real_t &NURBSPatch::operator()(int i, int j, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 2D");
   }
#endif

   return data[(i+j*ni)*Dim+l];
}

inline const real_t &NURBSPatch::operator()(int i, int j, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 2D");
   }
#endif

   return data[(i+j*ni)*Dim+l];
}

inline real_t &NURBSPatch::operator()(int i, int j, int k, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || k < 0 ||
       k >= nk || l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 3D");
   }
#endif

   return data[(i+(j+k*nj)*ni)*Dim+l];
}

inline const real_t &NURBSPatch::operator()(int i, int j, int k, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || k < 0 ||
       k >= nk ||  l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 3D");
   }
#endif

   return data[(i+(j+k*nj)*ni)*Dim+l];
}

inline int NURBSExtension::KnotInd(int edge) const
{
   const int kv = edge_to_ukv[edge];
   return kv >= 0 ? kv : -1 - kv;
}

inline int NURBSExtension::KnotSign(int edge) const
{
   return edge_to_ukv[edge] >= 0 ? 1 : -1;
}

inline KnotVector *NURBSExtension::KnotVec(int edge)
{
   return knotVectors[KnotInd(edge)];
}

inline const KnotVector *NURBSExtension::KnotVec(int edge) const
{
   return knotVectors[KnotInd(edge)];
}

inline const KnotVector *NURBSExtension::KnotVec(int edge, int oedge, int *okv)
const
{
   int kv = edge_to_ukv[edge];
   if (kv >= 0)
   {
      *okv = oedge;
      return knotVectors[kv];
   }
   else
   {
      *okv = -oedge;
      return knotVectors[-1-kv];
   }
}

inline int NURBSExtension::KnotVecNE(int edge) const
{
   return knotVectors[KnotInd(edge)]->GetNE();
}

// static method
inline int NURBSPatchMap::Or2D(const int n1, const int n2,
                               const int N1, const int N2, const int Or)
{
   switch (Or)
   {
      case 0: return n1 + n2*N1;
      case 1: return n2 + n1*N2;
      case 2: return n2 + (N1 - 1 - n1)*N2;
      case 3: return (N1 - 1 - n1) + n2*N1;
      case 4: return (N1 - 1 - n1) + (N2 - 1 - n2)*N1;
      case 5: return (N2 - 1 - n2) + (N1 - 1 - n1)*N2;
      case 6: return (N2 - 1 - n2) + n1*N2;
      case 7: return n1 + (N2 - 1 - n2)*N1;
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::Or2D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i) const
{
   const int i1 = i - 1;
   switch (F(i1, I))
   {
      case 0: return verts[0];
      case 1: return edgeMaster.Size() > 0 && edgeMaster[0] ?
                        GetMasterEdgeDof(0, Or1D(i1, I, opatch)) :
                        pOffset + Or1D(i1, I, opatch);
      case 2: return verts[1];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 1D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i, const int j) const
{
   const int i1 = i - 1, j1 = j - 1;
   switch (3*F(j1, J) + F(i1, I))
   {
      case 0: return verts[0];
      case 1: return EC(0, i1, I);
      case 2: return verts[1];
      case 3: return EC(3, j1, J, -1);
      case 4: return FCP(0, i1, j1, I, J);
      case 5: return EC(1, j1, J);
      case 6: return verts[3];
      case 7: return EC(2, i1, I, -1);
      case 8: return verts[2];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 2D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i, const int j, const int k)
const
{
   const int i1 = i - 1, j1 = j - 1, k1 = k - 1;
   switch (3*(3*F(k1, K) + F(j1, J)) + F(i1, I))
   {
      case  0: return verts[0];
      case  1: return EC(0, i1, I);
      case  2: return verts[1];
      case  3: return EC(3, j1, J);
      case  4: return FC(0, i1, J - 1 - j1, I, J);
      case  5: return EC(1, j1, J);
      case  6: return verts[3];
      case  7: return EC(2, i1, I);
      case  8: return verts[2];
      case  9: return EC(8, k1, K);
      case 10: return FC(1, i1, k1, I, K);
      case 11: return EC(9, k1, K);
      case 12: return FC(4, J - 1 - j1, k1, J, K);
      case 13: return pOffset + I*(J*k1 + j1) + i1;
      case 14: return FC(2, j1, k1, J, K);
      case 15: return EC(11, k1, K);
      case 16: return FC(3, I - 1 - i1, k1, I, K);
      case 17: return EC(10, k1, K);
      case 18: return verts[4];
      case 19: return EC(4, i1, I);
      case 20: return verts[5];
      case 21: return EC(7, j1, J);
      case 22: return FC(5, i1, j1, I, J);
      case 23: return EC(5, j1, J);
      case 24: return verts[7];
      case 25: return EC(6, i1, I);
      case 26: return verts[6];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 3D");
#endif
   return -1;
}

}

#endif
