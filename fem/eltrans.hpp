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

#ifndef MFEM_ELEMENTTRANSFORM
#define MFEM_ELEMENTTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "fe.hpp"

#include "kernel_dispatch.hpp"

namespace mfem
{

class GridFunction;

class ElementTransformation
{
protected:
   const IntegrationPoint *IntPoint;
   DenseMatrix dFdx, adjJ, invJ;
   DenseMatrix d2Fdx2, adjJT;
   real_t Wght;
   int EvalState;
   enum StateMasks
   {
      JACOBIAN_MASK = 1,
      WEIGHT_MASK   = 2,
      ADJUGATE_MASK = 4,
      INVERSE_MASK  = 8,
      HESSIAN_MASK  = 16,
      TRANS_ADJUGATE_MASK = 32
   };
   Geometry::Type geom;

   /** @brief Evaluate the Jacobian of the transformation at the IntPoint and
       store it in dFdx. */
   virtual const DenseMatrix &EvalJacobian() = 0;

   /** @brief Evaluate the Hessian of the transformation at the IntPoint and
       store it in d2Fdx2. */
   virtual const DenseMatrix &EvalHessian() = 0;

   real_t EvalWeight();
   const DenseMatrix &EvalAdjugateJ();
   const DenseMatrix &EvalTransAdjugateJ();
   const DenseMatrix &EvalInverseJ();

   /// @name Tolerance used for point comparisons
   ///@{
#ifdef MFEM_USE_DOUBLE
   static constexpr real_t tol_0 = 1e-15;
#elif defined(MFEM_USE_SINGLE)
   static constexpr real_t tol_0 = 1e-7;
#endif
   ///@}

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

   /// The Mesh object containing the element.
   /** If the element transformation belongs to a mesh, this will point to the
       containing Mesh object. ElementNo will be the number of the element in
       this Mesh. This will be NULL if the element does not belong to a mesh. */
   const Mesh *mesh;

   ElementTransformation();

   /** @brief Force the reevaluation of the Jacobian in the next call. */
   void Reset() { EvalState = 0; }

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
       The Weight evaluates to $ \sqrt{\lvert J^T J \rvert} $. */
   real_t Weight() { return (EvalState & WEIGHT_MASK) ? Wght : EvalWeight(); }

   /** @brief Return the adjugate of the Jacobian matrix of the transformation
        at the currently set IntegrationPoint. */
   const DenseMatrix &AdjugateJacobian()
   { return (EvalState & ADJUGATE_MASK) ? adjJ : EvalAdjugateJ(); }

   /** @brief Return the transpose of the adjugate of the Jacobian matrix of
        the transformation at the currently set IntegrationPoint. */
   const DenseMatrix &TransposeAdjugateJacobian()
   { return (EvalState & TRANS_ADJUGATE_MASK) ? adjJT : EvalTransAdjugateJ(); }

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

   /// Return the order of $ adj(J)^T \nabla fi $
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
       reference space and optionally can set a solver tolerance using @a phys_tol. */
   /** Attempt to find the IntegrationPoint that is transformed into the given
       point in physical space. If the inversion fails a non-zero value is
       returned. This method is not 100 percent reliable for non-linear
       transformations. */
   virtual int TransformBack(const Vector &pt, IntegrationPoint &ip,
                             const real_t phys_tol = tol_0) = 0;

   virtual ~ElementTransformation() { }
};


/// The inverse transformation of a given ElementTransformation.
class InverseElementTransformation
{
public:
   /// Algorithms for selecting an initial guess.
   enum InitGuessType
   {
      Center = 0,          ///< Use the center of the reference element.
      ClosestPhysNode = 1, /**<
       Use the point returned by FindClosestPhysPoint() from a reference-space
       grid of type and size controlled by SetInitGuessPointsType() and
       SetInitGuessRelOrder(), respectively. */
      ClosestRefNode = 2,  /**<
        Use the point returned by FindClosestRefPoint() from a reference-space
        grid of type and size controlled by SetInitGuessPointsType() and
        SetInitGuessRelOrder(), respectively. */
      GivenPoint = 3,      ///< Use a specific point, set with SetInitialGuess().
      EdgeScan =
         4, /**< Performs full solves on multiple points along the r/s/t=0 edges
              of the element. It is recommended that SetInitGuessRelOrder() is
              chosen such that max(trans_order+order,0)+1 <= 4 with
              SetInitGuessPointsType() as Quadrature1D::ClosedUniform. @see
              GeometryRefiner::EdgeScan */
   };

   /// Solution strategy.
   enum SolverType
   {
      Newton = 0,               /**<
                     Use Newton's algorithm, without restricting the reference-space points
                     (iterates) to the reference element. */
      NewtonSegmentProject = 1, /**<
       Use Newton's algorithm, restricting the reference-space points to the
       reference element by scaling back the Newton increments, i.e.
       projecting new iterates, x_new, lying outside the element, to the
       intersection of the line segment [x_old, x_new] with the boundary. */
      NewtonElementProject = 2, /**<
       Use Newton's algorithm, restricting the reference-space points to the
       reference element by projecting new iterates, x_new, lying outside the
       element, to the point on the boundary closest (in reference-space) to
       x_new. */
   };

   /// Values returned by Transform().
   enum TransformResult
   {
      Inside = 0,  ///< The point is inside the element
      Outside = 1, ///< The point is _probably_ outside the element
      Unknown = 2  ///< The algorithm failed to determine where the point is
   };

protected:
   // Pointer to the forward transformation. Not owned.
   ElementTransformation *T;

   // Parameters of the inversion algorithms:
   IntegrationPoint ip0;
   int init_guess_type; // algorithm to use
   GeometryRefiner refiner; // geometry refiner for initial guess
   int qpts_order;          // num_1D_qpts = rel_qpts_order + 1, or < 0 to use
   // rel_qpts_order.
   int rel_qpts_order; // num_1D_qpts = max(trans_order+rel_qpts_order,0)+1
   int solver_type; // solution strategy to use
   int max_iter; // max. number of Newton iterations
   real_t ref_tol; // reference space tolerance
   real_t phys_rtol; // physical space tolerance (relative)
   real_t ip_tol; // tolerance for checking if a point is inside the ref. elem.
   int print_level;

   void NewtonPrint(int mode, real_t val);
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
        init_guess_type(Center),
        refiner(Quadrature1D::OpenHalfUniform),
        qpts_order(-1),
        rel_qpts_order(-1),
        solver_type(NewtonElementProject),
        max_iter(16),
#ifdef MFEM_USE_DOUBLE
        ref_tol(1e-15),
        phys_rtol(4e-15),
        ip_tol(1e-8),
#elif defined(MFEM_USE_SINGLE)
        ref_tol(4e-7),
        phys_rtol(1e-6),
        ip_tol(1e-4),
#endif
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
   { ip0 = init_ip; SetInitialGuessType(GivenPoint); }

   /// Set the Quadrature1D type used for the `Closest*` and `EdgeScan` initial
   /// guess types.
   void SetInitGuessPointsType(int q_type) { refiner.SetType(q_type); }

   /// Set the relative order used for the `Closest*` initial guess types.
   /** The number of points in each spatial direction is given by the formula
       max(trans_order+order,0)+1, where trans_order is the order of the current
       ElementTransformation. */
   void SetInitGuessRelOrder(int order)
   {
      qpts_order = -1;
      rel_qpts_order = order;
   }

   /** The number of points in each spatial direction is given by the formula
       order+1. */
   void SetInitGuessOrder(int order)
   {
      qpts_order = order;
   }

   /** @brief Specify which algorithm to use for solving the transformation
       equation, i.e. when calling the Transform() method. */
   void SetSolverType(SolverType stype) { solver_type = stype; }

   /// Set the maximum number of iterations when solving for a reference point.
   void SetMaxIter(int max_it) { max_iter = max_it; }

   /// Set the reference-space convergence tolerance.
   void SetReferenceTol(real_t ref_sp_tol) { ref_tol = ref_sp_tol; }

   /// Set the relative physical-space convergence tolerance.
   void SetPhysicalRelTol(real_t phys_rel_tol) { phys_rtol = phys_rel_tol; }

   /** @brief Set the tolerance used to determine if a point lies inside or
       outside of the reference element. */
   /** This tolerance is used only with the pure #Newton solver. */
   void SetElementTol(real_t el_tol) { ip_tol = el_tol; }

   /// Set the desired print level, useful for debugging.
   /** The valid options are: -1 - never print (default); 0 - print only errors;
       1 - print the first and last iterations; 2 - print every iteration;
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

/**
 * @brief Performs batch inverse element transforms. Currently only supports
 * non-mixed meshes with SEGMENT, SQUARE, or CUBE geometries. Mixed
 * element order meshes are projected onto an equivalent uniform order mesh.
 */
class BatchInverseElementTransformation
{
   // nodes grid function, not owned
   const GridFunction *gf_ = nullptr;
   // initial guess algorithm to use
   InverseElementTransformation::InitGuessType init_guess_type =
      InverseElementTransformation::ClosestPhysNode;
   int qpts_order = -1; // num_1D_qpts = rel_qpts_order + 1, or < 0 to use
   // rel_qpts_order.
   // num_1D_qpts = max(trans_order+rel_qpts_order,0)+1
   int rel_qpts_order = 0;
   // solution strategy to use
   InverseElementTransformation::SolverType solver_type =
      InverseElementTransformation::NewtonElementProject;
   // basis type stored in points1d
   int basis_type = BasisType::Invalid;
   // initial guess points type. Quadrature1D::Invalid is used for match
   // basis_type.
   int guess_points_type = Quadrature1D::Invalid;
   // max. number of Newton iterations
   int max_iter = 16;
   // internal element node locations cache
   Vector node_pos;
#ifdef MFEM_USE_DOUBLE
   // reference space tolerance
   real_t ref_tol = 1e-15;
   // physical space tolerance (relative)
   real_t phys_rtol = 4e-15;
#else
   // reference space tolerance
   real_t ref_tol = 4e-7;
   // physical space tolerance (relative)
   real_t phys_rtol = 1e-6;
#endif
   // not owned, location of tensor product basis nodes in reference space
   const Array<real_t> *points1d = nullptr;

public:
   /// Uninitialized BatchInverseElementTransformation. Users must call
   /// UpdateNodes before Transform.
   BatchInverseElementTransformation();
   ///
   /// Constructs a BatchInverseElementTransformation given @a nodes representing
   /// the mesh nodes.
   ///
   BatchInverseElementTransformation(const GridFunction &nodes,
                                     MemoryType d_mt = MemoryType::DEFAULT);
   ///
   /// Constructs a BatchInverseElementTransformation for a given @a mesh.
   /// mesh.GetNodes() must not be null.
   ///
   BatchInverseElementTransformation(const Mesh &mesh,
                                     MemoryType d_mt = MemoryType::DEFAULT);

   ~BatchInverseElementTransformation();

   /** @brief Choose how the initial guesses for subsequent calls to Transform()
        will be selected. ClosestRefNode is currently not supported. */
   void SetInitialGuessType(InverseElementTransformation::InitGuessType itype)
   {
      MFEM_ASSERT(itype != InverseElementTransformation::ClosestRefNode,
                  "ClosestRefNode is currently not supported");
      init_guess_type = itype;
   }

   /// Set the Quadrature1D type used for the `Closest*` and `EdgeScan` initial
   /// guess types.
   void SetInitGuessPointsType(int q_type) { guess_points_type = q_type; }

   /// Set the relative order used for the `Closest*` initial guess types.
   /** The number of points in each spatial direction is given by the formula
        max(trans_order+order,0)+1, where trans_order is the order of the
      current ElementTransformation. */
   void SetInitGuessRelOrder(int order)
   {
      qpts_order = -1;
      rel_qpts_order = order;
   }

   /** The number of points in each spatial direction is given by the formula
       order+1. */
   void SetInitGuessOrder(int order) { qpts_order = order; }

   /// @b Gets the basis type nodes are projected onto, or BasisType::Invalid if
   /// uninitialized.
   int GetBasisType() const { return basis_type; }

   /** @brief Specify which algorithm to use for solving the transformation
       equation, i.e. when calling the Transform() method. NewtonSegmentProject
       is currently not supported. */
   void SetSolverType(InverseElementTransformation::SolverType stype)
   {
      MFEM_ASSERT(stype != InverseElementTransformation::NewtonSegmentProject,
                  "NewtonSegmentProject is currently not supported");
      solver_type = stype;
   }

   /// Set the maximum number of iterations when solving for a reference point.
   void SetMaxIter(int max_it) { max_iter = max_it; }

   /// Set the reference-space convergence tolerance.
   void SetReferenceTol(real_t ref_sp_tol) { ref_tol = ref_sp_tol; }

   /// Set the relative physical-space convergence tolerance.
   void SetPhysicalRelTol(real_t phys_rel_tol) { phys_rtol = phys_rel_tol; }

   /**
    * @brief Updates internal datastructures if @a nodes change. Some version
    * of UpdateNodes must be called at least once before calls to Transform if
    * nodes have changed.
    */
   void UpdateNodes(const GridFunction &nodes,
                    MemoryType d_mt = MemoryType::DEFAULT);
   /**
    * @brief Updates internal datastructures if @a mesh nodes change. Some version
    * of UpdateNodes must be called at least once before calls to Transform if
    * mesh nodes have changed. mesh.GetNodes() must not be null.
    */
   void UpdateNodes(const Mesh &mesh, MemoryType d_mt = MemoryType::DEFAULT);

   /** @brief Performs a batch request of a set of points belonging to the given
       elements.
        @a pts list of physical point coordinates ordered by
           Ordering::Type::byNODES.
        @a elems which element index to search for each corresponding point in
      @a pts
        @a types output search classification (@see
      InverseElementTransformation::TransformResult).
        @a refs result reference point coordinates ordered by
       Ordering::Type::byNODES. If using InitGuessType::GivenPoint, this should
       contain the initial guess for each point.
        @a use_device hint for if device acceleration should be used.
       Device acceleration is currently only implemented for meshes containing
       only a single tensor product basis element type.
        @a iters optional array storing how many iterations was spent on each
      tested point
     */
   void Transform(const Vector &pts, const Array<int> &elems, Array<int> &types,
                  Vector &refs, bool use_device = true,
                  Array<int> *iters = nullptr) const;

   using ClosestPhysPointKernelType = void (*)(int, int, int, int,
                                               const real_t *, const real_t *,
                                               const int *, const real_t *,
                                               const real_t *, real_t *);

   // specialization params: Geom, SDim, use_device
   MFEM_REGISTER_KERNELS(FindClosestPhysPoint, ClosestPhysPointKernelType,
                         (int, int, bool));

   using ClosestPhysDofKernelType = void (*)(int, int, int,
                                             const real_t *, const real_t *,
                                             const int *, const real_t *,
                                             real_t *);

   // specialization params: Geom, SDim, use_device
   MFEM_REGISTER_KERNELS(FindClosestPhysDof, ClosestPhysDofKernelType,
                         (int, int, bool));

   using ClosestRefDofKernelType = void (*)(int, int, int, const real_t *,
                                            const real_t *, const int *,
                                            const real_t *, real_t *);

   // specialization params: Geom, SDim, use_device
   MFEM_REGISTER_KERNELS(FindClosestRefDof, ClosestRefDofKernelType,
                         (int, int, bool));

   using ClosestRefPointKernelType = void (*)(int, int, int, int, const real_t *,
                                              const real_t *, const int *,
                                              const real_t *, const real_t *,
                                              real_t *);

   // specialization params: Geom, SDim, use_device
   MFEM_REGISTER_KERNELS(FindClosestRefPoint, ClosestRefPointKernelType,
                         (int, int, bool));

   using NewtonKernelType = void (*)(real_t, real_t, int, int, int, int,
                                     const real_t *, const real_t *,
                                     const int *, const real_t *, int *, int*,
                                     real_t *);

   // specialization params: Geom, SDim, SolverType, use_device
   MFEM_REGISTER_KERNELS(NewtonSolve, NewtonKernelType,
                         (int, int, InverseElementTransformation::SolverType,
                          bool));

   using NewtonEdgeScanKernelType = void (*)(real_t, real_t, int, int, int, int,
                                             const real_t *, const real_t *,
                                             const int *, const real_t *,
                                             const real_t *, int, int *, int *,
                                             real_t *);

   // specialization params: Geom, SDim, SolverType, use_device
   MFEM_REGISTER_KERNELS(NewtonEdgeScan, NewtonEdgeScanKernelType,
                         (int, int, InverseElementTransformation::SolverType,
                          bool));

   struct Kernels { Kernels(); };

   template <int Dim, int SDim>
   static void AddFindClosestSpecialization()
   {
      FindClosestPhysPoint::Specialization<Dim, SDim, true>::Add();
      FindClosestRefPoint::Specialization<Dim, SDim, true>::Add();
      FindClosestPhysPoint::Specialization<Dim, SDim, false>::Add();
      FindClosestRefPoint::Specialization<Dim, SDim, false>::Add();
      FindClosestPhysDof::Specialization<Dim, SDim, true>::Add();
      FindClosestRefDof::Specialization<Dim, SDim, true>::Add();
      FindClosestPhysDof::Specialization<Dim, SDim, false>::Add();
      FindClosestRefDof::Specialization<Dim, SDim, false>::Add();
   }

   template <int Dim, int SDim, InverseElementTransformation::SolverType SType>
   static void AddNewtonSolveSpecialization()
   {
      NewtonSolve::Specialization<Dim, SDim, SType, true>::Add();
      NewtonEdgeScan::Specialization<Dim, SDim, SType, true>::Add();
      NewtonSolve::Specialization<Dim, SDim, SType, false>::Add();
      NewtonEdgeScan::Specialization<Dim, SDim, SType, false>::Add();
   }
};

/// A standard isoparametric element transformation
class IsoparametricTransformation : public ElementTransformation
{
private:
   DenseMatrix dshape, d2shape;
   Vector shape;

   const FiniteElement *FElem;
   DenseMatrix PointMat; // dim x dof

   /** @brief Evaluate the Jacobian of the transformation at the IntPoint and
       store it in dFdx. */
   const DenseMatrix &EvalJacobian() override;
   // Evaluate the Hessian of the transformation at the IntPoint and store it
   // in d2Fdx2.
   const DenseMatrix &EvalHessian() override;

public:
   IsoparametricTransformation() : FElem(NULL) {}

   /// Set the element that will be used to compute the transformations
   void SetFE(const FiniteElement *FE)
   {
      MFEM_ASSERT(FE != NULL, "Must provide a valid FiniteElement object!");
      EvalState = (FE != FElem) ? 0 : EvalState;
      FElem = FE; geom = FE->GetGeomType();
   }

   /// Get the current element used to compute the transformations
   const FiniteElement* GetFE() const { return FElem; }

   /// @brief Set the underlying point matrix describing the transformation.
   /** The dimensions of the matrix are space-dim x dof. The transformation is
       defined as
           $ x = F( \hat x ) = P \phi( \hat x ) $

       where $ \hat x $  is the reference point, @a x is the corresponding
       physical point, @a P is the point matrix, and $ \phi( \hat x ) $ is
       the column-vector of all basis functions evaluated at $ \hat x $ .
       The columns of @a P represent the control points in physical space
       defining the transformation. */
   void SetPointMat(const DenseMatrix &pm) { PointMat = pm; EvalState = 0; }

   /// Return the stored point matrix.
   const DenseMatrix &GetPointMat() const { return PointMat; }

   /// @brief Write access to the stored point matrix. Use with caution.
   /** If the point matrix is altered using this member function the Reset
       function should also be called to force the reevaluation of the
       Jacobian, etc.. */
   DenseMatrix &GetPointMat() { return PointMat; }

   /// Set the FiniteElement Geometry for the reference elements being used.
   void SetIdentityTransformation(Geometry::Type GeomType);

   /** @brief Transform integration point from reference coordinates to
       physical coordinates and store them in the vector. */
   void Transform(const IntegrationPoint &, Vector &) override;

   /** @brief Transform all the integration points from the integration rule
       from reference coordinates to physical
      coordinates and store them as column vectors in the matrix. */
   void Transform(const IntegrationRule &, DenseMatrix &) override;

   /** @brief Transform all the integration points from the column vectors
       of @a matrix from reference coordinates to physical
       coordinates and store them as column vectors in @a result. */
   void Transform(const DenseMatrix &matrix, DenseMatrix &result) override;

   /// Return the order of the current element we are using for the transformation.
   int Order() const override { return FElem->GetOrder(); }

   /// Return the order of the elements of the Jacobian of the transformation.
   int OrderJ() const override;

   /** @brief Return the order of the determinant of the Jacobian (weight)
       of the transformation. */
   int OrderW() const override;

   /// Return the order of $ adj(J)^T \nabla fi $
   int OrderGrad(const FiniteElement *fe) const override;

   int GetSpaceDim() const override { return PointMat.Height(); }

   /** @brief Transform a point @a pt from physical space to a point @a ip in
       reference space and optionally can set a solver tolerance using @a phys_tol. */
   /** Attempt to find the IntegrationPoint that is transformed into the given
       point in physical space. If the inversion fails a non-zero value is
       returned. This method is not 100 percent reliable for non-linear
       transformations. */
   int TransformBack (const Vector & v, IntegrationPoint & ip,
                      const real_t phys_rel_tol = tol_0) override
   {
      InverseElementTransformation inv_tr(this);
      inv_tr.SetPhysicalRelTol(phys_rel_tol);
      return inv_tr.Transform(v, ip);
   }

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

/** @brief A specialized ElementTransformation class representing a face and
    its two neighboring elements.

    This class can be used as a container for the element transformation data
    needed for integrating discontinuous fields on element interfaces in a
    Discontinuous Galerkin (DG) context.

    The secondary purpose of this class is to enable the
    GridFunction::GetValue function, and various related functions, to properly
    evaluate fields with limited continuity on boundary elements.
*/
class FaceElementTransformations : public IsoparametricTransformation
{
private:

   // Bitwise OR of ConfigMasks
   int mask;

   IntegrationPoint eip1, eip2;

protected: // interface for Mesh to be able to configure this object.

   friend class Mesh;
#ifdef MFEM_USE_MPI
   friend class ParMesh;
#endif

   /// Set the mask indicating which portions of the object have been setup
   /** The argument @a m is a bitmask used in
       Mesh::GetFaceElementTransformations to indicate which portions of the
       FaceElementTransformations object have been configured.

       mask &  1: Elem1 is configured
       mask &  2: Elem2 is configured
       mask &  4: Loc1 is configured
       mask &  8: Loc2 is configured
       mask & 16: The Face transformation itself is configured
   */
   void SetConfigurationMask(int m) { mask = m; }

public:

   enum ConfigMasks
   {
      HAVE_ELEM1 =  1, ///< Element on side 1 is configured
      HAVE_ELEM2 =  2, ///< Element on side 2 is configured
      HAVE_LOC1  =  4, ///< Point transformation for side 1 is configured
      HAVE_LOC2  =  8, ///< Point transformation for side 2 is configured
      HAVE_FACE  = 16  ///< Face transformation is configured
   };

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

   /** @brief Return the mask defining the configuration state.

       The mask value indicates which portions of FaceElementTransformations
       object have been configured.

       mask &  1: Elem1 is configured
       mask &  2: Elem2 is configured
       mask &  4: Loc1 is configured
       mask &  8: Loc2 is configured
       mask & 16: The Face transformation itself is configured
   */
   int GetConfigurationMask() const { return mask; }

   /** @brief Set the integration point in the Face and the two neighboring
       elements, if present.

       The point @a face_ip must be in the reference coordinate system of the
       face.
   */
   void SetIntPoint(const IntegrationPoint *face_ip);

   /** @brief Set the integration point in the Face and the two neighboring
       elements, if present.

       This is a more expressive member function name than SetIntPoint, which
       in this special case, does the same thing. This function can be used for
       greater code clarity.
   */
   inline void SetAllIntPoints(const IntegrationPoint *face_ip)
   { FaceElementTransformations::SetIntPoint(face_ip); }

   /** @brief Get a const reference to the integration point in neighboring
       element 1 corresponding to the currently set integration point on the
       face.

       This IntegrationPoint object will only contain up-to-date data if
       SetIntPoint or SetAllIntPoints has been called with the latest
       integration point for the face and the appropriate point transformation
       has been configured. */
   const IntegrationPoint &GetElement1IntPoint() { return eip1; }

   /** @brief Get a const reference to the integration point in neighboring
       element 2 corresponding to the currently set integration point on the
       face.

       This IntegrationPoint object will only contain up-to-date data if
       SetIntPoint or SetAllIntPoints has been called with the latest
       integration point for the face and the appropriate point transformation
       has been configured. */
   const IntegrationPoint &GetElement2IntPoint() { return eip2; }

   void Transform(const IntegrationPoint &, Vector &) override;
   void Transform(const IntegrationRule &, DenseMatrix &) override;
   void Transform(const DenseMatrix &matrix, DenseMatrix &result) override;

   ElementTransformation & GetElement1Transformation();
   ElementTransformation & GetElement2Transformation();
   IntegrationPointTransformation & GetIntPoint1Transformation();
   IntegrationPointTransformation & GetIntPoint2Transformation();

   /** @brief Check for self-consistency: compares the result of mapping the
       reference face vertices to physical coordinates using the three
       transformations: face, element 1, and element 2.

       @param[in] print_level  If set to a positive number, print the physical
                               coordinates of the face vertices computed through
                               all available transformations: face, element 1,
                               and/or element 2.
       @param[in,out] out      The output stream to use for printing.

       @returns A maximal distance between physical coordinates of face vertices
                that should coincide. A successful check should return a small
                number relative to the mesh extents. If less than 2 of the three
                transformations are set, returns 0.

       @warning This check will generally fail on periodic boundary faces.
   */
   real_t CheckConsistency(int print_level = 0,
                           std::ostream &out = mfem::out);
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
