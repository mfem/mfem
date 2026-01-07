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

#ifndef MFEM_GRIDFUNC
#define MFEM_GRIDFUNC

#include "../config/config.hpp"
#include "fespace.hpp"
#include "coefficient.hpp"
#include "bilininteg.hpp"
#include "bounds.hpp"
#ifdef MFEM_USE_ADIOS2
#include "../general/adios2stream.hpp"
#endif
#include <limits>
#include <ostream>
#include <string>

namespace mfem
{

/** This enumerated type describes the three main projection types:
    - ELEMENT, assigns the degree of freedom per element, as specified in the
      specific element
    - GLOBAL_L2, solves a global L2 projection
    - ELEMENT_L2, solves a element level L2 projection. Inter element
      connectivity is dealt with similar as in:
      Bezier-Projection : A unified approach for local projection and
      quadrature-free refinement and coarsening of NURBS and T-splines with
      particular application to isogeometric design and analysis
      [CMAME (284) 2015 pg 55-105]
    - DEFAULT, for NURBS spaces this is ELEMENT_L2, while for all other spaces
      this ELEMENT.
   Note 1: ELEMENT_L2 also works for non NURBS elements
   Note 2: For NURBS elements the ELEMENT projection gives results without
   over and undershoots. However, the gradient near the boundary does not
   converge.*/
enum class ProjectType { DEFAULT, ELEMENT, GLOBAL_L2, ELEMENT_L2 };

/// Class for grid function - Vector with associated FE space.
class GridFunction : public Vector
{
protected:
   /// FE space on which the grid function lives. Owned if #fec_owned is not NULL.
   FiniteElementSpace *fes;

   /** @brief Used when the grid function is read from a file. It can also be
       set explicitly, see MakeOwner().

       If not NULL, this pointer is owned by the GridFunction. */
   FiniteElementCollection *fec_owned;

   long fes_sequence; // see FiniteElementSpace::sequence, Mesh::sequence

   /** Optional, internal true-dof vector: if the FiniteElementSpace #fes has a
       non-trivial (i.e. not NULL) prolongation operator, this Vector may hold
       associated true-dof values - either owned or external. */
   Vector t_vec;

   void SaveSTLTri(std::ostream &out, real_t p1[], real_t p2[], real_t p3[]);

   // Project the delta coefficient without scaling and return the (local)
   // integral of the projection.
   void ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                real_t &integral);

   // Sum fluxes to vertices and count element contributions
   void SumFluxAndCount(BilinearFormIntegrator &blfi,
                        GridFunction &flux,
                        Array<int>& counts,
                        bool wcoef,
                        int subdomain);

   /** Project a discontinuous vector coefficient in a continuous space and
       return in dof_attr the maximal attribute of the elements containing each
       degree of freedom. */
   void ProjectDiscCoefficient(VectorCoefficient &coeff, Array<int> &dof_attr);

   /** Helper function for ProjectCoefficientElementL2 */
   void ProjectCoefficientElementL2_(Coefficient &coeff, Vector &sol, Vector &Va);
   void ProjectCoefficientElementL2_(VectorCoefficient &vcoeff, Vector &sol,
                                     Vector &Va);

   /// Loading helper.
   void LegacyNCReorder();

   void Destroy();

public:
   GridFunction() { fes = NULL; fec_owned = NULL; fes_sequence = 0; UseDevice(true); }

   /// Copy constructor. The internal true-dof vector #t_vec is not copied.
   GridFunction(const GridFunction &orig)
      : Vector(orig), fes(orig.fes), fec_owned(NULL), fes_sequence(orig.fes_sequence)
   { UseDevice(true); }

   /// Construct a GridFunction associated with the FiniteElementSpace @a *f.
   GridFunction(FiniteElementSpace *f) : Vector(f->GetVSize())
   { fes = f; fec_owned = NULL; fes_sequence = f->GetSequence(); UseDevice(true); }

   /// Same as above but specify the memory type
   GridFunction(FiniteElementSpace *f, MemoryType mt) : Vector(f->GetVSize(), mt)
   { fes = f; fec_owned = NULL; fes_sequence = f->GetSequence(); UseDevice(true); }

   /// Construct a GridFunction using previously allocated array @a data.
   /** The GridFunction does not assume ownership of @a data which is assumed to
       be of size at least `f->GetVSize()`. Similar to the Vector constructor
       for externally allocated array, the pointer @a data can be NULL. The data
       array can be replaced later using the method SetData().
    */
   GridFunction(FiniteElementSpace *f, real_t *data)
      : Vector(data, f->GetVSize())
   { fes = f; fec_owned = NULL; fes_sequence = f->GetSequence(); UseDevice(true); }

   /** @brief Construct a GridFunction using previously allocated Vector @a base
       starting at the given offset, @a base_offset. */
   GridFunction(FiniteElementSpace *f, Vector &base, int base_offset = 0)
      : Vector(base, base_offset, f->GetVSize())
   { fes = f; fec_owned = NULL; fes_sequence = f->GetSequence(); UseDevice(true); }

   /// Construct a GridFunction on the given Mesh, using the data from @a input.
   /** The content of @a input should be in the format created by the method
       Save(). The reconstructed FiniteElementSpace and FiniteElementCollection
       are owned by the GridFunction. */
   GridFunction(Mesh *m, std::istream &input);

   GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces);

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** It is assumed that this object and @a rhs use FiniteElementSpace%s that
       have the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   GridFunction &operator=(const GridFunction &rhs)
   { return operator=((const Vector &)rhs); }

   /// Make the GridFunction the owner of #fec_owned and #fes.
   /** If the new FiniteElementCollection, @a fec_, is NULL, ownership of #fec_owned
       and #fes is taken away. */
   void MakeOwner(FiniteElementCollection *fec_) { fec_owned = fec_; }

   FiniteElementCollection *OwnFEC() { return fec_owned; }

   /// Shortcut for calling FiniteElementSpace::GetVectorDim() on the underlying #fes
   int VectorDim() const;

   /// Shortcut for calling FiniteElementSpace::GetCurlDim() on the underlying #fes
   int CurlDim() const;

   /// Read only access to the (optional) internal true-dof Vector.
   const Vector &GetTrueVector() const
   {
      MFEM_VERIFY(t_vec.Size() > 0, "SetTrueVector() before GetTrueVector()");
      return t_vec;
   }
   /// Read and write access to the (optional) internal true-dof Vector.
   /** Note that @a t_vec is set if it is not allocated or set already.*/
   Vector &GetTrueVector()
   { if (t_vec.Size() == 0) { SetTrueVector(); } return t_vec; }

   /// Extract the true-dofs from the GridFunction.
   void GetTrueDofs(Vector &tv) const;

   /// Shortcut for calling GetTrueDofs() with GetTrueVector() as argument.
   void SetTrueVector() { GetTrueDofs(t_vec); }

   /// Set the GridFunction from the given true-dof vector.
   virtual void SetFromTrueDofs(const Vector &tv);

   /// Shortcut for calling SetFromTrueDofs() with GetTrueVector() as argument.
   void SetFromTrueVector() { SetFromTrueDofs(GetTrueVector()); }

   /** @brief Returns the values at the vertices of element @a i for the 1-based
       dimension vdim. */
   void GetNodalValues(int i, Array<real_t> &nval, int vdim = 1) const;

   /** @name Element index Get Value Methods

       These methods take an element index and return the interpolated value of
       the field at a given reference point within the element.

       @warning These methods retrieve and use the ElementTransformation object
       from the mfem::Mesh. This can alter the state of the element
       transformation object and can also lead to unexpected results when the
       ElementTransformation object is already in use such as when these methods
       are called from within an integration loop. Consider using
       GetValue(ElementTransformation &T, ...) instead.
   */
   ///@{
   /** Return a scalar value from within the given element. */
   virtual real_t GetValue(int i, const IntegrationPoint &ip,
                           int vdim = 1) const;

   /** Return a vector value from within the given element. */
   virtual void GetVectorValue(int i, const IntegrationPoint &ip,
                               Vector &val) const;
   ///@}

   /** @name Element Index Get Values Methods

       These are convenience methods for repeatedly calling GetValue for
       multiple points within a given element. The GetValues methods are
       optimized and should perform better than repeatedly calling GetValue. The
       GetVectorValues method simply calls GetVectorValue repeatedly.

       @warning These methods retrieve and use the ElementTransformation object
       from the mfem::Mesh. This can alter the state of the element
       transformation object and can also lead to unexpected results when the
       ElementTransformation object is already in use such as when these methods
       are called from within an integration loop. Consider using
       GetValues(ElementTransformation &T, ...) instead.
   */
   ///@{
   /** Compute a collection of scalar values from within the element indicated
       by the index i. */
   void GetValues(int i, const IntegrationRule &ir, Vector &vals,
                  int vdim = 1) const;

   /** Compute a collection of vector values from within the element indicated
       by the index i. */
   void GetValues(int i, const IntegrationRule &ir, Vector &vals,
                  DenseMatrix &tr, int vdim = 1) const;

   void GetVectorValues(int i, const IntegrationRule &ir,
                        DenseMatrix &vals, DenseMatrix &tr) const;
   ///@}

   /** @name ElementTransformation Get Value Methods

       These member functions are designed for use within
       GridFunctionCoefficient objects. These can be used with
       ElementTransformation objects coming from either
       Mesh::GetElementTransformation() or Mesh::GetBdrElementTransformation().

       @note These methods do not reset the ElementTransformation object so they
       should be safe to use within integration loops or other contexts where
       the ElementTransformation is already in use.
   */
   ///@{
   /** Return a scalar value from within the element indicated by the
       ElementTransformation Object. */
   virtual real_t GetValue(ElementTransformation &T, const IntegrationPoint &ip,
                           int comp = 0, Vector *tr = NULL) const;

   /** Return a vector value from within the element indicated by the
       ElementTransformation Object. */
   virtual void GetVectorValue(ElementTransformation &T,
                               const IntegrationPoint &ip,
                               Vector &val, Vector *tr = NULL) const;
   ///@}

   /** @name ElementTransformation Get Values Methods

       These are convenience methods for repeatedly calling GetValue for
       multiple points within a given element. They work by calling either the
       ElementTransformation or FaceElementTransformations versions described
       above. Consequently, these methods should not be expected to run faster
       than calling the above methods in an external loop.

       @note These methods do not reset the ElementTransformation object so they
       should be safe to use within integration loops or other contexts where
       the ElementTransformation is already in use.

       @note These methods can also be used with FaceElementTransformations
       objects.
    */
   ///@{
   /** Compute a collection of scalar values from within the element indicated
       by the ElementTransformation object. */
   void GetValues(ElementTransformation &T, const IntegrationRule &ir,
                  Vector &vals, int comp = 0, DenseMatrix *tr = NULL) const;

   /** Compute a collection of vector values from within the element indicated
       by the ElementTransformation object. */
   void GetVectorValues(ElementTransformation &T, const IntegrationRule &ir,
                        DenseMatrix &vals, DenseMatrix *tr = NULL) const;
   ///@}

   /** @name Face Index Get Values Methods

       These methods are designed to work with Discontinuous Galerkin basis
       functions. They compute field values on the interface between elements,
       or on boundary elements, by interpolating the field in a neighboring
       element. The \a side argument indices which neighboring element should be
       used: 0, 1, or 2 (automatically chosen).

       @warning These methods retrieve and use the FaceElementTransformations
       object from the mfem::Mesh. This can alter the state of the face element
       transformations object and can also lead to unexpected results when the
       FaceElementTransformations object is already in use such as when these
       methods are called from within an integration loop. Consider using
       GetValues(ElementTransformation &T, ...) instead.
    */
   ///@{
   /** Compute a collection of scalar values from within the face
       indicated by the index i. */
   int GetFaceValues(int i, int side, const IntegrationRule &ir, Vector &vals,
                     DenseMatrix &tr, int vdim = 1) const;

   /** Compute a collection of vector values from within the face
       indicated by the index i. */
   int GetFaceVectorValues(int i, int side, const IntegrationRule &ir,
                           DenseMatrix &vals, DenseMatrix &tr) const;
   ///@}

   void GetLaplacians(int i, const IntegrationRule &ir, Vector &laps,
                      int vdim = 1) const;

   void GetLaplacians(int i, const IntegrationRule &ir, Vector &laps,
                      DenseMatrix &tr, int vdim = 1) const;

   void GetHessians(int i, const IntegrationRule &ir, DenseMatrix &hess,
                    int vdim = 1) const;

   void GetHessians(int i, const IntegrationRule &ir, DenseMatrix &hess,
                    DenseMatrix &tr, int vdim = 1) const;

   void GetValuesFrom(const GridFunction &orig_func);

   void GetBdrValuesFrom(const GridFunction &orig_func);

   void GetVectorFieldValues(int i, const IntegrationRule &ir,
                             DenseMatrix &vals,
                             DenseMatrix &tr, int comp = 0) const;

   /// For a vector grid function, makes sure that the ordering is byNODES.
   void ReorderByNodes();

   /** @brief Returns the values as a vector at mesh vertices, for the 1-based
       dimension vdim. */
   void GetNodalValues(Vector &nval, int vdim = 1) const;

   void GetVectorFieldNodalValues(Vector &val, int comp) const;

   void ProjectVectorFieldOn(GridFunction &vec_field, int comp = 0);

   /** @brief Compute a certain derivative of a function's component.
       Derivatives of the function are computed at the DOF locations of @a der,
       and averaged over overlapping DOFs. Thus this function projects the
       derivative to the FiniteElementSpace of @a der.
       @param[in]  comp  Index of the function's component to be differentiated.
                         The index is 1-based, i.e., use 1 for scalar functions.
       @param[in]  der_comp  Use 0/1/2 for derivatives in x/y/z directions.
       @param[out] der       The resulting derivative (scalar function). The
                             FiniteElementSpace of this function must be set
                             before the call. */
   void GetDerivative(int comp, int der_comp, GridFunction &der) const;

   real_t GetDivergence(ElementTransformation &tr) const;

   void GetCurl(ElementTransformation &tr, Vector &curl) const;

   /** @brief Gradient of a scalar function at a quadrature point.

       @note It is assumed that the IntegrationPoint of interest has been
       specified by ElementTransformation::SetIntPoint() before calling
       GetGradient().

       @note Can be used from a ParGridFunction when @a tr is an
       ElementTransformation of a face-neighbor element and face-neighbor data
       has been exchanged. */
   void GetGradient(ElementTransformation &tr, Vector &grad) const;

   /// Extension of GetGradient(...) for a collection of IntegrationPoints.
   void GetGradients(ElementTransformation &tr, const IntegrationRule &ir,
                     DenseMatrix &grad) const;

   /// Extension of GetGradient(...) for a collection of IntegrationPoints.
   void GetGradients(const int elem, const IntegrationRule &ir,
                     DenseMatrix &grad) const
   { GetGradients(*fes->GetElementTransformation(elem), ir, grad); }

   /** @brief Compute the vector gradient with respect to the physical element
       variable. */
   void GetVectorGradient(ElementTransformation &tr, DenseMatrix &grad) const;

   /** @brief Compute the vector gradient with respect to the reference element
       variable. */
   void GetVectorGradientHat(ElementTransformation &T, DenseMatrix &gh) const;

   /** @brief Evaluate the gradients of the GridFunction at the given quadrature
       points, @a ir, in all mesh elements. */
   /** This method assumes that all mesh elements are the same type and that the
       IntegrationRule @a ir is consistent with that type of element.

       @param[in] ir     Quadrature points at which the gradients are to be
                         evaluated.
       @param[out] grad  Output vector of size `SDIM*VDIM*NQ*NE` where `SDIM` is
                         the spatial dimention of the mesh, `VDIM` is the vector
                         dimension of the GridFunction, `NQ` is the number of
                         quadrature points in @a ir, and `NE` is the number of
                         elements in the mesh. The layout of @a grad is
                         determined by the parameter @a ql: when @a ql is
                         QVectorLayout::byNODES, the layout is
                         `NQ x VDIM x SDIM x NE`; when @a ql is
                         QVectorLayout::byVDIM, the layout is
                         `VDIM x SDIM x NQ x NE`.
       @param[in] ql     Determines the layout of the output vector @a grad; see
                         the description of @a grad for details.
       @param[in] d_mt   MemoryType to use for allocating the output vector
                         @a grad, as well the GeometricFactors and temporary
                         vector used by the method. By default, the current
                         device memory type is used. */
   void GetGradients(const IntegrationRule &ir, Vector &grad,
                     QVectorLayout ql = QVectorLayout::byNODES,
                     MemoryType d_mt = MemoryType::DEFAULT) const;

   /** Compute $ (\int_{\Omega} (*this) \psi_i)/(\int_{\Omega} \psi_i) $,
       where $ \psi_i $ are the basis functions for the FE space of avgs.
       Both FE spaces should be scalar and on the same mesh. */
   void GetElementAverages(GridFunction &avgs) const;

   /** Sets the output vector @a dof_vals to the values of the degrees of
       freedom of element @a el. */
   virtual void GetElementDofValues(int el, Vector &dof_vals) const;

   /** Impose the given bounds on the function's DOFs while preserving its local
    *  integral (described in terms of the given weights) on the i'th element
    *  through SLBPQ optimization.
    *  Intended to be used for discontinuous FE functions. */
   void ImposeBounds(int i, const Vector &weights,
                     const Vector &lo_, const Vector &hi_);
   void ImposeBounds(int i, const Vector &weights,
                     real_t min_ = 0.0, real_t max_ = infinity());

   /** On a non-conforming mesh, make sure the function lies in the conforming
       space by multiplying with R and then with P, the conforming restriction
       and prolongation matrices of the space, respectively. */
   void RestrictConforming();

   /** @brief Project the @a src GridFunction to @a this GridFunction, both of
       which must be on the same mesh. */
   /** The current implementation assumes that all elements use the same
       projection matrix. */
   void ProjectGridFunction(const GridFunction &src);

   /** @brief Project @a coeff Coefficient to @a this GridFunction. The
       projection computation depends on the choice of the FiniteElementSpace
       #fes. Note that this is usually interpolation at the degrees of freedom
       in each element (not L2 projection). For elements without a projection
       member function one could use ProjectCoefficientGlobalL2 instead.
       NOTE: For parallel simulations with NURBS elements some dofs might
       not be defined, if the evaluation point does not reside on this rank.
       If that is the case it is defined on another rank, and the issue is
       rectified with the appropriate communication, see in ParGridFunction.
       */
   virtual void ProjectCoefficient(Coefficient &coeff,
                                   ProjectType type = ProjectType::DEFAULT);

   /** @brief Project @a coeff Coefficient to @a this GridFunction. The
       projection is a global L2 projection. This routine can be used a
       fallback for elements without a projection member function.*/
   virtual void ProjectCoefficientGlobalL2(Coefficient &coeff,
                                           real_t rtol = 1e-12,
                                           int iter = 1000);

   /** @brief Project @a coeff Coefficient to @a this GridFunction. The
       projection is an element local L2 projection, with an appropriate
       weighting for Dofs that are shared between elements. Inspired on
       Bezier-Projection [CMAME (284) 2015 pg 55-105]
       This routine can be used a fallback for elements without a projection
       member function.*/
   virtual void ProjectCoefficientElementL2(Coefficient &coeff);

   /** @brief Project @a coeff Coefficient to @a this GridFunction, using one
       element for each degree of freedom in @a dofs and nodal interpolation on
       that element. */
   void ProjectCoefficient(Coefficient &coeff, Array<int> &dofs, int vd = 0);

   /** @brief Project @a vcoeff VectorCoefficient to @a this GridFunction. The
       projection computation depends on the choice of the FiniteElementSpace
       #fes. Note that this is usually interpolation at the degrees of freedom
       in each element (not L2 projection). For elements without a projection
       member function one could use ProjectCoefficientGlobalL2 instead.
       NOTE: For parallel simulations with NURBS elements some dofs might
       not be defined, if the evaluation point does not reside on this rank.
       If that is the case it is defined on another rank, and the issue is
       rectified with the appropriate communication, see in ParGridFunction.*/
   virtual void ProjectCoefficient(VectorCoefficient &vcoeff,
                                   ProjectType type = ProjectType::DEFAULT);

   /** @brief Project @a coeff Coefficient to @a this GridFunction. The
       projection is a global L2 projection. This routine can be used a
       fallback for elements without a projection member function.*/
   virtual void ProjectCoefficientGlobalL2(VectorCoefficient &vcoeff,
                                           real_t rtol = 1e-12,
                                           int iter = 1000);

   /** @brief Project @a coeff Coefficient to @a this GridFunction. The
       projection is a global L2 projection. This routine can be used a
       fallback for elements without a projection member function.*/
   virtual void ProjectCoefficientElementL2(VectorCoefficient &vcoeff);

   /** @brief Project @a vcoeff VectorCoefficient to @a this GridFunction, using
       one element for each degree of freedom in @a dofs and nodal interpolation
       on that element. */
   void ProjectCoefficient(VectorCoefficient &vcoeff, Array<int> &dofs);

   /** @brief Project @a vcoeff VectorCoefficient to @a this GridFunction, only
       projecting onto elements with the given @a attribute */
   void ProjectCoefficient(VectorCoefficient &vcoeff, int attribute);

   /** @brief Analogous to the version with argument @a vcoeff VectorCoefficient
       but using an array of scalar coefficients for each component. */
   void ProjectCoefficient(Coefficient *coeff[]);

   /** @brief Project a discontinuous vector coefficient as a grid function on
       a continuous finite element space. The values in shared dofs are
       determined from the element with maximal attribute. */
   virtual void ProjectDiscCoefficient(VectorCoefficient &coeff);

   enum AvgType {ARITHMETIC, HARMONIC};
   /** @brief Projects a discontinuous coefficient so that the values in shared
       vdofs are computed by taking an average of the possible values. */
   virtual void ProjectDiscCoefficient(Coefficient &coeff, AvgType type);
   /** @brief Projects a discontinuous _vector_ coefficient so that the values
       in shared vdofs are computed by taking an average of the possible values.
   */
   virtual void ProjectDiscCoefficient(VectorCoefficient &coeff, AvgType type);

   /** @brief Return a GridFunction with the values of this, prolongated to the
       maximum order of all elements in the mesh. */
   std::unique_ptr<GridFunction> ProlongateToMaxOrder() const;

protected:
   /** @brief Accumulates (depending on @a type) the values of @a coeff at all
       shared vdofs and counts in how many zones each vdof appears. */
   void AccumulateAndCountZones(Coefficient &coeff, AvgType type,
                                Array<int> &zones_per_vdof);

   /** @brief Accumulates (depending on @a type) the values of @a vcoeff at all
       shared vdofs and counts in how many zones each vdof appears. */
   void AccumulateAndCountZones(VectorCoefficient &vcoeff, AvgType type,
                                Array<int> &zones_per_vdof);

   /** @brief Used for the serial and parallel implementations of the
       GetDerivative() method; see its documentation. */
   void AccumulateAndCountDerivativeValues(int comp, int der_comp,
                                           GridFunction &der,
                                           Array<int> &zones_per_dof) const;

   void AccumulateAndCountBdrValues(Coefficient *coeff[],
                                    VectorCoefficient *vcoeff,
                                    const Array<int> &attr,
                                    Array<int> &values_counter);

   void AccumulateAndCountBdrTangentValues(VectorCoefficient &vcoeff,
                                           const Array<int> &bdr_attr,
                                           Array<int> &values_counter);

   // Complete the computation of averages; called e.g. after
   // AccumulateAndCountZones().
   void ComputeMeans(AvgType type, Array<int> &zones_per_vdof);

   /// P-refinement version of Update().
   void UpdatePRef();

public:
   /** @brief For each vdof, counts how many elements contain the vdof,
       as containment is determined by FiniteElementSpace::GetElementVDofs(). */
   virtual void CountElementsPerVDof(Array<int> &elem_per_vdof) const;

   /** @brief Project a Coefficient on the GridFunction, modifying only DOFs on
       the boundary associated with the boundary attributes marked in the
       @a attr array. */
   void ProjectBdrCoefficient(Coefficient &coeff, const Array<int> &attr)
   {
      Coefficient *coeff_p = &coeff;
      ProjectBdrCoefficient(&coeff_p, attr);
   }

   /** @brief Project a VectorCoefficient on the GridFunction, modifying only
       DOFs on the boundary associated with the boundary attributes marked in
       the @a attr array. */
   virtual void ProjectBdrCoefficient(VectorCoefficient &vcoeff,
                                      const Array<int> &attr);

   /** @brief Project a set of Coefficient%s on the components of the
       GridFunction, modifying only DOFs on the boundary associated with the
       boundary attributed marked in the @a attr array. */
   /** If a Coefficient pointer in the array @a coeff is NULL, that component
       will not be touched. */
   virtual void ProjectBdrCoefficient(Coefficient *coeff[],
                                      const Array<int> &attr);

   /** Project the normal component of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes RT-type VectorFE GridFunction. */
   void ProjectBdrCoefficientNormal(VectorCoefficient &vcoeff,
                                    const Array<int> &bdr_attr);

   /** @brief Project the tangential components of the given VectorCoefficient
       on the boundary. Only boundary attributes that are marked in @a bdr_attr
       are projected. Assumes ND-type VectorFE GridFunction. */
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                             const Array<int> &bdr_attr);

   /// @brief Returns ||exsol - u_h||_L2 for scalar or vector H1 or L2 elements
   ///
   /// @param[in] exsol   Pointer to an array of scalar Coefficient objects,
   ///                    one for each component of the vector field. The
   ///                    length of the array should be at least equal to
   ///                    FiniteElementSpace::GetVDim().
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   /// @param[in] elems   Optional pointer to a marker array, with a length
   ///                    equal to the number of local elements, indicating
   ///                    which elements to integrate over. Only those elements
   ///                    corresponding to non-zero entries in @a elems will
   ///                    contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(Coefficient *exsol[],
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||grad u_ex - grad u_h||_L2 in element ielem for
   ///        H1 or L2 elements
   ///
   /// @param[in] ielem   Index of the element in which to compute the L2 error.
   /// @param[in] exgrad  Pointer to a VectorCoefficient object reproducing the
   ///                    expected gradient of the scalar field, grad u_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeElementGradError(int ielem, VectorCoefficient *exgrad,
                                          const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns ||u_ex - u_h||_L2 for H1 or L2 elements
   ///
   /// @param[in] exsol  Coefficient object reproducing the anticipated values
   ///                   of the scalar field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const
   { return GridFunction::ComputeLpError(2.0, exsol, NULL, irs, elems); }

   /// @brief Returns ||u_ex - u_h||_L2 for vector fields
   ///
   /// @param[in] exsol  VectorCoefficient object reproducing the anticipated
   ///                   values of the vector field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||grad u_ex - grad u_h||_L2 for H1 or L2 elements
   ///
   /// @param[in] exgrad  Pointer to a VectorCoefficient object reproducing the
   ///                    expected gradient of the scalar field, grad u_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note This function only computes the error of the gradient in the
   ///       interior of the elements. In the context of discontinuous
   ///       Galerkin (DG) methods it may also be desirable to compute the
   ///       error in the jumps across element interfaces using
   ///       ComputeDGFaceJumpError().
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeGradError(VectorCoefficient *exgrad,
                                   const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns ||curl u_ex - curl u_h||_L2 for ND elements
   ///
   /// @param[in] excurl  Pointer to a VectorCoefficient object reproducing the
   ///                    expected curl of the vector field, curl u_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeCurlError(VectorCoefficient *excurl,
                                   const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns ||div u_ex - div u_h||_L2 for RT elements
   ///
   /// @param[in] exdiv  Pointer to a Coefficient object reproducing the
   ///                   expected divergence of the vector field, div u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeDivError(Coefficient *exdiv,
                                  const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns the Face Jumps error for L2 elements.
   ///
   /// Computes:
   ///   $$\sqrt{\sum_{f\in faces}\int_f js(f) ell(f)
   ///                                   (2 u_{ex} - u_1 - u_2)^2}$$
   ///
   /// Where js[f] is the jump_scaling evaluated on the face f and ell is the
   /// average of ell_coef evaluated in the two elements sharing the face f.
   ///
   /// @param[in] exsol         Pointer to a Coefficient object reproducing the
   ///                          anticipated values of the scalar field, u_ex.
   /// @param[in] ell_coeff     Pointer to a Coefficient object used to compute
   ///                          the averaged value ell in the above integral.
   /// @param[in] jump_scaling  Can be configured to provide scaling by
   ///                          nu, nu/h, or nu*p^2/h
   /// @param[in] irs           Optional pointer to an array of custom
   ///                          integration rules e.g. higher order than the
   ///                          default rules. If present the array will be
   ///                          indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of faces.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeDGFaceJumpError(Coefficient *exsol,
                                         Coefficient *ell_coeff,
                                         class JumpScaling jump_scaling,
                                         const IntegrationRule *irs[] = NULL)
   const;

   /// @brief Returns the Face Jumps error for L2 elements, with 1/h scaling.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @deprecated See @ref ComputeDGFaceJumpError(Coefficient *exsol,
   ///                                      Coefficient *ell_coeff,
   ///                                      class JumpScaling jump_scaling,
   ///                                      const IntegrationRule *irs[]) const
   ///             for the preferred implementation.
   MFEM_DEPRECATED
   real_t ComputeDGFaceJumpError(Coefficient *exsol,
                                 Coefficient *ell_coeff,
                                 real_t Nu,
                                 const IntegrationRule *irs[] = NULL) const;

   /** This method is kept for backward compatibility.

       Returns either the H1-seminorm, or the DG face jumps error, or both
       depending on norm_type = 1, 2, 3. Additional arguments for the DG face
       jumps norm: ell_coeff: mesh-depended coefficient (weight) Nu: scalar
       constant weight */
   virtual real_t ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                 Coefficient *ell_coef, real_t Nu,
                                 int norm_type) const;

   /// @brief Returns the error measured in H1-norm for H1 or L2 elements
   ///
   /// Computes the norm using the $L^2$ norms of the function and its gradient
   ///    $$\sqrt{norm\_u^2 + norm\_du^2}$$
   /// Where
   ///    $$norm\_u = \|u_{ex} - u_h\|_{L^2}$$
   /// and
   ///    $$norm\_du = \|du_{ex} - \nabla u_h\|_{L^2}$$
   ///
   /// @param[in] exsol   Coefficient object reproducing the anticipated values
   ///                    of the scalar field, u_ex.
   /// @param[in] exgrad  VectorCoefficient object reproducing the anticipated
   ///                    values of the gradient of the scalar field, du_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note For L2 elements this returns what could be called a "broken"
   ///       H1-norm.
   virtual real_t ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                 const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns the error measured in H(div)-norm for RT elements
   ///
   /// Computes the norm using the $L^2$ norms of the function and its
   /// divergence
   ///    $$\sqrt{norm\_u^2 + norm\_du^2}$$
   /// Where
   ///    $$norm\_u = \|u_{ex} - u_h\|_{L^2}$$
   /// and
   ///    $$norm\_du = \|du_{ex} - \nabla\cdot u_h\|_{L^2}$$
   ///
   /// @param[in] exsol  VectorCoefficient object reproducing the anticipated
   ///                   values of the vector field, u_ex.
   /// @param[in] exdiv  VectorCoefficient object reproducing the anticipated
   ///                   values of the divergence of the vector field, du_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeHDivError(VectorCoefficient *exsol,
                                   Coefficient *exdiv,
                                   const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns the error measured in H(curl)-norm for ND elements
   ///
   /// Computes the norm using the $L^2$ norms of the function and its curl
   ///    $$\sqrt{norm\_u^2 + norm\_du^2}$$
   /// Where
   ///    $$norm\_u = \|u_{ex} - u_h\|_{L^2}$$
   /// and
   ///    $$norm\_du = \|du_{ex} - \nabla\times u_h\|_{L^2}$$
   ///
   /// @param[in] exsol   VectorCoefficient object reproducing the anticipated
   ///                    values of the vector field, u_ex.
   /// @param[in] excurl  VectorCoefficient object reproducing the anticipated
   ///                    values of the curl of the vector field, du_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeHCurlError(VectorCoefficient *exsol,
                                    VectorCoefficient *excurl,
                                    const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns Max|u_ex - u_h| error for H1 or L2 elements
   ///
   /// Compute the $L^\infty$ error across the entire domain.
   ///
   /// @param[in] exsol      Coefficient object reproducing the anticipated
   ///                       values of the scalar field, u_ex.
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note Uses ComputeLpError internally. See the ComputeLpError
   ///       documentation for generalizations of this error computation.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   virtual real_t ComputeMaxError(Coefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, irs);
   }

   /// @brief Returns Max|u_ex - u_h| error for scalar or vector fields
   ///
   /// Compute the $L^\infty$ error across the entire domain.
   ///
   /// Computes:
   ///    $$max_{elems} (max_{elem} |scalar\_error|)$$
   ///
   /// Where
   ///    $$scalar\_error = max_{d=0\ldots vdim}|u_{ex}[d] - u_h[d]|$$
   ///
   /// @param[in] exsol  Pointer to an array of scalar Coefficient objects,
   ///                   one for each component of the vector field. The
   ///                   length of the array should be at least equal to
   ///                   FiniteElementSpace::GetVDim().
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   ///
   /// @note This implementation of the max error of a vector field computes
   ///       the max norm over vector components rather than the magnitude of
   ///       the vector.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   virtual real_t ComputeMaxError(Coefficient *exsol[],
                                  const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns Max|u_ex - u_h| error for vector fields
   ///
   /// Compute the $L^\infty$ error across the entire domain.
   ///
   /// Computes:
   ///    $$max_{elems} (max_{elem} |scalar\_error|)$$
   ///
   /// Where
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] exsol      VectorCoefficient object reproducing the
   ///                       anticipated values of the vector field, u_ex.
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note Uses ComputeLpError internally. See the ComputeLpError
   ///       documentation for generalizations of this error computation.
   ///
   /// @note Computes the maximum magnitude of the difference vector not the
   ///       component-wise maximum difference of the vector fields.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   virtual real_t ComputeMaxError(VectorCoefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, NULL, irs);
   }

   /// @brief Returns ||u_ex - u_h||_L1 for H1 or L2 elements
   ///
   /// Computes:
   ///    $$\sum_{elems} \int_{elem} |u_{ex} - u_h|$$
   ///
   /// @param[in] exsol   Coefficient object reproducing the anticipated values
   ///                    of the scalar field, u_ex.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeLpError internally. See the ComputeLpError
   ///       documentation for generalizations of this error computation.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   virtual real_t ComputeL1Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, irs); }

   /// @brief Returns ||u_ex - u_h||_L1 for H1 or L2 elements
   ///
   /// Computes:
   ///    $$\sum_{elems} \int_{elem} |u_{ex} - u_h|$$
   ///
   /// @param[in] exsol   Pointer to an array of Coefficient objects
   ///                    reproducing the anticipated values of the scalar
   ///                    field, u_ex. Only the first entry of this array will
   ///                    be accessed.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeW11Error internally. See the ComputeW11Error
   ///       documentation for generalizations of this error computation.
   ///
   /// @warning While this function is nominally equivalent to ComputeLpError,
   ///          with appropriate arguments, the returned errors may differ
   ///          noticeably because ComputeLpError uses a higher order
   ///          integration rule by default.
   ///
   /// @deprecated See @ref ComputeL1Error(Coefficient &exsol,
   ///                              const IntegrationRule *irs[]) const
   ///             for the preferred implementation.
   MFEM_DEPRECATED
   virtual real_t ComputeL1Error(Coefficient *exsol[],
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeW11Error(*exsol, NULL, 1, NULL, irs); }

   /// @brief Returns $W^1_1$ norm (or portions thereof) for H1 or L2 elements
   ///
   /// Computes for norm_type == 1 the $L^1$ norm of $u$:
   ///    $$(\sum_{elems} \int_{elem} |u_{ex} - u_h|$$
   ///
   /// Computes for norm_type == 2 the $L^1$ semi-norm of $\nabla u$:
   ///    $$(\sum_{elems} \int_{elem} |du_{ex} - \nabla u_h|$$
   ///
   /// Computes for norm_type == 3 the $W^1_1$ norm of $u$:
   ///    $$(\sum_{elems} \int_{elem} |u_{ex} - u_h| + |du_{ex} - \nabla u_h|$$
   ///
   /// @param[in] exsol     Pointer to Coefficient object reproducing the
   ///                      anticipated values of the scalar field, u_ex.
   /// @param[in] exgrad    Pointer to VectorCoefficient object reproducing the
   ///                      anticipated values of the gradient of the scalar
   ///                      field, du_ex.
   /// @param[in] norm_type Integer value of 1, 2, or 3 indicating the type of
   ///                      norm to compute (see above).
   /// @param[in] elems     Optional pointer to a marker array, with a length
   ///                      equal to the number of local elements, indicating
   ///                      which elements to integrate over. Only those
   ///                      elements corresponding to non-zero entries in
   ///                      @a elems will contribute to the computed $W^1_1$
   ///                      error.
   /// @param[in] irs       Optional pointer to an array of custom integration
   ///                      rules e.g. higher order than the default rules. If
   ///                      present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeW11Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                  int norm_type, const Array<int> *elems = NULL,
                                  const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns ||u_ex - u_h||_L1 for vector fields
   ///
   /// Computes:
   ///    $$\sum_{elems} \int_{elem} |scalar\_error|$$
   ///
   /// Where
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] exsol     VectorCoefficient object reproducing the anticipated
   ///                      values of the vector field, u_ex.
   /// @param[in] irs       Optional pointer to an array of custom integration
   ///                      rules e.g. higher order than the default rules. If
   ///                      present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeLpError internally. See the ComputeLpError
   ///       documentation for generalizations of this error computation.
   virtual real_t ComputeL1Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, NULL, irs); }

   /// @brief Returns ||u_ex - u_h||_Lp for H1 or L2 elements
   ///
   /// Computes:
   ///    $$(\sum_{elems} \int_{elem} w \, |u_{ex} - u_h|^p)^{1/p}$$
   ///
   /// @param[in] p       Real value indicating the exponent of the $L^p$ norm.
   ///                    To avoid domain errors p should have a positive value,
   ///                    either finite or infinite.
   /// @param[in] exsol   Coefficient object reproducing the anticipated values
   ///                    of the scalar field, u_ex.
   /// @param[in] weight  Optional pointer to a Coefficient object reproducing
   ///                    a weighting function, w.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   /// @param[in] elems   Optional pointer to a marker array, with a length
   ///                    equal to the number of local elements, indicating
   ///                    which elements to integrate over. Only those elements
   ///                    corresponding to non-zero entries in @a elems will
   ///                    contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeLpError(const real_t p, Coefficient &exsol,
                                 Coefficient *weight = NULL,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||u_ex - u_h||_Lp elementwise for H1 or L2 elements
   ///
   /// Compute the Lp error in each element of the mesh and store the results in
   /// the Vector @a error. The result should be of length number of elements,
   /// for example an L2 GridFunction of order zero using map type @ref
   /// map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$(\int_{elem} w \, |u_{ex} - u_h|^p)^{1/p}$$
   ///
   /// @param[in] p          Real value indicating the exponent of the $L^p$
   ///                       norm. To avoid domain errors p should have a
   ///                       positive value, either finite or infinite.
   /// @param[in] exsol      Coefficient object reproducing the anticipated
   ///                       values of the scalar field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^p$ errors
   /// @param[in] weight     Optional pointer to a Coefficient object
   ///                       reproducing a weighting function, w.
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual void ComputeElementLpErrors(const real_t p, Coefficient &exsol,
                                       Vector &error,
                                       Coefficient *weight = NULL,
                                       const IntegrationRule *irs[] = NULL
                                      ) const;

   /// @brief Returns ||u_ex - u_h||_L1 elementwise for H1 or L2 elements
   ///
   /// Compute the $L^1$ error in each element of the mesh and store the
   /// results in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// @param[in] exsol      Coefficient object reproducing the anticipated
   ///                       values of the scalar field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^1$ errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   virtual void ComputeElementL1Errors(Coefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(1.0, exsol, error, NULL, irs); }

   /// @brief Returns ||u_ex - u_h||_L2 elementwise for H1 or L2 elements
   ///
   /// Compute the $L^2$ error in each element of the mesh and store the results
   /// in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$(\int_{elem} |u_{ex} - u_h|^2)^{1/2}$$
   ///
   /// @param[in] exsol      Coefficient object reproducing the anticipated
   ///                       values of the scalar field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^2$ errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   virtual void ComputeElementL2Errors(Coefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(2.0, exsol, error, NULL, irs); }

   /// @brief Returns Max|u_ex - u_h| elementwise for H1 or L2 elements
   ///
   /// Compute the $L^\infty$ error in each element of the mesh and store the
   /// results in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// @param[in] exsol      Coefficient object reproducing the anticipated
   ///                       values of the scalar field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^\infty$
   ///                       errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   virtual void ComputeElementMaxErrors(Coefficient &exsol,
                                        Vector &error,
                                        const IntegrationRule *irs[] = NULL
                                       ) const
   { ComputeElementLpErrors(infinity(), exsol, error, NULL, irs); }

   /// @brief Returns ||u_ex - u_h||_Lp for vector fields
   ///
   /// When given a vector weight, compute the pointwise (scalar) error as the
   /// dot product of the vector error with the vector weight. Otherwise, the
   /// scalar error is the l_2 norm of the vector error.
   ///
   /// Computes:
   ///    $$(\sum_{elems} \int_{elem} w \, |scalar\_error|^p)^{1/p}$$
   ///
   /// Where
   ///    $$scalar\_error = |v\_weight \cdot (u_{ex} - u_h)|$$
   /// or
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] p         Real value indicating the exponent of the $L^p$
   ///                      norm. To avoid domain errors p should have a
   ///                      positive value, either finite or infinite.
   /// @param[in] exsol     VectorCoefficient object reproducing the anticipated
   ///                      values of the vector field, u_ex.
   /// @param[in] weight    Optional pointer to a Coefficient object reproducing
   ///                      a weighting function, w.
   /// @param[in] v_weight  Optional pointer to a VectorCoefficient object
   ///                      reproducing a weighting vector as shown above.
   /// @param[in] irs       Optional pointer to an array of custom integration
   ///                      rules e.g. higher order than the default rules. If
   ///                      present the array will be indexed by Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeLpError(const real_t p, VectorCoefficient &exsol,
                                 Coefficient *weight = NULL,
                                 VectorCoefficient *v_weight = NULL,
                                 const IntegrationRule *irs[] = NULL) const;

   /// @brief Returns ||u_ex - u_h||_Lp elementwise for vector fields
   ///
   /// Compute the $L^p$ error in each element of the mesh and store the results
   /// in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$(\int_{elem} w \, |scalar\_error|^p)^{1/p}$$
   ///
   /// Where
   ///    $$scalar\_error = |v\_weight \cdot (u_{ex} - u_h)|$$
   /// or
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] p          Real value indicating the exponent of the $L^p$
   ///                       norm. To avoid domain errors p should have a
   ///                       positive value, either finite or infinite.
   /// @param[in] exsol      VectorCoefficient object reproducing the
   ///                       anticipated values of the vector field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^p$ errors
   /// @param[in] weight     Optional pointer to a Coefficient object
   ///                       reproducing a weighting function, w.
   /// @param[in] v_weight   Optional pointer to a VectorCoefficient object
   ///                       reproducing a weighting vector as shown above.
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual void ComputeElementLpErrors(const real_t p, VectorCoefficient &exsol,
                                       Vector &error,
                                       Coefficient *weight = NULL,
                                       VectorCoefficient *v_weight = NULL,
                                       const IntegrationRule *irs[] = NULL
                                      ) const;

   /// @brief Returns ||u_ex - u_h||_L1 elementwise for vector fields
   ///
   /// Compute the $L^1$ error in each element of the mesh and store the
   /// results in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$\int_{elem} |scalar\_error|$$
   ///
   /// Where
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] exsol      VectorCoefficient object reproducing the
   ///                       anticipated values of the vector field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^1$ errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   virtual void ComputeElementL1Errors(VectorCoefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(1.0, exsol, error, NULL, NULL, irs); }

   /// @brief Returns ||u_ex - u_h||_L2 elementwise for vector fields
   ///
   /// Compute the $L^2$ error in each element of the mesh and store the
   /// results in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$(\int_{elem} |scalar\_error|^2)^{1/2}$$
   ///
   /// Where
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] exsol      VectorCoefficient object reproducing the
   ///                       anticipated values of the vector field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^2$ errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   virtual void ComputeElementL2Errors(VectorCoefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(2.0, exsol, error, NULL, NULL, irs); }

   /// @brief Returns Max|u_ex - u_h| elementwise for vector fields
   ///
   /// Compute the $L^\infty$ error in each element of the mesh and store the
   /// results in the Vector @a error. The result should be of length number of
   /// elements, for example an L2 GridFunction of order zero using map type
   /// @ref map_type_value "VALUE".
   ///
   /// Computes:
   ///    $$max_{elem} |scalar\_error|$$
   ///
   /// Where
   ///    $$scalar\_error = \sqrt{(u_{ex} - u_h) \cdot (u_{ex} - u_h)}$$
   ///
   /// @param[in] exsol      VectorCoefficient object reproducing the
   ///                       anticipated values of the vector field, u_ex.
   /// @param[in,out] error  Vector to contain the element-wise $L^\infty$
   ///                       errors
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Uses ComputeElementLpError internally. See the
   ///       ComputeElementLpError documentation for generalizations of this
   ///       error computation.
   ///
   /// @note Computes the maximum magnitude of the difference vector not the
   ///       component-wise maximum difference of the vector fields.
   virtual void ComputeElementMaxErrors(VectorCoefficient &exsol,
                                        Vector &error,
                                        const IntegrationRule *irs[] = NULL
                                       ) const
   { ComputeElementLpErrors(infinity(), exsol, error, NULL, NULL, irs); }

   virtual void ComputeFlux(BilinearFormIntegrator &blfi,
                            GridFunction &flux,
                            bool wcoef = true, int subdomain = -1);

   /// Redefine '=' for GridFunction = constant.
   GridFunction &operator=(real_t value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       FiniteElementSpace #fes. */
   GridFunction &operator=(const Vector &v);

   /// Transform by the Space UpdateMatrix (e.g., on Mesh change).
   virtual void Update();

   /** Return update counter, similar to Mesh::GetSequence(). Used to
       check if it is up to date with the space. */
   long GetSequence() const { return fes_sequence; }

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Associate a new FiniteElementSpace with the GridFunction.
   /** The GridFunction is resized using the SetSize() method. */
   virtual void SetSpace(FiniteElementSpace *f);

   using Vector::MakeRef;

   /** @brief Make the GridFunction reference external data on a new
       FiniteElementSpace. */
   /** This method changes the FiniteElementSpace associated with the
       GridFunction and sets the pointer @a v as external data in the
       GridFunction. */
   virtual void MakeRef(FiniteElementSpace *f, real_t *v);

   /** @brief Make the GridFunction reference external data on a new
       FiniteElementSpace. */
   /** This method changes the FiniteElementSpace associated with the
       GridFunction and sets the data of the Vector @a v (plus the @a v_offset)
       as external data in the GridFunction.
       @note This version of the method will also perform bounds checks when
       the build option MFEM_DEBUG is enabled. */
   virtual void MakeRef(FiniteElementSpace *f, Vector &v, int v_offset);

   /** @brief Associate a new FiniteElementSpace and new true-dof data with the
       GridFunction. */
   /** - If the prolongation matrix of @a f is trivial (i.e. its method
         FiniteElementSpace::GetProlongationMatrix() returns NULL), then the
         method MakeRef() is called with the same arguments.
       - Otherwise, the method SetSpace() is called with argument @a f.
       - The internal true-dof vector is set to reference @a tv. */
   void MakeTRef(FiniteElementSpace *f, real_t *tv);

   /** @brief Associate a new FiniteElementSpace and new true-dof data with the
       GridFunction. */
   /** - If the prolongation matrix of @a f is trivial (i.e. its method
         FiniteElementSpace::GetProlongationMatrix() returns NULL), this method
         calls MakeRef() with the same arguments.
       - Otherwise, this method calls SetSpace() with argument @a f.
       - The internal true-dof vector is set to reference the sub-vector of
         @a tv starting at the offset @a tv_offset. */
   void MakeTRef(FiniteElementSpace *f, Vector &tv, int tv_offset);

   /// Save the GridFunction to an output stream.
   virtual void Save(std::ostream &out) const;

   /// Save the GridFunction to a file. The given @a precision will be used for
   /// ASCII output.
   virtual void Save(const char *fname, int precision=16) const;

#ifdef MFEM_USE_ADIOS2
   /// Save the GridFunction to a binary output stream using adios2 bp format.
   virtual void Save(adios2stream &out, const std::string& variable_name,
                     const adios2stream::data_type
                     type = adios2stream::data_type::point_data) const;
#endif

   /** @brief Write the GridFunction in VTK format. Note that Mesh::PrintVTK
       must be called first. The parameter ref > 0 must match the one used in
       Mesh::PrintVTK. */
   void SaveVTK(std::ostream &out, const std::string &field_name, int ref);

#ifdef MFEM_USE_HDF5
   /// @brief Save the GridFunction in %VTKHDF format.
   ///
   /// If @a high-order is true, then @a ref controls the order of output. If
   /// @a ref is -1, then the order of the grid function will be used.
   ///
   /// If @a high-order is false, then low-order output will be used. @a ref
   /// controls the number of mesh refinements; if @a ref is -1, no refinements
   /// will be performed.
   void SaveVTKHDF(const std::string &fname, const std::string &name="u",
                   bool high_order=true, int ref=-1);
#endif

   /** @brief Write the GridFunction in STL format. Note that the mesh dimension
       must be 2 and that quad elements will be broken into two triangles.*/
   void SaveSTL(std::ostream &out, int TimesToRefine = 1);

   /** @name Methods to compute bounds on the grid function
       \brief See bounds.hpp for \ref PLBound that constructs piecewise linear
       bounds for a given set of bases. These piecewise bounds can be used to compute bounds on a grid function. Currently tensor-product elements are
       supported with Lagrange interpolants on Gauss Legendre nodes and Gauss Lobatto Legendre nodes, and Bernstein bases.
    */
   ///@{
   /// Computes the \ref PLBound for the gridfunction with number of control
   /// points based on @a ref_factor, and returns the overall bounds for each
   /// vdim (across all elements) in @b lower and @b upper. We also return the
   /// PLBound object used to compute the bounds.
   /// We compute the bounds for each vdim if @a vdim < 1.
   /// Note: For most cases, this method/interface will be sufficient.
   virtual PLBound GetBounds(Vector &lower, Vector &upper,
                             const int ref_factor=1, const int vdim=-1) const;

   /// Computes the \ref PLBound for the gridfunction with number of control
   /// points based on @a ref_factor, and returns the bounds for each element
   /// ordered byVDim:
   /// lower_{0,0}, lower_{1,0}, ..., lower_{ne-1,0},
   /// lower_{0,1}, ..., lower_{ne-1,vdim-1}. We also return the
   /// PLBound object used to compute the bounds.
   /// We compute the bounds for each vdim if @a vdim < 1.
   PLBound GetElementBounds(Vector &lower, Vector &upper,
                            const int ref_factor=1, const int vdim=-1) const;

   /// Compute piecewise linear bounds on the given element at the grid of
   /// [plb.ncp x plb.ncp x plb.ncp] control points for each of the vdim
   /// components of the gridfunction.
   void GetElementBoundsAtControlPoints(const int elem, const PLBound &plb,
                                        Vector &lower, Vector &upper,
                                        const int vdim = -1) const;

   /// Compute bounds on the grid function for the given element.
   /// The bounds are stored in @b lower and @b upper.
   void GetElementBounds(const int elem, const PLBound &plb,
                         Vector &lower, Vector &upper,
                         const int vdim = -1) const;

   /// Compute bounds on the grid function for all the elements. The bounds
   /// are returned in @b lower and @b upper, ordered byVDim:
   /// lower_{0,0}, lower_{1,0}, ..., lower_{ne-1,0},
   /// lower_{0,1}, ..., lower_{ne-1,vdim-1}
   void GetElementBounds(const PLBound &plb, Vector &lower, Vector &upper,
                         const int vdim=-1) const;
   ///@}

   /// Destroys grid function.
   virtual ~GridFunction() { Destroy(); }
};

/** Overload operator<< for std::ostream and GridFunction; valid also for the
    derived class ParGridFunction */
std::ostream &operator<<(std::ostream &out, const GridFunction &sol);

/// Class used to specify how the jump terms in
/// GridFunction::ComputeDGFaceJumpError are scaled.
class JumpScaling
{
public:
   enum JumpScalingType
   {
      CONSTANT,
      ONE_OVER_H,
      P_SQUARED_OVER_H
   };
private:
   real_t nu;
   JumpScalingType type;
public:
   JumpScaling(real_t nu_=1.0, JumpScalingType type_=CONSTANT)
      : nu(nu_), type(type_) { }
   real_t Eval(real_t h, int p) const
   {
      real_t val = nu;
      if (type != CONSTANT) { val /= h; }
      if (type == P_SQUARED_OVER_H) { val *= p*p; }
      return val;
   }
};

/// Overload operator<< for std::ostream and QuadratureFunction.
std::ostream &operator<<(std::ostream &out, const QuadratureFunction &qf);


real_t ZZErrorEstimator(BilinearFormIntegrator &blfi,
                        GridFunction &u,
                        GridFunction &flux,
                        Vector &error_estimates,
                        Array<int> *aniso_flags = NULL,
                        int with_subdomains = 1,
                        bool with_coeff = false);

/// Defines the global tensor product polynomial space used by NewZZErorrEstimator
/**
 *  See BoundingBox(...) for a description of @a angle and @a midpoint
 */
void TensorProductLegendre(int dim,                      // input
                           int order,                    // input
                           const Vector &x_in,           // input
                           const Vector &xmax,           // input
                           const Vector &xmin,           // input
                           Vector &poly,                 // output
                           real_t angle=0.0,             // input (optional)
                           const Vector *midpoint=NULL); // input (optional)

/// Defines the bounding box for the face patches used by NewZZErorrEstimator
/**
 *  By default, BoundingBox(...) computes the parameters of a minimal bounding box
 *  for the given @a face_patch that is aligned with the physical (i.e. global)
 *  Cartesian axes. This means that the size of the bounding box will depend on the
 *  orientation of the patch. It is better to construct an orientation-independent box.
 *  This is implemented for 2D patches. The parameters @a angle and @a midpoint encode
 *  the necessary additional geometric information.
 *
 *      @a iface     : Index of the face that the patch corresponds to.
 *                     This is used to compute @a angle and @a midpoint.
 *
 *      @a angle     : The angle the patch face makes with the x-axis.
 *      @a midpoint  : The midpoint of the face.
 */
void BoundingBox(const Array<int> &face_patch, // input
                 FiniteElementSpace *ufes,     // input
                 int order,                    // input
                 Vector &xmin,                 // output
                 Vector &xmax,                 // output
                 real_t &angle,                // output
                 Vector &midpoint,             // output
                 int iface=-1);                // input (optional)

/// A ``true'' ZZ error estimator that uses face-based patches for flux reconstruction.
/**
 *  Only two-element face patches are ever used:
 *   - For conforming faces, the face patch consists of its two neighboring elements.
 *   - In the non-conforming setting, only the face patches associated to fine-scale
 *     element faces are used. These face patches always consist of two elements
 *     delivered by mesh::GetFaceElements(Face, *Elem1, *Elem2).
 */
real_t LSZZErrorEstimator(BilinearFormIntegrator &blfi,         // input
                          GridFunction &u,                      // input
                          Vector &error_estimates,              // output
                          bool subdomain_reconstruction = true, // input (optional)
                          bool with_coeff = false,              // input (optional)
                          real_t tichonov_coeff = 0.0);         // input (optional)

/// Compute the Lp distance between two grid functions on the given element.
real_t ComputeElementLpDistance(real_t p, int i,
                                GridFunction& gf1, GridFunction& gf2);


/// Class used for extruding scalar GridFunctions
class ExtrudeCoefficient : public Coefficient
{
private:
   int n;
   Mesh *mesh_in;
   Coefficient &sol_in;
public:
   ExtrudeCoefficient(Mesh *m, Coefficient &s, int n_)
      : n(n_), mesh_in(m), sol_in(s) { }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
   virtual ~ExtrudeCoefficient() { }
};

/// Extrude a scalar 1D GridFunction, after extruding the mesh with Extrude1D.
GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny);

} // namespace mfem

#endif
