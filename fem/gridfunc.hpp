// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#ifdef MFEM_USE_ADIOS2
#include "../general/adios2stream.hpp"
#endif
#include <limits>
#include <ostream>
#include <string>

namespace mfem
{

/// Class for grid function - Vector with associated FE space.
class GridFunction : public Vector
{
protected:
   /// FE space on which the grid function lives. Owned if #fec is not NULL.
   FiniteElementSpace *fes;

   /** @brief Used when the grid function is read from a file. It can also be
       set explicitly, see MakeOwner().

       If not NULL, this pointer is owned by the GridFunction. */
   FiniteElementCollection *fec;

   long sequence; // see FiniteElementSpace::sequence, Mesh::sequence

   /** Optional, internal true-dof vector: if the FiniteElementSpace #fes has a
       non-trivial (i.e. not NULL) prolongation operator, this Vector may hold
       associated true-dof values - either owned or external. */
   Vector t_vec;

   void SaveSTLTri(std::ostream &out, double p1[], double p2[], double p3[]);

   void GetVectorGradientHat(ElementTransformation &T, DenseMatrix &gh) const;

   // Project the delta coefficient without scaling and return the (local)
   // integral of the projection.
   void ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                double &integral);

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

   void Destroy();

public:

   GridFunction() { fes = NULL; fec = NULL; sequence = 0; UseDevice(true); }

   /// Copy constructor. The internal true-dof vector #t_vec is not copied.
   GridFunction(const GridFunction &orig)
      : Vector(orig), fes(orig.fes), fec(NULL), sequence(orig.sequence)
   { UseDevice(true); }

   /// Construct a GridFunction associated with the FiniteElementSpace @a *f.
   GridFunction(FiniteElementSpace *f) : Vector(f->GetVSize())
   { fes = f; fec = NULL; sequence = f->GetSequence(); UseDevice(true); }

   /// Construct a GridFunction using previously allocated array @a data.
   /** The GridFunction does not assume ownership of @a data which is assumed to
       be of size at least `f->GetVSize()`. Similar to the Vector constructor
       for externally allocated array, the pointer @a data can be NULL. The data
       array can be replaced later using the method SetData().
    */
   GridFunction(FiniteElementSpace *f, double *data)
      : Vector(data, f->GetVSize())
   { fes = f; fec = NULL; sequence = f->GetSequence(); UseDevice(true); }

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
       assignemnt operator. */
   GridFunction &operator=(const GridFunction &rhs)
   { return operator=((const Vector &)rhs); }

   /// Make the GridFunction the owner of #fec and #fes.
   /** If the new FiniteElementCollection, @a _fec, is NULL, ownership of #fec
       and #fes is taken away. */
   void MakeOwner(FiniteElementCollection *_fec) { fec = _fec; }

   FiniteElementCollection *OwnFEC() { return fec; }

   int VectorDim() const;

   /// Read only access to the (optional) internal true-dof Vector.
   /** Note that the returned Vector may be empty, if not previously allocated
       or set. */
   const Vector &GetTrueVector() const { return t_vec; }
   /// Read and write access to the (optional) internal true-dof Vector.
   /** Note that the returned Vector may be empty, if not previously allocated
       or set. */
   Vector &GetTrueVector() { return t_vec; }

   /// @brief Extract the true-dofs from the GridFunction. If all dofs are true,
   /// then `tv` will be set to point to the data of `*this`.
   /** @warning This method breaks const-ness when all dofs are true. */
   void GetTrueDofs(Vector &tv) const;

   /// Shortcut for calling GetTrueDofs() with GetTrueVector() as argument.
   void SetTrueVector() { GetTrueDofs(GetTrueVector()); }

   /// Set the GridFunction from the given true-dof vector.
   virtual void SetFromTrueDofs(const Vector &tv);

   /// Shortcut for calling SetFromTrueDofs() with GetTrueVector() as argument.
   void SetFromTrueVector() { SetFromTrueDofs(GetTrueVector()); }

   /// Returns the values in the vertices of i'th element for dimension vdim.
   void GetNodalValues(int i, Array<double> &nval, int vdim = 1) const;

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
   virtual double GetValue(int i, const IntegrationPoint &ip,
                           int vdim = 1) const;

   /** Return a vector value from within the given element. */
   void GetVectorValue(int i, const IntegrationPoint &ip, Vector &val) const;
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
   double GetValue(ElementTransformation &T, const IntegrationPoint &ip,
                   int comp = 0, Vector *tr = NULL) const;

   /** Return a vector value from within the element indicated by the
       ElementTransformation Object. */
   void GetVectorValue(ElementTransformation &T, const IntegrationPoint &ip,
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

   /// Return the values as a vector on mesh vertices for dimension vdim.
   void GetNodalValues(Vector &nval, int vdim = 1) const;

   void GetVectorFieldNodalValues(Vector &val, int comp) const;

   void ProjectVectorFieldOn(GridFunction &vec_field, int comp = 0);

   void GetDerivative(int comp, int der_comp, GridFunction &der);

   double GetDivergence(ElementTransformation &tr) const;

   void GetCurl(ElementTransformation &tr, Vector &curl) const;

   void GetGradient(ElementTransformation &tr, Vector &grad) const;

   void GetGradients(ElementTransformation &tr, const IntegrationRule &ir,
                     DenseMatrix &grad) const;

   void GetGradients(const int elem, const IntegrationRule &ir,
                     DenseMatrix &grad) const
   { GetGradients(*fes->GetElementTransformation(elem), ir, grad); }

   void GetVectorGradient(ElementTransformation &tr, DenseMatrix &grad) const;

   /** Compute \f$ (\int_{\Omega} (*this) \psi_i)/(\int_{\Omega} \psi_i) \f$,
       where \f$ \psi_i \f$ are the basis functions for the FE space of avgs.
       Both FE spaces should be scalar and on the same mesh. */
   void GetElementAverages(GridFunction &avgs) const;

   /** Impose the given bounds on the function's DOFs while preserving its local
    *  integral (described in terms of the given weights) on the i'th element
    *  through SLBPQ optimization.
    *  Intended to be used for discontinuous FE functions. */
   void ImposeBounds(int i, const Vector &weights,
                     const Vector &_lo, const Vector &_hi);
   void ImposeBounds(int i, const Vector &weights,
                     double _min = 0.0, double _max = infinity());

   /** @brief Project the @a src GridFunction to @a this GridFunction, both of
       which must be on the same mesh. */
   /** The current implementation assumes that all elements use the same
       projection matrix. */
   void ProjectGridFunction(const GridFunction &src);

   virtual void ProjectCoefficient(Coefficient &coeff);

   void ProjectCoefficient(Coefficient &coeff, Array<int> &dofs, int vd = 0);

   void ProjectCoefficient(VectorCoefficient &vcoeff);

   void ProjectCoefficient(VectorCoefficient &vcoeff, Array<int> &dofs);

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

protected:
   /** @brief Accumulates (depending on @a type) the values of @a coeff at all
       shared vdofs and counts in how many zones each vdof appears. */
   void AccumulateAndCountZones(Coefficient &coeff, AvgType type,
                                Array<int> &zones_per_vdof);

   /** @brief Accumulates (depending on @a type) the values of @a vcoeff at all
       shared vdofs and counts in how many zones each vdof appears. */
   void AccumulateAndCountZones(VectorCoefficient &vcoeff, AvgType type,
                                Array<int> &zones_per_vdof);

   void AccumulateAndCountBdrValues(Coefficient *coeff[],
                                    VectorCoefficient *vcoeff, Array<int> &attr,
                                    Array<int> &values_counter);

   void AccumulateAndCountBdrTangentValues(VectorCoefficient &vcoeff,
                                           Array<int> &bdr_attr,
                                           Array<int> &values_counter);

   // Complete the computation of averages; called e.g. after
   // AccumulateAndCountZones().
   void ComputeMeans(AvgType type, Array<int> &zones_per_vdof);

public:
   /** @brief Project a Coefficient on the GridFunction, modifying only DOFs on
       the boundary associated with the boundary attributes marked in the
       @a attr array. */
   void ProjectBdrCoefficient(Coefficient &coeff, Array<int> &attr)
   {
      Coefficient *coeff_p = &coeff;
      ProjectBdrCoefficient(&coeff_p, attr);
   }

   /** @brief Project a VectorCoefficient on the GridFunction, modifying only
       DOFs on the boundary associated with the boundary attributes marked in
       the @a attr array. */
   virtual void ProjectBdrCoefficient(VectorCoefficient &vcoeff,
                                      Array<int> &attr);

   /** @brief Project a set of Coefficient%s on the components of the
       GridFunction, modifying only DOFs on the boundary associated with the
       boundary attributed marked in the @a attr array. */
   /** If a Coefficient pointer in the array @a coeff is NULL, that component
       will not be touched. */
   virtual void ProjectBdrCoefficient(Coefficient *coeff[], Array<int> &attr);

   /** Project the normal component of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes RT-type VectorFE GridFunction. */
   void ProjectBdrCoefficientNormal(VectorCoefficient &vcoeff,
                                    Array<int> &bdr_attr);

   /** @brief Project the tangential components of the given VectorCoefficient
       on the boundary. Only boundary attributes that are marked in @a bdr_attr
       are projected. Assumes ND-type VectorFE GridFunction. */
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                             Array<int> &bdr_attr);

   virtual double ComputeL2Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(2.0, exsol, NULL, irs); }

   virtual double ComputeL2Error(Coefficient *exsol[],
                                 const IntegrationRule *irs[] = NULL) const;

   virtual double ComputeL2Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const;

   virtual double ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                 Coefficient *ell_coef, double Nu,
                                 int norm_type) const;

   virtual double ComputeMaxError(Coefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, irs);
   }

   virtual double ComputeMaxError(Coefficient *exsol[],
                                  const IntegrationRule *irs[] = NULL) const;

   virtual double ComputeMaxError(VectorCoefficient &exsol,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsol, NULL, NULL, irs);
   }

   virtual double ComputeL1Error(Coefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, irs); }

   virtual double ComputeW11Error(Coefficient *exsol, VectorCoefficient *exgrad,
                                  int norm_type, Array<int> *elems = NULL,
                                  const IntegrationRule *irs[] = NULL) const;

   virtual double ComputeL1Error(VectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL) const
   { return ComputeLpError(1.0, exsol, NULL, NULL, irs); }

   virtual double ComputeLpError(const double p, Coefficient &exsol,
                                 Coefficient *weight = NULL,
                                 const IntegrationRule *irs[] = NULL) const;

   /** Compute the Lp error in each element of the mesh and store the results in
       the Vector @a error. The result should be of length number of elements,
       for example an L2 GridFunction of order zero using map type VALUE. */
   virtual void ComputeElementLpErrors(const double p, Coefficient &exsol,
                                       Vector &error,
                                       Coefficient *weight = NULL,
                                       const IntegrationRule *irs[] = NULL
                                      ) const;

   virtual void ComputeElementL1Errors(Coefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(1.0, exsol, error, NULL, irs); }

   virtual void ComputeElementL2Errors(Coefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(2.0, exsol, error, NULL, irs); }

   virtual void ComputeElementMaxErrors(Coefficient &exsol,
                                        Vector &error,
                                        const IntegrationRule *irs[] = NULL
                                       ) const
   { ComputeElementLpErrors(infinity(), exsol, error, NULL, irs); }

   /** When given a vector weight, compute the pointwise (scalar) error as the
       dot product of the vector error with the vector weight. Otherwise, the
       scalar error is the l_2 norm of the vector error. */
   virtual double ComputeLpError(const double p, VectorCoefficient &exsol,
                                 Coefficient *weight = NULL,
                                 VectorCoefficient *v_weight = NULL,
                                 const IntegrationRule *irs[] = NULL) const;

   /** Compute the Lp error in each element of the mesh and store the results in
       the Vector @ error. The result should be of length number of elements,
       for example an L2 GridFunction of order zero using map type VALUE. */
   virtual void ComputeElementLpErrors(const double p, VectorCoefficient &exsol,
                                       Vector &error,
                                       Coefficient *weight = NULL,
                                       VectorCoefficient *v_weight = NULL,
                                       const IntegrationRule *irs[] = NULL
                                      ) const;

   virtual void ComputeElementL1Errors(VectorCoefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(1.0, exsol, error, NULL, NULL, irs); }

   virtual void ComputeElementL2Errors(VectorCoefficient &exsol,
                                       Vector &error,
                                       const IntegrationRule *irs[] = NULL
                                      ) const
   { ComputeElementLpErrors(2.0, exsol, error, NULL, NULL, irs); }

   virtual void ComputeElementMaxErrors(VectorCoefficient &exsol,
                                        Vector &error,
                                        const IntegrationRule *irs[] = NULL
                                       ) const
   { ComputeElementLpErrors(infinity(), exsol, error, NULL, NULL, irs); }

   virtual void ComputeFlux(BilinearFormIntegrator &blfi,
                            GridFunction &flux,
                            bool wcoef = true, int subdomain = -1);

   /// Redefine '=' for GridFunction = constant.
   GridFunction &operator=(double value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       FiniteElementSpace #fes. */
   GridFunction &operator=(const Vector &v);

   /// Transform by the Space UpdateMatrix (e.g., on Mesh change).
   virtual void Update();

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
   virtual void MakeRef(FiniteElementSpace *f, double *v);

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
   void MakeTRef(FiniteElementSpace *f, double *tv);

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

   /** @brief Write the GridFunction in STL format. Note that the mesh dimension
       must be 2 and that quad elements will be broken into two triangles.*/
   void SaveSTL(std::ostream &out, int TimesToRefine = 1);

   /// Destroys grid function.
   virtual ~GridFunction() { Destroy(); }
};


/** Overload operator<< for std::ostream and GridFunction; valid also for the
    derived class ParGridFunction */
std::ostream &operator<<(std::ostream &out, const GridFunction &sol);


/** @brief Class representing a function through its values (scalar or vector)
    at quadrature points. */
class QuadratureFunction : public Vector
{
protected:
   QuadratureSpace *qspace; ///< Associated QuadratureSpace
   int vdim;                ///< Vector dimension
   bool own_qspace;         ///< QuadratureSpace ownership flag

public:
   /// Create an empty QuadratureFunction.
   /** The object can be initialized later using the SetSpace() methods. */
   QuadratureFunction()
      : qspace(NULL), vdim(0), own_qspace(false) { }

   /** @brief Copy constructor. The QuadratureSpace ownership flag, #own_qspace,
       in the new object is set to false. */
   QuadratureFunction(const QuadratureFunction &orig)
      : Vector(orig),
        qspace(orig.qspace), vdim(orig.vdim), own_qspace(false) { }

   /// Create a QuadratureFunction based on the given QuadratureSpace.
   /** The QuadratureFunction does not assume ownership of the QuadratureSpace.
       @note The Vector data is not initialized. */
   QuadratureFunction(QuadratureSpace *qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_->GetSize()),
        qspace(qspace_), vdim(vdim_), own_qspace(false) { }

   /** @brief Create a QuadratureFunction based on the given QuadratureSpace,
       using the external data, @a qf_data. */
   /** The QuadratureFunction does not assume ownership of neither the
       QuadratureSpace nor the external data. */
   QuadratureFunction(QuadratureSpace *qspace_, double *qf_data, int vdim_ = 1)
      : Vector(qf_data, vdim_*qspace_->GetSize()),
        qspace(qspace_), vdim(vdim_), own_qspace(false) { }

   /// Read a QuadratureFunction from the stream @a in.
   /** The QuadratureFunction assumes ownership of the read QuadratureSpace. */
   QuadratureFunction(Mesh *mesh, std::istream &in);

   virtual ~QuadratureFunction() { if (own_qspace) { delete qspace; } }

   /// Get the associated QuadratureSpace.
   QuadratureSpace *GetSpace() const { return qspace; }

   /// Change the QuadratureSpace and optionally the vector dimension.
   /** If the new QuadratureSpace is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data size is updated by calling Vector::SetSize(). */
   inline void SetSpace(QuadratureSpace *qspace_, int vdim_ = -1);

   /** @brief Change the QuadratureSpace, the data array, and optionally the
       vector dimension. */
   /** If the new QuadratureSpace is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data array is replaced by calling Vector::NewDataAndSize(). */
   inline void SetSpace(QuadratureSpace *qspace_, double *qf_data,
                        int vdim_ = -1);

   /// Get the vector dimension.
   int GetVDim() const { return vdim; }

   /// Set the vector dimension, updating the size by calling Vector::SetSize().
   void SetVDim(int vdim_)
   { vdim = vdim_; SetSize(vdim*qspace->GetSize()); }

   /// Get the QuadratureSpace ownership flag.
   bool OwnsSpace() { return own_qspace; }

   /// Set the QuadratureSpace ownership flag.
   void SetOwnsSpace(bool own) { own_qspace = own; }

   /// Redefine '=' for QuadratureFunction = constant.
   QuadratureFunction &operator=(double value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       QuadratureSpace #qspace. */
   QuadratureFunction &operator=(const Vector &v);

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** The QuadratureFunctions @a v and @a *this must have QuadratureSpaces with
       the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignemnt operator. */
   QuadratureFunction &operator=(const QuadratureFunction &v);

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return qspace->GetElementIntRule(idx); }

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a reference to the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, Vector &values);

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a copy of the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, Vector &values) const;

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a reference to the
       global values. */
   inline void GetElementValues(int idx, const int ip_num, Vector &values);

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a copy to the
       global values. */
   inline void GetElementValues(int idx, const int ip_num, Vector &values) const;

   /// Return all values associated with mesh element @a idx in a DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a reference to the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, DenseMatrix &values);

   /// Return all values associated with mesh element @a idx in a const DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a copy of the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetElementValues(int idx, DenseMatrix &values) const;

   /// Write the QuadratureFunction to the stream @a out.
   void Save(std::ostream &out) const;
};

/// Overload operator<< for std::ostream and QuadratureFunction.
std::ostream &operator<<(std::ostream &out, const QuadratureFunction &qf);


double ZZErrorEstimator(BilinearFormIntegrator &blfi,
                        GridFunction &u,
                        GridFunction &flux,
                        Vector &error_estimates,
                        Array<int> *aniso_flags = NULL,
                        int with_subdomains = 1,
                        bool with_coeff = false);

/// Compute the Lp distance between two grid functions on the given element.
double ComputeElementLpDistance(double p, int i,
                                GridFunction& gf1, GridFunction& gf2);


/// Class used for extruding scalar GridFunctions
class ExtrudeCoefficient : public Coefficient
{
private:
   int n;
   Mesh *mesh_in;
   Coefficient &sol_in;
public:
   ExtrudeCoefficient(Mesh *m, Coefficient &s, int _n)
      : n(_n), mesh_in(m), sol_in(s) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ExtrudeCoefficient() { }
};

/// Extrude a scalar 1D GridFunction, after extruding the mesh with Extrude1D.
GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny);


// Inline methods

inline void QuadratureFunction::SetSpace(QuadratureSpace *qspace_, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   SetSize(vdim*qspace->GetSize());
}

inline void QuadratureFunction::SetSpace(QuadratureSpace *qspace_,
                                         double *qf_data, int vdim_)
{
   if (qspace_ != qspace)
   {
      if (own_qspace) { delete qspace; }
      qspace = qspace_;
      own_qspace = false;
   }
   vdim = (vdim_ < 0) ? vdim : vdim_;
   NewDataAndSize(qf_data, vdim*qspace->GetSize());
}

inline void QuadratureFunction::GetElementValues(int idx, Vector &values)
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.NewDataAndSize(data + vdim*s_offset, vdim*sl_size);
}

inline void QuadratureFunction::GetElementValues(int idx, Vector &values) const
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.SetSize(vdim*sl_size);
   const double *q = data + vdim*s_offset;
   for (int i = 0; i<values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetElementValues(int idx, const int ip_num,
                                                 Vector &values)
{
   const int s_offset = qspace->element_offsets[idx] * vdim + ip_num * vdim;
   values.NewDataAndSize(data + s_offset, vdim);
}

inline void QuadratureFunction::GetElementValues(int idx, const int ip_num,
                                                 Vector &values) const
{
   const int s_offset = qspace->element_offsets[idx] * vdim + ip_num * vdim;
   values.SetSize(vdim);
   const double *q = data + s_offset;
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetElementValues(int idx, DenseMatrix &values)
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.Reset(data + vdim*s_offset, vdim, sl_size);
}

inline void QuadratureFunction::GetElementValues(int idx,
                                                 DenseMatrix &values) const
{
   const int s_offset = qspace->element_offsets[idx];
   const int sl_size = qspace->element_offsets[idx+1] - s_offset;
   values.SetSize(vdim, sl_size);
   const double *q = data + vdim*s_offset;
   for (int j = 0; j<sl_size; j++)
   {
      for (int i = 0; i<vdim; i++)
      {
         values(i,j) = *(q++);
      }
   }
}

} // namespace mfem

#endif
