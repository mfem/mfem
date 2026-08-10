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

#ifndef MFEM_DOFTRANSFORM
#define MFEM_DOFTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"

namespace mfem
{

/** The StatelessDofTransformation class is an abstract base class for a family
    of transformations that map local degrees of freedom (DoFs), contained
    within individual elements, to global degrees of freedom, stored within
    GridFunction objects.

    In this context "stateless" means that the concrete classes derived from
    StatelessDofTransformation do not store information about the relative
    orientations of the faces with respect to their neighboring elements. In
    other words there is no information specific to a particular element (aside
    from the element type e.g. tetrahedron, wedge, or pyramid). The
    StatelessDofTransformation provides access to the transformation operators
    for specific relative face orientations. These are useful, for example, when
    relating DoFs associated with distinct overlapping meshes such as parent and
    sub-meshes.

    These transformations are necessary to ensure that basis functions in
    neighboring (or overlapping) elements align correctly. Closely related but
    complementary transformations are required for the entries stored in
    LinearForm and BilinearForm objects. The StatelessDofTransformation class
    is designed to apply the action of both of these types of DoF
    transformations.

    Let the "primal transformation" be given by the operator T. This means that
    given a local element vector v the data that must be placed into a
    GridFunction object is v_t = T * v.

    We also need the inverse of the primal transformation T^{-1} so that we can
    recover the local element vector from data read out of a GridFunction
    e.g. v = T^{-1} * v_t.

    We need to preserve the action of our linear forms applied to primal
    vectors. In other words, if f is the local vector computed by a linear
    form then f * v = f_t * v_t (where "*" represents an inner product of
    vectors). This requires that f_t = T^{-T} * f i.e. the "dual transform" is
    given by the transpose of the inverse of the primal transformation.

    For bilinear forms we require that v^T * A * v = v_t^T * A_t * v_t. This
    implies that A_t = T^{-T} * A * T^{-1}. This can be accomplished by
    performing dual transformations of the rows and columns of the matrix A.

    For discrete linear operators the range must be modified with the primal
    transformation rather than the dual transformation because the result is a
    primal vector rather than a dual vector. This leads to the transformation
    D_t = T * D * T^{-1}. This can be accomplished by using a primal
    transformation on the columns of D and a dual transformation on its rows.
*/
class StatelessDofTransformation
{
protected:
   int size_;

   StatelessDofTransformation(int size)
      : size_(size) {}

public:
   inline int Size() const { return size_; }
   inline int Height() const { return size_; }
   inline int NumRows() const { return size_; }
   inline int Width() const { return size_; }
   inline int NumCols() const { return size_; }

   /// If the DofTransformation performs no transformation
   virtual bool IsIdentity() const = 0;

   /** Transform local DoFs to align with the global DoFs. For example, this
       transformation can be used to map the local vector computed by
       FiniteElement::Project() to the transformed vector stored within a
       GridFunction object. */
   virtual void TransformPrimal(const Array<int> & face_orientation,
                                real_t *v) const = 0;
   inline void TransformPrimal(const Array<int> & face_orientation,
                               Vector &v) const
   { TransformPrimal(face_orientation, v.GetData()); }

   /** Inverse transform local DoFs. Used to transform DoFs from a global vector
       back to their element-local form. For example, this must be used to
       transform the vector obtained using GridFunction::GetSubVector before it
       can be used to compute a local interpolation.
   */
   virtual void InvTransformPrimal(const Array<int> & face_orientation,
                                   real_t *v) const = 0;
   inline void InvTransformPrimal(const Array<int> & face_orientation,
                                  Vector &v) const
   { InvTransformPrimal(face_orientation, v.GetData()); }

   /** Transform dual DoFs as computed by a LinearFormIntegrator before summing
       into a LinearForm object. */
   virtual void TransformDual(const Array<int> & face_orientation,
                              real_t *v) const = 0;
   inline void TransformDual(const Array<int> & face_orientation,
                             Vector &v) const
   { TransformDual(face_orientation, v.GetData()); }

   /** Inverse Transform dual DoFs */
   virtual void InvTransformDual(const Array<int> & face_orientation,
                                 real_t *v) const = 0;
   inline void InvTransformDual(const Array<int> & face_orientation,
                                Vector &v) const
   { InvTransformDual(face_orientation, v.GetData()); }

   virtual ~StatelessDofTransformation() = default;
};

/** The DofTransformation class is an extension of the
    StatelessDofTransformation which stores the face orientations used to
    select the necessary transformations which allows it to offer a collection
    of convenience methods.

    DofTransformation objects are provided by the FiniteElementSpace which has
    access to the mesh and can therefore provide the face orientations. This is
    convenient when working with GridFunction, LinearForm, or BilinearForm
    objects or their parallel counterparts.

    StatelessDofTransformation objects are provided by FiniteElement or
    FiniteElementCollection objects which do not have access to face
    orientation information. This can be useful in non-standard contexts such as
    transferring finite element degrees of freedom between different meshes.
    For examples of its use see the TransferMap used by the SubMesh class.
   */
class DofTransformation
{
protected:
   Array<int> Fo_;
   const StatelessDofTransformation * dof_trans_;
   int vdim_;
   int ordering_;

public:
   /** @brief Default constructor which requires that SetDofTransformation be
       called before use. */
   DofTransformation(int vdim = 1, int ordering = 0)
      : dof_trans_(NULL)
      , vdim_(vdim)
      , ordering_(ordering)
   {}

   /// Constructor with a known StatelessDofTransformation
   DofTransformation(const StatelessDofTransformation & dof_trans,
                     int vdim = 1, int ordering = 0)
      : dof_trans_(&dof_trans)
      , vdim_(vdim)
      , ordering_(ordering)
   {}

   /** @brief Configure the transformation using face orientations for the
       current element. */
   /// The face_orientation array can be obtained from Mesh::GetElementFaces.
   inline void SetFaceOrientations(const Array<int> & Fo)
   { Fo_ = Fo; }

   /// Return the face orientations for the current element
   inline const Array<int> & GetFaceOrientations() const { return Fo_; }

   /// Set or change the nested StatelessDofTransformation object
   inline void SetDofTransformation(const StatelessDofTransformation & dof_trans)
   {
      dof_trans_ = &dof_trans;
   }
   inline void SetDofTransformation(const StatelessDofTransformation * dof_trans)
   {
      dof_trans_ = dof_trans;
   }

   /// Return the nested StatelessDofTransformation object
   inline const StatelessDofTransformation * GetDofTransformation() const
   { return dof_trans_; }

   /// Set or change the vdim and ordering parameter
   inline void SetVDim(int vdim = 1, int ordering = 0)
   {
      vdim_ = vdim;
      ordering_ = ordering;
   }

   /// Return the current vdim value
   inline int GetVDim() const { return vdim_; }

   inline int Size() const { return dof_trans_->Size(); }
   inline int Height() const { return dof_trans_->Height(); }
   inline int NumRows() const { return dof_trans_->NumRows(); }
   inline int Width() const { return dof_trans_->Width(); }
   inline int NumCols() const { return dof_trans_->NumCols(); }
   inline bool IsIdentity() const { return !dof_trans_ || dof_trans_->IsIdentity(); }

   /** Transform local DoFs to align with the global DoFs. For example, this
       transformation can be used to map the local vector computed by
       FiniteElement::Project() to the transformed vector stored within a
       GridFunction object. */
   void TransformPrimal(real_t *v) const;
   inline void TransformPrimal(Vector &v) const { TransformPrimal(v.GetData()); }

   /// Transform groups of DoFs stored as dense matrices
   inline void TransformPrimalCols(DenseMatrix &V) const
   {
      if (IsIdentity()) { return; }
      for (int c=0; c<V.Width(); c++)
      {
         TransformPrimal(V.GetColumn(c));
      }
   }

   /** Inverse transform local DoFs. Used to transform DoFs from a global vector
       back to their element-local form. For example, this must be used to
       transform the vector obtained using GridFunction::GetSubVector before it
       can be used to compute a local interpolation.
   */
   void InvTransformPrimal(real_t *v) const;
   inline void InvTransformPrimal(Vector &v) const
   { InvTransformPrimal(v.GetData()); }

   /** Transform dual DoFs as computed by a LinearFormIntegrator before summing
       into a LinearForm object. */
   void TransformDual(real_t *v) const;
   inline void TransformDual(Vector &v) const
   { TransformDual(v.GetData()); }

   /** Inverse Transform dual DoFs */
   void InvTransformDual(real_t *v) const;
   inline void InvTransformDual(Vector &v) const
   { InvTransformDual(v.GetData()); }

   /** Transform a matrix of dual DoFs entries as computed by a
       BilinearFormIntegrator before summing into a BilinearForm object. */
   inline void TransformDual(DenseMatrix &V) const
   {
      TransformDualCols(V);
      TransformDualRows(V);
   }

   /// Transform rows of a dense matrix containing dual DoFs
   inline void TransformDualRows(DenseMatrix &V) const
   {
      if (IsIdentity()) { return; }
      Vector row;
      for (int r=0; r<V.Height(); r++)
      {
         V.GetRow(r, row);
         TransformDual(row);
         V.SetRow(r, row);
      }
   }

   /// Transform columns of a dense matrix containing dual DoFs
   inline void TransformDualCols(DenseMatrix &V) const
   {
      if (IsIdentity()) { return; }
      for (int c=0; c<V.Width(); c++)
      {
         TransformDual(V.GetColumn(c));
      }
   }
};

/** Transform a matrix of DoFs entries from different finite element spaces as
    computed by a DiscreteInterpolator before copying into a
    DiscreteLinearOperator.
*/
void TransformPrimal(const DofTransformation &ran_dof_trans,
                     const DofTransformation &dom_dof_trans,
                     DenseMatrix &elmat);

/** Transform a matrix of dual DoFs entries from different finite element spaces
    as computed by a BilinearFormIntegrator before summing into a
    MixedBilinearForm object.
*/
void TransformDual(const DofTransformation &ran_dof_trans,
                   const DofTransformation &dom_dof_trans,
                   DenseMatrix &elmat);

/** Abstract base class for high-order Nedelec spaces on elements with
    triangular faces.

    The Nedelec DoFs on the interior of triangular faces come in pairs which
    share an interpolation point but have different vector directions. These
    directions depend on the orientation of the face and can therefore differ in
    neighboring elements. The mapping required to transform these DoFs can be
    implemented as series of 2x2 linear transformations. The raw data for these
    linear transformations is stored in the T_data and TInv_data arrays and can
    be accessed as DenseMatrices using the GetFaceTransform() and
    GetFaceInverseTransform() methods.
*/
class ND_DofTransformation : public StatelessDofTransformation
{
private:
   static const real_t T_data[24];
   static const real_t TInv_data[24];
   static const DenseTensor T, TInv;

protected:
   const int  order;  // basis function order
   const int  nedofs; // number of DoFs per edge
   const int  ntdofs; // number of DoFs per triangular face
   const int  nqdofs; // number of DoFs per quadrilateral face
   const int  nedges; // number of edges per element
   const int  nfaces; // number of faces per element
   const int *ftypes; // Pointer to array of Geometry::Type for each face

   ND_DofTransformation(int size, int order, int num_edges, int num_faces,
                        int *face_types);

public:
   // Return the 2x2 transformation operator for the given face orientation
   static const DenseMatrix & GetFaceTransform(int ori) { return T(ori); }

   // Return the 2x2 inverse transformation operator
   static const DenseMatrix & GetFaceInverseTransform(int ori)
   { return TInv(ori); }

   bool IsIdentity() const override { return ntdofs < 2; }

   void TransformPrimal(const Array<int> & Fo, real_t *v) const override;
   void InvTransformPrimal(const Array<int> & Fo, real_t *v) const override;
   void TransformDual(const Array<int> & Fo, real_t *v) const override;
   void InvTransformDual(const Array<int> & Fo, real_t *v) const override;
};

/// Stateless DoF transformation implementation for the Nedelec basis on
/// triangles
class ND_TriDofTransformation : public ND_DofTransformation
{
private:
   const int face_type[1] = { Geometry::TRIANGLE };
public:
   ND_TriDofTransformation(int order)
      : ND_DofTransformation(order*(order + 2), order, 3, 1, (int *)face_type)
   {}
};

/// DoF transformation implementation for the Nedelec basis on tetrahedra
class ND_TetDofTransformation : public ND_DofTransformation
{
public:
   ND_TetDofTransformation(int order)
      : ND_DofTransformation(order*(order + 2)*(order + 3)/2, order, 6, 4,
                             (int *)Geometry::Constants<Geometry::TETRAHEDRON>::
                             FaceTypes)
   {}
};

/// DoF transformation implementation for the Nedelec basis on wedge elements
class ND_WedgeDofTransformation : public ND_DofTransformation
{
public:
   ND_WedgeDofTransformation(int order)
      : ND_DofTransformation(3 * order * ((order + 1) * (order + 2))/2,
                             order, 9, 5,
                             (int *)Geometry::Constants<Geometry::PRISM>::
                             FaceTypes)
   {}
};

/// DoF transformation implementation for the Nedelec basis on pyramid elements
class ND_PyramidDofTransformation : public ND_DofTransformation
{
public:
   ND_PyramidDofTransformation(int order)
      : ND_DofTransformation(2 * order * (order * (order + 1) + 2),
                             order, 8, 5,
                             (int *)Geometry::Constants<Geometry::PYRAMID>::
                             FaceTypes)
   {}
};

} // namespace mfem

#endif // MFEM_DOFTRANSFORM
