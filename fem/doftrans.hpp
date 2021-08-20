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

#ifndef MFEM_DOFTRANSFORM
#define MFEM_DOFTRANSFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "fe.hpp"

namespace mfem
{

/** The DofTransformation class is an abstract base class for a family of
    transformations that map local degrees of freedom (DoFs), contained within
    individual elements, to global degrees of freedom, stored within
    GridFunction objects. These transformations are necessary to ensure that
    basis functions in neighboring elements align correctly. Closely related but
    complementary transformations are required for the entries stored in
    LinearForm and BilinearForm objects. The DofTransformation class is designed
    to apply the action of both of these types of DoF transformations.

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
class DofTransformation
{
protected:
   int size_;

   Array<int> Fo;

   DofTransformation(int size)
      : size_(size) {}

public:

   inline int Size() const { return size_; }
   inline int Height() const { return size_; }
   inline int NumRows() const { return size_; }
   inline int Width() const { return size_; }
   inline int NumCols() const { return size_; }

   /** @brief Configure the transformation using face orientations for the
       current element. */
   /// The face_orientation array can be obtained from Mesh::GetElementFaces.
   inline void SetFaceOrientations(const Array<int> & face_orientation)
   { Fo = face_orientation; }

   inline const Array<int> & GetFaceOrientations() const { return Fo; }

   /** Transform local DoFs to align with the global DoFs. For example, this
       transformation can be used to map the local vector computed by
       FiniteElement::Project() to the transformed vector stored within a
       GridFunction object. */
   virtual void TransformPrimal(double *v) const = 0;
   virtual void TransformPrimal(Vector &v) const;

   /// Transform groups of DoFs stored as dense matrices
   virtual void TransformPrimalCols(DenseMatrix &V) const;

   /** Inverse transform local DoFs. Used to transform DoFs from a global vector
       back to their element-local form. For example, this must be used to
       transform the vector obtained using GridFunction::GetSubVector before it
       can be used to compute a local interpolation.
   */
   virtual void InvTransformPrimal(double *v) const = 0;
   virtual void InvTransformPrimal(Vector &v) const;

   /** Transform dual DoFs as computed by a LinearFormIntegrator before summing
       into a LinearForm object. */
   virtual void TransformDual(double *v) const = 0;
   virtual void TransformDual(Vector &v) const;

   /** Transform a matrix of dual DoFs entries as computed by a
       BilinearFormIntegrator before summing into a BilinearForm object. */
   virtual void TransformDual(DenseMatrix &V) const;

   /// Transform groups of dual DoFs stored as dense matrices
   virtual void TransformDualRows(DenseMatrix &V) const;
   virtual void TransformDualCols(DenseMatrix &V) const;

   virtual ~DofTransformation() {}
};

/** Transform a matrix of DoFs entries from different finite element spaces as
    computed by a DiscreteInterpolator before copying into a
    DiscreteLinearOperator.
*/
void TransformPrimal(const DofTransformation *ran_dof_trans,
                     const DofTransformation *dom_dof_trans,
                     DenseMatrix &elmat);

/** Transform a matrix of dual DoFs entries from different finite element spaces
    as computed by a BilinearFormIntegrator before summing into a
    MixedBilinearForm object.
*/
void TransformDual(const DofTransformation *ran_dof_trans,
                   const DofTransformation *dom_dof_trans,
                   DenseMatrix &elmat);

/** The VDofTransformation class implements a nested transformation where an
    arbitrary DofTransformation is replicated with a vdim >= 1.
*/
class VDofTransformation : public DofTransformation
{
private:
   int vdim_;
   int ordering_;
   DofTransformation * doftrans_;

public:
   /** @brief Default constructor which requires that SetDofTransformation be
       called before use. */
   VDofTransformation(int vdim = 1, int ordering = 0)
      : DofTransformation(0),
        vdim_(vdim), ordering_(ordering),
        doftrans_(NULL) {}

   /// Constructor with a known DofTransformation
   VDofTransformation(DofTransformation & doftrans, int vdim = 1,
                      int ordering = 0)
      : DofTransformation(vdim * doftrans.Size()),
        vdim_(vdim), ordering_(ordering),
        doftrans_(&doftrans) {}

   /// Set or change the vdim parameter
   inline void SetVDim(int vdim)
   {
      vdim_ = vdim;
      if (doftrans_)
      {
         size_ = vdim_ * doftrans_->Size();
      }
   }

   /// Return the current vdim value
   inline int GetVDim() const { return vdim_; }

   /// Set or change the nested DofTransformation object
   inline void SetDofTransformation(DofTransformation & doftrans)
   {
      size_ = vdim_ * doftrans.Size();
      doftrans_ = &doftrans;
   }

   /// Return the nested DofTransformation object
   inline DofTransformation * GetDofTransformation() const { return doftrans_; }

   inline void SetFaceOrientation(const Array<int> & face_orientation)
   { Fo = face_orientation; doftrans_->SetFaceOrientations(face_orientation); }

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(double *v) const;
   void InvTransformPrimal(double *v) const;
   void TransformDual(double *v) const;
};

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
class ND_DofTransformation : public DofTransformation
{
protected:
   static const double T_data[24];
   static const double TInv_data[24];
   static const DenseTensor T, TInv;
   int order;

   ND_DofTransformation(int size, int order);

public:
   // Return the 2x2 transformation operator for the given face orientation
   static const DenseMatrix & GetFaceTransform(int ori) { return T(ori); }

   // Return the 2x2 inverse transformation operator
   static const DenseMatrix & GetFaceInverseTransform(int ori)
   { return TInv(ori); }
};

/// DoF transformation implementation for the Nedelec basis on triangles
class ND_TriDofTransformation : public ND_DofTransformation
{
public:
   ND_TriDofTransformation(int order);

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(double *v) const;

   void InvTransformPrimal(double *v) const;

   void TransformDual(double *v) const;
};

/// DoF transformation implementation for the Nedelec basis on tetrahedra
class ND_TetDofTransformation : public ND_DofTransformation
{
public:
   ND_TetDofTransformation(int order);

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(double *v) const;

   void InvTransformPrimal(double *v) const;

   void TransformDual(double *v) const;
};

/// DoF transformation implementation for the Nedelec basis on wedge elements
/** TODO: (Under development) */
class ND_WedgeDofTransformation : public ND_DofTransformation
{
public:
   ND_WedgeDofTransformation(int order);

   using DofTransformation::TransformPrimal;
   using DofTransformation::InvTransformPrimal;
   using DofTransformation::TransformDual;

   void TransformPrimal(double *v) const;

   void InvTransformPrimal(double *v) const;

   void TransformDual(double *v) const;
};

} // namespace mfem

#endif // MFEM_DOFTRANSFORM
