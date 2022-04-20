// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_QFUNCTION
#define MFEM_QFUNCTION

#include "../config/config.hpp"
#include "qspace.hpp"
#include "gridfunc.hpp"

namespace mfem
{

class QuadratureFunctionBase : public Vector
{
protected:
   QuadratureSpaceBase *qspace; ///< Associated QuadratureSpaceBase object
   bool own_qspace;             ///< QuadratureSpaceBase ownership flag
   int vdim;

   QuadratureFunctionBase() : qspace(nullptr), own_qspace(false), vdim(0) { }

   QuadratureFunctionBase(QuadratureSpaceBase *qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_->GetSize()),
        qspace(qspace_), own_qspace(false), vdim(vdim_)
   { }

public:
   /// Get the vector dimension.
   int GetVDim() const { return vdim; }

   /// Set the vector dimension, updating the size by calling Vector::SetSize().
   void SetVDim(int vdim_)
   { vdim = vdim_; SetSize(vdim*qspace->GetSize()); }

   /// Get the associated QuadratureSpaceBase object.
   QuadratureSpaceBase *GetSpace() { return qspace; }

   /// Get the associated QuadratureSpaceBase object (const version).
   const QuadratureSpaceBase *GetSpace() const { return qspace; }

   /// Get the QuadratureSpaceBase ownership flag.
   bool OwnsSpace() { return own_qspace; }

   /// Set the QuadratureSpaceBase ownership flag.
   void SetOwnsSpace(bool own) { own_qspace = own; }

   /// Set this equal to a constant value.
   QuadratureFunctionBase &operator=(double value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       QuadratureSpaceBase #qspace times the QuadratureFunctionBase vector
       dimension i.e. QuadratureFunction::Size(). */
   QuadratureFunctionBase &operator=(const Vector &v);

   /// Evaluate the given coefficient at each quadrature point.
   virtual void ProjectCoefficient(Coefficient &coeff) = 0;

   /// Evaluate the given vector coefficient at each quadrature point.
   virtual void ProjectCoefficient(VectorCoefficient &coeff) = 0;

   /// @brief Evaluate the given symmetric matrix coefficient at each
   /// quadrature point, and store the values in "symmetric format".
   ///
   /// @sa DenseSymmetricMatrix
   virtual void ProjectSymmetricCoefficient(SymmetricMatrixCoefficient &coeff) = 0;

   /// @brief Evaluate the given matrix coefficient at each quadrature point.
   ///
   /// @note The coefficient is stored as a full (non-symmetric) matrix.
   /// @sa ProjectSymmetricCoefficient.
   virtual void ProjectCoefficient(MatrixCoefficient &coeff, bool transpose=false) = 0;

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a reference to the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetValues(int idx, Vector &values);

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a copy of the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetValues(int idx, Vector &values) const;

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a reference to the
       global values. */
   inline void GetValues(int idx, const int ip_num, Vector &values);

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a copy to the
       global values. */
   inline void GetValues(int idx, const int ip_num, Vector &values) const;

   /// Return all values associated with mesh element @a idx in a DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a reference to the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetValues(int idx, DenseMatrix &values);

   /// Return all values associated with mesh element @a idx in a const DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a copy of the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   inline void GetValues(int idx, DenseMatrix &values) const;

   virtual ~QuadratureFunctionBase()
   {
      if (own_qspace) { delete qspace; }
   }
};

/** @brief Class representing a function through its values (scalar or vector)
    at quadrature points. */
class QuadratureFunction : public QuadratureFunctionBase
{
public:
   /// Create an empty QuadratureFunction.
   /** The object can be initialized later using the SetSpace() methods. */
   QuadratureFunction() { }

   /** @brief Copy constructor. The QuadratureSpace ownership flag, #own_qspace,
       in the new object is set to false. */
   QuadratureFunction(const QuadratureFunction &orig)
      : QuadratureFunctionBase(orig.qspace, orig.vdim)
   {
      Vector::operator=(orig);
   }

   /// Create a QuadratureFunction based on the given QuadratureSpace.
   /** The QuadratureFunction does not assume ownership of the QuadratureSpace.
       @note The Vector data is not initialized. */
   QuadratureFunction(QuadratureSpace *qspace_, int vdim_ = 1)
      : QuadratureFunctionBase(qspace_, vdim_) { }

   /// Read a QuadratureFunction from the stream @a in.
   /** The QuadratureFunction assumes ownership of the read QuadratureSpace. */
   QuadratureFunction(Mesh *mesh, std::istream &in);

   /// Get the associated QuadratureSpace.
   QuadratureSpace *GetSpace() const
   { return static_cast<QuadratureSpace*>(qspace); }

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

   /// Evaluate a grid function at each quadrature point.
   void ProjectGridFunction(const GridFunction &gf);

   /// Evaluate the given coefficient at each quadrature point.
   void ProjectCoefficient(Coefficient &coeff);

   /// Evaluate the given vector coefficient at each quadrature point.
   void ProjectCoefficient(VectorCoefficient &coeff);

   /// @brief Evaluate the given symmetric matrix coefficient at each
   /// quadrature point, and store the values in "symmetric format".
   ///
   /// @sa DenseSymmetricMatrix
   void ProjectSymmetricCoefficient(SymmetricMatrixCoefficient &coeff);

   /// @brief Evaluate the given matrix coefficient at each quadrature point.
   ///
   /// @note The coefficient is stored as a full (non-symmetric) matrix.
   /// @sa ProjectSymmetricCoefficient.
   void ProjectCoefficient(MatrixCoefficient &coeff, bool transpose=false);

   /// Get the IntegrationRule associated with mesh element @a idx.
   const IntegrationRule &GetElementIntRule(int idx) const
   { return GetSpace()->GetElementIntRule(idx); }

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a reference to the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   void GetElementValues(int idx, Vector &values)
   { GetValues(idx, values); }

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a copy of the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   void GetElementValues(int idx, Vector &values) const
   { GetValues(idx, values); }

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a reference to the
       global values. */
   void GetElementValues(int idx, const int ip_num, Vector &values)
   { GetValues(idx, ip_num, values); }

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a copy to the
       global values. */
   void GetElementValues(int idx, const int ip_num, Vector &values) const
   { GetValues(idx, ip_num, values); }

   /// Return all values associated with mesh element @a idx in a DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a reference to the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   void GetElementValues(int idx, DenseMatrix &values)
   { GetValues(idx, values); }

   /// Return all values associated with mesh element @a idx in a const DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a copy of the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   void GetElementValues(int idx, DenseMatrix &values) const
   { GetValues(idx, values); }

   /// Write the QuadratureFunction to the stream @a out.
   void Save(std::ostream &out) const;

   /// @brief Write the QuadratureFunction to @a out in VTU (ParaView) format.
   ///
   /// The data will be uncompressed if @a compression_level is zero, or if the
   /// format is VTKFormat::ASCII. Otherwise, zlib compression will be used for
   /// binary data.
   void SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0) const;

   /// @brief Save the QuadratureFunction to a VTU (ParaView) file.
   ///
   /// The extension ".vtu" will be appended to @a filename.
   /// @sa SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
   ///             int compression_level=0)
   void SaveVTU(const std::string &filename, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0) const;
};

class FaceQuadratureFunction : public QuadratureFunctionBase
{
public:
   /** @brief Copy constructor. The FaceQuadratureSpace ownership flag, #own_qspace,
       in the new object is set to false. */
   FaceQuadratureFunction(const FaceQuadratureFunction &orig)
      : QuadratureFunctionBase(orig.qspace, orig.vdim)
   {
      Vector::operator=(orig);
   }

   /// Create a FaceQuadratureFunction based on the given FaceQuadratureSpace.
   /** The FaceQuadratureFunction does not assume ownership of the
       FaceQuadratureSpace.
       @note The Vector data is not initialized. */
   FaceQuadratureFunction(FaceQuadratureSpace *qspace_, int vdim_ = 1)
      : QuadratureFunctionBase(qspace_, vdim_) { }

   /// Get the associated FaceQuadratureSpace.
   FaceQuadratureSpace *GetSpace() const
   { return static_cast<FaceQuadratureSpace*>(qspace); }

   /// Change the FaceQuadratureSpace and optionally the vector dimension.
   /** If the new FaceQuadratureSpace is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data size is updated by calling Vector::SetSize(). */
   inline void SetSpace(FaceQuadratureSpace *qspace_, int vdim_ = -1);

   /// Evaluate the given coefficient at each quadrature point.
   void ProjectCoefficient(Coefficient &coeff);

   /// Evaluate the given vector coefficient at each quadrature point.
   void ProjectCoefficient(VectorCoefficient &coeff);

   /// @brief Evaluate the given symmetric matrix coefficient at each
   /// quadrature point, and store the values in "symmetric format".
   ///
   /// @sa DenseSymmetricMatrix
   void ProjectSymmetricCoefficient(SymmetricMatrixCoefficient &coeff);

   /// @brief Evaluate the given matrix coefficient at each quadrature point.
   ///
   /// @note The coefficient is stored as a full (non-symmetric) matrix.
   /// @sa ProjectSymmetricCoefficient.
   void ProjectCoefficient(MatrixCoefficient &coeff, bool transpose=false);

   /// Get the IntegrationRule associated with mesh face @a idx.
   const IntegrationRule &GetFaceIntRule(int idx) const
   { return GetSpace()->GetFaceIntRule(idx); }

   /// Return all values associated with mesh face @a idx in a Vector.
   void GetFaceValues(int idx, Vector &values)
   { GetValues(idx, values); }

   /// Return all values associated with mesh face @a idx in a Vector.
   void GetFaceValues(int idx, Vector &values) const
   { GetValues(idx, values); }

   /// Return the quadrature function values at an integration point.
   void GetFaceValues(int idx, const int ip_num, Vector &values)
   { GetValues(idx, ip_num, values); }

   /// Return the quadrature function values at an integration point.
   void GetFaceValues(int idx, const int ip_num, Vector &values) const
   { GetValues(idx, ip_num, values); }

   /// Return all values associated with mesh face @a idx in a DenseMatrix.
   void GetFaceValues(int idx, DenseMatrix &values)
   { GetValues(idx, values); }

   /// Return all values associated with mesh face @a idx in a const DenseMatrix.
   void GetFaceValues(int idx, DenseMatrix &values) const
   { GetValues(idx, values); }
};

// Inline methods

inline void QuadratureFunctionBase::GetValues(
   int idx, Vector &values)
{
   const int s_offset = qspace->offsets[idx];
   const int sl_size = qspace->offsets[idx+1] - s_offset;
   values.NewDataAndSize(data + vdim*s_offset, vdim*sl_size);
}

inline void QuadratureFunctionBase::GetValues(
   int idx, Vector &values) const
{
   const int s_offset = qspace->offsets[idx];
   const int sl_size = qspace->offsets[idx+1] - s_offset;
   values.SetSize(vdim*sl_size);
   const double *q = data + vdim*s_offset;
   for (int i = 0; i<values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunctionBase::GetValues(
   int idx, const int ip_num, Vector &values)
{
   const int s_offset = qspace->offsets[idx] * vdim + ip_num * vdim;
   values.NewDataAndSize(data + s_offset, vdim);
}

inline void QuadratureFunctionBase::GetValues(
   int idx, const int ip_num, Vector &values) const
{
   const int s_offset = qspace->offsets[idx] * vdim + ip_num * vdim;
   values.SetSize(vdim);
   const double *q = data + s_offset;
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunctionBase::GetValues(
   int idx, DenseMatrix &values)
{
   const int s_offset = qspace->offsets[idx];
   const int sl_size = qspace->offsets[idx+1] - s_offset;
   values.Reset(data + vdim*s_offset, vdim, sl_size);
}

inline void QuadratureFunctionBase::GetValues(
   int idx, DenseMatrix &values) const
{
   const int s_offset = qspace->offsets[idx];
   const int sl_size = qspace->offsets[idx+1] - s_offset;
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
