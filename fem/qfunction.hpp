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

#ifndef MFEM_QFUNCTION
#define MFEM_QFUNCTION

#include "../config/config.hpp"
#include "qspace.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Represents values or vectors of values at quadrature points on a mesh.
class QuadratureFunction : public Vector
{
protected:
   QuadratureSpaceBase *qspace; ///< Associated QuadratureSpaceBase object.
   bool own_qspace; ///< Does this own the associated QuadratureSpaceBase?
   int vdim; ///< Vector dimension.

public:
   /// Default constructor, results in an empty vector.
   QuadratureFunction() : qspace(nullptr), own_qspace(false), vdim(0)
   { UseDevice(true); }

   /// Create a QuadratureFunction based on the given QuadratureSpaceBase.
   /** The QuadratureFunction does not assume ownership of the
       QuadratureSpaceBase.
       @note The Vector data is not initialized. */
   QuadratureFunction(QuadratureSpaceBase &qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_.GetSize()),
        qspace(&qspace_), own_qspace(false), vdim(vdim_)
   { UseDevice(true); }

   /// Create a QuadratureFunction based on the given QuadratureSpaceBase.
   /** The QuadratureFunction does not assume ownership of the
       QuadratureSpaceBase.
       @warning @a qspace_ may not be NULL. */
   QuadratureFunction(QuadratureSpaceBase *qspace_, int vdim_ = 1)
      : QuadratureFunction(*qspace_, vdim_) { }

   /** @brief Create a QuadratureFunction based on the given QuadratureSpaceBase,
       using the external (host) data, @a qf_data. */
   /** The QuadratureFunction does not assume ownership of the
       QuadratureSpaceBase or the external data.
       @warning @a qspace_ may not be NULL.
       @note @a qf_data must be a valid **host** pointer (see the constructor
       Vector::Vector(double *, int)). */
   QuadratureFunction(QuadratureSpaceBase *qspace_, real_t *qf_data, int vdim_ = 1)
      : Vector(qf_data, vdim_*qspace_->GetSize()),
        qspace(qspace_), own_qspace(false), vdim(vdim_) { UseDevice(true); }

   /** @brief Copy constructor. The QuadratureSpace ownership flag, #own_qspace,
       in the new object is set to false. */
   QuadratureFunction(const QuadratureFunction &orig)
      : QuadratureFunction(*orig.qspace, orig.vdim)
   {
      Vector::operator=(orig);
   }

   /// Read a QuadratureFunction from the stream @a in.
   /** The QuadratureFunction assumes ownership of the read QuadratureSpace. */
   QuadratureFunction(Mesh *mesh, std::istream &in);

   /// Get the vector dimension.
   int GetVDim() const { return vdim; }

   /// Set the vector dimension, updating the size by calling Vector::SetSize().
   void SetVDim(int vdim_)
   { vdim = vdim_; SetSize(vdim*qspace->GetSize()); }

   /// Get the associated QuadratureSpaceBase object.
   QuadratureSpaceBase *GetSpace() { return qspace; }

   /// Get the associated QuadratureSpaceBase object (const version).
   const QuadratureSpaceBase *GetSpace() const { return qspace; }

   /// Change the QuadratureSpaceBase and optionally the vector dimension.
   /** If the new QuadratureSpaceBase is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data size is updated by calling Vector::SetSize(). */
   inline void SetSpace(QuadratureSpaceBase *qspace_, int vdim_ = -1);

   /** @brief Change the QuadratureSpaceBase, the data array, and optionally the
       vector dimension. */
   /** If the new QuadratureSpaceBase is different from the current one, the
       QuadratureFunction will not assume ownership of the new space; otherwise,
       the ownership flag remains the same.

       If the new vector dimension @a vdim_ < 0, the vector dimension remains
       the same.

       The data array is replaced by calling Vector::NewDataAndSize(). */
   inline void SetSpace(QuadratureSpaceBase *qspace_, real_t *qf_data,
                        int vdim_ = -1);

   /// Get the QuadratureSpaceBase ownership flag.
   bool OwnsSpace() { return own_qspace; }

   /// Set the QuadratureSpaceBase ownership flag.
   void SetOwnsSpace(bool own) { own_qspace = own; }

   /// Set this equal to a constant value.
   QuadratureFunction &operator=(real_t value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       QuadratureSpaceBase #qspace times the QuadratureFunction vector
       dimension i.e. QuadratureFunction::Size(). */
   QuadratureFunction &operator=(const Vector &v);

   /// Evaluate a grid function at each quadrature point.
   void ProjectGridFunction(const GridFunction &gf);

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

   /// Get the IntegrationRule associated with entity (element or face) @a idx.
   const IntegrationRule &GetIntRule(int idx) const
   { return GetSpace()->GetIntRule(idx); }

   /// Write the QuadratureFunction to the stream @a out.
   void Save(std::ostream &out) const;

   /// @brief Write the QuadratureFunction to @a out in VTU (ParaView) format.
   ///
   /// The data will be uncompressed if @a compression_level is zero, or if the
   /// format is VTKFormat::ASCII. Otherwise, zlib compression will be used for
   /// binary data.
   void SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const;

   /// @brief Save the QuadratureFunction to a VTU (ParaView) file.
   ///
   /// The extension ".vtu" will be appended to @a filename.
   /// @sa SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
   ///             int compression_level=0)
   void SaveVTU(const std::string &filename, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const;


   /// Return the integral of the quadrature function (vdim = 1 only).
   real_t Integrate() const;

   /// @brief Integrate the (potentially vector-valued) quadrature function,
   /// storing the results in @a integrals (length @a vdim).
   void Integrate(Vector &integrals) const;

   virtual ~QuadratureFunction()
   {
      if (own_qspace) { delete qspace; }
   }
};

// Inline methods

inline void QuadratureFunction::GetValues(
   int idx, Vector &values)
{
   const int s_offset = qspace->Offset(idx);
   const int sl_size = qspace->Offset(idx + 1) - s_offset;
   values.MakeRef(*this, vdim*s_offset, vdim*sl_size);
}

inline void QuadratureFunction::GetValues(
   int idx, Vector &values) const
{
   const int s_offset = qspace->Offset(idx);
   const int sl_size = qspace->Offset(idx + 1) - s_offset;
   values.SetSize(vdim*sl_size);
   values.HostWrite();
   const real_t *q = HostRead() + vdim*s_offset;
   for (int i = 0; i<values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetValues(
   int idx, const int ip_num, Vector &values)
{
   const int s_offset = qspace->Offset(idx) * vdim + ip_num * vdim;
   values.MakeRef(*this, s_offset, vdim);
}

inline void QuadratureFunction::GetValues(
   int idx, const int ip_num, Vector &values) const
{
   const int s_offset = qspace->Offset(idx) * vdim + ip_num * vdim;
   values.SetSize(vdim);
   values.HostWrite();
   const real_t *q = HostRead() + s_offset;
   for (int i = 0; i < values.Size(); i++)
   {
      values(i) = *(q++);
   }
}

inline void QuadratureFunction::GetValues(
   int idx, DenseMatrix &values)
{
   const int s_offset = qspace->Offset(idx);
   const int sl_size = qspace->Offset(idx + 1) - s_offset;
   // Make the values matrix memory an alias of the quadrature function memory
   Memory<real_t> &values_mem = values.GetMemory();
   values_mem.Delete();
   values_mem.MakeAlias(GetMemory(), vdim*s_offset, vdim*sl_size);
   values.SetSize(vdim, sl_size);
}

inline void QuadratureFunction::GetValues(
   int idx, DenseMatrix &values) const
{
   const int s_offset = qspace->Offset(idx);
   const int sl_size = qspace->Offset(idx + 1) - s_offset;
   values.SetSize(vdim, sl_size);
   values.HostWrite();
   const real_t *q = HostRead() + vdim*s_offset;
   for (int j = 0; j<sl_size; j++)
   {
      for (int i = 0; i<vdim; i++)
      {
         values(i,j) = *(q++);
      }
   }
}


inline void QuadratureFunction::SetSpace(QuadratureSpaceBase *qspace_,
                                         int vdim_)
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

inline void QuadratureFunction::SetSpace(
   QuadratureSpaceBase *qspace_, real_t *qf_data, int vdim_)
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


} // namespace mfem

#endif
