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
#include "pointer_utils.hpp"
#include <memory>

namespace mfem
{

/// Represents values or vectors of values at quadrature points on a mesh.
class QuadratureFunction : public Vector
{
protected:
   std::shared_ptr<QuadratureSpaceBase> qspace; ///< Associated QuadratureSpaceBase object.
   int vdim; ///< Vector dimension.

public:
   /// Default constructor, results in an empty vector.
   QuadratureFunction() : qspace(nullptr), vdim(0)
   { UseDevice(true); }

   /// Create a QuadratureFunction based on the given QuadratureSpaceBase shared_ptr.
   /** @note The Vector data is not initialized. */
   QuadratureFunction(std::shared_ptr<QuadratureSpaceBase> qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_->GetSize()),
        qspace(std::move(qspace_)), vdim(vdim_)
   { UseDevice(true); }

   /// Create a QuadratureFunction based on the given QuadratureSpaceBase raw pointer (deprecated).
   /** @note The Vector data is not initialized. */
   [[deprecated("Use constructor with std::shared_ptr<QuadratureSpaceBase> instead")]]
   QuadratureFunction(QuadratureSpaceBase* qspace_, int vdim_ = 1)
      : Vector(vdim_*qspace_->GetSize()),
        qspace(ptr_utils::borrow_ptr(qspace_)), vdim(vdim_)
   { UseDevice(true); }

   /** @brief Create a QuadratureFunction based on the given QuadratureSpaceBase shared_ptr,
       using the external (host) data, @a qf_data. */
   /** @note @a qf_data must be a valid **host** pointer (see the constructor
       Vector::Vector(double *, int)). */
   QuadratureFunction(std::shared_ptr<QuadratureSpaceBase> qspace_, real_t *qf_data, int vdim_ = 1)
      : Vector(qf_data, vdim_*qspace_->GetSize()),
        qspace(std::move(qspace_)), vdim(vdim_) { UseDevice(true); }

   /** @brief Create a QuadratureFunction based on the given QuadratureSpaceBase raw pointer,
       using the external (host) data, @a qf_data (deprecated). */
   /** @note @a qf_data must be a valid **host** pointer. */
   [[deprecated("Use constructor with std::shared_ptr<QuadratureSpaceBase> instead")]]
   QuadratureFunction(QuadratureSpaceBase* qspace_, real_t *qf_data, int vdim_ = 1)
      : Vector(qf_data, vdim_*qspace_->GetSize()),
        qspace(ptr_utils::borrow_ptr(qspace_)), vdim(vdim_) { UseDevice(true); }

   /** @brief Copy constructor that shares the QuadratureSpace. */
   QuadratureFunction(const QuadratureFunction &orig)
      : Vector(orig.Size()), qspace(orig.qspace), vdim(orig.vdim)
   {
      UseDevice(true);
      Vector::operator=(orig);
   }

   /// Read a QuadratureFunction from the stream @a in using a Mesh shared_ptr.
   QuadratureFunction(std::shared_ptr<Mesh> mesh, std::istream &in);

   /// Read a QuadratureFunction from the stream @a in using a raw Mesh pointer (deprecated).
   [[deprecated("Use constructor with std::shared_ptr<Mesh> instead")]]
   QuadratureFunction(Mesh* mesh, std::istream &in) 
      : QuadratureFunction(ptr_utils::borrow_ptr(mesh), in) { }

   /// Get the vector dimension.
   [[nodiscard]] int GetVDim() const { return vdim; }

   /// Set the vector dimension, updating the size by calling Vector::SetSize().
   void SetVDim(int vdim_)
   { vdim = vdim_; SetSize(vdim*qspace->GetSize()); }

   /// Get the associated QuadratureSpaceBase object as shared_ptr.
   [[nodiscard]] std::shared_ptr<QuadratureSpaceBase> GetSpaceShared() { return qspace; }

   /// Get the associated QuadratureSpaceBase object as shared_ptr (const version).
   [[nodiscard]] const std::shared_ptr<QuadratureSpaceBase>& GetSpaceShared() const { return qspace; }

   /// Get the associated QuadratureSpaceBase object as raw pointer (deprecated).
   [[deprecated("Use GetSpaceShared() instead")]]
   [[nodiscard]] QuadratureSpaceBase* GetSpace() { return qspace.get(); }

   /// Get the associated QuadratureSpaceBase object as raw pointer (const version, deprecated).
   [[deprecated("Use GetSpaceShared() instead")]]
   [[nodiscard]] const QuadratureSpaceBase* GetSpace() const { return qspace.get(); }

   /// Change the QuadratureSpaceBase using shared_ptr and optionally the vector dimension.
   virtual void SetSpace(std::shared_ptr<QuadratureSpaceBase> qspace_, int vdim_ = -1);

   /// Change the QuadratureSpaceBase using raw pointer and optionally the vector dimension (deprecated).
   [[deprecated("Use SetSpace() with std::shared_ptr<QuadratureSpaceBase> instead")]]
   virtual void SetSpace(QuadratureSpaceBase* qspace_, int vdim_ = -1);

   /// Change the QuadratureSpaceBase (shared_ptr), the data array, and optionally the vector dimension.
   virtual void SetSpace(std::shared_ptr<QuadratureSpaceBase> qspace_, real_t *qf_data, int vdim_ = -1);

   /// Change the QuadratureSpaceBase (raw pointer), the data array, and optionally the vector dimension (deprecated).
   [[deprecated("Use SetSpace() with std::shared_ptr<QuadratureSpaceBase> instead")]]
   virtual void SetSpace(QuadratureSpaceBase* qspace_, real_t *qf_data, int vdim_ = -1);

   /// Set this equal to a constant value.
   virtual QuadratureFunction &operator=(real_t value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       QuadratureSpaceBase #qspace times the QuadratureFunction vector
       dimension i.e. QuadratureFunction::Size(). */
   virtual QuadratureFunction &operator=(const Vector &v);

   /// Evaluate a grid function at each quadrature point.
   virtual void ProjectGridFunction(const GridFunction &gf);

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a reference to the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   virtual void GetValues(int idx, Vector &values);

   /// Return all values associated with mesh element @a idx in a Vector.
   /** The result is stored in the Vector @a values as a copy of the
       global values.

       Inside the Vector @a values, the index `i+vdim*j` corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   virtual void GetValues(int idx, Vector &values) const;

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a reference to the
       global values. */
   virtual void GetValues(int idx, const int ip_num, Vector &values);

   /// Return the quadrature function values at an integration point.
   /** The result is stored in the Vector @a values as a copy to the
       global values. */
   virtual void GetValues(int idx, const int ip_num, Vector &values) const;

   /// Return all values associated with mesh element @a idx in a DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a reference to the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   virtual void GetValues(int idx, DenseMatrix &values);

   /// Return all values associated with mesh element @a idx in a const DenseMatrix.
   /** The result is stored in the DenseMatrix @a values as a copy of the
       global values.

       Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
       `i`-th vector component at the `j`-th quadrature point.
    */
   virtual void GetValues(int idx, DenseMatrix &values) const;

   /// Get the IntegrationRule associated with entity (element or face) @a idx.
   [[nodiscard]] virtual const IntegrationRule &GetIntRule(int idx) const
   { return GetSpaceShared()->GetIntRule(idx); }

   /// Write the QuadratureFunction to the stream @a out.
   virtual void Save(std::ostream &out) const;

   /// @brief Write the QuadratureFunction to @a out in VTU (ParaView) format.
   ///
   /// The data will be uncompressed if @a compression_level is zero, or if the
   /// format is VTKFormat::ASCII. Otherwise, zlib compression will be used for
   /// binary data.
   virtual void SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const;

   /// @brief Save the QuadratureFunction to a VTU (ParaView) file.
   ///
   /// The extension ".vtu" will be appended to @a filename.
   /// @sa SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
   ///             int compression_level=0)
   virtual void SaveVTU(const std::string &filename, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const;

   /// Return the integral of the quadrature function (vdim = 1 only).
   [[nodiscard]] virtual real_t Integrate() const;

   /// @brief Integrate the (potentially vector-valued) quadrature function,
   /// storing the results in @a integrals (length @a vdim).
   virtual void Integrate(Vector &integrals) const;

   // Factory methods for creating QuadratureFunction instances

   /// Create a shared_ptr QuadratureFunction from a QuadratureSpaceBase shared_ptr
   static std::shared_ptr<QuadratureFunction> Create(
      std::shared_ptr<QuadratureSpaceBase> qspace, int vdim = 1) {
      return std::make_shared<QuadratureFunction>(std::move(qspace), vdim);
   }

   /// Create a shared_ptr QuadratureFunction from a raw QuadratureSpaceBase pointer (deprecated)
   [[deprecated("Use Create() with std::shared_ptr<QuadratureSpaceBase> instead")]]
   static std::shared_ptr<QuadratureFunction> Create(QuadratureSpaceBase* qspace, int vdim = 1) {
      return std::make_shared<QuadratureFunction>(ptr_utils::borrow_ptr(qspace), vdim);
   }

   /// Create with explicit ownership semantics (deprecated)
   [[deprecated("Use Create() with std::shared_ptr<QuadratureSpaceBase> instead")]]
   static std::shared_ptr<QuadratureFunction> CreateWithOwnership(
      QuadratureSpaceBase* qspace, int vdim = 1, bool take_ownership = false) {
      if (take_ownership) {
         return std::make_shared<QuadratureFunction>(
            std::shared_ptr<QuadratureSpaceBase>(qspace), vdim);
      } else {
         return std::make_shared<QuadratureFunction>(qspace, vdim);
      }
   }

   virtual ~QuadratureFunction() = default;
};

// Inline method implementations

inline void QuadratureFunction::SetSpace(std::shared_ptr<QuadratureSpaceBase> qspace_,
                                         int vdim_)
{
   qspace = std::move(qspace_);
   vdim = (vdim_ < 0) ? vdim : vdim_;
   SetSize(vdim*qspace->GetSize());
}

inline void QuadratureFunction::SetSpace(QuadratureSpaceBase* qspace_, int vdim_)
{
   qspace = ptr_utils::borrow_ptr(qspace_);
   vdim = (vdim_ < 0) ? vdim : vdim_;
   SetSize(vdim*qspace->GetSize());
}

inline void QuadratureFunction::SetSpace(
   std::shared_ptr<QuadratureSpaceBase> qspace_, real_t *qf_data, int vdim_)
{
   qspace = std::move(qspace_);
   vdim = (vdim_ < 0) ? vdim : vdim_;
   NewDataAndSize(qf_data, vdim*qspace->GetSize());
}

inline void QuadratureFunction::SetSpace(
   QuadratureSpaceBase* qspace_, real_t *qf_data, int vdim_)
{
   qspace = ptr_utils::borrow_ptr(qspace_);
   vdim = (vdim_ < 0) ? vdim : vdim_;
   NewDataAndSize(qf_data, vdim*qspace->GetSize());
}

} // namespace mfem

#endif