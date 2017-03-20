// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA

#include "ovector.hpp"

namespace mfem {
  OccaVector::OccaVector() :
    size_(0),
    ownsData(false) {}

  /// @brief Creates vector of size s using the current OCCA device
  /// @warning Entries are not initialized to zero!
  OccaVector::OccaVector(const int64_t size) {
    SetSize(size);
  }

  /// Copy constructor.
  OccaVector::OccaVector(const OccaVector &other) {
    const int entries = other.Size();
    SetSize(other.data.getDevice(), entries);
    occa::memcpy(data, other.data, entries * sizeof(double));
  }

  /// @brief Creates vector of size s using the default OCCA device
  /// @warning Entries are not initialized to zero!
  OccaVector::OccaVector(occa::device device, const int64_t size) {
    SetSize(device, size);
  }

  /// Creates vector based on Vector using the current OCCA device
  OccaVector::OccaVector(const Vector &v) {
    SetSize(v.Size(), v.GetData());
  }

  /// Creates vector based on Vector using the passed OCCA device
  OccaVector::OccaVector(occa::device device, const Vector &v) {
    SetSize(device, v.Size(), v.GetData());
  }

  /// Convert to Vector
  OccaVector::operator Vector() const {
    Vector v(size_);
    occa::memcpy(v.GetData(), data, size_ * sizeof(double));
    return v;
  }

  /// Reads a vector from multiple files
  void OccaVector::Load(std::istream **in, int np, int *dim) {
    Vector v;
    v.Load(in, np, dim);
    *this = v;
  }

  /// Load a vector from an input stream.
  void OccaVector::Load(std::istream &in, int Size) {
    Vector v;
    v.Load(in, Size);
    *this = v;
  }

  /// @brief Resize the vector to size @a s.
  /** If the new size is less than or equal to Capacity() then the internal
      data array remains the same. Otherwise, the old array is deleted, if
      owned, and a new array of size @a s is allocated without copying the
      previous content of the Vector.
      @warning New entries are not initialized! */
  void OccaVector::SetSize(const int64_t size, const void *src) {
    if (data.isInitialized()) {
      SetSize(data.getDevice(), size, src);
    } else {
      SetSize(occa::currentDevice(), size, src);
    }
  }

  void OccaVector::SetSize(occa::device device, const int64_t size, const void *src) {
    size_ = size;
    if (size > (int64_t) Capacity()) {
      if (ownsData) {
        data.free();
      }
      data = device.malloc(size * sizeof(double), src);
      ownsData = true;
    } else if (size && src) {
      occa::memcpy(data, src, size * sizeof(double));
    }
  }

  double OccaVector::operator * (const OccaVector &v) const {
    return occa::linalg::dot<double, double, double>(data, v.data);
  }

  /// Redefine '=' for vector = vector.
  OccaVector& OccaVector::operator = (const Vector &v) {
    SetSize(v.Size(), v.GetData());
    return *this;
  }

  /// Redefine '=' for vector = vector.
  OccaVector& OccaVector::operator = (const OccaVector &v)
  {
    SetSize(v.Size());
    occa::memcpy(data, v.data, size_ * sizeof(double));
    return *this;
  }

  /// Redefine '=' for vector = constant.
  OccaVector& OccaVector::operator = (double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_op_eq",
                        "v0[i] = c0;");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), value, data);

    return *this;
  }

  OccaVector& OccaVector::operator *= (double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_op_mult",
                        "v0[i] *= c0;");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), value, data);

    return *this;
  }

  OccaVector& OccaVector::operator /= (double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_op_div",
                        "v0[i] /= c0;");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), value, data);

    return *this;
  }

  OccaVector& OccaVector::operator -= (double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_op_sub",
                        "v0[i] -= c0;");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), value, data);

    return *this;
  }

  OccaVector& OccaVector::operator += (double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_op_add",
                        "v0[i] += c0;");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), value, data);

    return *this;
  }

  OccaVector& OccaVector::operator -= (const OccaVector &v) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_vec_sub",
                        "v0[i] -= v1[i];");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), data, v.data);

    return *this;
  }

  OccaVector& OccaVector::operator += (const OccaVector &v) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_vec_add",
                        "v0[i] += v1[i];");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    kernel((int) Size(), data, v.data);

    return *this;
  }

  /// (*this) += a * Va
  OccaVector& OccaVector::Add(const double a, const OccaVector &Va) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axpy",
                        "v0[i] += c0 * v1[i];");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), a, data, Va.data);
    return *this;
  }

  /// (*this) = a * x
  OccaVector& OccaVector::Set(const double a, const OccaVector &x) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_set",
                        "v0[i] = c0 * v1[i];");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), a, data, x.data);
    return *this;
  }

  /// (*this) = -(*this)
  void OccaVector::Neg() {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_neg",
                        "v0[i] *= -1.0;");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), data);
  }

  /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
  void OccaVector::median(const OccaVector &lo, const OccaVector &hi) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_median",
                        "const double val = v0[i];"
                        "const double lo  = v1[i];"
                        "const double hi  = v2[i];"
                        "v0[i] = (val < lo) ? lo : ((hi < val) ? hi : val);");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), data, lo.data, hi.data);
  }

  void OccaVector::GetSubVector(const Array<int> &dofs,
                                Vector &elemvect) const {
    elemvect.SetSize(dofs.Size());
    GetSubVector(dofs, elemvect.GetData());
  }

  void OccaVector::GetSubVector(const Array<int> &dofs,
                                double *elem_data) const {
    const int dofCount = dofs.Size();
    OccaVector buffer(dofCount);
    GetSubVector(dofs, buffer);

    occa::memcpy(elem_data, buffer.data, dofCount * sizeof(double));
    buffer.data.free();
  }

  void OccaVector::GetSubVector(const Array<int> &dofs,
                                OccaVector &elemvect) const {
    occa::memory o_dofs = data.getDevice().malloc(dofs.Size() * sizeof(int),
                                                  dofs.GetData());
    GetSubVector(o_dofs, elemvect);
    o_dofs.free();
  }

  void OccaVector::GetSubVector(occa::memory dofs,
                                OccaVector &elemvect) const {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_get_subvector",
                        "const int dof_i = v2[i];"
                        "v0[i] = dof_i >= 0 ? v1[dof_i] : -v1[-dof_i - 1];",
                        "defines: { VTYPE2: 'int' }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    const int dofCount = dofs.entries<int>();
    elemvect.SetSize(dofCount);

    kernel(dofCount, elemvect.data, data, dofs);
  }

  /// Set the entries listed in `dofs` to the given `value`.

  void OccaVector::SetSubVector(const Array<int> &dofs,
                                const Vector &elemvect) {

    SetSubVector(dofs, elemvect.GetData());
  }

  void OccaVector::SetSubVector(const Array<int> &dofs,
                                double *elem_data) {
    const int dofCount = dofs.Size();

    OccaVector buffer(dofCount);
    occa::memcpy(buffer.data, elem_data, dofCount * sizeof(double));

    SetSubVector(dofs, buffer);

    buffer.data.free();
  }

  void OccaVector::SetSubVector(const Array<int> &dofs,
                                const OccaVector &elemvect) {
    occa::memory o_dofs = data.getDevice().malloc(dofs.Size() * sizeof(int),
                                                  dofs.GetData());
    SetSubVector(o_dofs, elemvect);
    o_dofs.free();
  }

  void OccaVector::SetSubVector(occa::memory dofs,
                                const OccaVector &elemvect) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_set_subvector",
                        "const int dof_i = v2[i];"
                        "if (dof_i >= 0) { v0[dof_i]      = v1[i]; }"
                        "else            { v0[-dof_i - 1] = -v1[i]; }",
                        "defines: { VTYPE2: 'int' }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    const int dofCount = dofs.entries<int>();
    kernel(dofCount, data, elemvect.data, dofs);
  }

  void OccaVector::SetSubVector(const Array<int> &dofs,
                                const double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_set_subvector_const",
                        "const int dof_i = v1[i];"
                        "if (dof_i >= 0) { v0[dof_i]      =  c0; }"
                        "else            { v0[-dof_i - 1] = -c0; }",
                        "defines: { VTYPE1: 'int' }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    occa::memory o_dofs = dev.malloc(dofs.Size() * sizeof(int),
                                     dofs.GetData());
    kernel((int) dofs.Size(), value, data, o_dofs);
    o_dofs.free();
  }

  /// Add (element) subvector to the vector.
  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    const Vector &elemvect) {

    AddElementVector(dofs, elemvect.GetData());
  }

  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    double *elem_data) {
    const int dofCount = dofs.Size();

    OccaVector buffer(dofCount);
    occa::memcpy(buffer.data, elem_data, dofCount * sizeof(double));

    AddElementVector(dofs, buffer);

    buffer.data.free();
  }

  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    const OccaVector &elemvect) {
    occa::memory o_dofs = data.getDevice().malloc(dofs.Size() * sizeof(int),
                                                  dofs.GetData());
    AddElementVector(o_dofs, elemvect);
    o_dofs.free();
  }

  void OccaVector::AddElementVector(occa::memory dofs,
                                    const OccaVector &elemvect) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_add_vec",
                        "const int dof_i = v2[i];"
                        "if (dof_i >= 0) { v0[dof_i]      += v1[i]; }"
                        "else            { v0[-dof_i - 1] -= v1[i]; }",
                        "defines: { VTYPE2: 'int' }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    const int dofCount = dofs.entries<int>();
    kernel(dofCount, data, elemvect.data, dofs);
  }

  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    const double a,
                                    const Vector &elemvect) {

    AddElementVector(dofs, a, elemvect.GetData());
  }

  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    const double a,
                                    double *elem_data) {
    const int dofCount = dofs.Size();

    OccaVector buffer(dofCount);
    occa::memcpy(buffer.data, elem_data, dofCount * sizeof(double));

    AddElementVector(dofs, a, buffer);

    buffer.data.free();
  }

  void OccaVector::AddElementVector(const Array<int> &dofs,
                                    const double a,
                                    const OccaVector &elemvect) {
    occa::memory o_dofs = data.getDevice().malloc(dofs.Size() * sizeof(int),
                                                  dofs.GetData());
    AddElementVector(o_dofs, a, elemvect);
    o_dofs.free();
  }

  void OccaVector::AddElementVector(occa::memory dofs,
                                    const double a,
                                    const OccaVector &elemvect) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_add_a_vec",
                        "const int dof_i = v2[i];"
                        "if (dof_i >= 0) { v0[dof_i]      += c0 * v1[i]; }"
                        "else            { v0[-dof_i - 1] -= c0 * v1[i]; }",
                        "defines: { VTYPE2: 'int' }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    const int dofCount = dofs.entries<int>();
    kernel(dofCount, a, data, elemvect.data, dofs);
  }

  /// Set all vector entries NOT in the 'dofs' array to the given 'val'.
  void OccaVector::SetSubVectorComplement(const Array<int> &dofs, const double val) {
    OccaVector dofs_vals;
    occa::memory o_dofs = data.getDevice().malloc(dofs.Size() * sizeof(int),
                                                  dofs.GetData());

    GetSubVector(o_dofs, dofs_vals);
    *this = val;
    SetSubVector(o_dofs, dofs_vals);

    o_dofs.free();
  }

  /// Prints vector to stream out.
  void OccaVector::Print(std::ostream & out, int width) const
  {
    Vector v(size_);
    occa::memcpy(v.GetData(), data, size_ * sizeof(double));
    v.Print(out, width);
  }

  /// Set random values in the vector.
  // void OccaVector::Randomize(int seed = 0);

  // Returns the l2 norm of the vector.
  double OccaVector::Norml2() const {
    return occa::linalg::l2Norm<double, double>(data);
  }

  // Returns the l_infinity norm of the vector.
  double OccaVector::Normlinf() const {
    return occa::linalg::lInfNorm<double, double>(data);
  }

  // Returns the l_1 norm of the vector.
  double OccaVector::Norml1() const {
    return occa::linalg::l1Norm<double, double>(data);
  }

  // Returns the l_p norm of the vector.
  double OccaVector::Normlp(double p) const {
    return occa::linalg::lpNorm<double, double>(p, data);
  }

  // Returns the maximal element of the vector.
  double OccaVector::Max() const {
    return occa::linalg::max<double, double>(data);
  }

  // Returns the minimal element of the vector.
  double OccaVector::Min() const {
    return occa::linalg::min<double, double>(data);
  }

  // Return the sum of the vector entries
  double OccaVector::Sum() const {
    return occa::linalg::sum<double, double>(data);
  }

  /// Compute the Euclidean distance to another vector.
  double OccaVector::DistanceTo(const OccaVector &other) const {
    return occa::linalg::distance<double, double, double>(data, other.data);
  }

  occa::kernelBuilder makeCustomBuilder(const std::string &kernelName,
                                        const std::string &formula,
                                        occa::properties props) {
    static occa::properties defaults("defines: {"
                                     "  CTYPE0: 'double',"
                                     "  CTYPE1: 'double',"
                                     "  VTYPE0: 'double',"
                                     "  VTYPE1: 'double',"
                                     "  VTYPE2: 'double',"
                                     "  TILESIZE: 128,"
                                     "}");

    return occa::linalg::customLinearMethod(kernelName,
                                            formula,
                                            defaults + props);
  }

  ///---[ Addition ]--------------------
  /// Set out = v1 + v2.
  void add(const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xpy",
                        "v0[i] = v1[i] + v2[i];");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), out.GetData(), v1.GetData(), v2.GetData());
  }

  /// Set v = v1 + alpha * v2.
  void add(const OccaVector &v1,
           const double alpha,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xpay",
                        "v0[i] = v1[i] + (c0 * v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, out.GetData(), v1.GetData(), v2.GetData());
  }

  /// z = a * (x + y)
  void add(const double alpha,
           const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_amxpy",
                        "v0[i] = c0 * (v1[i] + v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, out.GetData(), v1.GetData(), v2.GetData());
  }

  /// z = a * x + b * y
  void add(const double alpha,
           const OccaVector &v1,
           const double beta,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axpby",
                        "v0[i] = (c0 * v1[i]) + (c1 * v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, beta, out.GetData(), v1.GetData(), v2.GetData());
  }
  ///===================================

  ///---[ Subtraction ]-----------------
  /// Set out = v1 - v2.
  void subtract(const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xsy",
                        "v0[i] = v1[i] - v2[i];");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), out.GetData(), v1.GetData(), v2.GetData());
  }

  /// Set out = v1 - alpha * v2.
  void subtract(const OccaVector &v1,
                const double alpha,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xsay",
                        "v0[i] = v1[i] - (c0 * v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, out.GetData(), v1.GetData(), v2.GetData());
  }

  /// out = alpha * (v1 - v2)
  void subtract(const double alpha,
                const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_amxsy",
                        "v0[i] = c0 * (v1[i] - v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, out.GetData(), v1.GetData(), v2.GetData());
  }

  /// out = alpha * v1 - beta * v2
  void subtract(const double alpha,
                const OccaVector &v1,
                const double beta,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axsby",
                        "v0[i] = (c0 * v1[i]) - (c1 * v2[i]);");

    occa::kernel kernel = builder.build(out.GetDevice());
    kernel((int) out.Size(), alpha, beta, out.GetData(), v1.GetData(), v2.GetData());
  }
  ///===================================
}

#endif