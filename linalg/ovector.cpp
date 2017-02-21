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
  /// @brief Creates vector of size s using the current OCCA device
  /// @warning Entries are not initialized to zero!
  OccaVector::OccaVector(const int64_t size)
  {
    size_ = size;
    data = occa::currentDevice().malloc(size_ * sizeof(double));
  }

  /// @brief Creates vector of size s using the default OCCA device
  /// @warning Entries are not initialized to zero!
  OccaVector::OccaVector(occa::device device, const int64_t size)
  {
    size_ = size;
    data = device.malloc(size_ * sizeof(double));
  }

  /// Creates vector based on Vector using the current OCCA device
  OccaVector::OccaVector(const Vector &v) {
    size_ = v.Size();
    data = occa::currentDevice().malloc(size_ * sizeof(double), v.GetData());
  }

  /// Creates vector based on Vector using the passed OCCA device
  OccaVector::OccaVector(occa::device device, const Vector &v) {
    size_ = v.Size();
    data = device.malloc(size_ * sizeof(double), v.GetData());
  }

  /// @brief Resize the vector to size @a s.
  /** If the new size is less than or equal to Capacity() then the internal
      data array remains the same. Otherwise, the old array is deleted, if
      owned, and a new array of size @a s is allocated without copying the
      previous content of the Vector.
      @warning New entries are not initialized! */
  void OccaVector::SetSize(const int64_t size)
  {
    if (data.size() < size)
      {
        occa::device device = data.getDevice();
        data.free();
        data = device.malloc(size_ * sizeof(double));
      }
    size_ = size;
  }

  /// Destroy a vector
  void OccaVector::Destroy()
  { size_ = 0; data.free(); }

  double OccaVector::operator * (const OccaVector &v) const {
    return occa::linalg::dot<double, double, double>(data, v.data);
  }

  /// Redefine '=' for vector = vector.
  OccaVector& OccaVector::operator = (const OccaVector &v)
  {
    occa::memcpy(data, v.data, size_ * sizeof(double));
    return *this;
  }

  /// Redefine '=' for vector = constant.
  OccaVector& OccaVector::operator = (double value) {
    occa::linalg::operator_eq(data, value);
    return *this;
  }

  OccaVector& OccaVector::operator *= (double c) {
    occa::linalg::operator_mult_eq(data, c);
    return *this;
  }

  OccaVector& OccaVector::operator /= (double c) {
    occa::linalg::operator_div_eq(data, c);
    return *this;
  }

  OccaVector& OccaVector::operator -= (double c) {
    occa::linalg::operator_sub_eq(data, c);
    return *this;
  }

  OccaVector& OccaVector::operator -= (const OccaVector &v) {
    occa::linalg::operator_sub_eq<double, double>(v.data, data);
    return *this;
  }

  OccaVector& OccaVector::operator += (const OccaVector &v) {
    occa::linalg::operator_plus_eq<double, double>(v.data, data);
    return *this;
  }

  /// (*this) += a * Va
  OccaVector& OccaVector::Add(const double a, const OccaVector &Va) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axpy", "v0[i] += c0 * v1[i]");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), a, data, Va.data);
    return *this;
  }

  /// (*this) = a * x
  OccaVector& OccaVector::Set(const double a, const OccaVector &x) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_ax", "v0[i] = c0 * v1[i]");

    occa::kernel kernel = builder.build(data.getDevice());
    kernel((int) Size(), a, data, x.data);
    return *this;
  }

  /// (*this) = -(*this)
  void OccaVector::Neg() {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_ax", "v0[i] *= -1.0");

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
                        "v0[i] = (val < lo) ? lo : ((hi < val) ? hi : val)");

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
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_get_subvector",
                        "const int dof_i = v2[i];"
                        "v0[i] = dof_i >= 0 ? v1[dof_i] : -v1[-dof_i - 1]");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    const int dofCount = dofs.Size();
    elemvect.SetSize(dofCount);

    occa::memory o_dofs = occafy(dofs);
    kernel(dofCount, elemvect.data, data, o_dofs);
    o_dofs.free();
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
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_set_subvector",
                        "const int dof_i = v2[i];"
                        "if (dof_i >= 0) { v0[dof_i]      += v1[i]; }"
                        "else            { v0[-dof_i - 1] -= v1[i]; }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(dev);

    occa::memory o_dofs = occafy(dofs);
    kernel((int) dofs.Size(), data, elemvect.data, o_dofs);
    o_dofs.free();
  }

  void OccaVector::SetSubVector(const Array<int> &dofs,
                                const double value) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_set_subvector_const",
                        "const int dof_i = v1[i];"
                        "if (dof_i >= 0) { v0[dof_i]      =  c0; }"
                        "else            { v0[-dof_i - 1] = -c0; }");

    occa::device dev = data.getDevice();
    occa::kernel kernel = builder.build(data.getDevice());

    occa::memory o_dofs = occafy(dofs);
    kernel((int) dofs.Size(), value, data, o_dofs);
    o_dofs.free();
  }

  /// Set all vector entries NOT in the 'dofs' array to the given 'val'.
  void OccaVector::SetSubVectorComplement(const Array<int> &dofs, const double val) {
    Vector dofs_vals;
    GetSubVector(dofs, dofs_vals);
    *this = val;
    SetSubVector(dofs, dofs_vals);
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
                                        const std::string &formula) {
    return occa::linalg::customLinearMethod(kernelName, formula,
                                            "defines: {"
                                            "  CTYPE0: 'double',"
                                            "  CTYPE1: 'double',"
                                            "  VTYPE0: 'double',"
                                            "  VTYPE1: 'double',"
                                            "  VTYPE2: 'double',"
                                            "  TILESIZE: 128,"
                                            "}");
  }

  ///---[ Addition ]--------------------
  /// Set out = v1 + v2.
  void add(const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xpy", "v0[i] = v1[i] + v2[i]");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), out.data, v1.data, v2.data);
  }

  /// Set v = v1 + alpha * v2.
  void add(const OccaVector &v1,
           double alpha,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xpay", "v0[i] = v1[i] + (c0 * v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, out.data, v1.data, v2.data);
  }

  /// z = a * (x + y)
  void add(const double alpha,
           const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_amxpy", "v0[i] = c0 * (v1[i] + v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, out.data, v1.data, v2.data);
  }

  /// z = a * x + b * y
  void add(const double alpha,
           const OccaVector &v1,
           const double beta,
           const OccaVector &v2,
           OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axpby", "v0[i] = (c0 * v1[i]) + (c1 * v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, beta, out.data, v1.data, v2.data);
  }
  ///===================================

  ///---[ Subtraction ]-----------------
  /// Set out = v1 - v2.
  void subtract(const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xsy", "v0[i] = v1[i] - v2[i]");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), out.data, v1.data, v2.data);
  }

  /// Set out = v1 - alpha * v2.
  void subtract(const OccaVector &v1,
                double alpha,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_xsay", "v0[i] = v1[i] - (c0 * v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, out.data, v1.data, v2.data);
  }

  /// out = alpha * (v1 - v2)
  void subtract(const double alpha,
                const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_amxsy", "v0[i] = c0 * (v1[i] - v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, out.data, v1.data, v2.data);
  }

  /// out = alpha * v1 - beta * v2
  void subtract(const double alpha,
                const OccaVector &v1,
                const double beta,
                const OccaVector &v2,
                OccaVector &out) {
    static occa::kernelBuilder builder =
      makeCustomBuilder("vector_axsby", "v0[i] = (c0 * v1[i]) - (c1 * v2[i])");

    occa::kernel kernel = builder.build(out.data.getDevice());
    kernel((int) out.Size(), alpha, beta, out.data, v1.data, v2.data);
  }
  ///===================================
}

#endif