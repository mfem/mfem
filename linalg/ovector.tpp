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

// This file is only to separate declarations and definitions
// ovector.tpp is included at the end of ovector.hpp

namespace mfem {
  /// @brief Creates vector of size s using the current OCCA device
  /// @warning Entries are not initialized to zero!
  template <class TM>
  OccaTVector<TM>::OccaTVector(const int64_t size)
  {
    size_ = size;
    data = occa::currentDevice().malloc(size_ * sizeof(TM));
  }

  /// @brief Creates vector of size s using the default OCCA device
  /// @warning Entries are not initialized to zero!
  template <class TM>
  OccaTVector<TM>::OccaTVector(occa::device device, const int64_t size)
  {
    size_ = size;
    data = device.malloc(size_ * sizeof(TM));
  }

  /// Creates vector based on Vector using the current OCCA device
  template <class TM>
  OccaTVector<TM>::OccaTVector(const Vector &v) {
    size_ = v.Size();
    data = occa::currentDevice().malloc(size_ * sizeof(TM), v.GetData());
  }

  /// Creates vector based on Vector using the passed OCCA device
  template <class TM>
  OccaTVector<TM>::OccaTVector(occa::device device, const Vector &v) {
    size_ = v.Size();
    data = device.malloc(size_ * sizeof(TM), v.GetData());
  }

  /// @brief Resize the vector to size @a s.
  /** If the new size is less than or equal to Capacity() then the internal
      data array remains the same. Otherwise, the old array is deleted, if
      owned, and a new array of size @a s is allocated without copying the
      previous content of the Vector.
      @warning New entries are not initialized! */
  template <class TM>
  void OccaTVector<TM>::SetSize(const int64_t size)
  {
    if (data.size() < size)
      {
        size_ = size;
        occa::device device = data.getDevice();
        data.free();
        data = device.malloc(size_ * sizeof(TM));
      }
  }

  /// Destroy a vector
  template <class TM>
  void OccaTVector<TM>::Destroy()
  { size_ = 0; data.free(); }

  template <class TM>
  TM OccaTVector<TM>::operator * (const OccaTVector<TM> &v) const {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "dot",
                                      VecOpKernelProps<TM>());
    kernel[v.data.getDevice()](data, v.data);
    return 10;
  }

  /// Redefine '=' for vector = vector.
  template <class TM>
  OccaTVector<TM>& OccaTVector<TM>::operator = (const OccaTVector<TM> &v)
  {
    occa::memcpy(data, v.data);
    return *this;
  }

  /// Redefine '=' for vector = constant.
  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator = (TM value);

  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator *= (double c);

  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator /= (double c);

  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator -= (TM c);

  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator -= (const OccaTVector<TM> &v);

  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::operator += (const OccaTVector<TM> &v);

  /// (*this) += a * Va
  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::Add(const TM a, const OccaTVector<TM> &Va);

  /// (*this) = a * x
  // template <class TM>
  // OccaTVector<TM>& OccaTVector<TM>::Set(const TM a, const OccaTVector<TM> &x);

  /// (*this) = -(*this)
  // template <class TM>
  // void OccaTVector<TM>::Neg();

  /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
  // template <class TM>
  // void OccaTVector<TM>::median(const OccaTVector<TM> &lo, const OccaTVector<TM> &hi);

  /// Prints vector to stream out.
  template <class TM>
  void OccaTVector<TM>::Print(std::ostream & out, int width) const
  {
    Vector v(size_);
    occa::memcpy(v.GetData(), data);
    v.Print(out, width);
  }

  /// Set random values in the vector.
  // template <class TM>
  // void OccaTVector<TM>::Randomize(int seed = 0);

  /// Returns the l2 norm of the vector.
  // template <class TM>
  // double OccaTVector<TM>::Norml2() const;

  /// Returns the l_infinity norm of the vector.
  // template <class TM>
  // double OccaTVector<TM>::Normlinf() const;

  /// Returns the l_1 norm of the vector.
  // template <class TM>
  // double OccaTVector<TM>::Norml1() const;

  /// Returns the l_p norm of the vector.
  // template <class TM>
  // double OccaTVector<TM>::Normlp(double p) const;

  /// Returns the maximal element of the vector.
  // template <class TM>
  // TM OccaTVector<TM>::Max() const;

  /// Returns the minimal element of the vector.
  // template <class TM>
  // TM OccaTVector<TM>::Min() const;

  /// Return the sum of the vector entries
  // template <class TM>
  // TM OccaTVector<TM>::Sum() const;

  /// Compute the Euclidean distance to another vector.
  // template <class TM>
  // double OccaTVector<TM>::DistanceTo(const OccaTVector<TM> &other) const;

  ///---[ Addition ]--------------------
  /// Set out = v1 + v2.
  template <class TM>
  void add(const OccaTVector<TM> &v1,
           const OccaTVector<TM> &v2,
           OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vertorops.okl",
                                      "add_vv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](v1.data, v2.data, out.data);
  }

  /// Set v = v1 + alpha * v2.
  template <class TM>
  void add(const OccaTVector<TM> &v1,
           double alpha,
           const OccaTVector<TM> &v2,
           OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "add_vcv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](v1.data, alpha, v2.data, out.data);
  }

  /// z = a * (x + y)
  template <class TM>
  void add(const double alpha,
           const OccaTVector<TM> &v1,
           const OccaTVector<TM> &v2,
           OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "add_cvv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](alpha, v1.data, v2.data, out.data);
  }

  /// z = a * x + b * y
  template <class TM>
  void add(const double alpha,
           const OccaTVector<TM> &v1,
           const double beta,
           const OccaTVector<TM> &v2,
           OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "add_cvcv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](alpha, v1.data, beta, v2.data, out.data);
  }
  ///===================================

  ///---[ Subtraction ]-----------------
  /// Set out = v1 - v2.
  template <class TM>
  void subtract(const OccaTVector<TM> &v1,
                const OccaTVector<TM> &v2,
                OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "subtract_vv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](v1.data, v2.data, out.data);
  }

  /// Set out = v1 - alpha * v2.
  template <class TM>
  void subtract(const OccaTVector<TM> &v1,
                double alpha,
                const OccaTVector<TM> &v2,
                OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "subtract_vcv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](v1.data, alpha, v2.data, out.data);
  }

  /// out = alpha * (v1 - v2)
  template <class TM>
  void subtract(const double alpha,
                const OccaTVector<TM> &v1,
                const OccaTVector<TM> &v2,
                OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "subtract_cvv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](alpha, v1.data, v2.data, out.data);
  }

  /// out = alpha * v1 - beta * v2
  template <class TM>
  void subtract(const double alpha,
                const OccaTVector<TM> &v1,
                const double beta,
                const OccaTVector<TM> &v2,
                OccaTVector<TM> &out) {
    static occa::kernelBuilder kernel("vectorops.okl",
                                      "subtract_cvcv",
                                      VecOpKernelProps<TM>());
    kernel[out.data.getDevice()](alpha, v1.data, beta, v2.data, out.data);
  }
  ///===================================
}
