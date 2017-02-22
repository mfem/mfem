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
#  ifndef MFEM_OVECTOR
#  define MFEM_OVECTOR

#include "stdint.h"

#include "vector.hpp"

#include "occa.hpp"
#include "occa/array/linalg.hpp"
#include "occa/tools/sys.hpp"

namespace mfem
{
  /// Vector data type.
  class OccaVector
  {
  protected:
    uint64_t size_;
    occa::memory data;

  public:
    /// Default constructor for Vector. Sets size = 0 and data = NULL.
    inline OccaVector() :
      size_(0) {}

    /// Copy constructor.
    OccaVector(const OccaVector &other);

    /// @brief Creates vector of size s using the current OCCA device
    /// @warning Entries are not initialized to zero!
    OccaVector(const int64_t size);

    /// @brief Creates vector of size s using the passed OCCA device
    /// @warning Entries are not initialized to zero!
    OccaVector(occa::device device, const int64_t size);

    /// Creates vector based on Vector using the current OCCA device
    OccaVector(const Vector &v);

    /// Creates vector based on Vector using the passed OCCA device
    OccaVector(occa::device device, const Vector &v);

    /// @brief Resize the vector to size @a s.
    /** If the new size is less than or equal to Capacity() then the internal
        data array remains the same. Otherwise, the old array is deleted, if
        owned, and a new array of size @a s is allocated without copying the
        previous content of the Vector.
        @warning New entries are not initialized! */
    void SetSize(const int64_t size, const void *src = NULL);
    void SetSize(occa::device device, const int64_t size, const void *src = NULL);

    /// Destroy a vector
    void Destroy();

    /// Returns the size of the vector.
    inline uint64_t Size() const { return size_; }

    /// Return the size of the currently allocated data array.
    /** It is always true that Capacity() >= Size(). */
    inline uint64_t Capacity() const { return data.size() / sizeof(double); }

    inline occa::memory GetData() { return data; }
    inline const occa::memory GetData() const { return data; }

    inline occa::device GetDevice() { return data.getDevice(); }

    /// Return the inner-product.
    double operator * (const OccaVector &v) const;

    /// Redefine '=' for vector = vector.
    OccaVector& operator = (const OccaVector &v);

    /// Redefine '=' for vector = constant.
    OccaVector& operator = (double value);

    OccaVector& operator *= (double value);

    OccaVector& operator /= (double value);

    OccaVector& operator -= (double value);

    OccaVector& operator -= (const OccaVector &v);

    OccaVector& operator += (const OccaVector &v);

    /// (*this) += a * Va
    OccaVector& Add(const double a, const OccaVector &Va);

    /// (*this) = a * x
    OccaVector& Set(const double a, const OccaVector &x);

    /// (*this) = -(*this)
    void Neg();

    /// Swap the contents of two Vectors
    inline void Swap(OccaVector &other)
    { data.swap(other.data); }

    template <class TM>
    occa::memory occafy(const Array<TM> &array) const {
      return data.getDevice().malloc(array.Size() * sizeof(TM),
                                     array.GetData());
    }

    /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
    void median(const OccaVector &lo, const OccaVector &hi);

    void GetSubVector(const Array<int> &dofs, Vector &elemvect) const;
    void GetSubVector(const Array<int> &dofs, double *elem_data) const;
    void GetSubVector(const Array<int> &dofs, OccaVector &elemvect) const;
    void GetSubVector(occa::memory dofs, OccaVector &elemvect) const;

    /// Set the entries listed in `dofs` to the given `value`.
    void SetSubVector(const Array<int> &dofs, const Vector &elemvect);
    void SetSubVector(const Array<int> &dofs, double *elem_data);
    void SetSubVector(const Array<int> &dofs, const OccaVector &elemvect);
    void SetSubVector(occa::memory dofs, const OccaVector &elemvect);
    void SetSubVector(const Array<int> &dofs, const double value);

    /// Set all vector entries NOT in the 'dofs' array to the given 'val'.
    void SetSubVectorComplement(const Array<int> &dofs, const double val);

    /// Prints vector to stream out.
    void Print(std::ostream & out = std::cout, int width = 8) const;

    /// Set random values in the vector.
    // void Randomize(int seed = 0);
    /// Returns the l2 norm of the vector.
    double Norml2() const;
    /// Returns the l_infinity norm of the vector.
    double Normlinf() const;
    /// Returns the l_1 norm of the vector.
    double Norml1() const;
    /// Returns the l_p norm of the vector.
    double Normlp(double p) const;
    /// Returns the maximal element of the vector.
    double Max() const;
    /// Returns the minimal element of the vector.
    double Min() const;
    /// Return the sum of the vector entries
    double Sum() const;
    /// Compute the Euclidean distance to another vector.
    double DistanceTo(const OccaVector &other) const;

    /// Destroys vector.
    inline virtual ~OccaVector()
    { size_ = 0; data.free(); }
  };

  occa::kernelBuilder makeCustomBuilder(const std::string &kernelName,
                                        const std::string &formula,
                                        occa::properties props = occa::properties());

  ///---[ Addition ]------------------
  /// Set out = v1 + v2.
  void add(const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out);

  /// Set out = v1 + alpha * v2.
  void add(const OccaVector &v1,
           const double alpha,
           const OccaVector &v2,
           OccaVector &out);

  /// out = alpha * (v1 + v2)
  void add(const double alpha,
           const OccaVector &v1,
           const OccaVector &v2,
           OccaVector &out);

  /// out = alpha * v1 + beta * v2
  void add(const double alpha,
           const OccaVector &v1,
           const double beta,
           const OccaVector &v2,
           OccaVector &out);
  ///=================================

  ///---[ Subtraction ]---------------
  /// Set out = v1 - v2.
  void subtract(const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out);

  /// Set out = v1 - alpha * v2.
  void subtract(const OccaVector &v1,
                const double alpha,
                const OccaVector &v2,
                OccaVector &out);

  /// out = alpha * (v1 - v2)
  void subtract(const double alpha,
                const OccaVector &v1,
                const OccaVector &v2,
                OccaVector &out);

  /// out = alpha * v1 - beta * v2
  void subtract(const double alpha,
                const OccaVector &v1,
                const double beta,
                const OccaVector &v2,
                OccaVector &out);
  ///=================================
}

#  endif
#endif
