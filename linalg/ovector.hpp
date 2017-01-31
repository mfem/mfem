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
#include "../general/otypes.hpp"

#include "occa.hpp"
#include "occa/tools/sys.hpp"

namespace mfem
{
  /// Vector data type.
  template <class TM>
  class OccaTVector
  {
  protected:
    uint64_t size_;
    occa::memory data;

  public:
    /// Default constructor for Vector. Sets size = 0 and data = NULL.
    inline OccaTVector() {}

    /// Copy constructor.
    /// @warning Uses the same memory
    inline OccaTVector(const OccaTVector<TM> &other) :
      size_(other.size_),
      data(other.data) {}

    /// @brief Creates vector of size s using the current OCCA device
    /// @warning Entries are not initialized to zero!
    OccaTVector(const int64_t size);

    /// @brief Creates vector of size s using the passed OCCA device
    /// @warning Entries are not initialized to zero!
    OccaTVector(occa::device device, const int64_t size);

    /// Creates vector based on Vector using the current OCCA device
    OccaTVector(const Vector &v);

    /// Creates vector based on Vector using the passed OCCA device
    OccaTVector(occa::device device, const Vector &v);

    /// @brief Resize the vector to size @a s.
    /** If the new size is less than or equal to Capacity() then the internal
        data array remains the same. Otherwise, the old array is deleted, if
        owned, and a new array of size @a s is allocated without copying the
        previous content of the Vector.
        @warning New entries are not initialized! */
    void SetSize(const int64_t size);

    /// Destroy a vector
    void Destroy();

    /// Returns the size of the vector.
    inline uint64_t Size() const { return size_; }

    /// Return the size of the currently allocated data array.
    /** It is always true that Capacity() >= Size(). */
    inline uint64_t Capacity() const { return data.size() / sizeof(TM); }

    /// Return the inner-product.
    TM operator * (const OccaTVector<TM> &v) const;

    /// Redefine '=' for vector = vector.
    OccaTVector<TM>& operator = (const OccaTVector<TM> &v);

    /// Redefine '=' for vector = constant.
    //OccaTVector<TM>& operator = (TM value);

    //OccaTVector<TM>& operator *= (double c);

    //OccaTVector<TM>& operator /= (double c);

    //OccaTVector<TM>& operator -= (TM c);

    //OccaTVector<TM>& operator -= (const OccaTVector<TM> &v);

    //OccaTVector<TM>& operator += (const OccaTVector<TM> &v);

    /// (*this) += a * Va
    //OccaTVector<TM>& Add(const TM a, const OccaTVector<TM> &Va);

    /// (*this) = a * x
    //OccaTVector<TM>& Set(const TM a, const OccaTVector<TM> &x);

    /// (*this) = -(*this)
    //void Neg();

    /// Swap the contents of two Vectors
    inline void Swap(OccaTVector<TM> &other)
    { data.swap(other.data); }

    /// v = median(v,lo,hi) entrywise.  Implementation assumes lo <= hi.
    //void median(const OccaTVector<TM> &lo, const OccaTVector<TM> &hi);

    /// Prints vector to stream out.
    void Print(std::ostream & out = std::cout, int width = 8) const;

    /// Set random values in the vector.
    //void Randomize(int seed = 0);
    /// Returns the l2 norm of the vector.
    //double Norml2() const;
    /// Returns the l_infinity norm of the vector.
    //double Normlinf() const;
    /// Returns the l_1 norm of the vector.
    //double Norml1() const;
    /// Returns the l_p norm of the vector.
    //double Normlp(double p) const;
    /// Returns the maximal element of the vector.
    //TM Max() const;
    /// Returns the minimal element of the vector.
    //TM Min() const;
    /// Return the sum of the vector entries
    //TM Sum() const;
    /// Compute the Euclidean distance to another vector.
    //double DistanceTo(const OccaTVector<TM> &other) const;

    /// Destroys vector.
    inline virtual ~OccaTVector()
    { size_ = 0; data.free(); }
    ///---[ Addition ]------------------
    /// Set out = v1 + v2.
    template <class TM2>
    friend void add(const OccaTVector<TM2> &v1,
                    const OccaTVector<TM2> &v2,
                    OccaTVector<TM2> &out);

    /// Set out = v1 + alpha * v2.
    template <class TM2>
    friend void add(const OccaTVector<TM2> &v1,
                    double alpha,
                    const OccaTVector<TM2> &v2,
                    OccaTVector<TM2> &out);

    /// out = alpha * (v1 + v2)
    template <class TM2>
    friend void add(const double alpha,
                    const OccaTVector<TM2> &v1,
                    const OccaTVector<TM2> &v2,
                    OccaTVector<TM2> &out);

    /// out = alpha * v1 + beta * v2
    template <class TM2>
    friend void add(const double alpha,
                    const OccaTVector<TM2> &v1,
                    const double beta,
                    const OccaTVector<TM2> &v2,
                    OccaTVector<TM2> &out);
    ///=================================

    ///---[ Subtraction ]---------------
    /// Set out = v1 - v2.
    template <class TM2>
    friend void subtract(const OccaTVector<TM2> &v1,
                         const OccaTVector<TM2> &v2,
                         OccaTVector<TM2> &out);

    /// Set out = v1 - alpha * v2.
    template <class TM2>
    friend void subtract(const OccaTVector<TM2> &v1,
                         double alpha,
                         const OccaTVector<TM2> &v2,
                         OccaTVector<TM2> &out);

    /// out = alpha * (v1 - v2)
    template <class TM2>
    friend void subtract(const double alpha,
                         const OccaTVector<TM2> &v1,
                         const OccaTVector<TM2> &v2,
                         OccaTVector<TM2> &out);

    /// out = alpha * v1 - beta * v2
    template <class TM2>
    friend void subtract(const double alpha,
                         const OccaTVector<TM2> &v1,
                         const double beta,
                         const OccaTVector<TM2> &v2,
                         OccaTVector<TM2> &out);
    ///=================================
  };

  template <class TM>
  inline const occa::properties& VecOpKernelProps() {
    static occa::properties props;
    if (!props.isInitialized()) {
      props["kernel/defines/VTYPE"] = typeToString<TM>();
    }
    return props;
  }

  typedef OccaTVector<double> OccaVector;
}

#include "ovector.tpp"

#  endif
#endif
