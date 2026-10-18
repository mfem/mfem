// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COMPLEX_COEFFICIENT
#define MFEM_COMPLEX_COEFFICIENT

#include "../config/config.hpp"
#include "../linalg/complex_densemat.hpp"
#include "../linalg/complex_vector.hpp"
#include "coefficient.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"

namespace mfem
{

class ComplexCoefficient;
class ComplexVectorCoefficient;
class ComplexMatrixCoefficient;

/** @brief Abstract base class for complex coefficients that optionally depend on space and
    time. These are used by the SesquilinearForm, ComplexLinearForm, and
    ComplexGridFunction classes to represent the physical coefficients in
    the PDEs that are being discretized. This class can also be used in a more
    general way to represent functions that don't necessarily belong to a FE
    space, e.g., to project onto ComplexGridFunctions to use as initial
    conditions, exact solutions, etc. See, e.g., ex22 for these uses. */
class ComplexCoefficient
{
protected:
   real_t time;

private:
   template <bool real_part>
   class PartCoefficient : public Coefficient
   {
   private:
      ComplexCoefficient &complex_coef_;

      void SetLocalTime(real_t t);

      friend class ComplexCoefficient;

   public:
      PartCoefficient(ComplexCoefficient &complex_coef);

      void SetTime(real_t t) override;

      real_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override;
   };

   PartCoefficient<true> re_part_coef_;
   PartCoefficient<false> im_part_coef_;

public:

   ComplexCoefficient();

   /// @note This class is non-copyable and non-movable because its internal
   /// real and imaginary views reference this object.
   ComplexCoefficient(const ComplexCoefficient &) = delete;
   ComplexCoefficient(ComplexCoefficient &&) = delete;
   ComplexCoefficient &operator=(const ComplexCoefficient &) = delete;
   ComplexCoefficient &operator=(ComplexCoefficient &&) = delete;

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t);

   /// Get the time for time dependent coefficients
   real_t GetTime() const { return time; }

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual complex_t Eval(ElementTransformation &T,
                          const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip at time @a t. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   complex_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip, real_t t);

   /** @brief Access a standard Coefficient object reproducing the real part of
       the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its real part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual Coefficient & real() { return re_part_coef_; }

   /** @brief Access a standard Coefficient object reproducing the imaginary
       part of the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its imaginary part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual Coefficient & imag() { return im_part_coef_; }

   virtual ~ComplexCoefficient() { }
};

/** @brief A complex coefficient assembled from real and imaginary
    Coefficient objects.

    The supplied coefficients are not owned and must outlive this object. */
class ComplexCoefficientFromRealImag : public ComplexCoefficient
{
private:
   Coefficient &real_coef_;
   Coefficient &imag_coef_;

public:
   ComplexCoefficientFromRealImag(Coefficient &c_r, Coefficient &c_i);

   void SetTime(real_t t) override;

   using ComplexCoefficient::Eval;

   complex_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override;

   Coefficient & real() override { return real_coef_; }
   Coefficient & imag() override { return imag_coef_; }
};

/** @brief Abstract base class for complex vector coefficients that optionally depend
    on space and time. These are used by the SesquilinearForm,
    ComplexLinearForm, and ComplexGridFunction classes to represent
    the physical vector-valued coefficients in the PDEs that are being
    discretized. This class can also be used in a more general way to
    represent functions that don't necessarily belong to a FE space,
    e.g., to project onto ComplexGridFunctions to use as initial
    conditions, exact solutions, etc. See, e.g., ex22 for these uses. */
class ComplexVectorCoefficient
{
protected:
   int vdim;
   real_t time;

private:
   template <bool real_part>
   class PartCoefficient : public VectorCoefficient
   {
   private:
      ComplexVectorCoefficient &complex_vcoef_;
#ifndef MFEM_THREAD_SAFE
      ComplexVector value_;
#endif

      void SetLocalTime(real_t t);

      friend class ComplexVectorCoefficient;

   public:
      PartCoefficient(ComplexVectorCoefficient &complex_vcoef);

      void SetTime(real_t t) override;

      void Eval(Vector &V, ElementTransformation &T,
                const IntegrationPoint &ip) override;
   };

   PartCoefficient<true> re_part_vcoef_;
   PartCoefficient<false> im_part_vcoef_;

public:
   ComplexVectorCoefficient(int vd);

   /// @note This class is non-copyable and non-movable because its internal
   /// real and imaginary views reference this object.
   ComplexVectorCoefficient(const ComplexVectorCoefficient &) = delete;
   ComplexVectorCoefficient(ComplexVectorCoefficient &&) = delete;
   ComplexVectorCoefficient &operator=(const ComplexVectorCoefficient &) = delete;
   ComplexVectorCoefficient &operator=(ComplexVectorCoefficient &&) = delete;

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t);

   /// Get the time for time dependent coefficients
   real_t GetTime() const { return time; }

   /// Returns dimension of the vector.
   int GetVDim() { return vdim; }

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at the point @a ip, storing the result in @a V. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(ComplexVector &V, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at the point @a ip at time @a t, storing the result in @a V. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip, real_t t);

   /** @brief Access a standard Coefficient object reproducing the real part of
       the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its real part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual VectorCoefficient & real() { return re_part_vcoef_; }

   /** @brief Access a standard Coefficient object reproducing the imaginary
       part of the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its imaginary part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual VectorCoefficient & imag() { return im_part_vcoef_; }

   virtual ~ComplexVectorCoefficient() { }
};

/** @brief A complex vector coefficient assembled from real and imaginary
    VectorCoefficient objects.

    The supplied coefficients are not owned and must outlive this object. */
class ComplexVectorCoefficientFromRealImag : public ComplexVectorCoefficient
{
private:
   VectorCoefficient &real_vcoef_;
   VectorCoefficient &imag_vcoef_;
#ifndef MFEM_THREAD_SAFE
   Vector V_r_, V_i_;
#endif

public:
   ComplexVectorCoefficientFromRealImag(VectorCoefficient &v_r,
                                        VectorCoefficient &v_i);

   void SetTime(real_t t) override;

   using ComplexVectorCoefficient::Eval;

   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   VectorCoefficient & real() override { return real_vcoef_; }
   VectorCoefficient & imag() override { return imag_vcoef_; }
};

/** @brief Abstract base class for complex matrix coefficients that optionally depend
    on space and time. These are used by the SesquilinearForm,
    ComplexLinearForm, and ComplexGridFunction classes to represent
    the physical matrix-valued coefficients in the PDEs that are being
    discretized. This class can also be used in a more general way to
    represent functions that don't necessarily belong to a FE space.
    See, e.g., ex22 for these uses. */
class ComplexMatrixCoefficient
{
protected:
   int height, width;
   real_t time;

private:
   template <bool real_part>
   class PartCoefficient : public MatrixCoefficient
   {
   private:
      ComplexMatrixCoefficient &complex_mcoef_;
#ifndef MFEM_THREAD_SAFE
      ComplexDenseMatrixData value_;
#endif

      void SetLocalTime(real_t t);

      friend class ComplexMatrixCoefficient;

   public:
      PartCoefficient(ComplexMatrixCoefficient &complex_mcoef);

      void SetTime(real_t t) override;

      void Eval(DenseMatrix &M, ElementTransformation &T,
                const IntegrationPoint &ip) override;
   };

   PartCoefficient<true> re_part_mcoef_;
   PartCoefficient<false> im_part_mcoef_;

public:
   /// Construct a dim x dim matrix coefficient.
   explicit ComplexMatrixCoefficient(int dim);

   /// Construct a h x w matrix coefficient.
   ComplexMatrixCoefficient(int h, int w);

   /// @note This class is non-copyable and non-movable because its internal
   /// real and imaginary views reference this object.
   ComplexMatrixCoefficient(const ComplexMatrixCoefficient &) = delete;
   ComplexMatrixCoefficient(ComplexMatrixCoefficient &&) = delete;
   ComplexMatrixCoefficient &operator=(const ComplexMatrixCoefficient &) = delete;
   ComplexMatrixCoefficient &operator=(ComplexMatrixCoefficient &&) = delete;

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t);

   /// Get the time for time dependent coefficients
   real_t GetTime() const { return time; }

   /// Get the height of the matrix.
   int GetHeight() const { return height; }

   /// Get the width of the matrix.
   int GetWidth() const { return width; }

   /// For backward compatibility get the width of the matrix.
   int GetVDim() const { return width; }

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip, storing the result in @a K. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(ComplexDenseMatrixData &K, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip at time @a t, storing the result in @a K. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   void Eval(ComplexDenseMatrixData &K, ElementTransformation &T,
             const IntegrationPoint &ip, real_t t);

   /** @brief Access a standard Coefficient object reproducing the real part of
       the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its real part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual MatrixCoefficient & real() { return re_part_mcoef_; }

   /** @brief Access a standard Coefficient object reproducing the imaginary
       part of the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its imaginary part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual MatrixCoefficient & imag() { return im_part_mcoef_; }

   virtual ~ComplexMatrixCoefficient() { }
};

/** @brief A complex matrix coefficient assembled from real and imaginary
    MatrixCoefficient objects.

    The supplied coefficients are not owned and must outlive this object. */
class ComplexMatrixCoefficientFromRealImag : public ComplexMatrixCoefficient
{
private:
   MatrixCoefficient &real_mcoef_;
   MatrixCoefficient &imag_mcoef_;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix M_r_, M_i_;
#endif

public:
   ComplexMatrixCoefficientFromRealImag(MatrixCoefficient &m_r,
                                        MatrixCoefficient &m_i);

   void SetTime(real_t t) override;

   using ComplexMatrixCoefficient::Eval;

   void Eval(ComplexDenseMatrixData &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   MatrixCoefficient & real() override { return real_mcoef_; }
   MatrixCoefficient & imag() override { return imag_mcoef_; }
};

/// A complex-valued coefficient that is constant across space and time
class ComplexConstantCoefficient : public ComplexCoefficient
{
private:
   complex_t val;

   ConstantCoefficient real_coef;
   ConstantCoefficient imag_coef;

public:
   using ComplexCoefficient::Eval;

   ComplexConstantCoefficient(const complex_t z);

   ComplexConstantCoefficient(real_t z_r, real_t z_i = 0.);

   Coefficient & real() override { return real_coef; }
   Coefficient & imag() override { return imag_coef; }

   complex_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override { return val; }
};

/// Complex-valued vector coefficient that is constant in space and time.
class ComplexVectorConstantCoefficient : public ComplexVectorCoefficient
{
private:
   ComplexVector vec;

public:
   /// Construct the coefficient with constant vector @a v.
   ComplexVectorConstantCoefficient(const ComplexVector &v)
      : ComplexVectorCoefficient(v.Size()), vec(v) { }

   /// Construct the coefficient with constant vector @a v.
   ComplexVectorConstantCoefficient(const Vector &v)
      : ComplexVectorCoefficient(v.Size()), vec(v) { }

   using ComplexVectorCoefficient::Eval;

   ///  Evaluate the vector coefficient at @a ip.
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override { V = vec; }

   /// Return a reference to the constant vector in this class.
   const ComplexVector& GetVec() const { return vec; }
};

/// Complex-valued matrix coefficient that is constant in space and time.
class ComplexMatrixConstantCoefficient : public ComplexMatrixCoefficient
{
private:
   ComplexDenseMatrixData mat;

public:
   /// Construct the coefficient with constant matrix @a m.
   ComplexMatrixConstantCoefficient(const ComplexDenseMatrixData &m)
      : ComplexMatrixCoefficient(m.Height(), m.Width()), mat(m) { }

   /// Construct the coefficient with constant matrix @a m.
   ComplexMatrixConstantCoefficient(const DenseMatrix &m)
      : ComplexMatrixCoefficient(m.Height(), m.Width()), mat(m) { }

   using ComplexMatrixCoefficient::Eval;

   ///  Evaluate the matrix coefficient at @a ip.
   void Eval(ComplexDenseMatrixData &M, ElementTransformation &T,
             const IntegrationPoint &ip) override { M = mat; }

   /// Return a reference to the constant matrix in this class.
   const ComplexDenseMatrixData& GetMat() const { return mat; }
};

/// A general complex-valued function coefficient
class ComplexFunctionCoefficient : public ComplexCoefficient
{
protected:
   std::function<complex_t(const Vector &)> Function;
   std::function<complex_t(const Vector &, real_t)> TDFunction;

public:
   using ComplexCoefficient::Eval;

   /// Define a time-independent coefficient from a std function
   /** \param F time-independent std::function */
   ComplexFunctionCoefficient(std::function<complex_t
                              (const Vector &)> F)
      : Function(std::move(F))
   { }

   /// Define a time-dependent coefficient from a std function
   /** \param TDF time-dependent function */
   ComplexFunctionCoefficient(std::function<complex_t
                              (const Vector &, real_t)> TDF)
      : TDFunction(std::move(TDF))
   { }

   /// (DEPRECATED) Define a time-independent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a f, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED ComplexFunctionCoefficient(complex_t
                                              (*f)(Vector &))
   {
      // Cast first to (void*) to suppress a warning from newer version of
      // Clang when using -Wextra.
      Function = reinterpret_cast<complex_t(*)
                 (const Vector&)>((void*)f);
      TDFunction = NULL;
   }

   /// (DEPRECATED) Define a time-dependent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a tdf, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED ComplexFunctionCoefficient(complex_t
                                              (*tdf)(Vector &, real_t))
   {
      Function = NULL;
      // Cast first to (void*) to suppress a warning from newer version of
      // Clang when using -Wextra.
      TDFunction =
         reinterpret_cast<complex_t(*)(const Vector&,
                                       real_t)>((void*)tdf);
   }

   /// Evaluate the coefficient at @a ip.
   complex_t Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override;
};

/// A general vector function coefficient
class ComplexVectorFunctionCoefficient : public ComplexVectorCoefficient
{
private:
   std::function<void(const Vector &, ComplexVector &)> Function;
   std::function<void(const Vector &, real_t, ComplexVector &)> TDFunction;
   ComplexCoefficient *Q;

public:
   /// Define a time-independent complex-valued vector coefficient
   /// from a std function
   /** \param dim - the size of the vector
       \param F - time-independent function
       \param q - optional scalar Coefficient to scale the vector coefficient */
   ComplexVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &,
                                                       ComplexVector &)> F,
                                    ComplexCoefficient *q = nullptr)
      : ComplexVectorCoefficient(dim), Function(std::move(F)), Q(q)
   { }

   /// Define a time-dependent complex-valued vector coefficient from
   /// a std function
   /** \param dim - the size of the vector
       \param TDF - time-dependent function
       \param q - optional scalar ComplexCoefficient to scale the vector coefficient */
   ComplexVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &, real_t,
                                                       ComplexVector &)> TDF,
                                    ComplexCoefficient *q = nullptr)
      : ComplexVectorCoefficient(dim), TDFunction(std::move(TDF)), Q(q)
   { }

   using ComplexVectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~ComplexVectorFunctionCoefficient() { }
};

} // end namespace mfem

#endif
