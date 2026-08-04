#ifndef MFEM_GSLIB_KERNEL_HELPERS_HPP
#define MFEM_GSLIB_KERNEL_HELPERS_HPP

#include "../../config/config.hpp"

#include <cmath>

namespace mfem
{

namespace gslib
{

struct dbl_range_t
{
   double min, max;
};

template <int SDIM>
struct obbox_t
{
   double c0[SDIM], A[SDIM * SDIM];
   dbl_range_t x[SDIM];
};

template <int SDIM>
struct findptsLocalHashData_t
{
   int hash_n;
   dbl_range_t bnd[SDIM];
   double fac[SDIM];
   unsigned int *offset;
};

// Eval the ith Lagrange interpolant at x.
MFEM_HOST_DEVICE inline void lagrange_eval(double *p0, double x,
                                           int i, int p_Nq,
                                           double *z, double *lagrangeCoeff)
{
   double p_i = (1 << (p_Nq - 1));
   for (int j = 0; j < p_Nq; ++j)
   {
      const double d_j = x - z[j];
      p_i *= j == i ? 1 : d_j;
   }
   p0[i] = lagrangeCoeff[i] * p_i;
}

// Eval the ith Lagrange interpolant and its first derivative at x.
MFEM_HOST_DEVICE inline void lag_eval_first_der(double *p0, double x,
                                                int i, const double *z,
                                                const double *lCoeff,
                                                int pN)
{
   double u0 = 1, u1 = 0;
   for (int j = 0; j < pN; ++j)
   {
      if (i != j)
      {
         const double d_j = 2 * (x - z[j]);
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   p0[i] = lCoeff[i] * u0;
   p0[pN + i] = 2.0 * lCoeff[i] * u1;
}

// Eval the ith Lagrange interpolant and its first and second derivative at x.
MFEM_HOST_DEVICE inline void lag_eval_second_der(double *p0, double x,
                                                 int i, const double *z,
                                                 const double *lCoeff,
                                                 int pN)
{
   double u0 = 1, u1 = 0, u2 = 0;
   for (int j = 0; j < pN; ++j)
   {
      if (i != j)
      {
         const double d_j = 2 * (x - z[j]);
         u2 = d_j * u2 + u1;
         u1 = d_j * u1 + u0;
         u0 = d_j * u0;
      }
   }
   p0[i] = lCoeff[i] * u0;
   p0[pN + i] = 2.0 * lCoeff[i] * u1;
   p0[2 * pN + i] = 8.0 * lCoeff[i] * u2;
}

// Solve Ax=y where A is a symmetric 2x2 matrix packed as {a00, a01, a11}.
MFEM_HOST_DEVICE inline void lin_solve_sym_2(double x[2],
                                             const double A[3],
                                             const double y[2])
{
   const double idet = 1 / (A[0] * A[2] - A[1] * A[1]);
   x[0] = idet * (A[2] * y[0] - A[1] * y[1]);
   x[1] = idet * (A[0] * y[1] - A[1] * y[0]);
}

// Positive when the point is inside the axis-aligned bounding box.
template <int SDIM>
MFEM_HOST_DEVICE inline double AABB_test(const obbox_t<SDIM> *const b,
                                         const double (&x)[SDIM])
{
   double test = 1.0;
   for (int d = 0; d < SDIM; ++d)
   {
      const double b_d = (x[d] - b->x[d].min) * (b->x[d].max - x[d]);
      test = test < 0.0 ? test : b_d;
   }
   return test;
}

// Positive when the point is inside the oriented bounding box.
template <int SDIM>
MFEM_HOST_DEVICE inline double bbox_test(const obbox_t<SDIM> *const b,
                                         const double (&x)[SDIM])
{
   const double bxyz = AABB_test(b, x);
   if (bxyz < 0.0)
   {
      return bxyz;
   }

   double dxyz[SDIM];
   for (int d = 0; d < SDIM; ++d)
   {
      dxyz[d] = x[d] - b->c0[d];
   }

   double test = 1.0;
   for (int d = 0; d < SDIM; ++d)
   {
      double rst = 0.0;
      for (int e = 0; e < SDIM; ++e)
      {
         rst += b->A[d * SDIM + e] * dxyz[e];
      }
      const double brst = (rst + 1.0) * (1.0 - rst);
      test = test < 0.0 ? test : brst;
   }
   return test;
}

// Hash index in the hash table for the point x.
template <int SDIM>
MFEM_HOST_DEVICE inline int hash_index(
   const findptsLocalHashData_t<SDIM> *const p,
   const double (&x)[SDIM])
{
   const int n = p->hash_n;
   int sum = 0;
   for (int d = SDIM - 1; d >= 0; --d)
   {
      sum *= n;
      const int i = (int)floor((x[d] - p->bnd[d].min) * p->fac[d]);
      sum += i < 0 ? 0 : (n - 1 < i ? n - 1 : i);
   }
   return sum;
}

// Squared Euclidean norm.
template <int SDIM>
MFEM_HOST_DEVICE inline double l2norm2(const double (&x)[SDIM])
{
   double sum = 0.0;
   for (int d = 0; d < SDIM; ++d)
   {
      sum += x[d] * x[d];
   }
   return sum;
}

template <int SDIM>
MFEM_HOST_DEVICE inline double l2norm2(const double *x)
{
   double sum = 0.0;
   for (int d = 0; d < SDIM; ++d)
   {
      sum += x[d] * x[d];
   }
   return sum;
}

} // namespace gslib

} // namespace mfem

#endif
