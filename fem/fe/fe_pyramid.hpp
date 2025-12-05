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

#ifndef MFEM_FE_PYRAMID
#define MFEM_FE_PYRAMID

#include "fe_base.hpp"

namespace mfem
{

/** Base class for arbitrary order basis functions on pyramid-shaped elements

  This base class provides a common class to store temporary vectors,
  matrices, and tensors computed by various functions defined on
  pyramid-shaped elements.

  The function names defined here are chosen to reflect, as closely as
  possible, those used in the paper "Orientation embedded high order
  shape functions for the exact sequence elements of all shapes" by
  Federico Fuentes, Brendan Keith, Leszek Demkowicz, and Sriram
  Nagaraj, see https://doi.org/10.1016/j.camwa.2015.04.027.

  @note Many of the functions below, e.g. lam1, lam2, etc. and related
        functions, are singular or multi-valued at the apex of the pyramid. The
   values returned near the apex are computed in the limit z->1 using
   (x, y, z) = ((1-z)/2, (1-z)/2, z) i.e. along the line from the center
   of the base to the apex.
*/
class FuentesPyramid
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix phi_E_mtmp;
   mutable Vector      phi_Q_vtmp1;
   mutable Vector      phi_Q_vtmp2;
   mutable DenseMatrix phi_Q_mtmp1;
   mutable DenseMatrix phi_Q_mtmp2;
   mutable Vector      phi_T_vtmp1;
   mutable Vector      phi_T_vtmp2;
   mutable Vector      phi_T_vtmp3;
   mutable Vector      phi_T_vtmp4;
   mutable DenseMatrix phi_T_mtmp1;
   mutable Vector      E_E_vtmp;
   mutable Vector      E_Q_vtmp;
   mutable DenseMatrix E_Q_mtmp1;
   mutable DenseMatrix E_Q_mtmp2;
   mutable DenseMatrix E_Q_mtmp3;
   mutable Vector      E_T_vtmp1;
   mutable Vector      E_T_vtmp2;
   mutable Vector      E_T_vtmp3;
   mutable DenseMatrix E_T_mtmp1;
   mutable DenseMatrix E_T_mtmp2;
   mutable DenseMatrix V_Q_mtmp1;
   mutable DenseMatrix V_Q_mtmp2;
   mutable Vector      V_T_vtmp1;
   mutable Vector      V_T_vtmp2;
   mutable DenseMatrix V_T_mtmp1;
   mutable Vector      VT_T_vtmp1;
   mutable Vector      VT_T_vtmp2;
   mutable DenseMatrix VT_T_mtmp1;
   mutable DenseTensor VT_T_ttmp1;
   mutable Vector      V_L_vtmp1;
   mutable Vector      V_L_vtmp2;
   mutable DenseMatrix V_L_mtmp1;
   mutable DenseMatrix V_L_mtmp2;
   mutable Vector      V_R_vtmp;
   mutable DenseMatrix V_R_mtmp;
#endif

protected:
   static constexpr real_t one  = 1.0;
   static constexpr real_t zero = 0.0;
   static constexpr real_t apex_tol = 1e-8;

public:
   FuentesPyramid() = default;

   static bool CheckZ(real_t z) { return std::abs(z - 1.0) > apex_tol; }

   /// Pyramid "Affine" Coordinates
   static real_t lam1(real_t x, real_t y, real_t z)
   { return CheckZ(z) ? (1.0 - x - z) * (1.0 - y - z) / (1.0 - z): 0.0; }
   static real_t lam2(real_t x, real_t y, real_t z)
   { return CheckZ(z) ? x * (1.0 - y - z) / (1.0 - z): 0.0; }
   static real_t lam3(real_t x, real_t y, real_t z)
   { return CheckZ(z) ? x * y / (1.0 - z): 0.0; }
   static real_t lam4(real_t x, real_t y, real_t z)
   { return CheckZ(z) ? (1.0 - x - z) * y / (1.0 - z): 0.0; }
   static real_t lam5(real_t x, real_t y, real_t z)
   { return CheckZ(z) ? z : 1.0; }

   /// Gradients of the "Affine" Coordinates
   static Vector grad_lam1(real_t x, real_t y, real_t z);
   static Vector grad_lam2(real_t x, real_t y, real_t z);
   static Vector grad_lam3(real_t x, real_t y, real_t z);
   static Vector grad_lam4(real_t x, real_t y, real_t z);
   static Vector grad_lam5(real_t x, real_t y, real_t z);

   /// Two component vectors associated with edges touching the apex
   static Vector lam15(real_t x, real_t y, real_t z)
   { return Vector({lam1(x, y, z), lam5(x, y, z)}); }
   static Vector lam25(real_t x, real_t y, real_t z)
   { return Vector({lam2(x, y, z), lam5(x, y, z)}); }
   static Vector lam35(real_t x, real_t y, real_t z)
   { return Vector({lam3(x, y, z), lam5(x, y, z)}); }
   static Vector lam45(real_t x, real_t y, real_t z)
   { return Vector({lam4(x, y, z), lam5(x, y, z)}); }

   /// Gradients of the above two component vectors
   static DenseMatrix grad_lam15(real_t x, real_t y, real_t z);
   static DenseMatrix grad_lam25(real_t x, real_t y, real_t z);
   static DenseMatrix grad_lam35(real_t x, real_t y, real_t z);
   static DenseMatrix grad_lam45(real_t x, real_t y, real_t z);

   /// Computes $\lambda_i \nabla \lambda_5 -  \lambda_5 \nabla \lambda_i$
   static Vector lam15_grad_lam15(real_t x, real_t y, real_t z);
   static Vector lam25_grad_lam25(real_t x, real_t y, real_t z);
   static Vector lam35_grad_lam35(real_t x, real_t y, real_t z);
   static Vector lam45_grad_lam45(real_t x, real_t y, real_t z);

   /// Three component vectors associated with triangular faces
   static Vector lam125(real_t x, real_t y, real_t z)
   { return Vector({lam1(x, y, z), lam2(x, y, z), lam5(x, y, z)}); }
   static Vector lam235(real_t x, real_t y, real_t z)
   { return Vector({lam2(x, y, z), lam3(x, y, z), lam5(x, y, z)}); }
   static Vector lam345(real_t x, real_t y, real_t z)
   { return Vector({lam3(x, y, z), lam4(x, y, z), lam5(x, y, z)}); }
   static Vector lam435(real_t x, real_t y, real_t z)
   { return Vector({lam4(x, y, z), lam3(x, y, z), lam5(x, y, z)}); }
   static Vector lam415(real_t x, real_t y, real_t z)
   { return Vector({lam4(x, y, z), lam1(x, y, z), lam5(x, y, z)}); }
   static Vector lam145(real_t x, real_t y, real_t z)
   { return Vector({lam1(x, y, z), lam4(x, y, z), lam5(x, y, z)}); }

   /// Vector functions related to the normals to the triangular faces
   ///
   /// Computes
   /// $
   ///    \lambda_i \nabla\lambda_j \times \nabla \lambda_5
   ///    + \lambda_j \nabla\lambda_5 \times \nabla \lambda_i
   ///    + \lambda_5 \nabla\lambda_i \times \nabla \lambda_j
   /// $
   static Vector lam125_grad_lam125(real_t x, real_t y, real_t z);
   static Vector lam235_grad_lam235(real_t x, real_t y, real_t z);
   static Vector lam345_grad_lam345(real_t x, real_t y, real_t z);
   static Vector lam435_grad_lam435(real_t x, real_t y, real_t z);
   static Vector lam415_grad_lam415(real_t x, real_t y, real_t z);
   static Vector lam145_grad_lam145(real_t x, real_t y, real_t z);

   /// Divergences of the above "normal" vector functions divided by 3
   static real_t div_lam125_grad_lam125(real_t x, real_t y, real_t z);
   static real_t div_lam235_grad_lam235(real_t x, real_t y, real_t z);
   static real_t div_lam345_grad_lam345(real_t x, real_t y, real_t z);
   static real_t div_lam435_grad_lam435(real_t x, real_t y, real_t z);
   static real_t div_lam415_grad_lam415(real_t x, real_t y, real_t z);
   static real_t div_lam145_grad_lam145(real_t x, real_t y, real_t z);

   static real_t mu0(real_t z)
   { return 1.0 - z; }
   static real_t mu1(real_t z)
   { return z; }

   static Vector grad_mu0(real_t z)
   { return Vector({0.0, 0.0, -1.0}); }
   static Vector grad_mu1(real_t z)
   { return Vector({0.0, 0.0, 1.0}); }

   static Vector mu01(real_t z)
   { return Vector({mu0(z), mu1(z)}); }

   static DenseMatrix grad_mu01(real_t z);

   static real_t mu0(real_t z, const Vector &xy, unsigned int ab)
   { return 1.0 - xy[ab-1] / (1.0 - z); }
   static real_t mu1(real_t z, const Vector &xy, unsigned int ab)
   { return xy[ab-1] / (1.0 - z); }

   static Vector grad_mu0(real_t z, const Vector xy, unsigned int ab);
   static Vector grad_mu1(real_t z, const Vector xy, unsigned int ab);

   static Vector mu01(real_t z, Vector xy, unsigned int ab)
   { return Vector({mu0(z, xy, ab), mu1(z, xy, ab)}); }

   static DenseMatrix grad_mu01(real_t z, Vector xy, unsigned int ab);
   static Vector mu01_grad_mu01(real_t z, Vector xy, unsigned int ab);

   static real_t nu0(real_t z, Vector xy, unsigned int ab)
   { return 1.0 - xy[ab-1] - z; }
   static real_t nu1(real_t z, Vector xy, unsigned int ab) { return xy[ab-1]; }
   static real_t nu2(real_t z, Vector xy, unsigned int ab) { return z; }

   static Vector grad_nu0(real_t z, const Vector xy, unsigned int ab);
   static Vector grad_nu1(real_t z, const Vector xy, unsigned int ab);
   static Vector grad_nu2(real_t z, const Vector xy, unsigned int ab);

   static Vector nu01(real_t z, Vector xy, unsigned int ab)
   { return Vector({nu0(z, xy, ab), nu1(z, xy, ab)}); }
   static Vector nu12(real_t z, Vector xy, unsigned int ab)
   { return Vector({nu1(z, xy, ab), nu2(z, xy, ab)}); }
   static Vector nu012(real_t z, Vector xy, unsigned int ab)
   { return Vector({nu0(z, xy, ab), nu1(z, xy, ab), nu2(z, xy, ab)}); }
   static Vector nu120(real_t z, Vector xy, unsigned int ab)
   { return Vector({nu1(z, xy, ab), nu2(z, xy, ab), nu0(z, xy, ab)}); }

   static DenseMatrix grad_nu01(real_t z, Vector xy, unsigned int ab);
   static DenseMatrix grad_nu012(real_t z, Vector xy, unsigned int ab);
   static DenseMatrix grad_nu120(real_t z, Vector xy, unsigned int ab);

   static Vector nu01_grad_nu01(real_t z, Vector xy, unsigned int ab);
   static Vector nu12_grad_nu12(real_t z, Vector xy, unsigned int ab);
   static Vector nu012_grad_nu012(real_t z, Vector xy, unsigned int ab);

   /// Shifted and Scaled Legendre Polynomials
   /** Implements a scaled and shifted set of Legendre polynomials

         $P_i(x;t) = P_i(x / t) * t^i$

       where @a t >= 0.0, @a x $\in [0,t]$, and $P_i$ is the shifted Legendre
       polynomial defined on $[0,1]$ rather than the usual $[-1,1]$. The
       entries stored in @a u correspond to the values of
       $P_0$, $P_1$, ... $P_p$.

       @a u must be at least @a p + 1 in length
   */
   static void CalcScaledLegendre(int p, real_t x, real_t t,
                                  real_t *u);
   static void CalcScaledLegendre(int p, real_t x, real_t t,
                                  real_t *u, real_t *dudx, real_t *dudt);

   static void CalcScaledLegendre(int p, real_t x, real_t t,
                                  Vector &u);
   static void CalcScaledLegendre(int p, real_t x, real_t t,
                                  Vector &u, Vector &dudx, Vector &dudt);

   /// Integrated Legendre Polynomials
   /** These are the integrals of the shifted and scaled Legendre polynomials
       provided above and defined as:

         $L_i(x;t) = \int_0^x P_{i-1}(y;t)dy\mbox{ for }i>=1$

       These polynomials are computed as:

         $L_0(x;t) = 0$, $L_1(x;t) = x$,

         $2(2i-1)L_i(x;t) = P_i(x;t) - t^2 P_{i-2}(x;t)\mbox{ for }i>=2$

       @a u must be at least @a p + 1 in length
    */
   static void CalcIntegratedLegendre(int p, real_t x,
                                      real_t t, real_t *u);
   static void CalcIntegratedLegendre(int p, real_t x,
                                      real_t t, real_t *u,
                                      real_t *dudx, real_t *dudt);

   static void CalcIntegratedLegendre(int p, real_t x,
                                      real_t t, Vector &u);
   static void CalcIntegratedLegendre(int p, real_t x,
                                      real_t t, Vector &u,
                                      Vector &dudx, Vector &dudt);

   /** @a u must be at least @a p + 1 in length */
   static void CalcHomogenizedScaLegendre(int p, real_t s0, real_t s1,
                                          real_t *u);
   static void CalcHomogenizedScaLegendre(int p,
                                          real_t s0, real_t s1,
                                          real_t *u,
                                          real_t *duds0, real_t *duds1);
   static void CalcHomogenizedScaLegendre(int p, real_t s0, real_t s1,
                                          Vector &u);
   static void CalcHomogenizedScaLegendre(int p,
                                          real_t s0, real_t s1,
                                          Vector &u,
                                          Vector &duds0, Vector &duds1);

   /** @a u must be at least @a p + 1 in length */
   static void CalcHomogenizedIntLegendre(int p,
                                          real_t t0, real_t t1,
                                          real_t *u);
   static void CalcHomogenizedIntLegendre(int p,
                                          real_t t0, real_t t1,
                                          real_t *u,
                                          real_t *dudt0, real_t *dudt1);
   static void CalcHomogenizedIntLegendre(int p,
                                          real_t t0, real_t t1,
                                          Vector &u);
   static void CalcHomogenizedIntLegendre(int p,
                                          real_t t0, real_t t1,
                                          Vector &u,
                                          Vector &dudt0, Vector &dudt1);

   /// Shifted and Scaled Jacobi Polynomials
   /** Implements a scaled and shifted set of Jacobi polynomials

         $P^\alpha_i(x / t) * t^i$

       where @a alpha $= \alpha >-1$, @a t $>= 0.0$, @a x $\in [0,t]$, and
       $P^\alpha_i$ is the shifted Jacobi polynomial defined on $[0,1]$ rather
       than the usual $[-1,1]$. The entries stored in @a u correspond to the
       values of $P^\alpha_0$, $P^\alpha_1$, ... $P^\alpha_p$.

       @note Jacobi polynomials typically posses two parameters,
       $P^{\alpha, \beta}_i$, but we only consider the special case where
       $\beta=0$.

       @a u must be at least @a p + 1 in length
   */
   static void CalcScaledJacobi(int p, real_t alpha,
                                real_t x, real_t t,
                                real_t *u);
   static void CalcScaledJacobi(int p, real_t alpha,
                                real_t x, real_t t,
                                real_t *u, real_t *dudx, real_t *dudt);

   static void CalcScaledJacobi(int p, real_t alpha,
                                real_t x, real_t t,
                                Vector &u);
   static void CalcScaledJacobi(int p, real_t alpha,
                                real_t x, real_t t,
                                Vector &u, Vector &dudx, Vector &dudt);

   /// Integrated Jacobi Polynomials
   /** These are the integrals of the shifted and scaled Jacobi polynomials
       provided above and defined as:

         $L^\alpha_i(x;t) = \int_0^x P^\alpha_{i-1}(y;t)dy\mbox{ for }i>=1$

       These polynomials are computed as:

         $L^\alpha_0(x;t) = 0$, $L^\alpha_1(x;t) = x$,

         $L^\alpha_i(x;t) = a_i P^\alpha_i(x;t) + b_i t P^\alpha_{i-1}(x;t)
                        - c_i t^2 P^\alpha_{i-2}(x;t)\mbox{ for }i>=2$

       With

         $a_i = (i + \alpha) / (2i + \alpha - 1)(2i + \alpha)$

         $b_i = \alpha / (2i + \alpha - 2)(2i + \alpha)$

         $c_i = (i - 1) / (2i + \alpha - 2)(2i + \alpha - 1)$

       @a u must be at least @a p + 1 in length
    */
   static void CalcIntegratedJacobi(int p, real_t alpha,
                                    real_t x, real_t t,
                                    real_t *u);
   static void CalcIntegratedJacobi(int p, real_t alpha,
                                    real_t x, real_t t,
                                    real_t *u, real_t *dudx, real_t *dudt);

   static void CalcIntegratedJacobi(int p, real_t alpha,
                                    real_t x, real_t t,
                                    Vector &u)
   { CalcIntegratedJacobi(p, alpha, x, t, u.GetData()); }
   static void CalcIntegratedJacobi(int p, real_t alpha,
                                    real_t x, real_t t,
                                    Vector &u, Vector &dudx, Vector &dudt)
   {
      CalcIntegratedJacobi(p, alpha, x, t, u.GetData(),
                           dudx.GetData(), dudt.GetData());
   }

   /** @a u must be at least @a p + 1 in length */
   static void CalcHomogenizedScaJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        real_t *u)
   { CalcScaledJacobi(p, alpha, t1, t0 + t1, u); }
   static void CalcHomogenizedScaJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        real_t *u,
                                        real_t *dudt0, real_t *dudt1);
   static void CalcHomogenizedScaJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        Vector &u);
   static void CalcHomogenizedScaJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        Vector &u,
                                        Vector &dudt0, Vector &dudt1);

   /** @a u must be at least @a p + 1 in length */
   static void CalcHomogenizedIntJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        real_t *u)
   { CalcIntegratedJacobi(p, alpha, t1, t0 + t1, u); }
   static void CalcHomogenizedIntJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        real_t *u,
                                        real_t *dudt0, real_t *dudt1);
   static void CalcHomogenizedIntJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        Vector &u);
   static void CalcHomogenizedIntJacobi(int p, real_t alpha,
                                        real_t t0, real_t t1,
                                        Vector &u,
                                        Vector &dudt0, Vector &dudt1);

   /** @a u must be at least @a p + 1 in length */
   static void phi_E(int p, real_t s0, real_t s1, real_t *u);
   static void phi_E(int p, real_t s0, real_t s1, real_t *u,
                     real_t *duds0, real_t *duds1);
   static void phi_E(int p, Vector s, Vector &u);
   static void phi_E(int p, Vector s, Vector &u, DenseMatrix &duds);

   /** @a grad_s must be 2x3 */
   void phi_E(int p, Vector s, const DenseMatrix &grad_s,
              Vector &u, DenseMatrix &grad_u) const;

   /** @a u must be at least (p+1)x(p+1) in size */
   void phi_Q(int p, Vector s, Vector t, DenseMatrix &u) const;
   void phi_Q(int p, Vector s, const DenseMatrix &grad_s,
              Vector t, const DenseMatrix &grad_t,
              DenseMatrix &u, DenseTensor &grad_u) const;
   void phi_T(int p, Vector nu, DenseMatrix &u) const;
   void phi_T(int p, Vector nu, const DenseMatrix &grad_nu,
              DenseMatrix &u, DenseTensor &grad_u) const;

   /** This is a vector-valued function associated with an edge of a pyramid

   The vector @a s contains two coordinate values and @a ds is related to the
   gradient of these coordinates with respect to the reference coordinates i.e.
      sds = s0 grad s1 - s1 grad s0
   */
   void E_E(int p, Vector s, Vector sds, DenseMatrix &u) const;
   void E_E(int p, Vector s, const DenseMatrix &grad_s, DenseMatrix &u,
            DenseMatrix &curl_u) const;

   void E_Q(int p, Vector s, Vector ds, Vector t,
            DenseTensor &u) const;
   void E_Q(int p, Vector s, const DenseMatrix &grad_s,
            Vector t, const DenseMatrix &grad_t,
            DenseTensor &u, DenseTensor &curl_u) const;

   void E_T(int p, Vector s, Vector sds, DenseTensor &u) const;
   void E_T(int p, Vector s, const DenseMatrix &grad_s,
            DenseTensor &u, DenseTensor &curl_u) const;

   /** This is a vector-valued function associated with the quadrilateral face
       of a pyramid

   The vectors @a s and @a t contain pairs of coordinate values and @a ds and
   @a dt are related to derivatives of these coordinates:

      ds = s0 grad s1 - s1 grad s0

      dt = t0 grad t1 - t1 grad t0
   */
   void V_Q(int p, Vector s, Vector ds, Vector t, Vector dt,
            DenseTensor &u) const;

   /** This is a vector-valued function associated with the triangular faces of
       a pyramid

   The vector @a s contains three coordinate values and @a sdsxds is related to
   derivatives of these coordinates with respect to the reference coordinates:

      sdsxds = s0 grad s1 x grad s2 + s1 grad s2 x grad s0 +
               s2 grad s0 x grad s1
   */
   void V_T(int p, Vector s, Vector sdsxds, DenseTensor &u) const;

   /** This computes V_T as above and its divergence

   The vector @a s contains three coordinate values and @a sdsxds is related to
   derivatives of these coordinates with respect to the reference coordinates:

      sdsxds = s0 grad s1 x grad s2 + s1 grad s2 x grad s0 +
               s2 grad s0 x grad s1

   The scalar @a dsdsxds is the divergence of sdsxds:

      dsdsxds = grad s0 dot (grad s1 x grad s2)
   */
   void V_T(int p, Vector s, Vector sdsxds, real_t dsdsxds,
            DenseTensor &u, DenseMatrix &du) const;

   void VT_T(int p, Vector s, Vector sds, Vector sdsxds,
             real_t mu, Vector grad_mu, DenseTensor &u) const;
   void VT_T(int p, Vector s, Vector sds, Vector sdsxds,
             Vector grad_s2, real_t mu, Vector grad_mu,
             DenseTensor &u, DenseMatrix &du) const;

   /** This implements $V^\unlhd_{ij}$ from the Fuentes paper

      @a u must be at least (p+1)x(p+1)x3
   */
   void V_L(int p, Vector sx, const DenseMatrix &grad_sx,
            Vector sy, const DenseMatrix &grad_sy,
            real_t t, Vector grad_t, DenseTensor &u) const;

   /** This implements $V^\unrhd_i$ from the Fuentes paper

      @a u must be at least (p+1)x3 */
   void V_R(int p, Vector s, const DenseMatrix &grad_s,
            real_t mu, Vector grad_mu,
            real_t t, Vector grad_t, DenseMatrix &u) const;
};

} // namespace mfem

#endif

