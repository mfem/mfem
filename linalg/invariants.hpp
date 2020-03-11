// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_INVARIANTS_HPP
#define MFEM_INVARIANTS_HPP

#include "../config/config.hpp"
#include "../general/error.hpp"
#include <cmath>

namespace mfem
{

// Matrix invariants and their derivatives for 2x2 and 3x3 matrices.

/** @brief Auxiliary class used as the default for the second template parameter
    in the classes InvariantsEvaluator2D and InvariantsEvaluator3D. */
template <typename scalar_t>
struct ScalarOps
{
   static scalar_t sign(const scalar_t &a)
   { return (a >= scalar_t(0)) ? scalar_t(1) : scalar_t(-1); }

   static scalar_t pow(const scalar_t &x, int m, int n)
   { return std::pow(x, scalar_t(m)/n); }
};


/** @brief Auxiliary class for evaluating the 2x2 matrix invariants and their
    first and second derivatives. */
/**
    The type `scalar_t` must support the standard operations:

        =, +=, -=, +, -, *, /, unary -, int*scalar_t, int/scalar_t, scalar_t/int

    The type `scalar_ops` must define the static method:

        scalar_t sign(const scalar_t &);
*/
template <typename scalar_t, typename scalar_ops = ScalarOps<scalar_t> >
class InvariantsEvaluator2D
{
protected:
   // Transformation Jacobian
   const scalar_t *J;

   // Invariants: I_1 = ||J||_F^2, \bar{I}_1 = I_1/det(J), \bar{I}_2 = det(J).
   scalar_t I1, I1b, I2b;

   // Derivatives of I1, I1b, I2, and I2b using column-major storage.
   scalar_t dI1[4], dI1b[4], dI2[4], dI2b[4];

   int D_height, alloc_height;
   const scalar_t *D; // Always points to external data or is empty
   scalar_t *DaJ, *DJt, *DXt, *DYt;

   scalar_t sign_detJ;

   enum EvalMasks
   {
      HAVE_I1   = 1,
      HAVE_I1b  = 2,
      HAVE_I2b  = 4,
      HAVE_dI1  = 8,
      HAVE_dI1b = 16,
      HAVE_dI2  = 32,
      HAVE_dI2b = 64,
      HAVE_DaJ  = 128, // D adj(J) = D dI2b^t
      HAVE_DJt  = 256  // D J^t
   };

   // Bitwise OR of EvalMasks
   int eval_state;

   bool dont(int have_mask) const { return !(eval_state & have_mask); }

   void Eval_I1()
   {
      eval_state |= HAVE_I1;
      I1 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2] + J[3]*J[3];
   }
   void Eval_I1b()
   {
      eval_state |= HAVE_I1b;
      I1b = Get_I1()/Get_I2b();
   }
   void Eval_I2b()
   {
      eval_state |= HAVE_I2b;
      const scalar_t det = J[0]*J[3] - J[1]*J[2];
      sign_detJ = scalar_ops::sign(det);
      I2b = sign_detJ*det;
   }
   void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      dI1[0] = 2*J[0]; dI1[2] = 2*J[2];
      dI1[1] = 2*J[1]; dI1[3] = 2*J[3];
   }
   void Eval_dI1b()
   {
      eval_state |= HAVE_dI1b;
      // I1b = I1/I2b
      // dI1b = (1/I2b)*dI1 - (I1/I2b^2)*dI2b = (2/I2b)*[J - (I1b/2)*dI2b]
      const scalar_t c1 = 2/Get_I2b();
      const scalar_t c2 = Get_I1b()/2;
      Get_dI2b();
      dI1b[0] = c1*(J[0] - c2*dI2b[0]);
      dI1b[1] = c1*(J[1] - c2*dI2b[1]);
      dI1b[2] = c1*(J[2] - c2*dI2b[2]);
      dI1b[3] = c1*(J[3] - c2*dI2b[3]);
   }
   void Eval_dI2()
   {
      eval_state |= HAVE_dI2;
      // I2 = I2b^2
      // dI2 = 2*I2b*dI2b = 2*det(J)*adj(J)^T
      const scalar_t c1 = 2*Get_I2b();
      Get_dI2b();
      dI2[0] = c1*dI2b[0];
      dI2[1] = c1*dI2b[1];
      dI2[2] = c1*dI2b[2];
      dI2[3] = c1*dI2b[3];
   }
   void Eval_dI2b()
   {
      eval_state |= HAVE_dI2b;
      // I2b = det(J)
      // dI2b = adj(J)^T
      Get_I2b();
      dI2b[0] =  sign_detJ*J[3];
      dI2b[1] = -sign_detJ*J[2];
      dI2b[2] = -sign_detJ*J[1];
      dI2b[3] =  sign_detJ*J[0];
   }
   void Eval_DaJ() // D adj(J) = D dI2b^t
   {
      eval_state |= HAVE_DaJ;
      Get_dI2b();
      Eval_DZt(dI2b, &DaJ);
   }
   void Eval_DJt() // D J^t
   {
      eval_state |= HAVE_DJt;
      Eval_DZt(J, &DJt);
   }
   void Eval_DZt(const scalar_t *Z, scalar_t **DZt_ptr)
   {
      MFEM_ASSERT(D != NULL, "");
      const int nd = D_height;
      scalar_t *DZt = *DZt_ptr;
      if (DZt == NULL) { *DZt_ptr = DZt = new scalar_t[2*alloc_height]; }
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1;
         DZt[i0] = D[i0]*Z[0] + D[i1]*Z[2];
         DZt[i1] = D[i0]*Z[1] + D[i1]*Z[3];
      }
   }

public:
   /// The Jacobian should use column-major storage.
   InvariantsEvaluator2D(const scalar_t *Jac = NULL)
      : J(Jac), D_height(), alloc_height(), D(), DaJ(), DJt(), DXt(), DYt(),
        eval_state(0) { }

   ~InvariantsEvaluator2D()
   {
      delete [] DYt;
      delete [] DXt;
      delete [] DJt;
      delete [] DaJ;
   }

   /// The Jacobian should use column-major storage.
   void SetJacobian(const scalar_t *Jac) { J = Jac; eval_state = 0; }

   /// The @a Deriv matrix is `dof x 2`, using column-major storage.
   void SetDerivativeMatrix(int height, const scalar_t *Deriv)
   {
      eval_state &= ~(HAVE_DaJ | HAVE_DJt);
      if (alloc_height < height)
      {
         delete [] DYt; DYt = NULL;
         delete [] DXt; DXt = NULL;
         delete [] DJt; DJt = NULL;
         delete [] DaJ; DaJ = NULL;
         alloc_height = height;
      }
      D_height = height;
      D = Deriv;
   }

   scalar_t Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   scalar_t Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   scalar_t Get_I2()  { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b*I2b; }
   scalar_t Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }

   const scalar_t *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   const scalar_t *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   const scalar_t *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   const scalar_t *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }

   // Assemble operation for tensor X with components X_jslt:
   //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is X_jslt D_kt)
   //    0 <= i,k < nd,  0 <= j,l,s,t < 2
   // where nd is the height of D, i.e. the number of DOFs in one component.

   void Assemble_ddI1(scalar_t w, scalar_t *A)
   {
      // ddI1_jslt = 2 I_jslt = 2 δ_jl δ_st
      //    A(i+nd*j,k+nd*l) += (\sum_st  2 w D_is δ_jl δ_st D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (2 w) (\sum_s  D_is D_ks) δ_jl
      //    A(i+nd*j,k+nd*l) += (2 w) (D D^t)_ik δ_jl

      const int nd = D_height;
      const int ah = 2*nd;
      const scalar_t a = 2*w;
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1;
         const scalar_t aDi[2] = { a*D[i0], a*D[i1] };
         // k == i
         const scalar_t aDDt_ii = aDi[0]*D[i0] + aDi[1]*D[i1];
         A[i0+ah*i0] += aDDt_ii;
         A[i1+ah*i1] += aDDt_ii;
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const int k0 = k+nd*0, k1 = k+nd*1;
            const scalar_t aDDt_ik = aDi[0]*D[k0] + aDi[1]*D[k1];
            A[i0+ah*k0] += aDDt_ik;
            A[k0+ah*i0] += aDDt_ik;
            A[i1+ah*k1] += aDDt_ik;
            A[k1+ah*i1] += aDDt_ik;
         }
      }
   }
   void Assemble_ddI1b(scalar_t w, scalar_t *A)
   {
      // ddI1b = X1 + X2 + X3, where
      // X1_ijkl = (I1b/I2) [ (δ_ks δ_it + δ_kt δ_si) dI2b_tj dI2b_sl ]
      //         = (I1b/I2) [ dI2b_ij dI2b_kl + dI2b_kj dI2b_il ]
      // X2_ijkl = (2/I2b) δ_ik δ_jl = (1/I2b) ddI1_ijkl
      // X3_ijkl = -(2/I2) (δ_ks δ_it) (J_tj dI2b_sl + dI2b_tj J_sl)
      //         = -(2/I2) (J_ij dI2b_kl + dI2b_ij J_kl)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI1b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       w (I1b/I2) [(D dI2b^t)_ij (D dI2b^t)_kl +
      //                   (D dI2b^t)_il (D dI2b^t)_kj]
      //     + w (2/I2b)  δ_jl (D D^t)_ik
      //     - w (2/I2)   [(D J^t)_ij (D dI2b^t)_kl + (D dI2b^t)_ij (D J^t)_kl]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      if (dont(HAVE_DJt)) { Eval_DJt(); }
      const int nd = D_height;
      const int ah = 2*nd;
      const scalar_t a = w*Get_I1b()/Get_I2();
      const scalar_t b = 2*w/Get_I2b();
      const scalar_t c = -2*w/Get_I2();
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1;
         const scalar_t aDaJ_i[2] = { a*DaJ[i0], a*DaJ[i1] };
         const scalar_t bD_i[2] = { b*D[i0], b*D[i1] };
         const scalar_t cDJt_i[2] = { c*DJt[i0], c*DJt[i1] };
         const scalar_t cDaJ_i[2] = { c*DaJ[i0], c*DaJ[i1] };
         // k == i
         {
            // Symmetries: A2_ii_00 = A2_ii_11
            const scalar_t A2_ii = bD_i[0]*D[i0] + bD_i[1]*D[i1];

            A[i0+ah*i0] += 2*(aDaJ_i[0] + cDJt_i[0])*DaJ[i0] + A2_ii;

            // Symmetries: A_ii_01 = A_ii_10
            const scalar_t A_ii_01 =
               (2*aDaJ_i[0] + cDJt_i[0])*DaJ[i1] + cDaJ_i[0]*DJt[i1];
            A[i0+ah*i1] += A_ii_01;
            A[i1+ah*i0] += A_ii_01;

            A[i1+ah*i1] += 2*(aDaJ_i[1] + cDJt_i[1])*DaJ[i1] + A2_ii;
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const int k0 = k+nd*0, k1 = k+nd*1;
            // Symmetries: A1_ik_01 = A1_ik_10 = A1_ki_01 = A1_ki_10
            const scalar_t A1_ik_01 = aDaJ_i[0]*DaJ[k1] + aDaJ_i[1]*DaJ[k0];

            // Symmetries: A2_ik_00 = A2_ik_11 = A2_ki_00 = A2_ki_11
            const scalar_t A2_ik = bD_i[0]*D[k0] + bD_i[1]*D[k1];

            const scalar_t A_ik_00 =
               (2*aDaJ_i[0] + cDJt_i[0])*DaJ[k0] + A2_ik + cDaJ_i[0]*DJt[k0];
            A[i0+ah*k0] += A_ik_00;
            A[k0+ah*i0] += A_ik_00;

            const scalar_t A_ik_01 =
               A1_ik_01 + cDJt_i[0]*DaJ[k1] + cDaJ_i[0]*DJt[k1];
            A[i0+ah*k1] += A_ik_01;
            A[k1+ah*i0] += A_ik_01;

            const scalar_t A_ik_10 =
               A1_ik_01 + cDJt_i[1]*DaJ[k0] + cDaJ_i[1]*DJt[k0];
            A[i1+ah*k0] += A_ik_10;
            A[k0+ah*i1] += A_ik_10;

            const scalar_t A_ik_11 =
               (2*aDaJ_i[1] + cDJt_i[1])*DaJ[k1] + A2_ik + cDaJ_i[1]*DJt[k1];
            A[i1+ah*k1] += A_ik_11;
            A[k1+ah*i1] += A_ik_11;
         }
      }
   }
   void Assemble_ddI2(scalar_t w, scalar_t *A)
   {
      // ddI2_ijkl = 2 (2 δ_ks δ_it - δ_kt δ_si) dI2b_tj dI2b_sl
      //           = 4 dI2b_ij dI2b_kl - 2 dI2b_kj dI2b_il
      //           = 2 dI2b_ij dI2b_kl + 2 (dI2b_ij dI2b_kl - dI2b_kj dI2b_il)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_st  w D_is (4 dI2b_js dI2b_lt - 2 dI2b_ls dI2b_jt) D_kt)
      //    A(i+nd*j,k+nd*l) +=
      //       2 w [2 (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj]
      //
      // Note: the expression
      //    (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj
      // is the determinant of the 2x2 matrix formed by rows {i,k} and columns
      // {j,l} from the matrix (D dI2b^t).

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D_height;
      const int ah = 2*nd;
      const scalar_t a = 2*w;
      for (int i = 0; i < ah; i++)
      {
         const scalar_t avi = a*DaJ[i];
         A[i+ah*i] += avi*DaJ[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t aVVt_ij = avi*DaJ[j];
            A[i+ah*j] += aVVt_ij;
            A[j+ah*i] += aVVt_ij;
         }
      }
      const int j = 1, l = 0;
      for (int i = 0; i < nd; i++)
      {
         const int ij = i+nd*j, il = i+nd*l;
         const scalar_t aDaJ_ij = a*DaJ[ij], aDaJ_il = a*DaJ[il];
         for (int k = 0; k < i; k++)
         {
            const int kj = k+nd*j, kl = k+nd*l;
            const scalar_t A_ijkl = aDaJ_ij*DaJ[kl] - aDaJ_il*DaJ[kj];
            A[ij+ah*kl] += A_ijkl;
            A[kl+ah*ij] += A_ijkl;
            A[kj+ah*il] -= A_ijkl;
            A[il+ah*kj] -= A_ijkl;
         }
      }
   }
   void Assemble_ddI2b(scalar_t w, scalar_t *A)
   {
      // ddI2b_ijkl = (1/I2b) (δ_ks δ_it - δ_kt δ_si) dI2b_tj dI2b_sl
      //    [j -> u], [l -> v], [i -> j], [k -> l]
      // ddI2b_julv = (1/I2b) (δ_ls δ_jt - δ_lt δ_sj) dI2b_tu dI2b_sv
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (\sum_uv  w D_iu ddI2b_julv D_kv)
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_uvst (w/I2b)
      //          D_iu (δ_ls δ_jt - δ_lt δ_sj) dI2b_tu dI2b_sv D_kv)
      //    A(i+nd*j,k+nd*l) +=
      //       (\sum_st (w/I2b)
      //          (D dI2b^t)_it (δ_ls δ_jt - δ_lt δ_sj) (D dI2b^t)_ks)
      //    A(i+nd*j,k+nd*l) += (w/I2b)
      //       [ (D dI2b^t)_ij (D dI2b^t)_kl - (D dI2b^t)_il (D dI2b^t)_kj ]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D_height;
      const int ah = 2*nd;
      const int j = 1, l = 0;
      const scalar_t a = w/Get_I2b();
      for (int i = 0; i < nd; i++)
      {
         const int ij = i+nd*j, il = i+nd*l;
         const scalar_t aDaJ_ij = a*DaJ[ij], aDaJ_il = a*DaJ[il];
         for (int k = 0; k < i; k++)
         {
            const int kj = k+nd*j, kl = k+nd*l;
            const scalar_t A_ijkl = aDaJ_ij*DaJ[kl] - aDaJ_il*DaJ[kj];
            A[ij+ah*kl] += A_ijkl;
            A[kl+ah*ij] += A_ijkl;
            A[kj+ah*il] -= A_ijkl;
            A[il+ah*kj] -= A_ijkl;
         }
      }
   }
   // Assemble the contribution from the term: T_ijkl = X_ij Y_kl + Y_ij X_kl,
   // where X and Y are pointers to 2x2 matrices stored in column-major layout.
   //
   // The contribution to the matrix A is given by:
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is T_jslt D_kt
   // or
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is (X_js Y_lt + Y_js X_lt) D_kt
   // or
   //    A(i+nd*j,k+nd*l) +=
   //       \sum_st  w [ (D X^t)_ij (D Y^t)_kl + (D Y^t)_ij (D X^t)_kl ]
   void Assemble_TProd(scalar_t w, const scalar_t *X, const scalar_t *Y,
                       scalar_t *A)
   {
      Eval_DZt(X, &DXt);
      Eval_DZt(Y, &DYt);
      const int nd = D_height;
      const int ah = 2*nd;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t axi = w*DXt[i], ayi = w*DYt[i];
         A[i+ah*i] += 2*axi*DYt[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t A_ij = axi*DYt[j] + ayi*DXt[j];
            A[i+ah*j] += A_ij;
            A[j+ah*i] += A_ij;
         }
      }
   }

   // Assemble the contribution from the term: T_ijkl = X_ij X_kl, where X is a
   // pointer to a 2x2 matrix stored in column-major layout.
   //
   // The contribution to the matrix A is given by:
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is X_js X_lt D_kt
   // or
   //    A(i+nd*j,k+nd*l) += \sum_st  w [ (D X^t)_ij (D X^t)_kl ]
   void Assemble_TProd(scalar_t w, const scalar_t *X, scalar_t *A)
   {
      Eval_DZt(X, &DXt);
      const int nd = D_height;
      const int ah = 2*nd;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t axi = w*DXt[i];
         A[i+ah*i] += axi*DXt[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t A_ij = axi*DXt[j];
            A[i+ah*j] += A_ij;
            A[j+ah*i] += A_ij;
         }
      }
   }
};


/** @brief Auxiliary class for evaluating the 3x3 matrix invariants and their
    first and second derivatives. */
/**
    The type `scalar_t` must support the standard operations:

        =, +=, -=, +, -, *, /, unary -, int*scalar_t, int/scalar_t, scalar_t/int

    The type `scalar_ops` must define the static methods:

        scalar_t sign(const scalar_t &);
        scalar_t pow(const scalar_t &x, int a, int b); // x^(a/b)
*/
template <typename scalar_t, typename scalar_ops = ScalarOps<scalar_t> >
class InvariantsEvaluator3D
{
protected:
   // Transformation Jacobian
   const scalar_t *J;

   // Invariants:
   //    I_1 = ||J||_F^2, \bar{I}_1 = det(J)^{-2/3}*I_1,
   //    I_2 = (1/2)*(||J||_F^4-||J J^t||_F^2) = (1/2)*(I_1^2-||J J^t||_F^2),
   //    \bar{I}_2 = det(J)^{-4/3}*I_2,
   //    I_3 = det(J)^2, \bar{I}_3 = det(J).
   scalar_t I1, I1b, I2, I2b, I3b;
   scalar_t I3b_p; // I3b^{-2/3}

   // Derivatives of I1, I1b, I2, I2b, I3, and I3b using column-major storage.
   scalar_t dI1[9], dI1b[9], dI2[9], dI2b[9], dI3[9], dI3b[9];
   scalar_t B[6]; // B = J J^t (diagonal entries first, then off-diagonal)

   int D_height, alloc_height;
   const scalar_t *D; // Always points to external data or is empty
   scalar_t *DaJ, *DJt, *DdI2t, *DXt, *DYt;

   scalar_t sign_detJ;

   enum EvalMasks
   {
      HAVE_I1     = 1,
      HAVE_I1b    = 2,
      HAVE_B_offd = 4,
      HAVE_I2     = 8,
      HAVE_I2b    = 16,
      HAVE_I3b    = 1<<5,
      HAVE_I3b_p  = 1<<6,
      HAVE_dI1    = 1<<7,
      HAVE_dI1b   = 1<<8,
      HAVE_dI2    = 1<<9,
      HAVE_dI2b   = 1<<10,
      HAVE_dI3    = 1<<11,
      HAVE_dI3b   = 1<<12,
      HAVE_DaJ    = 1<<13, // D adj(J) = D dI3b^t
      HAVE_DJt    = 1<<14, // D J^t
      HAVE_DdI2t  = 1<<15  // D dI2^t
   };

   // Bitwise OR of EvalMasks
   int eval_state;

   bool dont(int have_mask) const { return !(eval_state & have_mask); }

   void Eval_I1()
   {
      eval_state |= HAVE_I1;
      B[0] = J[0]*J[0] + J[3]*J[3] + J[6]*J[6];
      B[1] = J[1]*J[1] + J[4]*J[4] + J[7]*J[7];
      B[2] = J[2]*J[2] + J[5]*J[5] + J[8]*J[8];
      I1 = B[0] + B[1] + B[2];
   }
   void Eval_I1b() // det(J)^{-2/3}*I_1 = I_1/I_3^{1/3}
   {
      eval_state |= HAVE_I1b;
      I1b = Get_I1()*Get_I3b_p();
   }
   void Eval_B_offd()
   {
      eval_state |= HAVE_B_offd;
      // B = J J^t
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      B[3] = J[0]*J[1] + J[3]*J[4] + J[6]*J[7]; // B(0,1)
      B[4] = J[0]*J[2] + J[3]*J[5] + J[6]*J[8]; // B(0,2)
      B[5] = J[1]*J[2] + J[4]*J[5] + J[7]*J[8]; // B(1,2)
   }
   void Eval_I2()
   {
      eval_state |= HAVE_I2;
      Get_I1();
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      const scalar_t BF2 = B[0]*B[0] + B[1]*B[1] + B[2]*B[2] +
                           2*(B[3]*B[3] + B[4]*B[4] + B[5]*B[5]);
      I2 = (I1*I1 - BF2)/2;
   }
   void Eval_I2b() // I2b = I2*I3b^{-4/3}
   {
      eval_state |= HAVE_I2b;
      Get_I3b_p();
      I2b = Get_I2()*I3b_p*I3b_p;
   }
   void Eval_I3b() // det(J)
   {
      eval_state |= HAVE_I3b;
      I3b = J[0]*(J[4]*J[8] - J[7]*J[5]) - J[1]*(J[3]*J[8] - J[5]*J[6]) +
            J[2]*(J[3]*J[7] - J[4]*J[6]);
      sign_detJ = scalar_ops::sign(I3b);
      I3b = sign_detJ*I3b;
   }
   scalar_t Get_I3b_p()  // I3b^{-2/3}
   {
      if (dont(HAVE_I3b_p))
      {
         eval_state |= HAVE_I3b_p;
         I3b_p = sign_detJ*scalar_ops::pow(Get_I3b(), -2, 3);
      }
      return I3b_p;
   }
   void Eval_dI1()
   {
      eval_state |= HAVE_dI1;
      for (int i = 0; i < 9; i++)
      {
         dI1[i] = 2*J[i];
      }
   }
   void Eval_dI1b()
   {
      eval_state |= HAVE_dI1b;
      // I1b = I3b^{-2/3}*I1
      // dI1b = 2*I3b^{-2/3}*(J - (1/3)*I1/I3b*dI3b)
      const scalar_t c1 = 2*Get_I3b_p();
      const scalar_t c2 = Get_I1()/(3*I3b);
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI1b[i] = c1*(J[i] - c2*dI3b[i]);
      }
   }
   void Eval_dI2()
   {
      eval_state |= HAVE_dI2;
      // dI2 = 2 I_1 J - 2 J J^t J = 2 (I_1 I - B) J
      Get_I1();
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      // B[0]=B(0,0), B[1]=B(1,1), B[2]=B(2,2)
      // B[3]=B(0,1), B[4]=B(0,2), B[5]=B(1,2)
      const scalar_t C[6] =
      {
         2*(I1 - B[0]), 2*(I1 - B[1]), 2*(I1 - B[2]),
         -2*B[3], -2*B[4], -2*B[5]
      };
      //       | C[0]  C[3]  C[4] |  | J[0]  J[3]  J[6] |
      // dI2 = | C[3]  C[1]  C[5] |  | J[1]  J[4]  J[7] |
      //       | C[4]  C[5]  C[2] |  | J[2]  J[5]  J[8] |
      dI2[0] = C[0]*J[0] + C[3]*J[1] + C[4]*J[2];
      dI2[1] = C[3]*J[0] + C[1]*J[1] + C[5]*J[2];
      dI2[2] = C[4]*J[0] + C[5]*J[1] + C[2]*J[2];

      dI2[3] = C[0]*J[3] + C[3]*J[4] + C[4]*J[5];
      dI2[4] = C[3]*J[3] + C[1]*J[4] + C[5]*J[5];
      dI2[5] = C[4]*J[3] + C[5]*J[4] + C[2]*J[5];

      dI2[6] = C[0]*J[6] + C[3]*J[7] + C[4]*J[8];
      dI2[7] = C[3]*J[6] + C[1]*J[7] + C[5]*J[8];
      dI2[8] = C[4]*J[6] + C[5]*J[7] + C[2]*J[8];
   }
   void Eval_dI2b()
   {
      eval_state |= HAVE_dI2b;
      // I2b = det(J)^{-4/3}*I2 = I3b^{-4/3}*I2
      // dI2b = (-4/3)*I3b^{-7/3}*I2*dI3b + I3b^{-4/3}*dI2
      //      = I3b^{-4/3} * [ dI2 - (4/3)*I2/I3b*dI3b ]
      Get_I3b_p();
      const scalar_t c1 = I3b_p*I3b_p;
      const scalar_t c2 = (4*Get_I2()/I3b)/3;
      Get_dI2();
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI2b[i] = c1*(dI2[i] - c2*dI3b[i]);
      }
   }
   void Eval_dI3()
   {
      eval_state |= HAVE_dI3;
      // I3 = I3b^2
      // dI3 = 2*I3b*dI3b = 2*det(J)*adj(J)^T
      const scalar_t c1 = 2*Get_I3b();
      Get_dI3b();
      for (int i = 0; i < 9; i++)
      {
         dI3[i] = c1*dI3b[i];
      }
   }
   void Eval_dI3b()
   {
      eval_state |= HAVE_dI3b;
      // I3b = det(J)
      // dI3b = adj(J)^T
      dI3b[0] = sign_detJ*(J[4]*J[8] - J[5]*J[7]);  // 0  3  6
      dI3b[1] = sign_detJ*(J[5]*J[6] - J[3]*J[8]);  // 1  4  7
      dI3b[2] = sign_detJ*(J[3]*J[7] - J[4]*J[6]);  // 2  5  8
      dI3b[3] = sign_detJ*(J[2]*J[7] - J[1]*J[8]);
      dI3b[4] = sign_detJ*(J[0]*J[8] - J[2]*J[6]);
      dI3b[5] = sign_detJ*(J[1]*J[6] - J[0]*J[7]);
      dI3b[6] = sign_detJ*(J[1]*J[5] - J[2]*J[4]);
      dI3b[7] = sign_detJ*(J[2]*J[3] - J[0]*J[5]);
      dI3b[8] = sign_detJ*(J[0]*J[4] - J[1]*J[3]);
   }
   void Eval_DZt(const scalar_t *Z, scalar_t **DZt_ptr)
   {
      MFEM_ASSERT(D != NULL, "");
      const int nd = D_height;
      scalar_t *DZt = *DZt_ptr;
      if (DZt == NULL) { *DZt_ptr = DZt = new scalar_t[3*alloc_height]; }
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1, i2 = i+nd*2;
         DZt[i0] = D[i0]*Z[0] + D[i1]*Z[3] + D[i2]*Z[6];
         DZt[i1] = D[i0]*Z[1] + D[i1]*Z[4] + D[i2]*Z[7];
         DZt[i2] = D[i0]*Z[2] + D[i1]*Z[5] + D[i2]*Z[8];
      }
   }
   void Eval_DaJ() // DaJ = D adj(J) = D dI3b^t
   {
      eval_state |= HAVE_DaJ;
      Get_dI3b();
      Eval_DZt(dI3b, &DaJ);
   }
   void Eval_DJt() // DJt = D J^t
   {
      eval_state |= HAVE_DJt;
      Eval_DZt(J, &DJt);
   }
   void Eval_DdI2t() // DdI2t = D dI2^t
   {
      eval_state |= HAVE_DdI2t;
      Get_dI2();
      Eval_DZt(dI2, &DdI2t);
   }

public:
   /// The Jacobian should use column-major storage.
   InvariantsEvaluator3D(const scalar_t *Jac = NULL)
      : J(Jac), D_height(), alloc_height(),
        D(), DaJ(), DJt(), DdI2t(), DXt(), DYt(), eval_state(0) { }

   ~InvariantsEvaluator3D()
   {
      delete [] DYt;
      delete [] DXt;
      delete [] DdI2t;
      delete [] DJt;
      delete [] DaJ;
   }

   /// The Jacobian should use column-major storage.
   void SetJacobian(const scalar_t *Jac) { J = Jac; eval_state = 0; }

   /// The @a Deriv matrix is `dof x 3`, using column-major storage.
   void SetDerivativeMatrix(int height, const scalar_t *Deriv)
   {
      eval_state &= ~(HAVE_DaJ | HAVE_DJt | HAVE_DdI2t);
      if (alloc_height < height)
      {
         delete [] DYt; DYt = NULL;
         delete [] DXt; DXt = NULL;
         delete [] DdI2t; DdI2t = NULL;
         delete [] DJt; DJt = NULL;
         delete [] DaJ; DaJ = NULL;
         alloc_height = height;
      }
      D_height = height;
      D = Deriv;
   }

   scalar_t Get_I1()  { if (dont(HAVE_I1 )) { Eval_I1();  } return I1; }
   scalar_t Get_I1b() { if (dont(HAVE_I1b)) { Eval_I1b(); } return I1b; }
   scalar_t Get_I2()  { if (dont(HAVE_I2 )) { Eval_I2();  } return I2; }
   scalar_t Get_I2b() { if (dont(HAVE_I2b)) { Eval_I2b(); } return I2b; }
   scalar_t Get_I3()  { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b*I3b; }
   scalar_t Get_I3b() { if (dont(HAVE_I3b)) { Eval_I3b(); } return I3b; }

   const scalar_t *Get_dI1()
   {
      if (dont(HAVE_dI1 )) { Eval_dI1();  } return dI1;
   }
   const scalar_t *Get_dI1b()
   {
      if (dont(HAVE_dI1b)) { Eval_dI1b(); } return dI1b;
   }
   const scalar_t *Get_dI2()
   {
      if (dont(HAVE_dI2)) { Eval_dI2(); } return dI2;
   }
   const scalar_t *Get_dI2b()
   {
      if (dont(HAVE_dI2b)) { Eval_dI2b(); } return dI2b;
   }
   const scalar_t *Get_dI3()
   {
      if (dont(HAVE_dI3)) { Eval_dI3(); } return dI3;
   }
   const scalar_t *Get_dI3b()
   {
      if (dont(HAVE_dI3b)) { Eval_dI3b(); } return dI3b;
   }

   // Assemble operation for tensor X with components X_jslt:
   //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is X_jslt D_kt)
   //    0 <= i,k < nd,  0 <= j,l,s,t < 3
   // where nd is the height of D, i.e. the number of DOFs in one component.

   void Assemble_ddI1(scalar_t w, scalar_t *A)
   {
      // ddI1_jslt = 2 I_jslt = 2 δ_jl δ_st
      //    A(i+nd*j,k+nd*l) += (\sum_st  2 w D_is δ_jl δ_st D_kt)
      // or
      //    A(i+nd*j,k+nd*l) += (2 w) (\sum_s  D_is D_ks) δ_jl
      //    A(i+nd*j,k+nd*l) += (2 w) (D D^t)_ik δ_jl

      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t a = 2*w;
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1, i2 = i+nd*2;
         const scalar_t aDi[3] = { a*D[i0], a*D[i1], a*D[i2] };
         // k == i
         const scalar_t aDDt_ii = aDi[0]*D[i0] + aDi[1]*D[i1] + aDi[2]*D[i2];
         A[i0+ah*i0] += aDDt_ii;
         A[i1+ah*i1] += aDDt_ii;
         A[i2+ah*i2] += aDDt_ii;
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const int k0 = k+nd*0, k1 = k+nd*1, k2 = k+nd*2;
            const scalar_t aDDt_ik = aDi[0]*D[k0] + aDi[1]*D[k1] + aDi[2]*D[k2];
            A[i0+ah*k0] += aDDt_ik;
            A[k0+ah*i0] += aDDt_ik;
            A[i1+ah*k1] += aDDt_ik;
            A[k1+ah*i1] += aDDt_ik;
            A[i2+ah*k2] += aDDt_ik;
            A[k2+ah*i2] += aDDt_ik;
         }
      }
   }
   void Assemble_ddI1b(scalar_t w, scalar_t *A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI1b():
      //
      // ddI1b = X1 + X2 + X3, where
      // X1_ijkl = (2/3*I1b/I3) [ (2/3 δ_ks δ_it + δ_kt δ_si) dI3b_tj dI3b_sl ]
      //         = (2/3*I1b/I3) [ 2/3 dI3b_ij dI3b_kl + dI3b_kj dI3b_il ]
      // X2_ijkl = (2*I3b^{-2/3}) δ_ik δ_jl = (I3b^{-2/3}) ddI1_ijkl
      // X3_ijkl = -(4/3*I3b^{-5/3}) (δ_ks δ_it) (J_tj dI3b_sl + dI3b_tj J_sl)
      //         = -(4/3*I3b^{-5/3}) (J_ij dI3b_kl + dI3b_ij J_kl)
      //
      //    A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI1b_jslt D_kt)
      // or
      //    A(i+nd*j,k+nd*l) +=
      //       w (2/3*I1b/I3) [ 2/3 DaJ_ij DaJ_kl + DaJ_il DaJ_kj ]
      //     + w (2*I3b^{-2/3}) (D D^t)_ik δ_jl
      //     - w (4/3*I3b^{-5/3}) [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      if (dont(HAVE_DJt)) { Eval_DJt(); }
      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t r23 = scalar_t(2)/3;
      const scalar_t r53 = scalar_t(5)/3;
      const scalar_t a = r23*w*Get_I1b()/Get_I3();
      const scalar_t b = 2*w*Get_I3b_p();
      const scalar_t c = -r23*b/I3b;
      for (int i = 0; i < nd; i++)
      {
         // A1a_ik_jl = 2/3 a DaJ_ij DaJ_kl, A1b_ik_jl = a DaJ_il DaJ_kj
         // Symmetries: A1a_ik_jl = A1a_ki_lj = 2/3 A1b_ik_lj = 2/3 A1b_ki_jl
         //    A1_ik_jl = A1_ki_lj =     A1b_ik_jl + 2/3 A1b_ik_lj
         //    A1_ik_lj = A1_ki_jl = 2/3 A1b_ik_jl +     A1b_ik_lj
         // k == i:
         //    A1_ii_jl = A1_ii_lj = (5/3) a DaJ_ij DaJ_il
         // l == j:
         //    A1_ik_jj = A1_ki_jj = (5/3) a DaJ_ij DaJ_kj
         // k == i && l == j:
         //    A1_ii_jj = (5/3) a DaJ_ij^2

         // A2_ik_jl = b (D D^t)_ik δ_jl
         // Symmetries:

         // A3_ik_jl = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
         // Symmetries:
         //    A3_ik_jl = A3_ki_lj = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
         //    A3_ik_lj = A3_ki_jl = c [ DJt_il DaJ_kj + DaJ_il DJt_kj ]
         // k == i:
         //    A3_ii_jl = A3_ii_lj = c [ DJt_ij DaJ_il + DaJ_ij DJt_il ]
         // l == j:
         //    A3_ik_jj = A3_ki_jj =  c [ DJt_ij DaJ_kj + DaJ_ij DJt_kj ]
         // k == i && l == j:
         //    A3_ii_jj = 2 c DJt_ij DaJ_ij

         const int i0 = i+nd*0, i1 = i+nd*1, i2 = i+nd*2;
         const scalar_t aDaJ_i[3] = { a*DaJ[i0], a*DaJ[i1], a*DaJ[i2] };
         const scalar_t bD_i[3] = { b*D[i0], b*D[i1], b*D[i2] };
         const scalar_t cDJt_i[3] = { c*DJt[i0], c*DJt[i1], c*DJt[i2] };
         const scalar_t cDaJ_i[3] = { c*DaJ[i0], c*DaJ[i1], c*DaJ[i2] };
         // k == i
         {
            // Symmetries: A2_ii_00 = A2_ii_11 = A2_ii_22
            const scalar_t A2_ii = bD_i[0]*D[i0]+bD_i[1]*D[i1]+bD_i[2]*D[i2];
            A[i0+ah*i0] += (r53*aDaJ_i[0] + 2*cDJt_i[0])*DaJ[i0] + A2_ii;
            A[i1+ah*i1] += (r53*aDaJ_i[1] + 2*cDJt_i[1])*DaJ[i1] + A2_ii;
            A[i2+ah*i2] += (r53*aDaJ_i[2] + 2*cDJt_i[2])*DaJ[i2] + A2_ii;

            // Symmetries: A_ii_jl = A_ii_lj
            for (int j = 1; j < 3; j++)
            {
               const int ij = i+nd*j;
               for (int l = 0; l < j; l++)
               {
                  const int il = i+nd*l;
                  const scalar_t A_ii_jl =
                     (r53*aDaJ_i[j] + cDJt_i[j])*DaJ[il] + cDaJ_i[j]*DJt[il];
                  A[ij+ah*il] += A_ii_jl;
                  A[il+ah*ij] += A_ii_jl;
               }
            }
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const int k0 = k+nd*0, k1 = k+nd*1, k2 = k+nd*2;
            // Symmetries: A2_ik_jj = A2_ki_ll
            const scalar_t A2_ik = bD_i[0]*D[k0]+bD_i[1]*D[k1]+bD_i[2]*D[k2];

            // l == j
            for (int j = 0; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               const scalar_t A_ik_jj = (r53*aDaJ_i[j] + cDJt_i[j])*DaJ[kj] +
                                        cDaJ_i[j]*DJt[kj] + A2_ik;
               A[ij+ah*kj] += A_ik_jj;
               A[kj+ah*ij] += A_ik_jj;
            }

            // 0 <= l < j
            for (int j = 1; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               for (int l = 0; l < j; l++)
               {
                  const int il = i+nd*l, kl = k+nd*l;
                  // A1b_ik_jl = a DaJ_il DaJ_kj
                  const scalar_t A1b_ik_jl = aDaJ_i[l]*DaJ[kj];
                  const scalar_t A1b_ik_lj = aDaJ_i[j]*DaJ[kl];
                  // A1_ik_jl = A1_ki_lj =     A1b_ik_jl + 2/3 A1b_ik_lj
                  // A1_ik_lj = A1_ki_jl = 2/3 A1b_ik_jl +     A1b_ik_lj
                  // A3_ik_jl = c [ DJt_ij DaJ_kl + DaJ_ij DJt_kl ]
                  const scalar_t A_ik_jl = A1b_ik_jl + r23*A1b_ik_lj +
                                           cDJt_i[j]*DaJ[kl]+cDaJ_i[j]*DJt[kl];
                  A[ij+ah*kl] += A_ik_jl;
                  A[kl+ah*ij] += A_ik_jl;
                  const scalar_t A_ik_lj = r23*A1b_ik_jl + A1b_ik_lj +
                                           cDJt_i[l]*DaJ[kj]+cDaJ_i[l]*DJt[kj];
                  A[il+ah*kj] += A_ik_lj;
                  A[kj+ah*il] += A_ik_lj;
               }
            }
         }
      }
   }
   void Assemble_ddI2(scalar_t w, scalar_t *A)
   {
      // dI2 = 2 I_1 J - 2 J J^t J = 2 (I_1 I - B) J
      //
      // ddI2 = X1 + X2 + X3
      //    X1_ijkl = (2 I_1) δ_ik δ_jl
      //    X2_ijkl = 2 ( 2 δ_ku δ_iv - δ_ik δ_uv - δ_kv δ_iu ) J_vj J_ul
      //    X3_ijkl = -2 (J J^t)_ik δ_jl = -2 B_ik δ_jl
      //
      // Apply: j->s, i->j, l->t, k->l
      //    X2_jslt = 2 ( δ_lu δ_jv - δ_jl δ_uv +
      //                  δ_lu δ_jv - δ_lv δ_ju ) J_vs J_ut
      //
      // A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2_jslt D_kt)
      //
      // \sum_st  w D_is X1_jslt D_kt =
      //    \sum_st  w D_is [ (2 I_1) δ_jl δ_st ] D_kt =
      //    (2 w I_1) D_is δ_jl D_ks = (2 w I_1) (D D^t)_ik δ_jl
      //
      // \sum_st  w D_is X2_jslt D_kt =
      //    \sum_stuv  w D_is [ 2 ( δ_lu δ_jv - δ_jl δ_uv +
      //                            δ_lu δ_jv - δ_lv δ_ju ) J_vs J_ut ] D_kt =
      //    \sum_uv  2 w [ δ_lu δ_jv - δ_jl δ_uv +
      //                   δ_lu δ_jv - δ_lv δ_ju ] (D J^t)_iv (D J^t)_ku =
      //    2 w ( DJt_ij DJt_kl - δ_jl (DJt DJt^t)_ik ) +
      //    2 w ( DJt_ij DJt_kl - DJt_il DJt_kj )
      //
      // \sum_st  w D_is X3_jslt D_kt = \sum_st  w D_is [ -2 B_jl δ_st ] D_kt =
      //    -2 w (D D^t)_ik B_jl
      //
      // A(i+nd*j,k+nd*l) +=
      //    (2 w I_1) (D D^t)_ik δ_jl - 2 w (D D^t)_ik B_jl +
      //    2 w DJt_ij DJt_kl - 2 w (DJt DJt^t)_ik δ_jl +
      //    2 w ( DJt_ij DJt_kl - DJt_il DJt_kj )
      //
      // The last term is a determinant: rows {i,k} and columns {j,l} of DJt:
      //    | DJt_ij  DJt_il |
      //    | DJt_kj  DJt_kl | = DJt_ij DJt_kl - DJt_il DJt_kj

      if (dont(HAVE_DJt)) { Eval_DJt(); }
      Get_I1(); // evaluates I1 and the diagonal of B
      if (dont(HAVE_B_offd)) { Eval_B_offd(); }
      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t a = 2*w;
      for (int i = 0; i < ah; i++)
      {
         const scalar_t avi = a*DJt[i];
         A[i+ah*i] += avi*DJt[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t aVVt_ij = avi*DJt[j];
            A[i+ah*j] += aVVt_ij;
            A[j+ah*i] += aVVt_ij;
         }
      }

      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1, i2 = i+nd*2;
         const scalar_t aD_i[3] = { a*D[i0], a*D[i1], a*D[i2] };
         const scalar_t aDJt_i[3] = { a*DJt[i0], a*DJt[i1], a*DJt[i2] };
         // k == i
         {
            const scalar_t aDDt_ii =
               aD_i[0]*D[i0] + aD_i[1]*D[i1] + aD_i[2]*D[i2];
            const scalar_t Z1_ii =
               I1*aDDt_ii - (aDJt_i[0]*DJt[i0] + aDJt_i[1]*DJt[i1] +
                             aDJt_i[2]*DJt[i2]);
            // l == j
            for (int j = 0; j < 3; j++)
            {
               const int ij = i+nd*j;
               A[ij+ah*ij] += Z1_ii - aDDt_ii*B[j];
            }
            // l != j
            const scalar_t Z2_ii_01 = aDDt_ii*B[3];
            const scalar_t Z2_ii_02 = aDDt_ii*B[4];
            const scalar_t Z2_ii_12 = aDDt_ii*B[5];
            A[i0+ah*i1] -= Z2_ii_01;
            A[i1+ah*i0] -= Z2_ii_01;
            A[i0+ah*i2] -= Z2_ii_02;
            A[i2+ah*i0] -= Z2_ii_02;
            A[i1+ah*i2] -= Z2_ii_12;
            A[i2+ah*i1] -= Z2_ii_12;
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            const int k0 = k+nd*0, k1 = k+nd*1, k2 = k+nd*2;
            const scalar_t aDDt_ik =
               aD_i[0]*D[k0] + aD_i[1]*D[k1] + aD_i[2]*D[k2];
            const scalar_t Z1_ik =
               I1*aDDt_ik - (aDJt_i[0]*DJt[k0] + aDJt_i[1]*DJt[k1] +
                             aDJt_i[2]*DJt[k2]);
            // l == j
            for (int j = 0; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               const scalar_t Z2_ik_jj = Z1_ik - aDDt_ik*B[j];
               A[ij+ah*kj] += Z2_ik_jj;
               A[kj+ah*ij] += Z2_ik_jj;
            }
            // l != j
            {
               const scalar_t Z2_ik_01 = aDDt_ik*B[3];
               A[i0+ah*k1] -= Z2_ik_01;
               A[i1+ah*k0] -= Z2_ik_01;
               A[k0+ah*i1] -= Z2_ik_01;
               A[k1+ah*i0] -= Z2_ik_01;
               const scalar_t Z2_ik_02 = aDDt_ik*B[4];
               A[i0+ah*k2] -= Z2_ik_02;
               A[i2+ah*k0] -= Z2_ik_02;
               A[k0+ah*i2] -= Z2_ik_02;
               A[k2+ah*i0] -= Z2_ik_02;
               const scalar_t Z2_ik_12 = aDDt_ik*B[5];
               A[i1+ah*k2] -= Z2_ik_12;
               A[i2+ah*k1] -= Z2_ik_12;
               A[k1+ah*i2] -= Z2_ik_12;
               A[k2+ah*i1] -= Z2_ik_12;
            }
            // 0 <= l < j
            for (int j = 1; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               for (int l = 0; l < j; l++)
               {
                  const int il = i+nd*l, kl = k+nd*l;
                  const scalar_t Z3_ik_jl =
                     aDJt_i[j]*DJt[kl] - aDJt_i[l]*DJt[kj];
                  A[ij+ah*kl] += Z3_ik_jl;
                  A[kl+ah*ij] += Z3_ik_jl;
                  A[il+ah*kj] -= Z3_ik_jl;
                  A[kj+ah*il] -= Z3_ik_jl;
               }
            }
         }
      }
   }
   void Assemble_ddI2b(scalar_t w, scalar_t *A)
   {
      // dI2b = (-4/3)*I3b^{-7/3}*I2*dI3b + I3b^{-4/3}*dI2
      //      = I3b^{-4/3} * [ dI2 - (4/3)*I2/I3b*dI3b ]
      //
      // ddI2b = X1 + X2 + X3
      //    X1_ijkl = 16/9 det(J)^{-10/3} I2 dI3b_ij dI3b_kl +
      //               4/3 det(J)^{-10/3} I2 dI3b_il dI3b_kj
      //    X2_ijkl = -4/3 det(J)^{-7/3} (dI2_ij dI3b_kl + dI2_kl dI3b_ij)
      //    X3_ijkl =      det(J)^{-4/3} ddI2_ijkl
      //
      // Apply: j->s, i->j, l->t, k->l
      //    X1_jslt = 16/9 det(J)^{-10/3} I2 dI3b_js dI3b_lt +
      //               4/3 det(J)^{-10/3} I2 dI3b_jt dI3b_ls
      //    X2_jslt = -4/3 det(J)^{-7/3} (dI2_js dI3b_lt + dI2_lt dI3b_js)
      //
      // A(i+nd*j,k+nd*l) += (\sum_st  w D_is ddI2b_jslt D_kt)
      //
      // (\sum_st  w D_is X1_jslt D_kt) =
      //    16/9 w det(J)^{-10/3} I2 DaJ_ij DaJ_kl +
      //     4/3 w det(J)^{-10/3} I2 DaJ_il DaJ_kj
      //
      // (\sum_st  w D_is X1_jslt D_kt) =
      //    -4/3 w det(J)^{-7/3} D_is (dI2_js dI3b_lt + dI2_lt dI3b_js) D_kt =
      //    -4/3 w det(J)^{-7/3} [ (D dI2^t)_ij DaJ_kl + DaJ_ij (D dI2^t)_kl ]
      //
      // A(i+nd*j,k+nd*l) +=
      //    16/9 w det(J)^{-10/3} I2 DaJ_ij DaJ_kl +
      //     4/3 w det(J)^{-10/3} I2 DaJ_il DaJ_kj -
      //     4/3 w det(J)^{-7/3} [ DdI2t_ij DaJ_kl + DaJ_ij DdI2t_kl ] +
      //         w det(J)^{-4/3} D_is D_kt ddI2_jslt

      Get_I3b_p(); // = det(J)^{-2/3}, evaluates I3b
      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      if (dont(HAVE_DdI2t)) { Eval_DdI2t(); }
      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t a = w*I3b_p*I3b_p;
      const scalar_t b = (-4*a)/(3*I3b);
      const scalar_t c = -b*Get_I2()/I3b;
      const scalar_t d = (4*c)/3;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t dvi = d*DaJ[i];
         A[i+ah*i] += dvi*DaJ[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t dVVt_ij = dvi*DaJ[j];
            A[i+ah*j] += dVVt_ij;
            A[j+ah*i] += dVVt_ij;
         }
      }
      Assemble_ddI2(a, A);
      for (int i = 0; i < nd; i++)
      {
         const int i0 = i+nd*0, i1 = i+nd*1, i2 = i+nd*2;
         const scalar_t cDaJ_i[3] = { c*DaJ[i0], c*DaJ[i1], c*DaJ[i2] };
         const scalar_t bDaJ_i[3] = { b*DaJ[i0], b*DaJ[i1], b*DaJ[i2] };
         const scalar_t bDdI2t_i[3] = { b*DdI2t[i0], b*DdI2t[i1], b*DdI2t[i2] };
         // k == i
         {
            // l == j
            for (int j = 0; j < 3; j++)
            {
               const int ij = i+nd*j;
               A[ij+ah*ij] += (cDaJ_i[j] + 2*bDdI2t_i[j])*DaJ[ij];
            }
            // 0 <= l < j
            for (int j = 1; j < 3; j++)
            {
               const int ij = i+nd*j;
               for (int l = 0; l < j; l++)
               {
                  const int il = i+nd*l;
                  const scalar_t Z_ii_jl =
                     (cDaJ_i[l] + bDdI2t_i[l])*DaJ[ij] + bDdI2t_i[j]*DaJ[il];
                  A[ij+ah*il] += Z_ii_jl;
                  A[il+ah*ij] += Z_ii_jl;
               }
            }
         }
         // 0 <= k < i
         for (int k = 0; k < i; k++)
         {
            // l == j
            for (int j = 0; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               const scalar_t Z_ik_jj =
                  (cDaJ_i[j] + bDdI2t_i[j])*DaJ[kj] + bDaJ_i[j]*DdI2t[kj];
               A[ij+ah*kj] += Z_ik_jj;
               A[kj+ah*ij] += Z_ik_jj;
            }
            // 0 <= l < j
            for (int j = 1; j < 3; j++)
            {
               const int ij = i+nd*j, kj = k+nd*j;
               for (int l = 0; l < j; l++)
               {
                  const int il = i+nd*l, kl = k+nd*l;
                  const scalar_t Z_ik_jl = cDaJ_i[l]*DaJ[kj] +
                                           bDdI2t_i[j]*DaJ[kl] +
                                           bDaJ_i[j]*DdI2t[kl];
                  A[ij+ah*kl] += Z_ik_jl;
                  A[kl+ah*ij] += Z_ik_jl;
                  const scalar_t Z_ik_lj = cDaJ_i[j]*DaJ[kl] +
                                           bDdI2t_i[l]*DaJ[kj] +
                                           bDaJ_i[l]*DdI2t[kj];
                  A[il+ah*kj] += Z_ik_lj;
                  A[kj+ah*il] += Z_ik_lj;
               }
            }
         }
      }
   }
   void Assemble_ddI3(scalar_t w, scalar_t *A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI2():
      //
      //    A(i+nd*j,k+nd*l) += 2 w [ 2 DaJ_ij DaJ_kl - DaJ_il DaJ_kj ]
      //
      // Note: the expression ( DaJ_ij DaJ_kl - DaJ_il DaJ_kj ) is the
      // determinant of the 2x2 matrix formed by rows {i,k} and columns {j,l}
      // from the matrix DaJ = D dI3b^t.

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t a = 2*w;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t avi = a*DaJ[i];
         A[i+ah*i] += avi*DaJ[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t aVVt_ij = avi*DaJ[j];
            A[i+ah*j] += aVVt_ij;
            A[j+ah*i] += aVVt_ij;
         }
      }
      for (int j = 1; j < 3; j++)
      {
         for (int l = 0; l < j; l++)
         {
            for (int i = 0; i < nd; i++)
            {
               const int ij = i+nd*j, il = i+nd*l;
               const scalar_t aDaJ_ij = a*DaJ[ij], aDaJ_il = a*DaJ[il];
               for (int k = 0; k < i; k++)
               {
                  const int kj = k+nd*j, kl = k+nd*l;
                  const scalar_t A_ijkl = aDaJ_ij*DaJ[kl] - aDaJ_il*DaJ[kj];
                  A[ij+ah*kl] += A_ijkl;
                  A[kl+ah*ij] += A_ijkl;
                  A[kj+ah*il] -= A_ijkl;
                  A[il+ah*kj] -= A_ijkl;
               }
            }
         }
      }
   }
   void Assemble_ddI3b(scalar_t w, scalar_t *A)
   {
      // Similar to InvariantsEvaluator2D::Assemble_ddI2b():
      //
      //    A(i+nd*j,k+nd*l) += (w/I3b) [ DaJ_ij DaJ_kl - DaJ_il DaJ_kj ]
      //
      // | DaJ_ij  DaJ_il | = determinant of rows {i,k}, columns {j,l} from DaJ
      // | DaJ_kj  DaJ_kl |

      if (dont(HAVE_DaJ)) { Eval_DaJ(); }
      const int nd = D_height;
      const int ah = 3*nd;
      const scalar_t a = w/Get_I3b();
      for (int j = 1; j < 3; j++)
      {
         for (int l = 0; l < j; l++)
         {
            for (int i = 0; i < nd; i++)
            {
               const int ij = i+nd*j, il = i+nd*l;
               const scalar_t aDaJ_ij = a*DaJ[ij], aDaJ_il = a*DaJ[il];
               for (int k = 0; k < i; k++)
               {
                  const int kj = k+nd*j, kl = k+nd*l;
                  const scalar_t A_ijkl = aDaJ_ij*DaJ[kl] - aDaJ_il*DaJ[kj];
                  A[ij+ah*kl] += A_ijkl;
                  A[kl+ah*ij] += A_ijkl;
                  A[kj+ah*il] -= A_ijkl;
                  A[il+ah*kj] -= A_ijkl;
               }
            }
         }
      }
   }
   // Assemble the contribution from the term: T_ijkl = X_ij Y_kl + Y_ij X_kl,
   // where X and Y are pointers to 3x3 matrices stored in column-major layout.
   //
   // The contribution to the matrix A is given by:
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is T_jslt D_kt
   // or
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is (X_js Y_lt + Y_js X_lt) D_kt
   // or
   //    A(i+nd*j,k+nd*l) +=
   //       \sum_st  w [ (D X^t)_ij (D Y^t)_kl + (D Y^t)_ij (D X^t)_kl ]
   void Assemble_TProd(scalar_t w, const scalar_t *X, const scalar_t *Y,
                       scalar_t *A)
   {
      Eval_DZt(X, &DXt);
      Eval_DZt(Y, &DYt);
      const int nd = D_height;
      const int ah = 3*nd;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t axi = w*DXt[i], ayi = w*DYt[i];
         A[i+ah*i] += 2*axi*DYt[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t A_ij = axi*DYt[j] + ayi*DXt[j];
            A[i+ah*j] += A_ij;
            A[j+ah*i] += A_ij;
         }
      }
   }
   // Assemble the contribution from the term: T_ijkl = X_ij X_kl, where X is a
   // pointer to a 3x3 matrix stored in column-major layout.
   //
   // The contribution to the matrix A is given by:
   //    A(i+nd*j,k+nd*l) += \sum_st  w D_is X_js X_lt D_kt
   // or
   //    A(i+nd*j,k+nd*l) += \sum_st  w [ (D X^t)_ij (D X^t)_kl ]
   void Assemble_TProd(scalar_t w, const scalar_t *X, scalar_t *A)
   {
      Eval_DZt(X, &DXt);
      const int nd = D_height;
      const int ah = 3*nd;

      for (int i = 0; i < ah; i++)
      {
         const scalar_t axi = w*DXt[i];
         A[i+ah*i] += axi*DXt[i];
         for (int j = 0; j < i; j++)
         {
            const scalar_t A_ij = axi*DXt[j];
            A[i+ah*j] += A_ij;
            A[j+ah*i] += A_ij;
         }
      }
   }
};

}

#endif
