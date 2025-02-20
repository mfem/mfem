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

#ifndef MFEM_TEMPLATE_BILININTEG
#define MFEM_TEMPLATE_BILININTEG

#include "../config/tconfig.hpp"
#include "tcoefficient.hpp"
#include "tbilinearform.hpp"

namespace mfem
{

// Templated local bilinear form integrator kernels, cf. bilininteg.?pp

/// The Integrator class combines a kernel and a coefficient
template <typename coeff_t, template<int,int,typename> class kernel_t>
class TIntegrator
{
public:
   typedef coeff_t coefficient_type;

   template <int SDim, int Dim, typename complex_t>
   struct kernel { typedef kernel_t<SDim,Dim,complex_t> type; };

   coeff_t coeff;

   TIntegrator(const coefficient_type &c) : coeff(c) { }
};


/// Mass kernel
template <int SDim, int Dim, typename complex_t>
struct TMassKernel
{
   typedef complex_t complex_type;

   /// Needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   /// @name Needed for the FieldEvaluator::Data class
   ///@{
   static const bool in_values     = true;
   static const bool in_gradients  = false;
   static const bool out_values    = true;
   static const bool out_gradients = false;
   ///@}

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in partial assembly, and partially
       assembled action. */
   template <int qpts>
   struct p_asm_data { typedef TVector<qpts,complex_t> type; };

   /** @brief Partially assembled data type for one element with the given
       number of quadrature points. This type is used in full element matrix
       assembly. */
   template <int qpts>
   struct f_asm_data { typedef TVector<qpts,complex_t> type; };

   template <typename IR, typename coeff_t, typename impl_traits_t>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,impl_traits_t>::Type Type;
   };

   /** @brief Method used for un-assembled (matrix free) action.
       @param k the element number
       @param F  Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q  CoefficientEval<>::Type
       @param q  CoefficientEval<>::Type::result_t
       @param R  val_qpts [M x NC x NE] - in/out data member in R
       val_qpts *= w det(J) */
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      typedef typename T_result_t::Jt_type::data_type real_t_;
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M*(1+NC)); // TDet counts its flops
      for (int i = 0; i < M; i++)
      {
         const complex_t wi =
            Q.get(q,i,k) * TDet<real_t_>(F.Jt.layout.ind14(i,k), F.Jt);
         for (int j = 0; j < NC; j++)
         {
            R.val_qpts(i,j,k) *= wi;
         }
      }
   }

   /** @brief Method defining partial assembly.
       Result in A is the quadrature-point dependent part of element matrix
       assembly (as opposed to part that is same for all elements),
       A = w det(J)
       @param k the element number
       @param F Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q CoefficientEval<>::Type
       @param q CoefficientEval<>::Type::result_t
       @param A [M] - partially assembled scalars
   */
   template <typename T_result_t, typename Q_t, typename q_t, int qpts>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, TVector<qpts,complex_t> &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t_;

      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M); // TDet counts its flops
      for (int i = 0; i < M; i++)
      {
         A[i] = Q.get(q,i,k) * TDet<real_t_>(F.Jt.layout.ind14(i,k), F.Jt);
      }
   }

   /** @brief Method for partially assembled action.
       @param k the element number
       @param A  [M] - partially assembled scalars
       @param R  val_qpts [M x NC x NE] - in/out data member in R
       val_qpts *= A
   */
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TVector<qpts,complex_t> &A, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M*NC);
      for (int i = 0; i < M; i++)
      {
         for (int j = 0; j < NC; j++)
         {
            R.val_qpts(i,j,k) *= A[i];
         }
      }
   }
};


/** @brief Diffusion kernel
    @tparam complex_t - type for the assembled data
*/
template <int SDim, int Dim, typename complex_t>
struct TDiffusionKernel;

/// Diffusion kernel in 1D
template <typename complex_t>
struct TDiffusionKernel<1,1,complex_t>
{
   typedef complex_t complex_type;

   /// Needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   /// Needed for the FieldEvaluator::Data class
   ///@{
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;
   ///@}

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in partial assembly, and partially
       assembled action. */
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,1,complex_t> type; };


   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in full element matrix assembly. */
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,1,1,complex_t> type; };

   template <typename IR, typename coeff_t, typename impl_traits_t>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,impl_traits_t>::Type Type;
   };

   /** @brief Method used for un-assembled (matrix free) action.
       @param k the element number
       @param F Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q - CoefficientEval<>::Type
       @param q - CoefficientEval<>::Type::result_t
       @param R grad_qpts [M x SDim x NC x NE]  - in/out data member in R
       grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts */
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M*(1+NC));
      for (int i = 0; i < M; i++)
      {
         const complex_t wi = Q.get(q,i,k) / F.Jt(i,0,0,k);
         for (int j = 0; j < NC; j++)
         {
            R.grad_qpts(i,0,j,k) *= wi;
         }
      }
   }


   /** @brief Method defining partial assembly.
       The pointwise Dim x Dim matrices are stored as symmetric (when
       asm_type == p_asm_data, i.e. A.layout.rank == 2) or
       non-symmetric (when asm_type == f_asm_data, i.e. A.layout.rank
       == 3) matrices.
       @param k the element number
       @param F Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q CoefficientEval<>::Type
       @param q CoefficientEval<>::Type::result_t
       @param A [M x Dim*(Dim+1)/2] - partially assembled Dim x Dim symm. matrices
              A [M x Dim x Dim]     - partially assembled Dim x Dim matrices
       A = (w/det(J)) adj(J) adj(J)^t
   */
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         // A[i] is A(i,0) or A(i,0,0)
         A[i] = Q.get(q,i,k) / F.Jt(i,0,0,k);
      }
   }
   /** @brief Method for partially assembled action.
       @param k the element number
       @param A  [M x Dim*(Dim+1)/2] partially assembled Dim x Dim symmetric
                                     matrices
       @param R  grad_qpts [M x SDim x NC x NE] - in/out data member in R
       grad_qpts = A grad_qpts
   */
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,1,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M*NC);
      for (int i = 0; i < M; i++)
      {
         for (int j = 0; j < NC; j++)
         {
            R.grad_qpts(i,0,j,k) *= A(i,0);
         }
      }
   }
};

/// Diffusion kernel in 2D
template <typename complex_t>
struct TDiffusionKernel<2,2,complex_t>
{
   typedef complex_t complex_type;

   /// Needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   /// Needed for the FieldEvaluator::Data class
   ///@{
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;
   ///@}

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in partial assembly, and partially
       assembled action. Stores one symmetric 2 x 2 matrix per point. */
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,3,complex_t> type; };

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in full element matrix assembly.
       Stores one general (non-symmetric) 2 x 2 matrix per point. */
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,2,2,complex_t> type; };

   template <typename IR, typename coeff_t, typename impl_traits_t>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,impl_traits_t>::Type Type;
   };

   /** @brief Method used for un-assembled (matrix free) action.
       @param k the element number
       @param F Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q CoefficientEval<>::Type
       @param q CoefficientEval<>::Type::result_t
       @param R grad_qpts [M x SDim x NC x NE]  - in/out data member in R
       grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts
   */
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M*(4+NC*14));
      for (int i = 0; i < M; i++)
      {
         typedef typename T_result_t::Jt_type::data_type real_t_;
         const real_t_ J11 = F.Jt(i,0,0,k);
         const real_t_ J12 = F.Jt(i,1,0,k);
         const real_t_ J21 = F.Jt(i,0,1,k);
         const real_t_ J22 = F.Jt(i,1,1,k);
         const complex_t w_det_J = Q.get(q,i,k) / (J11 * J22 - J21 * J12);
         for (int j = 0; j < NC; j++)
         {
            const complex_t x1 = R.grad_qpts(i,0,j,k);
            const complex_t x2 = R.grad_qpts(i,1,j,k);
            // z = adj(J)^t x
            const complex_t z1 = J22 * x1 - J21 * x2;
            const complex_t z2 = J11 * x2 - J12 * x1;
            R.grad_qpts(i,0,j,k) = w_det_J * (J22 * z1 - J12 * z2);
            R.grad_qpts(i,1,j,k) = w_det_J * (J11 * z2 - J21 * z1);
         }
      }
   }

   /** @brief Method defining partial assembly.
       The pointwise Dim x Dim matrices are stored as symmetric (when
       asm_type == p_asm_data, i.e. A.layout.rank == 2) or non-symmetric
       (when asm_type == f_asm_data, i.e. A.layout.rank == 3) matrices.
       A = (w/det(J)) adj(J) adj(J)^t
       @param k the element number
       @param F Jt [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       @param Q CoefficientEval<>::Type
       @param q CoefficientEval<>::Type::result_t
       @param A either [M x Dim*(Dim+1)/2] partially assembled Dim x Dim symm.
       matrices, or [M x Dim x Dim] partially assembled Dim x Dim matrices.
   */
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t_;
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(16*M);
      const bool Symm = (asm_type::layout_type::rank == 2);
      for (int i = 0; i < M; i++)
      {
         const real_t_ J11 = F.Jt(i,0,0,k);
         const real_t_ J12 = F.Jt(i,1,0,k);
         const real_t_ J21 = F.Jt(i,0,1,k);
         const real_t_ J22 = F.Jt(i,1,1,k);
         const complex_t w_det_J = Q.get(q,i,k) / (J11 * J22 - J21 * J12);
         internal::MatrixOps<2,2>::Symm<Symm>::Set(
            A.layout.ind1(i), A,
            + w_det_J * (J12*J12 + J22*J22), // (1,1)
            - w_det_J * (J11*J12 + J21*J22), // (2,1)
            + w_det_J * (J11*J11 + J21*J21)  // (2,2)
         );
      }
   }

   /** @brief  Method for partially assembled action.
       @param k the element number
       @param  A  [M x Dim*(Dim+1)/2]  - partially assembled Dim x Dim symmetric
                                         matrices
       @param R grad_qpts [M x SDim x NC x NE] - in/out data member in R
       grad_qpts = A grad_qpts
   */
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,3,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(6*M*NC);
      for (int i = 0; i < M; i++)
      {
         const complex_t A11 = A(i,0);
         const complex_t A21 = A(i,1);
         const complex_t A22 = A(i,2);
         for (int j = 0; j < NC; j++)
         {
            const complex_t x1 = R.grad_qpts(i,0,j,k);
            const complex_t x2 = R.grad_qpts(i,1,j,k);
            R.grad_qpts(i,0,j,k) = A11 * x1 + A21 * x2;
            R.grad_qpts(i,1,j,k) = A21 * x1 + A22 * x2;
         }
      }
   }
};

/// Diffusion kernel in 3D
template <typename complex_t>
struct TDiffusionKernel<3,3,complex_t>
{
   typedef complex_t complex_type;

   /// Needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   /// Needed for the FieldEvaluator::Data class
   ///@{
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;
   ///@}

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in partial assembly, and partially
       assembled action. Stores one symmetric 3 x 3 matrix per point. */
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,6,complex_t> type; };

   /** @brief Partially assembled data type for one element with the given number of
       quadrature points. This type is used in full element matrix assembly.
       Stores one general (non-symmetric) 3 x 3 matrix per point. */
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,3,3,complex_t> type; };

   template <typename IR, typename coeff_t, typename impl_traits_t>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,impl_traits_t>::Type Type;
   };

   /** @brief Method used for un-assembled (matrix free) action.
       grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts
       Jt        [M x Dim x SDim x NE] - Jacobian transposed, data member in F
       Q                               - CoefficientEval<>::Type
       q                               - CoefficientEval<>::Type::result_t
       grad_qpts [M x SDim x NC x NE]  - in/out data member in R
   */
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M); // just need to count Q/detJ
      for (int i = 0; i < M; i++)
      {
         typedef typename T_result_t::Jt_type::data_type real_t_;
         TMatrix<3,3,real_t_> adj_J;
         const complex_t w_det_J =
            (Q.get(q,i,k) /
             TAdjDet<real_t_>(F.Jt.layout.ind14(i,k).transpose_12(), F.Jt,
                              adj_J.layout, adj_J));
         TMatrix<3,NC,complex_t> z; // z = adj(J)^t x
         sMult_AB<false>(adj_J.layout.transpose_12(), adj_J,
                         R.grad_qpts.layout.ind14(i,k), R.grad_qpts,
                         z.layout, z);
         z.Scale(w_det_J);
         sMult_AB<false>(adj_J.layout, adj_J,
                         z.layout, z,
                         R.grad_qpts.layout.ind14(i,k), R.grad_qpts);
      }
   }

   /** @brief Method defining partial assembly.
      The pointwise Dim x Dim matrices are stored as symmetric (when
      asm_type == p_asm_data, i.e. A.layout.rank == 2) or
      non-symmetric (when asm_type == f_asm_data, i.e. A.layout.rank
      == 3) matrices.
      A = (w/det(J)) adj(J) adj(J)^t
      Jt   [M x Dim x SDim x NE] - Jacobian transposed, data member in F
      Q                          - CoefficientEval<>::Type
      q                          - CoefficientEval<>::Type::result_t
      A    [M x Dim*(Dim+1)/2]   - partially assembled Dim x Dim symm. matrices
      A    [M x Dim x Dim]       - partially assembled Dim x Dim matrices
   */
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t_;
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(37*M);
      const bool Symm = (asm_type::layout_type::rank == 2);
      for (int i = 0; i < M; i++)
      {
         TMatrix<3,3,real_t_> B; // = adj(J)
         const complex_t u =
            (Q.get(q,i,k) /
             TAdjDet<real_t_>(F.Jt.layout.ind14(i,k).transpose_12(), F.Jt,
                              B.layout, B));
         internal::MatrixOps<3,3>::Symm<Symm>::Set(
            A.layout.ind1(i), A,
            u*(B(0,0)*B(0,0)+B(0,1)*B(0,1)+B(0,2)*B(0,2)), // 1,1
            u*(B(0,0)*B(1,0)+B(0,1)*B(1,1)+B(0,2)*B(1,2)), // 2,1
            u*(B(0,0)*B(2,0)+B(0,1)*B(2,1)+B(0,2)*B(2,2)), // 3,1
            u*(B(1,0)*B(1,0)+B(1,1)*B(1,1)+B(1,2)*B(1,2)), // 2,2
            u*(B(1,0)*B(2,0)+B(1,1)*B(2,1)+B(1,2)*B(2,2)), // 3,2
            u*(B(2,0)*B(2,0)+B(2,1)*B(2,1)+B(2,2)*B(2,2))  // 3,3
         );
      }
   }

   /** @brief Method for partially assembled action.
       A         [M x Dim*(Dim+1)/2]  - partially assembled Dim x Dim symmetric
                                        matrices
       grad_qpts [M x SDim x NC x NE] - in/out data member in R
       grad_qpts = A grad_qpts
   */
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,6,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      const int NC = S_data_t::eval_type::vdim;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(15*M*NC);
      for (int i = 0; i < M; i++)
      {
         const complex_t A11 = A(i,0);
         const complex_t A21 = A(i,1);
         const complex_t A31 = A(i,2);
         const complex_t A22 = A(i,3);
         const complex_t A32 = A(i,4);
         const complex_t A33 = A(i,5);
         for (int j = 0; j < NC; j++)
         {
            const complex_t x1 = R.grad_qpts(i,0,j,k);
            const complex_t x2 = R.grad_qpts(i,1,j,k);
            const complex_t x3 = R.grad_qpts(i,2,j,k);
            R.grad_qpts(i,0,j,k) = A11*x1 + A21*x2 + A31*x3;
            R.grad_qpts(i,1,j,k) = A21*x1 + A22*x2 + A32*x3;
            R.grad_qpts(i,2,j,k) = A31*x1 + A32*x2 + A33*x3;
         }
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_BILININTEG
