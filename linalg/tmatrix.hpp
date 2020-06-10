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

#ifndef MFEM_TEMPLATE_MATRIX
#define MFEM_TEMPLATE_MATRIX

#include "../config/tconfig.hpp"
#include "../general/tassign.hpp"
#include "../general/cuda.hpp"
#include "../general/hip.hpp"

namespace mfem
{

// Matrix-matrix products

namespace internal
{

template <typename T> struct entry_type { typedef typename T::data_type type; };

template <typename T> struct entry_type<T*> { typedef T type; };

} // namespace mfem::internal


// C  {=|+=}  A.B -- simple version (no blocks)
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void sMult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 2, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   MFEM_STATIC_ASSERT(A2 == B1 && A1 == C1 && B2 == C2,
                      "invalid dimensions");

   MFEM_FLOPS_ADD(Add ? 2*A1*A2*B2 : 2*A1*A2*B2-A1*B2);
   for (int b2 = 0; b2 < B2; b2++)
   {
      for (int a1 = 0; a1 < A1; a1++)
      {
         typename internal::entry_type<C_data_t>::type c_a1_b2;
         if (Add)
         {
            // C(a1,b2) += A(a1,0) * B(0,b2);
            c_a1_b2 = C_data[C_layout.ind(a1,b2)];
            c_a1_b2.fma(A_data[A_layout.ind(a1,0)], B_data[B_layout.ind(0,b2)]);
         }
         else
         {
            // C(a1,b2) = A(a1,0) * B(0,b2);
            c_a1_b2.mul(A_data[A_layout.ind(a1,0)], B_data[B_layout.ind(0,b2)]);
         }
         for (int s = 1; s < A2; s++)
         {
            // C(a1,b2) += A(a1,s) * B(s,b2);
            c_a1_b2.fma(A_data[A_layout.ind(a1,s)], B_data[B_layout.ind(s,b2)]);
         }
         C_data[C_layout.ind(a1,b2)] = c_a1_b2;
      }
   }
}

// C  {=|+=}  A.B -- block version
template <int bA1, int bA2, int bB2, // block sizes
          bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void bMult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 2, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   MFEM_STATIC_ASSERT(A2 == B1 && A1 == C1 && B2 == C2,
                      "invalid dimensions");

   const int rA1 = A1%bA1;
   const int rA2 = A2%bA2;
   const int rB2 = B2%bB2;

   for (int b2_b = 0; b2_b < B2/bB2; b2_b++)
   {
      if (A2/bA2 > 0)
      {
         // s_b == 0
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<Add>(
               A_layout.template sub<bA1,bA2>(a1_b*bA1,0), A_data,
               B_layout.template sub<bA2,bB2>(0,b2_b*bB2), B_data,
               C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<Add>(
               A_layout.template sub<rA1,bA2>(A1-rA1,0), A_data,
               B_layout.template sub<bA2,bB2>(0,b2_b*bB2), B_data,
               C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
         }
         for (int s_b = 1; s_b < A2/bA2; s_b++)
         {
            for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
            {
               sMult_AB<true>(
                  A_layout.template sub<bA1,bA2>(a1_b*bA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,bB2>(s_b*bA2,b2_b*bB2), B_data,
                  C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
            }
            if (rA1)
            {
               sMult_AB<true>(
                  A_layout.template sub<rA1,bA2>(A1-rA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,bB2>(s_b*bA2,b2_b*bB2), B_data,
                  C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
            }
         }
      }
      if (rA2)
      {
         const bool rAdd = Add || (A2/bA2 > 0);
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<bA1,rA2>(a1_b*bA1,A2-rA2), A_data,
               B_layout.template sub<rA2,bB2>(A2-rA2,b2_b*bB2), B_data,
               C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<rA1,rA2>(A1-rA1,A2-rA2), A_data,
               B_layout.template sub<rA2,bB2>(A2-rA2,b2_b*bB2), B_data,
               C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
         }
      }
   }
   if (rB2)
   {
      if (A2/bA2 > 0)
      {
         // s_b == 0
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<Add>(
               A_layout.template sub<bA1,bA2>(a1_b*bA1,0), A_data,
               B_layout.template sub<bA2,rB2>(0,B2-rB2), B_data,
               C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<Add>(
               A_layout.template sub<rA1,bA2>(A1-rA1,0), A_data,
               B_layout.template sub<bA2,rB2>(0,B2-rB2), B_data,
               C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
         }
      }
      if (A2/bA2 > 1)
      {
         for (int s_b = 1; s_b < A2/bA2; s_b++)
         {
            for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
            {
               sMult_AB<true>(
                  A_layout.template sub<bA1,bA2>(a1_b*bA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,rB2>(s_b*bA2,B2-rB2), B_data,
                  C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
            }
            if (rA1)
            {
               sMult_AB<true>(
                  A_layout.template sub<rA1,bA2>(A1-rA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,rB2>(s_b*bA2,B2-rB2), B_data,
                  C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
            }
         }
      }
      if (rA2)
      {
         const bool rAdd = Add || (A2/bA2 > 0);
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<bA1,rA2>(a1_b*bA1,A2-rA2), A_data,
               B_layout.template sub<rA2,rB2>(A2-rA2,B2-rB2), B_data,
               C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<rA1,rA2>(A1-rA1,A2-rA2), A_data,
               B_layout.template sub<rA2,rB2>(A2-rA2,B2-rB2), B_data,
               C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
         }
      }
   }
}

template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
             const B_layout_t &B_layout, const B_data_t &B_data,
             const C_layout_t &C_layout, C_data_t &C_data)
{
   const int b = MFEM_TEMPLATE_BLOCK_SIZE;
   bMult_AB<b,b,b,Add>(A_layout, A_data, B_layout, B_data, C_layout, C_data);
}


// Small matrix operations (determinant, adjugate,...) defined by specialization

namespace internal
{

template <int N1, int N2>
struct MatrixOps { };

template <>
struct MatrixOps<1,1>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      return A[a.ind(0,0)];
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(D[i], A[a.ind(i,0,0)]);
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] = scalar_t(1);
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      Adjugate<scalar_t>(a, A, b, B);
      return Det<scalar_t>(a, A);
   }
};

template <>
struct MatrixOps<2,2>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      MFEM_FLOPS_ADD(3);
      return (A[a.ind(0,0)]*A[a.ind(1,1)] -
              A[a.ind(1,0)]*A[a.ind(0,1)]);
   }

   // Compute det(A), host+device version.
   template <typename scalar_t, typename layout_t, typename data_t>
   MFEM_HOST_DEVICE
   static inline scalar_t DetHD(const layout_t &a, const data_t &A)
   {
      MFEM_FLOPS_ADD(3);
      return (A[a.ind(0,0)]*A[a.ind(1,1)] -
              A[a.ind(1,0)]*A[a.ind(0,1)]);
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      MFEM_FLOPS_ADD(3*M);
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(D[i], (A[a.ind(i,0,0)]*A[a.ind(i,1,1)] -
                           A[a.ind(i,1,0)]*A[a.ind(i,0,1)]));
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] =  A[a.ind(1,1)];
      B[b.ind(0,1)] = -A[a.ind(0,1)];
      B[b.ind(1,0)] = -A[a.ind(1,0)];
      B[b.ind(1,1)] =  A[a.ind(0,0)];
   }

   // Compute B = adj(A), host+device version.
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   MFEM_HOST_DEVICE
   static inline void AdjugateHD(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] =  A[a.ind(1,1)];
      B[b.ind(0,1)] = -A[a.ind(0,1)];
      B[b.ind(1,0)] = -A[a.ind(1,0)];
      B[b.ind(1,1)] =  A[a.ind(0,0)];
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      Adjugate<scalar_t>(a, A, b, B);
      return Det<scalar_t>(a, A);
   }

   // Compute adj(A) and det(A), host+device version.
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   MFEM_HOST_DEVICE
   static inline scalar_t AdjDetHD(const A_layout_t &a, const A_data_t &A,
                                   const B_layout_t &b, B_data_t &B)
   {
      AdjugateHD<scalar_t>(a, A, b, B);
      return DetHD<scalar_t>(a, A);
   }

   template <bool symm> struct Symm;
};

template <>
struct MatrixOps<2,2>::Symm<true>
{
   template <typename A_layout_t, typename A_data_t, typename scalar_t>
   static inline MFEM_ALWAYS_INLINE
   void Set(const A_layout_t &a, A_data_t &A,
            const scalar_t a11, const scalar_t a21, const scalar_t a22)
   {
      A[a.ind(0)] = a11;
      A[a.ind(1)] = a21;
      A[a.ind(2)] = a22;
   }
};

template <>
struct MatrixOps<2,2>::Symm<false>
{
   template <typename A_layout_t, typename A_data_t, typename scalar_t>
   static inline MFEM_ALWAYS_INLINE
   void Set(const A_layout_t &a, A_data_t &A,
            const scalar_t a11, const scalar_t a21, const scalar_t a22)
   {
      A[a.ind(0,0)] = a11;
      A[a.ind(1,0)] = a21;
      A[a.ind(0,1)] = a21;
      A[a.ind(1,1)] = a22;
   }
};

template <>
struct MatrixOps<3,3>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      MFEM_FLOPS_ADD(14);
      return (A[a.ind(0,0)]*(A[a.ind(1,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(1,2)]) -
              A[a.ind(1,0)]*(A[a.ind(0,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(0,2)]) +
              A[a.ind(2,0)]*(A[a.ind(0,1)]*A[a.ind(1,2)] -
                             A[a.ind(1,1)]*A[a.ind(0,2)]));
   }

   // Compute det(A), host+device version.
   template <typename scalar_t, typename layout_t, typename data_t>
   MFEM_HOST_DEVICE
   static inline scalar_t DetHD(const layout_t &a, const data_t &A)
   {
      MFEM_FLOPS_ADD(14);
      return (A[a.ind(0,0)]*(A[a.ind(1,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(1,2)]) -
              A[a.ind(1,0)]*(A[a.ind(0,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(0,2)]) +
              A[a.ind(2,0)]*(A[a.ind(0,1)]*A[a.ind(1,2)] -
                             A[a.ind(1,1)]*A[a.ind(0,2)]));
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      MFEM_FLOPS_ADD(14*M);
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(
            D[i],
            A[a.ind(i,0,0)]*(A[a.ind(i,1,1)]*A[a.ind(i,2,2)] -
                             A[a.ind(i,2,1)]*A[a.ind(i,1,2)]) -
            A[a.ind(i,1,0)]*(A[a.ind(i,0,1)]*A[a.ind(i,2,2)] -
                             A[a.ind(i,2,1)]*A[a.ind(i,0,2)]) +
            A[a.ind(i,2,0)]*(A[a.ind(i,0,1)]*A[a.ind(i,1,2)] -
                             A[a.ind(i,1,1)]*A[a.ind(i,0,2)]));
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      MFEM_FLOPS_ADD(27);
      B[b.ind(0,0)] = A[a.ind(1,1)]*A[a.ind(2,2)] - A[a.ind(1,2)]*A[a.ind(2,1)];
      B[b.ind(0,1)] = A[a.ind(0,2)]*A[a.ind(2,1)] - A[a.ind(0,1)]*A[a.ind(2,2)];
      B[b.ind(0,2)] = A[a.ind(0,1)]*A[a.ind(1,2)] - A[a.ind(0,2)]*A[a.ind(1,1)];
      B[b.ind(1,0)] = A[a.ind(1,2)]*A[a.ind(2,0)] - A[a.ind(1,0)]*A[a.ind(2,2)];
      B[b.ind(1,1)] = A[a.ind(0,0)]*A[a.ind(2,2)] - A[a.ind(0,2)]*A[a.ind(2,0)];
      B[b.ind(1,2)] = A[a.ind(0,2)]*A[a.ind(1,0)] - A[a.ind(0,0)]*A[a.ind(1,2)];
      B[b.ind(2,0)] = A[a.ind(1,0)]*A[a.ind(2,1)] - A[a.ind(1,1)]*A[a.ind(2,0)];
      B[b.ind(2,1)] = A[a.ind(0,1)]*A[a.ind(2,0)] - A[a.ind(0,0)]*A[a.ind(2,1)];
      B[b.ind(2,2)] = A[a.ind(0,0)]*A[a.ind(1,1)] - A[a.ind(0,1)]*A[a.ind(1,0)];
   }

   // Compute B = adj(A), host+device version.
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   MFEM_HOST_DEVICE
   static inline void AdjugateHD(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      MFEM_FLOPS_ADD(27);
      B[b.ind(0,0)] = A[a.ind(1,1)]*A[a.ind(2,2)] - A[a.ind(1,2)]*A[a.ind(2,1)];
      B[b.ind(0,1)] = A[a.ind(0,2)]*A[a.ind(2,1)] - A[a.ind(0,1)]*A[a.ind(2,2)];
      B[b.ind(0,2)] = A[a.ind(0,1)]*A[a.ind(1,2)] - A[a.ind(0,2)]*A[a.ind(1,1)];
      B[b.ind(1,0)] = A[a.ind(1,2)]*A[a.ind(2,0)] - A[a.ind(1,0)]*A[a.ind(2,2)];
      B[b.ind(1,1)] = A[a.ind(0,0)]*A[a.ind(2,2)] - A[a.ind(0,2)]*A[a.ind(2,0)];
      B[b.ind(1,2)] = A[a.ind(0,2)]*A[a.ind(1,0)] - A[a.ind(0,0)]*A[a.ind(1,2)];
      B[b.ind(2,0)] = A[a.ind(1,0)]*A[a.ind(2,1)] - A[a.ind(1,1)]*A[a.ind(2,0)];
      B[b.ind(2,1)] = A[a.ind(0,1)]*A[a.ind(2,0)] - A[a.ind(0,0)]*A[a.ind(2,1)];
      B[b.ind(2,2)] = A[a.ind(0,0)]*A[a.ind(1,1)] - A[a.ind(0,1)]*A[a.ind(1,0)];
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      MFEM_FLOPS_ADD(5);
      Adjugate<scalar_t>(a, A, b, B);
      return (A[a.ind(0,0)]*B[b.ind(0,0)] +
              A[a.ind(1,0)]*B[b.ind(0,1)] +
              A[a.ind(2,0)]*B[b.ind(0,2)]);
   }

   // Compute adj(A) and det(A), host+device version.
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   MFEM_HOST_DEVICE
   static inline scalar_t AdjDetHD(const A_layout_t &a, const A_data_t &A,
                                   const B_layout_t &b, B_data_t &B)
   {
      MFEM_FLOPS_ADD(5);
      AdjugateHD<scalar_t>(a, A, b, B);
      return (A[a.ind(0,0)]*B[b.ind(0,0)] +
              A[a.ind(1,0)]*B[b.ind(0,1)] +
              A[a.ind(2,0)]*B[b.ind(0,2)]);
   }

   template <bool symm> struct Symm;
};

template <>
struct MatrixOps<3,3>::Symm<true>
{
   template <typename A_layout_t, typename A_data_t, typename scalar_t>
   static inline MFEM_ALWAYS_INLINE
   void Set(const A_layout_t &a, A_data_t &A,
            const scalar_t a11, const scalar_t a21, const scalar_t a31,
            const scalar_t a22, const scalar_t a32, const scalar_t a33)
   {
      A[a.ind(0)] = a11;
      A[a.ind(1)] = a21;
      A[a.ind(2)] = a31;
      A[a.ind(3)] = a22;
      A[a.ind(4)] = a32;
      A[a.ind(5)] = a33;
   }
};

template <>
struct MatrixOps<3,3>::Symm<false>
{
   template <typename A_layout_t, typename A_data_t, typename scalar_t>
   static inline MFEM_ALWAYS_INLINE
   void Set(const A_layout_t &a, A_data_t &A,
            const scalar_t a11, const scalar_t a21, const scalar_t a31,
            const scalar_t a22, const scalar_t a32, const scalar_t a33)
   {
      A[a.ind(0,0)] = a11;
      A[a.ind(1,0)] = a21;
      A[a.ind(2,0)] = a31;
      A[a.ind(0,1)] = a21;
      A[a.ind(1,1)] = a22;
      A[a.ind(2,1)] = a32;
      A[a.ind(0,2)] = a31;
      A[a.ind(1,2)] = a32;
      A[a.ind(2,2)] = a33;
   }
};

} // namespace mfem::internal

// Compute the determinant of a (small) matrix: det(A).
template <typename scalar_t, typename layout_t, typename data_t>
inline scalar_t TDet(const layout_t &a, const data_t &A)
{
   MFEM_STATIC_ASSERT(layout_t::rank == 2, "invalid rank");
#if !defined(__xlC__) || (__xlC__ >= 0x0d00)
   return internal::MatrixOps<layout_t::dim_1,layout_t::dim_2>::
          template Det<scalar_t>(a, A);
#else
   return internal::MatrixOps<layout_t::dim_1,layout_t::dim_2>::
          Det<scalar_t>(a, A);
#endif
}

// Compute the determinant of a (small) matrix: det(A). Host+device version.
template <typename scalar_t, typename layout_t, typename data_t>
MFEM_HOST_DEVICE
inline scalar_t TDetHD(const layout_t &a, const data_t &A)
{
   MFEM_STATIC_ASSERT(layout_t::rank == 2, "invalid rank");
#if !defined(__xlC__) || (__xlC__ >= 0x0d00)
   return internal::MatrixOps<layout_t::dim_1,layout_t::dim_2>::
          template DetHD<scalar_t>(a, A);
#else
   return internal::MatrixOps<layout_t::dim_1,layout_t::dim_2>::
          DetHD<scalar_t>(a, A);
#endif
}

// Compute the determinants of a set of (small) matrices: D[i] = det(A[i,*,*]).
// The layout of A is (M x N1 x N2) and the size of D is M.
template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
          typename D_data_t>
inline void TDet(const A_layout_t &a, const A_data_t &A, D_data_t &D)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 3, "invalid rank");
#if !defined(__xlC__) || (__xlC__ >= 0x0d00)
   internal::MatrixOps<A_layout_t::dim_2,A_layout_t::dim_3>::
   template Det<Op>(a, A, D);
#else
   internal::MatrixOps<A_layout_t::dim_2,A_layout_t::dim_3>::
   Det<Op>(a, A, D);
#endif
}

// Compute the adjugate matrix of a (small) matrix: B = adj(A).
template <typename scalar_t,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline void TAdjugate(const A_layout_t &a, const A_data_t &A,
                      const B_layout_t &b, B_data_t &B)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                      "invalid ranks");
   internal::MatrixOps<A_layout_t::dim_1,A_layout_t::dim_2>::
   template Adjugate<scalar_t>(a, A, b, B);
}

// Compute the adjugate and the determinant of a (small) matrix: B = adj(A),
// return det(A).
template <typename scalar_t,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline scalar_t TAdjDet(const A_layout_t &a, const A_data_t &A,
                        const B_layout_t &b, B_data_t &B)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                      "invalid ranks");
   return internal::MatrixOps<A_layout_t::dim_1,A_layout_t::dim_2>::
          template AdjDet<scalar_t>(a, A, b, B);
}

// Compute the adjugate and the determinant of a (small) matrix: B = adj(A),
// return det(A). Host+device version.
template <typename scalar_t,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
MFEM_HOST_DEVICE
inline scalar_t TAdjDetHD(const A_layout_t &a, const A_data_t &A,
                          const B_layout_t &b, B_data_t &B)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                      "invalid ranks");
   return internal::MatrixOps<A_layout_t::dim_1,A_layout_t::dim_2>::
          template AdjDetHD<scalar_t>(a, A, b, B);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_MATRIX
