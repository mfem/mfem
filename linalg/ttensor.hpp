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

#ifndef MFEM_TEMPLATE_TENSOR
#define MFEM_TEMPLATE_TENSOR

#include "../config/tconfig.hpp"
#include "../linalg/simd.hpp"
#include "../general/tassign.hpp"
#include "../general/backends.hpp"
#include "tlayout.hpp"
#include "tmatrix.hpp"

// Templated tensor implementation (up to order 4)

namespace mfem
{

// Element-wise tensor operations

namespace internal
{

template <int Rank>
struct TensorOps;

template <>
struct TensorOps<1> // rank = 1
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 1, "invalid rank");
      for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
      {
         mfem::Assign<Op>(A_data[A_layout.ind(i1)], value);
      }
   }

   // Assign: A {=,+=,*=} scalar_value, host+device version
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   MFEM_HOST_DEVICE
   static void AssignHD(const A_layout_t &A_layout, A_data_t &A_data,
                        const scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 1, "invalid rank");
      for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
      {
         mfem::AssignHD<Op>(A_data[A_layout.ind(i1)], value);
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 1 && B_layout_t::rank == 1,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1,
                         "invalid dimensions");
      for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
      {
         mfem::Assign<Op>(A_data[A_layout.ind(i1)], B_data[B_layout.ind(i1)]);
      }
   }
};

template <>
struct TensorOps<2> // rank = 2
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 2, "invalid rank");
      for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
      {
         for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
         {
            mfem::Assign<Op>(A_data[A_layout.ind(i1,i2)], value);
         }
      }
   }

   // Assign: A {=,+=,*=} scalar_value, host+device version
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   MFEM_HOST_DEVICE
   static void AssignHD(const A_layout_t &A_layout, A_data_t &A_data,
                        scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 2, "invalid rank");
      for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
      {
         for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
         {
            mfem::AssignHD<Op>(A_data[A_layout.ind(i1,i2)], value);
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2,
                         "invalid dimensions");
      for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
      {
         for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
         {
            mfem::Assign<Op>(A_data[A_layout.ind(i1,i2)],
                             B_data[B_layout.ind(i1,i2)]);
         }
      }
   }
};

template <>
struct TensorOps<3> // rank = 3
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 3, "invalid rank");
      for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
      {
         for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
         {
            for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
            {
               mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3)], value);
            }
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 3 && B_layout_t::rank == 3,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2 &&
                         A_layout_t::dim_3 == B_layout_t::dim_3,
                         "invalid dimensions");
      for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
      {
         for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
         {
            for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
            {
               mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3)],
                                B_data[B_layout.ind(i1,i2,i3)]);
            }
         }
      }
   }
};

template <>
struct TensorOps<4> // rank = 4
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 4, "invalid rank");
      for (int i4 = 0; i4 < A_layout_t::dim_4; i4++)
      {
         for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
         {
            for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
            {
               for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
               {
                  mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3,i4)], value);
               }
            }
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 4 && B_layout_t::rank == 4,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2 &&
                         A_layout_t::dim_3 == B_layout_t::dim_3 &&
                         A_layout_t::dim_4 == B_layout_t::dim_4,
                         "invalid dimensions");
      for (int i4 = 0; i4 < A_layout_t::dim_4; i4++)
      {
         for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
         {
            for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
            {
               for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
               {
                  mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3,i4)],
                                   B_data[B_layout.ind(i1,i2,i3,i4)]);
               }
            }
         }
      }
   }
};

} // namespace mfem::internal

// Tensor or sub-tensor assign function: A {=,+=,*=} scalar_value.
template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
          typename scalar_t>
inline void TAssign(const A_layout_t &A_layout, A_data_t &A_data,
                    const scalar_t value)
{
   internal::TensorOps<A_layout_t::rank>::
   template Assign<Op>(A_layout, A_data, value);
}

// Tensor or sub-tensor assign function: A {=,+=,*=} scalar_value.
// Host+device version.
template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
          typename scalar_t>
MFEM_HOST_DEVICE
inline void TAssignHD(const A_layout_t &A_layout, A_data_t &A_data,
                      const scalar_t value)
{
   internal::TensorOps<A_layout_t::rank>::
   template AssignHD<Op>(A_layout, A_data, value);
}

// Tensor assign function: A {=,+=,*=} B that allows different input and output
// layouts. With suitable layouts this function can be used to permute
// (transpose) tensors, extract sub-tensors, etc.
template <AssignOp::Type Op,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline void TAssign(const A_layout_t &A_layout, A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data)
{
   internal::TensorOps<A_layout_t::rank>::
   template Assign<Op>(A_layout, A_data, B_layout, B_data);
}


// classes TVector, TMatrix, TTensor3, TTensor4

template <int S, typename data_t = double, bool align = false>
struct TVector
{
public:
   static const int size = S;
   static const int aligned_size = align ? MFEM_ALIGN_SIZE(S,data_t) : size;
   typedef data_t data_type;
   data_t data[aligned_size>0?aligned_size:1];

   typedef StridedLayout1D<S,1> layout_type;
   static const layout_type layout;

   data_t &operator[](int i) { return data[i]; }
   const data_t &operator[](int i) const { return data[i]; }

   template <AssignOp::Type Op>
   void Assign(const data_t d)
   {
      TAssign<Op>(layout, data, d);
   }

   template <AssignOp::Type Op, typename src_data_t>
   void Assign(const src_data_t &src)
   {
      TAssign<Op>(layout, data, layout, src);
   }

   template <AssignOp::Type Op, typename dest_data_t>
   void AssignTo(dest_data_t &dest)
   {
      TAssign<Op>(layout, dest, layout, data);
   }

   void Set(const data_t d)
   {
      Assign<AssignOp::Set>(d);
   }

   template <typename src_data_t>
   void Set(const src_data_t &src)
   {
      Assign<AssignOp::Set>(src);
   }

   template <typename dest_data_t>
   void Assemble(dest_data_t &dest) const
   {
      AssignTo<AssignOp::Add>(dest);
   }

   void Scale(const data_t scale)
   {
      Assign<AssignOp::Mult>(scale);
   }
};

template <int S, typename data_t, bool align>
const typename TVector<S,data_t,align>::layout_type
TVector<S,data_t,align>::layout = layout_type();


template <int N1, int N2, typename data_t = double, bool align = false>
struct TMatrix : public TVector<N1*N2,data_t,align>
{
   typedef TVector<N1*N2,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout2D<N1,N2> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2) { return layout.ind(i1,i2); }

   data_t &operator()(int i, int j) { return data[ind(i,j)]; }
   const data_t &operator()(int i, int j) const { return data[ind(i,j)]; }

   inline data_t Det() const
   {
      return TDet<data_t>(layout, data);
   }

   inline void Adjugate(TMatrix<N1,N2,data_t> &adj) const
   {
      TAdjugate<data_t>(layout, data, layout, adj.data);
   }

   // Compute the adjugate and the determinant of a (small) matrix.
   inline data_t AdjDet(TMatrix<N2,N1,data_t> &adj) const
   {
      return TAdjDet<data_t>(layout, data, layout, adj.data);
   }
};

template <int N1, int N2, typename data_t, bool align>
const typename TMatrix<N1,N2,data_t,align>::layout_type
TMatrix<N1,N2,data_t,align>::layout = layout_type();


template <int N1, int N2, int N3, typename data_t = double, bool align = false>
struct TTensor3 : TVector<N1*N2*N3,data_t,align>
{
   typedef TVector<N1*N2*N3,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout3D<N1,N2,N3> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2, int i3)
   { return layout.ind(i1,i2,i3); }

   data_t &operator()(int i, int j, int k) { return data[ind(i,j,k)]; }
   const data_t &operator()(int i, int j, int k) const
   { return data[ind(i,j,k)]; }
};

template <int N1, int N2, int N3, typename data_t, bool align>
const typename TTensor3<N1,N2,N3,data_t,align>::layout_type
TTensor3<N1,N2,N3,data_t,align>::layout = layout_type();

template <int N1, int N2, int N3, int N4, typename data_t = double,
          bool align = false>
struct TTensor4 : TVector<N1*N2*N3*N4,data_t,align>
{
   typedef TVector<N1*N2*N3*N4,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout4D<N1,N2,N3,N4> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2, int i3, int i4)
   { return layout.ind(i1,i2,i3,i4); }

   data_t &operator()(int i, int j, int k, int l)
   { return data[ind(i,j,k,l)]; }
   const data_t &operator()(int i, int j, int k, int l) const
   { return data[ind(i,j,k,l)]; }
};

template <int N1, int N2, int N3, int N4, typename data_t, bool align>
const typename TTensor4<N1,N2,N3,N4,data_t,align>::layout_type
TTensor4<N1,N2,N3,N4,data_t,align>::layout = layout_type();


// Tensor products

// C_{i,j,k}  {=|+=}  \sum_s A_{s,j} B_{i,s,k}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_1_2(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 3, "invalid ranks");
   const int B3 = B_layout_t::dim_3;
   const int C3 = C_layout_t::dim_3;
   MFEM_STATIC_ASSERT(B3 == C3, "invalid dimentions");
   for (int k = 0; k < B3; k++)
   {
      Mult_AB<Add>(B_layout.ind3(k), B_data,
                   A_layout, A_data,
                   C_layout.ind3(k), C_data);
   }
}

// C_{i,j,k}  {=|+=}  \sum_s A_{i,s} B_{s,j,k}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_2_1(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 3, "invalid ranks");
   Mult_AB<Add>(A_layout, A_data,
                B_layout.merge_23(), B_data,
                C_layout.merge_23(), C_data);
}

// C_{i,k,j,l}  {=|+=}  \sum_s A_{s,i} A_{s,j} B_{k,s,l}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void TensorAssemble(const A_layout_t &A_layout, const A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data,
                    const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 4, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int B3 = B_layout_t::dim_3;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int C4 = C_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A1 == B2 && A2 == C1 && A2 == C3 && B1 == C2 && B3 == C4,
                      "invalid dimensions");

#if 1
   // Impl == 3
   MFEM_FLOPS_ADD(3*A1*A2*A2*B1*B3);
   if (!Add) { TAssign<AssignOp::Set>(C_layout, C_data, 0.0); }
   for (int j = 0; j < A2; j++)
   {
      for (int i = 0; i < A2; i++)
      {
         for (int l = 0; l < B3; l++)
         {
            for (int k = 0; k < B1; k++)
            {
               for (int s = 0; s < A1; s++)
               {
                  // C(i,k,j,l) += A(s,i) * A(s,j) * B(k,s,l);
                  C_data[C_layout.ind(i,k,j,l)] +=
                     A_data[A_layout.ind(s,i)] *
                     A_data[A_layout.ind(s,j)] *
                     B_data[B_layout.ind(k,s,l)];
               }
            }
         }
      }
   }
#else
   // Impl == 1
   if (!Add) { TAssign<AssignOp::Set>(C_layout, C_data, 0.0); }
   for (int s = 0; s < A1; s++)
   {
      for (int i = 0; i < A2; i++)
      {
         for (int k = 0; k < B1; k++)
         {
            for (int j = 0; j < A2; j++)
            {
               for (int l = 0; l < B3; l++)
               {
                  // C(i,k,j,l) += A(s,i) * A(s,j) * B(k,s,l);
                  C_data[C_layout.ind(i,k,j,l)] +=
                     A_data[A_layout.ind(s,i)] *
                     A_data[A_layout.ind(s,j)] *
                     B_data[B_layout.ind(k,s,l)];
               }
            }
         }
      }
   }
#endif
}

// D_{i,k,j,l}  {=|+=}  \sum_s A_{i,s} B_{s,j} C_{k,s,l}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t,
          typename D_layout_t, typename D_data_t>
MFEM_ALWAYS_INLINE inline
void TensorAssemble(const A_layout_t &A_layout, const A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data,
                    const C_layout_t &C_layout, const C_data_t &C_data,
                    const D_layout_t &D_layout, D_data_t &D_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 3 && D_layout_t::rank == 4,
                      "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int D1 = D_layout_t::dim_1;
   const int D2 = D_layout_t::dim_2;
   const int D3 = D_layout_t::dim_3;
   const int D4 = D_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A2 == B1 && A2 == C2 && A1 == D1 && B2 == D3 &&
                      C1 == D2 && C3 == D4, "invalid dimensions");

#if 0
   TTensor4<A1,C1,A2,C3> H;
   // H_{i,k,s,l} = A_{i,s} C_{k,s,l}
   for (int l = 0; l < C3; l++)
   {
      for (int s = 0; s < B1; s++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int i = 0; i < A1; i++)
            {
               H(i,k,s,l) = A_data[A_layout.ind(i,s)]*
                            C_data[C_layout.ind(k,s,l)];
            }
         }
      }
   }
   // D_{(i,k),j,l} = \sum_s B_{s,j} H_{(i,k),s,l}
   Mult_1_2<Add>(B_layout, B_data, H.layout.merge_12(), H,
                 D_layout.merge_12(), D_data);
#elif 1
   MFEM_FLOPS_ADD(A1*B1*C1*C3); // computation of H(l)
   for (int l = 0; l < C3; l++)
   {
      TTensor3<A1,C1,A2,typename C_data_t::data_type> H;
      // H(l)_{i,k,s} = A_{i,s} C_{k,s,l}
      for (int s = 0; s < B1; s++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int i = 0; i < A1; i++)
            {
               H(i,k,s) = A_data[A_layout.ind(i,s)]*
                          C_data[C_layout.ind(k,s,l)];
            }
         }
      }
      // D_{(i,k),j,l} = \sum_s H(l)_{(i,k),s} B_{s,j}
      Mult_AB<Add>(H.layout.merge_12(), H, B_layout, B_data,
                   D_layout.merge_12().ind3(l), D_data);
   }
#else
   TTensor4<B1,C1,B2,C3> F;
   for (int l = 0; l < C3; l++)
   {
      for (int j = 0; j < B2; j++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int s = 0; s < B1; s++)
            {
               F(s,k,j,l) = B_data[B_layout.ind(s,j)]*
                            C_data[C_layout.ind(k,s,l)];
            }
         }
      }
   }
   Mult_AB<Add>(A_layout, A_data, F.layout.merge_34().merge_23(), F,
                D_layout.merge_34().merge_23(), D_data);
#endif
}


// C_{i,j,k,l}  {=|+=}  A_{i,j,k} B_{j,l}
template <AssignOp::Type Op,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void TensorProduct(const A_layout_t &a, const A_data_t &A,
                   const B_layout_t &b, const B_data_t &B,
                   const C_layout_t &c, C_data_t &C)
{
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int A3 = A_layout_t::dim_3;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int C4 = C_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A1 == C1 && A2 == B1 && A2 == C2 && A3 == C3 && B2 == C4,
                      "invalid dimensions");

   MFEM_FLOPS_ADD(A1*A2*A3*B2);
   for (int l = 0; l < B2; l++)
   {
      for (int k = 0; k < A3; k++)
      {
         for (int j = 0; j < A2; j++)
         {
            for (int i = 0; i < A1; i++)
            {
               mfem::Assign<Op>(C[c.ind(i,j,k,l)],
                                A[a.ind(i,j,k)]*B[b.ind(j,l)]);
            }
         }
      }
   }
}

} // namespace mfem

#endif // MFEM_TEMPLATE_TENSOR
