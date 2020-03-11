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

#ifndef MFEM_TEMPLATE_LAYOUT
#define MFEM_TEMPLATE_LAYOUT

#include "../config/tconfig.hpp"
#include "../fem/fespace.hpp"
#include "../general/cuda.hpp"
#include "../general/hip.hpp"

namespace mfem
{

// Layout classes

template <int N1, int S1>
struct OffsetStridedLayout1D;
template <int N1, int S1, int N2, int S2>
struct StridedLayout2D;

template <int N1, int S1>
struct StridedLayout1D
{
   static const int rank = 1;
   static const int dim_1 = N1;
   static const int size = N1;

   MFEM_HOST_DEVICE static inline int ind(int i1)
   {
      return S1*i1;
   }

   template <int M1>
   static OffsetStridedLayout1D<M1,S1> sub(int o1)
   {
      return OffsetStridedLayout1D<M1,S1>(S1*o1);
   }

   // reshape methods

   template <int N1_1, int N1_2>
   static StridedLayout2D<N1_1,S1,N1_2,S1*N1_1> split_1()
   {
      // S1*i1 == S1*(i1_1+N1_1*i1_2)
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return StridedLayout2D<N1_1,S1,N1_2,S1*N1_1>();
   }
};

template <int N1, int S1, int N2, int S2>
struct OffsetStridedLayout2D;

template <int N1, int S1>
struct OffsetStridedLayout1D
{
   static const int rank = 1;
   static const int dim_1 = N1;
   static const int size = N1;

   int offset;

   OffsetStridedLayout1D() { }
   OffsetStridedLayout1D(int offset_) : offset(offset_) { }
   MFEM_HOST_DEVICE inline int ind(int i1) const
   {
      return offset+S1*i1;
   }

   template <int M1>
   OffsetStridedLayout1D<M1,S1> sub(int o1) const
   {
      return OffsetStridedLayout1D<M1,S1>(offset+S1*o1);
   }

   // reshape methods

   template <int N1_1, int N1_2>
   OffsetStridedLayout2D<N1_1,S1,N1_2,S1*N1_1> split_1() const
   {
      // S1*i1 == S1*(i1_1+N1_1*i1_2)
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return OffsetStridedLayout2D<N1_1,S1,N1_2,S1*N1_1>(offset);
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3>
struct StridedLayout3D;
template <int N1, int S1, int N2, int S2, int N3, int S3, int N4, int S4>
struct StridedLayout4D;

template <int N1, int S1, int N2, int S2>
struct StridedLayout2D
{
   static const int rank = 2;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int size = N1*N2;

   MFEM_HOST_DEVICE static inline int ind(int i1, int i2)
   {
      return (S1*i1+S2*i2);
   }
   static OffsetStridedLayout1D<N2,S2> ind1(int i1)
   {
      return OffsetStridedLayout1D<N2,S2>(S1*i1);
   }
   static OffsetStridedLayout1D<N1,S1> ind2(int i2)
   {
      return OffsetStridedLayout1D<N1,S1>(S2*i2);
   }

   template <int M1, int M2>
   static OffsetStridedLayout2D<M1,S1,M2,S2> sub(int o1, int o2)
   {
      return OffsetStridedLayout2D<M1,S1,M2,S2>(S1*o1+S2*o2);
   }

   // reshape methods

   template <int N1_1, int N1_2>
   static StridedLayout3D<N1_1,S1,N1_2,S1*N1_1,N2,S2> split_1()
   {
      // S1*i1+S2*i2 == S1*(i1_1+N1_1*i1_2)+S2*i2
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return StridedLayout3D<N1_1,S1,N1_2,S1*N1_1,N2,S2>();
   }
   template <int N2_1, int N2_2>
   static StridedLayout3D<N1,S1,N2_1,S2,N2_2,S2*N2_1> split_2()
   {
      // S1*i1+S2*i2 == S1*i1+S2*(i2_1*N2_1*i2_2)
      MFEM_STATIC_ASSERT(N2_1*N2_2 == N2, "invalid dimensions");
      return StridedLayout3D<N1,S1,N2_1,S2,N2_2,S2*N2_1>();
   }
   template <int N1_1, int N1_2, int N2_1, int N2_2>
   static StridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2_1,S2,N2_2,S2*N2_1> split_12()
   {
      // S1*i1+S2*i2 == S1*(i1_1+N1_1*i1_2)+S2*(i2_1+N2_1*i2_2)
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1 && N2_1*N2_2 == N2,
                         "invalid dimensions");
      return StridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2_1,S2,N2_2,S2*N2_1>();
   }
   static StridedLayout1D<N1*N2,(S1<S2)?S1:S2> merge_12()
   {
      // use: (S1*i1+S2*i2) == (S1*(i1+S2/S1*i2))
      //  or  (S1*i1+S2*i2) == (S2*(S1/S2*i1+i2))
      // assuming: S2 == S1*N1 || S1 == S2*N2
      MFEM_STATIC_ASSERT(S2 == S1*N1 || S1 == S2*N2, "invalid reshape");
      return StridedLayout1D<N1*N2,(S1<S2)?S1:S2>();
   }
   static StridedLayout2D<N2,S2,N1,S1> transpose_12()
   {
      return StridedLayout2D<N2,S2,N1,S1>();
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3>
struct OffsetStridedLayout3D;
template <int N1, int S1, int N2, int S2, int N3, int S3, int N4, int S4>
struct OffsetStridedLayout4D;

template <int N1, int S1, int N2, int S2>
struct OffsetStridedLayout2D
{
   static const int rank = 2;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int size = N1*N2;

   int offset;

   OffsetStridedLayout2D() { }
   OffsetStridedLayout2D(int offset_) : offset(offset_) { }
   MFEM_HOST_DEVICE inline int ind(int i1, int i2) const
   {
      return offset+S1*i1+S2*i2;
   }
   OffsetStridedLayout1D<N2,S2> ind1(int i1) const
   {
      return OffsetStridedLayout1D<N2,S2>(offset+S1*i1);
   }
   OffsetStridedLayout1D<N1,S1> ind2(int i2) const
   {
      return OffsetStridedLayout1D<N1,S1>(offset+S2*i2);
   }

   template <int M1, int M2>
   OffsetStridedLayout2D<M1,S1,M2,S2> sub(int o1, int o2) const
   {
      return OffsetStridedLayout2D<M1,S1,M2,S2>(offset+S1*o1+S2*o2);
   }

   // reshape methods

   template <int N1_1, int N1_2>
   OffsetStridedLayout3D<N1_1,S1,N1_2,S1*N1_1,N2,S2> split_1() const
   {
      // S1*i1+S2*i2 == S1*(i1_1+N1_1*i1_2)+S2*i2
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return OffsetStridedLayout3D<N1_1,S1,N1_2,S1*N1_1,N2,S2>(offset);
   }
   template <int N2_1, int N2_2>
   OffsetStridedLayout3D<N1,S1,N2_1,S2,N2_2,S2*N2_1> split_2() const
   {
      // S1*i1+S2*i2 == S1*i1+S2*(i2_1*N2_1*i2_2)
      MFEM_STATIC_ASSERT(N2_1*N2_2 == N2, "invalid dimensions");
      return OffsetStridedLayout3D<N1,S1,N2_1,S2,N2_2,S2*N2_1>(offset);
   }
   template <int N1_1, int N1_2, int N2_1, int N2_2>
   OffsetStridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2_1,S2,N2_2,S2*N2_1>
   split_12() const
   {
      // S1*i1+S2*i2 == S1*(i1_1+N1_1*i1_2)+S2*(i2_1+N2_1*i2_2)
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1 && N2_1*N2_2 == N2,
                         "invalid dimensions");
      return OffsetStridedLayout4D<
             N1_1,S1,N1_2,S1*N1_1,N2_1,S2,N2_2,S2*N2_1>(offset);
   }
   OffsetStridedLayout1D<N1*N2,(S1<S2)?S1:S2> merge_12() const
   {
      // use: (S1*i1+S2*i2) == (S1*(i1+S2/S1*i2))
      //  or  (S1*i1+S2*i2) == (S2*(S1/S2*i1+i2))
      // assuming: S2 == S1*N1 || S1 == S2*N2
      MFEM_STATIC_ASSERT(S2 == S1*N1 || S1 == S2*N2, "invalid reshape");
      return OffsetStridedLayout1D<N1*N2,(S1<S2)?S1:S2>(offset);
   }
   OffsetStridedLayout2D<N2,S2,N1,S1> transpose_12() const
   {
      return OffsetStridedLayout2D<N2,S2,N1,S1>(offset);
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3>
struct StridedLayout3D
{
   static const int rank = 3;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int dim_3 = N3;
   static const int size = N1*N2*N3;

   static inline int ind(int i1, int i2, int i3)
   {
      return S1*i1+S2*i2+S3*i3;
   }
   static OffsetStridedLayout2D<N2,S2,N3,S3> ind1(int i1)
   {
      return OffsetStridedLayout2D<N2,S2,N3,S3>(S1*i1);
   }
   static OffsetStridedLayout2D<N1,S1,N3,S3> ind2(int i2)
   {
      return OffsetStridedLayout2D<N1,S1,N3,S3>(S2*i2);
   }
   static OffsetStridedLayout2D<N1,S1,N2,S2> ind3(int i3)
   {
      return OffsetStridedLayout2D<N1,S1,N2,S2>(S3*i3);
   }

   // reshape methods

   static StridedLayout2D<N1*N2,S1,N3,S3> merge_12()
   {
      // use: (S1*i1+S2*i2+S3*i3) == (S1*(i1+S2/S1*i2)+S3*i3)
      // assuming: S2 == S1*N1
      MFEM_STATIC_ASSERT(S2 == S1*N1, "invalid reshape");
      return StridedLayout2D<N1*N2,S1,N3,S3>();
      // alternative:
      // use: (S1*i1+S2*i2+S3*i3) == (S2*(S1/S2*i1+i2)+S3*i3)
      // assuming: S1 == S2*N2
      // result is: StridedLayout2D<N1*N2,S2,N3,S3>
   }
   static StridedLayout2D<N1,S1,N2*N3,S2> merge_23()
   {
      // use: (S1*i1+S2*i2+S3*i3) == (S1*i1+S2*(i2+S3/S2*i3))
      // assuming: S3 == S2*N2
      MFEM_STATIC_ASSERT(S3 == S2*N2, "invalid reshape");
      return StridedLayout2D<N1,S1,N2*N3,S2>();
   }

   template <int N1_1, int N1_2>
   static StridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2,S2,N3,S3> split_1()
   {
      // S1*i1+S2*i2+S3*i3 == S1*(i1_1+N1_1*i1_2)+S2*i2+S3*i3
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return StridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2,S2,N3,S3>();
   }
   template <int N2_1, int N2_2>
   static StridedLayout4D<N1,S1,N2_1,S2,N2_2,S2*N2_1,N3,S3> split_2()
   {
      // S1*i1+S2*i2+S3*i3 == S1*i1+S2*(i2_1+N2_1*i2_2)+S3*i3
      MFEM_STATIC_ASSERT(N2_1*N2_2 == N2, "invalid dimensions");
      return StridedLayout4D<N1,S1,N2_1,S2,N2_2,S2*N2_1,N3,S3>();
   }
   template <int N3_1, int N3_2>
   static StridedLayout4D<N1,S1,N2,S2,N3_1,S3,N3_2,S3*N3_1> split_3()
   {
      // S1*i1+S2*i2+S3*i3 == S1*i1+S2*i2+S3*(i3_1+N3_1*i3_2)
      MFEM_STATIC_ASSERT(N3_1*N3_2 == N3, "invalid dimensions");
      return StridedLayout4D<N1,S1,N2,S2,N3_1,S3,N3_2,S3*N3_1>();
   }

   static StridedLayout3D<N2,S2,N1,S1,N3,S3> transpose_12()
   {
      return StridedLayout3D<N2,S2,N1,S1,N3,S3>();
   }
   static StridedLayout3D<N3,S3,N2,S2,N1,S1> transpose_13()
   {
      return StridedLayout3D<N3,S3,N2,S2,N1,S1>();
   }
   static StridedLayout3D<N1,S1,N3,S3,N2,S2> transpose_23()
   {
      return StridedLayout3D<N1,S1,N3,S3,N2,S2>();
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3>
struct OffsetStridedLayout3D
{
   static const int rank = 3;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int dim_3 = N3;
   static const int size = N1*N2*N3;

   int offset;

   OffsetStridedLayout3D() { }
   OffsetStridedLayout3D(int offset_) : offset(offset_) { }
   inline int ind(int i1, int i2, int i3) const
   {
      return offset+S1*i1+S2*i2+S3*i3;
   }
   OffsetStridedLayout2D<N2,S2,N3,S3> ind1(int i1) const
   {
      return OffsetStridedLayout2D<N2,S2,N3,S3>(offset+S1*i1);
   }
   OffsetStridedLayout2D<N1,S1,N3,S3> ind2(int i2) const
   {
      return OffsetStridedLayout2D<N1,S1,N3,S3>(offset+S2*i2);
   }
   OffsetStridedLayout2D<N1,S1,N2,S2> ind3(int i3) const
   {
      return OffsetStridedLayout2D<N1,S1,N2,S2>(offset+S3*i3);
   }

   // reshape methods

   OffsetStridedLayout2D<N1*N2,S1,N3,S3> merge_12() const
   {
      // use: (S1*i1+S2*i2+S3*i3) == (S1*(i1+S2/S1*i2)+S3*i3)
      // assuming: S2 == S1*N1
      MFEM_STATIC_ASSERT(S2 == S1*N1, "invalid reshape");
      return OffsetStridedLayout2D<N1*N2,S1,N3,S3>(offset);
   }
   OffsetStridedLayout2D<N1,S1,N2*N3,S2> merge_23() const
   {
      // use: (S1*i1+S2*i2+S3*i3) == (S1*i1+S2*(i2+S3/S2*i3))
      // assuming: S3 == S2*N2
      MFEM_STATIC_ASSERT(S3 == S2*N2, "invalid reshape");
      return OffsetStridedLayout2D<N1,S1,N2*N3,S2>(offset);
   }

   template <int N1_1, int N1_2>
   OffsetStridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2,S2,N3,S3> split_1() const
   {
      // S1*i1+S2*i2+S3*i3 == S1*(i1_1+N1_1*i1_2)+S2*i2+S3*i3
      MFEM_STATIC_ASSERT(N1_1*N1_2 == N1, "invalid dimensions");
      return OffsetStridedLayout4D<N1_1,S1,N1_2,S1*N1_1,N2,S2,N3,S3>(offset);
   }
   template <int N2_1, int N2_2>
   OffsetStridedLayout4D<N1,S1,N2_1,S2,N2_2,S2*N2_1,N3,S3> split_2() const
   {
      // S1*i1+S2*i2+S3*i3 == S1*i1+S2*(i2_1+N2_1*i2_2)+S3*i3
      MFEM_STATIC_ASSERT(N2_1*N2_2 == N2, "invalid dimensions");
      return OffsetStridedLayout4D<N1,S1,N2_1,S2,N2_2,S2*N2_1,N3,S3>(offset);
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3, int N4, int S4>
struct StridedLayout4D
{
   static const int rank = 4;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int dim_3 = N3;
   static const int dim_4 = N4;
   static const int size = N1*N2*N3*N4;

   static inline int ind(int i1, int i2, int i3, int i4)
   {
      return S1*i1+S2*i2+S3*i3+S4*i4;
   }
   static OffsetStridedLayout2D<N1,S1,N4,S4> ind23(int i2, int i3)
   {
      return OffsetStridedLayout2D<N1,S1,N4,S4>(S2*i2+S3*i3);
   }
   static OffsetStridedLayout2D<N2,S2,N3,S3> ind14(int i1, int i4)
   {
      return OffsetStridedLayout2D<N2,S2,N3,S3>(S1*i1+S4*i4);
   }
   static OffsetStridedLayout3D<N1,S1,N2,S2,N3,S3> ind4(int i4)
   {
      return OffsetStridedLayout3D<N1,S1,N2,S2,N3,S3>(S4*i4);
   }

   static StridedLayout3D<N1*N2,S1,N3,S3,N4,S4> merge_12()
   {
      // use: (S1*i1+S2*i2+S3*i3+S4*i4) == (S1*(i1+S2/S1*i2)+S3*i3+S4*i4)
      // assuming S2 == S1*N1
      MFEM_STATIC_ASSERT(S2 == S1*N1, "invalid reshape");
      return StridedLayout3D<N1*N2,S1,N3,S3,N4,S4>();
   }
   static StridedLayout3D<N1,S1,N2,S2,N3*N4,S3> merge_34()
   {
      // use: (S1*i1+S2*i2+S3*i3+S4*i4) == (S1*i1+S2*i2+S3*(i3+S4/S3*i4))
      // assuming S4 == S3*N3
      MFEM_STATIC_ASSERT(S4 == S3*N3, "invalid reshape");
      return StridedLayout3D<N1,S1,N2,S2,N3*N4,S3>();
   }
};

template <int N1, int S1, int N2, int S2, int N3, int S3, int N4, int S4>
struct OffsetStridedLayout4D
{
   static const int rank = 4;
   static const int dim_1 = N1;
   static const int dim_2 = N2;
   static const int dim_3 = N3;
   static const int dim_4 = N4;
   static const int size = N1*N2*N3*N4;

   int offset;

   OffsetStridedLayout4D() { }
   OffsetStridedLayout4D(int offset_) : offset(offset_) { }
   inline int ind(int i1, int i2, int i3, int i4) const
   {
      return offset+S1*i1+S2*i2+S3*i3+S4*i4;
   }
};

template <int N1, int N2>
struct ColumnMajorLayout2D
   : public StridedLayout2D<N1,1,N2,N1> { };

template <int N1, int N2, int N3>
struct ColumnMajorLayout3D
   : public StridedLayout3D<N1,1,N2,N1,N3,N1*N2> { };

template <int N1, int N2, int N3, int N4>
struct ColumnMajorLayout4D
   : public StridedLayout4D<N1,1,N2,N1,N3,N1*N2,N4,N1*N2*N3> { };


// Vector layout classes

class DynamicVectorLayout
{
public:
   static const int vec_dim = 0; // 0 - dynamic

protected:
   int scal_stride, comp_stride;
   int num_components;

   void Init(Ordering::Type ordering, int scalar_size, int num_comp)
   {
      num_components = num_comp;
      if (ordering == Ordering::byNODES)
      {
         scal_stride = 1;
         comp_stride = scalar_size;
      }
      else
      {
         scal_stride = num_comp;
         comp_stride = 1;
      }
   }

public:
   DynamicVectorLayout(Ordering::Type ordering, int scalar_size, int num_comp)
   {
      Init(ordering, scalar_size, num_comp);
   }
   DynamicVectorLayout(const FiniteElementSpace &fes)
   {
      Init(fes.GetOrdering(), fes.GetNDofs(), fes.GetVDim());
   }
   // default copy constructor

   int NumComponents() const { return num_components; }

   int ind(int scalar_idx, int comp_idx) const
   {
      return scal_stride * scalar_idx + comp_stride * comp_idx;
   }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return true;
   }
};

// The default value (NumComp = 0) indicates that the number of components is
// dynamic, i.e. it will be specified at run-time.
template <Ordering::Type Ord, int NumComp = 0>
class VectorLayout
{
public:
   static const int vec_dim = NumComp;

protected:
   int num_components, scalar_size;

public:
   VectorLayout(int scalar_size_, int num_comp_ = NumComp)
      : num_components(num_comp_),
        scalar_size(scalar_size_)
   {
      MFEM_ASSERT(NumComp == 0 || num_components == NumComp,
                  "invalid number of components");
   }

   VectorLayout(const FiniteElementSpace &fes)
      : num_components(fes.GetVDim()),
        scalar_size(fes.GetNDofs())
   {
      MFEM_ASSERT(fes.GetOrdering() == Ord, "ordering mismatch");
      MFEM_ASSERT(NumComp == 0 || num_components == NumComp,
                  "invalid number of components");
   }
   // default copy constructor

   int NumComponents() const { return (NumComp ? NumComp : num_components); }

   int ind(int scalar_idx, int comp_idx) const
   {
      if (Ord == Ordering::byNODES)
      {
         return scalar_idx + comp_idx * scalar_size;
      }
      else
      {
         return comp_idx + (NumComp ? NumComp : num_components) * scalar_idx;
      }
   }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return (Ord == fes.GetOrdering() &&
              (NumComp == 0 || NumComp == fes.GetVDim()));
   }
};

class ScalarLayout
{
public:
   static const int vec_dim = 1;

   ScalarLayout() { }

   ScalarLayout(const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetVDim() == 1, "invalid number of components");
   }

   int NumComponents() const { return 1; }

   int ind(int scalar_idx, int comp_idx) const { return scalar_idx; }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return (fes.GetVDim() == 1);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_LAYOUT
