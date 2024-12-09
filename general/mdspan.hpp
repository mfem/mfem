// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MDSPAN_HPP
#define MFEM_MDSPAN_HPP

#include <list>
#include <array>
#include <vector>
#include <utility>
#include <type_traits>

#include "device.hpp"
#include "backends.hpp"

namespace mfem
{

namespace internal // experimental helper functions for mfem::MDLayout
{

// md_sequence represents a compile-time sequence of integers
template <typename T, T... args> struct md_sequence { };

template <typename T, int N, bool left> struct make_md_sequence;

// make_sequence, specialized for left (default) and right layout
template <typename T, int N, bool left = true>
using make_sequence = typename make_md_sequence<T, N, left>::type;

} // namespace internal

/// @brief The MDOffset class computes the multi-dimensional offsets
template <int n, int N, typename T, typename... Ts>
struct MDOffset
{
   static MFEM_HOST_DEVICE inline
   T offset(const int (&Sd)[N], T nd, Ts... args)
   { return nd * Sd[n-1] + MDOffset<n+1, N, Ts...>::offset(Sd, args...); }
};

template <int N, typename T, typename... Ts>
struct MDOffset<N, N, T, Ts...>
{
   static MFEM_HOST_DEVICE inline
   T offset(const int (&Sd)[N], T nd) { return nd * Sd[N-1]; }
};

/// @brief The MDTensor class holds the pointer and strides for each dimension
template<int N, typename T> class MDTensor
{
   T *ptr;
   int Sd[N];

public:
   /// Default constructor
   MDTensor() = delete;

   /// Copy constructor (default)
   MDTensor(const MDTensor&) = default;

   /// Copy assignment (default)
   MDTensor& operator=(const MDTensor&) = default;

   /// Constructor to initialize a tensor from a pointer and strides
   template <typename... Args> MFEM_HOST_DEVICE
   MDTensor(T *ptr, const int (&sd)[N]): ptr(ptr)
   { for (int i = 0; i < N; ++i) { Sd[i] = sd[i]; } }

   /// Accessor for the data
   template <typename... Ts> MFEM_HOST_DEVICE inline
   T& operator()(Ts... args) { return ptr[Offset(args...)]; }

   /// Const accessor for the data
   template <typename... Ts> MFEM_HOST_DEVICE inline
   T& operator()(Ts... args) const { return ptr[Offset(args...)]; }

   /// Offset computation
   template <typename... Ts> MFEM_HOST_DEVICE inline
   int Offset(Ts... args) const
   {
      static_assert(sizeof...(args) == N, "Wrong number of dimensions");
      return MDOffset<1, N, Ts...>::offset(Sd, args...);
   }
};

/// \brief The MDLayout class, defaulted to a column-major (left) ordering
template<int N, bool left = true> struct MDLayout
{
   /// Create a layout with the internal::md_sequence
   template <int... args>
   static constexpr auto Make(internal::md_sequence<int, args...>)
   -> std::array<int, sizeof...(args)> { return {(static_cast<int>(args))...}; }

   /// Array holding the layout permutation
   using perm_type = std::array<int, N>;
   perm_type perm = Make(internal::make_sequence<int, N, left> {});

   /// Default constructor
   MDLayout() = default;

   /// Copy constructor (default)
   MDLayout(const MDLayout&) = default;

   /// Copy assignment (default)
   MDLayout& operator=(const MDLayout&) = default;

   /// Constructor to initialize a layout from an array of indices
   template <typename... Ts>
   MDLayout(int n, Ts... args) noexcept: MDLayout(args...)
   {
      constexpr int k = N - sizeof...(args) - 1;
      static_assert(0 <= k && k < N, "Index out of bounds!");
      perm[k] = n;
   }

   /// Access layout entries using operator()
   inline int operator()(int i) const
   { return Assert(i), perm[static_cast<typename perm_type::size_type>(i)]; }

   /// Access layout entries using operator[]
   inline int operator[](int i) const
   { return Assert(i), perm[static_cast<typename perm_type::size_type>(i)]; }

   /// Asserts the given index is valid (only in MFEM_DEBUG)
   inline void Assert(const int k) const
   {
      MFEM_CONTRACT_VAR(k);
      MFEM_ASSERT(0 <= k && k < N, "Index should be in [0," << (N-1) << "]");
   }
};

/// Left (Column-major (Fortran)) and Right (Row-major (C/C++)) layouts
template<int N> using MDLayoutLeft = MDLayout<N, true>;
template<int N> using MDLayoutRight = MDLayout<N, false>;

/// \brief The MDSpan base class is a generic non-owning mfem_type's view
/// that reinterprets it as a multidimensional type.
template<typename mfem_type, int N, class layout_type = MDLayoutLeft<N>>
class MDSpan : protected mfem_type
{
protected:
   using T = typename std::remove_pointer<decltype(mfem_type::data.h_ptr)>::type;

   int Nd[N], Sd[N]; // dimension sizes and strides, once the layout is set
   layout_type layout; // stored layout, useful for reshapes

   /// Set the dimensions (Nd) and strides (Sd) during contruction.
   /// When all the arguments have been processed, SetSize is called on the
   /// mfem_type with Device::GetMemoryType() as memory type and SetLayout is
   /// called using the layout.
   template <typename... Ts> void Setup(int dim, Ts... args)
   {
      constexpr int k = N - sizeof...(args) - 1;
      Sd[k] = Nd[k] = dim;
      if (k > 0) { return; }
      int psize = 1;
      for (int i = 0; i < N; i++) { psize *= Nd[i]; }
      mfem_type::SetSize(static_cast<int>(psize), Device::GetMemoryType());
      SetLayout(layout);
   }

public:

   /// Default constructor (recursion)
   MDSpan() noexcept: mfem_type() { }

   /// Recursion constructor
   template <typename... Ts>
   MDSpan(int n, Ts... args): MDSpan(args...) { Setup(n, args...); }

   /// Move constructor (delete)
   MDSpan(MDSpan&&) = delete;

   /// Copy constructor (delete)
   MDSpan(const MDSpan&) = delete;

   /// Move assignment (delete)
   MDSpan& operator=(MDSpan&&) = delete;

   /// Copy assignment (delete)
   MDSpan& operator=(const MDSpan&) = delete;

   /// Return the ith dimension
   int Extent(int i) const { return Nd[i]; }

   /// Return the size of the span.
   int Size() const { return mfem_type::Size(); }

   /// Store and use the given layout to update the strides
   template<typename Layout> void SetLayout(const Layout &l)
   {
      layout = l;
      Sd[l[0]] = 1;
      for (int i = 1; i < N; i++) { Sd[l[i]] = Nd[l[i-1]] * Sd[l[i-1]]; }
   }

   /// Variadic resize the mfem_type
   template <typename... Ts> inline void SetSize(int size, Ts... args)
   {
      constexpr int k = N - sizeof...(args) - 1;
      Sd[k] = Nd[k] = size;
      const int msize = mfem_type::Size();
      MFEM_VERIFY(size > 0, "Size should be positive!");
      mfem_type::SetSize(msize > 0 ? msize*size : size, Device::GetMemoryType());
      MDSpan::SetSize(args...);
   }

   /// Variadic terminal case of the mfem_type resize
   inline void SetSize(int size)
   {
      Sd[N-1] = Nd[N-1] = size;
      const int msize = mfem_type::Size();
      MFEM_VERIFY(size > 0, "Size should be positive!");
      mfem_type::SetSize(msize > 0 ? msize*size : size, Device::GetMemoryType());
      SetLayout(layout);
   }

   /// Access mfem_type data entries using operator()
   template <typename... Ts> inline
   T& operator()(Ts... args) { return mfem_type::data[Offset(args...)]; }

   /// Const access mfem_type data entries using operator()
   template <typename... Ts> inline const T& operator()(Ts... args) const
   {
      return mfem_type::data[Offset(args...)];
   }

   /// Offset computation
   template <typename... Ts> inline int Offset(Ts... args) const
   {
      static_assert(sizeof...(args) == N, "Wrong number of dimensions");
      return MDOffset<1,N,Ts...>::offset(Sd, args...);
   }

   /// Shortcut for mfem::Read(mfem_type::data, mfem_type::size, on_dev)
   /// and return an MDTensor with the MDSpan's pointer and strides
   const MDTensor<N,const T> MDRead(bool on_dev = true) const
   {
      const T *ptr = mfem::Read(mfem_type::data, mfem_type::size, on_dev);
      return MDTensor<N,const T>(ptr, Sd);
   }

   /// Shortcut for mfem::Read(mfem_type::data, mfem_type::size, false)
   /// and return an MDTensor with the MDSpan's pointer and strides
   const MDTensor<N,const T> MDHostRead() const
   {
      const T *ptr = mfem::Read(mfem_type::data, mfem_type::size, false);
      return MDTensor<N,const T>(ptr, Sd);
   }

   /// Shortcut for mfem::Write(mfem_type::data, mfem_type::size, on_dev)
   /// and return an MDTensor with the MDSpan's pointer and strides
   MDTensor<N,T> MDWrite(bool on_dev = true)
   {
      T *ptr = mfem::Write(mfem_type::data, mfem_type::size, on_dev);
      return MDTensor<N,T>(ptr, Sd);
   }

   /// Shortcut for mfem::Write(mfem_type::data, mfem_type::size, false)
   /// and return an MDTensor with the MDSpan's pointer and strides
   MDTensor<N,T> MDHostWrite()
   {
      T *ptr = mfem::Write(mfem_type::data, mfem_type::size, false);
      return MDTensor<N,T>(ptr, Sd);
   }

   /// Shortcut for mfem::ReadWrite(mfem_type::data, mfem_type::size, on_dev)
   /// and return an MDTensor with the MDSpan's pointer and strides
   MDTensor<N,T> MDReadWrite(bool on_dev = true)
   {
      T *ptr = mfem::ReadWrite(mfem_type::data, mfem_type::size, on_dev);
      return MDTensor<N,T>(ptr, Sd);
   }

   /// Shortcut for mfem::ReadWrite(mfem_type::data, mfem_type::size, false)
   /// and return an MDTensor with the MDSpan's pointer and strides
   MDTensor<N,T> MDHostReadWrite()
   {
      T *ptr = mfem::ReadWrite(mfem_type::data, mfem_type::size, false);
      return MDTensor<N,T>(ptr, Sd);
   }

   /// The MDReshape function allows to reshape the multi-dimentional view
   /// into a new multi-dimentional one, by the use of std::array blocks.
   /// For example, if 'this' has three dimensions {N1, N2, N3}, it could handle
   /// this->MDReshape<4>(ptr, N1, std::array<int,2> {2, N2/2}, N3);

   // Parameter R could be omitted with c++14 standard's deduced return types

   // first method with given data pointer and rest of arguments
   template <int R, int m = 0, int M = 0, typename... Ts>
   inline auto MDReshape(T *ptr, Ts&&... args) -> MDTensor<R,T>
   {
      rNd.clear();
      reshape_ptr = ptr;
      reshape_offset = 1, reshape_shifts[0] = reshape_shifts[1] = 0;
      return MDReshape<R,m,M>(std::forward<Ts>(args)...);
   }

   // variadic method, where a new block of reshape is given in argument
   template <int R, int m = 0, int M = 0, size_t P, typename... Ts>
   inline auto MDReshape(std::array<int,P> list, Ts&&... args) -> MDTensor<R,T>
   {
      reshape_shifts[0] = layout.perm[m]; // store layout shift begin
      int shifted_layout = reshape_shifts[1] + layout.perm[m];
      for (int dim: list)
      {
         rNd.push_back(dim);
         rLt[m].push_back(sub_layout_pair{shifted_layout,-1});
         shifted_layout += 1; // default left layout
      }
      reshape_shifts[1] += P-1; // update end
      return MDReshape<R,m+1,M+P>(std::forward<Ts>(args)...);
   }

   // variadic method, where a new dimension of reshape is given
   template <int R, int m = 0, int M = 0, typename... Ts>
   inline auto MDReshape(int dim, Ts&&... args) -> MDTensor<R,T>
   {
      rNd.push_back(dim);
      const int shift =
         reshape_shifts[0] < layout.perm[m] ? reshape_shifts[1] : 0;
      rLt[m].push_back(sub_layout_pair{layout.perm[m] + shift,-1});
      return MDReshape<R,m+1,M+1>(std::forward<Ts>(args)...);
   }

   // terminal case which returns the resulting MDTensor
   template <int R, int m = 0, int M = 0>
   inline MDTensor<R,T> MDReshape()
   {
      int k = 0, rLt_idx[M], rSd[M];
      // initialize sub_layout_pair's second
      for (sub_layout_type &sub: rLt)
      {
         for (sub_layout_pair &p: sub) { p.second = k++; }
      }
      // scan with the previous layout (N) order the reshaped layout (M)
      for (int i = 0, j = 0; i < N; i++)
      {
         for (sub_layout_pair &p: rLt[layout[i]])
         {
            rLt_idx[j++] = p.second;
         }
      }
      // apply the reshaped layout (M)
      rSd[rLt_idx[0]] = 1;
      for (int i = 1; i < M; i++)
      {
         rSd[rLt_idx[i]] = rNd[rLt_idx[i-1]] * rSd[rLt_idx[i-1]];
      }
      // construct the MDTensor with the given pointer and reshaped sizes
      static_assert(R == M, "R != M");
      return MDTensor<R,T>(reshape_ptr, rSd);
   }

private:
   T *reshape_ptr;
   std::vector<int> rNd; // reshape sizes
   int reshape_offset, reshape_shifts[2];// shift begin & end
   using sub_layout_pair = std::pair<int,int>;
   using sub_layout_type = std::list<sub_layout_pair>;
   std::array<sub_layout_type,N> rLt; // layout
};

// md_sequence, md_extend and make_md_sequence implementation
namespace internal
{

template <typename T, int N, int mod, bool left> struct md_extend;

template <typename T, T... args, int N>
struct md_extend<md_sequence<T, args...>, N, 0, true>
{
   using type = md_sequence<T, args..., (args + N)...>;
};

template <typename T, T... args, int N>
struct md_extend<md_sequence<T, args...>, N, 1, true>
{
   using type = md_sequence<T, args..., (args + N)..., 2*N>;
};

template <typename T, T... args, int N>
struct md_extend<md_sequence<T, args...>, N, 0, false>
{
   using type = md_sequence<T, (args + N)..., args...>;
};

template <typename T, T... args, int N>
struct md_extend<md_sequence<T, args...>, N, 1, false>
{
   using type = md_sequence<T, 2*N, (args + N)..., args...>;
};

template <typename T, int N, bool L> struct make_md_sequence
{
   using sequence_type = typename make_md_sequence<T,N/2,L>::type;
   using type = typename md_extend<sequence_type, N/2, N%2, L>::type;
};

template <typename T, bool L>
struct make_md_sequence<T,0,L> { using type = md_sequence<T>; };

} // namespace internal

} // namespace mfem

#endif // MFEM_MDSPAN_HPP
