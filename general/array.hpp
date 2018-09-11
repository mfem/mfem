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

#ifndef MFEM_ARRAY
#define MFEM_ARRAY

#include "../config/config.hpp"
#include "error.hpp"
#include "globals.hpp"

#ifdef MFEM_USE_BACKENDS
#include "../backends/base/backend.hpp"
#endif

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>

namespace mfem
{

template <typename> struct remove_const;

template <typename T>
struct remove_const { typedef T type; };

template <typename T>
struct remove_const<const T> { typedef T type; };


/// TODO: doxygen
template <typename entry_t, typename idx_t>
class BArray
{
public:
   /// The array entry type - must be a POD type.
   typedef entry_t entry_type;
   /// The array index and size type - must be a signed integer type.
   typedef idx_t idx_type;

   typedef typename mfem::remove_const<entry_t>::type non_const_entry_t;
   typedef BArray<non_const_entry_t,idx_t> non_const_array_t;
   typedef BArray<const non_const_entry_t,idx_t> const_array_t;

protected:
   /// Pointer to data
   mutable entry_t *data;
   /// Size of the array
   idx_t size;
   /** @brief Size of the allocated memory. Values <= 0 mean that the data is
       not owned and abs(allocsize) is the largest size the data array can hold.
   */
   idx_t allocsize;

   /// Default (empty) constructor. Can only be used by derived classes.
   BArray() { }

   /// Copy constructor. Can only be used by derived classes.
   BArray(const BArray &orig) { InitSize(orig.Size()); *this = orig.GetData(); }

   /// TODO: doxygen
   static inline entry_t *Alloc(idx_t allocsize)
   {
      // *** default initialization:
      // return new entry_t[allocsize];
      // *** value initialization:
      // return new entry_t[allocsize]();
      // *** no initialization:
      return (entry_t *)(new char[allocsize*sizeof(entry_t)]);
   }

   /// Reallocate the data array copying @a copy_size items to the new location
   inline void Realloc(idx_t allocsize_, idx_t copy_size)
   {
      if (allocsize_ > 0)
      {
         entry_t *data_ = Alloc(allocsize_);
         MFEM_ASSERT(0 <= copy_size && copy_size <= std::min(size, allocsize_),
                     "internal error");
         std::memcpy(data_, data, copy_size*sizeof(entry_t));
         Free();
         data = data_;
         allocsize = allocsize_;
      }
      else
      {
         SetEmpty();
      }
   }

   /// TODO: doxygen
   static inline void Free(entry_t *data)
   {
      // *** invoke the destructors for all (allocsize) entries:
      // delete [] data;
      // *** do not invoke any destructors:
      delete [] (char*)data;
   }

   /// TODO: doxygen
   inline void InitSize(idx_t size_)
   {
      if (size_ > 0)
      {
         data = Alloc((size = allocsize = size_));
      }
      else
      {
         SetEmpty();
      }
   }

   inline void InitAll(entry_t *data_, idx_t size_, idx_t allocsize_)
   {
      data = (size_ > 0) ? data_ : NULL;
      size = size_;
      allocsize = allocsize_;
   }

   /// TODO: doxygen
   inline void InitDataAndSize(entry_t *data_, idx_t size_, bool own_data)
   {
      InitAll(data_, size_, own_data ? size_ : -size_);
   }

   /// TODO: doxygen
   inline void UpdateFromDev(entry_t *new_data, idx_type new_size)
   {
      size = new_size;
      if (allocsize > 0)
      {
         if (!new_data && allocsize >= size) { return; }
         Free(data);
      }
      data = (size > 0) ? new_data : NULL;
      allocsize = data ? -size : size;
   }

   /** Check if the current state of the BArray is valid. Used mainly for
       debugging. */
   inline bool Good() const
   {
      if (size >= 0 && size <= std::abs(allocsize))
      {
         return (allocsize == 0) == (data == NULL);
      }
      return false;
   }

   /// Deallocate the data if owned without resetting any object fields
   void Free() { if (OwnsData()) { Free(data); } }

   /// Initialize the array to be empty without deallocating the data
   void SetEmpty() { data = NULL; size = allocsize = 0; }

   /// Swap the contents of two arrays
   inline void Swap(BArray &other)
   {
      std::swap(data, other.data);
      std::swap(size, other.size);
      std::swap(allocsize, other.allocsize);
   }

   /// TODO: doxygen
   // grow factor = ifact + frac/den, assuming that 0 <= frac < den, and that
   // den*den fits in an int variable.
   template <int ifact, int frac, int den>
   static inline idx_t GetGrowSize(idx_t capacity)
   {
      if (ifact == 1 || capacity == 0 ||
          std::numeric_limits<idx_t>::max()/capacity >= ifact)
      {
         idx_t new_allocsize = ifact*capacity;
         if (frac%den == 0)
         {
            return new_allocsize;
         }
         else
         {
            idx_t incr = (frac%den)*(capacity/den);
            if (std::numeric_limits<idx_t>::max() - new_allocsize >= incr)
            {
               new_allocsize += incr;
               incr = ((frac%den)*(capacity%den))/den;
               if (std::numeric_limits<idx_t>::max() - new_allocsize >= incr)
               {
                  return new_allocsize + incr;
               }
            }
         }
      }
      return std::numeric_limits<idx_t>::max();
   }

   /// Reallocate the data array without copying the old contents.
   /** Assumes that @a min_allocsize > Capacity(). The Size() of the array is
       not modified. The new capacity is the larger of the numbers
       @a min_allocsize and grow_factor*Capacity(), where the grow factor is
       given by the template parameters: grow_factor = ifact + frac/2^10. */
   template <int ifact, int frac>
   inline void GrowSizeNoCopy(idx_t min_allocsize)
   {
      static const int den = 1 << 10;
      idx_t grown_size = GetGrowSize<ifact+frac/den,frac%den,den>(Capacity());
      Free();
      InitSize(std::max(min_allocsize, grown_size));
   }

   /// TODO: doxygen
   inline void SetSizeNoCopy(idx_t new_size)
   {
      if (new_size > Capacity())
      {
         GrowSizeNoCopy<2,0>(new_size);
      }
      size = new_size;
   }

   /// Reallocate the data array and copy the old contents.
   /** Assumes that @a min_allocsize > Capacity(). The Size() of the array is
       not modified. The new capacity is the larger of the numbers
       @a min_allocsize and grow_factor*Capacity(), where the grow factor is
       given by the template parameters: grow_factor = ifact + frac/2^10. */
   template <int ifact, int frac>
   inline void GrowSizeAndCopy(idx_t min_allocsize, idx_t copy_size)
   {
      static const int den = 1 << 10;
      idx_t grown_size = GetGrowSize<ifact+frac/den,frac%den,den>(Capacity());
      Realloc(std::max(min_allocsize, grown_size), copy_size);
   }

   /// TODO: doxygen
   inline void SetSizeAndCopy(idx_t new_size)
   {
      if (new_size > Capacity())
      {
         GrowSizeAndCopy<2,0>(new_size, size);
      }
      size = new_size;
   }

public:
   /// TODO: doxygen
   ~BArray() { Free(); }

   /// Return the size of the array.
   idx_t Size() const { return size; }

   /** @brief Maximum number of entries the array can store without allocating
       more memory. */
   idx_t Capacity() const { return abs(allocsize); }

   /// Clear the contents of the array, deallocating the data if owned.
   void Clear() { Free(); SetEmpty(); }

   /// TODO: doxygen
   void SetSize(idx_t new_size) { SetSizeAndCopy(new_size); }

   /// TODO: doxygen
   inline void SetSize(idx_t new_size, const entry_t &value)
   {
      const idx_t old_size = size;
      SetSizeAndCopy(new_size);
      for (idx_t i = old_size; i < new_size; i++)
      {
         data[i] = value;
      }
   }

   /// Return the array data
   entry_t *GetData() { return data; }

   /// Return the array data (const version)
   const entry_t *GetData() const { return data; }

   /// Return true if the data will be deleted by the array.
   bool OwnsData() const { return (allocsize > 0); }

   /// Transfer ownership of the data to the array
   void MakeDataOwner() { allocsize = std::abs(allocsize); }

   /// Extract the data from the array and clear its contents
   void StealData(entry_t **data_) { *data_ = data; SetEmpty(); }

   /// Extract the data from the array and clear its contents
   entry_t *StealData() { entry_t *p = data; SetEmpty(); return p; }

   /// Read-write access to the @a i-th array element
   inline entry_t &operator[](idx_t i)
   {
      MFEM_ASSERT(i >= 0 && i < size,
                  "invalid access at index [" << i << "], size = " << size);
      MFEM_ASSERT(data, "invalid data");
      return data[i];
   }

   /// Read-only access to the @a i-th array element
   inline const entry_t &operator[](idx_t i) const
   {
      MFEM_ASSERT(i >= 0 && i < size,
                  "invalid access at index [" << i << "], size = " << size);
      MFEM_ASSERT(data, "invalid data");
      return data[i];
   }

   /// Initialize all entries of the array with @a value
   inline BArray &operator=(const entry_t &value)
   {
      for (idx_t i = 0; i < size; i++)
      {
         data[i] = value;
      }
      return *this;
   }

   /// Copy Size() entries from @a src to the array
   inline BArray &operator=(const entry_t *src)
   {
      if (src != data)
      {
         MFEM_ASSERT(data + size <= src || src + size <= data,
                     "source array overlaps with data!");
         std::memcpy(data, src, Size()*sizeof(entry_t));
      }
      return *this;
   }

   /// Convert and copy Size() entries from @a src to the array
   template <typename src_t>
   inline BArray &operator=(const src_t *src)
   {
      for (idx_t i = 0; i < size; i++)
      {
         data[i] = entry_t(src[i]);
      }
      return *this;
   }

   /// TODO: doxygen
   BArray &operator=(const BArray &rhs)
   { SetSizeNoCopy(rhs.Size()); *this = rhs.GetData(); return *this; }

   /// TODO: doxygen
   template <typename rhs_entry_t, typename rhs_idx_t>
   BArray &operator=(const BArray<rhs_entry_t,rhs_idx_t> &rhs)
   { SetSizeNoCopy(rhs.Size()); *this = rhs.GetData(); return *this; }

   /// TODO: doxygen
   void Reserve(idx_t min_capacity)
   {
      if (min_capacity > Capacity())
      {
         GrowSizeAndCopy<2,0>(min_capacity, size);
      }
   }

   /// TODO: doxygen
   inline void MakeRef(entry_type *master, idx_type master_size)
   {
      Free();
      InitDataAndSize(master, master_size, false);
   }
};


/// TODO: doxygen
/// Device extension for the array and vector classes
template <typename array_t, typename dev_ext_t>
class DevExtension : public array_t
{
public:
   typedef typename array_t::entry_type entry_type;
   typedef typename array_t::idx_type idx_type;
   typedef typename array_t::non_const_array_t non_const_array_t;
   typedef typename array_t::const_array_t const_array_t;

   friend class DevExtension<const_array_t,dev_ext_t>;

protected:
   using array_t::data;
   using array_t::size;
   using array_t::allocsize;

#ifdef MFEM_USE_BACKENDS
   dev_ext_t dev_ext;

   inline void InitLayout(const DLayout &layout)
   {
      if (!layout)
      {
         this->InitSize(0);
      }
      else
      {
         this->InitLayout(*layout);
      }
   }

   inline void InitLayout(PLayout &layout)
   {
      if (!layout.HasEngine())
      {
         this->InitSize(layout.Size());
      }
      else
      {
         dev_ext = layout.Make<dev_ext_t,entry_type>();
         entry_type *data_ = dev_ext->template PullData<entry_type>(NULL);
         // data_ is NULL if dev_ext's data isn't on host
         this->InitDataAndSize(data_, dev_ext->Size(), data_ == NULL);
      }
   }
#endif

   inline void InitClone(const DevExtension &orig, bool copy_data)
   {
#ifdef MFEM_USE_BACKENDS
      if (orig.dev_ext)
      {
         entry_type *data_;
         dev_ext = orig.dev_ext->Clone(copy_data, &data_);
         MFEM_ASSERT(dev_ext != NULL, "error in Clone()");
         this->InitDataAndSize(data_, dev_ext->Size(), data_ == NULL);
      }
      else
#endif
      {
         MFEM_ASSERT(orig.size == 0 || orig.data, "invalid data array");
         this->InitSize(orig.size);
         if (copy_data) { array_t::operator=(orig.data); }
      }
   }

   inline bool DGood() const
   {
      if (size >= 0 && size <= this->Capacity())
      {
#ifdef MFEM_USE_BACKENDS
         if (dev_ext) // have dev_ext
         {
            if ((std::size_t)size == dev_ext->Size())
            {
               if (allocsize > 0)
               {
                  return dev_ext->template PullData<entry_type>(NULL) == NULL;
               }
               else if (allocsize < 0)
               {
                  return data && (size == 0 || dev_ext->template
                                  PullData<entry_type>(NULL) == data);
               }
               else // allocsize == 0
               {
                  return data == NULL;
               }
            }
         }
         else
#endif
         {
            return (allocsize == 0) == (data == NULL);
         }
      }
      return false;
   }

   inline void AssertDGood() const
   { MFEM_ASSERT(DGood(), "invalid state"); }

   DevExtension() { }

public:
#ifdef MFEM_USE_BACKENDS
   /// TODO: wrap dev_array ctor
   inline DevExtension(typename dev_ext_t::stored_type &dev_array);

   /// TODO
   inline void SetEngine(const Engine &e);

   /// TODO
   inline void MakeRef(typename dev_ext_t::stored_type &dev_array);
#endif

   using array_t::MakeRef;

   /// TODO
   /** @warning After this call, resizing @a master or @a *this will generally
       make the other DevExtension invalid. */
   inline void MakeRef(DevExtension &master);
   
   /// TODO: doxygen
   inline void MakeRefOffset(DevExtension &master, std::size_t offset);

   /// TODO: doxygen
   inline void MakeConstRef(
      const DevExtension<non_const_array_t,dev_ext_t> &master);

   /// TODO: doxygen; supports device arrays.
   inline void Resize(int new_size);

#ifdef MFEM_USE_BACKENDS
   /// TODO: doxygen; supports device arrays.
   inline void Resize(const DLayout &new_layout);
#endif

   /** @brief If the Array has an associated DArray, copy the contents of the
       DArray to the #data array. Otherwise, do nothing. */
   /** If @a copy_data is false, the content of the DArray is not copied to the
       host, however a host buffer of the appropriate size is still allocated
       and the Array data is set to point to it. If the DArray uses contiguous
       data representation in host memory, only a pointer value is copied. */
   inline void Pull(bool copy_data = true) const;

   inline void Pull(DevExtension &dst, bool copy_data = true);

   /** @brief If the Array has an associated DArray, copy the contents of the
       #data array to the DArray. Otherwise, do nothing. */
   /** If the DArray uses contiguous data representation in host memory, the
       copy operation can be unnecessary. */
   inline void Push();

   /// Assign @a value to all entries of the Array.
   /** If the Array has an associated DArray, only the DArray entries are
       set. Otherwise, only the entries of the #data array are set. */
   inline void Fill(const entry_type &value);

   /** @brief If the Array has an associated DArray, copy the @a buffer
       contents to the DArray. Otherwise, copy the @a buffer contents to the
       #data array. In both cases, Size() entries are copied. */
   inline void PushData(const entry_type *buffer);

   /// TODO: doxygen
   inline void Assign(const DevExtension &src);
};


#ifndef MFEM_USE_BACKENDS
struct DArray { };
#endif

/**
   Abstract data type Array.

   Array<T> is an automatically increasing array containing elements of the
   generic type T. The allocated size may be larger then the logical size
   of the array.
   The elements can be accessed by the [] operator, the range is 0 to size-1.

   TODO: device arrays

   Only methods whose documention explicitly states so, support device arrays.
*/
template <class T>
class Array : public DevExtension<BArray<T,int>,DArray>
{
protected:
   typedef BArray<T,int> base_class;
   typedef DevExtension<BArray<T,int>,DArray> dev_class;

   using base_class::data;
   using base_class::size;
   using base_class::allocsize;
#ifdef MFEM_USE_BACKENDS
   using dev_class::dev_ext;
#endif

public:
   // FIXME
   // typedef typename base_class::entry_type entry_type;
   // typedef typename base_class::idx_type idx_type;

   /// Creates array of asize elements
   explicit Array(int asize = 0) { base_class::InitSize(asize); }

   /// Creates array using an existing c-array of @a asize elements.
   /** The field #allocsize is set to @a -asize to indicate that the data will
       not be deleted. */
   Array(T *data, int asize, bool own_data = false)
   { base_class::InitDataAndSize(data, asize, own_data); }

   /// TODO: doxygen
   Array(int asize, const T &value)
   { base_class::InitSize(asize); *this = value; }

   /// Copy constructor: deep copy; supports device arrays.
   Array(const Array<T> &src, bool copy_data = true)
   { dev_class::InitClone(src, copy_data); }

   /// Copy constructor (deep copy) from an Array of convertable type.
   /** Does NOT support device arrays. */
   template <typename CT>
   explicit Array(const Array<CT> &src) { base_class::operator=(src); }

#ifdef MFEM_USE_BACKENDS
   /// TODO: doxygen; supports device arrays.
   /** @warning Problem: if @a layout has engine and it is not allocated itself
       with ::operator::new().

       Should we just remove this ctor?
   */
   explicit Array(PLayout &layout) { dev_class::InitLayout(layout); }

   /// TODO: doxygen; supports device arrays.
   explicit Array(const DLayout &layout) { dev_class::InitLayout(layout); }

   /// TODO: doxygen; wraps a device array.
   /** @warning If the object @a dev_array is not itself dynamically allocated
       with ::operator::new(), e.g. if it is allocated in a block (on the stack)
       or as a sub-object of another object, it's inherited method
       RefCounted::DontDelete() must be called before the destruction of the
       wrapper object created by this constructor.

       Should we just remove this ctor?
   */
   explicit Array(PArray &dev_array)
      : dev_class(dev_array) { }
#endif

   /// Destructor; supports device arrays.
   inline ~Array() { }

#ifdef MFEM_USE_BACKENDS
   /// TODO: doxygen
   PArray *Get_PArray() { return dev_ext.Get(); }

   /// TODO: doxygen
   const PArray *Get_PArray() const { return dev_ext.Get(); }
#endif

   /// Swap the content with another array. Supports device arrays.
   inline void Swap(Array &other);

   /// Copy the data from @a *this to @a copy.
   /** The @a copy array is resized to match the size of @a *this. */
   inline void Copy(Array &copy) const { copy.base_class::operator=(*this); }

   /// Set all Size() entries of the array to @a value.
   Array &operator=(const T &value)
   { base_class::operator=(value); return *this; }

   // This next overload for operator= causes ambiguity for arrays of integers.
#if 0
   /// Copy Size() entries from the array starting at @a src.
   Array &operator=(const T *src)
   { base_class::operator=(src); return *this; }
#endif

   /// Assignment operator: copy the data from @a src to @a *this.
   /** The array is resized to match the size of @a src. */
   Array<T> &operator=(const Array<T> &src) { src.Copy(*this); return *this; }

   /** @brief Assignment operator: copy the data from the Array @a src to
       @a *this. The entries of @a src are explicitly converted to the type of
       the entries of @a *this. */
   template <typename CT>
   inline Array<T> &operator=(const Array<CT> &src);

   /// Return the data as 'T *'
   inline operator T *() { return data; }

   /// Return the data as 'const T *'
   inline operator const T *() const { return data; }

   /// Initialize the array to be empty without deallocating the data
   inline void LoseData() { base_class::SetEmpty(); }

   /// Append element to array, resize if necessary
   inline int Append(const T & el);

   /// Append another array to this array, resize if necessary
   inline int Append(const T *els, int nels);

   /// Append another array to this array, resize if necessary
   inline int Append(const Array<T> &els) { return Append(els, els.Size()); }

   /// Prepend an element to the array, resize if necessary
   inline int Prepend(const T &el);

   /// Return the last element in the array
   inline T &Last();
   inline const T &Last() const;

   /// Append element when it is not yet in the array, return index
   inline int Union(const T & el);

   /// Return the first index where 'el' is found; return -1 if not found
   inline int Find(const T &el) const;

   /// Do bisection search for 'el' in a sorted array; return -1 if not found.
   inline int FindSorted(const T &el) const;

   /// Delete the last entry
   inline void DeleteLast() { if (size > 0) { size--; } }

   /// Delete the first 'el' entry
   inline void DeleteFirst(const T &el);

   /// Delete whole array
   inline void DeleteAll() { base_class::Clear(); }

   inline void GetSubArray(int offset, int sa_size, Array<T> &sa);

   /// Prints array to stream with width elements per row
   void Print(std::ostream &out = mfem::out, int width = 4) const;

   /** @brief Save the Array to the stream @a out using the format @a fmt.
       The format @a fmt can be:

          0 - write the size followed by all entries
          1 - write only the entries
   */
   void Save(std::ostream &out, int fmt = 0) const;

   /** @brief Read an Array from the stream @a in using format @a fmt.
       The format @a fmt can be:

          0 - read the size then the entries
          1 - read Size() entries
   */
   void Load(std::istream &in, int fmt = 0);

   /** @brief Set the Array size to @a new_size and read that many entries from
       the stream @a in. */
   void Load(int new_size, std::istream &in)
   { this->SetSize(new_size); Load(in, 1); }

   /** @brief Find the maximal element in the array, using the comparison
       operator `<` for class T. */
   T Max() const;

   /** @brief Find the minimal element in the array, using the comparison
       operator `<` for class T. */
   T Min() const;

   /// Sorts the array. This requires operator< to be defined for T.
   void Sort() { std::sort((T*) data, (T*) data + size); }

   /// Sorts the array using the supplied comparison function object.
   template<class Compare>
   void Sort(Compare cmp) { std::sort((T*) data, (T*) data + size, cmp); }

   /** Removes duplicities from a sorted array. This requires operator== to be
       defined for T. */
   void Unique()
   {
      T* end = std::unique((T*) data, (T*) data + size);
      this->SetSize(end - (T*) data);
   }

   /// return true if the array is sorted.
   int IsSorted();

   /// Partial Sum
   void PartialSum();

   /// Sum all entries
   T Sum();

   /// Copy data from a pointer. Size() elements are copied.
   void Assign(const T *src) {base_class::operator=(src); }

   // STL-like begin/end
   inline T* begin() const { return (T*) data; }
   inline T* end() const { return (T*) data + size; }

   long MemoryUsage() const { return base_class::Capacity() * sizeof(T); }
};

template <class T>
inline bool operator==(const Array<T> &LHS, const Array<T> &RHS)
{
   if ( LHS.Size() != RHS.Size() ) { return false; }
   for (int i=0; i<LHS.Size(); i++)
      if ( LHS[i] != RHS[i] ) { return false; }
   return true;
}

template <class T>
inline bool operator!=(const Array<T> &LHS, const Array<T> &RHS)
{
   return !( LHS == RHS );
}


template <class T>
class Array2D;

template <class T>
void Swap(Array2D<T> &, Array2D<T> &);

/// Dynamic 2D array using row-major layout
template <class T>
class Array2D
{
private:
   friend void Swap<T>(Array2D<T> &, Array2D<T> &);

   Array<T> array1d;
   int M, N; // number of rows and columns

public:
   Array2D() { M = N = 0; }
   Array2D(int m, int n) : array1d(m*n) { M = m; N = n; }

   void SetSize(int m, int n) { array1d.SetSize(m*n); M = m; N = n; }

   int NumRows() const { return M; }
   int NumCols() const { return N; }

   inline const T &operator()(int i, int j) const;
   inline       T &operator()(int i, int j);

   inline const T *operator[](int i) const;
   inline T       *operator[](int i);

   const T *operator()(int i) const { return (*this)[i]; }
   T       *operator()(int i)       { return (*this)[i]; }

   const T *GetRow(int i) const { return (*this)[i]; }
   T       *GetRow(int i)       { return (*this)[i]; }

   /// Extract a copy of the @a i-th row into the Array @a sa.
   void GetRow(int i, Array<T> &sa) const
   {
      sa.SetSize(N);
      sa.Assign(GetRow(i));
   }

   /** @brief Save the Array2D to the stream @a out using the format @a fmt.
       The format @a fmt can be:

          0 - write the number of rows and columns, followed by all entries
          1 - write only the entries, using row-major layout
   */
   void Save(std::ostream &out, int fmt = 0) const
   {
      if (fmt == 0) { out << NumRows() << ' ' << NumCols() << '\n'; }
      array1d.Save(out, 1);
   }

   /** @brief Read an Array2D from the stream @a in using format @a fmt.
       The format @a fmt can be:

          0 - read the number of rows and columns, then the entries
          1 - read NumRows() x NumCols() entries, using row-major layout
   */
   void Load(std::istream &in, int fmt = 0)
   {
      if (fmt == 0) { in >> M >> N; array1d.SetSize(M*N); }
      array1d.Load(in, 1);
   }

   /// Read an Array2D from a file
   void Load(const char *filename, int fmt = 0);

   /** @brief Set the Array2D dimensions to @a new_size0 x @a new_size1 and read
       that many entries from the stream @a in. */
   void Load(int new_size0,int new_size1, std::istream &in)
   { SetSize(new_size0,new_size1); Load(in, 1); }

   void Copy(Array2D &copy) const
   { copy.M = M; copy.N = N; array1d.Copy(copy.array1d); }

   inline void operator=(const T &a)
   { array1d = a; }

   /// Make this Array a reference to 'master'
   inline void MakeRef(Array2D &master)
   { M = master.M; N = master.N; array1d.MakeRef(master.array1d); }

   /// Delete all dynamically allocated memory, reseting all dimentions to zero.
   inline void DeleteAll() { M = 0; N = 0; array1d.DeleteAll(); }

   /// Prints array to stream with width elements per row
   void Print(std::ostream &out = mfem::out, int width = 4);
};


template <class T>
class Array3D
{
private:
   Array<T> array1d;
   int N2, N3;

public:
   Array3D() { N2 = N3 = 0; }
   Array3D(int n1, int n2, int n3)
      : array1d(n1*n2*n3) { N2 = n2; N3 = n3; }

   void SetSize(int n1, int n2, int n3)
   { array1d.SetSize(n1*n2*n3); N2 = n2; N3 = n3; }

   inline const T &operator()(int i, int j, int k) const;
   inline       T &operator()(int i, int j, int k);
};


/** A container for items of type T. Dynamically grows as items are added.
 *  Each item is accessible by its index. Items are allocated in larger chunks
 *  (blocks), so the 'Append' method is very fast on average.
 */
template<typename T>
class BlockArray
{
public:
   BlockArray(int block_size = 16*1024);
   BlockArray(const BlockArray<T> &other); // deep copy
   ~BlockArray();

   /// Allocate and construct a new item in the array, return its index.
   int Append();

   /// Allocate and copy-construct a new item in the array, return its index.
   int Append(const T &item);

   /// Access item of the array.
   inline T& At(int index)
   {
      CheckIndex(index);
      return blocks[index >> shift][index & mask];
   }
   inline const T& At(int index) const
   {
      CheckIndex(index);
      return blocks[index >> shift][index & mask];
   }

   /// Access item of the array.
   inline T& operator[](int index) { return At(index); }
   inline const T& operator[](int index) const { return At(index); }

   /// Return the number of items actually stored.
   int Size() const { return size; }

   /// Return the current capacity of the BlockArray.
   int Capacity() const { return blocks.Size()*(mask+1); }

   void Swap(BlockArray<T> &other);

   long MemoryUsage() const;

protected:
   template <typename cA, typename cT>
   class iterator_base
   {
   public:
      cT& operator*() const { return *ptr; }
      cT* operator->() const { return ptr; }

      bool good() const { return !stop; }
      int index() const { return (ptr - ref); }

   protected:
      cA *array;
      cT *ptr, *b_end, *ref;
      int b_end_idx;
      bool stop;

      iterator_base() { }
      iterator_base(bool stop) : stop(stop) { }
      iterator_base(cA *a)
         : array(a), ptr(a->blocks[0]), ref(ptr), stop(false)
      {
         b_end_idx = std::min(a->size, a->mask+1);
         b_end = ptr + b_end_idx;
      }

      void next()
      {
         MFEM_ASSERT(!stop, "invalid use");
         if (++ptr == b_end)
         {
            if (b_end_idx < array->size)
            {
               ptr = &array->At(b_end_idx);
               ref = ptr - b_end_idx;
               b_end_idx = std::min(array->size, (b_end_idx|array->mask) + 1);
               b_end = &array->At(b_end_idx-1) + 1;
            }
            else
            {
               MFEM_ASSERT(b_end_idx == array->size, "invalid use");
               stop = true;
            }
         }
      }
   };

public:
   class iterator : public iterator_base<BlockArray, T>
   {
   protected:
      friend class BlockArray;
      typedef iterator_base<BlockArray, T> base;

      iterator() { }
      iterator(bool stop) : base(stop) { }
      iterator(BlockArray *a) : base(a) { }

   public:
      iterator &operator++() { base::next(); return *this; }

      bool operator==(const iterator &other) const { return base::stop; }
      bool operator!=(const iterator &other) const { return !base::stop; }
   };

   class const_iterator : public iterator_base<const BlockArray, const T>
   {
   protected:
      friend class BlockArray;
      typedef iterator_base<const BlockArray, const T> base;

      const_iterator() { }
      const_iterator(bool stop) : base(stop) { }
      const_iterator(const BlockArray *a) : base(a) { }

   public:
      const_iterator &operator++() { base::next(); return *this; }

      bool operator==(const const_iterator &other) const { return base::stop; }
      bool operator!=(const const_iterator &other) const { return !base::stop; }
   };

   iterator begin() { return size ? iterator(this) : iterator(true); }
   iterator end() { return iterator(); }

   const_iterator cbegin() const
   { return size ? const_iterator(this) : const_iterator(true); }
   const_iterator cend() const { return const_iterator(); }

protected:
   Array<T*> blocks;
   int size, shift, mask;

   int Alloc();

   inline void CheckIndex(int index) const
   {
      MFEM_ASSERT(index >= 0 && index < size,
                  "Out of bounds access: " << index << ", size = " << size);
   }
};


/// inlines ///


#ifdef MFEM_USE_BACKENDS

template <typename array_t, typename dev_ext_t>
inline DevExtension<array_t,dev_ext_t>::DevExtension(
   typename dev_ext_t::stored_type &dev_array)
   : dev_ext(&dev_array)
{
   entry_type *data_ = dev_ext->template PullData<entry_type>(NULL);
   this->InitDataAndSize(data_, dev_ext->Size(), data_ == NULL);
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::SetEngine(const Engine &e)
{
   if (!dev_ext || (&dev_ext->GetLayout().GetEngine() != &e))
   {
      this->Free();
      InitLayout(*e.MakeLayout(0));
   }
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::MakeRef(
   typename dev_ext_t::stored_type &dev_array)
{
   dev_ext.Reset(&dev_array);
   entry_type *new_data = dev_ext->template PullData<entry_type>(NULL);
   this->UpdateFromDev(new_data, dev_ext->Size());
}
#endif

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::MakeRef(DevExtension &master)
{
#ifdef MFEM_USE_BACKENDS
   dev_ext = master.dev_ext;
   if (dev_ext)
   {
      this->UpdateFromDev(master.OwnsData() ? NULL : master.GetData(),
                          master.Size());
   }
   else
#endif
   {
      this->Free();
      this->InitAll(master.GetData(), master.Size(), -master.Capacity());
   }
}

// *****************************************************************************
#include <cassert>
#define ddbg(...) //printf("\n\033[31;1m");printf(__VA_ARGS__);printf("\033[m");fflush(0);
template <typename array_t, typename dev_ext_t> inline
void DevExtension<array_t,dev_ext_t>::MakeRefOffset(DevExtension &src,
                                                    std::size_t offset)
{
   const std::size_t size = this->Size();
#ifdef MFEM_USE_BACKENDS
   dev_ext.Reset();
   if (src.dev_ext){
      if (src.OwnsData()){
         ddbg("[MakeRefOffset] device holds data");
         dev_ext = src.dev_ext->GetLayout().GetEngine().MakeLayout(0)->template Make<dev_ext_t,entry_type>();
         dev_ext->template MakeRefOffset<entry_type>(*src.dev_ext,offset,size);
      } else {
         ddbg("[MakeRefOffset] device ready, but data on the host (size=%ld)",size);
         dev_ext = src.dev_ext->GetLayout().GetEngine().MakeLayout(0)->template Make<dev_ext_t,entry_type>();
         dev_ext->template MakeRefOffset<entry_type>(*src.dev_ext,offset,size);
         entry_type *data_ = dev_ext->template PullData<entry_type>(NULL);
         // data_ is NULL if dev_ext's data isn't on host
         this->InitDataAndSize(data_, dev_ext->Size(), data_ == NULL);
      }
   }else
#endif
   {
      ddbg("[MakeRefOffset] host only");
      this->Free();
      this->InitAll(src.GetData()+offset, size, -size);
   }
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::MakeConstRef(
   const DevExtension<non_const_array_t,dev_ext_t> &master)
{
#ifdef MFEM_USE_BACKENDS
   dev_ext = master.dev_ext;
   if (dev_ext)
   {
      this->UpdateFromDev(master.OwnsData() ? NULL : master.GetData(),
                          master.Size());
   }
   else
#endif
   {
      this->Free();
      this->InitAll(master.GetData(), master.Size(), -master.Capacity());
   }
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Resize(int new_size)
{
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   if (dev_ext)
   {
      entry_type *new_data;
      MFEM_DEBUG_DO(int err =) dev_ext->Resize(new_size, &new_data);
      MFEM_ASSERT(err == 0, "memory overflow");
      this->UpdateFromDev(new_data, new_size);
   }
   else
#endif
   {
      this->SetSize(new_size);
   }
}

#ifdef MFEM_USE_BACKENDS
template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Resize(const DLayout &new_layout)
{
   AssertDGood();
   if (!new_layout || !new_layout->HasEngine())
   {
      Resize(new_layout ? new_layout->Size() : 0);
   }
   else if (!dev_ext)
   {
      this->Free();
      InitLayout(new_layout);
   }
   else
   {
      entry_type *new_data;
      MFEM_DEBUG_DO(int err =) dev_ext->Resize(*new_layout, &new_data);
      MFEM_ASSERT(err == 0, "memory overflow");
      this->UpdateFromDev(new_data, new_layout->Size());
   }
}
#endif

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Pull(bool copy_data) const
{
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   if (dev_ext && allocsize > 0)
   {
      if (!data) { data = this->Alloc(allocsize); }
      if (copy_data)
      {
         MFEM_DEBUG_DO(entry_type *new_data =) dev_ext->PullData(data);
         MFEM_ASSERT(size == 0 || new_data == data, "internal error");
      }
   }
#endif
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Pull(DevExtension &dst,
                                                  bool copy_data)
{
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   if (dev_ext)
   {
      if (allocsize > 0)
      {
         dst.SetSizeNoCopy(size);
         if (copy_data) { dev_ext->PullData(dst.data); }
      }
      else
      {
         dst.dev_ext.Reset();
         dst.Free();
         dst.InitAll(data, size, allocsize);
      }
   }
   else
#endif
   {
      dst.MakeRef(*this);
   }
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Push()
{
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   if (dev_ext && size > 0)
   {
      MFEM_ASSERT(data != NULL, "invalid data");
      dev_ext->PushData(data);
   }
#endif
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Fill(const entry_type &value)
{
#ifdef MFEM_USE_BACKENDS
   dev_ext ? dev_ext->Fill(value) : (void) array_t::operator=(value);
#else
   array_t::operator=(value);
#endif
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::PushData(const entry_type *buffer)
{
   MFEM_ASSERT(size == 0 || buffer != NULL, "");
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   dev_ext ? dev_ext->PushData(buffer) : (void) array_t::operator=(buffer);
#else
   array_t::operator=(buffer);
#endif
}

template <typename array_t, typename dev_ext_t>
inline void DevExtension<array_t,dev_ext_t>::Assign(const DevExtension &src)
{
#ifdef MFEM_USE_BACKENDS
   MFEM_ASSERT(bool(dev_ext) == bool(src.dev_ext), "incompatible arrays");
#endif
   AssertDGood();
#ifdef MFEM_USE_BACKENDS
   (dev_ext ? dev_ext->template Assign<entry_type>(*src.dev_ext) :
    (void) array_t::operator=(src));
#else
   array_t::operator=(src);
#endif
}


template <class T>
inline void Swap(T &a, T &b)
{
   T c = a;
   a = b;
   b = c;
}

template <class T>
inline void Swap(Array<T> &a, Array<T> &b)
{
   a.Swap(b);
}


template <typename T>
inline void Array<T>::Swap(Array &other)
{
   base_class::Swap(other);
#ifdef MFEM_USE_BACKENDS
   mfem::Swap(dev_ext, other.dev_ext);
#endif
}

template <typename T> template <typename CT>
inline Array<T> &Array<T>::operator=(const Array<CT> &src)
{
   base_class::operator=(src);
   return *this;
}

template <class T>
inline int Array<T>::Append(const T &el)
{
   this->SetSize(size+1, el);
   return size;
}

template <class T>
inline int Array<T>::Append(const T *els, int nels)
{
   const int old_size = size;

   this->SetSize(size + nels);
   for (int i = 0; i < nels; i++)
   {
      data[old_size+i] = els[i];
   }
   return size;
}

template <class T>
inline int Array<T>::Prepend(const T &el)
{
   this->SetSize(size+1);
   for (int i = size-1; i > 0; i--)
   {
      data[i] = data[i-1];
   }
   data[0] = el;
   return size;
}

template <class T>
inline T &Array<T>::Last()
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return data[size-1];
}

template <class T>
inline const T &Array<T>::Last() const
{
   MFEM_ASSERT(size > 0, "Array size is zero: " << size);
   return data[size-1];
}

template <class T>
inline int Array<T>::Union(const T &el)
{
   int i = 0;
   while ((i < size) && (data[i] != el)) { i++; }
   if (i == size)
   {
      Append(el);
   }
   return i;
}

template <class T>
inline int Array<T>::Find(const T &el) const
{
   for (int i = 0; i < size; i++)
   {
      if (data[i] == el) { return i; }
   }
   return -1;
}

template <class T>
inline int Array<T>::FindSorted(const T &el) const
{
   const T *begin = data, *end = begin + size;
   const T* first = std::lower_bound(begin, end, el);
   if (first == end || !(*first == el)) { return  -1; }
   return first - begin;
}

template <class T>
inline void Array<T>::DeleteFirst(const T &el)
{
   for (int i = 0; i < size; i++)
   {
      if (data[i] == el)
      {
         for (i++; i < size; i++)
         {
            data[i-1] = data[i];
         }
         size--;
         return;
      }
   }
}

template <class T>
inline void Array<T>::GetSubArray(int offset, int sa_size, Array<T> &sa)
{
   sa.SetSize(sa_size);
   for (int i = 0; i < sa_size; i++)
   {
      sa[i] = (*this)[offset+i];
   }
}


template <class T>
inline const T &Array2D<T>::operator()(int i, int j) const
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N && j>=0 && j<N,
                "Array2D: invalid access of element (" << i << ',' << j
                << ") in array of size (" << array1d.Size()/N << ',' << N
                << ")." );
   return array1d[i*N+j];
}

template <class T>
inline T &Array2D<T>::operator()(int i, int j)
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N && j>=0 && j<N,
                "Array2D: invalid access of element (" << i << ',' << j
                << ") in array of size (" << array1d.Size()/N << ',' << N
                << ")." );
   return array1d[i*N+j];
}

template <class T>
inline const T *Array2D<T>::operator[](int i) const
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N,
                "Array2D: invalid access of row " << i << " in array with "
                << array1d.Size()/N << " rows.");
   return &array1d[i*N];
}

template <class T>
inline T *Array2D<T>::operator[](int i)
{
   MFEM_ASSERT( i>=0 && i< array1d.Size()/N,
                "Array2D: invalid access of row " << i << " in array with "
                << array1d.Size()/N << " rows.");
   return &array1d[i*N];
}


template <class T>
inline void Swap(Array2D<T> &a, Array2D<T> &b)
{
   Swap(a.array1d, b.array1d);
   Swap(a.M, b.M);
   Swap(a.N, b.N);
}


template <class T>
inline const T &Array3D<T>::operator()(int i, int j, int k) const
{
   MFEM_ASSERT(i >= 0 && i < array1d.Size() / N2 / N3 && j >= 0 && j < N2
               && k >= 0 && k < N3,
               "Array3D: invalid access of element ("
               << i << ',' << j << ',' << k << ") in array of size ("
               << array1d.Size() / N2 / N3 << ',' << N2 << ',' << N3 << ").");
   return array1d[(i*N2+j)*N3+k];
}

template <class T>
inline T &Array3D<T>::operator()(int i, int j, int k)
{
   MFEM_ASSERT(i >= 0 && i < array1d.Size() / N2 / N3 && j >= 0 && j < N2
               && k >= 0 && k < N3,
               "Array3D: invalid access of element ("
               << i << ',' << j << ',' << k << ") in array of size ("
               << array1d.Size() / N2 / N3 << ',' << N2 << ',' << N3 << ").");
   return array1d[(i*N2+j)*N3+k];
}


template<typename T>
BlockArray<T>::BlockArray(int block_size)
{
   mask = block_size-1;
   MFEM_VERIFY(!(block_size & mask), "block_size must be a power of two.");

   size = shift = 0;
   while ((1 << shift) < block_size) { shift++; }
}

template<typename T>
BlockArray<T>::BlockArray(const BlockArray<T> &other)
{
   blocks.SetSize(other.blocks.Size());

   size = other.size;
   shift = other.shift;
   mask = other.mask;

   int bsize = mask+1;
   for (int i = 0; i < blocks.Size(); i++)
   {
      blocks[i] = (T*) new char[bsize * sizeof(T)];
   }

   // copy all items
   for (int i = 0; i < size; i++)
   {
      new (&At(i)) T(other[i]);
   }
}

template<typename T>
int BlockArray<T>::Alloc()
{
   int bsize = mask+1;
   if (size >= blocks.Size() * bsize)
   {
      T* new_block = (T*) new char[bsize * sizeof(T)];
      blocks.Append(new_block);
   }
   return size++;
}

template<typename T>
int BlockArray<T>::Append()
{
   int index = Alloc();
   new (&At(index)) T();
   return index;
}

template<typename T>
int BlockArray<T>::Append(const T &item)
{
   int index = Alloc();
   new (&At(index)) T(item);
   return index;
}

template<typename T>
void BlockArray<T>::Swap(BlockArray<T> &other)
{
   mfem::Swap(blocks, other.blocks);
   std::swap(size, other.size);
   std::swap(shift, other.shift);
   std::swap(mask, other.mask);
}

template<typename T>
long BlockArray<T>::MemoryUsage() const
{
   return blocks.Size()*(mask+1)*sizeof(T) +
          blocks.MemoryUsage();
}

template<typename T>
BlockArray<T>::~BlockArray()
{
   int bsize = size & mask;
   for (int i = blocks.Size(); i != 0; )
   {
      T *block = blocks[--i];
      for (int j = bsize; j != 0; )
      {
         block[--j].~T();
      }
      delete [] (char*) block;
      bsize = mask+1;
   }
}

} // namespace mfem

#endif
