// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ARRAYS_BY_NAME
#define MFEM_ARRAYS_BY_NAME

#include "../config/config.hpp"
#include "array.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>

namespace mfem
{

/**
   Container class for storing arrays indexed by strings.

   The Array<T> objects stored within this container must all be based on the
   same underlying generic type T, which must be a trivial type, see
   `std::is_trivial`.

   In order to provide some level of protection against typos this class will
   not create new named arrays when access to unrecognized names is requested.
   New named arrays must be explicitly created using `CreateArray()`. To
   facilitate this behavior and avoid such errors the method `EntryExists()` is
   provided.

   This container does not store pointers to pre-existing arrays. It will copy
   or move entries as appropriate from existing Array<T> objects into new
   Array<T> objects stored within this container.
*/
template <class T>
class ArraysByName
{
protected:
   /// Reusing STL map iterators
   using container      = std::map<std::string,Array<T> >;
   using iterator       = typename container::iterator;
   using const_iterator = typename container::const_iterator;

   /// Map containing the data sorted alphabetically by name
   container data;

public:

   /// Default constructor
   ArraysByName() = default;

   /// Copy constructor: deep copy from @a src
   ArraysByName(const ArraysByName &src) = default;

   /// Move constructor
   ArraysByName(ArraysByName &&src) noexcept = default;

   /// Return the number of named arrays in the container
   int Size() const { return static_cast<int>(data.size()); }

   /// Return an STL set of strings giving the names of the arrays
   inline std::set<std::string> GetNames() const;

   /// @brief Return true if an array with the given name is present in the
   /// container
   inline bool EntryExists(const std::string &name) const;

   /// @brief Reference access to the named entry.
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline Array<T> &operator[](const std::string &name);

   /// @brief Const reference access to the named entry.
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline const Array<T> &operator[](const std::string &name) const;

   /// @brief Create a new empty array with the given name
   ///
   /// @note Passing a name for an already existent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline Array<T> &CreateArray(const std::string &name);

   /// Delete all named arrays from the container
   inline void DeleteAll();

   /// @brief Delete the named array from the container
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline void DeleteArray(const std::string &name);

   /// Copy assignment operator: deep copy from 'src'.
   ArraysByName<T> &operator=(const ArraysByName<T> &src) = default;

   /// Move assignment operator
   ArraysByName<T> &operator=(ArraysByName<T> &&src) noexcept = default;

   /// @brief Print the contents of the container to an output stream
   ///
   /// @note Each array will be printed on at least three lines; the name on
   /// one line, the length of the associated array, lastly the array contents
   /// with @a width entries per line. A specific number of entries per line
   /// can be used by changing the @a width argument.
   inline void Print(std::ostream &out = mfem::out, int width = -1) const;

   /// @brief Load the contents of the container from an input stream
   ///
   /// @note This method will not first empty the container. First call
   /// DeleteAll if this behavior is needed.
   void Load(std::istream &in);

   /// Sort each named array in the container
   inline void SortAll();

   /// @brief Remove duplicates from each, previously sorted, named array
   ///
   /// @note Identical entries may exist in multiple arrays but will only occur
   /// at most once in each array.
   inline void UniqueAll();

   /// STL-like begin.  Returns pointer to the first entry of the container.
   iterator begin() { return data.begin(); }

   /// STL-like end.  Returns pointer after the last entry of the container.
   iterator end() { return data.end(); }

   /// @brief STL-like begin.  Returns const pointer to the first entry of the
   /// container.
   const_iterator begin() const { return data.cbegin(); }

   /// @brief STL-like end.  Returns const pointer after the last entry of the
   /// container.
   const_iterator end() const { return data.cend(); }
};

template <class T>
inline bool operator==(const ArraysByName<T> &LHS, const ArraysByName<T> &RHS)
{
   if ( LHS.Size() != RHS.Size() ) { return false; }
   for (auto it1 = LHS.begin(), it2 = RHS.begin();
        it1 != LHS.end() && it2 != RHS.end(); it1++, it2++)
   {
      if (it1->first != it2->first) { return false; }
      if (it1->second != it2->second) { return false; }
   }
   return true;
}

template<class T>
inline std::set<std::string> ArraysByName<T>::GetNames() const
{
   std::set<std::string> names;
   for (auto const &entry : data)
   {
      names.insert(entry.first);
   }
   return names;
}

template<class T>
inline bool ArraysByName<T>::EntryExists(const std::string &name) const
{
   return data.find(name) != data.end();
}

template<class T>
inline Array<T> &ArraysByName<T>::operator[](const std::string &name)
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Access to unknown named array \"" << name << "\"");
   return data[name];
}

template<class T>
inline const Array<T> &ArraysByName<T>::operator[](const std::string &name)
const
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Access to unknown named array \"" << name << "\"");
   return data.at(name);
}

template<class T>
inline Array<T> &ArraysByName<T>::CreateArray(const std::string &name)
{
   MFEM_VERIFY( data.find(name) == data.end(),
                "Named array \"" << name << "\" already exists");
   Array<T> empty_array;
   data.insert(std::pair<std::string,Array<T> >(name,empty_array));
   return data[name];
}

template<class T>
inline void ArraysByName<T>::DeleteAll()
{
   data.clear();
}

template<class T>
inline void ArraysByName<T>::DeleteArray(const std::string &name)
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Attempting to delete unknown named array \"" << name << "\"");
   data.erase(name);
}

template <class T>
inline void ArraysByName<T>::SortAll()
{
   for (auto &a : data)
   {
      a.second.Sort();
   }
}

template <class T>
inline void ArraysByName<T>::UniqueAll()
{
   for (auto &a : data)
   {
      a.second.Unique();
   }
}

template <class T>
inline void ArraysByName<T>::Print(std::ostream &os, int width) const
{
   os << data.size() << '\n';
   for (auto const &it : data)
   {
      os << '"' << it.first << '"' << '\n' << it.second.Size() << '\n';
      it.second.Print(os, width > 0 ? width : it.second.Size());
   }
}

template <class T>
void ArraysByName<T>::Load(std::istream &in)
{
   int NumArrays;
   in >> NumArrays;

   std::string ArrayLine, ArrayName;
   for (int i=0; i < NumArrays; i++)
   {
      in >> std::ws;
      getline(in, ArrayLine);

      std::size_t q0 = ArrayLine.find('"');
      std::size_t q1 = ArrayLine.rfind('"');

      if (q0 != std::string::npos && q1 > q0)
      {
         // Locate set name between first and last double quote
         ArrayName = ArrayLine.substr(q0+1,q1-q0-1);
      }
      else
      {
         // If no double quotes found locate set name using white space
         q1 = ArrayLine.find(' ');
         ArrayName = ArrayLine.substr(0,q1-1);
      }

      // Ignore the remainder of the line which may contain explanatory comments
      data[ArrayName].Load(in, 0);
   }

}

}

#endif
