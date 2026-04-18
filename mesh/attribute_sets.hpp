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

#ifndef MFEM_ATTRIBUTE_SETS
#define MFEM_ATTRIBUTE_SETS

#include "../config/config.hpp"
#include "../general/arrays_by_name.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>

namespace mfem
{

class AttributeSets
{
private:
   const Array<int> & attributes;

   const int def_width = 10;

public:
   /// Named sets of attributes
   ArraysByName<int> attr_sets;

   AttributeSets(const Array<int> &attr);

   /// @brief Create a copy of the internal data to the provided @a copy.
   void Copy(AttributeSets &copy) const;

   /// @brief Return true if any named sets are currently defined
   bool SetsExist() const;

   /// @brief Return all attribute set names as an STL set
   std::set<std::string> GetAttributeSetNames() const;

   /// @brief Return true is the named attribute set is present
   bool AttributeSetExists(const std::string &name) const;

   /// @brief Create an empty named attribute set
   Array<int> & CreateAttributeSet(const std::string &set_name);

   /// @brief Delete a named attribute set
   void DeleteAttributeSet(const std::string &set_name);

   /// @brief Create a new attribute set
   /**
       @param[in] set_name The name of the new set
       @param[in] attr An array of attribute numbers making up the new set

       @note If an attribute set matching this name already exists, that set
       will be replaced with this new attribute set.

       @note The attribute numbers are not checked for validity or
       existence within the mesh.
    */
   void SetAttributeSet(const std::string &set_name, const Array<int> &attr);

   /// @brief Add a single entry to an existing attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr A single attribute number to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToAttributeSet(const std::string &set_name, int attr);

   /// @brief Add an array of entries to an existing attribute set
   /**
       @param[in] set_name The name of the set being augmented
       @param[in] attr Array of attribute numbers to be inserted in the set

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.
       @note Duplicate entries will be ignored and the resulting sets will be
       sorted.
    */
   void AddToAttributeSet(const std::string &set_name, const Array<int> &attr);

   /// @brief Remove a single entry from an existing attribute set
   /**
       @param[in] set_name The name of the set being modified
       @param[in] attr A single attribute number to be removed from the set

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.
       @note If @a attr is not a member of the named set the set will not
       be modified and no error will occur.
    */
   void RemoveFromAttributeSet(const std::string &set_name, int attr);

   /// @brief Print the contents of the container to an output stream
   ///
   /// @note The array entries will contain 10 entries per line. A specific
   /// number of entries per line can be used by changing the @a width argument.
   void Print(std::ostream &out = mfem::out, int width = -1) const;

   /// @brief Access a named attribute set
   /**
       @param[in] set_name The name of the set being accessed

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.

       @note The reference returned by this method can be invalidated by
       subsequent calls to SetAttributeSet, ClearAttributeSet, or
       RemoveFromAttributeSet. AddToAttributeSet should not invalidate this
       reference.
    */
   Array<int> & GetAttributeSet(const std::string & set_name);

   /// @brief Access a constant reference to a named attribute set
   /**
       @param[in] set_name The name of the set being accessed

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.

       @note The reference returned by this method can be invalidated by
       subsequent calls to SetAttributeSet, ClearAttributeSet, or
       RemoveFromAttributeSet. AddToAttributeSet should not invalidate this
       reference.
    */
   const Array<int> & GetAttributeSet(const std::string & set_name) const;

   /// @brief Return a marker array corresponding to a named attribute set
   /**
       @param[in] set_name The name of the set being accessed

       @note If the named set does not exist an error message will be printed
       and execution will halt. `AttributeSetExists()` may be used to verify
       existence of a named set.
    */
   Array<int> GetAttributeSetMarker(const std::string & set_name) const;

   /// @brief Prepares a marker array corresponding to an array of element
   /// attributes
   /**
       @param[in] max_attr Number of entries to create in the @a marker array
       @param[in] attrs    An array of attribute numbers which should be
                           activated

       The returned marker array will be of size @a max_attr and it will contain
       only zeroes and ones. Ones indicate which attribute numbers are present
       in the @a attrs array.
    */
   static Array<int> AttrToMarker(int max_attr, const Array<int> &attrs);
};

} // namespace mfem

#endif
