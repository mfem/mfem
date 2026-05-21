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

#include "attribute_sets.hpp"

namespace mfem
{

AttributeSets::AttributeSets(const Array<int> &attr)
   : attributes(attr)
{}

void AttributeSets::Copy(AttributeSets &copy) const
{
   copy.attr_sets = attr_sets;
}

bool AttributeSets::SetsExist() const
{
   return attr_sets.Size() > 0;
}

std::set<std::string> AttributeSets::GetAttributeSetNames() const
{
   return attr_sets.GetNames();
}

bool AttributeSets::AttributeSetExists(const std::string &name) const
{
   return attr_sets.EntryExists(name);
}

Array<int> & AttributeSets::CreateAttributeSet(const std::string &set_name)
{
   return attr_sets.CreateArray(set_name);
}

void AttributeSets::DeleteAttributeSet(const std::string &set_name)
{
   attr_sets.DeleteArray(set_name);
}

void AttributeSets::SetAttributeSet(const std::string &set_name,
                                    const Array<int> &attr)
{
   if (!attr_sets.EntryExists(set_name))
   {
      attr_sets.CreateArray(set_name);
   }
   attr_sets[set_name] = attr;
   attr_sets[set_name].Sort();
   attr_sets[set_name].Unique();
}

void AttributeSets::AddToAttributeSet(const std::string &set_name, int attr)
{
   attr_sets[set_name].Append(attr);
   attr_sets[set_name].Sort();
   attr_sets[set_name].Unique();
}

void AttributeSets::AddToAttributeSet(const std::string &set_name,
                                      const Array<int> &attr)
{
   attr_sets[set_name].Append(attr);
   attr_sets[set_name].Sort();
   attr_sets[set_name].Unique();
}

void AttributeSets::RemoveFromAttributeSet(const std::string &set_name,
                                           int attr)
{
   if (!attr_sets.EntryExists(set_name))
   {
      mfem::err << "Unrecognized attribute set name \"" << set_name
                << "\" in AttributeSets::RemoveFromAttributeSet" << std::endl;
   }

   Array<int> &attr_set = attr_sets[set_name];

   attr_set.DeleteFirst(attr);
}

void AttributeSets::Print(std::ostream &os, int width) const
{
   attr_sets.Print(os, width > 0 ? width : def_width);
}

Array<int> & AttributeSets::GetAttributeSet(const std::string & set_name)
{
   return attr_sets[set_name];
}

const Array<int> &
AttributeSets::GetAttributeSet(const std::string & set_name) const
{
   return attr_sets[set_name];
}

Array<int>
AttributeSets::GetAttributeSetMarker(const std::string & set_name) const
{
   return AttrToMarker(attributes.Max(), GetAttributeSet(set_name));
}

Array<int> AttributeSets::AttrToMarker(int max_attr, const Array<int> &attrs)
{
   MFEM_VERIFY(attrs.Min() >= 1, "Found attribute less than one")
   MFEM_ASSERT(attrs.Max() <= max_attr, "Found attribute greater than max_attr")

   Array<int> marker(max_attr);
   marker = 0;
   for (auto const &attr : attrs)
   {
      marker[attr-1] = 1;
   }
   return marker;
}

} // namespace mfem
