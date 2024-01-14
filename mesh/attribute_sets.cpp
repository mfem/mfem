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

#include "attribute_sets.hpp"

namespace mfem
{

void AttributeSets::Copy(AttributeSets &copy) const
{
   attr_sets.Copy(copy.attr_sets);
   bdr_attr_sets.Copy(copy.bdr_attr_sets);
}

bool AttributeSets::SetsExist() const
{
   return attr_sets.Size() + bdr_attr_sets.Size() > 0;
}

void AttributeSets::GetAttributeSetNames(std::set<std::string> &names) const
{
   attr_sets.GetNames(names);
}

std::set<std::string> AttributeSets::GetAttributeSetNames() const
{
   return attr_sets.GetNames();
}

void AttributeSets::GetBdrAttributeSetNames(std::set<std::string> &names) const
{
   bdr_attr_sets.GetNames(names);
}

std::set<std::string> AttributeSets::GetBdrAttributeSetNames() const
{
   return bdr_attr_sets.GetNames();
}

bool AttributeSets::AttributeSetExists(const std::string &name) const
{
   return attr_sets.EntryExists(name);
}

bool AttributeSets::BdrAttributeSetExists(const std::string &name) const
{
   return bdr_attr_sets.EntryExists(name);
}

Array<int> & AttributeSets::CreateAttributeSet(const std::string &set_name)
{
   return attr_sets.CreateArray(set_name);
}

Array<int> & AttributeSets::CreateBdrAttributeSet(const std::string &set_name)
{
   return bdr_attr_sets.CreateArray(set_name);
}

void AttributeSets::DeleteAttributeSet(const std::string &set_name)
{
   attr_sets.DeleteArray(set_name);
}

void AttributeSets::DeleteBdrAttributeSet(const std::string &set_name)
{
   bdr_attr_sets.DeleteArray(set_name);
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

void AttributeSets::SetBdrAttributeSet(const std::string &set_name,
                                       const Array<int> &attr)
{
   if (!bdr_attr_sets.EntryExists(set_name))
   {
      bdr_attr_sets.CreateArray(set_name);
   }
   bdr_attr_sets[set_name] = attr;
   bdr_attr_sets[set_name].Sort();
   bdr_attr_sets[set_name].Unique();
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

void AttributeSets::AddToBdrAttributeSet(const std::string &set_name, int attr)
{
   bdr_attr_sets[set_name].Append(attr);
   bdr_attr_sets[set_name].Sort();
   bdr_attr_sets[set_name].Unique();
}

void AttributeSets::AddToBdrAttributeSet(const std::string &set_name,
                                         const Array<int> &attr)
{
   bdr_attr_sets[set_name].Append(attr);
   bdr_attr_sets[set_name].Sort();
   bdr_attr_sets[set_name].Unique();
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

void AttributeSets::RemoveFromBdrAttributeSet(const std::string &set_name,
                                              int attr)
{
   if (!bdr_attr_sets.EntryExists(set_name))
   {
      mfem::err << "Unrecognized boundary attribute set name \"" << set_name
                << "\" in AttributeSets::RemoveFromBdrAttributeSet"
                << std::endl;
   }

   Array<int> &bdr_attr_set = bdr_attr_sets[set_name];

   bdr_attr_set.DeleteFirst(attr);
}

Array<int> & AttributeSets::GetAttributeSet(const std::string & set_name)
{
   return attr_sets[set_name];
}

Array<int> & AttributeSets::GetBdrAttributeSet(const std::string & set_name)
{
   return bdr_attr_sets[set_name];
}

} // namespace mfem
