// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <typeindex>
#include <unordered_map>
#include <vector>

#include "../../general/array.hpp"
#include "fielddescriptor.hpp"

namespace mfem::future
{

/// @brief Data describing the entities an integrator is applied to.
///
/// @tparam entity_t the entity type (see Entity). Backends deduce it from the
/// context they receive, so it selects the entity-specific restrictions,
/// DofToQuad maps and dimensions without passing it separately.
template <typename entity_t>
struct IntegratorContext
{
   using entity_type = entity_t;

   const ParMesh &mesh;
   const Array<int> *elem_attr;
   Array<int> attr;
   const int nentities;
   const std::vector<FieldDescriptor> &infds;
   const std::vector<FieldDescriptor> &outfds;
   const std::vector<FieldDescriptor> &unionfds;
   const IntegrationRule &ir;
   std::unordered_map<std::type_index, std::vector<int>> &in_qlayouts;
   std::unordered_map<std::type_index, std::vector<int>> &out_qlayouts;
};

}
