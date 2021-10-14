// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "submesh.hpp"

using namespace mfem;

SubMesh::SubMesh(Mesh &parent, Array<int> element_ids,
                 Array<int> attributes) : parent_(parent),
   attributes_(attributes), element_ids_(element_ids)
{
}

SubMesh::~SubMesh() {}

SubMesh SubMesh::CreateFromDomain(Mesh &parent,
                                  Array<int> domain_attributes)
{
   Array<int> element_ids;
   return SubMesh(parent, domain_attributes, element_ids);
}


SubMesh SubMesh::CreateFromBoundary(Mesh &parent,
                                    Array<int> boundary_attributes)
{
   Array<int> element_ids;
   return SubMesh(parent, boundary_attributes, element_ids);
}
