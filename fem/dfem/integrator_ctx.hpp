#pragma once

#include "util.hpp"

namespace mfem::future
{

struct IntegratorContext
{
   const ParMesh &mesh;
   const Array<int> *elem_attr;
   Array<int> attr;
   int nentities;
   ThreadBlocks thread_blocks;
   const std::vector<FieldDescriptor> &unionfds;
   const IntegrationRule &ir;
};

}
