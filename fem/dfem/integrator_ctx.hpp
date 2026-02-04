#pragma once

#include "util.hpp"

namespace mfem::future
{

struct IntegratorContext
{
   const Array<int> *elem_attributes;
   Array<int> attributes;
   int num_entities;
   ThreadBlocks thread_blocks;
   const std::vector<FieldDescriptor> &infds;
   const IntegrationRule &ir;
};

}
