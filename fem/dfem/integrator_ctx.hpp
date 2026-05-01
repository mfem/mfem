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
   const std::vector<FieldDescriptor> &infds;
   const std::vector<FieldDescriptor> &outfds;
   const std::vector<FieldDescriptor> &unionfds;
   const IntegrationRule &ir;
   std::unordered_map<std::type_index, std::vector<int>> &in_qlayouts;
   std::unordered_map<std::type_index, std::vector<int>> &out_qlayouts;
};

}
