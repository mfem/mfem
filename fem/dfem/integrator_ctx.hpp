#pragma once

#include <typeindex>
#include <unordered_map>
#include <vector>

#include "linalg/vector.hpp"
#include "fielddescriptor.hpp"
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

   const struct
   {
      const bool use_kernel_specializations = false;
      const int num_entities = 0, d1d = 0, q1d = 0;
      const Array<int> *attributes = nullptr;
      const DofToQuadMap input_dtq_maps = {};
      const ThreadBlocks thread_blocks = {};
      const Array<int> *elem_attributes = nullptr;
      // ptrs
      std::function<void(std::vector<Vector> &,
                         const std::vector<Vector> &,
                         std::vector<Vector> &)> *local_restriction_callback = nullptr;
      std::vector<Vector> *local_fields_e = nullptr;
      Vector *local_residual_e = nullptr;
      std::function<void(Vector &, Vector &)> *output_restriction_transpose = nullptr;
   } local;
};

} // namespace mfem::future
