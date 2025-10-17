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

#ifndef MFEM_LIBCEED_MIXED_INTEGRATOR
#define MFEM_LIBCEED_MIXED_INTEGRATOR

#include "ceed.hpp"
#include "integrator.hpp"
#include <unordered_map>

namespace mfem
{

namespace ceed
{

/** @brief This class wraps a `ceed::PAIntegrator` or `ceed::MFIntegrator` to
    support mixed finite element spaces. */
template <typename CeedInteg>
class MixedIntegrator : public ceed::Operator
{
#ifdef MFEM_USE_CEED
   using ElementKey = std::pair<int, int>; //< Element::Type, Order >
   struct key_hash
   {
      std::size_t operator()(const ElementKey& k) const
      {
         return k.first + 2 * k.second;
      }
   };
   using ElementsMap = std::unordered_map<const ElementKey, int*, key_hash>;
   std::vector<CeedInteg*> sub_ops;

public:
   template <typename Integrator, typename CeedOperatorInfo, typename CoeffType>
   void Assemble(const Integrator &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 CoeffType *Q)
   {
      ElementsMap count;
      ElementsMap element_indices;
      ElementsMap offsets;

      // Count the number of elements of each type
      for (int i = 0; i < fes.GetNE(); i++)
      {
         ElementKey key(fes.GetElementType(i), fes.GetElementOrder(i));
         auto value = count.find(key);
         if (value == count.end())
         {
            count[key] = new int(1);
         }
         else
         {
            (*value->second)++;
         }
      }

      // Initialization of the arrays
      for ( const auto& value : count )
      {
         element_indices[value.first] = new int[*value.second];
         offsets[value.first] = new int(0);
      }

      // Populates the indices arrays for each element type
      for (int i = 0; i < fes.GetNE(); i++)
      {
         ElementKey key(fes.GetElementType(i), fes.GetElementOrder(i));
         int &offset = *(offsets[key]);
         int* indices_array = element_indices[key];
         indices_array[offset] = i;
         offset++;
      }

      // Create composite CeedOperator
      CeedOperatorCreateComposite(internal::ceed, &oper);

      // Create each sub-CeedOperator
      sub_ops.reserve(element_indices.size());
      for (const auto& value : element_indices)
      {
         const int* indices = value.second;
         const int first_index = indices[0];
         const mfem::FiniteElement &el = *fes.GetFE(first_index);
         auto &T = *fes.GetMesh()->GetElementTransformation(first_index);
         MFEM_ASSERT(!integ.GetIntRule(),
                     "Mixed mesh integrators should not have an"
                     " IntegrationRule.");
         const IntegrationRule &ir = GetRule(integ, el, el, T);
         auto sub_op = new CeedInteg();
         int nelem = *count[value.first];
         sub_op->Assemble(info, fes, ir, nelem, indices, Q);
         sub_ops.push_back(sub_op);
         CeedOperatorCompositeAddSub(oper, sub_op->GetCeedOperator());
      }

      const int ndofs = fes.GetVDim() * fes.GetNDofs();
      CeedVectorCreate(internal::ceed, ndofs, &u);
      CeedVectorCreate(internal::ceed, ndofs, &v);
   }

   virtual ~MixedIntegrator()
   {
      for (auto sub_op : sub_ops)
      {
         delete sub_op;
      }
   }
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_MIXED_INTEGRATOR
