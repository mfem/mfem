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

#ifndef MFEM_LIBCEED_MIXED_INTEGRATOR
#define MFEM_LIBCEED_MIXED_INTEGRATOR

#include <array>
#include <unordered_map>
#include "integrator.hpp"
#include "util.hpp"
#include "ceed.hpp"

namespace mfem
{

namespace ceed
{

/** @brief This class wraps one or more `ceed::Integrator` objects to support
    finite element spaces on mixed meshes. */
class MixedIntegrator : public Operator
{
#ifdef MFEM_USE_CEED
   using ElementKey =
      std::array<int, 3>; // <mfem::Element::Type, TrialOrder, TestOrder>
   struct ElementHash
   {
      std::size_t operator()(const ElementKey &k) const
      {
         return CeedHashCombine(
                   CeedHashCombine(CeedHash(k[0]), CeedHash(k[1])),
                   CeedHash(k[2]));
      }
   };
   using ElementsMap = std::unordered_map<const ElementKey, int *, ElementHash>;
   std::vector<Integrator *> sub_ops;

public:
   template <typename IntegratorType, typename CeedOperatorInfo, typename CoeffType>
   void Assemble(const IntegratorType &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(integ, info, fes, fes, Q, use_bdr, use_mf);
   }

   template <typename IntegratorType, typename CeedOperatorInfo, typename CoeffType>
   void Assemble(const IntegratorType &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      MFEM_VERIFY(trial_fes.GetMesh() == test_fes.GetMesh(),
                  "Trial and test basis must correspond to the same Mesh.");
      mfem::Mesh &mesh = *trial_fes.GetMesh();
      const bool mixed =
         mesh.GetNumGeometries(mesh.Dimension() - (use_bdr * 1)) > 1 ||
         trial_fes.IsVariableOrder() || test_fes.IsVariableOrder();
      if (!mixed)
      {
         const mfem::FiniteElement &trial_fe = use_bdr ? *trial_fes.GetBE(0) :
                                               *trial_fes.GetFE(0);
         const mfem::FiniteElement &test_fe = use_bdr ? *test_fes.GetBE(0) :
                                              *test_fes.GetFE(0);
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(0) :
                   *mesh.GetElementTransformation(0);
         const mfem::IntegrationRule &ir =
            integ.GetIntegrationRule() ? *integ.GetIntegrationRule() :
            integ.GetRule(trial_fe, test_fe, T);
         sub_ops.reserve(1);
         auto sub_op = new Integrator();
         sub_op->Assemble(info, trial_fes, test_fes, ir, Q, use_bdr, use_mf);
         sub_ops.push_back(sub_op);

         CeedOperatorReferenceCopy(sub_op->GetCeedOperator(), &oper);
         CeedVectorReferenceCopy(sub_op->GetCeedVectorU(), &u);
         CeedVectorReferenceCopy(sub_op->GetCeedVectorV(), &v);
         return;
      }

      // Count the number of elements of each type
      ElementsMap count;
      ElementsMap element_indices;
      ElementsMap offsets;

      const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
      for (int i = 0; i < ne; i++)
      {
         const mfem::FiniteElement &trial_fe = use_bdr ? *trial_fes.GetBE(i) :
                                               *trial_fes.GetFE(i);
         const mfem::FiniteElement &test_fe = use_bdr ? *test_fes.GetBE(i) :
                                              *test_fes.GetFE(i);
         mfem::Element::Type type = use_bdr ? mesh.GetBdrElementType(i) :
                                    mesh.GetElementType(i);
         ElementKey key = {type, trial_fe.GetOrder(), test_fe.GetOrder()};
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
      for (const auto &value : count)
      {
         element_indices[value.first] = new int[*value.second];
         offsets[value.first] = new int(0);
      }

      // Populates the indices arrays for each element type
      for (int i = 0; i < ne; i++)
      {
         const mfem::FiniteElement &trial_fe = use_bdr ? *trial_fes.GetBE(i) :
                                               *trial_fes.GetFE(i);
         const mfem::FiniteElement &test_fe = use_bdr ? *test_fes.GetBE(i) :
                                              *test_fes.GetFE(i);
         mfem::Element::Type type = use_bdr ? mesh.GetBdrElementType(i) :
                                    mesh.GetElementType(i);
         ElementKey key = {type, trial_fe.GetOrder(), test_fe.GetOrder()};
         int &offset = *(offsets[key]);
         int *indices_array = element_indices[key];
         indices_array[offset] = i;
         offset++;
      }

      // Create composite CeedOperator
      CeedCompositeOperatorCreate(internal::ceed, &oper);

      // Create each sub-CeedOperator
      sub_ops.reserve(element_indices.size());
      for (const auto &value : element_indices)
      {
         const int *indices = value.second;
         const int first_index = indices[0];
         const mfem::FiniteElement &trial_fe =
            use_bdr ? *trial_fes.GetBE(first_index) : *trial_fes.GetFE(first_index);
         const mfem::FiniteElement &test_fe =
            use_bdr ? *test_fes.GetBE(first_index) : *test_fes.GetFE(first_index);
         auto &T = use_bdr ? *mesh.GetBdrElementTransformation(first_index) :
                   *mesh.GetElementTransformation(first_index);
         MFEM_ASSERT(!integ.GetIntegrationRule(),
                     "Mixed mesh integrators should not have an"
                     " IntegrationRule.");
         const IntegrationRule &ir = integ.GetRule(el, el, T);
         auto sub_op = new Integrator();
         sub_op->Assemble(info, trial_fes, test_fes, ir, *count[value.first], indices, Q,
                          use_bdr, use_mf);
         sub_ops.push_back(sub_op);
         CeedCompositeOperatorAddSub(oper, sub_op->GetCeedOperator());
      }

      CeedVectorCreate(internal::ceed, trial_fes.GetVDim() * trial_fes.GetNDofs(),
                       &u);
      CeedVectorCreate(internal::ceed, test_fes.GetVDim() * test_fes.GetNDofs(), &v);
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
