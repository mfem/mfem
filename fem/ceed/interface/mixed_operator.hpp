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

#ifndef MFEM_LIBCEED_MIXED_OPERATOR
#define MFEM_LIBCEED_MIXED_OPERATOR

#include "../../fespace.hpp"
#include "operator.hpp"
#include "util.hpp"
#include "ceed.hpp"
#include <array>
#include <unordered_map>
#include <vector>
#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

namespace mfem
{

namespace ceed
{

/** @brief This class wraps one or more `OpType` objects to support finite
    element spaces on mixed meshes. */
template <typename OpType>
class MixedOperator : public Operator
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
#ifndef MFEM_USE_OPENMP
   std::vector<OpType *> sub_ops;
#else
   std::vector<std::vector<OpType *>> sub_ops;
#endif

public:
   template <typename IntegratorType, typename CeedOperatorInfo, typename CoeffType>
   void Assemble(const IntegratorType &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 CoeffType *Q,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(integ, info, fes, fes, Q, (mfem::Coefficient *)nullptr, use_bdr,
               use_mf);
   }

   template <typename IntegratorType, typename CeedOperatorInfo,
             typename CoeffType1, typename CoeffType2>
   void Assemble(const IntegratorType &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &fes,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      Assemble(integ, info, fes, fes, Q1, Q2, use_bdr, use_mf);
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
      Assemble(integ, info, trial_fes, test_fes, Q, (mfem::Coefficient *)nullptr,
               use_bdr, use_mf);
   }

   template <typename IntegratorType, typename CeedOperatorInfo,
             typename CoeffType1, typename CoeffType2>
   void Assemble(const IntegratorType &integ,
                 CeedOperatorInfo &info,
                 const mfem::FiniteElementSpace &trial_fes,
                 const mfem::FiniteElementSpace &test_fes,
                 CoeffType1 *Q1,
                 CoeffType2 *Q2,
                 const bool use_bdr = false,
                 const bool use_mf = false)
   {
      MFEM_VERIFY(trial_fes.GetMesh() == test_fes.GetMesh(),
                  "Trial and test basis must correspond to the same Mesh.");
      mfem::Mesh &mesh = *trial_fes.GetMesh();
#ifndef MFEM_USE_OPENMP
      const bool mixed =
         mesh.GetNumGeometries(mesh.Dimension() - use_bdr) > 1 ||
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
         sub_ops.push_back(new OpType);
         auto &sub_op = *sub_ops.back();
         sub_op.Assemble(internal::ceed, info, trial_fes, test_fes, ir,
                         Q1, Q2, use_bdr, use_mf);
         CeedOperatorReferenceCopy(sub_op.GetOperator(), &oper);
         if (sub_op.GetTransposeOperator())
         {
            CeedOperatorReferenceCopy(sub_op.GetTransposeOperator(), &oper_t);
         }
         CeedVectorCreate(internal::ceed,
                          trial_fes.GetVDim() * trial_fes.GetNDofs(), &u);
         CeedVectorCreate(internal::ceed,
                          test_fes.GetVDim() * test_fes.GetNDofs(), &v);
         return;
      }
#endif

#ifdef MFEM_USE_OPENMP
      #pragma omp parallel
#endif
      {
         const int ne = use_bdr ? mesh.GetNBE() : mesh.GetNE();
#ifdef MFEM_USE_OPENMP
         const int nt = omp_get_num_threads();
         #pragma omp master
         {
            thread_ops.resize(nt, nullptr);
            thread_ops_t.resize(nt, nullptr);
            thread_u.resize(nt, nullptr);
            thread_v.resize(nt, nullptr);
            sub_ops.resize(nt);
         }
         const int tid = omp_get_thread_num();
         const int stride = (ne + nt - 1) / nt;
         const int start = tid * stride;
         const int stop = std::min(start + stride, ne);
#else
         const int start = 0;
         const int stop = ne;
#endif

         // Count the number of elements of each type
         std::unordered_map<ElementKey, int, ElementHash> counts, offsets;
         std::unordered_map<ElementKey, std::vector<int>, ElementHash> element_indices;
         for (int i = start; i < stop; i++)
         {
            const mfem::FiniteElement &trial_fe = use_bdr ? *trial_fes.GetBE(i) :
                                                  *trial_fes.GetFE(i);
            const mfem::FiniteElement &test_fe = use_bdr ? *test_fes.GetBE(i) :
                                                 *test_fes.GetFE(i);
            mfem::Element::Type type = use_bdr ? mesh.GetBdrElementType(i) :
                                       mesh.GetElementType(i);
            ElementKey key = {type, trial_fe.GetOrder(), test_fe.GetOrder()};
            auto value = counts.find(key);
            if (value == counts.end())
            {
               counts[key] = 1;
            }
            else
            {
               value->second++;
            }
         }

         // Initialization of the arrays
         for (const auto &value : counts)
         {
            offsets[value.first] = 0;
            element_indices[value.first] = std::vector<int>(value.second);
         }

         // Populates the indices arrays for each element type
         for (int i = start; i < stop; i++)
         {
            const mfem::FiniteElement &trial_fe = use_bdr ? *trial_fes.GetBE(i) :
                                                  *trial_fes.GetFE(i);
            const mfem::FiniteElement &test_fe = use_bdr ? *test_fes.GetBE(i) :
                                                 *test_fes.GetFE(i);
            mfem::Element::Type type = use_bdr ? mesh.GetBdrElementType(i) :
                                       mesh.GetElementType(i);
            ElementKey key = {type, trial_fe.GetOrder(), test_fe.GetOrder()};
            int &offset = offsets[key];
            std::vector<int> &indices = element_indices[key];
            indices[offset++] = i;
         }

         // Create each sub-CeedOperator, some threads may be empty
#ifdef MFEM_USE_OPENMP
         #pragma omp barrier
         CeedOperator &loc_oper = thread_ops[tid];
         CeedOperator &loc_oper_t = thread_ops_t[tid];
         CeedVector &loc_u = thread_u[tid];
         CeedVector &loc_v = thread_v[tid];
         std::vector<OpType *> &loc_sub_ops = sub_ops[tid];
#else
         CeedOperator &loc_oper = oper;
         CeedOperator &loc_oper_t = oper_t;
         CeedVector &loc_u = u;
         CeedVector &loc_v = v;
         std::vector<OpType *> &loc_sub_ops = sub_ops;
#endif
         if (element_indices.size() > 0)
         {
            loc_sub_ops.reserve(element_indices.size());
            CeedCompositeOperatorCreate(internal::ceed, &loc_oper);
            IsoparametricTransformation T;  // Thread-safe
            for (const auto &value : element_indices)
            {
               const std::vector<int> &indices = value.second;
               const int first_index = indices[0];
               const mfem::FiniteElement &trial_fe =
                  use_bdr ? *trial_fes.GetBE(first_index) : *trial_fes.GetFE(first_index);
               const mfem::FiniteElement &test_fe =
                  use_bdr ? *test_fes.GetBE(first_index) : *test_fes.GetFE(first_index);
               if (use_bdr)
               {
                  mesh.GetBdrElementTransformation(first_index, &T);
               }
               else
               {
                  mesh.GetElementTransformation(first_index, &T);
               }
               MFEM_VERIFY(!integ.GetIntegrationRule(),
                           "Mixed mesh integrators should not have an IntegrationRule.");
               const IntegrationRule &ir = integ.GetRule(trial_fe, test_fe, T);
               loc_sub_ops.push_back(new OpType);
               auto &sub_op = *loc_sub_ops.back();
               sub_op.Assemble(internal::ceed, info, trial_fes, test_fes, ir,
                               static_cast<int>(indices.size()), indices.data(),
                               Q1, Q2, use_bdr, use_mf);
               CeedCompositeOperatorAddSub(loc_oper, sub_op.GetOperator());
               if (sub_op.GetTransposeOperator())
               {
                  if (!loc_oper_t) { CeedCompositeOperatorCreate(internal::ceed, &loc_oper_t); }
                  CeedCompositeOperatorAddSub(loc_oper_t, sub_op.GetTransposeOperator());
               }
            }
            CeedOperatorCheckReady(loc_oper);
            if (loc_oper_t) { CeedOperatorCheckReady(loc_oper_t); }
            CeedVectorCreate(internal::ceed,
                             trial_fes.GetVDim() * trial_fes.GetNDofs(), &loc_u);
            CeedVectorCreate(internal::ceed,
                             test_fes.GetVDim() * test_fes.GetNDofs(), &loc_v);
         }
      }
   }

   virtual ~MixedOperator()
   {
#ifndef MFEM_USE_OPENMP
      for (auto *sub_op : sub_ops)
      {
         delete sub_op;
      }
#else
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         for (auto *sub_op : sub_ops[tid])
         {
            delete sub_op;
         }
      }
#endif
   }
#endif
};

} // namespace ceed

} // namespace mfem

#endif // MFEM_LIBCEED_MIXED_OPERATOR
