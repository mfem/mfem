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

#include "../../config/config.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_MPI

#include "util.hpp"

namespace mfem::future
{

/// @brief Get the prolongation operator for a field descriptor.
///
/// @param f the field descriptor.
/// @returns the prolongation operator for the field descriptor.
inline
std::shared_ptr<const Operator> get_prolongation(const FieldDescriptor &f)
{
   return std::visit([](auto&& arg) -> std::shared_ptr<const Operator>
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, const FiniteElementSpace *> ||
                    std::is_same_v<T, const ParFiniteElementSpace *>)
      {
         // Non-owning shared_ptr with no-op deleter
         return std::shared_ptr<const Operator>(arg->GetProlongationMatrix(), [](
         const Operator*) {});
      }
      else if constexpr (std::is_same_v<T, const VectorQuadratureSpace *>)
      {
         return nullptr;
      }
      else if constexpr (std::is_same_v<T, const ParameterSpace *>)
      {
         // Non-owning shared_ptr with no-op deleter
         return std::shared_ptr<const Operator>(arg->GetProlongationMatrix(), [](
         const Operator*) {});
      }
      else
      {
         static_assert(dfem::always_false<T>, "can't use GetProlongation on type");
      }
      return nullptr; // Unreachable, but avoids compiler warning
   }, f.data);
}

/// @brief Apply the prolongation operator to a field.
///
/// @param field the field descriptor.
/// @param x the input vector in tdofs.
/// @param field_l the output vector in vdofs.
/// @param is_lvector whether @a x already stores L-vector data.
inline
void prolongation(const FieldDescriptor field, const Vector &x, Vector &field_l,
                  const bool is_lvector = false)
{
   const auto P = get_prolongation(field);

   // Check if input is already L-vector sized (skip prolongation if so)
   if (is_lvector && (P == nullptr || x.Size() == P->Height()))
   {
      field_l.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
      field_l.UseDevice(x.UseDevice());
      field_l.SyncMemory(x);
      return;
   }

   // If P is nullptr or Identity, just copy
   if (P == nullptr || dynamic_cast<const IdentityOperator *>(P.get()))
   {
      if (Device::Allows(Backend::DEBUG_DEVICE))
      {
         int n = x.Size();
         field_l.SetSize(n);
         bool use_dev = false; // x.UseDevice();
         field_l.UseDevice(use_dev);
         const auto xr = x.Read(use_dev);
         auto fw = field_l.ReadWrite(use_dev);
         mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { fw[i] = xr[i]; });
      }
      else
      {
         field_l = x;
      }
      return;
   }

   // Apply prolongation
   field_l.SetSize(P->Height());
   P->Mult(x, field_l);
}

inline
void prolongation_transpose(
   const FieldDescriptor &field, const Vector &field_l, Vector &x)
{
   const auto P = get_prolongation(field);
   if (P == nullptr)
   {
      x = field_l;
   }
   else
   {
      x.SetSize(P->Width());
      P->MultTranspose(field_l, x);
   }
}

/// @brief Apply the prolongation operator to a vector of fields.
///
/// Applies each field's prolongation operator blockwise from the input
/// BlockVector into the corresponding L-vector storage.
///
/// When @a is_lvector is true, the input blocks are assumed to already be in
/// L-vector layout and are copied directly without applying prolongation.
///
/// @param fields the field descriptors.
/// @param x the input BlockVector.
/// @param x_l the output vectors in L-vector layout.
/// @param is_lvector whether @a x already stores L-vector data.
inline
void prolongation(
   const std::vector<FieldDescriptor> &fields,
   const BlockVector &x,
   std::vector<Vector *> &x_l,
   const bool is_lvector = false)
{
   MFEM_ASSERT(x.NumBlocks() == static_cast<int>(x_l.size()),
               "error " << x.NumBlocks() << " vs " << x_l.size());
   for (int i = 0; i < x.NumBlocks(); i++)
   {
      const auto P = get_prolongation(fields[i]);
      const auto alias_lvector_block = [&]
      {
         const Vector &xi = x.GetBlock(i);
         x_l[i]->NewMemoryAndSize(xi.GetMemory(), xi.Size(), false);
         x_l[i]->UseDevice(xi.UseDevice());
         x_l[i]->SyncMemory(xi);
      };

      // If nullptr, assume Identity.
      if (P == nullptr)
      {
         alias_lvector_block();
      }
      // Check if input is already L-vector sized (skip prolongation if so)
      else if (is_lvector && x.GetBlock(i).Size() == P->Height())
      {
         alias_lvector_block();
      }
      else
      {
         MFEM_ASSERT(P->Width() == x.GetBlock(i).Size(),
                     "prolongation not applicable to given input data size " <<
                     P->Width() << " vs " << x.GetBlock(i).Size());
         MFEM_ASSERT(P->Height() == x_l[i]->Size(),
                     "prolongation not applicable to given output data size " <<
                     P->Height() << " vs " << x_l[i]->Size());
         P->Mult(x.GetBlock(i), *x_l[i]);
      }
   }
}

/// @brief Apply the prolongation operator to a vector of fields.
///
/// Applies each field's prolongation operator blockwise from the input
/// MultiVector into the corresponding L-vector storage.
///
/// When @a is_lvector is true, the input blocks are assumed to already be in
/// L-vector layout and their memory is reused directly without applying
/// prolongation.
///
/// @param fields the field descriptors.
/// @param x the input MultiVector.
/// @param x_l the output vectors in L-vector layout.
/// @param is_lvector whether @a x already stores L-vector data.
inline
void prolongation(
   const std::vector<FieldDescriptor> &fields,
   const MultiVector &x,
   std::vector<Vector *> &x_l,
   const bool is_lvector = false)
{
   MFEM_ASSERT(x.NumBlocks() == static_cast<int>(x_l.size()),
               "error " << x.NumBlocks() << " vs " << x_l.size());
   for (int i = 0; i < x.NumBlocks(); i++)
   {
      if (is_lvector)
      {
         x_l[i]->NewMemoryAndSize(x[i].GetMemory(), x[i].Size(), false);
         x_l[i]->UseDevice(x[i].UseDevice());
         x_l[i]->SyncMemory(x[i]);
         continue;
      }
      const auto P = get_prolongation(fields[i]);

      // If nullptr, assume Identity.
      if (P == nullptr || dynamic_cast<const IdentityOperator*>(P.get()))
      {
         if (Device::Allows(Backend::DEBUG_DEVICE))
         {
            int n = x[i].Size();
            x_l[i]->SetSize(n);
            bool use_dev = false; // x[i].UseDevice();
            x_l[i]->UseDevice(use_dev);
            const auto xr = x[i].Read(use_dev);
            auto fw = x_l[i]->ReadWrite(use_dev);
            mfem::forall_switch(use_dev, n, [=] MFEM_HOST_DEVICE (int i) { fw[i] = xr[i]; });
         }
         else
         {
            x_l[i]->NewMemoryAndSize(x[i].GetMemory(), x[i].Size(), false);
            x_l[i]->UseDevice(x[i].UseDevice());
            x_l[i]->SyncMemory(x[i]);
         }
      }
      else
      {
         // Classify each block using the field's true vs local dof sizes so
         // mixed T/L MultiVector inputs stay consistent across MPI ranks.
         const int true_sz = GetTrueVSize(fields[i]);
         const int local_sz = GetVSize(fields[i]);
         if (x[i].Size() == local_sz && x[i].Size() != true_sz)
         {
            x_l[i]->SetSize(x[i].Size());
            *x_l[i] = x[i];
         }
         else if (x[i].Size() == true_sz)
         {
            x_l[i]->SetSize(P->Height());
            P->Mult(x[i], *x_l[i]);
         }
         else
         {
            MFEM_ABORT("prolongation: input size " << x[i].Size()
                       << " does not match T-vector size " << true_sz
                       << " or L-vector size " << local_sz);
         }
      }
   }
}

inline
void prolongation_transpose(
   const std::vector<FieldDescriptor> &fields,
   const std::vector<Vector *> &x_l,
   BlockVector &x,
   const bool is_lvector = false)
{
   MFEM_ASSERT(static_cast<int>(x_l.size()) == x.NumBlocks(),
               "error " << x_l.size() << " vs " << x.NumBlocks());
   for (size_t i = 0; i < x_l.size(); i++)
   {
      const auto P = get_prolongation(fields[i]);

      // If nullptr, assume Identity.
      if (P == nullptr || is_lvector)
      {
         x.GetBlock(i) = *x_l[i];
      }
      else
      {
         MFEM_ASSERT(P->Height() == x_l[i]->Size(),
                     "prolongation not applicable to given input data size " <<
                     P->Height() << " vs " << x_l[i]->Size());
         MFEM_ASSERT(P->Width() == x.GetBlock(i).Size(),
                     "prolongation not applicable to given output data size " <<
                     P->Width() << " vs " << x.GetBlock(i).Size());
         P->MultTranspose(*x_l[i], x.GetBlock(i));
      }
   }
}

inline
void prolongation_transpose(
   const std::vector<FieldDescriptor> &fields,
   const std::vector<Vector *> &x_l,
   MultiVector &x,
   const bool is_lvector = false)
{
   MFEM_ASSERT(static_cast<int>(x_l.size()) == x.NumBlocks(),
               "error " << x_l.size() << " vs " << x.NumBlocks());
   for (size_t i = 0; i < x_l.size(); i++)
   {
      if (is_lvector)
      {
         x[i] = *x_l[i];
         continue;
      }

      const auto P = get_prolongation(fields[i]);

      if (P == nullptr || dynamic_cast<const IdentityOperator*>(P.get()) != nullptr)
      {
         x[i] = *x_l[i];
      }
      else
      {
         const int true_sz = GetTrueVSize(fields[i]);
         const int local_sz = GetVSize(fields[i]);
         if (x[i].Size() == local_sz && x[i].Size() != true_sz)
         {
            x[i] = *x_l[i];
         }
         else
         {
            x[i].SetSize(true_sz);
            P->MultTranspose(*x_l[i], x[i]);
         }
      }
   }
}

} // namespace mfem::future

#endif // MFEM_USE_MPI
