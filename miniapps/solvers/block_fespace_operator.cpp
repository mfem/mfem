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

#include "block_fespace_operator.hpp"

namespace mfem
{

BlockFESpaceOperator::BlockFESpaceOperator(const FESVector &fespaces):
   Operator(GetHeight(fespaces)),
   offsets(GetBlockOffsets(fespaces)),
   prolongColOffsets(GetProColBlockOffsets(fespaces)),
   restrictRowOffsets(GetResRowBlockOffsets(fespaces)),
   A(offsets),
   prolongation(offsets, prolongColOffsets),
   restriction(restrictRowOffsets, offsets)
{
   for (size_t i = 0; i <fespaces.size(); i++)
   {
      // Since const_cast is required here, be sure to avoid using
      // BlockOperator::GetBlock on restriction or prolongation.
      auto prolongation_matrix = fespaces[i]->GetProlongationMatrix();
      auto restriction_matrix = fespaces[i]->GetRestrictionOperator();
      prolongation.SetDiagonalBlock(i, const_cast<Operator *>(prolongation_matrix));
      restriction.SetDiagonalBlock(i, const_cast<Operator *>(restriction_matrix));
   }
}

int BlockFESpaceOperator::GetHeight(const FESVector &fespaces)
{
   int height = 0;
   for (size_t i = 0; i < fespaces.size(); i++)
   {
      height += fespaces[i]->GetVSize();
   }
   return height;
}

Array<int> BlockFESpaceOperator::GetBlockOffsets(const FESVector &fespaces)
{
   Array<int> offsets(fespaces.size()+1);
   offsets[0] = 0;
   for (size_t i = 1; i <=fespaces.size(); i++)
   {
      offsets[i] = fespaces[i-1]->GetVSize();
   }
   offsets.PartialSum();
   offsets.Print();
   return offsets;
}

Array<int> BlockFESpaceOperator::GetProColBlockOffsets(const FESVector
                                                       &fespaces)
{
   Array<int> offsets(fespaces.size()+1);
   offsets[0] = 0;
   for (size_t i = 1; i <=fespaces.size(); i++)
   {
      const auto *prolong = fespaces[i-1]->GetProlongationMatrix();
      if (prolong)
      {
         offsets[i] = prolong->Width();
      }
      else
      {
         offsets[i] = fespaces[i-1]->GetVSize();
      }
      offsets[i] = fespaces[i-1]->GetTrueVSize();
   }
   offsets.PartialSum();
   offsets.Print();
   return offsets;
}

Array<int> BlockFESpaceOperator::GetResRowBlockOffsets(const FESVector
                                                       &fespaces)
{
   Array<int> offsets(fespaces.size()+1);
   std::cout << "fespaces.size() = " << fespaces.size() << std::endl;
   offsets[0] = 0;
   for (size_t i = 1; i <=fespaces.size(); i++)
   {
      const auto *restriction = fespaces[i-1]->GetRestrictionOperator();
      if (restriction)
      {
         offsets[i] = restriction->Height();
      }
      else
      {
         offsets[i] = fespaces[i-1]->GetVSize();
      }
      offsets[i] = fespaces[i-1]->GetTrueVSize();
   }
   offsets.PartialSum();
   offsets.Print();
   return offsets;
}

const Operator* BlockFESpaceOperator::GetProlongation() const
{
   return &prolongation;
}

const Operator* BlockFESpaceOperator::GetRestriction() const
{
   return &restriction;
}

} // namespace mfem
