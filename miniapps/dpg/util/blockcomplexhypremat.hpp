// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BLOCKCOMPLEX
#define MFEM_BLOCKCOMPLEX

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

class ParBlockComplexSystem
{

protected:

   /// ess_tdof list for each space
   Array<Array<int> *> ess_tdofs;
   Array<int> toffsets;
   int nblocks;

   /** split ess_tdof_list given in global tdof (for all spaces)
    to individual lists for each space */
   void FillEssTdofLists(const Array<int> & ess_tdof_list);

   // Block Prolongation
   BlockOperator * P = nullptr;
   // Block Restriction
   BlockMatrix * R = nullptr;

   ComplexOperator * op = nullptr;
   BlockOperator * op_r = nullptr;
   BlockOperator * op_i = nullptr;
   BlockOperator * op_e_r = nullptr;
   BlockOperator * op_e_i = nullptr;

public:

   ParBlockComplexSystem() {}

   /// Creates bilinear form associated with FE spaces @a trial_pfes_.
   ParBlockComplexSystem(ComplexOperator * op_)
      : op(op_)
   {
      op_r = dynamic_cast<BlockOperator *>(&op->real());
      op_i = dynamic_cast<BlockOperator *>(&op->imag());
      toffsets = op_r->RowOffsets(); // (assumes square blockoperator)
      nblocks = toffsets.Size() - 1;
      ess_tdofs.SetSize(nblocks);
   }

   ComplexOperator * EliminateBC(const Array<int> ess_tdof_list, Vector &X,
                                 Vector & B);

   virtual ~ParBlockComplexSystem() {}


};

} // namespace mfem


#endif // MFEM_USE_MPI

#endif
