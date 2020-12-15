// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_CONFIG
#define MFEM_TENSOR_CONFIG

namespace mfem
{

template <int Dim, bool IsTensor=false, int TDofs=0, int TQuads=0, int BatchSize=1>
struct KernelConfig
{
   static constexpr int dim = Dim;
   static constexpr bool is_tensor = IsTensor;
   static constexpr int Dofs = TDofs; // IsTensor? pow(Dofs,Dim) : Dofs; ?
   static constexpr int Quads = TQuads;
   static constexpr int batch_size = BatchSize;
   const int dofs;
   const int quads;

   KernelConfig(int dofs, int quads): dofs(dofs), quads(quads)
   {
      // TODO check that if Dofs!= 0 then dofs==Dofs
      // TODO check that if Quads!= 0 then quads==Quads
   }
};

template <int Dim, bool IsTensor=false, int Dofs=0, int Quads=0, int BatchSize=1>
auto MakeConfig(int dofs, int quads)
{
   return KernelConfig<Dim,IsTensor,Dofs,Quads,BatchSize>(dofs,quads);
}

} // mfem namespace

#endif // MFEM_TENSOR_CONFIG
