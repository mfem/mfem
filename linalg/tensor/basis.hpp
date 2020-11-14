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

#ifndef MFEM_BASIS
#define MFEM_BASIS

namespace mfem
{

template <int Quads, int Dofs>//Dim?
struct FixedTensorBasis
{
   static const int D = Dofs;
   static const int Q = Quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   MFEM_HOST_DEVICE inline
   const double& operator()(int i, int j) const
   {
      return B[i+Q*j];
   }
};

template <typename Basis>
struct Transpose
{
   Basis &basis;
   MFEM_HOST_DEVICE inline
   const double& operator()(int i, int j) const
   {
      return basis.Bt[i+basis.D*j];
   }
};

template <typename Basis>
struct Gradient;

template <int Quads, int Dofs>
struct Gradient<FixedTensorBasis<Quads,Dofs>>
{
   FixedTensorBasis<Quads,Dofs> &basis;

   MFEM_HOST_DEVICE inline
   const double& operator()(int i, int j, int k) const
   {
      return basis.G[i+basis.Q*j];
   }
};



template <typename Basis>
inline Gradient<Basis> Grad(Basis &basis)
{
   return Gradient<Basis>{basis};
}

} // mfem namespace

#endif // MFEM_BASIS
