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

#include "util.hpp"
#include "tensor.hpp"

namespace mfem
{

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct Basis;

template <int Dim>
struct Basis<Dim,true,0,0>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = true;
   static constexpr int MaxSize1D = 16;
   const int dofs1D;
   const int quads1D;
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      SharedTensor<2,double,pow(MaxSize1D,2)> s_B(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      SharedTensor<2,double,pow(MaxSize1D,2)> s_Bt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      SharedTensor<2,double,pow(MaxSize1D,2)> s_G(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      SharedTensor<2,double,pow(MaxSize1D,2)> s_Gt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Gt(d,q) = Gt[d+dofs1D*q];
         }
      }
      return s_Gt;
   }
};

template <int Dim, int Dofs1D, int Quads1D>
struct Basis<Dim,true,Dofs1D,Quads1D>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = true;
   static constexpr int dofs1D = Dofs1D;
   static constexpr int quads1D = Quads1D;
   static constexpr int dofs = pow(Dofs1D,Dim);
   static constexpr int quads = pow(Quads1D,Dim);
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      StaticSharedTensor<double,quads1D,dofs1D> s_B(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_B(q,d) = B[q+quads1D*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      StaticSharedTensor<double,dofs1D,quads1D> s_Bt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Bt(d,q) = Bt[d+dofs1D*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      StaticSharedTensor<double,quads1D,dofs1D> s_G(quads1D,dofs1D);
      MFEM_FOREACH_THREAD(d,y,dofs1D)
      {
         MFEM_FOREACH_THREAD(q,x,quads1D)
         {
            s_G(q,d) = G[q+quads1D*d];
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      StaticSharedTensor<double,dofs1D,quads1D> s_Gt(dofs1D,quads1D);
      MFEM_FOREACH_THREAD(q,y,quads1D)
      {
         MFEM_FOREACH_THREAD(d,x,dofs1D)
         {
            s_Gt(d,q) = Gt[d+dofs1D*q];
         }
      }
      return s_Gt;
   }
};

template <int Dim>
struct Basis<Dim,false,0,0>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = false;
   static constexpr int MaxSize = pow(16,3);
   const int dofs;
   const int quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      SharedTensor<2,double,MaxSize> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      SharedTensor<2,double,MaxSize> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      SharedTensor<3,double,MaxSize> s_G(quads,dofs,dim);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_G(q,d,i) = G[q+quads*d+dofs*quads*i];
            }
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      SharedTensor<3,double,MaxSize> s_Gt(dofs,quads,dim);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_Gt(d,q,i) = Gt[d+dofs*q+dofs*quads*i];
            }
         }
      }
      return s_Gt;
   }
};

template <int Dim, int Dofs, int Quads>
struct Basis<Dim,false,Dofs,Quads>
{
   static constexpr int dim = Dim;
   static constexpr bool isTensor = false;
   static constexpr int dofs = Dofs;
   static constexpr int quads = Quads;
   const double *B;
   const double *Bt;
   const double *G;
   const double *Gt;

   auto GetB()
   {
      StaticSharedTensor<double,quads,dofs> s_B(quads,dofs);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            s_B(q,d) = B[q+quads*d];
         }
      }
      return s_B;
   }

   auto GetBt()
   {
      StaticSharedTensor<double,dofs,quads> s_Bt(dofs,quads);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            s_Bt(d,q) = Bt[d+dofs*q];
         }
      }
      return s_Bt;
   }

   auto GetG()
   {
      StaticSharedTensor<double,quads,dofs,dim> s_G(quads,dofs,dim);
      MFEM_FOREACH_THREAD(d,y,dofs)
      {
         MFEM_FOREACH_THREAD(q,x,quads)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_G(q,d,i) = G[q+quads*d+quads*dofs*i];
            }
         }
      }
      return s_G;
   }

   auto GetGt()
   {
      StaticSharedTensor<double,dofs,quads,dim> s_Gt(dofs,quads,dim);
      MFEM_FOREACH_THREAD(q,y,quads)
      {
         MFEM_FOREACH_THREAD(d,x,dofs)
         {
            for (size_t i = 0; i < dim; i++)
            {
               s_Gt(d,q,i) = Gt[d+dofs*q+quads*dofs*i];
            }
         }
      }
      return s_Gt;
   }
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisTranspose
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradient
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
struct BasisGradientTranspose
{
   Basis<Dim,IsTensor,Dofs,Quads> &basis;
};

template <int Dim, bool IsTensor, int Dofs, int Quads>
auto transpose(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisTranspose<Dim,IsTensor,Dofs,Quads>{basis};
}

template <int Dim, bool IsTensor, int Dofs, int Quads>
auto transpose(BasisGradient<Dim,IsTensor,Dofs,Quads> &G)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{G.basis};
}

template <int Dim, bool IsTensor, int Dofs, int Quads>
auto grad(Basis<Dim,IsTensor,Dofs,Quads> &basis)
{
   return BasisGradient<Dim,IsTensor,Dofs,Quads>{basis};
}

template <int Dim, bool IsTensor, int Dofs, int Quads>
auto grad(BasisTranspose<Dim,IsTensor,Dofs,Quads> &Bt)
{
   return BasisGradientTranspose<Dim,IsTensor,Dofs,Quads>{Bt.basis};
}

} // mfem namespace

#endif // MFEM_BASIS
