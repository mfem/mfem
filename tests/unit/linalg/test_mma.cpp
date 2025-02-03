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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

real_t obj0(mfem::Vector& x)
{
   const int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+x[i]*x[i];
   }

#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

real_t dobj0(mfem::Vector& x, mfem::Vector& dx)
{
   const int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+x[i]*x[i];
      dx[i]=2.0*x[i];
   }
#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

real_t g0(mfem::Vector& x)
{
   int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+x[i];
   }

   int gn = n;
#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   rez = grez;
#endif

   rez=rez/gn;
   return rez-2.0;
}

real_t dg0(mfem::Vector& x, mfem::Vector& dx)
{
   const int n=x.Size();

   int gn = n;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+x[i];
      dx[i]=1.0/gn;
   }

#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   rez = grez;
#endif

   rez=rez/gn;
   return rez-1.0;
}

#ifdef MFEM_USE_MPI
TEST_CASE("MMA Test", "[Parallel], [MMA]")
{
#else
TEST_CASE("MMA Test", "[MMA]")
{
#endif
   int world_size = 1;
#ifdef MFEM_USE_MPI
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

   int num_var=12 / world_size;

   mfem::Vector x(num_var);
   mfem::Vector dx(num_var);
   mfem::Vector xmin(num_var); xmin=-1.0;
   mfem::Vector xmax(num_var); xmax=2.0;
   x=xmin; x+=0.5;

   mfem::MMAOpt* mma = nullptr;

#ifdef MFEM_USE_MPI
   mma = new mfem::MMAOpt(MPI_COMM_WORLD,num_var,1,x);
#else
   mma = new mfem::MMAOpt(num_var,1,x);
#endif

   mfem::Vector g(1); g=-1.0;
   mfem::Vector dg(num_var); dg=0.0;

   real_t o;
   for (int it=0; it<30; it++)
   {
      o=dobj0(x,dx);
      g[0]=dg0(x,dg);

      std::cout<<"it="<<it<<" o="<<o<<" g="<<g[0]<<std::endl;

      for (int i=0; i<num_var; i++)
      {
         std::cout<<" "<<x[i];
      }
      std::cout<<std::endl;
      for (int i=0; i<num_var; i++)
      {
         std::cout<<" "<<dx[i];
      }
      std::cout<<std::endl;

      mma->Update(it,dx,g,dg,xmin,xmax,x);
      std::cout<<std::endl;
   }

   for (int i=0; i<num_var; i++)
   {
      std::cout<<" "<<x[i];
   }
   std::cout<<std::endl;

   o=obj0(x);
   std::cout<<"Final o="<<o<<std::endl;

   delete mma;

   REQUIRE( std::fabs(o - 0.0005790847638021212) < 1e-12 );
}