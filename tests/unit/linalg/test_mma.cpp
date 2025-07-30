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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

real_t obj0(Vector& x)
{
   const int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+x[i]*x[i];
   }

#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

real_t dobj0(Vector& x, Vector& dx)
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
   MPI_Allreduce(&rez, &grez, 1,MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

real_t g0(Vector& x)
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
   MPI_Allreduce(&rez, &grez, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   rez=rez/gn;
   return rez-2.0;
}

real_t dg0(Vector& x, Vector& dx)
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
   MPI_Allreduce(&rez, &grez, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   rez=rez/gn;
   return rez-2.0;
}


/** \brief Constrained Unit test
 *
 *    minimize   F(x) = \sum[ x*x ],
 *    subject to \sum[ x ]/ m - 2 <= 0  for m design variables
 *
 * */
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

   Vector x(num_var);
   Vector dx(num_var);
   Vector xmin(num_var); xmin=-1.0;
   Vector xmax(num_var); xmax=2.0;
   x=xmin; x+=0.5;

   MMA* mma = nullptr;

#ifdef MFEM_USE_MPI
   mma = new MMA(MPI_COMM_WORLD,num_var,1,x);
#else
   mma = new MMA(num_var,1,x);
#endif

   Vector g(1); g=-1.0;
   Vector dg(num_var); dg=0.0;

   real_t o;
   for (int it=0; it<30; it++)
   {
      o=dobj0(x,dx);
      g[0]=dg0(x,dg);

      mma->Update(dx,g,dg,xmin,xmax,x);
   }

   o=obj0(x);

   delete mma;

   REQUIRE( std::fabs(o - 0.00233310583131376) < 1e-12 );
}

real_t obj0_c(Vector& x)
{
   const int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+1.0/x[i]+10.0*x[i];
   }

#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

real_t dobj0_c(Vector& x, Vector& dx)
{
   const int n=x.Size();
   real_t rez=0.0;
   for (int i=0; i<n; i++)
   {
      rez=rez+1.0/x[i]+10.0*x[i];
      dx[i]= -1.0/(x[i]*x[i]) + 10.0;
   }
#ifdef MFEM_USE_MPI
   real_t grez;
   MPI_Allreduce(&rez, &grez, 1,MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   rez = grez;
#endif

   return rez;
}

/** \brief Unconstrained Unit test
 *
 *    minimize   F(x) = \sum[ (1 / x) + 10x ],
 *
 * */
#ifdef MFEM_USE_MPI
TEST_CASE("MMA Unconstrained Test", "[Parallel], [MMA_0CONSTR]")
{
#else
TEST_CASE("MMA Unconstrained Test", "[MMA_0CONSTR]")
{
#endif
   int world_size = 1;
#ifdef MFEM_USE_MPI
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

   const int num_var=12 / world_size;

   Vector x(num_var);
   Vector dx(num_var);
   Vector xmin(num_var); xmin=0.0;
   Vector xmax(num_var); xmax=2.0;
   x=xmin; x+=1.5;

   MMA* mma = nullptr;

#ifdef MFEM_USE_MPI
   mma = new MMA(MPI_COMM_WORLD,num_var,0,x);
#else
   mma = new MMA(num_var,0,x);
#endif

   real_t o;
   for (int it=0; it<30; it++)
   {
      o=dobj0_c(x,dx);

      mma->Update(dx,xmin,xmax,x);
   }

   o=obj0_c(x);

   delete mma;

   REQUIRE( std::fabs(o - 75.977534018859) < 1e-12 );
}
