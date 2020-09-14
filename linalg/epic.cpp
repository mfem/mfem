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

#include "epic.hpp"

#ifdef MFEM_USE_EPIC

namespace mfem
{
EPICSolver::EPICSolver()
{
   // Allocate an empty serial N_Vector
   temp = N_VNewEmpty_Serial(0);
   m[0] = 10;
   m[1] = 10;
   MFEM_VERIFY(temp, "error in N_VNewEmpty_Serial()");
}

#ifdef MFEM_USE_MPI
EPICSolver::EPICSolver(MPI_Comm comm)
{
   // Allocate an empty vector
   if (comm == MPI_COMM_NULL)
   {
      // Allocate an empty serial N_Vector
      temp = N_VNewEmpty_Serial(0);
      MFEM_VERIFY(temp, "error in N_VNewEmpty_Serial()");
   }
   else
   {
      // Allocate an empty parallel N_Vector
      temp = N_VNewEmpty_Parallel(comm, 0, 0);  // calls MPI_Allreduce()
      MFEM_VERIFY(temp, "error in N_VNewEmpty_Parallel()");
   }
}
#endif

int EPICSolver::RHS(realtype t, const N_Vector y, N_Vector ydot, void *user_data)
{
   // Get data from N_Vectors
   const Vector mfem_y(y);
   Vector mfem_ydot(ydot);
   EPICSolver *self = static_cast<EPICSolver*>(user_data);

   // Compute y' = f(t, y)
   self->f->SetTime(t);
   self->f->Mult(mfem_y, mfem_ydot);

   // Return success
   return (0);
}

void EPICSolver::Init(TimeDependentOperator &f)
{
    ODESolver::Init(f);

    long local_size = f.Height();
    long global_size = 0;
#ifdef MFEM_USE_MPI
    if (Parallel())
    {
        MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                      NV_COMM_P(temp));
    }
#endif

    Vector mfem_temp(local_size);
    mfem_temp.ToNVector(temp, global_size);
    integrator = new Epi2_KIOPS(EPICSolver::RHS, this, 100, temp ,local_size);
}

void EPICSolver::Step(Vector &x, double &t, double &dt)
{
   if (!Parallel())
   {
      NV_DATA_S(temp) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(temp) == x.Size(), "");
   }
   else
   {
#ifdef MFEM_USE_MPI
      NV_DATA_P(temp) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(temp) == x.Size(), "");
#endif
   }
   integrator->Integrate(dt, t, t+dt, 0, temp, 1e-10, m);
   t += dt;
}

EPICSolver::~EPICSolver()
{
    delete integrator;
}

}

#endif
