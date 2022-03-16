// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "vector_operator.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

ParVectorOperator::ParVectorOperator(MPI_Comm comm,
                                     int myid,
                                     int local_vec_size,
                                     int num_vecs)
   : Operator((myid == 0) ? num_vecs : 0, local_vec_size),
     comm(comm),
     myid(myid),
     vecs(num_vecs),
     coefs(num_vecs),
     owns(num_vecs)
{
   vecs = NULL;
   coefs = 1.0;
   owns = false;
}

ParVectorOperator::~ParVectorOperator()
{
   for (int i=0; i < vecs.Size(); i++)
   {
      if (owns[i]) { delete vecs[i]; }
      vecs[i] = NULL;
   }
}

void ParVectorOperator::SetVector(int idx, Vector *vec,
                                  double c, bool own_vec)
{
   MFEM_VERIFY(idx >= 0 && idx < vecs.Size(),
               "ParVectorOperator: Index out of range");

   vecs[idx] = vec;
   coefs[idx] = c;
   owns[idx] = own_vec;
}

void ParVectorOperator::Mult(const Vector &x, Vector &y) const
{
   for (int i=0; i<vecs.Size(); i++)
   {
      double vo = coefs[i] * (*vecs[i] * x);
      double vi = 0.0;
      MPI_Reduce(&vo, &vi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
      if (myid == 0) { y[i] = vi; }
   }
}

/// Action of the transpose operator: `y=A^t(x)`.
void ParVectorOperator::MultTranspose(const Vector &x, Vector &y) const
{
   y = 0.0;
   for (int i=0; i<vecs.Size(); i++)
   {
      double xi = (myid == 0) ? x[i] : 0.0;
      MPI_Bcast(&xi, 1, MPI_DOUBLE, 0, comm);
      y.Add(xi * coefs[i], *vecs[i]);
   }
}

}

#endif // MFEM_USE_MPI
