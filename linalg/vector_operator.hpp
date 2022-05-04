// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_VECTOR_OPERATOR
#define MFEM_VECTOR_OPERATOR

#include "operator.hpp"
#include "densemat.hpp"
#include "vector.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

class ParVectorOperator : public Operator
{
private:
   MPI_Comm comm;
   int myid;

   Array<Vector*> vecs;
   Array<double> coefs;
   Array<bool> owns;

public:
   ParVectorOperator(MPI_Comm comm,
                     int myid,
                     int local_vec_size,
                     int num_vecs);

   ~ParVectorOperator();

   void SetVector(int idx, Vector *vec,
                  double c = 1.0, bool own_vec = false);

   /// Operator application: `y=A(x)`.
   void Mult(const Vector &x, Vector &y) const;

   /// Action of the transpose operator: `y=A^t(x)`.
   void MultTranspose(const Vector &x, Vector &y) const;

   /// Compute LQ factorization of this operator
   /** This operator, A, represents a matrix with very few rows but
       many columns which are distributed across multiple
       processors. The LQ factorization LQ = A is related to the QR
       factorization of the transpose of A with L = R^T and the two Q
       operators being transposes of eachother.
   */
   void GetLQFactors(DenseMatrix &L, ParVectorOperator &Q);
};

#endif // MFEM_USE_MPI

} // namespace mfem

#endif // MFEM_VECTOR_OPERATOR
