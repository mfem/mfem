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

#ifndef MFEM_MUMPS
#define MFEM_MUMPS

#include "../config/config.hpp"

#ifdef MFEM_USE_MUMPS
#ifdef MFEM_USE_MPI
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>
#include "dmumps_c.h"

namespace mfem
{

class MUMPSSolver : public mfem::Solver
{
public:
   // Constructor with MPI_Comm parameter.
   MUMPSSolver( MPI_Comm comm );

   // Constructor with HypreParMatrix Object.
   MUMPSSolver( HypreParMatrix & A);

   // Default destructor.
   ~MUMPSSolver( void );

   // Factor and solve the linear system y = Op^{-1} x.
   void Mult( const Vector & x, Vector & y ) const;

   // Set the operator.
   void SetOperator( const Operator & op );

   // Set various solver options. Refer to MUMPSSolver documentation for details.

private:
   void Init();

protected:
   MPI_Comm      comm_;
   int           numProcs_;
   int           myid_;
   const HypreParMatrix * APtr;

   hypre_CSRMatrix *csr_op;
   int n_global;
   int n_loc;
   int nnz;
   // coordinate format storage
   int * I;
   int * J;
   double * data;

   DMUMPS_STRUC_C * id;

   Vector rhs_glob;
   Array<int> recv_counts;
   Array<int> displs;
}; // mfem::MUMPSSolver class

} // mfem namespace


#endif // MFEM_USE_MPI
#endif // MFEM_USE_MUMPS
#endif // MFEM_MUMPS
