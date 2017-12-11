// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

namespace mfem
{

using namespace std;
template <class TVector>
TIterativeSolver<TVector>::TIterativeSolver()
   : TSolver<TVector>(0, true)
{
   oper = NULL;
   prec = NULL;
   max_iter = 10;
   print_level = -1;
   rel_tol = abs_tol = 0.0;
#ifdef MFEM_USE_MPI
   dot_prod_type = 0;
#endif
}

#ifdef MFEM_USE_MPI
template <class TVector>
TIterativeSolver<TVector>::TIterativeSolver(MPI_Comm _comm)
   : TSolver<TVector>(0, true)
{
   oper = NULL;
   prec = NULL;
   max_iter = 10;
   print_level = -1;
   rel_tol = abs_tol = 0.0;
   dot_prod_type = 1;
   comm = _comm;
}
#endif

template <class TVector>
void TIterativeSolver<TVector>::SetPrintLevel(int print_lvl)
{
#ifndef MFEM_USE_MPI
   print_level = print_lvl;
#else
   if (dot_prod_type == 0)
   {
      print_level = print_lvl;
   }
   else
   {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
   }
#endif
}

template <class TVector>
void TIterativeSolver<TVector>::SetPreconditioner(TSolver<TVector> &pr)
{
   prec = &pr;
   prec->iterative_mode = false;
}

template <class TVector>
void TIterativeSolver<TVector>::SetOperator(const TOperator<TVector> &op)
{
   oper = &op;
   TIterativeSolver<TVector>::height = op.Height();
   TIterativeSolver<TVector>::width = op.Width();
   if (prec)
   {
      prec->SetOperator(*oper);
   }
}

  
}
