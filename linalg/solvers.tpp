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

template <class TVector>
void TSLISolver<TVector>::UpdateVectors()
{
   r.SetSize(this->width);
   z.SetSize(this->width);
}

template <class TVector>
void TSLISolver<TVector>::Mult(const TVector &b, TVector &x) const
{
   int i;

   // Optimized preconditioned SLI with fixed number of iterations and given
   // initial guess
   if (!this->rel_tol && this->iterative_mode && this->prec)
   {
      for (i = 0; i < this->max_iter; i++)
      {
         this->oper->Mult(x, r);  // r = A x
         subtract(b, r, r); // r = b - A x
         this->prec->Mult(r, z);  // z = B r
         add(x, 1.0, z, x); // x = x + B (b - A x)
      }
      this->converged = 1;
      this->final_iter = i;
      return;
   }

   // Optimized preconditioned SLI with fixed number of iterations and zero
   // initial guess
   if (!this->rel_tol && !this->iterative_mode && this->prec)
   {
      this->prec->Mult(b, x);     // x = B b (initial guess 0)
      for (i = 1; i < this->max_iter; i++)
      {
         this->oper->Mult(x, r);  // r = A x
         subtract(b, r, r); // r = b - A x
         this->prec->Mult(r, z);  // z = B r
         add(x, 1.0, z, x); // x = x + B (b - A x)
      }
      this->converged = 1;
      this->final_iter = i;
      return;
   }

   // General version of SLI with a relative tolerance, optional preconditioner
   // and optional initial guess
   double r0, nom, nom0, nomold = 1, cf;

   if (this->iterative_mode)
   {
      this->oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      r = b;
      x = 0.0;
   }

   if (this->prec)
   {
      this->prec->Mult(r, z); // z = B r
      nom0 = nom = this->Dot(z, r);
   }
   else
   {
      nom0 = nom = this->Dot(r, r);
   }

   if (this->print_level == 1)
      mfem::out << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
                << nom << '\n';

   r0 = std::max(nom*this->rel_tol*this->rel_tol, this->abs_tol*this->abs_tol);
   if (nom <= r0)
   {
      this->converged = 1;
      this->final_iter = 0;
      this->final_norm = sqrt(nom);
      return;
   }

   // start iteration
   this->converged = 0;
   this->final_iter = this->max_iter;
   for (i = 1; true; )
   {
      if (this->prec) //  x = x + B (b - A x)
      {
         add(x, 1.0, z, x);
      }
      else
      {
         add(x, 1.0, r, x);
      }

      this->oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x

      if (this->prec)
      {
         this->prec->Mult(r, z); //  z = B r
         nom = this->Dot(z, r);
      }
      else
      {
         nom = this->Dot(r, r);
      }

      cf = sqrt(nom/nomold);
      if (this->print_level == 1)
         mfem::out << "   Iteration : " << setw(3) << i << "  (B r, r) = "
                   << nom << "\tConv. rate: " << cf << '\n';
      nomold = nom;

      if (nom < r0)
      {
         if (this->print_level == 2)
            mfem::out << "Number of SLI iterations: " << i << '\n'
                      << "Conv. rate: " << cf << '\n';
         else if (this->print_level == 3)
            mfem::out << "(B r_0, r_0) = " << nom0 << '\n'
                      << "(B r_N, r_N) = " << nom << '\n'
                      << "Number of SLI iterations: " << i << '\n';
         this->converged = 1;
         this->final_iter = i;
         break;
      }

      if (++i > this->max_iter)
      {
         break;
      }
   }

   if (this->print_level >= 0 && !this->converged)
   {
      mfem::err << "SLI: No convergence!" << '\n';
      mfem::out << "(B r_0, r_0) = " << nom0 << '\n'
                << "(B r_N, r_N) = " << nom << '\n'
                << "Number of SLI iterations: " << this->final_iter << '\n';
   }
   if (this->print_level >= 1 || (this->print_level >= 0 && !this->converged))
   {
      mfem::out << "Average reduction factor = "
                << pow (nom/nom0, 0.5/this->final_iter) << '\n';
   }
   this->final_norm = sqrt(nom);
}

template <class TVector>
void TSLI(const TOperator<TVector> &A,
          const TVector &b, TVector &x,
          int print_iter, int max_num_iter,
          double RTOLERANCE, double ATOLERANCE)
{
   TSLISolver<TVector> sli;
   sli.SetPrintLevel(print_iter);
   sli.SetMaxIter(max_num_iter);
   sli.SetRelTol(sqrt(RTOLERANCE));
   sli.SetAbsTol(sqrt(ATOLERANCE));
   sli.SetOperator(A);
   sli.Mult(b, x);
}

template <class TVector>
void TSLI(const TOperator<TVector> &A, TSolver<TVector> &B,
          const TVector &b, TVector &x,
          int print_iter, int max_num_iter,
          double RTOLERANCE, double ATOLERANCE)
{
   TSLISolver<TVector> sli;
   sli.SetPrintLevel(print_iter);
   sli.SetMaxIter(max_num_iter);
   sli.SetRelTol(sqrt(RTOLERANCE));
   sli.SetAbsTol(sqrt(ATOLERANCE));
   sli.SetOperator(A);
   sli.SetPreconditioner(B);
   sli.Mult(b, x);
}

}
