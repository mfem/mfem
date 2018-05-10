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
#include "../raja.hpp"

namespace mfem {
  
  // ***************************************************************************
  static __device__ double Dot(const RajaVector &x,
                               const RajaVector &y,
                               const int dot_prod_type=1) {
#ifndef MFEM_USE_MPI
      return (x * y);
#else
      //if (dot_prod_type == 0) return (x * y);
      
      //const double local_dot = (x * y);
      const double global_dot = 0.0;
      //MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return global_dot;
#endif
    }
  // ***************************************************************************
  static __global__ void cuCG(const RajaVector &b, RajaVector &x,
                              const int N,
                              const RajaOperator *oper,
                              RajaSolverOperator *prec,
                              const bool iterative_mode,
                              RajaVector &r,
                              RajaVector &d,
                              RajaVector &z,
                              double &final_norm,
                              const double rel_tol = 0.0,
                              const double abs_tol = 0.0,
                              const int max_iter = 1024,
                              const int print_level =0) {
    int i,converged,final_iter;
    double r0, den, nom, nom0, betanom;//, alpha, beta;
    if (iterative_mode) {
      //oper->Mult(x, r);
      //r=b-r;//subtract(b, r, r); // r = b - A x
    }
    else
    {
      //r = b;
      //x = 0.0;
      //d_vector_op_eq(N,0.0,x);
    }

    if (prec)
    {
      //prec->Mult(r, z); // z = B r
      //d = z;
    }
    else
    {
      //d = r;
    }

    nom0 = nom = Dot(d, r);
    //MFEM_ASSERT(IsFinite(nom), "nom = " << nom);
      
    if (print_level == 1 || print_level == 3) {
      printf("   Iteration : %d (B r, r) = %f", 0, nom);
    }
      
    r0 = max(nom*rel_tol*rel_tol,abs_tol*abs_tol);

    if (nom <= r0)
    {
      converged = 1;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
    }

    //oper->Mult(d, z);  // z = A d

    den = Dot(z, d);
    MFEM_ASSERT(IsFinite(den), "den = " << den);

    if (print_level >= 0 && den < 0.0) {
      printf("Negative denominator in step 0 of PCG: %f\n", den);
    }

    if (den == 0.0)
    {
      converged = 0;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
    }

    // start iteration
    converged = 0;
    final_iter = max_iter;
    for (i = 1; true; ){
      //alpha = nom/den;
      //add(x,  alpha, d, x);     //  x = x + alpha d
      //add(r, -alpha, z, r);     //  r = r - alpha A d
      if (prec)
      {
        //prec->Mult(r, z);      //  z = B r
        betanom = Dot(r, z);
      }
      else
      {
        betanom = Dot(r, r);
      }        
        
      if (print_level == 1){
        printf("   Iteration : %d  (B r, r) = %f\n",i,betanom);
      }

      if (betanom < r0)
      {
        if (print_level == 2)
        {
          printf("Number of PCG iterations: %d\n",i);
        }
        else if (print_level == 3)
        {
          printf("   Iteration : %d (B r, r) = %f\n",i,betanom);
        }
        converged = 1;
        final_iter = i;
        break;
      }

      if (++i > max_iter)
      {
        break;
      }

      //beta = betanom/nom;
      if (prec)
      {
        //add(z, beta, d, d);   //  d = z + beta d
      }
      else
      {
        //add(r, beta, d, d);
      }
        
      //oper->Mult(d, z);       //  z = A d

      den = Dot(d, z);
        
      //assert(IsFinite(den))); 
      if (den <= 0.0)
      {
        if (print_level >= 0 && Dot(d, d) > 0.0)
          printf("PCG: The operator is not positive definite. (Ad, d) = %f\n",den);
      }
      nom = betanom;
    }
     
    if (print_level >= 0 && !converged)
    {
      if (print_level != 1)
      {
        if (print_level != 3)
        {
          printf("   Iteration : 0 (B r, r) = %f\n",nom0);
        }
        printf("   Iteration : %d (B r, r) = %f\n",final_iter, betanom);
      }
      printf("PCG: No convergence!\n");
    }
      
    if (print_level >= 1 || (print_level >= 0 && !converged)) {
      printf("Average reduction factor = %f\n",
             pow (betanom/nom0, 0.5/final_iter));
    }
    final_norm = sqrt(betanom);
  }

  // ***************************************************************************
  void d_Mult(const RajaVector &b,
              RajaVector &x,
              const int N,
              const RajaOperator *oper,
              RajaSolverOperator *prec,
              const bool iterative_mode,
              RajaVector &r,
              RajaVector &d,
              RajaVector &z,
              double &final_norm) {
    assert(false);
    cuCG<<<1,1>>>(b,x,b.Size(),oper,prec,iterative_mode,r,d,z,final_norm);
  }
 
} // mfem
