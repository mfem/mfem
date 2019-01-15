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

#include "mfem.hpp"
#include "mechanics_solver.hpp"
#include "../../linalg/linalg.hpp"
#include "../../general/globals.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>


namespace mfem
{

using namespace std;

void ExaNewtonSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
}

  void ExaNewtonSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_max;
   const bool have_b = (b.Size() == Height());

   Vector c_fix(x.Size());
   
   if (!iterative_mode)
   {
      x = 0.0;
   }

   oper->Mult(x, r);
   if (have_b)
   {
      r -= b;
   }

   norm0 = norm = Norm(r);
   //Set the value for the norm that we'll exit on
   norm_max = rel_tol*norm;//std::max(rel_tol*norm, abs_tol);

   printf("Max norm %lf\n", norm_max);
   printf("Relative Norm %lf\n", rel_tol);
   
   prec->iterative_mode = false;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      //Make sure the norm is finite
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }
      //See if our solution has converged and we can quit
      if (norm <= norm_max)
      {
         converged = 1;
         break;
      }
      //See if we've gone over the max number of desired iterations
      if (it >= max_iter)
      {
         converged = 0;
         break;
      }

      prec->SetOperator(oper->GetGradient(x));

      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
                         // ExaConstit may use GMRES here
      //The scaling factor is usually set to 1 by default
      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      
      add(x, -c_scale, c, x); // full update to the current config
                              // ExaConstit (srw)
      
      //We now get our new residual
      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

      //Find our new norm
      norm = Norm(r);
      //fix_me...
      //way to test the umat
      //norm = 1e-14;
   }

   final_iter = it;
   final_norm = norm;
}
  
}
