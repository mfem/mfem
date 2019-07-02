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
   double norm_prev, norm_ratio;
   const bool have_b = (b.Size() == Height());

   //Might want to use this to fix things later on for example when we have a
   //large residual. We might also want to eventually try and find a converged
   //relaxation factor which would mean resetting our solution vector a few times.
   Vector x_prev(x.Size());
   
   if (!iterative_mode)
   {
      x = 0.0;
   }
   
   x_prev = x;

   oper->Mult(x, r);
   if (have_b)
   {
      r -= b;
   }

   norm0 = norm = norm_prev = Norm(r);
   norm_ratio = 1.0;
   //Set the value for the norm that we'll exit on
   norm_max = std::max(rel_tol*norm, abs_tol);
   
   prec->iterative_mode = false;
   double scale = 1.0;

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
      //We'll want to edit this later on to automatically change the c_scale
      //We'll probably want to base it on this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1608280
      //doi:10.1109/TMAG.2006.871566
      //The change in scaling factor would become something like
      //d_alpha^(i) = (||R^(i+1)|| - ||R^(i)||) * 2 * R^(i+1)^T del_R^(i + 1)/del_A^(i + 1) * delta_A^(i) where the middle term is our jacobian at the current time step.
      //Our d_alpha^(i) could then be used to find our alpha term using a
      //backward eulerian integration term. So alpha^(i+1) = alpha^(i) + d_alpha^(i).
      //We can probably use this alpha^(i+1) in our next time step.
      //I wouldn't call this the best method, but it might work. I still need
      //to find a nice paper or program that actually implements this or
      //expands on the algorithm quite a bit more. The current paper is pretty
      //lacking...
//      const double c_scale = ComputeScalingFactor(x, b);
      const double c_scale = scale;
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

      //Find our new norm and save our previous time step value.
      norm_prev = norm;
      norm = Norm(r);
      //We're going to more or less use a heuristic method here for now if
      //our ratio is greater than 1e-1 then we'll set our scaling factor for
      //the next iteration to 0.5.
      //We want to do this since it's not uncommon for us to run into the case
      //where our solution is oscillating over the one we actually want.
      //Eventually, we'll fix this in our scaling factor function.
      norm_ratio = norm / norm_prev;
      
      if (norm_ratio > 1.0e-1){
         scale = 0.5;
         if (print_level >= 0)
         {
            mfem::out << "The relaxation factor for the next iteration has been reduced to " << scale << "\n";
         }
         
      }else{
         scale = 1.0;
      }
      
      //fix_me...
      //way to test the umat
      //norm = 1e-14;
   }

   final_iter = it;
   final_norm = norm;
}
  
}
