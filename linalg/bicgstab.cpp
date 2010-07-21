// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include <iostream>
#include <iomanip>
#include "vector.hpp"
#include "matrix.hpp"

//*****************************************************************
// Iterative template routine -- BiCGSTAB
//
// BiCGSTAB solves the unsymmetric linear system Ax = b
// using the Preconditioned BiConjugate Gradient Stabilized method
//
// BiCGSTAB follows the algorithm described on p. 27 of the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************

int
BiCGSTAB(const Operator &A, Vector &x, const Vector &b,
         const Operator &M, int &max_iter, double &tol,
         double atol, int printit)
{
   int i, n = A.Size();
   double resid;
   double rho_1, rho_2=1.0, alpha=1.0, beta, omega=1.0;
   Vector p(n), phat(n), s(n), shat(n), t(n), v(n), r(n), rtilde(n);

   A.Mult (x, r);  //  r = A * x
   subtract (b, r, r);  //  r = b - r
   rtilde = r;

   resid = (r * r);
   if (printit)
      cout << "   iter " << 0 << ",   (r, r) = " << resid << endl;
   tol *= resid;
   tol = (atol > tol) ? atol : tol;

   if (resid <= tol)
   {
      tol = resid;
      max_iter = 0;
      return 0;
   }

   for (i = 1; i <= max_iter; i++)
   {
      rho_1 = rtilde * r;
      if (rho_1 == 0)
      {
         tol = resid;
         if (printit)
            cout << "   iter " << i << ",   (r, r) = " << resid << endl;
         return 2;
      }
      if (i == 1)
         p = r;
      else
      {
         beta = (rho_1/rho_2) * (alpha/omega);
         add (p, -omega, v, p);  //  p = p - omega * v
         add (r, beta, p, p);    //  p = r + beta * p
      }
      M.Mult (p, phat);  //  phat = M^{-1} * p
      A.Mult (phat, v);  //  v = A * phat
      alpha = rho_1 / (rtilde * v);
      add (r, -alpha, v, s);  //  s = r - alpha * v
      resid = (s * s);
      if (resid < tol)
      {
         x.Add (alpha, phat);  //  x = x + alpha * phat
         tol = resid;
         if (printit)
            cout << "   iter " << i << ",   (s, s) = " << resid << endl;
         return 0;
      }
      if (printit)
         cout << "   iter " << i << ",   (s, s) = " << resid << ";   ";
      M.Mult (s, shat);  //  shat = M^{-1} * s
      A.Mult (shat, t);  //  t = A * shat
      omega = (t * s) / (t * t);
      x.Add (alpha, phat);  //  x += alpha * phat
      x.Add (omega, shat);  //  x += omega * shat
      add (s, -omega, t, r);  //  r = s - omega * t

      rho_2 = rho_1;
      resid = (r * r);
      if (printit)
         cout << "(r, r) = " << resid << endl;
      if (resid < tol)
      {
         tol = resid;
         max_iter = i;
         return 0;
      }
      if (omega == 0)
      {
         tol = resid;
         return 3;
      }
   }

   tol = resid;
   return 1;
}
