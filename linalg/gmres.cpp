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
#include <math.h>
#include "vector.hpp"
#include "matrix.hpp"
#include "densemat.hpp"

//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the
// Generalized Minimum Residual method
//
// GMRES follows the algorithm described on p. 20 of the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  initial approximation / approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************

inline void GeneratePlaneRotation (double &dx, double &dy, double &cs, double &sn)
{
   if (dy == 0.0) {
      cs = 1.0;
      sn = 0.0;
   } else if (fabs(dy) > fabs(dx)) {
      double temp = dx / dy;
      sn = 1.0 / sqrt( 1.0 + temp*temp );
      cs = temp * sn;
   } else {
      double temp = dy / dx;
      cs = 1.0 / sqrt( 1.0 + temp*temp );
      sn = temp * cs;
   }
}

inline void ApplyPlaneRotation (double &dx, double &dy, double &cs, double &sn)
{
   double temp  =  cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
}

inline void Update (Vector &x, int k, DenseMatrix &h, Vector &s, Array<Vector*> &v)
{
   Vector y(s.Size());  y = s;

   // Backsolve:
   for (int i = k; i >= 0; i--) {
      y(i) /= h(i,i);
      for (int j = i - 1; j >= 0; j--)
         y(j) -= h(j,i) * y(i);
   }

   for (int j = 0; j <= k; j++)
      x.Add(y(j),*v[j]);
}

inline double norm (Vector & u)
{
   return sqrt(u*u);
}

int
GMRES(const Operator &A, Vector &x, const Vector &b,
      const Operator &M, int &max_iter,
      int m, double &tol, double &atol, int printit)
{
   int n = A.Size();

   DenseMatrix H(m+1,m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector w(n), av(n);

   double resid;
   int i, j, k;

   M.Mult(b,w);
   double normb = norm(w); // normb = ||M b||
   if (normb == 0.0)
      normb = 1;

   Vector r(n);
   A.Mult(x, r);
   subtract(b,r,w);
   M.Mult(w, r);           // r = M (b - A x)
   double beta = norm(r);  // beta = ||r||

   resid = beta / normb;

   if (resid * resid <= tol) {
      tol = resid * resid;
      max_iter = 0;
      return 0;
   }

   if (printit)
      cout << "   Pass : " << setw(2) << 1
           << "   Iteration : " << setw(3) << 0
           << "  (r, r) = " << beta*beta << endl;

   tol *= (normb*normb);
   tol = (atol > tol) ? atol : tol;

   Array<Vector *> v(m+1);
   for (i= 0; i<=m; i++)
      v[i] = NULL;

   j = 1;
   while (j <= max_iter)
   {
      if (v[0] == NULL)  v[0] = new Vector(n);
      (*v[0]) = 0.0;
      v[0] -> Add (1.0/beta, r);   // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++) {
         A.Mult((*v[i]),av);
         M.Mult(av,w);              // w = M A v[i]

         for (k = 0; k <= i; k++) {
            H(k,i) = w * (*v[k]);    // H(k,i) = w * v[k]
            w.Add(-H(k,i), (*v[k])); // w -= H(k,i) * v[k]
         }

         H(i+1,i)  = norm(w);       // H(i+1,i) = ||w||
         if (v[i+1] == NULL)  v[i+1] = new Vector(n);
         (*v[i+1]) = 0.0;
         v[i+1] -> Add (1.0/H(i+1,i), w); // v[i+1] = w / H(i+1,i)

         for (k = 0; k < i; k++)
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         resid = fabs(s(i+1));
         if (printit)
            cout << "   Pass : " << setw(2) << j
                 << "   Iteration : " << setw(3) << i+1
                 << "  (r, r) = " << resid*resid << endl;

         if ( resid*resid < tol) {
            Update(x, i, H, s, v);
            tol = resid * resid;
            max_iter = j;
            for (i= 0; i<=m; i++)
               if (v[i])  delete v[i];
            return 0;
         }
      }

      if (printit)
         cout << "Restarting..." << endl;

      Update(x, i-1, H, s, v);

      A.Mult(x, r);
      subtract(b,r,w);
      M.Mult(w, r);           // r = M (b - A x)
      beta = norm(r);         // beta = ||r||
      if ( resid*resid < tol) {
         tol = resid * resid;
         max_iter = j;
         for (i= 0; i<=m; i++)
            if (v[i])  delete v[i];
         return 0;
      }

      j++;
   }

   tol = resid * resid;
   for (i= 0; i<=m; i++)
      if (v[i])  delete v[i];
   return 1;
}

int
aGMRES(const Operator &A, Vector &x, const Vector &b,
       const Operator &M, int &max_iter,
       int m_max, int m_min, int m_step, double cf,
       double &tol, double &atol, int printit)
{
   int n = A.Size();

   int m = m_max;

   DenseMatrix H(m+1,m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector w(n), av(n);

   double r1, resid;
   int i, j, k;

   M.Mult(b,w);
   double normb = norm(w); // normb = ||M b||
   if (normb == 0.0)
      normb = 1;

   Vector r(n);
   A.Mult(x, r);
   subtract(b,r,w);
   M.Mult(w, r);           // r = M (b - A x)
   double beta = norm(r);  // beta = ||r||

   resid = beta / normb;

   if (resid * resid <= tol) {
      tol = resid * resid;
      max_iter = 0;
      return 0;
   }

   if (printit)
      cout << "   Pass : " << setw(2) << 1
           << "   Iteration : " << setw(3) << 0
           << "  (r, r) = " << beta*beta << endl;

   tol *= (normb*normb);
   tol = (atol > tol) ? atol : tol;

   m = m_max;
   Array<Vector *> v(m+1);
   for (i= 0; i<=m; i++) {
      v[i] = new Vector(n);
      (*v[i]) = 0.0;
   }

   j = 1;
   while (j <= max_iter) {
      (*v[0]) = 0.0;
      v[0] -> Add (1.0/beta, r);   // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      r1 = beta;

      for (i = 0; i < m && j <= max_iter; i++) {
         A.Mult((*v[i]),av);
         M.Mult(av,w);              // w = M A v[i]

         for (k = 0; k <= i; k++) {
            H(k,i) = w * (*v[k]);    // H(k,i) = w * v[k]
            w.Add(-H(k,i), (*v[k])); // w -= H(k,i) * v[k]
         }

         H(i+1,i)  = norm(w);       // H(i+1,i) = ||w||
         (*v[i+1]) = 0.0;
         v[i+1] -> Add (1.0/H(i+1,i), w); // v[i+1] = w / H(i+1,i)

         for (k = 0; k < i; k++)
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         resid = fabs(s(i+1));
         if (printit)
            cout << "   Pass : " << setw(2) << j
                 << "   Iteration : " << setw(3) << i+1
                 << "  (r, r) = " << resid*resid << endl;

         if ( resid*resid < tol) {
            Update(x, i, H, s, v);
            tol = resid * resid;
            max_iter = j;
            for (i= 0; i<=m; i++)
               delete v[i];
            return 0;
         }
      }

      if (printit)
         cout << "Restarting..." << endl;

      Update(x, i-1, H, s, v);

      A.Mult(x, r);
      subtract(b,r,w);
      M.Mult(w, r);           // r = M (b - A x)
      beta = norm(r);         // beta = ||r||
      if ( resid*resid < tol) {
         tol = resid * resid;
         max_iter = j;
         for (i= 0; i<=m; i++)
            delete v[i];
         return 0;
      }

      if (beta/r1 > cf)
      {
         if (m - m_step >= m_min)
            m -= m_step;
         else
            m = m_max;
      }

      j++;
   }

   tol = resid * resid;
   for (i= 0; i<=m; i++)
      delete v[i];
   return 1;
}
