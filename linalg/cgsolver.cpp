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
#include "sparsemat.hpp"

// Conjugate Gradient solver

void CG( const Operator &A, const Vector &b, Vector &x,
         int print_iter=0, int max_num_iter=1000,
         double RTOLERANCE=10e-12, double ATOLERANCE=10e-24){

   int i, dim = x.Size();
   double den, nom, nom0, betanom, alpha, beta, r0;
   Vector r(dim), d(dim), Ad(dim);

   A.Mult( x, r);
   subtract( b, r, r);                         // r = b - A x
   d = r;
   nom0 = nom = r * r;

   if (print_iter == 1)
      cout << "   Iteration : " << setw(3) << 0 << "  (r, r) = "
           << nom << endl;

   if ( (r0 = nom * RTOLERANCE) < ATOLERANCE) r0 = ATOLERANCE;
   if (nom < r0)
      return;

   A.Mult( d, Ad);
   den = d * Ad;


   if (den <= 0.0) {
      if (nom0 > 0.0)
         cout <<"Operator A is not postive definite. (Ar,r) = "
              << den << endl;
      return;
   }

   // start iteration                          //  d = r, Ad = A r
   for(i = 1; i<max_num_iter ;i++) {
      alpha= nom/den;                           // alpha = (r_o,r_o)/(Ar_o,r_o)

      add(x, alpha, d, x);                      //   x =   x + alpha * d
      add(r, -alpha, Ad, r);                    // r_n = r_o - alpha * Ad
      betanom = r*r;                            // betanom = (r_o, r_o)

      if (print_iter == 1)
         cout << "   Iteration : " << setw(3) << i << "  (r, r) = "
              << betanom << endl;

      if ( betanom < r0) {
         if (print_iter == 2)
            cout << "Number of CG iterations: " << i << endl;
         else
            if (print_iter == 3)
               cout << "(r_0, r_0) = " << nom0 << endl
                    << "(r_N, r_N) = " << betanom << endl
                    << "Number of CG iterations: " << i << endl;
         break;
      }

      beta = betanom/nom;                       // beta = (r_n,r_n)/(r_o,r_o)
      add(r, beta, d, d);                       //    d = r_n + beta * d
      A.Mult(d, Ad);                            //   Ad = A d
      den = d * Ad;                             //  den = (d , A d)
      if (den <= 0.0)
      {
         if (d * d > 0.0)
            cout <<"Operator A is not postive definite. (Ad,d) = "
                 << den << endl;
      }
      nom = betanom;                            //  nom = (r_n, r_n)
   }
   if (i == max_num_iter && print_iter >= 0)
   {
      cerr << "CG: No convergence!" << endl;
      cout << "(r_0, r_0) = " << nom0 << endl
           << "(r_N, r_N) = " << betanom << endl
           << "Number of CG iterations: " << i << endl;
   }
}
