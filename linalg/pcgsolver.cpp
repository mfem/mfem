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
#include <fstream>
#include <iomanip>
#include <math.h>
#include "vector.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"

// Preconditined conjugate gradient

void PCG ( const Operator &A, const Operator &B,
           const Vector &b, Vector &x,
           int print_iter=0, int max_num_iter=1000,
           double RTOLERANCE=10e-12, double ATOLERANCE=10e-24,
           int save = 0)
{
   int i, dim = x.Size();
   double r0, den, nom, nom0, betanom, alpha, beta;
   Vector r(dim), d(dim), z(dim);

   A.Mult(x, r);                               //    r = A x
   subtract(b, r, r);                          //    r = b  - r
   B.Mult(r, z);                               //    z = B r
   d = z;
   nom0 = nom = z * r;

   if (print_iter == 1)
      cout << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
           << nom << endl;

   if ( (r0 = nom * RTOLERANCE) < ATOLERANCE) r0 = ATOLERANCE;
   if (nom < r0)
      return;

   A.Mult(d, z);
   den = z * d;

   if (den < 0.0) {
      cout << "Negative denominator in step 0 of PCG: ";
      cout << den << endl;
      //    return;
   }

   if (den == 0.0)
      return;

   // start iteration
   for(i= 1; i <= max_num_iter ; i++) {

      if (save)
         if (i % save == 0) {
            cout << "saving the solution vector on iteration " << i << endl;
            ofstream out("pcg.x");
            x.Print (out,1);
         }

      alpha = nom/den;
      add(x, alpha, d, x);                  //  x = x + alpha d
      add(r,-alpha, z, r);                  //  r = r - alpha z

      B.Mult( r, z);                        //  z = B r
      betanom = r * z;

      if (print_iter == 1)
         cout << "   Iteration : " << setw(3) << i << "  (B r, r) = "
              << betanom << endl;

      if ( betanom < r0) {
         if (print_iter == 2)
            cout << "Number of PCG iterations: " << i << endl;
         else
            if (print_iter == 3)
               cout << "(B r_0, r_0) = " << nom0 << endl
                    << "(B r_N, r_N) = " << betanom << endl
                    << "Number of PCG iterations: " << i << endl;
         break;
      }

      beta = betanom/nom;
      add(z, beta, d, d);                   //  d = z + beta d
      A.Mult(d, z);
      den = d * z;
      nom = betanom;
   }
   if (i > max_num_iter)
   {
      cerr << "PCG: No convergence!" << endl;
      cout << "(B r_0, r_0) = " << nom0 << endl
           << "(B r_N, r_N) = " << betanom << endl
           << "Number of PCG iterations: " << (i-1) << endl;
   }
   if (print_iter >= 1 || i > max_num_iter)
   {
      if (i > max_num_iter)  i--;
      cout << "Average reduction factor = "
           << pow (betanom/nom0, 0.5/i) << endl;
   }
}

// Preconditioned stationary linear iteration
void SLI (const Operator &A, const Operator &B,
          const Vector &b, Vector &x,
          int print_iter=0, int max_num_iter=1000,
          double RTOLERANCE=10e-12, double ATOLERANCE=10e-24)
{
   int i, dim = x.Size();
   double r0, nom, nomold = 1, nom0, cf;
   Vector r(dim),  z(dim);

   r0 = -1.0;

   for(i = 1; i < max_num_iter ; i++) {
      A.Mult(x, r);         //    r = A x
      subtract(b, r, r);    //    r = b  - A x
      B.Mult(r, z);         //    z = B r

      nom = z * r;

      if (r0 == -1.0) {
         nom0 = nom;
         r0 = nom * RTOLERANCE;
         if (r0 < ATOLERANCE) r0 = ATOLERANCE;
      }

      cf = sqrt(nom/nomold);
      if (print_iter == 1) {
         cout << "   Iteration : " << setw(3) << i << "  (B r, r) = "
              << nom;
         if (i > 1)
            cout << "\tConv. rate: " << cf;
         cout << endl;
      }
      nomold = nom;

      if (nom < r0) {
         if (print_iter == 2)
            cout << "Number of iterations: " << i << endl
                 << "Conv. rate: " << cf << endl;
         else
            if (print_iter == 3)
               cout << "(B r_0, r_0) = " << nom0 << endl
                    << "(B r_N, r_N) = " << nom << endl
                    << "Number of iterations: " << i << endl;
         break;
      }

      add(x, 1.0, z, x);  //  x = x + B (b - A x)
   }

   if (i == max_num_iter)
   {
      cerr << "No convergence!" << endl;
      cout << "(B r_0, r_0) = " << nom0 << endl
           << "(B r_N, r_N) = " << nom << endl
           << "Number of iterations: " << i << endl;
   }
}
