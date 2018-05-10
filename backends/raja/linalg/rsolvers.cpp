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

  // *************************************************************************
  void RajaCGSolver::h_Mult(const RajaVector &b, RajaVector &x) const {
    push(SkyBlue);
    //assert(false);
      
    int i;
    double r0, den, nom, nom0, betanom, alpha, beta;
    if (iterative_mode) {
      push(iMsub,SkyBlue);
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
      pop();
    }
    else
    {
      push(rbx0,SkyBlue);
      r = b;
      x = 0.0;
      pop();
    }

    push(d,SkyBlue);
    if (prec)
    {
      prec->Mult(r, z); // z = B r
      d = z;
    }
    else
    {
      d = r;
    }
    pop();

    push(nom,SkyBlue);
    nom0 = nom = Dot(d, r);
    MFEM_ASSERT(IsFinite(nom), "nom = " << nom);
    pop();
      
    if (print_level == 1
        || print_level == 3)
    {
      mfem::out << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                << nom << (print_level == 3 ? " ...\n" : "\n");
    }
    pop();
      
    push(r0,SkyBlue);
    r0 = std::max(nom*rel_tol*rel_tol,abs_tol*abs_tol);
    pop();

    push(nCvg?,SkyBlue);
    if (nom <= r0)
    {
      converged = 1;
      final_iter = 0;
      final_norm = sqrt(nom);
      pop();
      pop();
      return;
    }
    pop();

    push(z=Ad,SkyBlue);
    oper->Mult(d, z);  // z = A d
    pop();

    push(z.d,SkyBlue);
    den = Dot(z, d);
    MFEM_ASSERT(IsFinite(den), "den = " << den);
    pop();

    if (print_level >= 0 && den < 0.0)
    {
      mfem::out << "Negative denominator in step 0 of PCG: " << den << '\n';
    }

    push(dCvg?,SkyBlue);
    if (den == 0.0)
    {
      converged = 0;
      final_iter = 0;
      final_norm = sqrt(nom);
      pop();
      pop();
      return;
    }
    pop();

    // start iteration
    converged = 0;
    final_iter = max_iter;
    push(for,SkyBlue);
    for (i = 1; true; ){
      alpha = nom/den;
      push(x+ad,SkyBlue);
      add(x,  alpha, d, x);     //  x = x + alpha d
      pop();
      push(r-aAd,SkyBlue);
      add(r, -alpha, z, r);     //  r = r - alpha A d
      pop();

      push(z=Br,SkyBlue);
      if (prec)
      {
        prec->Mult(r, z);      //  z = B r
        betanom = Dot(r, z);
      }
      else
      {
        betanom = Dot(r, r);
      }
      MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);
      pop();
        
        
      if (print_level == 1)
      {
        mfem::out << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                  << betanom << '\n';
      }

      if (betanom < r0)
      {
        if (print_level == 2)
        {
          mfem::out << "Number of PCG iterations: " << i << '\n';
        }
        else if (print_level == 3)
        {
          mfem::out << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                    << betanom << '\n';
        }
        converged = 1;
        final_iter = i;
        break;
      }

      if (++i > max_iter)
      {
        break;
      }

      push(z+bd,SkyBlue);
      beta = betanom/nom;
      if (prec)
      {
        add(z, beta, d, d);   //  d = z + beta d
      }
      else
      {
        add(r, beta, d, d);
      }
      pop();
        
      push(Ad,SkyBlue);
      oper->Mult(d, z);       //  z = A d
      pop();

      push(d.z,SkyBlue);
      den = Dot(d, z);
      pop();
        
      MFEM_ASSERT(IsFinite(den), "den = " << den);
      if (den <= 0.0)
      {
        if (print_level >= 0 && Dot(d, d) > 0.0)
          mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                    << den << '\n';
      }
      nom = betanom;
    }
    pop();
     
    if (print_level >= 0 && !converged)
    {
      if (print_level != 1)
      {
        if (print_level != 3)
        {
          mfem::out << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                    << nom0 << " ...\n";
        }
        mfem::out << "   Iteration : " << std::setw(3) << final_iter << "  (B r, r) = "
                  << betanom << '\n';
      }
      mfem::out << "PCG: No convergence!" << '\n';
    }
      
    if (print_level >= 1 || (print_level >= 0 && !converged))
    {
      mfem::out << "Average reduction factor = "
                << pow (betanom/nom0, 0.5/final_iter) << '\n';
    }
    push(final_norm,SkyBlue);
    final_norm = sqrt(betanom);
    pop();
  }
    
} // mfem
