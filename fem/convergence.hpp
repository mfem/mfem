// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CONVERGENCE
#define MFEM_CONVERGENCE

#include "../linalg/linalg.hpp"
#include "gridfunc.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#endif

namespace mfem
{

/** @brief Class to compute error and convergence rates.
    It supports H1, H(curl) (ND elements), H(div) (RT elements) and L2 (DG). */
class Convergence
{
private:
   // counters for solutions/derivatives
   int counter=0;
   int dcounter=0;
   int fcounter=0;

   // space continuity type
   int cont_type=-1;

   // printing flag for helpful for MPI calls
   int print_flag=1;

   // exact solution and derivatives
   double CoeffNorm;
   double CoeffDNorm;

   // Arrays to store error/rates
   Array<double> L2Errors, DGFaceErrors, DErrors, EnErrors;
   Array<double> L2Rates, DGFaceRates, DRates, EnRates;
   Array<int> ndofs;

   void AddL2Error(GridFunction * gf, Coefficient * u, VectorCoefficient * U);
   void AddGf(GridFunction * gf, Coefficient * u, VectorCoefficient * grad,
              Coefficient * ell_coeff, double Nu);
   void AddGf(GridFunction * gf, VectorCoefficient * u,
              VectorCoefficient * curl, Coefficient * div);

   // returns the exact solution/grad/div/curl norm
   double GetNorm(GridFunction * gf, Coefficient * u, VectorCoefficient * U);

public:

   // Clear any internal data
   void Reset();

   // Add H1/L2 GridFunction,
   // the exact solution and possibly
   // its gradient and/or DG face jumps parameters
   void AddGridFunction(GridFunction * gf, Coefficient * u,
                        VectorCoefficient * grad=NULL,
                        Coefficient * ell_coeff=NULL, double Nu=1.0)
   {
      AddGf(gf, u, grad, ell_coeff,Nu);
   }

   // Add H(curl)/H(div) GridFunction,
   // and the exact solution
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u)
   {
      AddGf(gf,u, nullptr, nullptr);
   }

   // Add H(curl) GridFunction,
   // the exact solution and its curl
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u,
                        VectorCoefficient * curl)
   {
      AddGf(gf,u, curl, nullptr);
   }

   // Add H(div) GridFunction,
   // the exact solution and its div
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u,
                        Coefficient * div)
   {
      AddGf(gf,u, nullptr, div);
   }

   // Get L2 error at step n
   double GetL2Error(int n)
   {
      MFEM_VERIFY(n<= counter,"Step out of bounds")
      return L2Errors[n];
   }

   // Get all L2 errors
   void GetL2Errors(Array<double> & L2Errors_)
   {
      L2Errors_ = L2Errors;
   }

   // Get Grad/Curl/Div Error at step n
   double GetDError(int n)
   {
      MFEM_VERIFY(n<= dcounter,"Step out of bounds")
      return DErrors[n];
   }

   // Get all Grad/Curl/Div
   void GetDErrors(Array<double> & DErrors_)
   {
      DErrors_ = DErrors;
   }

   // Get DGFaceJumps Error at step n
   double GetDGFaceJumpsError(int n)
   {
      MFEM_VERIFY(n<= fcounter,"Step out of bounds")
      return DGFaceErrors[n];
   }

   // Get all DGFaceJumps
   void GetDGFaceJumpsErrors(Array<double> & DGFaceErrors_)
   {
      DGFaceErrors_ = DGFaceErrors;
   }

   // Print rates and errors
   void Print(bool relative = false, std::ostream &out = mfem::out);

};

} // namespace mfem

#endif // MFEM_CONVERGENCE
