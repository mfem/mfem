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
    It supports H1, H(curl) (ND elements), H(div) (RT elements) and L2 (DG).

    For "smooth enough" solutions the Galerkin error measured in the appropriate
    norm satisfies || u - u_h || ~ h^k

    Here, k is called the asymptotic rate of convergence

    For successive uniform h-refinements the rate can be estimated by
    k = log(||u - u_h|| / ||u - u_{h/2}||)/log(2)
*/
class ConvergenceStudy
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

   void AddL2Error(GridFunction *gf, Coefficient *scalar_u,
                   VectorCoefficient *vector_u);
   void AddGf(GridFunction *gf, Coefficient *scalar_u,
              VectorCoefficient *grad=nullptr,
              Coefficient *ell_coeff=nullptr, double Nu=1.0);
   void AddGf(GridFunction *gf, VectorCoefficient *vector_u,
              VectorCoefficient *curl, Coefficient *div);
   // returns the L2-norm of scalar_u or vector_u
   double GetNorm(GridFunction *gf, Coefficient *scalar_u,
                  VectorCoefficient *vector_u);

public:

   /// Clear any internal data
   void Reset();

   /// Add L2 GridFunction, the exact solution and possibly its gradient and/or
   /// DG face jumps parameters
   void AddL2GridFunction(GridFunction *gf, Coefficient *scalar_u,
                          VectorCoefficient *grad=nullptr,
                          Coefficient *ell_coeff=nullptr, double Nu=1.0)
   {
      AddGf(gf, scalar_u, grad, ell_coeff, Nu);
   }

   /// Add H1 GridFunction, the exact solution and possibly its gradient
   void AddH1GridFunction(GridFunction *gf, Coefficient *scalar_u,
                          VectorCoefficient *grad=nullptr)
   {
      AddGf(gf, scalar_u, grad);
   }

   /// Add H(curl) GridFunction, the exact solution and possibly its curl
   void AddHcurlGridFunction(GridFunction *gf, VectorCoefficient *vector_u,
                             VectorCoefficient *curl=nullptr)
   {
      AddGf(gf, vector_u, curl, nullptr);
   }

   /// Add H(div) GridFunction, the exact solution and possibly its div
   void AddHdivGridFunction(GridFunction *gf, VectorCoefficient *vector_u,
                            Coefficient *div=nullptr)
   {
      AddGf(gf,vector_u, nullptr, div);
   }

   /// Get the L2 error at step n
   double GetL2Error(int n)
   {
      MFEM_VERIFY( n <= counter,"Step out of bounds")
      return L2Errors[n];
   }

   /// Get all L2 errors
   void GetL2Errors(Array<double> & L2Errors_)
   {
      L2Errors_ = L2Errors;
   }

   /// Get the Grad/Curl/Div error at step n
   double GetDError(int n)
   {
      MFEM_VERIFY(n <= dcounter,"Step out of bounds")
      return DErrors[n];
   }

   /// Get all Grad/Curl/Div errors
   void GetDErrors(Array<double> & DErrors_)
   {
      DErrors_ = DErrors;
   }

   /// Get the DGFaceJumps error at step n
   double GetDGFaceJumpsError(int n)
   {
      MFEM_VERIFY(n<= fcounter,"Step out of bounds")
      return DGFaceErrors[n];
   }

   /// Get all DGFaceJumps errors
   void GetDGFaceJumpsErrors(Array<double> & DGFaceErrors_)
   {
      DGFaceErrors_ = DGFaceErrors;
   }

   /// Print rates and errors
   void Print(bool relative = false, std::ostream &out = mfem::out);
};

} // namespace mfem

#endif // MFEM_CONVERGENCE
