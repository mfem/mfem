// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_STAB_TAU_HPP
#define MFEM_STAB_TAU_HPP

#include "mfem.hpp"

namespace mfem
{

/// Enumerate to indicate the stabilisation type.
enum StabType
{
   GALERKIN = -2,
   SUPG = 0,
   GLS = -1,
   VMS = 1
};

/// This Class defines a generic stabilisation parameter.
class Tau: public Coefficient
{
protected:
   /// The advection field
   VectorCoefficient *adv;
   // The diffusion parameter field
   Coefficient *kappa;
   /// Dimension of the problem
   int dim;
   /// Velocity vector
   Vector a;

public:
   /** Construct a stabilized confection-diffusion integrator with:
    -  vector coefficient @a a the convection velocity
    -  scalar coefficient @a d the diffusion coefficient
    */
   Tau (VectorCoefficient *a_, Coefficient *k) : adv(a_), kappa(k)
   {
      dim = adv->GetVDim();
      a.SetSize(dim);
   };
   Tau () : adv(nullptr), kappa(nullptr) {};

   void SetConvection(VectorCoefficient *a_)
   {
      adv = a_;
      dim = adv->GetVDim();
      a.SetSize(dim);
   };
   void SetDiffusion(Coefficient *k_) { kappa = k_; };

};

/** This Class defines the stabilisation parameter for the multi-dimensional
    convection-diffusion problem. The parameter is defined as given in:

    Franca, L.P., Frey, S.L., & Hughes, T.J.R.
    Stabilized finite element methods:
    I. Application to the advective-diffusive model.
    Computer Methods in Applied Mechanics and Engineering, 95(2), 253-276.
*/
class FFH92Tau: public Tau
{
private:
   /// User provided inverse estimate of the element
   real_t Ci;
   /// If the user provided inverse estimate is negative 
   /// the actual precomputed inverse estimate is used.
   InverseEstimateCoefficient *iecf = nullptr;

   /// The norm used for the velocity vector
   real_t p;
   /// Temp variable
   Vector row;
   /// Element size in different directions
   Vector h;

   /// Flag for printing
   bool print = true;

    /** Returns element size according to:

        Harari, I, & Hughes, T.J.R.
        What are C and h?: Inequalities for the analysis and design of
        finite element methods.
        Computer methods in applied mechanics and engineering 97(2), 157-192.
   */
   real_t GetElementSize(ElementTransformation &T);

   real_t GetInverseEstimate(ElementTransformation &T,
                             const IntegrationPoint &ip, real_t scale = 1.0);

public:
   /** Construct a stabilized confection-diffusion integrator with:
    -  vector coefficient @a a the convection velocity
    -  scalar coefficient @a d the diffusion coefficient
    */
    FFH92Tau (VectorCoefficient *a, Coefficient *k,
              FiniteElementSpace *fes = nullptr,
              real_t c_ = -1.0, real_t p_ = 2.0)
      :  Tau(a,k), Ci(c_), p(p_)
   {
      if (Ci < 0.0)
      {
         iecf = new InverseEstimateCoefficient(fes, *kappa);
      }
   };

   FFH92Tau (FiniteElementSpace *fes = nullptr,
             real_t c_ = -1.0, real_t p_ = 2.0)
      : Ci(c_), p(p_)
   {
      if (Ci < 0.0)
      {
         iecf = new InverseEstimateCoefficient(fes, *kappa);
      }
   };

   /// Evaluate the coefficient at @a ip.
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) override;

    // Destructor
   ~FFH92Tau()
   { if (iecf) { delete iecf; } }
};

} // namespace mfem

#endif
