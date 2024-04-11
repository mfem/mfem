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

#ifndef MFEM_STAB_CONDIF_HPP
#define MFEM_STAB_CONDIF_HPP

#include "mfem.hpp"

namespace mfem
{

/** This Class defines the stabilisation parameter for the multi-dimensional
    convection-diffusion problem. The parameter is defined as:
    $tau = ({\bf a} \cdot G {\bf a} + C_{inv} \kappa^2 G:G)^{-1/2}$
    where:
     - ${\bf a}$ is the convection velocity
     - $\kappa$ is the diffusion parameter
     - $G$ is the metric tensor
     - $C_{inv}$ is a discretisation parameter dependent parameter
*/
class StabilizedParameter: public Coefficient
{
private:
   /// The advection field
   VectorCoefficient *adv;
   // The diffusion parameter field
   Coefficient *dif;
   /// The inverse estimate of the elements used
   real_t Ci;

   /// Dimension of the problem
   int dim;
   /// Current Metric tensor
   DenseMatrix Gij;
   /// Current Velocity vector
   Vector u;


public:
   /** Construct a stabilized confection-diffusion integrator with:
    -  vector coefficient @a a the convection velocity
    -  scalar coefficient @a d the diffusion coefficient
    */
   StabilizedParameter (VectorCoefficient &a, Coefficient &d, real_t c = 12.0)
   : adv(&a), dif(&d), Ci(c)
   {
      dim = adv->GetVDim();
      Gij.SetSize(dim,dim);
      u.SetSize(dim);
   }

   /// Evaluate the coefficient at @a ip.
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) override;

    // Destructor
   ~StabilizedParameter()
   {}
};

/// Enumerate to indicate the stabilisation type.
enum StabType
{
      GALERKIN,
      SUPG,
      GLS,
      VMS
};

/** $({\bf a}\cdot \nabla u,  v) + (\kappa \nabla u,  \nabla v)  \\
 + ({\bf a}\cdot \nabla u, \tau {\bf a}\cdot\nabla v)
 - (\kappa \Delta u, \tau {\bf a}\cdot\nabla v)\\
 + \gamma ({\bf a}\cdot \nabla u, \tau \kappa \Delta v)
 - \gamma (\kappa \Delta u, \tau \kappa \Delta v)
  = (f,v)
   + (f, \tau {\bf a}\cdot\nabla v) 
   + \gamma (f, \tau \kappa \Delta v)\\$
   Where:
   -$\gamma = 0$: SUPG
   -$\gamma = 1$: VMS
   -$\gamma =-1$: GLS
   */
class StabilizedConvectionDiffusion
{

private:
   /// The advection field
   VectorCoefficient *adv;
   // The diffusion parameter and force fields
   Coefficient *kappa, *force;

   StabilizedParameter *tau;
   //// Helper coefficients for defining the weak forms
   VectorCoefficient *adv_tau;
   Coefficient *kappa_tau;

   /// SUPG coefficients
   VectorCoefficient *adv_tau_force;
   VectorCoefficient *adv_tau_kappa;
   MatrixCoefficient *adv_tau_adv;

   /// GLS/VMS coefficients
   Coefficient *kappa_tau_kappa;
   Coefficient *kappa_tau_force;

   /// The bilinear form
   BilinearForm *a = nullptr;
   /// The linear form
   LinearForm *b = nullptr;

public:

   /** Constructor
       @a a: is the advection velocity field.
       @a k: is the diffusion param field.
       @a f: is the force field. */
   StabilizedConvectionDiffusion(VectorCoefficient &a, Coefficient &k, Coefficient &f);

   /// Destructor
   ~StabilizedConvectionDiffusion()
   {
      if (a) delete a;
      if (b) delete b;
      delete tau, adv_tau, adv_tau_kappa, adv_tau_adv, adv_tau_force,
                  kappa_tau, kappa_tau_kappa,kappa_tau_force;
   }

   /// This method destroys the current forms and creates new ones.
   /// The @a stype flag determines the stabilized formulation.
   void SetForms(FiniteElementSpace *fes, StabType stype = GALERKIN);
   /// Add weak boundary terms to the from --> DOES NOT WORK YET
   void AddWBC(Coefficient &u_dir, real_t &penalty);

   /// This method assembles the bilinearform and returns it.
   /// Caller does not get ownership.
   BilinearForm* GetBilinearForm()
   {
      a->Assemble();
      return a;
   }

   /// This method assembles the linearform and returns it.
   /// Caller does not get ownership.
   LinearForm* GetLinearForm()
   {
      b->Assemble();
      return b;
   }
};

} // namespace mfem

#endif
