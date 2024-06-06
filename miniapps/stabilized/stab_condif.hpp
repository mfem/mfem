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
#include "stab_tau.hpp"

namespace mfem
{

/** This Class defines a monolithic integrator for stabilized multi-dimensional
    convection-diffusion.

     $(a \cdot \nabla u, v) + (\kappa \nabla u, \nabla v) 
    + \sum (a \cdot \nabla u - \kappa \Delta u, \tau (a \cdot \nabla v + s \kappa \Delta v))_e$

     $(f, \nabla v) 
    + \sum (f, \tau (a \cdot \nabla v + s \kappa \Delta v))_e$
*/
class StabConDifIntegrator : public BilinearFormIntegrator,
                             public LinearFormIntegrator
{
protected:
   /// The advection field
   VectorCoefficient *adv;
   /// The diffusion parameter and force fields
   Coefficient *kappa, *force;

   /// The stabilization parameter
   Tau *tau;
   bool own_tau;

   StabType stab;

private:
   Vector laplace, shape, adshape, trail, test;
   DenseMatrix dshape;

public:
   StabConDifIntegrator(VectorCoefficient *a,
                        Coefficient *k,
                        Coefficient *f,
                        Tau *t = nullptr, StabType s = GALERKIN);

   ~StabConDifIntegrator();

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
}; 

/** This Class composes standard integrators to obtain a stabilized formulation for 
    multi-dimensional convection-diffusion.

     $(a \cdot \nabla u, v) + (\kappa \nabla u, \nabla v) 
    + \sum (a \cdot \nabla u - \kappa \Delta u, \tau (a \cdot \nabla v + s \kappa \Delta v))_e$

     $(f, \nabla v) 
    + \sum (f, \tau (a \cdot \nabla v + s \kappa \Delta v))_e$
*/
class StabConDifComposition
{

private:
   /// The advection field
   VectorCoefficient *adv;
   /// The diffusion parameter and force fields
   Coefficient *kappa, *force;

   /// The stabilization parameter
   Tau *tau;
   bool own_tau;

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

public:

   /** Constructor
       @a a: is the advection velocity field.
       @a k: is the diffusion param field.
       @a f: is the force field. */
   StabConDifComposition(VectorCoefficient *a,
                         Coefficient *k,
                         Coefficient *f,
                         Tau *t = nullptr);

   /// Destructor
   ~StabConDifComposition();

   /// This method sets the integrators for the bilinearform
   void SetBilinearIntegrators(BilinearForm *a, StabType s);

   /// This method sets the integrators for the linearform
   void SetLinearIntegrators(LinearForm *b, StabType s);
};

} // namespace mfem

#endif
