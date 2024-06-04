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
