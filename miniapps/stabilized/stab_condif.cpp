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

#include "stab_condif.hpp"

using namespace mfem;

real_t StabilizedParameter::Eval(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   MultAAt(T.InverseJacobian(),Gij);

   real_t nu = dif->Eval(T, ip);
   adv->Eval(u,T, ip);

   real_t tau = 0.0;
   for (int j = 0; j < dim; j++)
      for (int i = 0; i < dim; i++)
      {
         tau += Gij(i,j)*u[i]*u[j];
      }

   for (int j = 0; j < dim; j++)
      for (int i = 0; i < dim; i++)
      {
         tau += Ci*Gij(i,j)*Gij(i,j)*nu*nu;
      }

   return 1.0/sqrt(tau);
}

StabilizedConvectionDiffusion::StabilizedConvectionDiffusion
   (VectorCoefficient &a, Coefficient &k, Coefficient &f)
     :  adv(&a), kappa(&k), force(&f)
{
      tau = new StabilizedParameter(*adv, *kappa, 310);
      adv_tau = new ScalarVectorProductCoefficient(*tau, *adv);
      adv_tau_force = new ScalarVectorProductCoefficient(*force, *adv_tau);
      adv_tau_kappa = new ScalarVectorProductCoefficient(*kappa, *adv_tau);
      adv_tau_adv = new OuterProductCoefficient(*adv_tau, *adv);

      kappa_tau = new ProductCoefficient(*kappa, *tau);
      kappa_tau_kappa = new ProductCoefficient(*kappa_tau, *kappa);
      kappa_tau_force = new ProductCoefficient(*kappa_tau, *force);
}

void StabilizedConvectionDiffusion::SetForms(FiniteElementSpace *fes,
                                             StabType stype)
{
   // Delete old forms
   if (a) delete a;
   if (b) delete b;

   // Create new forms
   a = new BilinearForm(fes);
   b = new LinearForm(fes);

   // Add Galerkin terms
   //a->AddDomainIntegrator(new InverseEstimateIntegrator(*kappa));
   a->AddDomainIntegrator(new ConservativeConvectionIntegrator(*adv));
   a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
   b->AddDomainIntegrator(new DomainLFIntegrator(*force));
   if (stype == GALERKIN) return;

   // Add SUPG terms
   a->AddDomainIntegrator(new DiffusionIntegrator(*adv_tau_adv));
   a->AddDomainIntegrator(new GradLaplaceIntegrator(*adv_tau_kappa,-1.0));
   b->AddDomainIntegrator(new DomainLFGradIntegrator(*adv_tau_force));
   if (stype == SUPG) return;

   // Add VMS/GLS terms
   real_t  s = (stype == GLS)? -1.0: 1.0;
   mfem::out<<s<<std::endl;
   a->AddDomainIntegrator(new LaplaceGradIntegrator(*adv_tau_kappa,s));
   a->AddDomainIntegrator(new LaplaceLaplaceIntegrator(*kappa_tau_kappa,-s));
   b->AddDomainIntegrator(new DomainLFLaplaceIntegrator(*kappa_tau_force,s));
}

void StabilizedConvectionDiffusion::AddWBC(Coefficient &u_dir, real_t &penalty)
{
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*kappa, -1.0, penalty));

   b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_dir, *kappa, -1.0, penalty));
}
