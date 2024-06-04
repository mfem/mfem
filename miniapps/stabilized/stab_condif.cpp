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

real_t FFH92Tau::GetElementSize(ElementTransformation &T)
{
   const DenseMatrix &dxdxi = T.Jacobian();
   row.SetSize(dim);
   h.SetSize(dim);

   for (int i = 0; i < dim; i++)
   {
      dxdxi.GetRow(i, row);
      h[i] = row.Norml2();
   }
   switch (dim)
   {
      case 1:
         return h[0];
      case 2:
         return h[0]*h[1]*sqrt(2.0/(h[0]*h[0] + h[1]*h[1]));
      case 3:
         return h[0]*h[1]*h[2]*(3.0/(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]));
   }
   mfem_error("Wrong dim!");
   return -1.0;
}

real_t FFH92Tau::GetInverseEstimate(ElementTransformation &T,
                                    const IntegrationPoint &ip, real_t scale)
{
   if (Ci>0.0)
   {
      return Ci;
   }
   else
   {
      return iecf->Eval(T,ip)*scale;
   }
}

real_t FFH92Tau::Eval(ElementTransformation &T,
                      const IntegrationPoint &ip)
{
   real_t k = kappa->Eval(T, ip);
   adv->Eval(a, T, ip);
   real_t hk = GetElementSize(T);
   real_t ci = GetInverseEstimate(T, ip, hk*hk/k);
   real_t mk = std::min(1.0/3.0, 2/ci);
   real_t ap = a.Normlp(p);
   real_t pe = mk*ap*hk/(2*k);
   real_t xi = std::min(pe,1.0);
   real_t tau = hk*xi/(2*ap);

   if (print)
   {
      std::cout<<"  kappa = "<<k  <<std::endl;
      std::cout<<"  adv   = "; a.Print(std::cout);
      std::cout<<"  h     = "<<hk <<std::endl;
      std::cout<<"  Ci    = "<<ci<<" "
               <<( (Ci<0) ? "(Computed)" :"(Specified)")<<std::endl;
      std::cout<<"  mk    = "<<mk <<std::endl;
      std::cout<<"  |a|_p = "<<ap <<std::endl;
      std::cout<<"  Pe    = "<<pe <<std::endl;
      std::cout<<"  xi    = "<<xi <<std::endl;
      std::cout<<"  tau   = "<<tau<<std::endl;
      print = false;
   }

   return tau;
}


StabConDifIntegrator::StabConDifIntegrator(VectorCoefficient *a,
                                           Coefficient *k,
                                           Coefficient *f,
                                           Tau *t, StabType s)
 : adv(a), kappa(k), force(f), tau(t), stab(s), own_tau(false)
{
   if (tau == nullptr)
   {
      tau = new FFH92Tau(adv, kappa, nullptr, 12.0);
      own_tau = true;
   }
   else
   {
      tau->SetConvection(adv);
      tau->SetDiffusion(kappa);
   }
}

StabConDifIntegrator::~StabConDifIntegrator()
{
   if (own_tau) { delete tau; }
}

const IntegrationRule &StabConDifIntegrator::GetRule(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans)
{
   int order = trial_fe.GetOrder() + test_fe.GetOrder();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

void StabConDifIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                 ElementTransformation &Trans,
                                                 DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   real_t w,k,t = 0;
   Vector a(dim);

   elmat.SetSize(nd);
   shape.SetSize(nd);
   dshape.SetSize(nd,dim);
   adshape.SetSize(nd);
   laplace.SetSize(nd);
   trail.SetSize(nd);
   test.SetSize(nd);

   const IntegrationRule *ir = NonlinearFormIntegrator::IntRule ? NonlinearFormIntegrator::IntRule : &GetRule(el, el, Trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;

      // Calculate shapes
      el.CalcPhysShape(Trans, shape);
      el.CalcPhysDShape(Trans, dshape);

      // Evaluate coefficients
      k = kappa->Eval(Trans, ip);
      adv->Eval(a, Trans, ip);

      // Galerkin convection term
      dshape.Mult(a, adshape);
      AddMult_a_VWt(w, shape, adshape, elmat);

      // Galerkin diffusion term
      AddMult_a_AAt(w*k, dshape, elmat);

      if (stab != GALERKIN)
      {
         // Calculate shapes
         el.CalcPhysLaplacian(Trans, laplace);

         // Evaluate coefficients
         t = tau->Eval(Trans, ip);

         // Stablization term
         // - GLS:  stab = -1
         // - SUPG: stab =  0
         // - VMS:  stab = +1
         add(adshape, stab*k, laplace, test);
         add(adshape, -k, laplace, trail);
         AddMult_a_VWt(w*t, test, trail, elmat);
      }
   }
}

void StabConDifIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      Vector &elvect)
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   real_t w,k,t,f;
   Vector a(dim);

   elvect.SetSize(nd);
   shape.SetSize(nd);
   dshape.SetSize(nd,dim);
   adshape.SetSize(nd);
   laplace.SetSize(nd);
   test.SetSize(nd);

   const IntegrationRule *ir = LinearFormIntegrator::IntRule ? LinearFormIntegrator::IntRule : &GetRule(el, el, Trans);

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;

      // Calculate shapes
      el.CalcPhysShape(Trans, shape);

      // Evaluate coefficients
      f = force->Eval(Trans, ip);

      // Galerkin term
      elvect.Add(w*f, shape);

      if (stab != GALERKIN)
      {
         // Calculate shapes
         el.CalcPhysDShape(Trans, dshape);
         el.CalcPhysLaplacian(Trans, laplace);

         // Evaluate coefficients
         k = kappa->Eval(Trans, ip);
         adv->Eval(a, Trans, ip);
         t = tau->Eval(Trans, ip);

         // Advective derivative
         dshape.Mult(a, adshape);

         // Stablization term
         // - GLS:  stab = -1
         // - SUPG: stab =  0
         // - VMS:  stab = +1
         add(adshape, stab*k, laplace, test);
         elvect.Add(w*f*t, test);
      }
   }
}


StabConDifComposition::StabConDifComposition(VectorCoefficient *a,
                                             Coefficient *k,
                                             Coefficient *f,
                                             Tau *t)
  :  adv(a), kappa(k), force(f), tau(t), own_tau(false)
{
   if (tau == nullptr)
   {
      tau = new FFH92Tau(adv, kappa, nullptr, 12.0);
      own_tau = true;
   }
   else
   {
      tau->SetConvection(adv);
      tau->SetDiffusion(kappa);
   }

   // SUPG coefficients
   adv_tau = new ScalarVectorProductCoefficient(*tau, *adv);
   adv_tau_force = new ScalarVectorProductCoefficient(*force, *adv_tau);
   adv_tau_kappa = new ScalarVectorProductCoefficient(*kappa, *adv_tau);
   adv_tau_adv = new OuterProductCoefficient(*adv_tau, *adv);

   // GLS/VMS coefficients
   kappa_tau = new ProductCoefficient(*kappa, *tau);
   kappa_tau_kappa = new ProductCoefficient(*kappa_tau, *kappa);
   kappa_tau_force = new ProductCoefficient(*kappa_tau, *force);
}

StabConDifComposition::~StabConDifComposition()
{
   if (own_tau) { delete tau; }
   delete adv_tau, adv_tau_kappa, adv_tau_adv, adv_tau_force,
          kappa_tau, kappa_tau_kappa,kappa_tau_force;
}

void StabConDifComposition::SetBilinearIntegrators(BilinearForm *a, StabType stype)
{
   a->AddDomainIntegrator(new ConservativeConvectionIntegrator(*adv));
   a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
   if (stype == GALERKIN) return;

   // Add SUPG terms
   a->AddDomainIntegrator(new DiffusionIntegrator(*adv_tau_adv));
   a->AddDomainIntegrator(new GradLaplaceIntegrator(*adv_tau_kappa, 1.0));
   if (stype == SUPG) return;

   // Add VMS/GLS terms
   real_t  s = (stype == GLS)? -1.0: 1.0;
   a->AddDomainIntegrator(new LaplaceGradIntegrator(*adv_tau_kappa,-s));
   a->AddDomainIntegrator(new LaplaceLaplaceIntegrator(*kappa_tau_kappa,-s));
}

void StabConDifComposition::SetLinearIntegrators(LinearForm *b, StabType stype)
{
   b->AddDomainIntegrator(new DomainLFIntegrator(*force));
   if (stype == GALERKIN) return;

   // Add SUPG terms
   b->AddDomainIntegrator(new DomainLFGradIntegrator(*adv_tau_force));
   if (stype == SUPG) return;

   // Add VMS/GLS terms
   real_t  s = (stype == GLS)? -1.0: 1.0;
   b->AddDomainIntegrator(new DomainLFLaplaceIntegrator(*kappa_tau_force,-s));
}


