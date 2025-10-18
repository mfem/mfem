// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "elastoperator.hpp"

namespace mfem
{

void ElasticityOperator::Init()
{
   int dim = pmesh->Dimension();
   fec = new H1_FECollection(order,dim);
   fes = new ParFiniteElementSpace(pmesh,fec,dim,Ordering::byVDIM);
   globalntdofs = fes->GlobalTrueVSize();
   pmesh->SetNodalFESpace(fes);

   auto ref_func = [](const Vector & x, Vector & y) { y = x; };
   VectorFunctionCoefficient ref_cf(dim,ref_func);
   ParGridFunction xr(fes); xr.ProjectCoefficient(ref_cf);
   xr.GetTrueDofs(xref);
   SetEssentialBC();
   SetUpOperator();
}

void ElasticityOperator::SetEssentialBC()
{
   ess_tdof_list.SetSize(0);
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
   }
   ess_bdr = 0;
   Array<int> ess_tdof_list_temp;
   for (int i = 0; i < ess_bdr_attr.Size(); i++ )
   {
      ess_bdr[ess_bdr_attr[i]-1] = 1;
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list_temp,ess_bdr_attr_comp[i]);
      ess_tdof_list.Append(ess_tdof_list_temp);
      ess_bdr[ess_bdr_attr[i]-1] = 0;
   }
}

void ElasticityOperator::SetUpOperator()
{
   x.SetSpace(fes);  x = 0.0;
   b = new ParLinearForm(fes);
   if (nonlinear)
   {
      material_model = new NeoHookeanModel(c1_cf, c2_cf);
      op = new ParNonlinearForm(fes);
      dynamic_cast<ParNonlinearForm*>(op)->AddDomainIntegrator(
         new HyperelasticNLFIntegrator(material_model));
      dynamic_cast<ParNonlinearForm*>(op)->SetEssentialTrueDofs(ess_tdof_list);
   }
   else
   {
      op = new ParBilinearForm(fes);
      dynamic_cast<ParBilinearForm*>(op)->AddDomainIntegrator(
         new ElasticityIntegrator(c1_cf,c2_cf));
      K = new HypreParMatrix();
      dynamic_cast<ParBilinearForm*>(op)->Assemble();
      dynamic_cast<ParBilinearForm*>(op)->FormSystemMatrix(ess_tdof_list,*K);
   }
}

ElasticityOperator::ElasticityOperator(ParMesh * pmesh_,
                                       Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_,
                                       const Vector & E, const Vector & nu, bool nonlinear_)
   : nonlinear(nonlinear_), pmesh(pmesh_), ess_bdr_attr(ess_bdr_attr_),
     ess_bdr_attr_comp(ess_bdr_attr_comp_)
{
   comm = pmesh->GetComm();
   SetParameters(E,nu);
   Init();
}

void ElasticityOperator::SetParameters(const Vector & E, const Vector & nu)
{
   int n = (pmesh->attributes.Size()) ?  pmesh->attributes.Max() : 0;
   MFEM_VERIFY(E.Size() == n, "Incorrect parameter size E");
   MFEM_VERIFY(nu.Size() == n, "Incorrect parameter size nu");
   c1.SetSize(n);
   c2.SetSize(n);
   if (nonlinear)
   {
      for (int i = 0; i<n; i++)
      {
         c1(i) = 0.5*E(i) / (1+nu(i));
         c2(i) = E(i)/(1.0-2.0*nu(i))/3.0;
      }
   }
   else
   {
      for (int i = 0; i<n; i++)
      {
         c1(i) = E(i) * nu(i) / ( (1+nu(i)) * (1-2*nu(i)) );
         c2(i) = 0.5 * E(i)/(1+nu(i));
      }
   }
   c1_cf.UpdateConstants(c1);
   c2_cf.UpdateConstants(c2);
}

void ElasticityOperator::SetNeumanPressureData(ConstantCoefficient &f,
                                               Array<int> & bdr_marker)
{
   pressure_cf.constant = f.constant;
   b->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(pressure_cf),
                            bdr_marker);
}

void ElasticityOperator::SetDisplacementDirichletData(const Vector & delta,
                                                      Array<int> essbdr)
{
   VectorConstantCoefficient delta_cf(delta);
   x.ProjectBdrCoefficient(delta_cf,essbdr);
}

void ElasticityOperator::FormLinearSystem()
{
   if (!formsystem)
   {
      formsystem = true;
      b->Assemble();
      B.SetSize(fes->GetTrueVSize());
      b->ParallelAssemble(B);
      B.SetSubVector(ess_tdof_list, 0.0);
      if (!nonlinear)
      {
         x.GetTrueDofs(X);
         dynamic_cast<ParBilinearForm*>(op)->EliminateVDofsInRHS(ess_tdof_list, X, B);
      }
   }
}

void ElasticityOperator::UpdateRHS()
{
   formsystem = false;
   delete b;
   b = new ParLinearForm(fes);
   x = 0.0;
}

real_t ElasticityOperator::GetEnergy(const Vector & u) const
{
   if (nonlinear)
   {
      real_t energy = 0.0;
      Vector tu(xref); tu += u;
      ParGridFunction u_gf(fes);
      u_gf.SetFromTrueDofs(tu);
      energy += dynamic_cast<ParNonlinearForm*>(op)->GetEnergy(u_gf);
      energy -= InnerProduct(comm, B, u);
      return energy;
   }
   else
   {
      Vector ku(K->Height());
      K->Mult(u,ku);
      return 0.5 * InnerProduct(comm,u, ku) - InnerProduct(comm,u, B);
   }
}

void ElasticityOperator::GetGradient(const Vector & u, Vector & gradE) const
{
   if (nonlinear)
   {
      Vector tu(xref); tu += u;
      gradE.SetSize(op->Height());
      dynamic_cast<ParNonlinearForm*>(op)->Mult(tu, gradE);
   }
   else
   {
      gradE.SetSize(K->Height());
      K->Mult(u, gradE);
   }
   gradE.Add(-1.0, B);
}

HypreParMatrix * ElasticityOperator::GetHessian(const Vector & u)
{
   if (nonlinear)
   {
      Vector tu(xref); tu += u;
      return dynamic_cast<HypreParMatrix *>(&dynamic_cast<ParNonlinearForm*>
                                            (op)->GetGradient(tu));
   }
   else
   {
      return K;
   }
}

ElasticityOperator::~ElasticityOperator()
{
   delete op;
   delete b;
   delete fes;
   delete fec;
   if (K) { delete K; }
   if (material_model) { delete material_model; }
}

}
