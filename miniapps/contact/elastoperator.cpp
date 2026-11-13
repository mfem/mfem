// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "neo_hookean.hpp"

namespace mfem
{

void ElasticityOperator::Init()
{
   B.UseDevice(true);
   X.UseDevice(true);
   xref.UseDevice(true);
   state_work.UseDevice(true);
   action_work.UseDevice(true);

   int dim = pmesh->Dimension();
   fec = new H1_FECollection(order,dim);
   fes = new ParFiniteElementSpace(pmesh,fec,dim,Ordering::byVDIM);
   globalntdofs = fes->GlobalTrueVSize();
   state_work.SetSize(fes->GetTrueVSize());
   action_work.SetSize(fes->GetTrueVSize());
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
   use_rhs_fa = Device::IsEnabled();
   ResetLinearForm();
   if (nonlinear)
   {
      material = std::make_unique<DeviceNeoHookean>(*fes, c1_cf, c2_cf);
   }
   else
   {
      op = new ParBilinearForm(fes);
      auto *a = op;
      if (use_fa)
      {
         a->SetAssemblyLevel(AssemblyLevel::FULL);
         a->EnableSparseMatrixSorting(Device::IsEnabled());
      }
      a->AddDomainIntegrator(new ElasticityIntegrator(c1_cf,c2_cf));
      a->Assemble();
      a->FormSystemMatrix(ess_tdof_list, K);
   }
}

void ElasticityOperator::ResetLinearForm()
{
   delete b;
   b = new ParLinearForm(fes);
   for (int i = 0; i < pressure_cf.Size(); i++) { delete pressure_cf[i]; }
   for (int i = 0; i < pressure_bdr_marker.Size(); i++) { delete pressure_bdr_marker[i]; }
   pressure_cf.SetSize(0);
   pressure_bdr_marker.SetSize(0);
}

ElasticityOperator::ElasticityOperator(ParMesh * pmesh_,
                                       Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_,
                                       const Vector & E, const Vector & nu,
                                       bool nonlinear_, bool use_fa_)
   : nonlinear(nonlinear_), use_fa(use_fa_), pmesh(pmesh_),
     ess_bdr_attr(ess_bdr_attr_),
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

void ElasticityOperator::SetNeumannPressureData(ConstantCoefficient &f,
                                                Array<int> & bdr_marker)
{
   pressure_cf.Append(new ConstantCoefficient(f.constant));
   pressure_bdr_marker.Append(new Array<int>(bdr_marker));
   b->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(
                               *pressure_cf.Last()),
                            *pressure_bdr_marker.Last());
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
      b->UseFastAssembly(use_rhs_fa);
      b->Assemble();
      if (!nonlinear)
      {
         auto *a = op;
         if (use_fa)
         {
            // Transform only the RHS with the unconstrained operator. The
            // assembled stiffness matrix K is reused across load steps.
            Operator *rhs_operator;
            a->Operator::FormLinearSystem(ess_tdof_list, x, *b,
                                          rhs_operator, X, B);
            delete rhs_operator;
            if (IsIdentityProlongation(fes->GetProlongationMatrix()))
            {
               // FormLinearSystem modifies X, which aliases x in this case.
               x.SyncMemory(X);
            }
         }
         else
         {
            B.SetSize(fes->GetTrueVSize());
            b->ParallelAssemble(B);
            B.SetSubVector(ess_tdof_list, 0.0);
            x.GetTrueDofs(X);
            a->ParallelEliminateTDofsInRHS(ess_tdof_list, X, B);
         }
      }
      else
      {
         B.SetSize(fes->GetTrueVSize());
         b->ParallelAssemble(B);
         B.SetSubVector(ess_tdof_list, 0.0);
      }
   }
}

void ElasticityOperator::UpdateRHS()
{
   formsystem = false;
   ResetLinearForm();
   x = 0.0;
}

real_t ElasticityOperator::GetEnergy(const Vector & u) const
{
   if (nonlinear)
   {
      real_t energy = 0.0;
      Vector &tu = state_work; tu = xref; tu += u;
      material->Evaluate(tu, ess_tdof_list, energy);
      energy -= InnerProduct(comm, B, u);
      return energy;
   }
   else
   {
      Vector &ku = action_work;
      K->Mult(u,ku);
      return 0.5 * InnerProduct(comm,u, ku) - InnerProduct(comm,u, B);
   }
}

void ElasticityOperator::GetGradient(const Vector & u, Vector & gradE) const
{
   if (nonlinear)
   {
      Vector &tu = state_work; tu = xref; tu += u;
      real_t energy;
      material->Evaluate(tu, ess_tdof_list, energy, &gradE);
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
      Vector &tu = state_work; tu = xref; tu += u;
      real_t energy;
      material->Evaluate(tu, ess_tdof_list, energy, nullptr, &K);
   }
   return K.As<HypreParMatrix>();
}

void ElasticityOperator::GetReferenceData(const Vector &u, real_t &energy,
                                          Vector &gradient, HypreParMatrix *&hessian)
{
   if (nonlinear)
   {
      state_work = xref;
      state_work += u;
      material->Evaluate(state_work, ess_tdof_list, energy, &gradient, &K);
      energy -= InnerProduct(comm, B, u);
      gradient.Add(-1.0, B);
      hessian = K.As<HypreParMatrix>();
   }
   else
   {
      energy = GetEnergy(u);
      GetGradient(u, gradient);
      hessian = GetHessian(u);
   }
}

ElasticityOperator::~ElasticityOperator()
{
   K.Clear();
   material.reset();
   delete op;
   delete b;
   for (int i = 0; i < pressure_cf.Size(); i++) { delete pressure_cf[i]; }
   for (int i = 0; i < pressure_bdr_marker.Size(); i++) { delete pressure_bdr_marker[i]; }
   delete fes;
   delete fec;
}

}
