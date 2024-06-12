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

#include "stab_navsto.hpp"

using namespace mfem;

StabInNavStoIntegrator::StabInNavStoIntegrator(Coefficient &mu_,
                                               VectorCoefficient &force_,
                                               Tau &t, Tau &d, StabType s)
 : c_mu(&mu_), c_force(&force_), tau(&t), delta(&d), stab(s)
{ }

void StabInNavStoIntegrator::SetDim(int dim_)
{
  if (dim_ != dim)
  {
     dim = dim_;
     u.SetSize(dim);
     f.SetSize(dim);
     res.SetSize(dim);
     up.SetSize(dim);
     grad_u.SetSize(dim);
     hess_u.SetSize(dim, (dim*(dim+1))/2);
     grad_p.SetSize(dim);
     hmap.SetSize(dim,dim);

     if (dim == 2)
     {
        hmap(0,0) = 0;
        hmap(0,1) =  hmap(1,0) =  1;
        hmap(1,1) = 2; 
     }
     else if (dim == 2)
     {
        hmap(0,0) = 0;
        hmap(0,1) = hmap(1,0) = 1;
        hmap(0,2) = hmap(2,0) = 2;
        hmap(1,1) = 3;
        hmap(1,2) = hmap(2,1) = 4;
        hmap(2,2) = 5;
     }
     else
     {
        mfem_error("Only implemented for 2D and 3D");
     }
  }
}

real_t StabInNavStoIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   if (el.Size() != 2)
   {
      mfem_error("StabInNavStoIntegrator::GetElementEnergy"
                 " has incorrect block finite element space size!");
   }
   SetDim(el[0]->GetDim());
   int dof_u = el[0]->GetDof();

   sh_u.SetSize(dof_u);
   elf_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

   int intorder = 2*el[0]->GetOrder();
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   real_t energy = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);

      real_t w = ip.weight * Tr.Weight();

      el[0]->CalcPhysShape(Tr, sh_u);
      elf_u.MultTranspose(sh_u, u);

      energy += w*(u*u)/2;
   }

   return energy;
}

void StabInNavStoIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el.Size() != 2)
   {
      mfem_error("StabInNavStoIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }

   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   SetDim(el[0]->GetDim());
   int spaceDim = Tr.GetSpaceDim();
   bool hess = (el[0]->GetDerivType() == (int) FiniteElement::HESS);
   if (dim != spaceDim)
   {
      mfem_error("StabInNavStoIntegrator::AssembleElementVector"
                 " is not defined on manifold meshes");
   }
   elvec[0]->SetSize(dof_u*dim);
   elvec[1]->SetSize(dof_p);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;

   elf_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   elv_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);

   sh_u.SetSize(dof_u);
   shg_u.SetSize(dof_u, dim);
   ushg_u.SetSize(dof_u);
   shh_u.SetSize(dof_u, (dim*(dim+1))/2);
   sh_p.SetSize(dof_p);
   shg_p.SetSize(dof_p, dim);

   int intorder = 2*el[0]->GetOrder();
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      real_t w = ip.weight * Tr.Weight();
      real_t mu = c_mu->Eval(Tr, ip);
      c_force->Eval(f, Tr, ip);

      // Compute shape and interpolate
      el[0]->CalcPhysShape(Tr, sh_u);
      elf_u.MultTranspose(sh_u, u);

      el[0]->CalcPhysDShape(Tr, shg_u);
      shg_u.Mult(u, ushg_u);
      MultAtB(elf_u, shg_u, grad_u);

      if (hess)
      {
         el[0]->CalcPhysHessian(Tr,shh_u);
         MultAtB(elf_u, shh_u, hess_u);
      }
      else
      {
         shh_u = 0.0;
         hess_u = 0.0;
      }

      el[1]->CalcPhysShape(Tr, sh_p);
      real_t p = sh_p*(*elfun[1]);

      el[1]->CalcPhysDShape(Tr, shg_p);
      shg_p.MultTranspose(*elfun[1], grad_p);

      // Compute strong residual
      grad_u.Mult(u,res);   // Add convection
      res += grad_p;        // Add pressure
      res -= f;             // Subtract force
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            res[j] -= mu*(hess_u(j,hmap(i,i)) + 
                          hess_u(i,hmap(j,i))); // Add diffusion
         }
      }

      // Compute stability params
      real_t t = tau->Eval(Tr, ip);
      real_t d = delta->Eval(Tr, ip);

      // Compute momentum weak residual
      flux.Diag(-p + d*grad_u.Trace(),dim);  // Add pressure & LSIC to flux
      grad_u.Symmetrize();                   // Grad to strain
      flux.Add(2*mu,grad_u);                 // Add stress to flux
      AddMult_a_VVt(-1.0, u, flux);          // Add convection to flux
      AddMult_a_VWt(t, res, u, flux);        // Add SUPG to flux --> check order u and res
      AddMult_a_ABt(w, shg_u, flux, elv_u);  // Add flux term to rhs
      AddMult_a_VWt(-w, sh_u, f,    elv_u);  // Add force term to rhs

      // Compute momentum weak residual
      elvec[1]->Add(w*grad_u.Trace(), sh_p); // Add Galerkin term
      shg_p.Mult(res, sh_p);                 // PSPG help term
      elvec[1]->Add(w*t, sh_p);              // Add PSPG term  - sign looks worng?
   }
}

void StabInNavStoIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   SetDim(el[0]->GetDim());
   bool hess = (el[0]->GetDerivType() == (int) FiniteElement::HESS);

   elf_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

   elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
   elmats(0,1)->SetSize(dof_u*dim, dof_p);
   elmats(1,0)->SetSize(dof_p, dof_u*dim);
   elmats(1,1)->SetSize(dof_p, dof_p);

   *elmats(0,0) = 0.0;
   *elmats(0,1) = 0.0;
   *elmats(1,0) = 0.0;
   *elmats(1,1) = 0.0;

   sh_u.SetSize(dof_u);
   shg_u.SetSize(dof_u, dim);
   ushg_u.SetSize(dof_u);
   sh_p.SetSize(dof_p);
   shg_p.SetSize(dof_p, dim);

   int intorder = 2*el[0]->GetOrder();
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      real_t w = ip.weight * Tr.Weight();
      real_t mu = c_mu->Eval(Tr, ip);
      real_t t = tau->Eval(Tr, ip);
      real_t d = delta->Eval(Tr, ip);

      el[0]->CalcPhysShape(Tr, sh_u);
      elf_u.MultTranspose(sh_u, u);

      el[0]->CalcPhysDShape(Tr, shg_u);
      MultAtB(elf_u, shg_u, grad_u);

      shg_u.Mult(u, ushg_u);

      el[1]->CalcPhysShape(Tr, sh_p);
      real_t p = sh_p*(*elfun[1]);

      el[1]->CalcPhysDShape(Tr, shg_p);
      shg_p.MultTranspose(*elfun[1], grad_p);

      // u,u block
      for (int i_u = 0; i_u < dof_u; ++i_u)
      {
         for (int j_u = 0; j_u < dof_u; ++j_u)
         {
            // Diffusion
            real_t mat = 0.0;
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
              mat += shg_u(i_u,dim_u)*shg_u(j_u,dim_u);
            }
            mat *= mu;

            // Convection
            mat -= ushg_u(i_u)*sh_u(j_u);      // Galerkin
            mat += t*ushg_u(i_u)*ushg_u(j_u);  // SUPG

            mat *= w;
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
               (*elmats(0,0))(i_u + dim_u*dof_u, j_u + dim_u*dof_u) += mat;
            }

            for (int i_dim = 0; i_dim < dim; ++i_dim)
            {
               for (int j_dim = 0; j_dim < dim; ++j_dim)
               {
                  (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                     (mu + d)*shg_u(i_u,j_dim)*shg_u(j_u,i_dim)*w;
               }
            }
         }
      }

      // u,p and p,u blocks
      for (int i_p = 0; i_p < dof_p; ++i_p)
      {
         for (int j_u = 0; j_u < dof_u; ++j_u)
         {
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
               (*elmats(0,1))(j_u + dof_u * dim_u, i_p) += (shg_p(i_p, dim_u)*t*ushg_u(j_u)
                                                           -shg_u(j_u,dim_u)*sh_p(i_p))*w;
               (*elmats(1,0))(i_p, j_u + dof_u * dim_u) +=  shg_u(j_u,dim_u)*sh_p(i_p)*w;
            }
         }
      }

      // p,p block
      AddMult_a_AAt(w*t, shg_p, *elmats(1,1));
   }
}


void GeneralResidualMonitor::MonitorResidual(int it, real_t norm,
                                             const Vector &r, bool final)
{
   if (it == 0)
   {
      norm0 = norm;
   }

   if ((print_level > 0 &&  it%print_level == 0) || final)
   {
      mfem::out << prefix << " iteration " << std::setw(2) << it
                << " : ||r|| = " << norm
                << ",  ||r||/||r_0|| = " << 100*norm/norm0<<" % \n";
   }
}

void SystemResidualMonitor::MonitorResidual(int it, real_t norm,
                                            const Vector &r, bool final)
{
   if (dc && (it > 0))
   {
      if (rank > 1)
      {
         for (int i = 0; i < nvar; ++i)
         {
            pgf[i]->Distribute(xp->GetBlock(i));
         }
      }
      dc->SetCycle(it);
      dc->Save();
   }

   Vector vnorm(nvar);

   for (int i = 0; i < nvar; ++i)
   {
       Vector r_i(r.GetData() + bOffsets[i], bOffsets[i+1] - bOffsets[i]);
       if ( rank == 1 )
       {
          vnorm[i] = r_i.Norml2();
       }
       else
       {
          vnorm[i] = sqrt(InnerProduct(MPI_COMM_WORLD, r_i, r_i));
       }
       if (it == 0) norm0[i] = vnorm[i];
   }

   bool print = (print_level > 0 &&  it%print_level == 0) || final;
   if (print)
   {
      mfem::out << prefix << " iteration " << std::setw(3) << it <<"\n"
                << " ||r||  \t"<< "||r||/||r_0||  \n";
      for (int i = 0; i < nvar; ++i)
      {
         mfem::out <<vnorm[i]<<"\t"<< 100*vnorm[i]/norm0[i]<<" % \n";
      }
   }
}


void JacobianPreconditioner::SetOperator(const Operator &op)
{
   BlockOperator *jacobian = (BlockOperator *) &op;

   for (int i = 0; i < prec.Size(); ++i)
   {
      prec[i]->SetOperator(jacobian->GetBlock(i,i));
      SetDiagonalBlock(i, prec[i]);
   }

   SetBlock(1,0, const_cast<Operator*>(&jacobian->GetBlock(1,0)));
}

JacobianPreconditioner::~JacobianPreconditioner()
{
   for (int i = 0; i < prec.Size(); ++i)
   {
      delete prec[i];
   }
}

