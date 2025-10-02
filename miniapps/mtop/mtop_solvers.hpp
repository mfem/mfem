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
#pragma once

#include <memory>

#include "mfem.hpp"

using real_t = mfem::real_t;

class StokesSolver:public mfem::Operator
{
public:
   StokesSolver(mfem::ParMesh* mesh, int order_, bool zero_mean_press_=false);

   virtual
   ~StokesSolver();

   /// Set the Linear Solver
   void SetLinearSolver(const real_t rtol = 1e-8,
                        const real_t atol = 1e-12,
                        const int miter = 200)
   {
      linear_atol=atol;
      linear_rtol=rtol;
      linear_iter=miter;
   }

   /// Sets BC dofs, bilinear form, preconditioner and solver.
   /// Should be called before calling Mult of MultTranspose
   virtual void Assemble();

   /// Sets Brinkman coefficient
   void SetBrink(std::shared_ptr<mfem::Coefficient> br_){
       brink=br_;
   }

   /// Sets viscosity
   void SetVisc(std::shared_ptr<mfem::Coefficient> vs_){
       visc=vs_;
   }

   /// Solves the forward problem.
   void FSolve();

   /// Solves the adjoint with the provided rhs.
   void ASolve(mfem::Vector &rhs);

   /// Clear all  BC
   void DeleteBC();

   /// Set the values of the volumetric force.
   void SetVolForce(real_t fx, real_t fy, real_t fz = 0.0);

   //Set zero mean pressure BC
   void SetZeroMeanPressure(bool val_=true)
   {
      zero_mean_press=val_;
   }

   //velocity boundary conditions
   void AddVelocityBC(int id, std::shared_ptr<mfem::VectorCoefficient> val)
   {
      vel_bcs[id]=val;
   }

   /// Set the Velocity BC on a given ParGridFunction.
   void SetEssVBC(mfem::ParGridFunction& pgf);

   /// Extracts the true boundary doffs of the velocity
   void SetEssTDofsV(mfem::Array<int>& ess_dofs);

   /// Set the Velocity BC on a given true vector.
   void SetEssTDofsV(mfem::Vector& v) const;

   /// Forward solve with given RHS. x is the RHS vector.
   /// The BC are set in the Mult operation.
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Adjoint solve with given RHS. x is the RHS vector.
   /// The BC are set to zero in the MultTranspose operator.
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Return velocity grid function
   mfem::ParGridFunction& GetVelocity()
   {
      vel.SetFromTrueDofs(sol.GetBlock(0));
      return vel;
   }

   /// Return pressure grid function
   mfem::ParGridFunction& GetPressure()
   {
      pre.SetFromTrueDofs(sol.GetBlock(1));
      return pre;
   }

   /// Return velocity grid function
   mfem::ParGridFunction& GetAdjVelocity()
   {
      avel.SetFromTrueDofs(adj.GetBlock(0));
      return avel;
   }

   /// Return pressure grid function
   mfem::ParGridFunction& GetAdjPressure()
   {
      apre.SetFromTrueDofs(adj.GetBlock(1));
      return apre;
   }

   /// Return velocty FES
   mfem::ParFiniteElementSpace* GetVelocitySpace() {return vfes;}

   /// Return pressure FES
   mfem::ParFiniteElementSpace* GetPressureSpace() {return pfes;}

   /// Return the block offset for providing RHS vectors for
   /// Mult and MultTranspose.
   mfem::Array<int>& GetTrueBlockOffsets() {return block_true_offsets;}


private:

   int myrank;

   bool zero_mean_press;

   /// The parallel mesh.
   mfem::ParMesh *pmesh = nullptr;

   /// The order of the velocity  space.
   int order;

   /// linear system solvers parameters
   real_t linear_atol;
   real_t linear_rtol;
   int  linear_iter;
   int dim;

   std::shared_ptr<mfem::Coefficient> visc; //viscosity
   std::shared_ptr<mfem::Coefficient> brink; //Brinkman penalization

   mfem::H1_FECollection* vfec; //velocity collections
   mfem::FiniteElementCollection* pfec; //pressure collecation
   mfem::ParFiniteElementSpace* vfes;
   mfem::ParFiniteElementSpace* pfes;

   std::unique_ptr<mfem::IterativeSolver> ls;
   std::unique_ptr<mfem::IterativeSolver> prec;

   //boundary conditions
   std::map<int, std::shared_ptr<mfem::VectorCoefficient>> vel_bcs;

   // holds the velocity constrained DOFs
   mfem::Array<int> ess_tdofv;

   // Volume force coefficient
   std::shared_ptr<mfem::VectorCoefficient> vol_force;

   mfem::Array<int> block_true_offsets;
   int siz_u;
   int siz_p;

   mfem::ConstantCoefficient onecoeff;
   mfem::ConstantCoefficient zerocoef;

   mutable mfem::ParGridFunction vel; //velocity
   mutable mfem::ParGridFunction pre; //pressure

   mfem::ParGridFunction avel; //velocity
   mfem::ParGridFunction apre; //pressure

   std::unique_ptr<mfem::HypreParMatrix> A11;
   std::unique_ptr<mfem::HypreParMatrix> A12;
   std::unique_ptr<mfem::HypreParMatrix> A21;

   std::unique_ptr<mfem::HypreParMatrix> A11e;
   std::unique_ptr<mfem::HypreParMatrix> A12e;
   std::unique_ptr<mfem::HypreParMatrix> A21e;

   std::unique_ptr<mfem::ParBilinearForm> bf11;
   std::unique_ptr<mfem::ParMixedBilinearForm> bf12;
   std::unique_ptr<mfem::ParMixedBilinearForm> bf21;

   std::unique_ptr<mfem::BlockOperator> bop;
   mutable mfem::BlockVector rhs;
   mutable mfem::BlockVector sol;
   mutable mfem::BlockVector adj;

   mfem::Vector V; //used for removing the mean pressure
};

class LSCStokesPrec:public mfem::IterativeSolver
{
public:
   LSCStokesPrec(mfem::ParFiniteElementSpace* vfes_,
                 mfem::ParFiniteElementSpace* pfes_,
                 std::shared_ptr<mfem::Coefficient> visc_,
                 std::shared_ptr<mfem::Coefficient> brink_,
                 mfem::Array<int>& ess_vdofs_,
                 const mfem::Operator* O11_,
                 const mfem::Operator* O12_,
                 const mfem::Operator* O21_,
                 bool   zero_mean_press_=false)
      :O11(O11_),O12(O12_),O21(O21_),zero_mean_press(zero_mean_press_)

   {
      // set the preconditioner for the upper block
      mfem::ConstantCoefficient lambda(0.00);
      std::unique_ptr<mfem::ParLORDiscretization>
      lor_discr(new mfem::ParLORDiscretization(*vfes_));
      mfem::ParFiniteElementSpace& vlor=lor_discr->GetParFESpace();
      std::unique_ptr<mfem::ParBilinearForm>
      b11(new mfem::ParBilinearForm(&vlor));
      b11->AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda,*visc_));
      //b11->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*visc_));
      if (nullptr!=brink_.get())
      {
         b11->AddDomainIntegrator(new mfem::VectorMassIntegrator(*brink_));
      }

      b11->Assemble(0);
      b11->Finalize(0);
      A11.reset(b11->ParallelAssemble());
      std::unique_ptr<mfem::HypreParMatrix> Ae(A11->EliminateRowsCols(ess_vdofs_));

      std::cout<<"A11 assembled"<<std::endl;

      amg11.reset(new mfem::HypreBoomerAMG());
      if (mfem::Ordering::Type::byNODES==vfes_->GetOrdering())
      {
         int dim=vlor.GetParMesh()->Dimension();
         amg11->SetSystemsOptions(dim,true);
      }
      amg11->SetOperator(*A11);

      cg11.reset(new mfem::CGSolver(vlor.GetComm()));
      cg11->SetOperator(*O11);
      cg11->SetPreconditioner(*amg11);
      cg11->SetMaxIter(10);
      cg11->SetRelTol(1e-12);
      cg11->SetAbsTol(1e-12);
      cg11->SetPrintLevel(0);

      //assemble diagonal mass matrix on the velocity space
      {
         std::unique_ptr<mfem::ParBilinearForm>
         q11(new mfem::ParBilinearForm(vfes_));
         q11->AddDomainIntegrator(
            new mfem::LumpedIntegrator(
               //new mfem::VectorMassIntegrator(*brink_)));
               new mfem::VectorMassIntegrator()));
         q11->Assemble(0);
         q11->Finalize(0);
         Qv.reset(q11->ParallelAssemble());
         amgv.reset(new mfem::HypreBoomerAMG());
         amgv->SetOperator(*Qv);
      }

      const mfem::HypreParMatrix* m21=dynamic_cast<const mfem::HypreParMatrix*>(O21);
      const mfem::HypreParMatrix* m12=dynamic_cast<const mfem::HypreParMatrix*>(O12);

      if ((nullptr!=m12)&&(nullptr!=m21))
      {

         mfem::HypreParVector Sd(vfes_->GetComm(),
                                 Qv->GetGlobalNumRows(),
                                 Qv->GetRowStarts());
         Qv->GetDiag(Sd);

         mfem::HypreParMatrix T(*m12);
         T.InvScaleRows(Sd);
         A.reset(ParMult(m21,&T));
      }
      else
      {
         //Construct the discrete approximations for O12 and O21
         std::unique_ptr<mfem::HypreParMatrix> A12, A21;
         std::unique_ptr<mfem::ParMixedBilinearForm>
         bf12(new mfem::ParMixedBilinearForm(pfes_, vfes_));
         bf12->AddDomainIntegrator(
            new mfem::TransposeIntegrator(
               new mfem::VectorDivergenceIntegrator()));
         bf12->Assemble(0);
         bf12->Finalize(0);
         A12.reset(bf12->ParallelAssemble());
         A12->EliminateRows(ess_vdofs_);

         std::unique_ptr<mfem::ParMixedBilinearForm>
         bf21(new mfem::ParMixedBilinearForm(vfes_, pfes_));
         bf21->AddDomainIntegrator(
            new mfem::VectorDivergenceIntegrator());
         bf21->Assemble(0);
         bf21->Finalize(0);
         A21.reset(bf21->ParallelAssemble());
         std::unique_ptr<mfem::HypreParMatrix> A21e(A21->EliminateCols(ess_vdofs_));

         mfem::HypreParVector Sd(vfes_->GetComm(),
                                 Qv->GetGlobalNumRows(),
                                 Qv->GetRowStarts());
         Qv->GetDiag(Sd);

         A12->InvScaleRows(Sd);
         A.reset(ParMult(A21.get(),A12.get()));
      }

      amga.reset(new mfem::HypreBoomerAMG());
      amga->SetOperator(*A);

      if (zero_mean_press)
      {
         //use GMRES
         cga.reset(new mfem::GMRESSolver(vfes_->GetComm()));
         cga->SetOperator(*A);
      }
      else
      {
         //use CG
         cga.reset(new mfem::CGSolver(vfes_->GetComm()));
         cga->SetOperator(*A);
      }

      cga->SetPreconditioner(*amga);
      cga->SetPrintLevel(0);
      cga->SetMaxIter(20);
      cga->SetRelTol(1e-12);
      cga->SetAbsTol(1e-12);

      siz_u=O11->NumRows();
      siz_p=O21->NumRows();

      block_true_offsets.SetSize(3);
      block_true_offsets[0] = 0;
      block_true_offsets[1] = siz_u;
      block_true_offsets[2] = siz_p;
      block_true_offsets.PartialSum();
      //set the width and the height of the operator
      this->width=  block_true_offsets[2];
      this->height= block_true_offsets[2];

      v1.SetSize(siz_p); v1=0.0;
      v2.SetSize(siz_u); v2=0.0;
      v3.SetSize(siz_u); v3=0.0;
      v4.SetSize(siz_p); v4=0.0;

      myrank=vfes_->GetMyRank();


   }

   /// Operator application
   virtual
   void Mult (const mfem::Vector & x, mfem::Vector & y) const override
   {
      mfem::BlockVector xb,yb;

      cga->SetMaxIter(mfem::IterativeSolver::max_iter);
      cga->SetAbsTol(IterativeSolver::abs_tol);
      cga->SetRelTol(IterativeSolver::rel_tol);
      //cga->iterative_mode=true;

      cg11->SetMaxIter(mfem::IterativeSolver::max_iter);
      cg11->SetAbsTol(IterativeSolver::abs_tol);
      cg11->SetRelTol(IterativeSolver::rel_tol);
      //cg11->iterative_mode=true;


      xb.Update(const_cast<mfem::Vector&>(x), block_true_offsets);
      yb.Update(y, block_true_offsets);

      if(0==myrank){std::cout<<"Schur complement solve";}
      cga->Mult(xb.GetBlock(1),v1);
      O12->Mult(v1,v2);
      amgv->Mult(v2,v3);
      O11->Mult(v3,v2);
      amgv->Mult(v2,v3);
      O21->Mult(v3,v4);
      cga->Mult(v4,yb.GetBlock(1));
      yb.GetBlock(1).Neg();

      //construct modification of the rhs for block 0
      O12->Mult(yb.GetBlock(1),v2);
      add(xb.GetBlock(0), -1, v2, v3);

      //multiply the upper block
      if(0==myrank){std::cout<<"Upper block solve";}
      cg11->Mult(v3,yb.GetBlock(0));
   }

private:

   mutable mfem::Vector v1;
   mutable mfem::Vector v2;
   mutable mfem::Vector v3;
   mutable mfem::Vector v4;

   const mfem::Operator* O11;
   const mfem::Operator* O12;
   const mfem::Operator* O21;

   std::unique_ptr<mfem::HypreParMatrix> A11;
   std::unique_ptr<mfem::HypreBoomerAMG> amg11;
   std::unique_ptr<mfem::CGSolver> cg11;

   std::unique_ptr<mfem::HypreParMatrix> A;
   std::unique_ptr<mfem::HypreBoomerAMG> amga;
   std::unique_ptr<mfem::IterativeSolver> cga;


   int siz_u;
   int siz_p;

   mfem::Array<int> block_true_offsets;

   std::unique_ptr<mfem::HypreParMatrix> Qv;
   std::unique_ptr<mfem::HypreBoomerAMG> amgv;

   bool zero_mean_press;
   int myrank;
};
