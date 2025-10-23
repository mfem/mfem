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


class PenalizedDOFSelector
{
public:
    PenalizedDOFSelector(mfem::FiniteElementSpace* fes_):fes(fes_)
    {

    }

    /// return all true dofs where coeff > threshold
    void Selector(mfem::Coefficient* coeff, mfem::Array<int>& dof_list, real_t threshold=1e-9)
    {
        mfem::GridFunction gf(fes);
        if(fes->GetVDim()>1){
            mfem::VectorArrayCoefficient vc(fes->GetVDim());
            for(int i=0;i<fes->GetVDim();i++){
                vc.Set(i,coeff,false);
            }
            gf.ProjectCoefficient(vc);
        }else{
            gf.ProjectCoefficient(*coeff);
        }

        gf.SetTrueVector();
        mfem::Vector& tb=gf.GetTrueVector();

        for(int i=0;i<tb.Size();i++){
            if(tb[i]>threshold){
                dof_list.Append(i);
            }
        }
    }

private:

    mfem::FiniteElementSpace* fes;
};



class StokesSolver:public mfem::Operator
{
public:
   StokesSolver(mfem::ParMesh* mesh, int order_, int num_mesh_ref_=0);

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
   void SetGMRESPressureSolver(bool val_=true)
   {
      gmres_press=val_;
   }

   //velocity boundary conditions
   void AddVelocityBC(int id, std::shared_ptr<mfem::VectorCoefficient> val)
   {
      vel_bcs[id]=val;
   }

   void SetZeroVelocityBC(int id)
   {
       std::shared_ptr<mfem::VectorCoefficient> val;
       mfem::Vector vec(vfes->GetVectorDim()); vec=0.0;

       val.reset(new mfem::VectorConstantCoefficient(vec));
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

   /// Return the Stokes operator assembled by the solver
   mfem::Operator* GetStokesOperator(){ return bop.get();}

private:

   int myrank;


   /// The parallel mesh.
   mfem::ParMesh *pmesh = nullptr;

   /// The order of the velocity  space.
   int order;

   /// Geometric MG refinement levels
   int num_mesh_ref;

   mfem::Array<mfem::ParMesh*> meshes;

   //operators and smoothers for MG
   mfem::Array<StokesSolver*> solvers; //[nlevels-1], *this is not included in the array
   mfem::Array<mfem::Operator*> prolongations; // [nlevels-1]
   mfem::Array<mfem::Operator*> operators;     // [nlevels]
   mfem::Array<mfem::Solver*> smoothers;       // [nlevels]

   bool gmres_press;

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


   int sol_method; //by default use method 1
   void FSolve1();
   void FSolve2();
   void FSolve3();

};


/// Constructs reduced operator A[dof_list,dof_list], i.e.,
/// Ar=R*A*Rt, where R in R^{n \times m} is a restriction
/// built from a specified dof list, A in R^{m \times m},
/// Ar is in R^{n \timex n}, and Rt is the transpose of R.
class RestrictedOperator:public mfem::Operator
{
public:
    RestrictedOperator(const mfem::Operator* A_, mfem::Array<int>& dof_list_)
                : A(A_), dof_list(dof_list_)
    {
        mfem::Operator::width=dof_list.Size();
        mfem::Operator::height=dof_list.Size();

        tv.SetSize(A->Width()); tv=0.0;
        tr.SetSize(A->Width()); tr=0.0;

    }

    /// Operator application
    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        tv.SetSubVector(dof_list,x);
        A->Mult(tv,tr);
        tr.GetSubVector(dof_list,y);
    }

    /// Transpose operator application
    virtual
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
    {
        tv.SetSubVector(dof_list,x);
        A->MultTranspose(tv,tr);
        tr.GetSubVector(dof_list,y);
    }

    /// y=R*x
    void Restrict(const mfem::Vector& x, mfem::Vector& y)
    {
        x.GetSubVector(dof_list,y);
    }

    /// y=Rt*y
    void Prolongate(const mfem::Vector& x, mfem::Vector& y)
    {
        y=0.0;
        y.SetSubVector(dof_list,x);
    }

private:
    mutable mfem::Vector tv;
    mutable mfem::Vector tr;

    const mfem::Operator* A;
    mfem::Array<int>& dof_list;
};

class RestrictedSolver:public mfem::Solver
{
public:
    RestrictedSolver(mfem::Solver* solver_,  mfem::Array<int>& dof_list_)
        :solver(solver_),dof_list(dof_list_)
    {
        mfem::Operator::width=dof_list.Size();
        mfem::Operator::height=dof_list.Size();

        tv.SetSize(solver->Width()); tv=0.0;
        tr.SetSize(solver->Width()); tr=0.0;
    }

    /// Operator application
    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        tv.SetSubVector(dof_list,x);
        solver->Mult(tv,tr);
        tr.GetSubVector(dof_list,y);
    }

    /// Transpose operator application
    virtual
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
    {
        tv.SetSubVector(dof_list,x);
        solver->MultTranspose(tv,tr);
        tr.GetSubVector(dof_list,y);
    }

    /// Set/update the solver for the given operator.
    virtual void SetOperator(const Operator &op)
    {
        solver->SetOperator(op);
    }
private:
    mfem::Solver* solver;
    mfem::Array<int>& dof_list;

    mutable mfem::Vector tv;
    mutable mfem::Vector tr;
};

class DiagonalPrec:public mfem::Solver
{
public:
    DiagonalPrec(const mfem::Vector& v): diag(v)
    {
       mfem::Operator::width=diag.Size();
       mfem::Operator::height=diag.Size();
    }

    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        y=x;
        y*=diag;
    }

    virtual
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
    {
        y=x;
        y*=diag;
    }

    /// Set/update the solver for the given operator.
    virtual void SetOperator(const Operator &op)
    {

    }


private:
    mfem::Vector diag;
};

class DiagVelocityPrec:public mfem::IterativeSolver
{
public:
    DiagVelocityPrec(mfem::ParFiniteElementSpace* vfes_,
                 std::shared_ptr<mfem::Coefficient> visc_,
                 std::shared_ptr<mfem::Coefficient> brink_,
                 mfem::Array<int>& ess_vdofs_,                           const mfem::Operator* O11_,
                 const mfem::Operator* O12_,
                 const mfem::Operator* O21_)
                  :O11(O11_),O12(O12_),O21(O21_)
    {
        myrank=vfes_->GetMyRank();
        mfem::ConstantCoefficient lambda(0.00);
        mfem::ConstantCoefficient one(1.00);

        std::unique_ptr<mfem::ParBilinearForm>
                               b11(new mfem::ParBilinearForm(vfes_));
        b11->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*visc_));
        //b11->AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda,*visc_));
        if (nullptr!=brink_.get())
        {
            b11->AddDomainIntegrator(new mfem::VectorMassIntegrator(*brink_));
        }

        diag.SetSize(vfes_->GetTrueVSize());
        b11->AssembleDiagonal(diag);

        diag.Reciprocal();

        prec.reset(new mfem::OperatorJacobiSmoother(diag,ess_vdofs_));


        cga11.reset(new mfem::CGSolver(vfes_->GetComm()));
        cga11->SetOperator(*O11);
        cga11->SetPreconditioner(*prec);
    }

    /// Operator application
    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        cga11->SetAbsTol(mfem::IterativeSolver::abs_tol);
        cga11->SetRelTol(mfem::IterativeSolver::rel_tol);
        cga11->SetMaxIter(mfem::IterativeSolver::max_iter);
        cga11->iterative_mode=mfem::IterativeSolver::iterative_mode;
        cga11->SetPrintLevel(mfem::IterativeSolver::print_options);

        {
            //amg11->Mult(x,y);
            cga11->Mult(x,y);
        }
    }

private:

    const mfem::Operator* O11;
    const mfem::Operator* O12;
    const mfem::Operator* O21;

    int myrank;

    mfem::Vector diag;

    std::unique_ptr<mfem::Solver> prec;
    std::unique_ptr<mfem::CGSolver> cga11;
};


/// Preconditioner for the velocity block of the
/// Stokes problems.
class VelocityPrec:public mfem::IterativeSolver
{
public:
    VelocityPrec(mfem::ParFiniteElementSpace* vfes_,
                 std::shared_ptr<mfem::Coefficient> visc_,
                 std::shared_ptr<mfem::Coefficient> brink_,
                 mfem::Array<int>& ess_vdofs_,                           const mfem::Operator* O11_,
                 const mfem::Operator* O12_,
                 const mfem::Operator* O21_)
                  :O11(O11_),O12(O12_),O21(O21_)
    {
        myrank=vfes_->GetMyRank();
        mfem::ConstantCoefficient lambda(0.00);
        mfem::ConstantCoefficient one(1.00);

        std::unique_ptr<mfem::ParLORDiscretization>
                   lor_discr(new mfem::ParLORDiscretization(*vfes_));
        mfem::ParFiniteElementSpace& vlor=lor_discr->GetParFESpace();
        std::unique_ptr<mfem::ParBilinearForm>
                               b11(new mfem::ParBilinearForm(&vlor));
        b11->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*visc_));
        //b11->AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda,*visc_));
        if (nullptr!=brink_.get())
        {
            b11->AddDomainIntegrator(new mfem::VectorMassIntegrator(*brink_));
        }

        b11->Assemble(0);
        b11->Finalize(0);
        A11.reset(b11->ParallelAssemble());
        std::unique_ptr<mfem::HypreParMatrix> Ae(A11->EliminateRowsCols(ess_vdofs_));
        amg11.reset(new mfem::HypreBoomerAMG());
        amg11->SetOperator(*A11);
        int dim=vlor.GetParMesh()->Dimension();
        if (mfem::Ordering::Type::byNODES==vfes_->GetOrdering())
        {
            amg11->SetSystemsOptions(dim,true);
        }else{
            amg11->SetSystemsOptions(dim,false);
        }

        cga11.reset(new mfem::CGSolver(A11->GetComm()));
        cga11->SetOperator(*O11);
        cga11->SetPreconditioner(*amg11);

    }

    /// Operator application
    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        cga11->SetAbsTol(mfem::IterativeSolver::abs_tol);
        cga11->SetRelTol(mfem::IterativeSolver::rel_tol);
        cga11->SetMaxIter(mfem::IterativeSolver::max_iter);
        cga11->iterative_mode=mfem::IterativeSolver::iterative_mode;
        cga11->SetPrintLevel(mfem::IterativeSolver::print_options);

        {
            //amg11->Mult(x,y);
            cga11->Mult(x,y);
        }
    }

private:
    const mfem::Operator* O11;
    const mfem::Operator* O12;
    const mfem::Operator* O21;

    int myrank;
    std::unique_ptr<mfem::HypreParMatrix> A11;
    std::unique_ptr<mfem::HypreBoomerAMG> amg11;
    std::unique_ptr<mfem::CGSolver> cga11;
};

class SchurComplement:public mfem::Operator
{
public:
    SchurComplement(const mfem::Solver* invA11_,
                    const mfem::Operator* A12_,
                    const mfem::Operator* A21_):
        invA11(invA11_),A12(A12_), A21(A21_)
    {
        mfem::Operator::width=A12->Width();
        mfem::Operator::height=A21->Height();

        tv.SetSize(A12->Height()); tv=0.0;
        ty.SetSize(A12->Height()); ty=0.0;
    }

    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        A12->Mult(x,tv);
        invA11->Mult(tv,ty);
        A21->Mult(ty,y);
    }

    virtual
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
    {
        Mult(x,y);
    }

private:

    mutable mfem::Vector tv;
    mutable mfem::Vector ty;

    const mfem::Solver* invA11;
    const mfem::Operator* A12;
    const mfem::Operator* A21;
};


class SchurComplementLSCPrec:public mfem::IterativeSolver
{
public:
    SchurComplementLSCPrec(mfem::ParFiniteElementSpace* vfes_,
                           mfem::ParFiniteElementSpace* pfes_,
                           std::shared_ptr<mfem::Coefficient> visc_,
                           std::shared_ptr<mfem::Coefficient> brink_,
                           mfem::Array<int>& ess_vdofs_,
                           const mfem::Operator* O11_,
                           const mfem::Operator* O12_,
                           const mfem::Operator* O21_)
                            :O11(O11_),O12(O12_),O21(O21_)
    {
        myrank=vfes_->GetMyRank();
        // set the preconditioner for the upper block
        mfem::ConstantCoefficient lambda(0.00);
        mfem::ConstantCoefficient one(1.00);

        //assemble diagonal mass matrix on the velocity space
        {
           std::unique_ptr<mfem::ParLORDiscretization>
                       lor_discr(new mfem::ParLORDiscretization(*vfes_));
           std::unique_ptr<mfem::ParBilinearForm>
           //q11(new mfem::ParBilinearForm(vfes_));
           q11(new mfem::ParBilinearForm(&(lor_discr->GetParFESpace())));
           //q11->AddDomainIntegrator(new mfem::VectorMassIntegrator(one));

           q11->AddDomainIntegrator(
                       new mfem::LumpedIntegrator(
                           new mfem::VectorMassIntegrator(one)));
           q11->Assemble(0);
           q11->Finalize(0);
           Qv.reset(q11->ParallelAssemble());
           amgv.reset(new mfem::HypreBoomerAMG());
           amgv->SetOperator(*Qv);
        }

        //construct Laplace preconditioner
        {
            mfem::ConstantCoefficient epsone(1e-6);
            std::unique_ptr<mfem::ParLORDiscretization>
                       lor_pres(new mfem::ParLORDiscretization(*pfes_));
            std::unique_ptr<mfem::ParBilinearForm>
                    aa(new mfem::ParBilinearForm(&(lor_pres->GetParFESpace())));

            aa->AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
            aa->AddDomainIntegrator(new mfem::MassIntegrator(epsone));

            aa->Assemble(0);
            aa->Finalize(0);

            A.reset(aa->ParallelAssemble());
            if(0==myrank){ std::cout<<"Step 1"<<std::endl;}

            amga.reset(new mfem::HypreBoomerAMG());
            amga->SetOperator(*A);
        }

        siz_u=O11->NumRows();
        siz_p=O21->NumRows();
        this->width = siz_p;
        this->height = siz_p;

        tv.SetSize(siz_u); tv=0.0;
        ty.SetSize(siz_u); ty=0.0;
        tp.SetSize(siz_p); tp=0.0;
        ts.SetSize(siz_p); ts=0.0;

    }

    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        SchurComplement sc(amgv.get(), O12, O21);

        std::unique_ptr<mfem::GMRESSolver> gm;
        gm.reset(new mfem::GMRESSolver(A->GetComm()));
        gm->SetOperator(sc);
        gm->SetPreconditioner(*amga);
        gm->SetMaxIter(mfem::IterativeSolver::max_iter);
        gm->SetAbsTol(mfem::IterativeSolver::abs_tol);
        gm->SetRelTol(mfem::IterativeSolver::rel_tol);
        gm->iterative_mode=mfem::IterativeSolver::iterative_mode;
        gm->SetPrintLevel(mfem::IterativeSolver::print_options);

        gm->Mult(x,ts);
        O12->Mult(ts,tv);
        amgv->Mult(tv,ty);
        O11->Mult(ty,tv);
        amgv->Mult(tv,ty);
        O21->Mult(ty,tp);
        gm->Mult(tp,y);
    }

    virtual
    void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override
    {
        Mult(x,y);
    }

private:

    int myrank;

    const mfem::Operator* O11;
    const mfem::Operator* O12;
    const mfem::Operator* O21;

    std::unique_ptr<mfem::HypreParMatrix> A;
    std::unique_ptr<mfem::HypreBoomerAMG> amga;

    std::unique_ptr<mfem::HypreParMatrix> Qv;
    std::unique_ptr<mfem::HypreBoomerAMG> amgv;

    int siz_u;
    int siz_p;

    mutable mfem::Vector tv;
    mutable mfem::Vector ty;
    mutable mfem::Vector tp;
    mutable mfem::Vector ts;

};


class StokesLSCPrec: public mfem::IterativeSolver
{
public:
    StokesLSCPrec(mfem::ParFiniteElementSpace* vfes_,
                  mfem::ParFiniteElementSpace* pfes_,
                  std::shared_ptr<mfem::Coefficient> visc_,
                  std::shared_ptr<mfem::Coefficient> brink_,
                  mfem::Array<int>& ess_vdofs_,
                  const mfem::Operator* O11_,
                  const mfem::Operator* O12_,
                  const mfem::Operator* O21_)
                  :O11(O11_),O12(O12_),O21(O21_)
    {
        myrank=vfes_->GetMyRank();

        vprec.reset(new VelocityPrec(vfes_,visc_,brink_,ess_vdofs_,
                                      O11_,O12_,O21_));
        sprec.reset(new SchurComplementLSCPrec(
                        vfes_,pfes_,visc_,brink_,ess_vdofs_,
                        O11_,O12_,O21_));
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
    }

    /// Operator application
    virtual
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
       mfem::BlockVector xb,yb;

       xb.Update(const_cast<mfem::Vector&>(x), block_true_offsets);
       yb.Update(y, block_true_offsets);

       vprec->SetMaxIter(mfem::IterativeSolver::max_iter);
       vprec->SetAbsTol(mfem::IterativeSolver::abs_tol);
       vprec->SetRelTol(mfem::IterativeSolver::rel_tol);
       vprec->SetPrintLevel(mfem::IterativeSolver::print_options);
       vprec->iterative_mode=mfem::IterativeSolver::iterative_mode;

       sprec->SetMaxIter(2*mfem::IterativeSolver::max_iter);
       sprec->SetAbsTol(mfem::IterativeSolver::abs_tol);
       sprec->SetRelTol(mfem::IterativeSolver::rel_tol);
       sprec->SetPrintLevel(mfem::IterativeSolver::print_options);
       sprec->iterative_mode=mfem::IterativeSolver::iterative_mode;

       if(!mfem::IterativeSolver::iterative_mode){
           y=0.0;
       }

       O12->Mult(yb.GetBlock(1),v2);
       add(xb.GetBlock(0), -1.0, v2, v3);
       vprec->Mult(v3, yb.GetBlock(0));

       O21->Mult(yb.GetBlock(0),v1);
       add(xb.GetBlock(1),-1.0, v1, v4);
       sprec->Mult(v4, yb.GetBlock(1));
       yb.GetBlock(1).Neg();
    }


private:
    int myrank;

    const mfem::Operator* O11;
    const mfem::Operator* O12;
    const mfem::Operator* O21;

    int siz_u;
    int siz_p;

    mfem::Array<int> block_true_offsets;

    std::unique_ptr<VelocityPrec> vprec;
    std::unique_ptr<SchurComplementLSCPrec> sprec;

    mutable mfem::Vector v1;
    mutable mfem::Vector v2;
    mutable mfem::Vector v3;
    mutable mfem::Vector v4;

};
