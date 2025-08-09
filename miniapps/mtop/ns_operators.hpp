#ifndef NS_OPERATORS_HPP
#define NS_OPERATORS_HPP


#include "mfem.hpp"
#include <fstream>
#include <iostream>

namespace mfem{


class PCLSC:public IterativeSolver
{
public:

    /// Approximates the inverse of the Schur complement S
    /// Constructor taking A11 as operator amd A12 and A21 as matrices
    /// Ad is a diagonal matrix - if null no scaling is performed.
    /// if Ai=Ad^{-1} is the inverse of Ad
    ///
    /// S^{-1}=(A21*Ai*A12)^{-1} *(A21*Ai*A11*Ai*A12)*(A21*Ai*A12)^{-1}
    ///
    /// The matrix A12_ is scaled with 1/c12_, i.e., the true matrix is c12_*A12
    /// The matrix A21_ is scaled with 1/c21_, i.e., the true matrix us c21_*A21
    ///
    /// NOTE: Instead of rescaling the matrices in the construction of the prec,
    ///  the final result is scaled with 1/(c12*c21).
    PCLSC(Operator* A11_, HypreParMatrix* A12_, HypreParMatrix* A21_,
          real_t c12_, real_t c21_, mfem::HypreParVector* Ad_=nullptr)
    {
        A11=A11_;
        A12=A12_;
        A21=A21_;

        c12=c12_;
        c21=c21_;

        Ad=Ad_;

        if(Ad!=nullptr){
            MA21.reset(new HypreParMatrix(*A21));
            MA21->InvScaleRows(*Ad);
            M=std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(MA21.get(),A12));
        }else{
            M=std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(A21,A12));
        }

        prec=std::unique_ptr<mfem::HypreBoomerAMG>(new mfem::HypreBoomerAMG());
        prec->SetOperator(*M);
        prec->SetPrintLevel(1);

        p.SetSize(M->GetNumCols());
        u.SetSize(A12->GetNumRows());
        v.SetSize(A12->GetNumRows());

        mfem::Operator::width=M->GetNumCols();
        mfem::Operator::height=M->GetNumCols();

        ls=std::unique_ptr<mfem::IterativeSolver>(new mfem::CGSolver(A12->GetComm()));
        ls->SetOperator(*M);
        ls->SetPreconditioner(*prec);
        ls->SetAbsTol(1e-8);
        ls->SetRelTol(1e-8);
        ls->SetMaxIter(100);
        ls->SetPrintLevel(1);

        std::cout<<" PCLSC allocated"<<std::endl;
    }

    virtual
    ~PCLSC(){}

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        ls->Mult(x,p);
        A12->Mult(p,u);
        if(Ad!=nullptr){u/=*Ad;}
        A11->Mult(u,v);
        if(Ad!=nullptr){v/=*Ad;}
        A21->Mult(v,p);
        ls->Mult(p,y);
        //scale the result
        y/=(c12*c21);
    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {
        Mult(x,y);
    }

    virtual void SetPrintLevel(int print_lvl) override
    {
        prec->SetPrintLevel(print_lvl);
        ls->SetPrintLevel(print_lvl);
    }


    virtual void SetAbsTol(real_t tol_)
    {
        ls->SetAbsTol(tol_);
    }

    virtual void SetRelTol(real_t tol_)
    {
       ls->SetRelTol(tol_);
    }

    virtual void SetMaxIter(int it)
    {
        ls->SetMaxIter(it);
    }

private:
    mfem::HypreParMatrix *A12;
    mfem::HypreParMatrix *A21;
    mfem::Operator *A11;

    real_t c12;
    real_t c21;

    mfem::HypreParVector* Ad;

    std::unique_ptr<mfem::HypreParMatrix> MA21;

    std::unique_ptr<mfem::HypreParMatrix> M;
    std::unique_ptr<mfem::HypreBoomerAMG> prec;
    std::unique_ptr<mfem::IterativeSolver> ls;

    mutable mfem::Vector p;
    mutable mfem::Vector u;
    mutable mfem::Vector v;
};


class DBlockPrec:public IterativeSolver
{
public:
    DBlockPrec(HypreParMatrix* A11_, HypreParMatrix* A12_, HypreParMatrix* A21_,
               real_t c12_, real_t c21_, mfem::HypreParVector* Ad_=nullptr)
    {
        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = A11_->Height();
        block_true_offsets[2] = A21_->Height();
        block_true_offsets.PartialSum();

        schur_cmpl.reset(new PCLSC(A11_,A12_,A21_, c12_, c21_, Ad_));

        prec.reset(new mfem::HypreBoomerAMG());
        prec->SetPrintLevel(1);

        ls.reset(new mfem::CGSolver(A11_->GetComm()));
        ls->SetAbsTol(atol);
        ls->SetRelTol(rtol);
        ls->SetMaxIter(max_iter);
        ls->SetOperator(*A11_);

        prec->SetOperator(*A11_);
        ls->SetPreconditioner(*prec);

        Operator::width=block_true_offsets[2];
        Operator::height=block_true_offsets[2];

        bop.reset(new BlockDiagonalPreconditioner(block_true_offsets));
        bop->SetDiagonalBlock(0,ls.get());
        bop->SetDiagonalBlock(1,schur_cmpl.get());

    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        bop->Mult(x,y);
    }

    virtual void SetPrintLevel(int print_lvl) override
    {
        prec->SetPrintLevel(print_lvl);
        ls->SetPrintLevel(print_lvl);
        schur_cmpl->SetPrintLevel(print_lvl);
    }

    virtual void SetAbsTol(real_t tol_)
    {
       ls->SetAbsTol(tol_);
       schur_cmpl->SetAbsTol(tol_);
    }

    virtual void SetRelTol(real_t tol_)
    {
       ls->SetRelTol(tol_);
       schur_cmpl->SetRelTol(tol_);
    }

    virtual void SetMaxIter(int it)
    {
        ls->SetMaxIter(it);
        schur_cmpl->SetMaxIter(it);
    }

    virtual
    ~DBlockPrec(){}


private:
    Array<int> block_true_offsets;

    std::unique_ptr<IterativeSolver> schur_cmpl;

    std::unique_ptr<HypreBoomerAMG> prec;
    std::unique_ptr<IterativeSolver> ls;

    real_t atol=1e-8;
    real_t rtol=1e-8;
    int max_iter=10;

    std::unique_ptr<BlockDiagonalPreconditioner> bop;
};


class GSBlockSolver:public IterativeSolver
{
public:
    GSBlockSolver(HypreParMatrix* A11_, HypreParMatrix* A12_, HypreParMatrix* A21_,
                  real_t c11_, real_t c12_, real_t c21_){

        block_true_offsets.SetSize(3);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = A11_->Height();
        block_true_offsets[2] = A21_->Height();
        block_true_offsets.PartialSum();


        A11=A11_;
        A12=A12_;
        A21=A21_;

        c11=c11_;
        c12=c12_;
        c21=c21_;
    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {

    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {

    }


    ~GSBlockSolver(){

    }
protected:
    Array<int> block_true_offsets;


    std::unique_ptr<IterativeSolver> ls;
    std::unique_ptr<IterativeSolver> prec;
    std::unique_ptr<BlockOperator> bop;

    real_t c11;
    real_t c12;
    real_t c21;

    HypreParMatrix* A11;
    HypreParMatrix* A12;
    HypreParMatrix* A21;

};

// direct block solver for time dependent Stokes-Brinkman solver
class GSDirectBlockSolver:public GSBlockSolver{
public:
    GSDirectBlockSolver(HypreParMatrix* A11_, HypreParMatrix* A12_, HypreParMatrix* A21_, HypreParMatrix* A22_,
                        real_t c11_, real_t c12_, real_t c21_, real_t c22_):GSBlockSolver(A11_,A12_,A21_,c11_,c12_,c21_)
    {
        //allocate MUMPS
        Array2D<const HypreParMatrix*> am(2,2);
        am(0,0)=A11_;
        am(0,1)=A12_;
        am(1,0)=A21_;
        am(1,1)=A22_;

        Array2D<real_t> cm(2,2);
        cm(0,0)=c11_; cm(0,1)=c12_;
        cm(1,0)=c21_; cm(1,1)=c22_;

        std::unique_ptr<HypreParMatrix> bm;
        bm.reset(HypreParMatrixFromBlocks(am,&cm));

        solver.reset(new MUMPSSolver(A11_->GetComm()));
        //mumps->SetPrintLevel(2);
        solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
        //solver->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
        solver->SetOperator(*bm);
    }

    /// Operator application
    void Mult (const mfem::Vector & x, mfem::Vector & y) const override
    {
        solver->Mult(x,y);
    }

    /// Action of the transpose operator
    void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const override
    {
        solver->MultTranspose(x,y);
    }

    ~GSDirectBlockSolver()
    {

    }

private:
    std::unique_ptr<MUMPSSolver> solver;
};


class TimeDependentStokes:public TimeDependentOperator
{
public:
    TimeDependentStokes(ParMesh* mesh, int order_, std::shared_ptr<Coefficient> visc_,
                        bool partial_assembly_=false, bool verbose_=true);

    virtual
    ~TimeDependentStokes();


    //velocity boundary conditions
    void AddVelocityBC(int id, std::shared_ptr<VectorCoefficient> val)
    {
        vel_bcs[id]=val;
    }

    //pressure boundary conditions
    void AddPressureBC(int id, std::shared_ptr<Coefficient> val)
    {
        pre_bcs[id]=val;
    }

    /// Set the Velocity BC on a given ParGridFunction.
    void SetEssVBC(real_t t, ParGridFunction& pgf);

    /// Set the Pressure BC on a given ParGridFunction.
    void SetEssPBC(real_t t, ParGridFunction& pgf);




    /// Set Brinkman coefficient
    void SetBrinkman(std::shared_ptr<Coefficient> brink_)
    {
       brink = brink_;
    }

    /// Set Viscosity coefficient
    void SetViscosity(std::shared_ptr<Coefficient> visc_)
    {
       visc = visc_;
    }

    /// Set volumetric force
    void SetVolForce(std::shared_ptr<VectorCoefficient> force_)
    {
        vol_force=force_;
    }

    /// Set up BC and discretizations necessary for the time integration.
    /// Call this method before starting the time integration.
    void Assemble();


    virtual
    void ImplicitSolve(const real_t gamma, const Vector &u, Vector &k) override;

    ParFiniteElementSpace* GetVelocitySpace(){return vfes;}
    ParFiniteElementSpace* GetPressureSpace(){return pfes;}
    Array<int>& GetTrueBlockOffsets(){return block_true_offsets;}



private:
   /// assbled flag
   bool assembled = false;


   /// Enable/disable debug output.
   bool debug = false;

   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;

   /// The parallel mesh.
   ParMesh *pmesh = nullptr;

   /// The order of the velocity  space.
   int order;

   /// linear system solvers parameters
   real_t linear_atol;
   real_t linear_rtol;
   int  linear_iter;

   std::shared_ptr<Coefficient> visc; //viscosity
   std::shared_ptr<Coefficient> brink; //Brinkman penalization

   H1_FECollection* vfec; //velocity collections
   H1_FECollection* pfec; //pressure collecation
   ParFiniteElementSpace* vfes;
   ParFiniteElementSpace* pfes;

   std::unique_ptr<IterativeSolver> ls;
   std::unique_ptr<Solver> prec;

   //boundary conditions
   std::map<int, std::shared_ptr<VectorCoefficient>> vel_bcs;
   std::map<int, std::shared_ptr<Coefficient>> pre_bcs;

   // holds the velocity constrained DOFs
   mfem::Array<int> ess_tdofv;

   // holds the pressure constrained DOFs
   mfem::Array<int> ess_tdofp;

   // Volume force coefficient
   std::shared_ptr<VectorCoefficient> vol_force;

   Array<int> block_true_offsets;
   int siz_u;
   int siz_p;

   ConstantCoefficient onecoeff;
   ConstantCoefficient zerocoef;

   /// Extracts the true boundary doffs of the velocity
   void SetEssTDofsV(mfem::Array<int>& ess_dofs);
   void SetEssTDofsP(mfem::Array<int>& ess_dofs);


   std::unique_ptr<HypreParMatrix> M;
   std::unique_ptr<HypreParMatrix> K;
   std::unique_ptr<HypreParMatrix> P;
   std::unique_ptr<HypreParMatrix> B;
   std::unique_ptr<HypreParMatrix> C;
   std::unique_ptr<HypreParMatrix> H; //diagonal mass matrix for stabilizing the pressure

   std::unique_ptr<HypreParMatrix> A11;
   std::unique_ptr<HypreParMatrix> A12;
   std::unique_ptr<HypreParMatrix> A21;
   std::unique_ptr<HypreParMatrix> A22;

   std::unique_ptr<HypreParMatrix> A11e;
   std::unique_ptr<HypreParMatrix> A12e;
   std::unique_ptr<HypreParMatrix> A21e;
   std::unique_ptr<HypreParMatrix> A22e;

   mutable BlockVector rhs;
   mutable ParGridFunction vel;
   mutable ParGridFunction pre;


};



}



#endif // NS_OPERATORS_HPP
